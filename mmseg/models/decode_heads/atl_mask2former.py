# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmcv.ops import point_sample
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.utils import preprocess_panoptic_gt
from mmdet.models.utils.misc import multi_apply
# try:
#     from mmdet.models.dense_heads import \
#         Mask2FormerHead as MMDET_Mask2FormerHead
# except ModuleNotFoundError:
#     MMDET_Mask2FormerHead = BaseModule
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from mmengine.structures import InstanceData
from mmengine.utils import deprecated_api_warning
from torch import Tensor

from mmseg.registry import MODELS, TASK_UTILS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from mmseg.utils.misc import add_prefix
from .decode_head import BaseDecodeHead


# ======================= mmseg_custom/models/plugins/msdeformattn_pixel_decoder.py =======================
class MSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`mmcv.ConfigDict` | dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`mmcv.ConfigDict` | dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`mmcv.ConfigDict` | dict): Config for transformer
            encoder. Defaults to `DetrTransformerEncoder`.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`mmcv.ConfigDict` | dict): Initialization config dict.
    """

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        encoder=None,
        init_cfg=None,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
    ):
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder.layer_cfg.self_attn_cfg.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in self_attn_cfg must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)

        self.encoder = MODELS.build(encoder)
        self.postional_encoding = MODELS.build(positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for layer in self.encoder.layers:
            if isinstance(layer.self_attn, MultiScaleDeformableAttention):
                layer.self_attn.init_weights()

        # for m in self.encoder.layers.modules():
        #     if isinstance(m, MultiScaleDeformableAttention):
        #         m.init_weights()

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=0)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_query, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[:self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def get_uncertainty(mask_pred, labels):
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_pred' for the foreground class in `classes`.

    Args:
        mask_pred (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (list[Tensor]): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_pred.shape[1] == 1:
        gt_class_logits = mask_pred.clone()
    else:
        inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
        gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(mask_pred, labels, num_points,
                                               oversample_ratio,
                                               importance_sample_ratio):
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (list): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_pred.device)
    point_logits = point_sample(mask_pred, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size, num_random_points, 2, device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords
