# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer

from mmseg.models.losses import accuracy
from mmseg.registry import MODELS
from mmseg.utils import SampleList, resize
from .aspp_head import ASPPHead, ASPPModule

from typing import List, Tuple

class ProjectionHead(nn.Module):
    """ProjectionHead, project feature map to specific channels.

    Args:
        dim_in (int): Input channels.
        norm_cfg (dict): config of norm layer.
        proj_dim (int): Output channels. Default: 256.
        proj (str): Projection type, 'linear' or 'convmlp'. Default: 'convmlp'
    """

    def __init__(self,
                 dim_in: int,
                 norm_cfg: dict,
                 proj_dim: int = 256,
                 proj: str = 'convmlp'):
        super().__init__()
        assert proj in ['convmlp', 'linear']
        if proj == 'linear':  # 投影，线性的话，就用1x1的卷积，将输入通道数变为输出通道数
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':  # 如果是convmlp，就用两个1x1的卷积，中间加上BN和ReLU
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1], nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)



class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@MODELS.register_module()
class ATL_Hiera_DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        proj (str): The type of ProjectionHead, 'linear' or 'convmlp',
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, 
                 c1_in_channels, 
                 c1_channels,
                 proj: str = 'convmlp',
                 **kwargs):
        
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        self.proj_head = ProjectionHead(dim_in=2048,   # 2048-->256
                                        norm_cfg=self.norm_cfg, 
                                        proj=proj)
        self.register_buffer('step', torch.zeros(1))


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)

        self.step += 1
        embedding = self.proj_head(inputs[-1]) # 最后的一个特征图

        return output, embedding


    def loss_by_feat(
            self,
            seg_logits: Tuple[Tensor],  # (out, embedding)
            batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss. Will fix in future.

        Args:
            seg_logits (Tuple[Tensor]): The output from decode head
                forward function.
                For this decode_head output are (out, embedding): tuple
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits_ = seg_logits[0]
        tree_triplet_embedding = seg_logits[1]
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        seg_logit = resize(
            input=seg_logits_,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        
        seg_label = seg_label.squeeze(1)
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                loss[loss_decode.loss_name] = loss_decode(
                    self.step,
                    tree_triplet_embedding,
                    seg_logits_,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else: # 如果同名的话，累加loss值
                loss[loss_decode.loss_name] += loss_decode(
                    self.step,
                    tree_triplet_embedding,
                    seg_logits_,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        
        # # 缩放0.5啥意思？
        # seg_logit_before = resize(
        #     input=seg_logit_before,
        #     scale_factor=0.5,  
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        
        # 正好最后19个 19个类别
        loss['acc_seg'] = accuracy(seg_logit[:, :19], seg_label)
        return loss
