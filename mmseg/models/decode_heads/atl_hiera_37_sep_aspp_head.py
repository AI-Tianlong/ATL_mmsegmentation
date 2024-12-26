# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer

from mmseg.models.losses import accuracy
from mmseg.registry import MODELS
from mmseg.utils import SampleList, ConfigType
from ..utils import resize
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
                 num_classes,
                 dilations=(1, 12, 24, 36),
                 proj: str = 'convmlp',
                 **kwargs):
        

        self.hiera_num_classes = num_classes
        num_classes = sum(num_classes)
        

        super().__init__(num_classes=num_classes, 
                         dilations=dilations,
                          **kwargs)

        # import pdb; pdb.set_trace()

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
        seg_logit = seg_logits[0]
        tree_triplet_embedding = seg_logits[1]
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        seg_logit = resize(
            input=seg_logit,
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
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else: # 如果同名的话，累加loss值
                loss[loss_decode.loss_name] += loss_decode(
                    self.step,
                    tree_triplet_embedding,
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # import pdb; pdb.set_trace()
        # 正好最后19个 19个类别
        loss['acc_seg'] = accuracy(seg_logit[:, -self.hiera_num_classes[-1]:, :, :], seg_label)
        return loss



    # ==================== 以下是推理用的函数 ====================
    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)  # 过decode_head的forward--->[2,40,128,128]
        seg_logit = seg_logits[0]

        if self.test_cfg.merge_hirtal:
            # L3[:, 0:1] = L3[:, 0:1] + L2[:, 0:1] + L1[:, 0:1]


        return self.predict_by_feat(seg_logit, batch_img_metas)   # [2,40,512,512]
    
    
    
    # def predict_by_feat(self, seg_logit: Tuple[Tensor],
    #                     batch_img_metas: List[dict]) -> Tensor:
    #     """Transform a batch of output seg_logits to the input shape.

    #     Args:
    #         seg_logits (Tensor): The output from decode head forward function.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.

    #     Returns:
    #         Tensor: Outputs segmentation logits map.
    #     """
    #     # HSSN decode_head output is: (out, embedding): tuple
    #     # only need 'out' here.

    #     if seg_logit.size(1) == 26:  # For cityscapes dataset，19 + 7
    #         hiera_num_classes = 7
    #         seg_logit[:, 0:2] += seg_logit[:, -7]
    #         seg_logit[:, 2:5] += seg_logit[:, -6]
    #         seg_logit[:, 5:8] += seg_logit[:, -5]
    #         seg_logit[:, 8:10] += seg_logit[:, -4]
    #         seg_logit[:, 10:11] += seg_logit[:, -3]
    #         seg_logit[:, 11:13] += seg_logit[:, -2]
    #         seg_logit[:, 13:19] += seg_logit[:, -1]

    #     elif seg_logit.size(1) == 12:  # For Pascal_person dataset, 7 + 5
    #         hiera_num_classes = 5
    #         seg_logit[:, 0:1] = seg_logit[:, 0:1] + \
    #             seg_logit[:, 7] + seg_logit[:, 10]
    #         seg_logit[:, 1:5] = seg_logit[:, 1:5] + \
    #             seg_logit[:, 8] + seg_logit[:, 11]
    #         seg_logit[:, 5:7] = seg_logit[:, 5:7] + \
    #             seg_logit[:, 9] + seg_logit[:, 11]

    #     elif seg_logit.size(1) == 25:  # For LIP dataset, 20 + 5
    #         hiera_num_classes = 5
    #         seg_logit[:, 0:1] = seg_logit[:, 0:1] + \
    #             seg_logit[:, 20] + seg_logit[:, 23]
    #         seg_logit[:, 1:8] = seg_logit[:, 1:8] + \
    #             seg_logit[:, 21] + seg_logit[:, 24]
    #         seg_logit[:, 10:12] = seg_logit[:, 10:12] + \
    #             seg_logit[:, 21] + seg_logit[:, 24]
    #         seg_logit[:, 13:16] = seg_logit[:, 13:16] + \
    #             seg_logit[:, 21] + seg_logit[:, 24]
    #         seg_logit[:, 8:10] = seg_logit[:, 8:10] + \
    #             seg_logit[:, 22] + seg_logit[:, 24]
    #         seg_logit[:, 12:13] = seg_logit[:, 12:13] + \
    #             seg_logit[:, 22] + seg_logit[:, 24]
    #         seg_logit[:, 16:20] = seg_logit[:, 16:20] + \
    #             seg_logit[:, 22] + seg_logit[:, 24]

    #     # elif seg_logit.size(1) == 144 # For Mapillary dataset, 124+16+4
    #     # unofficial repository not release mapillary until 2023/2/6

    #     if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
    #         # slide inference
    #         size = batch_img_metas[0]['img_shape']
    #     elif 'pad_shape' in batch_img_metas[0]:
    #         size = batch_img_metas[0]['pad_shape'][:2]
    #     else:
    #         size = batch_img_metas[0]['img_shape']
    #     seg_logit = seg_logit[:, :-hiera_num_classes]
    #     seg_logit = resize(
    #         input=seg_logit,
    #         size=size,
    #         mode='bilinear',
    #         align_corners=self.align_corners)

    #     return seg_logit
    
    # def predict_by_feat(self, seg_logits: Tensor,
    #                     batch_img_metas: List[dict]) -> Tensor:
    #     """Transform a batch of output seg_logits to the input shape.  # 缩放！

    #     Args:
    #         seg_logits (Tensor): The output from decode head forward function.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.

    #     Returns:
    #         Tensor: Outputs segmentation logits map.
    #     """

    #     if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
    #         # slide inference
    #         size = batch_img_metas[0]['img_shape']
    #     elif 'pad_shape' in batch_img_metas[0]:
    #         size = batch_img_metas[0]['pad_shape'][:2]
    #     else:
    #         size = batch_img_metas[0]['img_shape']

    #     seg_logits = resize(
    #         input=seg_logits,
    #         size=size,
    #         mode='bilinear',
    #         align_corners=self.align_corners)
    #     return seg_logits