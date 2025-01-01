.0  # Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from venv import logger

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from torch import Tensor
import mmcv 

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..losses import accuracy
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM


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
                build_norm_layer(norm_cfg, dim_in)[1], 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)


@MODELS.register_module()
class ATL_UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 num_level_classes=None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        # 屏蔽掉BaseDecodeHead里的conv_seg,否则会报错
        # arameters which did not receive grad for rank 2: decode_head.conv_seg.bias, decode_head.conv_seg.weight

        # 分开输出的话
        if num_level_classes is not None:
            self.conv_seg_L1 = nn.Conv2d(self.channels, num_level_classes[0], kernel_size=1) #(1024-->6)
            self.conv_seg_L2 = nn.Conv2d(self.channels, num_level_classes[1], kernel_size=1) #(1024-->12)
            self.conv_seg_L3 = self.conv_seg #(1024-->22)
            # self.conv_seg_L3 = self.conv_seg

            self.num_level_classes = num_level_classes

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals,
        # 3xConvModule{Conv2d(1024, 1024, (1,1), (1,1), bias=False) + bn + ReLU}
        #
        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32]]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32], [2, 1024, 16, 16]]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])
        # fpn_outs: [[2,1024,128,128],[2,1024,64,64],[2,1024,32,32],[2,1024,16,16]]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # fpn_outs: [[2,1024,128,128],[2,1024,128,128],[2,1024,128,128],[2,1024,128,128]]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # fpn_outs: [2,4096,128,128]

        feats = self.fpn_bottleneck(fpn_outs)
        # ConvModule(
        # (conv): Conv2d(4096, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (activate): ReLU(inplace=True)
        # )
        # feats.shape: torch.Size([2, 1024, 128, 128])
        return feats

    # 分开输出的话
    def cls_seg(self, feat, conv_seg):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = conv_seg(feat)
        return output

    # 把最后三个分开，然后分别输出
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # [2,1024,128,128]
        # output_L1 = self.cls_seg(output, self.conv_seg_L1)  # [2,6,128,128]

        output_L1 = self.cls_seg(output, self.conv_seg_L1)  # [2,6,128,128]
        output_L2 = self.cls_seg(output, self.conv_seg_L2)  # [2,12,128,128]
        output_L3 = self.cls_seg(output, self.conv_seg_L3 )    # [2,22,128,128]
        # output = torch.cat([output_L1, output_L2, output_L3], dim=1)  # [2,40,128,128]
        # 消融实验1：只让模型输出最后output_L3的结果,然后用普通的Loss和普通的EncoderDecoder训练

        # import pdb; pdb.set_trace()
        return output_L1, output_L2, output_L3

    # # 合在一起输出一个40通道的特征图
    # def forward(self, inputs):
    #     """Forward function."""
    #     output = self._forward_feature(inputs)  # [2,1024,128,128]
    #     output = self.cls_seg(output)  # [2,40,128,128]

    #     # 消融实验1：只让模型输出最后output_L3的结果,然后用普通的Loss和普通的EncoderDecoder训练

    #     # import pdb; pdb.set_trace()
    #     return output


# 这里要把三个输出都合并，然后在cat在一起。其实还是[2,40,128,128]
# output_L1 = [2,6,128,128]  output_L2 = [2,12,128,128]  output_L3 = [2,22,128,128]
# 然后cat一下，还是[2,40,128,128]

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss. Overwrite the
        BaseDecodeHead.loss_by_feat() function.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # import pdb;pdb.set_trace()
        # seg_logits = seg_logits[2]  # output_L3
        # seg_logits: [2,40,128,128]，三个output拼起来的

        seg_label = self._stack_batch_gt(
            batch_data_samples)  # 从batch_data_samples里提取 [2,1,512,512]
        loss = dict()
        # 直接把，[2,40,128,128]---双线性插值--->[2,40,512,512]
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
            # print('走的这里')
        seg_label = seg_label.squeeze(
            1)  # 去掉标签的通道维度 [2,1,512,512]-->[2,512,512]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        # import pdb;pdb.set_trace()
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],去过atl_loss的forward
                # pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)  # cross_entropy
                # pdb.set_trace()

        if self.num_level_classes is not None:
            num_low_level_classes = self.num_level_classes[-1]  # 22
        seg_logits_low_level = seg_logits[:,
                                          -num_low_level_classes:]  # [2,22,512,512]  # 这里不准确。
        loss['acc_seg'] = accuracy(
            seg_logits_low_level, seg_label,
            ignore_index=self.ignore_index)  # 这里为什么低，因为参数没全导入。
        # pdb.set_trace()
        return loss  #这可以正常执行

    # ATL_EncoderDecoder里的predict函数会调用UperHead的predict，又会调这里。
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.  # 缩放！

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # [1,40,512,512]
        # seg_logits # [1,40,128,128]
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        # [1,40,512,512]
        return seg_logits


@MODELS.register_module()
class ATL_hiera_UPerHead_Multi_convseg(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 num_classes_level_list=[5,10,19],
                 merge_hiera: bool = True,
                 proj: str = 'convmlp',
                 **kwargs):

        num_classes = num_classes_level_list[-1] # 去创建 self.conv_seg
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select', 
                         **kwargs)

        # 屏蔽掉BaseDecodeHead里的conv_seg,否则会报错
        # arameters which did not receive grad for rank 2: decode_head.conv_seg.bias, decode_head.conv_seg.weight

        self.merge_hiera = merge_hiera

    
        # 分开输出的话
        if isinstance(num_classes_level_list, list):
            self.num_classes_level_list = num_classes_level_list  # [5,9,10]

            self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->6)
            self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1) #(1024-->12)
            self.conv_seg_L3 = self.conv_seg #(1024-->22)

        elif isinstance(num_classes_level_list, int):
            num_classes = num_classes

        # UperHead define
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # 将最后一个1024维度的特征图映到256维 
        self.proj_head = ProjectionHead(dim_in=self.in_channels[-1] if isinstance(self.in_channels, list) else self.in_channels,  # 2048-->256
                                        norm_cfg=self.norm_cfg, 
                                        proj=proj)
        self.register_buffer('step', torch.zeros(1))

    
    # 分开输出的话
    def cls_seg(self, feat, conv_seg):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = conv_seg(feat)
        return output
    

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals,
        # 3xConvModule{Conv2d(1024, 1024, (1,1), (1,1), bias=False) + bn + ReLU}
        #
        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32]]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32], [2, 1024, 16, 16]]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])
        # fpn_outs: [[2,1024,128,128],[2,1024,64,64],[2,1024,32,32],[2,1024,16,16]]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # fpn_outs: [[2,1024,128,128],[2,1024,128,128],[2,1024,128,128],[2,1024,128,128]]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # fpn_outs: [2,4096,128,128]

        feats = self.fpn_bottleneck(fpn_outs)
        # ConvModule(
        # (conv): Conv2d(4096, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (activate): ReLU(inplace=True)
        # )
        # feats.shape: torch.Size([2, 1024, 128, 128])
        return feats


    # 把最后三个分开，然后分别输出
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # [2,1024,128,128]
        # output_L1 = self.cls_seg(output, self.conv_seg_L1)  # [2,6,128,128]

        output_L1 = self.cls_seg(output, self.conv_seg_L1)  # [2,5,128,128]
        output_L2 = self.cls_seg(output, self.conv_seg_L2)  # [2,10,128,128]
        output_L3 = self.cls_seg(output, self.conv_seg)  # [2,19,128,128]
        
        output_cat = torch.cat([output_L1, output_L2, output_L3],dim=1)  # [2,34,128,128]
        output_list = [output_L1, output_L2, output_L3]
        self.step += 1
        embedding = self.proj_head(inputs[-1]) # 最后的一个特征图 # [2,1024,16,16]->[2,256,16,16]    

        return output_list, embedding

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if isinstance(seg_logits, tuple):
            if len(seg_logits) == 2:
                seg_logits, embedding = seg_logits

                if isinstance(seg_logits, list) and len(seg_logits) == 3:
                    seg_logits = seg_logits
                elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
                    seg_logits = seg_logits

        # seg_logits: [2,65,128,128]
        seg_label = self._stack_batch_gt(batch_data_samples)  # [2,1,512,512]
        loss = dict()
        # 原来是直接把，[2,65,128,128]---双线性插值--->[2,65,512,512]
        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            for i in range(len(seg_logits)):
                seg_logits[i] = resize(
                    input=seg_logits[i],
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = resize(
                    input=seg_logits,
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
            # print('走的这里')

        seg_label = seg_label.squeeze(1)  # [2,1,512,512]-->[2,512,512]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                # pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    self.step,
                    embedding,
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    self.step,
                    embedding,
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # pdb.set_trace()
        # pdb.set_trace()

        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            seg_logits = seg_logits[2]
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = seg_logits[:,-self.num_classes_level_list[-1]:,:,:]

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # pdb.set_trace()
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.  # 缩放！

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if isinstance(seg_logits, tuple):
            if len(seg_logits) == 2:
                seg_logits, embedding = seg_logits

                if isinstance(seg_logits, list) and len(seg_logits) == 3:
                    seg_logits = seg_logits[2]
                    assert seg_logits.shape[1] == self.num_classes_level_list[-1], f'please check seg_logits.shape'
                elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
                    seg_logits = seg_logits[:,-self.num_classes_level_list[-1]:,:,:]
                    assert seg_logits.shape[1] == self.num_classes_level_list[-1], f'please check seg_logits.shape'

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        
        
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits