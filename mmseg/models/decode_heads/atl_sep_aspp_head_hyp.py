# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer

from mmseg.registry import MODELS
from ..utils import resize
from .aspp_head import ASPPHead, ASPPModule
from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList

import geoopt as gt
from .atl_embedding_space.embedding_space import EmbeddingSpace


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
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
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
class DepthwiseSeparableASPPHead_hyp(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, hyperbolic, **kwargs):
        super().__init__(**kwargs)

        self.hyperbolic = hyperbolic

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
        
        # 改双曲空间
        self.proj_head = ProjectionHead(dim_in=2048, norm_cfg=self.norm_cfg)


    def forward(self, inputs, offsets, normals):
        """Forward function."""

        ball = gt.PoincareBall(c=1.0)  # PoincareBall manifold
        embedding = self.proj_head(inputs[-1])
        embedding = embedding.permute((0,2,3,1))  # [2,256,64,64]--->[2,64,64,256]
        embedding = ball.projx(embedding)         # 将欧几里得空间的embedding投影到PoincareBall上
        embedding = embedding.permute((0,3,1,2))  # [2,64,64,256]--->[2,256,64,64]


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

        if self.hyperbolic:
            output = output.permute((0,2,3,1))  # [2, 128, 128, 512]
            out_proj = ball.projx(output)       # [2, 128, 128, 512]  # 讲output 再次投影到PoincareBall上
            output = self.embedding_space.run_log_torch(out_proj, offsets, normals, 1.0) # 然后是在embedding_space中去运行log函数 [2, 128, 128, 19]
            output = output.permute((0,3,1,2)) # [2,19,128,128]  #计算在流形上
        else:
            output = self.cls_seg(output)


        return output



   

    def init_embedding_space(self, offsets, normals, curvature):
        return EmbeddingSpace(offsets, normals, curvature)

    # forward_train
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType,
             offsets, 
             normals) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        curvature = torch.tensor(1.0)
        self.embedding_space = self.init_embedding_space(offsets, normals, curvature)  #
        seg_logits = self.forward(inputs=inputs, offsets=offsets, normals=normals)  # [2,40,128,128]
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType,
                offsets, 
                normals) -> Tensor:
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
        curvature = torch.tensor(1.0)
        self.embedding_space = self.init_embedding_space(offsets, normals, curvature)  #
        seg_logits = self.forward(inputs, offsets, normals)  # 过decode_head的forward--->[2,40,128,128]

        return self.predict_by_feat(seg_logits, batch_img_metas)   # [2,40,512,512]
