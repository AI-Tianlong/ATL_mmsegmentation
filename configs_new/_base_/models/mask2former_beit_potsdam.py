from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         MultiScaleDeformableAttention)
from mmdet.models.layers import Mask2FormerTransformerDecoder
from mmdet.models.layers.msdeformattn_pixel_decoder import \
    MSDeformAttnPixelDecoder
from mmdet.models.layers.positional_encoding import SinePositionalEncoding

from mmdet.models.layers.transformer.detr_layers import (
    DetrTransformerDecoder, DetrTransformerDecoderLayer,
    DetrTransformerEncoder, DetrTransformerEncoderLayer)
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.dice_loss import DiceLoss
from mmdet.models.task_modules.assigners.hungarian_assigner import \
    HungarianAssigner
from mmdet.models.task_modules.assigners.match_cost import (
    ClassificationCost, CrossEntropyLossCost, DiceCost, FocalLossCost)
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from torch.nn.modules.activation import GELU, ReLU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN
from torch.nn.modules.normalization import LayerNorm as LN

from mmseg.models.backbones import BEiT
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


num_classes = 6  # loss 要用，也要加

norm_cfg = dict(type=SyncBN, requires_grad=True)

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type=EncoderDecoder,
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=BEiT,

    ),
    decode_head=dict(
        type=Mask2FormerHead,  # 千万别自己实现，全是坑
        in_channels=[1024, 1024, 1024,
                     1024],  # BEiT-Adapter [1024,1024,1024,1024]

        in_index=[0, 1, 2, 3],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type=
            MSDeformAttnPixelDecoder,  # MSDeformAttnPixelDecoder #用的自己实现的，vit-adapter
            num_outs=
            3,  # mmdet的在mmdet-->models-->layers-->msdeformattn_pixel_decoder.py
            norm_cfg=dict(type=GN, num_groups=32),
            act_cfg=dict(type=ReLU),
            encoder=dict(
                # type=DetrTransformerEncoder,# DetrTransformerEncoder # 用的mmdet实现的
                num_layers=6,
                layer_cfg=dict(
                    # type=DetrTransformerEncoderLayer,# DetrTransformerEncoder绑定了
                    self_attn_cfg=dict(
                        # type=MultiScaleDeformableAttention,  # DetrTransformerEncoderLayer绑定了
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        # type=FFN, #DetrTransformerEncoderLayer绑定了
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        # with_cp=True,
                        act_cfg=dict(type=ReLU, inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                # type=SinePositionalEncoding, # ATL 的 MSDeformAttnPixelDecoder 默认是这个
                num_feats=128,
                normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            #  type=SinePositionalEncoding, # Mask2FormerHead写死了
            num_feats=128,
            normalize=True),
        transformer_decoder=dict(
            # type=DetrTransformerDecoder,  # Mask2FormrtHead--->DetrTransformerDecoder写死了
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                # type=DetrTransformerDecoderLayer,  # DetrTransformerDecoder 写死了
                self_attn_cfg=dict(
                    # type=MultiheadAttention,  # DetrTransformerDecoderLayer 写死了
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type=ReLU, inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=ClassificationCost, weight=2.0),
                    dict(
                        type=CrossEntropyLossCost,
                        weight=5.0,
                        use_sigmoid=True),
                    dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type=MaskPseudoSampler))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
