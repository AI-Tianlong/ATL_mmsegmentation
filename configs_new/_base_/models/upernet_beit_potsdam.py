from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         MultiScaleDeformableAttention)
from mmdet.models.layers import Mask2FormerTransformerDecoder
from mmdet.models.layers.msdeformattn_pixel_decoder import \
    MSDeformAttnPixelDecoder
from mmdet.models.layers.positional_encoding import SinePositionalEncoding
from mmdet.models.layers.transformer.detr_layers import (
    DetrTransformerDecoder, DetrTransformerDecoderLayer,
    DetrTransformerEncoder, DetrTransformerEncoderLayer)
# from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
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
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.necks.featurepyramid import \
    Feature2Pyramid  # A neck structure connect ViT backbone and decoder_heads.
from mmseg.models.necks.multilevel_neck import \
    MultiLevelNeck  # A neck structure connect vit backbone and decoder_heads.
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

num_classes = 6  # loss 要用，也要加

norm_cfg = dict(type=SyncBN, requires_grad=True)

data_preprocessor = dict(
    # type=SegDataPreProcessor,
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    # bgr_to_rgb=True,
    # pad_val=0,
    # seg_pad_val=255
)

model = dict(
    type=EncoderDecoder,
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(type=BEiT, ),
    decode_head=dict(
        type=UPerHead,
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
