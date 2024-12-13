from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor

from mmseg.models.data_preprocessor import SegDataPreProcessor

from mmseg.evaluation import ATL_IoUMetric #多卡时有问题
from mmseg.models.decode_heads.atl_fcn_head import ATL_FCNHead
from mmseg.models.decode_heads.uper_head import UPerHead

from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_encoder_decoder import ATL_EncoderDecoder
from mmseg.models.segmentors.atl_encoder_decoder_hyp import EncoderDecoder_hyp

from mmseg.models.backbones.resnet import ResNetV1c
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.decode_heads.atl_sep_aspp_head_hyp import DepthwiseSeparableASPPHead_hyp
from mmseg.models.decode_heads.fcn_head import FCNHead
 
from mmseg.models.decode_heads.atl_uper_head import ATL_UPerHead, ATL_UPerHead_fenkai
from mmseg.models.losses.atl_loss import ATL_Loss, S2_5B_Dataset_21Classes_Map_nobackground
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from torch.optim.sgd import SGD
from mmseg.evaluation import IoUMetric

with read_base():
    from ..._base_.datasets.atl_0_paper_5b_s2_19class import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

crop_size = (512, 512)
num_classes = 19

# model settings
norm_cfg = dict(type=SyncBN, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type=EncoderDecoder_hyp,
    data_preprocessor=data_preprocessor,
    pretrained='/opt/AI-Tianlong/0-ATL-paper-work/0-预训练好的权重/2-对比实验的权重/deeplabv3plus/resnet101_v1c-10channel_BGR.pth',
    backbone=dict(
        type=ResNetV1c,
        depth=101,
        in_channels = 10,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type=DepthwiseSeparableASPPHead_hyp,
        hyperbolic = True,
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))



# optimizer
optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)



train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=1000)
default_hooks.update(dict(
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=10)))