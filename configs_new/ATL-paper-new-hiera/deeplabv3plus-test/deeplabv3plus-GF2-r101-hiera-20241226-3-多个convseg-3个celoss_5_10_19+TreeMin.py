# 2024-09-02 测试, 可以跑通,loss-从5开始降低。

from mmcv.transforms import RandomChoiceResize, RandomFlip
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,RandomCrop,
                                       ResizeShortestEdge)
from mmseg.models.decode_heads.uper_head import UPerHead


# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
# DecodeHead
from mmseg.models.decode_heads.atl_hiera_37_sep_aspp_head_multi_convseg import ATL_Hiera_DepthwiseSeparableASPPHead_Multi_convseg
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.decode_heads.atl_hiera_37_sep_aspp_head import ATL_Hiera_DepthwiseSeparableASPPHead
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# Evaluation
from mmseg.evaluation import IoUMetric

with read_base():
    from ..._base_.datasets.a_atl_0_paper_5b_GF2_19class import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

find_unuser_parameters = True

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 5  # number of L1 Level label   # 5
L2_num_classes = 10  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 19  # number of L1 Level label  # 21

# 总的类别数，包括背景，L1+L2+L3级标签数

num_classes = L1_num_classes + L2_num_classes + L3_num_classes # 37 

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (512, 512)
pretrained = 'checkpoints/2-对比实验的权重/deeplabv3plus/resnet101_v1c-4channel_BGR.pth'


# model settings
norm_cfg = dict(type=SyncBN, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=ATL_Hiera_EncoderDecoder,
    # type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type=ResNetV1c,
        depth=101,
        in_channels = 4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type=ATL_Hiera_DepthwiseSeparableASPPHead_Multi_convseg,
        merge_hiera=False,
        # type=DepthwiseSeparableASPPHead,
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes_level_list=[5,10,19],
        # num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=ATL_Hiera_Loss_convseg, num_classes=[5,10,19], loss_weight=1.0)),
        # loss_decode=dict(
        #     type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=L3_num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



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

train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks.update(
    dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=4000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook)))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)