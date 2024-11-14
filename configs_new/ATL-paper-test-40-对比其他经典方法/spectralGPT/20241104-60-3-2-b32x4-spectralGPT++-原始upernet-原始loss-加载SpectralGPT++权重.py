# 2024-09-02 测试, 可以跑通,loss-从5开始降低。

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
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile, LoadSingleRSImageFromFile_spectral_GPT
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor

from mmseg.evaluation import ATL_IoUMetric #多卡时有问题
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.decode_heads.atl_fcn_head import ATL_FCNHead
from mmseg.models.decode_heads.uper_head import UPerHead


from mmseg.models.segmentors.atl_encoder_decoder import ATL_EncoderDecoder
from mmseg.models.backbones import ViTAdapter
from mmseg.models.backbones.atl_vit_adapter_spectralgpt import ViTAdapter_SpectralGPT
from mmseg.models.backbones.atl_spectral_gpt_utils import VisionTransformer

from mmseg.models.decode_heads.atl_uper_head import ATL_UPerHead, ATL_UPerHead_fenkai
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses.atl_loss import ATL_Loss, ATL_Loss2, S2_5B_Dataset_21Classes_Map_nobackground
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.evaluation import IoUMetric
from functools import partial
import torch.nn as nn

with read_base():
    from ..._base_.datasets.atl_0_paper_5b_s2_19class_128 import *
    from ..._base_.default_runtime import *
    from ..._base_.models.upernet_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

find_unuser_parameters = True

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 5  # number of L1 Level label   # 5
L2_num_classes = 10  # number of L1 Level label  # 11 5+11+21=37类
L3_num_classes = 19  # number of L1 Level label  # 21

# 总的类别数，包括背景，L1+L2+L3级标签数

num_classes = L1_num_classes + L2_num_classes + L3_num_classes # 37

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (128, 128)
pretrained = '/data/AI-Tianlong/Checkpoints/2-对比实验的权重/spectral-GPT的权重/SpectralGPT+.pth'
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        mean=None,
        std=None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size))

model.update(
    dict(
        type=EncoderDecoder,
        # level_classes_map=S2_5B_Dataset_21Classes_Map_nobackground,  # 注意传参！！
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type=VisionTransformer,
            img_size=128,
            in_chans=1,
            patch_size=8,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            num_frames=12,
            t_patch_size=3,
            norm_cfg=dict(type='LN', eps=1e-6),
            out_indices=(2, 5, 8, 11),  # 输出第2个stage的，第5个stage的，第8个stage的，第11个stage的
            init_cfg=dict(type='Pretrained', checkpoint=pretrained) # 不加预训练权重
            # frozen_exclude=None,
        ),
        neck = dict(
            type=MultiLevelNeck,
            in_channels = [768, 768, 768, 768],
            out_channels = 768,
            scales = [4, 2, 1, 0.5]),
        decode_head=dict(
            type=UPerHead,
            in_channels=[768, 768, 768, 768],  # 和vit的结构保持一致，large的话1024
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,   # 这是个 啥参数来着？
            dropout_ratio=0.1,
            num_classes=L3_num_classes, #37
            # num_level_classes=[L1_num_classes, L2_num_classes, L3_num_classes],  # 这里需要和loss的map对应上
            norm_cfg=norm_cfg,
            align_corners=False,
            # loss_decode=dict(
            #     type=ATL_Loss,
            #     use_sigmoid=False,
            #     loss_weight=1.0,
            #     classes_map=S2_5B_Dataset_21Classes_Map_nobackground)),
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
        auxiliary_head=dict(
            type=FCNHead,
            in_channels=768, # 和上面的768 保持统一
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=L3_num_classes, #21
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(64, 64))))# 在512的图上推理


# dataset config
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile_spectral_GPT),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 128) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=512),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
train_dataloader.update(dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# learning policy
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        # begin=0,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

load_from = None
default_hooks.update(
    dict(logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
