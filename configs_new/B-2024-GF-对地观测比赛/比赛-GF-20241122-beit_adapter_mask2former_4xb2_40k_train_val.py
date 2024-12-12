# Copyright (c) Shanghai AI Lab. All rights reserved.
from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

with read_base():
    from .._base_.datasets.atl_2024_bisai_GF import *
    from .._base_.default_runtime import *
    from .._base_.models.mask2former_beit_potsdam import *
    from .._base_.schedules.schedule_80k import *

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！

# reduce_zero_label = True
num_classes = 9  # loss 要用，也要加 # 加上背景是25类

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误

crop_size = (512, 512)
# pretrained = None
pretrained = None #超微
# pretrained = '/data/AI-Tianlong/Checkpoints/2-对比实验的权重/vit-adapter-offical/beitv2_large_patch16_224_pt1k_ft21k.pth' #浪潮
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        # mean = [1042.4173119787,1518.1018093155,1774.4368727829,3078.1374645515],
        # std = [346.9202500676,456.20526627209,617.83057417844,361.47252556325],
        mean=[995.26933225455, 1452.7270343669, 1638.4348408118, 3150.9832206793],
        std=[317.36181350835, 406.4103774175, 546.77043273976, 501.33003719076],
        # mean=None,
        # std=None,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size))

model.update(
    dict(
        type=EncoderDecoder,
        pretrained=pretrained,
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type=BEiTAdapter,
            img_size=512,
            patch_size=16,
            embed_dim=1024,
            in_channels=4,  # 4个波段
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            init_values=1e-6,
            drop_path_rate=0.3,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            with_cp=False,  # set with_cp=True to save memory
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        ),  #backbone 完全一样
        decode_head=dict(
            in_channels=[1024, 1024, 1024, 1024],
            feat_channels=256,
            out_channels=256,
            num_queries=100,
            num_classes=num_classes,
            loss_cls=dict(
                type=CrossEntropyLoss,
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * num_classes + [0.1]),
        ),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(128, 128))))

# dataset config
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 512) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
train_dataloader.update(
    batch_size=2,
    dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optimizer = dict(
    type=AdamW,
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))

# learning policy

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1000,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]
train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=1000)
load_from = '/data/AI-Tianlong/Checkpoints/1-自己预训练的权重/mmseg1.x_beit_adapter_GIDimg_FiveBillionlabel-80000iter_miou_69_beitv2pr_4channel.pth'
# load_from = None
