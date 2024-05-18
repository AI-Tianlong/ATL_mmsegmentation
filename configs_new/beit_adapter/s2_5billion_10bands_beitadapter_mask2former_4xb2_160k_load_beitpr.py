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
    from .._base_.datasets.atl_s2_five_billion import *
    from .._base_.default_runtime import *
    from .._base_.models.mask2former_beit_potsdam import *
    from .._base_.schedules.schedule_160k import *

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！

# reduce_zero_label = True
num_classes = 25  # loss 要用，也要加 # 加上背景是25类

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误

crop_size = (512, 512)
# pretrained = None
pretrained = '/opt/AI-Tianlong/checkpoints/vit-adapter-offical/beitv2_large_patch16_224_pt1k_ft21k.pth'
# pretrained = None
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        #       B2       B3      B4      B5      B6      B7     B8       B8A     B11    B12
        # mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # std= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # bgr_to_rgb=True,
        mean=None,
        std=None,
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
            in_channels=10,  # 4个波段
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
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))))

# dataset config
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
train_dataloader.update(dataset=dict(pipeline=train_pipeline))  # potsdam的变量

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
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

load_from = None
# load_from = None
