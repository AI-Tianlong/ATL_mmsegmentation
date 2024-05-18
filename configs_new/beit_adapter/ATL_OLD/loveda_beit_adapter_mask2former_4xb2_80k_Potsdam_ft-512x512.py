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
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

with read_base():
    from .._base_.datasets.loveda import *
    from .._base_.default_runtime import *
    from .._base_.models.mask2former_beit_potsdam import *
    from .._base_.schedules.schedule_80k import *

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！

num_classes = 7  # loss 要用，也要加

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
"""
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
"""

crop_size = (512, 512)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
pretrained = None
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
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
            in_channels=3,  # 4个波段
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
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, reduce_zero_label=True),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion),
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

load_from = '/opt/AI-Tianlong/checkpoints/atl_beit_adapter_checkpoints/mmseg1.x-beit-adapter-loveda-potsdamft-iter-64000-miou_55.97.pth'
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)
