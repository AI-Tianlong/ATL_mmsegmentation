# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    './mask2former_beit_potsdam.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
pretrained = None
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BEiTAdapter',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        in_channels=3,
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
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),  #backbone 完全一样
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        feat_channels=256,
        out_channels=256,
        num_queries=100,
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))

# learning policy

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

load_from = '/opt/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/beit_adapter_mask2former_potsdam/mmseg1.x-beitv2_adapter_potsdam_iter_80000_mIoU80.57.pth'
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)
