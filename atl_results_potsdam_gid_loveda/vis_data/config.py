crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='mmseg.models.data_preprocessor.SegDataPreProcessor')
data_root = 'data/loveDA'
dataset_type = 'mmseg.datasets.loveda.LoveDADataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=8000, type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=50,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(draw=True, type='mmseg.engine.SegVisualizationHook'))
default_scope = None
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = 'checkpoints/atl_beit_adapter_checkpoints/mmseg1.x-beit-adapter-loveda-potsdamft-iter-64000-miou_55.97.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        cffn_ratio=0.25,
        conv_inplane=64,
        deform_num_heads=16,
        deform_ratio=0.5,
        depth=24,
        drop_path_rate=0.3,
        embed_dim=1024,
        img_size=512,
        in_channels=3,
        init_values=1e-06,
        interaction_indexes=[
            [
                0,
                5,
            ],
            [
                6,
                11,
            ],
            [
                12,
                17,
            ],
            [
                18,
                23,
            ],
        ],
        mlp_ratio=4,
        n_points=4,
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        type='mmseg.models.backbones.BEiTAdapter',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        with_cp=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmseg.models.data_preprocessor.SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.models.losses.cross_entropy_loss.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.models.losses.dice_loss.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.models.losses.cross_entropy_loss.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=7,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='torch.nn.modules.activation.ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(
                            inplace=True,
                            type='torch.nn.modules.activation.ReLU'),
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(
                num_groups=32,
                type='torch.nn.modules.normalization.GroupNorm'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type=
            'mmdet.models.layers.msdeformattn_pixel_decoder.MSDeformAttnPixelDecoder'
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(
                        type=
                        'mmdet.models.task_modules.assigners.match_cost.ClassificationCost',
                        weight=2.0),
                    dict(
                        type=
                        'mmdet.models.task_modules.assigners.match_cost.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type=
                        'mmdet.models.task_modules.assigners.match_cost.DiceCost',
                        weight=5.0),
                ],
                type=
                'mmdet.models.task_modules.assigners.hungarian_assigner.HungarianAssigner'
            ),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(
                type='mmdet.models.task_modules.samplers.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(
                        inplace=True, type='torch.nn.modules.activation.ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='mmseg.models.decode_heads.mask2former_head.Mask2FormerHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    train_cfg=dict(),
    type='mmseg.models.segmentors.encoder_decoder.EncoderDecoder')
norm_cfg = dict(
    requires_grad=True, type='torch.nn.modules.batchnorm.SyncBatchNorm')
num_classes = 7
optim_wrapper = dict(
    constructor='mmseg.engine.optimizers.LayerDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=2e-05,
        type='torch.optim.AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.9, num_layers=24),
    type='mmengine.optim.optimizer.OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=2e-05,
    type='torch.optim.AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=1500,
        start_factor=1e-06,
        type='mmengine.optim.scheduler.lr_scheduler.LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=80000,
        eta_min=0.0,
        power=1.0,
        type='mmengine.optim.scheduler.lr_scheduler.PolyLR'),
]
pretrained = None
resume = False
test_cfg = dict(type='mmengine.runner.loops.TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='data/loveDA',
        pipeline=[
            dict(type='mmcv.transforms.loading.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1024,
                    1024,
                ),
                type='mmcv.transforms.processing.Resize'),
            dict(
                reduce_zero_label=True,
                type='mmseg.datasets.transforms.loading.LoadAnnotations'),
            dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
        ],
        type='mmseg.datasets.loveda.LoveDADataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='mmseg.evaluation.IoUMetric')
test_pipeline = [
    dict(type='mmcv.transforms.loading.LoadImageFromFile'),
    dict(
        keep_ratio=True,
        scale=(
            1024,
            1024,
        ),
        type='mmcv.transforms.processing.Resize'),
    dict(
        reduce_zero_label=True,
        type='mmseg.datasets.transforms.loading.LoadAnnotations'),
    dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
]
train_cfg = dict(
    max_iters=80000,
    type='mmengine.runner.loops.IterBasedTrainLoop',
    val_interval=8000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='data/loveDA',
        pipeline=[
            dict(type='mmcv.transforms.LoadImageFromFile'),
            dict(
                reduce_zero_label=True,
                type='mmseg.datasets.transforms.LoadAnnotations'),
            dict(
                max_size=2048,
                resize_type='mmseg.datasets.transforms.ResizeShortestEdge',
                scales=[
                    320,
                    384,
                    448,
                    512,
                    576,
                    640,
                    704,
                    768,
                    832,
                    896,
                    960,
                    1024,
                    1088,
                    1152,
                    1216,
                    1280,
                ],
                type='mmcv.transforms.RandomChoiceResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    512,
                    512,
                ),
                type='mmseg.datasets.transforms.RandomCrop'),
            dict(prob=0.5, type='mmcv.transforms.RandomFlip'),
            dict(type='mmseg.datasets.transforms.PhotoMetricDistortion'),
            dict(type='mmseg.datasets.transforms.PackSegInputs'),
        ],
        type='mmseg.datasets.loveda.LoveDADataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(
        shuffle=True, type='mmengine.dataset.sampler.InfiniteSampler'))
train_pipeline = [
    dict(type='mmcv.transforms.LoadImageFromFile'),
    dict(
        reduce_zero_label=True,
        type='mmseg.datasets.transforms.LoadAnnotations'),
    dict(
        max_size=2048,
        resize_type='mmseg.datasets.transforms.ResizeShortestEdge',
        scales=[
            320,
            384,
            448,
            512,
            576,
            640,
            704,
            768,
            832,
            896,
            960,
            1024,
            1088,
            1152,
            1216,
            1280,
        ],
        type='mmcv.transforms.RandomChoiceResize'),
    dict(
        cat_max_ratio=0.75,
        crop_size=(
            512,
            512,
        ),
        type='mmseg.datasets.transforms.RandomCrop'),
    dict(prob=0.5, type='mmcv.transforms.RandomFlip'),
    dict(type='mmseg.datasets.transforms.PhotoMetricDistortion'),
    dict(type='mmseg.datasets.transforms.PackSegInputs'),
]
tta_model = dict(type='mmseg.models.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='mmcv.transforms.loading.LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scale_factor=0.5,
                    type='mmcv.transforms.processing.Resize'),
                dict(
                    keep_ratio=True,
                    scale_factor=0.75,
                    type='mmcv.transforms.processing.Resize'),
                dict(
                    keep_ratio=True,
                    scale_factor=1.0,
                    type='mmcv.transforms.processing.Resize'),
                dict(
                    keep_ratio=True,
                    scale_factor=1.25,
                    type='mmcv.transforms.processing.Resize'),
                dict(
                    keep_ratio=True,
                    scale_factor=1.5,
                    type='mmcv.transforms.processing.Resize'),
                dict(
                    keep_ratio=True,
                    scale_factor=1.75,
                    type='mmcv.transforms.processing.Resize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='mmcv.transforms.processing.RandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='mmcv.transforms.processing.RandomFlip'),
            ],
            [
                dict(type='mmseg.datasets.transforms.loading.LoadAnnotations'),
            ],
            [
                dict(
                    type='mmseg.datasets.transforms.formatting.PackSegInputs'),
            ],
        ],
        type='mmcv.transforms.processing.TestTimeAug'),
]
val_cfg = dict(type='mmengine.runner.loops.ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='data/loveDA',
        pipeline=[
            dict(type='mmcv.transforms.loading.LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    1024,
                    1024,
                ),
                type='mmcv.transforms.processing.Resize'),
            dict(
                reduce_zero_label=True,
                type='mmseg.datasets.transforms.loading.LoadAnnotations'),
            dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
        ],
        type='mmseg.datasets.loveda.LoveDADataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='mmseg.evaluation.IoUMetric')
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='./atl_vis_loveda3',
    type='mmseg.visualization.SegLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
work_dir = './work_dirs/loveda_beit_adapter_mask2former_4xb2_80k_Potsdam_ft-512x512'
