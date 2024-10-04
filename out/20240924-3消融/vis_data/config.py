L1_num_classes = 6
L2_num_classes = 12
L3_num_classes = 22
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    mean=None,
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=None,
    type='mmseg.models.data_preprocessor.SegDataPreProcessor')
data_root = 'data/0-atl-paper-s2/0-S2_5B-21类-包含雪21'
dataset_type = 'mmseg.datasets.atl_0_paper_5b_s2_22class.ATL_S2_5B_Dataset_22class'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=4000, type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=50,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(draw=True, type='mmseg.engine.SegVisualizationHook'))
default_scope = 'mmseg'
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
launcher = 'pytorch'
load_from = 'work_dirs/20240920-1消融-去掉loss里面的权重，变成mean/iter_80000_33.92.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4,
            type='mmseg.models.losses.cross_entropy_loss.CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(
            requires_grad=True,
            type='torch.nn.modules.batchnorm.SyncBatchNorm'),
        num_classes=22,
        num_convs=1,
        type='mmseg.models.decode_heads.fcn_head.FCNHead'),
    backbone=dict(
        cffn_ratio=0.25,
        conv_inplane=64,
        deform_num_heads=16,
        deform_ratio=0.5,
        depth=24,
        drop_path_rate=0.3,
        embed_dim=1024,
        img_size=512,
        in_channels=10,
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
        with_cp=False),
    data_preprocessor=dict(
        mean=None,
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=None,
        type='mmseg.models.data_preprocessor.SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        dropout_ratio=0.1,
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
        loss_decode=dict(
            classes_map=
            'mmseg.models.losses.atl_loss.S2_5B_Dataset_22Classes_Map',
            loss_weight=1.0,
            reduction='mean',
            type='mmseg.models.losses.atl_loss.ATL_Loss',
            use_sigmoid=False),
        norm_cfg=dict(
            requires_grad=True,
            type='torch.nn.modules.batchnorm.SyncBatchNorm'),
        num_classes=40,
        num_level_classes=[
            6,
            12,
            22,
        ],
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='mmseg.models.decode_heads.atl_uper_head.ATL_UPerHead'),
    level_classes_map=
    'mmseg.models.losses.atl_loss.S2_5B_Dataset_22Classes_Map',
    pretrained=None,
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    train_cfg=dict(),
    type='mmseg.models.segmentors.atl_encoder_decoder.ATL_EncoderDecoder')
norm_cfg = dict(
    requires_grad=True, type='torch.nn.modules.batchnorm.SyncBatchNorm')
num_classes = 40
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
        data_root='data/0-atl-paper-s2/0-S2_5B-21类-包含雪21',
        pipeline=[
            dict(
                type=
                'mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
            dict(type='mmseg.datasets.transforms.loading.LoadAnnotations'),
            dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
        ],
        type=
        'mmseg.datasets.atl_0_paper_5b_s2_22class.ATL_S2_5B_Dataset_22class'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ],
    keep_results=True,
    type='mmseg.evaluation.ATL_IoUMetric')
test_pipeline = [
    dict(type='mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
    dict(type='mmseg.datasets.transforms.loading.LoadAnnotations'),
    dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
]
train_cfg = dict(
    max_iters=80000,
    type='mmengine.runner.loops.IterBasedTrainLoop',
    val_interval=2000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='data/0-atl-paper-s2/0-S2_5B-21类-包含雪21',
        pipeline=[
            dict(
                type=
                'mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
            dict(type='mmseg.datasets.transforms.LoadAnnotations'),
            dict(
                max_size=2048,
                resize_type='mmseg.datasets.transforms.ResizeShortestEdge',
                scales=[
                    256,
                    307,
                    358,
                    409,
                    460,
                    512,
                    563,
                    614,
                    665,
                    716,
                    768,
                    819,
                    870,
                    921,
                    972,
                    1024,
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
            dict(type='mmseg.datasets.transforms.PackSegInputs'),
        ],
        type=
        'mmseg.datasets.atl_0_paper_5b_s2_22class.ATL_S2_5B_Dataset_22class'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        shuffle=True, type='mmengine.dataset.sampler.InfiniteSampler'))
train_pipeline = [
    dict(type='mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
    dict(type='mmseg.datasets.transforms.LoadAnnotations'),
    dict(
        max_size=2048,
        resize_type='mmseg.datasets.transforms.ResizeShortestEdge',
        scales=[
            256,
            307,
            358,
            409,
            460,
            512,
            563,
            614,
            665,
            716,
            768,
            819,
            870,
            921,
            972,
            1024,
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
    dict(type='mmseg.datasets.transforms.PackSegInputs'),
]
tta_model = dict(type='mmseg.models.SegTTAModel')
tta_pipeline = [
    dict(
        backend_args=None,
        type='mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
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
        data_root='data/0-atl-paper-s2/0-S2_5B-21类-包含雪21',
        pipeline=[
            dict(
                type=
                'mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(
                    512,
                    512,
                ),
                type='mmcv.transforms.processing.Resize'),
            dict(type='mmseg.datasets.transforms.loading.LoadAnnotations'),
            dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
        ],
        type=
        'mmseg.datasets.atl_0_paper_5b_s2_22class.ATL_S2_5B_Dataset_22class'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        shuffle=False, type='mmengine.dataset.sampler.DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='mmseg.evaluation.ATL_IoUMetric')
val_pipeline = [
    dict(type='mmseg.datasets.transforms.loading.LoadSingleRSImageFromFile'),
    dict(
        keep_ratio=True,
        scale=(
            512,
            512,
        ),
        type='mmcv.transforms.processing.Resize'),
    dict(type='mmseg.datasets.transforms.loading.LoadAnnotations'),
    dict(type='mmseg.datasets.transforms.formatting.PackSegInputs'),
]
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='./out/20240924-3消融',
    type='mmseg.visualization.SegLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
work_dir = './work_dirs/20240924-3消融-去掉loss里面的权重，变成mean-仅argmaxL3的特征图'
