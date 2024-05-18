
# dataset settings
# from mmseg.datasets.atl_2024_bisai import ATL2024Bisai
dataset_type = 'ATL2024Bisai'
data_root = 'data/atl_2024_Bisai/'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'), # 多通道 不太能用这个
    dict(type='PackSegInputs')
]

val_pipeline = [  #
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [  #
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # dict(type=Resize, scale=(6800, 7200), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type=LoadAnnotations),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_test/images', seg_map_path='train_test/labels'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_test/images', seg_map_path='train_test/labels'),
        pipeline=val_pipeline))
# 想用大图去推理
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        data_prefix=dict(
            img_path='/opt/AI-Tianlong/openmmlab/mmsegmentation/configs_new/Juesai/data/images', seg_map_path='train_set/Label_2'),
        # ann_file='/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/5billion-S2/Big-image/code/val_list.txt',
        pipeline=test_pipeline))

val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    keep_results=True)
