from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmseg.datasets.atl_2024_JL_bisai_road import ATL2024Bisai_ROAD
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)
from mmseg.evaluation import IoUMetric

# dataset settings
dataset_type = ATL2024Bisai_ROAD
data_root = 'data/算法赛道1高分辨率遥感数据道路提取初赛数据集/数据集'

crop_size = (512, 512)
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomResize,
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion), # 多通道 不太能用这个
    dict(type=PackSegInputs)
]

val_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    dict(type=Resize, scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]

test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    # dict(type=Resize, scale=(512, 512), keep_ratio=True),
    # dict(type=Resize, scale=(6800, 7200), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]

img_ratios = [0.5, 1.0,1.5]
tta_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(
        type=TestTimeAug,
        transforms=[[
            dict(type=Resize, scale_factor=r, keep_ratio=True)
            # dict(type=Resize, scale_factor=0.75, keep_ratio=True)
            # dict(type=Resize, scale_factor=1.0, keep_ratio=True)
            # dict(type=Resize, scale_factor=1.25, keep_ratio=True)
            for r in img_ratios
        ],
                    [
                        dict(type=RandomFlip, prob=0., direction='horizontal'),
                        dict(type=RandomFlip, prob=1., direction='horizontal')
                    ], # [dict(type=LoadAnnotations)],
                    [dict(type=PackSegInputs)]])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_val/img', seg_map_path='train_val/label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_val/img', seg_map_path='train_val/label'),
        pipeline=val_pipeline))
# 想用大图去推理
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    keep_results=True)
