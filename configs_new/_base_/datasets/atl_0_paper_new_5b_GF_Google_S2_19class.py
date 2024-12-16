from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler


from mmseg.datasets.transforms.formatting import PackSegInputs, ATL_3_embedding_PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)
from mmseg.evaluation import IoUMetric

from mmseg.datasets.transforms.loading import (LoadSingleRSImageFromFile,
                                               ATL_multi_embedding_LoadAnnotations,
                                               LoadSingleRSImageFromFile_with_data_preproocess,
                                               LoadMultiRSImageFromFile_with_data_preproocess)

from mmseg.datasets.atl_0_paper_new_5b_GF_Google_S2_19class import (ATL_5B_GF_Google_S2_Dataset_19class_train, 
                                                                    ATL_5B_GF_Google_S2_Dataset_19class_test)


from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmseg.datasets.transforms import (RandomCrop, ResizeShortestEdge)

# dataset settings
dataset_type_train = ATL_5B_GF_Google_S2_Dataset_19class_train
dataset_type_test = ATL_5B_GF_Google_S2_Dataset_19class_test

data_root = 'data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512'

crop_size = (512, 512)
albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]


train_pipeline = [
    dict(type=LoadMultiRSImageFromFile_with_data_preproocess),
    dict(type=ATL_multi_embedding_LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 512) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # # dict(type=PhotoMetricDistortion), # 多通道 不太能用这个, 就全不用了
    dict(type=ATL_3_embedding_PackSegInputs)
]

val_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile_with_data_preproocess),
    dict(type=Resize, scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]

test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile_with_data_preproocess),
    # dict(type=Resize, scale=(512, 512), keep_ratio=True),   # 不 Resize 按原图尺寸推理
    # dict(type=Resize, scale=(6800, 7200), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),  # 不需要验证，不用添加 Annotations
    dict(type=PackSegInputs)
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type=LoadSingleRSImageFromFile_with_data_preproocess, backend_args=None),
    dict(
        type=TestTimeAug,
        transforms=[[
            dict(type=Resize, scale_factor=r, keep_ratio=True)
            for r in img_ratios
        ],
                    [
                        dict(type=RandomFlip, prob=0., direction='horizontal'),
                        dict(type=RandomFlip, prob=1., direction='horizontal')
                    ], [dict(type=ATL_multi_embedding_LoadAnnotations)],
                    [dict(type=PackSegInputs)]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/train/Google_5B_19类_size512', seg_map_path='ann_dir/train/Google_5B_19类_size512'), # 3chan
            img_path_MSI_3chan='img_dir/train/Google_5B_19类_size512',
            img_path_MSI_4chan='img_dir/train/GF2_5B_19类_size512',         # 4chan GF2
            img_path_MSI_10chan='img_dir/train/S2_5B_19类_包含雪_size512',   # 10chan S2
            
            seg_map_path_MSI_3chan='ann_dir/train/Google_5B_19类_size512',
            seg_map_path_MSI_4chan='ann_dir/train/GF2_5B_19类_size512',     # 4chan
            seg_map_path_MSI_10chan='ann_dir/train/S2_5B_19类_包含雪_size512'),    # 10chan
            
            # seg_map_path_MSI_10chan='ann_dir/train/S2_5B_19类_包含雪_size512'),    # 10chan
            # img_path='img_dir/train/S2_5B_19类_包含雪_size512', seg_map_path='ann_dir/train/S2_5B_19类_包含雪_size512'), # 10chan
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type_test,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val/GF2_5B_19类_size512',         # 4chan GF2
            seg_map_path='ann_dir/val/GF2_5B_19类_size512'),    # 10chan
        pipeline=val_pipeline))
# 想用大图去推理
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type_test,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/train/Google_5B_19类_size512', seg_map_path='ann_dir/train/Google_5B_19类_size512'), # 3chan
            img_path='img_dir/val/GF2_5B_19类_size512',         # 4chan GF2
            seg_map_path='ann_dir/val/GF2_5B_19类_size512'),    # 10chan
        pipeline=test_pipeline))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
