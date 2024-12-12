from mmengine.registry import init_default_scope
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmseg.datasets.atl_0_paper_new_5b_GF_Google_S2_19class import ATL_5B_GF_Google_S2_Dataset_19class
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)


from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile_with_data_preproocess



init_default_scope('mmseg')

data_root = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512'
data_prefix=dict(
    img_path_MSI_4chan='img_dir/train/GF2_5B_19类_size512',         # 4chan GF2
    img_path_MSI_10chan='img_dir/train/S2_5B_19类_包含雪_size512',   # 10chan S2
    seg_map_path_MSI_4chan='ann_dir/train/GF2_5B_19类_size512',     # 4chan
    seg_map_path_MSI_10chan='ann_dir/train/GF2_5B_19类_size512')    # 10chan



train_pipeline = [
    dict(type=LoadSingleRSImageFromFile_with_data_preproocess),
    dict(type=LoadAnnotations),
    # dict(
    #     type=RandomResize,
    #     scale=crop_size,
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type=RandomFlip, prob=0.5),
    # # dict(type=PhotoMetricDistortion), # 多通道 不太能用这个, 就全不用了
    dict(type=PackSegInputs)
]

dataset = ATL_5B_GF_Google_S2_Dataset_19class(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)