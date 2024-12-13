from mmseg.datasets import atl_0_paper_new_5b_GF_Google_S2_19class
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

data_root = 'data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/img_dir/train/Google_5B_19类_size512'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile_with_data_preproocess'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

dataset = atl_0_paper_new_5b_GF_Google_S2_19class(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)