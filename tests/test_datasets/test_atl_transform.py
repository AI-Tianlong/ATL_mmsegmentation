# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest import TestCase

import mmcv
import numpy as np
import pytest
from mmengine.registry import init_default_scope
from PIL import Image

from mmseg.datasets.transforms import *  # noqa
from mmseg.datasets.transforms import (LoadBiomedicalData,
                                       LoadBiomedicalImageFromFile,
                                       PhotoMetricDistortion, RandomCrop,
                                       RandomDepthMix)
from mmseg.registry import TRANSFORMS
from osgeo import gdal
from PIL import Image

from mmseg.datasets.transforms import MultiImg_MultiAnn_RandomCrop

from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)


init_default_scope('mmseg')


def test_MultiImg_MultiAnn_RandomCrop():

    with pytest.raises(AssertionError):
        MultiImg_MultiAnn_RandomCrop(crop_size=(-1, 0))
    # transform = dict(type='MultiImg_MultiAnn_RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # TRANSFORMS.build(transform)

    img_path_MSI_3chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/img_dir/train/Google_5B_19类_size512/train/Google_GF2_PMS2__L1A0001537637-MSS2_36_53.tif'
    img_path_MSI_4chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/img_dir/train/GF2_5B_19类_size512/train/GF2_PMS2__L1A0001246645-MSS2_11_6.tif'
    img_path_MSI_10chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/img_dir/train/S2_5B_19类_包含雪_size512/train/S2_SR_2019_GF2_PMS2__L1A0001465929-MSS2_2_0.tif'
    
    seg_map_path_MSI_3chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/ann_dir/train/Google_5B_19类_size512/train/Google_GF2_PMS2__L1A0001537637-MSS2_36_53.tif'
    seg_map_path_MSI_4chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/ann_dir/train/GF2_5B_19类_size512/train/GF2_PMS2__L1A0001246645-MSS2_11_6.tif'
    seg_map_path_MSI_10chan = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-seg-裁切好的训练图像_S2_GF2_Google_size512/ann_dir/train/S2_5B_19类_包含雪_size512/train/S2_SR_2019_GF2_PMS2__L1A0001465929-MSS2_2_0.tif'

    img_MSI_3chan = gdal.Open(img_path_MSI_3chan).ReadAsArray().transpose(1, 2, 0)
    img_MSI_4chan = gdal.Open(img_path_MSI_4chan).ReadAsArray().transpose(1, 2, 0)
    img_MSI_10chan = gdal.Open(img_path_MSI_10chan).ReadAsArray().transpose(1, 2, 0)

    gt_semantic_seg_MSI_3chan = gdal.Open(seg_map_path_MSI_3chan).ReadAsArray()
    gt_semantic_seg_MSI_4chan = gdal.Open(seg_map_path_MSI_3chan).ReadAsArray()
    gt_semantic_seg_MSI_10chan = gdal.Open(seg_map_path_MSI_3chan).ReadAsArray()

    assert img_MSI_3chan is not None and img_MSI_4chan is not None and img_MSI_10chan is not None
    assert gt_semantic_seg_MSI_3chan is not None and gt_semantic_seg_MSI_4chan is not None and gt_semantic_seg_MSI_10chan is not None

    results = dict()
    results['img_MSI_3chan'] = img_MSI_3chan
    results['img_MSI_4chan'] = img_MSI_4chan
    results['img_MSI_10chan'] = img_MSI_10chan
    
    results['gt_semantic_seg_MSI_3chan'] = gt_semantic_seg_MSI_3chan
    results['gt_semantic_seg_MSI_4chan'] = gt_semantic_seg_MSI_4chan
    results['gt_semantic_seg_MSI_10chan'] = gt_semantic_seg_MSI_10chan

    results['seg_fields'] = ['gt_semantic_seg_MSI_3chan', 'gt_semantic_seg_MSI_4chan', 'gt_semantic_seg_MSI_10chan']
    results['img_shape'] = results['img_MSI_4chan'].shape
    results['ori_shape'] = results['img_MSI_4chan']
    results['pad_shape'] = results['img_MSI_4chan']
    results['scale_factor'] = 1.0

    h, w, _ = img_MSI_4chan.shape
    # pipeline = MultiImg_MultiAnn_RandomCrop(crop_size=(512,512), cat_max_ratio=0.75)
    pipeline = RandomChoiceResize(
        scales=[int(x * 0.1 * 512) for x in range(5, 21)],
        resize_type='MultiImg_MultiAnn_ResizeShortestEdge',
        max_size=2048)
    pipline2 = MultiImg_MultiAnn_RandomCrop(crop_size=(512,512), cat_max_ratio=0.75)
    
    results = pipeline(results)
    

    print(f"results[img_shape]{results['img_shape']}") 
    print(f"results[img_MSI_3chan].shape[:2]{results['img_MSI_3chan'].shape[:2]}")
    print(f"results[img_MSI_4chan].shape[:2]{results['img_MSI_4chan'].shape[:2]}")
    print(f"results[img_MSI_10chan].shape[:2]{results['img_MSI_10chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_3chan].shape[:2]{results['gt_semantic_seg_MSI_3chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_4chan].shape[:2]{results['gt_semantic_seg_MSI_4chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_10chan].shape[:2]{results['gt_semantic_seg_MSI_10chan'].shape[:2]}")

    results = pipline2(results)
    
    print(f"results[img_shape]{results['img_shape']}") 
    print(f"results[img_MSI_3chan].shape[:2]{results['img_MSI_3chan'].shape[:2]}")
    print(f"results[img_MSI_4chan].shape[:2]{results['img_MSI_4chan'].shape[:2]}")
    print(f"results[img_MSI_10chan].shape[:2]{results['img_MSI_10chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_3chan].shape[:2]{results['gt_semantic_seg_MSI_3chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_4chan].shape[:2]{results['gt_semantic_seg_MSI_4chan'].shape[:2]}")
    print(f"results[gt_semantic_seg_MSI_10chan].shape[:2]{results['gt_semantic_seg_MSI_10chan'].shape[:2]}")

    # assert results['img_MSI_3chan'].shape[:2] == (h - 20, w - 20)
    # assert results['img_MSI_4chan'].shape[:2] == (h - 20, w - 20)
    # assert results['img_MSI_10chan'].shape[:2] == (h - 20, w - 20)
    # assert results['img_shape'] == (h - 20, w - 20)


    # assert results['gt_semantic_seg_MSI_3chan'].shape[:2] == (h - 20, w - 20)
    # assert results['gt_semantic_seg_MSI_4chan'].shape[:2] == (h - 20, w - 20)
    # assert results['gt_semantic_seg_MSI_10chan'].shape[:2] == (h - 20, w - 20)


# if __name__ == '__main__':
#     test_MultiImg_MultiAnn_RandomCrop()