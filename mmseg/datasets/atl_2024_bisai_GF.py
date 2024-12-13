# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL2024Bisai_GF(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('其他', '小麦','玉米','向日葵','西瓜','西红柿','甜菜','葱','西葫芦'), 
        palette=[[255,255,255], [0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], 
                 [250, 200, 0  ],[200, 200, 0  ], [0,   0,   200], [0,   150, 200], 
                 [150, 200, 250]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
