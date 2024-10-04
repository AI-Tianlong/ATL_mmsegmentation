# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL_S2_TUM_13class(BaseSegDataset):
    """"""
    METAINFO = dict(
        classes=('Background', 'Arable land', 'Permanent crops',
                 'Pastures', 'Forests', 'Surface water', 'Shrub',
                 'Open spaces', 'Wetlands', 'Mine, dump', 'Artificial vegetation',
                 'Urban fabric', 'Bulidings'),
        palette=[[0, 0, 0], [255, 220, 130], [206, 1330, 65],
                 [188, 183, 108], [0, 255, 0], [7, 252, 250], [92, 155, 41],
                 [139, 69, 18], [131, 111, 255], [147, 0, 211], [255, 130, 255],
                 [254, 0, 0], [131, 12, 190]])

    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=False,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
