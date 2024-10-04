# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL_S2_Crop10m_Dataset_4class(BaseSegDataset):
    """"""
    METAINFO = dict(
        classes=('Others-land', 'Rice', 'Corn', 'soybean'),
        palette=[[190, 190, 190], [0, 200, 250], [250, 200, 0],
                 [150, 150, 250]])

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
