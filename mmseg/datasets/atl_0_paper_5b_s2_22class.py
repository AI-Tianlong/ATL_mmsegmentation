# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL_S2_5B_Dataset_22class(BaseSegDataset):
    """"""
    METAINFO = dict(
        classes=('Other land', 'Paddy field', 'Irrigated field',
                 'Dry cropland', 'Garden land', 'Forest', 'Natural meadow',
                 'Artificial meadow', 'River', 'Lake', 'Pond',
                 'Factory-Storage-Shopping malls', 'Urban residential',
                 'Rural residential', 'Stadium', 'Park Square', 'Road',
                 'Overpass', 'Railway station', 'Airport', 'Bare land',
                 'Glaciers Snow'),
        palette=[[190, 190, 190], [161, 243, 161], [198, 224, 180],
                 [169, 208, 142], [142, 169, 219], [0, 176,
                                                    80], [240, 238, 146],
                 [217, 206, 63], [0, 51, 204], [0, 102, 255], [87, 171, 255],
                 [212, 30, 26], [250, 0, 150], [209, 75, 187], [138, 151, 63],
                 [0, 255, 0], [250, 150, 155], [250, 150, 0], [250, 200, 250],
                 [200, 150, 0], [198, 89, 17], [255, 255, 255]])

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
