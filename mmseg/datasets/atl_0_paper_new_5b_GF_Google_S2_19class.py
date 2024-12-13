# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import Base_3embedding_Dataset
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL_5B_GF_Google_S2_Dataset_19class_train(Base_3embedding_Dataset):
    """"""
    METAINFO = dict(
        classes=('Paddy field', 'Other Field', 'Forest', 'Natural meadow',
                 'Artificial meadow', 'River', 'Lake', 'Pond',
                 'Factory-Storage-Shopping malls', 'Urban residential',
                 'Rural residential', 'Stadium', 'Park Square', 'Road',
                 'Overpass', 'Railway station', 'Airport', 'Bare land',
                 'Glaciers Snow'),
        palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                 [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                 [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                 [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                 [200, 150, 0  ], [200, 100, 50 ], [255, 255, 255]])       

    def __init__(
        self,
        img_suffix_MSI_4chan='.tif',
        img_suffix_MSI_10chan='.tif',
        seg_map_suffix_MSI_4chan='.tif',
        seg_map_suffix_MSI_10chan='.tif',
        reduce_zero_label=True,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix_MSI_4chan=img_suffix_MSI_4chan,
            img_suffix_MSI_10chan=img_suffix_MSI_10chan,
            seg_map_suffix_MSI_4chan=seg_map_suffix_MSI_4chan,
            seg_map_suffix_MSI_10chan=seg_map_suffix_MSI_10chan,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class ATL_5B_GF_Google_S2_Dataset_19class_test(BaseSegDataset):
    """"""
    METAINFO = dict(
        classes=('Paddy field', 'Other Field', 'Forest', 'Natural meadow',
                 'Artificial meadow', 'River', 'Lake', 'Pond',
                 'Factory-Storage-Shopping malls', 'Urban residential',
                 'Rural residential', 'Stadium', 'Park Square', 'Road',
                 'Overpass', 'Railway station', 'Airport', 'Bare land',
                 'Glaciers Snow'),
        palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                 [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                 [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                 [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                 [200, 150, 0  ], [200, 100, 50 ], [255, 255, 255]])       

    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
