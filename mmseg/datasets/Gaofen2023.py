# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Gaofen2023(BaseSegDataset):
    """Mapillary Vistas Dataset.

    Dataset paper link:
    http://ieeexplore.ieee.org/document/8237796/

    v1.2 contain 66 object classes.
    (37 instance-specific)

    v2.0 contain 124 object classes.
    (70 instance-specific, 46 stuff, 8 void or crowd).

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Mapillary Vistas Dataset.
    """
    METAINFO = dict(
        classes=('ShuiTian', 'HanDi','QiTaJianSheYongDi',
                 'YoulinDi','GuanMuLin','ShuLinDi','QiTaLinDi',
                 'GaoFugaiCao','ZhongFugaiCaoDi','DiFuGaiCaoDi',
                 'HeQu','ShuiKu','TanDi','Chengzhenyongdi','nongcunjumindain',
                 'hupo'

        ),
        palette=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                 [180, 165, 180], [90, 120, 150], [102, 102, 156],
                 [128, 64, 255], [140, 140, 200], [170, 170, 170],
                 [250, 170, 160], [96, 96, 96],
                 [230, 150, 140], [128, 64, 128], [110, 110, 110],
                 [244, 35, 232]])

    def __init__(self,
                 img_suffix='_image.tif',
                 seg_map_suffix='_label.tif',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, ignore_index=ignore_index, **kwargs)

