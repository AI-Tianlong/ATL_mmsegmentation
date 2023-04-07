# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATL_ISAR_Seg(BaseSegDataset):

    METAINFO = dict(
        classes=('Background', 'Empennage', 'Engine', 
                 'Fuselage', 'Head', 'Wing'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], 
                 [128, 128, 0], [0, 0, 128], [128, 0, 128]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


