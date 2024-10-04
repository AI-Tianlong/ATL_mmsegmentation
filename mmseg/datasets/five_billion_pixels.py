# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FiveBillionPixelsDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('unlabeled', 'industrial area', 'paddy field',
                 'irrigated field', 'dry cropland', 'garden land',
                 'arbor forest', 'shrub forest', 'park', 'natural meadow',
                 'artificial meadow', 'river', 'urban residential', 'lake',
                 'pond', 'fish pond', 'snow', 'bareland', 'rural residential',
                 'stadium', 'square', 'road', 'overpass', 'railway station',
                 'airport'),
        palette=[[191, 191, 191], [161, 243, 161], [198, 224, 180],
                 [169, 208, 142], [142, 169, 219], [0, 176,
                                                    80], [240, 238, 146],
                 [217, 206, 63], [0, 51, 204], [0, 102, 255], [87, 171, 255],
                 [212, 30, 26], [250, 0, 150], [209, 75, 187], [138, 151, 63],
                 [0, 255, 0], [250, 150, 155], [250, 150, 0], [250, 200, 250],
                 [200, 150, 0], [198, 89, 17], [255, 255, 255]])

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
