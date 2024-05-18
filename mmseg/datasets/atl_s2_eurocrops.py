# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATLS2EuroCrops(BaseSegDataset):

    METAINFO = dict(
        classes=('common_wheat_and_spelt', 'durum_wheat', 'rye', 'barley', 'oats',
                 'grain_maize', 'rice', 'triticale', 'millet', 'dried_pulses_and_protein_crops',
                 'potatoes', 'industrial_crops', 'tobacco', 'hops', 'cotton', 'rape_and_turnip_rape',
                 'sunflower', 'industrial_nonfood_crops', 'flax', 'flax_linseed_oil', 'oilseed_crops',
                 'hemp', 'fibre_crops', 'aromatic_plants_medicinal_culinary_plants',
                 'fresh_vegetables', 'flowers_ornamental_plants', 'plants_harvested_green',
                 'temporary_grass', 'arable_land_seed_and_seedlings', 'fallow_land_not_crop',
                 'kitchen_gardens', 'cucurbits', 'fodder_roots_brassicas', 'peasture_meadow',
                 'permanent_crops', 'mushrooms_energy_crops_genetically_modified_crops',
                 'energy_crops', 'genetically_modified_crops', 'greenhouse_under_foil',
                 'orchards_fruits', 'fruit_of_temperate_climate_zones', 'berry_species',
                 'nuts', 'citrus_plantations', 'olive_plantations', 'olives_for_oil_production',
                 'table_olives', 'vineyards', 'nursereis', 'arable_crops', 'cereals', 'background'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)
