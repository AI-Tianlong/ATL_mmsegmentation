# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .ade import ADE20KDataset
from .atl_2024_bisai import ATL2024Bisai
from .atl_s2_five_billion import (ATLS2FIveBillionDataset24,
                                    ATLS2FIveBillionDataset5)
from .basesegdataset import BaseCDDataset, BaseSegDataset
from .bdd100k import BDD100KDataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import MultiImageMixDataset
from .decathlon import DecathlonDataset
from .drive import DRIVEDataset
from .dsdl import DSDLSegDataset
from .five_billion_pixels import FiveBillionPixelsDataset
from .GID_25classes import GID25Dataset
from .hrf import HRFDataset
from .hsi_drive import HSIDrive20Dataset
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .levir import LEVIRCDDataset
from .lip import LIPDataset
from .loveda import LoveDADataset
from .mapillary import MapillaryDataset_v1, MapillaryDataset_v2
from .night_driving import NightDrivingDataset
from .nyu import NYUDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .refuge import REFUGEDataset
from .stare import STAREDataset
from .synapse import SynapseDataset

from .atl_2024_JL_bisai_road import ATL2024Bisai_ROAD
from .atl_2024_bisai_GF import ATL2024Bisai_GF

# yapf: disable
from .transforms import (CLAHE, AdjustGamma, Albu, BioMedical3DPad,
                         BioMedical3DRandomCrop, BioMedical3DRandomFlip,
                         BioMedicalGaussianBlur, BioMedicalGaussianNoise,
                         BioMedicalRandomGamma, ConcatCDInput, GenerateEdge,
                         LoadAnnotations, LoadBiomedicalAnnotation,
                         LoadBiomedicalData, LoadBiomedicalImageFromFile,
                         LoadImageFromNDArray, LoadMultipleRSImageFromFile,
                         LoadSingleRSImageFromFile, PackSegInputs,
                         PhotoMetricDistortion, RandomCrop, RandomCutOut,
                         RandomMosaic, RandomRotate, RandomRotFlip, Rerange,
                         ResizeShortestEdge, ResizeToMultiple, RGB2Gray,
                         SegRescale)
from .voc import PascalVOCDataset
from .atl_0_paper_TUM_13classes import ATL_S2_TUM_13class
from .atl_0_paper_5b_s2_22class import ATL_S2_5B_Dataset_22class, ATL_S2_5B_Dataset_19class
from .atl_0_paper_new_5b_GF_Google_S2_19class import ATL_5B_GF_Google_S2_Dataset_19class

# yapf: enable
__all__ = [
    'BaseSegDataset', 'BioMedical3DRandomCrop', 'BioMedical3DRandomFlip',
    'CityscapesDataset', 'PascalVOCDataset', 'ADE20KDataset',
    'PascalContextDataset', 'PascalContextDataset59', 'ChaseDB1Dataset',
    'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'DarkZurichDataset',
    'NightDrivingDataset', 'COCOStuffDataset', 'LoveDADataset',
    'MultiImageMixDataset', 'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset',
    'LoadAnnotations', 'RandomCrop', 'SegRescale', 'PhotoMetricDistortion',
    'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'RandomCutOut', 'RandomMosaic', 'PackSegInputs', 'ResizeToMultiple',
    'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'DecathlonDataset', 'LIPDataset', 'ResizeShortestEdge',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedicalRandomGamma', 'BioMedical3DPad', 'RandomRotFlip',
    'SynapseDataset', 'REFUGEDataset', 'MapillaryDataset_v1',
    'MapillaryDataset_v2', 'Albu', 'LEVIRCDDataset',
    'LoadMultipleRSImageFromFile', 'LoadSingleRSImageFromFile',
    'ConcatCDInput', 'BaseCDDataset', 'DSDLSegDataset', 'BDD100KDataset',
    'NYUDataset', 'FiveBillionPixelsDataset', 'GID25Dataset',
    'ATLS2FIveBillionDataset24','ATLS2FIveBillionDataset5' 'ATL2024Bisai',
    'NYUDataset', 'HSIDrive20Dataset','ATL_S2_TUM_13class',
    'ATL_S2_5B_Dataset_22class', 'ATL_S2_5B_Dataset_19class' , 'ATL_S2_5B_Dataset_24class',
    'ATL2024Bisai_ROAD', 'ATL2024Bisai_GF', 'ATL_5B_GF_Google_S2_Dataset_19class'
    ] # type: ignore
