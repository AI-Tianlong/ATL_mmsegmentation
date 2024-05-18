from argparse import ArgumentParser

from osgeo import gdal

from mmseg.apis import MMSegInferencer

model_cfg = '/opt/AI-Tianlong/openmmlab/mmsegmentation/configs_new/beit_adapter/five_billion_4_channels_beit_adapter_mask2former_2xb2_80k_512.py'
checkpoint_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/five_billion_beit_adapter_mask2former_4xb2_80k_Potsdam_loveda_ft-512x512/iter_80000.pth'

img_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/data/five_billion_pixels/img_8bit_NirRGB/train/GF2_PMS1__L1A0000564539-MSS1_0_0.tif'

image_gdal = gdal.Open(img_path)
img = image_gdal.ReadAsArray()
img = img.transpose((1, 2, 0))

mmseg_inferencer = MMSegInferencer(
    model_cfg,
    checkpoint_path,
    dataset_name='five_billion_pixels',
    device='cuda:0')

mmseg_inferencer(
    img, show=False, out_dir='./atl_out', opacity=0.5, with_labels=False)
