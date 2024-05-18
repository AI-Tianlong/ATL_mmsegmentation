import os

import numpy as np
import tiffile as tif
from ATL_Tools import find_data_list, mkdir_or_exist
from osgeo import gdal
from PIL import Image
from tqdm import tqdm, trange

Image.MAX_IMAGE_PIXELS = None

IMG_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/images_3channel'
RGB_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels_RGB'

img_lists = find_data_list(IMG_path, suffix='.tif')

for img_path in tqdm(img_lists):
    img_basename = os.path.basename(img_path)

    img_gdal_ori = gdal.Open(img_path, gdal.GA_ReadOnly)
    img_gdal_label = gdal.Open(
        os.path.join(RGB_path,
                     os.path.basename(img_path).replace('.tif', '.png')))

    trans = img_gdal_ori.GetGeoTransform()
    proj = img_gdal_ori.GetProjection()

    img_gdal_label.SetGeoTransform(trans)
    img_gdal_label.SetProjection(proj)

    output_path = img_path
    gdal.Warp(output_path, img_gdal_label)

    img_gdal_ori = None
    img_gdal_label = None
