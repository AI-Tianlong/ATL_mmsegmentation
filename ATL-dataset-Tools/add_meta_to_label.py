import numpy as np
from ATL_Tools import find_data_list, mkdir_or_exist
from osgeo import gdal

image_paths = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/images_3channel'
label_paths = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/labels'

img_lists = find_data_list(image_paths, suffix='.tif')
label_lists = find_data_list(label_paths, suffix='.tif')

for image_path, label_path in zip(img_lists, label_lists):
    image = gdal.Open(image_path, gdal.GA_ReadOnly)
    label = gdal.Open(label_path, gdal.GA_ReadOnly)

    img_trans = image.GetGeoTransform()
    img_proj = image.GetProjection()
