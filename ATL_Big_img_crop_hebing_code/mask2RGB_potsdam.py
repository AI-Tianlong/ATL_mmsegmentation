import os

import numpy as np
from ATL_Tools import find_data_list, mkdir_or_exist
from osgeo import gdal
from PIL import Image
from tqdm import tqdm, trange

Image.MAX_IMAGE_PIXELS = None

MASK_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/atl_nangangqu_potsdam'
RGB_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/atl_nangangqu_potsdam_RGB'

IMG_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/南岗区'
mkdir_or_exist(RGB_path)

label_lists = find_data_list(MASK_path, suffix='.png')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别

# 是否包含0类，classes和palette里没包含
reduce_zero_label = True
# 给生成的RGB图像添加空间
add_meta_info = True

METAINFO = dict(
    classes=('impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
             'clutter'),
    palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
             [255, 255, 0], [255, 0, 0]])
classes = METAINFO['classes']
palette = METAINFO['palette']

# palette = np.array(palette)

if reduce_zero_label:
    new_palette = [[0, 0, 0]] + palette
    print(f'palette: {new_palette}')
else:
    new_palette = palette
    print(f'palette: {new_palette}')

new_palette = np.array(new_palette)

for mask_label_path in tqdm(label_lists, colour='Green'):
    mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
    h, w = mask_label.shape

    RGB_label = new_palette[mask_label]
    output_path = os.path.join(
        RGB_path,
        os.path.basename(mask_label_path).replace('.png', '.tif'))

    driver = gdal.GetDriverByName('GTiff')
    RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

    RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:, :, 0])
    RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:, :, 1])
    RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:, :, 2])

    if add_meta_info:
        IMG_gdal = gdal.Open(
            os.path.join(
                IMG_path,
                os.path.basename(mask_label_path).replace('.png', '.tif')),
            gdal.GA_ReadOnly)

        trans = IMG_gdal.GetGeoTransform()
        proj = IMG_gdal.GetProjection()

        RGB_label_gdal.SetGeoTransform(trans)
        RGB_label_gdal.SetProjection(proj)

    RGB_label_gdal = None