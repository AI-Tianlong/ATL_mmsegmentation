import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

MASK_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels_12classes'
RGB_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels_12classes_RGB'

IMG_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/images_3channel'
mkdir_or_exist(RGB_path)

label_lists = find_data_list(MASK_path, suffix='.tif')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别

# 是否包含0类，classes和palette里没包含
reduce_zero_label = True
# 给生成的RGB图像添加空间
add_meta_info = True 

METAINFO = dict(
    classes=("工业区域", "农田","树木",
             "草地", "水域","城市住宅",   
             "农村住宅", "裸地","雪",   
             "人造公园", "道路","火车站机场",   
    ),
    palette=[[200, 0, 0], [150, 250, 0], [0, 200, 0],
             [150, 200, 150], [0, 0, 200], [250, 0, 150],
             [200, 150, 150], [200, 200, 200], [250, 250, 250],
             [150, 150, 0], [250, 150, 0], [250, 200, 250]]
             )  
classes = METAINFO['classes']
palette = METAINFO['palette']

# palette = np.array(palette)

if reduce_zero_label:
    new_palette = [[0, 0, 0]] + palette
    print(f"palette: {new_palette}")
else:
    new_palette = palette
    print(f"palette: {new_palette}")

new_palette = np.array(new_palette)


for mask_label_path in tqdm(label_lists, colour='Green'):
    mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
    h,w = mask_label.shape

    RGB_label = new_palette[mask_label]
    output_path = os.path.join(RGB_path, os.path.basename(mask_label_path))

    driver = gdal.GetDriverByName('GTiff')
    RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

    RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
    RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
    RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

    if add_meta_info:
        IMG_gdal = gdal.Open(os.path.join(IMG_path, os.path.basename(mask_label_path).replace('.tif', '.tif')), gdal.GA_ReadOnly)

        trans = IMG_gdal.GetGeoTransform()
        proj = IMG_gdal.GetProjection()

        RGB_label_gdal.SetGeoTransform(trans)
        RGB_label_gdal.SetProjection(proj)

    RGB_label_gdal = None