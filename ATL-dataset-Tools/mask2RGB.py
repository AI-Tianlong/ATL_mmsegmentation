import os

import numpy as np
from ATL_Tools import find_data_list, mkdir_or_exist
from PIL import Image
from tqdm import tqdm, trange

MASK_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/atl_gid_big_intetference'
RGB_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/atl_gid_big_intetference_vis'
mkdir_or_exist(RGB_path)

label_lists = find_data_list(MASK_path, suffix='.png')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别

# 是否包含0类，classes和palette里没包含
reduce_zero_label = False

METAINFO = dict(
    classes=('unlabeled', 'industrial area', 'paddy field', 'irrigated field',
             'dry cropland', 'garden land', 'arbor forest', 'shrub forest',
             'park', 'natural meadow', 'artificial meadow', 'river',
             'urban residential', 'lake', 'pond', 'fish pond', 'snow',
             'bareland', 'rural residential', 'stadium', 'square', 'road',
             'overpass', 'railway station', 'airport'),
    palette=[[0, 0, 0], [200, 0, 0], [0, 200, 0], [150, 250,
                                                   0], [150, 200, 150],
             [200, 0, 200], [150, 0, 250], [150, 150, 250], [200, 150, 200],
             [250, 200, 0], [200, 200, 0], [0, 0, 200], [250, 0, 150],
             [0, 150, 200], [0, 200, 250], [150, 200, 250], [250, 250, 250],
             [200, 200, 200], [200, 150, 150], [250, 200, 150], [150, 150, 0],
             [250, 150, 150], [250, 150, 0], [250, 200, 250], [200, 150, 0]])

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
for mask_label_path in tqdm(label_lists):
    mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
    # print(mask_label.shape)
    RGB_label = new_palette[mask_label]
    Image.fromarray(RGB_label.astype(np.uint8)).save(
        os.path.join(RGB_path,
                     os.path.basename(mask_label_path).replace('.png',
                                                               '.png')))
