import os
from PIL import Image
import numpy as np
from tqdm import trange
from ATL_path import scandir, mkdir_or_exist, find_data_list

print('\n')
print(f'-----------第 1 步  转换训练集标签为mask---------------')

labels100_path = '/home/xiaopengyou1/AITianlong/Datasets/2023-Gaofen/train/labels'
labels_list = find_data_list(labels100_path, suffix='.tif')

labels_mask_path = '/home/xiaopengyou1/AITianlong/Datasets/2023-Gaofen/train/labels_mask'
mkdir_or_exist(labels_mask_path)

class_list = [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 46, 51, 52, 53, 255]

print(f'---start label to mask')

for i in trange(len(labels_list),colour='GREEN',desc=f'---'):

    label = np.array(Image.open(labels_list[i]))

    for j in range(len(class_list)):

        label[np.where(label==class_list[j])] = j

    label = Image.fromarray(label)
    label.save(os.path.join(labels_mask_path, os.path.basename(labels_list[i])))
print(f'---label to mask have done')
print(f'---第 1 步 已完成')

print('\n')
print(f'-----------第 2 步  模型训练---------------')

