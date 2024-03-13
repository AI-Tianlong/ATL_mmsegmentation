# Copyright (c) OpenMMLab. All rights reserved.
from osgeo import gdal
import argparse
import glob
import math
import os
import os.path as osp

from PIL import Image

import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, save_path, splits, crop_size=512):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.

    img_or_label = None  # 用来标注是image还是label

    if 'label' in image_path:
        img_or_label = 'ann_dir'  ##GF2_PMS1__L1A0000189089-MSS1_24label
        img_basename = osp.basename(image_path).split('.')[0].split('_24label')[0] #GF2_PMS1__L1A0000189089-MSS1
        data_type = 'train' if img_basename in splits['train'] else 'val'
        save_path = os.path.join(save_path, img_or_label, data_type)

    elif 'image_RGB_black_side' in image_path:  # 8bit
        img_or_label = 'img_dir'
        img_basename = osp.basename(image_path).split('.')[0] #GF2_PMS1__L1A0000189089-MSS1
        data_type = 'train' if img_basename in splits['train'] else 'val'
        save_path = os.path.join(save_path, img_or_label, data_type)

    # 对图像进行裁切,只切512的部分
    if img_or_label == 'img_dir':
        image_gdal = gdal.Open(image_path)
        img = image_gdal.ReadAsArray()
        img = img.transpose((1,2,0))

        h, w, c = img.shape
        rows, cols, bands = img.shape

        hang = h - (h//crop_size)*crop_size
        lie =  w - (w//crop_size)*crop_size

        # print(f'可裁成{h//crop_size}行...{hang}')
        # print(f'可裁成{w//crop_size}列...{lie}')
        # print(f'共512*512：{((h//crop_size)*(w//crop_size))}张，边缘处')
        
        # 512的部分
        for i in range(h//crop_size):
            for j in range(w//crop_size):
                out_path = os.path.join(save_path, img_basename+'_'+str(i)+'_'+str(j)+'.tif')
                Driver = gdal.GetDriverByName("Gtiff")

                new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
                new_img = Driver.Create(out_path, crop_size, crop_size, c, gdal.GDT_Byte)
            
                new_512 = img[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size,:]   #横着来       

                for band_num in range(bands):
                    band = new_img.GetRasterBand(band_num + 1)
                    band.WriteArray(new_512[:, :, band_num])

    elif img_or_label == 'ann_dir':
        img_label = np.array(Image.open(image_path))
        h, w = img_label.shape
        hang = h - (h//crop_size)*crop_size
        lie =  w - (w//crop_size)*crop_size

        for i in range(h//crop_size):
            for j in range(w//crop_size):
                new_512 = np.zeros((crop_size,crop_size),dtype=np.uint8)
                new_512 = img_label[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size]   #横着来       
                out_path = os.path.join(save_path, img_basename+'_'+str(i)+'_'+str(j)+'.png')
                Image.fromarray(new_512).save(out_path)

        # print(f'--- 标签图像裁切完成')



def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = os.path.join(args.dataset_path, 'data', 'GID_25classes')
    else:   
        out_dir = args.out_dir
    
    mkdir_or_exist(out_dir)

    train_list = []
    val_list = []
    # 读取train_list.txt为列表
    with open(os.path.join(dataset_path, 'train_list.txt'), 'r') as f:
        for line in f.readlines():
            train_list.append(line.strip())
    print('train_list.txt包含', len(train_list), '张')

    # 读取val_list.txt为列表
    with open(os.path.join(dataset_path, 'val_list.txt'), 'r') as f:
        for line in f.readlines():
            val_list.append(line.strip())
    print('val_list.txt包含', len(val_list), '张')

    splits = {
        'train': train_list,
        'val': val_list
    }

    

    print('Making directories...')
    
    label_folder = os.path.join(dataset_path, 'Annotation__index')

    img_folder = os.path.join(dataset_path, 'image_RGB_black_side')
    file_list = [label_folder, img_folder]

    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    print('Find the data', file_list)

    for file_name in file_list:
        src_path_list=[]
        # 如果是图像的话
        if 'image_RGB_black_side' in file_name:
            src_path_list = glob.glob(os.path.join(file_name,'*.tif'))
            print(f"Found {len(src_path_list)} images in {file_name}.")

        # 如果是标签的话
        elif 'Annotation__index' in file_name:
            src_path_list = glob.glob(os.path.join(file_name,'*.png'))
            print(f"Found {len(src_path_list)} labels in {file_name}.")

        assert len(src_path_list) > 0, f'Found no images in {file_name}.'

        # prog_bar = ProgressBar(len(src_path_list))
        for src_path in tqdm(src_path_list,desc=f"---正在切分图像{os.path.basename(file_name)}",colour="GREEN"):
            clip_big_image(src_path, out_dir, splits, crop_size=512)
            # prog_bar.update()
    print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
