import numpy as np
from PIL import Image
import glob
import time
from tqdm import tqdm, trange
import os 

#转换数据集
"""因为，可以看数据集发现，masks_png都是0-1的小数，没有办法训练，所以这里需要转换一下数据集"""
"""给的代码示例是一次转换一张，现在试试一次性全部转换"""
#类别数 忽略-0 背景-1、建筑-2、道路-3、水-4、贫瘠-5、森林-6、农业-7。没有数据的区域被指定为0，应该忽略 
COLOR_MAP = dict(
    Others=(0, 0, 0),
    Built_up=(255, 0, 0),
    Farmland=(0, 255, 0),
    Forest=(0, 255, 255),
    Meadow=(255, 255, 0),
    Water=(0, 0, 255)
)

def render(mask_path, vis_path):
    new_mask = np.array(Image.open(mask_path)).astype(np.uint8) #把原来的图片读出来，存成uint8的格式
    cm = np.array(list(COLOR_MAP.values())).astype(np.uint8) #取出COLOR_MAP中的值，存到列表
    color_img = cm[new_mask]  #这里就变成1024*1024*3的了？ why？？
    color_img = Image.fromarray(np.uint8(color_img)) #从np的array格式转换成PIL格式
    color_img.save(vis_path)

#数据集的地址    
masks_file_path = r'D:\ATL\AI_work\OpenMMLab\mmsegmentation\data\gid\ann_dir\val'
vis_file_path = r'D:\ATL\AI_work\OpenMMLab\mmsegmentation\projects\gid_dataset\vis_atl'
if not os.path.exists(vis_file_path): os.mkdir(vis_file_path)

#数据集的地址列表
masks_list = os.listdir(masks_file_path)
# masks_list = np.array(sorted(glob.glob(masks_file_path + '*.png'))) #把这个文件夹中的图片的地址，以有序的列表形式存放 
print(len(masks_list))



if __name__ == '__main__':
    print('vis_png 转换ing...')
    start = time.time()
    for i in trange(len(masks_list)):
        render(os.path.join(masks_file_path, masks_list[i]), os.path.join(vis_file_path, masks_list[i])) #这里得改一下
    end = time.time()
    print(f'共转换vis_png {len(masks_list)} 张,耗时 {(end-start)} 秒')