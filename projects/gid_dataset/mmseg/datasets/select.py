import shutil


src_list = [
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0000647767-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001064454-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001348919-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001680851-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001680853-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001680857-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS1__L1A0001757429-MSS1.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0000607681-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0000635115-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0000658637-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0001206072-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0001471436-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0001642620-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0001787089-MSS2.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\image_RGB\GF2_PMS2__L1A0001838560-MSS2.tif',

]

label_list = [
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0000647767-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001064454-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001348919-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001680851-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001680853-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001680857-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS1__L1A0001757429-MSS1_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0000607681-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0000635115-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0000658637-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0001206072-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0001471436-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0001642620-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0001787089-MSS2_label.tif',
    r'D:\ATL\AI_work\Datasets\GID\GID\Large-scale Classification_5classes\label_5classes\GF2_PMS2__L1A0001838560-MSS2_label.tif',

]

dest_dir = r'D:\ATL\AI_work\Datasets\GID\GID\15\labels'

# 把src_list中的文件复制到dest_dir中


for src in label_list:
    shutil.copy(src, dest_dir)
