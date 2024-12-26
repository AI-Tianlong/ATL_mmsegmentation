# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .cross_entropy_loss import CrossEntropyLoss, cross_entropy
from .utils import get_class_weight, weight_reduce_loss

# S2_5B 数据集，分为L1、L2、L3三级标签，共21类
# L1: 包含5类, 0-4
# L2: 包含11类, 0-10
# L3: 包含21类, 0-20

# reduce_zero_label 后的值 

L1_map = [[0,1,2,3,4],[5,6,7],[8,9,10,11,12,13,14,15,16],[17],[18]]
L2_map = [[0,1],[2],[3,4],[5,6,7],[8],[9,10],[11,12],[13,14,15,16],[17],[18]]
L3_map = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

FiveBillion_19Classes_HieraMap_nobackground = dict(
    # class_L1_{L1中的标签号}_{L1中的标签名称}=[L3级标签的值]
    Classes_Map_L1=dict(
        class_L1_0_Vegetation=[0, 1, 2, 3, 4],
        class_L1_1_Water=[5,6,7],
        class_L1_2_Artificial_surface=[8,9,10,11,12,13,14,15,16],
        class_L1_3_Bare_land=[17],
        class_L1_4_Ice_snow=[18],
    ),
    # class_L2_{L1级标签中的标签值}_{L2级标签中的标签值}_{L2级标签中的标签名称}=[L3级标签中的值]
    Classes_Map_L2=dict(
        class_L2_0_Crop_land=[0, 1],
        class_L2_1_Forest=[2],
        class_L2_2_Grassland=[3,4],
        class_L2_3_Water=[5,6,7],
        class_L2_4_Factory_Shopping_malls=[8],
        class_L2_5_Residence=[9,10],
        class_L2_6_Public_area=[11,12],
        class_L2_7_Transportation_infrastructure=[13,14,15,16],
        class_L2_8_Bare_land=[17],
        class_L2_9_Ice_snow=[18],
    ),
    # class_L3_{L1级标签中的标签值}_{L2级标签中的标签值}_{L3级标签中的标签值}_{L3级标签中的标签名称}
    Classes_Map_L3=dict(
        class_L3_0_Paddy_field=[0],
        class_L3_1_Dry_cropland=[1],
        class_L3_2_Forest=[2],
        class_L3_3_Natural_meadow=[3],
        class_L3_4_Artificial_meadow=[4],
        class_L3_5_River=[5],
        class_L3_6_Lake=[6],
        class_L3_7_Pond=[7],
        class_L3_8_Factory_shopping_malls=[8],
        class_L3_9_Urban_residential=[9],
        class_L3_10_Rural_residential=[10],
        class_L3_11_Stadium=[11],
        class_L3_12_Park_Square=[12],
        class_L3_13_Road=[13],
        class_L3_14_Overpass=[14],
        class_L3_15_Railway_station=[15],
        class_L3_16_Airport=[16],
        class_L3_17_Bare_land=[17],
        class_L3_18_ice_snow=[18]))


def convert_low_level_label_to_High_level(label, classes_map):
    """Convert low level label to High level label.
        e.g.:
          convert L3 label (num_classes=22)  to
          L2 label (num_classes=12) or L1 label (num_classes=6)

    Args:
        label (Tensor): L3 label.  label.shape:[2, 512, 512]
        classes_map (dict): Classes map.

    Returns:
        Tensor: Lx label.
    """

    label_list = list()  # 转换 L3 (low level) --> L1 L2 (hgih level)
    for _, high_level_dict in list(classes_map.items())[:-1]:
        high_level_label = torch.zeros_like(label).fill_(255)  # [2,512,512] 255 is the ignore index #因为用了like，所以也在GPU上
        for high_level_label_value, high_level_key in enumerate(high_level_dict): # For L1:0 1 2 3 4
            low_level_label_list = high_level_dict[high_level_key]
            for low_level_label in low_level_label_list:
                high_level_label[label == low_level_label] = high_level_label_value

        label_list.append(high_level_label)
    label_list.append(label)  # add L3 label to label List
    return label_list  # [L1级label, L2级label, L3级label] #tensor [2,512,512][2,512,512][2,512,512]


def Tree_Min_Loss(pred_seg_logits, # [B,5+10+19,512,512]
                  label_level_list,  # [512,512](0-4) [512,512](0,10) [512,512](0,20)   
                  num_classes, # 19
                  eps=1e-8, 
                  gamma=2,
                  ignore_index=-100,
                  **kwargs):

    """计算子类和父类的损失，求和之后返回
    Args:   
        predictions: torch.Tensor, shape=(batch, num_classes+7, h, w), 预测的标签 # [2,37,512,512]
        targets: torch.Tensor, shape=(batch, h, w), 真实的标签[0~19]  # [2,31,512,512]
        targets_top: torch.Tensor, shape=(batch, h, w), 合并成父类的标签[0~7]
        num_classes: int, 类别数
        indices_high: list, 父类对应的子类的的索引范围
    """
    # import pdb; pdb.set_trace()
    # b, _, h, w = pred_seg_logits.shape # [2, 34, 128, 128]
    # 让所有推理过一个 sigmoid
    
    # sigmoid,每一个通道的值，自己去算sigmoid，和其他人没关系
    pred_sigmoid = torch.sigmoid(pred_seg_logits.float()) # 让预测值过一个sigmoid函数，变为0-1 (1, 34, 512, 512) # 之和自己的值有关系
    # softmax,每一个相同位置的像素去整体算softmax值，和其他通道有关系。
    pred_softmax = torch.softmax(pred_seg_logits, dim=1) # 让预测值过一个softmax函数，变为0-1 (1, 34, 512, 512) # 和其他通道的值也有关系
    # TODO: 这里的softmax，应该每一个层级自己做，还是整体做。

    label_L1 = label_level_list[0]  # 无效标签为255
    label_L2 = label_level_list[1] 
    label_L3 = label_level_list[2] 

    # 将标签转换为one-hot编码,并且将维度变换为(batch, num_classes, h, w), 例如(1, 19, 1024, 2048)，
    # 一个像素用19维的向量表示
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 代表2
    
    # ======================================= 生成 ont-hot 向量的标签 ============================
    # 处理L1标签       # 将ignore的像素值的位置设置为全0向量。
    invalid_pos = (label_L3==ignore_index) # 无效的位置 #ignore # 变成one hot向量
    label_L1[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L1_one_hot = F.one_hot(label_L1, num_classes=len(L1_map)).permute(0,3,1,2)  # [2,5,512,512]
    # 处理L2标签
    label_L2[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L2_one_hot = F.one_hot(label_L2, num_classes=len(L2_map)).permute(0,3,1,2)  # [2,10,512,512]
    # 处理L3标签
    label_L3[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L3_one_hot = F.one_hot(label_L3, num_classes=len(L3_map)).permute(0,3,1,2)  # [2,19,512,512]
    

    # 计算loss
    # 19 7 原文的loss。是这样的  [5,10,19] [0,5] [5,16] [16,37]
    # 0 1 2 3 4  [0,5) L1的概率值
    # 5 6 7 8 9 10 11 12 13 14  [5,15) L2的概率值
    # 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 [15,37) L3的概率值
    # import pdb; pdb.set_trace()
    pred_sigmoid_L1 = pred_sigmoid[:,0:len(L1_map),:,:]    # 前5个是L1
    pred_sigmoid_L2 = pred_sigmoid[:,len(L1_map):len(L1_map)+len(L2_map),:,:]  # 前5-15个是L2
    pred_sigmoid_L3 = pred_sigmoid[:,-len(L3_map):,:,:] # 最后的19个是L3   
    

    # ==============================《约束1 去更新L2 L1 使父类的特征图的值要大于等于子类的特征图的值包含其中最大的sigmoid值》==================
    # 根据法则1, 更新L2的sigmoid值，确保L2中的每个类特征图上的值，是L2、L3的最大值
    # 更新sigmoid的值，根据法则1
    update_max_pred_sigmoid_L1 = pred_sigmoid_L1.clone() # clone 修改Update 不会动原始的pred
    update_max_pred_sigmoid_L2 = pred_sigmoid_L2.clone()
    update_max_pred_sigmoid_L3 = pred_sigmoid_L3.clone()

    # 更新sigmoid的值，根据法则1
    # 更新L2  --> 原始的L3 和 原始L2的最大值
    for L2_index, L3_index_list_in_L2 in enumerate(L2_map):
        
        # 0，[0,1]   # L2的0 和 L3的0、1 这三个一起去找最大值，然后赋值给L2
        # 1，[2]
        # 2，[3,4]
        # 3，[5,6,7]
        # 4，[8]
        # 5，[9,10]
        # 6，[11,12]
        # 7，[13,14,15,16]   for L3_index in L3_index_list_in_L2:
        # 8，[17]
        # 9，[18]                # 如果不是按+1这样索引，会少一维度。
            update_max_pred_sigmoid_L2[:, L2_index:L2_index+1,:,:] = torch.max(torch.cat([pred_sigmoid_L3[:,L3_index:L3_index+1,:,:] for L3_index in L3_index_list_in_L2] +
                                                                                         [pred_sigmoid_L2[:,L2_index:L2_index+1,:,:]], dim=1), 1, True)[0]  # 0-4 5-16 17-36

    # 更新L1 --> 更新后的L2 和 原始L1的最大值
    for L1_index, L2_index_list_in_L1 in enumerate(L1_map):
            # 0, [0, 1, 2, 3, 4]
            # 1, [5, 6, 7]
            # 2, [8, 9, 10, 11, 12, 13, 14, 15, 16]
            # 3, [17]
            # 4, [18]
            update_max_pred_sigmoid_L1[:, L1_index:L1_index+1,:,:] = torch.max(torch.cat([update_max_pred_sigmoid_L2[:,L2_index:L2_index+1,:,:] for L2_index in L2_index_list_in_L1] + 
                                                                                         [pred_sigmoid_L1[:,L1_index:L1_index+1,:,:]], dim=1), 1, True)[0]

    # ==============================《约束2 去更新L2 L3 使子类的特征图要小于等于父类最大的sigmoid值》==================
    update_min_pred_sigmoid_L1 = pred_sigmoid_L1.clone() # clone 修改Update 不会动原始的pred
    update_min_pred_sigmoid_L2 = pred_sigmoid_L2.clone()
    update_min_pred_sigmoid_L3 = pred_sigmoid_L3.clone()

    # 更新L2的值---> 原始的L1 和原始L2的最小值
    for L1_index, L2_index_list_in_L1 in enumerate(L1_map):
        for L2_index in L2_index_list_in_L1:
            # L1:0, L2_list:[0, 1, 2, 3, 4]     0-0  0-1  0-2  0-3  0-4 分别去找最小，复制给L2
            # L1:1, L2_list:[5, 6, 7]
            # L1:2, L2_list:[8, 9, 10, 11, 12, 13, 14, 15, 16]
            # L1:3, L2_list:[17]
            # L1:4, L2_list:[18]
            update_min_pred_sigmoid_L2[:, L2_index:L2_index+1,:,:] = torch.min(torch.cat([pred_sigmoid_L2[:,L2_index:L2_index+1,:,:],
                                                                                          pred_sigmoid_L1[:,L1_index:L1_index+1,:,:]], dim=1), 1, True)[0]

    # 更新L3的值---> 更新后的L2 和原始L3的最小值
    for L2_index, L3_index_list_in_L2 in enumerate(L1_map):
        for L3_index in L3_index_list_in_L2:
            # 0，[0,1]   # L2的0 和 L3的0找最小，给L3的0   L2的0 和 L3的1找最小，给L3的1
            # 1，[2]
            # 2，[3,4]
            # 3，[5,6,7]
            # 4，[8]
            # 5，[9,10]
            # 6，[11,12]
            # 7，[13,14,15,16]   for L3_index in L3_index_list_in_L2:
            # 8，[17]
            # 9，[18]                # 如果不是按+1这样索引，会少一维度。
            update_min_pred_sigmoid_L3[:, L3_index:L3_index+1,:,:] = torch.min(torch.cat([update_min_pred_sigmoid_L2[:,L2_index:L2_index+1,:,:],
                                                                                          pred_sigmoid_L3[:,L3_index:L3_index+1,:,:]], dim=1), 1, True)[0]
    #  =================================== 更新 seg_logits 结束，去计算 Tree-min loss值 ================================
    # 有效区域的索引。    
    valid_pos = (~invalid_pos).unsqueeze(1) # 子类标签 有效的位置，把无效的位置取反，然后在通道纬度上增加一个维度

    num_valid = valid_pos.sum() # 子类标签 有效的位置的个数

    # 计算loss 交叉熵损失函数 交叉熵损失函数的定义是：-y*log(y_hat)-(1-y)*log(1-y_hat)
    # loss = 求和｛(-前19通道子类的标签*log(前19通道子类和父类预测之中小的那个) - 
    #           (1-前19通道子类的标签)*log(1-前19通道子类和父类预测之中小的那个))有效位置的索引，true和false，只算有效位置｝除以有效位置的个数/类别数(归一化)

    # 求L3的 predict 和 one-hot向量的 BCE loss
    # 二元交叉熵，             # ,:len(L3_map), = ,:,
    # import pdb; pdb.set_trace()
    loss = ((-label_L3_one_hot*torch.log(update_min_pred_sigmoid_L3+eps) # 求负类的loss
             -(1-label_L3_one_hot)*torch.log(1-update_max_pred_sigmoid_L3+eps))
        *valid_pos).sum()/num_valid/len(L3_map)

    # 求L2的 predict 和 one-hot向量的 BCE loss
    loss += ((-label_L2_one_hot*torch.log(update_min_pred_sigmoid_L2+eps) # 求负类的loss
             -(1-label_L2_one_hot)*torch.log(1-update_max_pred_sigmoid_L2+eps))
        *valid_pos).sum()/num_valid/len(L2_map)

    # 求L2的 predict 和 one-hot向量的 BCE loss
    loss += ((-label_L1_one_hot*torch.log(update_min_pred_sigmoid_L1+eps) # 求负类的loss
             -(1-label_L1_one_hot)*torch.log(1-update_max_pred_sigmoid_L1+eps))
        *valid_pos).sum()/num_valid/len(L1_map)

    # 然后放大5倍返回损失？why？
    # 权重调整：在多任务学习或多损失函数的情况下，不同的损失函数可能会有不同的量级。乘以一个系数可以平衡这些损失函数，使它们对最终的总损失有类似的重要性。
    # 训练动态调整：通过乘以一个系数，可以加快或减慢训练过程中的梯度更新速度。一个较大的系数会使梯度变得更大，从而加快参数更新的速度。
    # 强调特定损失：有时我们希望特定的损失在总损失中占据更大的比例，以便模型更加关注特定的目标。通过乘以一个系数，可以增加该损失在总损失中的权重。
    return 5*loss
    # return loss


def Focal_Tree_Min_Loss(pred_seg_logits, # [B,5+10+19,512,512]
                        label_level_list,  # [512,512](0-4) [512,512](0,10) [512,512](0,20)   
                        num_classes, # 19
                        eps=1e-8, 
                        gamma=2,
                        ignore_index=-100,
                        **kwargs):

    """计算子类和父类的损失，求和之后返回
    Args:   
        predictions: torch.Tensor, shape=(batch, num_classes+7, h, w), 预测的标签 # [2,37,512,512]
        targets: torch.Tensor, shape=(batch, h, w), 真实的标签[0~19]  # [2,31,512,512]
        targets_top: torch.Tensor, shape=(batch, h, w), 合并成父类的标签[0~7]
        num_classes: int, 类别数
        indices_high: list, 父类对应的子类的的索引范围
    """
    b, _, h, w = pred_seg_logits[0] # [2, _, 512, 512]
    # 让所有推理过一个 sigmoid
    pred_sigmoid = torch.sigmoid(pred_seg_logits.float()) # 让预测值过一个sigmoid函数，变为0-1 (1, 26, 1024, 2048)
    
    label_L1 = label_level_list[0]  # 无效标签为255
    label_L2 = label_level_list[1] 
    label_L3 = label_level_list[2] 

    # 将标签转换为one-hot编码,并且将维度变换为(batch, num_classes, h, w), 例如(1, 19, 1024, 2048)，
    # 一个像素用19维的向量表示
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 代表2
    
    # ======================================= 生成 ont-hot 向量的标签 ============================
    # 处理L1标签       # 将ignore的像素值的位置设置为全0向量。
    invalid_pos = (label_L3==ignore_index) # 无效的位置 #ignore # 变成one hot向量
    label_L1[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L1_one_hot = F.one_hot(label_L1, num_classes=len(L1_map)).permute(0,3,1,2) 
    # 处理L2标签
    label_L2[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L2_one_hot = F.one_hot(label_L2, num_classes=len(L2_map)).permute(0,3,1,2) 
    # 处理L3标签
    label_L3[invalid_pos]=0  # 无效的位置设置为19, 之后忽略掉
    label_L3_one_hot = F.one_hot(label_L3, num_classes=len(L3_map)).permute(0,3,1,2) 
    

    # 计算loss
    # 19 7 原文的loss。是这样的  [5,10,19] [0,5] [5,16] [16,37]
    # 0 1 2 3 4  [0,5) L1的概率值
    # 5 6 7 8 9 10 11 12 13 14  [5,15) L2的概率值
    # 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 [15,37) L3的概率值
    pred_sigmoid_L1 = pred_sigmoid[:,0:len(L1_map),:,:]    # 前5个是L1
    pred_sigmoid_L2 = pred_sigmoid[:,len(L1_map):len(L1_map)+len(L2_map),:,:]  # 前5-15个是L2
    pred_sigmoid_L3 = pred_sigmoid[:,-len(L3_map):,:,:] # 最后的19个是L3   
    

    # ==============================《约束1 去更新L2 L1 使父类的特征图的值要大于等于子类的特征图的值包含其中最大的sigmoid值》==================
    # 根据法则1, 更新L2的sigmoid值，确保L2中的每个类特征图上的值，是L2、L3的最大值
    # 更新sigmoid的值，根据法则1
    update_max_pred_sigmoid_L1 = pred_sigmoid_L1.clone() # clone 修改Update 不会动原始的pred
    update_max_pred_sigmoid_L2 = pred_sigmoid_L2.clone()
    update_max_pred_sigmoid_L3 = pred_sigmoid_L3.clone()

    # 更新sigmoid的值，根据法则1
    # 更新L2  --> 原始的L3 和 原始L2的最大值
    for L2_index, L3_index_list_in_L2 in enumerate(L2_map):
        
        # 0，[0,1]   # L2的0 和 L3的0、1 这三个一起去找最大值，然后赋值给L2
        # 1，[2]
        # 2，[3,4]
        # 3，[5,6,7]
        # 4，[8]
        # 5，[9,10]
        # 6，[11,12]
        # 7，[13,14,15,16]   for L3_index in L3_index_list_in_L2:
        # 8，[17]
        # 9，[18]                # 如果不是按+1这样索引，会少一维度。
            update_max_pred_sigmoid_L2[:, L2_index:L2_index+1,:,:] = torch.max(torch.cat([pred_sigmoid_L3[:,L3_index:L3_index+1,:,:] for L3_index in L3_index_list_in_L2] +
                                                                                         [pred_sigmoid_L2[:,L2_index:L2_index+1,:,:]], dim=1), 1, True)[0]  # 0-4 5-16 17-36

    # 更新L1 --> 更新后的L2 和 原始L1的最大值
    for L1_index, L2_index_list_in_L1 in enumerate(L1_map):
            # 0, [0, 1, 2, 3, 4]
            # 1, [5, 6, 7]
            # 2, [8, 9, 10, 11, 12, 13, 14, 15, 16]
            # 3, [17]
            # 4, [18]
            update_max_pred_sigmoid_L1[:, L1_index:L1_index+1,:,:] = torch.max(torch.cat([update_max_pred_sigmoid_L2[:,L2_index:L2_index+1,:,:] for L2_index in L2_index_list_in_L1] + 
                                                                                         [pred_sigmoid_L1[:,L1_index:L1_index+1,:,:]], dim=1), 1, True)[0]

    # ==============================《约束2 去更新L2 L3 使子类的特征图要小于等于父类最大的sigmoid值》==================
    update_min_pred_sigmoid_L1 = pred_sigmoid_L1.clone() # clone 修改Update 不会动原始的pred
    update_min_pred_sigmoid_L2 = pred_sigmoid_L2.clone()
    update_min_pred_sigmoid_L3 = pred_sigmoid_L3.clone()

    # 更新L2的值---> 原始的L1 和原始L2的最小值
    for L1_index, L2_index_list_in_L1 in enumerate(L1_map):
        for L2_index in L2_index_list_in_L1:
            # L1:0, L2_list:[0, 1, 2, 3, 4]     0-0  0-1  0-2  0-3  0-4 分别去找最小，复制给L2
            # L1:1, L2_list:[5, 6, 7]
            # L1:2, L2_list:[8, 9, 10, 11, 12, 13, 14, 15, 16]
            # L1:3, L2_list:[17]
            # L1:4, L2_list:[18]
            update_min_pred_sigmoid_L2[:, L2_index:L2_index+1,:,:] = torch.min(torch.cat([pred_sigmoid_L2[:,L2_index:L2_index+1,:,:],
                                                                                          pred_sigmoid_L1[:,L1_index:L1_index+1,:,:]], dim=1), 1, True)[0]

    # 更新L3的值---> 更新后的L2 和原始L3的最小值
    for L2_index, L3_index_list_in_L2 in enumerate(L1_map):
        for L3_index in L3_index_list_in_L2:
            # 0，[0,1]   # L2的0 和 L3的0找最小，给L3的0   L2的0 和 L3的1找最小，给L3的1
            # 1，[2]
            # 2，[3,4]
            # 3，[5,6,7]
            # 4，[8]
            # 5，[9,10]
            # 6，[11,12]
            # 7，[13,14,15,16]   for L3_index in L3_index_list_in_L2:
            # 8，[17]
            # 9，[18]                # 如果不是按+1这样索引，会少一维度。
            update_min_pred_sigmoid_L3[:, L3_index:L3_index+1,:,:] = torch.min(torch.cat([update_min_pred_sigmoid_L2[:,L2_index:L2_index+1,:,:],
                                                                                          pred_sigmoid_L3[:,L3_index:L3_index+1,:,:]], dim=1), 1, True)[0]
    #  =================================== 更新 seg_logits 结束，去计算 Tree-min loss值 ================================
    # 有效区域的索引。    
    valid_pos = (~invalid_pos).unsqueeze(1) # 子类标签 有效的位置，把无效的位置取反，然后在通道纬度上增加一个维度

    num_valid = valid_pos.sum() # 子类标签 有效的位置的个数

    # 计算loss 交叉熵损失函数 交叉熵损失函数的定义是：-y*log(y_hat)-(1-y)*log(1-y_hat)
    # loss = 求和｛(-前19通道子类的标签*log(前19通道子类和父类预测之中小的那个) - 
    #           (1-前19通道子类的标签)*log(1-前19通道子类和父类预测之中小的那个))有效位置的索引，true和false，只算有效位置｝除以有效位置的个数/类别数(归一化)

    # 求L3的 predict 和 one-hot向量的 BCE loss
    # 二元交叉熵，             # ,:len(L3_map), = ,:,
    loss = ((-label_L3_one_hot*torch.pow((1.0-update_min_pred_sigmoid_L3),gamma)*torch.log(update_min_pred_sigmoid_L3+eps) # 求负类的loss
             -(1-label_L3_one_hot)*torch.pow(update_max_pred_sigmoid_L3, gamma)**torch.log(1-update_max_pred_sigmoid_L3+eps))
        *valid_pos).sum()/num_valid/len(L3_map)

    # 求L2的 predict 和 one-hot向量的 BCE loss
    loss += ((-label_L2_one_hot*torch.pow((1.0-update_min_pred_sigmoid_L2),gamma)*torch.log(update_min_pred_sigmoid_L2+eps) # 求负类的loss
             -(1-label_L2_one_hot)*torch.pow(update_max_pred_sigmoid_L2, gamma)**torch.log(1-update_max_pred_sigmoid_L2+eps))
        *valid_pos).sum()/num_valid/len(L2_map)

    # 求L2的 predict 和 one-hot向量的 BCE loss
    loss += ((-label_L1_one_hot*torch.pow((1.0-update_min_pred_sigmoid_L1),gamma)*torch.log(update_min_pred_sigmoid_L1+eps) # 求负类的loss
             -(1-label_L1_one_hot)*torch.pow(update_max_pred_sigmoid_L1, gamma)**torch.log(1-update_max_pred_sigmoid_L1+eps))
        *valid_pos).sum()/num_valid/len(L1_map)

    # 然后放大5倍返回损失？why？
    # 权重调整：在多任务学习或多损失函数的情况下，不同的损失函数可能会有不同的量级。乘以一个系数可以平衡这些损失函数，使它们对最终的总损失有类似的重要性。
    # 训练动态调整：通过乘以一个系数，可以加快或减慢训练过程中的梯度更新速度。一个较大的系数会使梯度变得更大，从而加快参数更新的速度。
    # 强调特定损失：有时我们希望特定的损失在总损失中占据更大的比例，以便模型更加关注特定的目标。通过乘以一个系数，可以增加该损失在总损失中的权重。
    # return 5*loss
    return loss


class TreeTripletLoss(nn.Module):
    """ L^TT Tree-Triplet Loss

    Args:
        embedding (torch.Tensor): The input embedding feature map.
        labels (torch.Tensor): The input label map.
        max_triplet (int): The maximum number of triplets to sample.
    
    Returns:
        loss (torch.Tensor): The computed loss.
        class_count (torch.Tensor)
    """
    def __init__(self, ignore_index=255):
        super().__init__()

        self.ignore_index = ignore_index

    def forward(self, embedding, labels=None, max_triplet=200):
        """ L^TT Tree-Triplet Loss ``forward`` function.

        Args:
            embedding (torch.Tensor): The input embedding feature map.
            labels (torch.Tensor): The input label map.
            max_triplet (int): The maximum number of triplets to sample.
        
        Returns:
            loss (torch.Tensor): The computed loss.
            class_count (torch.Tensor)

        
        """
        batch_size = embedding.shape[0]
        labels = labels.unsqueeze(1).float().clone()

        # 将 labels resize成 和embedding一样大小的
        labels = torch.nn.functional.interpolate(labels,
                                                 (embedding.shape[2], embedding.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == embedding.shape[-1], '{} {}'.format(labels.shape, embedding.shape)

        labels = labels.view(-1)
        embedding = embedding.permute(0, 2, 3, 1)  # [2,256,64,64]-->[2,64,64,256]
        embedding = embedding.contiguous().view(-1, embedding.shape[-1]) # [2,64,64,256]-->[2*64*64,256]
        
        triplet_loss=0
        exist_classes = torch.unique(labels)  # 当前label中存在的类别
        exist_classes = [x for x in exist_classes if x != 255]
        class_count=0
        
        # hiera_map = [0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6]
        # hiera_index = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]]
        for class_ in exist_classes:  # [ 0,  1,  2,  8, 11, 12, 13, 18]
            index_anchor = labels==class_  # i^A anchor,当前类
                                           # 然后去找正例 i^P postive 和 反例 i^N negative
            # 只找 L2 L3之间的关系，不考虑 L1了？？？ 

           

            for postive_list in L2_map:
                if class_ in postive_list:
                    postive_list_cuda = torch.tensor(postive_list).cuda()
                    index_pos = torch.isin(labels, postive_list_cuda) & ~index_anchor
            
            index_neg = index_anchor.clone()
            index_neg = (index_anchor | index_pos)

            # import pdb; pdb.set_trace()
            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)
            
            
            embedding_anchor = embedding[index_anchor][:min_size]  # 前多少个
            embedding_pos = embedding[index_pos][:min_size]
            embedding_neg = embedding[index_neg][:min_size]
            
            distance = torch.zeros(min_size, 2).cuda()
            distance[:,0:1] = 1-(embedding_anchor*embedding_pos).sum(1, True)  # 1-余弦相似度
            distance[:,1:2] = 1-(embedding_anchor*embedding_neg).sum(1, True)  # 1-余弦相似度
            
            # margin always 0.1 + (4-2)/4 since the hierarchy is three level
            # TODO: should include label of pos is the same as anchor, i.e. margin=0.1
            margin = 0.6*torch.ones(min_size).cuda()
            
            tl = distance[:,0] - distance[:,1] + margin
            tl = F.relu(tl)

            if tl.size(0)>0:
                triplet_loss += tl.mean()
                class_count+=1
        if class_count==0:
            return None, torch.tensor([0]).cuda()
        triplet_loss /=class_count
        return triplet_loss, torch.tensor([class_count]).cuda()



@MODELS.register_module()
class ATL_Hiera_Loss(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_name = 'loss_hiera',
                 loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.tree_triplet_loss = TreeTripletLoss(ignore_index = 255)
        self.cross_entropy_loss = CrossEntropyLoss()

        self._loss_name = loss_name

    def forward(self,
                step,
                embedding,         # [2,256,64,64]
                pred_seg_logits,   # [2,34,128,128] [5+10+19]
                label,
                **kwargs):
        
        # import pdb; pdb.set_trace()

        
        hiera_label_list = convert_low_level_label_to_High_level(label, FiveBillion_19Classes_HieraMap_nobackground)
        
        # Tree-Min Loss
        tree_min_loss = Tree_Min_Loss(pred_seg_logits, hiera_label_list, self.num_classes, ignore_index=255)  # 10.9371
        # L1 cross entropy loss
        ce_loss_L1 = self.cross_entropy_loss(pred_seg_logits[:,:len(L1_map),:,:], hiera_label_list[0])
        # L2 cross entropy loss
        ce_loss_L2 = self.cross_entropy_loss(pred_seg_logits[:,len(L1_map):len(L1_map)+len(L2_map),:,:], hiera_label_list[1])
        # L3 cross entropy loss
        ce_loss_L3 = self.cross_entropy_loss(pred_seg_logits[:,-len(L3_map):,:,:], hiera_label_list[2])


        # loss = tree_min_loss + ce_loss_L1 + ce_loss_L2 + ce_loss_L3
        loss = ce_loss_L1 + ce_loss_L2 + ce_loss_L3

        # loss_triplet, class_count = self.tree_triplet_loss(embedding, label)
        # class_counts = [torch.ones_like(class_count) for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather(class_counts, class_count, async_op=False)
        # class_counts = torch.cat(class_counts, dim=0)

        # if torch.distributed.get_world_size()==torch.nonzero(class_counts, as_tuple=False).size(0):
        #     factor = 1/4*(1+torch.cos(torch.tensor((step.item()-80000)/80000*math.pi))) if step.item()<80000 else 0.5
        #     loss += factor*loss_triplet
            
        
        return loss*self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
