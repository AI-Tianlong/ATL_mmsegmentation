# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import warnings
from ATL_Tools import setup_logger

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
S2_5B_Dataset_21Classes_Map_nobackground = dict(
    # class_L1_{L1中的标签号}_{L1中的标签名称}=[L3级标签的值]
    Classes_Map_L1=dict(
        class_L1_0_Vegetation=[0, 1, 2, 3, 4, 5, 6],
        class_L1_1_Water=[7, 8, 9],
        class_L1_2_Artificial_surface=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        class_L1_3_Bare_land=[19],
        class_L1_4_Ice_snow=[20],
    ),
    # class_L2_{L1级标签中的标签值}_{L2级标签中的标签值}_{L2级标签中的标签名称}=[L3级标签中的值]
    Classes_Map_L2=dict(
        class_L2_0_0_Crop_land=[0, 1, 2],
        class_L2_0_1_Garden_land=[3],
        class_L2_0_2_Forest=[4],
        class_L2_0_3_Grassland=[5, 6],
        class_L2_1_4_Water=[7, 8, 9],
        class_L2_2_5_Factory_Shopping_malls=[10],
        class_L2_2_6_Residence=[11, 12],
        class_L2_2_7_Public_area=[13, 14],
        class_L2_2_8_Transportation_infrastructure=[15, 16, 17, 18],
        class_L2_3_9_Bare_land=[19],
        class_L2_4_10_Ice_snow=[20],
    ),
    # class_L3_{L1级标签中的标签值}_{L2级标签中的标签值}_{L3级标签中的标签值}_{L3级标签中的标签名称}
    Classes_Map_L3=dict(
        class_L3_0_0_0_Paddy_field=[0],
        class_L3_0_0_1_Irrigated_field=[1],
        class_L3_0_0_2_Dry_cropland=[2],
        class_L3_0_1_3_Garden_land=[3],
        class_L3_0_2_4_Forest=[4],
        class_L3_0_3_5_Natural_meadow=[5],
        class_L3_0_3_6_Artificial_meadow=[6],
        class_L3_1_4_7_River=[7],
        class_L3_1_4_8_Lake=[8],
        class_L3_1_4_9_Pond=[9],
        class_L3_2_5_10_Factory_shopping_malls=[10],
        class_L3_2_6_11_Urban_residential=[11],
        class_L3_2_6_12_Rural_residential=[12],
        class_L3_2_7_13_Stadium=[13],
        class_L3_2_7_14_Park_Square=[14],
        class_L3_2_8_15_Road=[15],
        class_L3_2_8_16_Overpass=[16],
        class_L3_2_8_17_Railway_station=[17],
        class_L3_2_8_18_Airport=[18],
        class_L3_1_9_19_Bare_land=[19],
        class_L3_1_10_20_ice_snow=[20]))


S2_5B_Dataset_22Classes_Map = dict(
    # class_L1_{L1中的标签号}_{L1中的标签名称}=[L3级标签的值]
    Classes_Map_L1=dict(
        class_L1_0_Other_land=[0],
        class_L1_1_Vegetation=[1, 2, 3, 4, 5, 6, 7],
        class_L1_2_Water=[8, 9, 10],
        class_L1_3_Artificial_surface=[11, 12, 13, 14, 15, 16, 17, 18, 19],
        class_L1_4_Bare_land=[20],
        class_L1_5_Ice_snow=[21],
    ),
    # class_L2_{L1级标签中的标签值}_{L2级标签中的标签值}_{L2级标签中的标签名称}=[L3级标签中的值]
    Classes_Map_L2=dict(
        class_L2_0_0_Other_land=[0],
        class_L2_1_1_Crop_land=[1, 2, 3],
        class_L2_1_2_Garden_land=[4],
        class_L2_1_3_Forest=[5],
        class_L2_1_4_Grassland=[6, 7],
        class_L2_2_5_Water=[8, 9, 10],
        class_L2_3_6_Factory_Shopping_malls=[11],
        class_L2_3_7_Residence=[12, 13],
        class_L2_3_8_Public_area=[14, 15],
        class_L2_3_9_Transportation_infrastructure=[16, 17, 18, 19],
        class_L2_4_10_Bare_land=[20],
        class_L2_5_11_Ice_snow=[21],
    ),
    # class_L3_{L1级标签中的标签值}_{L2级标签中的标签值}_{L3级标签中的标签值}_{L3级标签中的标签名称}
    Classes_Map_L3=dict(
        class_L3_0_0_0_Other_land=[0],
        class_L3_1_1_1_Paddy_field=[1],
        class_L3_1_1_2_Irrigated_field=[2],
        class_L3_1_1_3_Dry_cropland=[3],
        class_L3_1_2_4_Garden_land=[4],
        class_L3_1_3_5_Forest=[5],
        class_L3_1_4_6_Natural_meadow=[6],
        class_L3_1_4_7_Artificial_meadow=[7],
        class_L3_1_5_8_River=[8],
        class_L3_1_5_9_Lake=[9],
        class_L3_1_5_10_Pond=[10],
        class_L3_1_6_11_Factory_shopping_malls=[11],
        class_L3_1_7_12_Urban_residential=[12],
        class_L3_1_7_13_Rural_residential=[13],
        class_L3_1_8_14_Stadium=[14],
        class_L3_1_8_15_Park_Square=[15],
        class_L3_1_9_16_Road=[16],
        class_L3_1_9_17_Overpass=[17],
        class_L3_1_9_18_Railway_station=[18],
        class_L3_1_9_19_Airport=[19],
        class_L3_1_10_20_Bare_land=[20],
        class_L3_1_11_21_ice_snow=[21]))


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
    # import pdb
    # pdb.set_trace()         # label 在cuda:0上
    label_list = list()  # 转换 L3 (low level) --> L1 L2 (hgih level)
    for _, high_level_dict in list(classes_map.items())[:-1]:
        high_level_label = torch.zeros_like(label).fill_(255)  # [2,512,512] 255 is the ignore index #因为用了like，所以也在GPU上
        for high_level_label_value, high_level_key in enumerate(high_level_dict):
            low_level_label_list = high_level_dict[high_level_key]
            for low_level_label in low_level_label_list:
                high_level_label[label == low_level_label] = high_level_label_value
                # pdb.set_trace()         # label 在cuda:0上
        # pdb.set_trace()
        label_list.append(high_level_label)
    # pdb.set_trace()         # label 在cuda:0上
    label_list.append(label)  # L3 label
    # pdb.set_trace()         # label 在cuda:0上
    return label_list  # [L1级label, L2级label, L3级label] #tensor [2,512,512][2,512,512][2,512,512]


@MODELS.register_module()
class ATL_Loss(nn.Module):
    """ATL_Loss .

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
        classes_map (dict): Classes_Map for ATL_loss. Default: None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='atl_loss',
                 avg_non_ignore=False,
                 classes_map=None):
        super().__init__()

        # import pdb
        # pdb.set_trace()
        # logger.warning('进入到 ATL_Loss 的 _init_()函数')
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.classes_map = classes_map

        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self._loss_name = loss_name

        # 构造层级Loss函数
        self.loss_ce_dict = dict()
        if self.classes_map == None:
            # 不分级，默认为 1 级，只要一个交叉熵损失
            self.loss_ce_dict[f'{self._loss_name}_ce'] = CrossEntropyLoss(
                loss_name=f'{self._loss_name}_ce')
            # loss_ce = CrossEntropyLoss(loss_name='atl_loss_ce')
        else:
            self.loss_ce_num = len(self.classes_map)  # 识别分为几级标签, 3 --> L1, L2, L3
            for loss_ce_index in range(self.loss_ce_num):
                self.loss_ce_dict[f'{self._loss_name}_ce_L{loss_ce_index+1}'] = CrossEntropyLoss(
                        loss_name=f'{self._loss_name}_ce_L{loss_ce_index+1}')
        # ✔

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(   # 由uperhead的loss_by_feat()调用
        self,
        pred,  # [2,37,512,512] (由decode_head的三个输出拼成的)
        label, # [2,512,512]
        weight=None,
        ignore_index=-100,  #BaseDecodeHead,loss_by_feat()会复写掉这个值=255
        **kwargs):  # kwargs, 用于接收额外的参数 --> 用于接收loss_by_feat()传递过来的参数
        """Forward function."""
        # logger.warning('进入到 ATL_Loss 的forward()函数')

        # len(label_level_list)=3 [L1级label, L2级label, L3级label] 3个[2,512,512]
        # 产生层级标签 和 层级seg_logits 去算loss
        # L1级标签：0~4 5类
        # L2级标签：0~10 11类
        # L3级标签：0~20 21类

        label_level_list = convert_low_level_label_to_High_level(
            label, classes_map=self.classes_map)

        
        # [0, 5, 11, 21] 
        num_levels_classes = list()
        num_levels_classes.append(0)
        for level_name, high_level_dict in list(self.classes_map.items()):
            num_levels_classes.append(len(high_level_dict))

        # 巧妙地构造出了 [0,5,11,21]-->[[0,5],[5,16],[16,37]]
        classes_range = list()
        for i in range(len(num_levels_classes) - 1):
            classes_range.append([
                num_levels_classes[i],
                num_levels_classes[i] + num_levels_classes[i + 1]
            ])
            num_levels_classes[i + 1] = num_levels_classes[i] + num_levels_classes[i + 1]
        # 最终的classes_range=[[0,5],[5,16],[16,37]]
        # num_levels_classes=[0, 5, 16, 37]

        
        pred_level_list = list()
        for i in range(len(self.classes_map)):
            pred_level_list.append(pred[:, classes_range[i][0]:classes_range[i][1], ...]) # [2,37,512,512] [2,5,512,512] [2,11,512,512] [2,21,512,512]
        # pred_level_list [[2,6,512,512],[2,12,512,512],[2,22,512,512]]
        
        loss_cls_list = list()
        for pred_, label_, loss_ce_name in zip(pred_level_list,
                                               label_level_list,
                                               self.loss_ce_dict):
            # import pdb; pdb.set_trace()
            loss_cls = self.loss_ce_dict[loss_ce_name](
                cls_score=pred_,
                label=label_,
                weight=None,
                avg_factor=None,
                ignore_index=ignore_index,
                **kwargs)
            loss_cls_list.append(loss_cls)  # 3个tensor,这里的数是在cuda的tensor
        # import pdb; pdb.set_trace()
        # 这里的loss最后必须是给一个数值，因为不能动loss_by_feat,所以这里的loss必须是一个数值
        # 但是这里的loss是一个list，所以需要把这个list合并成一个数值
        # import pdb; pdb.set_trace()
        level_weight = [1.0, 1.0, 1.0] # 3个层级的权重
        level_weight = torch.tensor(
            level_weight, device=loss_cls_list[0].device)
        loss_cls_tensor = torch.stack(loss_cls_list)  # 将列表堆叠为一个张量
        weighted_loss_tensor = loss_cls_tensor * level_weight # 给不同层级一个权重
        # import pdb; pdb.set_trace()
        # loss return的是一个值，所以这里需要把loss_cls_list合并成一个值


        # 将多个loss 加起来，作为总的Loss
        loss = torch.sum(weighted_loss_tensor)

        # import pdb; pdb.set_trace()
        return loss* self.loss_weight  #当前loss的权重，主分割头1.0 辅助头0.4 

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






def losses_LTM(pred_level_list, # [B,5,512,512] [B,11,512,512] [B,21,512,512]
               label_level_list,  # [512,512](0-4) [512,512](0,10) [512,512](0,20)   
               num_classes, # 19
               eps=1e-8, 
               gamma=2):

    """计算子类和父类的损失，求和之后返回
    Args:   
        predictions: torch.Tensor, shape=(batch, num_classes+7, h, w), 预测的标签 # [2,37,512,512]
        targets: torch.Tensor, shape=(batch, h, w), 真实的标签[0~19]  # [2,31,512,512]
        targets_top: torch.Tensor, shape=(batch, h, w), 合并成父类的标签[0~7]
        num_classes: int, 类别数
        indices_high: list, 父类对应的子类的的索引范围
    """
    b, _, h, w = pred_level_list[0] # [2, _, 512, 512]
    predictions = torch.sigmoid(predictions.float()) # 让预测值过一个sigmoid函数，变为0-1 (1, 26, 1024, 2048)
    # 【1，26，512，512】---> 每一个像素都都是一个0-1的分布，只有一个最大值，代表其属于哪个类别，其他都近似0 # 但问题这样的话,我是三个分开输出的
                              # 这对于他那个方法来说是可以的。 这里并不能一起做sigmoid，需要三个层级的特征图分开sigmoid。
    # 处理标签
    void_indices = (targets==255) # 无效的位置
    targets[void_indices]=0 # 标签中无效的位置设置为0 ????????【ATL-LOG】 为什么要设置为0
    # 将标签转换为one-hot编码,并且将维度变换为(batch, num_classes, h, w), 例如(1, 19, 1024, 2048)，
    # 一个像素用19维的向量表示[0,0,0,0,0,0,1,0] [1,19,512,512]
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2) 

    void_indices2 = (targets_top==255) # 把父类标签中无效的位置也设置为0
    targets_top[void_indices2]=0  
    # [1,512,512]-->[1,7,512,512]，每个像素一个1*7的向量【0，0，0，0，0，1，0】
    targets_top = F.one_hot(targets_top, num_classes = 7).permute(0,3,1,2) # 父类标签转换为one-hot编码(1,7,1024,2048)
    
    # 计算loss
    # predictions (1, 19+7=26, 1024, 2048)
    MCMA = predictions[:,:num_classes,:,:]   # MCMA提取前19个通道的预测值概率 (1,19,1024,2048)，子类的
    # MCMB 为(1,7,1024,2048)的0矩阵,并放在CUDA上
    MCMB = torch.zeros((b,7,h, w), dtype=predictions.dtype, device=predictions.device)
    for ii in range(7):                             # 0      1     2     3       4       5       6
        indices = indices_high[ii] # indices_high = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]
        # ii=2 的时候，父类【0、1、2、3、4、5、6】         
        # MCMB[:,2:3,:,:] = torch.max(torch.cat([predictions[:,5:8,:,:], predictions[:,19+2:20+2,:,:]], dim=1), 1, True)[0]
        # 把子类的特征图的值，和父类的特征图的值拼起来（如5,6,7类[5,8]都属于同一个父类），这三个和父类的特征图拼起来   # 0-18 子类 19-25 父类
        # 然后把拼起来的特征图在通道维度取最大值出来，放到MCMB中                  #0                  #2                          19
        MCMB[:,ii:ii+1,:,:] = torch.max(torch.cat([predictions[:,indices_high[ii][0]:indices_high[ii][1],:,:], predictions[:,num_classes+ii:num_classes+ii+1,:,:]], dim=1), 1, True)[0]   

    # 这样经历了7次循环之后，得到的MCMB(b,7,h,w),是多个子类+父类中特征值最大的那个

    MCLA = predictions[:,:num_classes,:,:].clone()      # MCLA为(1,19,1024,2048)子类的预测值
    MCLB = predictions[:,num_classes:num_classes+7,:,:] # MCLA为(1,7,1024,2048)父类的预测值, 从19开始到25                          
    
    for ii in range(7):
        indices = indices_high[ii] # indices_high = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]
        for jj in range(indices_high[ii][0], indices_high[ii][1]): # 如 (5,8) jj分别为: 5 6 7
            # MCLA[:,5:6,:,:] = torch.min(torch.cat([predictions[:,5:6,:,:], MCLB[:,2:3,:,:]], dim=1), 1, True)[0]
            # 原先的MCLA的5:6通道，存的是模型预测出的5类别的预测值 5:'pole' 
            # MCLB中的2:3通道，存的是模型预测出的3父类的预测值 [5,8]:'object'
            # 现在把这个两个值拼起来，然后在通道纬度取最小值，放到MCLA中
            MCLA[:,jj:jj+1,:,:] = torch.min(torch.cat([predictions[:,jj:jj+1,:,:],MCLB[:,ii:ii+1,:,:]], dim=1), 1, True)[0]   
    
    # 这样经过了7次循环之后，得到的MCLA(b,26,h,w), 是单个子类和父类中特征值小的那个   # 最不像标签的
    # MCLA和MCMA是一样的，都是子类的预测值，只是MCLA中，子类和父类中特征值小的那个    # MCMB 最像标签的
    # MCLB是父类的预测值，只是MCLB中，子类和父类中特征值大的那个
    
    valid_indices = (~void_indices).unsqueeze(1) # 子类标签 有效的位置，把无效的位置取反，然后在通道纬度上增加一个维度
    num_valid = valid_indices.sum() # 子类标签 有效的位置的个数

    valid_indices2 = (~void_indices2).unsqueeze(1) #  父类标签 有效的位置，把无效的位置取反，然后在通道纬度上增加一个维度
    num_valid2 = valid_indices2.sum() # 父类标签 有效的位置的个数
    #channel_num*sum()/one_channel_valid already has a weight

    # 计算loss 交叉熵损失函数 交叉熵损失函数的定义是：-y*log(y_hat)-(1-y)*log(1-y_hat)
    # loss = 求和｛(-前19通道子类的标签*log(前19通道子类和父类预测之中小的那个) - 
    #           (1-前19通道子类的标签)*log(1-前19通道子类和父类预测之中小的那个))有效位置的索引，true和false，只算有效位置｝除以有效位置的个数/类别数(归一化)

    # MCLA: 【(子类) cat (子类和父类cat最大值)】最小值
    # MCMA: [0-18]子类的特征图，没区别啊
    loss = (( -targets[:,:num_classes,:,:]*torch.log(MCLA+eps) -(1.0-targets[:,:num_classes,:,:])*torch.log(1.0-MCMA+eps))*valid_indices).sum()/num_valid/num_classes
    

    # MCLB: [19-25] 父类的特征图
    # MCMB: 子类和父类特征图 cat后，的最大值
    loss+= ((-targets_top[:,:,:,:]*torch.log(MCLB+eps) -(1.0-targets_top[:,:,:,:])*torch.log(1.0-MCMB+eps)) *valid_indices2).sum()/num_valid2/7

    # 然后放大5倍返回损失？why？
    # 权重调整：在多任务学习或多损失函数的情况下，不同的损失函数可能会有不同的量级。乘以一个系数可以平衡这些损失函数，使它们对最终的总损失有类似的重要性。
    # 训练动态调整：通过乘以一个系数，可以加快或减慢训练过程中的梯度更新速度。一个较大的系数会使梯度变得更大，从而加快参数更新的速度。
    # 强调特定损失：有时我们希望特定的损失在总损失中占据更大的比例，以便模型更加关注特定的目标。通过乘以一个系数，可以增加该损失在总损失中的权重。
    
    return 5*loss

@MODELS.register_module()
class ATL_Loss2(nn.Module):
    """ATL_Loss2 . L1_CE + L2_CE + L3_CE + LTT loss

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
        classes_map (dict): Classes_Map for ATL_loss. Default: None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='atl_loss',
                 avg_non_ignore=False,
                 level_weight = [1.0, 1.0, 1.0], # 3个层级的权重
                 classes_map=None):
        super().__init__()

        # import pdb
        # pdb.set_trace()
        # logger.warning('进入到 ATL_Loss 的 _init_()函数')
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.classes_map = classes_map
        self.level_weight = level_weight

        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self._loss_name = loss_name

        # 构造层级Loss函数
        self.loss_ce_dict = dict()
        if self.classes_map == None:
            # 不分级，默认为 1 级，只要一个交叉熵损失
            self.loss_ce_dict[f'{self._loss_name}_ce'] = CrossEntropyLoss(
                loss_name=f'{self._loss_name}_ce')
            # loss_ce = CrossEntropyLoss(loss_name='atl_loss_ce')
        else:
            self.loss_ce_num = len(self.classes_map)  # 识别分为几级标签, 3 --> L1, L2, L3
            for loss_ce_index in range(self.loss_ce_num):
                self.loss_ce_dict[f'{self._loss_name}_ce_L{loss_ce_index+1}'] = CrossEntropyLoss(
                        loss_name=f'{self._loss_name}_ce_L{loss_ce_index+1}')
        # ✔

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(   # 由uperhead的loss_by_feat()调用
        self,
        pred,  # [2,37,512,512] (由decode_head的三个输出拼成的)
        label, # [2,512,512]
        weight=None,
        ignore_index=-100,  #BaseDecodeHead,loss_by_feat()会复写掉这个值=255
        **kwargs):  # kwargs, 用于接收额外的参数 --> 用于接收loss_by_feat()传递过来的参数
        """Forward function."""
        # logger.warning('进入到 ATL_Loss 的forward()函数')

        # len(label_level_list)=3 [L1级label, L2级label, L3级label] 3个[2,512,512]
        # 产生层级标签 和 层级seg_logits 去算loss
        # L1级标签：0~4 5类
        # L2级标签：0~10 11类
        # L3级标签：0~20 21类

        label_level_list = convert_low_level_label_to_High_level(
            label, classes_map=self.classes_map)

        # [0, 5, 11, 21] 
        num_levels_classes = list()
        num_levels_classes.append(0)
        for level_name, high_level_dict in list(self.classes_map.items()):
            num_levels_classes.append(len(high_level_dict))

        # 巧妙地构造出了 [0,5,11,21]-->[[0,5],[5,16],[16,37]]
        classes_range = list()
        for i in range(len(num_levels_classes) - 1):
            classes_range.append([
                num_levels_classes[i],
                num_levels_classes[i] + num_levels_classes[i + 1]
            ])
            num_levels_classes[i + 1] = num_levels_classes[i] + num_levels_classes[i + 1]
        # 最终的classes_range=[[0,5],[5,16],[16,37]]
        # num_levels_classes=[0, 5, 16, 37]

        
        pred_level_list = list()
        for i in range(len(self.classes_map)):
            pred_level_list.append(pred[:, classes_range[i][0]:classes_range[i][1], ...]) # [2,37,512,512] [2,5,512,512] [2,11,512,512] [2,21,512,512]
        # pred_level_list [[2,5,512,512],[2,11,512,512],[2,21,512,512]]
        
        # 第一个LOSS CE_LOSS 分别去算三个层级各自之间的CE_LOSS
        loss_ce_list = list()
        for pred_, label_, loss_ce_name in zip(pred_level_list,
                                               label_level_list,
                                               self.loss_ce_dict):
            loss_ce = self.loss_ce_dict[loss_ce_name](
                cls_score=pred_,
                label=label_,
                weight=None,
                avg_factor=None,
                ignore_index=ignore_index,
                **kwargs)
            loss_ce_list.append(loss_ce)  # 3个tensor,这里的数是在cuda的tensor
        
        level_weight_cuda = torch.tensor(self.level_weight, device=loss_ce_list[0].device)
        loss_ce_tensor = torch.stack(loss_ce_list)  # 将列表堆叠为一个张量
        weighted_ce_tensor = loss_ce_tensor * level_weight_cuda # 给不同层级一个权重
        # --------------------------------------------------------
        # 第一部分 loss ce loss, 3个层级的loss和
        loss_ce_total = torch.sum(weighted_ce_tensor) 
        # --------------------------------------------------------

        # 第二部分loss LTM loss
        loss_LTM_total = losses_LTM(pred_level_list),





        # import pdb; pdb.set_trace()
        return loss_ce_total* self.loss_weight  #当前loss的权重，主分割头1.0 辅助头0.4 

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
