# 2024-09-02 测试, 可以跑通,loss-从5开始降低。

from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor
from mmseg.evaluation import ATL_IoUMetric
from mmseg.evaluation import IoUMetric
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.decode_heads.atl_uper_head import ATL_UPerHead
from mmseg.models.decode_heads.atl_fcn_head import ATL_FCNHead
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.losses.atl_loss import ATL_Loss, S2_5B_Dataset_22Classes_Map
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.segmentors.atl_encoder_decoder import ATL_EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

with read_base():
    from .._base_.datasets.atl_0_paper_5b_s2_22class import *
    from .._base_.default_runtime import *
    from .._base_.models.upernet_beit_potsdam import *
    from .._base_.schedules.schedule_80k import *

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 6  # number of L1 Level label
L2_num_classes = 12  # number of L1 Level label
L3_num_classes = 22  # number of L1 Level label

# 总的类别数，包括背景，L1+L2+L3级标签数
# num_classes = 22
num_classes = L1_num_classes + L2_num_classes + L3_num_classes  # 这个不怎么用到?

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误

crop_size = (512, 512)
# pretrained = None
pretrained = None
# pretrained = None
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        #       B2       B3      B4      B5      B6      B7     B8       B8A     B11    B12
        # mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # std= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # bgr_to_rgb=True,
        mean=None,
        std=None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size))

model.update(
    dict(
        type=ATL_EncoderDecoder,
        level_classes_map=S2_5B_Dataset_22Classes_Map,  # 注意传参！！
        pretrained=pretrained,
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type=BEiTAdapter,
            img_size=512,
            patch_size=16,
            embed_dim=1024,
            in_channels=10,  # 4个波段
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            init_values=1e-6,
            drop_path_rate=0.3,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            with_cp=False,  # set with_cp=True to save memory
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        ),  #backbone 完全一样
        decode_head=dict(
            type=ATL_UPerHead,
            in_channels=[1024, 1024, 1024, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6), 
            channels=1024,   # 最后 [2,1024,128,128]-->[2,22,128,128]
            dropout_ratio=0.1,
            num_classes=num_classes, #22，和con_seg，和out_channels对应。在basedecodehead里 这里要设置成40，否则slide推理的时候都会有问题。
            num_level_classes=[L1_num_classes, L2_num_classes, L3_num_classes],  # 这里需要和loss的map对应上
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=ATL_Loss,
                reduction='sum',  # 消融1-->去掉loss里面的权重，sum->mean
                use_sigmoid=False,
                loss_weight=1.0,
                classes_map=S2_5B_Dataset_22Classes_Map)),
        auxiliary_head=dict(
            type=FCNHead,
            in_channels=1024,
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=L3_num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))))
    

# dataset config
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 512) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
train_dataloader.update(dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optimizer = dict(
    type=AdamW,
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))

# learning policy
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        # begin=0,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

load_from = None
# load_from = '/opt/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/20240920-s2_5B_S2-beit_uperner_large-b4x2-80k-ATL调试paper/iter_24000.pth'
default_hooks.update(
    dict(logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False)))

# test_dataloader.update(
#     dict(
#         dataset=dict(
#         data_root='data/0-atl-paper-s2/tiny-test',
#         data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
#         pipeline=test_pipeline)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
