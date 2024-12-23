# 2024-09-02 测试, 可以跑通,loss-从5开始降低。
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW


from mmseg.datasets.transforms.loading import (LoadSingleRSImageFromFile,
                                               ATL_multi_embedding_LoadAnnotations,
                                               LoadSingleRSImageFromFile_with_data_preproocess,
                                               LoadMultiRSImageFromFile_with_data_preproocess)


from mmseg.engine.optimizers import LayerDecayOptimizerConstructor

# EncoderDecoder
from mmseg.models.segmentors.atl_encoder_decoder_multi_embedding import ATL_Multi_Embedding_EncoderDecoder
from mmseg.models.segmentors.atl_encoder_decoder_multi_embedding_multi_decoder import ATL_Multi_Embedding_Multi_Decoder_EncoderDecoder
# data_preprocessor
from mmseg.models.data_preprocessor_atl import ATL_SegDataPreProcessor
# train pipline
from mmseg.datasets.transforms.formatting import PackSegInputs, ATL_3_embedding_PackSegInputs

# backbone
from mmseg.models.backbones.atl_vit_adapter_multi_embedding import ViTAdapter_multi_embedding
# decode_head auxiliary_head
from mmseg.models.decode_heads.atl_uper_head_multi_embedding import ATL_multi_embedding_UPerHead
from mmseg.models.decode_heads.atl_fcn_head_multi_embedding import ATL_multi_embedding_FCNHead
# loss & validation
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.evaluation import IoUMetric

with read_base():
    from .._base_.datasets.atl_0_paper_new_5b_GF_Google_S2_19class import *
    from .._base_.default_runtime import *
    from .._base_.models.upernet_beit_potsdam import *
    from .._base_.schedules.schedule_80k import *

find_unuser_parameters = True

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 5  # number of L1 Level label   # 5
L2_num_classes = 10  # number of L1 Level label  # 11 5+11+21=37类
L3_num_classes = 19  # number of L1 Level label  # 21

# 总的类别数，包括背景，L1+L2+L3级标签数

num_classes = L1_num_classes + L2_num_classes + L3_num_classes # 37

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (512, 512)
pretrained = '/data/AI-Tianlong/Checkpoints/2-对比实验的权重/vit-adapter-offical/mmpretrainformat-4chan-ViT-Adapter-Aug-L.pth'


# 这个data_preprocessor是在mmengine中去定义的flow,需要改mmengine的东西啊？
# 不需要改mmengine，这里的type_是mmseg定义的SegDataPreProcessor，而不是data_preprocessor
data_preprocessor = dict(               # 将归一化的操作，写到LoadSingleRSImage那里
        type=ATL_SegDataPreProcessor,
        mean=None,
        std=None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

model = dict(
    type=ATL_Multi_Embedding_Multi_Decoder_EncoderDecoder,
    # level_classes_map=S2_5B_Dataset_21Classes_Map_nobackground,  # 注意传参！！
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=ViTAdapter_multi_embedding,
        img_size=512,
        patch_size=16,
        arch='l', # embed_dims=1024, num_layers=24, num_heads=16
        in_channels=[3, 4, 10],  # 4个波段
        # mlp_ratio=4,  # mpl的通道数，是4倍的enbed_dim
        qkv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        # interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], # base
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) # 不加预训练权重
        # frozen_exclude=None,
    ),  
    decode_head_MSI_3chan=dict(
        type=ATL_multi_embedding_UPerHead,
        in_channels=[1024, 1024, 1024, 1024],  # 和vit的结构保持一致，large的话1024
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,   # 这是个 啥参数来着？
        dropout_ratio=0.1,
        num_classes=L3_num_classes, #37
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    decode_head_MSI_4chan=dict(
        type=ATL_multi_embedding_UPerHead,
        in_channels=[1024, 1024, 1024, 1024],  # 和vit的结构保持一致，large的话1024
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,   # 这是个 啥参数来着？
        dropout_ratio=0.1,
        num_classes=L3_num_classes, #37
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    decode_head_MSI_10chan=dict(
        type=ATL_multi_embedding_UPerHead,
        in_channels=[1024, 1024, 1024, 1024],  # 和vit的结构保持一致，large的话1024
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,   # 这是个 啥参数来着？
        dropout_ratio=0.1,
        num_classes=L3_num_classes, #37
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=ATL_multi_embedding_FCNHead,
        in_channels=1024, # 和上面的768 保持统一
        # in_channels=768, # 和上面的768 保持统一
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=L3_num_classes, #21
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# dataset config
train_pipeline = [
    dict(type=LoadMultiRSImageFromFile_with_data_preproocess),
    dict(type=ATL_multi_embedding_LoadAnnotations),
    dict(type=ATL_3_embedding_PackSegInputs)
]
train_dataloader.update(
    batch_size=2,
    num_workers=4,  
    dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optimizer = dict(
    type=AdamW,
    lr=6e-6, # 2e-5
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=4000, max_keep_ckpts=20),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

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
