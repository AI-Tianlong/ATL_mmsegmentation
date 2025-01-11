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
from mmseg.engine.optimizers.layer_decay_optimizer_constructor import ATL_LayerDecayOptimizerConstructor
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor


# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
from mmseg.models.segmentors.atl_multi_encoder_multi_decoder import ATL_Multi_Encoder_Multi_Decoder
from mmseg.models.segmentors.atl_multi_encoder_multi_decoder_cfglist import ATL_Multi_Encoder_Multi_Decoder_cfglist
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.data_preprocessor_atl import ATL_SegDataPreProcessor
# Backbone
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.backbones import ViTAdapter
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.atl_hiera_37_uper_head_multi_convseg import ATL_hiera_UPerHead_Multi_convseg
from mmseg.models.decode_heads.atl_multi_encoder_multi_decoder_uperhead import ATL_Multi_Encoder_Multi_Decoder_UPerHead
from mmseg.models.decode_heads.atl_fcn_head_multi_embedding import ATL_multi_embedding_FCNHead
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# Evaluation
from mmseg.evaluation import IoUMetric

with read_base():
    # from ..._base_.datasets.a_atl_0_paper_multi_GF2_Google_S2_19class import *
    from ..._base_.datasets.a_atl_0_paper_S2_crop10m_4class import *
    from ..._base_.default_runtime import *
    # from ..._base_.models.upernet_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

find_unuser_parameters = False

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

base_L1_num_classes = 5  # number of L1 Level label   # 5
base_L2_num_classes = 10  # number of L1 Level label  # 11  5+11+21=37类
base_L3_num_classes = 19  # number of L1 Level label  # 21


downstream_L1_num_classes = 2  # number of L1 Level label  # 农田 / 非农田  # 来自于base domain的 mask
downstream_L2_num_classes = 3  # number of L1 Level label  # 目标作物农田——非目标作物农田(base的mask基础上) / 非农田 （base）
downstream_L3_num_classes = 4  # number of L1 Level label  # 水稻 大豆 玉米 / 其他
# 总的类别数，包括背景，L1+L2+L3级标签数

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (512, 512)
pretrained_3chan = 'checkpoints/2-对比实验的权重/vit-adapter-offical/BEiT/beitv2_large_patch16_224_pt1k_ft21k-3chan.pth'
pretrained_4chan = 'checkpoints/2-对比实验的权重/vit-adapter-offical/BEiT/beitv2_large_patch16_224_pt1k_ft21k-4chan.pth'
pretrained_10chan = 'checkpoints/2-对比实验的权重/vit-adapter-offical/BEiT/beitv2_large_patch16_224_pt1k_ft21k-10chan.pth'

# 加载第一阶段训练好的Multi-Modal model
# load_from = 'work_dirs/BEiTAdapter-L-UpetNet-multi-多Encoder-多Decoder-普通upernet-20250107/iter_80000.pth'

data_preprocessor = dict(
        type=SegDataPreProcessor,
        mean = None,
        std = None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

# Encoder Config
backbone_config = dict(
        type=BEiTAdapter,
        img_size=512,
        patch_size=16,
        embed_dim=1024,  # B:768 L:1024
        in_channels=3,  
        depth=24,       # B:12 L:24
        num_heads=16,   # B:12 L:16
        deform_num_heads=16, # Adapter的参数： B:12 L:16
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=False,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]], # large
        # interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],  # base
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_3chan)) # 不加预训练权重

# Decoder Config
decode_head_config=dict(
        # type=ATL_Multi_Encoder_Multi_Decoder_UPerHead,
        type=UPerHead,
        in_channels=[1024, 1024, 1024, 1024],  # 和vit的结构保持一致，large的话1024
        # in_channels=[768, 768, 768, 768],  # 和vit的结构保持一致，large的话1024
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,   # 768  1024 这是个 啥参数来着？
        dropout_ratio=0.1,
        num_classes=base_L3_num_classes,
        # num_classes_level_list=[5,10,19], #37
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type=ATL_Hiera_Loss_convseg, num_classes=[5,10,19], loss_weight=1.0)),
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0))


# backbone config
backbone_MSI_3chan={**backbone_config, 'in_channels': 3, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_3chan)}
backbone_MSI_4chan={**backbone_config, 'in_channels': 4, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_4chan)}
backbone_MSI_10chan={**backbone_config, 'in_channels': 10, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_10chan)}

# decode_head_config
decode_head_MSI_3chan=decode_head_config
decode_head_MSI_4chan=decode_head_config
decode_head_MSI_10chan=decode_head_config
decode_head_10chan_crop10m={**decode_head_config, 'num_classes': downstream_L3_num_classes}

model=dict(
        type=ATL_Multi_Encoder_Multi_Decoder_cfglist,
        data_preprocessor=data_preprocessor,

        # backbone 3个   全冻结
        backbone_config_base_list=dict(backbone_MSI_3chan=backbone_MSI_3chan,
                                       backbone_MSI_4chan=backbone_MSI_4chan,
                                       backbone_MSI_10chan=backbone_MSI_10chan),
        # decode_head 3个 全冻结
        decode_head_config_base_list=dict(decode_head_MSI_3chan=decode_head_MSI_3chan,
                                          decode_head_MSI_4chan=decode_head_MSI_4chan,
                                          decode_head_MSI_10chan=decode_head_MSI_10chan), 

        # 只有这个有梯度 
        decode_head_config_downstream = decode_head_10chan_crop10m,

        auxiliary_head=dict(
            type=FCNHead,
            in_channels=1024, # 和上面的768 保持统一
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=downstream_L3_num_classes, #21
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
        test_cfg=dict(mode='whole'))

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
    constructor=ATL_LayerDecayOptimizerConstructor,
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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
