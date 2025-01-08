
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW


from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN

# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
from mmseg.models.segmentors.atl_multi_encoder_multi_decoder import ATL_Multi_Encoder_Multi_Decoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.data_preprocessor_atl import ATL_SegDataPreProcessor
# Backbone
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.backbones import ViTAdapter
from mmseg.models.backbones import MSCAN
# DecodeHead
from mmseg.models.decode_heads.atl_multi_encoder_multi_decoder_uperhead import ATL_Multi_Encoder_Multi_Decoder_UPerHead
from mmseg.models.decode_heads.atl_fcn_head_multi_embedding import ATL_multi_embedding_FCNHead
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.decode_heads.atl_multi_encoder_multi_decoder_ham_dead import ATL_Multi_Encoder_Multi_Decoder_LightHamHead
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# Evaluation
from mmseg.evaluation import IoUMetric

with read_base():
    from ..._base_.datasets.a_atl_0_paper_multi_GF2_Google_S2_19class import *
    from ..._base_.default_runtime import *
    # from ..._base_.models.upernet_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

find_unuser_parameters = False
ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)
# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 5  # number of L1 Level label   # 5
L2_num_classes = 10  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 19  # number of L1 Level label  # 21

# 总的类别数，包括背景，L1+L2+L3级标签数

# num_classes = L1_num_classes + L2_num_classes + L3_num_classes # 37 

num_classes = L3_num_classes

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (512, 512)
pretrained_3chan = 'checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_3chan.pth'
pretrained_4chan = 'checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_4chan.pth'
pretrained_10chan = 'checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_10chan.pth'

data_preprocessor = dict(
        type=ATL_SegDataPreProcessor,
        mean = None,
        std = None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

# Encoder Config
backbone_config = dict(
        type=MSCAN,
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=SyncBN, requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_3chan))

# Decoder Config
decode_head_config=dict(
        type=ATL_Multi_Encoder_Multi_Decoder_LightHamHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True))

model=dict(
        type=ATL_Multi_Encoder_Multi_Decoder,
        data_preprocessor=data_preprocessor,

        backbone_MSI_3chan={**backbone_config, 'in_channels': 3, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_3chan)},
        backbone_MSI_4chan={**backbone_config, 'in_channels': 4, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_4chan)},
        backbone_MSI_10chan={**backbone_config, 'in_channels': 10, 'init_cfg': dict(type='Pretrained', checkpoint=pretrained_10chan)},
        decode_head_MSI_3chan=decode_head_config,
        decode_head_MSI_4chan=decode_head_config,
        decode_head_MSI_10chan=decode_head_config,
        # auxiliary_head=dict(
        #     type=ATL_multi_embedding_FCNHead,
        #     in_channels=1024, # 和上面的768 保持统一
        #     in_index=3,
        #     channels=256,
        #     num_convs=1,
        #     concat_input=False,
        #     dropout_ratio=0.1,
        #     num_classes=L3_num_classes, #21
        #     norm_cfg=norm_cfg,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
        test_cfg=dict(mode='whole'))

# dataset config
train_pipeline = [
    dict(type=LoadMultiRSImageFromFile_with_data_preproocess),
    dict(type=ATL_multi_embedding_LoadAnnotations),
    # dict(
    #     type=RandomChoiceResize,
    #     scales=[int(x * 0.1 * 512) for x in range(5, 21)],
    #     resize_type=ResizeShortestEdge,
    #     max_size=2048),
    # dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=ATL_3_embedding_PackSegInputs)
]
train_dataloader.update(dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
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
