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


# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.backbones import ViTAdapter
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.atl_hiera_37_uper_head_multi_convseg import ATL_hiera_UPerHead_Multi_convseg
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# Evaluation
from mmseg.evaluation import IoUMetric

with read_base():
    from ..._base_.datasets.a_atl_0_paper_5b_GF2_19class import *
    from ..._base_.default_runtime import *
    from ..._base_.models.mask2former_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

find_unuser_parameters = False

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 5  # number of L1 Level label   # 5
L2_num_classes = 10  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 19  # number of L1 Level label  # 21

# 总的类别数，包括背景，L1+L2+L3级标签数

num_classes = L1_num_classes + L2_num_classes + L3_num_classes # 37 

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (512, 512)
pretrained = 'checkpoints/2-对比实验的权重/vit-adapter-offical/BEiT/beitv2_large_patch16_224_pt1k_ft21k-4chan.pth'

data_preprocessor = dict(
        type=SegDataPreProcessor,
        mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
        std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

model=dict(
        type=EncoderDecoder,
        data_preprocessor=data_preprocessor,
        # pretrained=pretrained,
        backbone=dict(
            type=BEiTAdapter,
            img_size=512,
            patch_size=16,
            embed_dim=1024,
            in_channels=4,  # 4个波段
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
            init_cfg=dict(type='Pretrained', checkpoint=pretrained) # 不加预训练权重
        ),  #backbone 完全一样
    
        decode_head=dict(
            type=Mask2FormerHead,  # 千万别自己实现，全是坑
            in_channels=[1024, 1024, 1024, 1024],  # BEiT-Adapter [1024,1024,1024,1024]
            in_index=[0, 1, 2, 3],
            feat_channels=256,  # 类别多的话：1024
            out_channels=256,   # 类别多的话：1024
            num_classes=num_classes,
            num_queries=100,
            num_transformer_feat_level=3,
            align_corners=False,
            pixel_decoder=dict(
                type=MSDeformAttnPixelDecoder,  # MSDeformAttnPixelDecoder #用的自己实现的，vit-adapter
                num_outs=3,  # mmdet的在mmdet-->models-->layers-->msdeformattn_pixel_decoder.py
                norm_cfg=dict(type=GN, num_groups=32),
                act_cfg=dict(type=ReLU),
                encoder=dict(
                    # type=DetrTransformerEncoder,# DetrTransformerEncoder # 用的mmdet实现的
                    num_layers=6,
                    layer_cfg=dict(
                        # type=DetrTransformerEncoderLayer,# DetrTransformerEncoder绑定了
                        self_attn_cfg=dict(
                            # type=MultiScaleDeformableAttention,  # DetrTransformerEncoderLayer绑定了
                            embed_dims=256, # 1024
                            num_heads=8,    # 32
                            num_levels=3,  
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=True,
                            norm_cfg=None,
                            init_cfg=None),
                        ffn_cfg=dict(
                            # type=FFN, #DetrTransformerEncoderLayer绑定了
                            embed_dims=256,            #1024
                            feedforward_channels=2048, #4096
                            num_fcs=2,
                            ffn_drop=0.0,
                            # with_cp=True,
                            act_cfg=dict(type=ReLU, inplace=True))),
                    init_cfg=None),
                positional_encoding=dict(
                    # type=SinePositionalEncoding, # ATL 的 MSDeformAttnPixelDecoder 默认是这个
                    num_feats=128, # 512
                    normalize=True),
                init_cfg=None),
            enforce_decoder_input_project=False,
            positional_encoding=dict(
                #  type=SinePositionalEncoding, # Mask2FormerHead写死了
                num_feats=128,  #512
                normalize=True),
            transformer_decoder=dict(
                # type=DetrTransformerDecoder,  # Mask2FormrtHead--->DetrTransformerDecoder写死了
                return_intermediate=True,
                num_layers=9,
                layer_cfg=dict(
                    # type=DetrTransformerDecoderLayer,  # DetrTransformerDecoder 写死了
                    self_attn_cfg=dict(
                        # type=MultiheadAttention,  # DetrTransformerDecoderLayer 写死了
                        embed_dims=256, # 1024
                        num_heads=8,    # 32
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,  #1024
                        num_heads=8,     #32
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,     # 1024
                        feedforward_channels=2048,  #4096
                        num_fcs=2,
                        act_cfg=dict(type=ReLU, inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True)),
                init_cfg=None),
            loss_cls=dict(
                type=CrossEntropyLoss,
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * num_classes + [0.1]),
            loss_mask=dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type=DiceLoss,
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type=HungarianAssigner,
                    match_costs=[
                        dict(type=ClassificationCost, weight=2.0),
                        dict(
                            type=CrossEntropyLossCost,
                            weight=5.0,
                            use_sigmoid=True),
                        dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
                    ]),
                sampler=dict(type=MaskPseudoSampler))),
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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=4000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
