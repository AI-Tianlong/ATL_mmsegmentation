# Copyright (c) Shanghai AI Lab. All rights reserved.
from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.segmentors.atl_encoder_decoder import ATL_EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

with read_base():
    from .._base_.datasets.atl_0_paper_eurocrops_s2_65class import *
    from .._base_.default_runtime import *
    from .._base_.models.mask2former_beit_potsdam import *
    from .._base_.schedules.schedule_80k import *

# 一定记得改类别数！！！！！！！！！！！！！！！！！！！！！！！

# reduce_zero_label = True 所以是24类
num_classes = 65  # loss 要用，也要加 # 加上背景是25类

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误

crop_size = (512, 512)
# pretrained = None
pretrained = '/opt/AI-Tianlong/checkpoints/atl_s2_checkpoint/10_channel_beitv2_large_patch16_224_pt1k_ft21k_BGR.pth'
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
            feat_channels=1024,  # 256 类别多的话：1024
            out_channels=1024,  # 256 类别多的话：1024
            num_classes=num_classes,
            num_queries=200,  # 100 200
            num_transformer_feat_level=3,
            align_corners=False,
            pixel_decoder=dict(
                type=
                MSDeformAttnPixelDecoder,  # MSDeformAttnPixelDecoder #用的自己实现的，vit-adapter
                num_outs=
                3,  # mmdet的在mmdet-->models-->layers-->msdeformattn_pixel_decoder.py
                norm_cfg=dict(type=GN, num_groups=32),
                act_cfg=dict(type=ReLU),
                encoder=dict(
                    # type=DetrTransformerEncoder,# DetrTransformerEncoder # 用的mmdet实现的
                    num_layers=6,
                    layer_cfg=dict(
                        # type=DetrTransformerEncoderLayer,# DetrTransformerEncoder绑定了
                        self_attn_cfg=dict(
                            # type=MultiScaleDeformableAttention,  # DetrTransformerEncoderLayer绑定了
                            embed_dims=1024,  # 512 1024
                            num_heads=32,  # 8 32
                            num_levels=3,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=True,
                            norm_cfg=None,
                            init_cfg=None),
                        ffn_cfg=dict(
                            # type=FFN, #DetrTransformerEncoderLayer绑定了
                            embed_dims=1024,  #256 1024
                            feedforward_channels=4096,  #2048 4096
                            num_fcs=2,
                            ffn_drop=0.0,
                            # with_cp=True,
                            act_cfg=dict(type=ReLU, inplace=True))),
                    init_cfg=None),
                positional_encoding=dict(
                    # type=SinePositionalEncoding, # ATL 的 MSDeformAttnPixelDecoder 默认是这个
                    num_feats=512,  # 128 512
                    normalize=True),
                init_cfg=None),
            enforce_decoder_input_project=False,
            positional_encoding=dict(
                #  type=SinePositionalEncoding, # Mask2FormerHead写死了
                num_feats=512,  # 128 512
                normalize=True),
            transformer_decoder=dict(
                # type=DetrTransformerDecoder,  # Mask2FormrtHead--->DetrTransformerDecoder写死了
                return_intermediate=True,
                num_layers=9,
                layer_cfg=dict(
                    # type=DetrTransformerDecoderLayer,  # DetrTransformerDecoder 写死了
                    self_attn_cfg=dict(
                        # type=MultiheadAttention,  # DetrTransformerDecoderLayer 写死了
                        embed_dims=1024,  # 256 1024
                        num_heads=32,  # 8 32
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=1024,  #256 1024
                        num_heads=32,  #8 32
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=1024,  # 256 1024
                        feedforward_channels=4096,  #2048 4096
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
                        dict(
                            type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
                    ]),
                sampler=dict(type=MaskPseudoSampler))),
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

# load_from =
load_from = None
default_hooks.update(
    dict(logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False)))
