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

from mmseg.models.data_preprocessor import SegDataPreProcessor

from mmseg.evaluation import ATL_IoUMetric #多卡时有问题
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.decode_heads.atl_fcn_head import ATL_FCNHead
from mmseg.models.decode_heads.uper_head import UPerHead

from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_encoder_decoder import ATL_EncoderDecoder
from mmseg.models.backbones import ViTAdapter
from mmseg.models.backbones import MSCAN

from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.decode_heads.atl_uper_head import ATL_UPerHead, ATL_UPerHead_fenkai
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses.atl_loss import ATL_Loss, ATL_Loss2, S2_5B_Dataset_21Classes_Map_nobackground
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.evaluation import IoUMetric


with read_base():
    from .._base_.datasets.atl_0_paper_5b_s2_19class import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_160k import *


# model settings
checkpoint_file = '/data/AI-Tianlong/Checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_10_chan_BGR.pth'   # noqa
ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=MSCAN,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=10,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=SyncBN, requires_grad=True)),
    decode_head=dict(
        type=LightHamHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=21,
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
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# # dataset settings
# train_dataloader = dict(batch_size=16)

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

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=ATL_IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)