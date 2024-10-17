# Copyright (c) OpenMMLab. All rights reserved.
import logging
import math
# from mmcv.ops.multi_scale_deform_attn \
#     import MultiScaleDeformableAttention as MSDeformAttn
import pdb
import warnings
from functools import partial

import mmcv
# =====================================================================================
# ========================== vit-adapter MSDeformAttn.py ==========================
import MultiScaleDeformableAttention as MSDA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)
from PIL import Image
from scipy import interpolate
from timm.models.layers import DropPath, drop_path, to_2tuple, trunc_normal_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.init import constant_, normal_, xavier_uniform_
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.registry import MODELS

# 为了和预训练保持一致，还是选择从mmpretrain中加载 VisionTransformer，可以权重对应上
# from mmseg.models.backbones.vit import VisionTransformer
from mmpretrain.models.backbones.vision_transformer import VisionTransformer

# 在做多尺度的时候，需要对输入的特征图进行插值，这里是插值的函数

class MSDeformAttnFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
                                             value_level_start_index,
                                             sampling_locations,
                                             attention_weights,
                                             ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
        sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(
            N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            'invalid input for _is_power_of_2: {} (type: {})'.format(
                n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):

    def __init__(self,
                 d_model=256,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model,
                                          n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model,
                                           n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()
        # 到这都一样

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                       self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape  # (1,1024,1024),([1, 5376, 1024])
        N, Len_in, _ = input_flatten.shape  # ([1, 5376, 1024]),(1,1024,1024)交替

        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).\
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes,
                                            input_level_start_index,
                                            sampling_locations,
                                            attention_weights,
                                            self.im2col_step)
        output = self.output_proj(output)
        return output


# =====================================================================================
# ============================ vit-adapter BEiT.py ====================================


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'



# =====================================================================================
# ============================ ViT-Adapter-MsDeformAttn.py ============================


# =====================================================================================
# ============================ ViT-Adapter-adapter_modules.py =========================
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),   # 512 --> 64  [64,64]
                                      (h // 16, w // 16), # 512 --> 32  [32,32]
                                      (h // 32, w // 32)],# 512 --> 16  [16,16]
                                     dtype=torch.long,
                                     device=x.device)
    # import pdb;pdb.set_trace()    
    # .new_zeros((1, )) 生成一个1维的0张量,与spatial_shapes的属性相同                    
    # .prod(1)计算每一行的乘积 [64*64, 32*32, 16*16]=[4096 1024 256]
    # .cumsum(0)计算元素的累加和[4096 1024 256] --> [4096 5120 5376]
    # [:-1] 提取到倒数第二个元素 [4096 5120 5376] --> [4096 5120]
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) #[0, 4096, 5120] 层级开始的index？
    reference_points = get_reference_points([(h // 16, w // 16)], x.device) # patch16 或 patch_8 [1,1024,1,2] 1个batch，1024个点，x,y 两个坐标

    deform_inputs1 = [reference_points, spatial_shapes, level_start_index] 

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], # [32,32]
                                     dtype=torch.long,
                                     device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0, 1024]
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)  # [1,5376,1,2] 1个batch，1024个点，1个通道，2个坐标
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    # [1,1024,1,2][[64,64],[32,32],[16,16]][0,4096,5120]  [1,5376,1,2][[32,32]][0,1024]
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2,
                                                    W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H,
                                                         W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2,
                                                   W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat),
                spatial_shapes, level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio)
        self.gamma = nn.Parameter(
            init_values * torch.ones(dim), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat),
                spatial_shapes, level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False,
                 with_cp=False):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp)
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    with_cp=with_cp) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            # print(f'【ATL-LOG-vit_adapter-518 行 】')
            # print(f'x.shape {x.shape}')
            # print(f'H {H}')
            # print(f'W {W}')
            # print(f'blk {blk}')
            # x = blk(x, H, W)  # VIT的 Transformer 块，都没有HW
            x = blk(x)  # VIT的 Transformer 块

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W)
        return x, c


class InteractionBlockWithCls(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False,
                 with_cp=False):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp)
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    with_cp=with_cp) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W)
        return x, c, cls


class SpatialPriorModule(nn.Module):

    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, in_chans=3):
        super().__init__()
        self.with_cp = with_cp

        # stem: [2,10,512,512]-->[2,64,128,128]
        self.stem = nn.Sequential(*[
            nn.Conv2d(
                in_chans,
                inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[  # [2,64,128,128]-->[2,128,64,64]
            nn.Conv2d(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[ # [2,128,64,64]-->[2,256,32,32] 
            nn.Conv2d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[ # [2,256,32,32] --> [2,256,16,16]
            nn.Conv2d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(  # [2,64,128,128]-->[2,1024,128,128]
            inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(  # [2,128,64,64]-->[2,1024,64,64]
            2 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc3 = nn.Conv2d(
            4 * inplanes,   # [2,256,32,32] -->[2,1024,32,32]
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc4 = nn.Conv2d(
            4 * inplanes,  # [2,256,16,16]-->[2,1024,16,16]
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)   # [2,64,128,128]
            c2 = self.conv2(c1) # [2, 128, 64, 64]
            c3 = self.conv3(c2) # [2, 256, 32, 32]
            c4 = self.conv4(c3) # [2, 256, 16, 16]
            # import pdb;pdb.set_trace()
            c1 = self.fc1(c1) # [2, 1024, 128,128]
            c2 = self.fc2(c2) # [2, 1024, 64, 64]
            c3 = self.fc3(c3) # [2, 1024, 32, 32]
            c4 = self.fc4(c4) # [2, 1024, 16, 16]

            bs, dim, _, _ = c1.shape # [2, 1024, 128,128]
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s  # [2,4096,1024]
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s # [2,1024,1024]
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s # [2,256,1024]

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

# ===================================== vit-adapter BEiTAdapter.py =====================================

@MODELS.register_module()
class ViTAdapter(VisionTransformer): 
    # 从mmpretrain中继承，权重什么的可以和预训练的对应上
    def __init__(self,
                 img_size=512,
                 pretrain_size=224, #重新reshape pos_embedding.
                 patch_size = 16,
                 in_channels=10,
                 arch='l',
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 with_cffn=True,
                 interaction_indexes=None,
                 add_vit_feature=True,
                 with_cp=False,
                 init_cfg=None,
                 drop_path_rate=0.3,
                 use_extra_extractor=True,
                 norm_cfg=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 *args,
                 **kwargs):
        print(f'【ATL-LOG】 进入到 ViTAdapter 初始化')


        super().__init__(
            arch=arch,
            in_channels=in_channels,
            init_cfg=init_cfg,
            img_size=img_size,
            patch_size = patch_size,
            *args,
            **kwargs)
        print(f'【ATL-LOG】初始化父类 VisionTransformer 完成')
        
        
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
    
        # self.num_classes = 80
        self.cls_token = None
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes # [[0, 5], [6, 11], [12, 17], [18, 23]], # large
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dims

        # 没懂这个 level_embed 是干嘛的
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim)) # [3, 1024] 3个level，每个level一个1024维的嵌入, 为什么是1024维度？
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, # 64
            embed_dim=embed_dim,   # 1024
            with_cp=False,
            in_chans=in_channels)  # 10
        
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 
                                  else False) and use_extra_extractor),
                with_cp=with_cp) 
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.sync_norm1 = nn.SyncBatchNorm(embed_dim) #这个syncBatchNorm会不会有问题，单卡没法用？
        self.sync_norm2 = nn.SyncBatchNorm(embed_dim)  # 
        self.sync_norm3 = nn.SyncBatchNorm(embed_dim)
        self.sync_norm4 = nn.SyncBatchNorm(embed_dim)
        # return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
        # RuntimeError: Given normalized_shape=[1024], expected input with shape [*, 1024], but got input of size[2, 1024, 128, 128]
        

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    # def init_weights(self):
    #     pass
    
    def print_all_parameters(self):
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter value: {param.data}\n")


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

                        #[1,32*32+1,1024]          
    # # 512的话，pos_embed为[1,1024+1,1024]
    # def _get_pos_embed(self, pos_embed, H, W):  # pos_embed原来是 [1, 196, 768]
    #     pos_embed = pos_embed.reshape(  # [1,1024,1024]-->[1,14,14, ]
    #         1,   
    #         self.pretrain_size[0] // patch_size, #patch_size=16
    #         self.pretrain_size[1] // patch_size, 
    #         -1).permute(0, 3, 1, 2)
    #     pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
    #         reshape(1, -1, H * W).permute(0, 2, 1)
    #     return pos_embed #[1,768,32*2=1024个]

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]  #层级的一个嵌入
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # import pdb;pdb.set_trace()
        # print(f'[ATL-LOG-BEiT-Adapter-forward] x.shape: {x.shape} x.dtype: {x.dtype} x.device: {x.device}')
        # [reference_points, spatial_shapes, level_start_index] 
        deform_inputs1, deform_inputs2 = deform_inputs(x.contiguous()) # Spatial Prior Module 
        # deform_inputs1 [[1,1024,1,2] [3,2] [3]]
        # deform_inputs2 [[1,5376,1,2] [1,2] [1]]

        # import pdb;pdb.set_trace()
        # SPM forward
        c1, c2, c3, c4 = self.spm(x) # [2,1024,128,128] # [2,4096,1024] [2,1024,1024] [2,256,1024]
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1) #  [2, 5376, 1024]
        # import pdb;pdb.set_trace()
        # Patch Embedding forward #一个512*512的图，打成1024个1024维度的patch，一个patch有16*16*10=2560个像素
        x, out_size = self.patch_embed(x) # [2,1024,1024] (32 32)  #如果是patch=8的话，[2,4096,1024] (64,64)

        # import pdb;pdb.set_trace()
        H, W = out_size
        bs, n, dim = x.shape

        # import pdb;pdb.set_trace()
        # 会报错 RuntimeError: shape '[1, 14, 14, -1]' is invalid for input of size 1048576（32*32*1024）
        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W) #有问题，自动去resize吧。
                                          #[1,32*32,1024]

        pos_embed = self.pos_embed[:, 1:] #[1,1024,1024]  | [1,4096,1024]

        x = self.drop_after_pos(x + pos_embed)  # Dropout #[1,1024,1024]  | [1,4096,1024]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]

            x, c = layer(x, c, 
                         self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
            # outs.append(x.transpose(1, 2).contiguous().view(bs, dim, H, W)) ATL瞎改的

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(
                x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(
                x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(
                x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        # import pdb;pdb.set_trace()
        f1 = self.sync_norm1(c1) # c1:[2, 1024, 128, 128]
        f2 = self.sync_norm2(c2) # c2:[2, 1024, 64, 64]
        f3 = self.sync_norm3(c3) # c3:[2, 1024, 32, 32]
        f4 = self.sync_norm4(c4) # c3:[2, 1024, 16, 16]
        return [f1, f2, f3, f4]
