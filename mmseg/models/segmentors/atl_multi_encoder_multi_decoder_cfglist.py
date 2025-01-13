# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from ..utils import resize
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
import copy

@MODELS.register_module()
class ATL_Multi_Encoder_Multi_Decoder_cfglist(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                #  backbone: ConfigType,
                 backbone_config_base_list: List[ConfigType],
                #  backbone_MSI_3chan: ConfigType,
                #  backbone_MSI_4chan: ConfigType,
                #  backbone_MSI_10chan: ConfigType,
                #  decode_head: ConfigType,
                 decode_head_config_base_list: List[ConfigType],
                 decode_head_config_downstream: ConfigType,
                #  decode_head_MSI_3chan: ConfigType,
                #  decode_head_MSI_4chan: ConfigType,
                #  decode_head_MSI_10chan: ConfigType,

                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # if pretrained is not None:
        #     assert backbone_config_base_list[0].get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone_config_base_list[0] = pretrained

        # Build ,Multi backbone 不需要梯度
        if isinstance(backbone_config_base_list, dict):
            assert isinstance(backbone_config_base_list, dict), f'please check the backbone_config_base_list, expect dict, but got {type(backbone_config_base_list)}'
            for _, backbone_name in enumerate(backbone_config_base_list):
                # 创建 self.backbone_MSI_3chan, self.backbone_MSI_4chan, self.backbone_MSI_10chan
                backbone_temp = MODELS.build(backbone_config_base_list[backbone_name])
                backbone_temp.eval()
                for param in backbone_temp.parameters():
                    param.requires_grad = False
                setattr(self, backbone_name, backbone_temp)

        # Build ,Multi decode_head 
        if isinstance(decode_head_config_base_list, dict):
            assert isinstance(decode_head_config_base_list, dict), f'please check the decode_head_config_base_list, expect dict, but got {type(decode_head_config_base_list)}'
            for _, decode_head_name in enumerate(decode_head_config_base_list):
                # 创建 self.backbone_MSI_3chan, self.backbone_MSI_4chan, self.backbone_MSI_10chan
                decode_head_temp = MODELS.build(decode_head_config_base_list[decode_head_name])
                decode_head_temp.eval()
                for param in decode_head_temp.parameters():
                    param.requires_grad = False
                setattr(self, decode_head_name, decode_head_temp)

        # Build downstream decode_head
        self.decode_head_downstream = MODELS.build(decode_head_config_downstream)

        # import pdb;pdb.set_trace()
        # Build Neck        
        if neck is not None:
            self.neck = MODELS.build(neck)

        self.align_corners = self.decode_head_MSI_4chan.align_corners
        self.out_channels = self.decode_head_MSI_4chan.out_channels

        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # assert self.with_decode_head

    # def _init_decode_head(self, decode_head: ConfigType) -> None:
    #     """Initialize ``decode_head``"""
    #     self.decode_head = MODELS.build(decode_head)
    #     self.align_corners = self.decode_head.align_corners
    #     self.num_classes = self.decode_head.num_classes
        # self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    # Multi Backbone   # predict & loss
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # import pdb;pdb.set_trace()
        if inputs.shape[1]==3:
            x = self.backbone_MSI_3chan(inputs)
        elif inputs.shape[1]==4:
            x = self.backbone_MSI_4chan(inputs)
        elif inputs.shape[1]==10:
            x = self.backbone_MSI_10chan(inputs)

        # x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        # import pdb;pdb.set_trace()
        return x
    

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head_downstream.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits
    

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head_downstream.loss(inputs, data_samples,
                                                       self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_downstream'))
        return losses
    
    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # import pdb;pdb.set_trace()
        x = self.extract_feat(inputs)  # [2,3,512,512] ---> [[],[],[],[]]

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples) # 只计算downstream头的loss
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    # def _auxiliary_head_forward_train(self, inputs: List[Tensor],
    #                                   data_samples: SampleList,
    #                                   gt_semantic_seg_name:str) -> dict:
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     if isinstance(self.auxiliary_head, nn.ModuleList):
    #         for idx, aux_head in enumerate(self.auxiliary_head):
    #             loss_aux = aux_head.loss(inputs, 
    #                                      data_samples, 
    #                                      self.train_cfg,
    #                                      gt_semantic_seg_name)
    #             losses.update(add_prefix(loss_aux, f'aux_{idx}'))
    #     else:
    #         loss_aux = self.auxiliary_head.loss(inputs, 
    #                                             data_samples,
    #                                             self.train_cfg,
    #                                             gt_semantic_seg_name)
    #         losses.update(add_prefix(loss_aux, 'aux'))
    #         # aux.loss_ce: 0.0877  aux.acc_seg: 98.7294
    #     return losses


    # For predict
    # def encode_decode(self, 
    #                   inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     x = self.extract_feat(inputs)  # [b,1024,128,128] [b,1024,64,64] [b,1024,32,32] [b,1024,16,16]
        
    #     if inputs.shape[1]==3:
    #         seg_logits = self.decode_head_MSI_3chan.predict(x, batch_img_metas, self.test_cfg)
    #     elif inputs.shape[1]==4:
    #         seg_logits = self.decode_head_MSI_4chan.predict(x, batch_img_metas, self.test_cfg)
    #     elif inputs.shape[1]==10:
    #         seg_logits = self.decode_head_MSI_10chan.predict(x, batch_img_metas, self.test_cfg)
    #     return seg_logits

    # def _decode_head_forward_train(self, inputs: List[Tensor],
    #                                data_samples: SampleList,
    #                                gt_semantic_seg_name:str) -> dict:
    #     """Run forward function and calculate loss for decode head in
    #     training."""
    #     losses = dict()
    #     if gt_semantic_seg_name == 'gt_semantic_seg_MSI_3chan':
    #         loss_decode = self.decode_head_MSI_3chan.loss(inputs, 
    #                                             data_samples,
    #                                             self.train_cfg,
    #                                             gt_semantic_seg_name)
            
    #     elif gt_semantic_seg_name == 'gt_semantic_seg_MSI_4chan':
    #         loss_decode = self.decode_head_MSI_4chan.loss(inputs, 
    #                                             data_samples,
    #                                             self.train_cfg,
    #                                             gt_semantic_seg_name)
            
    #     elif gt_semantic_seg_name == 'gt_semantic_seg_MSI_10chan':
    #         loss_decode = self.decode_head_MSI_10chan.loss(inputs, 
    #                                             data_samples,
    #                                             self.train_cfg,
    #                                             gt_semantic_seg_name)
    #     # import pdb;pdb.set_trace()
    #     losses.update(add_prefix(loss_decode, f'decode_{gt_semantic_seg_name[16:]}')) # decode_MSI_3chan.loss_ce  或者 decode_MSI_3chan.loss_hiera
    #     return losses

    # def loss(self, 
    #          inputs: Tensor, 
    #          data_samples: SampleList) -> dict:
    #     """Calculate losses from a batch of inputs and data samples.

    #     Args:
    #         inputs (List): [[2,4,512,512] [2,10,512,512]]  # 如果是两个输入的话。
    #         data_samples (list[:obj:`SegDataSample`]): The seg data samples.
    #             It usually includes information such as `metainfo` and
    #             `gt_sem_seg`.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """
    #     if isinstance(inputs, list) and isinstance(inputs[0], Tensor):
    #         if inputs[0].shape[1]==3:
    #             inputs_MSI_3chan = inputs[0]
    #         if inputs[1].shape[1]==4:
    #             inputs_MSI_4chan = inputs[1]
    #         if inputs[2].shape[1]==10:
    #             inputs_MSI_10chan = inputs[2]
            
    #         assert inputs_MSI_3chan.shape[1]==3 and inputs_MSI_4chan.shape[1]==4 and inputs_MSI_10chan.shape[1]==10, 'please check the input channel'

    #     x_MSI_3chan = self.extract_feat(inputs_MSI_3chan)   # [2,3,512,512] ---> [[],[],[],[]]
    #     x_MSI_4chan = self.extract_feat(inputs_MSI_4chan)    # x: list [x_MSI_4chan], [x_MSI_10chan] 分别是四个尺度的特征图
    #     x_MSI_10chan = self.extract_feat(inputs_MSI_10chan)
    #     # import pdb;pdb.set_trace()
    #     losses = dict()

    #     loss_decode_MSI_3chan = self._decode_head_forward_train(x_MSI_3chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_3chan') # x:[4个多尺度列表]]
    #     loss_decode_MSI_4chan = self._decode_head_forward_train(x_MSI_4chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_4chan') # x:[4个多尺度列表]]
    #     loss_decode_MSI_10chan = self._decode_head_forward_train(x_MSI_10chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_10chan') # x:[4个多尺度列表]]
        
    #     # import pdb;pdb.set_trace()
    #     # decode_head返回的loss值
    #     new_loss_value = loss_decode_MSI_3chan['decode_MSI_3chan.loss_ce'] + loss_decode_MSI_4chan['decode_MSI_4chan.loss_ce'] + loss_decode_MSI_10chan['decode_MSI_10chan.loss_ce']
    #     # 以GF2的为基准，其他的两个acc不要
    #     # new_acc_seg = (loss_decode_MSI_3chan['decode.acc_seg'] + loss_decode_MSI_4chan['decode.acc_seg']+loss_decode_MSI_10chan['decode.acc_seg'])/3
    #     new_acc_seg = loss_decode_MSI_4chan['decode_MSI_4chan.acc_seg']

    #     loss_decode = dict() 
    #     loss_decode['decode.loss_ce'] = new_loss_value
    #     loss_decode['decode.acc_seg'] = new_acc_seg

    #     losses.update(loss_decode)

    #     if self.with_auxiliary_head:
    #         loss_aux_MSI_3chan = self._auxiliary_head_forward_train(x_MSI_3chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_3chan')
    #         loss_aux_MSI_4chan = self._auxiliary_head_forward_train(x_MSI_4chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_4chan')
    #         loss_aux_MSI_10chan = self._auxiliary_head_forward_train(x_MSI_10chan, data_samples, gt_semantic_seg_name='gt_semantic_seg_MSI_10chan')
            
    #         loss_aux = loss_aux_MSI_4chan
    #         new_loss_aux_value = loss_aux_MSI_3chan['aux.loss_ce'] + loss_aux_MSI_4chan['aux.loss_ce'] + loss_aux_MSI_10chan['aux.loss_ce']
    #         # new_aux_acc_seg =  (loss_aux_MSI_3chan['aux.acc_seg'] + loss_aux_MSI_4chan['aux.acc_seg'] + loss_aux_MSI_10chan['aux.acc_seg'])/3
    #         new_aux_acc_seg = loss_aux_MSI_4chan['aux.acc_seg']
    #         loss_aux['aux.loss_ce'] = new_loss_aux_value
    #         loss_aux['aux.acc_seg'] = new_aux_acc_seg
    #         losses.update(loss_aux)

    #     return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        
        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """

        # import pdb;pdb.set_trace()
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        post_result = self.postprocess_result(seg_logits, data_samples)
    
        return post_result

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size

        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)

        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred



    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(
                    dim=0, keepdim=True)  # keepdim=True，保留第0维度，大小为1
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples