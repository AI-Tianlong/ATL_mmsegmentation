# Copyright (c) OpenMMLab. All rights reserved.
import logging
from copy import deepcopy
from re import T
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ATL_Tools import setup_logger
from mmengine.logging import print_log
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

atl_logger = setup_logger(show_file_path=True)


@MODELS.register_module()
class ATL_EncoderDecoder(BaseSegmentor):
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

    def __init__(
            self,
            backbone: ConfigType,
            decode_head: ConfigType,
            neck: OptConfigType = None,
            auxiliary_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            pretrained: Optional[str] = None,
            init_cfg: OptMultiConfig = None,
            level_classes_map: Optional[dict] = None,  # 层级分类的映射map
    ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.level_classes_map = level_classes_map  # ATL-添加的属性 层级分类的映射map,去做后处理的时候用

    # 初始化 decode_head 不用动
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    # 初始化 auxiliary_head 不用动
    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    # 过 backbone 和 neck 抽特征不用动
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)  #iniputs:[2,10,512,512]
        if self.with_neck:
            x = self.neck(x)
        return x

    # 这里也不用动，调用的 decode_head 的  predict 方法
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)  # 过backbone+neck抽特征，然后输入到decode_head--> [1024,1024,1024,1024]层级
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg) # [2, 40, 512, 512]

        # 所以在seg_logits里面存的是预测的结果,这里去做后处理的文章
        return seg_logits

    # 这里也不用动, 调用的是 decode_head 的 loss 方法，并且前缀加上 decode.d8.loss_cls decode.d4.loss_mask decode.d4.loss_dice
    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()  # 这里调用的是 BaseDecodeHead 里的 loss 方法
        loss_decode = self.decode_head.loss(
            inputs,
            data_samples,  # atl_loss:
            self.train_cfg)
        # 这里，有一个 acc_seg 的指标，是在 loss_by_feat 里面计算的

        losses.update(add_prefix(loss_decode, 'decode'))  # decode.atl_loss
        # pdb.set_trace()
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
            inputs (Tensor): Input images.  # torch.Size([2, 10, 512, 512])
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(inputs)
        # x: List[Tensor]  BEiT-Adapter 的输出
        # torch.Size([2, 1024, 128, 128]) [2, 1024, 64, 64] [2, 1024, 32, 32] [2, 1024, 16, 16]
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

    
        # pdb.set_trace()
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, data_samples)  # 辅助头的参数正确导入了，所以这里是对的，数值比较高
            losses.update(loss_aux)

        # pdb.set_trace()
        return losses

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

        seg_logits = self.inference(inputs, batch_img_metas)  # [2,10,512,512]-->[2, 37, 512, 512] UperHead 的输出
        # whole inference或者slide inference，都是调用的这个函数，这个函数里面调用的是 encode_decode

        # 就是这里，输出看一下特征图。
        # import pdb; pdb.set_trace() 
        
        # 没有softmax的logits
        # 后处理的时候，应该对每个层级的特征图，单独做一个softmax，然后再叠加

        return self.postprocess_result(seg_logits, data_samples) #然后是最后的后处理，保存单个通道的那种

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
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas) #[1,40,512,512]
                # import pdb;pdb.set_trace()
                # preds [1,22,512,512]

                # import pdb;pdb.set_trace()
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

        seg_logits = self.encode_decode(inputs, batch_img_metas) # [2,40,512,512]

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
            seg_logit = self.whole_inference(inputs, batch_img_metas)  # [2, 40, 512, 512]

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
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    

    # # 第二次覆写，最终的结果，只要子类的特征图来预测。
    # def postprocess_result(self,
    #                         seg_logits: Tensor,
    #                         data_samples: OptSampleList = None) -> SampleList:
    #         """ Convert results list to `SegDataSample`.
    #         Args:
    #             seg_logits (Tensor): The segmentation results, seg_logits from
    #                 model of each input image.
    #             data_samples (list[:obj:`SegDataSample`]): The seg data samples.
    #                 It usually includes information such as `metainfo` and
    #                 `gt_sem_seg`. Default to None.
    #         Returns:
    #             list[:obj:`SegDataSample`]: Segmentation results of the
    #             input images. Each SegDataSample usually contain:

    #             - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
    #             - ``seg_logits``(PixelData): Predicted logits of semantic
    #                 segmentation before normalization.
    #         """
    #         batch_size, C, H, W = seg_logits.shape  # val的时候，batch是1 [1,40,512,512]
    #         # 只要后22的通道, 消融实验，层级只在训练的时候有用！！！！！。
    #         seg_logits = seg_logits[:, -22:, :, :]  # [1,22,512,512] 是那个L3的吧？要么不cat呢?

    #         # import pdb;pdb.set_trace()

    #         if data_samples is None:
    #             data_samples = [SegDataSample() for _ in range(batch_size)]
    #             only_prediction = True
    #         else:
    #             only_prediction = False

    #         for i in range(batch_size):
    #             if not only_prediction:
    #                 img_meta = data_samples[i].metainfo
    #                 # remove padding area
    #                 if 'img_padding_size' not in img_meta:
    #                     padding_size = img_meta.get('padding_size', [0] * 4)
    #                 else:
    #                     padding_size = img_meta['img_padding_size']
    #                 padding_left, padding_right, padding_top, padding_bottom =\
    #                     padding_size
    #                 # i_seg_logits shape is 1, C, H, W after remove padding
    #                 i_seg_logits = seg_logits[i:i + 1, :,
    #                                         padding_top:H - padding_bottom,
    #                                         padding_left:W - padding_right]

    #                 flip = img_meta.get('flip', None)
    #                 if flip:
    #                     flip_direction = img_meta.get('flip_direction', None)
    #                     assert flip_direction in ['horizontal', 'vertical']
    #                     if flip_direction == 'horizontal':
    #                         i_seg_logits = i_seg_logits.flip(dims=(3, ))
    #                     else:
    #                         i_seg_logits = i_seg_logits.flip(dims=(2, ))

    #                 # resize as original shape
    #                 i_seg_logits = resize(
    #                     i_seg_logits,
    #                     size=img_meta['ori_shape'],
    #                     mode='bilinear',
    #                     align_corners=self.align_corners,
    #                     warning=False).squeeze(0)
    #             else:
    #                 i_seg_logits = seg_logits[i]

    #             if C > 1:
    #                 i_seg_pred = i_seg_logits.argmax(
    #                     dim=0, keepdim=True)  # keepdim=True，保留第0维度，大小为1
    #             else:
    #                 i_seg_logits = i_seg_logits.sigmoid()
    #                 i_seg_pred = (i_seg_logits >
    #                             self.decode_head.threshold).to(i_seg_logits)
    #             data_samples[i].set_data({
    #                 'seg_logits':
    #                 PixelData(**{'data': i_seg_logits}),
    #                 'pred_sem_seg':
    #                 PixelData(**{'data': i_seg_pred})
    #             })

    #         return data_samples
    


    # ATL-覆写的 BaseSegmentor 里的 `postprocess_result` 方法
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
        batch_size, C, H, W = seg_logits.shape  # [2, 37, 512, 512]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):  # 对每一个batch单独处理！！！
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
                i_seg_logits = seg_logits[
                    i:i + 1, :,  # 分离batch 为 1
                    padding_top:H -
                    padding_bottom,  # [1,40,512,512] [1,40,512,512]
                    padding_left:W - padding_right]

                flip = img_meta.get('flip', None)  # 如果翻转了，需要 flip 回来
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape       # 将预测的结果 resize 回原图大小
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)  # [1,40,512,512] --> [40,512,512]
                # 跑s2-five-billion-执行的是这个。 i_seg_logits: torch.Size([40, 512, 512])
            else:
                i_seg_logits = seg_logits[i]
                # atl_logger.info(f"i_seg_logits: {i_seg_logits.shape}")

            # import pdb; pdb.set_trace()
            if C > 1:  # [40,512,512] --> 在第0维上做 argmax，替换成，先求父类和子类的和。
                # 在这里改后处理就行！！！！

                if self.level_classes_map is not None:
                    # import pdb
                    # pdb.set_trace()

                    # 大思想：叠加父类和子类的特征图，作为共同的特征图值，再进行argmax。

                    # step1: 根据map,找到对应的L1级别的特征图索引，L2级别的特征图索引，L3级别的特征图索引。
                    num_levels = len(self.level_classes_map)  # L1 L3 L3,那就是3

                    # [0, 5, 11, 21]
                    num_levels_classes = list()
                    num_levels_classes.append(0)
                    for level_name, high_level_dict in list(
                            self.level_classes_map.items()):
                        num_levels_classes.append(len(high_level_dict))

                    num_levels_classes_original = deepcopy(num_levels_classes)
                    # 巧妙地构造出了 [0,5,11,21]-->[[0,5],[5,16],[16,37]]
                    classes_range = list()
                    for j in range(len(num_levels_classes) - 1):
                        classes_range.append([num_levels_classes[j], num_levels_classes[j] + num_levels_classes[j + 1]])
                        num_levels_classes[j+1] = num_levels_classes[j] + num_levels_classes[j+1]
                    # 最终的classes_range=[[0,5],[5,16],[16,37]]
                    # num_levels_classes=[0, 5, 16, 37]

                    # step2: 根据找的索引,创立一个新的[21,512,512]的特征图。注意这里的设备，要和i_seg_logits一样。
                    # 我去！！！这里的num_levels_classes 变了！！！ 注意，特征图变成了[37,512,512] !!!! 完蛋
                    # from ATL_Tools import setup_logger
                    # atl_logger = setup_logger(show_file_path=True)
                    # atl_logger.info(f'num_levels_classes: {num_levels_classes}')
                    # atl_logger.info(f'num_levels_classes_original: {num_levels_classes_original}')
                    # [2,21,512,512]
                    # import pdb; pdb.set_trace()

                    # num_levels_classes=[0, 5, 16, 37]
                    new_seg_logits = torch.zeros((num_levels_classes_original[-1], H, W), device=i_seg_logits.device)# [21,512,512]
                    L1_i_seg_logits = i_seg_logits[num_levels_classes[0]:num_levels_classes[1], :, :] # [5, 512, 512]  [0-4]
                    L2_i_seg_logits = i_seg_logits[num_levels_classes[1]:num_levels_classes[2], :, :] # [11, 512, 512] [5-15]
                    L3_i_seg_logits = i_seg_logits[num_levels_classes[2]:num_levels_classes[3], :, :] # [21, 512, 512] [16,36]

                    # 在argmax之前, 加一个softmax, 然后再做特征图的叠加

                    softmax_L1_i_seg_logits = F.softmax(L1_i_seg_logits, dim=0) # [5, 512, 512]
                    softmax_L2_i_seg_logits = F.softmax(L2_i_seg_logits, dim=0) # [11, 512, 512]
                    softmax_L3_i_seg_logits = F.softmax(L3_i_seg_logits, dim=0) # [21, 512, 512]

                    softmax_i_seg_logits = torch.cat([softmax_L1_i_seg_logits, 
                                                      softmax_L2_i_seg_logits, 
                                                      softmax_L3_i_seg_logits], dim=0) # [37, 512, 512]

                    # import pdb; pdb.set_trace()
                    # import pdb    
                    # pdb.set_trace()
                    # Step3：计算叠加的特征图
                    # 对于每一个L3级别的特征图，都要叠加他自己和L1和L2的特征图。
                    # 所以，需要知道映射关系，哪三个图往一起加。
                    for index_range, high_level_dict_key in enumerate(self.level_classes_map):
                        index_add_num = num_levels_classes[index_range]  #[0, 6, 18, 40] # 加索引的数
                        # print(f'index_add_num {index_add_num}')
                        for high_level_index_, high_level_name in enumerate(self.level_classes_map[high_level_dict_key]):
                            for low_index_ in self.level_classes_map[high_level_dict_key][high_level_name]:  #遍历[1,2,3,4,5,6,7]

                                high_label_index = index_add_num + high_level_index_
                                # print(f'特征图：{high_label_index} 加到新的 {low_index_} 类别上')

                                # print(f'原特征图 {high_label_index} --> 新特征图 {low_index_}')
                                # import pdb; pdb.set_trace()
                                #新特征图中的索引，               原先特征图的索引
                                new_seg_logits[low_index_] += softmax_i_seg_logits[high_label_index]

                    # 然后再对new_seg_logits进行argmax操作，得到最终的预测结果。

                    # 消融2,不加L1 L2 的特征图，只在算loss的时候用。
                    # new_new_seg_logits = i_seg_logits[-num_levels_classes_original[-1]:,:,:]
                    # print(f'new_new_seg_logits: {new_new_seg_logits.shape}')
                    i_seg_pred = new_seg_logits.argmax(dim=0, keepdim=True)  # [1,512,512] # keepdim=True，保留第0维度，大小为1

                else:
                    i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)  # [1,512,512]
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data(
                {
                    'seg_logits':
                    PixelData(**{'data': i_seg_logits}),  #[40,512,512]
                    'pred_sem_seg':  #[1,512,512]
                    PixelData(**{'data': i_seg_pred})
                })
        return data_samples



    