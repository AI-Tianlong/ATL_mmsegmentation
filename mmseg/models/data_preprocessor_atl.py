# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Union
from mmseg.utils.typing_utils import SampleList

from mmseg.registry import MODELS
import numpy as np
import torch.nn.functional as F
# from mmseg.utils import stack_batch

def multi_embedding_stack_batch(inputs: List[tuple],  # 这里规定输入的inputs是一个list，里面的元素是每一个图像的tensor。因为我是一个大list，里面套两个list, 然后里面还是一个
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """


    # assert isinstance(inputs, list), \
    #     f'Expected input type to be list, but got {type(inputs)}'
    # assert len({tensor.ndim for tensor in inputs}) == 1, \
    #     f'Expected the dimensions of all inputs must be the same, ' \
    #     f'but got {[tensor.ndim for tensor in inputs]}'
    # assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
    #     f'but got {inputs[0].ndim}'
    # assert len({tensor.shape[0] for tensor in inputs}) == 1, \
    #     f'Expected the channels of all inputs must be the same, ' \
    #     f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    batch_size = len(inputs[0])# 2 3 4 

    inputs = [input_MSI_xchan_.float()  for inputs_MSI_xchan in inputs for input_MSI_xchan_ in inputs_MSI_xchan]
    
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    # 开始拼接多个 tensor 为一个 4D 的batch
    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)

    # 所以应该分开去padd和stack
    # pad gt_sem_seg
    if data_samples is not None:

        for i in range(len(data_samples)):
            data_sample = data_samples[i]
            pad_shape = None

            # 看ATL_3_packSegInputs的代码，这里是把gt_sem_seg, gt_edge_map, gt_depth_map都pad了

            if 'gt_semantic_seg_MSI_3chan' in data_sample:
                gt_semantic_seg_MSI_3chan = data_sample.gt_semantic_seg_MSI_3chan.data
                del data_sample.gt_semantic_seg_MSI_3chan.data
                data_sample.gt_semantic_seg_MSI_3chan.data = F.pad(
                    gt_semantic_seg_MSI_3chan, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_semantic_seg_MSI_3chan.shape

            if 'gt_semantic_seg_MSI_4chan' in data_sample:
                gt_semantic_seg_MSI_4chan = data_sample.gt_semantic_seg_MSI_4chan.data
                del data_sample.gt_semantic_seg_MSI_4chan.data
                data_sample.gt_semantic_seg_MSI_4chan.data = F.pad(
                    gt_semantic_seg_MSI_4chan, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_semantic_seg_MSI_4chan.shape

            if 'gt_semantic_seg_MSI_10chan' in data_sample:
                gt_semantic_seg_MSI_10chan = data_sample.gt_semantic_seg_MSI_10chan.data
                del data_sample.gt_semantic_seg_MSI_10chan.data
                data_sample.gt_semantic_seg_MSI_10chan.data = F.pad(
                    gt_semantic_seg_MSI_10chan, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_semantic_seg_MSI_10chan.shape


            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
    else:
        padded_samples.append(
            dict(
                img_padding_size=padding_size,
                pad_shape=pad_img.shape[-2:]))
        
    batch_padded_inputs_MSI_3chan = torch.stack(padded_inputs[0:batch_size], dim=0)
    batch_padded_inputs_MSI_4chan = torch.stack(padded_inputs[batch_size:2*batch_size], dim=0)
    batch_padded_inputs_MSI_10chan = torch.stack(padded_inputs[2*batch_size:], dim=0)


    return [batch_padded_inputs_MSI_3chan, batch_padded_inputs_MSI_4chan, batch_padded_inputs_MSI_10chan], padded_samples


def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """

    import pdb; pdb.set_trace()
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    # 开始拼接多个 tensor 为一个 4D 的batch
    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if 'gt_sem_seg' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if 'gt_depth_map' in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples

@MODELS.register_module()
class ATL_SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model 1 input.
        """
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()
        # TODO: whether normalize should be after stack_batch
        # if self.channel_conversion and inputs[0][0].size(0) == 3:
        #     inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        
        
        # 把 inputs展开，变成一个list，然后用这个list去拼接成一个patch
        # inputs = [_input.float() for _input in inputs]  # 这才能把数据转换成float类型，不然全是255--->1367
        
        # import pdb;pdb.set_trace()
        # if self._enable_normalize:
        #     inputs = [(_input - self.mean) / self.std for _input in inputs]

        # import pdb;pdb.set_trace()
        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            
            # 将多个输入变成一个patch
            inputs, data_samples = multi_embedding_stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            # import pdb;pdb.set_trace()

            # if self.batch_augments is not None:
            #     inputs, data_samples = self.batch_augments(
            #         inputs, data_samples)
        else:
            inputs = [_input.float() for _input in inputs]  # 这才能把数据转换成float类型，不然全是255--->1367

            # import pdb;pdb.set_trace()
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})

            else:
                inputs = torch.stack(inputs, dim=0)
            # else:
            #     if isinstance(inputs, list) and isinstance(inputs[0], tuple) \
            #         and isinstance(inputs[0][0], torch.Tensor):
            #         batch_size = len(inputs[0])
            #         inputs = [input_MSI_xchan_.float()  for inputs_MSI_xchan in inputs for input_MSI_xchan_ in inputs_MSI_xchan]

            #         batch_inputs_MSI_4chan = torch.stack(inputs[0:batch_size], dim=0)
            #         batch_inputs_MSI_10chan = torch.stack(inputs[batch_size:], dim=0)

            #         inputs = [batch_inputs_MSI_4chan, batch_inputs_MSI_10chan]

        return dict(inputs=inputs, data_samples=data_samples)
