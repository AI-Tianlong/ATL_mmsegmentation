# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import CheckpointLoader, Runner
from mmengine.runner.checkpoint import (_load_checkpoint, load_checkpoint,
                                        load_state_dict)
from tqdm import tqdm

from mmseg.registry import RUNNERS


def convert_beit(original_ckpt):
    new_ckpt = OrderedDict()
    new_ckpt = copy.deepcopy(original_ckpt)

    cfg_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/configs/beit_adapter/beit_adapter_mask2former_4xb16_potsdam-512x512.py'
    cfg = Config.fromfile(cfg_path)

    cfg.work_dir = './'
    # resume training
    cfg.resume = False
    runner = Runner.from_cfg(cfg)
    model = runner.model
    # print(model.state_dict().keys())
    keys_to_remove = []
    model_keys_list = list(model.state_dict().keys())
    state_keys_list = list(original_ckpt['state_dict'].keys())

    miss_num = 0
    for i in range(len(original_ckpt['state_dict'].keys())):
        k1 = state_keys_list[i]
        k2 = model_keys_list[i]
        if k1 != k2:

            new_ckpt['state_dict'][k2] = original_ckpt['state_dict'][k1]
            new_ckpt['state_dict'].pop(k1)
            miss_num += 1
            print(f'{i+1} {state_keys_list[i]} ---> {model_keys_list[i]}')

    print(f'miss_num: {miss_num}')
    print(f"new_ckpt: {len(new_ckpt['state_dict'].keys())}")

    # 验证：
    a1 = original_ckpt['state_dict'][
        'decode_head.transformer_decoder.layers.8.ffns.0.layers.1.bias']
    a2 = new_ckpt['state_dict'][
        'decode_head.transformer_decoder.layers.8.ffn.layers.1.bias']

    print(a1)
    print(a2)

    if a1.equal(a2):
        print(f'a1 a2 相等')

    new_ckpt_keys_list = list(sorted(new_ckpt['state_dict'].keys()))
    model_keys_list = sorted(model_keys_list)
    for i in range(len(new_ckpt_keys_list)):
        if new_ckpt_keys_list[i] != model_keys_list[i]:
            print(f'{i+1} {new_ckpt_keys_list[i]} ---> {model_keys_list[i]}')
            print(f'验证失败！')
            break
    print(f'验证成功！,保存ing。。。')

    return new_ckpt


def main():

    original_ckpt_path = '/opt/AI-Tianlong/checkpoints/beit_adapter_mask2former_potsdam/mmseg0.x-beitv2_adapter_potsdam_iter_80000_mIoU80.57.pth'
    original_ckpt = CheckpointLoader.load_checkpoint(
        original_ckpt_path, map_location='cpu')

    new_ckpt_path = '/opt/AI-Tianlong/checkpoints/beit_adapter_mask2former_potsdam/mmseg1.x-beitv2_adapter_potsdam_iter_80000_mIoU80.57.pth'

    new_ckpt = convert_beit(original_ckpt)
    print(f'保存ing。。。')
    torch.save(new_ckpt, new_ckpt_path)


if __name__ == '__main__':
    main()
