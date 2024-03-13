from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer

model_cfg = '/opt/AI-Tianlong/openmmlab/mmsegmentation/configs/beit_adapter/beit_adapter_mask2former_4xb16_potsdam-512x512.py'
checkpoint_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/'


img_path = '/opt/AI-Tianlong/Datasets/potsdam/img_dir/train/2_10_0_0_512_512.png'
mmseg_inferencer = MMSegInferencer(
    model_cfg, checkpoint_path, dataset_name='potsdam', device='cuda:0')

print('推理ing...')
mmseg_inferencer(
    img_path,
    show=False,
    out_dir='atl_debug/atl_1.x_out',
    opacity=0.5,
    with_labels=True)
