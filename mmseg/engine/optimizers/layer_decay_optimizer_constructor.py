# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1

def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1

def atl_get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token','backbone.pos_embed','backbone.visual_embed') or \
       var_name in ('backbone_MSI_3chan.cls_token', 'backbone_MSI_3chan.mask_token','backbone_MSI_3chan.pos_embed','backbone_MSI_3chan.visual_embed') or \
       var_name in ('backbone_MSI_4chan.cls_token', 'backbone_MSI_4chan.mask_token','backbone_MSI_4chan.pos_embed','backbone_MSI_4chan.visual_embed') or \
       var_name in ('backbone_MSI_10chan.cls_token', 'backbone_MSI_10chan.mask_token','backbone_MSI_10chan.pos_embed','backbone_MSI_10chan.visual_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed') or \
         var_name.startswith('backbone_MSI_3chan.patch_embed') or \
         var_name.startswith('backbone_MSI_4chan.patch_embed') or \
         var_name.startswith('backbone_MSI_10chan.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks') or var_name.startswith('backbone.layers') or \
        var_name.startswith('backbone_MSI_3chan.blocks') or var_name.startswith('backbone_MSI_3chan.layers') or \
        var_name.startswith('backbone_MSI_4chan.blocks') or var_name.startswith('backbone_MSI_4chan.layers') or \
        var_name.startswith('backbone_MSI_10chan.blocks') or var_name.startswith('backbone_MSI_10chan.layers'):
        
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1



def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token','backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks') or var_name.startswith('backbone.layers'):
    # elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1




@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ATL_LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """
    """
    ATL_Multi_Encoder_Multi_Decoder
        - data_preprocessor
        - backbone_MSI_3chan
        - backbone_MSI_4chan
        - backbone_MSI_10chan
        - decode_head_MSI_3chan
        - decode_head_MSI_4chan
        - decode_head_MSI_10chan
        - auxiliary_head

    """
    
    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """

        parameter_groups = {}
        print_log(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        print_log('Build LearningRateDecayOptimizerConstructor  '
                  f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters(): # module: ATL_multi_encoder_multi_decoder
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                
                # 改成 backbone_MSI_3chan
                if getattr(module, 'backbone', None) is not None:  # 对于有backbone的模型
                    if 'ConvNeXt' in module.backbone.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(
                            name, self.paramwise_cfg.get('num_layers'))
                        print_log(f'set param {name} as id {layer_id}')
                    # BEiTAdapter 要包含 BEiT!!!!!
                    elif 'BEiT' in module.backbone.__class__.__name__ or \
                        'MAE' in module.backbone.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(name, num_layers)
                        print_log(f'set param {name} as id {layer_id}')

                    elif 'ViT' in module.backbone.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(name, num_layers)
                        print_log(f'set param {name} as id {layer_id}')

                    else:
                        raise NotImplementedError()
                
                elif getattr(module, 'backbone_MSI_3chan', None) or \
                     getattr(module, 'backbone_MSI_4chan', None) or \
                     getattr(module, 'backbone_MSI_10chan', None) is not None:  # 对于有backbone的模型

                    if 'ConvNeXt' in module.backbone_MSI_4chan.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(
                            name, self.paramwise_cfg.get('num_layers'))
                        print_log(f'set param {name} as id {layer_id}')
                    # BEiTAdapter 要包含 BEiT!!!!!
                    elif 'BEiT' in module.backbone_MSI_4chan.__class__.__name__ or \
                        'MAE' in module.backbone_MSI_4chan.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(name, num_layers)
                        print_log(f'set param {name} as id {layer_id}')
                        # import pdb; pdb.set_trace()

                    elif 'ViT' in module.backbone_MSI_4chan.__class__.__name__:
                        layer_id = atl_get_layer_id_for_vit(name, num_layers)
                        print_log(f'set param {name} as id {layer_id}')

                    else:
                        raise NotImplementedError()

            # elif decay_type == 'stage_wise':
            #     if 'ConvNeXt' in module.backbone.__class__.__name__:
            #         layer_id = get_stage_id_for_convnext(name, num_layers)
            #         print_log(f'set param {name} as id {layer_id}')
            #     else:
            #         raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'   # layer_id:13  group_name:'decay'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1) # 0.9**(14 - 13 - 1) = 0.9

                # {'layer_13_decay': {'weight_decay': 0.05, 'params': [], 'param_names': [], 'lr_scale': 1.0, 'group_name': 'layer_13_decay', 'lr': 2e-05}}
                parameter_groups[group_name] = {           
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ATL_LayerDecayOptimizerConstructor(ATL_LearningRateDecayOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for BEiT,
    and it will be deprecated.
    Please use ``LearningRateDecayOptimizerConstructor`` instead.
    """

    def __init__(self, optim_wrapper_cfg, paramwise_cfg):
        warnings.warn('DeprecationWarning: Original '
                      'LayerDecayOptimizerConstructor of BEiT '
                      'will be deprecated. Please use '
                      'LearningRateDecayOptimizerConstructor instead, '
                      'and set decay_type = layer_wise_vit in paramwise_cfg.')
        paramwise_cfg.update({'decay_type': 'layer_wise_vit'})
        warnings.warn('DeprecationWarning: Layer_decay_rate will '
                      'be deleted, please use decay_rate instead.')
        paramwise_cfg['decay_rate'] = paramwise_cfg.pop('layer_decay_rate')
        super().__init__(optim_wrapper_cfg, paramwise_cfg)


# ================================ ATL ==============================
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """
    """
    ATL_Multi_Encoder_Multi_Decoder
        - data_preprocessor
        - backbone_MSI_3chan
        - backbone_MSI_4chan
        - backbone_MSI_10chan
        - decode_head_MSI_3chan
        - decode_head_MSI_4chan
        - decode_head_MSI_10chan
        - auxiliary_head

    """
    
    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """

        parameter_groups = {}
        print_log(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        print_log('Build LearningRateDecayOptimizerConstructor  '
                  f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():  # backbone.cls_token
            if not param.requires_grad:                # param.requires_grad: True
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:

                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_convnext(
                        name, self.paramwise_cfg.get('num_layers'))
                    print_log(f'set param {name} as id {layer_id}')
                
                # BEiTAdapter 要包含 BEiT!!!!!  ，ViTAdapter要包含ViT
                
                elif 'BEiT' in module.backbone.__class__.__name__ or \
                     'MAE' in module.backbone.__class__.__name__:
                    
                    layer_id = get_layer_id_for_vit(name, num_layers)   # backbone.cls_token:0
                    print_log(f'set param {name} as id {layer_id}')     
                    # import pdb; pdb.set_trace()

                elif 'ViT' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                    print_log(f'set param {name} as id {layer_id}')

                else:
                    raise NotImplementedError()
            elif decay_type == 'stage_wise':
                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_stage_id_for_convnext(name, num_layers)
                    print_log(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())



@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(LearningRateDecayOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for BEiT,
    and it will be deprecated.
    Please use ``LearningRateDecayOptimizerConstructor`` instead.
    """

    def __init__(self, optim_wrapper_cfg, paramwise_cfg):
        warnings.warn('DeprecationWarning: Original '
                      'LayerDecayOptimizerConstructor of BEiT '
                      'will be deprecated. Please use '
                      'LearningRateDecayOptimizerConstructor instead, '
                      'and set decay_type = layer_wise_vit in paramwise_cfg.')
        paramwise_cfg.update({'decay_type': 'layer_wise_vit'})
        warnings.warn('DeprecationWarning: Layer_decay_rate will '
                      'be deleted, please use decay_rate instead.')
        paramwise_cfg['decay_rate'] = paramwise_cfg.pop('layer_decay_rate')
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
