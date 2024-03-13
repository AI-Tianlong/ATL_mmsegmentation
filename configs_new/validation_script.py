import argparse
import os.path as osp
import re
import tempfile
from collections import defaultdict
from importlib import import_module

import mmengine.registry.root as mmengine_root
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import DefaultScope, Registry
from mmengine.utils import get_object_from_string, scandir
from rich.progress import track

logger = MMLogger('config validate', log_file='work_dirs/debug.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Validate config')
    parser.add_argument('--old', help='config file path')
    parser.add_argument('--new', help='config file path')
    parser.add_argument('--scope', help='scope of repo')
    args = parser.parse_args()
    return args


def validate(origin_file, target_file, repo_scope):

    def get_obj_special(origin):
        module = getattr(get_object_from_string('mmcv.ops'), origin, None)
        if module is not None:
            return module
        module = getattr(
            get_object_from_string('albumentations.augmentations'), origin,
            None)
        if module is not None:
            return module
        raise NotImplementedError()

    def collect_all_modules():
        registry_module = import_module(f'{repo_scope}.registry')
        mmpretrain_registries = [
            value for value in registry_module.__dict__.values()
            if isinstance(value, Registry)
        ]
        mmengine_registries = [
            value for value in mmengine_root.__dict__.values()
            if isinstance(value, Registry)
        ]
        all_registries = set(mmpretrain_registries + mmengine_registries)
        all_modules = defaultdict(dict)
        for registry in all_registries:
            all_modules[registry.scope].update(registry.module_dict)
        return all_modules

    def _parse_type_name(type_name):
        matched = re.match(r'(.*)\.(.*)', type_name)
        if matched is not None:
            scope, type_name = matched.groups()
        else:
            scope = None
        return scope, type_name

    def _get_module(type_name, scope=None):
        if scope is None:
            scope = repo_scope
        if type_name in all_modules[scope]:
            return all_modules[scope][type_name]
        else:
            return all_modules['mmengine'].get(type_name)

    def update_all_modules(scope):
        utils = import_module(f'{scope}.utils')
        utils.register_all_modules()
        all_registries = [
            r for r in import_module(f'{scope}.registry').__dict__.values()
            if isinstance(r, Registry)
        ]
        for r in all_registries:
            all_modules[scope].update(r.module_dict)

    def check(origin, target):
        if isinstance(origin, dict):
            origin = {
                key: value
                for key, value in origin.items()
                if key != '_scope_' and key != 'default_scope'
            }
            target = {
                key: value
                for key, value in origin.items() if key != 'default_scope'
            }
            if '_scope_' in origin:
                scope = origin.get('_scope_')
                if scope not in all_modules:
                    update_all_modules(scope)
                if 'type' in origin:
                    type_name = origin['type']
                    assert target['type'] is all_modules[scope][type_name], (
                        f'module get from {scope}.{type_name} != {target["type"]} in {origin_file} and ./{target_file}'
                    )
                else:
                    raise RuntimeError(
                        'Special case for checking, please check it manually.')
            else:
                scope = None
            try:
                assert len(origin) == len(target)
            except Exception as e:
                if len(origin) > len(target):
                    error_mse = f'origin has extra keys: {set(origin) - set(target)}'
                else:
                    error_mse = f'target has extra keys: {set(target) - set(origin)}'
                raise type(e)(error_mse)

            for k, v in origin.items():
                with DefaultScope.overwrite_default_scope(scope):
                    check(v, target[k])

        elif isinstance(origin, list):
            assert len(origin) == len(target), (
                'origin: {}, target: {}'.format('\n'.join(origin),
                                                '\n'.join(target)))
            for item_a, item_b in zip(origin, target):
                check(item_a, item_b)
        else:
            if origin != target:
                if isinstance(target, str):
                    target = get_object_from_string(target)
                scope, type_name = _parse_type_name(origin)
                scope = scope or DefaultScope.get_current_instance().scope_name
                module = _get_module(type_name, scope)
                if module is None and origin is not None:
                    module = get_obj_special(origin)
                assert module == target, (
                    f'{module} != \n{target} in {origin_file} and ./{target_file}'
                )

    import_module(f'{repo_scope}.utils').register_all_modules()
    all_modules = collect_all_modules()
    ori_cfg = Config.fromfile(origin_file)._cfg_dict
    target_cfg_loaded = Config.fromfile(target_file)

    # check loaded
    imported_name = getattr(target_cfg_loaded, '_imported_names', None)
    # The imported variable in target_cfg should not be checked
    if imported_name is not None:
        target_cfg = {
            key: value
            for key, value in target_cfg_loaded.items()
            if key not in imported_name
        }
    else:
        target_cfg = target_cfg_loaded._cfg_dict
    default_scope = ori_cfg.get('default_scope')
    with DefaultScope.overwrite_default_scope(default_scope):
        check(ori_cfg, target_cfg)

    # check dumped
    with tempfile.NamedTemporaryFile(suffix='.py') as f:
        target_cfg_loaded.dump(file=f.name)
        target_cfg_dumped = Config.fromfile(f.name)

    with DefaultScope.overwrite_default_scope(default_scope):
        check(ori_cfg, target_cfg_dumped._cfg_dict)


def str_match(a, b):
    return sum(i == j for i, j in zip(a, b)) / len(a)


def main():
    args = parse_args()
    scope = args.scope
    if args.old is not None:
        validate(args.old, args.new, args.scope)
    else:
        old_cfgs = list(scandir('configs', recursive=True))
        new_cfgs = list(scandir(f'{scope}/configs', recursive=True))
        new_cfgs = list(new_cfg for new_cfg in new_cfgs
                        if '__init__.py' not in new_cfg)
        old_basenames = list(osp.basename(cfg) for cfg in old_cfgs)
        for new in track(new_cfgs):
            new_basename = osp.basename(new)
            all_matched = [
                str_match(new_basename, old_basename)
                for old_basename in old_basenames
            ]
            old = old_cfgs[all_matched.index(max(all_matched))]
            old = osp.join('configs', old)
            new = osp.join(f'{scope}/configs', new)
            try:
                validate(old, new, scope)
            except Exception as e:
                raise type(e)(f'Failed to check {old} and {new}') from e


if __name__ == '__main__':
    main()
