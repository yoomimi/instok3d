import functools
import importlib
import inspect
from typing import Callable, Union

from omegaconf import DictConfig, OmegaConf

from engine.exceptions import (ModuleImportException, InstantiateException, TargetLoadException,
                               UnknownKeyException)
from engine import utils
from .reserved import ReservedKeys

__all__ = ['instantiate', 'load_target_of_cfg']


def instantiate(cfg: Union[dict, DictConfig],
                target: Callable = None,
                as_callable: bool = False,
                filter_unknown_params: bool = False,
                **overrides):
    """Call ``target`` (default: the class named by ``cfg._target_``) with the non-reserved keys of ``cfg``.

    Args:
        cfg: keyword arguments of the target; reserved keys such as ``_target_`` are dropped.
        target: callable to use instead of ``cfg._target_``.
        as_callable: return ``functools.partial(target, **params)`` instead of calling the target.
        filter_unknown_params: drop the keys that are not parameters of the target instead of raising.
        overrides: keyword arguments that take precedence over ``cfg``.
    """
    if target is None:
        target = load_target_of_cfg(cfg)

    if not callable(target):
        raise TargetLoadException(target, f'target or {ReservedKeys.TARGET} must be callable, got {target!r}')
    params = _cfg_to_params(cfg, target, overrides, filter_unknown_params=filter_unknown_params)
    if as_callable:
        return functools.partial(target, **params)

    return call_target(target, params)


def load_target_of_cfg(cfg, current_module=None):
    """Import the object named by ``cfg._target_``; ``.Name`` is looked up in ``current_module``."""
    target_name = cfg.get(ReservedKeys.TARGET)
    if not target_name:
        raise TargetLoadException(target_name)
    return _load_target(target_name, current_module)


def _cfg_to_params(params, target, overrides, filter_unknown_params=False):
    if isinstance(params, DictConfig):
        params = OmegaConf.to_container(params, resolve=True)
    params = utils.merge_dict(params, overrides)
    _delete_reserved_keys_(params)

    unknown_keys = _get_unknown_param_keys(target, params)
    if filter_unknown_params:
        params = _filter_unknown_params(params, unknown_keys)
    elif len(unknown_keys) > 0:
        raise UnknownKeyException(target, unknown_keys)

    return params


def _delete_reserved_keys_(params_dict):
    reserved_keys = [*ReservedKeys.values(), *ReservedKeys.keys()]
    for key in list(params_dict.keys()):
        if key in reserved_keys:
            del params_dict[key]


def _get_unknown_param_keys(fn, params_dict):
    args = inspect.getfullargspec(fn).args
    if params_dict is None:
        return args
    if len(args) == 0:
        return []
    if args[0] == 'self':
        args = args[1:]

    unknown_keys = []
    for key in params_dict.keys():
        if key not in args:
            unknown_keys.append(key)

    return unknown_keys


def _filter_unknown_params(params_dict, unknown_keys):
    if params_dict is None:
        return {}

    filtered_params = {}
    for key in params_dict.keys():
        if key not in unknown_keys:
            filtered_params[key] = params_dict[key]

    return filtered_params


def call_target(target, params):
    try:
        return target(**params)
    except Exception as e:
        raise InstantiateException(e, target)


def _load_target(target: str, current_module=None):
    if target.startswith('.') and current_module is not None:
        return getattr(importlib.import_module(current_module), target[1:])
    return _import_target(target)


def _import_target(class_path):
    packages = class_path.split('.')
    class_name = packages[-1]
    packages = '.'.join(packages[:-1])

    try:
        module = importlib.import_module(packages)
    except Exception as e:
        raise ModuleImportException(class_name, e)

    if not hasattr(module, class_name):
        raise ModuleImportException(class_name, f'failed to find {class_name} from "{packages}"')

    return getattr(module, class_name)
