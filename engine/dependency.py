import inspect
from typing import Dict, Optional, get_origin

from omegaconf import DictConfig

from engine.state import get_current_run_state
from engine.components import Solver, DataModule
from engine.exceptions import DependencyResolveException
from engine.config import ConfigNode, instantiate, load_target_of_cfg
from engine.utils import Singleton

DEPENDENCY_OPTIONS_KEY = '__dependency__'

__all__ = ['inject', 'DependencyResolver', 'resolver']


class DependencyResolver(Singleton):
    """Instantiates the config nodes that an entry function and its ``@inject`` targets ask for."""

    def __init__(self):
        super().__init__()

        self._injected_modules_dict = {}
        self._injected_module_names_dict = {}
        self._filter_unknown_params = False
        self.root_node: Optional[ConfigNode] = None

    def get_initialized_module_names(self) -> Dict[str, str]:
        return self._injected_module_names_dict

    def resolve_all_dependencies(self, root_node: ConfigNode, entry_fn, filter_unknown_params: bool = False):
        self.root_node = root_node
        self._filter_unknown_params = filter_unknown_params

        args = inspect.getfullargspec(entry_fn).args
        entry_injection_args = []
        for arg in args:
            if not root_node.has_key(arg, recursive=False) and (arg != 'cfg'):
                continue
            entry_injection_args.append(arg)

        entry_fn = self.inject(*entry_injection_args)(entry_fn)
        return self._resolve_dependency_of_fn(entry_fn, root_node, [],
                                              as_callable=True, filter_unknown_params=True, is_entry_fn=True)

    def _resolve_dependency_of_fn(self, fn, current_cfg: ConfigNode, current_stack=(),
                                  as_callable: bool = False, filter_unknown_params: bool = False,
                                  is_entry_fn: bool = False):
        options = self._get_inject_options(fn)
        params = inspect.signature(fn).parameters
        if isinstance(current_cfg, DictConfig):
            current_cfg = ConfigNode.from_omegaconf(current_cfg)

        kwargs = current_cfg.to_container().copy()
        if is_entry_fn:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        for config_key in options.keys():
            param_key = config_key
            if isinstance(options[config_key], str):
                config_key = options[config_key]
            if param_key not in params:
                raise DependencyResolveException(f'cannot find parameter {config_key} in {fn.__name__}')
            if config_key in ['self', 'cls']:
                raise DependencyResolveException(f'cannot inject {config_key} in {fn.__name__}')

            if config_key == 'cfg':
                kwargs[param_key] = self.root_node.get_cfg()
                continue

            param = params[param_key]
            full_config_key = config_key
            should_search_locally = False
            if config_key.startswith('.'):
                config_key = config_key[1:]
                full_config_key = '.'.join([*current_stack, config_key])
                should_search_locally = True

            param_type = param.annotation if param.annotation is not param.empty else None
            param_as_callable = False
            should_check_type = False
            if full_config_key in self._injected_modules_dict:
                target = self._injected_modules_dict[full_config_key]
            else:
                config_node = current_cfg if should_search_locally else self.root_node
                try:
                    target_cfg = config_node.select(config_key)
                except AttributeError:
                    raise DependencyResolveException(f'cannot find {full_config_key} in config')
                if target_cfg is None:
                    kwargs[param_key] = None
                    continue
                if isinstance(target_cfg, DictConfig):
                    target_cfg = ConfigNode.from_omegaconf(target_cfg)
                if not isinstance(target_cfg, ConfigNode):
                    continue

                target_fn = load_target_of_cfg(target_cfg, fn.__module__)
                param_as_callable = target_cfg.is_provider
                should_check_type = True
                target = self._resolve_dependency_of_fn(target_fn, target_cfg,
                                                        current_stack=[*current_stack, config_key],
                                                        filter_unknown_params=self._filter_unknown_params,
                                                        as_callable=param_as_callable)
                if not should_search_locally:
                    module_name = self._get_target_module_name(target, target_cfg)
                    self._injected_modules_dict[full_config_key] = target
                    self._injected_module_names_dict[full_config_key] = module_name

            # String annotations (``from __future__ import annotations``) cannot be checked.
            if should_check_type and not param_as_callable and not isinstance(param_type, (str, type(None))):
                param_type = get_origin(param_type) or param_type
                if not isinstance(target, param_type):
                    raise DependencyResolveException(f'expected {config_key} to be a {param_type}, but got {target}')

            if target is None:
                raise DependencyResolveException(f'cannot resolve {config_key} for {fn}')
            kwargs[param_key] = target
            if config_key != param_key:
                kwargs.pop(config_key)

        instance = instantiate(kwargs, fn, as_callable=as_callable, filter_unknown_params=filter_unknown_params)
        if isinstance(instance, Solver):
            get_current_run_state().solver_name = current_cfg.get_module_name()
        elif isinstance(instance, DataModule):
            get_current_run_state().datamodule_name = current_cfg.get_module_name()
        return instance

    def _get_target_module_name(self, target, cfg):
        if isinstance(cfg, ConfigNode) and (cfg.get_module_name() is not None):
            return cfg.get_module_name()
        return target.__name__ if hasattr(target, '__name__') else target.__class__.__name__

    def _get_inject_options(self, fn):
        options = getattr(fn, DEPENDENCY_OPTIONS_KEY, {})
        if hasattr(fn, '__init__'):
            options.update(getattr(fn.__init__, DEPENDENCY_OPTIONS_KEY, {}))
        return options

    def inject(self, *args, **options):
        """Decorator: fill the named parameters from the config (``param='.key'`` reads a key of the own node)."""
        def decorator_fn(fn):
            for arg in args:
                options[arg] = True

            setattr(fn, DEPENDENCY_OPTIONS_KEY, options)
            return fn

        return decorator_fn


resolver = DependencyResolver.instance()
inject = resolver.inject
