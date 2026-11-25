import os
from os import path

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError

from .reserved import ReservedKeys
from engine.utils.common import is_yaml_file_path
from engine.exceptions import ConfigNotFound


class ConfigNode:
    """A yaml config with its ``_base_`` config merged in and one child node per sub-config group."""

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig):
        return cls('', '', cfg, with_sub_configs=False, with_base_config=False)

    def __init__(self, config_dir: str, config_name: str, cfg: DictConfig = None,
                 with_base_config: bool = True,
                 with_sub_configs: bool = True):
        super().__init__()

        self._config_dir = config_dir
        self._config_name = config_name
        self._sub_nodes = {}
        self._module_name = None

        if cfg is None:
            config_path = path.join(config_dir, config_name)
            if not is_yaml_file_path(config_path):
                config_path += '.yaml'

            if not path.exists(config_path):
                raise ConfigNotFound(config_path)
            cfg = OmegaConf.load(config_path)
            available_subgroups = [x for x in os.listdir(config_dir) if path.isdir(path.join(config_dir, x))]
        else:
            available_subgroups = []

        self.cfg = cfg
        self._available_subgroups = available_subgroups

        if with_base_config:
            self._load_base_config()

        if with_sub_configs:
            self.load_sub_configs()

    def _load_base_config(self):
        if ReservedKeys.BASE not in self.cfg:
            return

        node = ConfigNode(self._config_dir, self.cfg[ReservedKeys.BASE], with_sub_configs=False)
        self.cfg = _merge_cfgs(node.cfg, self.cfg)

    def get_sub_configs(self):
        return self._sub_nodes.values()

    def get_module_name(self):
        return self._module_name

    def merge_with_loaded_config(self, loaded_cfg, groups=None):
        available_subgroups = [*self._available_subgroups, 'checkpoint']
        if groups is not None:
            available_subgroups = [x for x in available_subgroups if x in groups]

        for key, value in loaded_cfg.cfg.items():
            if key not in available_subgroups:
                continue
            sub_node = ConfigNode('', '', cfg=value, with_base_config=False)
            self._sub_nodes[key] = sub_node
            self.cfg[key] = value

    def load_sub_configs(self):
        for group in self._available_subgroups:
            if group not in self.cfg:
                continue

            sub_config_name = self.cfg[group]
            if sub_config_name is None:
                continue
            if isinstance(sub_config_name, DictConfig):
                self._sub_nodes[group] = ConfigNode.from_omegaconf(sub_config_name)
                continue

            node = ConfigNode(path.join(self._config_dir, group), sub_config_name)
            self._sub_nodes[group] = node
            node._module_name = sub_config_name
            node.cfg[ReservedKeys.MODULE_NAME] = sub_config_name

    def __getattr__(self, key):
        instance = vars(self)
        if key in self._sub_nodes:
            return self._sub_nodes[key]
        if key in instance.keys():
            return instance.get(key)
        elif key in self.cfg:
            return self.cfg[key]

        raise AttributeError(f'config has no attribute: {key}')

    def __repr__(self):
        return f'<ConfigNode: {path.join(self._config_dir, self._config_name)}>'

    def __str__(self):
        return repr(self.to_container())

    def get_cfg(self) -> DictConfig:
        cfg = self.cfg.copy()
        for key, node in self._sub_nodes.items():
            cfg[key] = node.get_cfg()

        return cfg

    @property
    def is_provider(self):
        return ReservedKeys.PROVIDER in self.cfg

    def select(self, full_key):
        keys = full_key.split('.')
        node = self
        while len(keys) > 1:
            sub_configs = node._sub_nodes
            if keys[0] not in sub_configs:
                break
            node = sub_configs[keys.pop(0)]

        last_key = '.'.join(keys)
        try:
            value = getattr(node, last_key)
        except ConfigAttributeError:
            raise AttributeError(f'config has no attribute: {full_key}')

        return value

    def update(self, full_key, value, allow_unspecified_key=False):
        if full_key.startswith('+'):
            allow_unspecified_key = True
            full_key = full_key.replace('+', '')

        keys = full_key.split('.')
        node = self
        while len(keys) > 1:
            sub_configs = node._sub_nodes
            if keys[0] not in sub_configs:
                break
            node = sub_configs[keys.pop(0)]

        if not allow_unspecified_key:
            OmegaConf.set_struct(node.cfg, True)
        last_key = '.'.join(keys)
        try:
            OmegaConf.update(node.cfg, last_key, value)
        except ConfigAttributeError:
            raise AttributeError(f'config has no attribute: {full_key}')
        OmegaConf.set_struct(node.cfg, False)

    def delete(self, full_key):
        """Remove a (dotted) key from the config tree; missing keys are ignored."""
        keys = full_key.split('.')
        node = self
        while len(keys) > 1:
            sub_configs = node._sub_nodes
            if keys[0] not in sub_configs:
                break
            node = sub_configs[keys.pop(0)]

        OmegaConf.set_struct(node.cfg, False)
        target = node.cfg
        for p in keys[:-1]:
            if p not in target:
                return
            target = target[p]
        leaf = keys[-1]
        if leaf in target:
            del target[leaf]

    def has_key(self, key, recursive=False):
        cfg = self.cfg
        if recursive:
            cfg = self.get_cfg()
        return key in cfg

    def get(self, key):
        return getattr(self, key)

    def to_container(self, resolve=True):
        cfg = self.get_cfg()
        cfg = OmegaConf.to_container(cfg, resolve=resolve)
        return cfg


def _merge_cfgs(src, other):
    merged = OmegaConf.merge(src, other)
    for key, value in other.items():
        if not isinstance(value, DictConfig):
            continue
        overwrite = value.get(ReservedKeys.OVERWRITE, False)
        if overwrite:
            merged[key] = value
            del merged[key][ReservedKeys.OVERWRITE]
    return merged
