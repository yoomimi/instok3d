import sys
from os import path
import re

from omegaconf import OmegaConf

from engine.utils import Singleton
from engine.utils.common import is_yaml_file_path
from engine.config import ConfigNode
from .default_configs import get_engine_config, TrainingConfigGetter

_OMEGA_CONF_ARG_PATTERN = re.compile(r"^([^=]+)=(.*)$")
_OMEGA_CONF_DELETE_PATTERN = re.compile(r"^~(.+)$")

_SHORTCUTS_DICT = {
    'e': 'engine',
    'd': 'debugging',
    't': 'trainer',
    'l': 'logging',
}
_OUTPUT_CONFIG_NAME = 'config.yaml'


class EntryConfigParser(Singleton):
    """Loads the entry config named on the command line and applies the ``key=value`` / ``~key`` overrides."""

    def dump_current_config(self, output_dir: str, cfg: ConfigNode):
        cfg = cfg.get_cfg()
        content = OmegaConf.to_yaml(cfg, resolve=True)
        with open(path.join(output_dir, _OUTPUT_CONFIG_NAME), 'w') as f:
            f.write(content)

    def load_saved_config(self, output_dir: str):
        cfg = ConfigNode(output_dir, _OUTPUT_CONFIG_NAME, with_sub_configs=False, with_base_config=False)
        return cfg

    def parse(self, config_dir: str, default_config_name: str, is_script_entry: bool = False):
        config_dir, filename = self._get_entry_config_file_path(config_dir, default_config_name, is_script_entry)

        cfg = ConfigNode(config_dir, filename, with_sub_configs=False)
        cfg.update('+engine', get_engine_config(cfg))
        for key in TrainingConfigGetter.CONFIG_KEYS:
            self._add_default_cfg_if_exists(cfg, key)

        # The engine overrides are needed before the sub-configs are loaded.
        depth_overrides = self._parse_omega_conf_args(sys.argv[1:])
        engine_overrides = []
        for override in depth_overrides:
            engine_overrides.extend([x for x in override if x.startswith('engine')])

        self._update_overrides(cfg, engine_overrides)

        return cfg

    def _add_default_cfg_if_exists(self, cfg: ConfigNode, key):
        default_cfg = TrainingConfigGetter.get_default_config(cfg, key)
        if default_cfg is not None:
            cfg.update(f'+{key}', default_cfg)

    def handle_overrides(self, cfg: ConfigNode, allow_unspecified_key: bool = False):
        nodes = [cfg]

        depth_overrides = self._parse_omega_conf_args(sys.argv[1:])
        if len(depth_overrides) == 0:
            cfg.load_sub_configs()

        for depth, overrides in enumerate(depth_overrides):
            self._update_overrides(cfg, overrides, allow_unspecified_key)
            next_nodes = []
            for node in nodes:
                node.load_sub_configs()
                next_nodes.extend(node.get_sub_configs())
            nodes = next_nodes

    def _update_overrides(self, cfg, overrides, allow_unspecified_key=False):
        for override in overrides:
            if override.startswith('~'):
                cfg.delete(override[1:])
                continue
            tokens = override.split('=')
            key, value = tokens[0], '='.join(tokens[1:])
            # Let OmegaConf parse the value (numbers, booleans, null, lists).
            value = OmegaConf.from_dotlist([f'key={value}']).key
            cfg.update(key, value, allow_unspecified_key=allow_unspecified_key)

    def _get_entry_config_file_path(self, default_config_dir: str, default_config_name: str,
                                    is_script_entry: bool = False):
        tokens = sys.argv[1:]
        if is_script_entry:
            tokens = tokens[1:]

        if len(tokens) == 0:
            return default_config_dir, default_config_name

        config_dir = default_config_dir
        filename = tokens[0]
        if '=' in filename:
            return default_config_dir, default_config_name
        if '/' in filename:
            _tokens = filename.split('/')
            config_dir = '/'.join(_tokens[:-1])
            filename = _tokens[-1]

        file_path = path.join(config_dir, filename)
        if not is_yaml_file_path(file_path):
            file_path += '.yaml'
        if not path.exists(file_path):
            raise Exception(f'failed to find entry config: {file_path}')

        return config_dir, filename

    def _parse_omega_conf_args(self, args):
        """Group the overrides by the depth of their key: ``[[depth-0 overrides], [depth-1 overrides], ...]``."""
        depth_overrides_dict = {}
        for arg in args:
            m = _OMEGA_CONF_ARG_PATTERN.match(arg)
            if m is not None:
                key, value = _expand_shortcut(m.group(1)), m.group(2)
                depth_overrides_dict.setdefault(key.count('.'), []).append(f'{key}={value}')
                continue

            md = _OMEGA_CONF_DELETE_PATTERN.match(arg)
            if md is None:
                continue
            key = _expand_shortcut(md.group(1))
            depth_overrides_dict.setdefault(key.count('.'), []).append(f'~{key}')

        depth_overrides = []
        keys = list(depth_overrides_dict.keys())
        if len(keys) == 0:
            return []
        max_depth = max(keys)
        for depth in range(max_depth + 1):
            depth_overrides.append(depth_overrides_dict.get(depth, []))

        return depth_overrides


def _expand_shortcut(key: str) -> str:
    """Expand a leading shortcut group, e.g. ``e.debug`` -> ``engine.debug``."""
    first_group = key.split('.')[0]
    if first_group in _SHORTCUTS_DICT:
        key = key.replace(f'{first_group}.', f'{_SHORTCUTS_DICT[first_group]}.', 1)
    return key
