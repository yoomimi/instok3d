from dataclasses import dataclass
from typing import Optional, Union, Any

from omegaconf import OmegaConf

from engine.config import ConfigNode

__all__ = ['get_engine_config', 'TrainingConfigGetter', 'LoggingConfig']


@dataclass
class EngineConfig:
    name: Optional[str] = None
    seed: int = 1234
    gpus: int = -1
    load: Optional[str] = None
    debug: Union[bool, int] = False


@dataclass
class LoggingConfig:
    logger: Optional[str] = None
    logger_config_path: Optional[str] = None
    auto_group: bool = True
    auto_tag: Any = True
    project: Any = None


@dataclass
class CheckpointConfig:
    save_dir: Optional[str] = 'checkpoints'
    save_top_k: int = 1
    mode: str = 'max'
    every_n_epochs: int = 1
    weights_only: bool = False


@dataclass
class DebuggingConfig:
    profiler: Optional[str] = None
    batch_size: int = -1
    val_batch_size: int = -1
    with_mp: bool = False


def get_engine_config(cfg: ConfigNode) -> EngineConfig:
    engine_cfg = OmegaConf.structured(EngineConfig)
    if cfg.has_key('engine') and cfg.engine is not None:
        engine_cfg = OmegaConf.merge(engine_cfg, cfg.engine)

    return engine_cfg


class TrainingConfigGetter:
    CONFIG_KEYS = ['checkpoint', 'debugging', 'logging', 'trainer']

    @staticmethod
    def get_default_config(cfg: ConfigNode, name):
        schema = TrainingConfigGetter._get_schema(name)
        if schema is None:
            return cfg.get(name) if cfg.has_key(name) else None

        result_cfg = OmegaConf.structured(schema)
        if cfg.has_key(name) and cfg.get(name) is not None:
            result_cfg = OmegaConf.merge(result_cfg, cfg.get(name))
        return result_cfg

    @staticmethod
    def _get_schema(name):
        if name == 'checkpoint':
            return CheckpointConfig
        elif name == 'debugging':
            return DebuggingConfig
        elif name == 'logging':
            return LoggingConfig
        else:
            return None
