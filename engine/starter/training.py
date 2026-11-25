import os
from os import path
import time
from typing import Union, List
from collections.abc import Iterable

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, DictConfig, read_write

from engine.state import get_current_run_state, to_output_dir
from engine.dependency import DependencyResolver
from engine.components import DataModule, Solver
from .default_configs import EngineConfig, LoggingConfig


def get_checkpoint_dir(cfg):
    return to_output_dir(cfg.checkpoint.save_dir)


def prepare_trainer(cfg: DictConfig, dm: DataModule, solver: Solver):
    """Build the Lightning trainer of a run; ``engine.debug`` runs a few batches without logging or checkpoints."""
    logging_cfg = cfg.get('logging', OmegaConf.create({}))
    checkpoint_cfg = cfg.get('checkpoint', OmegaConf.create({}))
    debugging_cfg = cfg.get('debugging', OmegaConf.create({}))
    trainer_cfg = cfg.get('trainer', OmegaConf.create({}))
    engine_cfg: EngineConfig = cfg.get('engine')
    state = get_current_run_state()

    use_logger = True
    profiler = None
    strategy = trainer_cfg.get('strategy', 'ddp')

    if debugging_cfg.profiler is not None:
        use_logger = False
        profiler = debugging_cfg.profiler
        with read_write(cfg):
            trainer_cfg.max_epochs = 1

    for module in solver.modules():
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()

    # -1 uses every visible GPU.
    gpus = engine_cfg.gpus
    use_multi_gpu = ((gpus == -1) or (gpus > 1)) and (torch.cuda.device_count() >= 2)

    is_debug_mode = bool(engine_cfg.debug)
    if is_debug_mode:
        use_logger = False
        solver.enable_debug_mode()
        dm.enable_debug_mode(engine_cfg.debug, debugging_cfg.with_mp,
                             debugging_cfg.batch_size,
                             debugging_cfg.val_batch_size)
        use_multi_gpu = False
        gpus = 1
    if not use_multi_gpu:
        strategy = None

    logger = _create_logger(cfg, logging_cfg) if use_logger else False

    callbacks = []
    checkpoint_dir = checkpoint_cfg.save_dir
    if not is_debug_mode and isinstance(checkpoint_dir, str) and state.should_save_outputs:
        checkpoint_filename = 'best-{epoch}-{score:.4f}'
        checkpoint_callback = ModelCheckpoint(dirpath=to_output_dir(checkpoint_dir),
                                              save_top_k=checkpoint_cfg.save_top_k,
                                              monitor='score',
                                              mode=checkpoint_cfg.mode,
                                              filename=checkpoint_filename,
                                              save_weights_only=checkpoint_cfg.weights_only,
                                              every_n_epochs=checkpoint_cfg.every_n_epochs,
                                              save_last=True,
                                              )
        checkpoint_callback.CHECKPOINT_NAME_LAST = 'last'
        callbacks.append(checkpoint_callback)

    trainer_params = OmegaConf.to_container(trainer_cfg, resolve=True)
    trainer_params.update(
        logger=logger,
        devices=gpus,
        callbacks=callbacks,
        strategy=strategy,
        fast_dev_run=engine_cfg.debug,
        profiler=profiler,
    )
    effective_batch_size = trainer_params.get('effective_batch_size', None)
    if effective_batch_size is not None:
        accumulate_size = max(1, effective_batch_size // dm.batch_size)
        print(f'Setting gradient accumulation to {accumulate_size} (effective batch: {effective_batch_size})')
        trainer_params['accumulate_grad_batches'] = accumulate_size
        trainer_params.pop('effective_batch_size')

    if strategy is None:
        trainer_params.pop('strategy')

    return pl.Trainer(**trainer_params)


def _create_logger(cfg, logging_cfg: LoggingConfig):
    state = get_current_run_state()
    if (logging_cfg.logger is None) or (state.job_name is None):
        return False
    if logging_cfg.logger == 'wandb':
        has_wandb_config = (isinstance(logging_cfg.logger_config_path, str)
                            and path.exists(logging_cfg.logger_config_path))
        if has_wandb_config:
            return _initialize_wandb_logger(cfg, logging_cfg)
        print(f'{logging_cfg.logger_config_path} not found: logging to csv instead of wandb.')
    return CSVLogger(save_dir=state.output_dir, name='csv_logs')


def _initialize_wandb_logger(cfg: DictConfig, logging_cfg: LoggingConfig):
    state = get_current_run_state()
    name = state.job_name

    group = auto_generate_group() if logging_cfg.auto_group else None
    tags = auto_generate_tags(logging_cfg.auto_tag)

    wandb_cfg = OmegaConf.load(logging_cfg.logger_config_path)
    os.environ['WANDB_API_KEY'] = wandb_cfg.api_key
    run_id = f'{int(time.time())}_{name}'
    config = OmegaConf.to_container(cfg, resolve=True)

    return WandbLogger(
        save_dir=state.output_dir,
        name=name,
        group=group,
        id=run_id,
        project=logging_cfg.project,
        entity=wandb_cfg.entity,
        tags=tags,
        config=config,
    )


def auto_generate_group():
    state = get_current_run_state()
    names = [state.solver_name, state.datamodule_name]
    names = [x for x in names if isinstance(x, str) and len(x) > 0]
    if len(names) == 0:
        return None

    return '_'.join(names)


def auto_generate_tags(auto_tag: Union[str, bool, List[str]]):
    if (auto_tag is None) or (auto_tag is False) or (auto_tag == 'false'):
        return None

    if (auto_tag == 'true') or (auto_tag == 'all'):
        auto_tag = True
    resolver = DependencyResolver.instance()
    all_module_names_dict = resolver.get_initialized_module_names()
    if isinstance(auto_tag, Iterable):
        module_names_dict = {key: all_module_names_dict[key] for key in auto_tag if key in all_module_names_dict}
    else:
        module_names_dict = all_module_names_dict

    if len(module_names_dict.keys()) == 0:
        return None
    return [f'{key}={value}' for key, value in module_names_dict.items()]

