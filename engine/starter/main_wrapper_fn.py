from typing import Optional, List
import functools
import os
from os import path

import lightning.pytorch as pl

from engine.dependency import resolver
from engine.utils.common import remove_extensions
from .default_configs import EngineConfig
from .parser import EntryConfigParser
from engine.state import get_current_run_state


def main(config_dir: str = 'config/',
         default_config_name: str = 'train',
         output_dir: Optional[str] = 'logs/',
         fix_seed: bool = True,
         load_groups: Optional[List[str]] = None,
         require: Optional[str] = None,
         ):
    """Decorator of an entry point: parses the config and injects the parameters of the function from it.

    `engine.load=<dir>` merges the config saved in `<dir>` into the entry config;
    `load_groups` restricts which sub-configs are taken from the saved one.
    `require` names an engine argument that must be given: 'name' (unless `engine.debug`) or 'load'.
    """
    parser = EntryConfigParser.instance()
    cfg = parser.parse(config_dir, default_config_name)
    engine_cfg: EngineConfig = cfg.engine
    if (require == 'name') and (engine_cfg.name is None) and not engine_cfg.debug:
        raise SystemExit('Name the run with e.name=<run> (or pass e.debug=true for a quick check).')
    if (require == 'load') and (engine_cfg.load is None):
        raise SystemExit('Pass the model directory or training run with e.load=<dir>.')
    should_load_saved_config = isinstance(engine_cfg.load, str)
    should_save_outputs = False

    if (engine_cfg.seed is not None) and fix_seed:
        pl.seed_everything(engine_cfg.seed)

    state = get_current_run_state()
    state.job_name = _get_job_name(engine_cfg)
    if should_load_saved_config:
        if not path.exists(engine_cfg.load):
            raise SystemExit(f'e.load={engine_cfg.load} does not exist.')
        load_dir = engine_cfg.load if path.isdir(engine_cfg.load) else path.dirname(engine_cfg.load)
        if not path.exists(path.join(load_dir, 'config.yaml')):
            raise SystemExit(f'e.load={engine_cfg.load}: no config.yaml found in {load_dir}.')
        state.output_dir = load_dir
        loaded_cfg = parser.load_saved_config(load_dir)
        cfg.merge_with_loaded_config(loaded_cfg, groups=load_groups)

    elif (output_dir is not None) and (state.job_name is not None):
        state.output_dir = path.join(output_dir, state.job_name)
        if not engine_cfg.debug:
            should_save_outputs = True

    parser.handle_overrides(cfg)

    def decorator_fn(fn):
        if should_save_outputs:
            os.makedirs(state.output_dir, exist_ok=True)
            parser.dump_current_config(state.output_dir, cfg)
        state.should_save_outputs = should_save_outputs

        fn = resolver.resolve_all_dependencies(cfg, fn)

        @functools.wraps(fn)
        def main_wrapper_fn(*args, **kwargs):
            fn(*args, **kwargs)

        return main_wrapper_fn

    return decorator_fn


def _get_job_name(cfg: EngineConfig):
    if isinstance(cfg.name, str):
        job_name = cfg.name
    elif isinstance(cfg.load, str):
        job_name = cfg.load.rstrip('/').split('/')[-1]
        job_name = remove_extensions(job_name, ['.yaml', '.yml'])
    else:
        job_name = None

    return job_name
