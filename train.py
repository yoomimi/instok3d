from os import path

import torch
from lightning.pytorch.utilities import rank_zero_only

import engine
from engine.components import DataModule, Solver
from engine.starter.training import get_checkpoint_dir
from instok3d.utils.weights import find_best_checkpoint, load_model_state_dict

# Modules that exist only in the joint (Stage 2) model and may be absent from `init_from`.
_GROUP_HEAD_PREFIXES = (
    'instance_query',
    'instance_encoder.',
    'anchor_assign_proj.',
    'query_assign_proj.',
    'logit_scale',
)


@rank_zero_only
def print_available_devices():
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        print(f'GPU {device}: {props.name} ({props.total_memory / 1024 ** 3:.0f} GB)')


@engine.main(config_dir='config', default_config_name='train', require='name')
def main(cfg, dm: DataModule, solver: Solver):
    print_available_devices()

    checkpoint_path = None if cfg.engine.debug else _get_resume_checkpoint_path(cfg, cfg.resume)
    init_from = cfg.get('init_from', None)
    if checkpoint_path is not None:
        print(f'Resuming from {checkpoint_path}.')
        if init_from is not None:
            print('init_from is ignored because the run is resumed.')

    trainer = engine.prepare_trainer(cfg, dm, solver)
    if (init_from is not None) and (checkpoint_path is None):
        _init_model_weights(solver, str(init_from))
    trainer.fit(model=solver, datamodule=dm, ckpt_path=checkpoint_path)


def _get_resume_checkpoint_path(cfg, resume):
    """Checkpoint to resume from: `resume` is true / 'last', 'best', a path, or null / false."""
    checkpoint_dir = get_checkpoint_dir(cfg)
    if (resume is None) or (resume is False) or (checkpoint_dir is None):
        return None

    if resume == 'best':
        checkpoint_file = find_best_checkpoint(checkpoint_dir, cfg.checkpoint.mode)
        return path.join(checkpoint_dir, checkpoint_file) if checkpoint_file is not None else None
    if (resume == 'last') or (resume is True):
        checkpoint_path = path.join(checkpoint_dir, 'last.ckpt')
        return checkpoint_path if path.exists(checkpoint_path) else None
    return resume


def _init_model_weights(solver: Solver, init_from: str):
    """Initialise `solver.model` from a model directory, a checkpoint or a safetensors file."""
    init_from = path.expanduser(init_from)
    print(f'Initializing model weights from {init_from}.')
    state_dict = load_model_state_dict(init_from, is_best=True)
    missing, unexpected = solver.model.load_state_dict(state_dict, strict=False)

    if len(unexpected) > 0:
        raise RuntimeError(f'Unexpected keys in init_from weights: {unexpected}')
    not_allowed = [key for key in missing if not key.startswith(_GROUP_HEAD_PREFIXES)]
    if len(not_allowed) > 0:
        raise RuntimeError(f'Missing keys in init_from weights: {not_allowed}')
    if len(missing) > 0:
        print(f'{len(missing)} group-head parameters are newly initialized.')


if __name__ == '__main__':
    main()
