"""Helpers for reading weights from checkpoints and released model directories.

A model directory holds ``config.yaml`` next to its weights, which are either
``model.safetensors`` (released models) or ``checkpoints/*.ckpt`` (training runs).
"""

import os
import re
from os import path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file as _load_safetensors

SAFETENSORS_NAME = 'model.safetensors'
CHECKPOINT_DIR = 'checkpoints'
_BEST_PATTERN = re.compile(r'best-epoch=([0-9]+)-score=([0-9.]+)\.ckpt')


def subset_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return the entries whose key starts with ``prefix``, with the prefix removed."""
    n = len(prefix)
    return {k[n:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_lightning_state_dict(ckpt_path: str) -> Dict:
    """Load a Lightning checkpoint and return the whole checkpoint dict."""
    if not path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if not (isinstance(ckpt, dict) and 'state_dict' in ckpt):
        raise TypeError(f'Expected a Lightning checkpoint with "state_dict"; got {type(ckpt).__name__}.')
    return ckpt


def load_safetensors_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    """Load a safetensors file as a ``{name: tensor}`` mapping."""
    if not path.exists(weights_path):
        raise FileNotFoundError(f'Weights file not found: {weights_path}')
    return _load_safetensors(weights_path)


def find_best_checkpoint(checkpoint_dir: str, mode: str = 'max') -> Optional[str]:
    """Return the file name of the best-score checkpoint in ``checkpoint_dir``, if any."""
    if not path.isdir(checkpoint_dir):
        return None
    scored = []
    for name in os.listdir(checkpoint_dir):
        match = _BEST_PATTERN.fullmatch(name)
        if match is not None:
            scored.append((float(match.group(2)), name))
    if not scored:
        return None
    scored.sort(reverse=(mode == 'max'))
    return scored[0][1]


def find_weights_file(model_dir: str, is_best: bool = False, mode: str = 'max') -> str:
    """Locate the weights of a model directory (or accept a weights file as is)."""
    if path.isfile(model_dir):
        return model_dir

    safetensors_path = path.join(model_dir, SAFETENSORS_NAME)
    if path.exists(safetensors_path):
        return safetensors_path

    checkpoint_dir = path.join(model_dir, CHECKPOINT_DIR)
    name = find_best_checkpoint(checkpoint_dir, mode) if is_best else None
    ckpt_path = path.join(checkpoint_dir, name or 'last.ckpt')
    if not path.exists(ckpt_path):
        raise FileNotFoundError(f'No weights found in {model_dir}')
    return ckpt_path


def load_model_state_dict(model_dir: str, is_best: bool = False, mode: str = 'max',
                          prefix: str = 'model.') -> Dict[str, torch.Tensor]:
    """Load the ``model`` weights of a model directory, without the ``model.`` prefix."""
    weights_path = find_weights_file(model_dir, is_best=is_best, mode=mode)
    if weights_path.endswith('.safetensors'):
        return load_safetensors_state_dict(weights_path)
    state_dict = load_lightning_state_dict(weights_path)['state_dict']
    return subset_state_dict(state_dict, prefix)
