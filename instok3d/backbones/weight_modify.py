"""Adapts VGGT weights to a different patch size.

Adapted from Uni3R (https://github.com/HorizonRobotics/Uni3R).
"""

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_PROJ_KEY = 'aggregator.patch_embed.patch_embed.proj.weight'
POS_EMBED_KEY = 'aggregator.patch_embed.pos_embed'


def resample_patch_embed(
    patch_embed: torch.Tensor,
    new_size: List[int],
    interpolation: str = 'bicubic',
    antialias: bool = True,
) -> torch.Tensor:
    """Resample a patch-embedding kernel with the pseudo-inverse resize of FlexiViT.

    Args:
        patch_embed: [out, in, kh, kw] kernel.
        new_size: target spatial size (kh', kw').

    Returns:
        [out, in, kh', kw'] kernel.
    """
    old_size = tuple(patch_embed.shape[-2:])
    new_size = tuple(new_size)
    if old_size == new_size:
        return patch_embed

    def resize(basis: np.ndarray) -> np.ndarray:
        x = torch.tensor(basis, dtype=torch.float32)[None, None]
        return F.interpolate(x, size=new_size, mode=interpolation, antialias=antialias)[0, 0].numpy()

    rows = []
    for i in range(int(np.prod(old_size))):
        basis = np.zeros(old_size, dtype=np.float64)
        basis[np.unravel_index(i, old_size)] = 1.0
        rows.append(resize(basis).reshape(-1))
    resize_mat = np.stack(rows).T
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel: torch.Tensor) -> torch.Tensor:
        return (resize_mat_pinv @ kernel.reshape(-1)).reshape(new_size)

    resample = torch.vmap(torch.vmap(resample_kernel, 0, 0), 1, 1)
    return resample(patch_embed.float()).to(patch_embed.dtype)


def checkpoint_filter_fn(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    interpolation: str = 'bicubic',
    antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """Resample the patch-embedding kernel and position embedding of a VGGT state_dict to fit ``model``."""
    out = dict(state_dict)
    patch_embed = model.aggregator.patch_embed

    kernel_size = list(patch_embed.patch_embed.proj.weight.shape[-2:])
    out[PATCH_PROJ_KEY] = resample_patch_embed(
        state_dict[PATCH_PROJ_KEY], kernel_size, interpolation, antialias,
    )

    pos_embed = state_dict[POS_EMBED_KEY]  # [1, 1 + N, C], class token first
    num_tokens = patch_embed.pos_embed.shape[1]
    if pos_embed.shape[1] != num_tokens:
        cls_token, grid = pos_embed[:, :1], pos_embed[:, 1:]
        src_size = int(round(math.sqrt(grid.shape[1])))
        tgt_size = int(round(math.sqrt(num_tokens - 1)))
        dim = grid.shape[2]
        grid = grid.view(1, src_size, src_size, dim).permute(3, 0, 1, 2).contiguous()
        grid = resample_patch_embed(grid, [tgt_size, tgt_size], interpolation, antialias)
        grid = grid.permute(1, 2, 3, 0).contiguous().view(1, -1, dim)
        out[POS_EMBED_KEY] = torch.cat([cls_token, grid], dim=1)

    return out
