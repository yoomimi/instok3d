"""Helpers shared by the solvers."""

import math
from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F

from instok3d.utils.cam import unpack_cameras


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int,
                         base_lr: float, min_lr: float) -> float:
    """``LambdaLR`` factor of a linear warm-up to ``base_lr`` followed by a cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine
    return lr / max(base_lr, 1e-10)


def resize_views(images: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize (B, V, C, H, W) images to ``hw``; a no-op if they already have that size."""
    hw = (int(hw[0]), int(hw[1]))
    if tuple(images.shape[-2:]) == hw:
        return images
    resized = F.interpolate(images.flatten(0, 1), size=hw, mode='bilinear', align_corners=False)
    return resized.unflatten(0, images.shape[:2])


def forward_backbone(backbone, input_images: torch.Tensor, batch: Dict[str, Any]):
    """Run the backbone on the input views (B, V, 3, H, W) with their ground-truth intrinsics."""
    if backbone.input_hw is not None:
        input_images = resize_views(input_images, backbone.input_hw)
    _, fxfycxcy = unpack_cameras(batch['input_cams'])
    return backbone(input_images, fxfycxcy)
