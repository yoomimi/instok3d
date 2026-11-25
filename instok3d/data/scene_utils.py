"""Scene-list and image-resizing helpers shared by the data handlers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def resize_rgb_batch(images: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Lanczos-resize (V, H, W, 3) uint8 images to ``target_hw``."""
    h, w = int(target_hw[0]), int(target_hw[1])
    if images.shape[1:3] == (h, w):
        return images
    out = np.empty((images.shape[0], h, w, 3), dtype=np.uint8)
    for i in range(images.shape[0]):
        im = Image.fromarray(images[i].astype(np.uint8))
        out[i] = np.asarray(im.resize((w, h), resample=Image.LANCZOS), dtype=np.uint8)
    return out


def resize_mask_batch(masks: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-resize (V, H, W) integer label maps to ``target_hw``; returns int64."""
    h, w = int(target_hw[0]), int(target_hw[1])
    if masks.shape[1:3] == (h, w):
        return masks
    out = np.empty((masks.shape[0], h, w), dtype=np.int64)
    for i in range(masks.shape[0]):
        mk = Image.fromarray(masks[i].astype(np.int32))
        mk = mk.resize((w, h), resample=Image.NEAREST)
        out[i] = np.asarray(mk, dtype=np.int64)
    return out


def build_scene_lists(h5_dir: str, split_last_n_val: int) -> Tuple[List[str], List[str], List[str]]:
    """List the scenes (``<scene>.h5``) of a directory and hold out the last ones for validation.

    Returns:
        (all, train, val) sorted scene names. ``split_last_n_val == 0`` gives (all, all, []).
    """
    h5_root = Path(h5_dir)
    if not h5_root.exists():
        raise FileNotFoundError(f'scene h5 directory does not exist: {h5_root}')
    all_scenes = sorted(
        p.stem for p in h5_root.glob('*.h5') if p.is_file() and p.stem.startswith('scene')
    )
    if not all_scenes:
        all_scenes = sorted(p.stem for p in h5_root.glob('*.h5') if p.is_file())
    if not all_scenes:
        raise RuntimeError(f'no scene h5 files found under {h5_root}')

    split_last_n_val = int(split_last_n_val)
    if split_last_n_val < 0:
        raise ValueError(f'split_last_n_val must be >= 0, got {split_last_n_val}')
    if split_last_n_val == 0:
        return all_scenes, all_scenes, []
    if len(all_scenes) <= split_last_n_val:
        raise ValueError(
            f'Not enough scenes for split_last_n_val={split_last_n_val}: total={len(all_scenes)}'
        )
    return all_scenes, all_scenes[:-split_last_n_val], all_scenes[-split_last_n_val:]


def read_scene_list(list_path: Path) -> List[str]:
    """Read a text file with one scene name (optionally ``<scene>.h5``) per line."""
    if not list_path.is_file():
        raise FileNotFoundError(f'scene list file does not exist: {list_path}')
    scenes: List[str] = []
    seen = set()
    with open(list_path, 'r') as fp:
        for raw in fp:
            line = raw.strip()
            if (not line) or line.startswith('#'):
                continue
            token = line.split()[0]
            name = Path(token).stem if token.endswith('.h5') else token
            if name not in seen:
                seen.add(name)
                scenes.append(name)
    if not scenes:
        raise RuntimeError(f'scene list is empty: {list_path}')
    return scenes
