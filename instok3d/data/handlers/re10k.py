"""RealEstate10K data handler (h5 shards holding several scenes each).

A scene id is ``<file_key>/<group_key>``, e.g. ``train/000123/3a4e08a3f02ad769`` for the
group ``3a4e08a3f02ad769`` of ``<dataset_root>/train/000123.h5``.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
from io import BytesIO
from os import path as osp
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import torch
from PIL import Image

from instok3d.utils.cam import pack_cameras

from ._registry import registry
from .base import BaseDataHandler, SceneContext

logger = logging.getLogger(__name__)


@registry.register('re10k')
class RE10KDataHandler(BaseDataHandler):
    """RE10K scenes: JPEG frames resized and centre-cropped to a square, ground-truth cameras.

    Train scenes come from ``valid_list_path``; val and test both use every scene of
    ``<dataset_root>/test/*.h5``. Frames with an identity pose are invalid.
    """

    name = 're10k'

    def __init__(
            self,
            dataset_root: str,
            valid_list_path: str,
            image_resolution: int = 256,
            phases: Sequence[str] = ('train',),

            instance_mask_dir_fmt: Optional[str] = None,
            instance_mask_label_subdir: str = 'default/final-output-label-png',
            instance_mask_frame_fmt: str = '{:03d}',
            instance_mask_void_id: Optional[int] = 0,
    ):
        """
        Args:
            valid_list_path: json ``{file_key: [group_key, ...]}`` of the usable scenes.
            instance_mask_dir_fmt: enables instance masks, read from
                ``<dataset_root>/<instance_mask_dir_fmt.format(split=...)>/<group_key>/
                <instance_mask_label_subdir>/<instance_mask_frame_fmt.format(frame)>.png``
                (uint16 label maps at the native resolution). Scenes and frames without
                a mask are then dropped.
            instance_mask_void_id: raw id of unlabeled pixels, remapped to -1.
        """
        self.dataset_root = str(dataset_root)
        self.valid_list_path = str(valid_list_path)
        self.image_resolution = int(image_resolution)
        if self.image_resolution <= 0:
            raise ValueError(f'image_resolution must be positive, got {self.image_resolution}')

        self.instance_mask_dir_fmt = str(instance_mask_dir_fmt) if instance_mask_dir_fmt else None
        self.instance_mask_label_subdir = str(instance_mask_label_subdir)
        self.instance_mask_frame_fmt = str(instance_mask_frame_fmt)
        self.instance_mask_void_id = (
            int(instance_mask_void_id) if instance_mask_void_id is not None else None
        )
        self._masks_enabled = self.instance_mask_dir_fmt is not None
        self._mask_frame_cache: Dict[str, np.ndarray] = {}

        self._phases = tuple(p.strip().lower() for p in phases)
        for p in self._phases:
            if p not in ('train', 'val', 'test'):
                raise ValueError(
                    f"RE10KDataHandler supports phases train/val/test, got phases={phases!r}"
                )

        self._train_scenes: List[str] = (
            self._discover_train_scenes() if 'train' in self._phases else []
        )
        self._test_scenes: List[str] = (
            self._discover_test_scenes()
            if ('val' in self._phases or 'test' in self._phases) else []
        )
        if self._masks_enabled:
            self._train_scenes = self._filter_to_mask_scenes(self._train_scenes, 'train')
            self._test_scenes = self._filter_to_mask_scenes(self._test_scenes, 'test')

    def get_scenes(self, phase: str) -> List[str]:
        phase_norm = str(phase).strip().lower()
        if phase_norm not in self._phases:
            return []
        if phase_norm == 'train':
            return list(self._train_scenes)
        return list(self._test_scenes)

    def get_valid_frame_ids(self, scene_name: str) -> np.ndarray:
        file_key, group_key = self._parse_scene_id(scene_name)
        h5_path = osp.join(self.dataset_root, f'{file_key}.h5')
        with h5py.File(h5_path, 'r', swmr=True) as fp:
            poses = np.asarray(fp[group_key]['gt_camera_pose'], dtype=np.float32)
        invalid_mask = _identity_pose_mask(poses)
        valid_ids = np.arange(poses.shape[0], dtype=np.int64)[~invalid_mask]
        if len(valid_ids) == 0:
            raise RuntimeError(
                f'No valid frames after identity-pose filter for scene={scene_name!r}'
            )
        if self._masks_enabled:
            mask_ids = self._mask_frame_ids(scene_name)
            valid_ids = valid_ids[np.isin(valid_ids, mask_ids)]
            if len(valid_ids) == 0:
                raise RuntimeError(f'No mask-covered valid frames for scene={scene_name!r}')
        return valid_ids

    @contextlib.contextmanager
    def open_scene(self, scene_name: str):
        file_key, group_key = self._parse_scene_id(scene_name)
        h5_path = osp.join(self.dataset_root, f'{file_key}.h5')
        fp = h5py.File(h5_path, 'r', swmr=True)
        try:
            group = fp[group_key]
            yield RE10KSceneContext(
                group=group,
                handler=self,
                scene_name=scene_name,
                image_offsets=np.asarray(group['image_offsets'], dtype=np.int64),
                image_bytes=group['images'],
            )
        finally:
            fp.close()

    def _discover_train_scenes(self) -> List[str]:
        with open(self.valid_list_path, 'r') as fp:
            valid_dict = json.load(fp)

        scene_ids: List[str] = []
        for file_key, group_keys in sorted(valid_dict.items()):
            if file_key.split('/', 1)[0] != 'train' or '/' not in file_key:
                continue
            if not osp.exists(osp.join(self.dataset_root, f'{file_key}.h5')):
                continue
            for group_key in group_keys:
                scene_ids.append(f'{file_key}/{group_key}')
        return scene_ids

    def _discover_test_scenes(self) -> List[str]:
        test_root = osp.join(self.dataset_root, 'test')
        if not osp.isdir(test_root):
            logger.warning('RE10KDataHandler: %s does not exist; no test scenes', test_root)
            return []
        scene_ids: List[str] = []
        for shard_path in sorted(glob.glob(osp.join(test_root, '*.h5'))):
            file_key = 'test/' + osp.splitext(osp.basename(shard_path))[0]
            try:
                with h5py.File(shard_path, 'r', swmr=True) as fp:
                    for group_key in fp.keys():
                        scene_ids.append(f'{file_key}/{group_key}')
            except OSError as e:
                logger.warning('RE10K test shard %s unreadable: %s', shard_path, e)
        return scene_ids

    @staticmethod
    def _parse_scene_id(scene_name: str) -> Tuple[str, str]:
        parts = scene_name.rsplit('/', 1)
        if len(parts) != 2:
            raise ValueError(
                f'scene_name must be "<file_key>/<group_key>", got {scene_name!r}'
            )
        return parts[0], parts[1]

    def _filter_to_mask_scenes(self, scenes: List[str], split: str) -> List[str]:
        """Keep the scenes that have a mask directory with at least one label map."""
        root = osp.join(self.dataset_root, self.instance_mask_dir_fmt.format(split=split))
        groups: Set[str] = set()
        if osp.isdir(root):
            groups = {
                entry for entry in os.listdir(root)
                if not entry.startswith('.') and osp.isdir(osp.join(root, entry))
            }
        elif scenes:
            logger.warning('RE10KDataHandler: mask root does not exist: %s', root)
        return [
            s for s in scenes
            if self._parse_scene_id(s)[1] in groups and len(self._mask_frame_ids(s)) > 0
        ]

    def _mask_frame_ids(self, scene_name: str) -> np.ndarray:
        """Sorted int64 ids of the frames that have a label map."""
        cached = self._mask_frame_cache.get(scene_name)
        if cached is not None:
            return cached
        mask_dir = self._resolve_mask_dir(scene_name)
        ids: List[int] = []
        if osp.isdir(mask_dir):
            for fn in os.listdir(mask_dir):
                if fn.endswith('.png') and fn[:-4].isdigit():
                    ids.append(int(fn[:-4]))
        arr = np.asarray(sorted(ids), dtype=np.int64)
        self._mask_frame_cache[scene_name] = arr
        return arr

    def _resolve_mask_dir(self, scene_name: str) -> str:
        file_key, group_key = self._parse_scene_id(scene_name)
        split = file_key.split('/', 1)[0]
        return osp.join(
            self.dataset_root,
            self.instance_mask_dir_fmt.format(split=split),
            group_key,
            self.instance_mask_label_subdir,
        )


class RE10KSceneContext(SceneContext):
    """Open h5 group of one scene."""

    def __init__(
            self,
            *,
            group: h5py.Group,
            handler: RE10KDataHandler,
            scene_name: str,
            image_offsets: np.ndarray,
            image_bytes: h5py.Dataset,
    ):
        self._group = group
        self._handler = handler
        self._scene_name = scene_name
        self._image_offsets = image_offsets
        self._image_bytes = image_bytes  # concatenated JPEG buffers, 1-D uint8

    def load_images(self, indices: np.ndarray) -> np.ndarray:
        res = self._handler.image_resolution
        out = np.empty((len(indices), res, res, 3), dtype=np.uint8)
        for i, idx in enumerate(indices.tolist()):
            jpeg_buf = self._read_jpeg_buf(int(idx))
            img = Image.open(BytesIO(bytes(jpeg_buf))).convert('RGB')
            out[i] = _resize_and_center_crop(img, res, Image.LANCZOS)
        return out

    def load_cameras(self, indices: np.ndarray) -> torch.Tensor:
        intrinsics = np.asarray(self._group['gt_intrinsics'], dtype=np.float32)
        extrinsics = np.asarray(self._group['gt_camera_pose'], dtype=np.float32)
        loaded_intrinsics = intrinsics[indices]
        c2w = extrinsics[indices]

        # Intrinsics are stored for the square crop; its principal point is the centre.
        res = float(self._handler.image_resolution)
        fxfycxcy = np.stack(
            [
                loaded_intrinsics[:, 0, 0] / res,
                loaded_intrinsics[:, 1, 1] / res,
                np.full(loaded_intrinsics.shape[0], 0.5, dtype=np.float32),
                np.full(loaded_intrinsics.shape[0], 0.5, dtype=np.float32),
            ],
            axis=-1,
        )
        return pack_cameras(torch.from_numpy(c2w).float(), torch.from_numpy(fxfycxcy).float())

    def load_instance_masks(self, indices: np.ndarray) -> Optional[np.ndarray]:
        h = self._handler
        if not h._masks_enabled:
            return None
        mask_dir = h._resolve_mask_dir(self._scene_name)
        res = h.image_resolution
        out = np.empty((len(indices), res, res), dtype=np.int64)
        for i, idx in enumerate(indices.tolist()):
            png_path = osp.join(mask_dir, f'{h.instance_mask_frame_fmt.format(int(idx))}.png')
            if not osp.isfile(png_path):
                raise FileNotFoundError(f'instance mask PNG missing: {png_path}')
            label = Image.fromarray(np.asarray(Image.open(png_path)).astype(np.int32))
            out[i] = _resize_and_center_crop(label, res, Image.NEAREST)
        raw_void = h.instance_mask_void_id
        if raw_void is not None:
            out = np.where(out == raw_void, -1, out)
        return out

    def _read_jpeg_buf(self, idx: int) -> np.ndarray:
        offsets = self._image_offsets
        start = int(offsets[idx])
        end = int(offsets[idx + 1]) if idx + 1 < len(offsets) else int(self._image_bytes.shape[0])
        return np.asarray(self._image_bytes[start:end], dtype=np.uint8)


def _identity_pose_mask(poses: np.ndarray) -> np.ndarray:
    return (
        (poses[:, 0, 0] == 1)
        & (poses[:, 1, 1] == 1)
        & (poses[:, 2, 2] == 1)
        & (poses[:, 0, 3] == 0)
        & (poses[:, 1, 3] == 0)
        & (poses[:, 2, 3] == 0)
    )


def _resize_and_center_crop(img: Image.Image, res: int, resample: int) -> np.ndarray:
    """Scale the shorter side to ``res``, then centre-crop to (res, res)."""
    w, h = img.size
    scale = max(res / h, res / w)
    w_s = round(w * scale)
    h_s = round(h * scale)
    img = img.resize((w_s, h_s), resample)
    col = (w_s - res) // 2
    row = (h_s - res) // 2
    img = img.crop((col, row, col + res, row + res))
    return np.array(img)
