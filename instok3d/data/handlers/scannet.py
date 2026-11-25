"""ScanNet data handler (one h5 file per scene)."""

from __future__ import annotations

import contextlib
import json
import logging
from os import path as osp
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py
import numpy as np
import torch
from PIL import Image

from instok3d.data.scannet_label_remap import (
    DEFAULT_LABELS,
    DEFAULT_TSV,
    build_label_remap,
    remap_labelmap,
)
from instok3d.data.scene_utils import build_scene_lists, read_scene_list
from instok3d.utils.cam import pack_cameras

from ._registry import registry
from .base import BaseDataHandler, SceneContext

logger = logging.getLogger(__name__)

_DEFAULT_PHASES = ('train', 'val', 'test')


@registry.register('scannet')
class ScanNetDataHandler(BaseDataHandler):
    """ScanNet scenes: RGB + cameras from ``<dataset_root>/<gt_subdir>/<scene>.h5``.

    The sorted scene list is split into train / val by holding out the last
    ``split_last_n_val`` scenes; the test split is read from ``test_scene_list_file``.
    Instance masks and per-frame semantic label maps are optional.
    """

    name = 'scannet'

    def __init__(
            self,
            dataset_root: str,
            gt_subdir: str = 'gt_h5',
            instance_mask_subdir: Optional[str] = 'gt_instance_mask',
            instance_mask_void_id: Optional[int] = None,

            split_last_n_val: int = 50,
            test_scene_list_file: Optional[str] = None,
            val_use_test_scene_list: bool = False,

            invalid_json_path: Optional[str] = 'assets/scannet_invalid_frames.json',
            frame_load_stride: int = 10,
            phases: Optional[List[str]] = None,

            labels_root: Optional[str] = None,
            labels_subdir: str = 'labels',
            labels_frame_fmt: str = '{:06d}',
            labels_tsv_path: str = DEFAULT_TSV,
    ):
        """
        Args:
            instance_mask_void_id: raw id of unlabeled pixels, remapped to -1.
            test_scene_list_file: text file with one test scene name per line.
            val_use_test_scene_list: validate on the test scenes instead of the val split.
            invalid_json_path: json ``{scene: 'all_invalid' | [frame ids]}`` of unusable frames.
            frame_load_stride: keep every n-th frame of the original sequence.
            phases: phases this source takes part in (default: all).
            labels_root: ``<labels_root>/<scene>/<labels_subdir>/<frame>.png`` holds raw
                ScanNet class ids (uint16, 0 = void); None disables semantic label maps.
        """
        self.dataset_root = str(dataset_root)
        self.scannet_h5_dir = osp.join(self.dataset_root, str(gt_subdir))
        self.instance_mask_subdir = str(instance_mask_subdir) if instance_mask_subdir else None
        self.instance_mask_void_id = (
            int(instance_mask_void_id) if instance_mask_void_id is not None else None
        )

        self.test_scene_list_file = str(test_scene_list_file) if test_scene_list_file else None
        self.val_use_test_scene_list = bool(val_use_test_scene_list)
        if self.val_use_test_scene_list and self.test_scene_list_file is None:
            raise ValueError(
                'val_use_test_scene_list=True requires test_scene_list_file to be set.'
            )

        self.frame_load_stride = int(frame_load_stride)
        if self.frame_load_stride <= 0:
            raise ValueError(f'frame_load_stride must be positive, got {self.frame_load_stride}')

        self._phases = (
            tuple(p.strip().lower() for p in phases) if phases is not None else _DEFAULT_PHASES
        )

        self.labels_root = str(labels_root) if labels_root else None
        self.labels_subdir = str(labels_subdir)
        self.labels_frame_fmt = str(labels_frame_fmt)
        self._labels_remap: Optional[Dict[int, int]] = None
        self._labels_other_idx = DEFAULT_LABELS.index('other') + 1
        if self.labels_root is not None:
            self._labels_remap = build_label_remap(
                labels=DEFAULT_LABELS, tsv_path=str(labels_tsv_path),
            )

        self._invalid_frames: Dict[str, Set[int]] = {}
        self._all_invalid_scenes: Set[str] = set()
        if invalid_json_path is not None and osp.isfile(invalid_json_path):
            with open(invalid_json_path, 'r') as fp:
                raw = json.load(fp)
            for scene_name, val in raw.items():
                if val == 'all_invalid':
                    self._all_invalid_scenes.add(scene_name)
                elif val:
                    self._invalid_frames[scene_name] = set(val)

        all_scenes, train_scenes, val_scenes = build_scene_lists(
            self.scannet_h5_dir, split_last_n_val,
        )
        self._all_scenes: List[str] = all_scenes
        self._train_scenes: List[str] = self._drop_all_invalid(train_scenes)
        self._val_scenes: List[str] = self._drop_all_invalid(val_scenes)
        self._cached_test_scenes: Optional[List[str]] = None

    def get_scenes(self, phase: str) -> List[str]:
        phase_norm = str(phase).strip().lower()
        if phase_norm not in self._phases:
            return []
        if phase_norm == 'train':
            return list(self._train_scenes)
        if phase_norm == 'val' and not self.val_use_test_scene_list:
            return list(self._val_scenes)
        return list(self._load_test_scenes())

    def get_valid_frame_ids(self, scene_name: str) -> np.ndarray:
        h5_path = Path(self.scannet_h5_dir) / f'{scene_name}.h5'
        with h5py.File(h5_path, 'r') as fp:
            num_stored = int(fp['colors'].shape[0])
            embedded = int(fp.attrs.get('frame_stride', 1))

        stride = self.frame_load_stride
        if stride < embedded or stride % embedded != 0:
            raise ValueError(
                f"frame_load_stride={stride} incompatible with "
                f"attrs['frame_stride']={embedded} on {scene_name}"
            )
        # Frame ids refer to the original sequence; the h5 stores every `embedded`-th frame.
        valid_ids = np.arange(num_stored, dtype=np.int64)[::stride // embedded] * embedded
        invalid_set = self._invalid_frames.get(scene_name)
        if invalid_set:
            valid_ids = valid_ids[
                ~np.isin(valid_ids, np.array(list(invalid_set), dtype=np.int64))
            ]
        if len(valid_ids) == 0:
            raise RuntimeError(
                f'No valid frames after applying frame_load_stride={stride} '
                f'for scene={scene_name!r} (stored={num_stored}, embedded={embedded}).'
            )
        return valid_ids

    @contextlib.contextmanager
    def open_scene(self, scene_name: str):
        h5_path = Path(self.scannet_h5_dir) / f'{scene_name}.h5'
        fp = h5py.File(h5_path, 'r')
        try:
            embedded = int(fp.attrs.get('frame_stride', 1))
            yield ScanNetSceneContext(
                fp=fp, handler=self, scene_name=scene_name, embedded_stride=embedded,
            )
        finally:
            fp.close()

    def _drop_all_invalid(self, scenes) -> List[str]:
        if not self._all_invalid_scenes:
            return list(scenes)
        return [s for s in scenes if s not in self._all_invalid_scenes]

    def _load_test_scenes(self) -> List[str]:
        if self._cached_test_scenes is not None:
            return self._cached_test_scenes
        if self.test_scene_list_file is None:
            raise ValueError('The test split requires test_scene_list_file to be set.')

        list_path = Path(self.test_scene_list_file)
        requested = read_scene_list(list_path)

        available = {p.stem for p in Path(self.scannet_h5_dir).glob('*.h5')}
        matched = [s for s in requested if s in available]
        if len(matched) < len(requested):
            logger.warning(
                'Test scene list %s: %d scenes missing in %s',
                list_path, len(requested) - len(matched), self.scannet_h5_dir,
            )
        if not matched:
            raise RuntimeError(
                f'No test scenes from list matched available h5 files. '
                f'list={list_path}, h5_dir={self.scannet_h5_dir}'
            )

        self._cached_test_scenes = self._drop_all_invalid(matched)
        return self._cached_test_scenes

    def _resolve_instance_mask_h5_path(self, scene_name: str) -> Path:
        mask_dir = Path(self.dataset_root) / self.instance_mask_subdir / scene_name
        if not mask_dir.is_dir():
            raise FileNotFoundError(f'instance mask dir does not exist: {mask_dir}')

        default_h5 = mask_dir / f'{scene_name}_gt_instance.h5'
        if default_h5.is_file():
            return default_h5

        h5_files = sorted(mask_dir.glob('*.h5'))
        if not h5_files:
            raise FileNotFoundError(f'no .h5 instance mask file found in: {mask_dir}')
        return h5_files[0]


class ScanNetSceneContext(SceneContext):
    """Open scene h5; the instance-mask h5 is opened on demand."""

    def __init__(
            self, *, fp: h5py.File, handler: ScanNetDataHandler, scene_name: str,
            embedded_stride: int = 1,
    ):
        self._fp = fp
        self._handler = handler
        self._scene_name = scene_name
        self._embedded_stride = int(embedded_stride)

    def _to_pos(self, indices: np.ndarray) -> np.ndarray:
        """Original frame ids -> positions in the h5 arrays."""
        return (indices // self._embedded_stride).astype(np.int64)

    def load_images(self, indices: np.ndarray) -> np.ndarray:
        positions = self._to_pos(indices)
        return np.stack([
            np.asarray(self._fp['colors'][int(p)], dtype=np.uint8)
            for p in positions.tolist()
        ])

    def load_cameras(self, indices: np.ndarray) -> torch.Tensor:
        intrinsic = np.asarray(self._fp['intrinsic'], dtype=np.float32)
        extrinsics = np.asarray(self._fp['extrinsics'], dtype=np.float32)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        H, W = float(self._fp['colors'].shape[1]), float(self._fp['colors'].shape[2])
        fxfycxcy = torch.tensor(
            [fx / W, fy / H, cx / W, cy / H], dtype=torch.float32
        )
        positions = self._to_pos(indices)
        c2w = torch.from_numpy(extrinsics[positions])  # (V, 4, 4)
        v = c2w.shape[0]
        return pack_cameras(c2w, fxfycxcy.unsqueeze(0).expand(v, -1))

    def load_instance_masks(self, indices: np.ndarray) -> Optional[np.ndarray]:
        if self._handler.instance_mask_subdir is None:
            return None
        h5_path = self._handler._resolve_instance_mask_h5_path(self._scene_name)
        with h5py.File(h5_path, 'r') as fp:
            arr = np.stack([
                np.asarray(fp[f'image_{int(fid):05d}'], dtype=np.int64)
                for fid in indices.tolist()
            ])
        raw_void = self._handler.instance_mask_void_id
        if raw_void is not None:
            arr = np.where(arr == raw_void, -1, arr)
        return arr

    def load_target_labelmap(self, indices: np.ndarray) -> Optional[np.ndarray]:
        h = self._handler
        if h._labels_remap is None:
            return None
        scene_dir = Path(h.labels_root) / self._scene_name / h.labels_subdir
        out: List[np.ndarray] = []
        for fid in indices.tolist():
            png_path = scene_dir / f'{h.labels_frame_fmt.format(int(fid))}.png'
            if not png_path.is_file():
                raise FileNotFoundError(f'labels file missing: {png_path}')
            raw = np.array(Image.open(png_path)).astype(np.int64, copy=False)
            out.append(remap_labelmap(raw, h._labels_remap, other_idx=h._labels_other_idx))
        return np.stack(out, axis=0)
