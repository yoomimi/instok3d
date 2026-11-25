"""ScanNet test set curated by LSM (evaluation only).

Expected layout:

    <dataset_root>/selected_seqs_test.json    {scene: [frame, ...]}
    <dataset_root>/<scene>/images/<frame>.jpg
    <dataset_root>/<scene>/images/<frame>.npz  camera_pose (4x4 camera-to-world),
                                               camera_intrinsics (3x3)
    <dataset_root>/<scene>/labels/<frame>.png  raw ScanNet class ids
    <instance_mask_h5_root>/<scene>/<scene>_gt_instance.h5  image_{frame:05d} -> instance ids

The 'dust3r' crop is adapted from DUSt3R (https://github.com/naver/dust3r).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from engine import inject
from engine.components import DataModule
from instok3d.data.scannet_label_remap import (
    DEFAULT_LABELS,
    DEFAULT_TSV,
    build_label_remap,
    remap_labelmap,
)
from instok3d.utils.cam import normalize_cameras_first_frame, pack_cameras

logger = logging.getLogger(__name__)

DEFAULT_LLFF_HOLD = 8
DEFAULT_TEST_IDS: Tuple[int, ...] = (1, 4)
FRAME_LIST_FILE = 'selected_seqs_test.json'
CROP_MODES = ('dust3r', 'pp_square')


def _camera_matrix_of_crop(
    K: np.ndarray, in_res: np.ndarray, out_res: np.ndarray, scaling: float = 1.0,
) -> np.ndarray:
    """Intrinsics after rescaling by ``scaling`` and centre-cropping ``in_res`` to ``out_res``."""
    margins = np.asarray(in_res) * scaling - np.asarray(out_res)
    assert np.all(margins >= -1e-8)
    offset = 0.5 * margins
    K = K.copy()
    # The scaling is applied with the pixel origin at the corner of the top-left pixel.
    K[:2, 2] += 0.5
    K[:2, :] *= scaling
    K[:2, 2] -= offset
    K[:2, 2] -= 0.5
    return K


def crop_resize_dust3r(
    image: np.ndarray,
    K: np.ndarray,
    target_hw: Tuple[int, int],
    label_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Crop around the principal point and resize to ``target_hw`` as DUSt3R does at test time.

    Args:
        image: (H, W, 3) uint8.
        K: (3, 3) pixel intrinsics.
        target_hw: (H_t, W_t).
        label_map: optional (H, W) integer map, resized with nearest-neighbour sampling.
    Returns:
        the (H_t, W_t, 3) uint8 image, its (3, 3) intrinsics and the (H_t, W_t) label map.
    """
    resolution = np.array([int(target_hw[1]), int(target_hw[0])])  # (W, H)
    image_pil = Image.fromarray(image)
    K = K.copy().astype(np.float64)

    # Largest rectangle centred on the principal point.
    W, H = image_pil.size
    cx, cy = int(np.round(K[0, 2])), int(np.round(K[1, 2]))
    mx = min(cx, W - cx)
    my = min(cy, H - cy)
    l, t = cx - mx, cy - my
    r, b = cx + mx, cy + my
    image_pil = image_pil.crop((l, t, r, b))
    K[0, 2] -= l
    K[1, 2] -= t

    # Aspect-preserving rescale that covers the target resolution.
    in_res = np.array(image_pil.size)
    scale = max(resolution / in_res) + 1e-8
    out_res = np.floor(in_res * scale).astype(int)
    image_pil = image_pil.resize(tuple(out_res), resample=Image.LANCZOS)
    K = _camera_matrix_of_crop(K, in_res, out_res, scaling=scale)

    # Centre crop to the target resolution.
    K_centred = _camera_matrix_of_crop(K, out_res, resolution)
    l2, t2 = (int(v) for v in np.int32(np.round(K[:2, 2] - K_centred[:2, 2])))
    r2, b2 = l2 + int(resolution[0]), t2 + int(resolution[1])
    image_pil = image_pil.crop((l2, t2, r2, b2))
    K[0, 2] -= l2
    K[1, 2] -= t2

    if label_map is not None:
        label_map = cv2.resize(
            label_map[t:b, l:r], tuple(out_res), interpolation=cv2.INTER_NEAREST,
        )[t2:b2, l2:r2]
    return np.asarray(image_pil), K, label_map


def _crop_resize_pp_square(
    image: np.ndarray,
    K: np.ndarray,
    target_hw: Tuple[int, int],
    label_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Crop the largest square centred on the principal point and resize it to ``target_hw``.

    Returns the (H_t, W_t, 3) uint8 image, the (4,) float32 intrinsics (fx, fy, cx, cy)
    normalised by the image size, and the (H_t, W_t) label map.
    """
    H, W = int(image.shape[0]), int(image.shape[1])
    H_t, W_t = int(target_hw[0]), int(target_hw[1])
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    half = int(min(cx, W - cx, cy, H - cy))
    if half <= 0:
        side = min(H, W)
        left = (W - side) // 2
        top = (H - side) // 2
    else:
        side = 2 * half
        left = int(round(cx - half))
        top = int(round(cy - half))

    image = image[top:top + side, left:left + side]
    if image.shape[:2] != (H_t, W_t):
        image = np.asarray(
            Image.fromarray(image).resize((W_t, H_t), resample=Image.LANCZOS), dtype=np.uint8,
        )
    if label_map is not None:
        label_map = _resize_nearest(label_map[top:top + side, left:left + side], (H_t, W_t))

    scale_x = W_t / float(side)
    scale_y = H_t / float(side)
    fx *= scale_x
    fy *= scale_y
    cx = (cx - left) * scale_x
    cy = (cy - top) * scale_y
    fxfycxcy = np.array([fx / W_t, fy / H_t, cx / W_t, cy / H_t], dtype=np.float32)
    return image, fxfycxcy, label_map


def _resize_nearest(id_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an (H, W) integer id map to int64 ``target_hw``."""
    H_t, W_t = target_hw
    if id_map.shape == (H_t, W_t):
        return id_map.astype(np.int64, copy=False)
    pil = Image.fromarray(id_map.astype(np.int32, copy=False))
    return np.asarray(pil.resize((W_t, H_t), resample=Image.NEAREST), dtype=np.int64)


class ScanNetLSMTestDataset(Dataset):
    """Context / target view sets of the ScanNet test scenes.

    ``num_views == 2``: every frame index ``t`` (in sorted order) with ``t % llff_hold in test_ids``
    is a target view and its neighbours ``t - 1`` and ``t + 1`` are the two context views.
    ``num_views >= 3``: one sample per scene built from its first ``2 * num_views - 1`` frames;
    even positions are the context views and odd positions the target views.

    ``crop_mode`` selects the pre-processing of the frames: ``'dust3r'`` (DUSt3R-style crop) or
    ``'pp_square'`` (principal-point-centred square crop followed by a Lanczos resize).

    Sample (``N`` context views, ``M`` target views, ``H, W = target_hw``):
        input_images / target_images: (N | M, 3, H, W) float32 in [0, 1]
        input_cams / target_cams: (N | M, 5, 4) float32, relative to the first context view
        target_masks: (M, H, W) int64 instance ids, if ``instance_mask_h5_root`` is set
        target_labelmap / input_labelmap: (M | N, H, W) int64 class ids (0 = void), if
            ``labels_subdir`` is set
    """

    def __init__(
        self,
        scene_names: Sequence[str],
        dataset_root: str,
        target_hw: Sequence[int] = (224, 224),
        llff_hold: int = DEFAULT_LLFF_HOLD,
        test_ids: Sequence[int] = DEFAULT_TEST_IDS,
        invalid_json_path: Optional[str] = None,
        num_views: int = 2,
        crop_mode: str = 'dust3r',
        instance_mask_h5_root: Optional[str] = None,
        instance_mask_void_id: Optional[int] = None,
        labels_subdir: Optional[str] = None,
        labels_classes: Sequence[str] = DEFAULT_LABELS,
        labels_tsv_path: str = DEFAULT_TSV,
    ):
        """
        Args:
            invalid_json_path: json of ``{scene: [frame_id, ...] | "all_invalid"}``; samples
                that use a listed frame are dropped.
            instance_mask_void_id: instance id that is remapped to -1 in ``target_masks``.
            labels_subdir: per-scene directory of the class label maps, e.g. ``'labels'``.
        """
        self.dataset_root = Path(dataset_root)
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.llff_hold = int(llff_hold)
        self.test_ids = tuple(int(v) for v in test_ids)
        self.num_views = int(num_views)
        self.crop_mode = str(crop_mode)
        self.instance_mask_h5_root = (
            Path(instance_mask_h5_root) if instance_mask_h5_root else None
        )
        self.instance_mask_void_id = (
            int(instance_mask_void_id) if instance_mask_void_id is not None else None
        )
        self.labels_subdir = str(labels_subdir) if labels_subdir else None

        if self.num_views < 2:
            raise ValueError(f'num_views must be >= 2, got {self.num_views}')
        if self.crop_mode not in CROP_MODES:
            raise ValueError(f'crop_mode must be one of {CROP_MODES}, got {crop_mode!r}')

        self._labels_remap: Optional[Dict[int, int]] = None
        if self.labels_subdir is not None:
            classes = [str(c).lower() for c in labels_classes]
            self._labels_remap = build_label_remap(labels=classes, tsv_path=str(labels_tsv_path))
            self._labels_other_idx = classes.index('other') + 1

        invalid_frames, invalid_scenes = self._load_invalid_frames(invalid_json_path)

        frame_list_path = self.dataset_root / FRAME_LIST_FILE
        if not frame_list_path.is_file():
            raise FileNotFoundError(f'Frame list not found: {frame_list_path}')
        with open(frame_list_path, 'r') as f:
            frame_lists = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        missing: List[str] = []
        for scene in scene_names:
            if scene in invalid_scenes:
                continue
            if not frame_lists.get(scene):
                missing.append(scene)
                continue
            frames = sorted(str(x) for x in frame_lists[scene])
            invalid = invalid_frames.get(scene, set())
            if self.num_views == 2:
                view_sets = self._triplets(frames)
            else:
                view_sets = self._multiview(frames)
            for context_names, target_names in view_sets:
                if any(int(name) in invalid for name in context_names + target_names):
                    continue
                self.samples.append({
                    'scene': scene,
                    'context_names': context_names,
                    'target_names': target_names,
                })
        if missing:
            logger.warning(
                '%d scenes have no frames in %s (first 5: %s)',
                len(missing), frame_list_path, missing[:5],
            )
        logger.info(
            'ScanNetLSMTestDataset: %d samples from %d scenes (num_views=%d, crop=%s, hw=%s)',
            len(self.samples), len({s['scene'] for s in self.samples}),
            self.num_views, self.crop_mode, self.target_hw,
        )

    def _triplets(self, frames: List[str]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """Two context views around every held-out target frame."""
        n = len(frames)
        if n < 3:
            return []
        return [
            ((frames[max(t - 1, 0)], frames[min(t + 1, n - 1)]), (frames[t],))
            for t in range(n) if (t % self.llff_hold) in self.test_ids
        ]

    def _multiview(self, frames: List[str]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """Interleaved context / target views from the first ``2 * num_views - 1`` frames."""
        needed = 2 * self.num_views - 1
        if len(frames) < needed:
            return []
        return [(tuple(frames[0:needed:2]), tuple(frames[1:needed:2]))]

    @staticmethod
    def _load_invalid_frames(
        invalid_json_path: Optional[str],
    ) -> Tuple[Dict[str, Set[int]], Set[str]]:
        """Return the invalid frame ids per scene and the set of fully invalid scenes."""
        if invalid_json_path is None:
            return {}, set()
        with open(str(invalid_json_path), 'r') as f:
            raw = json.load(f)
        per_scene: Dict[str, Set[int]] = {}
        invalid_scenes: Set[str] = set()
        for scene, value in raw.items():
            if value == 'all_invalid':
                invalid_scenes.add(scene)
            elif value:
                per_scene[scene] = set(int(v) for v in value)
        return per_scene, invalid_scenes

    def __len__(self) -> int:
        return len(self.samples)

    def _load_view(
        self, scene_dir: Path, frame: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load a frame as (uint8 image, (4, 4) c2w, normalised (4,) fxfycxcy, class label map)."""
        image = np.asarray(
            Image.open(scene_dir / 'images' / f'{frame}.jpg').convert('RGB'), dtype=np.uint8,
        )
        camera = np.load(scene_dir / 'images' / f'{frame}.npz')
        K = np.asarray(camera['camera_intrinsics'], dtype=np.float64)
        c2w = np.asarray(camera['camera_pose'], dtype=np.float32)

        labels = None
        if self.labels_subdir is not None:
            labels = np.array(
                Image.open(scene_dir / self.labels_subdir / f'{frame}.png'),
            ).astype(np.int32, copy=False)

        if self.crop_mode == 'dust3r':
            H_t, W_t = self.target_hw
            image, K, labels = crop_resize_dust3r(image, K, self.target_hw, labels)
            fxfycxcy = np.array(
                [K[0, 0] / W_t, K[1, 1] / H_t, K[0, 2] / W_t, K[1, 2] / H_t], dtype=np.float32,
            )
        else:
            image, fxfycxcy, labels = _crop_resize_pp_square(image, K, self.target_hw, labels)

        if labels is not None:
            labels = remap_labelmap(
                labels.astype(np.int64, copy=False),
                self._labels_remap,
                other_idx=self._labels_other_idx,
            )
        return image, c2w, fxfycxcy, labels

    def _load_target_masks(self, scene: str, target_names: Sequence[str]) -> np.ndarray:
        """Load the (M, H, W) int64 instance masks of the target frames."""
        h5_path = self.instance_mask_h5_root / scene / f'{scene}_gt_instance.h5'
        if not h5_path.is_file():
            raise FileNotFoundError(f'Instance mask file not found: {h5_path}')
        masks = []
        with h5py.File(str(h5_path), 'r') as fp:
            for name in target_names:
                key = f'image_{int(name):05d}'
                if key not in fp:
                    raise KeyError(f'Mask {key} missing in {h5_path}')
                masks.append(_resize_nearest(np.asarray(fp[key]), self.target_hw))
        masks = np.stack(masks, axis=0)
        if self.instance_mask_void_id is not None:
            masks = np.where(masks == self.instance_mask_void_id, -1, masks)
        return masks

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        scene = sample['scene']
        scene_dir = self.dataset_root / scene
        context_names = sample['context_names']
        target_names = sample['target_names']
        n_ctx = len(context_names)

        views = [self._load_view(scene_dir, name) for name in context_names + target_names]
        images_u8 = np.stack([v[0] for v in views], axis=0)                # (V, H, W, 3)
        images = (images_u8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        c2w = torch.from_numpy(np.stack([v[1] for v in views], axis=0))    # (V, 4, 4)
        fxfycxcy = torch.from_numpy(np.stack([v[2] for v in views], axis=0))  # (V, 4)

        c2w = normalize_cameras_first_frame(c2w.unsqueeze(0)).squeeze(0)
        cams = pack_cameras(c2w, fxfycxcy)                                 # (V, 5, 4)

        out: Dict[str, Any] = {
            'idx': idx,
            'scene_name': scene,
            'input_images': images[:n_ctx],
            'target_images': images[n_ctx:],
            'input_cams': cams[:n_ctx],
            'target_cams': cams[n_ctx:],
            'input_frame_ids': np.array([int(n) for n in context_names], dtype=np.int64),
            'target_frame_ids': np.array([int(n) for n in target_names], dtype=np.int64),
        }
        if self.instance_mask_h5_root is not None:
            out['target_masks'] = torch.from_numpy(self._load_target_masks(scene, target_names))
        if self.labels_subdir is not None:
            labels = np.stack([v[3] for v in views], axis=0)               # (V, H, W)
            out['target_labelmap'] = torch.from_numpy(labels[n_ctx:])
            out['input_labelmap'] = torch.from_numpy(labels[:n_ctx])
        return out


class ScanNetLSMTestDataModule(DataModule):
    """Evaluation-only data module of the ScanNet test set."""

    @inject(dataset_builder='.dataset')
    def __init__(
        self,
        dataset_builder,
        batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        dataset_root: str = 'datasets/scannet_test',
    ):
        super().__init__(batch_size=batch_size, eval_batch_size=eval_batch_size)
        self.dataset_builder = dataset_builder
        self.dataset_root = str(dataset_root)
        self.num_workers = int(num_workers)
        self._scene_names = self._discover_scenes()
        logger.info(
            'ScanNetLSMTestDataModule: %d scenes under %s',
            len(self._scene_names), self.dataset_root,
        )

    def _discover_scenes(self) -> List[str]:
        """Scenes of ``selected_seqs_test.json`` that exist under the dataset root."""
        root = Path(self.dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset root does not exist: {root}')

        with open(root / FRAME_LIST_FILE, 'r') as f:
            requested = sorted(json.load(f).keys())

        scenes = [s for s in requested if (root / s).is_dir()]
        if len(scenes) < len(requested):
            logger.warning(
                '%d requested scenes are missing under %s', len(requested) - len(scenes), root,
            )
        if not scenes:
            raise RuntimeError(f'No test scenes found under {root}')
        return scenes

    def dataset(self, phase: str) -> Dataset:
        if str(phase).strip().lower() == 'train':
            raise NotImplementedError('ScanNetLSMTestDataModule has no training split.')
        return self.dataset_builder(
            scene_names=self._scene_names,
            dataset_root=self.dataset_root,
        )

    def dataloader(self, dataset: Dataset, phase: str) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
