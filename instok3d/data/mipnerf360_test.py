"""Mip-NeRF 360 test set for zero-shot two-view reconstruction (evaluation only).

Expected layout (the public Mip-NeRF 360 release):

    <dataset_root>/<scene>/<image_subdir>/<name>.JPG
    <dataset_root>/<scene>/sparse/0/cameras.bin   COLMAP intrinsics
    <dataset_root>/<scene>/sparse/0/images.bin    COLMAP world-to-camera poses
"""

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from engine import inject
from engine.components import DataModule
from instok3d.data.scannet_lsm_test import crop_resize_dust3r
from instok3d.utils.cam import normalize_cameras_first_frame, pack_cameras

logger = logging.getLogger(__name__)

DEFAULT_LLFF_HOLD = 8
DEFAULT_TEST_IDS: Tuple[int, ...] = (1, 4)
MIPNERF360_PUBLIC_SCENES: Tuple[str, ...] = (
    'bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump',
)

# COLMAP camera model id -> number of parameters.
_COLMAP_MODEL_NUM_PARAMS = {
    0: 3,    # SIMPLE_PINHOLE: f, cx, cy
    1: 4,    # PINHOLE: fx, fy, cx, cy
    2: 4,    # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,    # RADIAL: f, cx, cy, k1, k2
    4: 8,    # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,    # OPENCV_FISHEYE
    6: 12,   # FULL_OPENCV
    7: 5,    # FOV
    8: 4,    # SIMPLE_RADIAL_FISHEYE
    9: 5,    # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}
_COLMAP_SINGLE_FOCAL = {0, 2, 3, 8, 9}


def read_colmap_cameras_bin(path: Path) -> Dict[int, Dict[str, float]]:
    """Parse COLMAP ``cameras.bin`` into ``{camera_id: {W, H, fx, fy, cx, cy}}`` (no distortion)."""
    cameras: Dict[int, Dict[str, float]] = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack('<ii', f.read(8))
            width, height = struct.unpack('<QQ', f.read(16))
            n_params = _COLMAP_MODEL_NUM_PARAMS.get(model_id)
            if n_params is None:
                raise ValueError(f'Unsupported COLMAP camera model {model_id} in {path}')
            params = struct.unpack('<' + 'd' * n_params, f.read(8 * n_params))
            if model_id in _COLMAP_SINGLE_FOCAL:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            cameras[camera_id] = {
                'W': float(width), 'H': float(height),
                'fx': float(fx), 'fy': float(fy),
                'cx': float(cx), 'cy': float(cy),
            }
    return cameras


def read_colmap_images_bin(path: Path) -> List[Dict[str, Any]]:
    """Parse COLMAP ``images.bin`` into a list of ``{name, qvec, tvec, camera_id}``."""
    records: List[Dict[str, Any]] = []
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            f.read(4)                                               # image id
            qvec = struct.unpack('<dddd', f.read(32))               # qw, qx, qy, qz
            tvec = struct.unpack('<ddd', f.read(24))
            camera_id = struct.unpack('<i', f.read(4))[0]
            name_chars = bytearray()
            while True:
                c = f.read(1)
                if c == b'\x00' or c == b'':
                    break
                name_chars += c
            num_points2d = struct.unpack('<Q', f.read(8))[0]
            f.seek(num_points2d * 24, 1)                            # skip the 2D points
            records.append({
                'name': name_chars.decode('utf-8'),
                'qvec': np.array(qvec, dtype=np.float64),
                'tvec': np.array(tvec, dtype=np.float64),
                'camera_id': int(camera_id),
            })
    return records


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> (3, 3) rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _qt_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP world-to-camera (qvec, tvec) -> (4, 4) float32 camera-to-world matrix."""
    R_c2w = _qvec2rotmat(qvec).T
    t_c2w = -R_c2w @ tvec
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w.astype(np.float32)
    c2w[:3, 3] = t_c2w.astype(np.float32)
    return c2w


class MipNeRF360TestDataset(Dataset):
    """Two context views and one target view from Mip-NeRF 360 scenes.

    The images of a scene are sorted by name; every index ``t`` with ``t % llff_hold in test_ids``
    is a target view and its neighbours ``t - 1`` and ``t + 1`` are the context views.
    COLMAP poses have an arbitrary scale per scene, so translations are rescaled such that the
    two context cameras are ``baseline_target`` apart.

    Sample (``H, W = target_hw``):
        input_images: (2, 3, H, W), target_images: (1, 3, H, W) float32 in [0, 1]
        input_cams: (2, 5, 4), target_cams: (1, 5, 4) float32, relative to the first context view
    """

    def __init__(
        self,
        scene_names: Sequence[str],
        dataset_root: str,
        target_hw: Sequence[int] = (256, 256),
        image_subdir: str = 'images_4',
        llff_hold: int = DEFAULT_LLFF_HOLD,
        test_ids: Sequence[int] = DEFAULT_TEST_IDS,
        baseline_target: float = 1.0,
    ):
        self.dataset_root = Path(dataset_root)
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.image_subdir = str(image_subdir)
        self.llff_hold = int(llff_hold)
        self.test_ids = tuple(int(v) for v in test_ids)
        self.baseline_target = float(baseline_target)

        self._scene_cam: Dict[str, Dict[str, float]] = {}
        self.samples: List[Dict[str, Any]] = []
        missing: List[str] = []
        for scene in scene_names:
            sparse_dir = self.dataset_root / scene / 'sparse' / '0'
            if not (sparse_dir / 'images.bin').is_file():
                missing.append(scene)
                continue
            cameras = read_colmap_cameras_bin(sparse_dir / 'cameras.bin')
            records = read_colmap_images_bin(sparse_dir / 'images.bin')
            records.sort(key=lambda r: r['name'])
            # Mip-NeRF 360 scenes have a single shared camera.
            self._scene_cam[scene] = cameras[records[0]['camera_id']]
            n = len(records)
            if n < 3:
                continue
            for t in range(n):
                if (t % self.llff_hold) not in self.test_ids:
                    continue
                left, right = max(t - 1, 0), min(t + 1, n - 1)
                self.samples.append({
                    'scene': scene,
                    'records': (records[left], records[right], records[t]),
                    'context_positions': (left, right),
                    'target_positions': (t,),
                })
        if missing:
            logger.warning('%d scenes have no sparse/0/images.bin: %s', len(missing), missing)
        logger.info(
            'MipNeRF360TestDataset: %d samples from %d scenes (subdir=%s, hw=%s, baseline=%s)',
            len(self.samples), len({s['scene'] for s in self.samples}),
            self.image_subdir, self.target_hw, self.baseline_target,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_view(
        self, scene: str, record: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one view as (uint8 image, (4, 4) c2w, normalised (4,) fxfycxcy)."""
        image_path = self.dataset_root / scene / self.image_subdir / record['name']
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        img_h, img_w = image.shape[:2]

        # COLMAP intrinsics refer to the full-resolution images.
        cam = self._scene_cam[scene]
        sx = img_w / cam['W']
        sy = img_h / cam['H']
        K = np.array([
            [cam['fx'] * sx, 0.0, cam['cx'] * sx],
            [0.0, cam['fy'] * sy, cam['cy'] * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        H_t, W_t = self.target_hw
        image, K, _ = crop_resize_dust3r(image, K, self.target_hw)
        fxfycxcy = np.array(
            [K[0, 0] / W_t, K[1, 1] / H_t, K[0, 2] / W_t, K[1, 2] / H_t], dtype=np.float32,
        )
        return image, _qt_to_c2w(record['qvec'], record['tvec']), fxfycxcy

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        scene = sample['scene']
        n_ctx = 2

        views = [self._load_view(scene, record) for record in sample['records']]
        images_u8 = np.stack([v[0] for v in views], axis=0)                # (V, H, W, 3)
        images = (images_u8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        c2w = torch.from_numpy(np.stack([v[1] for v in views], axis=0))    # (V, 4, 4)
        fxfycxcy = torch.from_numpy(np.stack([v[2] for v in views], axis=0))  # (V, 4)

        c2w = normalize_cameras_first_frame(c2w.unsqueeze(0)).squeeze(0)
        baseline = torch.linalg.norm(c2w[1, :3, 3] - c2w[0, :3, 3])
        if float(baseline) > 1e-8:
            c2w = c2w.clone()
            c2w[:, :3, 3] *= (self.baseline_target / baseline)
        cams = pack_cameras(c2w, fxfycxcy)                                 # (V, 5, 4)

        return {
            'idx': idx,
            'scene_name': scene,
            'input_images': images[:n_ctx],
            'target_images': images[n_ctx:],
            'input_cams': cams[:n_ctx],
            'target_cams': cams[n_ctx:],
            'input_frame_ids': np.array(sample['context_positions'], dtype=np.int64),
            'target_frame_ids': np.array(sample['target_positions'], dtype=np.int64),
        }


class MipNeRF360TestDataModule(DataModule):
    """Evaluation-only data module of the Mip-NeRF 360 test set."""

    @inject(dataset_builder='.dataset')
    def __init__(
        self,
        dataset_builder,
        batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 4,
        dataset_root: str = 'datasets/mipnerf360',
        scene_names: Optional[Sequence[str]] = None,
    ):
        super().__init__(batch_size=batch_size, eval_batch_size=eval_batch_size)
        self.dataset_builder = dataset_builder
        self.dataset_root = str(dataset_root)
        self.num_workers = int(num_workers)

        root = Path(self.dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset root does not exist: {root}')
        requested = list(scene_names) if scene_names else list(MIPNERF360_PUBLIC_SCENES)
        self._scene_names = [s for s in requested if (root / s).is_dir()]
        if len(self._scene_names) < len(requested):
            logger.warning(
                'Mip-NeRF 360 scenes missing under %s: %s',
                root, [s for s in requested if s not in self._scene_names],
            )
        if not self._scene_names:
            raise RuntimeError(f'No Mip-NeRF 360 scenes found under {root}')

    def dataset(self, phase: str) -> Dataset:
        if str(phase).strip().lower() == 'train':
            raise NotImplementedError('MipNeRF360TestDataModule has no training split.')
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
