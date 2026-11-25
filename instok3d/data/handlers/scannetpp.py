"""ScanNet++ data handler (same h5 layout as ScanNet, explicit validation scene list, no instance masks)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from instok3d.data.scene_utils import read_scene_list

from ._registry import registry
from .scannet import ScanNetDataHandler

logger = logging.getLogger(__name__)


@registry.register('scannetpp')
class ScanNetppDataHandler(ScanNetDataHandler):
    """ScanNet++ scenes; the scenes listed in ``val_scene_list_path`` form the val split."""

    name = 'scannetpp'

    def __init__(
            self,
            dataset_root: str,
            gt_subdir: str = 'h5',
            val_scene_list_path: str = 'nvs_sem_val.txt',
            frame_load_stride: int = 1,
            phases: Optional[List[str]] = None,
    ):
        """
        Args:
            val_scene_list_path: text file of val scenes, relative to ``dataset_root``.
        """
        super().__init__(
            dataset_root=dataset_root,
            gt_subdir=gt_subdir,
            instance_mask_subdir=None,
            split_last_n_val=0,
            invalid_json_path=None,
            frame_load_stride=frame_load_stride,
            phases=phases,
        )

        self.val_scene_list_path = str(val_scene_list_path)
        self._train_scenes, self._val_scenes = self._partition_by_val_list(self._all_scenes)

    def _partition_by_val_list(self, all_scenes: List[str]) -> Tuple[List[str], List[str]]:
        list_path = Path(self.val_scene_list_path)
        if not list_path.is_absolute():
            list_path = Path(self.dataset_root) / list_path
        requested = read_scene_list(list_path)

        available = set(all_scenes)
        val = [s for s in requested if s in available]
        if len(val) < len(requested):
            logger.warning(
                'val scene list %s: %d scenes missing in %s',
                list_path, len(requested) - len(val), self.scannet_h5_dir,
            )
        if not val:
            raise RuntimeError(
                f'No val scenes from list matched available h5 files. '
                f'list={list_path}, h5_dir={self.scannet_h5_dir}'
            )
        val_set = set(val)
        train = [s for s in all_scenes if s not in val_set]
        return train, val
