"""Base classes of the dataset-specific data handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import List, Optional

import numpy as np
import torch


class SceneContext(ABC):
    """Open resources of one scene (e.g. its h5 file), released when the context exits."""

    @abstractmethod
    def load_images(self, indices: np.ndarray) -> np.ndarray:
        """Returns (V, H, W, 3) uint8 RGB images of the given frame ids."""

    @abstractmethod
    def load_cameras(self, indices: np.ndarray) -> torch.Tensor:
        """Returns (V, 5, 4) float32 packed cameras in the dataset's world frame."""

    @abstractmethod
    def load_instance_masks(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """Returns (V, H, W) int64 instance ids (-1 = void), or None without masks."""

    def load_target_labelmap(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """Returns (V, H, W) int64 semantic class ids (0 = void), or None without labels."""
        return None


class BaseDataHandler(ABC):
    """A data source: its scene split, the usable frames of a scene and how to load them."""

    name: str

    @abstractmethod
    def get_scenes(self, phase: str) -> List[str]:
        """Scenes of a phase; empty if the source does not take part in it."""

    @abstractmethod
    def get_valid_frame_ids(self, scene_name: str) -> np.ndarray:
        """int64 ids of the usable frames of a scene."""

    @abstractmethod
    def open_scene(self, scene_name: str) -> AbstractContextManager[SceneContext]:
        """Use as ``with handler.open_scene(name) as ctx: ...``."""
