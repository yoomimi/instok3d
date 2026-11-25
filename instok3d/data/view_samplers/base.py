"""Base class of the view samplers.

Lifecycle::

    sampler.init(scene_valid_ids, phase, train_epoch_size)
    for epoch in epochs:
        sampler.set_epoch(epoch)
        for sub_idx in range(sampler.epoch_size):
            scene, context_ids, target_ids = sampler.sample(sub_idx)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseViewSampler(ABC):
    @property
    @abstractmethod
    def num_views(self) -> int:
        """Number of context views of a sample."""

    @property
    @abstractmethod
    def epoch_size(self) -> int:
        """Number of samples per epoch; available after init()."""

    @abstractmethod
    def init(
            self,
            scene_valid_ids: List[Tuple[str, np.ndarray]],
            phase: str,
            train_epoch_size: int = -1,
    ) -> None:
        """Set up the sampler for a phase.

        Args:
            scene_valid_ids: (scene name, valid frame ids) of every scene of the phase.
            train_epoch_size: number of training samples per epoch; <= 0 uses the full pool.
        """

    def set_epoch(self, epoch: int) -> None:
        """Re-draw the training sample order for an epoch (no-op outside training)."""

    @abstractmethod
    def sample(self, sub_idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        """Returns (scene name, context ids, target ids) for sub_idx in [0, epoch_size)."""
