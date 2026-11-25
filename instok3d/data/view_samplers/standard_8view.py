"""N uniformly spaced context views with one target view between each consecutive pair.

train: the pool holds every (start, stride) combination of every scene, for the strides in
       ``train_strides`` (positions index the valid frames of a scene). An epoch draws
       ``train_epoch_size`` pool entries in a seeded random order, so long scenes are
       sampled more often.
val:   one sample per scene, starting at the first valid frame with ``eval_stride``.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from instok3d.data.view_sampling_policy import resolve_eval_stride, resolve_train_strides
from instok3d.utils.view_sampling import build_uniform_frame_ids

from ._registry import registry
from .base import BaseViewSampler

logger = logging.getLogger(__name__)


@registry.register('standard_8view')
class Standard8ViewSampler(BaseViewSampler):
    def __init__(
            self,
            num_views: int = 8,
            train_strides: Sequence[int] = (2, 4),
            eval_stride: int = 2,
    ):
        self._num_views = int(num_views)
        if self._num_views <= 1:
            raise ValueError(f'num_views must be >= 2, got {self._num_views}')

        self.train_strides = resolve_train_strides(train_strides)
        self.eval_stride = resolve_eval_stride(eval_stride)

        self._phase: Optional[str] = None
        self._pool: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self._scene_valid: List[Tuple[str, np.ndarray]] = []
        self._idxs_mapping: np.ndarray = np.zeros(0, dtype=np.int64)
        self._target_epoch_size: int = 0

    @property
    def num_views(self) -> int:
        return self._num_views

    @property
    def epoch_size(self) -> int:
        if self._phase is None:
            raise RuntimeError('Standard8ViewSampler.epoch_size called before init()')
        if self._phase == 'train':
            return int(len(self._idxs_mapping))
        return int(len(self._scene_valid))

    def init(
            self,
            scene_valid_ids: List[Tuple[str, np.ndarray]],
            phase: str,
            train_epoch_size: int = -1,
    ) -> None:
        phase_norm = str(phase).strip().lower()
        if phase_norm == 'train':
            self._pool = self._build_train_pool(scene_valid_ids)
            if not self._pool:
                raise RuntimeError(
                    f'Standard8ViewSampler.init: train pool is empty '
                    f'(scenes={len(scene_valid_ids)}, num_views={self.num_views}, '
                    f'train_strides={self.train_strides})'
                )
            self._phase = phase_norm
            self._target_epoch_size = int(train_epoch_size)
            self.set_epoch(0)
        elif phase_norm == 'val':
            min_required = (self.num_views - 1) * self.eval_stride + 1
            self._scene_valid = [
                (name, valid) for (name, valid) in scene_valid_ids
                if len(valid) > min_required
            ]
            if not self._scene_valid:
                raise RuntimeError(
                    f'Standard8ViewSampler.init: no val scene has more than '
                    f'{min_required} valid frames (scenes={len(scene_valid_ids)})'
                )
            self._phase = phase_norm
        else:
            raise NotImplementedError(f'Standard8ViewSampler does not support phase={phase!r}')

    def set_epoch(self, epoch: int) -> None:
        if self._phase != 'train':
            return
        rng = np.random.default_rng(seed=int(epoch) + 777)
        n_pool = len(self._pool)
        perm = rng.permutation(n_pool)
        target = self._target_epoch_size
        if target <= 0:
            self._idxs_mapping = perm.astype(np.int64)
            return
        repeats = 1 + (target - 1) // n_pool
        tiled = np.concatenate([perm] * repeats)
        self._idxs_mapping = tiled[:target].astype(np.int64)

    def sample(self, sub_idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        if self._phase == 'train':
            return self._pool[int(self._idxs_mapping[sub_idx])]

        scene_name, valid_ids = self._scene_valid[sub_idx]
        local_ctx = build_uniform_frame_ids(0, self.eval_stride, self.num_views)
        ctx = valid_ids[np.asarray(local_ctx, dtype=np.int64)]
        tgt = _unseen_between(valid_ids, ctx)
        return scene_name, ctx, tgt

    def _build_train_pool(
            self,
            scene_valid_ids: List[Tuple[str, np.ndarray]],
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        gaps = {(self.num_views - 1) * s for s in self.train_strides}
        min_required = max(gaps) + 1

        pool: List[Tuple[str, np.ndarray, np.ndarray]] = []
        skipped = 0
        for scene_name, valid_ids in scene_valid_ids:
            n = int(len(valid_ids))
            if n < min_required:
                skipped += 1
                continue
            for i, j in itertools.combinations(range(n), 2):
                if j - i not in gaps:
                    continue
                stride = (j - i) // (self.num_views - 1)
                ctx_local = np.arange(self.num_views, dtype=np.int64) * stride + i
                ctx = valid_ids[ctx_local]
                tgt = _unseen_between(valid_ids, ctx)
                pool.append((scene_name, ctx, tgt))
        if skipped:
            logger.info(
                'Standard8ViewSampler: skipped %d train scenes with fewer than %d valid frames',
                skipped, min_required,
            )
        return pool


def _unseen_between(valid_ids: np.ndarray, context_ids: np.ndarray) -> np.ndarray:
    """One target per pair of consecutive context views: the middle non-context valid frame
    between them, or the non-context frame closest to their midpoint if there is none."""
    context_set = set(context_ids.tolist())
    unseen_ids = np.array(
        [v for v in valid_ids if v not in context_set], dtype=np.int64
    )
    target_ids: List[int] = []
    for i in range(len(context_ids) - 1):
        lo, hi = int(context_ids[i]), int(context_ids[i + 1])
        between = unseen_ids[(unseen_ids > lo) & (unseen_ids < hi)]
        if len(between) > 0:
            target_ids.append(int(between[len(between) // 2]))
        else:
            mid = (lo + hi) // 2
            target_ids.append(int(unseen_ids[np.argmin(np.abs(unseen_ids - mid))]))
    return np.array(target_ids, dtype=np.int64)
