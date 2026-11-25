"""Two context views and the target view(s) between them.

``kind`` selects the protocol (positions index the valid frames of a scene):
  - 'scannet': targets are the positions with ``i % llff_hold in test_ids``; the context
    is the pair of neighbours ``(target - 1, target + 1)``. Training picks a target at
    random, evaluation picks ``sub_idx % len(targets)``.
  - 'scannetpp' (training only): draws a gap in ``[min_gap, max_gap]``, a random left view
    and a target strictly inside the pair.
  - 're10k': training draws a gap in ``[min_gap, max_gap]``, a random left view and a
    target in the closed interval. Evaluation replays the context pair and all target
    frames of an evaluation-index json (pixelSplat / MVSplat protocol).

A training epoch visits the scenes in a seeded random order, repeated up to
``train_epoch_size`` samples.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._registry import registry
from .base import BaseViewSampler

logger = logging.getLogger(__name__)

_SUPPORTED_KINDS = ('scannet', 'scannetpp', 're10k')


@registry.register('paired_2view_scannet')
@registry.register('paired_2view_scannetpp')
@registry.register('paired_2view_re10k')
class Paired2ViewSampler(BaseViewSampler):
    """The three registry names are aliases of this class; the protocol is always set by ``kind``."""

    def __init__(
            self,
            kind: str,
            llff_hold: int = 8,
            test_ids: Sequence[int] = (1, 4),
            min_gap: int = 5,
            max_gap: int = 15,
            eval_index_path: Optional[str] = None,
            max_eval_scenes: Optional[int] = None,
    ):
        """
        Args:
            eval_index_path: ('re10k') json ``{scene: {'context': [l, r], 'target': [...]}}``
                of raw frame ids used for validation / test.
            max_eval_scenes: ('re10k') evaluate an evenly strided subset of at most this
                many scenes; None uses all of them.
        """
        kind_norm = str(kind).strip().lower()
        if kind_norm not in _SUPPORTED_KINDS:
            raise ValueError(
                f'Unsupported kind={kind!r}. Expected one of {_SUPPORTED_KINDS}.'
            )
        self.kind = kind_norm
        self.llff_hold = int(llff_hold)
        if self.llff_hold <= 0:
            raise ValueError(f'llff_hold must be positive, got {self.llff_hold}')
        self.test_ids = tuple(int(v) for v in test_ids)
        self.min_gap = int(min_gap)
        self.max_gap = int(max_gap)
        if self.min_gap < 2:
            raise ValueError(
                f'min_gap must be >= 2 to leave an interior target slot, got {self.min_gap}'
            )
        if self.max_gap < self.min_gap:
            raise ValueError(f'max_gap < min_gap ({self.max_gap} < {self.min_gap})')

        self.eval_index_path = str(eval_index_path) if eval_index_path is not None else None
        self.max_eval_scenes = (
            int(max_eval_scenes)
            if max_eval_scenes is not None and int(max_eval_scenes) > 0 else None
        )

        self._phase: Optional[str] = None
        self._scene_valid: List[Tuple[str, np.ndarray]] = []
        self._idxs_mapping: np.ndarray = np.zeros(0, dtype=np.int64)
        self._target_epoch_size: int = 0
        self._eval_entries: List[Dict[str, Any]] = []  # aligned with _scene_valid

    @property
    def num_views(self) -> int:
        return 2

    @property
    def epoch_size(self) -> int:
        if self._phase is None:
            raise RuntimeError('Paired2ViewSampler.epoch_size called before init()')
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
        if self.kind == 'scannetpp' and phase_norm != 'train':
            raise NotImplementedError(
                f"Paired2ViewSampler(kind='scannetpp') supports only the train phase, got phase={phase!r}"
            )
        if self.kind == 're10k' and phase_norm != 'train':
            self._init_re10k_eval(scene_valid_ids)
            self._phase = phase_norm
            return

        # Every scene must be long enough for the narrowest context pair.
        min_len = self.min_gap + 1 if self.kind == 're10k' else 3
        self._scene_valid = [
            (name, valid) for (name, valid) in scene_valid_ids if len(valid) >= min_len
        ]
        if not self._scene_valid:
            raise RuntimeError(
                f'Paired2ViewSampler.init(kind={self.kind}, phase={phase_norm}): no scenes '
                f'with at least {min_len} valid frames (scenes={len(scene_valid_ids)})'
            )
        self._phase = phase_norm
        if phase_norm == 'train':
            self._target_epoch_size = int(train_epoch_size)
            self.set_epoch(0)

    def _init_re10k_eval(self, scene_valid_ids: List[Tuple[str, np.ndarray]]) -> None:
        if self.eval_index_path is None:
            raise RuntimeError(
                'Paired2ViewSampler(kind=re10k) requires eval_index_path for evaluation'
            )
        with open(self.eval_index_path, 'r') as fp:
            eval_idx: Dict[str, Any] = json.load(fp)
        kept_scenes: List[Tuple[str, np.ndarray]] = []
        kept_entries: List[Dict[str, Any]] = []
        for name, valid in scene_valid_ids:
            entry = eval_idx.get(name.rsplit('/', 1)[-1])
            if entry is None:
                continue
            ctx = entry.get('context')
            tgt = entry.get('target')
            if not ctx or not tgt:
                continue
            valid_set = set(int(v) for v in valid.tolist())
            if any(int(i) not in valid_set for i in list(ctx) + list(tgt)):
                continue
            kept_scenes.append((name, valid))
            kept_entries.append({'context': list(ctx), 'target': list(tgt)})
        if not kept_scenes:
            raise RuntimeError(
                f'Paired2ViewSampler.init(kind=re10k): no scenes after the evaluation-index '
                f'filter (in={len(scene_valid_ids)})'
            )
        logger.info(
            'Paired2ViewSampler(kind=re10k): %d of %d scenes are in the evaluation index',
            len(kept_scenes), len(scene_valid_ids),
        )
        if self.max_eval_scenes is not None and len(kept_scenes) > self.max_eval_scenes:
            stride = len(kept_scenes) / float(self.max_eval_scenes)
            sel = [int(i * stride) for i in range(self.max_eval_scenes)]
            kept_scenes = [kept_scenes[i] for i in sel]
            kept_entries = [kept_entries[i] for i in sel]
        self._scene_valid = kept_scenes
        self._eval_entries = kept_entries

    def set_epoch(self, epoch: int) -> None:
        if self._phase != 'train':
            return
        rng = np.random.default_rng(seed=int(epoch) + 777)
        n_scenes = len(self._scene_valid)
        perm = rng.permutation(n_scenes)
        target = self._target_epoch_size
        if target <= 0:
            self._idxs_mapping = perm.astype(np.int64)
            return
        repeats = 1 + (target - 1) // n_scenes
        tiled = np.concatenate([perm] * repeats)
        self._idxs_mapping = tiled[:target].astype(np.int64)

    def sample(self, sub_idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        is_train = self._phase == 'train'
        if is_train:
            scene_name, valid_ids = self._scene_valid[int(self._idxs_mapping[sub_idx])]
        else:
            scene_name, valid_ids = self._scene_valid[sub_idx]

        if self.kind == 're10k' and not is_train:
            # The evaluation index stores raw frame ids, not positions into valid_ids.
            entry = self._eval_entries[sub_idx]
            ctx = np.asarray(entry['context'][:2], dtype=np.int64)
            tgt = np.asarray(list(entry['target']), dtype=np.int64)
            return scene_name, ctx, tgt

        pool_size = int(valid_ids.shape[0])
        if self.kind == 'scannet':
            left, right, target = self._sample_scannet(pool_size, is_train, sub_idx)
        elif self.kind == 're10k':
            left, right = _sample_gap_pair(pool_size, self.min_gap, self.max_gap)
            target = int(np.random.randint(left, right + 1))
        else:
            left, right = _sample_gap_pair(pool_size, self.min_gap, self.max_gap)
            target = int(np.random.randint(left + 1, right))
        ctx = valid_ids[np.array([left, right], dtype=np.int64)]
        tgt = valid_ids[np.array([target], dtype=np.int64)]
        return scene_name, ctx, tgt

    def _sample_scannet(self, pool_size: int, is_train: bool, idx: int) -> Tuple[int, int, int]:
        selected = [i for i in range(pool_size) if (i % self.llff_hold) in self.test_ids]
        if not selected:
            selected = [pool_size // 2]
        if is_train:
            target = int(np.random.choice(selected))
        else:
            target = int(selected[int(idx) % len(selected)])
        left = max(target - 1, 0)
        right = min(target + 1, pool_size - 1)
        return left, right, target


def _sample_gap_pair(pool_size: int, min_gap: int, max_gap: int) -> Tuple[int, int]:
    """Random (left, right) positions whose gap is in [min_gap, max_gap], clamped to the pool."""
    eff_max = min(max_gap, pool_size - 1)
    eff_min = min(min_gap, eff_max)
    gap = int(np.random.randint(eff_min, eff_max + 1))
    left = int(np.random.randint(0, pool_size - gap))
    return left, left + gap
