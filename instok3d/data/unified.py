"""Scene-level data module over one or several data sources.

Each entry of ``data_sources`` names a data handler (``name``), the view sampler it uses
(``view_sampler``, a key of ``view_samplers``) and its training epoch size, either as
``train_epoch_size`` (samples) or as ``repeat`` (multiple of the number of train scenes);
the remaining keys are passed to the handler. All sources of a phase are concatenated
into a single dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

from engine import inject
from engine.components import DataModule
from instok3d.data.scene_utils import resize_mask_batch, resize_rgb_batch
from instok3d.utils.cam import (
    normalize_cameras_farthest_unit,
    normalize_cameras_first_frame,
    pack_cameras,
    unpack_cameras,
)

from .handlers import get_handler_cls
from .handlers.base import BaseDataHandler
from .view_samplers import get_view_sampler_cls
from .view_samplers.base import BaseViewSampler

logger = logging.getLogger(__name__)

_DEFAULT_TRAIN_EPOCH_SIZE = 5000


class UnifiedSceneDataModule(DataModule):
    @inject(dataset_builder='.dataset')
    def __init__(
            self,
            dataset_builder,

            batch_size: int,
            eval_batch_size: int = -1,
            num_workers: int = 0,

            data_sources: Optional[List[Dict[str, Any]]] = None,
            view_samplers: Optional[Dict[str, Any]] = None,

            input_hw: Optional[Sequence[int]] = None,
            target_hw: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            data_sources: list of handler configs (see the module docstring).
            view_samplers: ``{registered sampler name: sampler kwargs}``.
            input_hw: resolution of the context views; None keeps the native one.
            target_hw: resolution of the target views, masks and label maps.
        """
        super().__init__(batch_size=batch_size, eval_batch_size=eval_batch_size)

        self.dataset_builder = dataset_builder
        self.num_workers = int(num_workers)
        self.input_hw = input_hw
        self.target_hw = target_hw

        if not data_sources:
            raise ValueError('data_sources must contain at least one handler config')
        if not view_samplers:
            raise ValueError('view_samplers must contain at least one entry')

        sampler_factories: Dict[str, Callable[[], BaseViewSampler]] = {
            name: get_view_sampler_cls(name, dict(cfg) if cfg else {})
            for name, cfg in view_samplers.items()
        }

        self.handlers: List[BaseDataHandler] = []
        self._handler_sampler_factories: List[Callable[[], BaseViewSampler]] = []
        self._train_epoch_sizes: List[int] = []
        for src in data_sources:
            cfg = dict(src)
            handler_name = cfg.pop('name')
            sampler_ref = cfg.pop('view_sampler')
            if sampler_ref not in sampler_factories:
                raise ValueError(
                    f'view_sampler={sampler_ref!r} not in view_samplers {list(sampler_factories)}'
                )
            train_epoch_size = cfg.pop('train_epoch_size', None)
            repeat = cfg.pop('repeat', None)
            handler = get_handler_cls(handler_name)(**cfg)
            if train_epoch_size is None:
                train_epoch_size = (
                    int(repeat) * len(handler.get_scenes('train'))
                    if repeat is not None else _DEFAULT_TRAIN_EPOCH_SIZE
                )
            self.handlers.append(handler)
            self._handler_sampler_factories.append(sampler_factories[sampler_ref])
            self._train_epoch_sizes.append(int(train_epoch_size))

    def dataset(self, phase: str) -> Dataset:
        kwargs: Dict[str, Any] = {}
        if self.input_hw is not None:
            kwargs['input_hw'] = self.input_hw
        if self.target_hw is not None:
            kwargs['target_hw'] = self.target_hw
        return self.dataset_builder(
            handlers=self.handlers,
            phase=str(phase).strip().lower(),
            view_samplers=[factory() for factory in self._handler_sampler_factories],
            train_epoch_sizes=self._train_epoch_sizes,
            **kwargs,
        )

    def dataloader(self, dataset: Dataset, phase: str) -> DataLoader:
        is_train = str(phase).strip().lower() == 'train'
        bs = self.batch_size if is_train else self.eval_batch_size
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=is_train,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def get_callbacks(self):
        return [_SetEpochCallback()]


class UnifiedSceneDataset(Dataset):
    """Concatenation of the samples of every source; each source has its own view sampler.

    Sample (``V`` context views at ``input_hw``, ``T`` target views at ``target_hw``):
        idx: int
        scene_name: str
        input_images / target_images: (V | T, 3, H, W) float32 in [0, 1]
        input_cams / target_cams: (V | T, 5, 4) float32; rows 0-3 are the camera-to-world matrix
            relative to the first context view, row 4 is (fx / W, fy / H, cx / W, cy / H)
        input_frame_ids / target_frame_ids: (V | T,) int64 frame ids within the scene
        target_masks: (T, H, W) int64 instance ids, -1 = void; with ``use_instance_mask``
        has_seg_mask: bool, False if ``target_masks`` is a placeholder; with ``use_instance_mask``
        target_labelmap: (T, H, W) int64 class ids, 0 = void; evaluation phases of sources
            with semantic labels
    """

    def __init__(
            self,
            handlers: List[BaseDataHandler],
            phase: str,
            view_samplers: List[BaseViewSampler],
            train_epoch_sizes: List[int],

            num_views: int = 8,
            input_hw: Optional[Sequence[int]] = None,
            target_hw: Optional[Sequence[int]] = None,
            normalize_cameras: bool = False,
            use_instance_mask: bool = False,
    ):
        """
        Args:
            num_views: number of context views; must match the view samplers.
            normalize_cameras: divide the camera translations by the largest distance of a
                context camera from the first one.
            use_instance_mask: also return instance masks; sources without masks then
                return placeholders filled with -1 and ``has_seg_mask=False``.
        """
        self.handlers = list(handlers)
        self.phase = str(phase).strip().lower()
        self.view_samplers = list(view_samplers)
        if len(self.view_samplers) != len(self.handlers):
            raise ValueError(
                f'view_samplers length {len(self.view_samplers)} != '
                f'handlers length {len(self.handlers)}'
            )
        if len(train_epoch_sizes) != len(self.handlers):
            raise ValueError(
                f'train_epoch_sizes length {len(train_epoch_sizes)} != '
                f'handlers length {len(self.handlers)}'
            )
        self.num_views = int(num_views)
        self.input_hw = _coerce_hw(input_hw)
        self.target_hw = _coerce_hw(target_hw)
        self.normalize_cameras = bool(normalize_cameras)
        self.use_instance_mask = bool(use_instance_mask)

        for sampler in self.view_samplers:
            if sampler.num_views != self.num_views:
                raise ValueError(
                    f'sampler.num_views={sampler.num_views} != '
                    f'dataset.num_views={self.num_views}'
                )

        self._active_handler_idxs: List[int] = []
        for h_idx, (handler, sampler, epoch_size) in enumerate(
                zip(self.handlers, self.view_samplers, train_epoch_sizes)):
            scenes = handler.get_scenes(self.phase)
            if not scenes:
                continue
            scene_valid = [(s, handler.get_valid_frame_ids(s)) for s in scenes]
            sampler.init(scene_valid, phase=self.phase, train_epoch_size=epoch_size)
            self._active_handler_idxs.append(h_idx)
            logger.info(
                'UnifiedSceneDataset phase=%s source=%s scenes=%d samples=%d',
                self.phase, handler.name, len(scenes), sampler.epoch_size,
            )

        # (handler index, index within the handler's sampler)
        self._entries: List[Tuple[int, int]] = [
            (h_idx, sub_idx)
            for h_idx in self._active_handler_idxs
            for sub_idx in range(self.view_samplers[h_idx].epoch_size)
        ]
        if not self._entries and self.phase == 'train':
            raise RuntimeError('UnifiedSceneDataset has no training samples')

    def __len__(self) -> int:
        return len(self._entries)

    def set_epoch(self, epoch: int) -> None:
        if self.phase != 'train':
            return
        for h_idx in self._active_handler_idxs:
            self.view_samplers[h_idx].set_epoch(epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s_idx, sub_idx = self._entries[idx]
        handler = self.handlers[s_idx]
        sampler = self.view_samplers[s_idx]
        scene_name, context_ids, target_ids = sampler.sample(sub_idx)
        all_ids = np.concatenate([context_ids, target_ids])

        with handler.open_scene(scene_name) as ctx:
            images_u8 = ctx.load_images(all_ids)
            cams = ctx.load_cameras(all_ids)
            target_masks = ctx.load_instance_masks(target_ids) if self.use_instance_mask else None
            # Semantic label maps only exist for evaluation scenes.
            target_labelmap = (
                ctx.load_target_labelmap(target_ids) if self.phase != 'train' else None
            )

        n = len(context_ids)
        input_imgs_u8 = images_u8[:n]
        target_imgs_u8 = images_u8[n:]
        if self.input_hw is not None and input_imgs_u8.shape[1:3] != self.input_hw:
            input_imgs_u8 = resize_rgb_batch(input_imgs_u8, self.input_hw)
        if self.target_hw is not None and target_imgs_u8.shape[1:3] != self.target_hw:
            target_imgs_u8 = resize_rgb_batch(target_imgs_u8, self.target_hw)

        has_seg_mask = target_masks is not None
        if has_seg_mask:
            if self.target_hw is not None and target_masks.shape[1:3] != self.target_hw:
                target_masks = resize_mask_batch(target_masks, self.target_hw)
        elif self.use_instance_mask:
            # Placeholders keep mixed batches collatable; the loss skips them per sample.
            tg_hw = self.target_hw if self.target_hw is not None else target_imgs_u8.shape[1:3]
            target_masks = np.full(
                (len(target_ids), int(tg_hw[0]), int(tg_hw[1])), -1, dtype=np.int64,
            )

        cams = _apply_first_frame_normalize(cams)
        if self.normalize_cameras:
            cams = _apply_s_max_scale(cams, n_input=n)

        input_images = (input_imgs_u8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        target_images = (target_imgs_u8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

        result: Dict[str, Any] = {
            'idx': idx,
            'scene_name': scene_name,
            'input_images': input_images,
            'target_images': target_images,
            'input_frame_ids': context_ids,
            'target_frame_ids': target_ids,
            'input_cams': cams[:n],
            'target_cams': cams[n:],
        }
        if target_masks is not None:
            result['target_masks'] = target_masks
            result['has_seg_mask'] = has_seg_mask

        if target_labelmap is not None:
            if self.target_hw is not None and target_labelmap.shape[1:3] != self.target_hw:
                target_labelmap = resize_mask_batch(target_labelmap, self.target_hw)
            result['target_labelmap'] = torch.from_numpy(
                target_labelmap.astype(np.int64, copy=False)
            )
        return result


def _coerce_hw(spec: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if spec is None:
        return None
    return tuple(int(x) for x in spec)


def _apply_first_frame_normalize(cams: torch.Tensor) -> torch.Tensor:
    """Express packed (V, 5, 4) cameras in the frame of the first view."""
    c2w = normalize_cameras_first_frame(cams[:, :4, :].unsqueeze(0)).squeeze(0)
    return pack_cameras(c2w, cams[:, 4, :])


def _apply_s_max_scale(cams: torch.Tensor, n_input: int) -> torch.Tensor:
    """Divide all translations by the largest camera distance among the first n_input views."""
    c2w_all, fxfycxcy_all = unpack_cameras(cams)
    _, s_max = normalize_cameras_farthest_unit(c2w_all[:n_input].unsqueeze(0))
    c2w_all = c2w_all.clone()
    c2w_all[..., :3, 3] = c2w_all[..., :3, 3] / s_max.squeeze(0)
    return pack_cameras(c2w_all, fxfycxcy_all)


class _SetEpochCallback(Callback):
    """Passes the current epoch to the training dataset so that it re-draws its samples."""

    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        ds = getattr(loader, 'dataset', None) if loader is not None else None
        if ds is not None and hasattr(ds, 'set_epoch'):
            ds.set_epoch(int(trainer.current_epoch))
