from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch import nn
from tqdm import tqdm
import lightning.pytorch as pl

from engine.components import DataModule
from engine._internal import ResultCollector, parse_list_string
from engine._internal.loader import data_indices_dataloader


class Evaluator(ABC):
    """Runs ``step`` on the batches of a data module on a single device and hands the outputs to ``result``."""

    def __init__(self, progress_bar: bool = True):
        super().__init__()

        self.progress_bar: bool = progress_bar

    @abstractmethod
    def step(self, batch, batch_idx):
        """Evaluate one batch; a returned dict is collected for ``result``."""
        raise NotImplementedError

    def result(self, results):
        """Called once after the last batch with ``{key: [value of every step]}`` (None without outputs)."""

    def run(self,
            dm: DataModule,
            stage: str,
            data_indices: Union[List[int], int, str, None] = None,
            gpus: Union[int, str] = 1,
            seed: int = 123456,
            ):
        """
        Args:
            stage: phase of the data module to evaluate, e.g. ``'test'``.
            data_indices: samples to evaluate: an int, a list, ``"0,8,16"`` or ``"0-39"`` (inclusive);
                None evaluates every sample.
            gpus: number of GPUs (0 runs on the CPU) or a comma-separated list of GPU ids; the first one is used.
        """
        loader = self._create_dataloader(dm, stage, data_indices)
        device = self._move_modules(gpus)

        pl.seed_everything(seed)
        self._main_loop(loader, dm, device)

    @torch.no_grad()
    def _main_loop(self, loader, dm, device):
        if self.progress_bar:
            loader = tqdm(loader)

        collector = ResultCollector()
        for batch_idx, batch in enumerate(loader):
            batch = _handle_batch_device(batch, device)
            batch = dm.on_after_batch_transfer(batch, 0)
            result = self.step(batch, batch_idx)
            collector.add(result)

            del batch

        self.result(collector.result())

    def _create_dataloader(self, dm, phase, data_indices):
        loader = dm.create_dataloader(phase)
        if data_indices is None:
            return loader

        data_indices = _parse_data_indices(data_indices)
        _assert_all_indices_are_valid(data_indices, loader.dataset)
        return data_indices_dataloader(loader, data_indices)

    def _move_modules(self, gpus):
        """Put every module held by the evaluator on the evaluation device, in eval mode."""
        gpu_ids = parse_list_string(gpus, parser=int)
        device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'

        for module in vars(self).values():
            if isinstance(module, nn.Module):
                module.to(device)
                module.eval()

        return device


def _parse_data_indices(data_indices) -> List[int]:
    if isinstance(data_indices, int):
        return [data_indices]
    if not isinstance(data_indices, str):
        return [int(x) for x in data_indices]

    indices = []
    for token in data_indices.split(','):
        token = token.strip()
        if '-' in token:
            start, end = token.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(token))

    return indices


def _handle_batch_device(batch, device):
    if isinstance(batch, dict):
        for key, value in batch.items():
            batch[key] = _handle_batch_device(batch[key], device)

    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [_handle_batch_device(x, device) for x in batch]

    elif _is_device_attachable(batch):
        batch = batch.to(device)

    return batch


def _is_device_attachable(item):
    return hasattr(item, 'to') and callable(item.to)


def _assert_all_indices_are_valid(data_indices, dataset):
    for idx in data_indices:
        assert 0 <= idx < len(dataset), f'invalid data indices: {idx} (total {len(dataset)} items)'
