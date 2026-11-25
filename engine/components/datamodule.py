from abc import abstractmethod, ABC
from typing import Union

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

from engine import Phases
from engine._internal.seed import worker_seed_fn, get_dataloader_generator

__all__ = ['DataModule']


class DataModule(pl.LightningDataModule, ABC):
    """LightningDataModule that builds the loader of a phase from ``dataset(phase)`` and ``dataloader(...)``."""

    def __init__(self, batch_size: int, eval_batch_size: int = -1):
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size > 0 else batch_size
        self._debug = False
        self._debug_multiprocess = False

    @abstractmethod
    def dataset(self, phase: str) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def dataloader(self, dataset: Dataset, phase: str) -> DataLoader:
        raise NotImplementedError

    def enable_debug_mode(self, debug: Union[bool, int], debug_multiprocess: bool = False,
                          batch_size=-1, eval_batch_size=-1):
        """Debug runs load data in the main process unless ``debug_multiprocess``; batch sizes > 0 override."""
        self._debug = int(debug) > 0
        self._debug_multiprocess = debug_multiprocess

        if eval_batch_size < 0:
            eval_batch_size = batch_size
        if batch_size > 0:
            self.batch_size = batch_size
        if eval_batch_size > 0:
            self.eval_batch_size = eval_batch_size

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(Phases.TRAIN)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(Phases.VAL)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(Phases.TEST)

    def create_dataloader(self, phase) -> DataLoader:
        """Build a new dataset and loader of ``phase`` with seeded worker processes."""
        dataloader = self.dataloader(self.dataset(phase), phase)
        if dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = worker_seed_fn
        rank = self.trainer.global_rank if self.trainer is not None else 0
        g = get_dataloader_generator(global_rank=rank)
        if g is not None and dataloader.generator is None:
            dataloader.generator = g

        if self._debug and not self._debug_multiprocess:
            dataloader.num_workers = 0

        return dataloader
