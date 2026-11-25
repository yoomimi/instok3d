"""Data module that pairs a training data module with an evaluation-only one."""

from __future__ import annotations

from typing import List

from torch.utils.data import DataLoader, Dataset

from engine import inject
from engine.components import DataModule


class ComposedTrainEvalDataModule(DataModule):
    """Trains on ``train_dm`` and evaluates on ``eval_dm``.

    train: ``train_dm``'s train loader.
    val:   [``train_dm``'s val loader (if ``val_include_train``), ``eval_dm``'s val loader].
    test:  [``eval_dm``'s test loader].

    ``val_dataset_names`` holds one metric prefix per val loader: ``train_dm_name`` and
    ``eval_display``.
    """

    @inject(train_dm='.train_dm', eval_dm='.eval_dm')
    def __init__(
        self,
        train_dm: DataModule,
        eval_dm: DataModule,
        batch_size: int = 1,
        eval_batch_size: int = -1,
        train_dm_name: str = 'unified',
        eval_display: str = 'eval/test',
        val_include_train: bool = True,
    ):
        super().__init__(batch_size=batch_size, eval_batch_size=eval_batch_size)
        self._train_dm = train_dm
        self._eval_dm = eval_dm
        self.train_dm_name = str(train_dm_name)
        self.eval_display = str(eval_display)
        self.val_include_train = bool(val_include_train)

        self.val_dataset_names: List[str] = (
            ([self.train_dm_name] if self.val_include_train else [])
            + [self.eval_display]
        )

    def enable_debug_mode(self, debug, debug_multiprocess: bool = False,
                          batch_size: int = -1, eval_batch_size: int = -1):
        super().enable_debug_mode(debug, debug_multiprocess, batch_size, eval_batch_size)
        self._train_dm.enable_debug_mode(debug, debug_multiprocess, batch_size, eval_batch_size)
        self._eval_dm.enable_debug_mode(debug, debug_multiprocess, batch_size, eval_batch_size)

    def dataset(self, phase: str) -> Dataset:
        raise NotImplementedError('Use the dataloaders of ComposedTrainEvalDataModule.')

    def dataloader(self, dataset: Dataset, phase: str) -> DataLoader:
        raise NotImplementedError('Use the dataloaders of ComposedTrainEvalDataModule.')

    def _attach_trainer(self, dm: DataModule) -> DataModule:
        """Hand the trainer to a sub module: its loaders are seeded with the rank of the process."""
        dm.trainer = self.trainer
        return dm

    def train_dataloader(self) -> DataLoader:
        return self._attach_trainer(self._train_dm).train_dataloader()

    def val_dataloader(self) -> List[DataLoader]:
        loaders: List[DataLoader] = []
        if self.val_include_train:
            loaders.append(self._attach_trainer(self._train_dm).val_dataloader())
        loaders.append(self._attach_trainer(self._eval_dm).val_dataloader())
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
        return [self._attach_trainer(self._eval_dm).test_dataloader()]

    def get_callbacks(self):
        cbs = []
        for dm in (self._train_dm, self._eval_dm):
            cbs.extend(getattr(dm, 'get_callbacks', lambda: [])())
        return cbs
