from typing import List

from torch.utils.data import DataLoader

__all__ = ['data_indices_dataloader']


def data_indices_dataloader(loader: DataLoader, data_indices: List[int]) -> DataLoader:
    """Loader over the given samples of ``loader.dataset``, in order, with the same batch size and collation."""
    batch_size = loader.batch_size if isinstance(loader.batch_size, int) and loader.batch_size > 1 else 1
    return DataLoader(loader.dataset, sampler=data_indices, batch_size=batch_size, collate_fn=loader.collate_fn)
