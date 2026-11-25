import math

import numpy as np

from .common import tensor_to_numpy


def make_grid(items, num_rows, num_cols):
    assert (len(items) == num_rows * num_cols)
    rows = []
    for i in range(num_rows):
        row = np.concatenate(items[i * num_cols:(i + 1) * num_cols], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def make_image_grid(images, n_rows=None, n_cols=None):
    """Arrange images into a grid, padding the last cells with black.

    Args:
        images: list of np.ndarray or torch.Tensor with the same spatial shape.
        n_rows: number of rows (derived from ``n_cols`` or the image count if None).
        n_cols: number of columns (derived from ``n_rows`` or the image count if None).
    """
    items = [tensor_to_numpy(img) for img in images]
    n = len(items)
    assert n > 0

    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n / n_rows)
    elif n_rows is None:
        n_rows = math.ceil(n / n_cols)

    total = n_rows * n_cols
    if n < total:
        pad = np.zeros_like(items[0])
        items.extend([pad] * (total - n))

    return make_grid(items, n_rows, n_cols)


def views_to_grid(views, n_rows, max_views=4):
    """Tile the first ``max_views`` of (V, 3, H, W) images in [0, 1] into one uint8 grid."""
    views = tensor_to_numpy(views)[:max_views]
    images = [(view.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8) for view in views]
    return make_image_grid(images, n_rows=n_rows)
