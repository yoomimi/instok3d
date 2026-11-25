"""Mapping from raw ScanNet class ids to the 8-class evaluation label set."""

from typing import Dict, Sequence

import numpy as np
import pandas as pd

DEFAULT_LABELS = (
    'wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other',
)
DEFAULT_TSV = 'assets/scannetv2-labels.combined.tsv'


def build_label_remap(
    labels: Sequence[str] = DEFAULT_LABELS,
    tsv_path: str = DEFAULT_TSV,
) -> Dict[int, int]:
    """Build ``{raw_scannet_id: class_id}`` with ``class_id`` in ``1..len(labels)``.

    NYU40 classes that are not in ``labels`` map to the ``'other'`` class. The raw id 0
    (void) is not part of the mapping.
    """
    df = pd.read_csv(tsv_path, sep='\t')
    id_to_nyu40class = pd.Series(
        df['nyu40class'].str.lower().values, index=df['id'],
    ).to_dict()
    labels = [s.lower() for s in labels]
    if 'other' not in labels:
        raise ValueError(f'`labels` must include "other" as a fallback; got {labels!r}')
    other_idx = labels.index('other') + 1
    nyu40_to_newid = {
        cls: (labels.index(cls) + 1 if cls in labels else other_idx)
        for cls in set(id_to_nyu40class.values())
    }
    return {int(raw): nyu40_to_newid[cls] for raw, cls in id_to_nyu40class.items()}


def remap_labelmap(
    labelmap: np.ndarray,
    id_to_newid: Dict[int, int],
    other_idx: int = 8,
) -> np.ndarray:
    """Remap a raw-id label map to class ids (int64, same shape).

    Raw id 0 stays 0 (void); raw ids missing from ``id_to_newid`` become ``other_idx``.
    """
    out = np.zeros_like(labelmap, dtype=np.int64)
    flat_in = labelmap.ravel()
    flat_out = out.ravel()
    for raw in np.unique(flat_in):
        raw_int = int(raw)
        if raw_int == 0:
            continue
        flat_out[flat_in == raw] = id_to_newid.get(raw_int, other_idx)
    return out
