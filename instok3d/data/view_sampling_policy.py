"""Validation of the view-sampling stride settings."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

DEFAULT_TRAIN_STRIDES = (2, 4)


def _normalize_positive_ints(values: Iterable[int], field_name: str) -> Tuple[int, ...]:
    normalized = []
    for raw in values:
        val = int(raw)
        if val <= 0:
            raise ValueError(f"{field_name} must contain only positive integers, got {val}")
        if val not in normalized:
            normalized.append(val)
    if len(normalized) == 0:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(normalized)


def resolve_train_strides(
    train_strides: Optional[Sequence[int]],
    default: Sequence[int] = DEFAULT_TRAIN_STRIDES,
) -> Tuple[int, ...]:
    """Return the de-duplicated context strides used to build training samples."""
    if train_strides is None:
        return _normalize_positive_ints(default, "train_strides")
    return _normalize_positive_ints(train_strides, "train_strides")


def resolve_eval_stride(eval_stride: Optional[int], default: int = 2) -> int:
    """Return the context stride of the deterministic validation samples."""
    stride = int(default if eval_stride is None else eval_stride)
    if stride <= 0:
        raise ValueError(f"eval_stride must be positive, got {stride}")
    return stride
