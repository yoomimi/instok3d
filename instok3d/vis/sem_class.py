"""Colours for semantic class maps."""

from typing import Sequence, Tuple

import numpy as np

# 0 void, 1 wall, 2 floor, 3 ceiling, 4 chair, 5 table, 6 sofa, 7 bed, 8 other
CLASS_HEX: Tuple[str, ...] = (
    '#000000', '#E6194B', '#3CB44B', '#FFE119', '#4363D8',
    '#F58231', '#911EB4', '#42D4F4', '#808000',
)


def _hex_palette_to_rgb(palette_hex: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(palette_hex), 3), dtype=np.uint8)
    for i, h in enumerate(palette_hex):
        s = h.lstrip('#')
        out[i] = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return out


def colorize_class_map(
    class_map: np.ndarray,
    palette_hex: Sequence[str] = CLASS_HEX,
) -> np.ndarray:
    """Colour an (H, W) class-id map (0 = void) as uint8 (H, W, 3).

    Ids beyond the palette take its last colour.
    """
    palette = _hex_palette_to_rgb(palette_hex)
    last_idx = palette.shape[0] - 1
    flat = np.clip(class_map.astype(np.int64), 0, last_idx)
    return palette[flat]
