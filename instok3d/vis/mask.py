import numpy as np


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Colour an (H, W) integer label map with a fixed colour per id; ids <= 0 are black."""
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(label_map)
    for uid in unique_ids.tolist():
        if int(uid) <= 0:
            color = (0, 0, 0)
        else:
            x = int(uid)
            color = ((x * 37) % 255, (x * 67) % 255, (x * 97) % 255)
        out[label_map == uid] = color
    return out


def to_heatmap_u8(prob_map: np.ndarray) -> np.ndarray:
    """Show an (H, W) map with values in [0, 1] in the red channel of a uint8 image."""
    p = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    img = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
    img[..., 0] = (p * 255.0).astype(np.uint8)
    return img
