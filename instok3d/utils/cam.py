"""Camera helpers.

Cameras are packed as (..., 5, 4) tensors: rows 0-3 hold the camera-to-world matrix and
row 4 holds (fx, fy, cx, cy) normalised by the image width / height.
"""

from typing import Tuple

import torch


def pack_cameras(c2w, fxfycxcy):
    return torch.cat([c2w, fxfycxcy.unsqueeze(-2)], dim=-2)


def unpack_cameras(cams):
    c2w = cams[..., :4, :]
    fxfycxcy = cams[..., 4, :]

    return c2w, fxfycxcy


def dense_intrinsics_to_params(fxfycxcy):
    """(B, V, 4) intrinsic parameters -> (B, V, 3, 3) intrinsic matrices."""
    intrinsics = torch.zeros((fxfycxcy.shape[0], fxfycxcy.shape[1], 3, 3), device=fxfycxcy.device)
    intrinsics[:, :, 0, 0] = fxfycxcy[:, :, 0]
    intrinsics[:, :, 1, 1] = fxfycxcy[:, :, 1]
    intrinsics[:, :, 0, 2] = fxfycxcy[:, :, 2]
    intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 3]
    intrinsics[:, :, 2, 2] = 1.0
    return intrinsics


def normalize_cameras_first_frame(c2w: torch.Tensor) -> torch.Tensor:
    """Re-express (B, V, 4, 4) camera-to-world matrices in the frame of the first view."""
    c2w_ref = c2w[:, 0:1]
    w2c_ref = torch.linalg.inv(c2w_ref)
    return w2c_ref @ c2w


def normalize_cameras_farthest_unit(
    c2w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale translations so that the camera farthest from the origin is at unit distance.

    Args:
        c2w: (..., V, 4, 4), already expressed in the frame of the first view.
    Returns:
        the scaled (..., V, 4, 4) cameras and the (...,) scale that was divided out.
    """
    t = c2w[..., :3, 3]
    s_max = t.norm(dim=-1).amax(dim=-1).clamp(min=eps)
    out = c2w.clone()
    out[..., :3, 3] = t / s_max.unsqueeze(-1).unsqueeze(-1)
    return out, s_max
