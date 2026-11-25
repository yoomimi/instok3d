"""Nearest-neighbour based rescaling of Gaussian scales."""
from typing import Dict

import torch
from pointops.functions import pointops


def compute_knn_ctx(xyz: torch.Tensor, k: int = 3) -> Dict[str, torch.Tensor]:
    """RMS distance of every point to its ``k`` nearest neighbours (excluding itself).

    Args:
        xyz: (B, N, 3) point cloud.
    Returns:
        dict with ``nn_dist`` (B, N) and its per-batch median ``median_dist`` (B, 1).
    """
    with torch.no_grad():
        xyz = xyz.detach().float().contiguous()
        _, dists = pointops.knn(xyz, xyz, k=k + 1)       # (B, N, k+1), column 0 is the point itself
        d2 = dists[..., 1:k + 1].pow(2)
        nn_dist = d2.mean(dim=-1).clamp_min(1e-7).sqrt()
        median_dist = nn_dist.median(dim=-1, keepdim=True).values
    return {'nn_dist': nn_dist, 'median_dist': median_dist}


def apply_knn_rescale(
        scale: torch.Tensor,
        nn_dist: torch.Tensor,
        median_dist: torch.Tensor,
        clamp_min: float = 0.1,
        clamp_max: float = 3.0,
) -> torch.Tensor:
    """Multiply ``scale`` by ``nn_dist`` and clamp it to multiples of ``median_dist``.

    Args:
        scale: (B, N, 3) scale per Gaussian.
        nn_dist: (B, N) per-point factor.
        median_dist: (B, 1) median neighbour distance.
    Returns:
        (B, N, 3) rescaled scale.
    """
    scale = scale * nn_dist.unsqueeze(-1)
    lo = (clamp_min * median_dist).unsqueeze(-1)
    hi = (clamp_max * median_dist).unsqueeze(-1)
    return torch.clamp(scale, min=lo, max=hi)
