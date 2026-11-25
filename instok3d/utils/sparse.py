import torch
from pointops.functions import pointops


def fps_with_mask(x, mask, k):
    """Farthest point sampling over the valid points of each batch element.

    Args:
        x: (B, N, 3 + C) points; the first three channels are the coordinates.
        mask: (B, N) bool, True for points that may be sampled.
        k: number of points to sample per batch element.
    Returns:
        (B, k, 3 + C) sampled points.
    """
    b = x.size(0)
    offset = mask.view(mask.size(0), -1).sum(dim=-1).contiguous()
    new_offset = torch.full((b,), k, dtype=torch.long, device=x.device)
    offset = torch.cumsum(offset, dim=0).int()
    new_offset = torch.cumsum(new_offset, dim=0).int()

    x = x[mask].contiguous().float()
    coord = x[..., :3].contiguous()
    idx = pointops.furthestsampling(coord, offset, new_offset).long()
    return x[idx].view(b, k, x.size(-1))
