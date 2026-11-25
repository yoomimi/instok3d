from typing import Optional, Tuple

import torch
from torch.nn import functional as F


def build_hard_assignment(
    gaussian_query_probs: torch.Tensor,
    gaussian_void_prob: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Turn soft Gaussian-to-group probabilities into a one-hot assignment.

    Args:
        gaussian_query_probs: (B, N, Q)
        gaussian_void_prob: (B, N), or None to take the argmax over the groups only.
    Returns:
        hard_query_probs: (B, N, Q) one-hot per Gaussian (all zero where void wins).
        hard_void_prob: (B, N), 1.0 where void wins; None if no void probability was given.
    """
    Q = gaussian_query_probs.shape[-1]

    if gaussian_void_prob is None:
        hard_assign = gaussian_query_probs.argmax(dim=-1)
        hard_query_probs = F.one_hot(hard_assign, num_classes=Q).to(gaussian_query_probs.dtype)
        return hard_query_probs, None

    all_probs = torch.cat([
        gaussian_query_probs, gaussian_void_prob.unsqueeze(-1),
    ], dim=-1)  # (B, N, Q+1)
    hard_assign = all_probs.argmax(dim=-1)

    hard_full = F.one_hot(hard_assign, num_classes=Q + 1).to(gaussian_query_probs.dtype)
    hard_query_probs = hard_full[..., :Q]
    hard_void_prob = hard_full[..., Q]
    return hard_query_probs, hard_void_prob
