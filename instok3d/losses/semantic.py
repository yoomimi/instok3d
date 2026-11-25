"""Distillation loss of the semantic feature lifting."""

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class SemanticDistillationLoss(nn.Module):
    """Cosine distillation of rendered features and of group embeddings to a 2D teacher.

    The pixel term compares the rendered feature map with the teacher feature map. The group
    term compares the embedding of each group with the mean teacher feature of its pixels;
    groups that cover fewer than ``mass_min`` pixels are skipped.
    """

    def __init__(
        self,
        pixel_loss_weight: float = 1.0,
        instance_shared_weight: float = 1.0,
        mass_min: float = 1.0,
    ):
        super().__init__()
        self.pixel_loss_weight = float(pixel_loss_weight)
        self.instance_shared_weight = float(instance_shared_weight)
        self.mass_min = float(mass_min)

    def forward(
        self,
        *,
        final_sem: torch.Tensor,
        lseg_target: torch.Tensor,
        pixel_oh: Optional[torch.Tensor] = None,
        instance_full: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            final_sem: (B, V, C, H, W) rendered semantic features.
            lseg_target: (B, V, C, H, W) teacher features of the same views.
            pixel_oh: (B, V, L, H, W) one-hot group of every pixel (None without group branch).
            instance_full: (B, L, C) group embeddings (None without group branch).
        Returns:
            dict with ``total``, ``feat_loss``, ``inst_loss`` and ``inst_valid_count``, the
            number of groups that entered the group term.
        """
        lseg_target = lseg_target.detach()
        feat_loss = (1 - F.cosine_similarity(final_sem, lseg_target, dim=2)).mean()

        if pixel_oh is None or instance_full is None:
            inst_loss = torch.zeros((), device=final_sem.device, dtype=feat_loss.dtype)
            inst_valid_count = inst_loss.detach()
        else:
            weighted_sum = torch.einsum('bvqhw, bvchw -> bqc', pixel_oh, lseg_target)
            mass = pixel_oh.sum(dim=(1, 3, 4))  # (B, L) pixels per group
            valid = (mass >= self.mass_min).to(weighted_sum.dtype)
            target_q = weighted_sum / mass.clamp_min(self.mass_min).unsqueeze(-1)
            cos_per_q = F.cosine_similarity(instance_full, target_q, dim=-1)
            inst_loss = ((1 - cos_per_q) * valid).sum() / valid.sum().clamp_min(1.0)
            inst_valid_count = valid.sum().detach()

        total = self.pixel_loss_weight * feat_loss + self.instance_shared_weight * inst_loss
        return {
            'total': total,
            'feat_loss': feat_loss,
            'inst_loss': inst_loss,
            'inst_valid_count': inst_valid_count,
        }
