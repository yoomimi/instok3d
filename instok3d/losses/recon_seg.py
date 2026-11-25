"""Grouping loss: Dice + BCE between rendered group masks and Hungarian-matched instance masks."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from instok3d.models.anchor_recon.gaussians import Gaussians

VOID_ID = -1  # id of the pixels that belong to no instance


def normalize_mask_tensor(x: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    """Drop a singleton channel axis (trailing, or at dim 2 of a 5D tensor) from a mask tensor."""
    if x.ndim == expected_dim + 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    elif expected_dim == 4 and x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.ndim != expected_dim:
        raise ValueError(
            f'{name} must be {expected_dim}D after normalization, got shape={tuple(x.shape)}'
        )
    return x


def _hungarian_numpy(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the linear-sum assignment for a (Q, M) cost matrix; returns (row, col) indices."""
    if cost.ndim != 2:
        raise ValueError(f'cost must be 2D, got shape={cost.shape}')

    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty

    row_idx, col_idx = linear_sum_assignment(cost)
    return row_idx.astype(np.int64, copy=False), col_idx.astype(np.int64, copy=False)


def _build_gt_instance_masks(gt_ids: torch.Tensor) -> torch.Tensor:
    """(V, H, W) instance ids -> (M, V, H, W) binary masks, one per non-void id."""
    all_ids = torch.unique(gt_ids)
    fg_ids = all_ids[all_ids != VOID_ID]
    if fg_ids.numel() == 0:
        return gt_ids.new_zeros((0,) + tuple(gt_ids.shape), dtype=torch.float32)
    return (gt_ids.unsqueeze(0) == fg_ids.view(-1, 1, 1, 1)).float()


def _autocast_disabled(device: torch.device):
    if device.type in ('cuda', 'cpu'):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _soft_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice coefficient of paired masks: (N, ...), (N, ...) -> (N,)."""
    pred_f = pred.reshape(pred.shape[0], -1)
    gt_f = gt.reshape(gt.shape[0], -1)
    inter = (pred_f * gt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + gt_f.sum(dim=1)
    return (2.0 * inter + eps) / (denom + eps)


def _pairwise_dice_cost(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice for every pair: (Q, V, H, W), (M, V, H, W) -> (Q, M)."""
    pred_f = pred.reshape(pred.shape[0], -1)
    gt_f = gt.reshape(gt.shape[0], -1)
    inter = torch.matmul(pred_f, gt_f.t())
    pred_sum = pred_f.sum(dim=1, keepdim=True)
    gt_sum = gt_f.sum(dim=1, keepdim=True).t()
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    return 1.0 - dice


def _pairwise_bce_cost(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean per-pixel BCE for every pair: (Q, V, H, W), (M, V, H, W) -> (Q, M)."""
    pred_f = pred.clamp(1e-6, 1.0 - 1e-6).float().reshape(pred.shape[0], -1)
    gt_f = gt.float().reshape(gt.shape[0], -1)
    n = float(pred_f.shape[1])

    with _autocast_disabled(pred_f.device):
        log_p = torch.log(pred_f)
        log_1mp = torch.log1p(-pred_f)
        base = log_1mp.sum(dim=1, keepdim=True)
        pos = torch.matmul(log_p - log_1mp, gt_f.t())
        return -(base + pos) / n


def _binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean BCE on probabilities, computed in float32 outside autocast."""
    pred_f = pred.float()
    target_f = target.float()
    with _autocast_disabled(pred_f.device):
        return F.binary_cross_entropy(pred_f, target_f)


class ReconSegmentationLoss(nn.Module):
    """Hungarian-matched mask loss on rendered group-probability maps.

    For every sample the Q group channels are rendered into the target views, matched to
    the ground-truth instance masks with the cost
    ``seg_loss_dice_weight * (1 - Dice) + seg_loss_bce_weight * BCE``,
    and the same two terms are applied to the matched pairs.

    Args:
        valid_mass_thr, valid_area_thr: a group counts as active (``num_valid``) when its
            anchor mass and its mean rendered probability exceed these thresholds.
    """

    def __init__(
        self,
        seg_loss_dice_weight: float = 1.0,
        seg_loss_bce_weight: float = 1.0,
        valid_mass_thr: float = 0.01,
        valid_area_thr: float = 0.001,
    ):
        super().__init__()
        self.seg_loss_dice_weight = float(seg_loss_dice_weight)
        self.seg_loss_bce_weight = float(seg_loss_bce_weight)
        self.valid_mass_thr = float(valid_mass_thr)
        self.valid_area_thr = float(valid_area_thr)

    @staticmethod
    def _resize_mask_targets(masks: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Nearest-neighbour resize of (B, V, H, W) instance ids."""
        if masks.shape[-2:] == hw:
            return masks
        return F.interpolate(
            masks.float().flatten(0, 1).unsqueeze(1),
            size=hw,
            mode='nearest',
        ).squeeze(1).view(masks.shape[0], masks.shape[1], hw[0], hw[1]).long()

    @staticmethod
    def _slice_gaussians(gaussians: Gaussians, b_idx: int) -> Gaussians:
        """Keep sample ``b_idx`` of a batch of Gaussians (batch size 1)."""
        g = Gaussians()
        for key in ('xyz', 'scale', 'rotation', 'opacity', 'feature'):
            value = getattr(gaussians, key)
            if value is None:
                continue
            setattr(g, key, value[b_idx:b_idx + 1])
        return g

    def _render_query_masks(
        self,
        renderer: Any,
        gaussians: Gaussians,
        aux: Dict[str, torch.Tensor],
        b_idx: int,
        cams_b: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render the group (Q, V, H, W) and void (V, H, W) probability maps of one sample."""
        seg_probs_b, _ = renderer.render_segmentation_probs(
            gaussians=self._slice_gaussians(gaussians, b_idx),
            cams=cams_b,
            resolution=hw,
            gaussian_query_probs=aux['gaussian_query_probs'][b_idx:b_idx + 1],
            gaussian_void_prob=aux['gaussian_void_prob'][b_idx:b_idx + 1],
        )
        return seg_probs_b[0, :-1], seg_probs_b[0, -1]

    def _match(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        """Hungarian matching of (Q, V, H, W) predictions to (M, V, H, W) binary masks.

        Returns the matched (prediction, ground-truth) index tensors; empty if there is
        nothing to match.
        """
        device = pred_masks.device
        if gt_masks.shape[0] == 0 or pred_masks.shape[0] == 0:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty

        pred_d = pred_masks.detach()
        gt_d = gt_masks.detach().to(dtype=pred_masks.dtype)
        cost = (
            self.seg_loss_dice_weight * _pairwise_dice_cost(pred_d, gt_d)
            + self.seg_loss_bce_weight * _pairwise_bce_cost(pred_d, gt_d)
        )

        row_np, col_np = _hungarian_numpy(cost.cpu().numpy())
        row = torch.from_numpy(row_np).to(device=device, dtype=torch.long)
        col = torch.from_numpy(col_np).to(device=device, dtype=torch.long)
        return row, col

    def forward(
        self,
        gaussians: Gaussians,
        aux: Dict[str, torch.Tensor],
        target_cams: torch.Tensor,
        target_images: torch.Tensor,
        target_masks: torch.Tensor,
        renderer: Any,
        for_val: bool = False,
        has_seg_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            aux: model outputs with ``gaussian_query_probs`` (B, N, Q), ``gaussian_void_prob``
                (B, N) and ``query_mass`` (B, Q).
            target_cams: (B, V, 5, 4)
            target_images: (B, V, 3, H, W); only its resolution is used.
            target_masks: (B, V, H, W) instance ids, ``VOID_ID`` where there is no instance.
            for_val: also return the rendered masks of every sample in ``vis_list``.
            has_seg_mask: (B,) bool; samples marked False have no instance masks and are skipped.
        Returns:
            dict with ``total`` (mean over the samples that have masks), the monitoring values
            ``matched_dice``, ``num_gt``, ``num_matched``, ``num_valid`` and ``vis_list``
            (one entry per sample, None for skipped samples or when ``for_val`` is False).
        """
        zero = target_images.new_tensor(0.0)

        target_masks = normalize_mask_tensor(target_masks, expected_dim=4, name='target_masks').long()
        batch_size = int(target_masks.shape[0])
        hw = (target_images.shape[-2], target_images.shape[-1])
        target_masks = self._resize_mask_targets(target_masks, hw)

        if has_seg_mask is None:
            has_seg_mask = target_masks.new_ones(batch_size, dtype=torch.bool)
        else:
            has_seg_mask = has_seg_mask.to(device=target_masks.device).bool()
        num_seg_samples = float(max(int(has_seg_mask.sum().item()), 1))

        loss_mask = zero.clone()
        matched_dice = zero.clone()
        num_gt_total = 0
        num_matched_total = 0
        num_valid_total = 0
        vis_list: list = [None] * batch_size

        for bi in range(batch_size):
            if not bool(has_seg_mask[bi]):
                continue
            mask_b = target_masks[bi]  # (V, H, W)
            gt_masks = _build_gt_instance_masks(mask_b)
            num_gt_total += int(gt_masks.shape[0])

            pred_masks, pred_void = self._render_query_masks(
                renderer, gaussians, aux, bi, target_cams[bi:bi + 1], hw,
            )

            row, col = self._match(pred_masks, gt_masks)
            if row.numel() > 0:
                pred_m = pred_masks[row]
                gt_m = gt_masks.to(dtype=pred_masks.dtype)[col]
                dice_loss = (1.0 - _soft_dice(pred_m, gt_m)).mean()
                bce_loss = _binary_cross_entropy(pred_m.clamp(1e-6, 1.0 - 1e-6), gt_m)
                sample_loss = (
                    self.seg_loss_dice_weight * dice_loss
                    + self.seg_loss_bce_weight * bce_loss
                )
                loss_mask = loss_mask + sample_loss / num_seg_samples
                matched_dice = matched_dice + _soft_dice(pred_m.detach(), gt_m).mean() / num_seg_samples
                num_matched_total += int(row.numel())

            area_b = pred_masks.mean(dim=(1, 2, 3))
            mass_b = aux['query_mass'][bi]
            valid_b = (mass_b > self.valid_mass_thr) & (area_b > self.valid_area_thr)
            num_valid_total += int(valid_b.sum().item())

            if for_val:
                vis_list[bi] = {
                    'pred_masks': pred_masks.detach(),
                    'pred_void': pred_void.detach(),
                    'gt_ids': mask_b.detach(),
                    'gt_inst_masks': gt_masks.detach(),
                    'mass_sel': mass_b.detach(),
                    'area_sel': area_b.detach(),
                }

        return {
            'total': loss_mask,
            'matched_dice': matched_dice,
            'num_gt': float(num_gt_total) / num_seg_samples,
            'num_matched': float(num_matched_total) / num_seg_samples,
            'num_valid': float(num_valid_total) / num_seg_samples,
            'vis_list': vis_list,
        }
