"""Class-agnostic 2D mask Average Precision (COCO-style)."""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class MaskAPMetric(Metric):
    """AP over ``iou_thresholds`` plus the AP at each ``pivot_iou``.

    ``update`` is called once per image (one target view of one sample); the precision-recall
    curve is built from the score-sorted predictions of all images.
    """

    full_state_update: bool = False

    def __init__(
        self,
        iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05).round(4).tolist()),
        pivot_iou: Sequence[float] = (0.25, 0.5),
        score_threshold: float = 0.0,
        mask_threshold: float = 0.5,
    ):
        super().__init__()

        thr_set = sorted({float(round(t, 4)) for t in tuple(iou_thresholds) + tuple(pivot_iou)})
        self.register_buffer('thresholds', torch.tensor(thr_set, dtype=torch.float32), persistent=False)
        self._iou_thresholds: Tuple[float, ...] = tuple(round(float(t), 4) for t in iou_thresholds)
        self._pivot_iou: Tuple[float, ...] = tuple(round(float(t), 4) for t in pivot_iou)
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)

        # Both list states are concatenated along dim 0 when synchronised across processes.
        self.add_state('scores_list', default=[], dist_reduce_fx='cat')
        self.add_state('tp_list', default=[], dist_reduce_fx='cat')  # each (P_i, T) bool
        self.add_state('num_gt', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(
        self,
        pred_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> None:
        """
        Args:
            pred_masks: (Q, H, W) probabilities, binarised at ``mask_threshold``.
            pred_scores: (Q,) confidence used to rank the predictions.
            gt_masks: (M, H, W) binary masks.
        """
        device = pred_masks.device
        T = int(self.thresholds.numel())

        keep = pred_scores >= self.score_threshold
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]
        Q = int(pred_masks.shape[0])
        M = int(gt_masks.shape[0])

        self.num_gt = self.num_gt + int(M)

        if Q == 0:
            self.scores_list.append(torch.zeros(0, device=device, dtype=torch.float32))
            self.tp_list.append(torch.zeros((0, T), device=device, dtype=torch.bool))
            return

        pred_bin = (pred_masks > self.mask_threshold).float()

        if M == 0:
            order = torch.argsort(pred_scores, descending=True)
            self.scores_list.append(pred_scores[order].detach().to(torch.float32))
            self.tp_list.append(torch.zeros((Q, T), dtype=torch.bool, device=device))
            return

        pred_flat = pred_bin.reshape(Q, -1)
        gt_flat = gt_masks.float().reshape(M, -1)
        inter = pred_flat @ gt_flat.t()                           # (Q, M)
        pred_sum = pred_flat.sum(dim=1, keepdim=True)             # (Q, 1)
        gt_sum = gt_flat.sum(dim=1, keepdim=True).t()             # (1, M)
        union = (pred_sum + gt_sum - inter).clamp_min(1e-6)
        iou = inter / union

        order = torch.argsort(pred_scores, descending=True)
        scores_sorted = pred_scores[order]
        iou_sorted = iou[order]

        # Greedy matching per threshold: in score order, a prediction is a true positive
        # iff its best-IoU ground truth among those not yet taken reaches the threshold.
        thresholds = self.thresholds.to(device)
        tp_matrix = torch.zeros((T, Q), dtype=torch.bool, device=device)
        for ti in range(T):
            tau = float(thresholds[ti].item())
            gt_taken = torch.zeros(M, dtype=torch.bool, device=device)
            for qi in range(Q):
                ious_q = iou_sorted[qi].clone()
                ious_q[gt_taken] = -1.0
                best_iou, best_gt = ious_q.max(dim=0)
                if float(best_iou.item()) >= tau:
                    tp_matrix[ti, qi] = True
                    gt_taken[int(best_gt.item())] = True

        self.scores_list.append(scores_sorted.detach().to(torch.float32))
        self.tp_list.append(tp_matrix.detach().t().contiguous())  # (Q, T)

    def _zero_results(self) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.thresholds.device)
        results = {'AP': zero}
        for piv in self._pivot_iou:
            results[f'AP{int(round(piv * 100))}'] = zero.clone()
        return results

    def compute(self) -> Dict[str, torch.Tensor]:
        device = self.thresholds.device

        # After synchronisation the list states may already be single concatenated tensors.
        scores_state = self.scores_list
        tp_state = self.tp_list
        if (isinstance(scores_state, list) and len(scores_state) == 0) or int(self.num_gt.item()) == 0:
            return self._zero_results()

        all_scores = dim_zero_cat(scores_state)                   # (P,)
        all_tp = dim_zero_cat(tp_state).t().contiguous()          # (T, P)
        if all_scores.numel() == 0:
            return self._zero_results()

        order = torch.argsort(all_scores, descending=True)
        tp_sorted = all_tp[:, order].float()
        tp_cum = tp_sorted.cumsum(dim=1)
        fp_cum = (1.0 - tp_sorted).cumsum(dim=1)
        num_gt = float(int(self.num_gt.item()))
        recall = tp_cum / max(num_gt, 1.0)
        precision = tp_cum / (tp_cum + fp_cum).clamp_min(1e-12)

        # 101-point interpolation of the precision envelope, per IoU threshold.
        rec_pts = torch.linspace(0.0, 1.0, 101, device=device)
        ap_per_thr = torch.zeros(self.thresholds.numel(), device=device)
        for ti in range(self.thresholds.numel()):
            r = recall[ti]
            p = precision[ti]
            p_max = torch.flip(torch.cummax(torch.flip(p, dims=[0]), dim=0)[0], dims=[0])
            ge = r.unsqueeze(0) >= rec_pts.unsqueeze(1)           # (101, P)
            p_max_b = p_max.unsqueeze(0).expand_as(ge)
            p_at = torch.where(ge, p_max_b, torch.zeros_like(p_max_b)).max(dim=1).values
            ap_per_thr[ti] = p_at.mean()

        thr_list = [round(float(x.item()), 4) for x in self.thresholds]
        ap_indices = [thr_list.index(float(t)) for t in self._iou_thresholds]
        results: Dict[str, torch.Tensor] = {
            'AP': ap_per_thr[torch.tensor(ap_indices, device=device)].mean(),
        }
        for piv in self._pivot_iou:
            piv_r = round(float(piv), 4)
            if piv_r in thr_list:
                results[f'AP{int(round(piv_r * 100))}'] = ap_per_thr[thr_list.index(piv_r)]
        return results
