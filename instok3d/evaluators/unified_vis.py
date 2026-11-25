"""Evaluation of the instance tokenizer: novel-view synthesis and class-agnostic instance masks."""

import json
import os
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

import engine
from engine import inject
from engine.components import Evaluator
from instok3d.metrics.image import MultiViewLPIPS
from instok3d.metrics.seg_ap import MaskAPMetric
from instok3d.models.anchor_recon.assignment import build_hard_assignment
from instok3d.utils.cam import pack_cameras, unpack_cameras
from instok3d.utils.training import forward_backbone, resize_views
from instok3d.vis.mask import colorize_label_map

_MEAN_METRICS = (
    'psnr_all', 'psnr_input', 'psnr_target',
    'ssim_all', 'ssim_input', 'ssim_target',
    'pixel_acc_target', 'miou_target', 'miou_target_matched',
)


class UnifiedVisEvaluator(Evaluator):
    """Reconstruction and instance-grouping metrics of a tokenizer with instance queries.

    Reports PSNR / SSIM of the rendered context and target views, LPIPS of the target views,
    class-agnostic mask AP / AP50 / AP25 (confidence = mass of a group) and Hungarian-matched
    mask IoU / pixel accuracy on the target views. With ``pose_align_steps > 0`` the target
    cameras are first aligned to the prediction at test time.

    Outputs, under ``outputs/unified[_pa<pose_align_steps>]/`` of the run directory:
        metrics_report.json                aggregate and per-sample metrics
        merged/<index>_<scene>.png         target views; rows: rendering, predicted masks,
                                           ground-truth image, ground-truth masks
        per_scene/<scene>/t<k>_ctx<i>.png  context views of the k-th sample of the scene
        per_scene/<scene>/t<k>_v<i>_{rgb,seg,gt_rgb,gt_seg}.png  single target-view images
    """

    @inject('dm', 'solver')
    def __init__(
        self,
        dm,
        solver,
        pose_align_steps: int = 0,
        pose_align_lr: float = 5e-3,
        save_images: bool = True,
    ):
        super().__init__()

        self.dm = dm
        self.solver = solver
        self.model = solver.model
        self.backbone = solver.backbone
        self.renderer = self.model.renderer
        if not getattr(self.model, 'use_instance_queries', False):
            raise ValueError('UnifiedVisEvaluator needs a tokenizer with the grouping head; '
                             'use eval=benchmark_recon for reconstruction-only models.')

        self.dm.eval_batch_size = 1
        self.pose_align_steps = int(pose_align_steps)
        self.pose_align_lr = float(pose_align_lr)
        self.save_images = bool(save_images)

        self.target_lpips = MultiViewLPIPS()
        self.seg_ap_metric = MaskAPMetric(score_threshold=0.0, mask_threshold=0.5)

        suffix = f'_pa{self.pose_align_steps}' if self.pose_align_steps > 0 else ''
        self.report_dir = engine.to_output_dir(f'outputs/unified{suffix}')
        self.merged_dir = os.path.join(self.report_dir, 'merged')
        self.per_scene_dir = os.path.join(self.report_dir, 'per_scene')
        os.makedirs(self.report_dir, exist_ok=True)
        self._scene_sample_counts: Dict[str, int] = {}

    def step(self, batch, batch_idx):
        scene_name = batch['scene_name'][0]
        input_images = batch['input_images']    # (1, N, 3, H, W)
        target_images = batch['target_images']  # (1, M, 3, H, W)
        input_cams = batch['input_cams']        # (1, N, 5, 4)
        target_cams = batch['target_cams']      # (1, M, 5, 4)
        num_input = int(input_images.shape[1])

        gaussians, aux = self.model(input_images, forward_backbone(self.backbone, input_images, batch))

        if self.pose_align_steps > 0:
            target_cams = align_target_cams(
                self.model.decode_images, gaussians, target_cams, target_images,
                steps=self.pose_align_steps, lr=self.pose_align_lr,
            )

        all_cams = torch.cat([input_cams, target_cams], dim=1)
        rendered, _ = self.model.decode_images(gaussians, all_cams)
        resolution = tuple(rendered.shape[-2:])
        all_gt = resize_views(torch.cat([input_images, target_images], dim=1), resolution)

        hard_query_probs, hard_void_prob = build_hard_assignment(
            aux['gaussian_query_probs'], aux['gaussian_void_prob'],
        )
        seg_probs, _ = self.renderer.render_segmentation_probs(
            gaussians=gaussians,
            cams=all_cams,
            resolution=resolution,
            gaussian_query_probs=hard_query_probs,
            gaussian_void_prob=hard_void_prob,
        )
        num_queries = int(seg_probs.shape[1]) - 1
        pixel_query = seg_probs[0].argmax(dim=0)               # (N + M, H, W); num_queries = void
        pred_ids = torch.where(pixel_query == num_queries, 0, pixel_query + 1).cpu().numpy()

        gt_masks = None                                        # (M, H, W) instance ids
        if 'target_masks' in batch:
            gt_masks = _resize_masks(batch['target_masks'], resolution)[0]
            self._update_mask_ap(
                pixel_query[num_input:], num_queries,
                aux['query_mass'][0].clamp(0.0, 1.0), gt_masks,
            )
            gt_masks = gt_masks.cpu().numpy().astype(np.int64)

        self.target_lpips.update(
            rendered[:, num_input:].clamp(0, 1), all_gt[:, num_input:].clamp(0, 1),
        )

        if self.save_images:
            self._save_images(
                int(batch['idx'][0]), scene_name, rendered[0], all_gt[0], pred_ids, gt_masks, num_input,
            )

        return {
            'scene_name': scene_name,
            'num_input_views': num_input,
            'num_target_views': int(target_images.shape[1]),
            'psnr_all': compute_psnr(rendered, all_gt),
            'psnr_input': compute_psnr(rendered[:, :num_input], all_gt[:, :num_input]),
            'psnr_target': compute_psnr(rendered[:, num_input:], all_gt[:, num_input:]),
            'ssim_all': compute_ssim(rendered, all_gt),
            'ssim_input': compute_ssim(rendered[:, :num_input], all_gt[:, :num_input]),
            'ssim_target': compute_ssim(rendered[:, num_input:], all_gt[:, num_input:]),
            **match_instance_masks(pred_ids[num_input:], gt_masks),
        }

    def result(self, results):
        num_samples = len(results['scene_name'])
        agg = {key: _mean_valid(results[key]) for key in _MEAN_METRICS}
        agg['lpips_target'] = float(self.target_lpips.compute().item())
        has_masks = agg['miou_target'] is not None
        for key, value in self.seg_ap_metric.compute().items():
            agg[f'seg_{key}_target'] = float(value.item()) if has_masks else None

        report = {
            'num_scenes': num_samples,
            'aggregate': agg,
            'per_scene': [
                {key: _to_python_scalar(values[i]) for key, values in results.items()}
                for i in range(num_samples)
            ],
        }
        report_path = os.path.join(self.report_dir, 'metrics_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f'\n=== UnifiedVisEvaluator: {num_samples} samples, '
              f'pose_align_steps={self.pose_align_steps} ===')
        print('  views        PSNR     SSIM    LPIPS')
        print(f'  target    {_fmt(agg["psnr_target"], 7)}  {_fmt(agg["ssim_target"])}  '
              f'{_fmt(agg["lpips_target"])}')
        print(f'  context   {_fmt(agg["psnr_input"], 7)}  {_fmt(agg["ssim_input"])}')
        print(f'  all       {_fmt(agg["psnr_all"], 7)}  {_fmt(agg["ssim_all"])}')
        print('  instance masks (target views)')
        print(f'  AP / AP50 / AP25:      {_fmt(agg["seg_AP_target"])} / '
              f'{_fmt(agg["seg_AP50_target"])} / {_fmt(agg["seg_AP25_target"])}')
        print(f'  matched mIoU / acc:    {_fmt(agg["miou_target"])} / '
              f'{_fmt(agg["pixel_acc_target"])}')
        print(f'  report: {report_path}')

    def _update_mask_ap(
        self,
        pixel_query: torch.Tensor,
        num_queries: int,
        query_mass: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> None:
        """Add the target views of one sample to the AP metric.

        Args:
            pixel_query: (M, H, W) index of the group that wins each pixel.
            query_mass: (Q,) confidence of the groups.
            gt_masks: (M, H, W) instance ids; ids <= 0 are ignored.
        """
        query_ids = torch.arange(num_queries, device=pixel_query.device).view(-1, 1, 1)
        for view in range(int(gt_masks.shape[0])):
            ids = torch.unique(gt_masks[view])
            ids = ids[ids > 0]
            self.seg_ap_metric.update(
                pred_masks=(pixel_query[view].unsqueeze(0) == query_ids).float(),
                pred_scores=query_mass,
                gt_masks=(gt_masks[view].unsqueeze(0) == ids.view(-1, 1, 1)).float(),
            )

    def _save_images(
        self,
        sample_idx: int,
        scene_name: str,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        pred_ids: np.ndarray,
        gt_masks: Optional[np.ndarray],
        num_input: int,
    ) -> None:
        """Write the comparison grid and the single-view images of one sample.

        Args:
            rendered, gt: (N + M, 3, H, W) context views followed by target views.
            pred_ids: (N + M, H, W) predicted instance ids (0 = void).
            gt_masks: (M, H, W) ground-truth instance ids of the target views, or None.
        """
        context_imgs = _to_uint8_images(gt[:num_input])
        images = {
            'rgb': _to_uint8_images(rendered[num_input:]),
            'seg': [colorize_label_map(ids) for ids in pred_ids[num_input:]],
            'gt_rgb': _to_uint8_images(gt[num_input:]),
        }
        if gt_masks is not None:
            images['gt_seg'] = [colorize_label_map(ids) for ids in gt_masks]

        rows = [np.hstack(views) for views in images.values()]
        if gt_masks is None:
            rows.append(np.zeros_like(rows[0]))
        os.makedirs(self.merged_dir, exist_ok=True)
        _save_image(os.path.join(self.merged_dir, f'{sample_idx:04d}_{scene_name}.png'), np.vstack(rows))

        scene_dir = os.path.join(self.per_scene_dir, str(scene_name))
        os.makedirs(scene_dir, exist_ok=True)
        sample = self._scene_sample_counts.get(scene_name, 0)
        self._scene_sample_counts[scene_name] = sample + 1
        for i, img in enumerate(context_imgs):
            _save_image(os.path.join(scene_dir, f't{sample:02d}_ctx{i:02d}.png'), img)
        for name, views in images.items():
            for i, img in enumerate(views):
                _save_image(os.path.join(scene_dir, f't{sample:02d}_v{i:02d}_{name}.png'), img)


def align_target_cams(
    decode_images: Callable,
    gaussians,
    target_cams: torch.Tensor,
    target_images: torch.Tensor,
    steps: int,
    lr: float,
) -> torch.Tensor:
    """Test-time pose alignment: fit the target cameras to the images with an L1 photometric loss.

    Each camera gets a translation and a 6D rotation offset that are optimised with Adam for
    ``steps - 1`` updates while the Gaussians stay fixed.

    Args:
        decode_images: ``(gaussians, cams) -> (rendered (B, M, 3, H, W), aux)``.
        target_cams: (B, M, 5, 4)
        target_images: (B, M, 3, H, W)
    Returns:
        the aligned (B, M, 5, 4) cameras.
    """
    pose_embeds = nn.Parameter(torch.zeros(target_cams.shape[1], 9, device=target_cams.device))
    optimizer = torch.optim.Adam([pose_embeds], lr=lr)

    c2w, fxfycxcy = unpack_cameras(target_cams)
    w2c = torch.linalg.inv(c2w)

    # Evaluators run under no_grad.
    with torch.enable_grad():
        for _ in range(steps - 1):
            rendered, _ = decode_images(gaussians, _apply_pose_deltas(w2c, fxfycxcy, pose_embeds))
            target = resize_views(target_images, rendered.shape[-2:])
            loss = (rendered - target).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return _apply_pose_deltas(w2c, fxfycxcy, pose_embeds.detach())


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean per-view PSNR of (B, V, 3, H, W) images in [0, 1]."""
    mse = F.mse_loss(pred.detach().clamp(0, 1), gt.clamp(0, 1), reduction='none')
    mse = mse.flatten(2).mean(dim=-1)  # (B, V)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return float(psnr.mean().item())


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean per-view SSIM of (B, V, 3, H, W) images in [0, 1]."""
    ssim = structural_similarity_index_measure(
        pred.detach().clamp(0, 1).flatten(0, 1),
        gt.clamp(0, 1).flatten(0, 1),
        data_range=1.0,
        reduction='none',
    )
    return float(ssim.mean().item())


def match_instance_masks(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, Any]:
    """Hungarian-match predicted to ground-truth instances by IoU over the views of a sample.

    Args:
        pred: (M, H, W) predicted instance ids; 0 is void.
        gt: (M, H, W) ground-truth instance ids, or None; ids <= 0 are unlabeled and ignored.
    Returns:
        ``miou_target`` (matched IoU averaged over the ground-truth instances),
        ``miou_target_matched`` (averaged over the matched pairs), ``pixel_acc_target``
        (accuracy after relabelling the predictions with their matches) and instance counts.
        The metrics are None if there is no ground-truth instance.
    """
    pred = np.asarray(pred, dtype=np.int64)
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]
    if gt is None:
        gt_ids = np.zeros(0, dtype=np.int64)
    else:
        gt = np.asarray(gt, dtype=np.int64)
        if pred.shape != gt.shape:
            raise ValueError(f'Mask shape mismatch: pred={pred.shape}, gt={gt.shape}')
        gt_ids = np.unique(gt)
        gt_ids = gt_ids[gt_ids > 0]

    if gt_ids.size == 0:
        return {
            'pixel_acc_target': None,
            'miou_target': None,
            'miou_target_matched': None,
            'num_pred_instances_target': int(pred_ids.size),
            'num_gt_instances_target': 0,
            'num_matched_instances_target': 0,
        }

    gt_instance_masks = [gt == gid for gid in gt_ids.tolist()]
    iou = np.zeros((pred_ids.size, gt_ids.size), dtype=np.float64)
    for pi, pid in enumerate(pred_ids.tolist()):
        pred_mask = pred == pid
        for gi, gt_mask in enumerate(gt_instance_masks):
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou[pi, gi] = float(inter / union)

    if iou.size > 0:
        rows, cols = linear_sum_assignment(1.0 - iou)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
    matched_ious = iou[rows, cols]

    pred_remap = np.zeros_like(pred)
    for r, c in zip(rows.tolist(), cols.tolist()):
        pred_remap[pred == pred_ids[r]] = gt_ids[c]
    labelled = gt > 0

    return {
        'pixel_acc_target': float((pred_remap[labelled] == gt[labelled]).mean()),
        'miou_target': float(matched_ious.sum() / gt_ids.size),
        'miou_target_matched': float(matched_ious.mean()) if matched_ious.size > 0 else 0.0,
        'num_pred_instances_target': int(pred_ids.size),
        'num_gt_instances_target': int(gt_ids.size),
        'num_matched_instances_target': int(matched_ious.size),
    }


def _resize_masks(masks: torch.Tensor, hw) -> torch.Tensor:
    """Nearest-neighbour resize of (B, M, H, W) integer masks to ``hw``."""
    if tuple(masks.shape[-2:]) == tuple(hw):
        return masks
    return F.interpolate(masks.float(), size=tuple(hw), mode='nearest').long()


def _to_uint8_images(images: torch.Tensor) -> List[np.ndarray]:
    """(V, 3, H, W) float images in [0, 1] -> list of (H, W, 3) uint8 arrays."""
    images = images.detach().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return [(img * 255).astype(np.uint8) for img in images]


def _save_image(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _to_python_scalar(value):
    return value.item() if hasattr(value, 'item') else value


def _mean_valid(values) -> Optional[float]:
    """Mean of the finite entries of ``values``, skipping None; None if there is no such entry."""
    valid = [float(v) for v in map(_to_python_scalar, values) if v is not None]
    valid = [v for v in valid if np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def _fmt(value: Optional[float], width: int = 6) -> str:
    return f'{"N/A":>{width}}' if value is None else f'{float(value):{width}.4f}'


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation representation (..., 6) -> rotation matrices (..., 3, 3) (Zhou et al., 2019)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _apply_pose_deltas(
    w2c: torch.Tensor, fxfycxcy: torch.Tensor, pose_embeds: torch.Tensor,
) -> torch.Tensor:
    """Right-multiply (B, V, 4, 4) world-to-camera matrices by per-view pose offsets.

    Args:
        pose_embeds: (V, 9) translation followed by a 6D rotation offset from identity.
    Returns:
        the perturbed cameras packed as (B, V, 5, 4).
    """
    identity6 = pose_embeds.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rot = _rotation_6d_to_matrix(pose_embeds[..., 3:] + identity6)
    top = torch.cat([rot, pose_embeds[..., :3].unsqueeze(-1)], dim=-1)   # (V, 3, 4)
    bottom = pose_embeds.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(pose_embeds.shape[0], 1, 4)
    delta = torch.cat([top, bottom], dim=-2)                            # (V, 4, 4)

    w2c = w2c @ delta.unsqueeze(0)
    R_t = w2c[..., :3, :3].transpose(-1, -2)
    c2w = w2c.clone()
    c2w[..., :3, :3] = R_t
    c2w[..., :3, 3:4] = -R_t @ w2c[..., :3, 3:4]
    return pack_cameras(c2w, fxfycxcy)
