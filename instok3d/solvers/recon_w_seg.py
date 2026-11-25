"""Solver for joint 3D Gaussian reconstruction and instance grouping."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from engine import inject
from engine.components import DataModule, Solver
from instok3d.losses.image import ReconstructionImageLoss
from instok3d.losses.recon_seg import ReconSegmentationLoss
from instok3d.metrics.image import MultiViewLPIPS, MultiViewPSNR, MultiViewSSIM
from instok3d.metrics.seg_ap import MaskAPMetric
from instok3d.models.anchor_recon.assignment import build_hard_assignment
from instok3d.utils.training import forward_backbone, resize_views, warmup_cosine_factor
from instok3d.vis.image import make_image_grid, views_to_grid
from instok3d.vis.mask import colorize_label_map, to_heatmap_u8

logger = logging.getLogger(__name__)

_TRACK_METRICS = ('psnr', 'ssim', 'psnr+ssim', 'ap', 'ap25', 'ap50', 'ap50+psnr')


def _label_map(group_probs: torch.Tensor, void_probs: torch.Tensor) -> torch.Tensor:
    """Per-pixel argmax of (Q, V, H, W) group and (V, H, W) void maps; 0 is void, q + 1 group q."""
    num_groups = group_probs.shape[0]
    labels = torch.cat([group_probs, void_probs.unsqueeze(0)], dim=0).argmax(dim=0)
    return torch.where(labels == num_groups, torch.zeros_like(labels), labels + 1)


class ReconSegSolver(Solver):
    """Trains the tokenizer with the rendering loss and the Hungarian-matched grouping loss.

    Validation reports PSNR / SSIM / LPIPS on the target views of all validation loaders,
    the grouping loss on the batches that carry instance masks, and class-agnostic mask AP
    (groups ranked by their anchor mass) on the benchmark loader.

    Args:
        warmup_epochs: length of the linear learning-rate warm-up.
        min_lr: learning rate at the end of the cosine decay.
        coeff_perceptual: weight of the LPIPS loss.
        use_perceptual_compile: compile the LPIPS network with ``torch.compile``.
        seg_loss_weight: weight of the grouping loss.
        seg_loss_warmup_epochs: epochs over which ``seg_loss_weight`` is ramped up linearly from 0.
        seg_loss_dice_weight, seg_loss_bce_weight: weights of the Dice and BCE terms, in the
            matching cost and in the loss.
        valid_mass_thr, valid_area_thr: thresholds of the logged number of active groups.
        ap_score_threshold, ap_mask_threshold: see ``MaskAPMetric``.
        val_vis_interval: log the visualisations of every n-th validation sample.
        val_max_vis_queries: number of groups shown in 'vis/query_on_mask'.
        track_metric: checkpoint selection score, one of 'psnr', 'ssim', 'psnr+ssim', 'ap',
            'ap25', 'ap50', 'ap50+psnr'.
    """

    @inject('dm', 'model', 'backbone')
    def __init__(
        self,
        dm: DataModule,
        model: nn.Module,
        backbone: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: float = 2.5,
        min_lr: float = 1e-6,
        coeff_perceptual: float = 0.05,
        use_perceptual_compile: bool = True,
        seg_loss_weight: float = 0.1,
        seg_loss_warmup_epochs: float = 10.0,
        seg_loss_dice_weight: float = 1.0,
        seg_loss_bce_weight: float = 1.0,
        valid_mass_thr: float = 0.01,
        valid_area_thr: float = 0.001,
        ap_score_threshold: float = 0.0,
        ap_mask_threshold: float = 0.5,
        val_vis_interval: int = 5,
        val_max_vis_queries: int = 4,
        track_metric: str = 'psnr',
    ):
        super().__init__(track_max_score=True)
        if track_metric not in _TRACK_METRICS:
            raise ValueError(f'track_metric must be one of {_TRACK_METRICS}; got {track_metric!r}')
        self.track_metric = track_metric

        self.model = model
        self.dm = dm
        for callback in getattr(dm, 'get_callbacks', lambda: [])():
            self.add_callback(callback)
        self.backbone = backbone

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.use_perceptual_compile = use_perceptual_compile
        self.seg_loss_weight = float(seg_loss_weight)
        self.seg_loss_warmup_epochs = float(seg_loss_warmup_epochs)
        self.val_vis_interval = val_vis_interval
        self.val_max_vis_queries = val_max_vis_queries

        self.seg_loss_fn = ReconSegmentationLoss(
            seg_loss_dice_weight=seg_loss_dice_weight,
            seg_loss_bce_weight=seg_loss_bce_weight,
            valid_mass_thr=valid_mass_thr,
            valid_area_thr=valid_area_thr,
        )
        self.seg_ap_metric = MaskAPMetric(
            score_threshold=float(ap_score_threshold),
            mask_threshold=float(ap_mask_threshold),
        )
        self.recon_loss_fn = ReconstructionImageLoss(coeff_perceptual=coeff_perceptual)

        self.psnr_metric = MultiViewPSNR()
        self.ssim_metric = MultiViewSSIM()
        self.lpips_metric = MultiViewLPIPS()

        self.create_image_log_buffer('vis/rendered', log_once=False)
        self.create_image_log_buffer('vis/gt', log_once=True)
        self.create_image_log_buffer('vis/seg_pred', log_once=False)
        self.create_image_log_buffer('vis/seg_gt', log_once=True)
        self.create_image_log_buffer('vis/seg_pred_one_hot', log_once=False)
        self.create_image_log_buffer('vis/query_on_mask', log_once=False)

        self.total_steps: int = 1
        self.warmup_steps: int = 0
        self.seg_loss_warmup_steps: int = 0
        self._val_seg_stats = []

    def setup(self, stage: str = 'fit') -> None:
        self.total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        steps_per_epoch = self.total_steps / max(self.trainer.max_epochs, 1)
        self.warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        self.seg_loss_warmup_steps = int(self.seg_loss_warmup_epochs * steps_per_epoch)
        self.recon_loss_fn.setup(
            trainer=self.trainer,
            use_perceptual_compile=self.use_perceptual_compile,
            is_debug=self.is_debug,
        )
        logger.info('ReconSegSolver setup: total_steps=%d', self.total_steps)

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(
                step, self.warmup_steps, self.total_steps, self.lr, self.min_lr,
            ),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def _forward_backbone(self, input_images: torch.Tensor, batch: Dict[str, Any]):
        return forward_backbone(self.backbone, input_images, batch)

    def _run_segmentation_loss(
        self,
        batch: Dict[str, Any],
        gaussians,
        aux: Dict[str, Any],
        target_cams: torch.Tensor,
        target_images: torch.Tensor,
        for_val: bool,
    ):
        """Grouping loss of a batch, or None if the batch carries no instance masks."""
        if 'target_masks' not in batch:
            return None
        return self.seg_loss_fn(
            gaussians=gaussians,
            aux=aux,
            target_cams=target_cams,
            target_images=target_images,
            target_masks=batch['target_masks'].to(device=target_images.device),
            renderer=self.model.renderer,
            for_val=for_val,
            has_seg_mask=batch.get('has_seg_mask', None),
        )

    def _seg_weight_for_step(self) -> float:
        if self.seg_loss_weight <= 0.0:
            return 0.0
        if self.seg_loss_warmup_steps <= 0:
            return self.seg_loss_weight
        warmup_progress = float(self.global_step) / float(max(self.seg_loss_warmup_steps, 1))
        warmup_progress = min(max(warmup_progress, 0.0), 1.0)
        return self.seg_loss_weight * warmup_progress

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        input_images = batch['input_images']    # (B, V_in, 3, H, W)
        target_images = batch['target_images']  # (B, V_tgt, 3, H, W)
        input_cams = batch['input_cams']        # (B, V_in, 5, 4)
        target_cams = batch['target_cams']      # (B, V_tgt, 5, 4)

        bb_out = self._forward_backbone(input_images, batch)
        gaussians, aux = self.model(input_images, bb_out)

        rendered, _ = self.model.decode_images(gaussians, torch.cat([input_cams, target_cams], dim=1))
        targets = torch.cat([
            resize_views(input_images, rendered.shape[-2:]),
            resize_views(target_images, rendered.shape[-2:]),
        ], dim=1)
        recon_out = self.recon_loss_fn(rendered, targets)

        total_loss = recon_out.total
        seg_out = self._run_segmentation_loss(
            batch=batch,
            gaussians=gaussians,
            aux=aux,
            target_cams=target_cams,
            target_images=target_images,
            for_val=False,
        )
        if seg_out is not None:
            seg_weight = self._seg_weight_for_step()
            seg_total_weighted = seg_out['total'] * seg_weight
            total_loss = total_loss + seg_total_weighted

            self.log('train/seg_total', seg_out['total'], rank_zero_only=True)
            self.log('train/seg_total_weighted', seg_total_weighted, rank_zero_only=True)
            self.log('train/seg_weight', float(seg_weight), rank_zero_only=True)
            self.log('train/seg_matched_dice', seg_out['matched_dice'], rank_zero_only=True)
            self.log('train/seg_num_gt', seg_out['num_gt'], rank_zero_only=True)
            self.log('train/seg_num_matched', seg_out['num_matched'], rank_zero_only=True)
            self.log('train/seg_num_valid', seg_out['num_valid'], rank_zero_only=True)

        # Distributed training needs a gradient for every grouping-head parameter on every
        # step, also when no sample of the batch has instance masks.
        total_loss = total_loss + self.model.seg_head_zero_residual()

        self.log('train/loss', total_loss, rank_zero_only=True)
        self.log('train/recon_loss', recon_out.total, rank_zero_only=True)
        self.log('train/mse', recon_out.mse, rank_zero_only=True)
        self.log('train/perceptual', recon_out.perceptual, rank_zero_only=True)
        self.log('train/mean_opacity', gaussians.opacity.mean().item(), rank_zero_only=True)
        self.log('train/mean_scale', gaussians.scale.mean().item(), rank_zero_only=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], rank_zero_only=True)

        return total_loss

    def _is_benchmark_loader(self, dataloader_idx: int) -> bool:
        """Whether a validation loader is the benchmark split (its name starts with 'eval/')."""
        names = getattr(self.dm, 'val_dataset_names', None)
        if not names or not (0 <= dataloader_idx < len(names)):
            return False
        return str(names[dataloader_idx]).startswith('eval/')

    def on_validation_epoch_start(self) -> None:
        self._val_seg_stats = []
        self.seg_ap_metric.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        input_images = batch['input_images']
        target_images = batch['target_images']
        target_cams = batch['target_cams']

        bb_out = self._forward_backbone(input_images, batch)
        gaussians, aux = self.model(input_images, bb_out)

        rendered_target, _ = self.model.decode_images(gaussians, target_cams)
        gt_target = resize_views(target_images, rendered_target.shape[-2:])

        self.psnr_metric.update(rendered_target, gt_target)
        self.ssim_metric.update(rendered_target, gt_target)
        self.lpips_metric.update(rendered_target, gt_target)

        idx = batch['idx']
        n_rows = 2 if rendered_target.shape[1] >= 4 else 1
        for b in range(rendered_target.shape[0]):
            if idx[b].item() % self.val_vis_interval == 0:
                self.add_log_buffer_items('vis/rendered', views_to_grid(rendered_target[b], n_rows))
                self.add_log_buffer_items('vis/gt', views_to_grid(gt_target[b], n_rows))

        seg_out = self._run_segmentation_loss(
            batch=batch,
            gaussians=gaussians,
            aux=aux,
            target_cams=target_cams,
            target_images=target_images,
            for_val=True,
        )
        if seg_out is None:
            return

        self._val_seg_stats.append({
            'total': float(seg_out['total'].detach().cpu().item()),
            'dice': float(seg_out['matched_dice'].detach().cpu().item()),
            'num_gt': float(seg_out['num_gt']),
            'num_matched': float(seg_out['num_matched']),
            'num_valid': float(seg_out['num_valid']),
        })

        update_ap = self._is_benchmark_loader(dataloader_idx)
        for b, entry in enumerate(seg_out['vis_list']):
            if entry is None:
                continue
            if update_ap:
                self._update_ap(entry)
            if idx[b].item() % self.val_vis_interval == 0:
                self._log_seg_visuals(entry, gaussians, aux, target_cams, sample_idx=b)

    def _update_ap(self, entry: Dict[str, torch.Tensor]) -> None:
        """Add the target views of one sample to the AP metric, one image per view."""
        pred_masks = entry['pred_masks']        # (Q, V, H, W)
        gt_masks = entry['gt_inst_masks']       # (M, V, H, W)
        num_groups, num_views, H, W = pred_masks.shape

        # The rendered probabilities are spread over Q + 1 channels, so each group mask is
        # the set of pixels where it wins the argmax rather than a thresholded probability.
        labels = _label_map(pred_masks, entry['pred_void'])
        group_ids = torch.arange(1, num_groups + 1, device=labels.device).view(-1, 1, 1, 1)
        hard_masks = (labels.unsqueeze(0) == group_ids).to(pred_masks.dtype)
        scores = entry['mass_sel'].clamp(0.0, 1.0)

        for vi in range(num_views):
            self.seg_ap_metric.update(
                pred_masks=hard_masks[:, vi],
                pred_scores=scores,
                gt_masks=gt_masks[:, vi] if gt_masks.numel() > 0 else gt_masks.new_zeros((0, H, W)),
            )

    def _log_seg_visuals(
        self,
        entry: Dict[str, torch.Tensor],
        gaussians,
        aux: Dict[str, Any],
        target_cams: torch.Tensor,
        sample_idx: int,
    ) -> None:
        pred_masks = entry['pred_masks']        # (Q, V, H, W)
        num_views, H, W = pred_masks.shape[1:]
        n_vis = min(num_views, 4)

        def log_label_maps(name: str, labels: torch.Tensor) -> None:
            labels = labels.cpu().numpy().astype(np.int64)
            images = [colorize_label_map(labels[vi]) for vi in range(n_vis)]
            self.add_log_buffer_items(name, make_image_grid(images, n_rows=2 if n_vis >= 4 else 1))

        log_label_maps('vis/seg_pred', _label_map(pred_masks, entry['pred_void']))
        log_label_maps('vis/seg_gt', entry['gt_ids'])

        # Groups with the largest mass x area, each shown in the view where it is largest.
        vis_score = (entry['area_sel'] * entry['mass_sel']).float()
        k_vis = min(int(self.val_max_vis_queries), int(vis_score.numel()))
        if k_vis > 0:
            heatmaps = []
            for query_mask in pred_masks[torch.topk(vis_score, k=k_vis).indices]:
                view_idx = int(torch.argmax(query_mask.mean(dim=(1, 2))).item())
                heatmaps.append(to_heatmap_u8(query_mask[view_idx].cpu().numpy()))
            self.add_log_buffer_items('vis/query_on_mask', make_image_grid(heatmaps, n_rows=1))

        # Segmentation rendered from the hard (one-hot) assignment that is used at inference.
        sample = slice(sample_idx, sample_idx + 1)
        hard_query_probs, hard_void_prob = build_hard_assignment(
            aux['gaussian_query_probs'][sample], aux['gaussian_void_prob'][sample],
        )
        seg_probs_hard, _ = self.model.renderer.render_segmentation_probs(
            gaussians=self.seg_loss_fn._slice_gaussians(gaussians, sample_idx),
            cams=target_cams[sample],
            resolution=(H, W),
            gaussian_query_probs=hard_query_probs,
            gaussian_void_prob=hard_void_prob,
        )
        log_label_maps('vis/seg_pred_one_hot', _label_map(seg_probs_hard[0, :-1], seg_probs_hard[0, -1]))

    def _compute_track_score(self, psnr: float, ssim: float, ap_results: Dict[str, float]) -> float:
        if self.track_metric == 'psnr':
            return psnr
        if self.track_metric == 'ssim':
            return ssim
        if self.track_metric == 'psnr+ssim':
            return psnr + 10.0 * ssim
        if self.track_metric == 'ap':
            return float(ap_results.get('AP', 0.0))
        if self.track_metric == 'ap25':
            return float(ap_results.get('AP25', 0.0))
        if self.track_metric == 'ap50':
            return float(ap_results.get('AP50', 0.0))
        # 'ap50+psnr': AP50 in percent plus PSNR in dB.
        return 100.0 * float(ap_results.get('AP50', 0.0)) + psnr

    def on_validation_epoch_end(self) -> None:
        psnr = self.psnr_metric.compute()
        ssim = self.ssim_metric.compute()
        lpips = self.lpips_metric.compute()

        self.log('val/psnr', psnr)
        self.log('val/ssim', ssim)
        self.log('val/lpips', lpips)

        ap_results: Dict[str, float] = {}
        if len(self._val_seg_stats) > 0:
            for key in self._val_seg_stats[0].keys():
                self.log(f'val/seg_{key}', float(np.mean([x[key] for x in self._val_seg_stats])))

            ap_results = self.seg_ap_metric.compute()
            for key, value in ap_results.items():
                self.log(f'val/seg_{key}', value)

        self.track_score(self._compute_track_score(float(psnr.item()), float(ssim.item()), ap_results))
        self.log_buffered()
