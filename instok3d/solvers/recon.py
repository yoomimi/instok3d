"""Solver for feed-forward 3D Gaussian reconstruction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from engine import inject
from engine.components import DataModule, Solver
from instok3d.losses.image import ReconstructionImageLoss
from instok3d.metrics.image import MultiViewLPIPS, MultiViewPSNR, MultiViewSSIM
from instok3d.utils.training import forward_backbone, resize_views, warmup_cosine_factor
from instok3d.vis.image import views_to_grid

logger = logging.getLogger(__name__)

_TRACK_METRICS = ('psnr', 'ssim', 'psnr+ssim')


class ReconSolver(Solver):
    """Trains a reconstruction model, optionally fine-tuning the backbone with it.

    Training renders the input and the target views and applies MSE + LPIPS; validation
    reports PSNR / SSIM / LPIPS on the target views of every validation loader.

    Args:
        lr: learning rate of the model.
        backbone_lr_scale: learning rate of the trainable backbone parameters, relative to ``lr``.
        warmup_epochs: length of the linear learning-rate warm-up.
        min_lr: learning rate at the end of the cosine decay (scaled by ``backbone_lr_scale``
            for the backbone).
        coeff_perceptual: weight of the LPIPS loss.
        use_perceptual_compile: compile the LPIPS network with ``torch.compile``.
        val_vis_interval: log the renderings of every n-th validation sample.
        track_metric: checkpoint selection score, one of 'psnr', 'ssim', 'psnr+ssim'.
    """

    @inject('dm', 'model', 'backbone')
    def __init__(
        self,
        dm: DataModule,
        model: nn.Module,
        backbone: nn.Module,
        lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 0.05,
        warmup_epochs: float = 1.0,
        min_lr: float = 1e-6,
        coeff_perceptual: float = 0.05,
        use_perceptual_compile: bool = True,
        val_vis_interval: int = 1,
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
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = warmup_epochs
        self.min_lr = float(min_lr)
        self.val_vis_interval = val_vis_interval
        self.use_perceptual_compile = use_perceptual_compile

        self.recon_loss_fn = ReconstructionImageLoss(coeff_perceptual=coeff_perceptual)

        self._val_dataset_names: List[str] = list(getattr(dm, 'val_dataset_names', None) or [])
        n_val = max(1, len(self._val_dataset_names))
        self.psnr_metrics = nn.ModuleList([MultiViewPSNR() for _ in range(n_val)])
        self.ssim_metrics = nn.ModuleList([MultiViewSSIM() for _ in range(n_val)])
        self.lpips_metrics = nn.ModuleList([MultiViewLPIPS() for _ in range(n_val)])

        self.create_image_log_buffer('vis/rendered', log_once=False)
        self.create_image_log_buffer('vis/gt', log_once=True)

        self.total_steps: int = 1
        self.warmup_steps: int = 0

    def setup(self, stage: str = 'fit') -> None:
        self.total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        steps_per_epoch = self.total_steps / max(self.trainer.max_epochs, 1)
        self.warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        self.recon_loss_fn.setup(
            trainer=self.trainer,
            use_perceptual_compile=self.use_perceptual_compile,
            is_debug=self.is_debug,
        )
        logger.info('ReconSolver setup: total_steps=%d', self.total_steps)

    def configure_optimizers(self):
        param_groups = [{
            'name': 'model',
            'params': list(self.model.parameters()),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                'name': 'backbone',
                'params': backbone_params,
                'lr': self.lr * self.backbone_lr_scale,
                'weight_decay': self.weight_decay,
            })
        optimizer = torch.optim.AdamW(param_groups)

        def lr_lambda(group_lr: float):
            # Every group decays to the same fraction of its own learning rate.
            group_min_lr = self.min_lr * (group_lr / max(self.lr, 1e-10))
            return lambda step: warmup_cosine_factor(
                step, self.warmup_steps, self.total_steps, group_lr, group_min_lr,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lr_lambda(group['lr']) for group in param_groups],
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        input_images = batch['input_images']    # (B, V_in, 3, H, W)
        target_images = batch['target_images']  # (B, V_tgt, 3, H, W)
        input_cams = batch['input_cams']        # (B, V_in, 5, 4)
        target_cams = batch['target_cams']      # (B, V_tgt, 5, 4)

        bb_out = self._forward_backbone(input_images, batch)
        gaussians, _ = self.model(input_images, bb_out)

        rendered, _ = self.model.decode_images(gaussians, torch.cat([input_cams, target_cams], dim=1))
        targets = torch.cat([
            resize_views(input_images, rendered.shape[-2:]),
            resize_views(target_images, rendered.shape[-2:]),
        ], dim=1)
        loss_out = self.recon_loss_fn(rendered, targets)
        total_loss = loss_out.total

        self.log('train/loss', total_loss, rank_zero_only=True)
        self.log('train/mse', loss_out.mse, rank_zero_only=True)
        self.log('train/perceptual', loss_out.perceptual, rank_zero_only=True)
        self.log('train/mean_opacity', gaussians.opacity.mean().item(), rank_zero_only=True)
        self.log('train/mean_scale', gaussians.scale.mean().item(), rank_zero_only=True)
        group_lrs = {group['name']: group['lr'] for group in self.trainer.optimizers[0].param_groups}
        self.log('train/lr', group_lrs['model'], rank_zero_only=True)
        if 'backbone' in group_lrs:
            self.log('train/backbone_lr', group_lrs['backbone'], rank_zero_only=True)

        # A NaN / Inf loss would poison every weight despite gradient clipping, so the step
        # is skipped instead; all processes must take the same decision.
        finite = torch.isfinite(total_loss).detach()
        if self.trainer.world_size > 1:
            finite = self.all_gather(finite).all()
        if not bool(finite):
            logger.warning(
                'Non-finite training loss at global_step %d (loss=%s); skipping the optimizer step.',
                self.global_step, total_loss.detach(),
            )
            self.log('train/nonfinite_skip', 1.0, rank_zero_only=True)
            return None

        self.log('train/nonfinite_skip', 0.0, rank_zero_only=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        input_images = batch['input_images']
        target_images = batch['target_images']
        target_cams = batch['target_cams']

        bb_out = self._forward_backbone(input_images, batch)
        gaussians, _ = self.model(input_images, bb_out)

        rendered_target, _ = self.model.decode_images(gaussians, target_cams)
        gt_target = resize_views(target_images, rendered_target.shape[-2:])

        self.psnr_metrics[dataloader_idx].update(rendered_target, gt_target)
        self.ssim_metrics[dataloader_idx].update(rendered_target, gt_target)
        self.lpips_metrics[dataloader_idx].update(rendered_target, gt_target)

        if dataloader_idx == 0:
            idx = batch['idx']
            for b in range(rendered_target.shape[0]):
                if idx[b].item() % self.val_vis_interval == 0:
                    self._log_val_images(rendered_target[b], gt_target[b])

    def _log_val_images(self, rendered: torch.Tensor, gt: torch.Tensor) -> None:
        """Log the rendered and ground-truth views (V, 3, H, W) of one sample."""
        n_rows = 2 if rendered.shape[0] >= 4 else 1
        self.add_log_buffer_items('vis/rendered', views_to_grid(rendered, n_rows=n_rows))
        self.add_log_buffer_items('vis/gt', views_to_grid(gt, n_rows=n_rows))

    def _compute_track_score(self, psnr: float, ssim: float) -> float:
        if self.track_metric == 'psnr':
            return psnr
        if self.track_metric == 'ssim':
            return ssim
        return psnr + 10.0 * ssim

    def on_validation_epoch_end(self) -> None:
        # Metrics are logged under 'val/' for a single loader, under '<name>/' for a loader
        # whose name contains '/', and under 'val/<name>/' otherwise.
        names: List[Optional[str]] = list(self._val_dataset_names) or [None]

        scores: List[float] = []
        for i, name in enumerate(names):
            psnr = self.psnr_metrics[i].compute()
            ssim = self.ssim_metrics[i].compute()
            lpips = self.lpips_metrics[i].compute()
            self.psnr_metrics[i].reset()
            self.ssim_metrics[i].reset()
            self.lpips_metrics[i].reset()
            if name is None:
                prefix = 'val'
            elif '/' in name:
                prefix = name
            else:
                prefix = f'val/{name}'
            self.log(f'{prefix}/psnr', psnr)
            self.log(f'{prefix}/ssim', ssim)
            self.log(f'{prefix}/lpips', lpips)
            scores.append(self._compute_track_score(float(psnr.item()), float(ssim.item())))

        self.track_score(sum(scores) / len(scores))
        self.log_buffered()
