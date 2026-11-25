"""Solver for semantic feature lifting on top of a frozen tokenizer."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy

from engine import inject
from engine.components import DataModule, Solver
from instok3d.losses.semantic import SemanticDistillationLoss
from instok3d.utils.training import forward_backbone, warmup_cosine_factor
from instok3d.vis.image import make_image_grid
from instok3d.vis.mask import colorize_label_map
from instok3d.vis.sem_class import colorize_class_map

logger = logging.getLogger(__name__)

_TRACK_METRICS = ('mIoU', 'feat_cos')


def mask_void_predictions(
    pred: torch.Tensor, labelmap: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bring class predictions and labels to the form the segmentation metrics expect.

    Args:
        pred: (B, V, H, W) predicted class ids in 1..n.
        labelmap: (B, V, h, w) class ids with 0 = void; resized (nearest) to (H, W) if needed.
    Returns:
        (B*V, H, W) predictions, set to 0 at void pixels, and (B*V, H, W) labels.
    """
    pred = pred.flatten(0, 1)
    labelmap = labelmap.flatten(0, 1).long()
    if labelmap.shape[-2:] != pred.shape[-2:]:
        labelmap = F.interpolate(
            labelmap.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest',
        ).squeeze(1).long()
    pred = torch.where(labelmap != 0, pred, torch.zeros_like(pred))
    return pred, labelmap


class SemanticSolver(Solver):
    """Trains a ``FeatureLiftingModel`` by distilling the features of a frozen 2D teacher.

    Validation reports the cosine similarity to the teacher features on the target views and,
    for loaders that provide ``target_labelmap``, the mIoU / pixel accuracy of the classes
    ``miou_labels`` predicted from the rendered features with the teacher's text encoder.

    Args:
        warmup_epochs: length of the linear learning-rate warm-up.
        min_lr: learning rate at the end of the cosine decay.
        pixel_loss_weight: weight of the pixel-level distillation loss.
        instance_shared_weight: weight of the group-level distillation loss.
        mass_min: minimum number of pixels of a group that enters the group-level loss.
        val_vis_interval: log the predictions of every n-th validation sample.
        miou_labels: class names; label id i + 1 is ``miou_labels[i]`` and 0 is void.
        track_metric: checkpoint selection score, 'mIoU' or 'feat_cos'.
    """

    @inject('dm', 'model', 'backbone', 'teacher')
    def __init__(
        self,
        dm: DataModule,
        model: nn.Module,
        backbone: nn.Module,
        teacher: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: float = 1.0,
        min_lr: float = 1e-6,
        pixel_loss_weight: float = 1.0,
        instance_shared_weight: float = 1.0,
        mass_min: float = 1.0,
        val_vis_interval: int = 5,
        miou_labels: Sequence[str] = (
            'wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other',
        ),
        track_metric: str = 'mIoU',
    ):
        super().__init__(track_max_score=True)
        if track_metric not in _TRACK_METRICS:
            raise ValueError(f'track_metric must be one of {_TRACK_METRICS}; got {track_metric!r}')
        self.track_metric = track_metric

        self.dm = dm
        for callback in getattr(dm, 'get_callbacks', lambda: [])():
            self.add_callback(callback)
        self.model = model
        self.backbone = backbone
        self.teacher = teacher

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = float(warmup_epochs)
        self.min_lr = float(min_lr)
        self.val_vis_interval = int(val_vis_interval)
        self.miou_labels: List[str] = list(miou_labels)

        self.loss_fn = SemanticDistillationLoss(
            pixel_loss_weight=pixel_loss_weight,
            instance_shared_weight=instance_shared_weight,
            mass_min=mass_min,
        )

        # Class 0 is void and ignored.
        n_cls = len(self.miou_labels) + 1
        self.miou = JaccardIndex(task='multiclass', num_classes=n_cls, ignore_index=0)
        self.acc = Accuracy(task='multiclass', num_classes=n_cls, ignore_index=0)

        self.create_image_log_buffer('vis/sem_pred_class', log_once=False)
        if self.model.use_group:
            self.create_image_log_buffer('vis/sem_pred_class_shared', log_once=False)
        self.create_image_log_buffer('vis/sem_gt_class', log_once=True)
        self.create_image_log_buffer('vis/instance_mask', log_once=True)

        self.total_steps: int = 1
        self.warmup_steps: int = 0
        self._val_feat_cos: List[torch.Tensor] = []
        self._val_miou_updated: bool = False

    def setup(self, stage: str = 'fit') -> None:
        self.total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        steps_per_epoch = self.total_steps / max(self.trainer.max_epochs, 1)
        self.warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        logger.info('SemanticSolver setup: total_steps=%d', self.total_steps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay,
        )
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

    @torch.no_grad()
    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher features (B, V, C, h, w) of images (B, V, 3, H, W) in [0, 1]."""
        feat = self.teacher.extract_features(images.flatten(0, 1))
        return feat.unflatten(0, images.shape[:2])

    def _forward_decode(self, batch: Dict[str, Any]):
        """Lift the input views and render the semantic features of the target views.

        Returns the decoded feature dict, the aux dict of the lifting model and the teacher
        features (B, V_tgt, C, h, w) of the target views.
        """
        input_images = batch['input_images']
        target_images = batch['target_images']
        num_inputs = input_images.shape[1]

        bb_out = self._forward_backbone(input_images, batch)
        # One teacher call for the input and the target views.
        feat = self._teacher_features(torch.cat([input_images, target_images], dim=1))
        sem_in, lseg_target = feat[:, :num_inputs], feat[:, num_inputs:]

        gaussians, recon_aux, sem_aux = self.model(input_images, bb_out, sem_in)
        # Rendered on the grid of the teacher features.
        decoded = self.model.decode_features(
            gaussians, recon_aux, sem_aux, cams=batch['target_cams'],
            resolution=lseg_target.shape[-1],
        )
        return decoded, sem_aux, lseg_target

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        decoded, sem_aux, lseg_target = self._forward_decode(batch)

        out = self.loss_fn(
            final_sem=decoded['final_sem'],
            lseg_target=lseg_target,
            pixel_oh=decoded['pixel_oh'],
            instance_full=sem_aux.get('instance_full'),
        )

        self.log('train/loss', out['total'])
        self.log('train/feat_loss', out['feat_loss'])
        self.log('train/inst_loss', out['inst_loss'])
        self.log('train/inst_valid_count', out['inst_valid_count'])
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return out['total']

    def on_validation_epoch_start(self) -> None:
        self._val_feat_cos = []
        self._val_miou_updated = False
        self.miou.reset()
        self.acc.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        decoded, _, lseg_target = self._forward_decode(batch)
        final_sem = decoded['final_sem']  # (B, V, C, H, W)

        feat_cos = F.cosine_similarity(final_sem, lseg_target.detach(), dim=2)
        self._val_feat_cos.append(feat_cos.mean().detach().float())

        if 'target_labelmap' in batch:
            pred, labelmap = mask_void_predictions(
                self.predict_classes(final_sem), batch['target_labelmap'],
            )
            self.miou.update(pred, labelmap)
            self.acc.update(pred, labelmap)
            self._val_miou_updated = True

        idx = batch['idx']
        for b in range(final_sem.shape[0]):
            if int(idx[b]) % self.val_vis_interval == 0:
                self._log_val_vis(decoded, batch, sample_idx=b)

    @torch.no_grad()
    def predict_classes(self, features: torch.Tensor) -> torch.Tensor:
        """Classify semantic features (B, V, C, H, W) into ``miou_labels``.

        Returns (B, V, H', W') class ids in 1..n at the resolution of the teacher's logits.
        """
        logits = self.teacher.decode_feature(features.flatten(0, 1), self.miou_labels)
        return (logits.argmax(dim=1) + 1).unflatten(0, features.shape[:2])

    @torch.no_grad()
    def _log_val_vis(
        self, decoded: Dict[str, torch.Tensor], batch: Dict[str, Any], sample_idx: int,
    ) -> None:
        """Log the class predictions and the ground truth of the first views of one sample."""
        n_vis = min(decoded['final_sem'].shape[1], 4)

        def log_class_maps(key: str, class_maps: torch.Tensor) -> None:
            images = [
                colorize_class_map(class_maps[vi].long().cpu().numpy()) for vi in range(n_vis)
            ]
            self.add_log_buffer_items(key, make_image_grid(images, n_cols=n_vis))

        log_class_maps('vis/sem_pred_class', self.predict_classes(decoded['final_sem'])[sample_idx])
        if decoded['group_feat'] is not None:
            log_class_maps(
                'vis/sem_pred_class_shared',
                self.predict_classes(decoded['group_feat'])[sample_idx],
            )
        if 'target_labelmap' in batch:
            log_class_maps('vis/sem_gt_class', batch['target_labelmap'][sample_idx])

        if 'target_masks' in batch:
            inst = batch['target_masks'][sample_idx]  # (V, H, W) ids or (V, M, H, W) masks
            if inst.ndim == 4:
                inst = inst.argmax(dim=1)
            images = [colorize_label_map(inst[vi].long().cpu().numpy()) for vi in range(n_vis)]
            self.add_log_buffer_items('vis/instance_mask', make_image_grid(images, n_cols=n_vis))

    def on_validation_epoch_end(self) -> None:
        if self._val_feat_cos:
            cos_mean = torch.stack(self._val_feat_cos).mean()
            self.log('val/feat_cos', cos_mean, sync_dist=True)
        else:
            cos_mean = torch.tensor(0.0)

        miou_score = None
        if self._val_miou_updated:
            miou_score = self.miou.compute()
            self.log('val/sem_mIoU', miou_score, sync_dist=True)
            self.log('val/sem_acc', self.acc.compute(), sync_dist=True)
        self.miou.reset()
        self.acc.reset()

        # Without labelled validation data the feature similarity is tracked instead.
        if self.track_metric == 'mIoU' and miou_score is not None:
            self.track_score(float(miou_score.item()))
        else:
            self.track_score(float(cos_mean.item()))

        self.log_buffered()
