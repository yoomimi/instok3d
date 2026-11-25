"""Open-vocabulary semantic segmentation benchmark of the lifted features."""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy

import engine
from engine import inject
from engine.components import Evaluator
from instok3d.solvers.semantic import mask_void_predictions
from instok3d.vis.image import make_grid
from instok3d.vis.sem_class import colorize_class_map


class SemanticEvaluator(Evaluator):
    """mIoU / pixel accuracy of the classes predicted from rendered semantic features.

    The features are rendered at the source (input) and at the target cameras and classified
    with the teacher's text encoder into the solver's ``miou_labels``. Both metrics are
    accumulated over all pixels of the test set, separately for source and target views; void
    pixels (label 0) are ignored. ``metrics_report.json`` and one image per sample (rows:
    ground-truth views, rendered views, predicted classes, ground-truth classes) are written
    to ``<output_subdir>/`` of the run directory.

    Args:
        rgb_resolution: resolution of the rendered views in the saved images.
        feature_resolution: resolution of the rendered feature maps.
    """

    @inject('dm', 'solver')
    def __init__(
        self,
        dm,
        solver,
        output_subdir: str = 'outputs/semantic',
        rgb_resolution: int = 256,
        feature_resolution: int = 128,
    ):
        super().__init__(progress_bar=True)
        self.dm = dm
        self.solver = solver
        self.model = solver.model
        self.dm.eval_batch_size = 1

        self.rgb_resolution = int(rgb_resolution)
        self.feature_resolution = int(feature_resolution)

        self.report_dir = engine.to_output_dir(output_subdir)
        self.merged_dir = os.path.join(self.report_dir, 'merged')
        os.makedirs(self.merged_dir, exist_ok=True)

        # Class 0 is void and ignored.
        n_cls = len(solver.miou_labels) + 1
        self.miou_source = JaccardIndex(task='multiclass', num_classes=n_cls, ignore_index=0)
        self.acc_source = Accuracy(task='multiclass', num_classes=n_cls, ignore_index=0)
        self.miou_target = JaccardIndex(task='multiclass', num_classes=n_cls, ignore_index=0)
        self.acc_target = Accuracy(task='multiclass', num_classes=n_cls, ignore_index=0)

    def step(self, batch, batch_idx):
        if 'input_labelmap' not in batch or 'target_labelmap' not in batch:
            raise KeyError('SemanticEvaluator needs a dataset with semantic label maps.')

        input_images = batch['input_images']    # (1, N, 3, H, W)
        target_images = batch['target_images']  # (1, M, 3, H, W)
        num_inputs = input_images.shape[1]

        bb_out = self.solver._forward_backbone(input_images, batch)
        sem_in = self.solver._teacher_features(input_images)
        gaussians, recon_aux, sem_aux = self.model(input_images, bb_out, sem_in)

        all_cams = torch.cat([batch['input_cams'], batch['target_cams']], dim=1)
        final_sem = self.model.decode_features(
            gaussians, recon_aux, sem_aux, cams=all_cams, resolution=self.feature_resolution,
        )['final_sem']                          # (1, N + M, C, h, w)
        rendered, _ = self.model.recon_model.decode_images(
            gaussians, all_cams, resolution=self.rgb_resolution,
        )

        pred_source = self.solver.predict_classes(final_sem[:, :num_inputs])
        pred_target = self.solver.predict_classes(final_sem[:, num_inputs:])
        self._update_metrics(
            pred_source, batch['input_labelmap'], self.miou_source, self.acc_source,
        )
        self._update_metrics(
            pred_target, batch['target_labelmap'], self.miou_target, self.acc_target,
        )

        idx = int(batch['idx'][0])
        scene_name = batch['scene_name'][0]
        self._save_visualization(
            os.path.join(self.merged_dir, f'{idx:04d}_{scene_name}.png'),
            images=torch.cat([input_images, target_images], dim=1)[0],
            rendered=rendered[0],
            pred=torch.cat([pred_source, pred_target], dim=1)[0],
            labelmap=torch.cat([batch['input_labelmap'], batch['target_labelmap']], dim=1)[0],
        )
        return {'idx': idx, 'scene_name': scene_name}

    def result(self, results):
        agg = {
            'n_samples': len(results['idx']),
            'source_mIoU': float(self.miou_source.compute().item()),
            'source_acc': float(self.acc_source.compute().item()),
            'target_mIoU': float(self.miou_target.compute().item()),
            'target_acc': float(self.acc_target.compute().item()),
        }

        print()
        print('=== Semantic feature lifting ===')
        print(f'  samples={agg["n_samples"]}')
        print(f'  source views: mIoU={agg["source_mIoU"]:.4f}  Acc={agg["source_acc"]:.4f}')
        print(f'  target views: mIoU={agg["target_mIoU"]:.4f}  Acc={agg["target_acc"]:.4f}')

        report_path = os.path.join(self.report_dir, 'metrics_report.json')
        with open(report_path, 'w') as f:
            json.dump({'aggregate': agg}, f, indent=2)
        print(f'Report saved to {report_path}')
        return agg

    @staticmethod
    def _update_metrics(pred, labelmap, miou: JaccardIndex, acc: Accuracy) -> None:
        pred, labelmap = mask_void_predictions(pred, labelmap)
        miou.update(pred, labelmap)
        acc.update(pred, labelmap)

    @staticmethod
    def _save_visualization(path, images, rendered, pred, labelmap) -> None:
        """Save a grid with one column per view.

        Args:
            images: (V, 3, H, W) ground-truth views in [0, 1].
            rendered: (V, 3, H', W') rendered views.
            pred: (V, h, w) predicted class ids.
            labelmap: (V, H, W) ground-truth class ids.
        """
        hw = images.shape[-2:]
        if rendered.shape[-2:] != hw:
            rendered = F.interpolate(rendered, size=hw, mode='bilinear', align_corners=False)
        if pred.shape[-2:] != hw:
            pred = F.interpolate(pred[:, None].float(), size=hw, mode='nearest')[:, 0]

        num_views = images.shape[0]
        merged = make_grid(
            [
                *(_to_uint8_image(images[v]) for v in range(num_views)),
                *(_to_uint8_image(rendered[v]) for v in range(num_views)),
                *(colorize_class_map(pred[v].long().cpu().numpy()) for v in range(num_views)),
                *(colorize_class_map(labelmap[v].long().cpu().numpy()) for v in range(num_views)),
            ],
            num_rows=4, num_cols=num_views,
        )
        Image.fromarray(merged).save(path)


def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """(3, H, W) image in [0, 1] -> (H, W, 3) uint8."""
    arr = x.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).astype(np.uint8)
