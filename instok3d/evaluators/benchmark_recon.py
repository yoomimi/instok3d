"""Novel-view synthesis benchmark (PSNR / SSIM / LPIPS of the target views)."""

import json
import os

import lpips
import torch
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import engine
from engine import inject
from engine.components import Evaluator
from instok3d.evaluators.unified_vis import align_target_cams


class BenchmarkReconEvaluator(Evaluator):
    """Reconstruction metrics on the target views, optionally after test-time pose alignment.

    PSNR and SSIM are accumulated over all target views of the test set and LPIPS (VGG) is
    averaged over the target views. ``metrics_report.json`` (aggregate and per-sample metrics)
    and ``per_scene.csv`` (one row per sample) are written to
    ``<output_subdir>[_pa<pose_align_steps>]/`` of the run directory.
    """

    @inject('dm', 'solver')
    def __init__(
        self,
        dm,
        solver,
        pose_align_steps: int = 0,
        pose_align_lr: float = 5e-3,
        output_subdir: str = 'outputs/benchmark_recon',
    ):
        super().__init__()
        self.dm = dm
        self.solver = solver
        self.model = solver.model
        self.backbone = solver.backbone
        self.dm.eval_batch_size = 1

        self.pose_align_steps = int(pose_align_steps)
        self.pose_align_lr = float(pose_align_lr)

        suffix = f'_pa{self.pose_align_steps}' if self.pose_align_steps > 0 else ''
        self.report_dir = engine.to_output_dir(f'{output_subdir}{suffix}')
        os.makedirs(self.report_dir, exist_ok=True)

        self.psnr_agg = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_agg = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips_vgg = lpips.LPIPS(net='vgg', verbose=False)
        for p in self.lpips_vgg.parameters():
            p.requires_grad_(False)
        self.lpips_scores = []

    def step(self, batch, batch_idx):
        input_images = batch['input_images']    # (1, N, 3, H, W)
        target_images = batch['target_images']  # (1, M, 3, H, W)
        target_cams = batch['target_cams']      # (1, M, 5, 4)

        bb_out = self.solver._forward_backbone(input_images, batch)
        gaussians, _ = self.model(input_images, bb_out)

        if self.pose_align_steps > 0:
            target_cams = align_target_cams(
                self.model.decode_images, gaussians, target_cams, target_images,
                steps=self.pose_align_steps, lr=self.pose_align_lr,
            )
        rendered, _ = self.model.decode_images(gaussians, target_cams)
        if rendered.shape != target_images.shape:
            raise ValueError(
                f'Rendered views {tuple(rendered.shape)} and target views '
                f'{tuple(target_images.shape)} must have the same resolution.'
            )

        rendered = rendered.detach().clamp(0.0, 1.0).flatten(0, 1)  # (M, 3, H, W)
        target = target_images.clamp(0.0, 1.0).flatten(0, 1)

        # LPIPS takes the [0, 1] images as they are (no rescaling to [-1, 1]), which is the
        # protocol of the baselines this benchmark is compared with.
        view_lpips = self.lpips_vgg(rendered, target).flatten().tolist()

        self.psnr_agg.update(rendered, target)
        self.ssim_agg.update(rendered, target)
        self.lpips_scores.extend(view_lpips)

        return {
            'scene_name': batch['scene_name'][0],
            'psnr': float((-10.0 * torch.log10(F.mse_loss(rendered, target))).item()),
            'ssim': float(
                structural_similarity_index_measure(rendered, target, data_range=1.0).item()
            ),
            'lpips': sum(view_lpips) / len(view_lpips),
        }

    def result(self, results):
        num_samples = len(results['scene_name'])
        agg = {
            'psnr': float(self.psnr_agg.compute().item()),
            'ssim': float(self.ssim_agg.compute().item()),
            'lpips': float(sum(self.lpips_scores) / len(self.lpips_scores)),
        }
        per_scene = [
            {key: values[i] for key, values in results.items()} for i in range(num_samples)
        ]

        report = {
            'config': {
                'pose_align_steps': self.pose_align_steps,
                'pose_align_lr': self.pose_align_lr,
                'n_scenes': num_samples,
                'lpips_lib': 'lpips.LPIPS(net=vgg) on [0,1] inputs',
                'dm_class': type(self.dm).__name__,
            },
            'aggregate': agg,
            'per_scene': per_scene,
        }
        report_path = os.path.join(self.report_dir, 'metrics_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        csv_path = os.path.join(self.report_dir, 'per_scene.csv')
        with open(csv_path, 'w') as f:
            f.write('scene,psnr,ssim,lpips\n')
            for r in per_scene:
                f.write(f'{r["scene_name"]},{r["psnr"]:.4f},{r["ssim"]:.4f},{r["lpips"]:.4f}\n')

        print(f'\n=== BenchmarkReconEvaluator: {num_samples} samples, '
              f'pose_align_steps={self.pose_align_steps} ===')
        print('     PSNR     SSIM    LPIPS')
        print(f'  {agg["psnr"]:7.4f}   {agg["ssim"]:.4f}   {agg["lpips"]:.4f}')
        print(f'  report: {report_path}')
        print(f'  per-sample metrics: {csv_path}')
