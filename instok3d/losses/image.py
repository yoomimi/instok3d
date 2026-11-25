"""Image reconstruction loss: MSE + LPIPS."""

from dataclasses import dataclass

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lpips import LPIPS
from torch import nn

from instok3d.utils.freeze import convert_to_buffer


@dataclass
class ReconstructionLossResult:
    total: torch.Tensor
    mse: torch.Tensor
    perceptual: torch.Tensor


class ReconstructionImageLoss(nn.Module):
    """MSE + ``coeff_perceptual`` * LPIPS between rendered and target views."""

    def __init__(self, coeff_perceptual: float = 0.05):
        super().__init__()
        self.coeff_perceptual = float(coeff_perceptual)
        self.mse_criterion = nn.MSELoss()
        self.perceptual_loss = LpipsLoss()

    def setup(self, trainer=None, use_perceptual_compile: bool = False, is_debug: bool = False):
        """Load the LPIPS network; must be called once before the first forward pass."""
        self.perceptual_loss.setup(trainer)
        if use_perceptual_compile and not is_debug:
            self.perceptual_loss = torch.compile(self.perceptual_loss)

    def forward(self, rendered: torch.Tensor, targets: torch.Tensor) -> ReconstructionLossResult:
        """rendered, targets: (B, V, 3, H, W) in [0, 1]."""
        rendered_flat = rendered.flatten(0, 1)
        targets_flat = targets.flatten(0, 1)

        mse_loss = self.mse_criterion(rendered_flat, targets_flat)
        perceptual_loss = self.perceptual_loss(rendered_flat, targets_flat)
        total_loss = mse_loss + self.coeff_perceptual * perceptual_loss

        return ReconstructionLossResult(
            total=total_loss,
            mse=mse_loss,
            perceptual=perceptual_loss,
        )


class LpipsLoss(nn.Module):
    """Frozen LPIPS (VGG) network; its weights are not saved in checkpoints."""

    def __init__(self):
        super().__init__()
        self.lpips = None

    def train(self, mode: bool = True):
        return super().train(False)

    def setup(self, trainer):
        self._download_weights()
        if trainer is not None:
            trainer.strategy.barrier()
        self.lpips = LPIPS(net='vgg')
        convert_to_buffer(self.lpips, persistent=False)
        self.requires_grad_(False)
        self.eval()

    @rank_zero_only
    def _download_weights(self):
        LPIPS(net='vgg')

    def forward(self, pred_img, target_img):
        """pred_img, target_img: (N, 3, H, W) in [0, 1]."""
        return self.lpips(pred_img * 2 - 1, target_img * 2 - 1).mean()
