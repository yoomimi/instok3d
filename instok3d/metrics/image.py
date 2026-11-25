"""Image quality metrics averaged over the views of each sample."""

import lpips
import torch
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.functional.image import structural_similarity_index_measure

from instok3d.utils.freeze import convert_to_buffer


class MultiViewPSNR(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    @torch.autocast(device_type='cuda', enabled=False)
    def update(self, out, target):
        """out, target: (B, V, C, H, W); both are clamped to [0, 1] in place."""
        out = out.detach()
        out.clamp_(0, 1)
        target.clamp_(0, 1)
        mse = F.mse_loss(out, target, reduction='none')
        mse = mse.contiguous().view(mse.size(0), mse.size(1), -1).mean(dim=-1)
        psnr = -10 * torch.log10(mse)
        psnr = psnr.mean(dim=1)

        self.result += psnr.sum()
        self.total += psnr.numel()

    def compute(self):
        return self.result.float() / self.total


class MultiViewSSIM(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    @torch.autocast(device_type='cuda', enabled=False)
    def update(self, out, target):
        """out, target: (B, V, C, H, W) in [0, 1]."""
        out = out.detach().float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        B, V = out.shape[:2]
        ssim = structural_similarity_index_measure(
            out.flatten(0, 1), target.flatten(0, 1), data_range=1.0, reduction='none',
        )  # (B * V,)

        ssim = ssim.view(B, V).mean(dim=1)
        self.result += ssim.sum()
        self.total += ssim.numel()

    def compute(self):
        return self.result.float() / self.total


class MultiViewLPIPS(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

        self._lpips_net = lpips.LPIPS(net='vgg', verbose=False)
        # Fixed pretrained network: kept out of checkpoints.
        convert_to_buffer(self._lpips_net, persistent=False)
        self._lpips_net.eval()

    @torch.autocast(device_type='cuda', enabled=False)
    def update(self, out, target):
        """out, target: (B, V, C, H, W) in [0, 1]."""
        out = out.detach().clamp(0, 1)
        target = target.clamp(0, 1)
        B, V = out.shape[:2]
        out_flat = out.flatten(0, 1) * 2 - 1  # LPIPS expects [-1, 1]
        target_flat = target.flatten(0, 1) * 2 - 1
        lp = self._lpips_net(out_flat, target_flat)  # (B * V, 1, 1, 1)
        lp = lp.squeeze().view(B, V).mean(dim=1)
        self.result += lp.sum()
        self.total += lp.numel()

    def compute(self):
        return self.result.float() / self.total
