from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from einops import rearrange
import gsplat

from instok3d.utils.sh import compute_sh_mask
from instok3d.utils.cam import unpack_cameras, dense_intrinsics_to_params
from .gaussians import Gaussians


class GaussianRenderer(nn.Module):
    """Rasterises Gaussians into images and group probability maps with gsplat.

    Cameras ``cams`` are (B, V, 5, 4): a camera-to-world matrix on top of normalised (fx, fy, cx, cy).
    ``resolution`` is an int or an (H, W) pair.
    """

    def __init__(self,
                 sh_degree: int,
                 near_far: Tuple[float, float] = (0.01, 100.0),
                 eps2d: float = 0.2,
                 ):
        super().__init__()

        assert sh_degree >= 0
        self.near_far = near_far
        self.eps2d = eps2d
        self.sh_degree = sh_degree

        if sh_degree > 0:
            self.register_buffer('rgb_sh_mask', compute_sh_mask(sh_degree), persistent=False)
        else:
            self.rgb_sh_mask = None

    def _normalize_resolution(self, resolution):
        if isinstance(resolution, int):
            return (resolution, resolution)
        return resolution

    def _prepare_feature(self, gaussians: Gaussians):
        feature = rearrange(gaussians.feature, 'b n (k c) -> b n k c', c=3)
        if self.rgb_sh_mask is not None:
            feature = feature * self.rgb_sh_mask.view(1, 1, -1, 1)
        return feature

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, gaussians: Gaussians, cams, resolution):
        """Render RGB images (B, V, 3, H, W); aux holds ``depth`` and ``alpha``, both (B, V, H, W)."""
        resolution = self._normalize_resolution(resolution)
        cams = cams.float()

        feature = self._prepare_feature(gaussians)
        rendered, depth, alpha = self._render_gaussians(
            gaussians, feature, self.sh_degree, cams, resolution,
        )

        rendered = rearrange(rendered, 'b v h w c -> b v c h w')
        aux = {
            'depth': depth,
            'alpha': alpha,
        }
        return rendered, aux

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def render_segmentation_probs(
        self,
        gaussians: Gaussians,
        cams: torch.Tensor,
        resolution,
        gaussian_query_probs: torch.Tensor,
        gaussian_void_prob: torch.Tensor,
        query_indices: Optional[Sequence[int]] = None,
        add_background_to_void: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Splat the group and void probabilities of the Gaussians in a single pass.

        Args:
            gaussian_query_probs: (B, N, Q)
            gaussian_void_prob: (B, N)
            query_indices: groups to render; all if None.
            add_background_to_void: add the remaining transmittance (1 - alpha) to the void channel.
        Returns:
            seg_probs: (B, Q_sel + 1, V, H, W), normalised over the channels; void comes last.
            aux: dict with ``alpha`` and the unnormalised ``seg_logits``.
        """
        resolution = self._normalize_resolution(resolution)
        cams = cams.float()

        if query_indices is None:
            query_indices = list(range(int(gaussian_query_probs.shape[-1])))
        query_indices = [int(q) for q in query_indices]

        q_probs = gaussian_query_probs[:, :, query_indices].float()
        void_prob = gaussian_void_prob.float().unsqueeze(-1)
        seg_colors = torch.cat([q_probs, void_prob], dim=-1)  # (B, N, Q_sel + 1)

        seg_logits, _, alpha = self._render_gaussians(
            gaussians=gaussians,
            feature=seg_colors,
            sh_degree=None,
            cams=cams,
            resolution=resolution,
            render_mode='RGB',
        )

        seg_logits = rearrange(seg_logits, 'b v h w c -> b c v h w')
        if add_background_to_void:
            seg_logits[:, -1] = seg_logits[:, -1] + (1.0 - alpha).float()
        seg_probs = seg_logits / seg_logits.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return seg_probs, {'alpha': alpha, 'seg_logits': seg_logits}

    def _render_gaussians(self,
                          gaussians: Gaussians,
                          feature,
                          sh_degree,
                          cams,
                          resolution,
                          render_mode: str = 'RGB+ED'):
        """Rasterise ``feature`` (SH coefficients, or plain channels if ``sh_degree`` is None).

        Returns ``(rendered (B, V, H, W, C), depth (B, V, H, W) or None, alpha (B, V, H, W))``.
        """
        grid_h, grid_w = resolution
        near, far = self.near_far
        c2w, fxfycxcy = unpack_cameras(cams)

        intrinsic_rescaled = fxfycxcy.clone()
        intrinsic_rescaled[..., 0] *= grid_w
        intrinsic_rescaled[..., 1] *= grid_h
        intrinsic_rescaled[..., 2] *= grid_w
        intrinsic_rescaled[..., 3] *= grid_h
        intrinsic = dense_intrinsics_to_params(intrinsic_rescaled)

        means = gaussians.xyz.float()
        quats = gaussians.rotation.float()
        scales = gaussians.scale.float()
        opacities = gaussians.opacity.float()

        w2c = torch.linalg.inv(c2w)

        rendered, alpha, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=feature,
            viewmats=w2c,
            Ks=intrinsic,
            width=grid_w,
            height=grid_h,
            near_plane=near,
            far_plane=far,
            sh_degree=sh_degree,
            eps2d=self.eps2d,
            render_mode=render_mode,
            packed=False,
        )
        depth = None
        if render_mode == 'RGB+ED':
            depth = rendered[..., -1]
            rendered = rendered[..., :-1]
        return rendered, depth, alpha.squeeze(-1)
