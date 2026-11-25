import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from instok3d.modules.transformer import init_model_weights
from .gaussians import Gaussians


def _make_head(embed_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, embed_dim, bias=False),
        nn.GELU(),
        nn.Linear(embed_dim, out_dim, bias=False),
    )


class GaussianProjector(nn.Module):
    """Decodes every anchor token into ``num_gaussians_per_anchor`` Gaussians placed around its anchor point.

    The attribute ``K`` is the number of Gaussians per anchor (N_g in the paper), not the number of anchors.
    """

    def __init__(self,
                 embed_dim: int,
                 sh_degree: int,
                 scale_range: Tuple[float, float],
                 num_gaussians_per_anchor: int = 4,
                 opacity_bias: float = 0,
                 scale_bias: float = 0,
                 scale_mult: float = 1.,
                 xyz_scale_mult: float = 0.1,
                 offset_bound: float = -1,
                 base_xyz_scale_mult: float = 0.0,
                 ):
        """
        Args:
            scale_range: (min, max) scale before ``scale_mult``; only the maximum is enforced.
            xyz_scale_mult: multiplier of the raw per-Gaussian offsets.
            offset_bound: if > 0, offsets are squashed by tanh to at most this distance.
            base_xyz_scale_mult: if > 0, the anchor point is refined by a residual with this multiplier.
        """
        super().__init__()

        assert sh_degree >= 0
        rgb_dim = 3 * (sh_degree + 1) ** 2

        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.scale_range = scale_range
        self.opacity_bias = opacity_bias
        self.scale_bias = scale_bias
        self.scale_mult = scale_mult
        self.xyz_scale_mult = xyz_scale_mult
        self.offset_bound = float(offset_bound)
        self.base_xyz_scale_mult = float(base_xyz_scale_mult)
        self.K = num_gaussians_per_anchor

        self.norm = nn.LayerNorm(self.embed_dim)

        # rotation (4) + scale (3) + opacity (1) per Gaussian
        self.geometry_head = _make_head(self.embed_dim, self.K * 8)
        self.rgb_head = _make_head(self.embed_dim, self.K * rgb_dim)
        self.coord_head = _make_head(self.embed_dim, self.K * 3)

        self.anchor_coord_head = None
        if self.base_xyz_scale_mult > 0.0:
            self.anchor_coord_head = _make_head(self.embed_dim, 3)

        self.reset_parameters()

    def reset_parameters(self):
        init_model_weights(self)
        init_model_weights(self.coord_head, 0.01)
        if self.anchor_coord_head is not None:
            # The anchor refinement starts as an exact zero residual.
            nn.init.zeros_(self.anchor_coord_head[-1].weight)

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, x, base_xyz):
        """
        Args:
            x: (B, N, D) anchor tokens.
            base_xyz: (B, N, 3) anchor points.
        Returns:
            gaussians: Gaussians with N * K entries, the K Gaussians of an anchor being contiguous.
            base_xyz: (B, N, 3) refined anchor points.
        """
        x = x.float()
        B, N, _ = x.shape
        K = self.K

        x = self.norm(x)
        base_xyz = base_xyz.float()
        if self.anchor_coord_head is not None:
            base_xyz = base_xyz + self.anchor_coord_head(x) * self.base_xyz_scale_mult

        geo = self.geometry_head(x).view(B, N * K, 8)
        rgb = self.rgb_head(x).view(B, N * K, -1)
        xyz_delta = self.coord_head(x).view(B, N, K, 3)
        if self.offset_bound > 0:
            xyz_delta = torch.tanh(xyz_delta * self.xyz_scale_mult) * self.offset_bound
        else:
            xyz_delta = xyz_delta * self.xyz_scale_mult
        xyz = (base_xyz.unsqueeze(2) + xyz_delta).reshape(B, N * K, 3)

        g = Gaussians()
        g.xyz = xyz
        g.rotation = F.normalize(geo[..., :4], dim=-1, eps=1e-8)
        g.scale = self._apply_scale_activation(geo[..., 4:7])
        g.opacity = torch.sigmoid(geo[..., 7] + self.opacity_bias)
        g.feature = rgb

        return g, base_xyz

    def _apply_scale_activation(self, scale_raw: torch.Tensor) -> torch.Tensor:
        _, s_max = self.scale_range
        biased = scale_raw + self.scale_bias
        return torch.exp(biased.clamp(max=math.log(s_max))) * self.scale_mult
