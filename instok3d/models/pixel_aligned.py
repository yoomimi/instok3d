import math
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instok3d.backbones import BackboneOutput
from instok3d.models.anchor_recon.gaussians import Gaussians
from instok3d.models.anchor_recon.renderer import GaussianRenderer
from instok3d.utils.gaussian_scale import apply_knn_rescale, compute_knn_ctx
from external.vggt.heads.dpt_head import DPTHead


class PixelAlignedReconstructionModel(nn.Module):
    """Per-pixel Gaussian head on the backbone tokens, used to fine-tune the backbone.

    One Gaussian is predicted for every pixel of the backbone point map; its scale is relative
    to the distance to the neighbouring points.
    """

    def __init__(
            self,
            backbone_feature_dim: int = 2048,
            patch_size: int = 16,

            dpt_features: int = 256,
            dpt_intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),

            gs_attr_hidden: int = 256,

            sh_degree: int = 0,
            scale_range: Tuple[float, float] = (0.0, 22000.0),
            scale_bias: float = 0.0,
            scale_mult: float = 0.0025,
            opacity_bias: float = 0.0,

            knn_rescale_k: int = 3,
            knn_rescale_clamp_min: float = 0.1,
            knn_rescale_clamp_max: float = 3.0,

            near_far: Tuple[float, float] = (0.01, 100.0),
            eps2d: float = 0.3,
            rendering_resolution: int = 256,
    ):
        """
        Args:
            dpt_intermediate_layer_idx: backbone blocks whose tokens feed the DPT head.
            scale_range: (min, max) scale before ``scale_mult``; only the maximum is enforced.
            knn_rescale_k: number of neighbours that define the local point spacing.
            knn_rescale_clamp_min: lower bound of the final scale, in units of the median spacing.
            knn_rescale_clamp_max: upper bound of the final scale, in units of the median spacing.
        """
        super().__init__()
        assert sh_degree >= 0

        self.sh_degree = int(sh_degree)
        self.scale_range = tuple(scale_range)
        self.scale_bias = float(scale_bias)
        self.scale_mult = float(scale_mult)
        self.opacity_bias = float(opacity_bias)
        self.knn_rescale_k = int(knn_rescale_k)
        self.knn_rescale_clamp_min = float(knn_rescale_clamp_min)
        self.knn_rescale_clamp_max = float(knn_rescale_clamp_max)
        self.rendering_resolution = int(rendering_resolution)

        d_sh = (self.sh_degree + 1) ** 2
        # scale (3) + rotation (4) + opacity (1) + SH colour (3 * d_sh)
        self.attr_dim = 3 + 4 + 1 + 3 * d_sh

        self.dpt_head = DPTHead(
            dim_in=backbone_feature_dim,
            patch_size=patch_size,
            features=dpt_features,
            intermediate_layer_idx=list(dpt_intermediate_layer_idx),
            pos_embed=True,
            feature_only=True,
        )
        self.gs_attr_proj = nn.Sequential(
            nn.Conv2d(dpt_features, gs_attr_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gs_attr_hidden, self.attr_dim, kernel_size=1),
        )
        self.renderer = GaussianRenderer(
            sh_degree=self.sh_degree,
            near_far=tuple(near_far),
            eps2d=float(eps2d),
        )

    def forward(self, images: torch.Tensor,
                bb_outputs: BackboneOutput) -> Tuple[Gaussians, Dict[str, Any]]:
        """Predict V*H*W Gaussians from ``images`` (B, V, 3, H, W) and the backbone output."""
        # The DPT head derives the patch grid from the image size, which must be the backbone's.
        H_bb, W_bb = bb_outputs.point_map.shape[2], bb_outputs.point_map.shape[3]
        if (images.shape[-2], images.shape[-1]) != (H_bb, W_bb):
            B_img, V_img = images.shape[:2]
            images = F.interpolate(
                images.flatten(0, 1), size=(H_bb, W_bb),
                mode='bilinear', align_corners=False,
            ).view(B_img, V_img, 3, H_bb, W_bb)

        feats = self.dpt_head(
            bb_outputs.aggregated_tokens_list,
            images=images,
            patch_start_idx=bb_outputs.patch_start_idx,
        )                                                          # (B, V, dpt_features, H, W)

        B, V, _, H, W = feats.shape
        gs_attr = self.gs_attr_proj(feats.flatten(0, 1))           # (B*V, A, H, W)
        gs_attr = gs_attr.reshape(B, V, self.attr_dim, H, W) \
                         .permute(0, 1, 3, 4, 2)                   # (B, V, H, W, A)

        # The renderer runs in float32 with autocast disabled.
        gs_attr = gs_attr.float()

        d_sh = (self.sh_degree + 1) ** 2
        scale_raw, rot_raw, opa_raw, sh_raw = torch.split(
            gs_attr, [3, 4, 1, 3 * d_sh], dim=-1,
        )

        N = V * H * W
        means = bb_outputs.point_map.float().reshape(B, N, 3)
        knn_ctx = compute_knn_ctx(means, k=self.knn_rescale_k)

        gaussians = Gaussians()
        gaussians.xyz = means
        gaussians.rotation = F.normalize(rot_raw.reshape(B, N, 4), dim=-1, eps=1e-8)
        gaussians.scale = self._scale_activation(scale_raw.reshape(B, N, 3), knn_ctx)
        gaussians.opacity = torch.sigmoid(opa_raw.reshape(B, N) + self.opacity_bias)
        gaussians.feature = sh_raw.reshape(B, N, 3 * d_sh)
        return gaussians, {}

    def decode_images(self, gaussians: Gaussians,
                      cams: torch.Tensor,
                      resolution: int = None):
        """Render the Gaussians from ``cams`` (B, V, 5, 4); see ``GaussianRenderer.forward``."""
        if resolution is None:
            resolution = self.rendering_resolution
        return self.renderer(gaussians, cams, resolution)

    def _scale_activation(self, raw: torch.Tensor,
                          knn_ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, s_max = self.scale_range
        biased = raw + self.scale_bias
        scale = torch.exp(biased.clamp(max=math.log(s_max))) * self.scale_mult
        return apply_knn_rescale(
            scale, knn_ctx['nn_dist'], knn_ctx['median_dist'],
            self.knn_rescale_clamp_min, self.knn_rescale_clamp_max,
        )
