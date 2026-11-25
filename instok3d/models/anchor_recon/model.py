import math
from typing import Tuple

import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

from instok3d.backbones import BackboneOutput
from instok3d.modules.transformer import (
    TransformerParams, init_embedding, init_model_weights,
)
from instok3d.utils.sparse import fps_with_mask

from .encoder import CrossAttentionEncoder
from .projector import GaussianProjector
from .renderer import GaussianRenderer
from .gaussians import Gaussians


# Parameter-name prefixes of the grouping head (group tokens, anchor-group decoder, assignment).
_INSTANCE_HEAD_PREFIXES = (
    'instance_query',
    'instance_encoder.',
    'anchor_assign_proj.',
    'query_assign_proj.',
    'logit_scale',
)


class AnchorReconstructionModel(nn.Module):
    """Instance-structured 3D tokenizer.

    K anchor tokens are sampled on the backbone point map and refined by the image-anchor
    decoder (``encoder``); the Gaussian projector decodes each of them into N_g Gaussians.
    With ``use_instance_queries``, L group tokens are decoded from the anchor tokens by the
    anchor-group decoder (``instance_encoder``) and every anchor is assigned to one of the
    groups or to void by a softmax over the L groups and a zero-logit void channel.
    """

    def __init__(self,
                 patch_size: int = 16,
                 embed_dim: int = 1024,
                 backbone_feature_dim: int = 2048,
                 dino_feature_dim: int = 1024,
                 num_anchor_points: int = 1024,
                 num_encoder_layers: int = 6,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 use_instance_queries: bool = True,
                 num_instance_queries: int = 100,
                 num_instance_layers: int = 4,
                 assignment_dim: int = 384,
                 freeze_non_instance_modules: bool = False,
                 sh_degree: int = 2,
                 scale_range: Tuple[float, float] = (0.0, 0.3),
                 num_gaussians_per_anchor: int = 32,
                 opacity_bias: float = -2.0,
                 scale_bias: float = -4.0,
                 scale_mult: float = 1.0,
                 xyz_scale_mult: float = 1.0,
                 offset_bound: float = 0.1,
                 base_xyz_scale_mult: float = 0.1,
                 near_far: Tuple[float, float] = (0.01, 100.0),
                 eps2d: float = 0.3,
                 rendering_resolution: int = 256,
                 ):
        """
        Args:
            patch_size: patch size of the backbone.
            backbone_feature_dim: channels of the backbone patch features.
            dino_feature_dim: channels of the backbone's patch-embedding (DINO) features.
            num_anchor_points: number of anchor tokens K.
            num_encoder_layers: layers of the image-anchor decoder.
            use_instance_queries: build the grouping head (off for reconstruction-only training).
            num_instance_queries: number of group tokens L.
            num_instance_layers: layers of the anchor-group decoder.
            assignment_dim: dimension of the anchor-to-group similarity space.
            freeze_non_instance_modules: train the grouping head only.
            num_gaussians_per_anchor: number of Gaussians N_g per anchor.
            See ``GaussianProjector`` and ``GaussianRenderer`` for the remaining arguments.
        """
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.num_anchor_points = int(num_anchor_points)
        self.rendering_resolution = int(rendering_resolution)
        self.use_instance_queries = bool(use_instance_queries)
        self.num_instance_queries = int(num_instance_queries)

        patch_dim = 3 * self.patch_size * self.patch_size

        self.backbone_proj = nn.Sequential(
            nn.Linear(backbone_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.appearance_proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.xyz_patch_proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.anchor_xyz_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        transformer_params = TransformerParams(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.encoder = CrossAttentionEncoder(transformer_params, num_encoder_layers)

        if self.use_instance_queries:
            query_shape = (1, self.num_instance_queries, embed_dim)
            self.instance_query = nn.Parameter(torch.empty(query_shape))
            self.instance_query_pos = nn.Parameter(torch.empty(query_shape))
            self.instance_encoder = CrossAttentionEncoder(transformer_params, num_instance_layers)
            self.anchor_assign_proj = nn.Linear(embed_dim, assignment_dim)
            self.query_assign_proj = nn.Linear(embed_dim, assignment_dim)
            # Learnable scale of the cosine-similarity logits, initialised to 1 / 0.07.
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

        self.projector = GaussianProjector(
            embed_dim=embed_dim,
            sh_degree=sh_degree,
            scale_range=scale_range,
            num_gaussians_per_anchor=num_gaussians_per_anchor,
            opacity_bias=opacity_bias,
            scale_bias=scale_bias,
            scale_mult=scale_mult,
            xyz_scale_mult=xyz_scale_mult,
            offset_bound=offset_bound,
            base_xyz_scale_mult=base_xyz_scale_mult,
        )

        self.renderer = GaussianRenderer(
            sh_degree=sh_degree,
            near_far=near_far,
            eps2d=eps2d,
        )

        self.reset_parameters()

        if freeze_non_instance_modules:
            if not self.use_instance_queries:
                raise ValueError('freeze_non_instance_modules requires use_instance_queries.')
            for name, p in self.named_parameters():
                if not name.startswith(_INSTANCE_HEAD_PREFIXES):
                    p.requires_grad = False

    def seg_head_zero_residual(self) -> torch.Tensor:
        """Zero-valued scalar that depends on every trainable grouping-head parameter.

        Added to the loss on steps without a grouping loss so that these parameters still
        receive a (zero) gradient, as distributed training requires.
        """
        out = None
        if self.use_instance_queries:
            for name, p in self.named_parameters():
                if not name.startswith(_INSTANCE_HEAD_PREFIXES) or not p.requires_grad:
                    continue
                term = p.sum() * 0.0
                out = term if out is None else out + term
        if out is None:
            return next(self.parameters()).new_zeros(())
        return out

    def reset_parameters(self):
        init_model_weights(self)
        self.projector.reset_parameters()
        if self.use_instance_queries:
            scale = self.embed_dim ** -0.5
            init_embedding(self.instance_query, scale)
            init_embedding(self.instance_query_pos, scale)

    def _prepare_context(self, images: torch.Tensor, bb_outputs: BackboneOutput):
        """Fuse the per-patch inputs into context tokens, one anchor candidate per patch.

        Args:
            images: (B, V, 3, H, W)
            bb_outputs: backbone output with a patch grid of (h, w) per view.
        Returns:
            context: (B, V*h*w, D) context tokens.
            candidate_xyz: (B, V*h*w, 3) point-map samples at the patch centres.
        """
        patch_feats = bb_outputs.patch_features  # (B, V, C, h, w)
        point_map = bb_outputs.point_map         # (B, V, H, W, 3)
        B, V = patch_feats.shape[:2]
        H, W = point_map.shape[2], point_map.shape[3]
        ps = self.patch_size

        if (images.shape[-2], images.shape[-1]) != (H, W):
            images = F.interpolate(
                images.flatten(0, 1), size=(H, W),
                mode='bilinear', align_corners=False,
            ).view(B, V, 3, H, W)

        context = self.backbone_proj(rearrange(patch_feats, 'b v c h w -> b (v h w) c'))
        context = context + self.dino_proj(
            rearrange(bb_outputs.patch_features_dino_raw, 'b v c h w -> b (v h w) c'),
        )
        context = context + self.appearance_proj(rearrange(
            images, 'b v c (h ps1) (w ps2) -> b (v h w) (c ps1 ps2)', ps1=ps, ps2=ps,
        ))
        context = context + self.xyz_patch_proj(rearrange(
            point_map, 'b v (h ps1) (w ps2) c -> b (v h w) (c ps1 ps2)', ps1=ps, ps2=ps,
        ))

        stride = bb_outputs.patch_stride
        candidate_xyz = point_map[:, :, stride // 2::stride, stride // 2::stride, :]
        candidate_xyz = rearrange(candidate_xyz, 'b v h w c -> b (v h w) c')

        return context, candidate_xyz

    def _init_anchors(self, candidate_xyz: torch.Tensor, context: torch.Tensor):
        """Pick the anchors among the finite candidates by farthest point sampling.

        Returns the positions (B, K, 3) and the context tokens (B, K, D) of the picked candidates.
        """
        valid_mask = torch.isfinite(candidate_xyz).all(dim=-1)
        xyz_feat = torch.cat([
            candidate_xyz.float(),
            context.float(),
        ], dim=-1)
        # With fewer finite candidates than anchors, the sampler pads with repeats of the first candidate.
        out = fps_with_mask(xyz_feat, valid_mask, self.num_anchor_points)
        return out[..., :3], out[..., 3:]

    def _run_instance_query_head(self, anchor_features: torch.Tensor) -> dict:
        """Decode the group tokens and assign every anchor to a group or to void.

        Args:
            anchor_features: (B, K, D) anchor tokens.
        Returns:
            dict with ``instance_tokens`` (B, L, D), ``assignment_logits`` (B, K, L),
            ``anchor_to_query_probs`` (B, K, L), ``anchor_to_void_prob`` (B, K) and
            ``query_mass`` (B, L), the mean assignment probability of each group.
        """
        B, N_anchor, _ = anchor_features.shape

        anchor_proj = F.normalize(self.anchor_assign_proj(anchor_features), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        instance_tokens = self.instance_encoder(
            self.instance_query.expand(B, -1, -1), anchor_features,
            query_pos=self.instance_query_pos.expand(B, -1, -1),
        )

        query_proj = F.normalize(self.query_assign_proj(instance_tokens), dim=-1)
        assignment_logits = torch.einsum('bnd,bqd->bnq', anchor_proj, query_proj) * logit_scale

        # The void channel has a fixed logit of zero.
        void_logits = assignment_logits.new_zeros(B, N_anchor, 1)
        probs = torch.softmax(torch.cat([assignment_logits, void_logits], dim=-1), dim=-1)
        anchor_to_query_probs = probs[..., :-1]

        return {
            'instance_tokens': instance_tokens,
            'assignment_logits': assignment_logits,
            'anchor_to_query_probs': anchor_to_query_probs,
            'anchor_to_void_prob': probs[..., -1],
            'query_mass': anchor_to_query_probs.mean(dim=1),
        }

    def forward(self, images: torch.Tensor, bb_outputs: BackboneOutput):
        """Tokenize a set of views.

        Args:
            images: (B, V, 3, H, W)
            bb_outputs: backbone output for the same views.
        Returns:
            gaussians: Gaussians with K * N_g entries per batch element.
            aux: ``anchor_xyz`` (B, K, 3), ``anchor_features`` (B, K, D) and, with the grouping
                head, the outputs of ``_run_instance_query_head`` plus the assignment repeated
                for the Gaussians of each anchor: ``gaussian_query_probs`` (B, K*N_g, L) and
                ``gaussian_void_prob`` (B, K*N_g).
        """
        B = images.shape[0]

        context, candidate_xyz = self._prepare_context(images, bb_outputs)
        anchor_xyz, anchor_feat = self._init_anchors(candidate_xyz, context)

        anchor_tokens = anchor_feat + self.anchor_xyz_proj(anchor_xyz)
        anchor_features = self.encoder(anchor_tokens, context)
        gaussians, anchor_xyz = self.projector(anchor_features, base_xyz=anchor_xyz)

        aux = {
            'anchor_xyz': anchor_xyz,
            'anchor_features': anchor_features,
        }

        if self.use_instance_queries:
            aux.update(self._run_instance_query_head(anchor_features))

            K = int(self.projector.K)
            N_anchor = anchor_features.shape[1]
            Q = self.num_instance_queries
            aux['gaussian_query_probs'] = (
                aux['anchor_to_query_probs'].unsqueeze(2)
                .expand(B, N_anchor, K, Q)
                .reshape(B, N_anchor * K, Q)
            )
            aux['gaussian_void_prob'] = (
                aux['anchor_to_void_prob'].unsqueeze(2)
                .expand(B, N_anchor, K)
                .reshape(B, N_anchor * K)
            )

        return gaussians, aux

    def decode_images(self, gaussians: Gaussians, cams: torch.Tensor, resolution: int = None):
        """Render the Gaussians from ``cams`` (B, V, 5, 4); see ``GaussianRenderer.forward``."""
        if resolution is None:
            resolution = self.rendering_resolution
        return self.renderer(gaussians, cams, resolution)
