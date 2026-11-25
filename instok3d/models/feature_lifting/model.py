"""Semantic feature lifting on top of a frozen instance-structured tokenizer."""

import os

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from omegaconf import OmegaConf

from engine import instantiate
from instok3d.modules.transformer import TransformerParams
from instok3d.models.anchor_recon.encoder import CrossAttentionEncoder
from instok3d.models.anchor_recon.assignment import build_hard_assignment
from instok3d.utils.freeze import convert_to_buffer
from instok3d.utils.weights import load_model_state_dict

from .instill_encoder import InstillEncoder

_FEATURE_MODES = ('all', 'group', 'anchor')


class FeatureLiftingModel(nn.Module):
    """Distils 2D semantic features into the tokens of a frozen tokenizer.

    Semantic anchor tokens are decoded from the teacher features of the input views with the
    attention maps of the tokenizer's image-anchor decoder. From them the model predicts a
    ``semantic_dim`` embedding per group (shared by all Gaussians of the group) and an
    ``anchor_low_dim`` residual per anchor, which is rendered and projected to ``semantic_dim``.

    Args:
        recon_model_path: model directory of the tokenizer (``config.yaml`` and its weights).
        semantic_dim: channels of the teacher features.
        anchor_low_dim: channels of the anchor-level residual.
        feature_mode: 'all' (group embedding + anchor residual), 'group' or 'anchor'.
    """

    def __init__(
        self,
        recon_model_path: str,
        semantic_dim: int = 512,
        anchor_low_dim: int = 8,
        feature_mode: str = 'all',
    ):
        super().__init__()

        if feature_mode not in _FEATURE_MODES:
            raise ValueError(f'feature_mode must be one of {_FEATURE_MODES}; got {feature_mode!r}')
        self.feature_mode = feature_mode
        self.use_group = feature_mode in ('all', 'group')
        self.use_anchor = feature_mode in ('all', 'anchor')

        cfg = OmegaConf.load(os.path.join(recon_model_path, 'config.yaml'))
        self.recon_model = instantiate(cfg.model)
        if not self.recon_model.use_instance_queries:
            raise ValueError(f'The tokenizer at {recon_model_path} has no grouping head.')

        self.recon_model_path = str(recon_model_path)
        self._recon_loaded = False

        self.semantic_dim = int(semantic_dim)
        self.anchor_low_dim = int(anchor_low_dim)
        self.embed_dim = int(self.recon_model.embed_dim)
        self.num_instance_queries = int(self.recon_model.num_instance_queries)
        D = self.embed_dim

        encoder_layers = self.recon_model.encoder.layers
        params = TransformerParams(
            embed_dim=D,
            num_heads=int(encoder_layers[0].cross_attn.num_heads),
            mlp_ratio=4.0,
            dropout=0.0,
        )

        self.instill_encoder = InstillEncoder(
            params=params,
            num_layers=len(encoder_layers),
            sem_dim=self.semantic_dim,
        )

        if self.use_group:
            self.instance_decoder = CrossAttentionEncoder(
                params=params,
                num_layers=len(self.recon_model.instance_encoder.layers),
            )
            self.instance_head = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, self.semantic_dim, bias=False),
            )
        else:
            self.instance_decoder = None
            self.instance_head = None

        if self.use_anchor:
            self.anchor_low_head = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, self.anchor_low_dim, bias=False),
            )
            # Zero-initialised: the residual starts at exactly zero.
            self.residual_head = nn.Linear(self.anchor_low_dim, self.semantic_dim, bias=True)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        else:
            self.anchor_low_head = None
            self.residual_head = None

    def load_pretrained_weights(self) -> bool:
        """Load the tokenizer and freeze it as non-persistent buffers (kept out of checkpoints)."""
        if self._recon_loaded:
            return True
        print(f'loading tokenizer weights from {self.recon_model_path}')
        state_dict = load_model_state_dict(self.recon_model_path, is_best=True)
        self.recon_model.load_state_dict(state_dict, strict=True)
        self.recon_model.eval()
        convert_to_buffer(self.recon_model, persistent=False)
        self._recon_loaded = True
        return True

    def train(self, mode: bool = True):
        super().train(mode)
        self.recon_model.eval()
        return self

    @torch.no_grad()
    def _recon_forward_capture(self, images, bb_outputs):
        """Tokenizer forward pass; also returns the cross-attention (queries, keys) per layer."""
        recon = self.recon_model
        B = images.shape[0]

        context, candidate_xyz = recon._prepare_context(images, bb_outputs)
        anchor_xyz, anchor_feat = recon._init_anchors(candidate_xyz, context)

        x = anchor_feat + recon.anchor_xyz_proj(anchor_xyz)
        qk_list = []
        for layer in recon.encoder.layers:
            x, qk = layer(x, context, return_cross_qk=True)
            qk_list.append(qk)
        anchor_features = x

        gaussians, anchor_xyz = recon.projector(anchor_features, base_xyz=anchor_xyz)

        recon_aux = {
            'anchor_xyz': anchor_xyz,
            'anchor_features': anchor_features,
            **recon._run_instance_query_head(anchor_features),
        }

        K = int(recon.projector.K)
        N = anchor_features.shape[1]
        Q = recon.num_instance_queries
        recon_aux['gaussian_query_probs'] = (
            recon_aux['anchor_to_query_probs'].unsqueeze(2).expand(B, N, K, Q).reshape(B, N * K, Q)
        )
        recon_aux['gaussian_void_prob'] = (
            recon_aux['anchor_to_void_prob'].unsqueeze(2).expand(B, N, K).reshape(B, N * K)
        )
        return gaussians, recon_aux, qk_list

    def forward(self, images, bb_outputs, semantic_feature_map):
        """
        Args:
            images: (B, V, 3, H, W) in [0, 1].
            bb_outputs: backbone output for the same views.
            semantic_feature_map: (B, V, semantic_dim, H', W') teacher features of the views.
        Returns:
            gaussians: Gaussians of the frozen tokenizer.
            recon_aux: aux dict of the tokenizer.
            sem_aux: ``anchor_sem`` (B, K, D); with the group branch ``instance_sem`` (B, L, D)
                and ``instance_full`` (B, L, semantic_dim); with the anchor branch
                ``anchor_low`` (B, K, anchor_low_dim).
        """
        gaussians, recon_aux, qk_list = self._recon_forward_capture(images, bb_outputs)

        anchor_features = recon_aux['anchor_features']
        B, N = anchor_features.shape[:2]

        # Teacher features on the backbone patch grid: (B, V*h*w, semantic_dim).
        h, w = bb_outputs.patch_features.shape[-2:]
        sm = rearrange(semantic_feature_map, 'b v c h w -> (b v) c h w')
        sm = F.interpolate(sm, size=(int(h), int(w)), mode='bilinear', align_corners=False)
        patch_sem = rearrange(sm, '(b v) c h w -> b (v h w) c', b=B)

        anchor_sem = torch.zeros(
            B, N, self.embed_dim, device=anchor_features.device, dtype=anchor_features.dtype,
        )
        anchor_sem = self.instill_encoder(
            anchor_sem=anchor_sem, patch_sem=patch_sem, qk_list=qk_list,
        )

        sem_aux = {'anchor_sem': anchor_sem}

        if self.use_group:
            # Start from the tokenizer's group tokens so that the semantic group tokens share
            # their indexing with the anchor-to-group assignment.
            instance_sem = self.instance_decoder(
                self.recon_model.instance_query.expand(B, -1, -1),
                anchor_sem,
                query_pos=self.recon_model.instance_query_pos.expand(B, -1, -1),
            )
            sem_aux['instance_sem'] = instance_sem
            sem_aux['instance_full'] = self.instance_head(instance_sem)

        if self.use_anchor:
            sem_aux['anchor_low'] = self.anchor_low_head(anchor_sem)

        return gaussians, recon_aux, sem_aux

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def decode_features(self, gaussians, recon_aux, sem_aux, cams, resolution: int = 128):
        """Render the semantic feature maps of the views ``cams`` (B, V, 5, 4).

        The hard group assignment and the anchor residuals of the Gaussians are rasterised in
        one pass; each pixel takes the embedding of its dominant group plus the projected residual.

        Returns:
            dict with ``final_sem`` (B, V, semantic_dim, H, W) and, with the group branch (else
            None), ``group_feat`` (B, V, semantic_dim, H, W) and ``pixel_oh`` (B, V, L, H, W),
            the one-hot group of every pixel.
        """
        resolution = int(resolution)

        K = int(self.recon_model.projector.K)
        gaussian_query_probs = recon_aux['gaussian_query_probs']  # (B, K*N_g, L)
        B, NK, Q = gaussian_query_probs.shape

        cols = []
        if self.use_group:
            hard_q, _ = build_hard_assignment(gaussian_query_probs, None)
            cols.append(hard_q.float())
        if self.use_anchor:
            anchor_low = sem_aux['anchor_low']
            N, low = anchor_low.shape[1:]
            anchor_low_per_gauss = (
                anchor_low.unsqueeze(2).expand(B, N, K, low).reshape(B, N * K, low)
            )
            if anchor_low_per_gauss.shape[1] != NK:
                raise RuntimeError(
                    f'{N} anchors x {K} Gaussians do not match the {NK} Gaussians of the scene'
                )
            cols.append(anchor_low_per_gauss.float())
        gauss_colors = torch.cat(cols, dim=-1)

        rendered, _, _ = self.recon_model.renderer._render_gaussians(
            gaussians=gaussians, feature=gauss_colors, sh_degree=None,
            cams=cams.float(), resolution=(resolution, resolution),
            render_mode='RGB',
        )
        splatted = rearrange(rendered, 'b v h w c -> b v c h w')

        pixel_oh = group_feat = residual = None
        offset = 0
        if self.use_group:
            splatted_oh = splatted[:, :, offset:offset + Q]
            offset += Q
            instance_full = sem_aux['instance_full']
            pixel_oh = (
                F.one_hot(splatted_oh.argmax(dim=2), num_classes=Q)
                .permute(0, 1, 4, 2, 3)
                .to(instance_full.dtype)
            )
            group_feat = torch.einsum('bvqhw, bqd -> bvdhw', pixel_oh, instance_full)

        if self.use_anchor:
            splatted_low = splatted[:, :, offset:]
            # The rasteriser returns float32; the head may run in a lower precision.
            head_dtype = next(self.residual_head.parameters()).dtype
            low_pix = rearrange(splatted_low.to(head_dtype), 'b v c h w -> b v h w c')
            residual = rearrange(self.residual_head(low_pix), 'b v h w c -> b v c h w')

        if group_feat is not None and residual is not None:
            final_sem = group_feat + residual.to(group_feat.dtype)
        elif group_feat is not None:
            final_sem = group_feat
        else:
            final_sem = residual

        return {
            'final_sem': final_sem,
            'group_feat': group_feat,
            'pixel_oh': pixel_oh,
        }
