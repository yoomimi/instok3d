"""Image-to-anchor semantic decoder that reuses the attention maps of the frozen tokenizer."""

from torch import nn
from torch.nn import functional as F
from einops import rearrange

from instok3d.modules.transformer import FFN, TransformerParams


class InstillLayer(nn.Module):
    """Attends with the tokenizer's cross-attention queries / keys and semantic values."""

    def __init__(self, params: TransformerParams, sem_dim: int):
        super().__init__()
        D = params.embed_dim
        self.num_heads = params.num_heads
        self.head_dim = D // params.num_heads
        self.to_v_sem = nn.Linear(sem_dim, D, bias=False)
        self.norm_ffn = nn.LayerNorm(D)
        self.ffn = FFN(
            d_model=D,
            mlp_ratio=params.mlp_ratio,
            dropout=params.dropout,
        )

    def forward(self, anchor_sem, patch_sem, q_recon, k_recon):
        """
        Args:
            anchor_sem: (B, K, D) semantic anchor tokens.
            patch_sem: (B, M, sem_dim) semantic features of the image patches.
            q_recon: (B, num_heads, K, head_dim) queries of the tokenizer layer.
            k_recon: (B, num_heads, M, head_dim) keys of the tokenizer layer.
        """
        v = self.to_v_sem(patch_sem)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)
        attn = F.scaled_dot_product_attention(q_recon, k_recon, v)
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        anchor_sem = anchor_sem + attn
        anchor_sem = anchor_sem + self.ffn(self.norm_ffn(anchor_sem))
        return anchor_sem


class InstillEncoder(nn.Module):
    """Stack of ``InstillLayer``, one per layer of the tokenizer's image-anchor decoder."""

    def __init__(self, params: TransformerParams, num_layers: int, sem_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            InstillLayer(params, sem_dim) for _ in range(num_layers)
        ])

    def forward(self, anchor_sem, patch_sem, qk_list):
        """``qk_list`` holds the (queries, keys) of every tokenizer layer, in order."""
        if len(qk_list) != len(self.layers):
            raise ValueError(
                f'got {len(qk_list)} (query, key) pairs for {len(self.layers)} layers'
            )
        for layer, (q_i, k_i) in zip(self.layers, qk_list):
            anchor_sem = layer(anchor_sem, patch_sem, q_i, k_i)
        return anchor_sem
