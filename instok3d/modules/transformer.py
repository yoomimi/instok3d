import torch
from torch import nn

from .attention import EfficientFullAttention


class TransformerParams:
    """Hyper-parameters shared by the layers of one transformer stack."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout


class FFN(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionLayer(nn.Module):
    """Pre-norm decoder layer: cross-attention, self-attention, feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm_cross = nn.LayerNorm(embed_dim)
        self.norm_cross_source = nn.LayerNorm(embed_dim)
        self.cross_attn = EfficientFullAttention(dim=embed_dim, num_heads=num_heads)

        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = EfficientFullAttention(dim=embed_dim, num_heads=num_heads)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FFN(
            d_model=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            query_pos: torch.Tensor = None,
            *,
            return_cross_qk: bool = False,
    ):
        """Update queries ``x`` (B, Lq, D) from ``context`` (B, Lc, D).

        ``query_pos`` is added to the normalised queries of both attentions.
        With ``return_cross_qk`` the cross-attention (queries, keys) are returned as well.
        """
        q_in = self.norm_cross(x)
        if query_pos is not None:
            q_in = q_in + query_pos
        cross_out = self.cross_attn(
            q_in,
            context=self.norm_cross_source(context),
            return_qk=return_cross_qk,
        )
        if return_cross_qk:
            cross_out, cross_qk = cross_out
        x = x + cross_out

        s_in = self.norm_self(x)
        if query_pos is not None:
            s_in = s_in + query_pos
        x = x + self.self_attn(s_in)

        x = x + self.ffn(self.norm_ffn(x))
        if return_cross_qk:
            return x, cross_qk
        return x


def init_embedding(embedding, scale):
    nn.init.normal_(embedding)
    embedding.data *= scale


def init_model_weights(module, std=0.02):
    """Initialise every Linear layer with normal(std) weights and zero bias."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
