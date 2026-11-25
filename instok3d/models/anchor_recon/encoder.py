from torch import nn

from instok3d.modules.transformer import TransformerParams, CrossAttentionLayer


class CrossAttentionEncoder(nn.Module):
    """Stack of decoder layers that update query tokens from a fixed set of context tokens."""

    def __init__(self, params: TransformerParams, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=params.embed_dim,
                num_heads=params.num_heads,
                mlp_ratio=params.mlp_ratio,
                dropout=params.dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, context, query_pos=None):
        """``query_pos`` (B, Lq, D) is a positional embedding added to the queries in each layer."""
        for layer in self.layers:
            x = layer(x, context, query_pos=query_pos)
        return x
