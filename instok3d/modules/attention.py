from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .norm import RMSNorm


class EfficientFullAttention(nn.Module):
    """Multi-head attention with RMS-normalised queries and keys."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.fc = nn.Linear(dim, dim)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, context=None, *, return_qk=False):
        """Attend from ``x`` to ``context`` (self-attention if ``context`` is None).

        With ``return_qk`` the normalised per-head queries and keys,
        (B, num_heads, L, head_dim) each, are returned as well.
        """
        if context is None:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = (
            rearrange(t, "b l (nh dh) -> b nh l dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh l dh -> b l (nh dh)")
        x = self.fc(out)
        if return_qk:
            return x, (q, k)
        return x
