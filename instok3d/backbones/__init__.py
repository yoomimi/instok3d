from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BackboneOutput:
    point_map: torch.Tensor                                       # (B, V, H, W, 3), frame of the first view
    patch_features: torch.Tensor                                  # (B, V, C, h, w)
    patch_stride: int
    patch_features_dino_raw: Optional[torch.Tensor] = None        # (B, V, C_dino, h, w)
    aggregated_tokens_list: Optional[List[torch.Tensor]] = None   # per block: (B, V, P, 2 * embed_dim)
    patch_start_idx: Optional[int] = None                         # index of the first patch token
