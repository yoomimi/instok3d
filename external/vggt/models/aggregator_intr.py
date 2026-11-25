# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""VGGT aggregator with one intrinsics token per frame.

Follows the intrinsics embedding of Uni3R (https://github.com/HorizonRobotics/Uni3R).
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from external.vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class Aggregator_IntrEmb(Aggregator):
    """Aggregator whose per-frame tokens are ``[camera, register..., patches..., intrinsics]``."""

    def __init__(self, embed_dim: int = 1024, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.intrinsic_encoder = nn.Linear(9, embed_dim)
        nn.init.kaiming_normal_(self.intrinsic_encoder.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.intrinsic_encoder.bias)

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        """
        Args:
            images: [B, S, 3, H, W] in [0, 1].
            intrinsics: [B, S, 3, 3] normalized intrinsic matrices.

        Returns:
            output_list: per block, [B, S, P, 2 * embed_dim] with the intrinsics token last.
            patch_tokens: [B * S, num_patches + 1, embed_dim] patch embeddings and the intrinsics token.
            patch_start_idx: index of the first patch token.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f'Expected 3 input channels, got {C_in}')

        patch_tokens = self.run_patch_embed(images)
        intr_token = self.intrinsic_encoder(intrinsics.reshape(B * S, 9)).unsqueeze(1)
        patch_tokens = torch.cat([patch_tokens, intr_token], dim=1)

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device,
            )
            # The intrinsics token gets a position one step past the last patch.
            extra = pos[:, 0:1, :].clone()
            extra[:, :, 0] = pos[:, -1, 0].unsqueeze(-1) + 1
            pos = torch.cat([pos, extra], dim=1)

        if self.patch_start_idx > 0:
            # Position 0 disables the rotary embedding for the camera and register tokens.
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list: List[torch.Tensor] = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == 'frame':
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos,
                    )
                elif attn_type == 'global':
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos,
                    )
                else:
                    raise ValueError(f'Unknown attention type: {attn_type}')

            for i in range(len(frame_intermediates)):
                output_list.append(
                    torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                )

        return output_list, patch_tokens, self.patch_start_idx
