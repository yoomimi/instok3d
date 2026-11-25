"""VGGT backbone with a resampled patch size and one intrinsics token per view.

The intrinsics conditioning follows Uni3R (https://github.com/HorizonRobotics/Uni3R).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from external.vggt.heads.dpt_head import DPTHead
from external.vggt.models.aggregator_intr import Aggregator_IntrEmb

from instok3d.backbones import BackboneOutput
from instok3d.backbones.weight_modify import checkpoint_filter_fn
from instok3d.utils.cam import dense_intrinsics_to_params
from instok3d.utils.freeze import convert_to_buffer
from instok3d.utils.weights import load_safetensors_state_dict

_HEAD_MODULES = {'point': 'point_head'}
_UNUSED_VGGT_HEADS = ('camera_head.', 'depth_head.', 'track_head.')
_INTRINSIC_ENCODER = 'aggregator.intrinsic_encoder.'


class VGGTIntrBackbone(nn.Module):
    """VGGT-1B conditioned on the camera intrinsics; predicts a point map in the frame of the first view.

    Args:
        repo_id: Hugging Face repository of the VGGT weights fine-tuning starts from.
        cache_dir: Hugging Face cache directory (default cache when None).
        freeze: keep all weights fixed; they are then loaded from ``finetuned_weights_path``.
        input_hw: resolution the input views are resized to before the backbone.
        finetuning_heads: prediction heads to fine-tune (``'point'``).
        finetuning_num_layers: number of last aggregator block pairs to fine-tune;
            negative for the whole aggregator, 0 for none.
        finetuned_weights_path: ``.safetensors`` file of a fine-tuned backbone.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 1024,
        repo_id: str = 'facebook/VGGT-1B',
        cache_dir: Optional[str] = None,
        freeze: bool = False,
        input_hw: Optional[List[int]] = None,
        finetuning_heads: Optional[List[str]] = None,
        finetuning_num_layers: int = 0,
        finetuned_weights_path: Optional[str] = None,
    ):
        super().__init__()
        finetuning_heads = list(finetuning_heads) if finetuning_heads else None
        unknown = set(finetuning_heads or []) - set(_HEAD_MODULES)
        if unknown:
            raise ValueError(
                f'Unknown finetuning_heads {sorted(unknown)}; available: {sorted(_HEAD_MODULES)}.'
            )
        is_finetuned = (finetuning_heads is not None) or (finetuning_num_layers != 0)
        if freeze == is_finetuned:
            raise ValueError('Set either freeze=True or finetuning_heads / finetuning_num_layers.')
        if freeze and finetuned_weights_path is None:
            raise ValueError(
                'freeze=True requires finetuned_weights_path: VGGT-1B has no intrinsics encoder.'
            )

        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.freeze = bool(freeze)
        self.patch_size = int(patch_size)
        self.input_hw = list(input_hw) if input_hw is not None else None
        self.finetuning_heads = finetuning_heads
        self.finetuning_num_layers = int(finetuning_num_layers)
        self.finetuned_weights_path = finetuned_weights_path or None
        self._weights_loaded = False

        self.aggregator = Aggregator_IntrEmb(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            output_dim=4,
            activation='inv_log',
            conf_activation='expp1',
        )

    def load_pretrained_weights(self) -> bool:
        """Load the weights and apply the freezing policy (called by the engine before a run)."""
        if self._weights_loaded:
            return True

        if self.finetuned_weights_path is not None:
            print(f'Loading backbone weights from {self.finetuned_weights_path}.')
            self.load_state_dict(self._read_state_dict(self.finetuned_weights_path))
        else:
            print(f'Loading backbone weights from {self.repo_id}.')
            weights_path = hf_hub_download(
                self.repo_id, filename='model.safetensors', cache_dir=self.cache_dir,
            )
            state_dict = checkpoint_filter_fn(self._read_state_dict(weights_path), self)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            # VGGT has no intrinsics encoder: it keeps its initialization.
            missing = [key for key in missing if not key.startswith(_INTRINSIC_ENCODER)]
            if missing or unexpected:
                raise RuntimeError(f'Incompatible VGGT weights: missing {missing}, unexpected {unexpected}.')

        if self.freeze:
            # Non-persistent buffers keep the frozen backbone out of checkpoints and optimizers.
            convert_to_buffer(self, persistent=False)
        else:
            self._set_trainable_parameters()
        self._weights_loaded = True
        return True

    @staticmethod
    def _read_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
        state_dict = load_safetensors_state_dict(weights_path)
        return {key: value for key, value in state_dict.items() if not key.startswith(_UNUSED_VGGT_HEADS)}

    def _set_trainable_parameters(self) -> None:
        self.requires_grad_(False)
        for head in self.finetuning_heads or []:
            getattr(self, _HEAD_MODULES[head]).requires_grad_(True)

        n = self.finetuning_num_layers
        if n < 0:
            self.aggregator.requires_grad_(True)
            # The mask token is never used in the forward pass.
            self.aggregator.patch_embed.mask_token.requires_grad_(False)
        elif n > 0:
            self.aggregator.frame_blocks[-n:].requires_grad_(True)
            self.aggregator.global_blocks[-n:].requires_grad_(True)
        self.aggregator.intrinsic_encoder.requires_grad_(True)

        num_trainable = sum(p.numel() for p in self.trainable_parameters())
        num_total = sum(p.numel() for p in self.parameters())
        print(f'Backbone fine-tuning: {num_trainable:,} of {num_total:,} parameters are trainable.')

    def trainable_parameters(self):
        """Iterate over the parameters that are fine-tuned (none when frozen)."""
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter

    def train(self, mode: bool = True) -> 'VGGTIntrBackbone':
        """Set the training mode, keeping every sub-module that is not fine-tuned in eval mode."""
        if self.freeze:
            return super().train(False)

        super().train(mode)
        for head, module_name in _HEAD_MODULES.items():
            if head not in (self.finetuning_heads or []):
                getattr(self, module_name).eval()

        n = self.finetuning_num_layers
        if n >= 0:
            self.aggregator.eval()
        if n > 0:
            self.aggregator.frame_blocks[-n:].train(mode)
            self.aggregator.global_blocks[-n:].train(mode)
        return self

    def forward(self, images: torch.Tensor, fxfycxcy: torch.Tensor) -> BackboneOutput:
        """
        Args:
            images: (B, V, 3, H, W) RGB in [0, 1].
            fxfycxcy: (B, V, 4) intrinsics normalized by the image size.
        """
        B, V, _, H, W = images.shape
        h, w = H // self.patch_size, W // self.patch_size

        intrinsics = dense_intrinsics_to_params(fxfycxcy.to(images.dtype))
        aggregated_tokens_list, patch_tokens, patch_start_idx = self.aggregator(images, intrinsics)
        # The heads expect the VGGT token layout, without the trailing intrinsics token.
        aggregated_tokens_list = [tokens[:, :, :-1, :] for tokens in aggregated_tokens_list]

        with torch.autocast(device_type='cuda', enabled=False):
            point_map, _ = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
            )

        patch_features = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
        patch_features = patch_features.reshape(B, V, h, w, -1).permute(0, 1, 4, 2, 3).contiguous()
        dino_features = patch_tokens[:, :-1, :]
        dino_features = dino_features.reshape(B, V, h, w, -1).permute(0, 1, 4, 2, 3).contiguous()

        return BackboneOutput(
            point_map=point_map,
            patch_features=patch_features,
            patch_stride=self.patch_size,
            patch_features_dino_raw=dino_features,
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=int(patch_start_idx),
        )
