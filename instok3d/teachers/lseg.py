"""Frozen LSeg teacher for semantic feature distillation.

Adapted from Uni3R (https://github.com/HorizonRobotics/Uni3R); the network definition is the
vendored lang-seg code in ``external/lang_seg``.
"""

import contextlib
import os

import clip
import torch

from external.lang_seg.modules.models.lseg_net import LSegNet
from instok3d.utils.freeze import convert_to_buffer
from instok3d.utils.weights import subset_state_dict


@contextlib.contextmanager
def _clip_download_root(root: str):
    """Make ``clip.load`` read and store its weights under ``root`` inside the context."""
    os.makedirs(root, exist_ok=True)
    download = clip.clip._download
    clip.clip._download = lambda url, _root=None: download(url, root)
    try:
        yield
    finally:
        clip.clip._download = download


def _forward_layers(pretrained, x):
    """Run the ViT on ``x`` (B, 3, H, W) and return its four hooked activations as feature maps."""
    h, w = x.shape[-2:]
    pretrained.model.forward_flex(x)
    patch_size = pretrained.model.patch_size
    grid = torch.Size([h // patch_size[1], w // patch_size[0]])

    layers = []
    for i in range(1, 5):
        layer = getattr(pretrained, f'act_postprocess{i}')[0:2](pretrained.activations[str(i)])
        if layer.ndim == 3:
            layer = layer.unflatten(2, grid)
        layers.append(layer)
    return layers


class LSegTeacher(LSegNet):
    """LSeg image encoder and text-driven classifier, always frozen and in eval mode.

    Args:
        half_res: return features at half the input resolution (and let ``decode_feature``
            upsample its logits by 2) instead of full-resolution features.
        pretrained_path: lang-seg checkpoint (``demo_e200.ckpt``).
        clip_cache_dir: directory of the CLIP weights (downloaded there if missing).
    """

    def __init__(
        self,
        half_res: bool = True,
        pretrained_path: str = 'pretrained/lseg/demo_e200.ckpt',
        clip_cache_dir: str = 'pretrained/clip',
    ):
        with _clip_download_root(clip_cache_dir):
            super().__init__(
                labels='',
                backbone='clip_vitl16_384',
                features=256,
                crop_size=224,
                arch_option=0,
                block_depth=0,
                activation='lrelu',
            )
        self.half_res = bool(half_res)
        self._pretrained_path = str(pretrained_path)
        self._weights_loaded = False
        self.requires_grad_(False)
        super().train(False)

    @torch.no_grad()
    def extract_features(self, x):
        """Encode images ``x`` (B, 3, H, W) in [0, 1] into (B, 512, H/2, W/2) features
        (``half_res``) or (B, 512, H, W) features."""
        layer_1, layer_2, layer_3, layer_4 = _forward_layers(self.pretrained, x)

        pretrained = self.pretrained
        layer_1 = pretrained.act_postprocess1[3:len(pretrained.act_postprocess1)](layer_1)
        layer_2 = pretrained.act_postprocess2[3:len(pretrained.act_postprocess2)](layer_2)
        layer_3 = pretrained.act_postprocess3[3:len(pretrained.act_postprocess3)](layer_3)
        layer_4 = pretrained.act_postprocess4[3:len(pretrained.act_postprocess4)](layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        image_features = self.scratch.head1(path_1)
        if self.half_res:
            return image_features
        return self.scratch.output_conv(image_features)

    @torch.no_grad()
    def decode_feature(self, image_features, labelset):
        """Classify features (B, 512, H, W) against the text labels ``labelset``.

        Returns logits (B, len(labelset), H', W'), upsampled by 2 with ``half_res``.
        """
        imshape = image_features.shape

        text = clip.tokenize(labelset).to(image_features.device)
        self.logit_scale = self.logit_scale.to(image_features.device)
        text_features = self.clip_pretrained.encode_text(text)
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale * image_features.half() @ text_features.t()
        out = logits.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        if self.half_res:
            out = self.scratch.output_conv(out)
        return out

    def load_pretrained_weights(self) -> bool:
        """Load the checkpoint and turn all weights into non-persistent buffers."""
        if self._weights_loaded:
            return True
        if not os.path.isfile(self._pretrained_path):
            raise FileNotFoundError(f'LSeg checkpoint not found: {self._pretrained_path}')
        print(f'loading LSeg weights from {self._pretrained_path}')
        # The checkpoint is a full Lightning checkpoint, which cannot be read with weights_only.
        ckpt = torch.load(self._pretrained_path, map_location='cpu', weights_only=False)
        self.load_state_dict(subset_state_dict(ckpt['state_dict'], 'net.'), strict=True)
        convert_to_buffer(self, persistent=False)
        self._weights_loaded = True
        return True

    def train(self, mode: bool = True):
        return super().train(False)
