# Third-party notices

InsTok3D is released under CC BY-NC 4.0 (see `LICENSE`). The components below are
redistributed or adapted from other projects and remain under their own licenses.

## Vendored code (`external/`)

| Path | Upstream | License |
|---|---|---|
| `external/vggt/` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt): aggregator, heads, layers and pose utilities. `models/aggregator_intr.py` adds an intrinsics token per view, following Uni3R. | VGGT License (`external/vggt/LICENSE.txt`) |
| `external/lang_seg/` | [isl-org/lang-seg](https://github.com/isl-org/lang-seg): LSeg network definition (minimal subset). | MIT (`external/lang_seg/LICENSE`) |
| `external/pointops/` | [Silverster98/pointops](https://github.com/Silverster98/pointops): farthest point sampling and kNN CUDA operators, originally by Hengshuang Zhao. | see the upstream repository |

## Files adapted from other projects

| Path | Adapted from | License of the original |
|---|---|---|
| `instok3d/backbones/weight_modify.py` | [Uni3R](https://github.com/HorizonRobotics/Uni3R) (patch-embedding resampling 14 -> 16) | CC BY-NC-SA 4.0 |
| `instok3d/teachers/lseg.py` | Uni3R (LSeg feature extractor wrapper) | CC BY-NC-SA 4.0 |
| `crop_resize_dust3r` in `instok3d/data/scannet_lsm_test.py` | [DUSt3R](https://github.com/naver/dust3r) dataset cropping | CC BY-NC-SA 4.0 |
| `align_target_cams` in `instok3d/evaluators/unified_vis.py` | test-time pose alignment of the Uni3R evaluation | CC BY-NC-SA 4.0 |
| `instok3d/models/feature_lifting/instill_encoder.py` | follows the semantic-token design of [C3G](https://github.com/cvlab-kaist/C3G) (re-implemented) | MIT |

Files adapted from CC BY-NC-SA 4.0 sources remain available under CC BY-NC-SA 4.0.

## Model weights

- The backbone is initialised from [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B);
  the fine-tuned backbones and every model built on them are derivatives of those weights
  and are subject to the VGGT License in addition to CC BY-NC 4.0.
- The LSeg weights (`demo_e200.ckpt`) and the OpenAI CLIP weights are downloaded from their
  original sources and are not redistributed here.

## Data files (`assets/`)

| File | Source |
|---|---|
| `assets/evaluation_index_re10k.json` | RealEstate10K evaluation index of [pixelSplat](https://github.com/dcharatan/pixelsplat) / [MVSplat](https://github.com/donydchen/mvsplat) |
| `assets/re10k_valid.json` | list of the RealEstate10K training sequences that are used |
| `assets/scannetv2-labels.combined.tsv` | [ScanNet](https://github.com/ScanNet/ScanNet) label mapping (ScanNet Terms of Use apply) |
| `assets/scannet_invalid_frames.json` | ScanNet frames with invalid camera poses |
| `assets/scannet_test_scenes.txt` | the 40 test scenes of [LSM](https://github.com/NVlabs/LSM) |
