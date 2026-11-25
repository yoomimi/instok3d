# lang_seg (vendored, minimal)

Minimal vendor of [isl-org/lang-seg](https://github.com/isl-org/lang-seg)
limited to the files required by `instok3d.teachers.lseg.LSegTeacher`.

## What's kept

- `modules/models/lseg_net.py` — `LSegNet` (the body our teacher inherits)
- `modules/models/lseg_blocks.py` — `FeatureFusionBlock`, `_make_encoder`,
  `forward_vit` etc., used by `lseg_net`
- `modules/models/lseg_vit.py` — ViT backbone helpers, used by `lseg_blocks`
- `LICENSE` — upstream MIT license

## What's pruned

Everything not exercised by `LSegTeacher.{__init__, extract_features,
decode_feature, load_pretrained_weights}` — including:

- top-level training/eval scripts (`train_lseg.py`, `test_lseg.py`,
  `lseg_app.py`, `*.sh`, `lseg_demo.ipynb`),
- `modules/lsegmentation_module.py`, `modules/lseg_module.py`, and the
  `_zs` variants (depend on the `encoding` package, unused here),
- `additional_utils/`, `data/`, `inputs/`, `label_files/`,
- `fewshot_data/` (~25 MB of COCO splits + asset images, unused).

## Runtime deps

- `clip` — `pip install git+https://github.com/openai/CLIP.git`
- `timm`, `torch`, `numpy` — standard.
