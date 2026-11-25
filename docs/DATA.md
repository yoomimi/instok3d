# Data preparation

All data is read from `datasets/` in the repository root (symlinks are fine). Images are
used at 256 x 256; the loaders resize on the fly, so the stored resolution only affects I/O.

```
datasets/
├── scannetv2/      ScanNet training / validation scenes (+ instance masks)
├── scannet_test/   40-scene ScanNet test split
├── scannetpp/      ScanNet++ (RGB only)
├── re10k/          RealEstate10K (+ SAM2 pseudo instance masks)
└── mipnerf360/     Mip-NeRF 360 (zero-shot evaluation)
```

Camera conventions: poses are camera-to-world in OpenCV coordinates (x right, y down,
z forward); intrinsics are pinhole matrices in pixels of the stored image.

We do not ship converters from the raw releases of ScanNet / ScanNet++ / RealEstate10K;
any exporter that writes the files described below works.

## ScanNet (training)

```
datasets/scannetv2/
├── gt_h5/<scene>.h5
└── gt_instance_mask/<scene>/<scene>_gt_instance.h5
```

`gt_h5/<scene>.h5`

| Key | Shape / type | Content |
|---|---|---|
| `colors` | `(N, H, W, 3)` uint8 | RGB frames, square (ours: 256 x 256) |
| `intrinsic` | `(3, 3)` or `(4, 4)` float | pinhole intrinsics for the stored `H x W` |
| `extrinsics` | `(N, 4, 4)` float | camera-to-world poses |
| attribute `frame_stride` | int, optional | `s` if the file stores only every `s`-th frame of the sequence (default 1) |

The loader resizes the frames to 256 x 256 without cropping, so store square crops (ours are
cropped around the principal point and resized to 256 x 256). The configs use every 10th
frame of the original sequence (`frame_load_stride: 10`), so a file that stores every 10th
frame (`frame_stride = 10`) is sufficient and much smaller. Frame ids always refer to the
original sequence.

`gt_instance_mask/<scene>/<scene>_gt_instance.h5` holds one dataset per frame, named
`image_{frame_id:05d}`: an integer `(h, w)` instance-id map rendered from the ScanNet
instance annotation, `0` = unlabeled. Masks are resized (nearest) to the training
resolution, so they may be stored at a lower resolution than the images (ours are 224 x 224).

Split: all 1,513 scans of the ScanNet train/val release are sorted by name; the last 50
(`scene0686_01` ... `scene0706_00`) are held out for validation and the remaining 1,463 are
used for training. The 40 test scenes (`assets/scannet_test_scenes.txt`) are part of the
held-out scans. Frames with invalid poses are listed in `assets/scannet_invalid_frames.json`
(`{scene: [frame ids] | "all_invalid"}`).

## ScanNet test split (evaluation)

We use the 40-scene test split and frame selection of [LSM](https://github.com/NVlabs/LSM),
which is also used by Uni3R and other baselines. Follow the LSM instructions to obtain it:

```
datasets/scannet_test/
├── selected_seqs_test.json          {scene: [frame name, ...]}, 30 frames per scene
└── <scene>/
    ├── images/<frame>.jpg
    ├── images/<frame>.npz           camera_pose (4x4 camera-to-world), camera_intrinsics (3x3)
    └── labels/<frame>.png           uint16 raw ScanNet class ids, 0 = void (feature lifting only)
```

Instance metrics additionally need the instance masks of the test scenes in
`datasets/scannetv2/gt_instance_mask/` (same format as above; the frame names of the test
split are the zero-padded ScanNet frame ids).

Protocols (`config/dm/scannet_test_*.yaml`):

- **2 views**: in the sorted frame list of a scene, every index `t` with `t % 8 in {1, 4}`
  is a target and `(t - 1, t + 1)` are the context views: 8 triplets per scene, 320 in
  total; the first triplet of scene `j` has index `8 * j`.
- **8 views**: the first 15 frames of a scene; even positions are the 8 context views,
  odd positions the 7 targets (one sample per scene).
- `crop_mode` selects the pre-processing of the 512 x 384 frames: `pp_square` (largest
  square centred on the principal point, then Lanczos resize) or `dust3r` (the
  [DUSt3R](https://github.com/naver/dust3r) crop used by LSM / Uni3R).
- Semantic labels are mapped to the 8 classes wall, floor, ceiling, chair, table, sofa,
  bed, other with `assets/scannetv2-labels.combined.tsv`.

## ScanNet++

```
datasets/scannetpp/
├── scannetpph5_256/<scene>.h5     same keys as the ScanNet gt_h5 files, 256 x 256
└── nvs_sem_val.txt                official validation scene list (excluded from training)
```

Used without instance masks, for backbone fine-tuning, Stage 1 and feature lifting.

## RealEstate10K

```
datasets/re10k/
├── train/<shard>.h5
├── test/<shard>.h5
├── masks_train/<sequence>/default/final-output-label-png/<frame:03d>.png
└── masks_test/<sequence>/default/final-output-label-png/<frame:03d>.png
```

Each `<shard>.h5` contains one group per sequence (the group name is the sequence hash):

| Key | Content |
|---|---|
| `images` | 1-D uint8, concatenated JPEG buffers of all frames |
| `image_offsets` | `N + 1` byte offsets of the frames inside `images` |
| `gt_camera_pose` | `(N, 4, 4)` camera-to-world poses |
| `gt_intrinsics` | `(N, 3, 3)` intrinsics in pixels of the 256 x 256 centre crop |

Frames are resized so that the shorter side is 256 and centre-cropped. The evaluation
(`config/dm/re10k_test.yaml`) covers the test sequences of the evaluation index that have
pseudo-labels in `masks_test/`.
`assets/re10k_valid.json` lists the usable training sequences per shard and
`assets/evaluation_index_re10k.json` is the pixelSplat / MVSplat evaluation index
(context pair and target frames per test sequence).

**SAM2 pseudo-labels.** RealEstate10K has no instance annotation. We run SAM2 with automatic
mask generation and video propagation on each clip
([AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2)) and store one label image per
frame: a 16-bit PNG at the native 360 x 640 resolution, `0` = void, `1..N` = instance ids
that are consistent within the clip. Only sequences and frames with masks are used for
Stage 2 training.

## Mip-NeRF 360

Download `360_v2.zip` from the [Mip-NeRF 360 page](https://jonbarron.info/mipnerf360/):

```
datasets/mipnerf360/<scene>/
├── images_4/<name>.JPG
└── sparse/0/{cameras.bin, images.bin}
```

The seven public scenes (bicycle, bonsai, counter, garden, kitchen, room, stump) are
evaluated with the same 2-context / 1-target rule as the ScanNet 2-view protocol
(408 triplets). The translations are rescaled so that the two context cameras are
`baseline_target` (0.25) apart, which brings the SfM scale into the range of the
ScanNet-trained model.
