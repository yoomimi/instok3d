# Project Page — Instance-Structured 3D Tokenization

Modern single-file project page. Open `index.html` in any browser (no build step).

## Drop-in assets
Replace the placeholders by saving images to `assets/figures/`:

| File | Paper source | Used in |
|------|--------------|---------|
| `teaser.png` | Figure 1 | Hero teaser (+ social preview / og:image) |
| `method.png` | Figure 2 | Method section |
| `recon.png` | Figure 3 | Results — under Table 1 (reconstruction) |
| `semantic.png` | Figure 4 | Results — under Table 1 (open-vocab segmentation) |
| `instance.png` | Figure 5 | Results — under Table 2 (class-agnostic instance seg) |
| `manipulation.png` | **Figure 6 (with insertion)** | Applications |

The paper's three qualitative figures (recon / semantic / instance) are kept as
**three separate slots** placed next to their matching result table. (Figure 7,
open-vocab retrieval, isn't slotted yet — say the word to add a `retrieval.png`
slot in the Applications "Open-Vocab Retrieval" card.)

Each `<img>` falls back to a styled placeholder until the real file exists,
so the page looks fine before assets are added.

## TODO before publishing
- Fill in the `href="#"` links in the hero (Paper PDF, arXiv, Code).
- Update the `@inproceedings` venue/year in the BibTeX block.
- Drop in the six figures above (export from the paper or render fresh).
- (optional) Add an `assets/videos/manipulation.mp4` and swap the `<img>` for `<video autoplay muted loop>` to show insertion as a clip.

## Light / dark mode
Toggle button (sun/moon) in the top-right of the nav. Choice is saved to
`localStorage`; first visit follows the OS `prefers-color-scheme`.

## Interactive token demo (`#demo`)
Real ScanNet test scene (`scene0693_00`, 32 768 Gaussians) decomposed into 8
instance token groups by the model.

**Primary: real 3D Gaussian splats.** Loads the actual predicted Gaussians as
splats via [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D)
(three.js, CDN). **Drag to orbit, scroll to zoom, right-drag to pan.** Clicking a
token rebuilds the scene from the enabled instances, so switching a token off
removes exactly those Gaussians. Per-instance `.ply` files + `meta.json` live in
`assets/demo/splat/` (`tok_00.ply … tok_07.ply` = the 8 tokens, `rest.ply` =
background, always on).

**Fallback: 2.5D diorama.** If WebGL2 / the CDN / workers aren't available (e.g.
opening via `file://`), it stays on the depth-parallax PNG diorama
(`assets/demo/tok_NN.png`) — drag still tilts the scene and tokens still toggle.
**Run over a local server** (below) for the real splats.

> Needs a browser to actually render (WebGL). Everything else (export, ply
> validity, asset serving) is verified; if the initial camera/orientation looks
> off, tweak `cameraUp` / `initialCameraPosition` in the inline module.

### Re-exporting / swapping scenes
Splat plys are produced by `scripts/export_scene_splat.py` (reuses the
`insert_instance_AtoB` forward + ranking machinery, so `tok_NN` lines up with the
`q00…q07` white renders):

```bash
conda run -n inst3d python scripts/export_scene_splat.py \
    --scene scene0693_00 --topk 8 --out-dir project_page/assets/demo/splat --gpu 0
```

INRIA 3DGS `.ply` format (degree-0 SH): `log(scale)`, `logit(opacity)`, wxyz quat,
`f_dc = SH DC`. To use a different scene, re-run with `--scene <name>` and update
the `DEMO.TOKENS` names/colors in the inline script.

## Local preview
```bash
cd project_page && python -m http.server 8000
# open http://localhost:8000
```
