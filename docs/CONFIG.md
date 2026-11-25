# Configuration

Training, evaluation and scripts are driven by a small config layer (`engine/`) on top of
PyTorch Lightning and OmegaConf.

## Entry configs and sub-configs

```
config/
├── train_*.yaml, eval_*.yaml    entry configs (first positional argument)
├── dm/         data modules
├── model/      models
├── solver/     LightningModules: losses, optimisation, validation
├── backbone/   VGGT backbone (fine-tuning / frozen with the released weights)
├── teacher/    2D foundation model for feature lifting
├── eval/       evaluators
└── scripts/    defaults of the scripts run through run_script.py
```

An entry config names one sub-config per group:

```yaml
# config/train_tokenizer_scannet_2view.yaml
_base_: train
dm: scannet_2view                     # -> config/dm/scannet_2view.yaml
solver: tokenizer                     # -> config/solver/tokenizer.yaml
model: tokenizer                      # -> config/model/tokenizer.yaml
backbone: vggt_intr_scannet_2view     # -> config/backbone/vggt_intr_scannet_2view.yaml
```

Reserved keys:

| Key | Meaning |
|---|---|
| `_target_` | Python class to instantiate. A leading `.` means "in the module of the parent". |
| `_base_` | Inherit from another file of the same directory and override the listed keys. |
| `_provider_` | Inject a factory (`functools.partial`) instead of an instance. |

The other keys of a sub-config are the constructor arguments of its `_target_`; unknown
keys are an error.

## Command-line overrides

```bash
python train.py <entry> key=value group.key=value group=other_sub_config ~key.to.delete
```

- `dm.batch_size=8` changes one value, `model=tokenizer_8view` swaps a whole sub-config.
- Shortcuts: `e.` = `engine.`, `t.` = `trainer.`, `l.` = `logging.`, `d.` = `debugging.`
- A key that is not present in the yaml needs a leading `+`: `+backbone.cache_dir=<dir>`,
  or `+trainer.max_time=00:12:00:00` (`trainer.*` is passed to `lightning.Trainer`).

## Runs, checkpoints and logging

| Option | Effect |
|---|---|
| `e.name=<run>` | Run name; outputs go to `logs/<run>/` (`config.yaml`, `checkpoints/`). |
| `e.gpus=N` | Number of training GPUs (a count; `-1` = all visible). Select devices with `CUDA_VISIBLE_DEVICES`. |
| `e.debug=true` | Quick check: one training and one validation batch, no logging, no checkpoints. |
| `resume=true\|best\|last\|null\|<path>` | Resume the run `e.name` (default `true`: from `last.ckpt` if it exists; `init_from` is then ignored). |
| `init_from=<run dir \| model dir \| .ckpt \| .safetensors>` | Initialise the model weights. Group-head weights may be missing (Stage 1 -> Stage 2); anything else must match. |
| `e.load=<dir>` | `evaluate.py`: model / solver / backbone config and weights of a model directory or run. |

Checkpoints are written to `logs/<run>/checkpoints/` as `last.ckpt` and
`best-epoch=<E>-score=<S>.ckpt`; the score is the `track_metric` of the solver.

Weights of frozen modules (the VGGT backbone, the LSeg teacher, the frozen tokenizer
inside the feature-lifting model) are not stored in checkpoints. They are re-loaded from
the paths in the config: `backbone.finetuned_weights_path`, `teacher.pretrained_path`,
`model.recon_model_path`.

### Weights & Biases

Create `secrets/wandb.yaml` (git-ignored) to log to Weights & Biases:

```yaml
entity: <your wandb entity>
api_key: <your wandb api key>
```

Without it, metrics are written to `logs/<run>/csv_logs/`.

## Evaluation options

```bash
python evaluate.py eval_scannet_2view e.load=pretrained/instok3d_scannet_2view
```

| Option | Effect |
|---|---|
| `is_best=false` | For a training run: use `last.ckpt` instead of the best-score checkpoint. |
| `data_indices=0,8,16` / `0-39` / `null` | Evaluate a subset / all samples. |
| `eval.pose_align_steps=0` | Disable the test-time pose alignment of the target cameras. |
| `eval.save_images=false` | Metrics only. |
| `dm.dataset.crop_mode=dust3r` | Pre-processing of the ScanNet test frames (see docs/DATA.md). |
