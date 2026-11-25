"""Extract the fine-tuned backbone of a training run into a ``.safetensors`` file.

The result is what ``backbone.finetuned_weights_path`` expects.

Usage:
    python run_script.py scripts/extract_backbone_weights.py run=<run dir or .ckpt> output=<file.safetensors>
"""

from os import makedirs, path

from safetensors.torch import save_file

import engine
from instok3d.utils.weights import find_weights_file, load_lightning_state_dict, subset_state_dict

BACKBONE_PREFIX = 'backbone.'


@engine.entry_fn()
def main(cfg, run: str = None, output: str = None, is_best: bool = True):
    if run is None or output is None:
        raise ValueError('Both "run" and "output" are required.')
    if not output.endswith('.safetensors'):
        raise ValueError(f'"output" must be a .safetensors file; got {output}.')

    checkpoint_path = find_weights_file(path.expanduser(run), is_best=is_best)
    if not checkpoint_path.endswith('.ckpt'):
        raise ValueError(f'Expected a Lightning checkpoint; found {checkpoint_path}.')
    checkpoint = load_lightning_state_dict(checkpoint_path)

    state_dict = subset_state_dict(checkpoint['state_dict'], BACKBONE_PREFIX)
    if not state_dict:
        raise RuntimeError(
            f'{checkpoint_path} has no backbone weights; frozen backbones are not stored in checkpoints.'
        )

    output = path.expanduser(output)
    makedirs(path.dirname(path.abspath(output)), exist_ok=True)
    save_file(state_dict, output)

    num_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f'Checkpoint: {checkpoint_path} (epoch {checkpoint.get("epoch")})')
    print(f'Saved {len(state_dict)} tensors ({num_params:,} values) to {output}.')
