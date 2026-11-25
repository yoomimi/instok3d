"""Export a training run as a model directory: ``config.yaml`` + ``model.safetensors``.

Usage:
    python run_script.py scripts/export_model.py run=logs/<run> output=pretrained/<name>
"""

import shutil
from os import makedirs, path

from safetensors.torch import save_file

import engine
from instok3d.utils.weights import SAFETENSORS_NAME, find_weights_file, load_model_state_dict


@engine.entry_fn()
def main(cfg, run: str = None, output: str = None, is_best: bool = True):
    if run is None or output is None:
        raise ValueError('Both "run" and "output" are required.')

    weights_path = find_weights_file(run, is_best=is_best)
    state_dict = load_model_state_dict(run, is_best=is_best)

    makedirs(output, exist_ok=True)
    shutil.copyfile(path.join(run, 'config.yaml'), path.join(output, 'config.yaml'))
    save_file(state_dict, path.join(output, SAFETENSORS_NAME))

    num_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f'Exported {weights_path} ({num_params:,} values) to {output}.')
