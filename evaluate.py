import engine
from engine.components import DataModule, Evaluator, Solver
from instok3d.utils.weights import (
    find_weights_file, load_lightning_state_dict, load_safetensors_state_dict, subset_state_dict,
)


@engine.main(config_dir='config/', default_config_name='eval_scannet_2view', fix_seed=False,
             load_groups=['solver', 'model', 'backbone', 'teacher'], require='load')
def main(cfg, eval: Evaluator, solver: Solver, dm: DataModule):
    for module in solver.modules():
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()

    model_dir = engine.get_current_run_state().output_dir
    weights_path = find_weights_file(model_dir, is_best=cfg.is_best, mode=cfg.checkpoint.mode)
    print(f'Loading weights from {weights_path}')
    _load_weights(solver, weights_path)

    solver.eval()
    eval.run(dm, cfg.stage, data_indices=cfg.data_indices, gpus=cfg.gpus, seed=cfg.seed)


def _load_weights(solver: Solver, weights_path: str):
    if weights_path.endswith('.safetensors'):
        solver.model.load_state_dict(load_safetensors_state_dict(weights_path))
        return

    state_dict = load_lightning_state_dict(weights_path)['state_dict']
    solver.model.load_state_dict(subset_state_dict(state_dict, 'model.'))
    # Only a backbone that is being fine-tuned is stored in checkpoints.
    backbone_state_dict = subset_state_dict(state_dict, 'backbone.')
    if len(backbone_state_dict) > 0:
        solver.backbone.load_state_dict(backbone_state_dict)


if __name__ == '__main__':
    main()
