"""Export the Gaussians predicted by a tokenizer as 3D Gaussian Splatting ``.ply`` files."""

import json
import os
from typing import Dict

import numpy as np
import torch

import engine
from engine import inject
from engine.components import Evaluator
from instok3d.models.anchor_recon.assignment import build_hard_assignment
from instok3d.utils.training import forward_backbone


class GaussianExportEvaluator(Evaluator):
    """Writes the token groups of each sample to ``<output_subdir>/<index>_<scene>/``.

        scene.ply          all Gaussians
        group_<id>.ply     Gaussians of one group (groups that own no Gaussian are skipped)
        void.ply           Gaussians that belong to no group
        groups.json        per group: id, number of Gaussians, anchor mass, centroid

    The files use the layout of 3D Gaussian Splatting (log scales, opacity logits, wxyz
    quaternions, spherical harmonics) in the frame of the first context camera, so they open
    in standard viewers. A model without the grouping head only produces ``scene.ply``.
    """

    @inject('dm', 'solver')
    def __init__(self, dm, solver, output_subdir: str = 'outputs/gaussians'):
        super().__init__()
        self.dm = dm
        self.solver = solver
        self.model = solver.model
        self.backbone = solver.backbone
        self.dm.eval_batch_size = 1
        self.output_dir = engine.to_output_dir(output_subdir)

    def step(self, batch, batch_idx):
        input_images = batch['input_images']  # (1, V, 3, H, W)
        gaussians, aux = self.model(input_images, forward_backbone(self.backbone, input_images, batch))
        vertices = _to_ply_vertices(gaussians, self.model.renderer._prepare_feature(gaussians))

        sample_dir = os.path.join(
            self.output_dir, f"{int(batch['idx'][0]):04d}_{batch['scene_name'][0]}",
        )
        os.makedirs(sample_dir, exist_ok=True)
        _write_ply(os.path.join(sample_dir, 'scene.ply'), vertices)

        if getattr(self.model, 'use_instance_queries', False):
            groups = self._write_groups(sample_dir, vertices, aux)
            with open(os.path.join(sample_dir, 'groups.json'), 'w') as fp:
                json.dump(groups, fp, indent=2)
        print(f'{sample_dir}: {len(vertices)} Gaussians')

    def _write_groups(self, sample_dir: str, vertices: np.ndarray, aux: Dict[str, torch.Tensor]):
        hard_query_probs, hard_void_prob = build_hard_assignment(
            aux['gaussian_query_probs'], aux['gaussian_void_prob'],
        )
        group_of_gaussian = hard_query_probs[0].argmax(dim=-1)
        group_of_gaussian[hard_void_prob[0] > 0.5] = -1
        group_of_gaussian = group_of_gaussian.cpu().numpy()
        query_mass = aux['query_mass'][0].float().cpu().numpy()

        groups = []
        for group_id in np.unique(group_of_gaussian):
            members = vertices[group_of_gaussian == group_id]
            name = 'void.ply' if group_id < 0 else f'group_{group_id:03d}.ply'
            _write_ply(os.path.join(sample_dir, name), members)
            if group_id >= 0:
                centroid = [float(members[axis].mean()) for axis in ('x', 'y', 'z')]
                groups.append({
                    'id': int(group_id),
                    'file': name,
                    'num_gaussians': int(len(members)),
                    'mass': float(query_mass[group_id]),
                    'centroid': centroid,
                })
        return groups


def _to_ply_vertices(gaussians, sh_coeffs: torch.Tensor) -> np.ndarray:
    """Pack the Gaussians of the first batch element; ``sh_coeffs`` is (B, N, K, 3)."""
    xyz = gaussians.xyz[0].float()
    opacity = gaussians.opacity[0].float().clamp(1e-6, 1.0 - 1e-6)
    sh_coeffs = sh_coeffs[0].float()
    num_rest = 3 * (sh_coeffs.shape[1] - 1)

    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    names += [f'f_rest_{i}' for i in range(num_rest)]
    names += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    columns = torch.cat([
        xyz,
        torch.zeros_like(xyz),
        sh_coeffs[:, 0],
        sh_coeffs[:, 1:].transpose(1, 2).reshape(-1, num_rest),  # channel-major, as in 3DGS
        torch.logit(opacity).unsqueeze(-1),
        torch.log(gaussians.scale[0].float().clamp_min(1e-9)),
        gaussians.rotation[0].float(),
    ], dim=-1).cpu().numpy().astype(np.float32)

    vertices = np.empty(len(columns), dtype=[(name, '<f4') for name in names])
    for i, name in enumerate(names):
        vertices[name] = columns[:, i]
    return vertices


def _write_ply(file_path: str, vertices: np.ndarray):
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(vertices)}']
    header += [f'property float {name}' for name in vertices.dtype.names]
    header += ['end_header', '']
    with open(file_path, 'wb') as fp:
        fp.write('\n'.join(header).encode('ascii'))
        fp.write(vertices.tobytes())
