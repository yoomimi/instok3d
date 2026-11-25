import torch


def compute_sh_mask(sh_degree):
    """Per-coefficient damping of the spherical-harmonics bands above degree 0."""
    sh_mask = torch.ones((sh_degree + 1) ** 2, dtype=torch.float32)
    for degree in range(1, sh_degree + 1):
        sh_mask[degree ** 2: (degree + 1) ** 2] = 0.1 * 0.25 ** degree
    return sh_mask
