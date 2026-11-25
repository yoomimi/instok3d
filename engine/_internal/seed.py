import os
import random

import numpy as np
import torch


def get_dataloader_generator(global_rank: int = 0):
    seed = os.environ.get('PL_GLOBAL_SEED', None)
    if seed is None:
        return None

    # The generator of a DataLoader seeds its worker processes: the rank offset gives the
    # workers of every DDP process different seeds.
    g = torch.Generator()
    g.manual_seed(int(seed) + global_rank)

    return g


def worker_seed_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
