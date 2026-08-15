"""Reproducibility helpers.

Seed 42 everywhere.  ``cudnn.benchmark=True`` and ``deterministic=False`` are
deliberate: full determinism costs roughly 20-30% throughput on convolutions,
and the run-to-run spread it would remove is measured directly instead, by
repeating one experiment row with a second seed.  A delta smaller than that
spread is not a result.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def loader_generator(seed: int = SEED) -> torch.Generator:
    """Generator for a DataLoader, so shuffling is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int, seed: int = SEED) -> None:
    """Give every DataLoader worker its own deterministic numpy stream."""
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def device_report() -> str:
    """One line describing the compute the notebook actually ran on."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda: {name} ({total:.1f} GB)"
    return "cpu (no CUDA device visible)"
