"""Seed handling for Python, NumPy, and PyTorch, plus seeded DataLoader workers.

Note: bitwise-identical results across CPU and MPS are NOT guaranteed even
with all seeds fixed and deterministic settings enabled -- MPS kernels do not
provide the same determinism guarantees as CPU or CUDA. Seeding here controls
sampling/initialization, not floating-point reduction order on MPS.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python's random, NumPy, and PyTorch (CPU + all GPU backends)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.mps does not expose a separate manual_seed as of this torch
    # version's public API being used the same as CUDA's; guarded for
    # forward/backward compatibility.
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # CPU-side determinism knob; irrelevant on MPS but harmless.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    """A CPU torch.Generator for DataLoader(generator=...), seeded deterministically."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive a distinct, deterministic seed per worker."""
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
