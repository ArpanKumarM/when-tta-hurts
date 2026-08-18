"""Dedicated validation-only evaluation loader for Phase 2B.4A. Split is
STRUCTURALLY fixed to 'val' -- there is no split parameter of any kind on
load_validation_evaluation_split(), so no CLI flag or environment variable
anywhere in this project can make it load anything else. Built on top of
data.py::load_pilot_split(), which already has no test-split mechanism at
all (see its docstring) -- this function narrows that further to exactly
one split, never exposing 'train' either, since evaluation must never
retrain or peek at training-order data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from when_tta_hurts.data import load_pilot_split


@dataclass(frozen=True)
class ValidationEvaluationSplit:
    """Structurally two fields only: images/labels for the validation
    split, plus stable sample indices -- no test-related field exists on
    this type at all, so one cannot be constructed to hold test data."""

    images: torch.Tensor  # [N, C, H, W], float32 in [0, 1]
    labels: np.ndarray  # [N] int
    sample_indices: np.ndarray  # [N] int, stable positional identity (0..N-1)
    dataset: str
    resolution: int


def load_validation_evaluation_split(
    dataset: str, resolution: int, root: str | Path = "data/raw"
) -> ValidationEvaluationSplit:
    """Load the official validation split ONLY -- 'val' is hardcoded, not a
    parameter. Uses load_pilot_split(), which raises TestSplitAccessError
    for anything other than 'train'/'val'; this function never even
    attempts 'train' or 'test'. Returns tensors/arrays in stable dataset
    order (a plain DataLoader with shuffle=False, batch_size=len(dataset),
    single batch) so `sample_indices` is simply range(N) and remains a
    valid, reproducible identity across any evaluation batch size chosen
    downstream (batching for inference is a separate, later step -- this
    function does not itself impose any batch size on the returned data).
    """
    val_dataset = load_pilot_split(dataset, split="val", size=resolution, root=str(root))
    n = len(val_dataset)
    loader = DataLoader(val_dataset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
    sample_indices = np.arange(n, dtype=np.int64)
    return ValidationEvaluationSplit(
        images=images.float(),
        labels=labels_np,
        sample_indices=sample_indices,
        dataset=dataset,
        resolution=resolution,
    )
