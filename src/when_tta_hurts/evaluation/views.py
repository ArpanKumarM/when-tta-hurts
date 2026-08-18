"""Deterministic, batch/worker/restart-independent TTA view generation for
confirmatory (Phase 2B.4) validation-only evaluation.

The Phase 2A pilot's `evaluation/tta.py::compute_ordered_view_logits` seeds
the WHOLE-BATCH call via `torch.manual_seed(tta_seed + view_idx)` before
applying the policy to the entire batch tensor at once. Because kornia's
`same_on_batch=False` draws each sample's random parameters sequentially
from the single global RNG stream established by that one seed, the
transform a given sample receives depends on WHERE it sits within that
particular batch call -- i.e. on batch composition/boundaries, not on the
sample's own stable identity. That is fine for the pilot (a single fixed
run), but Phase 2B.4A's confirmatory requirement is stronger: "view
identity must be independent of ... batch boundaries" and "models sharing
the same dataset, resolution, and sample order must receive bit-identical
transformed inputs for every view index" (including across different
evaluation batch sizes/worker counts/restarts).

This module satisfies that by transforming ONE SAMPLE AT A TIME, seeded by
a STABLE, batch-independent combination of (tta_seed, dataset, resolution,
sample_index, view_index) -- via hashlib (never Python's randomized
`hash()`). The frozen transform POLICY itself (build_policy()) is reused
completely unchanged: only the seeding/batching mechanism is new here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

_SEED_MODULUS = 2**31 - 1  # fits comfortably in torch.manual_seed's accepted range


def stable_view_seed(tta_seed: int, dataset: str, resolution: int, sample_index: int, view_index: int) -> int:
    """Deterministic per-(sample, view) seed, independent of batch
    boundaries, worker count, execution order, or any mutable global RNG
    state. Uses hashlib.sha256, never Python's randomized `hash()` (whose
    output varies per-process via PYTHONHASHSEED randomization)."""
    payload = f"{tta_seed}|{dataset}|{resolution}|{sample_index}|{view_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


@dataclass(frozen=True)
class ViewSeedManifestEntry:
    sample_index: int
    view_index: int
    seed: int


def build_view_seed_manifest(
    tta_seed: int, dataset: str, resolution: int, sample_indices: Sequence[int], n_views: int
) -> list[ViewSeedManifestEntry]:
    """Full (sample, view) -> seed manifest for provenance persistence.
    Deliberately does not depend on model/checkpoint/training seed/batch
    size/worker count -- only on the five stable identifiers listed above."""
    return [
        ViewSeedManifestEntry(
            sample_index=sample_indices[i],
            view_index=v,
            seed=stable_view_seed(tta_seed, dataset, resolution, sample_indices[i], v),
        )
        for v in range(n_views)
        for i in range(len(sample_indices))
    ]


def generate_single_view(
    x: torch.Tensor,
    policy: nn.Module,
    tta_seed: int,
    dataset: str,
    resolution: int,
    sample_indices: Sequence[int],
    view_index: int,
) -> torch.Tensor:
    """Generate ONE view (view_index) for every sample in `x` ([N, C, H, W],
    any device -- moved to CPU internally, matching the pilot's documented
    MPS-performance workaround). Each sample is transformed in ISOLATION
    (a batch of size 1), seeded by stable_view_seed(), so the result for
    sample i is identical regardless of which other samples are in `x`,
    what order they appear in, or how many workers/processes are involved.

    Augmentation is applied EXACTLY ONCE per (sample, view_index) pair --
    this function is the single call site that performs it.
    """
    x_cpu = x.detach().to("cpu")
    policy_cpu = policy.to("cpu")
    n = x_cpu.shape[0]
    if len(sample_indices) != n:
        raise ValueError(f"len(sample_indices)={len(sample_indices)} does not match batch size {n}.")

    out = torch.empty_like(x_cpu)
    for i in range(n):
        seed = stable_view_seed(tta_seed, dataset, resolution, sample_indices[i], view_index)
        torch.manual_seed(seed)
        with torch.no_grad():
            transformed = policy_cpu(x_cpu[i : i + 1])
        out[i] = transformed[0]
    return out


def iter_deterministic_views(
    x: torch.Tensor,
    policy: nn.Module,
    tta_seed: int,
    dataset: str,
    resolution: int,
    sample_indices: Sequence[int],
    n_views: int,
):
    """Yield (view_index, view_batch) for view_index in [0, n_views), one
    view at a time -- callers should consume each view (e.g. run the model
    forward pass, take softmax, store only the resulting probabilities)
    and let the yielded tensor be garbage-collected before requesting the
    next view, so no more than one view's worth of transformed images is
    ever retained at once (per the "stream views/batches" requirement --
    raw image tensors are never persisted to disk by this module)."""
    for view_index in range(n_views):
        yield (
            view_index,
            generate_single_view(x, policy, tta_seed, dataset, resolution, sample_indices, view_index),
        )
