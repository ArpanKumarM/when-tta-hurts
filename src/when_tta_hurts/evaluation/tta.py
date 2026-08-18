"""Deterministic ordered-view TTA inference with nested-prefix aggregation.

Generates ONE deterministic ordered sequence of augmented views (up to
max_views), computes per-view logits once, and derives every tested view
count (a prefix of that sequence) from the same cached logits -- per
docs/compute_budget.md's evaluation-cache design and
docs/pilot_protocol.md's "nested deterministic prefixes" requirement.

Only mean-probability aggregation is implemented here (the pilot's only
aggregation method, per docs/pilot_protocol.md); majority-vote and
confidence-weighted aggregation are out of scope for Phase 2A.

KNOWN MPS PERFORMANCE ISSUE (measured, not assumed -- see the Phase 2A
completion report for full numbers): kornia's RandomResizedCrop (~186ms per
call at batch=256, 28px) and RandomGaussianBlur (~91ms/call) are roughly
15x slower on MPS than on CPU for this exact workload (full mixed-policy
call: ~350ms/call on MPS vs. ~22ms/call on CPU), likely because their
underlying grid_sample-based warping and separable-convolution blur have
poor MPS kernel support in this torch/kornia version. This first showed up
as a near-total stall during real pilot execution (a process that appeared
"running" but made almost no CPU-time progress over ~40 minutes). Per the
project's hard constraint ("if an op fails on MPS, implement a controlled
CPU preprocessing path instead of silently falling back inside an
experiment"), TTA augmentation is therefore performed on CPU here
UNCONDITIONALLY, regardless of the `device` argument -- only the resulting
augmented batch is moved to `device` for the (fast, MPS-native) model
forward pass. This is model inference; not silently changing the frozen
augmentation policy itself, which is unaffected in kind, values, or
outcome by which device executes it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from when_tta_hurts.metrics import softmax
from when_tta_hurts.transforms import sample_deterministic_view


@torch.no_grad()
def compute_ordered_view_logits(
    model: nn.Module,
    x: torch.Tensor,
    policy: nn.Module,
    tta_seed: int,
    max_views: int,
    device: torch.device,
) -> np.ndarray:
    """Return an array of shape [max_views, N, C]: per-view logits for a
    deterministic ordered sequence of `max_views` augmented views of `x`.
    View i uses seed `tta_seed + i` (a distinct, deterministic seed per
    view -- see transforms/policies.py::sample_deterministic_view).

    Augmentation runs on CPU regardless of `device` (see module docstring
    "KNOWN MPS PERFORMANCE ISSUE"); only the model forward pass uses
    `device`. `policy` and `x` may be passed on either device -- both are
    moved to CPU internally for the augmentation step.
    """
    model.eval()
    x_cpu = x.detach().to("cpu")
    policy_cpu = policy.to("cpu")
    all_view_logits = []
    for view_idx in range(max_views):
        view_cpu = sample_deterministic_view(x_cpu, policy_cpu, seed=tta_seed + view_idx)
        logits = model(view_cpu.to(device))
        all_view_logits.append(logits.detach().cpu().numpy())
    return np.stack(all_view_logits, axis=0)  # [max_views, N, C]


def aggregate_mean_prefix(ordered_view_logits: np.ndarray, n_views: int, eps: float = 1e-12) -> np.ndarray:
    """Mean-probability aggregation over the first `n_views` of an ordered
    [max_views, N, C] logits array (a nested prefix, per the pilot
    protocol). Returns a [N, C] array in LOG-PROBABILITY space so it can be
    fed directly into metrics.py's softmax-based functions unchanged:
    softmax(log(p)) == p exactly (p is already normalized), so this is a
    lossless representation, not an approximation.
    """
    if n_views < 1 or n_views > ordered_view_logits.shape[0]:
        raise ValueError(f"n_views={n_views} out of range [1, {ordered_view_logits.shape[0]}]")
    prefix = ordered_view_logits[:n_views]  # [n_views, N, C]
    probs_per_view = np.stack([softmax(v) for v in prefix], axis=0)  # [n_views, N, C]
    mean_probs = probs_per_view.mean(axis=0)  # [N, C]
    return np.log(np.clip(mean_probs, eps, 1.0))
