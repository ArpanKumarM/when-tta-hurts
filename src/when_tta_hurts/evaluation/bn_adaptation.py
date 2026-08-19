"""Frozen BN-adaptation semantics, per docs/phase2b_protocol.md sec.4.
Paper-constrained operationalization (source implementation unavailable).

Exact procedure (frozen, do not deviate):
1. Begin from an untouched COPY of the frozen best checkpoint.
2. Use only unlabeled inputs from the split being evaluated.
3. One deterministic, no-gradient pass over the relevant augmented inputs.
4. Update BatchNorm running mean/variance only.
5. Do not update convolutional, linear, affine-BN, or any other learned
   parameter.
6. Return the model to evaluation mode before prediction.
7. Reset from the original checkpoint separately for every
   split/N/seed/condition -- this module never carries state across calls;
   each call to bn_adapt() takes a fresh model copy.
8. Never use labels during adaptation (no labels parameter exists on
   bn_adapt() at all -- structurally impossible to pass any).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn


class BNAdaptationNotApplicableError(RuntimeError):
    """Raised when bn_adapt() is called on a model with no BatchNorm
    layers (e.g. a GroupNorm cell) -- BN adaptation is not applicable."""


def _has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def _run_bn_adaptation_pass(model: nn.Module, batches: Iterable[torch.Tensor]) -> nn.Module:
    """Shared core for bn_adapt()/bn_adapt_sequential(): deep-copy `model`,
    verify BatchNorm is present, run a deterministic, no-gradient, train()-
    mode forward pass over EACH tensor in `batches` in order (a
    single-element iterable reproduces the original one-shot single-batch
    behavior exactly -- same one forward call), verify every learned
    parameter is bit-identical before/after, return to eval mode. `model`
    itself is never mutated. Raises BNAdaptationNotApplicableError if
    `model` has no BatchNorm layers (GroupNorm cells)."""
    if not _has_batchnorm(model):
        raise BNAdaptationNotApplicableError(
            "bn_adapt() called on a model with no BatchNorm layers -- BN adaptation "
            "is not applicable to GroupNorm cells, per docs/phase2b_protocol.md sec.4."
        )

    adapted = copy.deepcopy(model)

    # Snapshot every learned parameter BEFORE the adaptation pass, so
    # immutability can be verified by the caller/tests after the fact.
    params_before = {name: p.detach().clone() for name, p in adapted.named_parameters()}

    adapted.train()  # BatchNorm only updates running stats in train() mode
    for batch in batches:
        _ = adapted(batch)  # deterministic, no-gradient forward pass (decorator enforces no_grad)
    adapted.eval()  # return to eval mode before prediction, per step 6

    params_after = dict(adapted.named_parameters())
    for name, before in params_before.items():
        after = params_after[name]
        if not torch.equal(before, after):
            raise RuntimeError(
                f"INVARIANT VIOLATION: learned parameter '{name}' changed during BN "
                f"adaptation. This must never happen -- only BatchNorm running "
                f"statistics may change. Aborting."
            )

    return adapted


@torch.no_grad()
def bn_adapt(model: nn.Module, adaptation_inputs: torch.Tensor) -> nn.Module:
    """Return a NEW model (deep copy of `model`) with BatchNorm running
    statistics adapted to `adaptation_inputs` via one forward pass.
    `model` itself is never mutated. Raises BNAdaptationNotApplicableError
    if `model` has no BatchNorm layers (GroupNorm cells).

    Unchanged, single-batch primitive -- kept for callers (and this
    module's own existing test suite) that already have their full
    adaptation population as one in-memory tensor. See
    bn_adapt_sequential() for the bounded-memory, multi-microbatch
    variant used by the production evaluation path (Phase 2B.4D OOM
    correction, docs/phase2b_validation_evaluation_batching_freeze.md).
    """
    return _run_bn_adaptation_pass(model, [adaptation_inputs])


@torch.no_grad()
def bn_adapt_sequential(model: nn.Module, adaptation_batches: Iterable[torch.Tensor]) -> nn.Module:
    """Return a NEW model (deep copy of `model`) with BatchNorm running
    statistics adapted via a deterministic SEQUENCE of no-gradient
    forward passes over `adaptation_batches` (an iterable of tensors,
    each at most the frozen adaptation batch size -- algorithm
    `sequential_microbatch_v1`, see
    docs/phase2b_validation_evaluation_batching_freeze.md sec.4.2).
    `model` itself is never mutated.

    NOT mathematically equivalent to bn_adapt() when `adaptation_batches`
    yields more than one tensor -- PyTorch BatchNorm's running-statistics
    update is an order-sensitive exponential moving average applied on
    EVERY forward call, so a sequence of microbatch calls computes a
    genuinely different result than one call over the pooled population
    (see the freeze document's deterministic synthetic proof). Passing a
    single-element iterable reproduces bn_adapt()'s result exactly, by
    construction (both delegate to the same shared core). Raises
    BNAdaptationNotApplicableError if `model` has no BatchNorm layers.
    """
    return _run_bn_adaptation_pass(model, adaptation_batches)
