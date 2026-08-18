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

import torch
from torch import nn


class BNAdaptationNotApplicableError(RuntimeError):
    """Raised when bn_adapt() is called on a model with no BatchNorm
    layers (e.g. a GroupNorm cell) -- BN adaptation is not applicable."""


def _has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


@torch.no_grad()
def bn_adapt(model: nn.Module, adaptation_inputs: torch.Tensor) -> nn.Module:
    """Return a NEW model (deep copy of `model`) with BatchNorm running
    statistics adapted to `adaptation_inputs` via one forward pass.
    `model` itself is never mutated. Raises BNAdaptationNotApplicableError
    if `model` has no BatchNorm layers (GroupNorm cells).
    """
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
    _ = adapted(adaptation_inputs)  # one deterministic, no-gradient pass (decorator enforces no_grad)
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
