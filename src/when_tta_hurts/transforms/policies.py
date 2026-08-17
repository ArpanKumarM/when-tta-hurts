"""Deterministic building blocks for the TTA augmentation policies described
in the source paper (arXiv:2604.09697, Section 4.3, verified full text --
see docs/literature_review.md):

  geometric: horizontal/vertical flip, rotation up to +/-15deg,
             random resized crop scale 0.8-1.0
  intensity: brightness/contrast jitter up to +/-0.3, Gaussian blur
  mixed:     geometric + intensity

The paper specifies the transform families and headline parameter ranges
but NOT every implementation detail (interpolation mode, fill/border
behavior, aspect-ratio range for the crop, exact blur kernel/sigma,
application probabilities, or operation order). ALL such details are
IMPLEMENTATION CHOICES, and per docs/experimental_protocol.md's frozen
augmentation table, they are fixed here and must not change silently after
observing pilot results -- any change requires a protocol amendment.

See docs/experimental_protocol.md "Frozen augmentation parameters" for the
authoritative table this module implements.
"""

from __future__ import annotations

import kornia.augmentation as K
import torch
from torch import nn

# --- Frozen parameters (see docs/experimental_protocol.md for the table) ---
_FLIP_P = 0.5
_ROTATION_DEGREES = 15.0
_ROTATION_RESAMPLE = (
    "BILINEAR"  # kornia default; no separate fill/border param exposed by this kornia version
)
_ROTATION_P = 1.0
_CROP_SCALE = (0.8, 1.0)  # paper-specified
_CROP_RATIO = (
    3.0 / 4.0,
    4.0 / 3.0,
)  # NOT specified by the paper; kornia's conventional default, frozen explicitly
_CROP_RESAMPLE = "BILINEAR"
_CROP_P = 1.0
_JITTER_BRIGHTNESS = 0.3
_JITTER_CONTRAST = 0.3
_JITTER_P = 1.0
_GAUSSIAN_BLUR_KERNEL = (3, 3)  # conservative choice per protocol; not specified by the paper
_GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)  # conservative choice per protocol; not specified by the paper
_GAUSSIAN_BLUR_P = 0.5  # NOT specified by the paper


def _geometric_ops() -> list[nn.Module]:
    return [
        K.RandomHorizontalFlip(p=_FLIP_P, same_on_batch=False),
        K.RandomVerticalFlip(p=_FLIP_P, same_on_batch=False),
        K.RandomRotation(
            degrees=_ROTATION_DEGREES, resample=_ROTATION_RESAMPLE, p=_ROTATION_P, same_on_batch=False
        ),
        K.RandomResizedCrop(
            size=(28, 28),
            scale=_CROP_SCALE,
            ratio=_CROP_RATIO,
            resample=_CROP_RESAMPLE,
            p=_CROP_P,
            same_on_batch=False,
        ),
    ]


def _intensity_ops() -> list[nn.Module]:
    return [
        K.ColorJitter(
            brightness=_JITTER_BRIGHTNESS, contrast=_JITTER_CONTRAST, p=_JITTER_P, same_on_batch=False
        ),
        K.RandomGaussianBlur(
            kernel_size=_GAUSSIAN_BLUR_KERNEL,
            sigma=_GAUSSIAN_BLUR_SIGMA,
            p=_GAUSSIAN_BLUR_P,
            same_on_batch=False,
        ),
    ]


def build_policy(name: str, output_size: tuple[int, int] = (28, 28)) -> nn.Sequential:
    """Build a policy ('geometric', 'intensity', or 'mixed') as an nn.Sequential
    of kornia augmentation ops. output_size controls RandomResizedCrop's target
    size so the policy works at 28/64/128px (see docs/compute_budget.md).

    Execution order (frozen): flips, then rotation, then resized crop, then
    (for 'mixed') color jitter, then Gaussian blur -- i.e. geometric ops
    strictly before intensity ops. Within ColorJitter, kornia internally
    randomizes brightness-vs-contrast application order per call; this is
    not independently configurable in this kornia version and is
    documented here rather than silently left unspecified.
    """
    name = name.lower()
    geometric = _geometric_ops()
    # RandomResizedCrop's size is baked in at construction time; rebuild if
    # output_size differs from the 28x28 default.
    geometric[-1] = K.RandomResizedCrop(
        size=output_size,
        scale=_CROP_SCALE,
        ratio=_CROP_RATIO,
        resample=_CROP_RESAMPLE,
        p=_CROP_P,
        same_on_batch=False,
    )
    intensity = _intensity_ops()

    if name == "geometric":
        ops = geometric
    elif name == "intensity":
        ops = intensity
    elif name == "mixed":
        ops = geometric + intensity  # frozen order: geometric before intensity
    else:
        raise ValueError(f"Unknown policy '{name}'; expected geometric/intensity/mixed.")

    return nn.Sequential(*ops)


def sample_deterministic_view(x: torch.Tensor, policy: nn.Module, seed: int) -> torch.Tensor:
    """Apply `policy` to `x` deterministically for a given integer seed.

    Kornia's augmentation ops draw their random parameters from PyTorch's
    global RNG by default, so seeding torch's RNG immediately before the
    call makes the sampled view reproducible. Each view is generated with a
    distinct seed by the caller (e.g. seed + view_index), so views are
    sampled independently of each other; within one view, a batch's samples
    are also sampled independently (same_on_batch=False on every op above).
    This is a CPU-generator seed; per docs/experimental_protocol.md and
    reproducibility.py, bitwise equivalence between CPU and MPS is not
    guaranteed even so, though same-device (MPS-to-MPS) reproducibility is
    expected and verified in Phase 1 -- see the Phase 1 completion report.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        return policy(x)
