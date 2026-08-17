"""SmallCNN: a PAPER-CONSTRAINED REIMPLEMENTATION, not an exact reproduction
of unavailable source code.

The source paper (arXiv:2604.09697, Table 1) specifies only "3-layer CNN,
BatchNorm, MaxPool, ~95K params" -- it does not disclose channel widths,
head design, or how spatial size is reduced to a fixed-size feature vector.
Because no reference implementation is available (see
docs/data_and_licensing.md -- the paper's linked code repo 404s), this is
necessarily an independent reimplementation constrained by the paper's
stated facts, not a byte-for-byte reproduction.

RESOLUTION-INDEPENDENCE REQUIREMENT (H2 correctness): the 28-vs-64(-vs-128)
resolution comparison requires the SAME trainable parameter count at every
resolution, so that any TTA-degradation difference across resolutions is
not confounded by the model having a different number of parameters. A
flatten-based head (the Phase-1-draft version of this file) does NOT
satisfy this, because a flattened conv feature map's size scales with input
resolution. This version uses global average pooling instead, which is
resolution-independent by construction.

IMPLEMENTATION CHOICES (not specified by the source paper -- documented
here rather than presented as paper-verified):
  - 3x3 conv kernels, padding=1, stride=1 for all three conv layers.
  - Conventional channel widths 32 -> 64 -> 128 (not tuned to hit any
    specific parameter-count target; the paper's "~95K" is approximate and
    resolution-invariance takes priority over matching it precisely -- see
    the measured count in tests/test_models.py).
  - MaxPool(kernel=2, stride=2) after each of the first two conv blocks,
    per the paper's "MaxPool" description.
  - Adaptive average pooling to 1x1 after the third conv block, THEN a
    single linear layer 128 -> num_classes. This is what makes the
    architecture resolution-independent; the paper does not specify how
    spatial reduction to a classifier head is done.
  - GroupNorm uses a fixed 8 groups, which evenly divides all three channel
    widths (32, 64, 128), so a valid grouping exists at every layer without
    per-resolution or per-layer adjustment.
  - Convolution bias is enabled (nn.Conv2d default, bias=True) for BOTH the
    BatchNorm and GroupNorm variants, specifically so the two normalization
    choices have IDENTICAL trainable parameter counts (some
    BatchNorm-specific implementations omit conv bias since BatchNorm can
    absorb it; that asymmetry is deliberately avoided here).
"""

from __future__ import annotations

import torch
from torch import nn

# Fixed GroupNorm group count: must evenly divide every channel width used
# (32, 64, 128). 8 does: 32/8=4, 64/8=8, 128/8=16.
GROUPNORM_GROUPS = 8

# Conventional channel widths; NOT tuned to hit a specific parameter count
# (see module docstring). Resolution-independent by construction via
# adaptive average pooling, so this tuple fully determines the parameter
# count regardless of input_size.
DEFAULT_CHANNELS: tuple[int, int, int] = (32, 64, 128)


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if norm == "groupnorm":
        if num_channels % GROUPNORM_GROUPS != 0:
            raise ValueError(
                f"GROUPNORM_GROUPS={GROUPNORM_GROUPS} does not evenly divide "
                f"num_channels={num_channels}; fix DEFAULT_CHANNELS or GROUPNORM_GROUPS."
            )
        return nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=num_channels)
    raise ValueError(f"Unknown normalization '{norm}'; expected 'batchnorm' or 'groupnorm'.")


class SmallCNN(nn.Module):
    """Paper-constrained reimplementation of the source paper's SmallCNN.

    Resolution-independent: parameter count does NOT depend on input_size,
    because global average pooling replaces a flatten-based head. See
    module docstring for which architectural details are implementation
    choices vs. paper-specified, and why this matters for H2.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        normalization: str = "batchnorm",
        channels: tuple[int, int, int] = DEFAULT_CHANNELS,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def build_small_cnn(
    num_classes: int,
    input_size: int = 28,
    in_channels: int = 3,
    normalization: str = "batchnorm",
) -> SmallCNN:
    """Construct a SmallCNN. `input_size` is accepted for interface
    compatibility with callers that pass it (e.g. scripts/smoke_test.py)
    but does NOT affect parameter count or architecture, by design -- see
    the resolution-independence requirement in the module docstring.
    """
    del input_size  # intentionally unused: resolution-independent by design
    return SmallCNN(num_classes=num_classes, in_channels=in_channels, normalization=normalization)
