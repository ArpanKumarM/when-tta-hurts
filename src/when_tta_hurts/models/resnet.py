"""ResNet-18 adapted for 28x28 input, per the source paper's Table 1 spec:
"~11M params, adapted for 28x28 (no initial pool)".

IMPLEMENTATION CHOICES (not specified by the source paper beyond "no initial
pool"): we follow the standard small-image ResNet adaptation used widely in
CIFAR-style literature -- replace the stem's 7x7 stride-2 conv with a 3x3
stride-1 conv, and replace the stem's MaxPool with nn.Identity (rather than
deleting it) so the module structure stays introspectable. Everything else
(stage widths/depths, block structure) is torchvision's standard ResNet-18.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def build_resnet18_small_input(num_classes: int, in_channels: int = 3) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # "no initial pooling layer" per the source paper.
    model.maxpool = nn.Identity()

    return model
