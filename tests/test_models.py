import pytest
import torch
from torch import nn

from when_tta_hurts.models import build_resnet18_small_input
from when_tta_hurts.models.small_cnn import GROUPNORM_GROUPS, build_small_cnn

# Reproduction target from the source paper (arXiv:2604.09697, Table 1):
# SmallCNN ~95K params, ResNet-18 ~11M params. SmallCNN is a
# "paper-constrained reimplementation" (see models/small_cnn.py docstring)
# using conventional 32/64/128 channel widths chosen for resolution
# independence, not tuned to hit ~95K exactly -- hence the wider tolerance.
SMALL_CNN_TARGET_PARAMS = 95_000
SMALL_CNN_TOLERANCE = 0.20  # 20%: paper only reports "~95K", not exact widths
SMALL_CNN_EXACT_PARAMS_9CLASS = 94_857  # measured; num_classes=9 (PathMNIST)
RESNET18_TARGET_PARAMS = 11_000_000
RESNET18_TOLERANCE = 0.10


def test_small_cnn_output_shape_batchnorm():
    model = build_small_cnn(num_classes=9, input_size=28, normalization="batchnorm")
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 9)


def test_small_cnn_output_shape_groupnorm():
    model = build_small_cnn(num_classes=8, input_size=28, normalization="groupnorm")
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 8)


def test_small_cnn_normalization_selection():
    bn_model = build_small_cnn(num_classes=9, normalization="batchnorm")
    gn_model = build_small_cnn(num_classes=9, normalization="groupnorm")
    bn_norm_layers = [m for m in bn_model.modules() if isinstance(m, nn.BatchNorm2d)]
    gn_norm_layers = [m for m in gn_model.modules() if isinstance(m, nn.GroupNorm)]
    assert len(bn_norm_layers) == 3
    assert len(gn_norm_layers) == 3
    assert not any(isinstance(m, nn.GroupNorm) for m in bn_model.modules())
    assert not any(isinstance(m, nn.BatchNorm2d) for m in gn_model.modules())


def test_groupnorm_group_count_divides_all_channel_widths():
    from when_tta_hurts.models.small_cnn import DEFAULT_CHANNELS

    for width in DEFAULT_CHANNELS:
        assert width % GROUPNORM_GROUPS == 0, (
            f"GROUPNORM_GROUPS={GROUPNORM_GROUPS} must evenly divide every "
            f"channel width; {width} is not divisible."
        )


def test_small_cnn_invalid_normalization_raises():
    with pytest.raises(ValueError):
        build_small_cnn(num_classes=9, normalization="layernorm")


@pytest.mark.parametrize("resolution", [28, 64, 128])
@pytest.mark.parametrize("normalization", ["batchnorm", "groupnorm"])
def test_small_cnn_param_count_is_resolution_independent(resolution, normalization):
    """H2 correctness requirement: the 28/64/128 comparison must use a model
    with an identical parameter count at every resolution, or any observed
    TTA-degradation difference across resolutions is confounded by model
    capacity differing across resolutions."""
    model = build_small_cnn(num_classes=9, input_size=resolution, normalization=normalization)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == SMALL_CNN_EXACT_PARAMS_9CLASS, (
        f"SmallCNN({normalization}) at {resolution}px has {n_params:,} params, "
        f"expected exactly {SMALL_CNN_EXACT_PARAMS_9CLASS:,} at every resolution."
    )


def test_small_cnn_batchnorm_and_groupnorm_have_equal_param_counts():
    bn_model = build_small_cnn(num_classes=9, normalization="batchnorm")
    gn_model = build_small_cnn(num_classes=9, normalization="groupnorm")
    n_bn = sum(p.numel() for p in bn_model.parameters())
    n_gn = sum(p.numel() for p in gn_model.parameters())
    assert n_bn == n_gn, f"BatchNorm has {n_bn:,} params, GroupNorm has {n_gn:,} -- must be equal for H1."


def test_small_cnn_param_count_within_tolerance_of_paper_target():
    model = build_small_cnn(num_classes=9, input_size=28, normalization="batchnorm")
    n_params = sum(p.numel() for p in model.parameters())
    lower = SMALL_CNN_TARGET_PARAMS * (1 - SMALL_CNN_TOLERANCE)
    upper = SMALL_CNN_TARGET_PARAMS * (1 + SMALL_CNN_TOLERANCE)
    assert lower <= n_params <= upper, (
        f"SmallCNN has {n_params:,} params, target ~{SMALL_CNN_TARGET_PARAMS:,} "
        f"+/-{SMALL_CNN_TOLERANCE:.0%}. Per CLAUDE.md/experimental_protocol.md, "
        f"a material mismatch must be investigated, not silently accepted."
    )


def test_resnet18_output_shape():
    model = build_resnet18_small_input(num_classes=9)
    x = torch.randn(2, 3, 28, 28)
    out = model(x)
    assert out.shape == (2, 9)


def test_resnet18_has_no_initial_maxpool():
    model = build_resnet18_small_input(num_classes=9)
    assert isinstance(model.maxpool, nn.Identity)


def test_resnet18_param_count_within_tolerance_of_paper_target():
    model = build_resnet18_small_input(num_classes=9)
    n_params = sum(p.numel() for p in model.parameters())
    lower = RESNET18_TARGET_PARAMS * (1 - RESNET18_TOLERANCE)
    upper = RESNET18_TARGET_PARAMS * (1 + RESNET18_TOLERANCE)
    assert lower <= n_params <= upper, f"ResNet18 has {n_params:,} params, target ~{RESNET18_TARGET_PARAMS:,}"
