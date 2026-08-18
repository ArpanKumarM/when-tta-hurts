"""Tests for evaluation/bn_adaptation.py. Uses synthetic tensors and the
already-tested SmallCNN builder only."""

import torch

from when_tta_hurts.evaluation.bn_adaptation import BNAdaptationNotApplicableError, bn_adapt
from when_tta_hurts.models.small_cnn import build_small_cnn


def test_bn_adapt_changes_running_stats_only():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)

    bn_layers_before = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    running_means_before = [bn.running_mean.clone() for bn in bn_layers_before]

    adapted = bn_adapt(model, x)

    bn_layers_after = [m for m in adapted.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    running_means_after = [bn.running_mean.clone() for bn in bn_layers_after]

    assert any(
        not torch.equal(before, after) for before, after in zip(running_means_before, running_means_after)
    )


def test_bn_adapt_does_not_mutate_original_model():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    bn_adapt(model, x)

    for k, v in model.state_dict().items():
        assert torch.equal(v, original_state[k]), f"original model mutated at {k}"


def test_bn_adapt_no_learned_parameter_changes():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)
    params_before = {name: p.clone() for name, p in model.named_parameters()}

    adapted = bn_adapt(model, x)

    for name, p_after in adapted.named_parameters():
        assert torch.equal(p_after, params_before[name]), f"learned parameter {name} changed"


def test_bn_adapt_returns_eval_mode_model():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)
    adapted = bn_adapt(model, x)
    assert not adapted.training


def test_bn_adapt_rejects_groupnorm():
    model = build_small_cnn(num_classes=9, normalization="groupnorm")
    x = torch.rand(16, 3, 28, 28)
    try:
        bn_adapt(model, x)
        raise AssertionError("expected BNAdaptationNotApplicableError")
    except BNAdaptationNotApplicableError:
        pass


def test_bn_adapt_resets_independently_across_calls():
    """Two separate bn_adapt() calls on the SAME original model with
    DIFFERENT input batches must not carry state between them -- each
    starts from the untouched original checkpoint."""
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x1 = torch.rand(16, 3, 28, 28)
    x2 = torch.rand(16, 3, 28, 28) + 5.0  # very different distribution

    adapted1 = bn_adapt(model, x1)
    adapted2 = bn_adapt(model, x2)  # from the SAME original `model`, not adapted1

    rm1 = adapted1.features[1].running_mean.clone()
    rm2 = adapted2.features[1].running_mean.clone()
    assert not torch.equal(rm1, rm2)  # different adaptation inputs -> different stats

    # Critically: adapting again with x1 after adapting with x2 (in either
    # order, from the ORIGINAL model) reproduces the SAME result as the
    # first x1 adaptation -- proving no state leaked across calls.
    adapted1_again = bn_adapt(model, x1)
    assert torch.equal(adapted1_again.features[1].running_mean, rm1)


def test_bn_adapt_never_uses_labels():
    """Structural check: bn_adapt()'s signature has no labels/targets
    parameter at all."""
    import inspect

    sig = inspect.signature(bn_adapt)
    param_names = set(sig.parameters.keys())
    assert "labels" not in param_names
    assert "targets" not in param_names
    assert "y" not in param_names
