import numpy as np
import pytest
import torch

from when_tta_hurts.evaluation.tta import aggregate_mean_prefix, compute_ordered_view_logits
from when_tta_hurts.metrics import softmax
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.transforms import build_policy


def test_compute_ordered_view_logits_shape():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    x = torch.rand(4, 3, 28, 28)
    policy = build_policy("mixed")
    view_logits = compute_ordered_view_logits(model, x, policy, tta_seed=271828, max_views=5, device=device)
    assert view_logits.shape == (5, 4, 9)


def test_compute_ordered_view_logits_deterministic():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    x = torch.rand(2, 3, 28, 28)
    policy = build_policy("mixed")
    a = compute_ordered_view_logits(model, x, policy, tta_seed=1, max_views=3, device=device)
    b = compute_ordered_view_logits(model, x, policy, tta_seed=1, max_views=3, device=device)
    assert np.allclose(a, b)


def test_nested_prefix_property():
    """A key correctness requirement: the first n views of a max_views
    sequence must be IDENTICAL to a sequence generated with max_views=n
    directly (same seeds -> same views -> same logits), not resampled."""
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    x = torch.rand(2, 3, 28, 28)
    policy = build_policy("mixed")

    full = compute_ordered_view_logits(model, x, policy, tta_seed=5, max_views=10, device=device)
    prefix_5 = compute_ordered_view_logits(model, x, policy, tta_seed=5, max_views=5, device=device)

    assert np.allclose(full[:5], prefix_5)


def test_aggregate_mean_prefix_matches_manual_mean_probability():
    rng = np.random.default_rng(0)
    ordered_logits = rng.normal(size=(10, 3, 9))  # [views, N, C]

    n = 4
    agg_log_probs = aggregate_mean_prefix(ordered_logits, n_views=n)
    recovered_probs = softmax(agg_log_probs)

    manual_probs = np.stack([softmax(v) for v in ordered_logits[:n]], axis=0).mean(axis=0)
    assert np.allclose(recovered_probs, manual_probs, atol=1e-6)


def test_aggregate_mean_prefix_out_of_range_raises():
    ordered_logits = np.zeros((5, 2, 9))
    with pytest.raises(ValueError):
        aggregate_mean_prefix(ordered_logits, n_views=6)
    with pytest.raises(ValueError):
        aggregate_mean_prefix(ordered_logits, n_views=0)


def test_aggregate_mean_prefix_single_view_recovers_original_probs():
    rng = np.random.default_rng(1)
    ordered_logits = rng.normal(size=(3, 2, 9))
    agg_log_probs = aggregate_mean_prefix(ordered_logits, n_views=1)
    recovered = softmax(agg_log_probs)
    expected = softmax(ordered_logits[0])
    assert np.allclose(recovered, expected, atol=1e-6)
