"""Tests for the frozen N=25 secondary analyses (augmentation-strategy and
aggregation ablations, docs/phase2b_protocol.md sec.3) and nested-prefix
equivalence for every new aggregation method at N=25, using synthetic
data and the CPU-only transform policies already verified in Phase 2A."""

import numpy as np
import torch

from when_tta_hurts.evaluation.aggregation import (
    confidence_weighted_average,
    majority_vote,
    mean_probability,
    original_anchored_mean_probability,
)
from when_tta_hurts.evaluation.tta import compute_ordered_view_logits
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.transforms import build_policy


def _ordered_logits(n_views=100, n_samples=4, n_classes=9, seed=7):
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=n_classes, normalization="batchnorm").to(device)
    x = torch.rand(n_samples, 3, 28, 28)
    policy = build_policy("mixed")
    return compute_ordered_view_logits(model, x, policy, tta_seed=seed, max_views=n_views, device=device)


def test_mean_probability_n25_is_prefix_of_n100():
    ordered = _ordered_logits(n_views=100)
    full_100 = mean_probability(ordered, n_views=100)
    prefix_25 = mean_probability(ordered[:25], n_views=25)
    direct_25 = mean_probability(ordered, n_views=25)
    assert np.allclose(prefix_25, direct_25)
    assert not np.allclose(full_100, direct_25)  # sanity: genuinely different aggregates


def test_majority_vote_n25_is_prefix_of_n100():
    ordered = _ordered_logits(n_views=100)
    pred_25_a, vf_25_a = majority_vote(ordered[:25], n_views=25)
    pred_25_b, vf_25_b = majority_vote(ordered, n_views=25)
    assert np.array_equal(pred_25_a, pred_25_b)
    assert np.allclose(vf_25_a, vf_25_b)


def test_confidence_weighted_n25_is_prefix_of_n100():
    ordered = _ordered_logits(n_views=100)
    a = confidence_weighted_average(ordered[:25], n_views=25)
    b = confidence_weighted_average(ordered, n_views=25)
    assert np.allclose(a, b)


def test_original_anchored_n25_is_prefix_of_n100():
    # Phase 2B.6J: original_anchored_mean_probability now takes the clean
    # PROBABILITY array directly (must sum to 1 per row) -- see
    # docs/phase2b_final_test_semantic_metric_contract_freeze.md.
    from when_tta_hurts.metrics import softmax

    ordered = _ordered_logits(n_views=100, n_samples=4)
    clean = softmax(torch.rand(4, 9).numpy())
    a = original_anchored_mean_probability(clean, ordered[:25], n_views=25)
    b = original_anchored_mean_probability(clean, ordered, n_views=25)
    assert np.allclose(a, b)


def test_geometric_intensity_mixed_policies_all_produce_valid_views_at_n25():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    x = torch.rand(2, 3, 28, 28)
    for policy_name in ("geometric", "intensity", "mixed"):
        policy = build_policy(policy_name)
        ordered = compute_ordered_view_logits(model, x, policy, tta_seed=1, max_views=25, device=device)
        assert ordered.shape == (25, 2, 9)
        assert np.isfinite(ordered).all()


def test_aggregation_ablation_all_three_methods_run_at_n25():
    ordered = _ordered_logits(n_views=25, n_samples=3)
    mp = mean_probability(ordered, n_views=25)
    mv_pred, mv_probs = majority_vote(ordered, n_views=25)
    cw = confidence_weighted_average(ordered, n_views=25)
    assert mp.shape == (3, 9)
    assert mv_pred.shape == (3,)
    assert mv_probs.shape == (3, 9)
    assert cw.shape == (3, 9)
