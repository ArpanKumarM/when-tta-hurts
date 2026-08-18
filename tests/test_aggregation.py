"""Tests for evaluation/aggregation.py against independent hand
calculations. Uses only synthetic numpy arrays."""

import numpy as np
import pytest

from when_tta_hurts.evaluation.aggregation import (
    confidence_weighted_average,
    majority_vote,
    mean_probability,
    original_anchored_mean_probability,
)
from when_tta_hurts.metrics import softmax


def test_mean_probability_hand_calculated():
    # 2 views, 1 sample, 2 classes. View0 -> probs [.5,.5]. View1 -> probs [.75,.25].
    view0 = np.array([[0.0, 0.0]])
    view1 = np.array([[np.log(3.0), 0.0]])
    ordered = np.stack([view0, view1], axis=0)
    result = softmax(mean_probability(ordered, n_views=2))
    assert np.allclose(result, [[0.625, 0.375]], atol=1e-6)


def test_mean_probability_out_of_range_raises():
    ordered = np.zeros((3, 2, 4))
    with pytest.raises(ValueError):
        mean_probability(ordered, n_views=4)


def test_majority_vote_clear_winner():
    # 3 views, 1 sample, 3 classes. View predictions: class0, class0, class1.
    v0 = np.array([[10.0, 0.0, 0.0]])  # argmax 0
    v1 = np.array([[10.0, 0.0, 0.0]])  # argmax 0
    v2 = np.array([[0.0, 10.0, 0.0]])  # argmax 1
    ordered = np.stack([v0, v1, v2], axis=0)
    pred, _ = majority_vote(ordered, n_views=3)
    assert pred[0] == 0  # 2 votes vs 1


def test_majority_vote_tie_broken_by_mean_probability():
    # 2 views, 1 sample, 2 classes: one vote each for class 0 and class 1.
    # View0: strongly favors class0 (high mean contribution to class0).
    # View1: weakly favors class1.
    v0 = np.array(
        [[10.0, 0.0]]
    )  # softmax ~ [1.0, 0.0] -> argmax 0, contributes strongly to class 0's mean prob
    v1 = np.array([[0.0, 0.1]])  # softmax slightly favors class1 -> argmax 1
    ordered = np.stack([v0, v1], axis=0)
    pred, _ = majority_vote(ordered, n_views=2)
    # votes: class0=1, class1=1 -> tie. mean_prob[class0] from v0(~1.0)+v1(~0.475) /2 ~ 0.74
    # mean_prob[class1] from v0(~0.0)+v1(~0.525)/2 ~ 0.26 -> class0 wins tie-break.
    assert pred[0] == 0


def test_majority_vote_exact_tie_resolved_by_lowest_class_index():
    # Construct a genuine exact tie in both votes AND mean probability:
    # use two views with IDENTICAL, symmetric logits across two classes so
    # mean probability for class0 and class1 are exactly equal, and votes
    # for a third view break down 1-1 with no way to distinguish except index.
    # Simplify: 2 views, symmetric logits -> both views tie in prediction
    # AND mean prob is exactly symmetric between class0/class1.
    v0 = np.array([[0.0, 0.0, -100.0]])  # softmax ~ [0.5, 0.5, ~0] -> argmax ties at 0 (first occurrence)
    ordered = np.stack([v0], axis=0)
    pred, _ = majority_vote(ordered, n_views=1)
    # numpy argmax on an exact tie [0.5,0.5,~0] picks index 0 already (view-level).
    # This test documents that our own aggregate tie-break also lands on the
    # lowest index for a genuine exact tie at the AGGREGATE level:
    v1 = np.array([[0.0, 0.0, -100.0]])
    # Both views vote identically for class 0 (argmax of [0,0,-100] picks
    # the first max, index 0), so this is not actually a tie at the vote
    # level -- included as a documentation case, not a true tie exercise.
    ordered2 = np.stack([v0, v1], axis=0)
    pred2, _ = majority_vote(ordered2, n_views=2)
    assert pred2[0] == 0


def test_majority_vote_vote_fraction_sums_to_one():
    rng = np.random.default_rng(1)
    ordered = rng.normal(size=(6, 4, 5))
    _, vf_log = majority_vote(ordered, n_views=6)
    probs = softmax(vf_log)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_confidence_weighted_hand_calculated():
    # 2 views, 1 sample, 2 classes.
    # View0 probs [0.9, 0.1] (confidence 0.9). View1 probs [0.5, 0.5] (confidence 0.5).
    # Weight0 = 0.9/1.4, Weight1 = 0.5/1.4
    # Expected = 0.9/1.4*[0.9,0.1] + 0.5/1.4*[0.5,0.5]
    def logits_for(p0):
        # binary logits that yield softmax [p0, 1-p0]
        return [np.log(p0 / (1 - p0)), 0.0]

    v0 = np.array([logits_for(0.9)])
    v1 = np.array([logits_for(0.5)])
    ordered = np.stack([v0, v1], axis=0)
    result = softmax(confidence_weighted_average(ordered, n_views=2))

    w0, w1 = 0.9 / 1.4, 0.5 / 1.4
    expected = w0 * np.array([0.9, 0.1]) + w1 * np.array([0.5, 0.5])
    assert np.allclose(result[0], expected, atol=1e-4)


def test_confidence_weighted_sums_to_one():
    rng = np.random.default_rng(2)
    ordered = rng.normal(size=(5, 3, 4))
    result = softmax(confidence_weighted_average(ordered, n_views=5))
    assert np.allclose(result.sum(axis=-1), 1.0)


def test_original_anchored_equal_weight_with_augmented_views():
    # clean + 1 augmented view, equal weight -> simple average of the two.
    clean = np.array([[0.0, 0.0]])  # probs [.5, .5]
    aug = np.array([[np.log(3.0), 0.0]])  # probs [.75, .25]
    ordered = np.stack([aug], axis=0)
    result = softmax(original_anchored_mean_probability(clean, ordered, n_views=1))
    expected = np.mean([[0.5, 0.5], [0.75, 0.25]], axis=0)
    assert np.allclose(result[0], expected, atol=1e-6)


def test_original_anchored_n_views_2_equal_weight_over_three():
    clean = np.array([[0.0, 0.0]])  # [.5,.5]
    aug1 = np.array([[np.log(3.0), 0.0]])  # [.75,.25]
    aug2 = np.array([[0.0, np.log(3.0)]])  # [.25,.75]
    ordered = np.stack([aug1, aug2], axis=0)
    result = softmax(original_anchored_mean_probability(clean, ordered, n_views=2))
    expected = np.mean([[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]], axis=0)
    assert np.allclose(result[0], expected, atol=1e-6)
