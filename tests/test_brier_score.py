import numpy as np

from when_tta_hurts.metrics import brier_score


def test_brier_score_perfect_confident_correct_is_zero():
    logits = np.array([[100.0, -100.0], [-100.0, 100.0]])
    labels = np.array([0, 1])
    assert brier_score(logits, labels) < 1e-6


def test_brier_score_hand_calculated_two_class():
    # 1 sample, 2 classes, softmax([0,0]) = [0.5, 0.5], true label = 0.
    # one_hot = [1, 0]. sum((p - onehot)^2) = (0.5-1)^2 + (0.5-0)^2 = 0.25+0.25=0.5
    logits = np.array([[0.0, 0.0]])
    labels = np.array([0])
    assert np.isclose(brier_score(logits, labels), 0.5, atol=1e-6)


def test_brier_score_worst_case_confident_wrong():
    # Confident wrong prediction: predicted [~1, ~0], true label = 1.
    # sum((1-0)^2 + (0-1)^2) = 1 + 1 = 2 (worst possible for 2-class)
    logits = np.array([[100.0, -100.0]])
    labels = np.array([1])
    assert np.isclose(brier_score(logits, labels), 2.0, atol=1e-4)


def test_brier_score_matches_manual_multiclass_formula():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 5))
    labels = rng.integers(0, 5, size=20)

    from when_tta_hurts.metrics import softmax

    probs = softmax(logits)
    one_hot = np.zeros((20, 5))
    one_hot[np.arange(20), labels] = 1.0
    manual = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))

    assert np.isclose(brier_score(logits, labels), manual, atol=1e-10)


def test_brier_score_is_averaged_over_samples():
    logits = np.array([[100.0, -100.0], [0.0, 0.0]])
    labels = np.array([0, 0])
    # sample0: brier ~0, sample1: brier = (0.5-1)^2+(0.5-0)^2 = 0.5
    expected = (0.0 + 0.5) / 2
    assert np.isclose(brier_score(logits, labels), expected, atol=1e-3)
