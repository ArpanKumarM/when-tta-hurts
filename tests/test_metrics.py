import numpy as np

from when_tta_hurts.metrics import (
    accuracy,
    compute_all_metrics,
    expected_calibration_error,
    harm_rescue_rates,
    macro_f1,
    negative_log_likelihood,
    softmax,
)


def test_softmax_sums_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_accuracy_perfect_predictions():
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    labels = np.array([0, 1, 0])
    assert accuracy(logits, labels) == 1.0


def test_accuracy_all_wrong():
    logits = np.array([[10.0, 0.0], [10.0, 0.0]])
    labels = np.array([1, 1])
    assert accuracy(logits, labels) == 0.0


def test_macro_f1_perfect():
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1, 0, 1])
    assert macro_f1(logits, labels) == 1.0


def test_nll_lower_for_confident_correct_predictions():
    confident_correct = np.array([[10.0, 0.0]])
    unsure = np.array([[0.1, 0.0]])
    labels = np.array([0])
    assert negative_log_likelihood(confident_correct, labels) < negative_log_likelihood(unsure, labels)


def test_ece_zero_when_perfectly_calibrated_and_correct():
    # Extremely confident and always correct -> low ECE (near 0, some
    # binning slack allowed).
    logits = np.tile([10.0, -10.0], (20, 1))
    labels = np.zeros(20, dtype=int)
    ece = expected_calibration_error(logits, labels)
    assert ece < 0.05


def test_harm_rescue_rates_basic():
    labels = np.array([0, 0, 1, 1])
    # clean predicts [0,0,1,1] (all correct); tta predicts [1,0,1,0] (harms sample 0 and 3)
    clean_logits = np.array([[10, 0], [10, 0], [0, 10], [0, 10]])
    tta_logits = np.array([[0, 10], [10, 0], [0, 10], [10, 0]])
    result = harm_rescue_rates(clean_logits, tta_logits, labels)
    assert result["n_clean_correct"] == 4
    assert result["n_clean_wrong"] == 0
    assert result["n_harmed"] == 2
    assert result["harm_rate"] == 0.5
    assert result["rescue_rate"] == 0.0


def test_harm_rescue_rates_rescue_case():
    labels = np.array([0, 1])
    # clean predicts [1,0] (both wrong); tta predicts [0,1] (both rescued)
    clean_logits = np.array([[0, 10], [10, 0]])
    tta_logits = np.array([[10, 0], [0, 10]])
    result = harm_rescue_rates(clean_logits, tta_logits, labels)
    assert result["n_clean_wrong"] == 2
    assert result["n_rescued"] == 2
    assert result["rescue_rate"] == 1.0
    assert result["harm_rate"] == 0.0


def test_compute_all_metrics_has_expected_keys():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1])
    out = compute_all_metrics(logits, labels)
    assert set(out.keys()) == {
        "accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "expected_calibration_error",
    }
