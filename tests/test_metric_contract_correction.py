"""Phase 2B.4D Metric-Contract Correction (Part H): tests for the
probability_native_v1 metric-input contract fix. All synthetic tensors and
temporary repositories only; attempt-3 persisted predictions are used
read-only for diagnosis and are never modified. No real MPS, checkpoint,
dataset, or test-split path is exercised anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from when_tta_hurts.evaluation.aggregation import (
    confidence_weighted_average,
    majority_vote,
    mean_probability,
    original_anchored_mean_probability,
)
from when_tta_hurts.evaluation_result_artifacts import EvaluationPersistenceError
from when_tta_hurts.metrics import (
    InvalidProbabilityArrayError,
    accuracy,
    brier_score,
    compute_metrics_from_logits,
    compute_metrics_from_probabilities,
    macro_f1,
    negative_log_likelihood,
    softmax,
    validate_probability_array,
)
from when_tta_hurts.validation_evaluation import (
    AGGREGATORS,
    PREFIX_SEQUENCE,
    _per_prefix_metrics,
    _recompute_all_conditions_from_predictions,
    _verify_metrics_semantically,
)

# ---------------------------------------------------------------------------
# 1-3: hand-calculated probability-native NLL/Brier/ECE
# ---------------------------------------------------------------------------

_P = np.array([[0.9, 0.1]])
_LABELS0 = np.array([0])


def test_hand_calculated_probability_native_nll():
    expected = -np.log(0.9)
    assert np.isclose(compute_metrics_from_probabilities(_P, _LABELS0)["negative_log_likelihood"], expected)


def test_hand_calculated_probability_native_brier():
    # Brier = (0.9-1)^2 + (0.1-0)^2 = 0.02
    expected = 0.01 + 0.01
    assert np.isclose(compute_metrics_from_probabilities(_P, _LABELS0)["brier_score"], expected)


def test_hand_calculated_probability_native_ece():
    # Single sample, confidence 0.9, correct -> |acc-conf| = |1.0-0.9| = 0.1
    expected = 0.1
    ece = compute_metrics_from_probabilities(_P, _LABELS0)["expected_calibration_error"]
    assert np.isclose(ece, expected)


# ---------------------------------------------------------------------------
# 4: logit-native metrics equal softmax-once probability-native metrics
# ---------------------------------------------------------------------------


def test_logit_native_equals_softmax_once_probability_native():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(30, 4))
    labels = rng.integers(0, 4, size=30)
    from_logits = compute_metrics_from_logits(logits, labels)
    from_probs = compute_metrics_from_probabilities(softmax(logits), labels)
    for key in from_logits:
        assert np.isclose(from_logits[key], from_probs[key], atol=1e-10)


# ---------------------------------------------------------------------------
# 5: extra/double softmax produces a DIFFERENT result in a known example
# ---------------------------------------------------------------------------


def test_double_softmax_produces_different_nll_and_brier():
    correct = compute_metrics_from_probabilities(_P, _LABELS0)
    once_extra = negative_log_likelihood(_P, _LABELS0)  # applies softmax(p) internally -- wrong
    assert not np.isclose(correct["negative_log_likelihood"], once_extra)
    twice_extra = negative_log_likelihood(softmax(_P), _LABELS0)
    assert not np.isclose(correct["negative_log_likelihood"], twice_extra)

    correct_brier = correct["brier_score"]
    once_extra_brier = brier_score(_P, _LABELS0)
    assert not np.isclose(correct_brier, once_extra_brier)


# ---------------------------------------------------------------------------
# 6: accuracy/F1 invariant to reapplied softmax, but this is NOT calibration
# correctness -- NLL/ECE/Brier still diverge for the identical input.
# ---------------------------------------------------------------------------


def test_accuracy_f1_invariant_but_calibration_metrics_are_not():
    rng = np.random.default_rng(1)
    probs = softmax(rng.normal(size=(40, 5)))
    labels = rng.integers(0, 5, size=40)

    correct = compute_metrics_from_probabilities(probs, labels)
    once_extra_acc = accuracy(probs, labels)
    once_extra_f1 = macro_f1(probs, labels)
    once_extra_nll = negative_log_likelihood(probs, labels)

    assert np.isclose(correct["accuracy"], once_extra_acc)
    assert np.isclose(correct["macro_f1"], once_extra_f1)
    assert not np.isclose(correct["negative_log_likelihood"], once_extra_nll)


# ---------------------------------------------------------------------------
# 7-9: mean-probability / original-anchoring / confidence-weighted receive
# no post-aggregation softmax before metric computation
# ---------------------------------------------------------------------------


def _view_log_probs(rng, n_views=3, n_samples=5, n_classes=3):
    probs = softmax(rng.normal(size=(n_views, n_samples, n_classes)))
    return np.log(np.clip(probs, 1e-12, 1.0))


def test_mean_probability_receives_no_post_aggregation_softmax():
    rng = np.random.default_rng(2)
    view_log_probs = _view_log_probs(rng)
    labels = rng.integers(0, 3, size=5)
    clean_probs = softmax(rng.normal(size=(5, 3)))

    agg = softmax(mean_probability(view_log_probs, 3))
    assert np.allclose(agg.sum(axis=-1), 1.0)
    result = _per_prefix_metrics(clean_probs, agg, labels)
    expected = compute_metrics_from_probabilities(agg, labels)
    assert np.isclose(result["negative_log_likelihood"], expected["negative_log_likelihood"])


def test_original_anchoring_receives_no_post_aggregation_softmax():
    rng = np.random.default_rng(3)
    view_log_probs = _view_log_probs(rng)
    labels = rng.integers(0, 3, size=5)
    clean_probs = softmax(rng.normal(size=(5, 3)))
    clean_logp = np.log(np.clip(clean_probs, 1e-12, 1.0))

    agg = softmax(original_anchored_mean_probability(clean_logp, view_log_probs, 3))
    result = _per_prefix_metrics(clean_probs, agg, labels)
    expected = compute_metrics_from_probabilities(agg, labels)
    assert np.isclose(result["brier_score"], expected["brier_score"])


def test_confidence_weighted_aggregation_follows_probability_native_contract():
    rng = np.random.default_rng(4)
    view_log_probs = _view_log_probs(rng)
    labels = rng.integers(0, 3, size=5)
    clean_probs = softmax(rng.normal(size=(5, 3)))

    agg = softmax(confidence_weighted_average(view_log_probs, 3))
    result = _per_prefix_metrics(clean_probs, agg, labels)
    expected = compute_metrics_from_probabilities(agg, labels)
    assert np.isclose(result["expected_calibration_error"], expected["expected_calibration_error"])


# ---------------------------------------------------------------------------
# 10: majority-vote / tie-breaking behavior unchanged by this correction
# ---------------------------------------------------------------------------


def test_majority_vote_tie_break_unchanged():
    logits = np.log(
        np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.1, 0.9], [0.3, 0.7]],
            ]
        )
    )
    predicted, _ = majority_vote(logits, 2)
    assert predicted[0] == 0
    assert predicted[1] == 1


# ---------------------------------------------------------------------------
# 11: clean and BN-adapted model outputs receive exactly one softmax
# ---------------------------------------------------------------------------


def test_clean_and_bn_adapted_receive_exactly_one_softmax():
    rng = np.random.default_rng(5)
    clean_logits = rng.normal(size=(6, 3))
    bn_logits = rng.normal(size=(6, 3))
    labels = rng.integers(0, 3, size=6)

    clean_probs = softmax(clean_logits)
    bn_probs = softmax(bn_logits)
    validate_probability_array(clean_probs)
    validate_probability_array(bn_probs)

    clean_metrics = _per_prefix_metrics(clean_probs, clean_probs, labels)
    bn_metrics = _per_prefix_metrics(clean_probs, bn_probs, labels)
    clean_expected = compute_metrics_from_logits(clean_logits, labels)["negative_log_likelihood"]
    bn_expected = compute_metrics_from_logits(bn_logits, labels)["negative_log_likelihood"]
    assert np.isclose(clean_metrics["negative_log_likelihood"], clean_expected)
    assert np.isclose(bn_metrics["negative_log_likelihood"], bn_expected)


# ---------------------------------------------------------------------------
# 12: semantic persistence catches corrupted NLL/ECE/Brier
# ---------------------------------------------------------------------------


def _fake_predictions(rng, n_samples=6, n_classes=3, n_views=100):
    labels = rng.integers(0, n_classes, size=n_samples)
    clean_probs = softmax(rng.normal(size=(n_samples, n_classes))).astype(np.float32)
    view_probs = softmax(rng.normal(size=(n_views, n_samples, n_classes))).astype(np.float32)
    return {"labels": labels, "clean_probs": clean_probs, "view_probs": view_probs}


def _fake_metrics(predictions, prefix_sequence):
    conditions = _recompute_all_conditions_from_predictions(predictions, prefix_sequence)
    from when_tta_hurts.metrics import compute_metrics_from_probabilities as _cmp

    clean = _cmp(predictions["clean_probs"], predictions["labels"])
    return {
        "clean": clean,
        "conditions": {
            "naive_tta": {agg: dict(conditions["naive_tta"][agg]) for agg in AGGREGATORS},
            "original_anchored_tta": dict(conditions["original_anchored_tta"]),
            "bn_adapted_tta": None,
        },
    }


def test_semantic_verification_passes_on_consistent_metrics():
    rng = np.random.default_rng(6)
    predictions = _fake_predictions(rng)
    metrics = _fake_metrics(predictions, PREFIX_SEQUENCE)
    _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)  # must not raise


def test_semantic_verification_catches_corrupted_nll():
    rng = np.random.default_rng(7)
    predictions = _fake_predictions(rng)
    metrics = _fake_metrics(predictions, PREFIX_SEQUENCE)
    metrics["conditions"]["naive_tta"]["mean_probability"][50]["negative_log_likelihood"] += 5.0
    with pytest.raises(EvaluationPersistenceError):
        _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)


def test_semantic_verification_catches_corrupted_ece_and_brier():
    rng = np.random.default_rng(8)
    predictions = _fake_predictions(rng)
    metrics = _fake_metrics(predictions, PREFIX_SEQUENCE)
    metrics["conditions"]["original_anchored_tta"][10]["expected_calibration_error"] += 1.0
    with pytest.raises(EvaluationPersistenceError):
        _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)

    metrics2 = _fake_metrics(predictions, PREFIX_SEQUENCE)
    metrics2["conditions"]["naive_tta"]["majority_vote"][1]["brier_score"] += 1.0
    with pytest.raises(EvaluationPersistenceError):
        _verify_metrics_semantically(predictions, metrics2, PREFIX_SEQUENCE)


# ---------------------------------------------------------------------------
# 13: every reported BN-adapted metric has persisted recomputation evidence
# ---------------------------------------------------------------------------


def test_bn_adapted_metrics_have_persisted_recomputation_evidence():
    rng = np.random.default_rng(9)
    predictions = _fake_predictions(rng)
    n_samples, n_classes = predictions["clean_probs"].shape
    bn_stack = softmax(rng.normal(size=(len(PREFIX_SEQUENCE), n_samples, n_classes))).astype(np.float32)
    predictions["bn_adapted_probs"] = bn_stack
    predictions["bn_adapted_prefix_sequence"] = np.array(PREFIX_SEQUENCE, dtype=np.int64)

    conditions = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)
    assert conditions["bn_adapted_tta"] is not None
    assert set(conditions["bn_adapted_tta"].keys()) == set(PREFIX_SEQUENCE)


def test_semantic_verification_requires_bn_evidence_if_metrics_report_it():
    rng = np.random.default_rng(10)
    predictions = _fake_predictions(rng)
    n_samples, n_classes = predictions["clean_probs"].shape
    bn_stack = softmax(rng.normal(size=(len(PREFIX_SEQUENCE), n_samples, n_classes))).astype(np.float32)
    predictions["bn_adapted_probs"] = bn_stack
    predictions["bn_adapted_prefix_sequence"] = np.array(PREFIX_SEQUENCE, dtype=np.int64)

    metrics = _fake_metrics(predictions, PREFIX_SEQUENCE)
    # metrics claims bn_adapted_tta is None even though predictions has evidence for it --
    # not an error by itself (recomputed-but-not-reported is fine); the failure mode this
    # guards is the reverse: reported without persisted evidence.
    conditions = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)
    assert conditions["bn_adapted_tta"] is not None
    metrics["conditions"]["bn_adapted_tta"] = dict(conditions["bn_adapted_tta"])
    _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)  # must not raise

    del predictions["bn_adapted_probs"]
    del predictions["bn_adapted_prefix_sequence"]
    with pytest.raises(EvaluationPersistenceError):
        _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)


# ---------------------------------------------------------------------------
# InvalidProbabilityArrayError / validate_probability_array fail-closed
# ---------------------------------------------------------------------------


def test_validate_probability_array_rejects_unnormalized():
    with pytest.raises(InvalidProbabilityArrayError):
        validate_probability_array(np.array([[0.5, 0.6]]))


def test_validate_probability_array_rejects_non_finite():
    with pytest.raises(InvalidProbabilityArrayError):
        validate_probability_array(np.array([[np.nan, 1.0]]))


def test_compute_metrics_from_probabilities_never_calls_softmax():
    import inspect

    from when_tta_hurts import metrics as metrics_module

    source = inspect.getsource(metrics_module.compute_metrics_from_probabilities)
    source += inspect.getsource(metrics_module.accuracy_from_probabilities)
    source += inspect.getsource(metrics_module.macro_f1_from_probabilities)
    source += inspect.getsource(metrics_module.nll_from_probabilities)
    source += inspect.getsource(metrics_module.ece_from_probabilities)
    source += inspect.getsource(metrics_module.brier_from_probabilities)
    assert "softmax(" not in source
