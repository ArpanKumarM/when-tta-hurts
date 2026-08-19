"""Metrics computed from saved logits + labels, per docs/pilot_protocol.md
and docs/experimental_protocol.md's secondary endpoints. All functions take
numpy arrays (logits: [N, C] float, labels: [N] int) so they can be reused
identically on cached/loaded artifacts, not just live tensors.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=-1)
    return float((preds == labels).mean())


def macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=-1)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = softmax(logits)
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = softmax(logits)
    confidences = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (
            (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        )
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def harm_rescue_rates(
    clean_logits: np.ndarray, tta_logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Clean-correct -> TTA-wrong harm rate, and clean-wrong -> TTA-correct
    rescue rate, per docs/experimental_protocol.md's secondary endpoints.
    """
    clean_preds = clean_logits.argmax(axis=-1)
    tta_preds = tta_logits.argmax(axis=-1)
    clean_correct = clean_preds == labels
    tta_correct = tta_preds == labels

    n_clean_correct = int(clean_correct.sum())
    n_clean_wrong = int((~clean_correct).sum())

    harmed = clean_correct & ~tta_correct
    rescued = ~clean_correct & tta_correct

    harm_rate = float(harmed.sum() / n_clean_correct) if n_clean_correct > 0 else 0.0
    rescue_rate = float(rescued.sum() / n_clean_wrong) if n_clean_wrong > 0 else 0.0

    return {
        "harm_rate": harm_rate,
        "rescue_rate": rescue_rate,
        "n_clean_correct": n_clean_correct,
        "n_clean_wrong": n_clean_wrong,
        "n_harmed": int(harmed.sum()),
        "n_rescued": int(rescued.sum()),
    }


def brier_score(logits: np.ndarray, labels: np.ndarray) -> float:
    """Frozen multiclass Brier score, per docs/phase2b_protocol.md sec.3:
    mean over samples of [sum over classes of (predicted_probability - one_hot_label)^2].
    """
    probs = softmax(logits)
    n, c = probs.shape
    one_hot = np.zeros((n, c))
    one_hot[np.arange(n), labels] = 1.0
    per_sample = np.sum((probs - one_hot) ** 2, axis=-1)
    return float(np.mean(per_sample))


def compute_all_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels),
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
        "expected_calibration_error": expected_calibration_error(logits, labels),
    }


# ---------------------------------------------------------------------------
# Probability-native metric API (Phase 2B.4D Metric-Contract Correction,
# metric_input_contract: probability_native_v1). These functions NEVER call
# softmax -- they operate directly on an already-normalized probability
# distribution. Use compute_metrics_from_probabilities() for any aggregate
# (mean-probability, majority-vote, confidence-weighted, original-anchored,
# BN-adapted) that is already a genuine probability distribution, and
# compute_metrics_from_logits() only for raw model logits. Passing an
# already-normalized probability array into a logits-native function above
# (accuracy/macro_f1/negative_log_likelihood/expected_calibration_error/
# brier_score) applies an extra, incorrect softmax -- this is exactly the
# defect these functions exist to make structurally impossible to repeat
# (see docs/phase2b_validation_evaluation_metric_contract_freeze.md).
# ---------------------------------------------------------------------------


class InvalidProbabilityArrayError(ValueError):
    """Raised when an array passed to a probability-native metric function
    is not a valid probability distribution -- non-finite, wrong rank,
    outside [0,1] beyond tolerance, or rows that do not sum to 1 within
    tolerance. Fails closed rather than silently treating a logits array
    (or an already-double-softmaxed array) as valid probabilities."""


def validate_probability_array(probs: np.ndarray, *, atol: float = 1e-4) -> None:
    """Validate `probs` as a genuine, already-normalized probability
    array: finite, 2-D [N, C], values within [0,1] (within `atol`), and
    rows summing to 1 (within `atol`). Raises InvalidProbabilityArrayError
    on any violation."""
    if not np.all(np.isfinite(probs)):
        raise InvalidProbabilityArrayError("probability array contains non-finite values.")
    if probs.ndim != 2:
        raise InvalidProbabilityArrayError(f"probability array must be 2-D [N, C], got shape {probs.shape}.")
    if probs.min() < -atol or probs.max() > 1.0 + atol:
        raise InvalidProbabilityArrayError(
            f"probability array out of [0,1] range (atol={atol}): min={probs.min()}, max={probs.max()}."
        )
    row_sums = probs.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise InvalidProbabilityArrayError(
            f"probability array rows do not sum to 1 within atol={atol} "
            f"(max deviation {np.abs(row_sums - 1.0).max():.6f})."
        )


def accuracy_from_probabilities(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=-1)
    return float((preds == labels).mean())


def macro_f1_from_probabilities(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=-1)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def nll_from_probabilities(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def ece_from_probabilities(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Identical binning formula/bin-boundary convention to
    expected_calibration_error() above (`(lo, hi]`, first bin's lower edge
    inclusive) -- the only difference is no internal softmax call."""
    confidences = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (
            (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        )
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_from_probabilities(probs: np.ndarray, labels: np.ndarray) -> float:
    n, c = probs.shape
    one_hot = np.zeros((n, c))
    one_hot[np.arange(n), labels] = 1.0
    per_sample = np.sum((probs - one_hot) ** 2, axis=-1)
    return float(np.mean(per_sample))


def compute_metrics_from_probabilities(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15, eps: float = 1e-12, validate: bool = True
) -> dict[str, float]:
    """Compute accuracy/macro_f1/negative_log_likelihood/
    expected_calibration_error/brier_score directly from an
    already-normalized probability distribution. NEVER calls softmax.
    Validates `probs` as a genuine probability array first (raises
    InvalidProbabilityArrayError, not silently on malformed input) unless
    `validate=False` is explicitly passed (only for callers that already
    validated, e.g. repeated calls in a hot loop)."""
    if validate:
        validate_probability_array(probs)
    return {
        "accuracy": accuracy_from_probabilities(probs, labels),
        "macro_f1": macro_f1_from_probabilities(probs, labels),
        "negative_log_likelihood": nll_from_probabilities(probs, labels, eps),
        "expected_calibration_error": ece_from_probabilities(probs, labels, n_bins),
        "brier_score": brier_from_probabilities(probs, labels),
    }


def compute_metrics_from_logits(
    logits: np.ndarray, labels: np.ndarray, *, n_bins: int = 15, eps: float = 1e-12
) -> dict[str, float]:
    """Apply softmax EXACTLY ONCE to raw logits, then delegate to
    compute_metrics_from_probabilities(). Use this entry point only for
    genuine model logits (or a valid logit-equivalent such as
    log(probabilities)) -- never for an already-normalized probability
    array (use compute_metrics_from_probabilities() directly for that)."""
    return compute_metrics_from_probabilities(softmax(logits), labels, n_bins=n_bins, eps=eps, validate=False)
