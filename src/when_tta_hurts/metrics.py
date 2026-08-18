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


def compute_all_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels),
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
        "expected_calibration_error": expected_calibration_error(logits, labels),
    }
