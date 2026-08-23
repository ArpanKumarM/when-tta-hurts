"""Frozen aggregation methods, per docs/phase2b_protocol.md sec.3. All
functions operate on an ordered [views, N, C] logits array (the same
format produced by evaluation/tta.py::compute_ordered_view_logits) and a
separate clean_logits [N, C] array where relevant.

Return convention: aggregate functions return a [N, C] array. Where the
aggregate is a genuine probability distribution (mean_probability,
confidence_weighted, original_anchored, majority_vote's probability
representation), it is returned in LOG-PROBABILITY space (same convention
as evaluation/tta.py::aggregate_mean_prefix), so it composes directly with
metrics.py's softmax-based functions: softmax(log(p)) == p exactly.
"""

from __future__ import annotations

import numpy as np

from when_tta_hurts.metrics import softmax

_EPS = 1e-12


def _to_log_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, _EPS, 1.0))


def mean_probability(ordered_view_logits: np.ndarray, n_views: int) -> np.ndarray:
    """Arithmetic mean of per-view softmax probabilities. Softmax is
    applied BEFORE aggregation, per the frozen definition."""
    if n_views < 1 or n_views > ordered_view_logits.shape[0]:
        raise ValueError(f"n_views={n_views} out of range [1, {ordered_view_logits.shape[0]}]")
    prefix = ordered_view_logits[:n_views]
    probs_per_view = np.stack([softmax(v) for v in prefix], axis=0)
    return _to_log_probs(probs_per_view.mean(axis=0))


def majority_vote(ordered_view_logits: np.ndarray, n_views: int) -> tuple[np.ndarray, np.ndarray]:
    """Each view votes for its argmax class. Ties are resolved by the
    highest mean probability among tied classes; any remaining exact tie
    is resolved by lowest class index.

    Returns (predicted_class [N], vote_fraction_log_probs [N, C]) --
    predicted_class implements the EXACT frozen tie-break rule (do not
    derive it via argmax on vote_fraction_log_probs, which would only
    implement the final "lowest index" tie-break step and skip the
    "highest mean probability" step). vote_fraction_log_probs (votes/N
    per class, in log space) is provided as the probability representation
    for NLL/ECE/Brier -- standard practice for majority-vote TTA metrics.
    """
    if n_views < 1 or n_views > ordered_view_logits.shape[0]:
        raise ValueError(f"n_views={n_views} out of range [1, {ordered_view_logits.shape[0]}]")
    prefix = ordered_view_logits[:n_views]  # [n_views, N, C]
    probs_per_view = np.stack([softmax(v) for v in prefix], axis=0)  # [n_views, N, C]
    n_samples, n_classes = probs_per_view.shape[1], probs_per_view.shape[2]

    view_preds = probs_per_view.argmax(axis=-1)  # [n_views, N]
    votes = np.zeros((n_samples, n_classes), dtype=np.int64)
    for v in range(n_views):
        for c in range(n_classes):
            votes[:, c] += view_preds[v] == c

    mean_probs = probs_per_view.mean(axis=0)  # [N, C], used only for tie-break

    predicted_class = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        max_votes = votes[i].max()
        tied_classes = np.flatnonzero(votes[i] == max_votes)
        if len(tied_classes) == 1:
            predicted_class[i] = tied_classes[0]
        else:
            tied_mean_probs = mean_probs[i, tied_classes]
            best_prob = tied_mean_probs.max()
            still_tied = tied_classes[tied_mean_probs == best_prob]
            predicted_class[i] = still_tied.min()  # lowest class index

    vote_fraction = votes.astype(np.float64) / n_views
    return predicted_class, _to_log_probs(vote_fraction)


def confidence_weighted_average(ordered_view_logits: np.ndarray, n_views: int) -> np.ndarray:
    """Each view is weighted by its own maximum softmax probability;
    weights normalized to sum to one per sample, then probability vectors
    are averaged using those normalized weights."""
    if n_views < 1 or n_views > ordered_view_logits.shape[0]:
        raise ValueError(f"n_views={n_views} out of range [1, {ordered_view_logits.shape[0]}]")
    prefix = ordered_view_logits[:n_views]  # [n_views, N, C]
    probs_per_view = np.stack([softmax(v) for v in prefix], axis=0)  # [n_views, N, C]
    confidences = probs_per_view.max(axis=-1)  # [n_views, N]
    weights = confidences / confidences.sum(axis=0, keepdims=True)  # normalize across views, per sample
    weighted = (probs_per_view * weights[:, :, None]).sum(axis=0)  # [N, C]
    return _to_log_probs(weighted)


def original_anchored_mean_probability(
    clean_probs: np.ndarray, ordered_view_logits: np.ndarray, n_views: int
) -> np.ndarray:
    """One clean view plus n_views augmented views, equal-weight mean
    probability over the n_views+1 views (clean view has equal weight with
    each augmented view, not double-weighted or half-weighted).

    `clean_probs` MUST be the already-computed, canonical clean
    probability array (the exact array persisted to predictions.npz) --
    NEVER raw clean logits. Phase 2B.6J: this function previously
    accepted raw clean logits and computed `softmax(clean_logits)`
    internally; every caller now supplies the persisted probability
    array directly instead, so the live-computation and semantic-
    verification-recompute call sites use the bit-identical clean anchor
    representation. Passing raw logits here (re-softmaxing them
    yourself first) is no longer correct: `clean_probs` must already sum
    to 1 per row. See
    docs/phase2b_final_test_semantic_metric_contract_freeze.md for the
    incident this eliminates -- an internal `softmax(clean_logits)` here
    was NEVER clipped, while the recompute path's caller previously had
    to synthesize an equivalent input via `log(clip(clean_probs, eps,
    1))`, which silently diverges from the true clean anchor whenever
    any class probability underflows the `1e-12` clip floor."""
    if n_views < 1 or n_views > ordered_view_logits.shape[0]:
        raise ValueError(f"n_views={n_views} out of range [1, {ordered_view_logits.shape[0]}]")
    aug_probs = np.stack([softmax(v) for v in ordered_view_logits[:n_views]], axis=0)  # [n_views, N, C]
    all_probs = np.concatenate([clean_probs[None, :, :], aug_probs], axis=0)  # [n_views+1, N, C]
    return _to_log_probs(all_probs.mean(axis=0))
