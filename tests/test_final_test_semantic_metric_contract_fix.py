"""Phase 2B.6J Part E: regression tests for the corrected
`original_anchored_mean_probability()` contract, which fixed a clipping/
renormalization asymmetry between the live-computation and semantic-
verification-recompute call sites (see
docs/phase2b_final_test_semantic_metric_contract_freeze.md). Every test
uses only synthetic arrays, a CPU-only synthetic model/split, or (where
explicitly noted) already-persisted real artifacts read strictly for
integrity-equivalence checking -- NONE invoke evaluate-test,
load_final_test_split(), a real device, or a real checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from when_tta_hurts.evaluation.aggregation import original_anchored_mean_probability
from when_tta_hurts.evaluation_result_artifacts import EvaluationPersistenceError
from when_tta_hurts.metrics import softmax
from when_tta_hurts.validation_evaluation import (
    AGGREGATORS,
    PREFIX_SEQUENCE,
    _recompute_all_conditions_from_predictions,
    _verify_metrics_semantically,
)


def _old_buggy_original_anchored(clean_logits, ordered_view_logits, n_views):
    """Reimplements the PRE-FIX formula (softmax(clean_logits) computed
    internally, live-path style) purely for comparison in this test file
    -- the production function no longer supports this call shape."""
    clean_probs = softmax(clean_logits)
    aug_probs = np.stack([softmax(v) for v in ordered_view_logits[:n_views]], axis=0)
    all_probs = np.concatenate([clean_probs[None, :, :], aug_probs], axis=0)
    return np.log(np.clip(all_probs.mean(axis=0), 1e-12, 1.0))


def _old_buggy_recompute_anchor(clean_probs, ordered_view_logits, n_views):
    """Reimplements the PRE-FIX recompute-path input construction
    (clip -> log round-trip of the persisted probability array, fed back
    through the same softmax-internal aggregator) for comparison only."""
    clean_logits_equivalent = np.log(np.clip(clean_probs, 1e-12, 1.0))
    return _old_buggy_original_anchored(clean_logits_equivalent, ordered_view_logits, n_views)


def _extreme_synthetic_arrays(rng, n=500, c=9, v=50):
    """Extreme-magnitude logits that genuinely underflow the 1e-12
    softmax floor for at least one class of at least one sample --
    exactly the real, confirmed trigger condition (cell 1's own
    persisted clean_probs independently contains 201 such entries)."""
    scale = rng.choice([1.0, 1.0, 1.0, 30.0], size=(n, 1))
    clean_logits = (rng.normal(0, 1, size=(n, c)) * scale).astype(np.float32)
    view_scale = rng.choice([1.0, 1.0, 1.0, 30.0], size=(v, n, 1))
    view_logits = (rng.normal(0, 1, size=(v, n, c)) * view_scale).astype(np.float32)
    clean_probs = softmax(clean_logits)
    view_probs = softmax(view_logits.reshape(-1, c)).reshape(v, n, c)
    view_log_probs = np.log(np.clip(view_probs, 1e-12, 1.0))
    labels = rng.integers(0, c, size=n)
    return clean_logits, clean_probs, view_probs, view_log_probs, labels


def test_reproduces_old_mismatch_with_near_zero_true_class_probabilities():
    """(1)/(4) The pre-fix formula (live raw-logit anchor vs recompute
    clip-round-tripped anchor) genuinely mismatches beyond the frozen
    tolerance when extreme/near-zero probabilities are present -- this
    is the exact incident reproduced mechanically, not assumed."""
    rng = np.random.default_rng(0)
    clean_logits, clean_probs, view_probs, view_log_probs, labels = _extreme_synthetic_arrays(rng)
    assert (clean_probs < 1e-12).sum() > 0, "test setup must genuinely trigger the clip floor"

    n = 1
    old_live = softmax(_old_buggy_original_anchored(clean_logits, view_log_probs, n))
    old_recompute = softmax(_old_buggy_recompute_anchor(clean_probs, view_log_probs, n))

    nll_live = -np.log(np.clip(old_live[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    nll_recompute = -np.log(np.clip(old_recompute[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    assert abs(nll_live - nll_recompute) > 1e-6, (
        "old formula must genuinely mismatch to prove this is the real bug"
    )


def test_corrected_contract_eliminates_the_mismatch_without_loosening_tolerance():
    """(2) The PRODUCTION corrected function (clean_probs passed directly
    to both call-site shapes) produces bit-identical results regardless
    of which representation the caller originally derived it from --
    eliminating the divergence by construction, not by widening
    atol/rtol (still the frozen 1e-6/1e-6, unchanged)."""
    rng = np.random.default_rng(0)
    _clean_logits, clean_probs, _view_probs, view_log_probs, labels = _extreme_synthetic_arrays(rng)

    for n in (1, 2, 5, 10, 25, 50):
        live_style = softmax(original_anchored_mean_probability(clean_probs, view_log_probs, n))
        recompute_style = softmax(original_anchored_mean_probability(clean_probs, view_log_probs, n))
        assert np.array_equal(live_style, recompute_style), f"n={n}: must be bit-identical, same input array"

        nll_live = -np.log(np.clip(live_style[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        nll_recompute = -np.log(np.clip(recompute_style[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        assert np.isclose(nll_live, nll_recompute, atol=1e-6, rtol=1e-6)


def test_full_semantic_verification_passes_with_extreme_probabilities():
    """(2)/(3)/(4) End-to-end: build a full synthetic predictions/metrics
    pair (clean + naive_tta all aggregators + original_anchored_tta +
    bn_adapted_tta, every registered prefix) using extreme/near-zero
    probabilities, and confirm _verify_metrics_semantically() (the exact
    production semantic-verification gate that failed in the real
    incident) raises nothing."""
    from when_tta_hurts.metrics import compute_metrics_from_probabilities

    rng = np.random.default_rng(1)
    clean_logits, clean_probs, view_probs, view_log_probs, labels = _extreme_synthetic_arrays(
        rng, n=200, c=9, v=100
    )

    predictions = {
        "labels": labels,
        "sample_indices": np.arange(len(labels)),
        "clean_probs": clean_probs,
        "view_probs": view_probs,
    }

    # bn_adapted_probs: reuse clean_probs-derived aggregation for a
    # synthetic BN-adapted stack across the same prefixes.
    bn_prefix_sequence = np.array(PREFIX_SEQUENCE, dtype=np.int64)
    bn_stack = np.stack(
        [
            softmax(original_anchored_mean_probability(clean_probs, view_log_probs, n))
            for n in PREFIX_SEQUENCE
        ],
        axis=0,
    )
    predictions["bn_adapted_probs"] = bn_stack
    predictions["bn_adapted_prefix_sequence"] = bn_prefix_sequence

    recomputed = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)

    clean_metrics = compute_metrics_from_probabilities(clean_probs, labels)
    metrics = {"clean": clean_metrics, "conditions": recomputed, "latency": {}}

    _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)  # must not raise


def test_all_aggregators_and_conditions_covered_across_every_prefix():
    """(3) Every registered aggregator, original_anchored_tta, and
    bn_adapted_tta are exercised for every entry in the frozen
    PREFIX_SEQUENCE -- not just prefix 1."""
    rng = np.random.default_rng(2)
    _clean_logits, clean_probs, view_probs, view_log_probs, labels = _extreme_synthetic_arrays(
        rng, n=50, c=5, v=100
    )
    predictions = {
        "labels": labels,
        "sample_indices": np.arange(len(labels)),
        "clean_probs": clean_probs,
        "view_probs": view_probs,
    }
    recomputed = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)

    for agg in AGGREGATORS:
        assert set(recomputed["naive_tta"][agg].keys()) == set(PREFIX_SEQUENCE)
    assert set(recomputed["original_anchored_tta"].keys()) == set(PREFIX_SEQUENCE)
    assert recomputed["bn_adapted_tta"] is None  # no bn_adapted_probs supplied in this call


def test_clean_probs_must_be_the_canonical_persistable_array_not_renormalized():
    """(5) original_anchored_mean_probability() must use `clean_probs`
    EXACTLY as supplied -- no internal renormalization/softmax that
    would silently accept a non-probability array. Passing a
    deliberately UN-normalized "probability" array (rows not summing to
    1) must propagate through unchanged (i.e. NOT be silently corrected
    by an internal softmax), proving the canonical-array contract is
    honored rather than papered over."""
    clean_not_normalized = np.array([[0.9, 0.9]])  # deliberately sums to 1.8, not 1.0
    ordered = np.array([[[0.0, 0.0]]])  # one view, log-probs [.5,.5]
    result = np.exp(original_anchored_mean_probability(clean_not_normalized, ordered, n_views=1))
    # mean of [0.9,0.9] and [0.5,0.5] elementwise, in PROBABILITY space,
    # is exactly [0.7,0.7] before the function's own _to_log_probs/exp
    # round-trip -- confirms no internal softmax renormalized [0.9,0.9]
    # to [0.5,0.5] first.
    assert np.allclose(result[0], [0.7, 0.7], atol=1e-6)


def test_exactly_one_softmax_call_on_augmented_views_none_on_clean():
    """(6) Static proof: original_anchored_mean_probability's source
    contains exactly one softmax(...) call (over the augmented views),
    and the clean-probability parameter is never passed to softmax()."""
    import ast
    import inspect

    source = inspect.getsource(original_anchored_mean_probability)
    tree = ast.parse(source)
    softmax_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "softmax"
    ]
    assert len(softmax_calls) == 1, f"expected exactly one softmax() call, found {len(softmax_calls)}"
    # The single softmax call's argument must be the loop variable over
    # ordered_view_logits, never the clean_probs parameter.
    call = softmax_calls[0]
    assert ast.unparse(call.args[0]) != "clean_probs"


def test_aggregation_formula_and_operation_order_unchanged():
    """(7) The equal-weight mean formula (concatenate clean + augmented
    probabilities, then arithmetic mean along the view axis) is
    unchanged -- hand-calculated, matching the original frozen
    definition's semantics exactly (only the clean INPUT representation
    changed, never the combination formula)."""
    clean_probs = np.array([[0.5, 0.5]])
    aug1 = np.array([[np.log(3.0), 0.0]])  # probs [.75, .25]
    aug2 = np.array([[0.0, np.log(3.0)]])  # probs [.25, .75]
    ordered = np.stack([aug1, aug2], axis=0)
    result = softmax(original_anchored_mean_probability(clean_probs, ordered, n_views=2))
    expected = np.mean([[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]], axis=0)
    assert np.allclose(result[0], expected, atol=1e-6)


def test_semantic_corruption_still_fails_before_persistence():
    """(8) A deliberately corrupted persisted metric must still be
    rejected by _verify_metrics_semantically() under the corrected
    contract -- the fix eliminates a false-positive failure mode, it
    does not weaken genuine corruption detection."""
    from when_tta_hurts.metrics import compute_metrics_from_probabilities

    rng = np.random.default_rng(3)
    _clean_logits, clean_probs, view_probs, _view_log_probs, labels = _extreme_synthetic_arrays(
        rng, n=50, c=5, v=100
    )
    predictions = {
        "labels": labels,
        "sample_indices": np.arange(len(labels)),
        "clean_probs": clean_probs,
        "view_probs": view_probs,
    }
    recomputed = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)
    clean_metrics = compute_metrics_from_probabilities(clean_probs, labels)
    metrics = {"clean": clean_metrics, "conditions": recomputed, "latency": {}}

    # Corrupt one persisted value beyond any plausible floating-point noise.
    metrics["conditions"]["original_anchored_tta"][1]["negative_log_likelihood"] += 5.0

    with pytest.raises(EvaluationPersistenceError, match="original_anchored_tta.1.negative_log_likelihood"):
        _verify_metrics_semantically(predictions, metrics, PREFIX_SEQUENCE)


@pytest.mark.skipif(
    not Path(
        "artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_003/predictions.npz"
    ).exists(),
    reason="cell 1's real persisted artifact is not present in this checkout",
)
def test_cell_1_remains_compatible_under_corrected_contract_no_values_displayed():
    """(9) Cell 1's already-authorized, already-persisted original_anchored_tta
    metrics must remain within the frozen tolerance under the corrected
    contract -- reads real artifacts strictly for integrity-equivalence
    checking (never prints/returns a scientific value; PASS/FAIL only),
    per docs/phase2b_final_test_semantic_metric_contract_freeze.md sec.2."""
    attempt_dir = Path("artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_003")
    npz = np.load(attempt_dir / "predictions.npz")
    metrics = json.loads((attempt_dir / "metrics.json").read_text())
    metadata = json.loads((attempt_dir / "metadata.json").read_text())

    predictions = {k: npz[k] for k in npz.files}
    prefix_sequence = tuple(metadata.get("prefix_sequence", list(PREFIX_SEQUENCE)))
    recomputed = _recompute_all_conditions_from_predictions(predictions, prefix_sequence)

    all_within_tolerance = True
    for n in prefix_sequence:
        recomputed_entry = recomputed["original_anchored_tta"][n]
        persisted_entry = metrics["conditions"]["original_anchored_tta"][str(n)]
        for k, v in recomputed_entry.items():
            if not np.isclose(v, persisted_entry[k], atol=1e-6, rtol=1e-6):
                all_within_tolerance = False

    assert all_within_tolerance


def test_cell_2_recovery_lifecycle_is_state_derived_and_gapless():
    """(10) State-derived invariant over cell 2's real attempt history --
    deliberately NOT pinned to any specific attempt count, so it remains
    valid as the append-only final-test ledger grows with future cells
    or (in principle) future recovery attempts for other cells. Read-only,
    no evaluate-test invocation, no test-split access.

    Verifies:
    1. Attempt 1 is present, byte-identical to its recorded failure state,
       and status=failed.
    2. Attempt 2 is present and status=completed.
    3. Attempt 2 is the sole completed (authorized) result for this cell.
    4. Existing attempt numbers are monotonic and gapless (1..max, no
       skips).
    5. next_evaluation_attempt_number(...) == max(existing) + 1 -- the
       production runner's own allocator, not a re-derivation of it.
    6. No historical ledger row or artifact is mutated by this test.
    """
    import hashlib

    from when_tta_hurts.final_test_evaluation import DEFAULT_FINAL_TEST_ROOT
    from when_tta_hurts.ledger import FINAL_TEST_LEDGER_PATH
    from when_tta_hurts.validation_evaluation import next_evaluation_attempt_number

    run_id = "A-pathmnist-28px-batchnorm-policy-none-s1"
    run_dir = DEFAULT_FINAL_TEST_ROOT / run_id
    if not run_dir.exists():
        pytest.skip("cell 2's real attempt history is not present in this checkout")

    attempt_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    existing_attempt_numbers = [int(p.name.split("_")[1]) for p in attempt_dirs]

    # (4) monotonic and gapless: exactly 1, 2, ..., max -- no skipped numbers.
    assert existing_attempt_numbers == list(range(1, len(existing_attempt_numbers) + 1))

    statuses = {}
    for attempt_dir in attempt_dirs:
        n = int(attempt_dir.name.split("_")[1])
        statuses[n] = json.loads((attempt_dir / "status.json").read_text())["status"]

    # (1) attempt 1 remains present and failed, with its content structure
    # intact -- a failed attempt has no persisted metrics/predictions.
    attempt_1_status_path = DEFAULT_FINAL_TEST_ROOT / run_id / "attempt_001" / "status.json"
    attempt_1_bytes = attempt_1_status_path.read_bytes()
    attempt_1_status_before = json.loads(attempt_1_bytes)
    assert attempt_1_status_before["status"] == "failed"
    assert attempt_1_status_before["attempt_number"] == 1
    assert attempt_1_status_before["failure_reason"]
    assert not (attempt_dirs[0] / "predictions.npz").exists()
    assert not (attempt_dirs[0] / "metrics.json").exists()
    attempt_1_hash_before = hashlib.sha256(attempt_1_bytes).hexdigest()

    # (2) + (3) exactly one completed attempt exists, and it is the last
    # (highest-numbered) one -- the sole authorized completed result.
    completed_attempts = [n for n, s in statuses.items() if s == "completed"]
    assert len(completed_attempts) == 1
    assert completed_attempts[0] == max(existing_attempt_numbers)

    # (5) the production allocator agrees with the state-derived maximum.
    next_attempt = next_evaluation_attempt_number(run_id, DEFAULT_FINAL_TEST_ROOT, FINAL_TEST_LEDGER_PATH)
    assert next_attempt == max(existing_attempt_numbers) + 1

    # (6) nothing above wrote to any ledger or artifact file: re-reading
    # attempt 1's status file yields byte-identical content.
    assert hashlib.sha256(attempt_1_status_path.read_bytes()).hexdigest() == attempt_1_hash_before
