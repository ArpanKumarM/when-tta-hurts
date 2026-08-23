"""Phase 2B.6K Part B: tests for the offline validation-metric
reconciliation mechanism (validation_metric_reconciliation.py). Every
test uses tmp_path-rooted synthetic ledgers/artifacts; NONE touch the
real validation ledger/artifacts, a real device, a real checkpoint, or
any dataset array.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from when_tta_hurts.artifacts import atomic_write_json, atomic_write_npz, hash_file
from when_tta_hurts.evaluation_result_artifacts import build_evaluation_artifact_manifest
from when_tta_hurts.ledger import (
    append_evaluation_entry,
    ensure_evaluation_ledger_exists,
)
from when_tta_hurts.metrics import compute_metrics_from_probabilities, softmax
from when_tta_hurts.validation_evaluation import (
    AGGREGATORS,
    PREFIX_SEQUENCE,
    _recompute_all_conditions_from_predictions,
)
from when_tta_hurts.validation_metric_reconciliation import (
    OLD_EVALUATOR_FINGERPRINT,
    ReconciliationError,
    is_reconciled_compatible,
    reconcile_validation_cell,
    resolve_canonical_pre_fix_row,
)

RUN_ID = "fake-run-a"


def _synthetic_predictions_and_metrics(n=80, c=9, v=100, seed=0):
    rng = np.random.default_rng(seed)
    scale = rng.choice([1.0, 1.0, 1.0, 25.0], size=(n, 1))
    clean_logits = (rng.normal(0, 1, size=(n, c)) * scale).astype(np.float32)
    clean_probs = softmax(clean_logits)
    view_scale = rng.choice([1.0, 1.0, 1.0, 25.0], size=(v, n, 1))
    view_logits = (rng.normal(0, 1, size=(v, n, c)) * view_scale).astype(np.float32)
    view_probs = softmax(view_logits.reshape(-1, c)).reshape(v, n, c)
    labels = rng.integers(0, c, size=n)

    predictions = {
        "labels": labels,
        "sample_indices": np.arange(n),
        "clean_probs": clean_probs,
        "view_probs": view_probs,
    }
    recomputed = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)
    clean_metrics = compute_metrics_from_probabilities(clean_probs, labels)
    # Serialize conditions with string prefix keys (matches real
    # metrics.json's on-disk JSON convention).
    conditions_json = {
        "naive_tta": {
            agg: {str(n_): recomputed["naive_tta"][agg][n_] for n_ in PREFIX_SEQUENCE} for agg in AGGREGATORS
        },
        "original_anchored_tta": {str(n_): recomputed["original_anchored_tta"][n_] for n_ in PREFIX_SEQUENCE},
        "bn_adapted_tta": None,
    }
    metrics = {"clean": clean_metrics, "conditions": conditions_json, "latency": {}}
    return predictions, metrics


def _write_fixture_cell(
    tmp_path,
    run_id=RUN_ID,
    attempt=5,
    old_fp=OLD_EVALUATOR_FINGERPRINT,
    corrupt_unaffected=False,
    seed=0,
):
    validation_root = tmp_path / "validation_evaluation"
    ledger_path = tmp_path / "ledger_validation_evaluation.csv"
    attempt_dir = validation_root / run_id / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True)

    predictions, metrics = _synthetic_predictions_and_metrics(seed=seed)
    if corrupt_unaffected:
        metrics["conditions"]["naive_tta"]["mean_probability"]["1"]["negative_log_likelihood"] += 5.0

    metadata = {"evaluator_fingerprint": old_fp, "training_run_id": run_id}
    atomic_write_npz(predictions, attempt_dir / "predictions.npz")
    atomic_write_json(metrics, attempt_dir / "metrics.json")
    atomic_write_json(metadata, attempt_dir / "metadata.json")
    atomic_write_json({"seed_manifest_sha256": "x"}, attempt_dir / "view_manifest.json")
    atomic_write_json(
        {"status": "completed", "attempt_number": attempt, "training_run_id": run_id},
        attempt_dir / "status.json",
    )
    manifest = build_evaluation_artifact_manifest(attempt_dir)
    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")

    ensure_evaluation_ledger_exists(ledger_path)
    append_evaluation_entry(
        evaluation_id=f"eval-{run_id}-{attempt}",
        training_run_id=run_id,
        training_attempt=1,
        checkpoint_hash="chk-a",
        evaluation_config_hash=f"eval-{run_id}-{attempt}",
        evaluation_attempt=attempt,
        status="completed",
        primary_artifact_hash="hash",
        started_at=time.time(),
        ended_at=time.time(),
        runtime_seconds=1.0,
        ledger_path=ledger_path,
    )
    return validation_root, ledger_path, attempt_dir


def test_reconcile_never_touches_mps_device_checkpoint_or_dataset(tmp_path, monkeypatch):
    """(zero inference/data/device/checkpoint access) Static + dynamic
    proof: patch torch.load/select_device to hard-fail; reconciliation
    must never call them."""
    import inspect

    import when_tta_hurts.validation_metric_reconciliation as vmr

    source = inspect.getsource(vmr)
    assert "select_device" not in source
    assert "load_and_verify_canonical_checkpoint" not in source
    assert "load_validation_evaluation_split" not in source
    assert ".to(device)" not in source

    validation_root, ledger_path, attempt_dir = _write_fixture_cell(tmp_path)
    reconciliation_root = tmp_path / "reconciliation"
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"

    result = reconcile_validation_cell(
        RUN_ID,
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=reconciliation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )
    assert result["status"] == "completed"


def test_original_artifacts_remain_byte_identical_after_reconciliation(tmp_path):
    validation_root, ledger_path, attempt_dir = _write_fixture_cell(tmp_path)
    before = {
        fn: hash_file(attempt_dir / fn)
        for fn in ("predictions.npz", "metrics.json", "metadata.json", "artifact_manifest.json")
    }

    reconcile_validation_cell(
        RUN_ID,
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=tmp_path / "reconciliation",
        reconciliation_ledger_path=tmp_path / "ledger_reconciliation.csv",
    )

    after = {fn: hash_file(attempt_dir / fn) for fn in before}
    assert before == after


def test_duplicate_reconciliation_rejected(tmp_path):
    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path)
    reconciliation_root = tmp_path / "reconciliation"
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"

    reconcile_validation_cell(
        RUN_ID,
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=reconciliation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )
    with pytest.raises(ReconciliationError, match="already has a reconciliation record"):
        reconcile_validation_cell(
            RUN_ID,
            validation_root=validation_root,
            ledger_path=ledger_path,
            reconciliation_root=reconciliation_root,
            reconciliation_ledger_path=reconciliation_ledger_path,
        )


def test_mismatched_old_fingerprint_rejected(tmp_path):
    """(reject mismatched old/new fingerprints) A cell whose persisted
    metadata does NOT carry the expected pre-fix fingerprint must be
    refused, never silently reconciled."""
    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path, old_fp="some-other-fingerprint")
    with pytest.raises(ReconciliationError):
        resolve_canonical_pre_fix_row(RUN_ID, ledger_path=ledger_path, validation_root=validation_root)


def test_unaffected_metric_divergence_fails_closed(tmp_path):
    """(prove primary/unaffected endpoints remain unchanged -- and that
    a genuine divergence is caught, not silently accepted) A cell whose
    persisted naive_tta metric has been corrupted (simulating the
    assumption 'only original_anchored_tta is affected' being wrong)
    must be rejected outright."""
    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path, corrupt_unaffected=True)
    with pytest.raises(ReconciliationError, match="UNAFFECTED metric diverged"):
        reconcile_validation_cell(
            RUN_ID,
            validation_root=validation_root,
            ledger_path=ledger_path,
            reconciliation_root=tmp_path / "reconciliation",
            reconciliation_ledger_path=tmp_path / "ledger_reconciliation.csv",
        )


def test_tampered_reconciliation_record_not_recognized_as_compatible(tmp_path):
    """(reject tampered reconciliation evidence) A reconciliation ledger
    row claiming a corrected_evaluator_fingerprint that does not match
    what the caller is actually asking about must not be treated as
    compatible."""
    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path)
    reconciliation_root = tmp_path / "reconciliation"
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"
    reconcile_validation_cell(
        RUN_ID,
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=reconciliation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )
    compatible, _row = is_reconciled_compatible(
        RUN_ID, "a-fingerprint-that-does-not-match", reconciliation_ledger_path
    )
    assert compatible is False


def test_partial_reconciliation_across_multiple_cells_reports_incomplete(tmp_path):
    """(reject partial N/M reconciliation) With two required cells, only
    one reconciled, compatibility must be False for the unreconciled one
    and the overall set must not be reported complete."""
    validation_root = tmp_path / "validation_evaluation"
    ledger_path = tmp_path / "ledger_validation_evaluation.csv"
    reconciliation_root = tmp_path / "reconciliation"
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"

    for run_id, seed in (("fake-run-a", 0), ("fake-run-b", 1)):
        attempt_dir = validation_root / run_id / "attempt_005"
        attempt_dir.mkdir(parents=True)
        predictions, metrics = _synthetic_predictions_and_metrics(seed=seed)
        atomic_write_npz(predictions, attempt_dir / "predictions.npz")
        atomic_write_json(metrics, attempt_dir / "metrics.json")
        atomic_write_json({"evaluator_fingerprint": OLD_EVALUATOR_FINGERPRINT}, attempt_dir / "metadata.json")
        atomic_write_json({"x": 1}, attempt_dir / "view_manifest.json")
        atomic_write_json({"status": "completed"}, attempt_dir / "status.json")
        manifest = build_evaluation_artifact_manifest(attempt_dir)
        atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
        ensure_evaluation_ledger_exists(ledger_path)
        append_evaluation_entry(
            evaluation_id=f"eval-{run_id}",
            training_run_id=run_id,
            training_attempt=1,
            checkpoint_hash="chk",
            evaluation_config_hash=f"eval-{run_id}",
            evaluation_attempt=5,
            status="completed",
            primary_artifact_hash="h",
            started_at=1.0,
            ended_at=2.0,
            runtime_seconds=1.0,
            ledger_path=ledger_path,
        )

    reconcile_validation_cell(
        "fake-run-a",
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=reconciliation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )

    from when_tta_hurts.validation_evaluation import compute_evaluator_fingerprint

    current_fp, _ = compute_evaluator_fingerprint()
    compat_a, _ = is_reconciled_compatible("fake-run-a", current_fp, reconciliation_ledger_path)
    compat_b, _ = is_reconciled_compatible("fake-run-b", current_fp, reconciliation_ledger_path)
    assert compat_a is True
    assert compat_b is False
    assert not (compat_a and compat_b)  # the pair as a whole is incomplete


def test_statistical_analysis_resolver_recognizes_reconciled_stale_row(tmp_path):
    """(item 8: resolvers recognize an old evaluation + valid
    reconciliation record as current-contract-compatible) Before
    reconciliation, the resolver must report 'stale'; after
    reconciliation, it must report 'eligible' with via_reconciliation."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity
    from when_tta_hurts.validation_evaluation import compute_evaluator_fingerprint

    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path)
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"
    current_fp, _ = compute_evaluator_fingerprint()

    before = _resolve_canonical_evaluation_identity(
        RUN_ID,
        current_fp,
        ledger_path=ledger_path,
        validation_evaluation_root=validation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )
    assert before["evaluation_status"] == "stale"

    reconcile_validation_cell(
        RUN_ID,
        validation_root=validation_root,
        ledger_path=ledger_path,
        reconciliation_root=tmp_path / "reconciliation",
        reconciliation_ledger_path=reconciliation_ledger_path,
    )

    after = _resolve_canonical_evaluation_identity(
        RUN_ID,
        current_fp,
        ledger_path=ledger_path,
        validation_evaluation_root=validation_root,
        reconciliation_ledger_path=reconciliation_ledger_path,
    )
    assert after["evaluation_status"] == "eligible"
    assert after.get("via_reconciliation") is True


def test_plan_mode_and_reconciliation_helpers_are_side_effect_free_when_reading(tmp_path):
    """(plan-mode side-effect freedom) is_reconciled_compatible() and
    resolve_canonical_pre_fix_row() never write anything."""
    validation_root, ledger_path, _ = _write_fixture_cell(tmp_path)
    reconciliation_ledger_path = tmp_path / "ledger_reconciliation.csv"
    before = set(tmp_path.rglob("*"))
    resolve_canonical_pre_fix_row(RUN_ID, ledger_path=ledger_path, validation_root=validation_root)
    is_reconciled_compatible(RUN_ID, "whatever", reconciliation_ledger_path)
    after = set(tmp_path.rglob("*"))
    assert before == after
