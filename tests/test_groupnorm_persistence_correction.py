"""Phase 2B.4F: GroupNorm persistence-schema correction tests.

Covers the frozen contract in
docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md: BN
adaptation is applicable only when normalization == "batchnorm";
bn_adaptation_applicable and bn_adaptation_microbatches_at_primary_n must
be consistent with each other, with the presence/absence of BN-adapted
metrics/probability arrays, and with the authoritative persisted
normalization value; None is never an accepted substitute value.

All synthetic tensors, temporary repositories, and tiny fresh models
only. No real MPS, checkpoint, dataset, or test-split path runs here --
the one real GroupNorm attempt referenced (evaluation_id
2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b) is
read only from the already-persisted, real production ledger, never
re-executed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
    persist_and_verify_evaluation_completion,
)
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.validation_evaluation import (
    ConflictingEvaluationImplementationError,
    check_evaluation_skip,
    compute_validation_evaluation,
    next_evaluation_attempt_number,
)

_VALID_PREFIX_SEQUENCE = (1, 2, 5, 10, 25, 50, 100)

_REAL_GROUPNORM_FAILED_EVAL_ID = "2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b"
_REAL_GROUPNORM_RUN_ID = "A-pathmnist-28px-groupnorm-policy-none-s0"


def _synthetic_split(n=4, n_classes=3, resolution=28, dataset="pathmnist", seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, resolution, resolution, generator=g)
    labels = np.array([i % n_classes for i in range(n)])
    return ValidationEvaluationSplit(
        images=images, labels=labels, sample_indices=np.arange(n), dataset=dataset, resolution=resolution
    )


def _valid_dataset_verification(dataset="pathmnist", resolution=28, checksum="a" * 32):
    return {
        "dataset": dataset,
        "resolution": resolution,
        "expected_checksum_md5": checksum,
        "actual_checksum_md5": checksum,
        "checksum_verified": True,
        "resized": False,
        "verification_method": "dataset_verification.verify_official_dataset_artifact",
        "verification_version": 1,
        "artifact_path": f"data/raw/{dataset}.npz",
    }


def _valid_latency(n_samples, prefix_sequence=_VALID_PREFIX_SEQUENCE, clean_latency=0.01):
    by_n = {}
    for n in prefix_sequence:
        tta = clean_latency * n
        by_n[str(n)] = {
            "tta_latency_seconds": tta,
            "per_sample_latency_seconds": tta / n_samples,
            "compute_multiplier": tta / clean_latency,
        }
    return {"clean_latency_seconds": clean_latency, "n_samples": n_samples, "by_n": by_n}


def _base_metadata(normalization: str, batching: dict, n_samples: int) -> dict:
    return {
        "evaluation_id": "e1",
        "training_run_id": "r1",
        "training_attempt": 1,
        "checkpoint_hash": "c1",
        "dataset": "pathmnist",
        "resolution": 28,
        "model": "small_cnn",
        "normalization": normalization,
        "training_policy": "none",
        "seed": 0,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "prefix_sequence": list(_VALID_PREFIX_SEQUENCE),
        "aggregators": ["mean_probability"],
        "secondary_analyses": ["scaling_curve"],
        "protocol_commit": "ce4c962",
        "matrix_hash": "m1",
        "source_commit": "s1",
        "evaluator_fingerprint": "fp1",
        "evaluator_fingerprint_manifest": {"src/when_tta_hurts/metrics.py": "abc123"},
        "dataset_expected_checksum_md5": "a" * 32,
        "dataset_verification": _valid_dataset_verification(),
        "batching": batching,
        "evaluation_config_hash": "e1",
        "split": "validation",
        "n_validation_samples": n_samples,
        "metric_input_contract": "probability_native_v1",
    }


def _view_manifest():
    return {
        "dataset": "pathmnist",
        "resolution": 28,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "n_views": 100,
        "seed_formula": "sha256(...)",
        "sample_indices": [0, 1, 2, 3],
        "seed_manifest_sha256": "abc",
    }


# ---------------------------------------------------------------------------
# 1: full synthetic GroupNorm evaluation through persistence completes
# 2: GroupNorm persists applicable=false and microbatch count 0
# 5: BatchNorm scientific outputs are unchanged by the correction
# 9: no real MPS/checkpoint/dataset/test-split path runs (CPU only, tiny
#    synthetic model/images, no dataset file touched)
# ---------------------------------------------------------------------------


def test_full_synthetic_groupnorm_evaluation_persists_successfully(tmp_path):
    model = build_small_cnn(num_classes=3, normalization="groupnorm")
    split = _synthetic_split(n=4, n_classes=3)
    outcome = compute_validation_evaluation(model, split, 111111, torch.device("cpu"))

    batching = outcome["batching"]
    assert batching["bn_adaptation_applicable"] is False
    assert batching["bn_adaptation_microbatches_at_primary_n"] == 0
    assert "bn_adapted_probs" not in outcome["predictions"]
    assert "bn_adapted_prefix_sequence" not in outcome["predictions"]
    assert outcome["metrics"]["conditions"]["bn_adapted_tta"] is None

    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    metrics = dict(outcome["metrics"])
    metrics["training_run_id"] = "r1"
    metrics["evaluation_config_hash"] = "e1"
    metrics["latency"] = _valid_latency(n_samples=4)

    manifest = persist_and_verify_evaluation_completion(
        tmp_path,
        predictions=outcome["predictions"],
        metrics=metrics,
        metadata=metadata,
        view_manifest=_view_manifest(),
        prefix_sequence=_VALID_PREFIX_SEQUENCE,
    )
    assert "artifacts" in manifest


def test_full_synthetic_batchnorm_evaluation_persists_successfully_and_scientific_outputs_unchanged(
    tmp_path,
):
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    split = _synthetic_split(n=4, n_classes=3)
    outcome = compute_validation_evaluation(model, split, 222222, torch.device("cpu"))

    batching = outcome["batching"]
    assert batching["bn_adaptation_applicable"] is True
    assert isinstance(batching["bn_adaptation_microbatches_at_primary_n"], int)
    assert batching["bn_adaptation_microbatches_at_primary_n"] > 0
    assert "bn_adapted_probs" in outcome["predictions"]
    assert "bn_adapted_prefix_sequence" in outcome["predictions"]
    assert outcome["metrics"]["conditions"]["bn_adapted_tta"] is not None
    # every registered prefix has a bn_adapted_tta entry
    assert set(outcome["metrics"]["conditions"]["bn_adapted_tta"].keys()) == set(_VALID_PREFIX_SEQUENCE)

    # scientific outputs (accuracy, macro_f1, NLL, ECE, Brier) are computed
    # by the same, unchanged _per_prefix_metrics()/compute_metrics_from_
    # probabilities() path -- this correction only touches the batching
    # metadata dict, never the metrics themselves.
    clean = outcome["metrics"]["clean"]
    assert set(clean.keys()) == {
        "accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "expected_calibration_error",
        "brier_score",
    }

    metadata = _base_metadata("batchnorm", batching, n_samples=4)
    metrics = dict(outcome["metrics"])
    metrics["training_run_id"] = "r1"
    metrics["evaluation_config_hash"] = "e1"
    metrics["latency"] = _valid_latency(n_samples=4)

    manifest = persist_and_verify_evaluation_completion(
        tmp_path,
        predictions=outcome["predictions"],
        metrics=metrics,
        metadata=metadata,
        view_manifest=_view_manifest(),
        prefix_sequence=_VALID_PREFIX_SEQUENCE,
    )
    assert "artifacts" in manifest


# ---------------------------------------------------------------------------
# 3: GroupNorm rejects None, positive counts, or any BN-adapted metrics/arrays
# ---------------------------------------------------------------------------


def _groupnorm_predictions(n=4, c=3):
    return {
        "labels": np.arange(n) % c,
        "sample_indices": np.arange(n),
        "clean_probs": np.full((n, c), 1.0 / c, dtype=np.float32),
        "view_probs": np.full((100, n, c), 1.0 / c, dtype=np.float32),
    }


def _groupnorm_metrics(n_samples=4):
    return {
        "training_run_id": "r1",
        "evaluation_config_hash": "e1",
        "clean": {"accuracy": 1 / 3},
        "conditions": {"bn_adapted_tta": None},
        "latency": _valid_latency(n_samples=n_samples),
    }


def _groupnorm_batching(**overrides):
    batching = {
        "inference_batch_size": 256,
        "bn_adaptation_batch_size": 256,
        "bn_adaptation_algorithm": "sequential_microbatch_v1",
        "bn_adaptation_enumeration_order": "view_major_then_sample_major",
        "bn_adaptation_applicable": False,
        "bn_adaptation_microbatches_at_primary_n": 0,
    }
    batching.update(overrides)
    return batching


def test_groupnorm_rejects_none_microbatch_count(tmp_path):
    batching = _groupnorm_batching()
    batching["bn_adaptation_microbatches_at_primary_n"] = None
    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    with pytest.raises((EvaluationSchemaValidationError, EvaluationPersistenceError)):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_groupnorm_predictions(),
            metrics=_groupnorm_metrics(),
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_groupnorm_rejects_positive_microbatch_count(tmp_path):
    batching = _groupnorm_batching()
    batching["bn_adaptation_microbatches_at_primary_n"] = 50
    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_groupnorm_predictions(),
            metrics=_groupnorm_metrics(),
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_groupnorm_rejects_reported_bn_adapted_metrics(tmp_path):
    batching = _groupnorm_batching()
    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    metrics = _groupnorm_metrics()
    metrics["conditions"]["bn_adapted_tta"] = {1: {"accuracy": 0.5}}  # must never be reported
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_groupnorm_predictions(),
            metrics=metrics,
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_groupnorm_rejects_persisted_bn_adapted_probability_arrays(tmp_path):
    batching = _groupnorm_batching()
    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    predictions = _groupnorm_predictions()
    predictions["bn_adapted_probs"] = np.full((len(_VALID_PREFIX_SEQUENCE), 4, 3), 1.0 / 3, dtype=np.float32)
    predictions["bn_adapted_prefix_sequence"] = np.array(_VALID_PREFIX_SEQUENCE, dtype=np.int64)
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=predictions,
            metrics=_groupnorm_metrics(),
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_groupnorm_applicability_must_match_normalization_field(tmp_path):
    """bn_adaptation_applicable=True paired with normalization=groupnorm
    is a contradiction and must fail closed even if every other field
    would otherwise look self-consistent."""
    batching = _groupnorm_batching(bn_adaptation_applicable=True, bn_adaptation_microbatches_at_primary_n=50)
    metadata = _base_metadata("groupnorm", batching, n_samples=4)
    metrics = _groupnorm_metrics()
    metrics["conditions"]["bn_adapted_tta"] = {1: {"accuracy": 0.5}}
    predictions = _groupnorm_predictions()
    predictions["bn_adapted_probs"] = np.full((len(_VALID_PREFIX_SEQUENCE), 4, 3), 1.0 / 3, dtype=np.float32)
    predictions["bn_adapted_prefix_sequence"] = np.array(_VALID_PREFIX_SEQUENCE, dtype=np.int64)
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=predictions,
            metrics=metrics,
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


# ---------------------------------------------------------------------------
# 4: BatchNorm still requires applicable=true, a positive count, every
# registered BN prefix, and recomputable evidence
# ---------------------------------------------------------------------------


def _batchnorm_predictions(n=4, c=3):
    preds = _groupnorm_predictions(n, c)
    preds["bn_adapted_probs"] = np.full((len(_VALID_PREFIX_SEQUENCE), n, c), 1.0 / c, dtype=np.float32)
    preds["bn_adapted_prefix_sequence"] = np.array(_VALID_PREFIX_SEQUENCE, dtype=np.int64)
    return preds


def _batchnorm_metrics(n_samples=4):
    metrics = _groupnorm_metrics(n_samples)
    metrics["conditions"]["bn_adapted_tta"] = {n: {"accuracy": 0.5} for n in _VALID_PREFIX_SEQUENCE}
    return metrics


def _batchnorm_batching(**overrides):
    batching = {
        "inference_batch_size": 256,
        "bn_adaptation_batch_size": 256,
        "bn_adaptation_algorithm": "sequential_microbatch_v1",
        "bn_adaptation_enumeration_order": "view_major_then_sample_major",
        "bn_adaptation_applicable": True,
        "bn_adaptation_microbatches_at_primary_n": 50,
    }
    batching.update(overrides)
    return batching


def test_batchnorm_rejects_zero_microbatch_count(tmp_path):
    batching = _batchnorm_batching(bn_adaptation_microbatches_at_primary_n=0)
    metadata = _base_metadata("batchnorm", batching, n_samples=4)
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_batchnorm_predictions(),
            metrics=_batchnorm_metrics(),
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_batchnorm_rejects_missing_bn_adapted_metrics(tmp_path):
    batching = _batchnorm_batching()
    metadata = _base_metadata("batchnorm", batching, n_samples=4)
    metrics = _batchnorm_metrics()
    metrics["conditions"]["bn_adapted_tta"] = None
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_batchnorm_predictions(),
            metrics=metrics,
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_batchnorm_rejects_missing_bn_adapted_probability_arrays(tmp_path):
    batching = _batchnorm_batching()
    metadata = _base_metadata("batchnorm", batching, n_samples=4)
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_groupnorm_predictions(),  # no bn_adapted_probs
            metrics=_batchnorm_metrics(),
            metadata=metadata,
            view_manifest=_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


def test_batchnorm_valid_case_persists_successfully(tmp_path):
    batching = _batchnorm_batching()
    metadata = _base_metadata("batchnorm", batching, n_samples=4)
    manifest = persist_and_verify_evaluation_completion(
        tmp_path,
        predictions=_batchnorm_predictions(),
        metrics=_batchnorm_metrics(),
        metadata=metadata,
        view_manifest=_view_manifest(),
        prefix_sequence=_VALID_PREFIX_SEQUENCE,
    )
    assert "artifacts" in manifest


# ---------------------------------------------------------------------------
# 6-8: the real failed GroupNorm attempt (read-only, from the real ledger)
# ---------------------------------------------------------------------------


def test_real_failed_groupnorm_attempt_1_remains_unchanged():
    """Attempt 1's row is a permanently fixed historical fact -- checked
    by attempt number, not by assuming it is the only row for this
    run_id. A later real attempt (e.g. the corrected attempt 2 canary)
    legitimately adds more rows without invalidating this record."""
    import csv

    with open("artifacts/ledger_validation_evaluation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    by_attempt = {
        int(r["evaluation_attempt"]): r for r in rows if r["training_run_id"] == _REAL_GROUPNORM_RUN_ID
    }
    assert 1 in by_attempt
    row = by_attempt[1]
    assert row["status"] == "failed"
    assert row["evaluation_id"] == _REAL_GROUPNORM_FAILED_EVAL_ID
    assert (
        row["failure_reason"]
        == "batching.bn_adaptation_microbatches_at_primary_n must be a nonnegative integer, got None."
    )
    assert row["test_metrics_observed"] == "False"


def test_real_groupnorm_next_attempt_is_monotonic_and_gapless():
    """next_evaluation_attempt_number() must always equal
    max(existing attempt numbers for this run_id) + 1, read dynamically
    from the real ledger's current state -- not a hardcoded number that
    goes stale the instant another real attempt runs (exactly what
    happened to this test's predecessor once the attempt-2 canary
    completed)."""
    import csv

    with open("artifacts/ledger_validation_evaluation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    attempt_numbers = {
        int(r["evaluation_attempt"]) for r in rows if r["training_run_id"] == _REAL_GROUPNORM_RUN_ID
    }
    assert 1 in attempt_numbers  # historical floor must never shrink
    expected_next = max(attempt_numbers) + 1
    assert next_evaluation_attempt_number(_REAL_GROUPNORM_RUN_ID) == expected_next


def test_failed_attempt_under_stale_fingerprint_never_conflicts(tmp_path):
    """A completed evaluation under a DIFFERENT config hash is a hard
    conflict (ConflictingEvaluationImplementationError) -- but a FAILED
    attempt never is, regardless of which evaluator fingerprint it ran
    under, since check_evaluation_skip() only considers status=
    "completed" attempts for the matching/conflicting buckets. Uses a
    synthetic, isolated tmp_path scenario (not the mutable real ledger,
    which now also has a real completed attempt 2 for this run_id and so
    can no longer represent the "only a failed attempt exists" case)."""
    from when_tta_hurts.ledger import append_evaluation_entry
    from when_tta_hurts.validation_evaluation import (
        EvaluationRunStatus as _RunStatus,
    )
    from when_tta_hurts.validation_evaluation import (
        finish_evaluation_attempt,
        start_evaluation_attempt,
    )

    run_id = "synthetic-groupnorm-stale-fingerprint-run"
    ledger_path = tmp_path / "ledger.csv"
    old_hash = "old-fingerprint-hash-before-correction"

    attempt_dir, status = start_evaluation_attempt(run_id, old_hash, root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.FAILED, failure_reason="schema bug")
    append_evaluation_entry(
        evaluation_id=old_hash,
        training_run_id=run_id,
        training_attempt=1,
        checkpoint_hash="ckpt",
        evaluation_config_hash=old_hash,
        evaluation_attempt=status.attempt_number,
        status="failed",
        primary_artifact_hash="",
        started_at=status.started_at,
        ended_at="",
        runtime_seconds="",
        ledger_path=ledger_path,
    )

    new_hash = "new-fingerprint-hash-after-correction"
    try:
        skip = check_evaluation_skip(run_id, new_hash, root=tmp_path, ledger_path=ledger_path)
    except ConflictingEvaluationImplementationError:
        pytest.fail(
            "A failed attempt under the old evaluator fingerprint must never trigger "
            "ConflictingEvaluationImplementationError."
        )
    assert skip is None  # no completed attempt exists -- permits a new attempt
