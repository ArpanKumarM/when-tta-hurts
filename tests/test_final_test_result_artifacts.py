"""Phase 2B.6A: synthetic tests for final_test_result_artifacts.py's
schema validation and persistence. All fixtures are hand-constructed
in-memory or written to tmp_path -- no real evaluation artifact is ever
read or written.
"""

from __future__ import annotations

import numpy as np
import pytest

from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
)
from when_tta_hurts.final_test_result_artifacts import (
    build_final_test_artifact_manifest,
    persist_and_verify_final_test_completion,
    verify_final_test_artifact_manifest,
)


def _valid_metadata(**overrides):
    metadata = {
        "final_test_evaluation_id": "ft-id",
        "training_run_id": "run-a",
        "training_attempt": 1,
        "checkpoint_hash": "chk",
        "dataset": "pathmnist",
        "resolution": 28,
        "model": "small_cnn",
        "normalization": "batchnorm",
        "training_policy": "none",
        "seed": 0,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "sha",
        "tta_seed_freeze_commit": "commit",
        "tta_seed_derivation_sha256": "deriv",
        "prefix_sequence": [1, 2, 5, 10, 25, 50, 100],
        "aggregators": ["mean_probability"],
        "secondary_analyses": [],
        "protocol_commit": "ce4c962",
        "matrix_hash": "matrix-hash",
        "source_commit": "src-commit",
        "evaluator_fingerprint": "fp",
        "evaluator_fingerprint_manifest": {},
        "dataset_expected_checksum_md5": "0" * 32,
        "dataset_verification": {
            "dataset": "pathmnist",
            "resolution": 28,
            "expected_checksum_md5": "0" * 32,
            "actual_checksum_md5": "0" * 32,
            "checksum_verified": True,
            "resized": False,
            "verification_method": "m",
            "verification_version": 1,
        },
        "batching": {
            "inference_batch_size": 256,
            "bn_adaptation_batch_size": 256,
            "bn_adaptation_algorithm": "sequential_microbatch_v1",
            "bn_adaptation_enumeration_order": "ascending",
            "bn_adaptation_applicable": True,
            "bn_adaptation_microbatches_at_primary_n": 1,
        },
        "metric_input_contract": "probability_native_v1",
        "evaluation_config_hash": "ft-id",
        "split": "test",
        "n_test_samples": 2,
        "statistical_analysis_fingerprint": "sa-fp",
        "cross_condition_analysis_fingerprint": "cc-fp",
        "final_test_runner_fingerprint": "runner-fp",
        "authorization_artifact_sha256": "auth-sha",
        "authorization_commit": "auth-commit",
        "test_split_accessed": True,
        "test_predictions_computed": True,
        "test_metrics_computed": True,
    }
    metadata.update(overrides)
    return metadata


def _valid_view_manifest():
    return {
        "dataset": "pathmnist",
        "resolution": 28,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "sha",
        "tta_seed_freeze_commit": "commit",
        "tta_seed_derivation_sha256": "deriv",
        "n_views": 100,
        "seed_formula": "formula",
        "sample_indices": [0, 1],
        "seed_manifest_sha256": "seed-sha",
    }


def _valid_predictions():
    labels = np.array([0, 1])
    clean_probs = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    view_probs = np.stack([clean_probs] * 100, axis=0)
    bn_adapted_probs = np.stack([clean_probs, clean_probs], axis=0)
    return {
        "labels": labels,
        "sample_indices": np.array([0, 1]),
        "clean_probs": clean_probs,
        "view_probs": view_probs,
        "bn_adapted_probs": bn_adapted_probs,
        "bn_adapted_prefix_sequence": np.array([50, 100]),
    }


def _valid_metrics():
    from when_tta_hurts.metrics import accuracy

    labels = np.array([0, 1])
    log_probs = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    acc = accuracy(log_probs, labels)
    return {
        "training_run_id": "run-a",
        "evaluation_config_hash": "ft-id",
        "clean": {"accuracy": acc},
        "conditions": {"bn_adapted_tta": {50: {"accuracy": acc}, 100: {"accuracy": acc}}},
        "latency": {
            "clean_latency_seconds": 1.0,
            "n_samples": 2,
            "by_n": {
                str(n): {
                    "tta_latency_seconds": 1.0,
                    "per_sample_latency_seconds": 0.5,
                    "compute_multiplier": 1.0,
                }
                for n in (1, 2, 5, 10, 25, 50, 100)
            },
        },
    }


def test_split_must_be_test_not_validation(tmp_path):
    with pytest.raises(EvaluationSchemaValidationError, match="split must be 'test'"):
        persist_and_verify_final_test_completion(
            tmp_path,
            predictions=_valid_predictions(),
            metrics=_valid_metrics(),
            metadata=_valid_metadata(split="validation"),
            view_manifest=_valid_view_manifest(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        )
    assert not (tmp_path / "metadata.json").exists()


def test_test_split_accessed_must_be_true(tmp_path):
    with pytest.raises(EvaluationSchemaValidationError, match="test_split_accessed must be True"):
        persist_and_verify_final_test_completion(
            tmp_path,
            predictions=_valid_predictions(),
            metrics=_valid_metrics(),
            metadata=_valid_metadata(test_split_accessed=False),
            view_manifest=_valid_view_manifest(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        )


def test_missing_metadata_key_rejected(tmp_path):
    metadata = _valid_metadata()
    del metadata["final_test_runner_fingerprint"]
    with pytest.raises(EvaluationSchemaValidationError, match="missing required keys"):
        persist_and_verify_final_test_completion(
            tmp_path,
            predictions=_valid_predictions(),
            metrics=_valid_metrics(),
            metadata=metadata,
            view_manifest=_valid_view_manifest(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        )


def test_round_trip_persists_and_verifies(tmp_path):
    manifest = persist_and_verify_final_test_completion(
        tmp_path,
        predictions=_valid_predictions(),
        metrics=_valid_metrics(),
        metadata=_valid_metadata(),
        view_manifest=_valid_view_manifest(),
        prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
    )
    assert (tmp_path / "predictions.npz").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metadata.json").exists()
    verify_final_test_artifact_manifest(tmp_path, manifest)


def test_manifest_tamper_detection(tmp_path):
    manifest = persist_and_verify_final_test_completion(
        tmp_path,
        predictions=_valid_predictions(),
        metrics=_valid_metrics(),
        metadata=_valid_metadata(),
        view_manifest=_valid_view_manifest(),
        prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
    )
    (tmp_path / "metrics.json").write_text('{"tampered": true}')
    with pytest.raises(EvaluationPersistenceError):
        verify_final_test_artifact_manifest(tmp_path, manifest)


def test_build_manifest_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(EvaluationPersistenceError, match="does not exist"):
        build_final_test_artifact_manifest(tmp_path)


def test_bn_adaptation_inapplicable_groupnorm_contract(tmp_path):
    metadata = _valid_metadata(
        normalization="groupnorm",
        batching={
            "inference_batch_size": 256,
            "bn_adaptation_batch_size": 256,
            "bn_adaptation_algorithm": "sequential_microbatch_v1",
            "bn_adaptation_enumeration_order": "ascending",
            "bn_adaptation_applicable": False,
            "bn_adaptation_microbatches_at_primary_n": 0,
        },
    )
    predictions = _valid_predictions()
    del predictions["bn_adapted_probs"]
    del predictions["bn_adapted_prefix_sequence"]
    metrics = _valid_metrics()
    metrics["conditions"] = {"bn_adapted_tta": None}

    manifest = persist_and_verify_final_test_completion(
        tmp_path,
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        view_manifest=_valid_view_manifest(),
        prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
    )
    assert manifest["artifacts"]


def test_bn_adaptation_contradiction_rejected(tmp_path):
    """BatchNorm claims bn_adaptation_applicable=True but predictions has
    no bn_adapted_probs evidence -- must fail closed."""
    metadata = _valid_metadata()
    predictions = _valid_predictions()
    del predictions["bn_adapted_probs"]
    del predictions["bn_adapted_prefix_sequence"]
    metrics = _valid_metrics()
    metrics["conditions"] = {"bn_adapted_tta": None}

    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_final_test_completion(
            tmp_path,
            predictions=predictions,
            metrics=metrics,
            metadata=metadata,
            view_manifest=_valid_view_manifest(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        )


def test_metric_recomputation_mismatch_rejected(tmp_path):
    with pytest.raises(EvaluationPersistenceError, match="recomputation mismatch"):
        persist_and_verify_final_test_completion(
            tmp_path,
            predictions=_valid_predictions(),
            metrics=_valid_metrics(),
            metadata=_valid_metadata(),
            view_manifest=_valid_view_manifest(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
            metric_recomputers={"fake.metric": (0.5, lambda: 0.999)},
        )
    assert not (tmp_path / "metadata.json").exists()


def test_validation_and_final_test_schemas_never_mixed():
    """The two modules' metadata schemas are structurally disjoint enough
    that a valid final-test metadata dict fails the VALIDATION module's
    schema (wrong split), and vice versa -- artifacts are never mixed."""
    from when_tta_hurts.evaluation_result_artifacts import _validate_metadata_schema as validate_validation
    from when_tta_hurts.final_test_result_artifacts import _validate_metadata_schema as validate_final_test

    final_test_metadata = _valid_metadata()
    with pytest.raises(EvaluationSchemaValidationError):
        validate_validation(final_test_metadata)

    validation_metadata = dict(final_test_metadata)
    validation_metadata["split"] = "validation"
    validation_metadata["n_validation_samples"] = 2
    with pytest.raises(EvaluationSchemaValidationError):
        validate_final_test(validation_metadata)
