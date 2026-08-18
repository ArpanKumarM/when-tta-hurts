"""Tests for block_d_gate.py -- uses ONLY fabricated benchmark records.
Never downloads or benchmarks anything real."""

import pytest

from when_tta_hurts.block_d_gate import (
    MAX_END_TO_END_MINUTES_PER_CELL,
    MAX_PESSIMISTIC_TOTAL_HOURS,
    MAX_TRAINING_MINUTES_PER_RUN,
    DatasetBenchmarkRecord,
    evaluate_block_d_gate,
)


def _passing_record(dataset: str) -> DatasetBenchmarkRecord:
    return DatasetBenchmarkRecord(
        dataset=dataset,
        artifact_is_native_128px=True,
        checksum_expected="abc123",
        checksum_actual="abc123",
        device="mps",
        oom_occurred=False,
        non_finite_loss_occurred=False,
        projected_training_minutes_per_run=45.0,
        projected_end_to_end_minutes_per_cell=60.0,
    )


def test_both_pass_activates():
    result = evaluate_block_d_gate(_passing_record("pathmnist"), _passing_record("bloodmnist"), 10.0)
    assert result.activated is True
    assert result.per_dataset_pass == {"pathmnist": True, "bloodmnist": True}


def test_pathmnist_fails_omits_entire_block():
    bad = DatasetBenchmarkRecord(**{**_passing_record("pathmnist").__dict__, "oom_occurred": True})
    result = evaluate_block_d_gate(bad, _passing_record("bloodmnist"), 10.0)
    assert result.activated is False
    assert result.per_dataset_pass["pathmnist"] is False
    assert result.per_dataset_pass["bloodmnist"] is True  # still individually true, but block omitted


def test_resized_proxy_rejected():
    bad = DatasetBenchmarkRecord(
        **{**_passing_record("pathmnist").__dict__, "artifact_is_native_128px": False}
    )
    result = evaluate_block_d_gate(bad, _passing_record("bloodmnist"), 10.0)
    assert result.activated is False
    assert any("resized proxy rejected" in r for r in result.reasons)


def test_checksum_mismatch_rejected():
    bad = DatasetBenchmarkRecord(**{**_passing_record("bloodmnist").__dict__, "checksum_actual": "wrong"})
    result = evaluate_block_d_gate(_passing_record("pathmnist"), bad, 10.0)
    assert result.activated is False


def test_non_mps_device_rejected():
    bad = DatasetBenchmarkRecord(**{**_passing_record("pathmnist").__dict__, "device": "cpu"})
    result = evaluate_block_d_gate(bad, _passing_record("bloodmnist"), 10.0)
    assert result.activated is False


def test_non_finite_loss_rejected():
    bad = DatasetBenchmarkRecord(
        **{**_passing_record("pathmnist").__dict__, "non_finite_loss_occurred": True}
    )
    result = evaluate_block_d_gate(bad, _passing_record("bloodmnist"), 10.0)
    assert result.activated is False


def test_training_minutes_at_boundary_passes():
    record = DatasetBenchmarkRecord(
        **{
            **_passing_record("pathmnist").__dict__,
            "projected_training_minutes_per_run": MAX_TRAINING_MINUTES_PER_RUN,
        }
    )
    result = evaluate_block_d_gate(record, _passing_record("bloodmnist"), 10.0)
    assert result.activated is True


def test_training_minutes_just_over_boundary_fails():
    record = DatasetBenchmarkRecord(
        **{
            **_passing_record("pathmnist").__dict__,
            "projected_training_minutes_per_run": MAX_TRAINING_MINUTES_PER_RUN + 0.1,
        }
    )
    result = evaluate_block_d_gate(record, _passing_record("bloodmnist"), 10.0)
    assert result.activated is False


def test_end_to_end_minutes_boundary():
    record = DatasetBenchmarkRecord(
        **{
            **_passing_record("bloodmnist").__dict__,
            "projected_end_to_end_minutes_per_cell": MAX_END_TO_END_MINUTES_PER_CELL,
        }
    )
    result = evaluate_block_d_gate(_passing_record("pathmnist"), record, 10.0)
    assert result.activated is True

    record_over = DatasetBenchmarkRecord(
        **{
            **_passing_record("bloodmnist").__dict__,
            "projected_end_to_end_minutes_per_cell": MAX_END_TO_END_MINUTES_PER_CELL + 0.1,
        }
    )
    result_over = evaluate_block_d_gate(_passing_record("pathmnist"), record_over, 10.0)
    assert result_over.activated is False


def test_pessimistic_total_boundary():
    result_pass = evaluate_block_d_gate(
        _passing_record("pathmnist"), _passing_record("bloodmnist"), MAX_PESSIMISTIC_TOTAL_HOURS - 0.01
    )
    assert result_pass.activated is True

    result_fail = evaluate_block_d_gate(
        _passing_record("pathmnist"), _passing_record("bloodmnist"), MAX_PESSIMISTIC_TOTAL_HOURS
    )
    assert result_fail.activated is False  # strictly less-than required


def test_decision_uses_no_accuracy_field():
    """Structural check: DatasetBenchmarkRecord has no accuracy/TTA-outcome
    field at all, so the decision cannot possibly be influenced by one."""
    fields = set(DatasetBenchmarkRecord.__dataclass_fields__.keys())
    for forbidden in ("accuracy", "delta_accuracy", "tta_outcome", "harm_rate", "rescue_rate"):
        assert forbidden not in fields


def test_wrong_dataset_name_raises():
    with pytest.raises(ValueError):
        evaluate_block_d_gate(_passing_record("bloodmnist"), _passing_record("pathmnist"), 10.0)
