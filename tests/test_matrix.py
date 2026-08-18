"""Tests for src/when_tta_hurts/matrix.py -- the fail-closed confirmatory
matrix parser. Only reads configs/experiment_matrix.yaml and temporary
fixture YAML files; never loads any dataset."""

import copy

import pytest
import yaml

from when_tta_hurts.matrix import (
    BLOCK_ORDER,
    FROZEN_TRAINING_SETTINGS,
    MatrixValidationError,
    parse_and_validate_matrix,
)


def test_real_matrix_parses_and_counts_are_exact():
    expanded = parse_and_validate_matrix()
    assert len(expanded.cells_by_block["A_core_normalization_resolution"]) == 24
    assert len(expanded.cells_by_block["B_policy_matching"]) == 6
    assert len(expanded.cells_by_block["C_positive_control_reproduction"]) == 3
    assert len(expanded.cells) == 33
    assert expanded.block_d_included is False


def test_real_matrix_with_block_d_gate_passed():
    expanded = parse_and_validate_matrix(block_d_gate_passed=True)
    assert len(expanded.cells_by_block["D_conditional_128px"]) == 6
    assert len(expanded.cells) == 39
    assert expanded.block_d_included is True


def test_expansion_order_is_literal_committed_order():
    expanded = parse_and_validate_matrix()
    blocks_seen = [cell.block for cell in expanded.cells]
    # Block A's 24 cells first, then B's 6, then C's 3 -- matching BLOCK_ORDER.
    assert blocks_seen == (
        ["A_core_normalization_resolution"] * 24
        + ["B_policy_matching"] * 6
        + ["C_positive_control_reproduction"] * 3
    )
    # Within block A: dataset-major, then resolution, then normalization, then seed.
    a_cells = expanded.cells_by_block["A_core_normalization_resolution"]
    assert [c.dataset for c in a_cells[:12]] == ["pathmnist"] * 12
    assert [c.dataset for c in a_cells[12:]] == ["bloodmnist"] * 12
    assert [c.seed for c in a_cells[:3]] == [0, 1, 2]


def test_no_duplicate_run_ids():
    expanded = parse_and_validate_matrix(block_d_gate_passed=True)
    run_ids = [c.run_id() for c in expanded.cells]
    assert len(run_ids) == len(set(run_ids))


def test_seed_314159_forbidden_in_matrix(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["seeds"]["confirmatory"] = [0, 1, 314159]
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="314159"):
        parse_and_validate_matrix(bad_path)


def test_confirmatory_seeds_must_be_exactly_0_1_2(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["seeds"]["confirmatory"] = [0, 1, 2, 3]
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match=r"\[0, 1, 2\]"):
        parse_and_validate_matrix(bad_path)


def test_draft_status_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["status"] = "draft"
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="draft"):
        parse_and_validate_matrix(bad_path)


def test_unknown_top_level_field_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["some_unknown_field"] = "surprise"
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="unknown field"):
        parse_and_validate_matrix(bad_path)


def test_unknown_block_field_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["training_matrix"]["A_core_normalization_resolution"]["mystery_field"] = 1
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="unknown field"):
        parse_and_validate_matrix(bad_path)


def test_missing_required_field_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    del raw["training_matrix"]["A_core_normalization_resolution"]["seeds"]
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="missing required field"):
        parse_and_validate_matrix(bad_path)


def test_wrong_training_runs_count_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["training_matrix"]["A_core_normalization_resolution"]["training_runs"] = 999
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="count mismatch"):
        parse_and_validate_matrix(bad_path)


def test_unregistered_dataset_reference_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["training_matrix"]["C_positive_control_reproduction"]["datasets"] = ["not_a_real_dataset"]
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="unregistered dataset"):
        parse_and_validate_matrix(bad_path)


def test_unregistered_normalization_for_model_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/experiment_matrix.yaml"))
    raw["training_matrix"]["C_positive_control_reproduction"]["normalization"] = "groupnorm"
    bad_path = tmp_path / "bad_matrix.yaml"
    bad_path.write_text(yaml.dump(raw))
    with pytest.raises(MatrixValidationError, match="not registered for"):
        parse_and_validate_matrix(bad_path)


def test_block_d_excluded_by_default():
    expanded = parse_and_validate_matrix()
    assert expanded.cells_by_block["D_conditional_128px"] == ()


def test_frozen_training_settings_match_protocol():
    # Cross-check against docs/phase2b_protocol.md sec.2's frozen values.
    assert FROZEN_TRAINING_SETTINGS.optimizer == "adam"
    assert FROZEN_TRAINING_SETTINGS.learning_rate == 0.001
    assert FROZEN_TRAINING_SETTINGS.weight_decay == 0.0
    assert FROZEN_TRAINING_SETTINGS.max_epochs == 30
    assert FROZEN_TRAINING_SETTINGS.early_stopping_patience == 5
    assert FROZEN_TRAINING_SETTINGS.early_stopping_min_delta == 0.0
    assert FROZEN_TRAINING_SETTINGS.batch_size_28_64px == 256
    assert FROZEN_TRAINING_SETTINGS.precision == "float32"
    assert FROZEN_TRAINING_SETTINGS.mixed_precision is False
    assert FROZEN_TRAINING_SETTINGS.label_smoothing is False
    assert FROZEN_TRAINING_SETTINGS.class_weighting is False
    assert FROZEN_TRAINING_SETTINGS.channel_standardization is False


def test_block_order_is_literal_tuple_not_dict():
    assert BLOCK_ORDER == (
        "A_core_normalization_resolution",
        "B_policy_matching",
        "C_positive_control_reproduction",
        "D_conditional_128px",
    )
    assert isinstance(BLOCK_ORDER, tuple)


def test_reparsing_does_not_mutate_shared_state():
    m1 = parse_and_validate_matrix()
    m2 = parse_and_validate_matrix()
    assert m1.cells == m2.cells
    # deep copy sanity: mutating one dataclass instance's tuple contents is
    # not possible (frozen), confirming no shared mutable state leaks.
    assert copy.deepcopy(m1.cells) == m2.cells
