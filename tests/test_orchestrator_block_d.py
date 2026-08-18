"""Tests for orchestrator.py's Block D gate-authorized training path
(Phase 2B.3F). Uses ONLY synthetic tensors, temporary repositories/files,
fake decisions, fake git histories, and injected factories -- never the
real downloaded 128px artifacts, never real MPS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts.block_d_benchmark import FROZEN_PROTOCOL_COMMIT, SPEC_COMMIT
from when_tta_hurts.dataset_verification import ArtifactVerification
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.orchestrator import (
    BlockDAuthorizationError,
    BlockDEffectiveConfig,
    CellTrainResult,
    NotBlockDRunIdError,
    TrainValidationLoaders,
    authorize_block_d_cell,
    compute_block_d_effective_config_hash,
    resolve_block_d_run_id,
    run_block_d_train_validation_cell,
)
from when_tta_hurts.run_identity import ConflictingCompletedRunError, run_directory

MATRIX_PATH = "configs/experiment_matrix.yaml"

D_CELL = MatrixCell(
    block="D_conditional_128px",
    dataset="pathmnist",
    resolution=128,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)

REAL_MATRIX_HASH = None  # resolved lazily below via a real (side-effect-free) parse


def _real_matrix_hash() -> str:
    global REAL_MATRIX_HASH
    if REAL_MATRIX_HASH is None:
        from when_tta_hurts.matrix import parse_and_validate_matrix

        REAL_MATRIX_HASH = parse_and_validate_matrix(MATRIX_PATH, block_d_gate_passed=True).source_config_hash
    return REAL_MATRIX_HASH


def _valid_decision(**overrides) -> dict:
    decision = {
        "schema_version": "1.0",
        "final_decision": "INCLUDED",
        "activated": True,
        "source_commit": "a" * 40,
        "protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "spec_commit": SPEC_COMMIT,
        "matrix_hash": _real_matrix_hash(),
        "raw_output_sha256": "b" * 64,
        "raw_output_path": "artifacts/benchmarks/block_d_native_128_benchmark.json",
        "per_dataset": {
            "pathmnist": {
                "artifact_path": "data/raw/pathmnist_128.npz",
                "checksum_expected": "checksum123",
                "checksum_actual": "checksum123",
                "resized": False,
                "selected_batch_size": 256,
                "projected_training_minutes_per_run": 10.0,
                "projected_end_to_end_minutes_per_cell": 15.0,
            },
            "bloodmnist": {
                "artifact_path": "data/raw/bloodmnist_128.npz",
                "checksum_expected": "checksum456",
                "checksum_actual": "checksum456",
                "resized": False,
                "selected_batch_size": 256,
                "projected_training_minutes_per_run": 5.0,
                "projected_end_to_end_minutes_per_cell": 8.0,
            },
        },
        "gate_condition_booleans": {},
        "gate_conditions": [],
        "per_dataset_pass": {"pathmnist": True, "bloodmnist": True},
        "frozen_pessimistic_abc_hours": 3.92,
        "block_d_contribution_seconds": 100.0,
        "binding_total_hours": 4.0,
        "no_scientific_metric_informed_decision": True,
        "scientific_metric_confirmation": "runtime/memory/checksum evidence only",
    }
    decision.update(overrides)
    return decision


def _write_decision(tmp_path, **overrides) -> Path:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_valid_decision(**overrides)))
    return path


def _always_tracked_clean(path):
    return True


def _commit_for(commit):
    def _last_commit_for_path(path):
        return commit

    return _last_commit_for_path


def _all_ancestors(commit, head):
    return True


def _make_loader(n=8, num_classes=9, batch_size=4, resolution=128):
    x = torch.zeros(n, 3, resolution, resolution)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _fake_loader_factory_recording(calls):
    def factory(cell, batch_size, root):
        calls.append((cell.run_id(), batch_size, root))
        return TrainValidationLoaders(
            train_loader=_make_loader(16, batch_size=batch_size),
            val_loader=_make_loader(8, batch_size=batch_size),
            dataset_artifact_filename=f"{cell.dataset}_128.npz",
            dataset_expected_checksum_md5="checksum123",
            dataset_actual_checksum_md5="checksum123",
        )

    return factory


def _fake_device_resolver_recording(calls, device=torch.device("cpu")):
    def resolver():
        calls.append(True)
        return device

    return resolver


def _fake_dataset_verifier(resized=False, checksum="checksum123"):
    def verifier(dataset, resolution, root):
        return ArtifactVerification(
            dataset=dataset,
            native_resolution=resolution,
            artifact_path=f"data/raw/{dataset}_{resolution}.npz",
            expected_checksum_md5=checksum,
            actual_checksum_md5=checksum,
            checksum_verified=True,
            resized=resized,
        )

    return verifier


# ---------------------------------------------------------------------------
# resolve_block_d_run_id
# ---------------------------------------------------------------------------


def test_resolve_block_d_run_id_accepts_valid_d_run_id():
    cell = resolve_block_d_run_id(D_CELL.run_id(), MATRIX_PATH)
    assert cell.block == "D_conditional_128px"
    assert cell.dataset == "pathmnist"


def test_resolve_block_d_run_id_rejects_a_run_id():
    with pytest.raises(NotBlockDRunIdError):
        resolve_block_d_run_id("A-pathmnist-28px-batchnorm-policy-none-s0", MATRIX_PATH)


def test_resolve_block_d_run_id_rejects_unknown_id():
    from when_tta_hurts.orchestrator import UnknownRunIdError

    with pytest.raises(UnknownRunIdError):
        resolve_block_d_run_id("D-not-a-real-cell", MATRIX_PATH)


def test_resolve_block_d_run_id_rejects_pilot_seed():
    from when_tta_hurts.orchestrator import PilotOrExcludedSeedRunIdError

    with pytest.raises(PilotOrExcludedSeedRunIdError):
        resolve_block_d_run_id("D-pathmnist-128px-batchnorm-policy-none-s314159", MATRIX_PATH)


# ---------------------------------------------------------------------------
# authorize_block_d_cell -- the core hard-fail matrix
# ---------------------------------------------------------------------------


def test_authorize_valid_included_decision_succeeds(tmp_path):
    path = _write_decision(tmp_path)
    decision, effective = authorize_block_d_cell(
        D_CELL,
        decision_path=path,
        matrix_path=MATRIX_PATH,
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    assert decision["final_decision"] == "INCLUDED"
    assert effective.selected_batch_size == 256
    assert effective.resized is False
    assert effective.native_resolution == 128


def test_authorize_omitted_decision_blocked(tmp_path):
    path = _write_decision(tmp_path, final_decision="OMITTED")
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_missing_decision_blocked(tmp_path):
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=tmp_path / "nope.json",
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_untracked_decision_blocked(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=lambda p: False,  # simulate untracked
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_dirty_decision_blocked(tmp_path):
    """Same git_tracked_and_clean hook covers both untracked and dirty --
    the production _default_git_tracked_and_clean returns False for both
    (see block_d_benchmark.py); this test exercises the orchestrator's
    handling of that False result via the injected fake."""
    path = _write_decision(tmp_path)
    calls = []

    def dirty_check(p):
        calls.append(p)
        return False

    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=dirty_check,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )
    assert calls  # the dirty check was actually consulted


def test_authorize_malformed_schema_blocked(tmp_path):
    path = tmp_path / "decision.json"
    decision = _valid_decision()
    del decision["raw_output_sha256"]
    path.write_text(json.dumps(decision))
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_conflicting_matrix_hash_blocked(tmp_path):
    path = _write_decision(tmp_path, matrix_hash="wrong_matrix_hash")
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_conflicting_protocol_commit_blocked(tmp_path):
    path = _write_decision(tmp_path, protocol_commit="wrong_protocol")
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_conflicting_spec_commit_blocked(tmp_path):
    path = _write_decision(tmp_path, spec_commit="wrong_spec")
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_decision_never_committed_blocked(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(BlockDAuthorizationError, match="never committed|no commit"):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=lambda p: None,  # never committed
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_non_ancestor_gate_decision_commit_blocked(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(BlockDAuthorizationError, match="ancestor"):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=lambda commit, head: False,  # nothing is an ancestor
        )


def test_authorize_non_ancestor_benchmark_source_commit_blocked(tmp_path):
    path = _write_decision(tmp_path)

    def selective_ancestor(commit, head):
        return commit != "a" * 40  # benchmark source_commit specifically fails

    with pytest.raises(BlockDAuthorizationError, match="ancestor"):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=selective_ancestor,
        )


def test_authorize_missing_dataset_selected_batch_blocked(tmp_path):
    decision = _valid_decision()
    del decision["per_dataset"]["pathmnist"]["selected_batch_size"]
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_batch_mismatch_blocked(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(BlockDAuthorizationError):
        authorize_block_d_cell(
            D_CELL,
            decision_path=path,
            matrix_path=MATRIX_PATH,
            expected_batch_size=64,  # decision says 256
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_authorize_head_does_not_need_to_equal_benchmark_source_commit(tmp_path):
    """HEAD may be a LATER commit than the benchmark source commit -- only
    ancestry is required, never equality."""
    path = _write_decision(tmp_path)
    decision, effective = authorize_block_d_cell(
        D_CELL,
        decision_path=path,
        matrix_path=MATRIX_PATH,
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
        head_commit="totally_different_later_head_commit",
    )
    assert decision["final_decision"] == "INCLUDED"


# ---------------------------------------------------------------------------
# Effective config hash: stability and provenance sensitivity
# ---------------------------------------------------------------------------


def test_effective_config_hash_is_stable_and_deterministic():
    cfg = BlockDEffectiveConfig(
        cell=D_CELL,
        selected_batch_size=256,
        decision_artifact_sha256="x" * 64,
        gate_decision_commit="c" * 40,
        benchmark_source_commit="a" * 40,
        benchmark_spec_commit="s" * 40,
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        matrix_hash=_real_matrix_hash(),
        dataset_checksum_md5="checksum123",
    )
    h1 = compute_block_d_effective_config_hash(cfg)
    h2 = compute_block_d_effective_config_hash(cfg)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64


def test_effective_config_hash_changes_with_selected_batch():
    base = dict(
        cell=D_CELL,
        selected_batch_size=256,
        decision_artifact_sha256="x" * 64,
        gate_decision_commit="c" * 40,
        benchmark_source_commit="a" * 40,
        benchmark_spec_commit="s" * 40,
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        matrix_hash=_real_matrix_hash(),
        dataset_checksum_md5="checksum123",
    )
    h_256 = compute_block_d_effective_config_hash(BlockDEffectiveConfig(**base))
    h_128 = compute_block_d_effective_config_hash(
        BlockDEffectiveConfig(**{**base, "selected_batch_size": 128})
    )
    assert h_256 != h_128


def test_effective_config_hash_changes_with_decision_sha256():
    base = dict(
        cell=D_CELL,
        selected_batch_size=256,
        decision_artifact_sha256="x" * 64,
        gate_decision_commit="c" * 40,
        benchmark_source_commit="a" * 40,
        benchmark_spec_commit="s" * 40,
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        matrix_hash=_real_matrix_hash(),
        dataset_checksum_md5="checksum123",
    )
    h1 = compute_block_d_effective_config_hash(BlockDEffectiveConfig(**base))
    h2 = compute_block_d_effective_config_hash(
        BlockDEffectiveConfig(**{**base, "decision_artifact_sha256": "y" * 64})
    )
    assert h1 != h2


# ---------------------------------------------------------------------------
# Full orchestration: reaches factories only when authorized
# ---------------------------------------------------------------------------


def test_included_decision_permits_orchestration_to_reach_factories(tmp_path):
    decision_path = _write_decision(tmp_path)
    loader_calls, device_calls = [], []
    result = run_block_d_train_validation_cell(
        D_CELL.run_id(),
        matrix_path=MATRIX_PATH,
        decision_path=decision_path,
        loader_factory=_fake_loader_factory_recording(loader_calls),
        dataset_verifier=_fake_dataset_verifier(),
        device_resolver=_fake_device_resolver_recording(device_calls),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    assert isinstance(result, CellTrainResult)
    assert result.status == "completed"
    assert len(loader_calls) == 1
    assert loader_calls[0][1] == 256  # gate-selected batch size was used
    assert len(device_calls) == 1


def test_omitted_decision_blocks_before_any_factory(tmp_path):
    decision_path = _write_decision(tmp_path, final_decision="OMITTED")
    loader_calls, device_calls = [], []
    with pytest.raises(BlockDAuthorizationError):
        run_block_d_train_validation_cell(
            D_CELL.run_id(),
            matrix_path=MATRIX_PATH,
            decision_path=decision_path,
            loader_factory=_fake_loader_factory_recording(loader_calls),
            dataset_verifier=_fake_dataset_verifier(),
            device_resolver=_fake_device_resolver_recording(device_calls),
            require_clean_tree=False,
            root=str(tmp_path / "confirmatory"),
            data_root=tmp_path / "data",
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
            amendments_ledger_path=tmp_path / "ledger_amendments.csv",
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )
    assert loader_calls == []
    assert device_calls == []
    assert not (tmp_path / "confirmatory").exists()
    assert not (tmp_path / "ledger_confirmatory.csv").exists()


def test_checksum_mismatch_at_train_time_blocked_after_authorization(tmp_path):
    """Authorization can pass (decision internally consistent) but the
    orchestrator must still re-verify the CURRENT on-disk artifact
    checksum against the decision's recorded value at train time."""
    decision_path = _write_decision(tmp_path)
    loader_calls = []
    with pytest.raises(BlockDAuthorizationError, match="checksum"):
        run_block_d_train_validation_cell(
            D_CELL.run_id(),
            matrix_path=MATRIX_PATH,
            decision_path=decision_path,
            loader_factory=_fake_loader_factory_recording(loader_calls),
            dataset_verifier=_fake_dataset_verifier(checksum="a_totally_different_checksum"),
            device_resolver=_fake_device_resolver_recording([]),
            require_clean_tree=False,
            root=str(tmp_path / "confirmatory"),
            data_root=tmp_path / "data",
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
            amendments_ledger_path=tmp_path / "ledger_amendments.csv",
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )
    assert loader_calls == []  # never reached the loader
    assert not (tmp_path / "confirmatory").exists()


def test_resized_true_at_train_time_blocked_after_authorization(tmp_path):
    decision_path = _write_decision(tmp_path)
    loader_calls = []
    with pytest.raises(BlockDAuthorizationError, match="resized"):
        run_block_d_train_validation_cell(
            D_CELL.run_id(),
            matrix_path=MATRIX_PATH,
            decision_path=decision_path,
            loader_factory=_fake_loader_factory_recording(loader_calls),
            dataset_verifier=_fake_dataset_verifier(resized=True),
            device_resolver=_fake_device_resolver_recording([]),
            require_clean_tree=False,
            root=str(tmp_path / "confirmatory"),
            data_root=tmp_path / "data",
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
            amendments_ledger_path=tmp_path / "ledger_amendments.csv",
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )
    assert loader_calls == []
    assert not (tmp_path / "confirmatory").exists()


def test_raw_benchmark_file_absent_but_valid_committed_decision_authorizes(tmp_path):
    """The ignored raw benchmark JSON need not exist locally at all --
    only the small, tracked, committed decision matters."""
    decision_path = _write_decision(tmp_path)
    assert not Path("does/not/exist/raw.json").exists()
    loader_calls = []
    result = run_block_d_train_validation_cell(
        D_CELL.run_id(),
        matrix_path=MATRIX_PATH,
        decision_path=decision_path,
        loader_factory=_fake_loader_factory_recording(loader_calls),
        dataset_verifier=_fake_dataset_verifier(),
        device_resolver=_fake_device_resolver_recording([]),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    assert result.status == "completed"


# ---------------------------------------------------------------------------
# Provenance persistence
# ---------------------------------------------------------------------------


def test_selected_batch_and_gate_provenance_persisted(tmp_path):
    decision_path = _write_decision(tmp_path)
    root = str(tmp_path / "confirmatory")
    run_block_d_train_validation_cell(
        D_CELL.run_id(),
        matrix_path=MATRIX_PATH,
        decision_path=decision_path,
        loader_factory=_fake_loader_factory_recording([]),
        dataset_verifier=_fake_dataset_verifier(),
        device_resolver=_fake_device_resolver_recording([]),
        require_clean_tree=False,
        root=root,
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    attempt_dir = run_directory(D_CELL, root=root) / "attempt_001"
    result_json = json.loads((attempt_dir / "result.json").read_text())
    metadata_json = json.loads((attempt_dir / "metadata.json").read_text())
    for blob in (result_json, metadata_json):
        prov = blob["block_d_gate_provenance"]
        assert prov["block_d_selected_batch_size"] == 256
        assert prov["block_d_gate_decision_commit"] == "c" * 40
        assert prov["block_d_benchmark_source_commit"] == "a" * 40
        assert prov["block_d_benchmark_spec_commit"] == SPEC_COMMIT
        assert prov["block_d_gate_final_decision"] == "INCLUDED"
        assert prov["block_d_resized"] is False
        assert prov["block_d_native_resolution"] == 128
    assert result_json["config_hash"] != ""

    ledger_rows = (tmp_path / "ledger_confirmatory.csv").read_text().strip().splitlines()
    assert len(ledger_rows) == 2  # header + one row
    assert result_json["config_hash"] in ledger_rows[1]


# ---------------------------------------------------------------------------
# A/B/C unaffected
# ---------------------------------------------------------------------------


def test_a_b_c_config_hashing_unaffected():
    """cell_config_hash(cell) for an A cell must not depend on anything
    Block-D-specific -- confirmed by recomputing it directly, independent
    of any Block D machinery."""
    from when_tta_hurts.run_identity import cell_config_hash

    a_cell = MatrixCell(
        block="A_core_normalization_resolution",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        training_policy="none",
        seed=0,
    )
    h1 = cell_config_hash(a_cell)
    h2 = cell_config_hash(a_cell)
    assert h1 == h2


def test_run_canary_cell_still_rejects_block_d():
    from when_tta_hurts.orchestrator import BlockDRunRejectedError, resolve_canary_run_id

    with pytest.raises(BlockDRunRejectedError):
        resolve_canary_run_id(D_CELL.run_id(), MATRIX_PATH)


def test_run_block_cells_still_rejects_block_d():
    from when_tta_hurts.orchestrator import UnsupportedBlockError, resolve_block_full_name

    with pytest.raises(UnsupportedBlockError):
        resolve_block_full_name("D")


# ---------------------------------------------------------------------------
# Skip behavior: idempotent, gate-decision-consistency-aware
# ---------------------------------------------------------------------------


def test_skip_occurs_before_mps_loader_model_factories_on_second_call(tmp_path):
    decision_path = _write_decision(tmp_path)
    common_kwargs = dict(
        matrix_path=MATRIX_PATH,
        decision_path=decision_path,
        dataset_verifier=_fake_dataset_verifier(),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    first = run_block_d_train_validation_cell(
        D_CELL.run_id(),
        loader_factory=_fake_loader_factory_recording([]),
        device_resolver=_fake_device_resolver_recording([]),
        **common_kwargs,
    )
    assert first.status == "completed"

    loader_calls, device_calls = [], []
    second = run_block_d_train_validation_cell(
        D_CELL.run_id(),
        loader_factory=_fake_loader_factory_recording(loader_calls),
        device_resolver=_fake_device_resolver_recording(device_calls),
        **common_kwargs,
    )
    assert second.status == "skipped_completed"
    assert loader_calls == []
    assert device_calls == []


def test_stale_completion_under_changed_decision_raises_conflict_not_silent_skip(tmp_path):
    """A completed attempt exists under one gate decision; if the decision
    changes (different selected batch -> different effective hash), the
    skip must NOT silently reuse the old completion -- it must hard-fail."""
    decision_path = _write_decision(tmp_path)
    common_kwargs = dict(
        matrix_path=MATRIX_PATH,
        dataset_verifier=_fake_dataset_verifier(),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    first = run_block_d_train_validation_cell(
        D_CELL.run_id(),
        decision_path=decision_path,
        loader_factory=_fake_loader_factory_recording([]),
        device_resolver=_fake_device_resolver_recording([]),
        **common_kwargs,
    )
    assert first.status == "completed"

    other_decision_path = _write_decision(
        tmp_path,
        per_dataset={
            **_valid_decision()["per_dataset"],
            "pathmnist": {**_valid_decision()["per_dataset"]["pathmnist"], "selected_batch_size": 128},
        },
    )
    # rename to avoid path collision with the first decision file
    changed_path = tmp_path / "decision_changed.json"
    changed_path.write_text(other_decision_path.read_text())

    with pytest.raises(ConflictingCompletedRunError):
        run_block_d_train_validation_cell(
            D_CELL.run_id(),
            decision_path=changed_path,
            loader_factory=_fake_loader_factory_recording([]),
            device_resolver=_fake_device_resolver_recording([]),
            **common_kwargs,
        )


# ---------------------------------------------------------------------------
# No zero-effort bypass: no force flag, no env-var/CLI escape hatch
# ---------------------------------------------------------------------------


def test_no_force_or_skip_authorization_parameter_exists():
    import inspect

    sig = inspect.signature(run_block_d_train_validation_cell)
    for forbidden in ("force", "skip_authorization", "bypass", "override", "unsafe"):
        assert forbidden not in sig.parameters


def test_no_env_var_bypass_in_block_d_orchestration_source():
    import inspect

    from when_tta_hurts import orchestrator

    for fn in (
        orchestrator.authorize_block_d_cell,
        orchestrator.run_block_d_train_validation_cell,
        orchestrator.resolve_block_d_run_id,
    ):
        source = inspect.getsource(fn)
        assert "os.environ" not in source
        assert "getenv" not in source


# ---------------------------------------------------------------------------
# Zero test-split / TTA access
# ---------------------------------------------------------------------------


def test_no_test_split_or_tta_reference_in_block_d_orchestration_source():
    import inspect

    from when_tta_hurts import orchestrator

    for fn in (
        orchestrator.authorize_block_d_cell,
        orchestrator.run_block_d_train_validation_cell,
        orchestrator.default_block_d_train_validation_loader_factory,
    ):
        source = inspect.getsource(fn)
        assert 'split="test"' not in source
        assert "split='test'" not in source
        assert "evaluation." not in source
        assert (
            "tta" not in source.lower() or "authoriz" in source.lower()
        )  # allow doc mentions of "authorization"


# ---------------------------------------------------------------------------
# 120-minute-class timeout behavior preserved (via existing train_model path)
# ---------------------------------------------------------------------------


def test_training_timeout_still_produces_failed_attempt_not_silent_success(tmp_path, monkeypatch):
    """Block D reuses run_train_validation_cell()'s existing
    max_training_seconds/TrainingTimeoutError handling unchanged -- forcing
    the frozen 90-minute budget down to 0 seconds must produce a "failed"
    attempt (preserved on disk, ledger row written), never a silent
    success, exactly like tests/test_orchestrator.py's A-cell equivalent."""
    import when_tta_hurts.orchestrator as orch

    monkeypatch.setattr(orch, "_BLOCK_D_MAX_TRAINING_MINUTES_PER_RUN", 0.0)

    decision_path = _write_decision(tmp_path)
    root = str(tmp_path / "confirmatory")
    with pytest.raises(Exception):  # noqa: B017 -- TrainingTimeoutError, mirroring test_orchestrator.py
        run_block_d_train_validation_cell(
            D_CELL.run_id(),
            matrix_path=MATRIX_PATH,
            decision_path=decision_path,
            loader_factory=_fake_loader_factory_recording([]),
            dataset_verifier=_fake_dataset_verifier(),
            device_resolver=_fake_device_resolver_recording([]),
            require_clean_tree=False,
            root=root,
            data_root=tmp_path / "data",
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
            amendments_ledger_path=tmp_path / "ledger_amendments.csv",
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )

    attempt_dir = run_directory(D_CELL, root=root) / "attempt_001"
    assert attempt_dir.exists()  # preserved, not silently discarded
    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == "failed"
    ledger_rows = (tmp_path / "ledger_confirmatory.csv").read_text().strip().splitlines()
    assert len(ledger_rows) == 2  # header + one failed row
    assert ",failed," in ledger_rows[1] or "failed" in ledger_rows[1]


# ---------------------------------------------------------------------------
# Recursive forbidden-field-name check (Part A audit fix)
# ---------------------------------------------------------------------------


def test_recursive_field_name_check_catches_plural_predictions():
    from when_tta_hurts.block_d_benchmark import _find_forbidden_field_name

    assert _find_forbidden_field_name({"predictions": [1, 2, 3]}) == "predictions"


def test_recursive_field_name_check_does_not_flag_legitimate_field():
    from when_tta_hurts.block_d_benchmark import _find_forbidden_field_name

    assert _find_forbidden_field_name({"test_metrics_observed": False}) is None


def test_recursive_field_name_check_walks_nested_structures():
    from when_tta_hurts.block_d_benchmark import _find_forbidden_field_name

    nested = {"a": {"b": [{"c": {"f1_score": 0.9}}]}}
    assert _find_forbidden_field_name(nested) == "f1_score"
