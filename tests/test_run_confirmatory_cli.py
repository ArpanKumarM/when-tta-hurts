"""Phase 2B.3A Part A tests: scripts/run_confirmatory.py CLI argument
handling. Never actually executes train-validation mode against real data
-- those code paths are exercised via orchestrator.run_canary_cell directly
in tests/test_canary_resolution.py. This file only checks argparse-level
behavior and static source guarantees.

Phase 2B.3G-Engineering adds: CLI-level dispatch tests for the Block D
--run-id route (resolve-by-matrix-block -> run_block_d_train_validation_cell
vs run_canary_cell). SAFETY: the real repo has an INCLUDED Block D gate
decision and real downloaded 128px artifacts on disk, so EVERY test that
exercises the D dispatch path monkeypatches cli_module's bound name for
run_block_d_train_validation_cell (never leaving production defaults live)
-- either with a pure recording fake, or with functools.partial(REAL
run_block_d_train_validation_cell, decision_path=<tmp>, device_resolver=
<fake cpu>, loader_factory=<fake>, dataset_verifier=<fake>,
git_tracked_and_clean=<fake>, ...) so authorization logic is exercised
for real while every side-effecting boundary (MPS, real dataset, real
decision path) is safely substituted."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_confirmatory.py"
MATRIX_PATH = "configs/experiment_matrix.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_confirmatory", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


class _FakeCellResult:
    """Minimal stand-in for orchestrator.CellTrainResult, matching every
    attribute _print_cell_result() reads."""

    def __init__(self, status="completed", run_id="fake", attempt_number=1):
        self.status = status
        self.run_id = run_id
        self.attempt_number = attempt_number
        self.checkpoint_hash = "fakehash"
        self.config_hash = "fakeconfighash"
        self.manifest_verified = True


def test_omitted_run_id_fails(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "requires exactly one of --run-id or --block" in capsys.readouterr().err


def test_multiple_run_ids_fail(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confirmatory.py",
            "train-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s0",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s1",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "only be specified once" in capsys.readouterr().err


def test_block_d_run_id_dispatches_to_authorized_function_not_ordinary_canary(
    cli_module, monkeypatch, capsys
):
    """CRITICAL SAFETY NOTE: the real repo has an INCLUDED gate decision and
    real downloaded 128px artifacts on disk, so this test must NEVER let
    cli_module.main() call the real run_block_d_train_validation_cell()
    with its production defaults (that would really initialize MPS and
    really train a cell). It always monkeypatches BOTH dispatch targets
    with recording fakes before invoking main()."""
    d_calls = []
    canary_calls = []

    def fake_run_block_d(run_id, matrix_path):
        d_calls.append((run_id, matrix_path))
        return _FakeCellResult()

    def fake_run_canary(run_id, matrix_path):
        canary_calls.append((run_id, matrix_path))
        return _FakeCellResult()

    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", fake_run_block_d)
    monkeypatch.setattr(cli_module, "run_canary_cell", fake_run_canary)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_confirmatory.py", "train-validation", "--run-id", "D-pathmnist-128px-batchnorm-policy-none-s0"],
    )
    rc = cli_module.main()
    assert rc == 0
    assert len(d_calls) == 1
    assert d_calls[0][0] == "D-pathmnist-128px-batchnorm-policy-none-s0"
    assert canary_calls == []


def test_pilot_seed_run_id_fails(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confirmatory.py",
            "train-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s314159",
        ],
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_unknown_run_id_fails(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", "not-a-real-id"])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_no_all_range_block_wildcard_flags_exist():
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        'add_argument("--all"',
        'add_argument("--range"',
        'add_argument("--block"',
        'add_argument("--wildcard"',
        "argparse.REMAINDER",
        "nargs=",  # --run-id must be a plain single string, not a list-accepting nargs
    ):
        assert forbidden not in source


def test_no_cli_or_env_bypass_of_synthetic_backend():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source
    assert "synthetic" not in source.lower()
    assert "--force" not in source
    assert "--skip-checksum" not in source
    assert "--cpu" not in source  # no explicit CPU-fallback override flag


def test_plan_mode_unaffected_and_still_side_effect_free(cli_module, monkeypatch, capsys, tmp_path):
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        matrix_path = os.path.join(cwd, "configs/experiment_matrix.yaml")
        monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "plan", "--matrix", matrix_path])
        rc = cli_module.main()
        assert rc == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        os.chdir(cwd)


def test_final_test_mode_unchanged_fails_closed(cli_module, monkeypatch):
    from when_tta_hurts.authorization import AuthorizationError

    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "final-test"])
    with pytest.raises(AuthorizationError):
        cli_module.main()


# ---------------------------------------------------------------------------
# Phase 2B.3G-Engineering: Block D --run-id dispatch (all synthetic/injected)
# ---------------------------------------------------------------------------


def _real_matrix_hash():
    from when_tta_hurts.matrix import parse_and_validate_matrix

    return parse_and_validate_matrix(MATRIX_PATH, block_d_gate_passed=True).source_config_hash


def _valid_decision(**overrides):
    from when_tta_hurts.block_d_benchmark import FROZEN_PROTOCOL_COMMIT, SPEC_COMMIT

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


def _write_decision(tmp_path, name="decision.json", **overrides):
    path = tmp_path / name
    path.write_text(json.dumps(_valid_decision(**overrides)))
    return path


def _always_tracked_clean(path):
    return True


def _commit_for(commit):
    return lambda path: commit


def _all_ancestors(commit, head):
    return True


def _make_loader(n=8, num_classes=9, batch_size=4, resolution=128):
    x = torch.zeros(n, 3, resolution, resolution)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _fake_loader_factory(calls=None):
    from when_tta_hurts.orchestrator import TrainValidationLoaders

    def factory(cell, batch_size, root):
        if calls is not None:
            calls.append((cell.run_id(), batch_size))
        return TrainValidationLoaders(
            train_loader=_make_loader(16, batch_size=batch_size),
            val_loader=_make_loader(8, batch_size=batch_size),
            dataset_artifact_filename=f"{cell.dataset}_128.npz",
            dataset_expected_checksum_md5="checksum123",
            dataset_actual_checksum_md5="checksum123",
        )

    return factory


def _fake_device_resolver(calls=None, device=torch.device("cpu")):
    def resolver():
        if calls is not None:
            calls.append(True)
        return device

    return resolver


def _fake_dataset_verifier(resized=False, checksum="checksum123"):
    from when_tta_hurts.dataset_verification import ArtifactVerification

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


def _bound_real_block_d(
    tmp_path,
    decision_path=None,
    loader_calls=None,
    device_calls=None,
    resized=False,
    checksum="checksum123",
    root_suffix="confirmatory",
):
    """functools.partial-bind the REAL run_block_d_train_validation_cell to
    entirely temporary/synthetic dependencies -- exercises the actual
    authorization/dispatch/persistence logic without ever touching real
    MPS, real 128px data, or the real decision/ledger paths."""
    from when_tta_hurts.orchestrator import run_block_d_train_validation_cell

    if decision_path is None:
        decision_path = _write_decision(tmp_path)
    return functools.partial(
        run_block_d_train_validation_cell,
        decision_path=decision_path,
        loader_factory=_fake_loader_factory(loader_calls),
        dataset_verifier=_fake_dataset_verifier(resized=resized, checksum=checksum),
        device_resolver=_fake_device_resolver(device_calls),
        require_clean_tree=False,
        root=str(tmp_path / root_suffix),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )


D_S0 = "D-pathmnist-128px-batchnorm-policy-none-s0"
D_S1 = "D-pathmnist-128px-batchnorm-policy-none-s1"
D_S2 = "D-pathmnist-128px-batchnorm-policy-none-s2"
D_BLOOD_S0 = "D-bloodmnist-128px-batchnorm-policy-none-s0"


@pytest.mark.parametrize("run_id", [D_S0, D_S1, D_S2, D_BLOOD_S0])
def test_all_block_d_run_ids_dispatch_to_authorized_function(cli_module, monkeypatch, run_id):
    d_calls, canary_calls = [], []
    monkeypatch.setattr(
        cli_module,
        "run_block_d_train_validation_cell",
        lambda rid, matrix_path: (d_calls.append((rid, matrix_path)), _FakeCellResult(run_id=rid))[1],
    )
    monkeypatch.setattr(
        cli_module,
        "run_canary_cell",
        lambda rid, matrix_path: (canary_calls.append((rid, matrix_path)), _FakeCellResult(run_id=rid))[1],
    )
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", run_id])
    rc = cli_module.main()
    assert rc == 0
    assert len(d_calls) == 1 and d_calls[0][0] == run_id
    assert canary_calls == []


def test_a_run_id_still_dispatches_to_run_canary_cell_unchanged_args(cli_module, monkeypatch):
    d_calls, canary_calls = [], []
    monkeypatch.setattr(
        cli_module,
        "run_block_d_train_validation_cell",
        lambda rid, matrix_path: (d_calls.append((rid, matrix_path)), _FakeCellResult(run_id=rid))[1],
    )
    monkeypatch.setattr(
        cli_module,
        "run_canary_cell",
        lambda rid, matrix_path: (canary_calls.append((rid, matrix_path)), _FakeCellResult(run_id=rid))[1],
    )
    run_id = "A-pathmnist-28px-batchnorm-policy-none-s0"
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", run_id])
    rc = cli_module.main()
    assert rc == 0
    assert canary_calls == [(run_id, MATRIX_PATH)]  # same positional/keyword shape as before this change
    assert d_calls == []


def test_unknown_run_id_never_dispatches_to_block_d_authorized_function(cli_module, monkeypatch, capsys):
    """An unknown run_id resolves to block=None, so dispatch always falls
    to run_canary_cell() (which itself rejects it as UnknownRunIdError) --
    the D-authorized function must never be reached for it."""
    d_calls = []
    monkeypatch.setattr(
        cli_module,
        "run_block_d_train_validation_cell",
        lambda rid, matrix_path: d_calls.append((rid, matrix_path)),
    )
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", "not-a-real-id"])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert d_calls == []


def test_omitted_decision_blocks_d_cli_command_before_any_factory(cli_module, monkeypatch, capsys, tmp_path):
    loader_calls, device_calls = [], []
    decision_path = _write_decision(tmp_path, final_decision="OMITTED")
    bound = _bound_real_block_d(
        tmp_path, decision_path=decision_path, loader_calls=loader_calls, device_calls=device_calls
    )
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert loader_calls == []
    assert device_calls == []
    assert not (tmp_path / "confirmatory").exists()
    assert not (tmp_path / "ledger_confirmatory.csv").exists()


@pytest.mark.parametrize(
    "make_decision_path",
    [
        lambda tmp_path: tmp_path / "does_not_exist.json",  # missing
        lambda tmp_path: _write_decision(tmp_path, matrix_hash="wrong_hash"),  # conflicting hash
    ],
)
def test_broken_decision_variants_block_d_cli_command(
    cli_module, monkeypatch, capsys, tmp_path, make_decision_path
):
    loader_calls, device_calls = [], []
    decision_path = make_decision_path(tmp_path)
    bound = _bound_real_block_d(
        tmp_path, decision_path=decision_path, loader_calls=loader_calls, device_calls=device_calls
    )
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert loader_calls == []
    assert device_calls == []


def test_untracked_decision_blocks_d_cli_command(cli_module, monkeypatch, capsys, tmp_path):
    from when_tta_hurts.orchestrator import run_block_d_train_validation_cell

    loader_calls, device_calls = [], []
    decision_path = _write_decision(tmp_path)
    bound = functools.partial(
        run_block_d_train_validation_cell,
        decision_path=decision_path,
        loader_factory=_fake_loader_factory(loader_calls),
        dataset_verifier=_fake_dataset_verifier(),
        device_resolver=_fake_device_resolver(device_calls),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        data_root=tmp_path / "data",
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        git_tracked_and_clean=lambda p: False,  # untracked/dirty
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert loader_calls == []


def test_malformed_decision_blocks_d_cli_command(cli_module, monkeypatch, capsys, tmp_path):
    loader_calls, device_calls = [], []
    decision = _valid_decision()
    del decision["raw_output_sha256"]
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision))
    bound = _bound_real_block_d(
        tmp_path, decision_path=decision_path, loader_calls=loader_calls, device_calls=device_calls
    )
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert loader_calls == []
    assert device_calls == []


def test_successful_synthetic_d_completion_exit_code_and_output(cli_module, monkeypatch, capsys, tmp_path):
    bound = _bound_real_block_d(tmp_path)
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    rc = cli_module.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "completed" in out
    assert D_S0 in out
    assert "attempt=1" in out


def test_synthetic_completed_skip_handled_correctly(cli_module, monkeypatch, capsys, tmp_path):
    decision_path = _write_decision(tmp_path)

    def _bound():
        return _bound_real_block_d(tmp_path, decision_path=decision_path)

    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", _bound())
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    first_rc = cli_module.main()
    assert first_rc == 0
    capsys.readouterr()

    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", _bound())
    second_rc = cli_module.main()
    out = capsys.readouterr().out
    assert second_rc == 0
    assert "skipped_completed" in out


def test_training_failure_nonzero_and_terminal_record(cli_module, monkeypatch, tmp_path):
    import when_tta_hurts.orchestrator as orch

    monkeypatch.setattr(orch, "_BLOCK_D_MAX_TRAINING_MINUTES_PER_RUN", 0.0)
    bound = _bound_real_block_d(tmp_path)
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    with pytest.raises(Exception):  # noqa: B017 -- TrainingTimeoutError, existing terminal-failure convention
        cli_module.main()

    from when_tta_hurts.matrix import parse_and_validate_matrix
    from when_tta_hurts.run_identity import run_directory

    expanded = parse_and_validate_matrix(MATRIX_PATH, block_d_gate_passed=True)
    cell = next(c for c in expanded.cells if c.run_id() == D_S0)
    attempt_dir = run_directory(cell, root=str(tmp_path / "confirmatory")) / "attempt_001"
    assert attempt_dir.exists()
    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_run_canary_cell_direct_invocation_still_rejects_block_d():
    from when_tta_hurts.orchestrator import BlockDRunRejectedError, run_canary_cell

    with pytest.raises(BlockDRunRejectedError):
        run_canary_cell(D_S0, matrix_path=MATRIX_PATH)


def test_block_d_flag_argument_rejected(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confirmatory.py",
            "train-validation",
            "--block",
            "D",
            "--expected-total",
            "6",
            "--expected-pending",
            "6",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_decision_path_or_authorization_bypass_flags_exist():
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        "--decision-path",
        "--skip-authorization",
        "--force",
        "--skip-gate",
        "os.environ",
        "getenv",
    ):
        assert forbidden not in source


def test_cli_refusal_creates_zero_files_and_ledger_rows(cli_module, monkeypatch, capsys, tmp_path):
    loader_calls, device_calls = [], []
    decision_path = _write_decision(tmp_path, final_decision="OMITTED")
    bound = _bound_real_block_d(
        tmp_path, decision_path=decision_path, loader_calls=loader_calls, device_calls=device_calls
    )
    monkeypatch.setattr(cli_module, "run_block_d_train_validation_cell", bound)
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation", "--run-id", D_S0])
    before = set(tmp_path.rglob("*"))
    cli_module.main()
    after = set(tmp_path.rglob("*"))
    # only the decision.json we wrote ourselves should exist -- nothing new
    assert after == before


def test_no_test_split_or_tta_reference_in_cli_source():
    source = SCRIPT_PATH.read_text()
    assert 'split="test"' not in source
    assert "split='test'" not in source
    assert "evaluation." not in source
