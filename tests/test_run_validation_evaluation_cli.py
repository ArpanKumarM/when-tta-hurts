"""CLI tests for scripts/run_validation_evaluation.py.

SAFETY (read before editing this file): the real repo has genuinely
completed confirmatory checkpoints, a valid committed frozen TTA-seed
config, and a working MPS device. If `cli_module.main()` is ever invoked
with a VALID run_id and the real `run_validation_evaluation` function
left unpatched, it WILL start a real, hours-long MPS evaluation against
real data (this happened once during development of this file -- a
stale test omitted the required patch after --tta-seed was removed from
the CLI, and `evaluate-validation --run-id <real cell>` ran to
completion of argument parsing and proceeded into real execution before
being caught and killed; the resulting incomplete, gitignored
artifacts/validation_evaluation/ attempt directory -- status.json only,
no predictions/metrics ever computed -- was removed).

To prevent a repeat: every test in this file that reaches
`evaluate-validation` with ANY run_id that could resolve to a real
canonical training completion ALWAYS patches `cli_module.run_validation_evaluation`
with either a pure recording fake, or a `functools.partial` binding of
the REAL function to entirely fake DI parameters (device_resolver,
tta_seed git-tracked/commit-ancestor hooks, ledger/root paths) BEFORE
calling `cli_module.main()`. Tests that only need argparse-level
behavior (missing/duplicate flags) never reach `main()`'s dispatch logic
at all, so they are safe without patching -- but this is verified by
inspection here, not assumed."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_validation_evaluation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_validation_evaluation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


def _always_tracked_clean(path):
    return True


def _commit_for(commit):
    return lambda path: commit


def _all_ancestors(commit, head):
    return True


def test_plan_mode_runs_and_is_side_effect_free(cli_module, monkeypatch, capsys, tmp_path):
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        matrix_path = os.path.join(cwd, "configs/experiment_matrix.yaml")
        monkeypatch.setattr(sys, "argv", ["run_validation_evaluation.py", "plan", "--matrix", matrix_path])
        rc = cli_module.main()
        assert rc == 0
        assert list(tmp_path.iterdir()) == []
        out = capsys.readouterr().out
        assert "run_id" in out
        # from tmp_path's cwd, configs/validation_evaluation.yaml (a relative
        # path) does not exist -- plan mode reports that as an error field
        # rather than raising, which is itself the side-effect-free contract
        # under test here (see test_plan_mode_reports_frozen_seed_config_hash_
        # and_freeze_commit in test_validation_evaluation.py for the
        # real-cwd, seed-config-present case).
        assert "tta_seed_config" in out
    finally:
        os.chdir(cwd)


def test_evaluate_validation_requires_run_id(cli_module, monkeypatch, capsys):
    """argparse-only: --run-id is missing, so main() never reaches
    run_validation_evaluation() at all -- safe without patching."""
    monkeypatch.setattr(sys, "argv", ["run_validation_evaluation.py", "evaluate-validation"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "requires --run-id" in capsys.readouterr().err


def test_evaluate_validation_has_no_tta_seed_flag(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s0",
            "--tta-seed",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "unrecognized" in capsys.readouterr().err.lower()


def test_pilot_seed_run_id_refused_before_dispatch(cli_module, monkeypatch, capsys):
    """Rejected inside resolve_canonical_training_completion() via an
    explicit string check, which fires before MPS/dataset/checkpoint
    access regardless of run_validation_evaluation()'s internal ordering
    -- safe to exercise with the REAL function (the frozen TTA-seed
    config load that precedes it is a harmless local file+git read)."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s314159",
        ],
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_unknown_run_id_refused_with_faked_dependencies(cli_module, monkeypatch, capsys, tmp_path):
    """Uses the REAL run_validation_evaluation (to prove the CLI's except
    clause maps its real exceptions to REFUSED/rc=1), but with EVERY
    side-effecting dependency faked -- device_resolver, seed-config git
    hooks, and disposable root/ledger paths -- so even if resolution
    somehow proceeded further than expected, nothing real would ever be
    touched. run_id is unknown, so resolve_canonical_training_completion()
    raises before any of those faked dependencies are even consulted."""
    from when_tta_hurts.devices import DeviceUnavailableError

    def fail_device_resolver():
        raise DeviceUnavailableError("must never be called for an unknown run_id")

    bound = functools.partial(
        cli_module.run_validation_evaluation,
        device_resolver=fail_device_resolver,
        root=tmp_path / "eval",
        training_root=tmp_path / "training",
        data_root=tmp_path / "data",
        evaluation_ledger_path=tmp_path / "ledger.csv",
        tta_seed_git_tracked_and_clean=_always_tracked_clean,
        tta_seed_last_commit_for_path=_commit_for("c" * 40),
        tta_seed_commit_is_ancestor=_all_ancestors,
    )
    monkeypatch.setattr(cli_module, "run_validation_evaluation", bound)
    monkeypatch.setattr(
        sys, "argv", ["run_validation_evaluation.py", "evaluate-validation", "--run-id", "not-a-real-run-id"]
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "eval").exists()
    assert not (tmp_path / "ledger.csv").exists()


def test_valid_run_id_dispatches_to_run_validation_evaluation_recording_fake(cli_module, monkeypatch, capsys):
    """Proves the CLI dispatches a valid --run-id through to
    run_validation_evaluation() with the expected arguments -- via a pure
    recording fake, never the real function."""
    calls = []

    def fake(run_id, matrix_path):
        calls.append((run_id, matrix_path))
        return {"status": "completed", "training_run_id": run_id, "evaluation_id": "eid"}

    monkeypatch.setattr(cli_module, "run_validation_evaluation", fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s0",
        ],
    )
    rc = cli_module.main()
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == "A-pathmnist-28px-batchnorm-policy-none-s0"


def test_no_all_cells_or_block_flag_exists():
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        'add_argument("--block"',
        'add_argument("--all"',
        'add_argument("--split"',
        "argparse.REMAINDER",
    ):
        assert forbidden not in source


def test_no_tta_seed_flag_registered():
    source = SCRIPT_PATH.read_text()
    assert 'add_argument("--tta-seed"' not in source


def test_no_bypass_flags_or_env_vars():
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        "--force",
        "--skip-authorization",
        "--allow-test",
        "--synthetic",
        "--policy-override",
        "--prefix-override",
        "--test-unlock",
        "--config-path",
        "--seed-config",
        "os.environ",
        "getenv",
    ):
        assert forbidden not in source


def test_no_split_argument_of_any_kind():
    source = SCRIPT_PATH.read_text()
    assert '"--split"' not in source
    assert "'--split'" not in source


def test_multiple_run_id_flags_rejected(cli_module, monkeypatch, capsys):
    """argparse-only: rejected before dispatch, safe without patching."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
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
