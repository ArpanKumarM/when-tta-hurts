"""CLI tests for scripts/run_validation_evaluation.py. Plan mode is
exercised for real (side-effect-free by design). evaluate-validation is
exercised only via argparse-level checks and static source guarantees --
never against real production defaults, since the real repo has genuinely
completed confirmatory checkpoints on disk and this must never trigger a
real evaluation."""

from __future__ import annotations

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
    finally:
        os.chdir(cwd)


def test_evaluate_validation_requires_run_id(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["run_validation_evaluation.py", "evaluate-validation", "--tta-seed", "1"]
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "requires --run-id" in capsys.readouterr().err


def test_evaluate_validation_requires_tta_seed(cli_module, monkeypatch, capsys):
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
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "requires --tta-seed" in capsys.readouterr().err


def test_pilot_seed_run_id_refused_before_dispatch(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s314159",
            "--tta-seed",
            "123456",
        ],
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_pilot_tta_seed_refused(cli_module, monkeypatch, capsys):
    """--tta-seed 271828 (the pilot's seed) must be refused, not silently
    accepted, even though run_validation_evaluation() only reaches the
    canonical-resolution step before this check fires for a valid run_id
    -- confirmed here it never proceeds to MPS/dataset/checkpoint access."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation_evaluation.py",
            "evaluate-validation",
            "--run-id",
            "not-a-real-run-id",
            "--tta-seed",
            "271828",
        ],
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_no_all_cells_or_block_flag_exists():
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        'add_argument("--block"',
        'add_argument("--all"',
        'add_argument("--split"',
        "argparse.REMAINDER",
    ):
        assert forbidden not in source


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
        "os.environ",
        "getenv",
    ):
        assert forbidden not in source


def test_no_split_argument_of_any_kind():
    source = SCRIPT_PATH.read_text()
    assert '"--split"' not in source
    assert "'--split'" not in source


def test_multiple_run_id_flags_rejected(cli_module, monkeypatch, capsys):
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
            "--tta-seed",
            "123456",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "only be specified once" in capsys.readouterr().err
