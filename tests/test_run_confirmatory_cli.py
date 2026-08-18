"""Phase 2B.3A Part A tests: scripts/run_confirmatory.py CLI argument
handling. Never actually executes train-validation mode against real data
-- those code paths are exercised via orchestrator.run_canary_cell directly
in tests/test_canary_resolution.py. This file only checks argparse-level
behavior and static source guarantees."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_confirmatory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_confirmatory", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


def test_omitted_run_id_fails(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_confirmatory.py", "train-validation"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert "--run-id is required" in capsys.readouterr().err


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


def test_block_d_run_id_fails(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_confirmatory.py", "train-validation", "--run-id", "D-pathmnist-128px-batchnorm-policy-none-s0"],
    )
    rc = cli_module.main()
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


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
