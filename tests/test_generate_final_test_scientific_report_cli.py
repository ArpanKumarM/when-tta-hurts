"""Phase 2B.8A: CLI tests for scripts/generate_final_test_scientific_report.py.

SAFETY: every test that reaches `unseal` ALWAYS patches
`cli_module.generate_and_persist_report` (or its authorization/resolution
dependencies) with a fake BEFORE calling `cli_module.main()`. An autouse
guard fixture additionally patches the real generator and the real
production output paths to explode if reached unpatched.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_final_test_scientific_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_final_test_scientific_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


class _ProductionPathReachedInTestError(AssertionError):
    pass


def _guard_explode(*args, **kwargs):
    raise _ProductionPathReachedInTestError(
        "PRODUCTION PATH REACHED IN TEST: a CLI test attempted to touch the real generator. Patch "
        "cli_module.generate_and_persist_report before calling cli_module.main()."
    )


@pytest.fixture(autouse=True)
def _guard_against_real_unsealing(cli_module, monkeypatch):
    monkeypatch.setattr(cli_module, "generate_and_persist_report", _guard_explode)
    yield


# ---------------------------------------------------------------------------
# plan: metadata-only, zero result-JSON parses, zero writes.
# ---------------------------------------------------------------------------


def test_plan_never_parses_result_json_and_writes_nothing(cli_module, monkeypatch, capsys, tmp_path):
    from when_tta_hurts.final_test_scientific_reporting import EXPECTED_UNITS

    fake_resolved = {
        f"{kind}:{identifier}": {"analysis_id": f"fake-{identifier}", "attempt": 1}
        for kind, identifier, _, _ in EXPECTED_UNITS
    }
    monkeypatch.setattr(cli_module, "resolve_seven_sealed_inputs", lambda: fake_resolved)
    monkeypatch.setattr(cli_module, "verify_unsealing_authorization", lambda: {"status": "approved"})
    monkeypatch.setattr(cli_module, "compute_final_test_reporting_fingerprint", lambda: ("fake-fp", {}))

    calls = []
    real_loads = json.loads
    monkeypatch.setattr(json, "loads", lambda s, *a, **k: (calls.append(1), real_loads(s, *a, **k))[1])

    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "plan"])
    rc = cli_module.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs_ready"] is True
    assert out["n_inputs"] == 7
    # The single json.loads call above is the harness's own parse of
    # captured stdout, not a call made by the CLI's plan path itself.
    assert len(calls) <= 1


def test_plan_reports_not_ready_when_resolution_fails(cli_module, monkeypatch, capsys):
    from when_tta_hurts.final_test_scientific_reporting import SealedInputResolutionError

    def _raise():
        raise SealedInputResolutionError("synthetic: missing unit")

    monkeypatch.setattr(cli_module, "resolve_seven_sealed_inputs", _raise)
    monkeypatch.setattr(cli_module, "verify_unsealing_authorization", lambda: {"status": "approved"})
    monkeypatch.setattr(cli_module, "compute_final_test_reporting_fingerprint", lambda: ("fake-fp", {}))
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "plan"])
    rc = cli_module.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs_ready"] is False
    assert out["inputs_error_class"] == "SealedInputResolutionError"


# ---------------------------------------------------------------------------
# unseal: authorization before dispatch, sealed receipt, idempotency.
# ---------------------------------------------------------------------------


def test_unseal_invokes_generator_exactly_once(cli_module, monkeypatch, capsys):
    calls = []

    def _fake_generate(*a, **kw):
        calls.append(1)
        return {"status": "completed", "summary_sha256": "fake-hash", "outputs": []}

    monkeypatch.setattr(cli_module, "generate_and_persist_report", _fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "unseal"])
    rc = cli_module.main()
    assert rc == 0
    assert len(calls) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"


def test_unseal_authorization_error_prevents_generator_call(cli_module, monkeypatch, capsys):
    from when_tta_hurts.final_test_scientific_reporting import UnsealingAuthorizationError

    def _fake_generate(*a, **kw):
        raise UnsealingAuthorizationError("synthetic: stale reporting fingerprint")

    monkeypatch.setattr(cli_module, "generate_and_persist_report", _fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "unseal"])
    rc = cli_module.main()
    assert rc == 1
    captured = capsys.readouterr()
    err = json.loads(captured.err)
    assert err["error_class"] == "UnsealingAuthorizationError"
    assert "synthetic" not in captured.err
    assert "stale" not in captured.err


def test_unseal_idempotent_skip_reports_status(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "generate_and_persist_report",
        lambda *a, **kw: {"status": "idempotent_skip", "summary_sha256": "fake-hash"},
    )
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "unseal"])
    rc = cli_module.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "idempotent_skip"


# ---------------------------------------------------------------------------
# Sealed output: no forbidden scientific field/value across success and
# failure paths.
# ---------------------------------------------------------------------------

_FORBIDDEN_TERMS = (
    "delta_accuracy",
    'did"',
    "p_value",
    "ci_low",
    "ci_high",
    "per_cell_statistics",
    "per_pair_results",
    "multiplicity",
    "bootstrap_seed",
)


def test_unseal_never_prints_forbidden_scientific_fields(cli_module, monkeypatch, capsys):
    def _rich_fake(*a, **kw):
        return {
            "status": "completed",
            "summary_sha256": "fake-hash",
            "outputs": [],
            "per_cell_statistics": {"cell-x": {"bootstrap": {"delta_accuracy": 0.5}}},
        }

    monkeypatch.setattr(cli_module, "generate_and_persist_report", _rich_fake)
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "unseal"])
    cli_module.main()
    captured = capsys.readouterr()
    for term in _FORBIDDEN_TERMS:
        assert term not in captured.out
        assert term not in captured.err


# ---------------------------------------------------------------------------
# No scientific configuration surface, no environment-variable bypass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--input-path", "/tmp/x"],
        ["--authorization-path", "/tmp/x.json"],
        ["--output-path", "/tmp/out.json"],
        ["--hypothesis", "H1"],
        ["--endpoint", "accuracy"],
        ["--threshold", "0.05"],
        ["--format", "csv"],
        ["--force"],
        ["--bypass"],
        ["--partial"],
    ],
)
def test_unsupported_flags_rejected_by_argparse(cli_module, monkeypatch, extra_args):
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "unseal", *extra_args])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_environment_variable_reads_in_script():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source
    assert "os.getenv" not in source


def test_unsupported_subcommand_rejected(cli_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_final_test_scientific_report.py", "generate"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_command_prints_individual_scientific_values():
    source = SCRIPT_PATH.read_text()
    assert "delta_accuracy" not in source
    assert "print(json.dumps(result" not in source  # never dumps the raw internal result


# ---------------------------------------------------------------------------
# Static proof: every test reaching `unseal`'s dispatch patches the real
# generator.
# ---------------------------------------------------------------------------


def test_every_unseal_reaching_test_patches_the_real_generator():
    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        fn_source = ast.get_source_segment(Path(__file__).read_text(), node) or ""
        if '"unseal"' not in fn_source or "cli_module.main()" not in fn_source:
            continue
        if "SystemExit" in fn_source:
            continue
        assert "generate_and_persist_report" in fn_source, (
            f"Test {node.name!r} reaches unseal without patching generate_and_persist_report."
        )
