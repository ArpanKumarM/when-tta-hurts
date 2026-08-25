"""Phase 2B.7C-Engineering: CLI tests for
scripts/run_final_test_statistical_analysis.py.

SAFETY: every test that reaches an `analyze-*` subcommand ALWAYS patches
`cli_module.compute_final_test_family_analysis` and/or
`cli_module.compute_final_test_hypothesis_did` with a fake BEFORE calling
`cli_module.main()`. An autouse guard fixture additionally patches both
real dispatch targets and `numpy.load` to explode if reached unpatched,
so a test that forgets to patch fails loudly instead of silently running
a real, hours-long analysis or touching real repository artifacts.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_final_test_statistical_analysis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_final_test_statistical_analysis", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


class _ProductionPathReachedInTestError(AssertionError):
    """Raised by the autouse guard below -- a test reached a real
    final-test-analysis dispatch function or numpy.load without
    patching it first."""


def _guard_explode(*args, **kwargs):
    raise _ProductionPathReachedInTestError(
        "PRODUCTION PATH REACHED IN TEST: a CLI test attempted to touch a real final-test-analysis "
        "dispatch function or numpy.load. Patch cli_module.compute_final_test_family_analysis / "
        "cli_module.compute_final_test_hypothesis_did before calling cli_module.main()."
    )


@pytest.fixture(autouse=True)
def _guard_against_real_analysis(cli_module, monkeypatch):
    monkeypatch.setattr(cli_module, "compute_final_test_family_analysis", _guard_explode)
    monkeypatch.setattr(cli_module, "compute_final_test_hypothesis_did", _guard_explode)
    import numpy as np

    monkeypatch.setattr(np, "load", _guard_explode)
    yield


# ---------------------------------------------------------------------------
# plan: side-effect-free, zero np.load calls, against the real repo.
# ---------------------------------------------------------------------------


def test_plan_is_side_effect_free_and_makes_zero_prediction_loads(cli_module, monkeypatch, capsys):
    # plan does not call numpy.load at all (verified at the library level
    # already) -- the autouse guard above would explode if it tried.
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "plan"])
    rc = cli_module.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["preregistered"]["authorization_status"] == "approved"
    assert out["cross_condition"]["authorization_status"] == "approved"


# ---------------------------------------------------------------------------
# Atomic dispatch: each analyze subcommand calls exactly its own target,
# never the other.
# ---------------------------------------------------------------------------


def _fake_family_result(family, **kwargs):
    return {"status": "completed", "analysis_id": f"fake-analysis-id-{family}"}


def _fake_hypothesis_result(hypothesis, **kwargs):
    return {"status": "completed", "analysis_id": f"fake-cross-id-{hypothesis}"}


def _patch_authorization_ok(cli_module, monkeypatch):
    monkeypatch.setattr(
        cli_module, "verify_final_test_analysis_authorization", lambda: {"status": "approved"}
    )


def test_analyze_preregistered_calls_only_the_preregistered_target(cli_module, monkeypatch, capsys):
    _patch_authorization_ok(cli_module, monkeypatch)
    family_calls = []
    hypothesis_calls = []
    monkeypatch.setattr(
        cli_module,
        "compute_final_test_family_analysis",
        lambda family, **kw: (family_calls.append(family), _fake_family_result(family))[1],
    )
    monkeypatch.setattr(
        cli_module,
        "compute_final_test_hypothesis_did",
        lambda hypothesis, **kw: (hypothesis_calls.append(hypothesis), _fake_hypothesis_result(hypothesis))[
            1
        ],
    )
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])
    rc = cli_module.main()
    assert rc == 0
    assert set(family_calls) == set(cli_module.KNOWN_FAMILIES)
    assert hypothesis_calls == [], "analyze-preregistered must never call the cross-condition target"


def test_analyze_cross_condition_calls_only_the_cross_condition_target(cli_module, monkeypatch, capsys):
    _patch_authorization_ok(cli_module, monkeypatch)
    family_calls = []
    hypothesis_calls = []
    monkeypatch.setattr(
        cli_module,
        "compute_final_test_family_analysis",
        lambda family, **kw: (family_calls.append(family), _fake_family_result(family))[1],
    )
    monkeypatch.setattr(
        cli_module,
        "compute_final_test_hypothesis_did",
        lambda hypothesis, **kw: (hypothesis_calls.append(hypothesis), _fake_hypothesis_result(hypothesis))[
            1
        ],
    )
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-cross-condition"])
    rc = cli_module.main()
    assert rc == 0
    assert set(hypothesis_calls) == set(cli_module.KNOWN_HYPOTHESES)
    assert family_calls == [], "analyze-cross-condition must never call the preregistered target"


# ---------------------------------------------------------------------------
# Validation-stage functions are never imported or invoked.
# ---------------------------------------------------------------------------


def test_script_never_imports_validation_stage_analysis_functions():
    source = SCRIPT_PATH.read_text()
    forbidden = (
        "from when_tta_hurts.statistical_analysis import",
        "from when_tta_hurts.cross_condition_addendum import",
        "compute_family_analysis(",
        "plan_statistical_analysis(",
        "compute_hypothesis_did(",
        "plan_cross_condition_addendum(",
    )
    for term in forbidden:
        assert term not in source, f"CLI script must never reference validation-stage symbol: {term!r}"


def test_script_imports_only_final_test_analysis_api():
    tree = ast.parse(SCRIPT_PATH.read_text())
    when_tta_hurts_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("when_tta_hurts")
    ]
    allowed_modules = {
        "when_tta_hurts.final_test_analysis_ledger",
        "when_tta_hurts.final_test_authorization",
        "when_tta_hurts.final_test_statistical_analysis",
    }
    assert set(when_tta_hurts_imports) <= allowed_modules, when_tta_hurts_imports


# ---------------------------------------------------------------------------
# Authorization-first ordering: invalid authorization refuses before any
# attempt allocation or prediction load.
# ---------------------------------------------------------------------------


def test_invalid_authorization_refuses_before_dispatch(cli_module, monkeypatch, capsys):
    from when_tta_hurts.final_test_statistical_analysis import FinalTestAnalysisAuthorizationError

    def _raise():
        raise FinalTestAnalysisAuthorizationError("synthetic: stale fingerprint")

    monkeypatch.setattr(cli_module, "verify_final_test_analysis_authorization", _raise)
    # compute_final_test_family_analysis/hypothesis_did remain the exploding
    # guard fakes -- if the CLI reached them, the test would fail with
    # _ProductionPathReachedInTestError instead of the expected sealed error.
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])
    rc = cli_module.main()
    assert rc == 1
    captured = capsys.readouterr()
    err = json.loads(captured.err)
    assert err["error_class"] == "FinalTestAnalysisAuthorizationError"
    assert "synthetic" not in captured.err
    assert "stale fingerprint" not in captured.err


# ---------------------------------------------------------------------------
# Idempotent completed-skip and ambiguous/conflicting hard-fail.
# ---------------------------------------------------------------------------


def test_idempotent_completed_skip_returns_without_error(cli_module, monkeypatch, capsys):
    _patch_authorization_ok(cli_module, monkeypatch)
    monkeypatch.setattr(
        cli_module, "compute_final_test_family_analysis", lambda family, **kw: _fake_family_result(family)
    )
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])
    rc = cli_module.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"
    assert len(out["analysis_ids"]) == len(cli_module.KNOWN_FAMILIES)


def test_conflicting_completion_hard_fails_and_is_sealed(cli_module, monkeypatch, capsys):
    from when_tta_hurts.final_test_analysis_ledger import FinalTestAnalysisLedgerConflictError

    _patch_authorization_ok(cli_module, monkeypatch)

    def _raise(family, **kw):
        raise FinalTestAnalysisLedgerConflictError(
            "conflicting completed rows: delta_accuracy=0.873, ci_low=0.81"
        )

    monkeypatch.setattr(cli_module, "compute_final_test_family_analysis", _raise)
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])
    rc = cli_module.main()
    assert rc == 1
    captured = capsys.readouterr()
    err = json.loads(captured.err)
    assert err["error_class"] == "FinalTestAnalysisLedgerConflictError"
    assert "0.873" not in captured.err
    assert "delta_accuracy" not in captured.err
    assert "ci_low" not in captured.err


# ---------------------------------------------------------------------------
# Sealed output: no forbidden scientific field/value in stdout or stderr,
# across success and failure paths.
# ---------------------------------------------------------------------------

_FORBIDDEN_TERMS = (
    "delta_accuracy",
    "accuracy",
    "harm_rate",
    "rescue_rate",
    "p_value",
    "ci_low",
    "ci_high",
    "bootstrap_seed",
    "per_cell_statistics",
    "per_pair_results",
    "multiplicity",
    "effect_sizes",
    "mcnemar",
)


def test_successful_run_never_prints_forbidden_scientific_fields(cli_module, monkeypatch, capsys):
    _patch_authorization_ok(cli_module, monkeypatch)

    def _rich_fake_result(family, **kw):
        # A deliberately "rich" fake result, as if the real function
        # returned its full internal dictionary -- proves the CLI's
        # allowlist strips it rather than trusting the callee.
        return {
            "status": "completed",
            "analysis_id": f"fake-{family}",
            "per_cell_statistics": {"cell-x": {"bootstrap": {"delta_accuracy": 0.5, "ci_low": 0.1}}},
            "multiplicity": {"raw_p_values": [0.01]},
        }

    monkeypatch.setattr(cli_module, "compute_final_test_family_analysis", _rich_fake_result)
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])
    rc = cli_module.main()
    assert rc == 0
    captured = capsys.readouterr()
    for term in _FORBIDDEN_TERMS:
        assert term not in captured.out
        assert term not in captured.err


def test_scientific_receipt_leak_is_caught_by_seal_guard(cli_module):
    with pytest.raises(RuntimeError, match="non-allowlisted"):
        cli_module._seal_receipt({"command": "x", "delta_accuracy": 0.5})


# ---------------------------------------------------------------------------
# Argparse-level rejection: no scientific configuration surface, no
# environment-variable bypass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--family", "H1"],
        ["--hypothesis", "H1"],
        ["--run-id", "some-run"],
        ["-n", "50"],
        ["--aggregator", "mean_probability"],
        ["--bootstrap-resamples", "10000"],
        ["--seed", "1"],
        ["--authorization-path", "/tmp/x.json"],
        ["--final-test-root", "/tmp"],
        ["--ledger-path", "/tmp/l.csv"],
        ["--split", "test"],
        ["--force"],
        ["--retry"],
        ["--bypass"],
        ["--unseal"],
        ["--print-results"],
        ["--debug-results"],
    ],
)
def test_unsupported_scientific_flags_rejected_by_argparse(cli_module, monkeypatch, extra_args):
    monkeypatch.setattr(
        sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered", *extra_args]
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_environment_variable_reads_anywhere_in_script():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "import os" not in source


def test_unsupported_subcommand_rejected(cli_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze"])
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Static proof: every test in THIS file that reaches an analyze
# subcommand patches the corresponding real runner first.
# ---------------------------------------------------------------------------


def test_every_analyze_reaching_test_patches_the_real_runner():
    tree = ast.parse(Path(__file__).read_text())
    target_by_mode = {
        "analyze-preregistered": "compute_final_test_family_analysis",
        "analyze-cross-condition": "compute_final_test_hypothesis_did",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        fn_source = ast.get_source_segment(Path(__file__).read_text(), node) or ""
        if "SystemExit" in fn_source:
            # argparse-only tests raise SystemExit before dispatch logic
            # is ever reached -- never call the real runner, so no patch
            # is required.
            continue
        for mode, target in target_by_mode.items():
            if f'"{mode}"' not in fn_source and f"'{mode}'" not in fn_source:
                continue
            if "cli_module.main()" not in fn_source:
                continue
            # A test that reaches this mode's main() must patch its
            # target explicitly, OR rely on the module-level fake
            # dispatch defined once and referenced via monkeypatch.setattr
            # with the target's name appearing in the same test body.
            assert f'"{target}"' in fn_source or target in fn_source, (
                f"Test {node.name!r} reaches {mode!r} without patching {target!r}."
            )


# ---------------------------------------------------------------------------
# Expected call trace, loading no real data, creating no production side
# effect.
# ---------------------------------------------------------------------------


def test_monkeypatched_proof_run_expected_call_trace_no_real_side_effects(cli_module, monkeypatch, capsys):
    _patch_authorization_ok(cli_module, monkeypatch)
    trace = []
    monkeypatch.setattr(
        cli_module,
        "compute_final_test_family_analysis",
        lambda family, **kw: (trace.append(("family", family)), _fake_family_result(family))[1],
    )
    monkeypatch.setattr(sys, "argv", ["run_final_test_statistical_analysis.py", "analyze-preregistered"])

    from pathlib import Path as _Path

    before = (
        _Path("artifacts/final_test_analysis").exists(),
        _Path("artifacts/ledger_final_test_analysis.csv").exists(),
    )
    rc = cli_module.main()
    after = (
        _Path("artifacts/final_test_analysis").exists(),
        _Path("artifacts/ledger_final_test_analysis.csv").exists(),
    )

    assert rc == 0
    assert sorted(trace) == sorted(("family", f) for f in cli_module.KNOWN_FAMILIES)
    assert before == after


# ---------------------------------------------------------------------------
# Fingerprint manifest placement.
# ---------------------------------------------------------------------------


def test_cli_script_is_in_final_test_analysis_manifest_only():
    from when_tta_hurts.cross_condition_addendum import CROSS_CONDITION_ADDENDUM_MANIFEST
    from when_tta_hurts.final_test_identity import FINAL_TEST_RUNNER_MANIFEST
    from when_tta_hurts.final_test_statistical_analysis import FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST
    from when_tta_hurts.statistical_analysis import ANALYSIS_FINGERPRINT_MANIFEST

    rel = "scripts/run_final_test_statistical_analysis.py"
    assert rel in FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST
    assert rel not in ANALYSIS_FINGERPRINT_MANIFEST
    assert rel not in CROSS_CONDITION_ADDENDUM_MANIFEST
    assert rel not in FINAL_TEST_RUNNER_MANIFEST


def test_output_receipt_keys_are_fully_allowlisted(cli_module):
    sig_source = inspect.getsource(cli_module._seal_receipt)
    assert "_ALLOWED_RECEIPT_KEYS" in sig_source
