#!/usr/bin/env python3
"""Phase 2B.7C-Engineering: sealed final-test statistical-analysis CLI.

Per docs/phase2b_final_test_analysis_cli_freeze.md. Distinct from, and
never routing to, scripts/run_statistical_analysis.py (the VALIDATION-
stage CLI, unmodified). This script imports only from
when_tta_hurts.final_test_statistical_analysis -- never
when_tta_hurts.statistical_analysis's or
when_tta_hurts.cross_condition_addendum's own analysis-dispatch
functions directly.

Modes:
    plan                    -- side-effect-free, zero prediction loads.
    analyze-preregistered   -- real analysis: every preregistered family
                                (H1, H2, H3, BLOCK_C). Never touches the
                                cross-condition addendum.
    analyze-cross-condition -- real analysis: every cross-condition
                                hypothesis (H1, H2, H3). Never touches the
                                preregistered families.

No scientific configuration flag exists on this parser (no run-id,
family/hypothesis selector, N/aggregator/endpoint override, bootstrap
override, seed override, alternate authorization/root/ledger path, split
selector, force/retry/bypass/unseal/print-results/debug-results flag).
No environment variable is read anywhere in this script.

Output is sealed: only lifecycle status, opaque IDs, hashes, paths, and
verification status are ever printed. The full internal result
dictionary returned by a real-analysis call is NEVER serialized to
stdout/stderr.

Usage:
    uv run python scripts/run_final_test_statistical_analysis.py plan
    uv run python scripts/run_final_test_statistical_analysis.py analyze-preregistered
    uv run python scripts/run_final_test_statistical_analysis.py analyze-cross-condition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.final_test_analysis_ledger import FinalTestAnalysisLedgerConflictError
from when_tta_hurts.final_test_authorization import FinalTestAuthorizationError
from when_tta_hurts.final_test_statistical_analysis import (
    KNOWN_FAMILIES,
    KNOWN_HYPOTHESES,
    FinalTestAnalysisAuthorizationError,
    FinalTestAnalysisInputError,
    FinalTestAnalysisSemanticVerificationError,
    compute_final_test_family_analysis,
    compute_final_test_hypothesis_did,
    plan_final_test_cross_condition_addendum,
    plan_final_test_statistical_analysis,
    verify_final_test_analysis_authorization,
)

# Every exception type this CLI is prepared to seal. Any other exception
# is re-raised after printing only its class name (never its message).
_SEALED_EXCEPTION_TYPES = (
    FinalTestAnalysisAuthorizationError,
    FinalTestAuthorizationError,
    FinalTestAnalysisInputError,
    FinalTestAnalysisSemanticVerificationError,
    FinalTestAnalysisLedgerConflictError,
)

_ALLOWED_RECEIPT_KEYS = frozenset(
    {
        "command",
        "mode",
        "status",
        "analysis_id",
        "analysis_ids",
        "attempt",
        "runtime_seconds",
        "n_inputs_required",
        "n_inputs_resolved",
        "manifest_verification",
        "semantic_verification",
    }
)


def _seal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Fails closed if a caller accidentally includes a non-allowlisted
    key -- this is the last line of defense against ever serializing a
    scientific value to stdout."""
    unexpected = set(receipt) - _ALLOWED_RECEIPT_KEYS
    if unexpected:
        raise RuntimeError(f"Internal error: receipt contains non-allowlisted key(s): {sorted(unexpected)}")
    return receipt


def _run_preregistered() -> dict[str, Any]:
    analysis_ids = []
    for family in KNOWN_FAMILIES:
        result = compute_final_test_family_analysis(family)
        if result["status"] != "completed":
            raise RuntimeError(f"Unexpected non-completed status for family {family!r}.")
        analysis_ids.append(result["analysis_id"])
    return _seal_receipt(
        {
            "command": "analyze-preregistered",
            "mode": "analyze-preregistered",
            "status": "completed",
            "analysis_ids": analysis_ids,
            "n_inputs_required": len(KNOWN_FAMILIES),
            "n_inputs_resolved": len(analysis_ids),
        }
    )


def _run_cross_condition() -> dict[str, Any]:
    analysis_ids = []
    for hypothesis in KNOWN_HYPOTHESES:
        result = compute_final_test_hypothesis_did(hypothesis)
        if result["status"] != "completed":
            raise RuntimeError(f"Unexpected non-completed status for hypothesis {hypothesis!r}.")
        analysis_ids.append(result["analysis_id"])
    return _seal_receipt(
        {
            "command": "analyze-cross-condition",
            "mode": "analyze-cross-condition",
            "status": "completed",
            "analysis_ids": analysis_ids,
            "n_inputs_required": len(KNOWN_HYPOTHESES),
            "n_inputs_resolved": len(analysis_ids),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "analyze-preregistered", "analyze-cross-condition"])
    args = parser.parse_args()

    if args.mode == "plan":
        report = {
            "command": "plan",
            "preregistered": plan_final_test_statistical_analysis(),
            "cross_condition": plan_final_test_cross_condition_addendum(),
        }
        print(json.dumps(report, indent=2, default=str))
        return 0

    try:
        verify_final_test_analysis_authorization()
    except FinalTestAnalysisAuthorizationError as e:
        print(json.dumps({"command": args.mode, "error_class": type(e).__name__}), file=sys.stderr)
        return 1

    try:
        if args.mode == "analyze-preregistered":
            receipt = _run_preregistered()
        else:
            receipt = _run_cross_condition()
    except _SEALED_EXCEPTION_TYPES as e:
        print(json.dumps({"command": args.mode, "error_class": type(e).__name__}), file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
