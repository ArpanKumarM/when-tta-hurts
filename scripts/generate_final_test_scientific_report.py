#!/usr/bin/env python3
"""Phase 2B.8A: sealed final-test scientific-report generator CLI.

Per docs/phase2b_final_test_unsealing_freeze.md. Imports exclusively
from when_tta_hurts.final_test_scientific_reporting -- the only module
in the repository permitted to parse a final-test analysis result's
scientific contents.

Modes:
    plan   -- metadata-only: resolves the seven sealed inputs via ledger
              rows and manifest-file byte hashes, NEVER json.loads of a
              result file. Reports readiness, opaque identities, expected
              counts, the reporting fingerprint, and output-path status.
              Zero writes.
    unseal -- real generation: requires the hardcoded
              artifacts/final_test_unsealing_authorization.json to be
              approved and current, verified BEFORE any result JSON is
              parsed. Invokes the deterministic generator exactly once
              and writes only the three frozen output paths. Prints only
              output paths, hashes, sizes, counts, lifecycle status, and
              the reporting fingerprint -- never a scientific value.

No scientific configuration flag exists on this parser (no alternate
input/output/authorization path, no hypothesis/endpoint/threshold
selector, no formatting-policy flag, no force/bypass/partial-generation
flag). No environment variable is read anywhere in this script.

Usage:
    uv run python scripts/generate_final_test_scientific_report.py plan
    uv run python scripts/generate_final_test_scientific_report.py unseal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.final_test_scientific_reporting import (
    EXPECTED_UNITS,
    INTERPRETATION_MARKDOWN_PATH,
    RESULTS_MARKDOWN_PATH,
    SCIENTIFIC_SUMMARY_PATH,
    ReportGenerationError,
    SealedInputResolutionError,
    SealedInputTamperError,
    SealedResultSchemaError,
    UnsealingAuthorizationError,
    compute_final_test_reporting_fingerprint,
    generate_and_persist_report,
    resolve_seven_sealed_inputs,
    verify_unsealing_authorization,
)

_SEALED_EXCEPTION_TYPES = (
    UnsealingAuthorizationError,
    SealedInputResolutionError,
    SealedInputTamperError,
    SealedResultSchemaError,
    ReportGenerationError,
)


def _hash_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_plan() -> dict[str, Any]:
    report: dict[str, Any] = {"command": "plan"}
    try:
        reporting_fp, _ = compute_final_test_reporting_fingerprint()
        report["reporting_fingerprint"] = reporting_fp
    except Exception as e:
        report["reporting_fingerprint_status"] = f"error:{type(e).__name__}"

    try:
        resolved = resolve_seven_sealed_inputs()
        report["inputs_ready"] = True
        report["n_inputs"] = len(resolved)
        report["inputs"] = [
            {
                "kind": kind,
                "identifier": identifier,
                "expected_count": expected_count,
                "analysis_id": resolved[f"{kind}:{identifier}"]["analysis_id"],
                "attempt": resolved[f"{kind}:{identifier}"]["attempt"],
            }
            for kind, identifier, expected_count, _ in EXPECTED_UNITS
        ]
    except (SealedInputResolutionError, SealedInputTamperError) as e:
        report["inputs_ready"] = False
        report["inputs_error_class"] = type(e).__name__

    try:
        verify_unsealing_authorization()
        report["unsealing_authorization_status"] = "approved"
    except UnsealingAuthorizationError as e:
        report["unsealing_authorization_status"] = "not_approved"
        report["unsealing_authorization_error_class"] = type(e).__name__

    report["output_paths"] = {
        "summary": str(SCIENTIFIC_SUMMARY_PATH),
        "results_markdown": str(RESULTS_MARKDOWN_PATH),
        "interpretation_markdown": str(INTERPRETATION_MARKDOWN_PATH),
    }
    report["outputs_exist"] = {
        "summary": SCIENTIFIC_SUMMARY_PATH.exists(),
        "results_markdown": RESULTS_MARKDOWN_PATH.exists(),
        "interpretation_markdown": INTERPRETATION_MARKDOWN_PATH.exists(),
    }
    return report


def _run_unseal() -> dict[str, Any]:
    result = generate_and_persist_report()
    receipt: dict[str, Any] = {"command": "unseal", "status": result["status"]}
    if result["status"] == "completed":
        outputs = []
        for p in result["outputs"]:
            path = Path(p)
            outputs.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _hash_file(path)})
        receipt["outputs"] = outputs
    receipt["summary_sha256"] = result["summary_sha256"]
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "unseal"])
    args = parser.parse_args()

    if args.mode == "plan":
        print(json.dumps(_run_plan(), indent=2, default=str))
        return 0

    try:
        receipt = _run_unseal()
    except _SEALED_EXCEPTION_TYPES as e:
        print(json.dumps({"command": "unseal", "error_class": type(e).__name__}), file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
