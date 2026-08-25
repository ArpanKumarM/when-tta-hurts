#!/usr/bin/env python3
"""Phase 2B.9A: sealed paper-evidence generator CLI.

Downstream presentation only -- renders figures/tables from the already-
sealed, already-committed canonical generation-2 final-test scientific
summary (`artifacts/final_test_scientific_summary.json`). Never reads
raw predictions, datasets, checkpoints, validation artifacts, or sealed
per-family analysis results. Never computes a new statistic.

This script is intended to be invoked ONLY via the isolated toolchain
per docs/phase2b_paper_evidence_toolchain_freeze.md:

    uv sync --project tools/paper_evidence --frozen
    uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py plan
    uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py generate

Modes:
    plan     -- metadata-only: verifies the canonical summary, reports
                exact expected figure/table counts and output paths.
                Never imports matplotlib. Zero writes.
    generate -- real generation: verifies the canonical summary, renders
                all 5 figures (PDF+PNG) and 7 tables (Markdown), writes
                the binding manifest. Requires matplotlib (only present
                in the isolated tools/paper_evidence environment).

No scientific configuration flag exists on this parser (no alternate
input/output path, no hypothesis/endpoint/threshold selector, no
formatting-policy flag, no force/bypass/partial-generation flag). No
environment variable is read anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.final_test_scientific_reporting import UnsealingAuthorizationError
from when_tta_hurts.paper_evidence import (
    MANIFEST_PATH,
    CanonicalSummaryVerificationError,
    PaperEvidenceFingerprintError,
    build_evidence_plan,
    compute_paper_evidence_fingerprint,
    generate_all_evidence,
    load_and_verify_canonical_summary,
)

_SEALED_EXCEPTION_TYPES = (
    CanonicalSummaryVerificationError,
    UnsealingAuthorizationError,
    PaperEvidenceFingerprintError,
)


def _run_plan() -> dict[str, Any]:
    report: dict[str, Any] = {"command": "plan"}
    try:
        fingerprint, _ = compute_paper_evidence_fingerprint()
        report["paper_evidence_fingerprint"] = fingerprint
    except PaperEvidenceFingerprintError as e:
        report["paper_evidence_fingerprint_status"] = f"error:{type(e).__name__}"

    try:
        summary = load_and_verify_canonical_summary()
        report["canonical_summary_ready"] = True
        report["canonical_summary_reporting_fingerprint"] = summary["reporting_fingerprint"]
        report.update(build_evidence_plan(summary))
    except CanonicalSummaryVerificationError as e:
        report["canonical_summary_ready"] = False
        report["canonical_summary_error_class"] = type(e).__name__
    except UnsealingAuthorizationError as e:
        report["canonical_summary_ready"] = False
        report["unsealing_authorization_error_class"] = type(e).__name__

    report["manifest_exists"] = MANIFEST_PATH.exists()
    return report


def _run_generate() -> dict[str, Any]:
    manifest = generate_all_evidence()
    return {"command": "generate", "status": "completed", "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "generate"])
    args = parser.parse_args()

    if args.mode == "plan":
        print(json.dumps(_run_plan(), indent=2, default=str))
        return 0

    try:
        receipt = _run_generate()
    except _SEALED_EXCEPTION_TYPES as e:
        print(json.dumps({"command": "generate", "error_class": type(e).__name__}), file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
