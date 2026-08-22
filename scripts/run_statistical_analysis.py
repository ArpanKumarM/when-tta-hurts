#!/usr/bin/env python3
"""Phase 2B.5A: read-only statistical-analysis CLI.

Modes:
    plan     -- side-effect-free: resolves every known analysis family
                (H1, H2, H3, BLOCK_C) to its required cells' canonical
                identity (run_id, checkpoint hash, evaluation ID,
                evaluator fingerprint) via the SAME production selection
                logic the evaluation pipeline uses. Never opens
                predictions.npz/metrics.json, never writes a file, never
                touches the test split.
    analyze  -- explicit real-analysis mode: --family {H1,H2,H3,BLOCK_C}.
                Computes the fully-specified within-cell statistics
                (paired bootstrap CI, McNemar, effect sizes) at the
                frozen primary N=50/mean_probability naive_tta condition,
                and prints the result as JSON. Does NOT persist anything
                to a ledger or attempt directory -- persistence via
                statistical_analysis_artifacts.persist_and_verify_analysis_completion
                is available as a library call but this CLI mode does not
                invoke it, since no confirmatory verdict may be produced
                yet (see docs/phase2b_statistical_analysis_engineering_freeze.md).

There is no --test-split flag, no environment variable, and no reachable
code path anywhere in this script or the modules it imports that can read
official test-split data -- validation_evaluation.py's own module
docstring states it "never accesses the official test split," and this
script imports only from when_tta_hurts.statistical_analysis, which
itself only reads artifacts/validation_evaluation/.

Usage:
    uv run python scripts/run_statistical_analysis.py plan
    uv run python scripts/run_statistical_analysis.py analyze --family H1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.statistical_analysis import (
    KNOWN_FAMILIES,
    AnalysisInputError,
    compute_family_analysis,
    plan_statistical_analysis,
)


class _SingleValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once.")
        setattr(namespace, self.dest, values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "analyze"])
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--family", action=_SingleValueAction, default=None, choices=KNOWN_FAMILIES)
    args = parser.parse_args()

    if args.mode == "plan":
        report = plan_statistical_analysis(args.matrix)
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.mode == "analyze":
        if not args.family:
            parser.error("analyze requires --family.")
        try:
            result = compute_family_analysis(args.family, matrix_path=args.matrix)
        except AnalysisInputError as e:
            print(f"ANALYSIS INPUT ERROR: {e}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
