#!/usr/bin/env python3
"""Confirmatory-matrix runner CLI. Phase 2B.2.

Modes:
    plan             -- parse/validate/expand the matrix, print it, no side effects.
    train-validation -- (implemented, NOT invoked against real data in Phase 2B.2)
    final-test       -- (implemented as a hard-gated stub; requires an
                         authorization artifact that does not yet exist,
                         so this always fails closed. NOT invoked at all
                         in Phase 2B.2.)

Usage:
    uv run python scripts/run_confirmatory.py plan
    uv run python scripts/run_confirmatory.py plan --with-block-d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.orchestrator import print_plan, run_final_test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "train-validation", "final-test"])
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument(
        "--with-block-d",
        action="store_true",
        help="Include Block D (only meaningful if its gate has already passed; "
        "this flag alone does NOT prove the gate passed and plan mode does not "
        "verify it -- it only reflects what would be included IF the gate had passed).",
    )
    args = parser.parse_args()

    if args.mode == "plan":
        print_plan(args.matrix, block_d_gate_passed=args.with_block_d)
        return 0

    if args.mode == "train-validation":
        print(
            "train-validation mode is implemented in src/when_tta_hurts/orchestrator.py "
            "but is not invoked against real data via this CLI in Phase 2B.2. Refusing.",
            file=sys.stderr,
        )
        return 1

    if args.mode == "final-test":
        # This will raise AuthorizationError -- the real authorization
        # artifact does not exist. Intentionally not caught: the failure
        # IS the expected, correct behavior for this phase.
        run_final_test()
        return 1  # unreachable while unauthorized

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
