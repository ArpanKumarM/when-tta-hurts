#!/usr/bin/env python3
"""Confirmatory-matrix runner CLI. Phase 2B.3A: train-validation mode is
now enabled, but ONLY as a strict single-cell canary -- exactly one
explicit --run-id, resolving to one approved unconditional (Block A/B/C)
matrix cell. No --all, range, block, or wildcard execution exists.

Modes:
    plan             -- parse/validate/expand the matrix, print it, no side effects.
    train-validation -- train ONE cell named by --run-id (Block D rejected;
                         pilot/excluded-seed IDs rejected; unknown IDs
                         rejected; requires a clean working tree and a
                         real MPS device -- no CPU fallback).
    final-test       -- implemented as a hard-gated lock; requires an
                         authorization artifact that does not yet exist,
                         so this always fails closed. NOT invoked at all
                         in Phase 2B.3A.

Usage:
    uv run python scripts/run_confirmatory.py plan
    uv run python scripts/run_confirmatory.py plan --with-block-d
    uv run python scripts/run_confirmatory.py train-validation \
        --run-id A-pathmnist-28px-batchnorm-policy-none-s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.devices import DeviceUnavailableError
from when_tta_hurts.orchestrator import (
    BlockDRunRejectedError,
    DirtyWorkingTreeError,
    PilotOrExcludedSeedRunIdError,
    UnknownRunIdError,
    print_plan,
    run_canary_cell,
    run_final_test,
)


class _SingleValueAction(argparse.Action):
    """Rejects a repeated --run-id -- this canary CLI permits exactly one
    explicit run ID and nothing resembling multi-cell execution."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once (single-cell canary execution only).")
        setattr(namespace, self.dest, values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "train-validation", "final-test"])
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument(
        "--with-block-d",
        action="store_true",
        help="Include Block D (only meaningful if its gate has already passed; "
        "this flag alone does NOT prove the gate passed and plan mode does not "
        "verify it -- it only reflects what would be included IF the gate had passed).",
    )
    parser.add_argument(
        "--run-id",
        action=_SingleValueAction,
        default=None,
        help="EXACTLY one run ID to train (train-validation mode only). Must resolve to "
        "one approved unconditional Block A/B/C matrix cell. Block D, pilot, "
        "excluded-seed, and unknown IDs are all rejected.",
    )
    args = parser.parse_args()

    if args.mode == "plan":
        print_plan(args.matrix, block_d_gate_passed=args.with_block_d)
        return 0

    if args.mode == "train-validation":
        if not args.run_id:
            parser.error(
                "--run-id is required for train-validation mode (exactly one, single-cell canary only)."
            )
        try:
            result = run_canary_cell(args.run_id, matrix_path=args.matrix)
        except (
            UnknownRunIdError,
            BlockDRunRejectedError,
            PilotOrExcludedSeedRunIdError,
            DirtyWorkingTreeError,
            DeviceUnavailableError,
        ) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
        print(
            f"{result.status}\trun_id={result.run_id}\t"
            f"attempt={result.attempt_number}\tcheckpoint_hash={result.checkpoint_hash}"
        )
        return 0 if result.status in ("completed", "skipped_completed") else 1

    if args.mode == "final-test":
        # This will raise AuthorizationError -- the real authorization
        # artifact does not exist. Intentionally not caught: the failure
        # IS the expected, correct behavior for this phase.
        run_final_test()
        return 1  # unreachable while unauthorized

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
