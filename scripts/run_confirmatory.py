#!/usr/bin/env python3
"""Confirmatory-matrix runner CLI.

Modes:
    plan               -- parse/validate/expand the matrix, print it, no side effects.
    train-validation    -- either:
                            (a) --run-id: train ONE cell, single-cell canary mode.
                            (b) --block {A,B,C} --expected-total N --expected-pending M:
                                execute every cell of that block sequentially, in
                                committed matrix order. --run-id and --block are
                                mutually exclusive. Block D is always rejected.
                                Both --expected-total/--expected-pending are
                                mandatory with --block and are verified BEFORE any
                                MPS/dataset/model activity for any cell.
    verify-completions  -- metadata-only: --block {A,B,C} --expected-total N.
                            Never touches MPS, datasets, models, or DataLoaders,
                            and makes no filesystem/ledger changes.
    final-test          -- hard-gated lock; requires an authorization artifact
                            that does not yet exist, so this always fails closed.

Usage:
    uv run python scripts/run_confirmatory.py plan
    uv run python scripts/run_confirmatory.py plan --with-block-d
    uv run python scripts/run_confirmatory.py train-validation \
        --run-id A-pathmnist-28px-batchnorm-policy-none-s0
    uv run python scripts/run_confirmatory.py train-validation \
        --block A --expected-total 24 --expected-pending 23
    uv run python scripts/run_confirmatory.py verify-completions \
        --block A --expected-total 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.devices import DeviceUnavailableError
from when_tta_hurts.orchestrator import (
    AmbiguousCanonicalCompletionError,
    BlockDRunRejectedError,
    DirtyWorkingTreeError,
    PersistenceVerificationError,
    PilotOrExcludedSeedRunIdError,
    UnknownRunIdError,
    UnsupportedBlockError,
    print_plan,
    run_block_cells,
    run_canary_cell,
    run_final_test,
    verify_block_completions,
)
from when_tta_hurts.run_identity import ConflictingCompletedRunError


class _SingleValueAction(argparse.Action):
    """Rejects a repeated flag -- this CLI permits exactly one explicit
    value per identifying argument, never a list/range/wildcard."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once.")
        setattr(namespace, self.dest, values)


def _print_cell_result(result) -> None:
    print(
        f"{result.status}\trun_id={result.run_id}\tattempt={result.attempt_number}\t"
        f"checkpoint_hash={result.checkpoint_hash}\tconfig_hash={result.config_hash}\t"
        f"manifest_verified={result.manifest_verified}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "train-validation", "verify-completions", "final-test"])
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
        help="EXACTLY one run ID to train (train-validation mode only, mutually exclusive "
        "with --block). Must resolve to one approved unconditional Block A/B/C matrix cell.",
    )
    parser.add_argument(
        "--block",
        choices=["A", "B", "C"],
        action=_SingleValueAction,
        default=None,
        help="Execute (train-validation) or report (verify-completions) an entire block "
        "sequentially, in committed matrix order. Block D is never accepted here.",
    )
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--expected-pending", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "plan":
        print_plan(args.matrix, block_d_gate_passed=args.with_block_d)
        return 0

    if args.mode == "train-validation":
        if args.run_id and args.block:
            parser.error("--run-id and --block are mutually exclusive.")
        if not args.run_id and not args.block:
            parser.error("train-validation requires exactly one of --run-id or --block.")

        if args.run_id:
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
            _print_cell_result(result)
            return 0 if result.status in ("completed", "skipped_completed") else 1

        # --block path
        if args.expected_total is None or args.expected_pending is None:
            parser.error("--block requires both --expected-total and --expected-pending.")
        try:
            results = run_block_cells(
                args.block,
                expected_total=args.expected_total,
                expected_pending=args.expected_pending,
                matrix_path=args.matrix,
            )
        except (
            UnsupportedBlockError,
            ValueError,
            DirtyWorkingTreeError,
            DeviceUnavailableError,
            ConflictingCompletedRunError,
            AmbiguousCanonicalCompletionError,
            PersistenceVerificationError,
        ) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
        for result in results:
            _print_cell_result(result)
        any_failed = any(r.status == "failed" for r in results)
        print(f"\nBlock {args.block}: {len(results)} cell(s) processed this invocation.")
        return 1 if any_failed else 0

    if args.mode == "verify-completions":
        if not args.block:
            parser.error("verify-completions requires --block.")
        if args.expected_total is None:
            parser.error("verify-completions requires --expected-total.")
        try:
            report = verify_block_completions(
                args.block, expected_total=args.expected_total, matrix_path=args.matrix
            )
        except (UnsupportedBlockError, ValueError) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
        print(
            f"Block {report['block']}: {report['canonical_count']}/{report['expected_total']} "
            f"canonical-eligible completions, missing={len(report['missing'])}, "
            f"ambiguous={len(report['ambiguous'])}, corrupt={len(report['corrupt'])}"
        )
        for cell_report in report["cells"]:
            print(cell_report)
        return 0 if report["canonical_count"] == report["expected_total"] else 1

    if args.mode == "final-test":
        # This will raise AuthorizationError -- the real authorization
        # artifact does not exist. Intentionally not caught: the failure
        # IS the expected, correct behavior for this phase.
        run_final_test()
        return 1  # unreachable while unauthorized

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
