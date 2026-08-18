#!/usr/bin/env python3
"""Phase 2B.4A: validation-only TTA evaluation CLI.

Modes:
    plan                 -- side-effect-free: lists all eligible training
                             cells, their canonical training completion (if
                             any), and the required frozen analyses. Never
                             initializes MPS, opens a dataset/checkpoint,
                             creates a file/directory, writes a ledger, or
                             computes a prediction.
    evaluate-validation   -- exact single-run evaluation:
                             --run-id <exact-training-run-id> --tta-seed N.
                             Dispatches to validation_evaluation.run_validation_evaluation(),
                             which enforces canonical-checkpoint resolution,
                             deterministic evaluation identity, idempotent
                             skip, and the full frozen validation-only
                             computation before writing any artifact.

There is no --block/--all-cells route in this phase (a single real canary
must pass first), and no flag anywhere in this script exists to override
authorization, the split, the augmentation policy, the prefix sequence, or
to select a synthetic production backend. No environment variable is read
anywhere in this script or the module it calls. The evaluated split is
always "validation" -- there is no split argument
of any kind.

Usage:
    uv run python scripts/run_validation_evaluation.py plan
    uv run python scripts/run_validation_evaluation.py evaluate-validation \
        --run-id A-pathmnist-28px-batchnorm-policy-none-s0 --tta-seed <N>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.devices import DeviceUnavailableError
from when_tta_hurts.orchestrator import (
    AmbiguousCanonicalCompletionError,
    BlockDAuthorizationError,
    PilotOrExcludedSeedRunIdError,
    UnknownRunIdError,
)
from when_tta_hurts.result_artifacts import PersistenceVerificationError
from when_tta_hurts.run_identity import ConflictingCompletedRunError
from when_tta_hurts.validation_evaluation import (
    EvaluationStaleAttemptError,
    NoCanonicalTrainingCompletionError,
    NoConfirmatoryTTASeedYetError,
    plan_validation_evaluation,
    run_validation_evaluation,
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
    parser.add_argument("mode", choices=["plan", "evaluate-validation"])
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--run-id", action=_SingleValueAction, default=None)
    parser.add_argument("--tta-seed", type=int, action=_SingleValueAction, default=None)
    args = parser.parse_args()

    if args.mode == "plan":
        rows = plan_validation_evaluation(args.matrix)
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if args.mode == "evaluate-validation":
        if not args.run_id:
            parser.error("evaluate-validation requires --run-id.")
        if args.tta_seed is None:
            parser.error("evaluate-validation requires --tta-seed.")
        try:
            result = run_validation_evaluation(args.run_id, args.tta_seed, matrix_path=args.matrix)
        except (
            UnknownRunIdError,
            PilotOrExcludedSeedRunIdError,
            NoCanonicalTrainingCompletionError,
            NoConfirmatoryTTASeedYetError,
            BlockDAuthorizationError,
            DeviceUnavailableError,
            EvaluationStaleAttemptError,
            ConflictingCompletedRunError,
            AmbiguousCanonicalCompletionError,
            PersistenceVerificationError,
        ) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] in ("completed", "skipped_completed") else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
