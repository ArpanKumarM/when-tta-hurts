#!/usr/bin/env python3
"""Phase 2B.6A: final-test-evaluation CLI for the Phase 2B 39-cell
confirmatory matrix.

Modes:
    plan               -- side-effect-free, TEST-DATA-FREE: reports every
                           eligible cell's canonical-training-completion
                           identity, existing final-test attempt counts,
                           and whether a committed, current final-test
                           authorization exists. Never initializes MPS,
                           opens a checkpoint/dataset, creates a file/
                           directory, writes a ledger, or computes a
                           prediction.
    evaluate-test      -- exact single-run final-test evaluation:
                           --run-id <exact-training-run-id>. Dispatches to
                           final_test_evaluation.run_final_test_evaluation(),
                           which verifies the committed final-test
                           authorization artifact BEFORE device
                           initialization, checkpoint loading, or any
                           test-array access. Without that committed
                           artifact, this command refuses before any
                           heavy dependency is touched.

There is no --block/--all-cells mode. There is no flag anywhere in this
script for: split selection, an authorization-path override, --force,
--retry, an alternate evaluator config, a TTA-seed override, a policy
override, a batch-size override, a synthetic backend, or an environment-
variable bypass of any kind. The evaluated split is always "test" -- there
is no split argument of any kind.

NO SCIENTIFIC METRIC VALUE IS EVER PRINTED BY THIS SCRIPT. `evaluate-test`
prints only identity (run ID, final-test evaluation ID), attempt number,
status, and the primary artifact's hash -- never a value from metrics.json
or predictions.npz.

Usage:
    uv run python scripts/run_final_test_evaluation.py plan
    uv run python scripts/run_final_test_evaluation.py evaluate-test \
        --run-id A-pathmnist-28px-batchnorm-policy-none-s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.devices import DeviceUnavailableError
from when_tta_hurts.evaluation.test_loader import TestLoaderError
from when_tta_hurts.final_test_authorization import FinalTestAuthorizationError
from when_tta_hurts.final_test_evaluation import (
    FinalTestAuthorizationRequiredError,
    plan_final_test_evaluation,
    run_final_test_evaluation,
)
from when_tta_hurts.final_test_identity import FinalTestFingerprintError
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
    FrozenTTASeedConfigError,
    NoCanonicalTrainingCompletionError,
)

# Every exception that must cause a clean REFUSAL (exit 1, no traceback,
# no partial output) rather than crashing -- deliberately does NOT include
# a generic `Exception` catch-all, so an unanticipated bug surfaces loudly
# rather than being silently reported as a routine refusal.
_REFUSAL_EXCEPTIONS = (
    UnknownRunIdError,
    PilotOrExcludedSeedRunIdError,
    NoCanonicalTrainingCompletionError,
    FrozenTTASeedConfigError,
    BlockDAuthorizationError,
    DeviceUnavailableError,
    EvaluationStaleAttemptError,
    ConflictingCompletedRunError,
    AmbiguousCanonicalCompletionError,
    PersistenceVerificationError,
    FinalTestAuthorizationError,
    FinalTestAuthorizationRequiredError,
    FinalTestFingerprintError,
    TestLoaderError,
)


class _SingleValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once.")
        setattr(namespace, self.dest, values)


def _redact_for_print(result: dict) -> dict:
    """Return ONLY identity/attempt/status/artifact-hash fields -- never a
    scientific metric value. Used for every evaluate-test print, on both
    the completed and skipped-completed paths."""
    allowed_keys = (
        "status",
        "training_run_id",
        "final_test_evaluation_id",
        "attempt_number",
    )
    redacted = {k: result[k] for k in allowed_keys if k in result}
    manifest = result.get("artifact_manifest")
    if isinstance(manifest, dict) and "artifacts" in manifest:
        redacted["artifact_hashes"] = {
            entry["path"]: entry["sha256"] for entry in manifest["artifacts"] if "path" in entry
        }
    return redacted


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["plan", "evaluate-test"])
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--run-id", action=_SingleValueAction, default=None)
    args = parser.parse_args()

    if args.mode == "plan":
        report = plan_final_test_evaluation(args.matrix)
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.mode == "evaluate-test":
        if not args.run_id:
            parser.error("evaluate-test requires --run-id.")
        try:
            result = run_final_test_evaluation(args.run_id, matrix_path=args.matrix)
        except _REFUSAL_EXCEPTIONS as e:
            print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print(json.dumps(_redact_for_print(result), indent=2, default=str))
        return 0 if result["status"] in ("completed", "skipped_completed") else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
