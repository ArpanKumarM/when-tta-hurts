#!/usr/bin/env python3
"""CLI for the native-128px Block D runtime-gate benchmark, per
docs/phase2b_block_d_benchmark_spec.md.

Three explicit, separate subcommands -- no subcommand has a side effect
implied by another:

    plan       Side-effect-free. No downloads, no directories, no MPS init,
               no files written. Reports datasets/cells/formulas/paths only.
    prefetch   Explicit, separate download+checksum-verify of both Block D
               datasets at native 128px. Never called implicitly by `plan`
               or `benchmark`.
    benchmark  Real execution. Requires MPS. Fails closed on any missing or
               non-native artifact (does NOT download). Writes the gate
               decision to artifacts/benchmarks/block_d_native_128_benchmark.json.

This script never trains a real Block D matrix cell, never allocates a
confirmatory attempt, and never unlocks Block D training -- it only
produces the runtime-gate decision artifact.

Usage:
    uv run python scripts/run_block_d_benchmark.py plan
    uv run python scripts/run_block_d_benchmark.py prefetch
    uv run python scripts/run_block_d_benchmark.py benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.block_d_benchmark import (
    plan_block_d_benchmark,
    prefetch_block_d_artifacts,
    run_block_d_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("prefetch")
    subparsers.add_parser("benchmark")
    args = parser.parse_args()

    if args.command == "plan":
        result = plan_block_d_benchmark()
    elif args.command == "prefetch":
        result = prefetch_block_d_artifacts()
    else:
        result = run_block_d_benchmark()

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
