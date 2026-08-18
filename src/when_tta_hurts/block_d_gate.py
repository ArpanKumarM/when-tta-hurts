"""Frozen Block D numeric-trigger decision, per docs/phase2b_protocol.md
sec.6. Pure function over SUPPLIED benchmark metadata -- this module does
not download artifacts, does not run a benchmark, and does not itself
measure anything. See docs/phase2b_protocol.md for the frozen thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_TRAINING_MINUTES_PER_RUN = 90
MAX_END_TO_END_MINUTES_PER_CELL = 120
MAX_PESSIMISTIC_TOTAL_HOURS = 24.0


@dataclass(frozen=True)
class DatasetBenchmarkRecord:
    """One dataset's (PathMNIST-128 or BloodMNIST-128) supplied benchmark
    evidence -- must come from a REAL native benchmark; this dataclass does
    not verify that itself (see field docs), the caller must supply
    honest data."""

    dataset: str
    artifact_is_native_128px: bool  # must be True; resized 28/64px proxies never satisfy the gate
    checksum_expected: str
    checksum_actual: str
    device: str  # must be "mps"
    oom_occurred: bool
    non_finite_loss_occurred: bool
    projected_training_minutes_per_run: float
    projected_end_to_end_minutes_per_cell: float


@dataclass(frozen=True)
class BlockDGateResult:
    activated: bool
    reasons: tuple[str, ...]  # human-readable pass/fail reasons, one per checked condition
    per_dataset_pass: dict[str, bool]


def _evaluate_dataset(record: DatasetBenchmarkRecord) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True

    if not record.artifact_is_native_128px:
        ok = False
        reasons.append(f"{record.dataset}: artifact is NOT native 128px (resized proxy rejected)")
    else:
        reasons.append(f"{record.dataset}: native 128px artifact confirmed")

    if record.checksum_expected != record.checksum_actual:
        ok = False
        reasons.append(
            f"{record.dataset}: checksum MISMATCH (expected {record.checksum_expected}, "
            f"got {record.checksum_actual})"
        )
    else:
        reasons.append(f"{record.dataset}: checksum verified")

    if record.device != "mps":
        ok = False
        reasons.append(f"{record.dataset}: device is '{record.device}', not 'mps'")
    else:
        reasons.append(f"{record.dataset}: device confirmed mps")

    if record.oom_occurred:
        ok = False
        reasons.append(f"{record.dataset}: OOM occurred during benchmark")
    else:
        reasons.append(f"{record.dataset}: no OOM")

    if record.non_finite_loss_occurred:
        ok = False
        reasons.append(f"{record.dataset}: non-finite loss occurred during benchmark")
    else:
        reasons.append(f"{record.dataset}: loss finite throughout")

    if record.projected_training_minutes_per_run > MAX_TRAINING_MINUTES_PER_RUN:
        ok = False
        reasons.append(
            f"{record.dataset}: projected training {record.projected_training_minutes_per_run:.1f}min/run "
            f"exceeds {MAX_TRAINING_MINUTES_PER_RUN}min limit"
        )
    else:
        reasons.append(
            f"{record.dataset}: projected training "
            f"{record.projected_training_minutes_per_run:.1f}min/run within limit"
        )

    if record.projected_end_to_end_minutes_per_cell > MAX_END_TO_END_MINUTES_PER_CELL:
        ok = False
        reasons.append(
            f"{record.dataset}: projected end-to-end "
            f"{record.projected_end_to_end_minutes_per_cell:.1f}min/cell exceeds "
            f"{MAX_END_TO_END_MINUTES_PER_CELL}min limit"
        )
    else:
        reasons.append(
            f"{record.dataset}: projected end-to-end "
            f"{record.projected_end_to_end_minutes_per_cell:.1f}min/cell within limit"
        )

    return ok, reasons


def evaluate_block_d_gate(
    pathmnist_record: DatasetBenchmarkRecord,
    bloodmnist_record: DatasetBenchmarkRecord,
    pessimistic_total_hours_estimate: float,
) -> BlockDGateResult:
    """Evaluate the frozen Block D trigger. Activates only if BOTH datasets
    pass every condition AND the pessimistic A+B+C+D total stays under 24h.
    Uses runtime/memory evidence ONLY -- no accuracy/TTA field exists on
    DatasetBenchmarkRecord by design, so it cannot influence this decision.
    """
    if pathmnist_record.dataset != "pathmnist":
        raise ValueError(f"pathmnist_record.dataset must be 'pathmnist', got {pathmnist_record.dataset!r}")
    if bloodmnist_record.dataset != "bloodmnist":
        raise ValueError(f"bloodmnist_record.dataset must be 'bloodmnist', got {bloodmnist_record.dataset!r}")

    path_ok, path_reasons = _evaluate_dataset(pathmnist_record)
    blood_ok, blood_reasons = _evaluate_dataset(bloodmnist_record)

    reasons = list(path_reasons) + list(blood_reasons)

    total_ok = pessimistic_total_hours_estimate < MAX_PESSIMISTIC_TOTAL_HOURS
    if total_ok:
        reasons.append(
            f"pessimistic A+B+C+D total {pessimistic_total_hours_estimate:.2f}h "
            f"within {MAX_PESSIMISTIC_TOTAL_HOURS}h limit"
        )
    else:
        reasons.append(
            f"pessimistic A+B+C+D total {pessimistic_total_hours_estimate:.2f}h "
            f"EXCEEDS {MAX_PESSIMISTIC_TOTAL_HOURS}h limit"
        )

    activated = path_ok and blood_ok and total_ok

    return BlockDGateResult(
        activated=activated,
        reasons=tuple(reasons),
        per_dataset_pass={"pathmnist": path_ok, "bloodmnist": blood_ok},
    )
