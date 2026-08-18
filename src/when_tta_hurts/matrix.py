"""Typed, fail-closed parser and expander for configs/experiment_matrix.yaml.

Phase 2B.2: implements Part B of the Phase 2B.2 task -- parsing and cell
expansion ONLY. Does not train, does not load data, does not construct a
DataLoader. See scripts/run_confirmatory.py for the CLI that uses this
module in "plan" mode.

Design notes:
- Block iteration order is a HARDCODED LITERAL TUPLE, not dict iteration,
  per your instruction to never depend on unordered dictionary iteration
  (Python dicts are insertion-ordered since 3.7, but this module does not
  rely on that guarantee for block order -- it looks each block up by an
  explicit, frozen key name).
- Within a block, cell expansion order is dataset-major, then resolution,
  then normalization, then seed -- matching the field order documented in
  docs/phase2b_protocol.md's expanded matrix and the Phase 2B preflight's
  row-by-row report. All loop variables come from literal YAML lists
  (order-preserving by YAML/JSON semantics), never from dict iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from when_tta_hurts.config import config_hash, load_config

PILOT_SEED = 314159
PILOT_TTA_SEED = 271828
FROZEN_CONFIRMATORY_SEEDS = (0, 1, 2)

BLOCK_ORDER: tuple[str, ...] = (
    "A_core_normalization_resolution",
    "B_policy_matching",
    "C_positive_control_reproduction",
    "D_conditional_128px",
)

# Top-level keys this parser understands. Anything else in the YAML is
# rejected (fail-closed on unknown fields), per your instruction.
_ALLOWED_TOP_LEVEL_KEYS = {
    "status",
    "hardware",
    "seeds",
    "datasets",
    "resolution_note",
    "models",
    "normalization_variant_note",
    "tta_view_counts",
    "augmentation_policies",
    "aggregation_methods",
    "conditions",
    "training_matrix",
    "training_run_totals",
    "evaluation_jobs_note",
    "confirmatory_promotion_rule",
}

_ALLOWED_BLOCK_KEYS = {
    "A_core_normalization_resolution": {
        "description",
        "datasets",
        "resolutions",
        "normalization_variants",
        "seeds",
        "training_runs",
    },
    "B_policy_matching": {
        "description",
        "datasets",
        "resolution",
        "normalization",
        "training_policy",
        "seeds",
        "training_runs",
        "reuses_checkpoints_from",
    },
    "C_positive_control_reproduction": {
        "description",
        "datasets",
        "model",
        "resolution",
        "normalization",
        "seeds",
        "training_runs",
    },
    "D_conditional_128px": {
        "description",
        "datasets",
        "resolution",
        "normalization",
        "seeds",
        "training_runs",
        "status",
    },
}

EXPECTED_TRAINING_RUNS = {
    "A_core_normalization_resolution": 24,
    "B_policy_matching": 6,
    "C_positive_control_reproduction": 3,
    "D_conditional_128px": 6,
}
EXPECTED_BEFORE_CONDITIONAL = 33
EXPECTED_WITH_CONDITIONAL = 39


class MatrixValidationError(ValueError):
    """Raised on any fail-closed validation failure: missing/unknown
    fields, draft status, wrong seeds, forbidden seed, count mismatch."""


@dataclass(frozen=True)
class FrozenTrainingSettings:
    """Frozen training hyperparameters, per docs/phase2b_protocol.md sec.2.
    Not encoded in configs/experiment_matrix.yaml itself (that file
    specifies WHICH cells exist, not their shared hyperparameters) --
    hardcoded here as the single source of truth for the runner, validated
    against the frozen protocol document by tests/test_matrix.py.
    """

    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    loss: str = "cross_entropy"
    max_epochs: int = 30
    lr_schedule: str = "cosine_annealing"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    restore_best: bool = True
    batch_size_28_64px: int = 256
    precision: str = "float32"
    mixed_precision: bool = False
    device: str = "mps"
    label_smoothing: bool = False
    class_weighting: bool = False
    channel_standardization: bool = False


FROZEN_TRAINING_SETTINGS = FrozenTrainingSettings()

# Canonical, filesystem-safe short tokens for each registered training
# policy value -- used EXPLICITLY (not just presence/absence) in run_id(),
# per the Phase 2B.2 audit requirement that the stable identity encode
# training policy directly, not imply it from which block a cell belongs
# to. The matched policy applied in Block B is specifically the frozen
# "mixed" TTA policy (docs/experimental_protocol.md), hence "matched_mixed".
# Any NEW training_policy value not in this map is rejected (fail-closed)
# rather than silently falling through to a possibly-colliding token.
TRAINING_POLICY_TOKENS: dict[str, str] = {
    "none": "none",
    "matched_to_approved_tta_policy": "matched_mixed",
}


@dataclass(frozen=True)
class MatrixCell:
    """One planned training run (a single confirmatory checkpoint)."""

    block: str
    dataset: str
    resolution: int
    model: str
    normalization: str
    training_policy: str  # "none" or "matched_to_approved_tta_policy" -- see TRAINING_POLICY_TOKENS
    seed: int

    def run_id(self) -> str:
        """Deterministic, human-readable stable run ID -- see run_identity.py
        for the canonical implementation this delegates to conceptually;
        kept here too since MatrixCell is the natural place to derive it.

        training_policy is ALWAYS explicitly encoded (via TRAINING_POLICY_TOKENS),
        not merely implied by block membership -- two cells identical in
        every field except training_policy always get different run IDs,
        even if that combination doesn't happen to occur in the current
        matrix (e.g. a hypothetical future Block A cell with a non-'none'
        policy would not collide with an actual Block A 'none'-policy cell).
        """
        if self.training_policy not in TRAINING_POLICY_TOKENS:
            raise ValueError(
                f"Unregistered training_policy '{self.training_policy}' -- add it to "
                f"TRAINING_POLICY_TOKENS with an explicit, unambiguous token before use."
            )
        policy_token = TRAINING_POLICY_TOKENS[self.training_policy]
        parts = [self.block.split("_")[0], self.dataset, f"{self.resolution}px"]
        if self.model != "small_cnn":
            parts.append(self.model)
        parts.append(self.normalization)
        parts.append(f"policy-{policy_token}")
        parts.append(f"s{self.seed}")
        return "-".join(parts)


@dataclass(frozen=True)
class ExpandedMatrix:
    cells: tuple[MatrixCell, ...]
    cells_by_block: dict[str, tuple[MatrixCell, ...]]
    block_d_included: bool
    source_config_hash: str


def _require_keys(d: dict, allowed: set[str], context: str) -> None:
    if not isinstance(d, dict):
        raise MatrixValidationError(f"{context}: expected a mapping, got {type(d)}")
    unknown = set(d.keys()) - allowed
    if unknown:
        raise MatrixValidationError(f"{context}: unknown field(s) {sorted(unknown)}")


def _require_field(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise MatrixValidationError(f"{context}: missing required field '{key}'")
    return d[key]


def _validate_seeds(seeds: Any, context: str) -> tuple[int, ...]:
    if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
        raise MatrixValidationError(f"{context}: 'seeds' must be a list of ints, got {seeds!r}")
    if PILOT_SEED in seeds:
        raise MatrixValidationError(
            f"{context}: pilot seed {PILOT_SEED} is PERMANENTLY EXCLUDED from confirmatory "
            f"seeds and must never appear in the matrix. Hard failure."
        )
    return tuple(seeds)


def parse_and_validate_matrix(
    path: str | Path = "configs/experiment_matrix.yaml",
    block_d_gate_passed: bool = False,
) -> ExpandedMatrix:
    """Parse, validate, and expand the frozen confirmatory matrix.

    `block_d_gate_passed` defaults to False (fail-safe: Block D is EXCLUDED
    unless the caller explicitly proves the frozen runtime-only gate
    passed -- see block_d_gate.py::evaluate_block_d_gate). This function
    performs no I/O beyond reading the YAML file itself: no dataset
    loading, no DataLoader construction, no artifact/ledger writes.
    """
    raw = load_config(path)
    _require_keys(raw, _ALLOWED_TOP_LEVEL_KEYS, "top-level matrix")

    status = _require_field(raw, "status", "top-level matrix")
    if status != "approved":
        raise MatrixValidationError(
            f"Matrix status is '{status}', not 'approved' -- refusing to expand a "
            f"draft/unapproved baseline protocol. Hard failure."
        )

    seeds_block = _require_field(raw, "seeds", "top-level matrix")
    _require_keys(seeds_block, {"confirmatory"}, "seeds")
    confirmatory_seeds = _validate_seeds(seeds_block["confirmatory"], "seeds.confirmatory")
    if confirmatory_seeds != FROZEN_CONFIRMATORY_SEEDS:
        raise MatrixValidationError(
            f"seeds.confirmatory is {list(confirmatory_seeds)}, expected "
            f"{list(FROZEN_CONFIRMATORY_SEEDS)} exactly. Hard failure."
        )

    dataset_names = {d["name"] for d in raw.get("datasets", [])}
    model_names = {m["name"] for m in raw.get("models", [])}
    model_norms: dict[str, set[str]] = {
        m["name"]: set(m.get("normalization_variants", [])) for m in raw.get("models", [])
    }

    training_matrix = _require_field(raw, "training_matrix", "top-level matrix")
    if set(training_matrix.keys()) != set(BLOCK_ORDER):
        raise MatrixValidationError(
            f"training_matrix keys {sorted(training_matrix.keys())} do not exactly match "
            f"the frozen block set {sorted(BLOCK_ORDER)}. Hard failure."
        )

    cells_by_block: dict[str, tuple[MatrixCell, ...]] = {}
    all_cells: list[MatrixCell] = []

    for block_name in BLOCK_ORDER:  # literal tuple order, not dict iteration
        block = training_matrix[block_name]
        _require_keys(block, _ALLOWED_BLOCK_KEYS[block_name], block_name)

        block_seeds = _validate_seeds(_require_field(block, "seeds", block_name), f"{block_name}.seeds")
        stated_runs = _require_field(block, "training_runs", block_name)

        block_datasets = _require_field(block, "datasets", block_name)
        if not isinstance(block_datasets, list):
            raise MatrixValidationError(f"{block_name}.datasets must be a list")
        for ds in block_datasets:
            if ds not in dataset_names:
                raise MatrixValidationError(
                    f"{block_name} references unregistered dataset '{ds}' "
                    f"(not in top-level datasets: {sorted(dataset_names)})"
                )

        block_model = block.get("model", "small_cnn")
        if block_model not in model_names:
            raise MatrixValidationError(f"{block_name} references unregistered model '{block_model}'")

        if "resolutions" in block:
            block_resolutions = block["resolutions"]
        else:
            block_resolutions = [_require_field(block, "resolution", block_name)]

        if "normalization_variants" in block:
            block_norms = block["normalization_variants"]
        else:
            block_norms = [_require_field(block, "normalization", block_name)]
        for norm in block_norms:
            if norm not in model_norms.get(block_model, set()):
                raise MatrixValidationError(
                    f"{block_name} references normalization '{norm}' not registered for "
                    f"model '{block_model}' ({sorted(model_norms.get(block_model, set()))})"
                )

        training_policy = block.get("training_policy", "none")

        block_cells: list[MatrixCell] = []
        if block_name == "D_conditional_128px" and not block_d_gate_passed:
            # Excluded unless the frozen runtime-only gate has explicitly
            # passed -- fail-safe default is exclusion, not inclusion.
            cells_by_block[block_name] = ()
        else:
            for ds in block_datasets:  # dataset-major
                for res in block_resolutions:  # then resolution
                    for norm in block_norms:  # then normalization
                        for seed in block_seeds:  # then seed
                            block_cells.append(
                                MatrixCell(
                                    block=block_name,
                                    dataset=ds,
                                    resolution=res,
                                    model=block_model,
                                    normalization=norm,
                                    training_policy=training_policy,
                                    seed=seed,
                                )
                            )
            if len(block_cells) != stated_runs:
                raise MatrixValidationError(
                    f"{block_name}: expanded {len(block_cells)} cells, but YAML states "
                    f"training_runs={stated_runs}. Hard failure (count mismatch)."
                )
            expected = EXPECTED_TRAINING_RUNS[block_name]
            if len(block_cells) != expected:
                raise MatrixValidationError(
                    f"{block_name}: expanded {len(block_cells)} cells, expected exactly "
                    f"{expected} per the frozen protocol. Hard failure."
                )
            cells_by_block[block_name] = tuple(block_cells)
            all_cells.extend(block_cells)

    mandatory_count = sum(len(cells_by_block[b]) for b in BLOCK_ORDER[:3])
    if mandatory_count != EXPECTED_BEFORE_CONDITIONAL:
        raise MatrixValidationError(
            f"A+B+C expanded to {mandatory_count} cells, expected exactly "
            f"{EXPECTED_BEFORE_CONDITIONAL}. Hard failure."
        )

    total_count = len(all_cells)
    block_d_included = len(cells_by_block["D_conditional_128px"]) > 0
    if block_d_included and total_count != EXPECTED_WITH_CONDITIONAL:
        raise MatrixValidationError(
            f"A+B+C+D expanded to {total_count} cells, expected exactly "
            f"{EXPECTED_WITH_CONDITIONAL}. Hard failure."
        )
    if not block_d_included and total_count != EXPECTED_BEFORE_CONDITIONAL:
        raise MatrixValidationError(
            f"A+B+C (Block D excluded) expanded to {total_count} cells, expected exactly "
            f"{EXPECTED_BEFORE_CONDITIONAL}. Hard failure."
        )

    for cell in all_cells:
        if cell.seed == PILOT_SEED:
            raise MatrixValidationError(f"Cell {cell} uses forbidden pilot seed {PILOT_SEED}. Hard failure.")

    return ExpandedMatrix(
        cells=tuple(all_cells),
        cells_by_block=cells_by_block,
        block_d_included=block_d_included,
        source_config_hash=config_hash(raw),
    )
