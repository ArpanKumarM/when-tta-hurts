"""Phase 2B.5C: fixed-model, image-paired difference-in-differences (DiD)
addendum for H1/H2/H3, per
docs/phase2b_final_test_cross_condition_addendum.md and
configs/final_test_cross_condition_addendum.yaml.

Post-validation, pre-test-specified SECONDARY analysis only:
- never originally preregistered (see the freeze doc's provenance section);
- never a general method-level H1/H2/H3 verdict;
- never pools seeds, datasets, resolutions, or normalizations into a
  single model-population statistic or p-value;
- structurally separate from (a) the original within-cell confirmatory
  statistics in statistical_analysis.py and (b) all descriptive/
  exploratory analyses.

This module has no test-split argument, flag, environment variable, or
loader anywhere. `plan_cross_condition_addendum()` (side-effect-free) and
this module's own tests never read predictions.npz/metrics.json for real
data; the real DiD-analysis mode (`compute_hypothesis_did`) is
implemented per the frozen spec but is NEVER invoked outside this
module's own tests during Phase 2B.5C.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash, load_config
from when_tta_hurts.matrix import parse_and_validate_matrix
from when_tta_hurts.statistical_analysis import (
    ANALYSIS_FINGERPRINT_MANIFEST,
    _resolve_canonical_evaluation_identity,
)
from when_tta_hurts.validation_evaluation import compute_evaluator_fingerprint

ADDENDUM_CONFIG_PATH = "configs/final_test_cross_condition_addendum.yaml"

CLASSIFICATION = "post_validation_pre_test_secondary"

KNOWN_HYPOTHESES: tuple[str, ...] = ("H1", "H2", "H3")

# Every file whose content could change a reported cross-condition DiD
# number: the addendum's own frozen spec, this implementation, and every
# file ANALYSIS_FINGERPRINT_MANIFEST already tracks (matrix parsing,
# canonical-selection logic, metrics/artifact code, the dependency lock),
# plus ledger.py -- load-bearing for canonical-evaluation-identity
# resolution (amendment exclusion, ledger reads) but not itself in
# ANALYSIS_FINGERPRINT_MANIFEST. Deliberately excludes docs/ -- a
# documentation-only commit must never change this identity.
CROSS_CONDITION_ADDENDUM_MANIFEST: tuple[str, ...] = ANALYSIS_FINGERPRINT_MANIFEST + (
    "src/when_tta_hurts/ledger.py",
    ADDENDUM_CONFIG_PATH,
    "src/when_tta_hurts/cross_condition_addendum.py",
)


class AddendumSpecError(RuntimeError):
    """Raised on a missing/malformed addendum config file, or when the
    on-disk config no longer matches the frozen contract this module
    requires (wrong endpoint, a pooling rule flipped on, 128px re-enabled
    for H2 inference, etc). Fails closed -- never silently proceeds under
    a different contract than the one frozen in
    docs/phase2b_final_test_cross_condition_addendum.md."""


class AddendumInputError(RuntimeError):
    """Raised when a fixed pair's inputs are missing, ambiguous, stale,
    mismatched (labels/sample-indices), or otherwise unusable for a real
    DiD computation. Fails closed -- never silently substitutes a
    different cell, drops a mismatched sample, or reorders an array."""


def load_addendum_spec(config_path: str | Path = ADDENDUM_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen addendum spec. Raises AddendumSpecError
    if the file is missing or has drifted from the frozen contract (wrong
    endpoint, pooling re-enabled, 128px re-enabled for H2, etc) -- this
    module never proceeds under an unrecognized contract."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise AddendumSpecError(f"Addendum spec file missing: {config_path}")
    spec = load_config(config_path)

    endpoint = spec.get("endpoint", {})
    if (
        endpoint.get("metric") != "accuracy"
        or endpoint.get("condition") != "naive_tta"
        or endpoint.get("aggregator") != "mean_probability"
        or endpoint.get("tta_view_count") != 50
    ):
        raise AddendumSpecError(f"Addendum endpoint spec does not match the frozen contract: {endpoint!r}")

    for key in ("h1_pairs", "h2_pairs", "h3_pairs", "bootstrap", "reporting"):
        if key not in spec:
            raise AddendumSpecError(f"Addendum spec missing required section: {key!r}")

    reporting = spec["reporting"]
    if reporting.get("produce_model_population_p_value") is not False:
        raise AddendumSpecError("Addendum reporting spec must forbid model-population p-values.")
    if reporting.get("produce_seed_level_sign_or_permutation_test") is not False:
        raise AddendumSpecError("Addendum reporting spec must forbid the seed-level sign/permutation test.")
    if reporting.get("fit_mixed_effects_or_hierarchical_model") is not False:
        raise AddendumSpecError("Addendum reporting spec must forbid mixed-effects/hierarchical modeling.")
    if spec["h2_pairs"].get("block_d_128px_included_in_inference") is not False:
        raise AddendumSpecError("Addendum spec must exclude Block D 128px from H2 inference.")

    return spec


def compute_cross_condition_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = CROSS_CONDITION_ADDENDUM_MANIFEST,
) -> tuple[str, dict[str, str]]:
    """Deterministic content fingerprint of every file that could change a
    reported cross-condition DiD number. Fails closed on a missing
    manifest file -- never computes a partial fingerprint. Independent of
    documentation-only commits; changes whenever the addendum spec or its
    implementation/dependencies change."""
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise AddendumSpecError(
                f"Cross-condition fingerprint manifest file missing: {rel_path}. Refusing to compute a "
                f"partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


def derive_bootstrap_seed(hypothesis: str, pair_id: str, analysis_fingerprint: str) -> int:
    """Deterministic uint64 bootstrap seed, per
    configs/final_test_cross_condition_addendum.yaml's `bootstrap.seed_derivation`:
    a pure, collision-resistant function of the pair's own identity, so a
    re-run of the same frozen analysis against the same frozen inputs is
    always exactly reproducible, and no two distinct pairs can accidentally
    share a seed."""
    digest = hashlib.sha256(
        f"phase2b5c_did_bootstrap_seed{hypothesis}{pair_id}{analysis_fingerprint}".encode()
    ).digest()
    return struct.unpack(">Q", digest[:8])[0]


# ---------------------------------------------------------------------------
# Fixed-pair derivation -- mechanically derived from the frozen matrix and
# the frozen addendum spec, never hand-typed run-ID lists.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedPair:
    hypothesis: str
    pair_id: str
    dataset: str
    seed: int
    condition_a_run_id: str
    condition_b_run_id: str
    condition_a_label: str
    condition_b_label: str
    directionality: str  # "two_sided" or "one_sided"


def _matrix_cells(matrix_path: str = "configs/experiment_matrix.yaml"):
    return list(parse_and_validate_matrix(matrix_path, block_d_gate_passed=True).cells)


def derive_fixed_pairs(
    hypothesis: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    addendum_spec: dict[str, Any] | None = None,
) -> tuple[FixedPair, ...]:
    """Mechanically derive `hypothesis`'s fixed pairs from the frozen
    matrix and the frozen addendum spec. Raises ValueError for any
    hypothesis not in KNOWN_HYPOTHESES. Returns pairs sorted by pair_id for
    determinism. Never pools -- one pair per (dataset, seed[, normalization
    or resolution as applicable])."""
    if hypothesis not in KNOWN_HYPOTHESES:
        raise ValueError(f"Unknown addendum hypothesis: {hypothesis!r}. Known: {KNOWN_HYPOTHESES}.")

    spec = addendum_spec if addendum_spec is not None else load_addendum_spec()
    cells = _matrix_cells(matrix_path)
    pairs: list[FixedPair] = []

    if hypothesis == "H1":
        h1 = spec["h1_pairs"]
        a_norm, b_norm = h1["condition_a_normalization"], h1["condition_b_normalization"]
        resolutions = set(h1["eligible_resolutions"])
        a_cells = {
            (c.dataset, c.resolution, c.seed): c
            for c in cells
            if c.block == "A_core_normalization_resolution"
            and c.normalization == a_norm
            and c.resolution in resolutions
        }
        b_cells = {
            (c.dataset, c.resolution, c.seed): c
            for c in cells
            if c.block == "A_core_normalization_resolution"
            and c.normalization == b_norm
            and c.resolution in resolutions
        }
        for key in sorted(set(a_cells) & set(b_cells)):
            dataset, resolution, seed = key
            a, b = a_cells[key], b_cells[key]
            pairs.append(
                FixedPair(
                    hypothesis="H1",
                    pair_id=f"H1-{dataset}-{resolution}px-s{seed}",
                    dataset=dataset,
                    seed=seed,
                    condition_a_run_id=a.run_id(),
                    condition_b_run_id=b.run_id(),
                    condition_a_label=a_norm,
                    condition_b_label=b_norm,
                    directionality=h1["directionality"],
                )
            )

    elif hypothesis == "H2":
        h2 = spec["h2_pairs"]
        res_a, res_b = h2["condition_a_resolution"], h2["condition_b_resolution"]
        norms = set(h2["eligible_normalizations"])
        a_cells = {
            (c.dataset, c.normalization, c.seed): c
            for c in cells
            if c.block == "A_core_normalization_resolution"
            and c.resolution == res_a
            and c.normalization in norms
        }
        b_cells = {
            (c.dataset, c.normalization, c.seed): c
            for c in cells
            if c.block == "A_core_normalization_resolution"
            and c.resolution == res_b
            and c.normalization in norms
        }
        for key in sorted(set(a_cells) & set(b_cells)):
            dataset, normalization, seed = key
            a, b = a_cells[key], b_cells[key]
            pairs.append(
                FixedPair(
                    hypothesis="H2",
                    pair_id=f"H2-{dataset}-{normalization}-s{seed}",
                    dataset=dataset,
                    seed=seed,
                    condition_a_run_id=a.run_id(),
                    condition_b_run_id=b.run_id(),
                    condition_a_label=f"{res_a}px",
                    condition_b_label=f"{res_b}px",
                    directionality=h2["directionality"],
                )
            )

    elif hypothesis == "H3":
        h3 = spec["h3_pairs"]
        a_cells = {
            (c.dataset, c.seed): c
            for c in cells
            if c.block == h3["condition_a_block"]
            and c.training_policy == h3["condition_a_training_policy"]
            and c.resolution == h3["eligible_resolution"]
            and c.normalization == h3["eligible_normalization"]
        }
        b_cells = {
            (c.dataset, c.seed): c
            for c in cells
            if c.block == h3["condition_b_block"]
            and c.training_policy == h3["condition_b_training_policy"]
            and c.resolution == h3["eligible_resolution"]
            and c.normalization == h3["eligible_normalization"]
        }
        for key in sorted(set(a_cells) & set(b_cells)):
            dataset, seed = key
            a, b = a_cells[key], b_cells[key]
            pairs.append(
                FixedPair(
                    hypothesis="H3",
                    pair_id=f"H3-{dataset}-s{seed}",
                    dataset=dataset,
                    seed=seed,
                    condition_a_run_id=a.run_id(),
                    condition_b_run_id=b.run_id(),
                    condition_a_label="unmatched",
                    condition_b_label="matched",
                    directionality=h3["directionality"],
                )
            )

    return tuple(sorted(pairs, key=lambda p: p.pair_id))


# ---------------------------------------------------------------------------
# Difference-in-differences statistics -- fully specified, no invention.
# ---------------------------------------------------------------------------


def _validate_four_arrays(clean_a, tta_a, clean_b, tta_b) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(x, dtype=bool) for x in (clean_a, tta_a, clean_b, tta_b))
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"clean_a/tta_a/clean_b/tta_b must all share the same aligned shape (same images, same "
            f"order); got shapes {[a.shape for a in arrays]}."
        )
    if arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("Correctness arrays must be non-empty 1-D arrays.")
    return arrays


def per_image_did(
    clean_a: np.ndarray, tta_a: np.ndarray, clean_b: np.ndarray, tta_b: np.ndarray
) -> np.ndarray:
    """Per-image DiD contribution: d_i = (T_B,i - C_B,i) - (T_A,i - C_A,i).
    All four arrays must be aligned index-for-index to the same test
    images (same model pair, same sample order)."""
    clean_a, tta_a, clean_b, tta_b = (
        a.astype(np.float64) for a in _validate_four_arrays(clean_a, tta_a, clean_b, tta_b)
    )
    return (tta_b - clean_b) - (tta_a - clean_a)


def did_point_estimate(
    clean_a: np.ndarray, tta_a: np.ndarray, clean_b: np.ndarray, tta_b: np.ndarray
) -> float:
    """DiD = mean_i(d_i), algebraically equal to
    (accuracy_TTA,B - accuracy_clean,B) - (accuracy_TTA,A - accuracy_clean,A)."""
    return float(per_image_did(clean_a, tta_a, clean_b, tta_b).mean())


def did_bootstrap_ci(
    clean_a: np.ndarray,
    tta_a: np.ndarray,
    clean_b: np.ndarray,
    tta_b: np.ndarray,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Paired bootstrap CI on the fixed-pair DiD estimand, reusing the
    frozen SAP's bootstrap parameters (resample test-image indices with
    replacement, >= 10,000 resamples, 95% CI). Every replicate resamples
    ONE joint index set and applies it to all four correctness arrays --
    independent resampling of the four arrays is never performed (that
    would break the pairing structure and inflate variance non-physically).
    Fails closed on mismatched shapes, empty input, or invalid
    n_resamples/ci_level."""
    clean_a, tta_a, clean_b, tta_b = _validate_four_arrays(clean_a, tta_a, clean_b, tta_b)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")

    rng = np.random.default_rng(seed)
    n = clean_a.size
    point = did_point_estimate(clean_a, tta_a, clean_b, tta_b)

    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = did_point_estimate(clean_a[idx], tta_a[idx], clean_b[idx], tta_b[idx])

    alpha = 1.0 - ci_level
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "did": point,
        "ci_level": ci_level,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": n_resamples,
        "n_samples": n,
        "bootstrap_seed": seed,
    }


# ---------------------------------------------------------------------------
# Plan mode -- side-effect-free, identity/manifest only, never reads a
# scientific metric value.
# ---------------------------------------------------------------------------


def plan_cross_condition_addendum(
    matrix_path: str = "configs/experiment_matrix.yaml",
    hypotheses: tuple[str, ...] = KNOWN_HYPOTHESES,
) -> dict[str, Any]:
    """SIDE-EFFECT-FREE. Never opens predictions.npz or metrics.json --
    only resolves each hypothesis's required fixed pairs to their
    canonical evaluation identity via the same production selection logic
    the evaluation pipeline and statistical_analysis.py's plan mode use.
    Reports missing/ambiguous/stale cells explicitly rather than raising."""
    spec = load_addendum_spec()
    current_fp, _ = compute_evaluator_fingerprint()
    cross_fp, _ = compute_cross_condition_fingerprint()

    report: dict[str, Any] = {
        "classification": CLASSIFICATION,
        "current_evaluator_fingerprint": current_fp,
        "cross_condition_analysis_fingerprint": cross_fp,
        "hypotheses": {},
    }

    for hypothesis in hypotheses:
        pairs = derive_fixed_pairs(hypothesis, matrix_path, spec)
        pair_reports = []
        n_eligible = 0
        for p in pairs:
            entry: dict[str, Any] = {
                "pair_id": p.pair_id,
                "dataset": p.dataset,
                "seed": p.seed,
                "condition_a_run_id": p.condition_a_run_id,
                "condition_b_run_id": p.condition_b_run_id,
                "condition_a_label": p.condition_a_label,
                "condition_b_label": p.condition_b_label,
                "directionality": p.directionality,
            }
            id_a = _resolve_canonical_evaluation_identity(p.condition_a_run_id, current_fp)
            id_b = _resolve_canonical_evaluation_identity(p.condition_b_run_id, current_fp)
            entry["condition_a_status"] = id_a.get("evaluation_status")
            entry["condition_b_status"] = id_b.get("evaluation_status")
            eligible = (
                id_a.get("evaluation_status") == "eligible" and id_b.get("evaluation_status") == "eligible"
            )
            entry["eligible"] = eligible
            if eligible:
                n_eligible += 1
                entry["condition_a_evaluation_id"] = id_a["evaluation_id"]
                entry["condition_b_evaluation_id"] = id_b["evaluation_id"]
            pair_reports.append(entry)

        report["hypotheses"][hypothesis] = {
            "pairs": pair_reports,
            "n_pairs_required": len(pairs),
            "n_pairs_eligible": n_eligible,
            "complete": n_eligible == len(pairs),
        }

    return report


# ---------------------------------------------------------------------------
# Real analysis mode -- implemented per the frozen spec, but NEVER invoked
# anywhere in the Phase 2B.5C engineering task. Only exercised by this
# module's own tests against synthetic/temporary fixtures.
# ---------------------------------------------------------------------------


def _load_pair_correctness(
    run_id: str,
    current_fp: str,
    validation_evaluation_root: str | Path,
    n: int,
) -> dict[str, Any]:
    """Read-only: resolve run_id's canonical evaluation, verify its
    artifact manifest and fingerprint, and compute clean/TTA-correct
    boolean arrays plus the aligned labels/sample_indices needed for
    cross-pair alignment checks. Raises AddendumInputError on anything
    other than a clean, verified, current-fingerprint completion."""
    import json as _json

    from when_tta_hurts.evaluation_result_artifacts import verify_evaluation_artifact_manifest

    identity = _resolve_canonical_evaluation_identity(run_id, current_fp)
    status = identity.get("evaluation_status")
    if status != "eligible":
        raise AddendumInputError(
            f"Cell {run_id!r} is not eligible for the cross-condition addendum (status={status!r})."
        )
    attempt = identity["evaluation_attempt"]
    attempt_dir = Path(validation_evaluation_root) / run_id / f"attempt_{attempt:03d}"

    manifest = _json.loads((attempt_dir / "artifact_manifest.json").read_text())
    verify_evaluation_artifact_manifest(attempt_dir, manifest)

    metadata = _json.loads((attempt_dir / "metadata.json").read_text())
    if metadata.get("evaluator_fingerprint") != current_fp:
        raise AddendumInputError(f"Cell {run_id!r} metadata fingerprint does not match current fingerprint.")

    predictions = dict(np.load(attempt_dir / "predictions.npz"))
    labels = predictions["labels"]
    sample_indices = predictions["sample_indices"]
    clean_probs = predictions["clean_probs"]
    view_probs = predictions["view_probs"][:n]
    agg_probs = view_probs.mean(axis=0)

    clean_correct = clean_probs.argmax(axis=-1) == labels
    tta_correct = agg_probs.argmax(axis=-1) == labels

    return {
        "run_id": run_id,
        "evaluation_id": identity["evaluation_id"],
        "evaluation_attempt": attempt,
        "checkpoint_hash": metadata["checkpoint_hash"],
        "labels": labels,
        "sample_indices": sample_indices,
        "clean_correct": clean_correct,
        "tta_correct": tta_correct,
        "artifact_manifest_sha256s": {a["path"]: a["sha256"] for a in manifest["artifacts"]},
    }


def compute_pair_did(
    pair: FixedPair,
    current_fp: str,
    analysis_fingerprint: str,
    n: int = 50,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    validation_evaluation_root: str | Path = "artifacts/validation_evaluation",
) -> dict[str, Any]:
    """Compute one fixed pair's DiD point estimate and bootstrap CI. Hard-
    fails on label mismatch, sample-index mismatch, or ineligible/stale/
    ambiguous inputs for either condition. Never invoked outside this
    module's own tests during Phase 2B.5C."""
    a = _load_pair_correctness(pair.condition_a_run_id, current_fp, validation_evaluation_root, n)
    b = _load_pair_correctness(pair.condition_b_run_id, current_fp, validation_evaluation_root, n)

    if a["labels"].shape != b["labels"].shape or not np.array_equal(a["labels"], b["labels"]):
        raise AddendumInputError(
            f"Pair {pair.pair_id!r}: label mismatch between {pair.condition_a_run_id!r} and "
            f"{pair.condition_b_run_id!r} -- refusing to compute a DiD on misaligned test images."
        )
    if a["sample_indices"].shape != b["sample_indices"].shape or not np.array_equal(
        a["sample_indices"], b["sample_indices"]
    ):
        raise AddendumInputError(
            f"Pair {pair.pair_id!r}: sample-index mismatch between {pair.condition_a_run_id!r} and "
            f"{pair.condition_b_run_id!r} -- refusing to compute a DiD on misaligned test images."
        )

    seed = derive_bootstrap_seed(pair.hypothesis, pair.pair_id, analysis_fingerprint)
    bootstrap = did_bootstrap_ci(
        a["clean_correct"],
        a["tta_correct"],
        b["clean_correct"],
        b["tta_correct"],
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )

    return {
        "pair_id": pair.pair_id,
        "hypothesis": pair.hypothesis,
        "dataset": pair.dataset,
        "seed": pair.seed,
        "directionality": pair.directionality,
        "condition_a": {
            "run_id": a["run_id"],
            "label": pair.condition_a_label,
            "evaluation_id": a["evaluation_id"],
            "evaluation_attempt": a["evaluation_attempt"],
            "checkpoint_hash": a["checkpoint_hash"],
            "artifact_manifest_sha256s": a["artifact_manifest_sha256s"],
        },
        "condition_b": {
            "run_id": b["run_id"],
            "label": pair.condition_b_label,
            "evaluation_id": b["evaluation_id"],
            "evaluation_attempt": b["evaluation_attempt"],
            "checkpoint_hash": b["checkpoint_hash"],
            "artifact_manifest_sha256s": b["artifact_manifest_sha256s"],
        },
        "bootstrap": bootstrap,
        "n_samples": int(a["labels"].shape[0]),
    }


def compute_hypothesis_did(
    hypothesis: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    validation_evaluation_root: str | Path = "artifacts/validation_evaluation",
) -> dict[str, Any]:
    """Compute every eligible fixed pair's DiD for `hypothesis`. Requires
    ALL expected pairs to be eligible; raises AddendumInputError on any
    missing/ambiguous/stale/mismatched pair rather than silently proceeding
    with a partial result. Never pools pairs into a single p-value or
    model-population verdict -- returns one independent result per pair.
    Never invoked outside this module's own tests during Phase 2B.5C."""
    spec = load_addendum_spec()
    current_fp, _ = compute_evaluator_fingerprint()
    analysis_fp, _ = compute_cross_condition_fingerprint()

    pairs = derive_fixed_pairs(hypothesis, matrix_path, spec)
    bootstrap_spec = spec["bootstrap"]

    per_pair_results: dict[str, Any] = {}
    for p in pairs:
        per_pair_results[p.pair_id] = compute_pair_did(
            p,
            current_fp,
            analysis_fp,
            n=spec["endpoint"]["tta_view_count"],
            n_resamples=bootstrap_spec["n_resamples"],
            ci_level=bootstrap_spec["ci_level"],
            validation_evaluation_root=validation_evaluation_root,
        )

    return {
        "classification": CLASSIFICATION,
        "hypothesis": hypothesis,
        "cross_condition_analysis_fingerprint": analysis_fp,
        "current_evaluator_fingerprint": current_fp,
        "pairs": [p.pair_id for p in pairs],
        "per_pair_results": per_pair_results,
        "status": "completed",
        "test_split_accessed": False,
    }
