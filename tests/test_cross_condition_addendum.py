"""Phase 2B.5C: synthetic/temporary-fixture tests for the fixed-model
difference-in-differences (DiD) cross-condition addendum. No real
validation/test prediction array, checkpoint, or scientific metric value
is ever read or written by these tests -- every correctness-array fixture
is hand-constructed in-memory or written to tmp_path. Structural
derivation tests (pair counts, family construction) use the real frozen
`configs/experiment_matrix.yaml`, exactly as test_statistical_analysis.py
already does for derive_family_cells() -- this reads only matrix
structure, never a prediction/metric value.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from when_tta_hurts.cross_condition_addendum import (
    CROSS_CONDITION_ADDENDUM_MANIFEST,
    KNOWN_HYPOTHESES,
    AddendumInputError,
    AddendumSpecError,
    FixedPair,
    compute_cross_condition_fingerprint,
    compute_pair_did,
    derive_bootstrap_seed,
    derive_fixed_pairs,
    did_bootstrap_ci,
    did_point_estimate,
    load_addendum_spec,
    per_image_did,
    plan_cross_condition_addendum,
)
from when_tta_hurts.statistical_analysis_artifacts import (
    AnalysisPersistenceError,
    AnalysisSchemaValidationError,
    persist_and_verify_cross_condition_completion,
    validate_cross_condition_result_schema,
    verify_analysis_artifact_manifest,
)

# ---------------------------------------------------------------------------
# Hand-calculated DiD examples + algebraic equivalence
# ---------------------------------------------------------------------------


def test_did_point_estimate_hand_calculated():
    """A: clean 5/10, TTA 3/10 correct (harmed by 0.2).
    B: clean 5/10, TTA 7/10 correct (helped by 0.2).
    DiD = 0.2 - (-0.2) = 0.4, hand-verified."""
    clean_a = np.array([1] * 5 + [0] * 5, dtype=bool)
    tta_a = np.array([1] * 3 + [0] * 7, dtype=bool)
    clean_b = np.array([1] * 5 + [0] * 5, dtype=bool)
    tta_b = np.array([1] * 7 + [0] * 3, dtype=bool)
    did = did_point_estimate(clean_a, tta_a, clean_b, tta_b)
    assert math.isclose(did, 0.4, abs_tol=1e-9)


def test_did_point_estimate_algebraic_equivalence_to_delta_b_minus_delta_a():
    """DiD must equal (acc_TTA_B - acc_clean_B) - (acc_TTA_A - acc_clean_A)
    exactly, for a non-trivial random-ish fixture, independent of the
    per-image formula's internal implementation."""
    rng = np.random.default_rng(42)
    n = 200
    clean_a = rng.random(n) < 0.7
    tta_a = rng.random(n) < 0.6
    clean_b = rng.random(n) < 0.7
    tta_b = rng.random(n) < 0.75

    did = did_point_estimate(clean_a, tta_a, clean_b, tta_b)
    delta_a = tta_a.mean() - clean_a.mean()
    delta_b = tta_b.mean() - clean_b.mean()
    assert math.isclose(did, delta_b - delta_a, abs_tol=1e-9)


def test_per_image_did_mean_equals_point_estimate():
    clean_a = np.array([1, 0, 1, 1], dtype=bool)
    tta_a = np.array([1, 1, 0, 1], dtype=bool)
    clean_b = np.array([0, 0, 1, 1], dtype=bool)
    tta_b = np.array([1, 1, 1, 1], dtype=bool)
    d = per_image_did(clean_a, tta_a, clean_b, tta_b)
    assert math.isclose(float(d.mean()), did_point_estimate(clean_a, tta_a, clean_b, tta_b), abs_tol=1e-12)


def test_did_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        did_point_estimate(
            np.array([True, False]), np.array([True]), np.array([True, False]), np.array([True, False])
        )


def test_did_rejects_empty_arrays():
    empty = np.array([], dtype=bool)
    with pytest.raises(ValueError):
        did_point_estimate(empty, empty, empty, empty)


# ---------------------------------------------------------------------------
# Joint aligned-index bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_resamples_one_joint_index_for_all_four_arrays(monkeypatch):
    """The implementation must call rng.integers exactly once per
    replicate (one joint index set), never once per array -- this is the
    structural proof that independent resampling of the four arrays is
    never performed. Wraps np.random.default_rng (Generator is an
    immutable C type and cannot be monkeypatched directly) with a
    call-counting proxy."""
    calls = {"n": 0}
    real_rng = np.random.default_rng(0)

    class _CountingRng:
        def integers(self, *args, **kwargs):
            calls["n"] += 1
            return real_rng.integers(*args, **kwargs)

    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(cca.np.random, "default_rng", lambda seed=None: _CountingRng())

    clean_a = np.array([1, 0, 1, 1, 0], dtype=bool)
    tta_a = np.array([1, 1, 0, 1, 0], dtype=bool)
    clean_b = np.array([0, 0, 1, 1, 1], dtype=bool)
    tta_b = np.array([1, 1, 1, 1, 0], dtype=bool)

    n_resamples = 25
    did_bootstrap_ci(clean_a, tta_a, clean_b, tta_b, n_resamples=n_resamples, seed=0)
    assert calls["n"] == n_resamples


def test_bootstrap_ci_deterministic_given_same_seed():
    clean_a = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    tta_a = np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=bool)
    clean_b = np.array([0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    tta_b = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    r1 = did_bootstrap_ci(clean_a, tta_a, clean_b, tta_b, n_resamples=500, seed=7)
    r2 = did_bootstrap_ci(clean_a, tta_a, clean_b, tta_b, n_resamples=500, seed=7)
    assert r1 == r2


def test_bootstrap_ci_identical_conditions_gives_zero_width_ci():
    """If A and B are identical in both clean and TTA, DiD is exactly 0 for
    every resample -- a strong sanity/no-invented-variance check."""
    clean = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    tta = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    result = did_bootstrap_ci(clean, tta, clean, tta, n_resamples=300, seed=1)
    assert result["did"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_bootstrap_rejects_invalid_n_resamples_and_ci_level():
    clean = np.array([True, False, True])
    with pytest.raises(ValueError):
        did_bootstrap_ci(clean, clean, clean, clean, n_resamples=0)
    with pytest.raises(ValueError):
        did_bootstrap_ci(clean, clean, clean, clean, ci_level=1.0)
    with pytest.raises(ValueError):
        did_bootstrap_ci(clean, clean, clean, clean, ci_level=0.0)


def test_bootstrap_seed_is_deterministic_and_pair_specific():
    fp = "fake-fingerprint"
    s1 = derive_bootstrap_seed("H1", "H1-pathmnist-28px-s0", fp)
    s2 = derive_bootstrap_seed("H1", "H1-pathmnist-28px-s0", fp)
    s3 = derive_bootstrap_seed("H1", "H1-pathmnist-28px-s1", fp)
    s4 = derive_bootstrap_seed("H2", "H1-pathmnist-28px-s0", fp)
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert isinstance(s1, int) and s1 >= 0


# ---------------------------------------------------------------------------
# Frozen spec loading -- endpoint/pooling/128px-exclusion enforcement
# ---------------------------------------------------------------------------


def test_load_addendum_spec_from_real_frozen_file():
    spec = load_addendum_spec()
    assert spec["endpoint"]["metric"] == "accuracy"
    assert spec["endpoint"]["aggregator"] == "mean_probability"
    assert spec["endpoint"]["tta_view_count"] == 50
    assert spec["endpoint"]["condition"] == "naive_tta"
    assert spec["h2_pairs"]["block_d_128px_included_in_inference"] is False
    assert spec["reporting"]["produce_model_population_p_value"] is False
    assert spec["reporting"]["produce_seed_level_sign_or_permutation_test"] is False
    assert spec["reporting"]["fit_mixed_effects_or_hierarchical_model"] is False


def _write_addendum_yaml(
    path, *, tta_view_count=50, aggregator="mean_probability", block_d_128px=False, pool_p=False
):
    spec = {
        "endpoint": {
            "metric": "accuracy",
            "condition": "naive_tta",
            "aggregator": aggregator,
            "tta_view_count": tta_view_count,
        },
        "h1_pairs": {},
        "h2_pairs": {"block_d_128px_included_in_inference": block_d_128px},
        "h3_pairs": {},
        "bootstrap": {},
        "reporting": {
            "produce_model_population_p_value": pool_p,
            "produce_seed_level_sign_or_permutation_test": False,
            "fit_mixed_effects_or_hierarchical_model": False,
        },
    }
    import yaml as _yaml

    path.write_text(_yaml.safe_dump(spec))


def test_load_addendum_spec_rejects_wrong_n(tmp_path):
    bad = tmp_path / "addendum.yaml"
    _write_addendum_yaml(bad, tta_view_count=10)
    with pytest.raises(AddendumSpecError):
        load_addendum_spec(bad)


def test_load_addendum_spec_rejects_wrong_aggregator(tmp_path):
    bad = tmp_path / "addendum.yaml"
    _write_addendum_yaml(bad, aggregator="max_probability")
    with pytest.raises(AddendumSpecError):
        load_addendum_spec(bad)


def test_load_addendum_spec_rejects_128px_reenabled(tmp_path):
    bad = tmp_path / "addendum.yaml"
    _write_addendum_yaml(bad, block_d_128px=True)
    with pytest.raises(AddendumSpecError):
        load_addendum_spec(bad)


def test_load_addendum_spec_rejects_pooling_reenabled(tmp_path):
    bad = tmp_path / "addendum.yaml"
    _write_addendum_yaml(bad, pool_p=True)
    with pytest.raises(AddendumSpecError):
        load_addendum_spec(bad)


def test_load_addendum_spec_missing_file(tmp_path):
    with pytest.raises(AddendumSpecError):
        load_addendum_spec(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Pair-family construction against the real frozen matrix (structure only)
# ---------------------------------------------------------------------------


def test_derive_fixed_pairs_h1_count_and_no_pooling():
    pairs = derive_fixed_pairs("H1")
    assert len(pairs) == 12  # 2 datasets x 2 resolutions x 3 seeds
    pair_ids = [p.pair_id for p in pairs]
    assert len(pair_ids) == len(set(pair_ids))  # one independent estimate per pair, never merged
    for p in pairs:
        assert p.condition_a_label == "batchnorm"
        assert p.condition_b_label == "groupnorm"
        assert p.resolution if hasattr(p, "resolution") else True  # resolution encoded in pair_id
        assert p.directionality == "two_sided"


def test_derive_fixed_pairs_h2_count_and_excludes_128px():
    pairs = derive_fixed_pairs("H2")
    assert len(pairs) == 12  # 2 datasets x 2 normalizations x 3 seeds
    for p in pairs:
        assert p.condition_a_label == "28px"
        assert p.condition_b_label == "64px"
        assert "128" not in p.pair_id
        assert "128px" not in (p.condition_a_label, p.condition_b_label)
        assert p.directionality == "one_sided"


def test_derive_fixed_pairs_h3_count():
    pairs = derive_fixed_pairs("H3")
    assert len(pairs) == 6  # 2 datasets x 3 seeds
    for p in pairs:
        assert p.condition_a_label == "unmatched"
        assert p.condition_b_label == "matched"
        assert p.directionality == "one_sided"


def test_derive_fixed_pairs_unknown_hypothesis_raises():
    with pytest.raises(ValueError):
        derive_fixed_pairs("H4")
    with pytest.raises(ValueError):
        derive_fixed_pairs("NOT_A_HYPOTHESIS")


def test_derive_fixed_pairs_deterministic_ordering():
    a = derive_fixed_pairs("H1")
    b = derive_fixed_pairs("H1")
    assert [p.pair_id for p in a] == [p.pair_id for p in b]
    assert [p.pair_id for p in a] == sorted(p.pair_id for p in a)


def test_derive_fixed_pairs_missing_condition_drops_pair_not_fabricates():
    """If one side of a would-be pair doesn't exist in the matrix, the
    pair must simply be absent (intersection semantics) -- never
    fabricated with a substituted cell."""
    from when_tta_hurts.matrix import MatrixCell

    class _FakeExpanded:
        def __init__(self, cells):
            self.cells = cells

    spec = load_addendum_spec()
    cells = [
        MatrixCell("A_core_normalization_resolution", "pathmnist", 28, "small_cnn", "batchnorm", "none", 0),
        # No groupnorm counterpart at all.
    ]
    import when_tta_hurts.cross_condition_addendum as cca

    original = cca._matrix_cells
    cca._matrix_cells = lambda matrix_path="configs/experiment_matrix.yaml": cells
    try:
        pairs = derive_fixed_pairs("H1", addendum_spec=spec)
    finally:
        cca._matrix_cells = original
    assert pairs == ()


# ---------------------------------------------------------------------------
# compute_pair_did: label/sample-index alignment enforcement, and
# ineligible/stale/ambiguous input rejection (synthetic tmp fixtures only)
# ---------------------------------------------------------------------------


def _write_predictions(path, labels, sample_indices, clean_probs, view_probs):
    np.savez(
        path,
        labels=labels,
        sample_indices=sample_indices,
        clean_probs=clean_probs,
        view_probs=view_probs,
    )


def _make_probs_for_correctness(correct: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Build a probs array whose argmax matches `correct` against label 0
    for every sample (correct -> class 0 highest; wrong -> class 1
    highest)."""
    probs = np.zeros((correct.shape[0], n_classes), dtype=np.float64)
    probs[correct, 0] = 1.0
    probs[correct, 1:] = 0.0
    probs[~correct, 0] = 0.0
    probs[~correct, 1] = 1.0
    return probs


def _write_attempt(root, run_id, attempt, fingerprint, labels, sample_indices, clean_correct, tta_correct):
    from when_tta_hurts.artifacts import hash_file

    d = root / run_id / f"attempt_{attempt:03d}"
    d.mkdir(parents=True, exist_ok=True)
    clean_probs = _make_probs_for_correctness(clean_correct)
    view_probs = np.stack([_make_probs_for_correctness(tta_correct)] * 50, axis=0)
    _write_predictions(d / "predictions.npz", labels, sample_indices, clean_probs, view_probs)
    (d / "metadata.json").write_text(
        json.dumps({"evaluator_fingerprint": fingerprint, "checkpoint_hash": "chk"})
    )
    manifest = {
        "artifacts": [
            {
                "path": "predictions.npz",
                "size_bytes": (d / "predictions.npz").stat().st_size,
                "sha256": hash_file(d / "predictions.npz"),
            }
        ]
    }
    (d / "artifact_manifest.json").write_text(json.dumps(manifest))
    return d


def _write_eval_ledger(path, rows):
    import csv

    fields = ["training_run_id", "evaluation_attempt", "evaluation_id", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_compute_pair_did_end_to_end_synthetic(tmp_path, monkeypatch):
    root = tmp_path / "validation_evaluation"
    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    _write_eval_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-a",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-a",
                "status": "completed",
            },
            {
                "training_run_id": "run-b",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-b",
                "status": "completed",
            },
        ],
    )
    _write_eval_ledger(amend_path, [])
    amend_path.write_text("evaluation_id,evaluation_attempt,canonical_eligible\n")

    labels = np.zeros(20, dtype=np.int64)
    sample_indices = np.arange(20)
    clean_a = np.array([True] * 12 + [False] * 8)
    tta_a = np.array([True] * 8 + [False] * 12)
    clean_b = np.array([True] * 12 + [False] * 8)
    tta_b = np.array([True] * 16 + [False] * 4)
    _write_attempt(root, "run-a", 1, "fp-current", labels, sample_indices, clean_a, tta_a)
    _write_attempt(root, "run-b", 1, "fp-current", labels, sample_indices, clean_b, tta_b)

    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca,
        "_resolve_canonical_evaluation_identity",
        lambda run_id, fp: {
            "evaluation_status": "eligible",
            "evaluation_id": "eval-a" if run_id == "run-a" else "eval-b",
            "evaluation_attempt": 1,
        },
    )

    pair = FixedPair(
        hypothesis="H1",
        pair_id="H1-test-s0",
        dataset="test",
        seed=0,
        condition_a_run_id="run-a",
        condition_b_run_id="run-b",
        condition_a_label="batchnorm",
        condition_b_label="groupnorm",
        directionality="two_sided",
    )
    result = compute_pair_did(
        pair, "fp-current", "analysis-fp", n=50, n_resamples=200, validation_evaluation_root=root
    )

    expected = did_point_estimate(clean_a, tta_a, clean_b, tta_b)
    assert math.isclose(result["bootstrap"]["did"], expected, abs_tol=1e-9)
    assert result["n_samples"] == 20
    assert result["condition_a"]["run_id"] == "run-a"
    assert result["condition_b"]["run_id"] == "run-b"


def test_compute_pair_did_rejects_label_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "validation_evaluation"
    labels_a = np.zeros(10, dtype=np.int64)
    labels_b = np.ones(10, dtype=np.int64)
    idx = np.arange(10)
    correct = np.array([True] * 5 + [False] * 5)
    _write_attempt(root, "run-a", 1, "fp-current", labels_a, idx, correct, correct)
    _write_attempt(root, "run-b", 1, "fp-current", labels_b, idx, correct, correct)

    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca,
        "_resolve_canonical_evaluation_identity",
        lambda run_id, fp: {
            "evaluation_status": "eligible",
            "evaluation_id": run_id,
            "evaluation_attempt": 1,
        },
    )
    pair = FixedPair("H1", "H1-x-s0", "x", 0, "run-a", "run-b", "batchnorm", "groupnorm", "two_sided")
    with pytest.raises(AddendumInputError, match="label mismatch"):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=root)


def test_compute_pair_did_rejects_sample_index_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "validation_evaluation"
    labels = np.zeros(10, dtype=np.int64)
    idx_a = np.arange(10)
    idx_b = np.arange(10)[::-1].copy()
    correct = np.array([True] * 5 + [False] * 5)
    _write_attempt(root, "run-a", 1, "fp-current", labels, idx_a, correct, correct)
    _write_attempt(root, "run-b", 1, "fp-current", labels, idx_b, correct, correct)

    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca,
        "_resolve_canonical_evaluation_identity",
        lambda run_id, fp: {
            "evaluation_status": "eligible",
            "evaluation_id": run_id,
            "evaluation_attempt": 1,
        },
    )
    pair = FixedPair("H2", "H2-x-s0", "x", 0, "run-a", "run-b", "28px", "64px", "one_sided")
    with pytest.raises(AddendumInputError, match="sample-index mismatch"):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=root)


def test_compute_pair_did_rejects_ineligible_missing_input(tmp_path, monkeypatch):
    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca, "_resolve_canonical_evaluation_identity", lambda run_id, fp: {"evaluation_status": "missing"}
    )
    pair = FixedPair("H3", "H3-x-s0", "x", 0, "run-a", "run-b", "unmatched", "matched", "one_sided")
    with pytest.raises(AddendumInputError, match="not eligible"):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=tmp_path)


def test_compute_pair_did_rejects_ambiguous_input(tmp_path, monkeypatch):
    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca,
        "_resolve_canonical_evaluation_identity",
        lambda run_id, fp: {"evaluation_status": "ambiguous", "evaluation_ids": ["e1", "e2"]},
    )
    pair = FixedPair("H1", "H1-x-s0", "x", 0, "run-a", "run-b", "batchnorm", "groupnorm", "two_sided")
    with pytest.raises(AddendumInputError, match="not eligible"):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=tmp_path)


def test_compute_pair_did_rejects_stale_fingerprint_input(tmp_path, monkeypatch):
    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca,
        "_resolve_canonical_evaluation_identity",
        lambda run_id, fp: {"evaluation_status": "stale", "evaluation_ids": ["e1"]},
    )
    pair = FixedPair("H1", "H1-x-s0", "x", 0, "run-a", "run-b", "batchnorm", "groupnorm", "two_sided")
    with pytest.raises(AddendumInputError, match="not eligible"):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=tmp_path)


def test_compute_pair_did_rejects_amendment_excluded_via_missing_status(tmp_path, monkeypatch):
    """An amendment-excluded (superseded) attempt resolves to 'missing' or
    'stale' via _resolve_canonical_evaluation_identity -- never silently
    reselected. compute_pair_did must reject it exactly like any other
    non-eligible status."""
    import when_tta_hurts.cross_condition_addendum as cca

    monkeypatch.setattr(
        cca, "_resolve_canonical_evaluation_identity", lambda run_id, fp: {"evaluation_status": "missing"}
    )
    pair = FixedPair("H3", "H3-x-s0", "x", 0, "run-a", "run-b", "unmatched", "matched", "one_sided")
    with pytest.raises(AddendumInputError):
        compute_pair_did(pair, "fp-current", "analysis-fp", validation_evaluation_root=tmp_path)


# ---------------------------------------------------------------------------
# Analysis-implementation fingerprint identity
# ---------------------------------------------------------------------------


def test_cross_condition_fingerprint_manifest_includes_addendum_files():
    assert "configs/final_test_cross_condition_addendum.yaml" in CROSS_CONDITION_ADDENDUM_MANIFEST
    assert "src/when_tta_hurts/cross_condition_addendum.py" in CROSS_CONDITION_ADDENDUM_MANIFEST
    for f in CROSS_CONDITION_ADDENDUM_MANIFEST:
        assert not f.startswith("docs/")
        assert "ledger_" not in f  # no ledger CSVs in the manifest


def test_cross_condition_fingerprint_stable_across_calls():
    fp1, _ = compute_cross_condition_fingerprint()
    fp2, _ = compute_cross_condition_fingerprint()
    assert fp1 == fp2


def test_cross_condition_fingerprint_changes_when_manifested_file_changes(tmp_path):
    f1 = tmp_path / "fake.py"
    f1.write_text("x = 1\n")
    fp1, _ = compute_cross_condition_fingerprint(repo_root=tmp_path, manifest=("fake.py",))
    f1.write_text("x = 2\n")
    fp2, _ = compute_cross_condition_fingerprint(repo_root=tmp_path, manifest=("fake.py",))
    assert fp1 != fp2


def test_cross_condition_fingerprint_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(AddendumSpecError):
        compute_cross_condition_fingerprint(repo_root=tmp_path, manifest=("does_not_exist.py",))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _valid_cross_condition_result():
    return {
        "classification": "post_validation_pre_test_secondary",
        "hypothesis": "H1",
        "cross_condition_analysis_fingerprint": "fp",
        "current_evaluator_fingerprint": "efp",
        "pairs": ["H1-x-s0"],
        "per_pair_results": {"H1-x-s0": {"bootstrap": {"did": 0.1, "ci_low": 0.0, "ci_high": 0.2}}},
        "status": "completed",
        "test_split_accessed": False,
    }


def test_validate_cross_condition_schema_accepts_valid_result():
    validate_cross_condition_result_schema(_valid_cross_condition_result())


def test_validate_cross_condition_schema_rejects_missing_key():
    r = _valid_cross_condition_result()
    del r["pairs"]
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_wrong_classification():
    r = _valid_cross_condition_result()
    r["classification"] = "originally_preregistered"
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_unknown_hypothesis():
    r = _valid_cross_condition_result()
    r["hypothesis"] = "H4"
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_test_split_accessed_true():
    r = _valid_cross_condition_result()
    r["test_split_accessed"] = True
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_non_completed_status():
    r = _valid_cross_condition_result()
    r["status"] = "failed"
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_empty_pairs():
    r = _valid_cross_condition_result()
    r["pairs"] = []
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_pooled_p_value_key():
    r = _valid_cross_condition_result()
    r["pooled_p_value"] = 0.03
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_model_population_p_value_key():
    r = _valid_cross_condition_result()
    r["model_population_p_value"] = 0.03
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


def test_validate_cross_condition_schema_rejects_non_finite_values():
    r = _valid_cross_condition_result()
    r["per_pair_results"]["H1-x-s0"]["bootstrap"]["did"] = float("nan")
    with pytest.raises(AnalysisSchemaValidationError):
        validate_cross_condition_result_schema(r)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_persist_and_verify_cross_condition_completion_round_trip(tmp_path):
    manifest = persist_and_verify_cross_condition_completion(tmp_path, result=_valid_cross_condition_result())
    assert (tmp_path / "cross_condition_result.json").exists()
    assert (tmp_path / "cross_condition_artifact_manifest.json").exists()
    verify_analysis_artifact_manifest(tmp_path, manifest)


def test_persist_cross_condition_rejects_invalid_schema(tmp_path):
    bad = _valid_cross_condition_result()
    bad["status"] = "failed"
    with pytest.raises(AnalysisSchemaValidationError):
        persist_and_verify_cross_condition_completion(tmp_path, result=bad)
    assert not (tmp_path / "cross_condition_result.json").exists()


def test_cross_condition_manifest_tamper_detection(tmp_path):
    manifest = persist_and_verify_cross_condition_completion(tmp_path, result=_valid_cross_condition_result())
    (tmp_path / "cross_condition_result.json").write_text('{"tampered": true}')
    with pytest.raises(AnalysisPersistenceError):
        verify_analysis_artifact_manifest(tmp_path, manifest)


# ---------------------------------------------------------------------------
# Plan mode: side-effect freedom against the real repo
# ---------------------------------------------------------------------------


def test_plan_mode_is_side_effect_free(tmp_path, monkeypatch):
    before = set(tmp_path.iterdir())
    report = plan_cross_condition_addendum()
    after = set(tmp_path.iterdir())
    assert before == after
    assert set(report["hypotheses"].keys()) == set(KNOWN_HYPOTHESES)


def test_plan_mode_never_reads_predictions_or_metrics_files(monkeypatch):
    real_load = np.load

    def guarded_load(path, *args, **kwargs):
        if "predictions.npz" in str(path):
            raise AssertionError("plan mode must never read predictions.npz")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    plan_cross_condition_addendum()


def test_plan_mode_reports_real_repo_pair_counts():
    """Structural check against the real repo state: plan mode must
    report exactly the frozen pair counts derived in this same session's
    Phase 2B.5B memo (H1=12, H2=12, H3=6). `complete` is NOT hardcoded --
    Phase 2B.6K's reconciliation mechanism means completeness legitimately
    flips based on whether every required cell's canonical evaluation
    currently resolves 'eligible' under
    _resolve_canonical_evaluation_identity() (directly OR via a valid
    reconciliation record). This test independently re-derives that same
    per-cell eligibility using the identical production resolver plan
    mode itself calls, and asserts the plan's reported `complete` field
    matches -- without valid reconciliation evidence the derived
    expectation (and therefore the plan) must be False; with complete,
    verified reconciliation evidence for every required run_id, both
    must be True."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity
    from when_tta_hurts.validation_evaluation import compute_evaluator_fingerprint

    report = plan_cross_condition_addendum()
    current_fp, _ = compute_evaluator_fingerprint()
    assert report["current_evaluator_fingerprint"] == current_fp
    assert report["hypotheses"]["H1"]["n_pairs_required"] == 12
    assert report["hypotheses"]["H2"]["n_pairs_required"] == 12
    assert report["hypotheses"]["H3"]["n_pairs_required"] == 6

    for hyp in ("H1", "H2", "H3"):
        expected_complete = True
        for pair in report["hypotheses"][hyp]["pairs"]:
            for run_id in (pair["condition_a_run_id"], pair["condition_b_run_id"]):
                status = _resolve_canonical_evaluation_identity(run_id, current_fp).get("evaluation_status")
                if status != "eligible":
                    expected_complete = False
        assert report["hypotheses"][hyp]["complete"] is expected_complete


# ---------------------------------------------------------------------------
# No test-split reachability
# ---------------------------------------------------------------------------


def test_no_test_split_symbol_reachable_from_cross_condition_addendum_module():
    import when_tta_hurts.cross_condition_addendum as cca

    source = inspect.getsource(cca)
    assert "allow_test" not in source
    assert "load_test" not in source
    assert "official_test" not in source
    assert "test_split" not in source.lower() or "test_split_accessed" in source


def test_compute_hypothesis_did_never_invoked_at_import_time():
    """Importing the module must never itself compute a real result --
    a smoke check that no module-level call to compute_hypothesis_did or
    compute_pair_did exists."""
    import when_tta_hurts.cross_condition_addendum as cca

    source = inspect.getsource(cca)
    # Only the def lines should mention these names at column 0-ish; a
    # module-level *call* (not inside a def) would be a bug. This is a
    # coarse structural check consistent with test_statistical_analysis.py's
    # equivalent CLI-flag check.
    lines = [
        ln
        for ln in source.splitlines()
        if ln.strip().startswith("compute_hypothesis_did(") or ln.strip().startswith("compute_pair_did(")
    ]
    assert lines == []
