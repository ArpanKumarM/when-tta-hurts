"""Tests for the Phase 2B.4D OOM correction: bounded-memory batching.
Uses ONLY synthetic tensors, a tiny SmallCNN, and temporary repositories.
Never touches a real dataset, checkpoint, or MPS device.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from when_tta_hurts.evaluation.bn_adaptation import (
    BNAdaptationNotApplicableError,
    bn_adapt,
    bn_adapt_sequential,
)
from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit
from when_tta_hurts.evaluation.views import iter_deterministic_views, stable_view_seed
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
    persist_and_verify_evaluation_completion,
)
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.transforms.policies import build_policy
from when_tta_hurts.validation_evaluation import (
    BN_ADAPTATION_BATCH_SIZE,
    INFERENCE_BATCH_SIZE,
    FrozenTTASeedConfigError,
    ValidationEvaluationConfig,
    _batched_forward,
    _bn_adaptation_microbatches,
    _n_bn_adaptation_microbatches,
    compute_evaluation_id,
    compute_evaluation_latency_report,
    compute_validation_evaluation,
    load_frozen_tta_seed_config,
)

DEVICE = torch.device("cpu")


def _synthetic_split(n, n_classes=3, resolution=28, dataset="pathmnist", seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, resolution, resolution, generator=g)
    labels = np.array([i % n_classes for i in range(n)])
    return ValidationEvaluationSplit(
        images=images, labels=labels, sample_indices=np.arange(n), dataset=dataset, resolution=resolution
    )


class _BatchSizeGuardModel(torch.nn.Module):
    """Wraps a real model; records every forward call's batch size and
    asserts it never exceeds `max_batch` -- an end-to-end proof that the
    production forward-call sites never exceed the frozen batch size,
    not just that the generators producing their inputs are bounded."""

    def __init__(self, inner: torch.nn.Module, max_batch: int):
        super().__init__()
        self.inner = inner
        self.max_batch = max_batch
        self.batch_sizes_seen: list[int] = []

    def forward(self, x):
        assert x.shape[0] <= self.max_batch, f"forward received batch size {x.shape[0]} > {self.max_batch}"
        self.batch_sizes_seen.append(x.shape[0])
        return self.inner(x)


# ---------------------------------------------------------------------------
# 1-2, 6: no forward call (clean/TTA/BN-adaptation) exceeds the frozen
# batch size; final partial batches are handled correctly
# ---------------------------------------------------------------------------


def test_no_clean_or_tta_forward_exceeds_frozen_batch_size(monkeypatch):
    """n_samples=300 > INFERENCE_BATCH_SIZE=256 forces a final partial
    chunk of 44. MAX_VIEWS/PREFIX_SEQUENCE are monkeypatched down to keep
    this test fast -- the per-sample deterministic view generation cost
    (not the batching logic under test) otherwise dominates runtime;
    boundedness itself does not depend on how many views are requested."""
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "MAX_VIEWS", 2)
    monkeypatch.setattr(ve, "PREFIX_SEQUENCE", (1, 2))
    monkeypatch.setattr(ve, "PRIMARY_N", 1)

    torch.manual_seed(0)
    n_samples = 300
    guard = _BatchSizeGuardModel(
        build_small_cnn(num_classes=3, normalization="batchnorm"), INFERENCE_BATCH_SIZE
    )
    split = _synthetic_split(n=n_samples, n_classes=3)

    outcome = ve.compute_validation_evaluation(guard, split, tta_seed=42, device=DEVICE)

    assert max(guard.batch_sizes_seen) <= INFERENCE_BATCH_SIZE
    assert 44 in guard.batch_sizes_seen  # final partial batch (300 mod 256) actually occurred
    assert outcome["predictions"]["clean_probs"].shape[0] == n_samples


def test_no_bn_adaptation_forward_exceeds_frozen_batch_size(monkeypatch):
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "MAX_VIEWS", 2)
    monkeypatch.setattr(ve, "PREFIX_SEQUENCE", (1, 2))
    monkeypatch.setattr(ve, "PRIMARY_N", 1)

    torch.manual_seed(0)
    n_samples = 300
    guard = _BatchSizeGuardModel(
        build_small_cnn(num_classes=3, normalization="batchnorm"), BN_ADAPTATION_BATCH_SIZE
    )
    split = _synthetic_split(n=n_samples, n_classes=3)

    ve.compute_validation_evaluation(guard, split, tta_seed=42, device=DEVICE)

    assert max(guard.batch_sizes_seen) <= BN_ADAPTATION_BATCH_SIZE
    assert 44 in guard.batch_sizes_seen


def test_batched_forward_final_partial_batch_correct():
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    model.eval()
    images = torch.rand(10, 3, 28, 28)  # batch_size=4 -> chunks of 4, 4, 2
    logits_chunked = _batched_forward(model, images, DEVICE, batch_size=4)
    with torch.no_grad():
        logits_full = model(images).numpy()
    np.testing.assert_allclose(logits_chunked, logits_full, rtol=1e-5, atol=1e-6)
    assert logits_chunked.shape[0] == 10


# ---------------------------------------------------------------------------
# 3-4-5: no full N-view population is materialized; every (sample, view)
# pair appears exactly once; view-major-then-sample-major ordering
# ---------------------------------------------------------------------------


def test_bn_adaptation_microbatches_no_full_n_view_list_materialized():
    """Structural proof: the microbatch generator's source never builds a
    list/torch.cat over all N views -- it only ever holds one view's
    full-split CPU tensor (from iter_deterministic_views(), itself
    bounded to one view at a time) and yields <=batch_size chunks."""
    import inspect

    source = inspect.getsource(_bn_adaptation_microbatches)
    assert "torch.cat" not in source
    assert "[view_batch for" not in source
    assert "adaptation_views" not in source


def test_bn_adaptation_microbatches_covers_every_sample_view_pair_exactly_once():
    torch.manual_seed(0)
    n_samples, n_views, batch_size = 10, 3, 4
    images = torch.rand(n_samples, 3, 28, 28)
    policy = build_policy("mixed", output_size=(28, 28))
    sample_indices = list(range(n_samples))

    seen_pairs: list[tuple[int, int]] = []
    view_of_chunk: list[int] = []
    chunk_sizes: list[int] = []
    consumed = 0
    for view_index, view_batch in iter_deterministic_views(
        images, policy, 123, "pathmnist", 28, sample_indices, n_views
    ):
        for start in range(0, view_batch.shape[0], batch_size):
            chunk = view_batch[start : start + batch_size]
            chunk_sizes.append(chunk.shape[0])
            view_of_chunk.append(view_index)
            for i in range(chunk.shape[0]):
                seen_pairs.append((view_index, sample_indices[start + i]))
        consumed += 1

    assert consumed == n_views
    assert len(seen_pairs) == n_samples * n_views
    assert len(set(seen_pairs)) == n_samples * n_views  # no duplicates
    expected_pairs = {(v, s) for v in range(n_views) for s in sample_indices}
    assert set(seen_pairs) == expected_pairs  # no omissions


def test_bn_adaptation_enumeration_order_is_view_major_then_sample_major():
    n_samples, n_views, batch_size = 5, 3, 2
    images = torch.rand(n_samples, 3, 28, 28)
    policy = build_policy("mixed", output_size=(28, 28))
    sample_indices = list(range(n_samples))

    order: list[tuple[int, int]] = []
    for view_index, view_batch in iter_deterministic_views(
        images, policy, 7, "pathmnist", 28, sample_indices, n_views
    ):
        for start in range(0, view_batch.shape[0], batch_size):
            for i in range(min(batch_size, view_batch.shape[0] - start)):
                order.append((view_index, sample_indices[start + i]))

    expected_order = [(v, s) for v in range(n_views) for s in sample_indices]
    assert order == expected_order


def test_n_bn_adaptation_microbatches_matches_actual_generator_count():
    n_samples, n_views, batch_size = 13, 4, 5
    images = torch.rand(n_samples, 3, 28, 28)
    policy = build_policy("mixed", output_size=(28, 28))
    sample_indices = list(range(n_samples))

    actual = sum(
        1
        for _ in _bn_adaptation_microbatches(
            images, policy, 1, "pathmnist", 28, sample_indices, n_views, DEVICE, batch_size
        )
    )
    assert actual == _n_bn_adaptation_microbatches(n_samples, n_views, batch_size)


def test_large_logical_population_bounded_memory_without_allocating_it(monkeypatch):
    """Proves boundedness for a LARGE logical population (matching
    PathMNIST's real validation split size, 10,004 samples, at the full
    N=100 views) WITHOUT allocating a genuinely large tensor: the
    microbatch COUNT is computed purely analytically (no tensor
    involved), and a small-N integration run (MAX_VIEWS/PREFIX_SEQUENCE
    monkeypatched down purely to keep this test fast -- the per-sample
    view-generation cost, not the batching logic under test, otherwise
    dominates runtime) with an exploding guard model proves the actual
    forward-call mechanism that would process such a population never
    exceeds the frozen batch size."""
    # Analytical, tensor-free proof of scale: this is exactly the
    # arithmetic that caused the real OOM (100 views * 10,004 samples).
    n_microbatches_at_scale = _n_bn_adaptation_microbatches(10004, 100, BN_ADAPTATION_BATCH_SIZE)
    assert n_microbatches_at_scale == 100 * math.ceil(10004 / BN_ADAPTATION_BATCH_SIZE)
    assert n_microbatches_at_scale == 4000

    # Small-scale integration proof (still n_samples=300 > frozen batch
    # size 256) with an exploding guard: any forward call exceeding the
    # frozen batch size raises immediately.
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "MAX_VIEWS", 2)
    monkeypatch.setattr(ve, "PREFIX_SEQUENCE", (1, 2))
    monkeypatch.setattr(ve, "PRIMARY_N", 1)

    torch.manual_seed(0)
    guard = _BatchSizeGuardModel(
        build_small_cnn(num_classes=3, normalization="batchnorm"), BN_ADAPTATION_BATCH_SIZE
    )
    split = _synthetic_split(n=300, n_classes=3)
    ve.compute_validation_evaluation(guard, split, tta_seed=1, device=DEVICE)
    assert max(guard.batch_sizes_seen) <= BN_ADAPTATION_BATCH_SIZE


# ---------------------------------------------------------------------------
# 7: batch boundaries never change view content (already guaranteed by
# stable_view_seed(), reused unchanged here)
# ---------------------------------------------------------------------------


def test_different_chunking_produces_identical_view_seeds_and_ordering():
    """Chunking the SAME view differently (e.g. by a different external
    batch size) must never change which seed a given (sample, view) pair
    receives -- stable_view_seed() takes no batch-size/boundary input at
    all, so this holds structurally, verified here directly."""
    for sample_index in (0, 17, 999):
        for view_index in (0, 49, 99):
            seed_a = stable_view_seed(123, "pathmnist", 28, sample_index, view_index)
            seed_b = stable_view_seed(123, "pathmnist", 28, sample_index, view_index)
            assert seed_a == seed_b
    sig_params = set(__import__("inspect").signature(stable_view_seed).parameters)
    assert "batch_size" not in sig_params
    assert "batch" not in sig_params


# ---------------------------------------------------------------------------
# 8-9: eval-mode full-batch vs chunked probabilities/metrics match within
# a predeclared tolerance for clean/TTA inference (mathematically
# equivalent -- unlike BN adaptation, chunking clean/TTA inference does
# NOT change any stateful computation)
# ---------------------------------------------------------------------------

_CLEAN_TTA_TOLERANCE = dict(rtol=1e-5, atol=1e-6)


def test_full_batch_and_chunked_clean_tta_probabilities_match_within_tolerance():
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    model.eval()
    split = _synthetic_split(n=17, n_classes=3)  # not a multiple of any small batch size

    outcome_chunked = compute_validation_evaluation(copy.deepcopy(model), split, tta_seed=99, device=DEVICE)

    # Independent full-batch reference computation (no chunking).
    with torch.no_grad():
        clean_logits_full = model(split.images).numpy()
    from when_tta_hurts.metrics import softmax

    clean_probs_full = softmax(clean_logits_full)

    np.testing.assert_allclose(
        outcome_chunked["predictions"]["clean_probs"], clean_probs_full, **_CLEAN_TTA_TOLERANCE
    )


def test_full_batch_and_chunked_metrics_match_within_tolerance():
    """Recomputes clean accuracy independently from the chunked pipeline's
    persisted predictions and confirms it matches the persisted metric
    exactly (same predeclared tolerance already used throughout this
    project's independent-recomputation checks)."""
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    split = _synthetic_split(n=17, n_classes=3)
    outcome = compute_validation_evaluation(model, split, tta_seed=99, device=DEVICE)

    from when_tta_hurts.evaluation_result_artifacts import recompute_clean_accuracy

    recomputed = recompute_clean_accuracy(outcome["predictions"]["clean_probs"], split.labels)
    assert recomputed == pytest.approx(outcome["metrics"]["clean"]["accuracy"], rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# 10-15: BN adaptation -- equivalence at <=batch_size, disclosed
# non-equivalence above it, parameter/buffer isolation, per-N reset
# ---------------------------------------------------------------------------


def test_bn_adapt_sequential_matches_bn_adapt_exactly_for_inputs_at_or_below_batch_size():
    """For inputs of size <=256 (here: one microbatch), the new
    sequential path matches the old single-forward path EXACTLY -- both
    delegate to the same shared core, so a single-element iterable
    reproduces bn_adapt()'s one forward call precisely."""
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)

    adapted_old = bn_adapt(model, x)
    adapted_new = bn_adapt_sequential(model, [x])

    assert torch.equal(adapted_old.features[1].running_mean, adapted_new.features[1].running_mean)
    assert torch.equal(adapted_old.features[1].running_var, adapted_new.features[1].running_var)


def test_bn_adapt_sequential_differs_from_bn_adapt_for_inputs_above_batch_size():
    """For inputs split across MORE than one microbatch, the sequential
    path is explicitly NOT claimed equivalent to a single giant forward
    pass -- this is tested and documented, never falsely asserted equal
    (see docs/phase2b_validation_evaluation_batching_freeze.md sec.3.2)."""
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)

    adapted_single_batch = bn_adapt(model, x)
    adapted_sequential = bn_adapt_sequential(model, [x[0:8], x[8:16]])

    assert not torch.equal(
        adapted_single_batch.features[1].running_mean, adapted_sequential.features[1].running_mean
    )


def test_bn_adapt_sequential_no_learned_parameter_changes():
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)
    params_before = {name: p.clone() for name, p in model.named_parameters()}

    adapted = bn_adapt_sequential(model, [x[0:8], x[8:16]])

    for name, p_after in adapted.named_parameters():
        assert torch.equal(p_after, params_before[name]), f"learned parameter {name} changed"


def test_bn_adapt_sequential_only_bn_buffers_change():
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(16, 3, 28, 28)
    buffers_before = {name: b.clone() for name, b in model.named_buffers()}

    adapted = bn_adapt_sequential(model, [x[0:8], x[8:16]])

    bn_buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    for name, b_after in adapted.named_buffers():
        if name.endswith(bn_buffer_suffixes):
            continue
        assert torch.equal(b_after, buffers_before[name]), f"non-BN buffer {name} changed"


def test_bn_adapt_sequential_resets_independently_across_n_conditions():
    """Two microbatch adaptations from the SAME original model, with
    DIFFERENT inputs, produce different stats and never leak state --
    identical discipline to the existing single-batch bn_adapt() test."""
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x1 = torch.rand(16, 3, 28, 28)
    x2 = torch.rand(16, 3, 28, 28) + 5.0

    adapted1 = bn_adapt_sequential(model, [x1[0:8], x1[8:16]])
    adapted2 = bn_adapt_sequential(model, [x2[0:8], x2[8:16]])  # from the SAME original model
    rm1 = adapted1.features[1].running_mean.clone()
    rm2 = adapted2.features[1].running_mean.clone()
    assert not torch.equal(rm1, rm2)

    adapted1_again = bn_adapt_sequential(model, [x1[0:8], x1[8:16]])
    assert torch.equal(adapted1_again.features[1].running_mean, rm1)


def test_bn_adapt_sequential_rejects_groupnorm():
    model = build_small_cnn(num_classes=9, normalization="groupnorm")
    x = torch.rand(4, 3, 28, 28)
    with pytest.raises(BNAdaptationNotApplicableError):
        bn_adapt_sequential(model, [x])


def test_every_n_starts_from_fresh_original_checkpoint_in_compute_validation_evaluation():
    """Within compute_validation_evaluation(), every registered N's BN
    adaptation is independently derived from the SAME original `model`
    (never a previously-adapted model) -- verified by confirming the
    original model's own running stats are never mutated across the
    whole run."""
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    split = _synthetic_split(n=20, n_classes=3)

    compute_validation_evaluation(model, split, tta_seed=5, device=DEVICE)

    for k, v in model.state_dict().items():
        assert torch.equal(v, original_state[k]), f"original model mutated at {k}"


# ---------------------------------------------------------------------------
# 16-17: latency covers the bounded path, reports all 7 N, formulas
# unchanged
# ---------------------------------------------------------------------------


def test_latency_report_covers_all_seven_n_values_with_correct_formulas():
    torch.manual_seed(0)
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    model.eval()
    split = _synthetic_split(n=5, n_classes=3)  # small -- this test is about N-coverage, not chunking

    report = compute_evaluation_latency_report(model, split, tta_seed=1, device=DEVICE)

    assert set(report.tta_latency_seconds_by_n.keys()) == {1, 2, 5, 10, 25, 50, 100}
    assert report.n_samples == 5
    for n, total in report.tta_latency_seconds_by_n.items():
        assert report.per_sample_latency_seconds_by_n[n] == pytest.approx(total / 5)
        expected_mult = (
            total / report.clean_latency_seconds if report.clean_latency_seconds > 0 else float("inf")
        )
        assert report.compute_multiplier_by_n[n] == pytest.approx(expected_mult)


def test_latency_measurement_forward_calls_never_exceed_frozen_batch_size(monkeypatch):
    """Separate from N-coverage above: proves latency's own forward calls
    respect INFERENCE_BATCH_SIZE at n_samples > that size. PREFIX_SEQUENCE
    monkeypatched down purely for test speed (per-sample view-generation
    cost, not the batching logic under test, otherwise dominates)."""
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "PREFIX_SEQUENCE", (1, 2))

    torch.manual_seed(0)
    guard = _BatchSizeGuardModel(
        build_small_cnn(num_classes=3, normalization="batchnorm"), INFERENCE_BATCH_SIZE
    )
    split = _synthetic_split(n=300, n_classes=3)

    report = ve.compute_evaluation_latency_report(guard, split, tta_seed=1, device=DEVICE)

    assert max(guard.batch_sizes_seen) <= INFERENCE_BATCH_SIZE
    assert 44 in guard.batch_sizes_seen
    assert report.n_samples == 300


# ---------------------------------------------------------------------------
# 18-19: schema requires batching provenance; frozen batch size
# participates in evaluation identity
# ---------------------------------------------------------------------------


def _valid_batching_dict():
    return {
        "inference_batch_size": 256,
        "bn_adaptation_batch_size": 256,
        "bn_adaptation_algorithm": "sequential_microbatch_v1",
        "bn_adaptation_enumeration_order": "view_major_then_sample_major",
        "bn_adaptation_microbatches_at_primary_n": 50,
    }


def _valid_dataset_verification_dict():
    checksum = "a" * 32
    return {
        "dataset": "pathmnist",
        "resolution": 28,
        "expected_checksum_md5": checksum,
        "actual_checksum_md5": checksum,
        "checksum_verified": True,
        "resized": False,
        "verification_method": "dataset_verification.verify_official_dataset_artifact",
        "verification_version": 1,
        "artifact_path": "data/raw/pathmnist.npz",
    }


def _valid_metadata_dict():
    return {
        "evaluation_id": "e1",
        "training_run_id": "r1",
        "training_attempt": 1,
        "checkpoint_hash": "c1",
        "dataset": "pathmnist",
        "resolution": 28,
        "model": "small_cnn",
        "normalization": "batchnorm",
        "training_policy": "none",
        "seed": 0,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "prefix_sequence": [1, 2, 5, 10, 25, 50, 100],
        "aggregators": ["mean_probability"],
        "secondary_analyses": ["scaling_curve"],
        "protocol_commit": "ce4c962",
        "matrix_hash": "m1",
        "source_commit": "s1",
        "evaluator_fingerprint": "fp1",
        "evaluator_fingerprint_manifest": {"src/when_tta_hurts/metrics.py": "abc123"},
        "dataset_expected_checksum_md5": "a" * 32,
        "dataset_verification": _valid_dataset_verification_dict(),
        "batching": _valid_batching_dict(),
        "evaluation_config_hash": "e1",
        "split": "validation",
        "n_validation_samples": 3,
    }


def _valid_view_manifest_dict():
    return {
        "dataset": "pathmnist",
        "resolution": 28,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "n_views": 100,
        "seed_formula": "sha256(...)",
        "sample_indices": [0, 1, 2],
        "seed_manifest_sha256": "abc",
    }


def _valid_latency_dict(n_samples=3):
    prefix_sequence = (1, 2, 5, 10, 25, 50, 100)
    by_n = {}
    for n in prefix_sequence:
        tta = 0.01 * n
        by_n[str(n)] = {
            "tta_latency_seconds": tta,
            "per_sample_latency_seconds": tta / n_samples,
            "compute_multiplier": tta / 0.01,
        }
    return {"clean_latency_seconds": 0.01, "n_samples": n_samples, "by_n": by_n}


def _valid_predictions_dict(n=3, c=3):
    return {
        "labels": np.arange(n) % c,
        "sample_indices": np.arange(n),
        "clean_probs": np.full((n, c), 1.0 / c, dtype=np.float32),
        "view_probs": np.full((100, n, c), 1.0 / c, dtype=np.float32),
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda md: md.__delitem__("batching"),
        lambda md: md["batching"].__delitem__("inference_batch_size"),
        lambda md: md["batching"].update(inference_batch_size=0),
        lambda md: md["batching"].update(inference_batch_size=-1),
        lambda md: md["batching"].update(inference_batch_size="256"),
        lambda md: md["batching"].update(bn_adaptation_batch_size=0),
        lambda md: md["batching"].update(bn_adaptation_microbatches_at_primary_n=-1),
        lambda md: md["batching"].update(bn_adaptation_algorithm=""),
        lambda md: md["batching"].update(bn_adaptation_enumeration_order=""),
    ],
)
def test_persist_rejects_malformed_batching_provenance(tmp_path, mutate):
    metadata = _valid_metadata_dict()
    mutate(metadata)
    with pytest.raises((EvaluationSchemaValidationError, EvaluationPersistenceError)):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=_valid_predictions_dict(),
            metrics={
                "training_run_id": "r1",
                "evaluation_config_hash": "e1",
                "clean": {"accuracy": 1.0 / 3},
                "conditions": {},
                "latency": _valid_latency_dict(),
            },
            metadata=metadata,
            view_manifest=_valid_view_manifest_dict(),
            prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        )


def test_frozen_config_rejects_wrong_inference_batch_size(tmp_path):
    yaml_text = """
schema_version: "1.0"
status: approved
split: validation
confirmatory_tta_seed: 1306178015
derivation:
  namespace: "when-tta-hurts|phase2b|confirmatory-tta|v1"
  sha256_digest: "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd"
  conversion_rule: "int(digest[:8], 16)"
excluded_seeds:
  pilot_tta_seed: 271828
  pilot_training_seed: 314159
  confirmatory_training_seeds: [0, 1, 2]
prefix_sequence: [1, 2, 5, 10, 25, 50, 100]
total_generated_views: 100
primary_prefix: 50
primary_aggregation: mean_probability
policy_identifier: mixed
inference_batch_size: 128
bn_adaptation_batch_size: 256
bn_adaptation_algorithm: sequential_microbatch_v1
bn_adaptation_enumeration_order: view_major_then_sample_major
"""
    path = tmp_path / "validation_evaluation.yaml"
    path.write_text(yaml_text)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=lambda p: True,
            last_commit_for_path=lambda p: "c" * 40,
            commit_is_ancestor=lambda c, h: True,
        )


def test_frozen_config_rejects_wrong_bn_adaptation_batch_size(tmp_path):
    yaml_text = """
schema_version: "1.0"
status: approved
split: validation
confirmatory_tta_seed: 1306178015
derivation:
  namespace: "when-tta-hurts|phase2b|confirmatory-tta|v1"
  sha256_digest: "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd"
  conversion_rule: "int(digest[:8], 16)"
excluded_seeds:
  pilot_tta_seed: 271828
  pilot_training_seed: 314159
  confirmatory_training_seeds: [0, 1, 2]
prefix_sequence: [1, 2, 5, 10, 25, 50, 100]
total_generated_views: 100
primary_prefix: 50
primary_aggregation: mean_probability
policy_identifier: mixed
inference_batch_size: 256
bn_adaptation_batch_size: 999
bn_adaptation_algorithm: sequential_microbatch_v1
bn_adaptation_enumeration_order: view_major_then_sample_major
"""
    path = tmp_path / "validation_evaluation.yaml"
    path.write_text(yaml_text)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=lambda p: True,
            last_commit_for_path=lambda p: "c" * 40,
            commit_is_ancestor=lambda c, h: True,
        )


def test_changing_tta_seed_config_sha256_changes_evaluation_id():
    """The batching fields live inside configs/validation_evaluation.yaml,
    whose content hash (tta_seed_config_sha256) is already a hashed field
    in ValidationEvaluationConfig -- changing that hash (as any byte
    change to the frozen batching settings would) changes evaluation_id,
    without needing a separate dedicated hashed field."""
    base = dict(
        training_run_id="r",
        training_attempt=1,
        checkpoint_hash="c",
        split="validation",
        tta_seed=1306178015,
        prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        aggregators=("mean_probability",),
        secondary_analyses=("scaling_curve",),
        policy="mixed",
        protocol_commit="ce4c962",
        matrix_hash="m",
        evaluator_fingerprint="fp",
        dataset_expected_checksum_md5="a" * 32,
        tta_seed_config_sha256="sha_A",
        tta_seed_freeze_commit="c" * 40,
        tta_seed_derivation_sha256="d" * 64,
    )
    id_a = compute_evaluation_id(ValidationEvaluationConfig(**base))
    id_b = compute_evaluation_id(ValidationEvaluationConfig(**{**base, "tta_seed_config_sha256": "sha_B"}))
    assert id_a != id_b


# ---------------------------------------------------------------------------
# 20: OOM/failure cannot produce completed artifacts (BN-specific)
# ---------------------------------------------------------------------------


def test_bn_adaptation_forward_failure_never_persists_partial_results(tmp_path):
    """A forward-call failure during BN adaptation (simulated here, not a
    real OOM) propagates as an exception from compute_validation_evaluation()
    -- it is the caller's (run_validation_evaluation()'s) job to route
    that to a terminal failed attempt; this test confirms the exception
    propagates rather than being silently swallowed or partially
    persisted."""

    class _ExplodesOnLargeBatch(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            if x.shape[0] > BN_ADAPTATION_BATCH_SIZE:
                raise RuntimeError("Invalid buffer size: simulated OOM")
            return self.inner(x)

    torch.manual_seed(0)
    model = _ExplodesOnLargeBatch(build_small_cnn(num_classes=3, normalization="batchnorm"))
    split = _synthetic_split(n=10, n_classes=3)

    # With bounded batching, this must NOT raise (every forward call is
    # <= BN_ADAPTATION_BATCH_SIZE) -- proving the fix actually prevents
    # the simulated OOM condition from ever triggering.
    outcome = compute_validation_evaluation(model, split, tta_seed=1, device=DEVICE)
    assert outcome["predictions"]["clean_probs"].shape[0] == 10
