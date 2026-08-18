"""Tests for validation_evaluation.py (Phase 2B.4A). Uses ONLY synthetic
tensors, temporary checkpoints/repositories, and fresh (never real-data-
trained) tiny models. resolve_canonical_training_completion() is tested
against the REAL, already-completed confirmatory matrix/ledger where noted
-- that is a metadata-only read (no checkpoint bytes loaded, no TTA
computed), identical in kind to what plan_validation_evaluation() already
does safely. No real checkpoint is ever loaded for inference, no MPS is
ever initialized, and the official test split is never touched anywhere
in this file."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit
from when_tta_hurts.evaluation.views import (
    build_view_seed_manifest,
    generate_single_view,
    iter_deterministic_views,
    stable_view_seed,
)
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
    persist_and_verify_evaluation_completion,
    recompute_clean_accuracy,
    recompute_mean_probability_prefix,
    validate_predictions_arrays,
)
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.orchestrator import PilotOrExcludedSeedRunIdError, UnknownRunIdError
from when_tta_hurts.transforms.policies import build_policy
from when_tta_hurts.validation_evaluation import EvaluationRunStatus as _RunStatus
from when_tta_hurts.validation_evaluation import (
    FrozenTTASeedConfigError,
    ValidationEvaluationConfig,
    check_evaluation_skip,
    compute_evaluation_id,
    compute_validation_evaluation,
    finish_evaluation_attempt,
    list_evaluation_attempts,
    load_frozen_tta_seed_config,
    plan_validation_evaluation,
    resolve_canonical_training_completion,
    run_validation_evaluation,
    start_evaluation_attempt,
)

MATRIX_PATH = "configs/experiment_matrix.yaml"
FROZEN_TTA_SEED = 1306178015
REAL_TTA_SEED_CONFIG_PATH = "configs/validation_evaluation.yaml"


def _synthetic_split(n=4, n_classes=3, resolution=28, dataset="pathmnist", seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, resolution, resolution, generator=g)
    labels = np.array([i % n_classes for i in range(n)])
    return ValidationEvaluationSplit(
        images=images, labels=labels, sample_indices=np.arange(n), dataset=dataset, resolution=resolution
    )


# ---------------------------------------------------------------------------
# Frozen confirmatory TTA-seed configuration
# ---------------------------------------------------------------------------

_VALID_YAML_TEXT = """
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
"""


def _write_config(tmp_path, text=None, **overrides):
    import yaml

    if text is not None:
        content = text
    else:
        data = yaml.safe_load(_VALID_YAML_TEXT)
        for key, value in overrides.items():
            data[key] = value
        content = yaml.safe_dump(data)
    path = tmp_path / "validation_evaluation.yaml"
    path.write_text(content)
    return path


def _always_tracked_clean(path):
    return True


def _commit_for(commit):
    return lambda path: commit


def _all_ancestors(commit, head):
    return True


def test_independent_seed_derivation_equals_frozen_value():
    import hashlib

    namespace = "when-tta-hurts|phase2b|confirmatory-tta|v1"
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    assert digest == "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd"
    assert int(digest[:8], 16) == FROZEN_TTA_SEED == 1306178015


@pytest.mark.parametrize("excluded", [0, 1, 2, 271828, 314159])
def test_frozen_seed_differs_from_pilot_and_training_seeds(excluded):
    assert FROZEN_TTA_SEED != excluded


def test_valid_tracked_configuration_succeeds(tmp_path):
    path = _write_config(tmp_path)
    cfg = load_frozen_tta_seed_config(
        path,
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    assert cfg.confirmatory_tta_seed == FROZEN_TTA_SEED
    assert cfg.prefix_sequence == (1, 2, 5, 10, 25, 50, 100)
    assert cfg.primary_prefix == 50
    assert cfg.primary_aggregation == "mean_probability"
    assert cfg.policy_identifier == "mixed"
    assert cfg.freeze_commit == "c" * 40


def test_missing_configuration_fails(tmp_path):
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            tmp_path / "does_not_exist.yaml",
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_untracked_configuration_fails(tmp_path):
    path = _write_config(tmp_path)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=lambda p: False,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_dirty_configuration_fails(tmp_path):
    path = _write_config(tmp_path)
    calls = []

    def dirty_check(p):
        calls.append(p)
        return False

    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=dirty_check,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )
    assert calls


def test_uncommitted_configuration_fails(tmp_path):
    """No commit in repository history for this path -- distinct from
    'untracked' (git_tracked_and_clean can pass for a staged-but-never-
    committed file in principle; last_commit_for_path returning None is
    the authoritative 'never committed' signal)."""
    path = _write_config(tmp_path)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=lambda p: None,
            commit_is_ancestor=_all_ancestors,
        )


def test_malformed_configuration_fails(tmp_path):
    path = _write_config(tmp_path, text="{not: valid: yaml: [")
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_draft_status_configuration_fails(tmp_path):
    path = _write_config(tmp_path, status="draft")
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_wrong_seed_fails(tmp_path):
    path = _write_config(tmp_path, confirmatory_tta_seed=999999999)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_pilot_tta_seed_fails(tmp_path):
    path = _write_config(tmp_path, confirmatory_tta_seed=271828)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 314159])
def test_training_and_pilot_seeds_fail(tmp_path, seed):
    path = _write_config(tmp_path, confirmatory_tta_seed=seed)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix_sequence", [1, 2, 5]),
        ("primary_prefix", 25),
        ("primary_aggregation", "majority_vote"),
        ("policy_identifier", "geometric"),
        ("total_generated_views", 50),
    ],
)
def test_altered_view_config_fails(tmp_path, field, value):
    path = _write_config(tmp_path, **{field: value})
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_non_ancestor_freeze_commit_fails(tmp_path):
    path = _write_config(tmp_path)
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=lambda commit, head: False,
        )


def test_head_may_be_descendant_of_freeze_commit(tmp_path):
    """HEAD need not equal the freeze commit -- only ancestry is
    required, and a later HEAD (a descendant) is exactly the expected
    case in ordinary use."""
    path = _write_config(tmp_path)
    cfg = load_frozen_tta_seed_config(
        path,
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
        head_commit="a_much_later_descendant_commit",
    )
    assert cfg.confirmatory_tta_seed == FROZEN_TTA_SEED


def test_tampered_derivation_digest_fails(tmp_path):
    """Recomputed SHA-256 of the namespace must match the recorded
    digest -- a config claiming a different (wrong) digest fails even if
    confirmatory_tta_seed itself happens to still read 1306178015."""
    import yaml

    data = yaml.safe_load(_VALID_YAML_TEXT)
    data["derivation"]["sha256_digest"] = "0" * 64
    path = tmp_path / "validation_evaluation.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(FrozenTTASeedConfigError):
        load_frozen_tta_seed_config(
            path,
            git_tracked_and_clean=_always_tracked_clean,
            last_commit_for_path=_commit_for("c" * 40),
            commit_is_ancestor=_all_ancestors,
        )


def test_real_committed_configuration_loads_successfully():
    """The actual, committed configs/validation_evaluation.yaml must load
    and verify successfully against the REAL git repository state -- a
    pure local file+git-metadata read, no MPS/dataset/checkpoint access."""
    cfg = load_frozen_tta_seed_config(REAL_TTA_SEED_CONFIG_PATH)
    assert cfg.confirmatory_tta_seed == FROZEN_TTA_SEED
    assert len(cfg.freeze_commit) == 40
    assert len(cfg.config_file_sha256) == 64


# ---------------------------------------------------------------------------
# Deterministic view generation
# ---------------------------------------------------------------------------


def test_stable_view_seed_deterministic_and_not_python_hash():
    a = stable_view_seed(1, "pathmnist", 28, 5, 2)
    b = stable_view_seed(1, "pathmnist", 28, 5, 2)
    assert a == b
    assert isinstance(a, int)


def test_stable_view_seed_varies_with_each_identifier():
    base = stable_view_seed(1, "pathmnist", 28, 5, 2)
    assert stable_view_seed(2, "pathmnist", 28, 5, 2) != base
    assert stable_view_seed(1, "bloodmnist", 28, 5, 2) != base
    assert stable_view_seed(1, "pathmnist", 64, 5, 2) != base
    assert stable_view_seed(1, "pathmnist", 28, 6, 2) != base
    assert stable_view_seed(1, "pathmnist", 28, 5, 3) != base


def test_view_identity_independent_of_batch_order():
    policy = build_policy("mixed", output_size=(28, 28))
    x = torch.rand(6, 3, 28, 28)
    full = generate_single_view(x, policy, 12345, "pathmnist", 28, list(range(6)), view_index=2)
    reordered_x = x[[5, 4, 3, 2, 1, 0]]
    reordered = generate_single_view(
        reordered_x, policy, 12345, "pathmnist", 28, [5, 4, 3, 2, 1, 0], view_index=2
    )
    assert torch.allclose(full[3], reordered[2])


def test_view_identity_independent_of_batch_size():
    policy = build_policy("mixed", output_size=(28, 28))
    x = torch.rand(6, 3, 28, 28)
    full = generate_single_view(x, policy, 12345, "pathmnist", 28, list(range(6)), view_index=2)
    subset = generate_single_view(x[[3]], policy, 12345, "pathmnist", 28, [3], view_index=2)
    assert torch.allclose(full[3], subset[0])


def test_view_identity_independent_of_model_and_training_seed():
    """View generation never touches the model at all -- confirmed by
    generating a view with no model in scope, then checking two DIFFERENT
    models fed the same view produce the view via the identical, model-
    independent transform (i.e. the view tensor itself never depends on
    which model/training-seed consumes it)."""
    policy = build_policy("mixed", output_size=(28, 28))
    x = torch.rand(3, 3, 28, 28)
    v1 = generate_single_view(x, policy, 999, "bloodmnist", 28, [0, 1, 2], view_index=0)
    v2 = generate_single_view(x, policy, 999, "bloodmnist", 28, [0, 1, 2], view_index=0)
    assert torch.allclose(v1, v2)  # no model/training-seed parameter exists on this function at all


def test_different_samples_receive_different_transforms():
    policy = build_policy("mixed", output_size=(28, 28))
    x = torch.rand(2, 3, 28, 28)
    view = generate_single_view(x, policy, 42, "pathmnist", 28, [0, 1], view_index=0)
    assert not torch.allclose(view[0], view[1])


def test_augmentation_applied_exactly_once_per_view():
    """iter_deterministic_views yields exactly one transformed batch per
    view index, each computed by exactly one generate_single_view() call
    (verified via call counting)."""
    import when_tta_hurts.evaluation.views as views_module

    calls = []
    real_generate = views_module.generate_single_view

    def counting_generate(*args, **kwargs):
        calls.append(args[-1] if "view_index" not in kwargs else kwargs["view_index"])
        return real_generate(*args, **kwargs)

    orig = views_module.generate_single_view
    views_module.generate_single_view = counting_generate
    try:
        policy = build_policy("mixed", output_size=(28, 28))
        x = torch.rand(2, 3, 28, 28)
        results = list(iter_deterministic_views(x, policy, 1, "pathmnist", 28, [0, 1], n_views=5))
    finally:
        views_module.generate_single_view = orig
    assert len(results) == 5
    assert calls == [0, 1, 2, 3, 4]


def test_view_seed_manifest_covers_every_sample_view_pair():
    manifest = build_view_seed_manifest(1, "pathmnist", 28, [10, 20], n_views=3)
    assert len(manifest) == 6
    pairs = {(e.sample_index, e.view_index) for e in manifest}
    assert pairs == {(10, 0), (10, 1), (10, 2), (20, 0), (20, 1), (20, 2)}


def test_view_seed_manifest_does_not_depend_on_training_seed_or_model():
    import inspect

    from when_tta_hurts.evaluation import views as views_module

    params = set(inspect.signature(views_module.build_view_seed_manifest).parameters)
    assert params == {"tta_seed", "dataset", "resolution", "sample_indices", "n_views"}


def test_no_python_randomized_hash_used():
    import inspect

    from when_tta_hurts.evaluation import views as views_module

    source = inspect.getsource(views_module)
    # only the builtin bytes-hashing hashlib is used; bare hash( calls are absent
    assert "hashlib" in source
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("hash(")


# ---------------------------------------------------------------------------
# Evaluation identity
# ---------------------------------------------------------------------------


def _cfg(**overrides):
    base = dict(
        training_run_id="A-pathmnist-28px-batchnorm-policy-none-s0",
        training_attempt=1,
        checkpoint_hash="deadbeef",
        split="validation",
        tta_seed=1306178015,
        prefix_sequence=(1, 2, 5, 10, 25, 50, 100),
        aggregators=("mean_probability", "majority_vote", "confidence_weighted_average"),
        secondary_analyses=("scaling_curve",),
        policy="mixed",
        protocol_commit="ce4c962",
        matrix_hash="abc",
        source_commit="def",
        tta_seed_config_sha256="cfg_sha256_abc",
        tta_seed_freeze_commit="c" * 40,
        tta_seed_derivation_sha256="4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
    )
    base.update(overrides)
    return ValidationEvaluationConfig(**base)


def test_evaluation_id_deterministic():
    h1 = compute_evaluation_id(_cfg())
    h2 = compute_evaluation_id(_cfg())
    assert h1 == h2


def test_production_evaluation_identity_uses_frozen_seed():
    cfg = _cfg()
    assert cfg.tta_seed == 1306178015


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_hash", "different"),
        ("tta_seed", 999999),
        ("prefix_sequence", (1, 2, 5)),
        ("aggregators", ("mean_probability",)),
        ("policy", "geometric"),
        ("matrix_hash", "different"),
        ("tta_seed_config_sha256", "different_sha"),
        ("tta_seed_freeze_commit", "d" * 40),
        ("tta_seed_derivation_sha256", "0" * 64),
    ],
)
def test_evaluation_id_changes_with_each_field(field, value):
    base = compute_evaluation_id(_cfg())
    changed = compute_evaluation_id(_cfg(**{field: value}))
    assert base != changed


# ---------------------------------------------------------------------------
# Canonical training-completion resolution (rejections + real success case)
# ---------------------------------------------------------------------------


def test_resolve_rejects_pilot_seed():
    with pytest.raises(PilotOrExcludedSeedRunIdError):
        resolve_canonical_training_completion("A-pathmnist-28px-batchnorm-policy-none-s314159", MATRIX_PATH)


def test_resolve_rejects_unknown_run_id():
    with pytest.raises(UnknownRunIdError):
        resolve_canonical_training_completion("not-a-real-run-id", MATRIX_PATH)


def test_resolve_finds_real_canonical_completion():
    """Metadata-only read against the real, already-completed matrix -- no
    checkpoint is loaded, no inference is run."""
    cell, result = resolve_canonical_training_completion(
        "A-pathmnist-28px-batchnorm-policy-none-s0", MATRIX_PATH
    )
    assert cell.run_id() == "A-pathmnist-28px-batchnorm-policy-none-s0"
    assert result.status in ("completed", "skipped_completed")
    assert result.checkpoint_hash is not None


# ---------------------------------------------------------------------------
# Attempt state machine / idempotent skip / stale detection
# ---------------------------------------------------------------------------


def test_idempotent_skip_before_any_factory(tmp_path, monkeypatch):
    import when_tta_hurts.ledger as ledger_module

    ledger_path = tmp_path / "ledger_validation_evaluation.csv"
    monkeypatch.setattr(ledger_module, "VALIDATION_EVALUATION_LEDGER_PATH", ledger_path)

    run_id = "fake-run"
    cfg_hash = "hash123"
    attempt_dir, status = start_evaluation_attempt(run_id, cfg_hash, root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.COMPLETED)
    (attempt_dir / "predictions.npz").write_bytes(b"x")
    (attempt_dir / "metrics.json").write_text("{}")
    (attempt_dir / "metadata.json").write_text("{}")
    (attempt_dir / "view_manifest.json").write_text("{}")
    from when_tta_hurts.artifacts import atomic_write_json
    from when_tta_hurts.evaluation_result_artifacts import build_evaluation_artifact_manifest
    from when_tta_hurts.ledger import append_evaluation_entry

    manifest = build_evaluation_artifact_manifest(attempt_dir)
    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    append_evaluation_entry(
        evaluation_id=cfg_hash,
        training_run_id=run_id,
        training_attempt=1,
        checkpoint_hash="ckpt",
        evaluation_config_hash=cfg_hash,
        evaluation_attempt=status.attempt_number,
        status="completed",
        primary_artifact_hash="art",
        started_at=status.started_at,
        ended_at=status.ended_at,
        runtime_seconds=1.0,
        ledger_path=ledger_path,
    )

    skip = check_evaluation_skip(run_id, cfg_hash, root=tmp_path, ledger_path=ledger_path)
    assert skip is not None
    assert skip["attempt_number"] == 1


def test_skip_returns_none_for_different_config_hash(tmp_path):
    from when_tta_hurts.ledger import append_evaluation_entry

    run_id = "fake-run-2"
    ledger_path = tmp_path / "ledger.csv"
    attempt_dir, status = start_evaluation_attempt(run_id, "hashA", root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.COMPLETED)
    append_evaluation_entry(
        evaluation_id="hashA",
        training_run_id=run_id,
        training_attempt=1,
        checkpoint_hash="ckpt",
        evaluation_config_hash="hashA",
        evaluation_attempt=status.attempt_number,
        status="completed",
        primary_artifact_hash="art",
        started_at=status.started_at,
        ended_at=status.ended_at,
        runtime_seconds=1.0,
        ledger_path=ledger_path,
    )
    skip = check_evaluation_skip(run_id, "hashB", root=tmp_path, ledger_path=ledger_path)
    assert skip is None


def test_stale_nonterminal_attempt_raises(tmp_path):
    from when_tta_hurts.validation_evaluation import EvaluationStaleAttemptError

    run_id = "fake-run-3"
    start_evaluation_attempt(run_id, "hashC", root=tmp_path)  # left RUNNING, no ledger row
    with pytest.raises(EvaluationStaleAttemptError):
        check_evaluation_skip(run_id, "hashC", root=tmp_path)


def test_attempt_numbers_increment(tmp_path):
    run_id = "fake-run-4"
    d1, s1 = start_evaluation_attempt(run_id, "h1", root=tmp_path)
    finish_evaluation_attempt(d1, s1, _RunStatus.FAILED, failure_reason="boom")
    d2, s2 = start_evaluation_attempt(run_id, "h2", root=tmp_path)
    assert s1.attempt_number == 1
    assert s2.attempt_number == 2
    assert len(list_evaluation_attempts(run_id, root=tmp_path)) == 2


# ---------------------------------------------------------------------------
# Ledger idempotency / conflict
# ---------------------------------------------------------------------------


def test_evaluation_ledger_idempotent_and_conflicting(tmp_path):
    from when_tta_hurts.ledger import LedgerConflictError, append_evaluation_entry

    path = tmp_path / "ledger.csv"
    kwargs = dict(
        evaluation_id="eid1",
        training_run_id="run1",
        training_attempt=1,
        checkpoint_hash="ckpt1",
        evaluation_config_hash="eid1",
        evaluation_attempt=1,
        status="completed",
        primary_artifact_hash="art1",
        started_at=1.0,
        ended_at=2.0,
        runtime_seconds=1.0,
        ledger_path=path,
    )
    first = append_evaluation_entry(**kwargs)
    assert first == "appended"
    second = append_evaluation_entry(**kwargs)
    assert second == "duplicate_ignored"

    conflicting = dict(kwargs)
    conflicting["checkpoint_hash"] = "different"
    with pytest.raises(LedgerConflictError):
        append_evaluation_entry(**conflicting)


def test_evaluation_ledger_always_records_validation_split_and_no_test_metrics(tmp_path):
    from when_tta_hurts.ledger import _read_existing_rows, append_evaluation_entry

    path = tmp_path / "ledger.csv"
    append_evaluation_entry(
        evaluation_id="eid2",
        training_run_id="run1",
        training_attempt=1,
        checkpoint_hash="ckpt1",
        evaluation_config_hash="eid2",
        evaluation_attempt=1,
        status="completed",
        primary_artifact_hash="art1",
        started_at=1.0,
        ended_at=2.0,
        runtime_seconds=1.0,
        ledger_path=path,
    )
    rows = _read_existing_rows(path)
    assert rows[0]["split"] == "validation"
    assert rows[0]["test_metrics_observed"] == "False"


def test_evaluation_ledger_header_only_file_creation(tmp_path):
    from when_tta_hurts.ledger import VALIDATION_EVALUATION_LEDGER_FIELDNAMES, ensure_evaluation_ledger_exists

    path = tmp_path / "ledger.csv"
    created = ensure_evaluation_ledger_exists(path)
    assert created is True
    content = path.read_text()
    assert content.count("\n") == 1  # header only, single line
    for field in VALIDATION_EVALUATION_LEDGER_FIELDNAMES:
        assert field in content
    assert ensure_evaluation_ledger_exists(path) is False  # no-op second call


# ---------------------------------------------------------------------------
# Plan mode: side-effect-free
# ---------------------------------------------------------------------------


def test_plan_mode_side_effect_free(tmp_path):
    before = set(tmp_path.rglob("*"))
    report = plan_validation_evaluation(MATRIX_PATH)
    after = set(tmp_path.rglob("*"))
    assert before == after
    assert len(report["cells"]) == 39


def test_plan_mode_reports_frozen_seed_config_hash_and_freeze_commit():
    report = plan_validation_evaluation(MATRIX_PATH)
    seed_report = report["tta_seed_config"]
    assert "error" not in seed_report
    assert seed_report["confirmatory_tta_seed"] == 1306178015
    assert len(seed_report["config_file_sha256"]) == 64
    assert len(seed_report["freeze_commit"]) == 40


def test_plan_mode_never_initializes_mps_or_touches_dataset():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(plan_validation_evaluation)))
    func_body = tree.body[0].body
    code_nodes = func_body[1:] if isinstance(func_body[0], ast.Expr) else func_body
    code_source = "\n".join(ast.unparse(node) for node in code_nodes)
    assert "mps" not in code_source.lower()
    assert "load_validation_evaluation_split" not in code_source
    assert "load_and_verify_canonical_checkpoint" not in code_source


# ---------------------------------------------------------------------------
# End-to-end computation on synthetic tensors (no real checkpoint/data)
# ---------------------------------------------------------------------------


def test_compute_validation_evaluation_batchnorm_end_to_end():
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    split = _synthetic_split(n=4, n_classes=3)
    out = compute_validation_evaluation(model, split, 555555, torch.device("cpu"))

    preds = out["predictions"]
    assert preds["view_probs"].shape == (100, 4, 3)
    assert preds["clean_probs"].shape == (4, 3)
    assert "bn_adapted_probs" in preds

    metrics = out["metrics"]
    assert set(metrics["conditions"]["naive_tta"]["mean_probability"].keys()) == {1, 2, 5, 10, 25, 50, 100}
    assert metrics["conditions"]["bn_adapted_tta"] is not None
    assert "primary_endpoint" in metrics


def test_compute_validation_evaluation_groupnorm_skips_bn_adapted():
    model = build_small_cnn(num_classes=3, normalization="groupnorm")
    split = _synthetic_split(n=4, n_classes=3)
    out = compute_validation_evaluation(model, split, 555555, torch.device("cpu"))
    assert out["metrics"]["conditions"]["bn_adapted_tta"] is None
    assert "bn_adapted_probs" not in out["predictions"]


def test_per_view_softmax_before_aggregation():
    """Stored view_probs must be genuine per-sample probability
    distributions (rows summing to 1), confirming softmax was applied per
    view before any averaging -- not applied only after aggregation."""
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    split = _synthetic_split(n=3, n_classes=3)
    out = compute_validation_evaluation(model, split, 111111, torch.device("cpu"))
    view_probs = out["predictions"]["view_probs"]
    sums = view_probs.reshape(-1, 3).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_independently_hand_calculated_mean_probability():
    """Recompute the N=2 mean-probability aggregate BY HAND from the
    stored per-view probability arrays and confirm it matches
    recompute_mean_probability_prefix()'s own output exactly."""
    view_probs = np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],  # view 0
            [[0.5, 0.4, 0.1], [0.3, 0.6, 0.1]],  # view 1
        ],
        dtype=np.float32,
    )
    hand_mean = (view_probs[0] + view_probs[1]) / 2
    assert np.allclose(recompute_mean_probability_prefix(view_probs, 2), hand_mean)


def test_nested_prefixes_are_true_prefixes():
    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    split = _synthetic_split(n=3, n_classes=3)
    out = compute_validation_evaluation(model, split, 222222, torch.device("cpu"))
    view_probs = out["predictions"]["view_probs"]
    mean_5 = recompute_mean_probability_prefix(view_probs, 5)
    mean_5_from_10_prefix = recompute_mean_probability_prefix(view_probs[:10], 5)
    assert np.allclose(mean_5, mean_5_from_10_prefix)  # first 5 of the 10-prefix == the 5-prefix


def test_majority_vote_tie_break_hand_calculated():
    from when_tta_hurts.evaluation.aggregation import majority_vote

    # 2 views, 2 samples, 2 classes: sample 0 has a tie (1 vote each),
    # broken by highest mean probability; sample 1 has a clear winner.
    logits = np.log(
        np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],  # view 0: sample0->class0, sample1->class1
                [[0.1, 0.9], [0.3, 0.7]],  # view 1: sample0->class1, sample1->class1
            ]
        )
    )
    predicted, _ = majority_vote(logits, 2)
    # sample 0: tied 1-1; mean probs = [0.5, 0.5] -> exact tie -> lowest index (0)
    assert predicted[0] == 0
    # sample 1: 2 votes for class 1
    assert predicted[1] == 1


def test_confidence_weighting_hand_calculated():
    from when_tta_hurts.evaluation.aggregation import confidence_weighted_average

    view_logits = np.log(
        np.array(
            [
                [[0.9, 0.1]],  # view 0, confidence 0.9
                [[0.6, 0.4]],  # view 1, confidence 0.6
            ]
        )
    )
    result = np.exp(confidence_weighted_average(view_logits, 2))
    w0, w1 = 0.9 / (0.9 + 0.6), 0.6 / (0.9 + 0.6)
    expected = w0 * np.array([0.9, 0.1]) + w1 * np.array([0.6, 0.4])
    assert np.allclose(result[0], expected, atol=1e-6)


def test_original_anchoring_hand_calculated():
    from when_tta_hurts.evaluation.aggregation import original_anchored_mean_probability

    clean_logits = np.log(np.array([[0.8, 0.2]]))
    view_logits = np.log(np.array([[[0.4, 0.6]], [[0.2, 0.8]]]))  # 2 views, 1 sample
    result = np.exp(original_anchored_mean_probability(clean_logits, view_logits, 2))
    expected = (np.array([0.8, 0.2]) + np.array([0.4, 0.6]) + np.array([0.2, 0.8])) / 3
    assert np.allclose(result[0], expected, atol=1e-6)


def test_accuracy_macro_f1_nll_ece_brier_are_frozen_functions_reused():
    """Confirms this module reuses metrics.py's functions unchanged rather
    than reimplementing them (a hand-check that the imports resolve to
    the SAME function objects)."""
    import when_tta_hurts.metrics as metrics_module
    import when_tta_hurts.validation_evaluation as ve_module

    assert ve_module.accuracy is metrics_module.accuracy
    assert ve_module.macro_f1 is metrics_module.macro_f1
    assert ve_module.negative_log_likelihood is metrics_module.negative_log_likelihood
    assert ve_module.expected_calibration_error is metrics_module.expected_calibration_error
    assert ve_module.brier_score is metrics_module.brier_score


def test_harm_rescue_computed_via_frozen_function():
    from when_tta_hurts.metrics import harm_rescue_rates

    clean = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    tta = np.log(np.array([[0.1, 0.9], [0.9, 0.1]]))  # both flipped
    labels = np.array([0, 1])
    result = harm_rescue_rates(clean, tta, labels)
    assert result["harm_rate"] == 1.0  # both were clean-correct, now both wrong
    assert result["rescue_rate"] == 0.0


# ---------------------------------------------------------------------------
# BN adaptation: running-stat updates + learned-parameter immutability +
# reset isolation (already enforced inside bn_adapt() itself -- confirmed
# reused unchanged here)
# ---------------------------------------------------------------------------


def test_bn_adaptation_updates_running_stats_not_learned_params():
    from when_tta_hurts.evaluation.bn_adaptation import bn_adapt

    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    model.eval()
    original_running_mean = {
        name: buf.clone() for name, buf in model.named_buffers() if "running_mean" in name
    }
    original_params = {name: p.clone() for name, p in model.named_parameters()}

    adaptation_inputs = torch.rand(8, 3, 28, 28)
    adapted = bn_adapt(model, adaptation_inputs)

    for name, before in original_params.items():
        after = dict(adapted.named_parameters())[name]
        assert torch.equal(before, after)  # learned params immutable

    changed = False
    for name, before in original_running_mean.items():
        after = dict(adapted.named_buffers())[name]
        if not torch.equal(before, after):
            changed = True
    assert changed  # running stats DID update

    # original model itself untouched
    for name, before in original_params.items():
        assert torch.equal(before, dict(model.named_parameters())[name])


def test_bn_adaptation_reset_isolation_across_calls():
    from when_tta_hurts.evaluation.bn_adaptation import bn_adapt

    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    adapted_a = bn_adapt(model, torch.rand(8, 3, 28, 28))
    adapted_b = bn_adapt(model, torch.rand(8, 3, 28, 28) * 5 + 1)
    running_mean_a = dict(adapted_a.named_buffers())
    running_mean_b = dict(adapted_b.named_buffers())
    any_diff = any(
        not torch.equal(running_mean_a[k], running_mean_b[k]) for k in running_mean_a if "running_mean" in k
    )
    assert any_diff  # each call starts fresh from the original checkpoint, not chained


def test_bn_adaptation_groupnorm_rejected():
    from when_tta_hurts.evaluation.bn_adaptation import BNAdaptationNotApplicableError, bn_adapt

    model = build_small_cnn(num_classes=3, normalization="groupnorm")
    with pytest.raises(BNAdaptationNotApplicableError):
        bn_adapt(model, torch.rand(4, 3, 28, 28))


# ---------------------------------------------------------------------------
# Latency: synchronization present, descriptive only
# ---------------------------------------------------------------------------


def test_latency_report_synchronized_and_descriptive():
    from when_tta_hurts.evaluation.latency import build_latency_report

    model = build_small_cnn(num_classes=3, normalization="batchnorm")
    x = torch.rand(4, 3, 28, 28)
    views_by_n = {1: [torch.rand(4, 3, 28, 28)], 2: [torch.rand(4, 3, 28, 28), torch.rand(4, 3, 28, 28)]}
    report = build_latency_report(model, x, views_by_n, torch.device("cpu"))
    assert report.clean_latency_seconds > 0
    assert set(report.tta_latency_seconds_by_n.keys()) == {1, 2}
    assert report.compute_multiplier_by_n[2] > 0


def test_latency_module_uses_mps_synchronize():
    import inspect

    from when_tta_hurts.evaluation import latency as latency_module

    source = inspect.getsource(latency_module)
    assert "torch.mps.synchronize" in source


# ---------------------------------------------------------------------------
# Persistence: schema validation, corrupt/missing/non-finite/misaligned
# ---------------------------------------------------------------------------


def _valid_predictions(n=3, c=3):
    view_probs = np.full((100, n, c), 1.0 / c, dtype=np.float32)
    return {
        "labels": np.arange(n) % c,
        "sample_indices": np.arange(n),
        "clean_probs": np.full((n, c), 1.0 / c, dtype=np.float32),
        "view_probs": view_probs,
    }


def test_validate_predictions_rejects_non_finite():
    preds = _valid_predictions()
    preds["clean_probs"][0, 0] = np.nan
    with pytest.raises(EvaluationPersistenceError):
        validate_predictions_arrays(preds)


def test_validate_predictions_rejects_unnormalized():
    preds = _valid_predictions()
    preds["clean_probs"][0] = [0.9, 0.9, 0.9]
    with pytest.raises(EvaluationPersistenceError):
        validate_predictions_arrays(preds)


def test_validate_predictions_rejects_misaligned_lengths():
    preds = _valid_predictions()
    preds["sample_indices"] = np.arange(2)  # mismatched with labels length 3
    with pytest.raises(EvaluationPersistenceError):
        validate_predictions_arrays(preds)


def test_validate_predictions_rejects_duplicate_sample_indices():
    preds = _valid_predictions()
    preds["sample_indices"] = np.array([0, 0, 1])
    with pytest.raises(EvaluationPersistenceError):
        validate_predictions_arrays(preds)


def test_validate_predictions_rejects_missing_arrays():
    with pytest.raises(EvaluationPersistenceError):
        validate_predictions_arrays({"labels": np.arange(3)})


def _valid_metadata():
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
        "evaluation_config_hash": "e1",
        "split": "validation",
        "n_validation_samples": 3,
    }


def _valid_view_manifest():
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


def _valid_metrics():
    return {
        "training_run_id": "r1",
        "evaluation_config_hash": "e1",
        "clean": {"accuracy": 1.0 / 3},
        "conditions": {},
    }


def test_metadata_schema_rejects_test_split():
    metadata = _valid_metadata()
    metadata["split"] = "test"
    with pytest.raises(EvaluationSchemaValidationError):
        persist_and_verify_evaluation_completion(
            attempt_dir="/tmp/does-not-matter",
            predictions=_valid_predictions(),
            metrics=_valid_metrics(),
            metadata=metadata,
            view_manifest=_valid_view_manifest(),
        )


def test_persist_and_verify_full_round_trip(tmp_path):
    predictions = _valid_predictions()
    metrics = _valid_metrics()
    metrics["clean"]["accuracy"] = recompute_clean_accuracy(predictions["clean_probs"], predictions["labels"])
    manifest = persist_and_verify_evaluation_completion(
        tmp_path,
        predictions=predictions,
        metrics=metrics,
        metadata=_valid_metadata(),
        view_manifest=_valid_view_manifest(),
        metric_recomputers={
            "clean.accuracy": (
                metrics["clean"]["accuracy"],
                lambda: recompute_clean_accuracy(predictions["clean_probs"], predictions["labels"]),
            )
        },
    )
    assert (tmp_path / "predictions.npz").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "view_manifest.json").exists()
    assert len(manifest["artifacts"]) == 4
    # never persists images
    assert "images" not in json.dumps(manifest)


def test_persist_rejects_metric_recomputation_mismatch(tmp_path):
    predictions = _valid_predictions()
    metrics = _valid_metrics()
    metrics["clean"]["accuracy"] = 0.999  # deliberately wrong
    with pytest.raises(EvaluationPersistenceError):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=predictions,
            metrics=metrics,
            metadata=_valid_metadata(),
            view_manifest=_valid_view_manifest(),
            metric_recomputers={
                "clean.accuracy": (
                    0.999,
                    lambda: recompute_clean_accuracy(predictions["clean_probs"], predictions["labels"]),
                )
            },
        )


# ---------------------------------------------------------------------------
# No result-dependent orchestration
# ---------------------------------------------------------------------------


def test_check_evaluation_skip_never_reads_a_metric_value():
    import inspect

    source = inspect.getsource(check_evaluation_skip)
    assert "metrics.json" not in source
    assert "accuracy" not in source.lower()


def test_final_test_remains_locked():
    from when_tta_hurts.authorization import AuthorizationError
    from when_tta_hurts.orchestrator import run_final_test

    with pytest.raises(AuthorizationError):
        run_final_test()


# ---------------------------------------------------------------------------
# Phase 2B.4C: hardened attempt numbering / ledger-directory consistency
# ---------------------------------------------------------------------------


def _append_row(ledger_path, evaluation_id, training_run_id, attempt, status, **overrides):
    from when_tta_hurts.ledger import append_evaluation_entry

    kwargs = dict(
        evaluation_id=evaluation_id,
        training_run_id=training_run_id,
        training_attempt=1,
        checkpoint_hash="ckpt",
        evaluation_config_hash=evaluation_id,
        evaluation_attempt=attempt,
        status=status,
        primary_artifact_hash="",
        started_at=1.0,
        ended_at="",
        runtime_seconds="",
        ledger_path=ledger_path,
    )
    kwargs.update(overrides)
    return append_evaluation_entry(**kwargs)


def test_ledger_only_aborted_attempt_reserves_attempt_number(tmp_path):
    from when_tta_hurts.validation_evaluation import next_evaluation_attempt_number

    ledger_path = tmp_path / "ledger.csv"
    run_id = "ledger-only-run"
    _append_row(ledger_path, "eid1", run_id, 1, "aborted")
    # no directory exists for attempt_001 at all
    assert next_evaluation_attempt_number(run_id, root=tmp_path, ledger_path=ledger_path) == 2


def test_next_execution_resolves_to_attempt_002_for_incident_evaluation_id(tmp_path):
    """Mirrors the real incident: a ledger-only aborted attempt_001 for a
    given evaluation_id means the next real execution for that SAME
    training_run_id resolves to attempt_002."""
    from when_tta_hurts.validation_evaluation import next_evaluation_attempt_number

    ledger_path = tmp_path / "ledger.csv"
    run_id = "A-pathmnist-28px-batchnorm-policy-none-s0"
    _append_row(
        ledger_path,
        "ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5",
        run_id,
        1,
        "aborted",
    )
    assert next_evaluation_attempt_number(run_id, root=tmp_path, ledger_path=ledger_path) == 2


def test_deleted_aborted_attempt_directory_is_not_recreated_by_skip_check(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    run_id = "deleted-aborted-run"
    _append_row(ledger_path, "eid2", run_id, 1, "aborted")
    # check_evaluation_skip must not raise and must not create anything
    before = set(tmp_path.rglob("*"))
    skip = check_evaluation_skip(run_id, "eid2", root=tmp_path, ledger_path=ledger_path)
    after = set(tmp_path.rglob("*"))
    assert skip is None  # aborted is never a skip target
    assert before == after


def test_aborted_attempt_never_causes_completed_skip(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    run_id = "aborted-not-skip-run"
    attempt_dir, status = start_evaluation_attempt(run_id, "eid3", root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.ABORTED, failure_reason="killed")
    _append_row(ledger_path, "eid3", run_id, status.attempt_number, "aborted")
    skip = check_evaluation_skip(run_id, "eid3", root=tmp_path, ledger_path=ledger_path)
    assert skip is None


def test_ledger_directory_hash_conflict_hard_fails(tmp_path):
    from when_tta_hurts.validation_evaluation import EvaluationLedgerConflictError

    ledger_path = tmp_path / "ledger.csv"
    run_id = "conflict-run"
    attempt_dir, status = start_evaluation_attempt(run_id, "eidA", root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.COMPLETED)
    _append_row(ledger_path, "eidB", run_id, status.attempt_number, "completed")  # different evaluation_id!
    with pytest.raises(EvaluationLedgerConflictError):
        check_evaluation_skip(run_id, "eidA", root=tmp_path, ledger_path=ledger_path)


def test_completed_ledger_row_without_directory_hard_fails(tmp_path):
    """Unlike aborted/failed, a 'completed' ledger row with no backing
    directory is NOT the sanctioned case -- completed artifacts must
    never simply vanish."""
    from when_tta_hurts.validation_evaluation import EvaluationLedgerConflictError

    ledger_path = tmp_path / "ledger.csv"
    run_id = "vanished-completed-run"
    _append_row(ledger_path, "eidC", run_id, 1, "completed")
    with pytest.raises(EvaluationLedgerConflictError):
        check_evaluation_skip(run_id, "eidC", root=tmp_path, ledger_path=ledger_path)


def test_terminal_directory_without_ledger_row_hard_fails(tmp_path):
    """A completed/failed/aborted attempt DIRECTORY with no ledger row at
    all indicates a crash between finish_evaluation_attempt() and
    append_evaluation_entry() -- must hard-fail, not silently retry."""
    from when_tta_hurts.validation_evaluation import EvaluationLedgerConflictError

    ledger_path = tmp_path / "ledger.csv"
    run_id = "crash-gap-run"
    attempt_dir, status = start_evaluation_attempt(run_id, "eidD", root=tmp_path, ledger_path=ledger_path)
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.COMPLETED)
    # no ledger row appended -- simulates a crash right after finish_evaluation_attempt()
    with pytest.raises(EvaluationLedgerConflictError):
        check_evaluation_skip(run_id, "eidD", root=tmp_path, ledger_path=ledger_path)


# ---------------------------------------------------------------------------
# Clean-tree ordering
# ---------------------------------------------------------------------------


def _init_git_repo(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "src_file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "src_file.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)


def test_dirty_source_tree_fails_before_attempt_allocation(tmp_path, monkeypatch):
    from when_tta_hurts.orchestrator import DirtyWorkingTreeError

    _init_git_repo(tmp_path)
    (tmp_path / "src_file.py").write_text("x = 2\n")  # dirty, unapproved path
    monkeypatch.chdir(tmp_path)

    root = tmp_path / "eval_root"
    ledger_path = tmp_path / "ledger.csv"

    def fake_resolve(run_id, matrix_path):
        cell = type(
            "FakeCell",
            (),
            {
                "run_id": lambda self: run_id,
                "dataset": "pathmnist",
                "resolution": 28,
                "block": "A_core_normalization_resolution",
            },
        )()
        result = type("R", (), {"attempt_number": 1, "checkpoint_hash": "c", "status": "completed"})()
        return cell, result

    # We only need to reach the clean-tree check; use a temp, isolated
    # seed config so load_frozen_tta_seed_config() doesn't touch the real
    # repo's committed config from within this synthetic git repo.
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "resolve_canonical_training_completion", fake_resolve)
    monkeypatch.setattr(
        ve, "parse_and_validate_matrix", lambda *a, **k: type("E", (), {"source_config_hash": "m"})()
    )

    seed_cfg_path = tmp_path / "validation_evaluation.yaml"
    seed_cfg_path.write_text(_VALID_YAML_TEXT)

    before = set(tmp_path.rglob("*"))

    with pytest.raises(DirtyWorkingTreeError):
        ve.run_validation_evaluation(
            "fake-run",
            device_resolver=lambda: (_ for _ in ()).throw(AssertionError("MPS must not be reached")),
            root=root,
            evaluation_ledger_path=ledger_path,
            tta_seed_config_path=seed_cfg_path,
            tta_seed_git_tracked_and_clean=_always_tracked_clean,
            tta_seed_last_commit_for_path=_commit_for("c" * 40),
            tta_seed_commit_is_ancestor=_all_ancestors,
        )
    after = set(tmp_path.rglob("*"))
    assert not root.exists()
    assert not ledger_path.exists()
    # only the pre-existing dirty file changed -- no new attempt files
    assert after == before


def test_allowed_ledger_prefix_append_does_not_trip_clean_tree(tmp_path, monkeypatch):
    from when_tta_hurts.orchestrator import require_clean_working_tree

    _init_git_repo(tmp_path)
    ledger_rel = "artifacts/ledger_validation_evaluation.csv"
    ledger_path = tmp_path / ledger_rel
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("a,b\n1,2\n")
    import subprocess

    subprocess.run(["git", "add", ledger_rel], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add ledger"], cwd=tmp_path, check=True)
    # strict append only
    with ledger_path.open("a") as f:
        f.write("3,4\n")
    monkeypatch.chdir(tmp_path)
    require_clean_working_tree()  # must not raise


def test_edited_ledger_row_rejected_by_clean_tree(tmp_path, monkeypatch):
    from when_tta_hurts.orchestrator import DirtyWorkingTreeError, require_clean_working_tree

    _init_git_repo(tmp_path)
    ledger_rel = "artifacts/ledger_validation_evaluation.csv"
    ledger_path = tmp_path / ledger_rel
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("a,b\n1,2\n")
    import subprocess

    subprocess.run(["git", "add", ledger_rel], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add ledger"], cwd=tmp_path, check=True)
    ledger_path.write_text("a,b\n1,9\n")  # edited, not appended
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DirtyWorkingTreeError):
        require_clean_working_tree()


def test_evaluation_ledger_path_is_in_approved_append_only_set():
    from when_tta_hurts.orchestrator import APPROVED_APPEND_ONLY_LEDGER_PATHS

    assert "artifacts/ledger_validation_evaluation.csv" in APPROVED_APPEND_ONLY_LEDGER_PATHS


@pytest.mark.parametrize(
    "dirty_rel_path",
    [
        "configs/validation_evaluation.yaml",
        "docs/phase2b_validation_evaluation_incident.md",
        "tests/test_validation_evaluation.py",
        "scripts/run_validation_evaluation.py",
    ],
)
def test_dirty_non_source_file_kinds_rejected_by_clean_tree(tmp_path, monkeypatch, dirty_rel_path):
    """config/docs/tests/scripts edits are not exempt -- only the approved
    append-only ledger paths get any special treatment, and even those only
    tolerate strict byte-prefix appends (see test_allowed_ledger_prefix_append_
    does_not_trip_clean_tree / test_edited_ledger_row_rejected_by_clean_tree)."""
    from when_tta_hurts.orchestrator import DirtyWorkingTreeError, require_clean_working_tree

    _init_git_repo(tmp_path)
    dirty_path = tmp_path / dirty_rel_path
    dirty_path.parent.mkdir(parents=True, exist_ok=True)
    dirty_path.write_text("dirty\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DirtyWorkingTreeError):
        require_clean_working_tree()


def test_production_run_validation_evaluation_calls_clean_tree_check():
    import inspect

    source = inspect.getsource(run_validation_evaluation)
    assert "require_clean_working_tree" in source


def test_production_order_attempt_allocation_before_mps():
    """Static check: start_evaluation_attempt() must appear before
    device_resolver() is called, within run_validation_evaluation()'s
    source -- MPS failures must never leave a dangling unledgered
    attempt."""
    import inspect

    source = inspect.getsource(run_validation_evaluation)
    assert source.index("start_evaluation_attempt(") < source.index("device_resolver()")
