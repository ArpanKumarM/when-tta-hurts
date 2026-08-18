"""Tests for block_d_benchmark.py -- the native-128px Block D runtime-gate
benchmark pipeline (docs/phase2b_block_d_benchmark_spec.md).

Uses ONLY synthetic tensors, fake NPZ-shaped fixtures, monkeypatched
metadata, fake timers, and fake device/model objects -- never real 128px
data, never real MPS initialization, never a real download. Boundary
thresholds (90min/120min/24h) and the pure gate decision logic are already
exhaustively tested in tests/test_block_d_gate.py; this file tests the
benchmark plumbing that feeds that gate.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts import block_d_benchmark as bdb
from when_tta_hurts.dataset_verification import ArtifactVerification, ArtifactVerificationError
from when_tta_hurts.devices import DeviceUnavailableError


def _fake_verification(dataset: str, resolution: int, *, resized: bool = False) -> ArtifactVerification:
    return ArtifactVerification(
        dataset=dataset,
        native_resolution=resolution,
        artifact_path=f"data/raw/{dataset}_{resolution}.npz",
        expected_checksum_md5="deadbeef",
        actual_checksum_md5="deadbeef",
        checksum_verified=True,
        resized=resized,
    )


def _synthetic_loaders(
    n_classes: int, batch_size: int, n_train: int = 40, n_val: int = 8
) -> bdb.BlockDLoaders:
    train_x = torch.zeros(n_train, 3, 128, 128)
    train_y = torch.randint(0, n_classes, (n_train,))
    val_x = torch.zeros(n_val, 3, 128, 128)
    val_y = torch.randint(0, n_classes, (n_val,))
    return bdb.BlockDLoaders(
        train_loader=DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False),
        train_split_size=n_train,
        val_split_size=n_val,
    )


class _NaNModel(nn.Module):
    """Fake model producing non-finite logits, to test non-finite handling
    without needing real data to actually diverge."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 128 * 128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.reshape(x.shape[0], -1))
        return out * float("nan")


class _OOMModel(nn.Module):
    """Fake model that raises an MPS-style OOM RuntimeError on forward."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 128 * 128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("MPS backend out of memory (MPSAllocator)")


# ---------------------------------------------------------------------------
# Data rules: native-128 requirement, resized-proxy/synthetic rejection
# ---------------------------------------------------------------------------


def test_default_loader_factory_rejects_resized_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=True),
    )
    with pytest.raises(bdb.SyntheticInputRejectedError):
        bdb.default_block_d_loader_factory("pathmnist", 128, 64, root=tmp_path)


def test_default_loader_factory_rejects_non_native_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, 64, resized=False),
    )
    with pytest.raises(bdb.NonNative128Error):
        bdb.default_block_d_loader_factory("pathmnist", 128, 64, root=tmp_path)


def test_default_loader_factory_propagates_checksum_mismatch(monkeypatch, tmp_path):
    def _raise(ds, res, root):
        raise ArtifactVerificationError("CHECKSUM MISMATCH")

    monkeypatch.setattr(bdb, "verify_official_dataset_artifact", _raise)
    with pytest.raises(ArtifactVerificationError, match="CHECKSUM MISMATCH"):
        bdb.default_block_d_loader_factory("pathmnist", 128, 64, root=tmp_path)


def test_benchmark_one_dataset_rejects_resized_proxy_before_any_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=True),
    )
    with pytest.raises(bdb.SyntheticInputRejectedError):
        bdb._benchmark_one_dataset(
            "pathmnist", lambda: torch.device("cpu"), None, tmp_path, tmp_path / "bench"
        )


def test_benchmark_one_dataset_rejects_non_native_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, 64, resized=False),
    )
    with pytest.raises(bdb.NonNative128Error):
        bdb._benchmark_one_dataset(
            "pathmnist", lambda: torch.device("cpu"), None, tmp_path, tmp_path / "bench"
        )


def test_zero_test_split_access_in_default_loader_factory():
    """Static check: the production loader factory must never request
    split='test' -- load_pilot_split has no such mechanism, but this
    confirms the call sites never even try."""
    import inspect

    source = inspect.getsource(bdb.default_block_d_loader_factory)
    assert 'split="test"' not in source
    assert "split='test'" not in source
    assert 'split="train"' in source
    assert 'split="val"' in source


def test_prefetch_never_called_implicitly_by_plan_or_benchmark():
    import inspect

    plan_source = inspect.getsource(bdb.plan_block_d_benchmark)
    benchmark_source = inspect.getsource(bdb.run_block_d_benchmark)
    assert "prefetch_block_d_artifacts" not in plan_source
    assert "prefetch_block_d_artifacts" not in benchmark_source


# ---------------------------------------------------------------------------
# Model rules: dataset-specific class counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset,expected_classes", [("pathmnist", 9), ("bloodmnist", 8)])
def test_dataset_specific_class_count(monkeypatch, tmp_path, dataset, expected_classes):
    monkeypatch.setattr(bdb, "BATCH_CANDIDATES", (2,))
    monkeypatch.setattr(bdb, "WARMUP_STEPS", 1)
    monkeypatch.setattr(bdb, "MEASURED_STEPS", 2)
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    captured_n_classes = {}
    real_build = bdb.build_small_cnn

    def _spy_build(num_classes, normalization="batchnorm"):
        captured_n_classes["value"] = num_classes
        return real_build(num_classes=num_classes, normalization=normalization)

    monkeypatch.setattr(bdb, "build_small_cnn", _spy_build)

    def loader_factory(ds, resolution, batch_size, root):
        return _synthetic_loaders(expected_classes, batch_size, n_train=8, n_val=4)

    bdb._benchmark_one_dataset(
        dataset, lambda: torch.device("cpu"), loader_factory, tmp_path, tmp_path / "bench"
    )
    assert captured_n_classes["value"] == expected_classes


# ---------------------------------------------------------------------------
# Batch-candidate selection: deterministic order, safe-selection, all-failed
# ---------------------------------------------------------------------------


def test_batch_candidates_are_frozen_ascending_order():
    assert bdb.BATCH_CANDIDATES == (64, 128, 256)


def test_deterministic_candidate_order_in_results(monkeypatch, tmp_path):
    monkeypatch.setattr(bdb, "BATCH_CANDIDATES", (2, 4, 8))
    monkeypatch.setattr(bdb, "WARMUP_STEPS", 1)
    monkeypatch.setattr(bdb, "MEASURED_STEPS", 2)
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def loader_factory(ds, resolution, batch_size, root):
        return _synthetic_loaders(9, batch_size, n_train=16, n_val=4)

    result = bdb._benchmark_one_dataset(
        "pathmnist", lambda: torch.device("cpu"), loader_factory, tmp_path, tmp_path / "bench"
    )
    assert [c["batch_size"] for c in result["candidate_results"]] == [2, 4, 8]


def test_largest_safe_candidate_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(bdb, "BATCH_CANDIDATES", (2, 4, 8))
    monkeypatch.setattr(bdb, "WARMUP_STEPS", 1)
    monkeypatch.setattr(bdb, "MEASURED_STEPS", 2)
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def loader_factory(ds, resolution, batch_size, root):
        return _synthetic_loaders(9, batch_size, n_train=16, n_val=4)

    result = bdb._benchmark_one_dataset(
        "pathmnist", lambda: torch.device("cpu"), loader_factory, tmp_path, tmp_path / "bench"
    )
    assert result["selected_batch_size"] == 8
    assert all(c["safe"] for c in result["candidate_results"])


def test_no_safe_candidate_fails_entire_dataset_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(bdb, "BATCH_CANDIDATES", (2, 4))
    monkeypatch.setattr(bdb, "WARMUP_STEPS", 1)
    monkeypatch.setattr(bdb, "MEASURED_STEPS", 2)
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def loader_factory(ds, resolution, batch_size, root):
        loaders = _synthetic_loaders(9, batch_size, n_train=16, n_val=4)
        return loaders

    def device_resolver():
        return torch.device("cpu")

    def fake_benchmark_training_steps(model, loader, device, batch_size):
        return {
            "status": "ok",
            "oom_occurred": False,
            "non_finite_loss_occurred": True,
            "raw_step_times_seconds": [0.01],
            "mean_step_seconds": 0.01,
            "median_step_seconds": 0.01,
            "peak_memory": {},
            "memory_fraction_of_recommended_max": None,
            "within_safe_memory_fraction": None,
        }

    monkeypatch.setattr(bdb, "_benchmark_training_steps", fake_benchmark_training_steps)

    with pytest.raises(bdb.AllBatchCandidatesFailedError):
        bdb._benchmark_one_dataset("pathmnist", device_resolver, loader_factory, tmp_path, tmp_path / "bench")


# ---------------------------------------------------------------------------
# OOM / non-finite-loss handling
# ---------------------------------------------------------------------------


def test_oom_marks_candidate_unsafe_and_sets_flag():
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=4)
    result = bdb._benchmark_training_steps(_OOMModel(9), loaders.train_loader, device, batch_size=2)
    assert result["status"] == "failed"
    assert result["oom_occurred"] is True


def test_non_finite_loss_marks_candidate_unsafe_and_sets_flag():
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=4)
    result = bdb._benchmark_training_steps(_NaNModel(9), loaders.train_loader, device, batch_size=2)
    assert result["status"] == "failed"
    assert result["non_finite_loss_occurred"] is True


def test_validation_non_finite_loss_detected():
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=4)
    result = bdb._benchmark_validation_steps(_NaNModel(9), loaders.val_loader, device)
    assert result["status"] == "failed"
    assert result["non_finite_loss_occurred"] is True


# ---------------------------------------------------------------------------
# 70% safe-memory-fraction boundary (direct unit test, no real MPS needed)
# ---------------------------------------------------------------------------


def test_memory_fraction_at_boundary_is_safe(monkeypatch):
    monkeypatch.setattr(
        bdb,
        "_mps_memory_snapshot",
        lambda: {
            "current_allocated_bytes": 700,
            "driver_allocated_bytes": 700,
            "recommended_max_bytes": 1000,
        },
    )
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=4)
    result = bdb._benchmark_training_steps(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 128 * 128, 9)), loaders.train_loader, device, batch_size=2
    )
    assert result["within_safe_memory_fraction"] is True


def test_memory_fraction_just_over_boundary_is_unsafe(monkeypatch):
    monkeypatch.setattr(
        bdb,
        "_mps_memory_snapshot",
        lambda: {
            "current_allocated_bytes": 701,
            "driver_allocated_bytes": 701,
            "recommended_max_bytes": 1000,
        },
    )
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=4)
    result = bdb._benchmark_training_steps(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 128 * 128, 9)), loaders.train_loader, device, batch_size=2
    )
    assert result["within_safe_memory_fraction"] is False


# ---------------------------------------------------------------------------
# Projection formulas (exact, per docs/phase2b_block_d_benchmark_spec.md sec.7)
# ---------------------------------------------------------------------------


def test_projection_formulas_exact(monkeypatch, tmp_path):
    monkeypatch.setattr(bdb, "BATCH_CANDIDATES", (4,))
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def loader_factory(ds, resolution, batch_size, root):
        return _synthetic_loaders(9, batch_size, n_train=40, n_val=8)

    def fake_train(model, loader, device, batch_size):
        return {
            "status": "ok",
            "oom_occurred": False,
            "non_finite_loss_occurred": False,
            "raw_step_times_seconds": [0.1] * 30,
            "mean_step_seconds": 0.1,
            "median_step_seconds": 0.1,
            "peak_memory": {},
            "memory_fraction_of_recommended_max": None,
            "within_safe_memory_fraction": None,
        }

    def fake_val(model, loader, device):
        return {
            "status": "ok",
            "non_finite_loss_occurred": False,
            "raw_step_times_seconds": [0.05] * 30,
            "mean_step_seconds": 0.05,
            "median_step_seconds": 0.05,
            "validation_batches_available": 2,
        }

    def fake_persistence(model, benchmark_dir, n_classes):
        return 2.0

    monkeypatch.setattr(bdb, "_benchmark_training_steps", fake_train)
    monkeypatch.setattr(bdb, "_benchmark_validation_steps", fake_val)
    monkeypatch.setattr(bdb, "_measure_persistence_overhead", fake_persistence)

    result = bdb._benchmark_one_dataset(
        "pathmnist", lambda: torch.device("cpu"), loader_factory, tmp_path, tmp_path / "bench"
    )

    train_batches_per_epoch = -(-40 // 4)
    val_batches_per_epoch = -(-8 // 4)
    expected_training = 30 * train_batches_per_epoch * 0.1
    expected_validation = 30 * val_batches_per_epoch * 0.05

    assert result["train_batches_per_epoch"] == train_batches_per_epoch
    assert result["validation_batches_per_epoch"] == val_batches_per_epoch
    assert result["projected_training_seconds"] == pytest.approx(expected_training)
    assert result["projected_validation_seconds"] == pytest.approx(expected_validation)
    assert result["persistence_verification_seconds"] == 2.0
    expected_end_to_end = (
        result["candidate_results"][0]["setup_seconds"] + expected_training + expected_validation + 2.0
    )
    assert result["projected_end_to_end_seconds"] == pytest.approx(expected_end_to_end)


def test_validation_split_smaller_than_measured_steps_cycles_without_crashing():
    device = torch.device("cpu")
    loaders = _synthetic_loaders(9, batch_size=2, n_train=8, n_val=2)  # only 1 validation batch
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 128 * 128, 9))
    result = bdb._benchmark_validation_steps(model, loaders.val_loader, device)
    assert result["status"] == "ok"
    assert len(result["raw_step_times_seconds"]) == bdb.MEASURED_STEPS


# ---------------------------------------------------------------------------
# Persistence overhead: real production code path, disposable directory only
# ---------------------------------------------------------------------------


def test_persistence_overhead_uses_disposable_dir_and_is_positive(tmp_path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 128 * 128, 9))
    # give the model the exact param shapes build_small_cnn(num_classes=9) would produce
    from when_tta_hurts.models.small_cnn import build_small_cnn

    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    benchmark_dir = tmp_path / "disposable" / "pathmnist"
    seconds = bdb._measure_persistence_overhead(model, benchmark_dir, n_classes=9)
    assert seconds > 0
    assert benchmark_dir.exists()
    assert (benchmark_dir / "best_checkpoint.pt").exists()
    assert "artifacts/confirmatory" not in str(benchmark_dir)


# ---------------------------------------------------------------------------
# Both datasets required; all-or-nothing decision; result schema
# ---------------------------------------------------------------------------


def _stub_full_pipeline(monkeypatch, tmp_path, *, pathmnist_ok=True, bloodmnist_ok=True):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def fake_benchmark_one_dataset(dataset, device_resolver, loader_factory, root, benchmark_root):
        n_classes = 9 if dataset == "pathmnist" else 8
        ok = pathmnist_ok if dataset == "pathmnist" else bloodmnist_ok
        return {
            "dataset": dataset,
            "n_classes": n_classes,
            "artifact_path": f"data/raw/{dataset}_128.npz",
            "expected_checksum_md5": "deadbeef",
            "actual_checksum_md5": "deadbeef" if ok else "wrong",
            "resized": False,
            "device": "mps",
            "candidate_results": [],
            "selected_batch_size": 64,
            "train_batches_per_epoch": 10,
            "validation_batches_per_epoch": 2,
            "persistence_verification_seconds": 1.0,
            "projected_training_seconds": 60.0,
            "projected_validation_seconds": 10.0,
            "projected_end_to_end_seconds": 75.0,
            "oom_occurred": False,
            "non_finite_loss_occurred": False,
        }

    monkeypatch.setattr(bdb, "_benchmark_one_dataset", fake_benchmark_one_dataset)


def test_both_datasets_benchmarked_for_all_or_nothing_decision(monkeypatch, tmp_path):
    _stub_full_pipeline(monkeypatch, tmp_path)
    output_path = tmp_path / "decision.json"
    result = bdb.run_block_d_benchmark(
        device_resolver=lambda: torch.device("cpu"),
        output_path=output_path,
    )
    assert set(result["per_dataset"].keys()) == {"pathmnist", "bloodmnist"}
    assert result["final_decision"] in ("INCLUDED", "OMITTED")


def test_one_dataset_failing_checksum_omits_entire_block(monkeypatch, tmp_path):
    _stub_full_pipeline(monkeypatch, tmp_path, pathmnist_ok=False)
    output_path = tmp_path / "decision.json"
    result = bdb.run_block_d_benchmark(
        device_resolver=lambda: torch.device("cpu"),
        output_path=output_path,
    )
    assert result["final_decision"] == "OMITTED"
    assert result["per_dataset_pass"]["pathmnist"] is False
    assert result["per_dataset_pass"]["bloodmnist"] is True


def test_result_schema_contains_no_scientific_metric_fields(monkeypatch, tmp_path):
    _stub_full_pipeline(monkeypatch, tmp_path)
    output_path = tmp_path / "decision.json"
    result = bdb.run_block_d_benchmark(
        device_resolver=lambda: torch.device("cpu"),
        output_path=output_path,
    )
    serialized = json.dumps(result).lower()
    for term in ("accuracy", "f1", "nll", "ece", "brier", "tta_delta", "test_metric"):
        assert term not in serialized


def test_validate_output_schema_rejects_forbidden_term():
    output = {
        "schema_version": "1.0",
        "source_commit": "abc",
        "protocol_commit": "def",
        "matrix_hash": "ghi",
        "per_dataset": {"note": "accuracy leaked in here"},
        "gate_conditions": [],
        "per_dataset_pass": {},
        "activated": False,
        "final_decision": "OMITTED",
    }
    with pytest.raises(bdb.IncompleteBenchmarkResultError):
        bdb._validate_output_schema(output)


def test_validate_output_schema_rejects_missing_field():
    with pytest.raises(bdb.IncompleteBenchmarkResultError):
        bdb._validate_output_schema({"schema_version": "1.0"})


def test_atomic_output_write_produces_valid_json_no_tmp_leftover(monkeypatch, tmp_path):
    _stub_full_pipeline(monkeypatch, tmp_path)
    output_path = tmp_path / "sub" / "decision.json"
    bdb.run_block_d_benchmark(device_resolver=lambda: torch.device("cpu"), output_path=output_path)
    assert output_path.exists()
    json.loads(output_path.read_text())
    assert not output_path.with_suffix(output_path.suffix + ".tmp").exists()


def test_no_ledger_or_confirmatory_artifact_writes(monkeypatch, tmp_path):
    _stub_full_pipeline(monkeypatch, tmp_path)
    output_path = tmp_path / "decision.json"
    bdb.run_block_d_benchmark(device_resolver=lambda: torch.device("cpu"), output_path=output_path)
    assert not (tmp_path / "confirmatory").exists()
    assert not (tmp_path / "ledger.csv").exists()
    assert not (tmp_path / "ledger_confirmatory.csv").exists()


def test_run_block_d_benchmark_fails_closed_on_device_unavailable():
    def raising_resolver():
        raise DeviceUnavailableError("mps not available")

    with pytest.raises(DeviceUnavailableError):
        bdb.run_block_d_benchmark(device_resolver=raising_resolver)


def test_all_batch_candidates_failed_prevents_any_decision_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bdb,
        "verify_official_dataset_artifact",
        lambda ds, res, root: _fake_verification(ds, res, resized=False),
    )

    def fake_benchmark_one_dataset(dataset, device_resolver, loader_factory, root, benchmark_root):
        raise bdb.AllBatchCandidatesFailedError(f"{dataset}: no safe candidate")

    monkeypatch.setattr(bdb, "_benchmark_one_dataset", fake_benchmark_one_dataset)
    output_path = tmp_path / "decision.json"
    with pytest.raises(bdb.AllBatchCandidatesFailedError):
        bdb.run_block_d_benchmark(device_resolver=lambda: torch.device("cpu"), output_path=output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Plan mode: side-effect-free
# ---------------------------------------------------------------------------


def test_plan_mode_creates_no_files_or_directories(tmp_path):
    before = set(tmp_path.rglob("*"))
    result = bdb.plan_block_d_benchmark()
    after = set(tmp_path.rglob("*"))
    assert before == after
    assert result["datasets"] == ["pathmnist", "bloodmnist"]
    assert result["resolution"] == 128
    assert len(result["cells"]) == 6


def test_plan_mode_never_touches_mps():
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(bdb.plan_block_d_benchmark))
    tree = ast.parse(source)
    func_body = tree.body[0].body
    # skip the docstring (first statement) -- only inspect executable code
    code_nodes = func_body[1:] if isinstance(func_body[0], ast.Expr) else func_body
    code_source = "\n".join(ast.unparse(node) for node in code_nodes)
    assert "mps" not in code_source.lower()


def test_plan_mode_reports_frozen_formulas():
    result = bdb.plan_block_d_benchmark()
    formulas = result["formulas"]
    assert (
        formulas["projected_training_seconds"] == "30 * train_batches_per_epoch * mean_training_step_seconds"
    )
    assert (
        formulas["projected_validation_seconds"]
        == "30 * validation_batches_per_epoch * mean_validation_step_seconds"
    )
    assert formulas["training_gate_seconds"] == 90 * 60
    assert formulas["cell_gate_seconds"] == 120 * 60
    assert formulas["pessimistic_total_hours_ceiling"] == 24.0
    assert formulas["frozen_pessimistic_abc_hours"] == 3.92


# ---------------------------------------------------------------------------
# Future Block D training consumption point: hard-fail behavior
# ---------------------------------------------------------------------------


def _write_decision(tmp_path, **overrides):
    decision = {
        "final_decision": "INCLUDED",
        "matrix_hash": "matrixhash123",
        "protocol_commit": bdb.FROZEN_PROTOCOL_COMMIT,
        "per_dataset": {"pathmnist": {"selected_batch_size": 128}},
    }
    decision.update(overrides)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    return path


def test_missing_decision_file_blocks_training(tmp_path):
    with pytest.raises(bdb.BlockDDecisionError):
        bdb.load_and_verify_block_d_decision(tmp_path / "nope.json", expected_matrix_hash="x")


def test_omitted_decision_blocks_training(tmp_path):
    path = _write_decision(tmp_path, final_decision="OMITTED")
    with pytest.raises(bdb.BlockDDecisionError):
        bdb.load_and_verify_block_d_decision(path, expected_matrix_hash="matrixhash123")


def test_matrix_hash_mismatch_blocks_training(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(bdb.BlockDDecisionError):
        bdb.load_and_verify_block_d_decision(path, expected_matrix_hash="different_hash")


def test_protocol_commit_mismatch_blocks_training(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(bdb.BlockDDecisionError):
        bdb.load_and_verify_block_d_decision(
            path, expected_matrix_hash="matrixhash123", expected_protocol_commit="different_commit"
        )


def test_batch_size_mismatch_blocks_training(tmp_path):
    path = _write_decision(tmp_path)
    with pytest.raises(bdb.BlockDDecisionError):
        bdb.load_and_verify_block_d_decision(
            path, expected_matrix_hash="matrixhash123", dataset="pathmnist", requested_batch_size=64
        )


def test_matching_included_decision_loads_successfully(tmp_path):
    path = _write_decision(tmp_path)
    decision = bdb.load_and_verify_block_d_decision(
        path, expected_matrix_hash="matrixhash123", dataset="pathmnist", requested_batch_size=128
    )
    assert decision["final_decision"] == "INCLUDED"


def test_decision_loader_not_wired_into_any_training_entry_point():
    """Confirm this task did not unlock Block D training: orchestrator.py's
    block-execution entry points must not reference the decision loader."""
    import inspect

    from when_tta_hurts import orchestrator

    orchestrator_source = inspect.getsource(orchestrator)
    assert "load_and_verify_block_d_decision" not in orchestrator_source


# ---------------------------------------------------------------------------
# No CLI/env-var synthetic-backend bypass
# ---------------------------------------------------------------------------


def test_no_environment_variable_bypass_in_production_paths():
    import inspect

    for fn in (bdb.default_block_d_loader_factory, bdb.run_block_d_benchmark, bdb._benchmark_one_dataset):
        source = inspect.getsource(fn)
        assert "os.environ" not in source
        assert "getenv" not in source
