"""Phase 2B.3B Part 1 tests: sequential block execution and metadata-only
verify-completions. Uses the REAL committed matrix (its cell counts are
frozen and hard-validated -- a shrunk synthetic matrix would fail
validation), but a completely synthetic EXECUTION environment: temporary
attempt roots, temporary ledgers, a CPU device resolver, and a synthetic
in-memory loader factory. Never touches the real attempt_001-004 of
A-pathmnist-28px-batchnorm-policy-none-s0, the real artifacts/confirmatory
tree, the real ledgers, or any real dataset file."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import when_tta_hurts.orchestrator as orch
from when_tta_hurts.matrix import parse_and_validate_matrix
from when_tta_hurts.run_identity import run_directory

REAL_MATRIX = "configs/experiment_matrix.yaml"


def _synthetic_loader_factory(cell):
    from when_tta_hurts.data import get_dataset_metadata

    n_classes = get_dataset_metadata(cell.dataset).n_classes
    g = torch.Generator().manual_seed(cell.seed)
    x = torch.rand(16, 3, cell.resolution, cell.resolution, generator=g)
    y = torch.randint(0, n_classes, (16,), generator=g)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=8, shuffle=False)
    return orch.TrainValidationLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_artifact_filename="synthetic.npz",
        dataset_expected_checksum_md5="synthetic",
        dataset_actual_checksum_md5="synthetic",
    )


def _run_block(tmp_path, block, expected_total, expected_pending, **kwargs):
    return orch.run_block_cells(
        block,
        expected_total=expected_total,
        expected_pending=expected_pending,
        matrix_path=REAL_MATRIX,
        loader_factory=_synthetic_loader_factory,
        device_resolver=lambda: torch.device("cpu"),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
        **kwargs,
    )


# --- order/count (against the real, frozen 24-cell Block A) ---


def test_exact_block_a_order_and_count():
    expanded = parse_and_validate_matrix(REAL_MATRIX, block_d_gate_passed=False)
    cells = expanded.cells_by_block["A_core_normalization_resolution"]
    assert len(cells) == 24
    run_ids = [c.run_id() for c in cells]
    assert run_ids[0] == "A-pathmnist-28px-batchnorm-policy-none-s0"
    assert len(set(run_ids)) == 24  # every run ID unique
    assert all(c.training_policy == "none" for c in cells)
    assert all(c.seed in (0, 1, 2) for c in cells)
    assert all(c.seed != 314159 for c in cells)


def test_expected_total_mismatch_hard_fails_before_data(tmp_path):
    def _forbidden(cell):
        raise AssertionError("loader_factory must never be called on a count mismatch")

    with pytest.raises(ValueError, match="expected-total mismatch"):
        orch.run_block_cells(
            "A",
            expected_total=999,
            expected_pending=999,
            matrix_path=REAL_MATRIX,
            loader_factory=_forbidden,
            device_resolver=_forbidden,
            require_clean_tree=False,
            root=str(tmp_path / "confirmatory"),
        )


def test_expected_pending_mismatch_hard_fails_before_data(tmp_path):
    def _forbidden(cell):
        raise AssertionError("loader_factory must never be called on a count mismatch")

    with pytest.raises(ValueError, match="expected-pending mismatch"):
        orch.run_block_cells(
            "A",
            expected_total=24,
            expected_pending=1,  # wrong -- all 24 are pending in a fresh tmp root
            matrix_path=REAL_MATRIX,
            loader_factory=_forbidden,
            device_resolver=_forbidden,
            require_clean_tree=False,
            root=str(tmp_path / "confirmatory"),
        )


# --- skip / sequencing / fresh state (limited to the first few cells to keep tests fast) ---


def _first_n_cells(n):
    expanded = parse_and_validate_matrix(REAL_MATRIX, block_d_gate_passed=False)
    return list(expanded.cells_by_block["A_core_normalization_resolution"])[:n]


def _subset_matrix_factory(cells):
    """Returns a parse_and_validate_matrix replacement that yields the
    REAL expanded matrix (preserving source_config_hash and every other
    field run_train_validation_cell/run_block_cells depend on) but with
    Block A's cell tuple restricted to `cells` -- lets tests exercise only
    the first few real cells without waiting on all 24."""
    real_expanded = parse_and_validate_matrix(REAL_MATRIX, block_d_gate_passed=False)

    def _fake(*args, **kwargs):
        import dataclasses

        return dataclasses.replace(
            real_expanded,
            cells_by_block={
                **real_expanded.cells_by_block,
                "A_core_normalization_resolution": tuple(cells),
            },
        )

    return _fake


def test_existing_canonical_cell_skipped_before_data(tmp_path, monkeypatch):
    """Restrict execution to the first 2 cells via a monkeypatched matrix
    expansion, to keep this test fast while proving the skip contract."""
    cells = _first_n_cells(2)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))

    results = _run_block(tmp_path, "A", expected_total=2, expected_pending=2)
    assert all(r.status == "completed" for r in results)

    root = tmp_path / "confirmatory"
    attempt_dirs_before = sorted(p.name for p in run_directory(cells[0], root).iterdir())

    results2 = _run_block(tmp_path, "A", expected_total=2, expected_pending=0)
    assert all(r.status == "skipped_completed" for r in results2)
    attempt_dirs_after = sorted(p.name for p in run_directory(cells[0], root).iterdir())
    assert attempt_dirs_before == attempt_dirs_after == ["attempt_001"]


def test_sequential_execution_order_matches_matrix(tmp_path, monkeypatch):
    cells = _first_n_cells(3)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    order = []
    real_run = orch.run_train_validation_cell

    def spy_run(cell, *args, **kwargs):
        order.append(cell.run_id())
        return real_run(cell, *args, **kwargs)

    monkeypatch.setattr(orch, "run_train_validation_cell", spy_run)
    _run_block(tmp_path, "A", expected_total=3, expected_pending=3)
    assert order == [c.run_id() for c in cells]


def test_fresh_seed_model_loader_state_per_cell(tmp_path, monkeypatch):
    """Two cells identical except seed must NOT produce identical checkpoints."""
    cells = _first_n_cells(2)  # seeds s0, s1 for the same dataset/res/norm
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    results = _run_block(tmp_path, "A", expected_total=2, expected_pending=2)
    assert results[0].checkpoint_hash != results[1].checkpoint_hash


def test_no_cross_cell_state_leakage(tmp_path, monkeypatch):
    """Two independent invocations from scratch must reproduce identical
    per-cell checkpoints -- proving no cell depends on execution-order
    side effects or leaked RNG state."""
    cells = _first_n_cells(2)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    results_a = _run_block(tmp_path / "run_a", "A", expected_total=2, expected_pending=2)
    results_b = _run_block(tmp_path / "run_b", "A", expected_total=2, expected_pending=2)
    hashes_a = {r.run_id: r.checkpoint_hash for r in results_a}
    hashes_b = {r.run_id: r.checkpoint_hash for r in results_b}
    assert hashes_a == hashes_b


def test_stop_on_first_failure_preserves_earlier_cells(tmp_path, monkeypatch):
    cells = _first_n_cells(3)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    call_count = {"n": 0}
    real_run = orch.run_train_validation_cell

    def flaky_run(cell, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure on cell 2")
        return real_run(cell, *args, **kwargs)

    monkeypatch.setattr(orch, "run_train_validation_cell", flaky_run)

    with pytest.raises(RuntimeError, match="simulated failure"):
        _run_block(tmp_path, "A", expected_total=3, expected_pending=3)

    root = tmp_path / "confirmatory"
    assert run_directory(cells[0], root).exists()  # cell 1 preserved
    assert not run_directory(cells[2], root).exists()  # cell 3 never reached


def test_resume_safely_after_earlier_completed_cells(tmp_path, monkeypatch):
    cells = _first_n_cells(3)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    call_count = {"n": 0}
    real_run = orch.run_train_validation_cell

    def flaky_run(cell, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure")
        return real_run(cell, *args, **kwargs)

    monkeypatch.setattr(orch, "run_train_validation_cell", flaky_run)
    with pytest.raises(RuntimeError):
        _run_block(tmp_path, "A", expected_total=3, expected_pending=3)

    monkeypatch.setattr(orch, "run_train_validation_cell", real_run)
    results = _run_block(tmp_path, "A", expected_total=3, expected_pending=2)
    assert results[0].status == "skipped_completed"
    assert all(r.status == "completed" for r in results[1:])

    root = tmp_path / "confirmatory"
    attempt_dirs = sorted(p.name for p in run_directory(cells[0], root).iterdir())
    assert attempt_dirs == ["attempt_001"]  # no duplicate on resume


# --- working-tree policy within block execution ---


def test_append_only_ledger_changes_accepted_within_block(tmp_path, monkeypatch):
    cells = _first_n_cells(1)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M artifacts/ledger_confirmatory.csv\n")
    monkeypatch.setattr(orch, "_git_show_head", lambda path: "header\n")
    monkeypatch.setattr(orch.Path, "read_text", lambda self: "header\nnew_row\n")
    results = orch.run_block_cells(
        "A",
        expected_total=1,
        expected_pending=1,
        matrix_path=REAL_MATRIX,
        loader_factory=_synthetic_loader_factory,
        device_resolver=lambda: torch.device("cpu"),
        require_clean_tree=True,
        root=str(tmp_path / "confirmatory"),
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
    )
    assert all(r.status == "completed" for r in results)


def test_dirty_source_rejected_within_block(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M src/when_tta_hurts/matrix.py\n")

    def _forbidden(cell):
        raise AssertionError("loader_factory must never be called with a dirty source tree")

    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.run_block_cells(
            "A",
            expected_total=24,
            expected_pending=24,
            matrix_path=REAL_MATRIX,
            loader_factory=_forbidden,
            device_resolver=_forbidden,
            require_clean_tree=True,
            root=str(tmp_path / "confirmatory"),
        )


# --- block letter exclusion / rejection ---


def test_block_b_excluded_from_block_a_run(tmp_path):
    results = orch.run_block_cells(
        "B",
        expected_total=6,
        expected_pending=6,
        matrix_path=REAL_MATRIX,
        loader_factory=_synthetic_loader_factory,
        device_resolver=lambda: torch.device("cpu"),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
    )
    assert len(results) == 6
    assert all(r.run_id.startswith("B-") for r in results)
    a_dir = tmp_path / "confirmatory" / "A"
    assert not a_dir.exists()


def test_block_c_excluded_from_block_a_run(tmp_path):
    results = orch.run_block_cells(
        "C",
        expected_total=3,
        expected_pending=3,
        matrix_path=REAL_MATRIX,
        loader_factory=_synthetic_loader_factory,
        device_resolver=lambda: torch.device("cpu"),
        require_clean_tree=False,
        root=str(tmp_path / "confirmatory"),
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
    )
    assert len(results) == 3
    assert all(r.run_id.startswith("C-") for r in results)
    a_dir = tmp_path / "confirmatory" / "A"
    assert not a_dir.exists()


def test_block_d_rejected_by_run_block_cells(tmp_path):
    root = str(tmp_path / "confirmatory")
    with pytest.raises(orch.UnsupportedBlockError):
        orch.run_block_cells("D", expected_total=6, expected_pending=6, matrix_path=REAL_MATRIX, root=root)


def test_unknown_block_letter_rejected(tmp_path):
    root = str(tmp_path / "confirmatory")
    with pytest.raises(orch.UnsupportedBlockError):
        orch.run_block_cells("Z", expected_total=1, expected_pending=1, matrix_path=REAL_MATRIX, root=root)


# --- no parallel execution ---


def test_no_parallel_execution_cells_run_strictly_sequentially(tmp_path, monkeypatch):
    cells = _first_n_cells(3)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    import time

    intervals = []
    real_run = orch.run_train_validation_cell

    def timed_run(cell, *args, **kwargs):
        t0 = time.perf_counter()
        result = real_run(cell, *args, **kwargs)
        t1 = time.perf_counter()
        intervals.append((t0, t1))
        return result

    monkeypatch.setattr(orch, "run_train_validation_cell", timed_run)
    _run_block(tmp_path, "A", expected_total=3, expected_pending=3)

    for (start_a, end_a), (start_b, end_b) in zip(intervals, intervals[1:]):
        assert start_b >= end_a  # no overlap


# --- skip output includes verified hashes ---


def test_skip_output_includes_verified_hashes(tmp_path, monkeypatch):
    cells = _first_n_cells(1)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    _run_block(tmp_path, "A", expected_total=1, expected_pending=1)
    results = _run_block(tmp_path, "A", expected_total=1, expected_pending=0)
    for r in results:
        assert r.status == "skipped_completed"
        assert r.checkpoint_hash is not None
        assert r.config_hash is not None
        assert r.manifest_verified is True


# --- verify-completions: metadata-only ---


def test_verify_completions_no_device_data_model_filesystem_side_effects(tmp_path):
    root = tmp_path / "confirmatory"
    report = orch.verify_block_completions("A", expected_total=24, matrix_path=REAL_MATRIX, root=str(root))
    assert report["canonical_count"] == 0
    assert len(report["missing"]) == 24
    assert not root.exists()  # nothing was created


def test_verify_completions_reports_canonical_after_execution(tmp_path, monkeypatch):
    cells = _first_n_cells(2)
    monkeypatch.setattr(orch, "parse_and_validate_matrix", _subset_matrix_factory(cells))
    _run_block(tmp_path, "A", expected_total=2, expected_pending=2)
    report = orch.verify_block_completions(
        "A",
        expected_total=2,
        matrix_path=REAL_MATRIX,
        root=str(tmp_path / "confirmatory"),
        amendments_ledger_path=tmp_path / "ledger_amendments.csv",
    )
    assert report["canonical_count"] == 2
    assert report["missing"] == []
    assert report["ambiguous"] == []
    assert report["corrupt"] == []


def test_verify_completions_never_touches_mps_selector():
    import inspect

    source = inspect.getsource(orch.verify_block_completions)
    assert "select_device" not in source
    assert "loader_factory" not in source
    assert "_build_model" not in source


# --- no test/TTA path anywhere in block execution ---


def test_no_test_or_tta_access_in_block_execution_source():
    import inspect

    source = inspect.getsource(orch.run_block_cells) + inspect.getsource(orch.verify_block_completions)
    assert "allow_test" not in source
    assert 'split="test"' not in source
    assert "tta" not in source.lower()
