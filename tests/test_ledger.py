import csv

from when_tta_hurts.ledger import append_pilot_entry


def test_append_pilot_entry_tags_required_fields(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    append_pilot_entry(
        run_id="test-run-1",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        seed=314159,
        tta_seed=271828,
        config_hash="deadbeef",
        git_commit="abc123",
        best_epoch=3,
        epochs_completed=8,
        early_stopped=True,
        clean_val_accuracy=0.85,
        status="completed",
        artifact_dir="artifacts/pilots/test-run-1",
        ledger_path=ledger_path,
    )
    with ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["phase"] == "pilot"
    assert row["confirmatory"] == "False"
    assert row["split"] == "validation"
    assert row["seed"] == "314159"


def test_append_pilot_entry_is_append_only(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    for i in range(3):
        append_pilot_entry(
            run_id=f"run-{i}",
            dataset="pathmnist",
            resolution=28,
            model="small_cnn",
            normalization="batchnorm",
            seed=314159,
            tta_seed=271828,
            config_hash="hash",
            git_commit="commit",
            best_epoch=1,
            epochs_completed=1,
            early_stopped=False,
            clean_val_accuracy=0.5,
            status="completed",
            artifact_dir=f"artifacts/pilots/run-{i}",
            ledger_path=ledger_path,
        )
    with ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [r["run_id"] for r in rows] == ["run-0", "run-1", "run-2"]


def test_append_pilot_entry_bad_run_is_still_recorded(tmp_path):
    """Per CLAUDE.md: a badly-performing/failed run must still get a
    ledger row, not be silently dropped."""
    ledger_path = tmp_path / "ledger.csv"
    append_pilot_entry(
        run_id="bad-run",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        seed=314159,
        tta_seed=271828,
        config_hash="hash",
        git_commit="commit",
        best_epoch=1,
        epochs_completed=1,
        early_stopped=False,
        clean_val_accuracy=0.11,  # near-random, "bad" result
        status="completed",
        artifact_dir="artifacts/pilots/bad-run",
        ledger_path=ledger_path,
    )
    with ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "bad-run"
