import json

from when_tta_hurts.artifacts import append_ledger_row, write_environment_manifest
from when_tta_hurts.devices import capture_environment, select_device
from when_tta_hurts.evaluation.cache import CacheKey, cache_key_hash


def test_write_environment_manifest(tmp_path):
    device = select_device("cpu")
    manifest = capture_environment(device)
    out_path = tmp_path / "env.json"
    write_environment_manifest(manifest, out_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["device"] == "cpu"


def test_append_ledger_row_creates_header_once(tmp_path):
    path = tmp_path / "ledger.csv"
    append_ledger_row({"run_id": "a", "status": "ok"}, path)
    append_ledger_row({"run_id": "b", "status": "failed"}, path)
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "run_id,status"
    assert len(lines) == 3


def test_cache_key_hash_deterministic():
    key = CacheKey(
        checkpoint_hash="abc123",
        dataset_version="pathmnist-28-v1",
        split="test",
        policy="mixed",
        seed=0,
        preprocessing_config_hash="deadbeef",
    )
    assert cache_key_hash(key) == cache_key_hash(key)


def test_cache_key_hash_changes_with_any_field():
    base = CacheKey("a", "b", "c", "d", 0, "e")
    variant = CacheKey("a", "b", "c", "d", 1, "e")  # seed differs
    assert cache_key_hash(base) != cache_key_hash(variant)
