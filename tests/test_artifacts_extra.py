import json

import numpy as np
import torch

from when_tta_hurts.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    hash_file,
    hash_state_dict,
    save_checkpoint,
)
from when_tta_hurts.models.small_cnn import build_small_cnn


def test_atomic_write_json_no_leftover_tmp(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json({"a": 1}, path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text()) == {"a": 1}


def test_atomic_write_npz_roundtrip(tmp_path):
    path = tmp_path / "out.npz"
    arrays = {"x": np.arange(10), "y": np.ones((2, 2))}
    atomic_write_npz(arrays, path)
    assert path.exists()
    loaded = np.load(path)
    assert np.array_equal(loaded["x"], arrays["x"])
    assert np.array_equal(loaded["y"], arrays["y"])


def test_hash_state_dict_deterministic():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    h1 = hash_state_dict(model.state_dict())
    h2 = hash_state_dict(model.state_dict())
    assert h1 == h2


def test_hash_state_dict_changes_with_weights():
    model_a = build_small_cnn(num_classes=9, normalization="batchnorm")
    torch.manual_seed(0)
    for p in model_a.parameters():
        p.data.fill_(0.0)
    h_zero = hash_state_dict(model_a.state_dict())

    for p in model_a.parameters():
        p.data.fill_(1.0)
    h_one = hash_state_dict(model_a.state_dict())

    assert h_zero != h_one


def test_save_checkpoint_and_hash_file_consistent(tmp_path):
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    ckpt_path = tmp_path / "model.pt"
    returned_hash = save_checkpoint(model.state_dict(), ckpt_path)
    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(".pt.tmp").exists()

    reloaded = torch.load(ckpt_path, weights_only=True)
    assert hash_state_dict(reloaded) == returned_hash

    # hash_file is a different (whole-file-bytes) hash than hash_state_dict
    # (which hashes tensor contents only) -- just confirm it runs and is stable.
    f1 = hash_file(ckpt_path)
    f2 = hash_file(ckpt_path)
    assert f1 == f2
