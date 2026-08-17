from when_tta_hurts.config import config_hash, config_hash_short, load_config


def test_load_config_smoke_yaml():
    cfg = load_config("configs/smoke.yaml")
    assert cfg["seed"] == 0
    assert cfg["dataset"]["name"] == "pathmnist"


def test_config_hash_deterministic():
    cfg = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_order_independent():
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"b": 2, "a": 1}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_changes_with_content():
    cfg1 = {"a": 1}
    cfg2 = {"a": 2}
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_short_is_prefix():
    cfg = {"a": 1}
    assert config_hash(cfg).startswith(config_hash_short(cfg))
    assert len(config_hash_short(cfg)) == 12
