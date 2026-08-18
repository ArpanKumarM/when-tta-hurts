from when_tta_hurts.config import config_hash, load_config


def test_pilot_config_loads():
    cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    assert cfg["phase"] == "pilot"
    assert cfg["confirmatory"] is False
    assert cfg["seed"] == 314159


def test_pilot_seed_not_in_confirmatory_seeds():
    pilot_cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    matrix_cfg = load_config("configs/experiment_matrix.yaml")
    confirmatory_seeds = matrix_cfg["seeds"]["confirmatory"]
    assert pilot_cfg["seed"] not in confirmatory_seeds


def test_pilot_config_test_split_disallowed():
    cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    assert cfg["dataset"]["test_split_allowed"] is False
    assert cfg["dataset"]["eval_split"] == "val"


def test_pilot_config_hash_deterministic():
    cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    assert config_hash(cfg) == config_hash(cfg)


def test_pilot_config_no_train_time_augmentation_or_label_smoothing():
    cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    assert cfg["preprocessing"]["train_time_augmentation"] is False
    assert cfg["preprocessing"]["label_smoothing"] is False
    assert cfg["preprocessing"]["class_weighting"] is False


def test_pilot_config_matches_frozen_small_cnn_param_count():
    from when_tta_hurts.models.small_cnn import build_small_cnn

    cfg = load_config("configs/pilot_pathmnist_28_bn.yaml")
    model = build_small_cnn(
        num_classes=cfg["model"]["num_classes"], normalization=cfg["model"]["normalization"]
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == cfg["model"]["expected_param_count"]
