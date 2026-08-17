import pytest
import torch

from when_tta_hurts.devices import DeviceUnavailableError, capture_environment, select_device


def test_select_device_cpu_always_works():
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_returns_valid_device():
    d = select_device("auto")
    assert d.type in ("cpu", "mps")


def test_select_device_unknown_raises():
    with pytest.raises(ValueError):
        select_device("cuda")


def test_select_device_mps_raises_if_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError):
        select_device("mps")


def test_capture_environment_has_expected_fields():
    device = select_device("cpu")
    manifest = capture_environment(device)
    d = manifest.to_dict()
    for key in (
        "device",
        "mps_built",
        "mps_available",
        "macos_version",
        "chip",
        "python_version",
        "torch_version",
        "torchvision_version",
        "package_versions",
    ):
        assert key in d
    assert d["device"] == "cpu"
