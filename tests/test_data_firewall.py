import pytest

from when_tta_hurts.data import TestSplitAccessError, load_dataset, load_pilot_split


def test_load_dataset_rejects_test_split_by_default():
    with pytest.raises(TestSplitAccessError):
        load_dataset("pathmnist", split="test", download=False)


def test_load_dataset_allows_test_split_only_with_explicit_flag(monkeypatch):
    """We don't want to actually download/touch test data in this offline
    test -- just confirm the guard is bypassed ONLY when allow_test=True is
    passed, by checking it gets past the guard and fails later for an
    unrelated reason (download=False + no cached file) rather than
    TestSplitAccessError."""
    with pytest.raises(Exception) as exc_info:
        load_dataset(
            "pathmnist", split="test", download=False, allow_test=True, root="/tmp/does_not_exist_xyz"
        )
    assert not isinstance(exc_info.value, TestSplitAccessError)


def test_load_pilot_split_rejects_test():
    with pytest.raises(TestSplitAccessError):
        load_pilot_split("pathmnist", split="test")


def test_load_pilot_split_rejects_arbitrary_strings():
    with pytest.raises(TestSplitAccessError):
        load_pilot_split("pathmnist", split="official_test")


def test_load_pilot_split_has_no_test_override_parameter():
    """The pilot loader must not gain a convenience override -- confirm its
    signature has no allow_test (or similarly named) parameter at all."""
    import inspect

    sig = inspect.signature(load_pilot_split)
    param_names = set(sig.parameters.keys())
    assert "allow_test" not in param_names
    assert not any("test" in name.lower() and name != "split" for name in param_names)
