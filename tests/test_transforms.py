import torch

from when_tta_hurts.transforms import build_policy, sample_deterministic_view


def test_build_policy_geometric():
    policy = build_policy("geometric")
    x = torch.rand(2, 3, 28, 28)
    out = policy(x)
    assert out.shape == x.shape


def test_build_policy_intensity():
    policy = build_policy("intensity")
    x = torch.rand(2, 3, 28, 28)
    out = policy(x)
    assert out.shape == x.shape


def test_build_policy_mixed():
    policy = build_policy("mixed")
    x = torch.rand(2, 3, 28, 28)
    out = policy(x)
    assert out.shape == x.shape


def test_build_policy_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        build_policy("nonsense")


def test_sample_deterministic_view_reproducible_given_same_seed():
    policy = build_policy("mixed")
    x = torch.rand(2, 3, 28, 28)
    v1 = sample_deterministic_view(x, policy, seed=1)
    v2 = sample_deterministic_view(x, policy, seed=1)
    assert torch.allclose(v1, v2)


def test_sample_deterministic_view_differs_across_seeds():
    policy = build_policy("mixed")
    x = torch.rand(2, 3, 28, 28)
    v1 = sample_deterministic_view(x, policy, seed=1)
    v2 = sample_deterministic_view(x, policy, seed=2)
    assert not torch.allclose(v1, v2)


def test_transform_output_shape_dtype_range():
    policy = build_policy("mixed")
    x = torch.rand(4, 3, 28, 28)
    out = sample_deterministic_view(x, policy, seed=0)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_transform_at_64px():
    policy = build_policy("mixed", output_size=(64, 64))
    x = torch.rand(2, 3, 64, 64)
    out = sample_deterministic_view(x, policy, seed=0)
    assert out.shape == x.shape
