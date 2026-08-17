import numpy as np
import torch

from when_tta_hurts.devices import select_device
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything, seeded_generator, worker_init_fn
from when_tta_hurts.transforms import build_policy, sample_deterministic_view


def test_seed_everything_reproduces_torch_randn():
    seed_everything(123)
    a = torch.randn(10)
    seed_everything(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_seed_everything_reproduces_numpy():
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(42)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_seeded_generator_reproducible():
    g1 = seeded_generator(7)
    g2 = seeded_generator(7)
    t1 = torch.randn(5, generator=g1)
    t2 = torch.randn(5, generator=g2)
    assert torch.equal(t1, t2)


def test_worker_init_fn_runs_without_error():
    torch.manual_seed(0)
    worker_init_fn(0)
    worker_init_fn(1)


def test_same_device_reproducibility_model_and_view():
    """Same-device (not cross-device) reproducibility check: see
    scripts/verify_reproducibility.py for the fuller MPS-targeted version
    and the Phase 1 completion report for its results. This test uses
    select_device('auto') so it also runs correctly in CPU-only CI."""
    device = select_device("auto")

    def run_once():
        seed_everything(0)
        model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
        params = [p.detach().clone() for p in model.parameters()]
        x = torch.rand(2, 3, 28, 28, device=device)
        with torch.no_grad():
            logits = model(x).detach().clone()
        policy = build_policy("mixed").to(device)
        view = sample_deterministic_view(x, policy, seed=0).detach().clone()
        return params, logits, view

    params_a, logits_a, view_a = run_once()
    params_b, logits_b, view_b = run_once()

    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa, pb)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(view_a, view_b)
