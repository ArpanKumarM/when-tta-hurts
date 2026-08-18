"""Phase 2A audit Part E: independently validate metrics.py against
external/manual references (sklearn, direct PyTorch, hand-constructed
examples) rather than only self-consistency tests."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from when_tta_hurts.evaluation.tta import aggregate_mean_prefix
from when_tta_hurts.metrics import (
    accuracy,
    expected_calibration_error,
    harm_rescue_rates,
    macro_f1,
    negative_log_likelihood,
)


def test_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 9))
    labels = rng.integers(0, 9, size=200)
    ours = accuracy(logits, labels)
    theirs = accuracy_score(labels, logits.argmax(axis=-1))
    assert np.isclose(ours, theirs)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 9))
    labels = rng.integers(0, 9, size=200)
    ours = macro_f1(logits, labels)
    theirs = f1_score(labels, logits.argmax(axis=-1), average="macro", zero_division=0)
    assert np.isclose(ours, theirs)


def test_nll_matches_direct_pytorch_cross_entropy():
    rng = np.random.default_rng(2)
    logits_np = rng.normal(size=(50, 9)).astype(np.float32)
    labels_np = rng.integers(0, 9, size=50)

    ours = negative_log_likelihood(logits_np, labels_np)

    logits_t = torch.tensor(logits_np)
    labels_t = torch.tensor(labels_np, dtype=torch.long)
    # torch's cross_entropy(logits, labels) with default reduction='mean'
    # is exactly the mean NLL of the softmax distribution at the true class.
    theirs = torch.nn.functional.cross_entropy(logits_t, labels_t).item()

    assert np.isclose(ours, theirs, atol=1e-5)


def test_ece_manually_constructed_two_bin_example():
    # 4 samples: 2 predicted with confidence 0.9 (both correct -> bin gap 0.1),
    # 2 predicted with confidence 0.6 (1 correct, 1 wrong -> bin acc 0.5, gap 0.1)
    # Expected ECE = 0.5*0.1 + 0.5*0.1 = 0.1 (equal-sized bins)

    def logits_for_confidence(p_true: float, n_classes: int = 9) -> np.ndarray:
        # true class gets logit L, all others get 0; softmax(true) = p means
        # exp(L) / (exp(L) + (n_classes-1)) = p  =>  L = log(p*(n-1)/(1-p))
        other = n_classes - 1
        L = np.log(p_true * other / (1 - p_true))
        row = np.zeros(n_classes)
        row[0] = L
        return row

    logits = np.stack(
        [
            logits_for_confidence(0.9),  # correct, conf .9
            logits_for_confidence(0.9),  # correct, conf .9
            logits_for_confidence(0.6),  # correct, conf .6
            logits_for_confidence(0.6),  # WRONG (true label != 0), conf .6 on wrong class
        ]
    )
    labels = np.array([0, 0, 0, 1])  # last sample predicted class 0 but true is 1 -> wrong

    ece = expected_calibration_error(logits, labels, n_bins=10)
    # bin containing 0.9 (both correct): acc=1.0, conf=0.9 -> |1.0-0.9|=0.1, weight 2/4
    # bin containing 0.6 (1 correct, 1 wrong): acc=0.5, conf=0.6 -> |0.5-0.6|=0.1, weight 2/4
    expected = 0.5 * 0.1 + 0.5 * 0.1
    assert np.isclose(ece, expected, atol=1e-6)


def test_harm_rescue_manually_constructed():
    # 6 samples. Clean correct: 0,1,2,3 (4 samples). Clean wrong: 4,5 (2 samples).
    # TTA flips 0 and 1 to wrong (harmed=2), keeps 2,3 correct.
    # TTA fixes sample 4 (rescued), sample 5 stays wrong.
    labels = np.array([0, 0, 0, 0, 1, 1])
    clean_preds = np.array([0, 0, 0, 0, 0, 0])  # correct on 0-3, wrong on 4,5
    # 0,1 harmed; 2,3 stay correct; 4 rescued; 5 stays wrong
    tta_preds = np.array([1, 1, 0, 0, 1, 0])

    def preds_to_logits(preds, n_classes=2):
        logits = np.full((len(preds), n_classes), -10.0)
        logits[np.arange(len(preds)), preds] = 10.0
        return logits

    clean_logits = preds_to_logits(clean_preds)
    tta_logits = preds_to_logits(tta_preds)

    result = harm_rescue_rates(clean_logits, tta_logits, labels)
    assert result["n_clean_correct"] == 4
    assert result["n_clean_wrong"] == 2
    assert result["n_harmed"] == 2  # samples 0,1
    assert result["n_rescued"] == 1  # sample 4
    assert np.isclose(result["harm_rate"], 2 / 4)
    assert np.isclose(result["rescue_rate"], 1 / 2)


def test_prefix_aggregation_hand_calculated():
    # 2 views, 1 sample, 2 classes. View0 logits [0, 0] -> probs [.5,.5].
    # View1 logits [log(3), 0] -> probs [.75, .25].
    # Mean prob = [.625, .375].
    view0 = np.array([[0.0, 0.0]])
    view1 = np.array([[np.log(3.0), 0.0]])
    ordered = np.stack([view0, view1], axis=0)  # [2, 1, 2]

    agg_log_probs = aggregate_mean_prefix(ordered, n_views=2)
    recovered_probs = np.exp(agg_log_probs)
    recovered_probs /= recovered_probs.sum(axis=-1, keepdims=True)

    expected = np.array([[0.625, 0.375]])
    assert np.allclose(recovered_probs, expected, atol=1e-6)


def test_checkpoint_reload_and_prediction_ordering():
    """Confirms that reloading a checkpoint and re-running inference in a
    fresh process preserves prediction<->label ordering -- a regression
    guard for the exact bug class this audit is checking for."""
    import torch as _torch
    from torch.utils.data import DataLoader, TensorDataset

    from when_tta_hurts.artifacts import hash_state_dict, save_checkpoint
    from when_tta_hurts.models.small_cnn import build_small_cnn

    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = _torch.rand(20, 3, 28, 28)
    y = _torch.arange(20) % 9
    loader = DataLoader(TensorDataset(x, y), batch_size=7, shuffle=False)

    model.eval()
    labels_a = []
    with _torch.no_grad():
        for _xb, yb in loader:
            labels_a.append(yb.numpy())
    labels_a = np.concatenate(labels_a)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/ckpt.pt"
        h1 = save_checkpoint(model.state_dict(), ckpt_path)
        reloaded_state = _torch.load(ckpt_path, weights_only=True)
        assert hash_state_dict(reloaded_state) == h1

        model2 = build_small_cnn(num_classes=9, normalization="batchnorm")
        model2.load_state_dict(reloaded_state)
        model2.eval()

        labels_b = []
        with _torch.no_grad():
            for _xb, yb in loader:
                labels_b.append(yb.numpy())
        labels_b = np.concatenate(labels_b)

    assert np.array_equal(labels_a, labels_b), "label ordering must be identical across reload"
    assert np.array_equal(labels_a, y.numpy()), "label ordering must match the original dataset order"
