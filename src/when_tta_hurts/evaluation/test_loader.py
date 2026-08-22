"""Phase 2B.6F: dedicated TEST-ONLY evaluation loader. Split is
STRUCTURALLY fixed to 'test' -- there is no split parameter of any kind
on load_final_test_split(), so no CLI flag or environment variable
anywhere in this project can make it load anything else.

Mirrors evaluation/validation_loader.py's ValidationEvaluationSplit shape
exactly, and reuses that exact dataclass, so
validation_evaluation.py::compute_validation_evaluation() and
compute_evaluation_latency_report() accept a test split completely
unchanged -- the only scientific-data difference from a validation
evaluation is which split's arrays were loaded. The validation loader
itself is NEVER broadened to accept 'test' -- this is a wholly separate
module and function.

Requires an already-verified `VerifiedFinalTestReceipt` (produced once by
the orchestrator's single upfront verify_final_test_authorization() call,
per docs/phase2b_final_test_authorization_receipt_freeze.md) rather than
re-invoking the full, dynamic verifier itself. The prior design's
redundant re-invocation caused a real, universal, mechanically-diagnosed
failure (docs/phase2b_final_test_attempt2_preaccess_failure.md): schema
v2's exact-attempt-binding check recomputes the next allocatable attempt
LIVE on every call, so a second call after the active attempt's directory
had already been allocated always observed one attempt number higher
than the first (correct) call did. This module now performs only a
STATIC, comparative recheck (verify_receipt_still_valid()) against the
already-issued receipt -- never a fresh dynamic attempt-number
computation -- immediately before test-data access.

This is the only call site in the entire project that passes
allow_test=True to data.py::load_dataset() -- see
tests/test_split_firewall_static.py, which statically enforces that this
(and only this) guarded call site exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from medmnist import INFO
from torch.utils.data import DataLoader

from when_tta_hurts.data import load_dataset
from when_tta_hurts.dataset_verification import verify_official_dataset_artifact
from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit
from when_tta_hurts.final_test_authorization import VerifiedFinalTestReceipt, verify_receipt_still_valid


class TestLoaderError(RuntimeError):
    """Raised on any test-only-loader failure: authorization failure,
    checksum mismatch, resized/proxy artifact, or shape/label/class-
    count/resolution mismatch. Always fails closed before returning any
    array."""


def load_final_test_split(
    dataset: str,
    resolution: int,
    root: str | Path = "data/raw",
    *,
    receipt: VerifiedFinalTestReceipt,
) -> ValidationEvaluationSplit:
    """Load the official TEST split ONLY -- 'test' is hardcoded, never a
    parameter. `receipt` is REQUIRED (keyword-only, no default) --
    produced once by the orchestrator's single
    verify_final_test_authorization() call before attempt allocation.
    This function never independently invokes the full verifier. Order,
    in this function:
    1. statically recheck the already-issued receipt (bytes/tracked-
       clean/fingerprints/checksum unchanged, and bound to THIS exact
       dataset/resolution) -- never recomputes an attempt number, never
       scans the active run's attempt directory/ledger;
    2. verify the official whole-file checksum of the exact native
       artifact for (dataset, resolution) BEFORE any array access, and
       reject a resized/proxy substitute;
    3. construct the medmnist test-split dataset -- accesses exactly the
       official test-split arrays via the official package's own split
       handling; this function never references any other split's
       image/label identifier anywhere;
    4. validate shapes/labels/class-count/resolution;
    5. return a ValidationEvaluationSplit (the SAME type
       validation_evaluation.py's scientific functions already consume,
       so no scientific code needs to change for split=test).
    """
    verify_receipt_still_valid(receipt, dataset, resolution)

    verification = verify_official_dataset_artifact(dataset, resolution, root=root)
    if verification.resized:
        raise TestLoaderError(f"Refusing a resized/proxy artifact for {dataset}@{resolution}px.")

    test_dataset = load_dataset(dataset, split="test", size=resolution, root=str(root), allow_test=True)
    n = len(test_dataset)
    loader = DataLoader(test_dataset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
    sample_indices = np.arange(n, dtype=np.int64)

    expected_n_classes = len(INFO[dataset]["label"])
    if images.shape[0] != n or labels_np.shape[0] != n:
        raise TestLoaderError(
            f"Image/label count mismatch in loaded test split: images={images.shape[0]}, "
            f"labels={labels_np.shape[0]}."
        )
    if images.shape[-1] != resolution or images.shape[-2] != resolution:
        raise TestLoaderError(
            f"Loaded test images have resolution {tuple(images.shape[-2:])}, expected "
            f"({resolution}, {resolution})."
        )
    if n > 0 and (labels_np.min() < 0 or labels_np.max() >= expected_n_classes):
        raise TestLoaderError(
            f"Test labels out of range for {expected_n_classes} classes: min={int(labels_np.min())}, "
            f"max={int(labels_np.max())}."
        )

    return ValidationEvaluationSplit(
        images=images.float(),
        labels=labels_np,
        sample_indices=sample_indices,
        dataset=dataset,
        resolution=resolution,
    )
