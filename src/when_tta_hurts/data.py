"""MedMNIST dataset loading via the official `medmnist` package only.

No images are redistributed by this repository: datasets are downloaded at
runtime into a git-ignored cache directory (data/raw/), never committed.
DermaMNIST is CC BY-NC 4.0 (non-commercial) -- callers must display
`DERMAMNIST_LICENSE_NOTICE` wherever DermaMNIST metadata is surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medmnist import INFO

DERMAMNIST_LICENSE_NOTICE = (
    "DermaMNIST is distributed under CC BY-NC 4.0 (non-commercial use only). "
    "Source: HAM10000 (Tschandl et al.). See docs/data_and_licensing.md."
)

SUPPORTED_DATASETS = ("pathmnist", "bloodmnist", "dermamnist")


class TestSplitAccessError(PermissionError):
    """Raised when code tries to load the official test split without
    explicit, narrow authorization. See docs/experimental_protocol.md's
    test firewall and the Phase 2A pilot protocol -- the pilot must never
    be able to reach the test split, accidentally or otherwise."""

    __test__ = False  # not a pytest test class; name starts with "Test" incidentally


# Expected official split counts, verified against the source paper and
# cross-checked programmatically against medmnist.INFO in
# verify_split_counts() below -- see docs/data_and_licensing.md.
EXPECTED_SPLITS: dict[str, dict[str, int]] = {
    "pathmnist": {"train": 89996, "val": 10004, "test": 7180},
    "bloodmnist": {"train": 11959, "val": 1712, "test": 3421},
    "dermamnist": {"train": 7007, "val": 1003, "test": 2005},
}


@dataclass
class DatasetMetadata:
    name: str
    n_channels: int
    n_classes: int
    license: str
    splits: dict[str, int]
    md5_by_resolution: dict[int, str]
    description: str


def get_dataset_metadata(name: str) -> DatasetMetadata:
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'; expected one of {SUPPORTED_DATASETS}")
    info = INFO[name]

    md5_by_resolution: dict[int, str] = {}
    if "MD5" in info:
        md5_by_resolution[28] = info["MD5"]
    for res in (64, 128, 224):
        key = f"MD5_{res}"
        if key in info:
            md5_by_resolution[res] = info[key]

    return DatasetMetadata(
        name=name,
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        license=info["license"],
        splits=dict(info["n_samples"]),
        md5_by_resolution=md5_by_resolution,
        description=info["description"],
    )


def verify_split_counts(name: str) -> dict[str, Any]:
    """Compare medmnist's shipped INFO split counts (package metadata only)
    against our recorded expectation (docs/data_and_licensing.md).

    IMPORTANT: this checks metadata only and does NOT require or imply that
    the dataset's image .npz artifact was downloaded or inspected. For an
    empirical check against the actual downloaded array, see
    verify_split_counts_from_artifact() below. See
    docs/data_and_licensing.md "Split-count verification" for which
    datasets have which level of verification.
    """
    meta = get_dataset_metadata(name)
    expected = EXPECTED_SPLITS[name]
    mismatches = {
        split: (expected[split], meta.splits.get(split))
        for split in expected
        if meta.splits.get(split) != expected[split]
    }
    return {
        "dataset": name,
        "verification_level": "metadata_only",
        "matches": len(mismatches) == 0,
        "expected": expected,
        "actual": meta.splits,
        "mismatches": mismatches,
    }


def verify_split_counts_from_artifact(name: str, root: str | Path = "data/raw") -> dict[str, Any]:
    """Empirically verify split counts by reading array shapes directly out of
    the downloaded .npz artifact -- requires the file to already exist at
    `root/{name}.npz` (does NOT download it). This is the stronger check:
    metadata could in principle be stale, but this reads the actual data.
    """
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'; expected one of {SUPPORTED_DATASETS}")

    npz_path = Path(root) / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} does not exist -- this function verifies against an "
            f"already-downloaded artifact and does not download one itself."
        )

    import numpy as np

    with np.load(npz_path) as data:
        actual = {
            "train": int(data["train_images"].shape[0]),
            "val": int(data["val_images"].shape[0]),
            "test": int(data["test_images"].shape[0]),
        }

    expected = EXPECTED_SPLITS[name]
    mismatches = {
        split: (expected[split], actual.get(split))
        for split in expected
        if actual.get(split) != expected[split]
    }
    return {
        "dataset": name,
        "verification_level": "empirical_artifact",
        "artifact_path": str(npz_path),
        "matches": len(mismatches) == 0,
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def load_dataset(
    name: str,
    split: str,
    size: int = 28,
    root: str | Path = "data/raw",
    download: bool = True,
    transform=None,
    allow_test: bool = False,
):
    """Load a MedMNIST split via the official package.

    `root` is expected to be a git-ignored cache directory (see
    data/README.md and .gitignore) -- this function never writes images
    anywhere else. Returns torch.Tensor images (not PIL) by default via
    torchvision.transforms.ToTensor(), unless a different `transform` is
    supplied.

    TEST-SET FIREWALL: split='test' is rejected with TestSplitAccessError
    unless allow_test=True is passed explicitly. allow_test defaults to
    False everywhere in this project's training/pilot code paths, per
    docs/experimental_protocol.md's test firewall. This is a generic
    loader used by scripts that ARE allowed test access (e.g. an eventual
    exact-baseline-reproduction script); the Phase 2A pilot does NOT use
    this parameter -- see load_pilot_split() below, which has no override
    at all.
    """
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'; expected one of {SUPPORTED_DATASETS}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split '{split}'; expected train/val/test")
    if split == "test" and not allow_test:
        raise TestSplitAccessError(
            "Attempted to load the official test split without allow_test=True. "
            "Per docs/experimental_protocol.md's test firewall, test-split access "
            "must be explicit and narrow, not a default or an incidental side effect."
        )

    import medmnist
    from torchvision import transforms as T

    if transform is None:
        transform = T.ToTensor()

    python_class_name = INFO[name]["python_class"]
    dataset_class = getattr(medmnist, python_class_name)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    return dataset_class(
        split=split,
        size=size,
        download=download,
        root=str(root),
        transform=transform,
    )


def load_pilot_split(name: str, split: str, size: int = 28, root: str | Path = "data/raw", transform=None):
    """Load a split for the Phase 2A pilot -- ONLY 'train' or 'val'.

    This function deliberately has NO allow_test override of any kind: the
    pilot configuration (configs/pilot_pathmnist_28_bn.yaml) prohibits test
    access entirely, and per your instruction this must not gain a
    convenience override. Requesting split='test' (or anything other than
    'train'/'val') raises TestSplitAccessError / ValueError before any
    data is touched.
    """
    if split not in ("train", "val"):
        raise TestSplitAccessError(
            f"load_pilot_split() only accepts split in ('train', 'val'); got '{split}'. "
            "The Phase 2A pilot has no mechanism to request the test split -- "
            "see docs/pilot_protocol.md."
        )
    return load_dataset(name, split=split, size=size, root=root, download=True, transform=transform)
