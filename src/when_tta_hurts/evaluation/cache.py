"""Cache-key design for the future per-view logit cache (Phase 2+).

This module implements ONLY the cache-key primitive, per Phase 1 scope --
the actual view-generation/caching pipeline (deterministic 100-view
sequence, nested-prefix aggregation for view counts 1/2/5/10/25/50/100,
Validation-Gated TTA) is explicitly out of scope for Phase 1 (see
docs/compute_budget.md "Evaluation job accounting" and CLAUDE.md).

Design (per docs/compute_budget.md): for a given (checkpoint, dataset
version, split, policy, seed, preprocessing config), a deterministic
ordered sequence of up to 100 view transforms is generated once and cached;
every tested view count (1,2,5,10,25,50,100) is a prefix of that same
sequence, and mean/majority/confidence-weighted aggregation are computed
from the cached per-view logits rather than re-running the model. BN-adapted
evaluation is NOT cacheable this way because it mutates model state, and
must be tracked as a separate inference pass.

Estimated storage (CORRECTED -- see docs/compute_budget.md "Evaluation
job accounting" for the full derivation): the formula is

    samples x views x classes x 4 bytes (float32 logits)

Only probabilities/aggregates are DERIVED from cached logits, never cached
separately (mean/majority/confidence-weighted aggregation and view-count
prefixes 1/2/5/10/25/50/100 are all recomputable from the same cached
100-view logit tensor). For validation+test at 100 views, per
checkpoint/policy:

  PathMNIST:  (10,004 + 7,180)  x 100 x 9 x 4 bytes =~ 59 MiB
  BloodMNIST: (1,712 + 3,421)   x 100 x 8 x 4 bytes =~ 15.7 MiB
  DermaMNIST: (1,003 + 2,005)   x 100 x 7 x 4 bytes =~ 8 MiB

With 15 PathMNIST + 15 BloodMNIST + 3 DermaMNIST distinct checkpoints
(blocks A+B+C, before the conditional 128px tier D):

  15 x 59 MiB + 15 x 15.7 MiB + 3 x 8 MiB =~ 1.1 GiB for ONE policy,
  =~ 3.3 GiB for all three policies (geometric/intensity/mixed).

With block D's 6 additional PathMNIST+BloodMNIST checkpoints included,
total is =~ 4 GiB for three policies -- BEFORE checkpoints themselves,
labels/metadata, temporary files, BN-adapted inference (which is NOT
cacheable this way, since it mutates model state -- see module docstring),
and additional validation artifacts. A practical working-storage allowance
of ~5-8 GB is recommended, not the 1-4GB logit-only figure alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheKey:
    checkpoint_hash: str
    dataset_version: str
    split: str
    policy: str
    seed: int
    preprocessing_config_hash: str

    def as_tuple(self) -> tuple:
        return (
            self.checkpoint_hash,
            self.dataset_version,
            self.split,
            self.policy,
            self.seed,
            self.preprocessing_config_hash,
        )


def cache_key_hash(key: CacheKey) -> str:
    """Stable, deterministic hash for a CacheKey, used as a cache filename."""
    canonical = "|".join(str(part) for part in key.as_tuple())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
