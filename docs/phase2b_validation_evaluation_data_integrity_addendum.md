# Phase 2B.4D Data-Integrity Addendum: checksum-verified evaluation loading

This addendum freezes a data-integrity correction to the validation-
evaluation runner, found during ongoing engineering audit: the production
evaluation loader (`evaluation/validation_loader.py::load_validation_evaluation_split`,
via `data.py::load_pilot_split`) currently loads the official MedMNIST
validation array without first verifying the artifact's checksum --
unlike the *training* loader path
(`orchestrator.py::default_train_validation_loader_factory`), which
already verifies via `dataset_verification.py::verify_official_dataset_artifact()`
before constructing any DataLoader. This addendum closes that gap for
evaluation. It does not alter the frozen scientific protocol
(`docs/phase2b_protocol.md`, `configs/validation_evaluation.yaml`): no
seed, prefix, aggregation, policy, or endpoint definition changes.

No real evaluation result has yet been observed anywhere in this
project.

## 1. Requirement (frozen here)

1. Every new validation-evaluation attempt must verify the exact official
   MedMNIST artifact before loading any array from it.
2. Verification must use the existing production verifier,
   `dataset_verification.py::verify_official_dataset_artifact()` -- no
   second checksum implementation.
3. The evaluator must verify: dataset identifier; native resolution;
   expected official MD5 from `medmnist.INFO`; actual full-file MD5;
   exact checksum equality; `resized=False`.
4. Missing, unsupported, resized/proxy, or checksum-mismatched artifacts
   must fail closed (a terminal `status="failed"` attempt, never
   `"completed"`).
5. Evaluation must never download data automatically --
   `verify_official_dataset_artifact()` has no download parameter or
   fallback of any kind; a missing artifact is a hard failure, not a
   trigger to fetch one.
6. Only the validation arrays may be indexed after verification --
   `load_validation_evaluation_split()` already hardcodes `split="val"`
   with no override of any kind (unchanged by this addendum).
7. Hashing the complete NPZ container for integrity (an MD5 over the
   whole file, including its `train`/`val`/`test` arrays as they exist
   inside the single `.npz` archive) is allowed and is **not** test-split
   evaluation: verification never reads, decodes, or indexes
   `test_images`/`test_labels` as arrays -- it only computes a checksum
   over the undifferentiated file bytes, exactly as the existing training
   path already does for the same reason. `test_images`/`test_labels`
   must never be indexed anywhere in the evaluation path (unchanged;
   `load_pilot_split()` structurally has no test-split mechanism).
8. A completed evaluation must persist the verification evidence
   (dataset, resolution, expected MD5, actual MD5, checksum-match,
   native/resized status, verification method/version) as a required,
   manifest-covered section of `metadata.json`.
9. The idempotent completed-result skip (`check_evaluation_skip()`) must
   occur before any checksum computation or dataset loading -- unchanged
   control-flow position (already before attempt allocation, which is
   itself before dataset verification in the corrected order).
10. Local file paths are provenance only: the absolute artifact path
    (`data/raw/pathmnist.npz` vs. any other location containing
    byte-identical official content) must never participate in scientific
    identity -- only the expected/actual checksum values do.
11. The expected official dataset checksum (a metadata-only lookup from
    `medmnist.INFO`, requiring no disk access to the artifact itself)
    must be included in the stable evaluation configuration/identity
    (`ValidationEvaluationConfig`, hashed into `evaluation_id`).
12. The actual checksum, computed from the real on-disk artifact
    immediately before array loading, must equal the expected checksum
    baked into `evaluation_id` before evaluation proceeds --
    `verify_official_dataset_artifact()` already raises on any mismatch,
    which this addendum wires into the production path as a hard,
    completion-blocking failure.

## 2. `dataset_verification.py` is now evaluation-relevant

`dataset_verification.py` is added to `EVALUATOR_FINGERPRINT_MANIFEST`
(`src/when_tta_hurts/validation_evaluation.py`) -- its verification logic
now runs on the evaluation path, not only the training path, so a change
to its checksum-comparison or resolution-registration logic must change
`evaluation_id` exactly like every other evaluation-relevant file already
does. See `docs/phase2b_validation_evaluation_engineering_addendum.md`
sec.3.2 for the exclusion table this updates: `dataset_verification.py`
moves from "safely excluded (training-loader-path only)" to "manifested
(now reachable from the evaluation path)."

## 3. Single source of truth for the expected checksum

`dataset_verification.py` gains one new, small function,
`expected_official_checksum(dataset, resolution) -> str` -- a pure
`medmnist.INFO` metadata lookup (no disk I/O), extracted from
`verify_official_dataset_artifact()`'s existing lookup logic so both the
identity-time lookup (item 11 above, called before any attempt exists)
and the full artifact verification (item 12, called just before array
loading) resolve the expected checksum through the exact same code path.
`verify_official_dataset_artifact()` itself is unchanged in behavior --
it now calls `expected_official_checksum()` internally instead of
duplicating the lookup inline. This is a refactor for a single source of
truth, not a second checksum implementation.
