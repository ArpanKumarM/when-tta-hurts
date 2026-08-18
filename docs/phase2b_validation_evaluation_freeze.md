# Phase 2B.4B: freezing the confirmatory TTA seed

This document freezes `configs/validation_evaluation.yaml`'s
`confirmatory_tta_seed` (and the rest of that file's frozen view/
aggregation configuration) **before any confirmatory validation
prediction, metric, or TTA outcome of any kind exists.**

## 1. Why the seed remained deferred

`docs/phase2b_protocol.md` sec.1 states explicitly:

> Pilot TTA seed 271828 -- not reused for any confirmatory TTA view
> sequence; confirmatory runs require their own TTA seed(s), distinct
> from 271828 (exact confirmatory TTA seed value(s) to be set when the
> runner is implemented -- not fixed by this document, since it is a
> new, not-yet-existing parameter; see sec.7's determinism requirements
> for how it must be derived once chosen).

At the time that sentence was written (Phase 2B.1), no evaluation runner
existed, so no seed could yet be meaningfully fixed. Phase 2B.4A
(commit `ea31d822940e3788879db24217092e9ad37f66ad`) implemented that
runner and exposed `--tta-seed` as a CLI parameter so its own synthetic
tests could inject arbitrary values -- but left the production value
unfrozen, which is a genuine defect for confirmatory use: a per-run
`--tta-seed` would let different Block D/A/B/C cells receive different
view banks, breaking the "one registered 100-view sequence per sample"
requirement across the confirmatory set and reintroducing exactly the
kind of undocumented researcher-discretion point the project's frozen-
protocol discipline exists to prevent.

## 2. Frozen before any confirmatory prediction

Reconfirmed immediately before this document and `configs/validation_evaluation.yaml`
were written:

- `artifacts/ledger_validation_evaluation.csv` contains a header row
  only -- zero data rows.
- No `artifacts/validation_evaluation/` directory exists anywhere in
  the repository.
- No real checkpoint has been loaded for evaluation/inference at any
  point in this project's history (Phase 2A's pilot used its own,
  separate, permanently-excluded seed 271828/314159 and is not a
  confirmatory result).
- No validation prediction, TTA metric, accuracy delta, or any other
  scientific outcome under the confirmatory protocol has been computed,
  observed, or inspected by anyone at any point before this freeze.

## 3. Exact mechanical derivation

```
namespace = "when-tta-hurts|phase2b|confirmatory-tta|v1"
digest    = sha256(namespace.encode("utf-8")).hexdigest()
          = "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd"
seed      = int(digest[:8], 16)
          = int("4ddab1df", 16)
          = 1306178015
```

Independently re-executed in Python during this freeze (`hashlib.sha256`,
stdlib, not a hand computation) and confirmed to match the specified
digest and integer exactly, character-for-character and value-for-value.
This is a **fixed, mechanical namespace-hash derivation** -- there is no
free parameter, no random draw, no candidate pool, and nothing to
"choose" beyond the namespace string itself, which names the exact
purpose (`confirmatory-tta`), the project, the phase, and a version tag
(`v1`, so a future genuinely-necessary re-derivation would use a new,
distinguishable namespace string rather than silently reusing this one).

## 4. Distinction from the pilot seed and every training seed

| Value | Meaning | Equal to 1306178015? |
|---|---|---|
| 271828 | Pilot TTA seed (`docs/pilot_protocol.md`) | No |
| 314159 | Pilot training seed, permanently excluded from all confirmatory work | No |
| 0, 1, 2 | Confirmatory **training** seeds (`configs/experiment_matrix.yaml`) -- a structurally different parameter (weight init/data-loader shuffling/training-time augmentation sampling), never a TTA view seed | No |

`1306178015` is trivially distinct from all five values by inspection
(the derivation produces a full 32-bit-range integer, not a small
hand-picked number), and this is additionally verified programmatically
by both `configs/validation_evaluation.yaml`'s own `excluded_seeds`
section and the production loader (`validation_evaluation.py`), which
hard-fails if the loaded seed ever coincides with any of them.

## 5. No candidate seeds were tried or compared; no outcome informed the choice

There is exactly one namespace string, computed once, yielding exactly
one seed. No alternative namespace strings were hashed, no alternative
seed values were considered, and -- as confirmed in section 2 above --
no confirmatory prediction, metric, or outcome of any kind existed to
influence this choice even if someone had wanted it to. This is a
pre-registration act, not a post-hoc rationalization of an already-run
result.

## 6. This document supplements, not rewrites, the frozen training protocol

`docs/phase2b_protocol.md` and `configs/experiment_matrix.yaml` are
**not modified** by this freeze. Their committed content, and therefore
their content hashes, remain exactly what they were when Blocks A/B/C/D
were trained against them -- verified unchanged as part of this freeze
(see the commit's verification record). `configs/validation_evaluation.yaml`
is an **additional**, narrowly-scoped, separately-versioned file that
fills in exactly the one parameter `docs/phase2b_protocol.md` sec.1
explicitly deferred, plus the view/aggregation configuration
(prefix sequence, total views, primary N, primary aggregation, policy
identifier) that must travel alongside the seed as a single frozen unit
so production can verify all of them together, not just the seed in
isolation.

## 7. Status

`configs/validation_evaluation.yaml` is committed with `status: approved`
in this same commit. Production evaluation code (Phase 2B.4B Part 2)
will refuse to run if this file is missing, untracked, dirty, malformed,
`status != approved`, uses any seed other than `1306178015`, or diverges
from the frozen view/aggregation configuration recorded here.
