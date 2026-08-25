# Table 7 — Claim Adjudication

| Claim | Evidence tier | Status |
|---|---|---|
| Naive TTA harmed all 30 distinct unmatched-policy base cells | Preregistered within-cell | Supported |
| Matched-policy training mitigates TTA harm | Secondary fixed-model DiD, descriptively corroborated by separate within-cell patterns | Supported only secondarily/descriptively -- not a preregistered cross-condition test |
| Normalization changes the magnitude of harm | Secondary fixed-model DiD only | Supported only secondarily; direction is dataset-dependent |
| Higher resolution reduces TTA harm | Secondary fixed-model DiD only | Contradicted for BloodMNIST; mixed/near-null for PathMNIST |
| BLOCK_C reproduces the source paper's positive TTA improvement | Preregistered within-cell (positive control) | Contradicted -- expected positive improvement not reproduced in any seed |
| Any H4 (Validation-Gated TTA) verdict | None -- no derivable family exists | Not made, anywhere |
| Any model-population or general medical-imaging generalization | None -- three seeds, fixed policy/budget | Not permitted, anywhere |
