# Table 6 — Complete Three-Seed BLOCK_C Table

| run_id | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |
|---|---|---|---|---|---|
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s0 | 0 | 0.25 | [-0.95, 1.45] | 0.751 | 1 |
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s1 | 1 | -2.89 | [-4.04, -1.80] | 5.76e-07 | 1.73e-06 |
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s2 | 2 | 0.05 | [-1.50, 1.65] | 1 | 1 |

External reference (descriptive only, not an acceptance threshold): the source paper's own reported TTA improvement at N=50 views was approximately +1.6 percentage points (docs/phase2b_validation_evaluation_block_c_audit.md sec.7). This project's frozen operationalization did not reproduce that expected positive improvement in any of the three seeds above.
