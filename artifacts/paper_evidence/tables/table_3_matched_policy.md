# Table 3 — Matched-Policy Within-Cell and Secondary DiD Table

## Within-cell (preregistered, H3 matched arm)

| run_id | dataset | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |
|---|---|---|---|---|---|---|
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0 | bloodmnist | 0 | 1.11 | [-0.03, 2.22] | 0.0603 | 0.0658 |
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1 | bloodmnist | 1 | 2.95 | [1.99, 3.92] | 2.78e-09 | 4.16e-09 |
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2 | bloodmnist | 2 | 0.35 | [-0.50, 1.23] | 0.468 | 0.468 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s0 | pathmnist | 0 | -1.21 | [-2.19, -0.24] | 0.0155 | 0.0186 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s1 | pathmnist | 1 | 4.09 | [3.27, 4.90] | 1.13e-22 | 1.94e-22 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s2 | pathmnist | 2 | 1.80 | [0.93, 2.66] | 5.04e-05 | 6.72e-05 |

## Secondary (post-validation/pre-test-specified, fixed-model DiD -- not a preregistered cross-condition test)

| pair_id | condition A | condition B | DiD (pp) | 95% CI (pp) |
|---|---|---|---|---|
| H3-bloodmnist-s0 | A-bloodmnist-28px-batchnorm-policy-none-s0 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0 | 31.22 | [29.29, 33.21] |
| H3-bloodmnist-s1 | A-bloodmnist-28px-batchnorm-policy-none-s1 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1 | 36.45 | [34.46, 38.41] |
| H3-bloodmnist-s2 | A-bloodmnist-28px-batchnorm-policy-none-s2 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2 | 49.58 | [47.65, 51.45] |
| H3-pathmnist-s0 | A-pathmnist-28px-batchnorm-policy-none-s0 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s0 | 17.55 | [16.13, 18.97] |
| H3-pathmnist-s1 | A-pathmnist-28px-batchnorm-policy-none-s1 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s1 | 44.32 | [42.67, 45.93] |
| H3-pathmnist-s2 | A-pathmnist-28px-batchnorm-policy-none-s2 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s2 | 44.07 | [42.59, 45.57] |
