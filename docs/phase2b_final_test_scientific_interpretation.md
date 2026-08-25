# Phase 2B Final-Test Scientific Interpretation

Cautious interpretation, limitations, and claim adjudication only. Governed entirely by the rules frozen in docs/phase2b_final_test_unsealing_freeze.md sec.8/9 -- no rule is introduced here that was not already frozen before this result was unsealed.

## Scientific classification (reaffirmed)

- Preregistered final-test analyses (H1/H2/H3/BLOCK_C) are the confirmatory within-cell analyses specified by the frozen SAP. They do not, by themselves, establish the cross-condition differences implied by H1/H2/H3.
- Cross-condition difference-in-differences analyses are explicitly secondary, post-validation/pre-test specified, fixed-model-only, and not preregistered.
- BLOCK_C is a positive-control analysis, reported regardless of direction.
- No H4 claim is made anywhere in this document.
- No population-level or model-population inference is made anywhere in this document.

## Required limitations

- Only three training seeds per cell; sample-level paired tests do not substitute for a model-seed population replication study.
- Limited dataset/architecture coverage (the specific MedMNIST subsets and ResNet variants actually used).
- A fixed augmentation policy and a fixed TTA view budget (N=50).
- The cross-condition addendum was frozen after validation-stage results were already observed, but before the official test split was opened.

## Incident disclosure

- The accidental final-test access incident for cell 1 (attempt 1, aborted).
- Two failed final-test engineering attempts (cell 1 attempt 2; cell 2 attempt 1), neither of which persisted any scientific value.
- The shared-aggregation-contract correction and the validation-metric-reconciliation mechanism it required.
- All 39 canonical final-test results were produced under the final, corrected evaluator/aggregation pipeline; cell 1's compatibility under its historical generation-3 binding was independently established via 56/56 recomputation checks, never assumed.
- No scientific result from any of the seven sealed artifacts was examined by any person or process before this controlled-unsealing phase.

