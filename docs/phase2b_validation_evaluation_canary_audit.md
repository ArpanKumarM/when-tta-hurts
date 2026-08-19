# Phase 2B.4D Canonical Validation-Evaluation Canary: attempt 4

**Recorded: 2026-08-19.** This document records the first technically
valid, eligible completion of the confirmatory validation-evaluation
pipeline for one cell of the experiment matrix
(`A-pathmnist-28px-batchnorm-policy-none-s0`), together with the honest
history of the three prior attempts that preceded it. Nothing in this
document is used to select, tune, or justify a policy, threshold, or
hypothesis change -- it is a pipeline-correctness canary, not a
confirmatory conclusion. The test split was never accessed while
producing any result recorded here.

## 1. Provenance

| Field | Value |
|---|---|
| Training run | `A-pathmnist-28px-batchnorm-policy-none-s0` |
| Canonical training attempt | `3` |
| Canonical checkpoint hash | `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` |
| Evaluation attempt | `4` |
| Evaluation ID / evaluation-config hash | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` |
| Evaluator fingerprint | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` |
| Execution source commit | `34528ee36c529829b42ca8c2ab670479516f4b5a` |
| Dataset | `pathmnist`, resolution 28, split `validation`, `n_validation_samples=10004` |
| Dataset checksum (expected == actual) | `a8b06965200029087d5bd730944a56c1`, `resized=False` |
| Frozen TTA seed | `1306178015` |
| TTA-seed config SHA-256 / freeze commit | `5037bc2f0ebd2723e984947485d6e332be842b69a4d929266eb29882affcc586` / `27fe46232f7b3b6c34bbfbbe82c3f363e3650eba` |
| Metric-input contract | `probability_native_v1` |
| Batching | `inference_batch_size=256`, `bn_adaptation_batch_size=256`, `bn_adaptation_algorithm=sequential_microbatch_v1`, `bn_adaptation_enumeration_order=view_major_then_sample_major`, `bn_adaptation_microbatches_at_primary_n=2000` |
| Runtime | `6785.219862937927` s (~1h53m) |
| Dependency versions | Python `3.12.2`, PyTorch `2.13.0`, torchvision `0.28.0`, NumPy `2.5.2` (pinned via `uv.lock`, itself content-hashed into the evaluator fingerprint) |

### Artifact manifest

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `predictions.npz` | 39,057,252 | `48b6ff9cf6900853043426ed3381537a84dba29b944670302229008ee1e3ba07` |
| `metrics.json` | 16,295 | `b9f6a0b309224aeeb2cbd29e33616cd570cee2276c7a7a391be4de763f287ba2` |
| `metadata.json` | 4,656 | `7193d84b39e6c6c6395f86148f554b451b58c53dbf404bcbc5bc381271d9a7b1` |
| `view_manifest.json` | 99,517 | `764f531cb816fa47cced92a7faa40090489bf46143bf39a92a31a0d8a7385b1e` |
| `artifact_manifest.json` | 647 | `f9765d6d228b4f9788ef778b66246adf55bd8b4eac1e8160fe0c21d911898cf3` |
| `status.json` | 306 | `2d8eef0a9599903ba648b66bd9dc0146854511885f564ba9407d16648bcd6f36` |

Location: `artifacts/validation_evaluation/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_004/`.

## 2. Attempt history (honest, nothing deleted or concealed)

| Attempt | evaluation_id | Status | Disposition |
|---|---|---|---|
| 1 | `ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5` | `aborted` | Test-harness escape; MPS initialized, process manually terminated shortly after; no probabilities/metrics ever computed or persisted; permanently reserved, never reused. |
| 2 | `96fbf4705bf93f4e2115fb33b9837df1095c90549d1f86ed1b1c1c160cc7fffe` | `failed` | BN-adaptation out-of-memory (`Invalid buffer size: 9.35 GiB`) at N=100 under the pre-bounded-memory-batching implementation. Fixed by the bounded-memory-batching correction (commit `b826338`). |
| 3 | `75aa7e37a9fe5454bf8edf6483d676a182d6dde9ff4a3730e4ada7195e09eb9e` | `completed`, **amendment-excluded** | Completed mechanically under the pre-`probability_native_v1` metric contract. `_per_prefix_metrics()` applied a spurious second `softmax()` to every aggregate condition's already-normalized probabilities before computing NLL/ECE/Brier. Accuracy/macro-F1/harm/rescue are argmax-invariant and unaffected; NLL/ECE/Brier for every aggregate condition are invalid. Recorded canonical-ineligible via `artifacts/ledger_validation_evaluation_amendments.csv` (`reason=probability_metric_double_softmax`); see `docs/phase2b_validation_evaluation_metric_contract_incident.md` for the full defect adjudication. **Artifacts preserved byte-identical, not deleted or rewritten.** |
| 4 | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` | `completed`, **eligible** | First technically valid completion under the corrected `probability_native_v1` metric contract. Sole canonical-compatible completed evaluation for this training run (`check_evaluation_skip()` selects it without ambiguity or conflict). |

Attempts 1-3 and the attempt-3 amendment row remain unchanged by this
document and by attempt 4's execution -- confirmed by hash comparison
(Section 5).

## 3. Complete results (attempt 4, exact persisted values)

### Clean

| Metric | Value |
|---|---|
| Accuracy | `0.7388044782087165` |
| Macro-F1 | `0.7243569247933442` |
| NLL | `0.91493159532547` |
| ECE | `0.08999954740770431` |
| Brier | `0.381816953275236` |

### Primary endpoint (frozen: N=50, mean-probability, stated neutrally)

```text
Clean accuracy:                          0.7388044782087165
TTA N=50 mean-probability accuracy:      0.45841663334666133
TTA N=50 mean-probability delta accuracy: -0.2803878448620552  (approximately -28.04pp)
```

### Mean-probability aggregation, all prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.30697720911635346 | 0.2799952733175686 | -0.4318272690923631 | 0.633473143011771 | 0.1385380788365863 | 5.078988552093506 | 0.46690835797789093 | 1.0895098232505374 |
| 2 | 0.35035985605757697 | 0.3161832640683311 | -0.38844462215113956 | 0.575429576511974 | 0.14045158821278225 | 3.272747039794922 | 0.23166633941313589 | 0.8937917227464572 |
| 5 | 0.38044782087165135 | 0.3453936874030481 | -0.3583566573370652 | 0.5336219726694629 | 0.13738997321086874 | 2.165464401245117 | 0.1032287886515945 | 0.7744721542608098 |
| 10 | 0.40913634546181527 | 0.3684649555477133 | -0.32966813274690127 | 0.49492626166959813 | 0.1377726750861079 | 1.8851244449615479 | 0.04572393191827149 | 0.7362185269288434 |
| 25 | 0.4412235105957617 | 0.40049357345711734 | -0.2975809676129548 | 0.4536598565823299 | 0.14389590508993494 | 1.7611045837402344 | 0.046877013246079346 | 0.7077923071601068 |
| **50** | **0.45841663334666133** | **0.41674392877069666** | **-0.2803878448620552** | **0.42957651197402247** | **0.1415996938384998** | **1.7255840301513672** | **0.06311160995394459** | **0.6988638629651448** |
| 100 | 0.46941223510595764 | 0.42669494623242393 | -0.2693922431027589 | 0.4134758490055473 | 0.1381553769613471 | 1.7069746255874634 | 0.0808504660446636 | 0.6953742478796293 |

### Majority-vote aggregation, all prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.30697720911635346 | 0.2799952733175686 | -0.4318272690923631 | 0.633473143011771 | 0.1385380788365863 | 19.148927368728224 | 0.6930227908756466 | 1.3860455817548187 |
| 2 | 0.2828868452618952 | 0.25051034558990026 | -0.4559176329468213 | 0.6597212826410499 | 0.12055109070034443 | 15.011669206545896 | 0.3712514993953525 | 1.0475809676066392 |
| 5 | 0.34106357457017195 | 0.31271252936091387 | -0.3977409036385446 | 0.5847652550399134 | 0.1312667432070417 | 9.279634206904992 | 0.20075969611812802 | 0.8404878048756207 |
| 10 | 0.3749500199920032 | 0.34153177993068085 | -0.3638544582167133 | 0.541198755242863 | 0.1377726750861079 | 6.496520433742604 | 0.1123950419808764 | 0.7734666133534939 |
| 25 | 0.41293482606957216 | 0.37709919261149477 | -0.3258696521391444 | 0.48964957380598023 | 0.13738997321086874 | 4.784690677156852 | 0.0586605357851594 | 0.7270931627344389 |
| 50 | 0.4348260695721711 | 0.39854170289511404 | -0.3039784086365454 | 0.46150723853335135 | 0.1415996938384998 | 4.077647571304569 | 0.04420431827267367 | 0.71231715313849 |
| 100 | 0.4544182327069172 | 0.4169553738596325 | -0.2843862455017993 | 0.43539439859288326 | 0.14274779946421737 | 3.56442627614543 | 0.03869652139292892 | 0.7062369452217718 |

### Confidence-weighted aggregation, all prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.30697720911635346 | 0.2799952733175686 | -0.4318272690923631 | 0.633473143011771 | 0.1385380788365863 | 5.078988552093506 | 0.46690835797789093 | 1.0895098232505374 |
| 2 | 0.34976009596161534 | 0.31601621165281657 | -0.3890443822471012 | 0.576376674333649 | 0.14083429008802142 | 3.2938153743743896 | 0.27721094637692145 | 0.9089579574734851 |
| 5 | 0.38794482207117154 | 0.35189317333567005 | -0.350859656137545 | 0.5238803950750913 | 0.1385380788365863 | 2.188520669937134 | 0.1310765190896798 | 0.7790883220330725 |
| 10 | 0.4171331467413035 | 0.37654732082442693 | -0.32167133146741306 | 0.48586118251928023 | 0.14274779946421737 | 1.903883934020996 | 0.06561431070951308 | 0.734942518901092 |
| 25 | 0.446421431427429 | 0.40563029179130633 | -0.2923830467812875 | 0.4470301718306048 | 0.1450440107156525 | 1.7743507623672485 | 0.04219793713316828 | 0.7018402582963525 |
| 50 | 0.4619152339064374 | 0.42113379048367816 | -0.2768892443022791 | 0.42605872006494383 | 0.1450440107156525 | 1.737414836883545 | 0.05408162289544467 | 0.6913216229785599 |
| 100 | 0.4728108756497401 | 0.43112039573996935 | -0.2659936025589764 | 0.41117575429576514 | 0.1446613088404133 | 1.7178012132644653 | 0.06374915323403776 | 0.687121461388292 |

### Original-anchored aggregation, all prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6388444622151139 | 0.6048421443388348 | -0.0999600159936026 | 0.17223650385604114 | 0.1044776119402985 | 1.2195560932159424 | 0.07218924364963562 | 0.5255822173975361 |
| 2 | 0.5652738904438225 | 0.5326525357486287 | -0.17353058776489405 | 0.27722906237315653 | 0.11978568694986605 | 1.355376958847046 | 0.04648106209918862 | 0.5808214778529895 |
| 5 | 0.5098960415833667 | 0.47290605048407897 | -0.22890843662534988 | 0.3548910837505074 | 0.12743972445464982 | 1.506988525390625 | 0.054777031288486336 | 0.6371154769937273 |
| 10 | 0.49020391843262695 | 0.4518058253042889 | -0.24860055977608958 | 0.38276281964551484 | 0.13088404133180254 | 1.5874170064926147 | 0.06515273156731143 | 0.6644145166197131 |
| 25 | 0.4819072371051579 | 0.4427951693379337 | -0.2568972411035586 | 0.3961574888377757 | 0.13700727133562954 | 1.6425498723983765 | 0.0787440255993321 | 0.6785613368832487 |
| 50 | 0.4790083966413435 | 0.4389083374304734 | -0.25979608156737305 | 0.40008117981328645 | 0.13700727133562954 | 1.6655033826828003 | 0.0826477246140776 | 0.6841445093746685 |
| 100 | 0.4802079168332667 | 0.43842096947729714 | -0.25859656137544984 | 0.39832228385874713 | 0.13662456946039037 | 1.6766754388809204 | 0.09311573652596007 | 0.6879733487552291 |

### BN-adapted aggregation, all seven registered prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6129548180727709 | 0.5959725279394923 | -0.12584966013594567 | 0.3505614937085645 | 0.5097588978185993 | 1.1252079010009766 | 0.13716627047752822 | 0.5445161202842096 |
| 2 | 0.6027588964414234 | 0.5844389211292417 | -0.1360455817672931 | 0.362197266946286 | 0.5036356678147723 | 1.1905168294906616 | 0.14820097349896905 | 0.5686349053398605 |
| 5 | 0.5920631747301079 | 0.5746751244216708 | -0.1467413034786086 | 0.37680963333784334 | 0.5040183696900115 | 1.1962921619415283 | 0.15226817869397696 | 0.5741701936985001 |
| 10 | 0.6296481407437026 | 0.6115465735982855 | -0.10915633746501396 | 0.3272899472331214 | 0.5078453884424033 | 1.087498426437378 | 0.12607440191422734 | 0.5260152634092423 |
| 25 | 0.6017592962814874 | 0.5830297557706617 | -0.13704518192722914 | 0.36246786632390743 | 0.5005740528128588 | 1.1896916627883911 | 0.1481136213280544 | 0.5675256636152245 |
| 50 | 0.5780687724910036 | 0.5592906187213443 | -0.1607357057177129 | 0.39318089568393993 | 0.4967470340604669 | 1.250497817993164 | 0.16504118026274056 | 0.5977966491076432 |
| 100 | 0.593562574970012 | 0.574336792057934 | -0.14524190323870456 | 0.37261534298471116 | 0.4978951396861845 | 1.2090015411376953 | 0.15426075826140223 | 0.577227121497971 |

All 7 BN-adapted prefixes have independently recomputable persisted
probability evidence (`bn_adapted_probs`, shape `[7, 10004, 9]`, indexed
by `bn_adapted_prefix_sequence=[1,2,5,10,25,50,100]`) -- see Section 5.

### Latency

| N | TTA latency (s) | Per-sample (s) | Compute multiplier |
|---|---|---|---|
| clean | 0.18177650100551546 | — | 1.0 (reference) |
| 1 | 0.20069878682261333 | 2.006185394068506e-05 | 1.1040964355261944 |
| 2 | 0.40049379598349333 | 4.003336625184859e-05 | 2.203220954128397 |
| 5 | 1.0539092881372198 | 0.00010534878929800278 | 5.797830205265323 |
| 10 | 2.0062332050292753 | 0.00020054310326162288 | 11.036812755948043 |
| 25 | 5.0343311165925115 | 0.0005032318189316785 | 27.69517010584201 |
| 50 | 10.047197701700497 | 0.0010043180429528686 | 55.27225821887531 |
| 100 | 20.138201316585764 | 0.0020130149256883012 | 110.78550420537984 |

`n_samples=10004`. Latency is descriptive telemetry, computed after all
scientific results, and never influenced any computed metric.

### Interpretation, stated neutrally

- This is **one validation cell and one seed**. It is a pipeline canary
  confirming the corrected metric-input contract executes end-to-end
  correctly on real data -- **not the final confirmatory conclusion**.
- **No threshold, policy, or analysis was changed because of this
  result.** The metric-contract correction that made attempt 4 possible
  was derived and frozen (`docs/phase2b_validation_evaluation_metric_contract_freeze.md`)
  entirely from a formula-level defect proof, before this attempt ran.
- **The test split remains untouched** (`test_metrics_observed=False`
  throughout; no test-loading code path exists in this pipeline).
- The observed harm (mean-probability TTA@N=50 costing ~28pp of
  accuracy relative to clean) is **consistent with the project's overall
  hypothesis that TTA can hurt**, but this single cell/seed cannot by
  itself establish that conclusion across the matrix -- it must be
  evaluated across all 39 cells and the full seed set before any such
  claim is made.

## 4. Independent verification

- **224+ independent metric recomputations, zero mismatches** (clean x5
  metrics; 3 aggregators x 7 prefixes x 8 fields for `naive_tta`;
  `original_anchored_tta` x 7 prefixes x 8 fields; `bn_adapted_tta` x 7
  prefixes x 8 fields), each compared under `atol=1e-6, rtol=1e-6`
  (`np.isclose`), recomputed independently and exactly from
  `predictions.npz` using `compute_metrics_from_probabilities()` /
  `_recompute_all_conditions_from_predictions()` -- functions that never
  call `softmax()`.
- **Probability normalization**: `clean_probs`, `view_probs`, and
  `bn_adapted_probs` are all finite, within `[0,1]`, and row-normalized
  to within `2.98e-7` of `1.0` (well inside the frozen `1e-4` validation
  tolerance).
- **Sample/label alignment**: `labels`/`sample_indices` are the same
  length as `clean_probs`' leading dimension, `sample_indices` are
  unique and exactly `0..10003` contiguous (no missing/duplicated
  sample).
- **Complete deterministic seed coverage**: the persisted
  `seed_manifest_sha256` was independently reconstructed from the
  frozen `stable_view_seed()` formula over all `10004 x 100 = 1,000,400`
  (sample, view) pairs and matched the persisted value exactly --
  `2374e80cc3f9da5f9c7b9412c2a645251d4c206f7cf54c7aeda077f02e226d71`.
- **Attempt-3/attempt-4 common arrays reproduced bitwise**: `labels`,
  `sample_indices`, `clean_probs`, and `view_probs` are byte-for-byte
  identical between attempts 3 and 4 (same checkpoint, dataset, seed,
  views, batching). The N=100 slice of attempt 4's stacked
  `bn_adapted_probs` array is also byte-for-byte identical to attempt
  3's single legacy array -- confirming the BN-adaptation computation
  itself is unaffected by the metric-contract/persistence-format
  correction; only the corrected metrics and the new stacked-array
  persistence format differ.
- **Corrected calibration metrics**: attempt 4's NLL/ECE/Brier differ
  from attempt 3's invalid persisted values everywhere (e.g. N=50
  mean-probability NLL `1.7255840301513672` vs. attempt 3's
  `2.024258613586426`; ECE `0.06311160995394459` vs. `0.3097403257793305`;
  Brier `0.6988638629651448` vs. `0.8438889114289245`), and match the
  predeclared recomputation from attempt-3's own persisted probabilities
  exactly -- confirming the fix, not a new inconsistency.
- **Full BN probability persistence**: all 7 registered N values have
  independently recomputable `bn_adapted_probs` evidence (Section 3),
  closing the artifact-design gap attempt 3 had (only N=100 persisted).
- **Semantic verification before completion**:
  `_verify_metrics_semantically()` recomputed every persisted metric
  from persisted probabilities and compared against `metrics.json`
  before `status="completed"` was ever written; attempt 4 completed,
  confirming this gate passed with zero discrepancies.
- **Checksum/manifest verification**: `artifact_manifest.json`
  independently reverified against the six files on disk
  (`verify_evaluation_artifact_manifest()`); dataset expected/actual MD5
  both equal `a8b06965200029087d5bd730944a56c1`, `resized=False`.

## 5. Limitations

- Single cell (`A-pathmnist-28px-batchnorm-policy-none-s0`), single
  seed. No population-level or cross-seed inference is licensed by this
  result alone.
- Validation split only -- the frozen test firewall means no
  test-split number exists anywhere in this document or its artifacts.
- Peak MPS memory was **not persisted** by this evaluation pipeline
  (that instrumentation exists only in the separate Block D benchmark
  module, not the validation-evaluation path) -- not reported here
  rather than estimated.
- Latency figures are descriptive telemetry only, measured after all
  scientific results were already computed; never used to select or
  gate anything.
- **Attempt 4 alone cannot establish novelty, a confirmed hypothesis
  outcome, or any publication-level conclusion.** It establishes that
  the corrected pipeline executes correctly, completely, and
  reproducibly end-to-end on real data for one cell.

## 6. Next steps

The remaining 38 confirmatory validation-evaluation cells may proceed
using this same corrected pipeline and configuration. A nonbinding
runtime projection and a recommended staged execution order (with a
stop-on-first-failure discipline) were reported alongside this audit
but are not persisted as a separate frozen document, since they are
estimates, not frozen protocol.
