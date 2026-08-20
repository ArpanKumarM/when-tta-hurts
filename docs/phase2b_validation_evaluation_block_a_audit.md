# Phase 2B Block A Validation-Evaluation Audit

**Recorded: 2026-08-20.** This document is the permanent record of Block
A's confirmatory validation-evaluation results: all 24 cells
(`A-*`, core normalization × resolution: PathMNIST + BloodMNIST ×
{28px, 64px} × {BatchNorm, GroupNorm} × 3 seeds). Every number in this
document was extracted programmatically from persisted
`metrics.json`/`metadata.json`/`status.json` files under
`artifacts/validation_evaluation/`, not transcribed by hand. No test
split was accessed at any point. No threshold, policy, or hypothesis
was changed because of any result reported here.

## 1. Exact 24-cell canonical mapping

Evaluation attempt column reflects the corrected mapping: the original
BatchNorm canary (`A-pathmnist-28px-batchnorm-policy-none-s0`) is
canonical at **evaluation attempt 4** (attempts 1-3 aborted/failed/
amendment-excluded); `A-pathmnist-28px-groupnorm-policy-none-s0` is
canonical at **evaluation attempt 2** (attempt 1 is the preserved
GroupNorm persistence-schema failure); every other cell is canonical at
evaluation attempt 1.

| run_id | dataset | res | norm | seed | train attempt | checkpoint hash | eval attempt | evaluation_id | evaluator_fingerprint | canonical-eligible | current-fp-compatible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-bloodmnist-28px-batchnorm-policy-none-s0` | bloodmnist | 28 | batchnorm | 0 | 1 | `e96e3b5615c71fba0fd05d21439c3d2a551ddba00985dbcc0b8affcfda5c519f` | 1 | `13e61b68dc9c34af49b11eeb394208ae506b26de654ac3d2ed7d829788f4170c` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-28px-batchnorm-policy-none-s1` | bloodmnist | 28 | batchnorm | 1 | 1 | `1448dad9629f956c195e301dabe42047502ccf2981d6f4f3154f5994a35da98f` | 1 | `ea473df526d2afe27b1328a2affd59f984d75437338be622ee339cddd910b47f` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-28px-batchnorm-policy-none-s2` | bloodmnist | 28 | batchnorm | 2 | 1 | `552c975c860585e9b57298d2e1fd3550420b9ad696066d2fe7bef49401a235f0` | 1 | `fb80a0bb0a5c5c3f6e7669058e707d87f9e9e9c383623ac3a86e29b6cd70a79a` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-28px-groupnorm-policy-none-s0` | bloodmnist | 28 | groupnorm | 0 | 1 | `cd804a09e799f7d89c1e98dad10a63a953a9a5204d98d5fa4c5677b7937765bd` | 1 | `5062aa9ea633b9ab670a53aa2890dc0934c869fa40e43bb5427f80a1f268e525` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-28px-groupnorm-policy-none-s1` | bloodmnist | 28 | groupnorm | 1 | 1 | `d914f4ecd0d597b9cc422dd7e5b07413c900eaa084b0f0992575010dfd899928` | 1 | `73767823eb091e4026e869580c5800484f8f46cc4a102cbd556fcec4e6b35568` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-28px-groupnorm-policy-none-s2` | bloodmnist | 28 | groupnorm | 2 | 1 | `34a1b18158c3572f281035592f8a3af33eb2baa11055aa9a8c2e6a60d495741d` | 1 | `187099e7c35bc78e61a67de2e0d604689df5d8ba5cdc78ffd5999a018b660c2c` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-batchnorm-policy-none-s0` | bloodmnist | 64 | batchnorm | 0 | 1 | `8572b19b0cb24b9c46ae07a1114eaa65ddad7690826265fe3af8984b2d0b6235` | 1 | `eaffb081d0e51dc592d032e5aeb0d721a890261759d228d75c0c92785a0fbcb0` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-batchnorm-policy-none-s1` | bloodmnist | 64 | batchnorm | 1 | 1 | `ff75298a2d1733dd1e43dd60d90a74a51450f80704c1e648e41eabf9ba7f4017` | 1 | `1cf8d8fec08f00c10459453f55414ad6a0d8081e261c39d9560ae9e9616720ed` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-batchnorm-policy-none-s2` | bloodmnist | 64 | batchnorm | 2 | 1 | `b2e83aa7e85050f6c3743c9e0a269428b3e1795d369a622328eb5b5e2abe2869` | 1 | `a978f829cad19f6343a5893abd8fef7bc95df89bb1ccdc19411960a3c29a00a5` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-groupnorm-policy-none-s0` | bloodmnist | 64 | groupnorm | 0 | 1 | `6dc1eb925bf71525353ddec56ac41b78e4d3e04b761ad47fb13ca5ae8d1443b0` | 1 | `c1b56b632c4e984027710c5a3ee1c541d456b1f97420ebe6771685a8928589cc` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-groupnorm-policy-none-s1` | bloodmnist | 64 | groupnorm | 1 | 1 | `2278d9cfd09a165210e53528eaa0f8567dba6b7303c0f5b21a09a20a652ec98b` | 1 | `eefa275d47905d5a1e7fd0071c6b1151d154c7aa79d69534832c3a73fdd61153` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-bloodmnist-64px-groupnorm-policy-none-s2` | bloodmnist | 64 | groupnorm | 2 | 1 | `9cbeb71c38c6d50873d5972ced6ca962cc1f38cde2bd080362351dd06df20fd9` | 1 | `f6c7484e058643a2d932fa3b939abee2c3551c15bada4e1034d7b6d6fc697f09` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-28px-batchnorm-policy-none-s0` | pathmnist | 28 | batchnorm | 0 | 3 | `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` | 4 | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | True | False |
| `A-pathmnist-28px-batchnorm-policy-none-s1` | pathmnist | 28 | batchnorm | 1 | 1 | `f3be88438078ce362528dc8c3919e7d395fb095902cca48159c861b6e308e6bf` | 1 | `d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | True | False |
| `A-pathmnist-28px-batchnorm-policy-none-s2` | pathmnist | 28 | batchnorm | 2 | 1 | `b8b971407b6b149d8e64e4bf55ec87f95a753cad5256aa5f3a57d419145d9c06` | 1 | `add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | True | False |
| `A-pathmnist-28px-groupnorm-policy-none-s0` | pathmnist | 28 | groupnorm | 0 | 1 | `fcf6a2f41c136cadc012bab8726249062ed1a16290a98504b65903a96c234e98` | 2 | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-28px-groupnorm-policy-none-s1` | pathmnist | 28 | groupnorm | 1 | 2 | `b63d8c9b9a38b3520f5aeb971a4afb60d760a7bffc707a63ce6c344abe5a4650` | 1 | `59fddf260eacdcf5907159914b9505f09245c5fdf139ceda7a5631dcf5a16ed9` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-28px-groupnorm-policy-none-s2` | pathmnist | 28 | groupnorm | 2 | 1 | `483eaa5957f570981a43933d3985b714b56911abc75ccbdebf5db8c6cdc9cdd3` | 1 | `8dd7923944a5409b9c842f5a002299acc175a484f0fc5006072926e37951b277` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-batchnorm-policy-none-s0` | pathmnist | 64 | batchnorm | 0 | 1 | `b5dea833b8985d7bf38e4092e5400a03452b0ec55a79ec7e52edd0091e6b3b54` | 1 | `bd78f398e9473f6de13d1454e81e8bdaf4b14ce0cc293355bc722d9da679dea0` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-batchnorm-policy-none-s1` | pathmnist | 64 | batchnorm | 1 | 1 | `b898155b2c6ab101ca2940f30bceff8a23a7f67ef9adc56d7c07ab2275831c8e` | 1 | `d7410d7afb0da70ca3ab89df9428bac05158dafd2a6119018079769df42a3ab0` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-batchnorm-policy-none-s2` | pathmnist | 64 | batchnorm | 2 | 1 | `39950a5ac56c50aeee11d3a6ca879baabfcc19506ce041f046fbff10243fe293` | 1 | `1a3c3c7e2479439166df8eea3fe27987669c19c6daed807801ab7229c4c00405` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-groupnorm-policy-none-s0` | pathmnist | 64 | groupnorm | 0 | 1 | `d54b5cd32b64882bc11030992db97ce57973df2078e37317871d1c66c92e484b` | 1 | `4f844fa44b7f8842632fc14b1c871545dd4e9e4b00a23368bd47ba7a6e406fde` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-groupnorm-policy-none-s1` | pathmnist | 64 | groupnorm | 1 | 1 | `ebfef2c0e60ae1ee1095f6c47afd6dbad3c7abc2c32ebc5f21e502a9f33ee44e` | 1 | `7a94cc438eec41cfd7e6ab6bc97d2584b8ad72829bde5601bcdd5c2d2367909f` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
| `A-pathmnist-64px-groupnorm-policy-none-s2` | pathmnist | 64 | groupnorm | 2 | 2 | `92c8739bfc1063b03782514c7c2b259cb3014ea5f0262f64e138be29e8c1ecd2` | 1 | `90ba07e9fd99e802b65b754807ab9e196999b5fb5c24100ca10894f533afbf46` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | True | True |
## 2. Exact historical-attempt accounting

For `A-pathmnist-28px-batchnorm-policy-none-s0`:

| Attempt | evaluation_id | Status | Disposition |
|---|---|---|---|
| 1 | `ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5` | `aborted` | Test-harness escape; no metric ever computed |
| 2 | `96fbf4705bf93f4e2115fb33b9837df1095c90549d1f86ed1b1c1c160cc7fffe` | `failed` | BN-adaptation OOM (`Invalid buffer size: 9.35 GiB`) |
| 3 | `75aa7e37a9fe5454bf8edf6483d676a182d6dde9ff4a3730e4ada7195e09eb9e` | `completed`, **amendment-excluded** | Double-softmax metric-contract defect (`reason=probability_metric_double_softmax`); accuracy/macro-F1/harm/rescue unaffected, NLL/ECE/Brier invalid |
| **4** | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` | `completed`, **canonical** | First technically valid completion under `probability_native_v1` |

For `A-pathmnist-28px-groupnorm-policy-none-s0`:

| Attempt | evaluation_id | Status | Disposition |
|---|---|---|---|
| 1 | `2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b` | `failed` | GroupNorm persistence-schema defect (`batching.bn_adaptation_microbatches_at_primary_n must be a nonnegative integer, got None.`) |
| **2** | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` | `completed`, **canonical** | First technically valid completion under the corrected GroupNorm applicability contract |

All other 22 Block A cells completed cleanly on evaluation attempt 1,
first try, no prior failed/aborted attempt.

Ledger totals: **28 data rows** -- **25 completed** (24 distinct
canonical Block A cells + the historical, amendment-excluded attempt 3
above, counted once each; attempt 3 and attempt 4 are two separate
completed rows for the same `training_run_id`), **2 failed** (the
BN-adaptation OOM and the GroupNorm schema defect), **1 aborted** (the
original test-harness escape). The evaluation-amendments ledger remains
byte-identical to its first commit (`6001bf0`), exactly one row
(attempt 3's exclusion).


## 3. Per-cell scientific results (all 24 cells)

Every cell's clean metrics and N=50 results across all four aggregators
(mean-probability, majority-vote, confidence-weighted, original-
anchored) plus BN-adapted (BatchNorm only -- GroupNorm cells correctly
report **"not applicable"**, never a fabricated zero value). Values are
exact, extracted directly from each cell's persisted `metrics.json`.

#### `A-bloodmnist-28px-batchnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.892523 | 0.584112 | 0.586449 | 0.547313 | 0.599883 | 0.785631 |
| Delta accuracy | n/a | -0.308411 | -0.306075 | -0.345210 | -0.292640 | -0.106893 |
| Macro-F1 | 0.887763 | 0.479058 | 0.478050 | 0.458413 | 0.489835 | 0.776794 |
| NLL | 0.330177 | 1.033449 | 1.216235 | 1.049646 | 0.999954 | 0.567968 |
| ECE | 0.022967 | 0.107315 | 0.114357 | 0.068949 | 0.123424 | 0.040502 |
| Brier | 0.166526 | 0.539842 | 0.533278 | 0.558426 | 0.524215 | 0.293672 |
| Harm rate | n/a | 0.380236 | 0.378927 | 0.414921 | 0.361911 | 0.136126 |
| Rescue rate | n/a | 0.288043 | 0.298913 | 0.233696 | 0.282609 | 0.135870 |


#### `A-bloodmnist-28px-batchnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.918224 | 0.571846 | 0.566005 | 0.571262 | 0.595210 | 0.734813 |
| Delta accuracy | n/a | -0.346379 | -0.352220 | -0.346963 | -0.323014 | -0.183411 |
| Macro-F1 | 0.907287 | 0.467532 | 0.460661 | 0.466423 | 0.493977 | 0.727469 |
| NLL | 0.242306 | 1.170985 | 1.600038 | 1.146304 | 1.111802 | 0.688762 |
| ECE | 0.017754 | 0.099176 | 0.064790 | 0.079132 | 0.110131 | 0.058544 |
| Brier | 0.123539 | 0.556272 | 0.551768 | 0.545754 | 0.539424 | 0.362504 |
| Harm rate | n/a | 0.405852 | 0.410941 | 0.402672 | 0.380407 | 0.208651 |
| Rescue rate | n/a | 0.321429 | 0.307143 | 0.278571 | 0.321429 | 0.100000 |


#### `A-bloodmnist-28px-batchnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.956776 | 0.456776 | 0.463785 | 0.463785 | 0.471379 | 0.812500 |
| Delta accuracy | n/a | -0.500000 | -0.492991 | -0.492991 | -0.485397 | -0.144276 |
| Macro-F1 | 0.952301 | 0.349863 | 0.354952 | 0.362039 | 0.370248 | 0.788128 |
| NLL | 0.130131 | 1.365915 | 2.530972 | 1.369406 | 1.227196 | 0.575122 |
| ECE | 0.008684 | 0.129137 | 0.140152 | 0.140423 | 0.115715 | 0.055745 |
| Brier | 0.066272 | 0.650423 | 0.657652 | 0.651295 | 0.627554 | 0.269985 |
| Harm rate | n/a | 0.546398 | 0.538462 | 0.539072 | 0.531136 | 0.163004 |
| Rescue rate | n/a | 0.527027 | 0.513514 | 0.527027 | 0.527027 | 0.270270 |


#### `A-bloodmnist-28px-groupnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.946846 | 0.741238 | 0.737734 | 0.738318 | 0.752921 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.205607 | -0.209112 | -0.208528 | -0.193925 | not applicable (GroupNorm) |
| Macro-F1 | 0.941182 | 0.656302 | 0.656525 | 0.655199 | 0.674955 | not applicable (GroupNorm) |
| NLL | 0.166194 | 0.845843 | 1.758675 | 0.836879 | 0.807255 | not applicable (GroupNorm) |
| ECE | 0.028275 | 0.137478 | 0.088341 | 0.117070 | 0.146685 | not applicable (GroupNorm) |
| Brier | 0.083711 | 0.393318 | 0.389732 | 0.384220 | 0.381707 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.241826 | 0.244911 | 0.244911 | 0.229488 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.439560 | 0.428571 | 0.439560 | 0.439560 | not applicable (GroupNorm) |


#### `A-bloodmnist-28px-groupnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.936332 | 0.712033 | 0.700350 | 0.717290 | 0.724883 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.224299 | -0.235981 | -0.219042 | -0.211449 | not applicable (GroupNorm) |
| Macro-F1 | 0.930064 | 0.656338 | 0.632519 | 0.648439 | 0.670759 | not applicable (GroupNorm) |
| NLL | 0.200026 | 0.986111 | 2.234738 | 0.964739 | 0.911941 | not applicable (GroupNorm) |
| ECE | 0.029033 | 0.131557 | 0.070117 | 0.126610 | 0.142368 | not applicable (GroupNorm) |
| Brier | 0.102379 | 0.428240 | 0.428209 | 0.413848 | 0.415880 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.262633 | 0.275733 | 0.257642 | 0.248908 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.339450 | 0.348624 | 0.348624 | 0.339450 | not applicable (GroupNorm) |


#### `A-bloodmnist-28px-groupnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.938668 | 0.675234 | 0.678738 | 0.679907 | 0.689252 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.263435 | -0.259930 | -0.258762 | -0.249416 | not applicable (GroupNorm) |
| Macro-F1 | 0.931881 | 0.574413 | 0.585189 | 0.582948 | 0.592677 | not applicable (GroupNorm) |
| NLL | 0.191903 | 0.957830 | 2.137716 | 0.944764 | 0.906201 | not applicable (GroupNorm) |
| ECE | 0.031850 | 0.113045 | 0.075245 | 0.103948 | 0.108484 | not applicable (GroupNorm) |
| Brier | 0.096572 | 0.436350 | 0.436777 | 0.425573 | 0.423663 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.300560 | 0.297449 | 0.294337 | 0.285625 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.304762 | 0.314286 | 0.285714 | 0.304762 | not applicable (GroupNorm) |


#### `A-bloodmnist-64px-batchnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.907710 | 0.375000 | 0.382009 | 0.352804 | 0.401285 | 0.764603 |
| Delta accuracy | n/a | -0.532710 | -0.525701 | -0.554907 | -0.506425 | -0.143107 |
| Macro-F1 | 0.912985 | 0.299895 | 0.317175 | 0.291146 | 0.328970 | 0.690245 |
| NLL | 0.345386 | 1.390797 | 1.660493 | 1.394440 | 1.345196 | 0.625257 |
| ECE | 0.102276 | 0.123882 | 0.125911 | 0.155070 | 0.107028 | 0.073480 |
| Brier | 0.163500 | 0.687786 | 0.678364 | 0.695656 | 0.669555 | 0.312779 |
| Harm rate | n/a | 0.597812 | 0.588160 | 0.620978 | 0.568855 | 0.214286 |
| Rescue rate | n/a | 0.107595 | 0.088608 | 0.094937 | 0.107595 | 0.556962 |


#### `A-bloodmnist-64px-batchnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.922897 | 0.251168 | 0.254089 | 0.232477 | 0.273364 | 0.798481 |
| Delta accuracy | n/a | -0.671729 | -0.668808 | -0.690421 | -0.649533 | -0.124416 |
| Macro-F1 | 0.914989 | 0.201962 | 0.207772 | 0.182215 | 0.224871 | 0.731906 |
| NLL | 0.272056 | 1.558961 | 1.882834 | 1.621021 | 1.492599 | 0.617842 |
| ECE | 0.082586 | 0.187010 | 0.206741 | 0.240166 | 0.169112 | 0.116675 |
| Brier | 0.133176 | 0.783829 | 0.784664 | 0.817019 | 0.760326 | 0.301339 |
| Harm rate | n/a | 0.736709 | 0.732911 | 0.756329 | 0.713291 | 0.167722 |
| Rescue rate | n/a | 0.106061 | 0.098485 | 0.098485 | 0.113636 | 0.393939 |


#### `A-bloodmnist-64px-batchnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.923481 | 0.420561 | 0.439252 | 0.398364 | 0.445678 | 0.745327 |
| Delta accuracy | n/a | -0.502921 | -0.484229 | -0.525117 | -0.477804 | -0.178154 |
| Macro-F1 | 0.915508 | 0.315268 | 0.338984 | 0.302036 | 0.346362 | 0.644515 |
| NLL | 0.311542 | 1.385558 | 1.465202 | 1.424329 | 1.336726 | 0.730516 |
| ECE | 0.107325 | 0.146798 | 0.150304 | 0.163672 | 0.134761 | 0.093506 |
| Brier | 0.140922 | 0.665507 | 0.641345 | 0.684525 | 0.647434 | 0.352006 |
| Harm rate | n/a | 0.550917 | 0.531309 | 0.575585 | 0.523719 | 0.217584 |
| Rescue rate | n/a | 0.076336 | 0.083969 | 0.083969 | 0.076336 | 0.297710 |


#### `A-bloodmnist-64px-groupnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.966121 | 0.502921 | 0.492407 | 0.501168 | 0.510514 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.463201 | -0.473715 | -0.464953 | -0.455607 | not applicable (GroupNorm) |
| Macro-F1 | 0.965245 | 0.453817 | 0.439881 | 0.450075 | 0.463583 | not applicable (GroupNorm) |
| NLL | 0.127737 | 1.295840 | 2.150489 | 1.326537 | 1.203214 | not applicable (GroupNorm) |
| ECE | 0.036650 | 0.140608 | 0.153143 | 0.150914 | 0.144817 | not applicable (GroupNorm) |
| Brier | 0.059761 | 0.582290 | 0.597921 | 0.593427 | 0.562864 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.485490 | 0.495768 | 0.487304 | 0.477025 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.172414 | 0.155172 | 0.172414 | 0.155172 | not applicable (GroupNorm) |


#### `A-bloodmnist-64px-groupnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.960864 | 0.553738 | 0.533294 | 0.564836 | 0.564836 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.407126 | -0.427570 | -0.396028 | -0.396028 | not applicable (GroupNorm) |
| Macro-F1 | 0.957624 | 0.491118 | 0.455242 | 0.503345 | 0.502500 | not applicable (GroupNorm) |
| NLL | 0.149551 | 1.068422 | 1.563390 | 1.077075 | 1.017838 | not applicable (GroupNorm) |
| ECE | 0.043779 | 0.088275 | 0.094194 | 0.088823 | 0.098485 | not applicable (GroupNorm) |
| Brier | 0.070989 | 0.524214 | 0.537648 | 0.522767 | 0.507509 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.434650 | 0.457143 | 0.421277 | 0.423100 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.268657 | 0.298507 | 0.223881 | 0.268657 | not applicable (GroupNorm) |


#### `A-bloodmnist-64px-groupnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.971963 | 0.387850 | 0.384930 | 0.389603 | 0.404206 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.584112 | -0.587033 | -0.582360 | -0.567757 | not applicable (GroupNorm) |
| Macro-F1 | 0.972004 | 0.351121 | 0.350104 | 0.354225 | 0.373370 | not applicable (GroupNorm) |
| NLL | 0.124045 | 1.240469 | 1.801655 | 1.271803 | 1.168234 | not applicable (GroupNorm) |
| ECE | 0.039133 | 0.241924 | 0.263540 | 0.257153 | 0.224298 | not applicable (GroupNorm) |
| Brier | 0.055570 | 0.642303 | 0.679767 | 0.660090 | 0.620495 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.602764 | 0.606370 | 0.600962 | 0.585938 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.062500 | 0.083333 | 0.062500 | 0.062500 | not applicable (GroupNorm) |


#### `A-pathmnist-28px-batchnorm-policy-none-s0` (evaluation attempt 4)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.738804 | 0.458417 | 0.434826 | 0.461915 | 0.479008 | 0.578069 |
| Delta accuracy | n/a | -0.280388 | -0.303978 | -0.276889 | -0.259796 | -0.160736 |
| Macro-F1 | 0.724357 | 0.416744 | 0.398542 | 0.421134 | 0.438908 | 0.559291 |
| NLL | 0.914932 | 1.725584 | 4.077648 | 1.737415 | 1.665503 | 1.250498 |
| ECE | 0.090000 | 0.063112 | 0.044204 | 0.054082 | 0.082648 | 0.165041 |
| Brier | 0.381817 | 0.698864 | 0.712317 | 0.691322 | 0.684145 | 0.597797 |
| Harm rate | n/a | 0.429577 | 0.461507 | 0.426059 | 0.400081 | 0.393181 |
| Rescue rate | n/a | 0.141600 | 0.141600 | 0.145044 | 0.137007 | 0.496747 |


#### `A-pathmnist-28px-batchnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.785286 | 0.222011 | 0.221311 | 0.219612 | 0.233507 | 0.465914 |
| Delta accuracy | n/a | -0.563275 | -0.563974 | -0.565674 | -0.551779 | -0.319372 |
| Macro-F1 | 0.784397 | 0.171661 | 0.172442 | 0.164227 | 0.186218 | 0.450428 |
| NLL | 0.636928 | 1.939851 | 4.396403 | 2.012502 | 1.831391 | 1.521468 |
| ECE | 0.050372 | 0.264110 | 0.289772 | 0.302160 | 0.249850 | 0.256877 |
| Brier | 0.308310 | 0.861296 | 0.891824 | 0.894075 | 0.838188 | 0.725839 |
| Harm rate | n/a | 0.747581 | 0.748218 | 0.750509 | 0.732943 | 0.467032 |
| Rescue rate | n/a | 0.110801 | 0.109870 | 0.110335 | 0.110801 | 0.220670 |


#### `A-pathmnist-28px-batchnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.870152 | 0.261196 | 0.260296 | 0.255398 | 0.272591 | 0.609256 |
| Delta accuracy | n/a | -0.608956 | -0.609856 | -0.614754 | -0.597561 | -0.260896 |
| Macro-F1 | 0.870362 | 0.179904 | 0.181275 | 0.171899 | 0.195146 | 0.601649 |
| NLL | 0.399272 | 1.911149 | 3.825993 | 1.975026 | 1.753149 | 1.361288 |
| ECE | 0.034341 | 0.228525 | 0.226897 | 0.259660 | 0.220244 | 0.202836 |
| Brier | 0.190049 | 0.809174 | 0.821431 | 0.830838 | 0.784123 | 0.597358 |
| Harm rate | n/a | 0.704767 | 0.706146 | 0.711200 | 0.691901 | 0.382654 |
| Rescue rate | n/a | 0.033102 | 0.035412 | 0.031563 | 0.034642 | 0.555042 |


#### `A-pathmnist-28px-groupnorm-policy-none-s0` (evaluation attempt 2)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.930228 | 0.366653 | 0.345562 | 0.365654 | 0.382047 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.563575 | -0.584666 | -0.564574 | -0.548181 | not applicable (GroupNorm) |
| Macro-F1 | 0.930285 | 0.315438 | 0.293812 | 0.312305 | 0.332468 | not applicable (GroupNorm) |
| NLL | 0.208438 | 2.073582 | 5.963950 | 2.094491 | 1.741959 | not applicable (GroupNorm) |
| ECE | 0.016133 | 0.112321 | 0.148950 | 0.132028 | 0.107119 | not applicable (GroupNorm) |
| Brier | 0.106254 | 0.756608 | 0.784957 | 0.759490 | 0.731758 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.614227 | 0.635611 | 0.614765 | 0.597679 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.111748 | 0.094556 | 0.104585 | 0.111748 | not applicable (GroupNorm) |


#### `A-pathmnist-28px-groupnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.945222 | 0.277689 | 0.273790 | 0.270492 | 0.291583 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.667533 | -0.671431 | -0.674730 | -0.653639 | not applicable (GroupNorm) |
| Macro-F1 | 0.945055 | 0.198880 | 0.192548 | 0.190519 | 0.215323 | not applicable (GroupNorm) |
| NLL | 0.162055 | 2.589658 | 7.365030 | 2.646130 | 1.992970 | not applicable (GroupNorm) |
| ECE | 0.013889 | 0.175657 | 0.196893 | 0.209296 | 0.163525 | not applicable (GroupNorm) |
| Brier | 0.082852 | 0.841425 | 0.868228 | 0.859833 | 0.812264 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.711083 | 0.714573 | 0.718063 | 0.696489 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.083942 | 0.072993 | 0.072993 | 0.085766 | not applicable (GroupNorm) |


#### `A-pathmnist-28px-groupnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.951719 | 0.266393 | 0.262995 | 0.258796 | 0.278089 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.685326 | -0.688725 | -0.692923 | -0.673631 | not applicable (GroupNorm) |
| Macro-F1 | 0.951697 | 0.204280 | 0.200989 | 0.194005 | 0.219421 | not applicable (GroupNorm) |
| NLL | 0.147401 | 2.567515 | 6.453902 | 2.631331 | 1.874999 | not applicable (GroupNorm) |
| ECE | 0.009040 | 0.221926 | 0.245342 | 0.266025 | 0.207380 | not applicable (GroupNorm) |
| Brier | 0.074431 | 0.832409 | 0.854738 | 0.860667 | 0.803187 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.722508 | 0.725974 | 0.730385 | 0.710325 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.047619 | 0.045549 | 0.045549 | 0.049689 | not applicable (GroupNorm) |


#### `A-pathmnist-64px-batchnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.910036 | 0.477409 | 0.476809 | 0.470512 | 0.499100 | 0.753898 |
| Delta accuracy | n/a | -0.432627 | -0.433227 | -0.439524 | -0.410936 | -0.156138 |
| Macro-F1 | 0.910905 | 0.417084 | 0.413309 | 0.417318 | 0.446293 | 0.752648 |
| NLL | 0.262940 | 1.378968 | 2.454875 | 1.382086 | 1.307410 | 0.699302 |
| ECE | 0.008190 | 0.063472 | 0.065356 | 0.080032 | 0.055902 | 0.087925 |
| Brier | 0.134738 | 0.627039 | 0.636876 | 0.630967 | 0.608544 | 0.358827 |
| Harm rate | n/a | 0.481986 | 0.482865 | 0.488027 | 0.457821 | 0.205185 |
| Rescue rate | n/a | 0.066667 | 0.068889 | 0.051111 | 0.063333 | 0.340000 |


#### `A-pathmnist-64px-batchnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.910136 | 0.493303 | 0.472911 | 0.492403 | 0.511096 | 0.820772 |
| Delta accuracy | n/a | -0.416833 | -0.437225 | -0.417733 | -0.399040 | -0.089364 |
| Macro-F1 | 0.906103 | 0.447248 | 0.427547 | 0.444390 | 0.469074 | 0.812291 |
| NLL | 0.255148 | 1.417769 | 2.108550 | 1.422268 | 1.340583 | 0.602020 |
| ECE | 0.023306 | 0.089681 | 0.085494 | 0.087647 | 0.094594 | 0.066922 |
| Brier | 0.127749 | 0.641033 | 0.653231 | 0.638949 | 0.620717 | 0.276012 |
| Harm rate | n/a | 0.470621 | 0.491378 | 0.472268 | 0.451181 | 0.171334 |
| Rescue rate | n/a | 0.127920 | 0.111235 | 0.134594 | 0.129032 | 0.740823 |


#### `A-pathmnist-64px-batchnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.936925 | 0.581168 | 0.569072 | 0.582867 | 0.599860 | 0.744402 |
| Delta accuracy | n/a | -0.355758 | -0.367853 | -0.354058 | -0.337065 | -0.192523 |
| Macro-F1 | 0.937995 | 0.533421 | 0.516709 | 0.543148 | 0.555410 | 0.722447 |
| NLL | 0.185776 | 1.312642 | 2.240571 | 1.299025 | 1.234225 | 0.817154 |
| ECE | 0.012805 | 0.097660 | 0.057269 | 0.067806 | 0.115486 | 0.120879 |
| Brier | 0.097922 | 0.577504 | 0.582354 | 0.571106 | 0.559113 | 0.378916 |
| Harm rate | n/a | 0.385256 | 0.398272 | 0.383442 | 0.365091 | 0.232156 |
| Rescue rate | n/a | 0.082409 | 0.083994 | 0.082409 | 0.079239 | 0.396197 |


#### `A-pathmnist-64px-groupnorm-policy-none-s0` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.979608 | 0.465614 | 0.447721 | 0.468213 | 0.489004 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.513994 | -0.531887 | -0.511395 | -0.490604 | not applicable (GroupNorm) |
| Macro-F1 | 0.979496 | 0.408753 | 0.387709 | 0.413177 | 0.436565 | not applicable (GroupNorm) |
| NLL | 0.072099 | 1.483892 | 2.439037 | 1.483755 | 1.359324 | not applicable (GroupNorm) |
| ECE | 0.012695 | 0.128499 | 0.134506 | 0.127045 | 0.122209 | not applicable (GroupNorm) |
| Brier | 0.034148 | 0.652161 | 0.667878 | 0.648313 | 0.628511 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.528469 | 0.546531 | 0.525612 | 0.504694 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.181373 | 0.171569 | 0.171569 | 0.186275 | not applicable (GroupNorm) |


#### `A-pathmnist-64px-groupnorm-policy-none-s1` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.981507 | 0.331467 | 0.325170 | 0.324870 | 0.344462 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.650040 | -0.656337 | -0.656637 | -0.637045 | not applicable (GroupNorm) |
| Macro-F1 | 0.981141 | 0.271202 | 0.263279 | 0.265818 | 0.287145 | not applicable (GroupNorm) |
| NLL | 0.068478 | 1.865346 | 3.867721 | 1.918378 | 1.637843 | not applicable (GroupNorm) |
| ECE | 0.014552 | 0.165867 | 0.192255 | 0.198506 | 0.151871 | not applicable (GroupNorm) |
| Brier | 0.031718 | 0.780999 | 0.804084 | 0.798226 | 0.752265 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.664019 | 0.670027 | 0.670639 | 0.650779 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.091892 | 0.070270 | 0.086486 | 0.091892 | not applicable (GroupNorm) |


#### `A-pathmnist-64px-groupnorm-policy-none-s2` (evaluation attempt 1)

| Metric | Clean | N=50 mean-prob | N=50 majority-vote | N=50 conf-weighted | N=50 original-anchored | N=50 BN-adapted |
|---|---|---|---|---|---|---|
| Accuracy | 0.983107 | 0.417033 | 0.388844 | 0.422231 | 0.438824 | not applicable (GroupNorm) |
| Delta accuracy | n/a | -0.566074 | -0.594262 | -0.560876 | -0.544282 | not applicable (GroupNorm) |
| Macro-F1 | 0.982952 | 0.349830 | 0.318931 | 0.357084 | 0.377281 | not applicable (GroupNorm) |
| NLL | 0.062615 | 1.599676 | 2.654930 | 1.597914 | 1.453541 | not applicable (GroupNorm) |
| ECE | 0.011519 | 0.096863 | 0.111401 | 0.107126 | 0.093242 | not applicable (GroupNorm) |
| Brier | 0.028663 | 0.691385 | 0.710842 | 0.686834 | 0.665951 | not applicable (GroupNorm) |
| Harm rate | n/a | 0.577224 | 0.605694 | 0.571734 | 0.555058 | not applicable (GroupNorm) |
| Rescue rate | n/a | 0.082840 | 0.071006 | 0.071006 | 0.082840 | not applicable (GroupNorm) |

## 4. Three-seed descriptive summaries

Per dataset × resolution × normalization condition (8 conditions, 3
seeds each). **Descriptive only** -- no threshold selection, no
hypothesis modification, no test-set claim. These are validation-stage
summaries of a 3-seed sample per condition; they characterize this
project's seed variability, not a confirmed population effect.

### bloodmnist 28px batchnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.892523 | 0.918224 | 0.956776 | 0.922508 | 0.032340 | 0.892523 | 0.956776 |
| Clean macro-F1 | 0.887763 | 0.907287 | 0.952301 | 0.915784 | 0.033098 | 0.887763 | 0.952301 |
| Clean NLL | 0.330177 | 0.242306 | 0.130131 | 0.234205 | 0.100269 | 0.130131 | 0.330177 |
| Clean ECE | 0.022967 | 0.017754 | 0.008684 | 0.016468 | 0.007228 | 0.008684 | 0.022967 |
| Clean Brier | 0.166526 | 0.123539 | 0.066272 | 0.118779 | 0.050296 | 0.066272 | 0.166526 |
| N=50 mean-prob accuracy | 0.584112 | 0.571846 | 0.456776 | 0.537578 | 0.070245 | 0.456776 | 0.584112 |
| N=50 mean-prob delta accuracy | -0.308411 | -0.346379 | -0.500000 | -0.384930 | 0.101446 | -0.500000 | -0.308411 |
| N=50 mean-prob macro-F1 | 0.479058 | 0.467532 | 0.349863 | 0.432151 | 0.071496 | 0.349863 | 0.479058 |
| N=50 mean-prob NLL | 1.033449 | 1.170985 | 1.365915 | 1.190117 | 0.167057 | 1.033449 | 1.365915 |
| N=50 mean-prob ECE | 0.107315 | 0.099176 | 0.129137 | 0.111876 | 0.015493 | 0.099176 | 0.129137 |
| N=50 mean-prob Brier | 0.539842 | 0.556272 | 0.650423 | 0.582179 | 0.059669 | 0.539842 | 0.650423 |
| N=50 mean-prob harm rate | 0.380236 | 0.405852 | 0.546398 | 0.444162 | 0.089461 | 0.380236 | 0.546398 |
| N=50 mean-prob rescue rate | 0.288043 | 0.321429 | 0.527027 | 0.378833 | 0.129421 | 0.288043 | 0.527027 |
| N=50 majority-vote accuracy | 0.586449 | 0.566005 | 0.463785 | 0.538746 | 0.065718 | 0.463785 | 0.586449 |
| N=50 confidence-weighted accuracy | 0.547313 | 0.571262 | 0.463785 | 0.527453 | 0.056424 | 0.463785 | 0.571262 |
| N=50 original-anchored accuracy | 0.599883 | 0.595210 | 0.471379 | 0.555491 | 0.072881 | 0.471379 | 0.599883 |
| N=50 BN-adapted accuracy | 0.785631 | 0.734813 | 0.812500 | 0.777648 | 0.039454 | 0.734813 | 0.812500 |
| N=50 BN-adapted delta accuracy | -0.106893 | -0.183411 | -0.144276 | -0.144860 | 0.038263 | -0.183411 | -0.106893 |
| N=50 BN-adapted NLL | 0.567968 | 0.688762 | 0.575122 | 0.610617 | 0.067770 | 0.567968 | 0.688762 |

### bloodmnist 28px groupnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.946846 | 0.936332 | 0.938668 | 0.940615 | 0.005521 | 0.936332 | 0.946846 |
| Clean macro-F1 | 0.941182 | 0.930064 | 0.931881 | 0.934376 | 0.005964 | 0.930064 | 0.941182 |
| Clean NLL | 0.166194 | 0.200026 | 0.191903 | 0.186041 | 0.017661 | 0.166194 | 0.200026 |
| Clean ECE | 0.028275 | 0.029033 | 0.031850 | 0.029720 | 0.001884 | 0.028275 | 0.031850 |
| Clean Brier | 0.083711 | 0.102379 | 0.096572 | 0.094221 | 0.009554 | 0.083711 | 0.102379 |
| N=50 mean-prob accuracy | 0.741238 | 0.712033 | 0.675234 | 0.709502 | 0.033075 | 0.675234 | 0.741238 |
| N=50 mean-prob delta accuracy | -0.205607 | -0.224299 | -0.263435 | -0.231114 | 0.029510 | -0.263435 | -0.205607 |
| N=50 mean-prob macro-F1 | 0.656302 | 0.656338 | 0.574413 | 0.629018 | 0.047289 | 0.574413 | 0.656338 |
| N=50 mean-prob NLL | 0.845843 | 0.986111 | 0.957830 | 0.929928 | 0.074180 | 0.845843 | 0.986111 |
| N=50 mean-prob ECE | 0.137478 | 0.131557 | 0.113045 | 0.127360 | 0.012746 | 0.113045 | 0.137478 |
| N=50 mean-prob Brier | 0.393318 | 0.428240 | 0.436350 | 0.419303 | 0.022866 | 0.393318 | 0.436350 |
| N=50 mean-prob harm rate | 0.241826 | 0.262633 | 0.300560 | 0.268340 | 0.029780 | 0.241826 | 0.300560 |
| N=50 mean-prob rescue rate | 0.439560 | 0.339450 | 0.304762 | 0.361257 | 0.069995 | 0.304762 | 0.439560 |
| N=50 majority-vote accuracy | 0.737734 | 0.700350 | 0.678738 | 0.705607 | 0.029847 | 0.678738 | 0.737734 |
| N=50 confidence-weighted accuracy | 0.738318 | 0.717290 | 0.679907 | 0.711838 | 0.029585 | 0.679907 | 0.738318 |
| N=50 original-anchored accuracy | 0.752921 | 0.724883 | 0.689252 | 0.722352 | 0.031909 | 0.689252 | 0.752921 |
| N=50 BN-adapted (all) | not applicable (GroupNorm) | | | | | | |

### bloodmnist 64px batchnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.907710 | 0.922897 | 0.923481 | 0.918030 | 0.008942 | 0.907710 | 0.923481 |
| Clean macro-F1 | 0.912985 | 0.914989 | 0.915508 | 0.914494 | 0.001332 | 0.912985 | 0.915508 |
| Clean NLL | 0.345386 | 0.272056 | 0.311542 | 0.309661 | 0.036701 | 0.272056 | 0.345386 |
| Clean ECE | 0.102276 | 0.082586 | 0.107325 | 0.097396 | 0.013071 | 0.082586 | 0.107325 |
| Clean Brier | 0.163500 | 0.133176 | 0.140922 | 0.145866 | 0.015755 | 0.133176 | 0.163500 |
| N=50 mean-prob accuracy | 0.375000 | 0.251168 | 0.420561 | 0.348910 | 0.087658 | 0.251168 | 0.420561 |
| N=50 mean-prob delta accuracy | -0.532710 | -0.671729 | -0.502921 | -0.569120 | 0.090102 | -0.671729 | -0.502921 |
| N=50 mean-prob macro-F1 | 0.299895 | 0.201962 | 0.315268 | 0.272375 | 0.061462 | 0.201962 | 0.315268 |
| N=50 mean-prob NLL | 1.390797 | 1.558961 | 1.385558 | 1.445105 | 0.098637 | 1.385558 | 1.558961 |
| N=50 mean-prob ECE | 0.123882 | 0.187010 | 0.146798 | 0.152563 | 0.031956 | 0.123882 | 0.187010 |
| N=50 mean-prob Brier | 0.687786 | 0.783829 | 0.665507 | 0.712374 | 0.062876 | 0.665507 | 0.783829 |
| N=50 mean-prob harm rate | 0.597812 | 0.736709 | 0.550917 | 0.628479 | 0.096618 | 0.550917 | 0.736709 |
| N=50 mean-prob rescue rate | 0.107595 | 0.106061 | 0.076336 | 0.096664 | 0.017621 | 0.076336 | 0.107595 |
| N=50 majority-vote accuracy | 0.382009 | 0.254089 | 0.439252 | 0.358450 | 0.094803 | 0.254089 | 0.439252 |
| N=50 confidence-weighted accuracy | 0.352804 | 0.232477 | 0.398364 | 0.327882 | 0.085706 | 0.232477 | 0.398364 |
| N=50 original-anchored accuracy | 0.401285 | 0.273364 | 0.445678 | 0.373442 | 0.089467 | 0.273364 | 0.445678 |
| N=50 BN-adapted accuracy | 0.764603 | 0.798481 | 0.745327 | 0.769470 | 0.026909 | 0.745327 | 0.798481 |
| N=50 BN-adapted delta accuracy | -0.143107 | -0.124416 | -0.178154 | -0.148559 | 0.027281 | -0.178154 | -0.124416 |
| N=50 BN-adapted NLL | 0.625257 | 0.617842 | 0.730516 | 0.657872 | 0.063021 | 0.617842 | 0.730516 |

### bloodmnist 64px groupnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.966121 | 0.960864 | 0.971963 | 0.966316 | 0.005552 | 0.960864 | 0.971963 |
| Clean macro-F1 | 0.965245 | 0.957624 | 0.972004 | 0.964958 | 0.007194 | 0.957624 | 0.972004 |
| Clean NLL | 0.127737 | 0.149551 | 0.124045 | 0.133778 | 0.013784 | 0.124045 | 0.149551 |
| Clean ECE | 0.036650 | 0.043779 | 0.039133 | 0.039854 | 0.003619 | 0.036650 | 0.043779 |
| Clean Brier | 0.059761 | 0.070989 | 0.055570 | 0.062107 | 0.007973 | 0.055570 | 0.070989 |
| N=50 mean-prob accuracy | 0.502921 | 0.553738 | 0.387850 | 0.481503 | 0.084992 | 0.387850 | 0.553738 |
| N=50 mean-prob delta accuracy | -0.463201 | -0.407126 | -0.584112 | -0.484813 | 0.090451 | -0.584112 | -0.407126 |
| N=50 mean-prob macro-F1 | 0.453817 | 0.491118 | 0.351121 | 0.432018 | 0.072500 | 0.351121 | 0.491118 |
| N=50 mean-prob NLL | 1.295840 | 1.068422 | 1.240469 | 1.201577 | 0.118593 | 1.068422 | 1.295840 |
| N=50 mean-prob ECE | 0.140608 | 0.088275 | 0.241924 | 0.156936 | 0.078115 | 0.088275 | 0.241924 |
| N=50 mean-prob Brier | 0.582290 | 0.524214 | 0.642303 | 0.582936 | 0.059047 | 0.524214 | 0.642303 |
| N=50 mean-prob harm rate | 0.485490 | 0.434650 | 0.602764 | 0.507635 | 0.086217 | 0.434650 | 0.602764 |
| N=50 mean-prob rescue rate | 0.172414 | 0.268657 | 0.062500 | 0.167857 | 0.103154 | 0.062500 | 0.268657 |
| N=50 majority-vote accuracy | 0.492407 | 0.533294 | 0.384930 | 0.470210 | 0.076632 | 0.384930 | 0.533294 |
| N=50 confidence-weighted accuracy | 0.501168 | 0.564836 | 0.389603 | 0.485202 | 0.088701 | 0.389603 | 0.564836 |
| N=50 original-anchored accuracy | 0.510514 | 0.564836 | 0.404206 | 0.493185 | 0.081705 | 0.404206 | 0.564836 |
| N=50 BN-adapted (all) | not applicable (GroupNorm) | | | | | | |

### pathmnist 28px batchnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.738804 | 0.785286 | 0.870152 | 0.798081 | 0.066602 | 0.738804 | 0.870152 |
| Clean macro-F1 | 0.724357 | 0.784397 | 0.870362 | 0.793039 | 0.073385 | 0.724357 | 0.870362 |
| Clean NLL | 0.914932 | 0.636928 | 0.399272 | 0.650377 | 0.258093 | 0.399272 | 0.914932 |
| Clean ECE | 0.090000 | 0.050372 | 0.034341 | 0.058238 | 0.028651 | 0.034341 | 0.090000 |
| Clean Brier | 0.381817 | 0.308310 | 0.190049 | 0.293392 | 0.096750 | 0.190049 | 0.381817 |
| N=50 mean-prob accuracy | 0.458417 | 0.222011 | 0.261196 | 0.313874 | 0.126701 | 0.222011 | 0.458417 |
| N=50 mean-prob delta accuracy | -0.280388 | -0.563275 | -0.608956 | -0.484206 | 0.177984 | -0.608956 | -0.280388 |
| N=50 mean-prob macro-F1 | 0.416744 | 0.171661 | 0.179904 | 0.256103 | 0.139180 | 0.171661 | 0.416744 |
| N=50 mean-prob NLL | 1.725584 | 1.939851 | 1.911149 | 1.858861 | 0.116310 | 1.725584 | 1.939851 |
| N=50 mean-prob ECE | 0.063112 | 0.264110 | 0.228525 | 0.185249 | 0.107260 | 0.063112 | 0.264110 |
| N=50 mean-prob Brier | 0.698864 | 0.861296 | 0.809174 | 0.789778 | 0.082935 | 0.698864 | 0.861296 |
| N=50 mean-prob harm rate | 0.429577 | 0.747581 | 0.704767 | 0.627308 | 0.172574 | 0.429577 | 0.747581 |
| N=50 mean-prob rescue rate | 0.141600 | 0.110801 | 0.033102 | 0.095168 | 0.055913 | 0.033102 | 0.141600 |
| N=50 majority-vote accuracy | 0.434826 | 0.221311 | 0.260296 | 0.305478 | 0.113702 | 0.221311 | 0.434826 |
| N=50 confidence-weighted accuracy | 0.461915 | 0.219612 | 0.255398 | 0.312308 | 0.130793 | 0.219612 | 0.461915 |
| N=50 original-anchored accuracy | 0.479008 | 0.233507 | 0.272591 | 0.328369 | 0.131913 | 0.233507 | 0.479008 |
| N=50 BN-adapted accuracy | 0.578069 | 0.465914 | 0.609256 | 0.551080 | 0.075386 | 0.465914 | 0.609256 |
| N=50 BN-adapted delta accuracy | -0.160736 | -0.319372 | -0.260896 | -0.247001 | 0.080226 | -0.319372 | -0.160736 |
| N=50 BN-adapted NLL | 1.250498 | 1.521468 | 1.361288 | 1.377751 | 0.136233 | 1.250498 | 1.521468 |

### pathmnist 28px groupnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.930228 | 0.945222 | 0.951719 | 0.942390 | 0.011022 | 0.930228 | 0.951719 |
| Clean macro-F1 | 0.930285 | 0.945055 | 0.951697 | 0.942346 | 0.010960 | 0.930285 | 0.951697 |
| Clean NLL | 0.208438 | 0.162055 | 0.147401 | 0.172631 | 0.031863 | 0.147401 | 0.208438 |
| Clean ECE | 0.016133 | 0.013889 | 0.009040 | 0.013021 | 0.003625 | 0.009040 | 0.016133 |
| Clean Brier | 0.106254 | 0.082852 | 0.074431 | 0.087846 | 0.016489 | 0.074431 | 0.106254 |
| N=50 mean-prob accuracy | 0.366653 | 0.277689 | 0.266393 | 0.303579 | 0.054916 | 0.266393 | 0.366653 |
| N=50 mean-prob delta accuracy | -0.563575 | -0.667533 | -0.685326 | -0.638811 | 0.065761 | -0.685326 | -0.563575 |
| N=50 mean-prob macro-F1 | 0.315438 | 0.198880 | 0.204280 | 0.239533 | 0.065792 | 0.198880 | 0.315438 |
| N=50 mean-prob NLL | 2.073582 | 2.589658 | 2.567515 | 2.410252 | 0.291775 | 2.073582 | 2.589658 |
| N=50 mean-prob ECE | 0.112321 | 0.175657 | 0.221926 | 0.169968 | 0.055023 | 0.112321 | 0.221926 |
| N=50 mean-prob Brier | 0.756608 | 0.841425 | 0.832409 | 0.810148 | 0.046585 | 0.756608 | 0.841425 |
| N=50 mean-prob harm rate | 0.614227 | 0.711083 | 0.722508 | 0.682606 | 0.059493 | 0.614227 | 0.722508 |
| N=50 mean-prob rescue rate | 0.111748 | 0.083942 | 0.047619 | 0.081103 | 0.032159 | 0.047619 | 0.111748 |
| N=50 majority-vote accuracy | 0.345562 | 0.273790 | 0.262995 | 0.294116 | 0.044879 | 0.262995 | 0.345562 |
| N=50 confidence-weighted accuracy | 0.365654 | 0.270492 | 0.258796 | 0.298314 | 0.058610 | 0.258796 | 0.365654 |
| N=50 original-anchored accuracy | 0.382047 | 0.291583 | 0.278089 | 0.317240 | 0.056529 | 0.278089 | 0.382047 |
| N=50 BN-adapted (all) | not applicable (GroupNorm) | | | | | | |

### pathmnist 64px batchnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.910036 | 0.910136 | 0.936925 | 0.919032 | 0.015496 | 0.910036 | 0.936925 |
| Clean macro-F1 | 0.910905 | 0.906103 | 0.937995 | 0.918334 | 0.017195 | 0.906103 | 0.937995 |
| Clean NLL | 0.262940 | 0.255148 | 0.185776 | 0.234621 | 0.042481 | 0.185776 | 0.262940 |
| Clean ECE | 0.008190 | 0.023306 | 0.012805 | 0.014767 | 0.007746 | 0.008190 | 0.023306 |
| Clean Brier | 0.134738 | 0.127749 | 0.097922 | 0.120136 | 0.019553 | 0.097922 | 0.134738 |
| N=50 mean-prob accuracy | 0.477409 | 0.493303 | 0.581168 | 0.517293 | 0.055885 | 0.477409 | 0.581168 |
| N=50 mean-prob delta accuracy | -0.432627 | -0.416833 | -0.355758 | -0.401739 | 0.040597 | -0.432627 | -0.355758 |
| N=50 mean-prob macro-F1 | 0.417084 | 0.447248 | 0.533421 | 0.465918 | 0.060374 | 0.417084 | 0.533421 |
| N=50 mean-prob NLL | 1.378968 | 1.417769 | 1.312642 | 1.369793 | 0.053161 | 1.312642 | 1.417769 |
| N=50 mean-prob ECE | 0.063472 | 0.089681 | 0.097660 | 0.083605 | 0.017886 | 0.063472 | 0.097660 |
| N=50 mean-prob Brier | 0.627039 | 0.641033 | 0.577504 | 0.615192 | 0.033380 | 0.577504 | 0.641033 |
| N=50 mean-prob harm rate | 0.481986 | 0.470621 | 0.385256 | 0.445954 | 0.052873 | 0.385256 | 0.481986 |
| N=50 mean-prob rescue rate | 0.066667 | 0.127920 | 0.082409 | 0.092332 | 0.031809 | 0.066667 | 0.127920 |
| N=50 majority-vote accuracy | 0.476809 | 0.472911 | 0.569072 | 0.506264 | 0.054428 | 0.472911 | 0.569072 |
| N=50 confidence-weighted accuracy | 0.470512 | 0.492403 | 0.582867 | 0.515261 | 0.059563 | 0.470512 | 0.582867 |
| N=50 original-anchored accuracy | 0.499100 | 0.511096 | 0.599860 | 0.536685 | 0.055039 | 0.499100 | 0.599860 |
| N=50 BN-adapted accuracy | 0.753898 | 0.820772 | 0.744402 | 0.773024 | 0.041622 | 0.744402 | 0.820772 |
| N=50 BN-adapted delta accuracy | -0.156138 | -0.089364 | -0.192523 | -0.146008 | 0.052320 | -0.192523 | -0.089364 |
| N=50 BN-adapted NLL | 0.699302 | 0.602020 | 0.817154 | 0.706159 | 0.107731 | 0.602020 | 0.817154 |

### pathmnist 64px groupnorm (seeds [0, 1, 2])

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample StdDev | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.979608 | 0.981507 | 0.983107 | 0.981407 | 0.001751 | 0.979608 | 0.983107 |
| Clean macro-F1 | 0.979496 | 0.981141 | 0.982952 | 0.981196 | 0.001729 | 0.979496 | 0.982952 |
| Clean NLL | 0.072099 | 0.068478 | 0.062615 | 0.067731 | 0.004786 | 0.062615 | 0.072099 |
| Clean ECE | 0.012695 | 0.014552 | 0.011519 | 0.012922 | 0.001529 | 0.011519 | 0.014552 |
| Clean Brier | 0.034148 | 0.031718 | 0.028663 | 0.031510 | 0.002748 | 0.028663 | 0.034148 |
| N=50 mean-prob accuracy | 0.465614 | 0.331467 | 0.417033 | 0.404705 | 0.067918 | 0.331467 | 0.465614 |
| N=50 mean-prob delta accuracy | -0.513994 | -0.650040 | -0.566074 | -0.576703 | 0.068643 | -0.650040 | -0.513994 |
| N=50 mean-prob macro-F1 | 0.408753 | 0.271202 | 0.349830 | 0.343262 | 0.069011 | 0.271202 | 0.408753 |
| N=50 mean-prob NLL | 1.483892 | 1.865346 | 1.599676 | 1.649638 | 0.195573 | 1.483892 | 1.865346 |
| N=50 mean-prob ECE | 0.128499 | 0.165867 | 0.096863 | 0.130409 | 0.034542 | 0.096863 | 0.165867 |
| N=50 mean-prob Brier | 0.652161 | 0.780999 | 0.691385 | 0.708182 | 0.066041 | 0.652161 | 0.780999 |
| N=50 mean-prob harm rate | 0.528469 | 0.664019 | 0.577224 | 0.589904 | 0.068659 | 0.528469 | 0.664019 |
| N=50 mean-prob rescue rate | 0.181373 | 0.091892 | 0.082840 | 0.118702 | 0.054463 | 0.082840 | 0.181373 |
| N=50 majority-vote accuracy | 0.447721 | 0.325170 | 0.388844 | 0.387245 | 0.061291 | 0.325170 | 0.447721 |
| N=50 confidence-weighted accuracy | 0.468213 | 0.324870 | 0.422231 | 0.405105 | 0.073190 | 0.324870 | 0.468213 |
| N=50 original-anchored accuracy | 0.489004 | 0.344462 | 0.438824 | 0.424097 | 0.073388 | 0.344462 | 0.489004 |
| N=50 BN-adapted (all) | not applicable (GroupNorm) | | | | | | |

## 5. Registered prefix curves (mean-probability), N = 1, 2, 5, 10, 25, 50, 100

Primary aggregation condition (mean-probability), accuracy and NLL
across every registered prefix, per seed, for all 8 dataset×resolution×
normalization conditions.

### bloodmnist 28px batchnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.437500 | 0.419393 | 0.403621 | 0.420171 |
| 2 | 0.451519 | 0.464953 | 0.436332 | 0.450935 |
| 5 | 0.498832 | 0.509346 | 0.453271 | 0.487150 |
| 10 | 0.539136 | 0.543808 | 0.467290 | 0.516745 |
| 25 | 0.573598 | 0.558995 | 0.458528 | 0.530374 |
| 50 | 0.584112 | 0.571846 | 0.456776 | 0.537578 |
| 100 | 0.594626 | 0.574182 | 0.452687 | 0.540498 |

### bloodmnist 28px batchnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 3.831318 | 3.276831 | 5.037395 | 4.048514 |
| 2 | 2.209141 | 2.125943 | 3.057908 | 2.464331 |
| 5 | 1.302308 | 1.450883 | 1.908499 | 1.553897 |
| 10 | 1.117786 | 1.267495 | 1.564060 | 1.316447 |
| 25 | 1.044426 | 1.187930 | 1.405997 | 1.212784 |
| 50 | 1.033449 | 1.170985 | 1.365915 | 1.190117 |
| 100 | 1.025607 | 1.158774 | 1.352153 | 1.178845 |

### bloodmnist 28px groupnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.579439 | 0.547897 | 0.557827 | 0.561721 |
| 2 | 0.642523 | 0.609229 | 0.608645 | 0.620132 |
| 5 | 0.681075 | 0.656542 | 0.650701 | 0.662773 |
| 10 | 0.713201 | 0.676402 | 0.666472 | 0.685358 |
| 25 | 0.738902 | 0.698598 | 0.678154 | 0.705218 |
| 50 | 0.741238 | 0.712033 | 0.675234 | 0.709502 |
| 100 | 0.738902 | 0.715537 | 0.679322 | 0.711254 |

### bloodmnist 28px groupnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 1.519197 | 1.703247 | 1.574182 | 1.598875 |
| 2 | 1.112294 | 1.322124 | 1.230888 | 1.221769 |
| 5 | 0.923354 | 1.088777 | 1.041555 | 1.017895 |
| 10 | 0.874568 | 1.029268 | 0.989162 | 0.964333 |
| 25 | 0.850543 | 0.992143 | 0.962649 | 0.935112 |
| 50 | 0.845843 | 0.986111 | 0.957830 | 0.929928 |
| 100 | 0.840443 | 0.981787 | 0.955094 | 0.925775 |

### bloodmnist 64px batchnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.334112 | 0.288551 | 0.361565 | 0.328076 |
| 2 | 0.348715 | 0.269276 | 0.360981 | 0.326324 |
| 5 | 0.376752 | 0.302570 | 0.403037 | 0.360787 |
| 10 | 0.376168 | 0.289136 | 0.403621 | 0.356308 |
| 25 | 0.391355 | 0.264019 | 0.412383 | 0.355919 |
| 50 | 0.375000 | 0.251168 | 0.420561 | 0.348910 |
| 100 | 0.365654 | 0.241238 | 0.414136 | 0.340343 |

### bloodmnist 64px batchnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 4.789126 | 5.931649 | 4.288114 | 5.002963 |
| 2 | 2.827369 | 3.481666 | 2.573938 | 2.960991 |
| 5 | 1.712909 | 2.027272 | 1.646512 | 1.795564 |
| 10 | 1.497684 | 1.724338 | 1.475695 | 1.565906 |
| 25 | 1.415792 | 1.598432 | 1.407077 | 1.473767 |
| 50 | 1.390797 | 1.558961 | 1.385558 | 1.445105 |
| 100 | 1.381583 | 1.547389 | 1.377839 | 1.435604 |

### bloodmnist 64px groupnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.442173 | 0.487150 | 0.432827 | 0.454050 |
| 2 | 0.485397 | 0.534463 | 0.447430 | 0.489097 |
| 5 | 0.489486 | 0.536799 | 0.428154 | 0.484813 |
| 10 | 0.492991 | 0.545561 | 0.417640 | 0.485397 |
| 25 | 0.510514 | 0.551402 | 0.405374 | 0.489097 |
| 50 | 0.502921 | 0.553738 | 0.387850 | 0.481503 |
| 100 | 0.501752 | 0.561332 | 0.392523 | 0.485202 |

### bloodmnist 64px groupnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 2.691884 | 2.135902 | 2.439811 | 2.422533 |
| 2 | 2.051359 | 1.581949 | 1.852043 | 1.828450 |
| 5 | 1.581453 | 1.216124 | 1.438517 | 1.412032 |
| 10 | 1.417414 | 1.133812 | 1.330138 | 1.293788 |
| 25 | 1.320229 | 1.080651 | 1.256969 | 1.219283 |
| 50 | 1.295840 | 1.068422 | 1.240469 | 1.201577 |
| 100 | 1.280735 | 1.060251 | 1.229677 | 1.190221 |

### pathmnist 28px batchnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.306977 | 0.253798 | 0.261495 | 0.274090 |
| 2 | 0.350360 | 0.253199 | 0.266393 | 0.289984 |
| 5 | 0.380448 | 0.241603 | 0.271092 | 0.297714 |
| 10 | 0.409136 | 0.229608 | 0.271991 | 0.303579 |
| 25 | 0.441224 | 0.224210 | 0.264294 | 0.309909 |
| 50 | 0.458417 | 0.222011 | 0.261196 | 0.313874 |
| 100 | 0.469412 | 0.223011 | 0.254198 | 0.315540 |

### pathmnist 28px batchnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 5.078989 | 5.113330 | 6.771387 | 5.654568 |
| 2 | 3.272747 | 3.575924 | 4.435321 | 3.761331 |
| 5 | 2.165464 | 2.420792 | 2.710473 | 2.432243 |
| 10 | 1.885124 | 2.117874 | 2.211609 | 2.071536 |
| 25 | 1.761105 | 1.977247 | 1.975672 | 1.904675 |
| 50 | 1.725584 | 1.939851 | 1.911149 | 1.858861 |
| 100 | 1.706975 | 1.919338 | 1.883315 | 1.836543 |

### pathmnist 28px groupnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.283287 | 0.240804 | 0.266094 | 0.263395 |
| 2 | 0.319572 | 0.256997 | 0.276789 | 0.284453 |
| 5 | 0.339664 | 0.262595 | 0.278888 | 0.293716 |
| 10 | 0.350460 | 0.274390 | 0.275190 | 0.300013 |
| 25 | 0.361655 | 0.279488 | 0.273990 | 0.305045 |
| 50 | 0.366653 | 0.277689 | 0.266393 | 0.303579 |
| 100 | 0.364954 | 0.275290 | 0.261096 | 0.300446 |

### pathmnist 28px groupnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 4.588325 | 5.360421 | 5.582506 | 5.177084 |
| 2 | 3.365742 | 4.095977 | 4.325189 | 3.928969 |
| 5 | 2.511156 | 3.186527 | 3.313620 | 3.003768 |
| 10 | 2.271489 | 2.886338 | 2.951439 | 2.703089 |
| 25 | 2.128986 | 2.681352 | 2.688668 | 2.499669 |
| 50 | 2.073582 | 2.589658 | 2.567515 | 2.410252 |
| 100 | 2.045332 | 2.534003 | 2.487843 | 2.355726 |

### pathmnist 64px batchnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.374550 | 0.349360 | 0.396242 | 0.373384 |
| 2 | 0.408637 | 0.392243 | 0.450620 | 0.417166 |
| 5 | 0.453419 | 0.433926 | 0.507797 | 0.465047 |
| 10 | 0.467913 | 0.462515 | 0.535786 | 0.488738 |
| 25 | 0.482707 | 0.478709 | 0.565574 | 0.508996 |
| 50 | 0.477409 | 0.493303 | 0.581168 | 0.517293 |
| 100 | 0.478209 | 0.499200 | 0.587665 | 0.521691 |

### pathmnist 64px batchnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 3.970643 | 4.770932 | 3.919297 | 4.220291 |
| 2 | 2.544806 | 3.018441 | 2.567345 | 2.710197 |
| 5 | 1.708659 | 1.882443 | 1.712735 | 1.767946 |
| 10 | 1.492226 | 1.576165 | 1.454587 | 1.507659 |
| 25 | 1.403727 | 1.451099 | 1.345044 | 1.399957 |
| 50 | 1.378968 | 1.417769 | 1.312642 | 1.369793 |
| 100 | 1.368524 | 1.404442 | 1.298531 | 1.357166 |

### pathmnist 64px groupnorm -- mean-probability accuracy across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 0.341563 | 0.297381 | 0.316773 | 0.318573 |
| 2 | 0.387445 | 0.310976 | 0.363155 | 0.353858 |
| 5 | 0.417833 | 0.327769 | 0.382647 | 0.376083 |
| 10 | 0.437225 | 0.329068 | 0.398641 | 0.388311 |
| 25 | 0.457917 | 0.333567 | 0.416433 | 0.402639 |
| 50 | 0.465614 | 0.331467 | 0.417033 | 0.404705 |
| 100 | 0.470512 | 0.329668 | 0.416933 | 0.405704 |

### pathmnist 64px groupnorm -- mean-probability NLL across N

| N | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 | 4.226413 | 4.870882 | 4.303709 | 4.467001 |
| 2 | 2.876852 | 3.503688 | 3.087095 | 3.155878 |
| 5 | 1.971158 | 2.503967 | 2.152336 | 2.209154 |
| 10 | 1.691257 | 2.160354 | 1.825147 | 1.892253 |
| 25 | 1.527682 | 1.936187 | 1.648836 | 1.704235 |
| 50 | 1.483892 | 1.865346 | 1.599676 | 1.649638 |
| 100 | 1.459699 | 1.826423 | 1.574164 | 1.620095 |

## 6. Exact runtime accounting

Per-cell wall-clock runtime (`status.json`'s `ended_at - started_at`),
category totals, and category mean/median.

| dataset | resolution | normalization | seed | runtime (s) | runtime (h:m:s) |
|---|---|---|---|---|---|
| bloodmnist | 28 | batchnorm | 0 | 1156.792 | 0h19m16.8s |
| bloodmnist | 28 | batchnorm | 1 | 1168.915 | 0h19m28.9s |
| bloodmnist | 28 | batchnorm | 2 | 1150.687 | 0h19m10.7s |
| bloodmnist | 28 | groupnorm | 0 | 705.866 | 0h11m45.9s |
| bloodmnist | 28 | groupnorm | 1 | 687.727 | 0h11m27.7s |
| bloodmnist | 28 | groupnorm | 2 | 687.285 | 0h11m27.3s |
| bloodmnist | 64 | batchnorm | 0 | 1265.539 | 0h21m05.5s |
| bloodmnist | 64 | batchnorm | 1 | 1262.471 | 0h21m02.5s |
| bloodmnist | 64 | batchnorm | 2 | 1257.238 | 0h20m57.2s |
| bloodmnist | 64 | groupnorm | 0 | 749.270 | 0h12m29.3s |
| bloodmnist | 64 | groupnorm | 1 | 739.861 | 0h12m19.9s |
| bloodmnist | 64 | groupnorm | 2 | 743.326 | 0h12m23.3s |
| pathmnist | 28 | batchnorm | 0 | 6785.220 | 1h53m05.2s |
| pathmnist | 28 | batchnorm | 1 | 6914.231 | 1h55m14.2s |
| pathmnist | 28 | batchnorm | 2 | 6963.303 | 1h56m03.3s |
| pathmnist | 28 | groupnorm | 0 | 4102.957 | 1h08m23.0s |
| pathmnist | 28 | groupnorm | 1 | 3904.791 | 1h05m04.8s |
| pathmnist | 28 | groupnorm | 2 | 3959.688 | 1h05m59.7s |
| pathmnist | 64 | batchnorm | 0 | 7345.785 | 2h02m25.8s |
| pathmnist | 64 | batchnorm | 1 | 7367.657 | 2h02m47.7s |
| pathmnist | 64 | batchnorm | 2 | 7402.262 | 2h03m22.3s |
| pathmnist | 64 | groupnorm | 0 | 4393.676 | 1h13m13.7s |
| pathmnist | 64 | groupnorm | 1 | 4392.979 | 1h13m13.0s |
| pathmnist | 64 | groupnorm | 2 | 4381.788 | 1h13m01.8s |

| Category | Total (s) | Total (h) | Mean (s) | Median (s) |
|---|---|---|---|---|
| bloodmnist 28px batchnorm | 3476.394 | 0.9657 | 1158.798 | 1156.792 |
| bloodmnist 28px groupnorm | 2080.879 | 0.5780 | 693.626 | 687.727 |
| bloodmnist 64px batchnorm | 3785.248 | 1.0515 | 1261.749 | 1262.471 |
| bloodmnist 64px groupnorm | 2232.458 | 0.6201 | 744.153 | 743.326 |
| pathmnist 28px batchnorm | 20662.754 | 5.7397 | 6887.585 | 6914.231 |
| pathmnist 28px groupnorm | 11967.436 | 3.3243 | 3989.145 | 3959.688 |
| pathmnist 64px batchnorm | 22115.704 | 6.1433 | 7371.901 | 7367.657 |
| pathmnist 64px groupnorm | 13168.443 | 3.6579 | 4389.481 | 4392.979 |
| **All 24 cells (sum)** | **79489.315** | **22.0804** | | |
### Complete Block A canonical evaluation runtime

**Total experimental runtime (sum of the 24 canonical cells' own
wall-clock runtimes): 79,489.315s = 22.0804h = 0.9200 days.** This is
the true experimental compute time -- pure evaluation wall-clock,
summed across cells, independent of when each cell happened to run in
calendar time.

### Separated: non-canonical attempts (not counted in the total above)

| Attempt | Runtime | Included in "22.08h total" above? |
|---|---|---|
| `A-pathmnist-28px-batchnorm-*-s0` attempt 1 (aborted) | not measured (`ended_at` empty -- manually terminated) | No |
| `A-pathmnist-28px-batchnorm-*-s0` attempt 2 (failed, OOM) | 1,597.689s (~26.6min) | No |
| `A-pathmnist-28px-batchnorm-*-s0` attempt 3 (completed, excluded) | 15,331.649s (~4.26h) | No -- real compute, but not part of the canonical 24-cell total since this attempt is not canonical |
| `A-pathmnist-28px-groupnorm-*-s0` attempt 1 (failed, schema) | 4,165.632s (~1.16h) | No |

**Grand total actual compute time across every attempt ever run for
Block A (canonical + non-canonical):** 79,489.315 + 1,597.689 +
15,331.649 + 4,165.632 = **100,584.285s = 27.940h = 1.164 days.**

### Idle time / interruptions (explicitly separated, not counted as experimental runtime)

The 24 canonical cells did not run back-to-back in a single unbroken
session -- their real calendar timestamps span multiple sessions across
this research phase (the original canary, the GroupNorm-fix
investigation, and the final 20-cell serial batch), separated by
periods where no evaluation process was running (design/documentation/
verification work, waiting for user authorization between phases, and
the interval consumed by diagnosing and fixing the GroupNorm schema
defect itself). This idle/investigation time is real calendar time but
is **not** experimental compute and is excluded from both totals above.
As a concrete reference point: `A-pathmnist-28px-batchnorm-*-s0`
(attempt 4) and its two BatchNorm siblings (`-s1`, `-s2`) ran in an
earlier session; `A-pathmnist-28px-groupnorm-*-s0`'s canonical attempt
2 ran as a standalone canary in a separate session; the remaining 20
cells (both remaining PathMNIST-28px GroupNorm seeds, all of
PathMNIST-64px, and all of BloodMNIST) ran consecutively in one final
unattended batch, 00:32:49 to 15:45:52 EDT on 2026-08-20 (~15h13m
wall-clock, matching the sum of those 20 cells' own runtimes plus
essentially zero idle gap between them -- sequential `set -e`
execution, no parallelism, no retries).

## 7. Integrity evidence

All of the following were independently verified, live, during
execution (not merely asserted after the fact) for every one of the 24
canonical cells:

- **Manifest verification**: `verify_evaluation_artifact_manifest()` against every persisted `artifact_manifest.json` -- OK for all 24.
- **Independent semantic recomputation**: clean accuracy/NLL and N=50 mean-probability accuracy/NLL recomputed directly from persisted `predictions.npz` via `compute_metrics_from_probabilities()` (never calls softmax) and compared to `metrics.json` -- exact match within `1e-6` for all 24, zero mismatches found.
- **Probability finiteness/range/normalization**: `clean_probs`, `view_probs`, and (BatchNorm only) `bn_adapted_probs` confirmed finite, within `[0,1]`, row-summing to 1 within tolerance -- for all 24 cells.
- **Dataset checksums**: `dataset_verification.checksum_verified == True` and `resized == False` for all 24 cells (`pathmnist`/`pathmnist_64`/`bloodmnist`/`bloodmnist_64` official MD5s all matched expected).
- **Checkpoint binding**: `metadata.checkpoint_hash` matched `resolve_canonical_training_completion()`'s canonical training result for all 24 cells.
- **BatchNorm/GroupNorm applicability consistency**: all 12 BatchNorm cells show `bn_adaptation_applicable=True`, a positive microbatch count, full `bn_adapted_probs`/`bn_adapted_prefix_sequence` evidence, and non-null `bn_adapted_tta` metrics; all 12 GroupNorm cells show `bn_adaptation_applicable=False`, `bn_adaptation_microbatches_at_primary_n=0`, **no** `bn_adapted_probs`/`bn_adapted_prefix_sequence` arrays, and **no** `bn_adapted_tta` entry -- zero fabricated BN-adapted results anywhere.
- **`test_metrics_observed=False`**: true for all 28 ledger rows (all attempts, canonical and non-canonical alike) -- no test-loading code path exists anywhere in this pipeline.
- **Phase 2A ledger MD5**: unchanged throughout, `e2dbdcd757cb13d77201c24cd746c05a`.
- **No Block B/C/D access**: zero `B-`/`C-`/`D-` prefixed rows or directories exist anywhere in the evaluation ledger or `artifacts/validation_evaluation/`.

## 8. Fingerprint-cohort disclosure

Three distinct, precisely-scoped claims:

- **Scientifically valid**: all 24 canonical Block A cells. Every
  cell's persisted probabilities, metrics, and artifacts independently
  reverified and internally consistent under the frozen
  `probability_native_v1` metric contract.
- **Canonical-eligible under its own persisted identity**: all 24
  cells. Each cell's own `evaluation_id`/`evaluator_fingerprint` is
  self-consistent, `check_evaluation_skip()` resolves each to its own
  attempt cleanly as the sole compatible completion when queried with
  that cell's own persisted identity.
- **Compatible with the current evaluator fingerprint**
  (`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`):
  **only 21 of 24**. `A-pathmnist-28px-batchnorm-policy-none-s0`
  (attempt 4), `-s1` (attempt 1), and `-s2` (attempt 1) were completed
  under the prior, metric-contract-only fingerprint
  (`f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7`)
  before the GroupNorm persistence-schema correction changed the
  fingerprint a second time.

**Block A is 24/24 scientifically evaluated but only 21/24
current-fingerprint-compatible.** The 3 stale-fingerprint cells do
**not** cleanly skip under the current implementation -- invoking
`check_evaluation_skip()` for any of them with a freshly-computed,
current-fingerprint evaluation ID raises
`ConflictingEvaluationImplementationError` (verified directly, see
`docs/phase2b_validation_evaluation_evaluator_fingerprint_drift_addendum.md`),
not a silent pass-through. Reconciliation (most likely a single rerun
of these three cells under the final frozen evaluator fingerprint, once
it is confirmed final across the remaining Blocks B/C/D) remains
**explicitly deferred** and is not resolved by this document.
