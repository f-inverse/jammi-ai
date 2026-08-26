# ci/doc_number_allowlist.txt classification

Every entry seeded into `ci/doc_number_allowlist.txt` from `origin/main`, hand-classified real (a genuine unproduced-measurement claim -- still needs a real producer citation or a `no-producer:` tag on its own branch, not fixed here) or noise (not actually a claimed measurement -- a defined/derived constant, a hand-computed worked example, a design threshold, a statistical convention, or similar false-positive shape this gate's lexical heuristics cannot fully rule out).

**Noise rate: 35/80 = 43.8%** (real: 45/80 = 56.2%).

Computed directly from the table below -- count the `noise` rows over the total row count, not asserted separately.

| # | file:line | number | kind | real/noise | note |
|---|---|---|---|---|---|
| 1 | `crates/jammi-kernels/src/admission.rs:976` | `8/200` | N/M | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 2 | `crates/jammi-kernels/src/cuda/adamw_step.cu:21` | `5145/16384` | N/M | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 3 | `crates/jammi-kernels/src/cuda/adamw_step.cu:131` | `5145/16384` | N/M | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 4 | `crates/jammi-kernels/src/ops/adamw_step.rs:97` | `5145/16384` | N/M | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 5 | `crates/jammi-kernels/src/ops/adamw_step.rs:123` | `2.44%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 6 | `crates/jammi-kernels/src/ops/adamw_step.rs:124` | `7.25%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 7 | `crates/jammi-kernels/src/ops/adamw_step.rs:125` | `2.45%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 8 | `crates/jammi-kernels/src/ops/adamw_step.rs:125` | `11.82%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 9 | `crates/jammi-kernels/src/ops/attention_block.rs:50` | `0.125` | cosine | noise | derived scale constant `1/sqrt(head_dim)`, not measured |
| 10 | `crates/jammi-kernels/src/ops/attention_block.rs:2242` | `0.08` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 11 | `crates/jammi-kernels/src/ops/attention_block.rs:2245` | `0.166` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 12 | `crates/jammi-kernels/src/ops/dropout.rs:453` | `0.05` | cosine | noise | explicitly-derived Binomial std-dev bound |
| 13 | `crates/jammi-kernels/src/ops/dropout.rs:454` | `0.05` | cosine | noise | explicitly-derived Binomial std-dev bound |
| 14 | `crates/jammi-kernels/src/ops/dropout.rs:455` | `0.0013` | cosine | noise | explicitly-derived Binomial std-dev bound ("a generous, explicitly derived, non-arbitrary bound") |
| 15 | `crates/jammi-kernels/src/ops/geglu.rs:173` | `0.398942` | cosine | noise | design choice: truncated literal vs full-precision constant |
| 16 | `crates/jammi-kernels/src/ops/geglu.rs:189` | `0.398942` | cosine | noise | same design choice, second mention |
| 17 | `crates/jammi-kernels/src/ops/softmax.rs:2775` | `0.01` | cosine | noise | adversarial-fixture arithmetic setup (`2000 + 0.01`), not an observed result |
| 18 | `crates/jammi-kernels/src/ops/softmax.rs:2778` | `0.01` | cosine | noise | same fixture arithmetic, second mention |
| 19 | `crates/jammi-kernels/tests/cuda_parity.rs:3214` | `5%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 20 | `crates/jammi-kernels/tests/cuda_parity.rs:3297` | `0.008` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 21 | `crates/jammi-kernels/tests/cuda_parity.rs:3537` | `0.15` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 22 | `crates/jammi-kernels/tests/cuda_parity.rs:3538` | `0.15` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 23 | `crates/jammi-kernels/tests/cuda_parity.rs:4462` | `5.5%` | % | noise | derived formula `7 * 2^-7 = 5.5%` |
| 24 | `crates/jammi-kernels/tests/cuda_parity.rs:4464` | `5%` | % | noise | design bound description ("a ~1-5% bound") |
| 25 | `crates/jammi-kernels/tests/cuda_parity.rs:4794` | `0.031` | cosine | noise | derived floor `4 * 2^-7`, explicitly "no derivation behind it" (a REJECTED prior floor) |
| 26 | `crates/jammi-kernels/tests/cuda_parity.rs:4808` | `0.06` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 27 | `crates/jammi-kernels/tests/cuda_parity.rs:4808` | `0.94` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 28 | `crates/jammi-kernels/tests/cuda_parity.rs:5944` | `0.273` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 29 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.619` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 30 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.641` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 31 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.321` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 32 | `crates/jammi-kernels/tests/cuda_parity.rs:6049` | `5.2%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 33 | `crates/jammi-kernels/tests/cuda_parity.rs:6049` | `53%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 34 | `crates/jammi-kernels/tests/cuda_parity.rs:6260` | `5145/16384` | N/M | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 35 | `crates/jammi-kernels/tests/flash_smoke.rs:302` | `0.39` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 36 | `crates/jammi-kernels/tests/flash_smoke.rs:578` | `0.06` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 37 | `crates/jammi-kernels/tests/flash_smoke.rs:578` | `0.60` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 38 | `crates/jammi-kernels/tests/geglu_oracles.rs:162` | `0.398942` | cosine | noise | design choice: truncated literal vs full-precision constant |
| 39 | `crates/jammi-kernels/tests/geglu_oracles.rs:184` | `2%` | % | noise | derived consequence of an existing bound, not itself measured |
| 40 | `crates/jammi-kernels/tests/geglu_oracles.rs:195` | `0.125` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 41 | `crates/jammi-kernels/tests/geglu_oracles.rs:196` | `1.02%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 42 | `crates/jammi-kernels/tests/geglu_oracles.rs:196` | `0.125` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 43 | `crates/jammi-kernels/tests/geglu_oracles.rs:201` | `1.5625%` | % | noise | pure formula `2 * 2^-7 = 2^-6 = 1.5625%` |
| 44 | `crates/jammi-kernels/tests/geglu_oracles.rs:212` | `0.0120` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 45 | `crates/jammi-kernels/tests/geglu_oracles.rs:213` | `0.03125` | cosine | noise | defined constant `BF16_ABS_FLOOR = 2^-5 = 0.03125` |
| 46 | `crates/jammi-kernels/tests/geglu_oracles.rs:218` | `1.06%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 47 | `crates/jammi-kernels/tests/geglu_oracles.rs:219` | `1.5625%` | % | noise | repeated mention of the same derived REL bound |
| 48 | `crates/jammi-kernels/tests/geglu_oracles.rs:242` | `0.125` | cosine | noise | generic explanatory sentence about what a bare 0.125 would mean, not itself an assertion |
| 49 | `crates/jammi-kernels/tests/geglu_oracles.rs:421` | `0.125` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 50 | `crates/jammi-kernels/tests/geglu_oracles.rs:424` | `0.125` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 51 | `crates/jammi-kernels/tests/oracles.rs:620` | `0.125` | cosine | noise | design constant `scale = 0.125`, an exact power of two |
| 52 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:185` | `0.0625` | cosine | noise | derived ULP-at-magnitude calculation |
| 53 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:193` | `0.015625` | cosine | noise | derived ULP-at-magnitude calculation |
| 54 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:412` | `0.333` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 55 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:414` | `0.4%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 56 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:414` | `0.333` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 57 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `21%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 58 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `0.174` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 59 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `0.82` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 60 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:421` | `0.25` | cosine | noise | defined constant `FLOOR = 2^-2 = 0.25` |
| 61 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:422` | `0.174` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 62 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:423` | `0.333` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 63 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:426` | `0.52%` | % | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 64 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:426` | `1.5625%` | % | noise | defined constant `REL = 2^-6 = 1.5625%` |
| 65 | `crates/jammi-encoders/src/htsat_audio.rs:128` | `0.75` | cosine | noise | well-known Keys cubic-convolution kernel parameter `a = -0.75` |
| 66 | `crates/jammi-bench/src/eval.rs:247` | `75%` | % | noise | self-referential synthetic-data-generator design constant |
| 67 | `crates/jammi-bench/src/finetune_step.rs:1055` | `0.001` | cosine | noise | known library default (`Default::default()`'s `lr = 0.001`) |
| 68 | `crates/jammi-bench/src/finetune_step.rs:1078` | `0.001` | cosine | noise | same known library default, second mention |
| 69 | `crates/jammi-bench/src/finetune_step.rs:1243` | `0.65` | cosine | noise | hand-computed worked example (fn name literally says `hand_computed_value`) |
| 70 | `crates/jammi-bench/src/fixture.rs:46` | `0.04` | cosine | noise | constant matched to another already-established margin |
| 71 | `crates/jammi-bench/src/fixture.rs:284` | `100%` | % | noise | hedged disclaimer quoting a hypothetical claim, not asserting it |
| 72 | `crates/jammi-bench/src/rate_gate.rs:20` | `30%` | % | noise | named design threshold `DEFAULT_REGRESSION_THRESHOLD` |
| 73 | `crates/jammi-bench/src/rate_gate.rs:37` | `30%` | % | noise | same named design threshold, second mention |
| 74 | `crates/jammi-bench/src/recall.rs:102` | `95%` | % | noise | fixed statistical convention (a chosen significance level) |
| 75 | `crates/jammi-bench/src/recall.rs:675` | `0.95` | cosine | noise | defined gate floor/threshold, not an observed value |
| 76 | `crates/jammi-bench/src/recall.rs:882` | `0.27` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 77 | `crates/jammi-bench/src/recall.rs:882` | `0.17` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 78 | `crates/jammi-bench/src/recall.rs:882` | `0.10` | cosine | real | explicit "measured"/"observed" claim with no producer cited nearby |
| 79 | `crates/jammi-bench/src/report.rs:1185` | `0.30` | cosine | noise | illustrative example value for an ULP-bucket explanation |
| 80 | `crates/jammi-bench/src/report.rs:1185` | `0.25` | cosine | noise | illustrative example value, same line |
