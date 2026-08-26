# ci/doc_number_allowlist.txt classification

Every entry seeded into `ci/doc_number_allowlist.txt` from `origin/main`, hand-classified real (a genuine unproduced-measurement claim -- still needs a real producer citation or a `no-producer:` tag on its own branch, not fixed here) or noise, with a distinct, per-row reason for EVERY row in both classes (round-4 audit fix: a prior revision of this table gave every `real` row the same templated note and counted a repeated restatement of an already-measured number -- e.g. the same `5145/16384` finding retold in four files, or a derived relative-percentage computed from a number two lines up -- as a fresh, independent claim; restatements are now classified noise, with a note naming which row carries the original).

**Noise rate: 36/69 = 52.2%** (real: 33/69 = 47.8%).

Computed directly from the table below -- count the `noise` rows over the total row count, not asserted separately.

| # | file:line | number | kind | real/noise | note |
|---|---|---|---|---|---|
| 1 | `crates/jammi-kernels/src/admission.rs:976` | `8/200` | N/M | real | flaky-test count explicitly framed as 'observed failing', no citation nearby |
| 2 | `crates/jammi-kernels/src/cuda/adamw_step.cu:21` | `5145/16384` | N/M | real | primary report of the AdamW FMA-contraction finding: 'measured on jammi-a100: 5145/16384 differed', no citation |
| 3 | `crates/jammi-kernels/src/cuda/adamw_step.cu:131` | `5145/16384` | N/M | noise | restatement of adamw_step.cu:21's same 5145/16384 finding, second mention in the same file |
| 4 | `crates/jammi-kernels/src/ops/adamw_step.rs:97` | `5145/16384` | N/M | noise | restatement of adamw_step.cu:21's same finding, mirrored in the Rust wrapper's doc |
| 5 | `crates/jammi-kernels/src/ops/adamw_step.rs:123` | `2.44%` | % | real | roofline measurement table entry (shape [16,1024]): 'Measured on jammi-a100 ... same-run', no citation |
| 6 | `crates/jammi-kernels/src/ops/adamw_step.rs:124` | `7.25%` | % | real | roofline measurement table entry (shape [3072,16]), same uncited table |
| 7 | `crates/jammi-kernels/src/ops/adamw_step.rs:125` | `2.45%` | % | real | roofline measurement table entry (shape [1024,16]), same uncited table |
| 8 | `crates/jammi-kernels/src/ops/adamw_step.rs:125` | `11.82%` | % | real | roofline measurement table entry (shape [5248,16]), same uncited table |
| 9 | `crates/jammi-kernels/src/ops/attention_block.rs:2242` | `0.08` | cosine | real | gradcheck ratio from 'Measured on this fixture: |sum|=2.46 ... ratio 0.08', no citation |
| 10 | `crates/jammi-kernels/src/ops/attention_block.rs:2245` | `0.166` | cosine | real | gradient-discrimination experiment result ('moves dqkv[0] from 0.166 to 1.33'), no citation |
| 11 | `crates/jammi-kernels/src/ops/dropout.rs:15` | `0.05` | cosine | noise | parameter-value mention (dropout rate) inside a vague 'see module doc' citation to another file, not itself the measured quantity |
| 12 | `crates/jammi-kernels/src/ops/dropout.rs:453` | `0.05` | cosine | noise | explicitly-derived Binomial std-dev bound input, not measured |
| 13 | `crates/jammi-kernels/src/ops/dropout.rs:454` | `0.05` | cosine | noise | same derived Binomial calculation, second mention |
| 14 | `crates/jammi-kernels/src/ops/dropout.rs:455` | `0.0013` | cosine | noise | explicitly-derived Binomial std-dev bound ("a generous, explicitly derived, non-arbitrary bound") |
| 15 | `crates/jammi-kernels/src/ops/geglu.rs:173` | `0.398942` | cosine | noise | design choice: truncated literal vs full-precision PDF-normalizing constant |
| 16 | `crates/jammi-kernels/src/ops/geglu.rs:189` | `0.398942` | cosine | noise | same design choice, second mention |
| 17 | `crates/jammi-kernels/src/ops/softmax.rs:278` | `0.001` | cosine | noise | derived IEEE-754 ULP magnitude at value 10000, not measured |
| 18 | `crates/jammi-kernels/src/ops/softmax.rs:539` | `128/512` | N/M | noise | ModernBERT-large's tested seq-length enumeration ('128/512'), not a mismatch ratio |
| 19 | `crates/jammi-kernels/src/ops/softmax.rs:2775` | `0.01` | cosine | noise | adversarial-fixture arithmetic setup ('2000 + 0.01'), not an observed result |
| 20 | `crates/jammi-kernels/src/ops/softmax.rs:2778` | `0.01` | cosine | noise | same fixture arithmetic, second mention |
| 21 | `crates/jammi-kernels/tests/cuda_parity.rs:3214` | `5%` | % | real | '~5% error once measured against the cancelled ~2385 output', no citation |
| 22 | `crates/jammi-kernels/tests/cuda_parity.rs:3297` | `0.008` | cosine | real | 'measured ~0.008 by the same standalone probe' -- names no grep-verifiable fn or tracked path, unresolvable citation |
| 23 | `crates/jammi-kernels/tests/cuda_parity.rs:3537` | `0.15` | cosine | real | da_tol/da-max ratio: 'measured here at da_tol ~= 6.3 ... ratio ~0.15', no citation |
| 24 | `crates/jammi-kernels/tests/cuda_parity.rs:3538` | `0.15` | cosine | real | independent db_tol/db-max ratio (different variable, same paragraph), same uncited measurement |
| 25 | `crates/jammi-kernels/tests/cuda_parity.rs:4808` | `0.06` | cosine | real | softmax weight split: 'measured on this exact fixture ... 0.06/0.94 point measured'; citation is a vague 'pinned in the diagnostic below' forward-reference naming nothing resolvable |
| 26 | `crates/jammi-kernels/tests/cuda_parity.rs:4808` | `0.94` | cosine | real | same softmax-split measurement, complementary value |
| 27 | `crates/jammi-kernels/tests/cuda_parity.rs:5944` | `0.273` | cosine | real | perf re-run timing sample 1 of 4 ('a re-run on an otherwise-idle box found ... 0.273/0.619/0.641/0.321 ms'), no citation |
| 28 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.619` | cosine | real | same perf re-run, timing sample 2 of 4 |
| 29 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.641` | cosine | real | same perf re-run, timing sample 3 of 4 |
| 30 | `crates/jammi-kernels/tests/cuda_parity.rs:5945` | `0.321` | cosine | real | same perf re-run, timing sample 4 of 4 |
| 31 | `crates/jammi-kernels/tests/cuda_parity.rs:6049` | `5.2%` | % | real | re-measured roofline result ('a clean, exclusive-box re-run ... still showed cast_scale_bf16_f32 at ~2.1 ms (5.2% roofline)'), no citation |
| 32 | `crates/jammi-kernels/tests/cuda_parity.rs:6049` | `53%` | % | real | the earlier (superseded) roofline claim being corrected, same uncited paragraph |
| 33 | `crates/jammi-kernels/tests/cuda_parity.rs:6260` | `5145/16384` | N/M | noise | third restatement of adamw_step.cu:21's same 5145/16384 finding, in the test file's own historical narrative |
| 34 | `crates/jammi-kernels/tests/flash_smoke.rs:302` | `0.39` | cosine | real | mutation-discrimination measurement ('measured on the A100: 3.1e-3 = 0.39 of the sup'), no citation |
| 35 | `crates/jammi-kernels/tests/flash_smoke.rs:578` | `0.06` | cosine | real | per-slot error-bound measurement ('measured on the A100: 0.06-0.60 of the worst case per slot'), no citation |
| 36 | `crates/jammi-kernels/tests/flash_smoke.rs:578` | `0.60` | cosine | real | same per-slot measurement, upper end of the observed range |
| 37 | `crates/jammi-kernels/tests/geglu_oracles.rs:162` | `0.398942` | cosine | noise | design choice: truncated literal vs full-precision constant, comparison framing |
| 38 | `crates/jammi-kernels/tests/geglu_oracles.rs:195` | `0.125` | cosine | real | primary report of the full-scan bf16-backward-divergence measurement ('measured 2026-08-24 ... worst such element measured is 2 bf16 ULP (0.125 absolute)'), no citation |
| 39 | `crates/jammi-kernels/tests/geglu_oracles.rs:196` | `1.02%` | % | noise | derived relative-percentage restatement of line 195's same 0.125 measurement (0.125/12.3) |
| 40 | `crates/jammi-kernels/tests/geglu_oracles.rs:196` | `0.125` | cosine | noise | restatement of line 195's same measured value |
| 41 | `crates/jammi-kernels/tests/geglu_oracles.rs:201` | `1.5625%` | % | noise | pure formula 2 * 2^-7 = 2^-6 = 1.5625%, derived |
| 42 | `crates/jammi-kernels/tests/geglu_oracles.rs:212` | `0.0120` | cosine | real | independent full-scan measurement ('measured max |abs diff| ... EITHER side is exactly zero: 0.0120'), no citation |
| 43 | `crates/jammi-kernels/tests/geglu_oracles.rs:213` | `0.03125` | cosine | noise | defined constant BF16_ABS_FLOOR = 2^-5 = 0.03125 |
| 44 | `crates/jammi-kernels/tests/geglu_oracles.rs:242` | `0.125` | cosine | noise | generic explanatory sentence about what a bare 0.125 would mean, not itself an assertion |
| 45 | `crates/jammi-kernels/tests/geglu_oracles.rs:421` | `0.125` | cosine | noise | restatement of line 195's same measured value, diagnostic-printer context |
| 46 | `crates/jammi-kernels/tests/geglu_oracles.rs:424` | `0.125` | cosine | noise | restatement of line 195's same measured value, continued |
| 47 | `crates/jammi-kernels/tests/oracles.rs:620` | `0.125` | cosine | noise | design constant 'scale = 0.125, an exact power of two' |
| 48 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:185` | `0.0625` | cosine | noise | derived ULP-at-magnitude calculation |
| 49 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:193` | `0.015625` | cosine | noise | derived ULP-at-magnitude calculation ('ULP 2^-6 = 0.015625 at this magnitude') |
| 50 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:412` | `0.333` | cosine | real | primary report of the swept-scaling sweep's worst ordinary-divergence measurement ('Verified directly ... worst observed 0.333 at scaling = 8/3'), no citation |
| 51 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:414` | `0.4%` | % | noise | derived relative-percentage restatement of line 412's same 0.333 measurement (0.333/83) |
| 52 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:414` | `0.333` | cosine | noise | restatement of line 412's same measured value |
| 53 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `21%` | % | real | independent near-zero-crossing-divergence measurement ('worst observed absolute diff 0.174 at magnitude 0.82 (~21% relative)'), no citation |
| 54 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `0.174` | cosine | real | primary report of the same near-zero-crossing measurement's absolute value |
| 55 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:416` | `0.82` | cosine | real | the magnitude at which that same near-zero-crossing measurement was taken |
| 56 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:421` | `0.25` | cosine | noise | defined constant FLOOR = 2^-2 = 0.25 |
| 57 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:422` | `0.174` | cosine | noise | restatement of line 416's same measured value |
| 58 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:423` | `0.333` | cosine | noise | restatement of line 412's same measured value |
| 59 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:426` | `0.52%` | % | real | independent residual-relative-error measurement ('the RESIDUAL relative requirement ... measures 0.52%'), no citation |
| 60 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:426` | `1.5625%` | % | noise | defined constant REL = 2^-6 = 1.5625% |
| 61 | `crates/jammi-bench/src/finetune_step.rs:1055` | `0.001` | cosine | noise | known library default (Default::default()'s lr = 0.001) |
| 62 | `crates/jammi-bench/src/fixture.rs:46` | `0.04` | cosine | noise | constant matched to another already-established margin |
| 63 | `crates/jammi-bench/src/fixture.rs:284` | `100%` | % | noise | hedged disclaimer quoting a hypothetical claim, not asserting it |
| 64 | `crates/jammi-bench/src/rate_gate.rs:20` | `30%` | % | noise | named design threshold DEFAULT_REGRESSION_THRESHOLD |
| 65 | `crates/jammi-bench/src/rate_gate.rs:37` | `30%` | % | noise | same named design threshold, second mention |
| 66 | `crates/jammi-bench/src/recall.rs:102` | `95%` | % | noise | fixed statistical convention (a chosen significance level for a CI) |
| 67 | `crates/jammi-bench/src/recall.rs:882` | `0.27` | cosine | real | independent recall-gap measurement at k=1 ('Measured on the committed fixture the Int8 gap is 0.27/0.17/0.10 at k=1/10/100'), no citation |
| 68 | `crates/jammi-bench/src/recall.rs:882` | `0.17` | cosine | real | same measurement, k=10 |
| 69 | `crates/jammi-bench/src/recall.rs:882` | `0.10` | cosine | real | same measurement, k=100 |
