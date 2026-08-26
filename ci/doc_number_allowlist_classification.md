# ci/doc_number_allowlist.txt classification

Every entry seeded into `ci/doc_number_allowlist.txt` from `origin/main`, hand-classified real (a genuine unproduced-measurement claim -- still needs a real producer citation or a `no-producer:` tag on its own branch, not fixed here) or noise, with a distinct, per-row reason for EVERY row in both classes (round-4 audit fix: a prior revision of this table gave every `real` row the same templated note and counted a repeated restatement of an already-measured number -- e.g. the same `5145/16384` finding retold in four files, or a derived relative-percentage computed from a number two lines up -- as a fresh, independent claim; restatements are now classified noise, with a note naming which row carries the original).

Rows 70-87 (round-5 audit A5) are the `kind = floor` class -- KO-4's category (A)/(B) `.max(<float>)`/`+<float>`/`_floor`-suffix detector, whose own statement-boundary depth-tracking fix (this round's own K1/K4 work) surfaced these 18 findings for the first time; the table carried zero `floor`-kind rows before this round. Rows 88-110 (round-6 audit advisory) classify the 23 non-floor findings that were, as of round 5, live on the merged tree's bare scan but not yet in this table at all (new content from the flash-attention merge, `#387`-`#392`) -- CLOSING that gap. Classified the same way as every other row here (real/noise by content, never by allowlist membership) -- NOT added to `ci/doc_number_allowlist.txt`, which only ever shrinks and is a separate mechanism entirely.

**Noise rate: 58/110 = 52.7%** (real: 52/110 = 47.3%).

This rate is computed over the CLASSIFIED set (the 110 rows below: the 69 rows seeded from `origin/main` plus the 41 rows this table has since added), not over "every unproduced-measurement number that has ever existed on any tree" -- a rate over a set that keeps growing as new content lands is a snapshot of what has been HAND-REVIEWED to date, not a claim that the classified set is exhaustive of the live bare scan at any given moment (it was NOT exhaustive between round 5 and this update: 23 real findings were live and unclassified in that window). Computed directly from the table below -- count the `noise` rows over the total row count, not asserted separately.

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
| 70 | `crates/jammi-kernels/src/ops/cast_scale.rs:584` | `0.0f32` | floor | noise | IEEE-754 negative-zero normalization epilogue term (`+ 0.0f32`), checked for exact bit equality later -- not a tolerance floor |
| 71 | `crates/jammi-kernels/tests/cuda_parity.rs:2766` | `2f32` | floor | noise | mantissa base of `2f32.powi(-10)`, a derived bf16-precision-scale (2^-10) amplitude-fraction floor -- not measured |
| 72 | `crates/jammi-kernels/tests/cuda_parity.rs:2794` | `2f32` | floor | noise | same 2^-10 bf16-precision-scale derivation as row 71, independent expression (the `dwi_out` counterpart) |
| 73 | `crates/jammi-kernels/tests/cuda_parity.rs:3266` | `2.0` | floor | noise | hand-computed expected-value LoRA scaling multiplier inside `let expected = ...map(...).collect()` -- the `.map().collect()` variant of the already-excluded `let expected = [...]` hand-computed-value shape, not a tolerance floor |
| 74 | `crates/jammi-kernels/tests/cuda_parity.rs:3444` | `1.0` | floor | noise | the literal `1.0` in `(1.0 - p)`, the dropout keep-probability complement -- a mathematical identity, not a chosen or measured tolerance |
| 75 | `crates/jammi-kernels/tests/cuda_parity.rs:3507` | `1.0` | floor | noise | same `(1.0 - p)` identity as row 74, the backward/`dx` counterpart |
| 76 | `crates/jammi-kernels/tests/cuda_parity.rs:4304` | `1e-1f64` | floor | real | `abs_floor` set with reference to 'measured ~0.008 by the same standalone probe cited in this test's doc' (the same probe row 22 already names), no resolvable citation |
| 77 | `crates/jammi-kernels/tests/cuda_parity.rs:4571` | `3e-1f64` | floor | real | `abs_floor` widened after 'two real pod runs' found the sign-mixed-cotangent divergence exceeded the prior `0.1` floor -- no resolvable producer citation for the new value |
| 78 | `crates/jammi-kernels/tests/cuda_parity.rs:6013` | `0.05` | floor | noise | comment-labeled 'a loose, non-derived check' sanity bound (the derived assertion is a separate one below it) -- deliberately generous, not a measurement |
| 79 | `crates/jammi-kernels/tests/cuda_parity.rs:7206` | `0.05` | floor | noise | `GROSS_REGRESSION_FLOOR`, comment states 'generous on purpose' -- a deliberate design margin, not a measured value |
| 80 | `crates/jammi-kernels/tests/flash_decisive_timing.rs:152` | `1e-9` | floor | noise | divide-by-zero guard on `mean.max(1e-9)` in a relative steady-state-drift ratio, not a measured tolerance |
| 81 | `crates/jammi-kernels/tests/geglu_oracles.rs:229` | `0.03125` | floor | noise | the same `BF16_ABS_FLOOR = 2^-5 = 0.03125` constant already classified noise at row 43 (then at line 213; line-shifted by later additions to the file) -- KO-4's `_floor`-suffix category (B) rule re-flags the same declaration under the `floor` kind |
| 82 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:429` | `0.25` | floor | noise | `F32_BASE_BF16_LORA_ABS_FLOOR = 2^-2 = 0.25`, the same power-of-two design-choice headroom pattern already classified noise at row 56 (the LoRA-specific counterpart of that row's base `FLOOR` constant) |
| 83 | `crates/jammi-kernels/tests/scaled_cast_add_oracles.rs:436` | `1e-12` | floor | noise | divide-by-zero guard on `magnitude.max(1e-12)` -- the exact `scaled_cast_add_oracles.rs:436` shape this round's own K1 statement-boundary depth-tracking fix was written to target; a machine-epsilon-scale guard, not a measured tolerance |
| 84 | `crates/jammi-encoders/src/htsat_audio.rs:2033` | `1.0` | floor | real | gradcheck relative-with-floor tolerance's `.max(1.0)` floor term; no producer citation or `no-producer:` tag for why `1.0` was chosen |
| 85 | `crates/jammi-bench/src/finetune_step.rs:1362` | `1.0` | floor | noise | `.max(1.0)` floor guarding a pure algebraic-identity check (`triplets_per_s * p50 == batch size`) against a degenerate zero magnitude -- the identity is derivable, not measured |
| 86 | `crates/jammi-bench/src/conformal.rs:505` | `0.0` | floor | real | non-negativity clamp (`.max(0.0)`) on a conformal spec floor computed from `pair.measured - spec.margin`; the clamp's own justification (a floor cannot be negative) is undocumented at this site -- no producer/`no-producer:` tag |
| 87 | `crates/jammi-bench/src/conformal.rs:614` | `0.0` | floor | real | the same `.max(0.0)` non-negativity clamp as row 86, an independent call site (the round-trip-through-the-gate test) |
| 88 | `crates/jammi-kernels/tests/cuda_parity.rs:7665` | `0.05` | cosine | noise | hypothetical comparison to a DIFFERENT test's `+ 0.05` floor shape ('as it would be under a `+ 0.05` floor'), not an assertion this fixture makes |
| 89 | `crates/jammi-kernels/tests/cuda_parity.rs:7973` | `0.02` | cosine | real | 'measured this round on a100b: ... dqkv ratio-to-bound 0.02-0.16 across a 5-seed sweep', no citation (range low end) |
| 90 | `crates/jammi-kernels/tests/cuda_parity.rs:7973` | `0.16` | cosine | real | same 5-seed sweep as row 89, range high end |
| 91 | `crates/jammi-kernels/tests/flash_decisive_timing.rs:29` | `5%` | % | noise | doc-comment restatement of the `STEADY_STATE_REL_TOL = 0.05` design threshold declared later in the same file -- a definition, not a measurement |
| 92 | `crates/jammi-kernels/tests/flash_torch_parity.rs:28` | `0.0036` | cosine | real | primary report of a hand-computed numpy divergence measurement ('measured by hand with numpy ... `\|f64_truth - ref_o\|max = 0.0036`'), no citation |
| 93 | `crates/jammi-kernels/tests/geglu_oracles.rs:218` | `1.06%` | % | real | 'verified directly, not just argued: the maximum (\|diff\| - FLOOR) / magnitude ... measures `1.06%`', no citation |
| 94 | `crates/jammi-kernels/tests/geglu_oracles.rs:219` | `1.5625%` | % | noise | restatement of `BF16_REL_TOL = 2^-6 = 1.5625%` (declared two lines below), a pure formula, not measured -- same pattern as the already-classified row 41 |
| 95 | `crates/jammi-encoders/src/layer_norm.rs:289` | `74734/262144` | N/M | real | measured value citing its own producing test by NAME in prose ('see that test's own printed count') but not in the grammar-recognized `see [`fn`]` form -- no resolvable citation |
| 96 | `crates/jammi-encoders/src/layer_norm.rs:293` | `74734/262144` | N/M | noise | restatement of row 95's same measurement, continued paragraph |
| 97 | `crates/jammi-encoders/src/layer_norm.rs:924` | `371/148` | N/M | real | one of 'the values measured on this branch' (the 93/371 budget constants), shown here in a derived-headroom ratio (371/148 ~= 2.5x); no citation for the underlying 371 measurement |
| 98 | `crates/jammi-encoders/src/layer_norm.rs:948` | `1%` | % | noise | design threshold ('a partial regression touching only ~1% of rows'), a chosen test sensitivity, not measured |
| 99 | `crates/jammi-encoders/src/layer_norm.rs:950` | `0.01` | cosine | noise | the same ~1% design threshold as row 98, expressed as the decimal literal used in code (`rows * 0.01`) |
| 100 | `crates/jammi-encoders/src/layer_norm.rs:954` | `1%` | % | noise | third mention of the same ~1% design threshold (rows 98-99) |
| 101 | `crates/jammi-encoders/src/layer_norm.rs:1664` | `59/262144` | N/M | real | cites `[`F16_REDUCTION_ORDER_BUDGET_FRACTION`]`, a CONST not a `#[test]`/pub-fn-under-`tests/`, so the citation grammar cannot resolve it -- no valid producer |
| 102 | `crates/jammi-encoders/src/modernbert.rs:3392` | `0.80` | cosine | real | 'the auditor's own 8-seed sweep at b8-s512 found the healthy pooled flash/block ratio ranging `0.80`-`1.48`', no grep-verifiable citation |
| 103 | `crates/jammi-encoders/src/modernbert.rs:3418` | `0.18823` | cosine | real | 'the committed GREEN run ... showed ... (`0.18823` > `0.17482` at b8_s512', an independently-measured error value (flash leg, b8_s512), no citation |
| 104 | `crates/jammi-encoders/src/modernbert.rs:3418` | `0.17482` | cosine | real | same measurement event as row 103, the complementary block-leg value at b8_s512 |
| 105 | `crates/jammi-encoders/src/modernbert.rs:3418` | `0.16754` | cosine | real | same measurement event as row 103, the flash-leg value at b1_s128 |
| 106 | `crates/jammi-encoders/src/modernbert.rs:3419` | `0.15673` | cosine | real | same measurement event as row 103, the complementary block-leg value at b1_s128 |
| 107 | `crates/jammi-encoders/src/modernbert.rs:3469` | `0.1275` | cosine | noise | diagnostic-only 'class-sweep probe ... run this round as a diagnostic (not committed' -- a parameter-perturbation description, not a claimed result requiring a producer |
| 108 | `crates/jammi-encoders/src/modernbert.rs:3469` | `0.125` | cosine | noise | same not-committed diagnostic as row 107, the production `softmax_scale` baseline value being perturbed |
| 109 | `crates/jammi-encoders/src/modernbert.rs:3512` | `8%` | % | real | measured range (~4-8%) citing 'the committed `2026-08-25-flash-arm-encoder-oracle-*.json` artifact' by prose description, not the recognized `measured by <path>` form -- unresolvable by the grammar despite being genuinely trackable |
| 110 | `crates/jammi-encoders/src/modernbert.rs:3931` | `0.23` | cosine | real | primary report of a measured cosine-distance range ('ranging `0.23`-`1.04` across seeds'), cites a pod run/commit but not the grammar-recognized citation form |
