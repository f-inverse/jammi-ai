# Unit 63 H5 — kernel-mutant RED column

CONTRACT 63 (PLAN v2 delta 7; corrected design in CONTRACT.md amendment
2026-08-29b, item 3) requires a kernel-mutant RED column whose sensitivity
claim is precisely bounded: **"detects a regression >= mutant M"**, never
"detects movement" and never "detects hitting some unspecified floor." This
directory pins that mutant family — its exact patches, base sha, sha256s,
and the honest, narrow claim the campaign is allowed to make from each dose.

This file is rewritten in place (current-truth discipline) to retire M1 as
the active design and replace it with the pre-registered dose family.

## M1 — honest non-detection record (superseded design, kept for the record)

**M1 was measured UNDETECTED.** Its patch (`M1.patch`, sha256
`68d2023b936fe613c75e85a49ad4c6de01fb27442ffb967db00107fbc012d926`, base
`4257cde6d51184475b3e798f5d7e9c3885a763ca`) is an off-by-one bias-correction
defect (`t+1` instead of `t` in `adamw_step_fused_t`'s `scale_m`/`scale_v`
derivation). It ran as the campaign-v1 mutant column: mutant-vs-fused
n_pos=8/12 mean -0.0035; mutant-vs-alloff n_neg=8/12 mean -0.0236 — **NOT
RED**. Full measured record: `measurements/campaign-v1/mutant-m1/` (per-seed
JSONs + `mutant63.log`) and `measurements/campaign-v1/README.md`'s
`mutant-m1/` section.

Root cause (pressure-tested, CONTRACT.md amendment 2026-08-29b item 3): M1's
net update multiplier `[(1-beta1^t)/(1-beta1^(t+1))]*sqrt[(1-beta2^(t+1))/(1-beta2^t)]`
is a **sign-flipping early transient** — `0.744` at `t=1`, converging to
`~1.009` by `t=50` — not a sustained one-direction dose. The reasoning that
picked M1 was in the wrong space (per-step magnitude of the bias-correction
scalar) instead of the space the gate actually measures (a fixed-sign,
non-vanishing perturbation to every step's effective learning rate, summed
over an entire training run). M1 stands as an honestly-recorded
non-detection, **not** a sensitivity bound; it is not deleted, and `M1.patch`
remains committed as a file (never applied to tree state on this branch).

## The new dose family: sustained silent-lr-inflation, `M_eps_{0.02,0.10,0.50}`

**Shape:** the fused AdamW update's effective learning rate is scaled by
`(1+eps)`, applied uniformly at **every** step — a one-parameter, monotone,
**sustained** perturbation (unlike M1's transient, this does not decay
toward 1.0 as `t` grows; `(1+eps)` is constant for all `t`). Three doses:
`eps in {0.02, 0.10, 0.50}`.

**Where it lives:** `adamw_step_fused_t` in
`crates/jammi-kernels/src/ops/adamw_step.rs`, immediately before the
`AdamThetaUpdate` that consumes `params.lr`:

```diff
+    const LR_INFLATION_FACTOR: f64 = 1.02_f64;  // 1.10 / 1.50 for the other doses
+    let lr_dose = params.lr * LR_INFLATION_FACTOR;
     super::apply_inplace3(
         theta,
         first_moment,
         second_moment,
-        AdamThetaUpdate::new(params.lr, params.weight_decay, scale_m, scale_v, params.eps),
+        AdamThetaUpdate::new(lr_dose, params.weight_decay, scale_m, scale_v, params.eps),
     )
```

Each patch touches exactly this one file, inside `adamw_step_fused_t` — the
production fused-AdamW entry point `jammi-ai::fine_tune::adamw` calls on the
campaign's fused arm. Nothing else changes:

- `InplaceOp2::name()` / `InplaceOp3::name()` — unchanged.
- `validate_step_domain` (dtype/shape/device/aliasing/contiguity/injectivity)
  — unchanged: a dose leg's ADMISSION behavior is identical to a clean fused
  leg.
- Every `DispatchCounters` field / the `admission` module — unchanged: a dose
  leg's premise fields (which kernel ran, whether it was admitted, how many
  times it was invoked) read IDENTICAL to a clean fused leg. This is the
  point: only the DECISION statistic (held-out loss, via the sign test)
  catches the perturbation — legs stay premise-clean, per the task's own
  "no dispatch/identity surface" constraint.
- `AdamMomentUpdate`/`AdamThetaUpdate`'s per-element rounding chain — the
  arithmetic itself is untouched; only the SCALAR fed into it (`lr`) is
  scaled. `weight_decay` is passed through as `params.weight_decay`
  unscaled by the caller's convention already (it multiplies `lr` inside
  `AdamThetaUpdate` to form `one_minus_lr_lambda`), so the dose's `(1+eps)`
  factor also raises the effective weight-decay shrink proportionally — the
  dose scales the ENTIRE per-step update (both the gradient step and the
  decay term), matching the "the fused AdamW update scaled by (1+eps)"
  framing exactly, not just the gradient term in isolation.

**Files:**
- `M_eps_0.02.patch` — sha256 `c173ae2bee0db87ac7719c3cd9ae8d12480e81bf5cab1832c18c0788741275f1`
- `M_eps_0.10.patch` — sha256 `63b7f53c5b19a5e2081551fb25564154460f6c0c4bcc55520d964f75990a607c`
- `M_eps_0.50.patch` — sha256 `30abfc1cf4d81be321de45d600e0741bb8a00169f34253fa3009e2ade55d2e98`

All three are unified diffs against base sha
`ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` (CONTRACT amendment 2026-08-29b's
commit), verified `git apply --check` clean at that sha, and each verified
apply -> `cargo build -p jammi-kernels` (exit 0) -> `git checkout --` revert,
independently, one dose at a time (never two doses applied simultaneously).

## Prediction table — REQUIRED BEFORE ANY SPEND

Derived ONLY from committed measured values in
`measurements/campaign-v1/raw/` (numeric extraction only — `seed{1,2}` are
the only seeds with a committed `lr0` leg) and
`measurements/campaign-v1/finetune_run_ab_report.json`'s top-level
`determinism_floor.cross_seed_spread` and `decision.mean_d` /
`sign_test.n_neg` fields.

### Step 1 — the only committed operating-point data: lr0 vs trained, fused arm

The mutant only fires on the fused arm (ALLOFF disables `adamw_step_fused`
entirely, per the Frame's `ALLOFF=attention_block_flash,adamw_step_fused`),
so only the **fused-arm** lr0-vs-trained pair is the relevant secant. LoRA
init is `ZerosB`, so the untrained (`lr=0`) held-out mean is identical
across seeds and arms:

| field | value |
|---|---|
| `held_out_example_mean`, lr=0 (seed1, fused) | `3.422172799706459` |
| `held_out_example_mean`, lr=0 (seed2, fused) | `3.422172799706459` |
| `held_out_example_mean`, lr=2e-4 (seed1, fused, r1) | `3.218041628599167` |
| `held_out_example_mean`, lr=2e-4 (seed2, fused, r1) | `3.4061567336320877` |

Only 2 of the 12 gate seeds have a committed `lr0` leg — this derivative
estimate is therefore an **n=2** estimate with no error bars available from
committed data.

Per-seed secant slope (decrease in held-out mean per unit increase of the
effective-lr scale `s`, over the full range `s: 0 -> 1`, `s=1` being the
campaign's trained lr=2e-4):

```
slope(seed) = held_out_mean(s=0) - held_out_mean(s=1)

slope(seed1) = 3.422172799706459 - 3.218041628599167   = 0.204131171107292
slope(seed2) = 3.422172799706459 - 3.4061567336320877  = 0.016016066074371

avg_slope    = (0.204131171107292 + 0.016016066074371) / 2 = 0.110073618590832
```

**Caveat (family K — diagnose the structure, don't over-trust a small-n
extrapolation):** the two available seeds disagree by **12.75x**
(`0.204 / 0.016`) on this slope. This is itself the headline finding of
this section: committed data does NOT tightly constrain
`d(held-out-mean)/d(effective-lr-scale)`; any point estimate below is a
weak central tendency over `n=2`, and the true range spans the two
per-seed slopes. Separately, this secant is measured over the **entire**
`[0, 1]` range (untrained -> trained) — using it as a stand-in for the
LOCAL derivative at the operating point `s=1` (where the doses actually
perturb) assumes the held-out-mean-vs-lr-scale response is linear over a
much larger interval than any dose probes; this is an assumption, not a
measured fact, and is progressively less trustworthy as `eps` grows
(worst at `eps=0.50`, a 50% relative lr perturbation, well outside any
region this secant can vouch for).

### Step 2 — extrapolated per-dose predicted shift

`Δmean(eps) = -slope * eps` (linear extrapolation beyond `s=1` to `s=1+eps`,
negative sign because the measured secant direction is DECREASING
held-out-mean as scale increases):

| eps | using slope(seed2)=0.016016 (weak end) | using avg_slope=0.110074 | using slope(seed1)=0.204131 (strong end) |
|---|---|---|---|
| 0.02 | `-0.016016 * 0.02 = -0.00032032` | `-0.110074 * 0.02 = -0.00220147` | `-0.204131 * 0.02 = -0.00408262` |
| 0.10 | `-0.016016 * 0.10 = -0.00160161` | `-0.110074 * 0.10 = -0.01100736` | `-0.204131 * 0.10 = -0.02041312` |
| 0.50 | `-0.016016 * 0.50 = -0.00800803` | `-0.110074 * 0.50 = -0.05503681` | `-0.204131 * 0.50 = -0.10206559` |

**Predicted DIRECTION (all doses, both ends of the range): IMPROVEMENT
(held-out mean DECREASES), not degradation.** This is the naive
linear-extrapolation prediction from the ONLY committed operating-point
data and is stated in full knowledge that it runs counter to the intuitive
framing of "silent lr inflation" as a harmful regression. If real training
dynamics at these doses instead show curvature (e.g., the well-known
too-large-lr degradation regime, most plausible at `eps=0.50`), the sign
could reverse — that reversal is exactly the falsifiable case this
prediction exists to be checked against; per family F/K discipline this is
reported as a prediction to be measured, not assumed.

### Step 3 — predicted detection verdict under the >=11/12+mean rule

Committed benchmarks to compare the predicted shift against:
- `determinism_floor.cross_seed_spread = 0.08264997071681932` (measured
  cross-seed spread; zero repeat noise — r1/r2 bit-identical, `max_delta =
  0.0`, so this spread is the only noise source relevant to whether a fixed
  per-seed shift can flip >=11/12 seeds concordantly).
- `decision.mean_d = -0.023799017749049446`, `sign_test.n_neg = 8/12` (the
  campaign's own REAL, measured fused-vs-alloff effect — and it did NOT
  reach the 11/12 threshold). This is an empirical calibration point: an
  effect of this magnitude, on this same 12-seed gate, produced only 8/12
  concordance, not RED.

| eps | predicted abs(Δmean) (avg_slope) | vs. cross_seed_spread (0.0826) | vs. already-non-detected effect (0.0238, 8/12) | predicted verdict |
|---|---|---|---|---|
| 0.02 | 0.00220 | 2.7% of spread | 9.2% of the effect that already failed to reach 11/12 | **predicted NOT detected** (very high confidence — an order of magnitude below both benchmarks) |
| 0.10 | 0.01101 | 13.3% of spread | 46.3% of the effect that already failed to reach 11/12 | **predicted NOT detected** (high confidence — still smaller than an effect that already failed) |
| 0.50 | 0.05504 | 66.6% of spread | 2.31x the effect that already failed to reach 11/12 | **predicted NOT reliably detected**, but this is the closest call: the shift exceeds the already-non-detected effect by >2x, yet stays below 1 full cross-seed-spread-unit, so >=11/12 concordance is not confidently predicted either way. Using the strong-end per-seed slope (`0.204131`) instead of the average, eps=0.50's predicted shift (`-0.10207`) would EXCEED the cross-seed spread — the two available seeds disagree sharply on whether this dose is detectable at all. **eps=0.50 is the dose most worth actually spending on**, both because it is the pre-registered largest dose and because the n=2 slope estimate leaves its predicted magnitude genuinely uncertain. |

**Important design-level flag, not a refusal:** because the predicted
direction (Step 2) is IMPROVEMENT under the naive linear-extrapolation
model, even a dose that DID reach >=11/12+mean would read RED-for-
improvement under the Frame's two-sided rule, not RED-for-degradation.
Amendment 2026-08-29b item 3's acceptance criterion ("mutant column proven
RED", discharged by "the smallest detected dose") is scoped to the
DEGRADATION direction (mirroring M1's pass criterion). If the actual pod
run confirms the naive-linear direction rather than reversing it at higher
doses, that is itself a finding requiring the dose family's sign to be
revisited (e.g., whether `(1-eps)` — a silent lr DEFLATION — would be the
degradation-producing member of this same family in this particular
small-scale/short-run regime) rather than silently reinterpreting an
improvement-direction RED as satisfying the degradation-sensitivity claim.
This report does not redesign the family (the `(1+eps)` shape is what
amendment 2026-08-29b item 3 pre-registers); it records the prediction so
the eventual pod result can be checked against it before any RED/GREEN
verdict is claimed.

No dose is refused for lack of data — all three are supported by the same
committed two-seed secant; the confidence intervals above (not point
estimates) are the honest product of what is actually committed.

## On-pod procedure (dose legs only — substituted INTO the fused arm)

The mutant-vs-fused pairing used by M1 is explicitly RETIRED. Per amendment
2026-08-29b item 3, each dose column is produced by substituting the mutant
into the fused arm itself and merging against the SAME v2 `alloff` legs
under the SAME `>=11/12+mean` rule — the gate's own statistic, not a
separate ad hoc comparison.

1. On the pod, clone/checkout at the recorded base sha into a **scratch
   worktree**, separate from the campaign's production checkout:
   ```sh
   git worktree add /workspace/scratch-mutant-dose ca559b4f16cd1129a2f95ccdd82288b3418e0d0a --detach
   cd /workspace/scratch-mutant-dose
   ```
2. Verify the patch applies cleanly and record its hash before touching
   anything (repeat per dose, one at a time, never two doses applied to the
   same scratch checkout simultaneously):
   ```sh
   sha256sum docs/plans/63-how-well/mutants/M_eps_<dose>.patch   # must equal this README's recorded hash
   git apply --check docs/plans/63-how-well/mutants/M_eps_<dose>.patch && echo CLEAN
   git apply docs/plans/63-how-well/mutants/M_eps_<dose>.patch
   ```
3. Build with the campaign's exact feature list (same CUDA/bf16 features as
   the fused arm — do not add or drop features relative to the campaign's
   recorded build invocation):
   ```sh
   cargo build --release -p jammi-kernels --features cuda
   cargo build --release -p jammi-ai --features <campaign's exact feature list>
   ```
4. Run the dose's 12 legs through the same `run_leg` vector the campaign's
   fused arm uses (same shapes, same seeds, same step count, same
   `attention_block_flash` + `adamw_step_fused` dispatch wiring) — the ONLY
   difference between a dose leg and a clean fused leg is that this scratch
   build's `jammi-kernels` binary was compiled from the patched source. This
   dose leg SUBSTITUTES for the fused arm's leg at that seed — it is not a
   third, separately-merged column.
5. Merge each dose's 12 substituted-fused legs against the SAME campaign
   `alloff` legs already on record (identity permitting, per H5(1)'s
   ALLOFF-reuse rule) using the gate's own merger and its own
   `>=11/12+mean` sign-test statistic — the same code path the real A/B
   verdict uses, not a bespoke mutant-vs-fused comparison.
6. Record, per dose leg, in the column artifact:
   - `base_sha = ca559b4f16cd1129a2f95ccdd82288b3418e0d0a`
   - `patch_sha256` = the dose's recorded sha256 above
   - `dose_label = eps_0.02` / `eps_0.10` / `eps_0.50`
   - the same held-out-loss / step-count / dispatch-counter fields the
     clean fused legs record, so each dose column is diff-able against the
     `alloff`/fused columns field-for-field.
7. The reported sensitivity is the pair of ADJACENT doses straddling
   detection (per amendment item 3) — run doses in ascending order and stop
   describing the sweep as "complete" only once a straddling pair is found
   or all three have been run without one.
8. Tear down the scratch worktree and its build artifacts after the legs
   complete; do not leave a patched binary or scratch checkout on the pod
   past the dose-leg run. Each patch is committed to this repo as a FILE
   only — it must never land as tree state on any branch that builds the
   production `jammi-kernels` binary.

## Pass criterion (per dose)

Exactly as M1's, restated for the retired-pairing correction: **a sign test
over the dose's substituted-fused column's held-out loss vs. the campaign's
`alloff` legs' held-out loss reads RED (degradation) at the campaign's
pre-registered significance threshold (`alpha2=0.0064`, `>=11/12` +
mean-agrees-in-sign).** If a dose's sign test does NOT read RED, that is a
finding about the gate's sensitivity floor at that dose and must be reported
as such — never suppressed or silently retried with an undocumented larger
dose. No `--allow-premise-failure`, no waived-seed list, no rescale switch
anywhere in the merger (amendment item 4).

## CPU-verifiable demonstration (not committed to the production suite)

A temporary, non-committed unit test
(`mutant_dose_demonstration_diverges_from_the_correct_oracle`) was added
locally to `crates/jammi-kernels/src/ops/adamw_step.rs`, once per dose (one
dose patch applied at a time, test added, run, then BOTH the test and the
patch reverted via `git checkout --` before moving to the next dose — never
committed). It runs the dose-patched `adamw_step_fused_t` against the
file's own correct `eager_step` oracle (unpatched formula) for 5
consecutive steps on the same fixed 4-element input M1 used
(`beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=0.01, eps=1e-8`), printing
the per-step L2 divergence in `theta` (`cargo test -p jammi-kernels --lib
mutant_dose_demonstration -- --nocapture`, patch applied, CPU only):

```
eps=0.02:
step=1 l2_divergence=3.508513e-5
step=2 l2_divergence=7.017026e-5
step=3 l2_divergence=1.0525539e-4
step=4 l2_divergence=1.4034052e-4
step=5 l2_divergence=1.7542565e-4

eps=0.10:
step=1 l2_divergence=1.7608417e-4
step=2 l2_divergence=3.5216834e-4
step=3 l2_divergence=5.28184e-4
step=4 l2_divergence=7.0419966e-4
step=5 l2_divergence=8.801983e-4

eps=0.50:
step=1 l2_divergence=8.7980856e-4
step=2 l2_divergence=1.7595316e-3
step=3 l2_divergence=2.6392546e-3
step=4 l2_divergence=3.5189774e-3
step=5 l2_divergence=4.3986836e-3
```

The step-5 divergence scales linearly with `eps` as the design intends
(sustained, monotone family): `8.801983e-4 / 1.7542565e-4 = 5.017` (expected
`5x`, since `0.10 / 0.02 = 5`); `4.3986836e-3 / 1.7542565e-4 = 25.076`
(expected `25x`, since `0.50 / 0.02 = 25`) — both within floating-point
accumulation tolerance of the exact ratio, confirming each dose is a real,
proportionally-scaled numeric perturbation on this crate's own oracle, not
a no-op and not a copy-paste of a different dose's constant.

`git checkout -- crates/jammi-kernels/src/ops/adamw_step.rs` restored the
clean tree after each dose; neither the patches nor the temporary test are
part of any commit on this branch (all three patches ARE committed, as
files only, alongside this README).

## Dose ladder (CONTRACT amendment 2026-08-29b item 3) — merger interface

M1 above is recorded as a NON-DETECTION (a sign-flipping early transient,
never a sensitivity bound — see the campaign-v1 finding this amendment
corrects). The corrected design is a one-parameter, monotone, SUSTAINED
dose family — the fused AdamW update scaled by `(1+eps)`,
`eps in {0.02, 0.10, 0.50}` — each dose run as its own scratch-worktree
mutant (the SAME on-pod procedure above: patched `jammi-kernels`, the SAME
`run_leg` vector, torn down after). This section documents ONLY the
merger's own consuming interface (`ci/scripts/perf/ab_merge.py`); the
per-dose PREDICTED effect (held-out example-mean units, stated BEFORE the
spend, per the amendment's own falsifiability requirement) is a separate,
not-yet-authored record this section does not substitute for.

- **Leg naming**: a dose's legs are recorded under the SAME `raw_dir` a
  campaign already uses, tagged `seed{seed}__fused__mutant-{dose_label}`
  (`ab_merge.mutant_leg_repeat_tag(dose_label)`) — a `repeat` value that can
  never collide with `r1`/`r2` (the main A/B pool) or `lr0` (the RED
  control), the SAME file-naming isolation the lr0 control already relies
  on. `dose_label` is an operator-chosen string (e.g. `"eps0.02"`), never
  reinterpreted by the merger.
- **Per-leg recorded fields**: every field a clean `fused` leg already
  carries, PLUS this section's own three (`mutant_id`, `base_sha`,
  `patch_sha256`, step 5 above) — a mutant leg missing any of the three, or
  whose own `patch_sha256` disagrees with the dose column it is merged
  under, is refused (`ab_merge.finetune_run_mutant_column_violations`).
- **Merger CLI**: `ab_merge.py finetune-run RAW_DIR OUT_DIR SEEDS
  [LR0_SEEDS] [--allow-missing-lr0-control] [--mutant-legs
  DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,...]` — `--mutant-legs` is repeatable,
  once per dose column; `finetune_run_ab.sh`'s own
  `FINETUNE_RUN_AB_MUTANT_LEGS` env var (`;`-separated specs) is a pure
  pass-through into the same flag, never a second leg-running mechanism.
- **Per-dose output** (`mutant_dose_ladder.doses[i]` in the merged
  artifact): `{dose_label, patch_sha256, detected, n_pos, n_neg, mean_d,
  p_value, clean_pair_count, violations, ...}` — `detected` is `"RED"` iff
  the SAME `>=11/12` threshold the primary decision uses is met in the
  DEGRADATION direction (mutant worse than alloff); `"not-detected"` if the
  threshold is not met, or is met in the opposite (anomalous-improvement)
  direction (M1's own sign-flipping-transient shape); `"INVALID"` if the
  premise-clean pair count is not exactly the pre-registered 12, or the
  sign test itself refuses — a correctness-of-measurement carve-out beyond
  the amendment's own literal `RED`/`not-detected` pair, added so a
  malformed dose column is never silently read as "not-detected" (a
  substantive finding) when it is really "this column could not be
  evaluated at all".
- **Sensitivity statement**: `mutant_dose_ladder.sensitivity` — the first
  adjacent `(not-detected, RED)` pair in the CALLER-SUPPLIED dose order
  (`ab_merge.mutant_dose_ladder_sensitivity`), or `null` if no such
  transition exists in that order.
- **Mutant legs never enter the primary A/B set**: proven structurally (the
  `mutant-<dose_label>` repeat tag can never equal `r1`/`r2`/`lr0`) and
  empirically (`MutantDoseLadderTests.test_mutant_leg_never_leaks_into_the_ab_set`,
  `ci/scripts/perf/test_ab_merge.py`).

## Files

- `M1.patch` — the retired mutant's committed unified diff (patch-file-only;
  never applied to tree state on this branch). Kept for the record; see
  `measurements/campaign-v1/mutant-m1/` for its measured non-detection.
- `M_eps_0.02.patch`, `M_eps_0.10.patch`, `M_eps_0.50.patch` — the new dose
  family's committed unified diffs against `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a`
  (patch-file-only; never applied to tree state on this branch).
- `README.md` — this file.
