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
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c3=meand('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c4=negcount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c5=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c6=meand('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
n_pos=8/12 mean -0.0035; mutant-vs-alloff n_neg=8/12 mean -0.0236 — **NOT
RED**. Full measured record: `measurements/campaign-v1/mutant-m1/` (per-seed
JSONs + `mutant63.log`) and `measurements/campaign-v1/README.md`'s
`mutant-m1/` section.

Root cause (pressure-tested, CONTRACT.md amendment 2026-08-29b item 3): M1's
<!-- claims63: c1=const; c2=const; c3=const; c4=const -->
net update multiplier `[(1-beta1^t)/(1-beta1^(t+1))]*sqrt[(1-beta2^(t+1))/(1-beta2^t)]`
<!-- claims63: c1=ledger -->
is a **sign-flipping early transient** — `0.744` at `t=1`, converging to
<!-- claims63: c1=ledger -->
`~1.009` by `t=50` — not a sustained one-direction dose. The reasoning that
picked M1 was in the wrong space (per-step magnitude of the bias-correction
scalar) instead of the space the gate actually measures (a fixed-sign,
non-vanishing perturbation to every step's effective learning rate, summed
over an entire training run). M1 stands as an honestly-recorded
non-detection, **not** a sensitivity bound; it is not deleted, and `M1.patch`
remains committed as a file (never applied to tree state on this branch).

## The dose family: sustained silent-lr-(in/de)flation, signed `eps`

**Shape:** the fused AdamW update's effective learning rate is scaled by
`(1+eps)`, applied uniformly at **every** step — a one-parameter, monotone,
**sustained** perturbation (unlike M1's transient, this does not decay
<!-- claims63: c1=ledger -->
toward 1.0 as `t` grows; `(1+eps)` is constant for all `t`).

**Superseded (falsified pre-spend, NOT scheduled to run):** amendment
<!-- claims63: c1=const; c2=const; c3=const -->
2026-08-29b item 3 originally named `eps in {0.02, 0.10, 0.50}` (positive
only — silent lr INFLATION). The REQUIRED pre-spend prediction table below
(Step 2/3, original version) showed the `(1+eps)` direction as held-out
IMPROVEMENT, not degradation, for both measured seeds — a positive dose
therefore cannot discharge "mutant column proven RED (degradation)". `M_eps_0.02.patch`
and `M_eps_0.10.patch` stay committed as the falsified-but-recorded
doses (documentation of the falsification, current-truth discipline); they
are not part of the scheduled ladder below.

**Current (CONTRACT.md addendum 2026-08-29c): the signed ladder,
<!-- claims63: c1=const; c2=const; c3=const -->
`eps in {-0.50, -0.10, +0.50}`.** Same one-parameter monotone family,
`eps` now SIGNED:
<!-- claims63: c1=const; c2=const -->
- Negative doses (`-0.10`, `-0.50`) are silent lr DEFLATION — the
  undertrained-regression class — and carry the predicted DEGRADATION
  direction (Step 2 below: secant symmetric, predicted per-seed shift
  `+|eps|*slope_seed`, sign-consistent across both measured seeds).
<!-- claims63: c1=const -->
- `+0.50` is retained deliberately as the two-sided falsification cell for
  the improvement prediction itself: if it reads RED-for-investigation
  shaped as improvement, the original prediction is confirmed; if it
  instead degrades, the secant extrapolation is refuted and that must be
  recorded here.
- Acceptance 5's "mutant column proven RED" is discharged by the smallest
  detected DEGRADATION dose (expected among the negative eps values); the
  reported sensitivity is the adjacent-dose pair straddling detection
  within the negative branch.

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
- `M_eps_0.02.patch` — sha256 `c173ae2bee0db87ac7719c3cd9ae8d12480e81bf5cab1832c18c0788741275f1` — **falsified, not scheduled**
- `M_eps_0.10.patch` — sha256 `63b7f53c5b19a5e2081551fb25564154460f6c0c4bcc55520d964f75990a607c` — **falsified, not scheduled**
- `M_eps_0.50.patch` — sha256 `30abfc1cf4d81be321de45d600e0741bb8a00169f34253fa3009e2ade55d2e98` — **scheduled** (two-sided falsification cell)
- `M_eps_-0.10.patch` — sha256 `8e3df8e66f7ab5850fac15ce97a109159963683c74526509051cb8661892895e` — **scheduled**
- `M_eps_-0.50.patch` — sha256 `1b0389f1ff117846259f47b850b0112b99bfb2ae96b868cd612694bada45c372` — **scheduled**

`M_eps_0.02.patch`/`M_eps_0.10.patch`/`M_eps_0.50.patch` are unified diffs
against base sha `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` (CONTRACT
amendment 2026-08-29b's commit); `M_eps_-0.10.patch`/`M_eps_-0.50.patch` are
unified diffs against base sha `cba0b835` (CONTRACT addendum 2026-08-29c's
commit — `crates/jammi-kernels/src/ops/adamw_step.rs` is byte-identical
between the two base shas, so all five patches apply against either
checkout). Every patch is verified `git apply --check` clean at its base
sha, and each is verified apply -> `cargo build -p jammi-kernels` (exit 0)
-> `git checkout --` revert, independently, one dose at a time (never two
doses applied simultaneously).

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
<!-- claims63: c1=abs(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean) -->
| `held_out_example_mean`, lr=0 (seed1, fused) | `3.422172799706459` |
<!-- claims63: c1=abs(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__lr0.json#/tiers/finetune_run/held_out_example_mean) -->
| `held_out_example_mean`, lr=0 (seed2, fused) | `3.422172799706459` |
<!-- claims63: c1=abs(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean) -->
| `held_out_example_mean`, lr=2e-4 (seed1, fused, r1) | `3.218041628599167` |
<!-- claims63: c1=abs(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean) -->
| `held_out_example_mean`, lr=2e-4 (seed2, fused, r1) | `3.4061567336320877` |

<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__lr0.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
Only 2 of the 12 gate seeds have a committed `lr0` leg — this derivative
<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__lr0.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
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
<!-- claims63: c1=ratio(absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean), absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean)) -->
extrapolation):** the two available seeds disagree by **12.75x**
<!-- claims63: c1=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean); c2=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean) -->
(`0.204 / 0.016`) on this slope. This is itself the headline finding of
this section: committed data does NOT tightly constrain
`d(held-out-mean)/d(effective-lr-scale)`; any point estimate below is a
<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__lr0.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
weak central tendency over `n=2`, and the true range spans the two
per-seed slopes. Separately, this secant is measured over the **entire**
`[0, 1]` range (untrained -> trained) — using it as a stand-in for the
LOCAL derivative at the operating point `s=1` (where the doses actually
perturb) assumes the held-out-mean-vs-lr-scale response is linear over a
much larger interval than any dose probes; this is an assumption, not a
measured fact, and is progressively less trustworthy as `eps` grows
<!-- claims63: c1=ratio(0.50, 0.01) -->
(worst at `eps=0.50`, a 50% relative lr perturbation, well outside any
region this secant can vouch for).

### Step 2 — extrapolated per-dose predicted shift

`Δmean(eps) = -slope * eps` (linear extrapolation beyond `s=1` to `s=1+eps`,
negative sign because the measured secant direction is DECREASING
held-out-mean as scale increases):

<!-- claims63: c1=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean); c2=ledger; c3=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean) -->
| eps | using slope(seed2)=0.016016 (weak end) | using avg_slope=0.110074 | using slope(seed1)=0.204131 (strong end) |
|---|---|---|---|
<!-- claims63: c1=const; c2=const; c3=const; c4=const; c5=const; c6=const; c7=const; c8=const; c9=const; c10=const -->
| 0.02 | `-0.016016 * 0.02 = -0.00032032` | `-0.110074 * 0.02 = -0.00220147` | `-0.204131 * 0.02 = -0.00408262` |
<!-- claims63: c1=const; c2=const; c3=const; c4=const; c5=const; c6=const; c7=const; c8=const; c9=const; c10=const -->
| 0.10 | `-0.016016 * 0.10 = -0.00160161` | `-0.110074 * 0.10 = -0.01100736` | `-0.204131 * 0.10 = -0.02041312` |
<!-- claims63: c1=const; c2=const; c3=const; c4=const; c5=const; c6=const; c7=const; c8=const; c9=const; c10=const -->
| 0.50 | `-0.016016 * 0.50 = -0.00800803` | `-0.110074 * 0.50 = -0.05503681` | `-0.204131 * 0.50 = -0.10206559` |
<!-- claims63: c1=const; c2=const; c3=const; c4=const; c5=const; c6=const; c7=const; c8=const; c9=const; c10=const -->
| -0.10 (addendum 2026-08-29c) | `-0.016016 * -0.10 = +0.00160161` | `-0.110074 * -0.10 = +0.01100736` | `-0.204131 * -0.10 = +0.02041312` |
<!-- claims63: c1=const; c2=const; c3=const; c4=const; c5=const; c6=const; c7=const; c8=const; c9=const; c10=const -->
| -0.50 (addendum 2026-08-29c) | `-0.016016 * -0.50 = +0.00800803` | `-0.110074 * -0.50 = +0.05503681` | `-0.204131 * -0.50 = +0.10206559` |

**Signed-family cross-check:** the same linear-secant formula
(`Δmean(eps) = -slope*eps`) applied to negative `eps` predicts POSITIVE
`Δmean` (held-out mean INCREASES) with IDENTICAL magnitude to the
corresponding positive dose — `eps=-0.10` mirrors `eps=+0.10`'s magnitude
<!-- claims63: c1=const -->
exactly (`0.01100736` either way, sign flipped), and `eps=-0.50` mirrors
<!-- claims63: c1=const -->
`eps=+0.50`'s magnitude exactly (`0.05503681`). Both measured seed slopes
<!-- claims63: c1=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean); c2=absdiff(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean) -->
are positive (`0.016016`, `0.204131`), so this sign flip is
**sign-consistent across both seeds** — a negative dose is predicted to
DEGRADE held-out loss regardless of which of the two seed slopes turns out
to be closer to the true local derivative. This is the "undertrained-
regression" direction addendum 2026-08-29c names: less effective lr moves
the model back toward its untrained (`s=0`) state, which is measured
<!-- claims63: c1=abs(docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json#/tiers/finetune_run/held_out_example_mean) -->
WORSE (`3.422173`) than the trained (`s=1`) state at both seeds.

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
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/determinism_floor/cross_seed_spread -->
- `determinism_floor.cross_seed_spread = 0.08264997071681932` (measured
  cross-seed spread; zero repeat noise — r1/r2 bit-identical, `max_delta =
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/determinism_floor/max_delta -->
  0.0`, so this spread is the only noise source relevant to whether a fixed
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/threshold; c2=#/decision/gate_seed_count -->
  per-seed shift can flip >=11/12 seeds concordantly).
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/mean_d; c2=#/sign_test/n_neg; c3=#/sign_test/n -->
- `decision.mean_d = -0.023799017749049446`, `sign_test.n_neg = 8/11` (the
  campaign's own REAL, measured fused-vs-alloff effect — and it did NOT
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/threshold; c2=#/decision/gate_seed_count -->
  reach the 11/12 threshold). This is an empirical calibration point: an
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/gate_seed_count; c2=#/sign_test/n_neg; c3=#/sign_test/n -->
  effect of this magnitude, on this same 12-seed gate, produced only 8/11
  concordance, not RED.

<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/determinism_floor/cross_seed_spread; c2=abs(#/decision/mean_d); c3=#/sign_test/n_neg; c4=#/sign_test/n -->
| eps | predicted abs(Δmean) (avg_slope) | vs. cross_seed_spread (0.0826) | vs. already-non-detected effect (0.0238, 8/11) | predicted verdict |
|---|---|---|---|---|
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=const; c2=const; c3=const; c4=const; c5=#/decision/threshold; c6=#/decision/gate_seed_count -->
| 0.02 (falsified, not scheduled) | 0.00220 | 2.7% of spread | 9.2% of the effect that already failed to reach 11/12 | **predicted NOT detected** (very high confidence — an order of magnitude below both benchmarks) |
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=const; c2=const; c3=const; c4=const; c5=#/decision/threshold; c6=#/decision/gate_seed_count -->
| 0.10 (falsified, not scheduled) | 0.01101 | 13.3% of spread | 46.3% of the effect that already failed to reach 11/12 | **predicted NOT detected** (high confidence — still smaller than an effect that already failed) |
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=const; c2=const; c3=ratio(ratio(0.05504, #/determinism_floor/cross_seed_spread), 0.01); c4=ratio(0.05504, abs(#/decision/mean_d)); c5=#/decision/threshold; c6=#/decision/gate_seed_count; c7=rel(ratio(0.05504, abs(#/decision/mean_d)), '>', 2); c8=rel(ratio(0.05504, #/determinism_floor/cross_seed_spread), '<', 1); c9=#/decision/threshold; c10=#/decision/gate_seed_count; c11=const; c12=const -->
| +0.50 (scheduled — two-sided falsification cell) | 0.05504 | 66.6% of spread | 2.31x the effect that already failed to reach 11/12 | **predicted NOT reliably detected**, but this is the closest call: the shift exceeds the already-non-detected effect by >2x, yet stays below 1 full cross-seed-spread-unit, so >=11/12 concordance is not confidently predicted either way. Using the strong-end per-seed slope (`0.204131`) instead of the average, eps=+0.50's predicted shift (`-0.10207`) would EXCEED the cross-seed spread — the two available seeds disagree sharply on whether this dose is detectable at all. If a sign test DOES fire here, the predicted DIRECTION is still improvement (RED-for-investigation, not RED-for-degradation) — that is what makes this the falsification cell for the Step-2 direction prediction, not a degradation candidate. |
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=const; c2=const; c3=const; c4=const; c5=#/decision/threshold; c6=#/decision/gate_seed_count; c7=const; c8=#/decision/threshold; c9=#/decision/gate_seed_count -->
| -0.10 (scheduled — addendum 2026-08-29c) | 0.01101 | 13.3% of spread | 46.3% of the effect that already failed to reach 11/12 | **predicted NOT detected** (same magnitude reasoning as `+0.10`, direction now DEGRADATION — the correct sign for the acceptance-5 claim, but the predicted magnitude is still smaller than an effect that already failed to reach 11/12 on this same gate) |
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=const; c2=const; c3=ratio(ratio(0.05504, #/determinism_floor/cross_seed_spread), 0.01); c4=ratio(0.05504, abs(#/decision/mean_d)); c5=#/decision/threshold; c6=#/decision/gate_seed_count; c7=const; c8=const; c9=const; c10=const; c11=const; c12=const -->
| -0.50 (scheduled — addendum 2026-08-29c) | 0.05504 | 66.6% of spread | 2.31x the effect that already failed to reach 11/12 | **predicted NOT reliably detected, but the best DEGRADATION-direction candidate of the three scheduled doses** — same magnitude/confidence caveats as `+0.50` (strong-end seed slope would push it to `0.10207`, above the cross-seed spread), but now in the direction that could actually discharge acceptance 5's "mutant column proven RED (degradation)" if the true local derivative tracks closer to the strong-end (seed1) slope than the weak-end (seed2) one. **This is the dose most likely, among the three scheduled, to produce the pair straddling detection that amendment 2026-08-29c's "reported sensitivity" clause asks for** — though the adjacent pair straddling detection, if one exists, is expected between `-0.50` and `-0.10` within the negative branch, per the addendum's own framing, not between `-0.50` and `+0.50`. |

**Design-level flag, resolved by addendum 2026-08-29c:** the original
positive-only ladder's predicted direction (Step 2) was IMPROVEMENT under
the naive linear-extrapolation model, so even a dose that DID reach
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/threshold; c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
`>=11/12+mean` would have read RED-for-improvement under the Frame's
two-sided rule, not RED-for-degradation — unable to discharge amendment
2026-08-29b item 3's acceptance criterion, which is scoped to the
DEGRADATION direction (mirroring M1's pass criterion). This is exactly the
`(1-eps)`-shaped revisit flagged in the prior revision of this section:
<!-- claims63: c1=const; c2=const -->
CONTRACT.md addendum 2026-08-29c signs the family (`eps in {-0.50, -0.10,
<!-- claims63: c1=const -->
+0.50}`) so the negative doses carry the predicted DEGRADATION direction
(same secant, sign-consistent across both measured seeds — see the
<!-- claims63: c1=const -->
signed-family cross-check above), while `+0.50` is RETAINED, not dropped,
as the two-sided falsification cell for the original improvement
prediction itself. No family redesign beyond signing `eps` was needed; the
`(1+eps)` shape is unchanged, only which values of `eps` are scheduled to
run changed.

No dose is refused for lack of data — all five doses (falsified and
scheduled) are supported by the same committed two-seed secant; the
confidence intervals above (not point estimates) are the honest product of
what is actually committed.

## On-pod procedure (dose legs only — substituted INTO the fused arm)

<!-- claims63: c1=const -->
**Scheduled ladder (CONTRACT.md addendum 2026-08-29c): `eps in {-0.50,
<!-- claims63: c1=const; c2=const; c3=const; c4=const -->
-0.10, +0.50}`.** `eps in {0.02, 0.10}` are falsified pre-spend (Step 2/3
above) and are NOT run on the pod; their patches stay committed as the
falsification record only.

The mutant-vs-fused pairing used by M1 is explicitly RETIRED. Per amendment
2026-08-29b item 3 (merge procedure) and addendum 2026-08-29c (signed
ladder), each dose column is produced by substituting the mutant into the
fused arm itself and merging against the SAME v2 `alloff` legs under the
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/threshold; c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
SAME `>=11/12+mean` rule — the gate's own statistic, not a separate ad hoc
comparison.

1. On the pod, clone/checkout at the recorded base sha into a **scratch
   worktree**, separate from the campaign's production checkout (either
   `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` or
   `cba0b835` — `adamw_step.rs` is byte-identical between them):
   ```sh
   git worktree add /workspace/scratch-mutant-dose cba0b835 --detach
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
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
4. Run the dose's 12 legs through the same `run_leg` vector the campaign's
   fused arm uses (same shapes, same seeds, same step count, same
   `attention_block_flash` + `adamw_step_fused` dispatch wiring) — the ONLY
   difference between a dose leg and a clean fused leg is that this scratch
   build's `jammi-kernels` binary was compiled from the patched source. This
   dose leg SUBSTITUTES for the fused arm's leg at that seed — it is not a
   third, separately-merged column.
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
5. Merge each dose's 12 substituted-fused legs against the SAME campaign
   `alloff` legs already on record (identity permitting, per H5(1)'s
   ALLOFF-reuse rule) using the gate's own merger and its own
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/threshold; c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
   `>=11/12+mean` sign-test statistic — the same code path the real A/B
   verdict uses, not a bespoke mutant-vs-fused comparison.
6. Record each dose leg's own provenance via the **three producer-stamped
   CLI flags**, passed to that leg's `run_leg` invocation itself — this is
   the ONLY mechanism by which a mutant leg's provenance is recorded; no
   artifact is ever hand-edited to add or correct these fields, ever:
   - `--mutant-id <M_eps_...>` — this dose's own patch identifier.
   - `--mutant-base-sha <base sha>` — the scratch checkout's base sha
     (`ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` or `cba0b835`).
   - `--mutant-patch-sha256 <sha256>` — the dose's recorded sha256 above.

   These stamp the leg's own `FinetuneRunTier` report with
   `mutant_id`/`mutant_base_sha`/`mutant_patch_sha256` (serde-skipped when
   absent — a non-mutant leg carries none of the three), which
   `ab_merge.finetune_run_mutant_column_violations` then checks are present
   and that `mutant_patch_sha256` agrees with the dose column's own
   caller-supplied `patch_sha256`. Separately, choose:
   - `dose_label = eps-0.50` / `eps-0.10` / `eps0.50` (operator-chosen
     string per the merger's `--mutant-legs DOSE_LABEL:PATCH_SHA256:SEEDS`
     convention — never `eps0.02`/`eps0.10`, which are not scheduled)

   Every other field a clean fused leg already records (held-out-loss /
   step-count / dispatch-counter fields) is recorded exactly as it always
   is, so each dose column is diff-able against the `alloff`/fused columns
   field-for-field.
7. The reported sensitivity is the pair of ADJACENT doses straddling
   detection WITHIN THE DEGRADATION-DIRECTION (negative-`eps`) BRANCH ONLY,
   ordered by `abs(eps)` (per amendment item 3 and addendum 2026-08-29c,
   unit-63 round-7 audit finding 4) — run the scheduled ladder in ascending
<!-- claims63: c1=const; c2=const; c3=const -->
   SIGNED `eps` order (`-0.50`, `-0.10`, `+0.50`) and stop describing the
   sweep as "complete" only once a straddling pair is found or all three
   have been run without one. The merger's own `abs(eps)`-ordered scan of
<!-- claims63: c1=const; c2=const; c3=const; c4=const -->
   the negative branch (`-0.10` -> `-0.50`, since `|{-0.10}| < |{-0.50}|`)
<!-- claims63: c1=const -->
   is deliberately NOT the same order the legs are RUN in (`-0.50` run
<!-- claims63: c1=const -->
   before `-0.10`) — reporting the straddle in run order would either miss
<!-- claims63: c1=const -->
   a real straddle (a detection at the larger-magnitude `-0.50`, run first,
   reads `(RED, not-detected)` in run order, not the `(not-detected, RED)`
   shape a straddle needs) or, worse, misreport a cross-sign
<!-- claims63: c1=const; c2=const -->
   `(-0.10 not-detected, +0.50 RED)` run-order-adjacent pair as though it
<!-- claims63: c1=const -->
   were a degradation-direction finding. `+0.50` reading RED is instead the
   two-sided-falsification finding — and it REFUTES, not confirms, the
   Step-2 improvement prediction (`"RED"` is always the DEGRADATION-
   concordant arm; more effective lr made held-out loss worse, so the
<!-- claims63: c1=const -->
   secant extrapolation was wrong). Only a `+0.50` dose reading
   `RED_FOR_INVESTIGATION`-shaped improvement CONFIRMS the prediction
   (unit-63 round-9 audit finding 1, correcting round-8 finding 1's own
   inverted-polarity phrasing, which survived here). Either outcome is
   reported separately, never folded into sensitivity.
8. Tear down the scratch worktree and its build artifacts after the legs
   complete; do not leave a patched binary or scratch checkout on the pod
   past the dose-leg run. Each patch is committed to this repo as a FILE
   only — it must never land as tree state on any branch that builds the
   production `jammi-kernels` binary.

## Pass criterion (per dose)

Exactly as M1's, restated for the retired-pairing correction: **a sign test
over the dose's substituted-fused column's held-out loss vs. the campaign's
`alloff` legs' held-out loss reads RED (degradation) at the campaign's
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/threshold; c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
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
<!-- claims63: c1=const -->
file's own correct `eager_step` oracle (unpatched formula) for 5
consecutive steps on the same fixed 4-element input M1 used
(`beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=0.01, eps=1e-8`), printing
the per-step L2 divergence in `theta` (`cargo test -p jammi-kernels --lib
mutant_dose_demonstration -- --nocapture`, patch applied, CPU only):

```
eps=0.02 (falsified, not scheduled):
step=1 l2_divergence=3.508513e-5
step=2 l2_divergence=7.017026e-5
step=3 l2_divergence=1.0525539e-4
step=4 l2_divergence=1.4034052e-4
step=5 l2_divergence=1.7542565e-4

eps=0.10 (falsified, not scheduled):
step=1 l2_divergence=1.7608417e-4
step=2 l2_divergence=3.5216834e-4
step=3 l2_divergence=5.28184e-4
step=4 l2_divergence=7.0419966e-4
step=5 l2_divergence=8.801983e-4

eps=0.50 (scheduled — two-sided falsification cell):
step=1 l2_divergence=8.7980856e-4
step=2 l2_divergence=1.7595316e-3
step=3 l2_divergence=2.6392546e-3
step=4 l2_divergence=3.5189774e-3
step=5 l2_divergence=4.3986836e-3

eps=-0.10 (scheduled — addendum 2026-08-29c):
step=1 l2_divergence=1.7599607e-4
step=2 l2_divergence=3.5199214e-4
step=3 l2_divergence=5.279882e-4
step=4 l2_divergence=7.039843e-4
step=5 l2_divergence=8.799804e-4

eps=-0.50 (scheduled — addendum 2026-08-29c):
step=1 l2_divergence=8.7986013e-4
step=2 l2_divergence=1.7597203e-3
step=3 l2_divergence=2.6395635e-3
step=4 l2_divergence=3.5194065e-3
step=5 l2_divergence=4.39925e-3
```

The step-5 divergence scales linearly with `|eps|` as the design intends
<!-- claims63: c1=code('0.10', 5); c2=code('0.02', 5); c3=ratio(code('0.10', 5), code('0.02', 5)) -->
(sustained, monotone family): `8.801983e-4 / 1.7542565e-4 = 5.018` (expected
<!-- claims63: c1=ratio(0.10, 0.02); c2=const; c3=const; c4=const; c5=code('0.50', 5); c6=code('0.02', 5); c7=ratio(code('0.50', 5), code('0.02', 5)) -->
`5x`, since `0.10 / 0.02 = 5`); `4.3986836e-3 / 1.7542565e-4 = 25.074`
<!-- claims63: c1=ratio(0.50, 0.02); c2=const; c3=const; c4=const -->
(expected `25x`, since `0.50 / 0.02 = 25`) — both within floating-point
accumulation tolerance of the exact ratio, confirming each dose is a real,
proportionally-scaled numeric perturbation on this crate's own oracle, not
a no-op and not a copy-paste of a different dose's constant. The two new
signed doses confirm the same magnitude relationship holds for negative
<!-- claims63: c1=code('-0.10', 5) -->
`eps`: `eps=-0.10`'s step-5 divergence (`8.799804e-4`) matches `eps=+0.10`'s
<!-- claims63: c1=code('0.10', 5); c2=absdiff(code('-0.10', 5), code('0.10', 5)) -->
(`8.801983e-4`) to within `~2.2e-7` (a floating-point accumulation-order
<!-- claims63: c1=const; c2=const -->
difference between `1.10*lr` and `0.90*lr`, not a design defect), and
<!-- claims63: c1=code('-0.50', 5); c2=code('0.50', 5) -->
`eps=-0.50`'s (`4.39925e-3`) matches `eps=+0.50`'s (`4.3986836e-3`) to
<!-- claims63: c1=absdiff(code('-0.50', 5), code('0.50', 5)) -->
within `~6e-7` — confirming `|1+eps|`-driven magnitude symmetry of the
perturbation at the kernel level, independent of sign.

`git checkout -- crates/jammi-kernels/src/ops/adamw_step.rs` restored the
clean tree after each dose; neither the patches nor the temporary test are
part of any commit on this branch (all five patches ARE committed, as
files only, alongside this README).

## Dose ladder (CONTRACT amendment 2026-08-29b item 3; addendum 2026-08-29c signs it) — merger interface

M1 above is recorded as a NON-DETECTION (a sign-flipping early transient,
never a sensitivity bound — see the campaign-v1 finding this amendment
corrects). The corrected design is a one-parameter, monotone, SUSTAINED
dose family — the fused AdamW update scaled by `(1+eps)`. Amendment
<!-- claims63: c1=const; c2=const; c3=const -->
2026-08-29b item 3 originally named `eps in {0.02, 0.10, 0.50}`
(positive-only); this README's own required pre-spend prediction table
falsified that direction before any spend (Step 2/3 above: predicted
IMPROVEMENT, not degradation, on both measured seeds). CONTRACT.md
addendum 2026-08-29c resolves this by SIGNING the family:
<!-- claims63: c1=const; c2=const; c3=const -->
**the scheduled ladder is now `eps in {-0.50, -0.10, +0.50}`** —
<!-- claims63: c1=const; c2=const -->
`eps in {0.02, 0.10}` remain committed as the falsified-but-recorded
doses, not scheduled to run. Each scheduled dose runs as its own
scratch-worktree mutant (the SAME on-pod procedure above: patched
`jammi-kernels`, the SAME `run_leg` vector, torn down after). This
section documents ONLY the merger's own consuming interface
(`ci/scripts/perf/ab_merge.py`); the per-dose PREDICTED effect (held-out
example-mean units, stated BEFORE the spend, per the amendment's own
falsifiability requirement) is the Step 1-3 prediction table above, not a
separate record.

- **Leg naming**: a dose's legs are recorded under the SAME `raw_dir` a
  campaign already uses, tagged `seed{seed}__fused__mutant-{dose_label}`
  (`ab_merge.mutant_leg_repeat_tag(dose_label)`) — a `repeat` value that can
  never collide with `r1`/`r2` (the main A/B pool) or `lr0` (the RED
  control), the SAME file-naming isolation the lr0 control already relies
  on. `dose_label` is an operator-chosen string from the scheduled set
  (e.g. `"eps-0.50"`), but it IS reinterpreted by the merger — parsed as a
  signed `eps` float (`ab_merge._dose_label_eps`) to place the column in
  the degradation/improvement branch and (unit-63 round-8 audit finding 3;
  round-9 audit finding 2 makes the domain ASYMMETRIC) validated finite,
  non-zero, and within this family's own sane domain — `eps in
  (MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND, -MUTANT_DOSE_LADDER_MIN_ABS_EPS]
  union [MUTANT_DOSE_LADDER_MIN_ABS_EPS, MUTANT_DOSE_LADDER_MAX_EPS]`
  (unit-63 round-10 audit advisory (b): `MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND`
<!-- claims63: c1=const; c2=const -->
  is `-1.0`, named rather than a bare literal so the `(1+eps)==0` rationale
  below has one place to live; `MUTANT_DOSE_LADDER_MAX_EPS` was renamed
  from `..._MAX_ABS_EPS` since every call site compares the SIGNED value
  directly, never `abs()` — a one-sided positive-branch ceiling, not a
  magnitude cap). `eps == MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND`
  itself is refused, EXCLUSIVE: the update-scale multiplier
  `(1+eps)` is zero there, a zero-update leg, not a member of this
  family — a single symmetric `|eps| <= MAX` check would have let it
  through; a magnitude below `MUTANT_DOSE_LADDER_MIN_ABS_EPS` is refused
  as below this family's own sanity floor, set deliberately BELOW the
<!-- claims63: c1=const -->
  smallest ever-SCHEDULED dose (`|eps| = 0.10`) so a genuine sub-schedule
  diagnostic dose (e.g. `eps=0.02` above) is still admitted while a
  manufactured near-zero eps is refused); an unparseable, non-finite,
  zero, or out-of-domain label is refused loudly, never silently
  passed through as an opaque tag.
- **Per-leg recorded fields**: every field a clean `fused` leg already
  carries, PLUS this section's own three producer-stamped fields
  (`mutant_id`/`mutant_base_sha`/`mutant_patch_sha256`, serde-skipped when
  absent — the on-pod procedure's own step 6 `--mutant-id`/
  `--mutant-base-sha`/`--mutant-patch-sha256` CLI flags, never hand-edited
  into the artifact) — a mutant leg missing any of the three, or whose own
  `mutant_patch_sha256` disagrees with the dose column it is merged under,
  is refused (`ab_merge.finetune_run_mutant_column_violations`).
- **Merger CLI**: `ab_merge.py finetune-run RAW_DIR OUT_DIR SEEDS
  [LR0_SEEDS] [--allow-missing-lr0-control] [--mutant-legs
  DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,...]` — `--mutant-legs` is repeatable,
  once per dose column; `finetune_run_ab.sh`'s own
  `FINETUNE_RUN_AB_MUTANT_LEGS` env var (`;`-separated specs) is a pure
  pass-through into the same flag, never a second leg-running mechanism.
- **Per-dose output** (`mutant_dose_ladder.doses[i]` in the merged
  artifact): `{dose_label, patch_sha256, detected, n_pos, n_neg, mean_d,
  p_value, clean_pair_count, violations, ...}` — `detected` is `"RED"` iff
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/threshold; c2=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
  the SAME `>=11/12` threshold the primary decision uses is met in the
  DEGRADATION direction (mutant worse than alloff); `"RED_FOR_INVESTIGATION"`
  (unit-63 round-8 audit finding 2) iff the SAME threshold is met in the
  OPPOSITE, IMPROVEMENT-concordant direction instead (mirrors the primary
  decision's own `RED_FOR_INVESTIGATION` state) — this is the confirming
<!-- claims63: c1=const -->
  outcome the `+0.50` two-sided-falsification cell needs to be able to
  report; before this fix the column had no state for this arm and it
  collapsed into `"not-detected"`, so the confirming outcome could never be
  reported. `"not-detected"` if NEITHER threshold is met at all (M1's own
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
  sign-flipping-transient shape: 8/12, well under either threshold);
  `"INVALID"` if the premise-clean pair count is not exactly the
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
  pre-registered 12, or the sign test itself refuses — a
  correctness-of-measurement carve-out beyond the amendment's own literal
  `RED`/`not-detected` pair, added so a malformed dose column is never
  silently read as "not-detected" (a substantive finding) when it is really
  "this column could not be evaluated at all".
- **Sensitivity statement** (unit-63 round-7 audit finding 4, addendum
  2026-08-29c): `mutant_dose_ladder.sensitivity` — the first adjacent
  `(not-detected, RED)` pair WITHIN THE DEGRADATION-DIRECTION (`eps < 0`)
  BRANCH ONLY, ordered by `abs(eps)` (each dose's SIGNED eps parsed from its
  own `dose_label`, `ab_merge._dose_label_eps`; a label that fails to parse
  is a merge-level refusal, `mutant_dose_ladder.sensitivity_error`, never a
  silent skip) — never the caller-supplied/run order, and never a
  cross-sign pair. Returns `null` if no such transition exists in that
  branch. `sensitivity` is scoped to the degradation branch ONLY —
  `"RED_FOR_INVESTIGATION"` never enters it, by construction (the straddle
  predicate matches the literal string `"RED"` only). This is true
  REGARDLESS of eps sign — `"RED_FOR_INVESTIGATION"` is NOT restricted to
  positive-eps doses (unit-63 round-9 audit finding 3 corrects that prior
  claim here): a NEGATIVE-eps dose can read `"RED_FOR_INVESTIGATION"` too
  (an anomalous improvement detected under DEFLATION, the opposite of that
  branch's own predicted degradation direction) — never folded into
  `sensitivity` either way, but reported separately under
  `mutant_dose_ladder.dose_anomalies` (see below) rather than silently
  dropped.
- **Dose anomalies** (unit-63 round-9 audit finding 3):
  `mutant_dose_ladder.dose_anomalies` — every NEGATIVE-eps (`eps < 0`) dose
  column whose `detected` is `"RED_FOR_INVESTIGATION"`
  (`ab_merge.mutant_dose_ladder_anomalies`), each recorded as
  `{dose_label, eps, detected, finding: "anomalous improvement under
  deflation (eps < 0)"}`. A NON-EMPTY list gates this merge's own exit
  code exactly as the primary decision's own `RED_FOR_INVESTIGATION`
  state does (`ab_merge.main`'s own three-outcome gate) — "investigated,
  never silently celebrated" applies here too, not just to the primary
  decision.
  A POSITIVE-eps dose reading `"RED_FOR_INVESTIGATION"` is NEVER a member
  of this list; that is the ORDINARY, PREDICTED two-sided-falsification
  confirming arm instead (see below), not an anomaly.
  A positive-eps (`eps > 0`) dose reading `"RED"` OR `"RED_FOR_INVESTIGATION"`
  is reported separately, under `mutant_dose_ladder.two_sided_falsification`
  (unit-63 round-8 audit finding 2: BOTH arms are reported there, with the
  correct polarity — `"RED"` is `"secant refuted (degradation at +eps)"`,
  `"RED_FOR_INVESTIGATION"` is `"secant confirmed (improvement at +eps)"` —
  never a positive-eps `"RED"` described as "confirming" the improvement
  prediction; that inverted-polarity phrasing was unit-63 round-8 audit
  finding 1 — round-8 itself missed two survivors (this file's own on-pod
  procedure step 7, and `ab_merge.mutant_dose_ladder_sensitivity`'s own
  docstring), corrected by unit-63 round-9 audit finding 1, confirmed
  corrected everywhere it appears by that round's own
<!-- claims63: c1=ledger -->
  `grep -rn -i 'confirm|refut'` / `'reading RED|reads RED|\+0.50'` sweep over
  `ab_merge.py`, `test_ab_merge.py`, this file, and `CONTRACT.md` — a
  completeness claim re-established by sweep each round it is touched,
  never merely asserted), never folded into
  `sensitivity` (`ab_merge.mutant_dose_ladder_sensitivity` /
  `ab_merge.mutant_dose_ladder_two_sided_falsification`).
- **Mutant legs never enter the primary A/B set**: proven structurally (the
  `mutant-<dose_label>` repeat tag can never equal `r1`/`r2`/`lr0`) and
  empirically (`MutantDoseLadderTests.test_mutant_leg_never_leaks_into_the_ab_set`,
  `ci/scripts/perf/test_ab_merge.py`).

## RED-proof mutants (outside the lr-scale family)

**Why the `(1+eps)` lr-scale family cannot supply acceptance 5's RED.** The
signed dose ladder (`measurements/dose-ladder/README.md`, unit 63 addendum
<!-- claims63: c1=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json#/mutant_dose_ladder/doses/0/n_neg; c2=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json#/mutant_dose_ladder/doses/0/gate_seed_count -->
2026-08-29c, run at base `494fb3e3`) DEMONSTRATED the detector — 11/12
<!-- claims63: c1=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json#/mutant_dose_ladder/doses/0/p_value -->
sign-concordance at `p=0.00635 < alpha2` on a real effect — but landed every
<!-- claims63: c1=const; c2=const; c3=const -->
scheduled dose (`eps in {-0.50, -0.10, +0.50}`) in the IMPROVEMENT direction:
`eps-0.50` and `eps-0.10` both read `RED_FOR_INVESTIGATION` (anomalous
improvement under deflation, `mean_d` negative both times), and `eps0.50`
read not-detected. Per that README's own finding 3: "no degradation-RED
among the (1+eps) family: the lr-scale knob cannot produce a degradation
demonstration near this operating point." The lr surface near the
campaign's `lr=2e-4` operating point is favorable in every tested
<!-- claims63: c1=ratio(0.50, 0.01) -->
direction (inflating OR deflating effective lr by up to 50% improves or is
neutral to held-out loss) — this is a genuine measured property of THIS
fixture's loss surface near THIS lr, not a detector-sensitivity gap, and no
member of the `(1+eps)` family (however far `eps` is pushed within its
`ab_merge.py`-validated domain) is going to flip that. Acceptance 5's
"mutant column proven RED" therefore needs a mutant OUTSIDE the lr-scale
shape — a defect that is not merely "the same update, rescaled" but a
qualitatively different corruption of the fused update.

Two mutants were originally pinned here (`M_nobc`, `M_signflip`); a third
(`M_signflip_v2`) supersedes `M_signflip` after its own GPU measurement
(see that section's "Measured"/lesson notes below). All three are against
base `e340391c` (`M_signflip_v2.patch` is cut against `74fd69ef` — same
byte-identical `adamw_step.rs`, blob `58a1418c`), all patch-file-only
(never applied to tree state on this branch), all touching ONLY
`adamw_step_fused_t`'s fused path inside
`crates/jammi-kernels/src/ops/adamw_step.rs` (no dispatch/identity surface
touched — `InplaceOp2::name()`/`InplaceOp3::name()`, `validate_step_domain`,
and every `DispatchCounters`/`admission` field read identical to a clean
fused leg, exactly as the `(1+eps)` family's own isolation note above
requires):

### `M_nobc` — bias correction removed entirely

<!-- claims63: c1=const; c2=const -->
**Definition:** `adamw_step_fused_t` pins `scale_m = 1.0`, `scale_v = 1.0`
<!-- claims63: c1=const; c2=const; c3=const; c4=const -->
instead of deriving them from `t` (`1/(1-beta1^t)`, `1/(1-beta2^t)`) — the
fused update runs on RAW (uncorrected) `m`/`v` moment estimates. Models a
realistic silent regression: a kernel that skips bias correction while
still computing the right EMA moments and the right theta-update
arithmetic otherwise. This is a SUSTAINED family-outside perturbation —
unlike M1's `t+1` off-by-one (a sign-flipping early transient that decays
<!-- claims63: c1=ledger -->
toward a `~1.009` multiplier by `t=50`, `measurements/campaign-v1/`'s own
<!-- claims63: c1=ledger -->
finding), the uncorrected-scale multiplier is monotone-decaying-toward-1.0
<!-- claims63: c1=const; c2=const -->
from a LARGE start (`scale_m` ratio to the correct value is `1/(1-beta1^t)`,
<!-- claims63: c1=ledger; c2=ledger -->
`~10x` at `t=1` for `beta1=0.9`; `scale_v`'s ratio is `~1000x` at `t=1` for
`beta2=0.999`) — a one-directional (never sign-flipping) effective-lr
<!-- claims63: c1=ledger; c2=ledger -->
blowup, matching the "sustained 3.2-6.5x effective-lr blowup" the earlier
pressure-test (CONTRACT.md amendment 2026-08-29b item 3's own root-cause
analysis of M1) computed over a realistic step range.

**Predicted direction: UNCERTAIN, stated honestly.** The measured
`(1+eps)` lr-scale data gives no clean extrapolation to this mutant's
<!-- claims63: c1=const; c2=const -->
magnitude: `eps=+-0.10` (a 1.10x/0.90x effective-lr multiplier) read
<!-- claims63: c1=ledger; c2=ledger -->
neutral-to-improving on this fixture, but `M_nobc`'s blowup is 3-6x LARGER
than anything the ladder measured — well outside the range the ladder's own
secant (measurements/dose-ladder's own README, and the earlier prediction
<!-- claims63: c1=ledger; c2=ledger -->
table above) can vouch for. A 3-6x effective-lr blowup at early steps
plausibly destabilizes training (the "too-large-lr degradation regime" the
prediction table above names as the mechanism that could reverse the
secant's sign) — but it is equally plausible that a large early-step
transient (m/v are zero-initialized, so v_hat's `eps` floor and the
`sqrt(v_hat)` denominator both damp the blowup's practical effect on
`adjusted_grad`) washes out over a full training run the way M1's transient
did. No claim of degradation is made for `M_nobc` before it is actually
run; this is the honest complement to `M_signflip` below.

<!-- claims63: c1=ledger -->
**Measured (12-leg GPU, a100, `redproof-nobc`): NOT DETECTED (raw).**
Committed artifact: `docs/plans/63-how-well/measurements/red-proof/` at
<!-- claims63: c1=ledger -->
dc1cfc3b (36 raw legs + the gated merge artifact). `M_nobc`
<!-- claims63: c1=zerocount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
IS a genuine perturbation on hardware — 0/12 legs bit-identical to the
clean fused column (unlike `M_signflip` v1 below, this mutant fires on the
CUDA arm exactly as designed, since it changes a Rust-side scalar
(`scale_m`/`scale_v`) fed into `AdamThetaUpdate`, a struct field both
`cpu_fwd` and `cuda_fwd` read identically — see `M_signflip_v2`'s own
"dispatch-invariant site" framing below, which this mutant already
<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
satisfied by construction). The RAW 12-pair concordance (all 12 mutant/alloff
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c3=meand('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
pairs, premise violations included) reads `n_pos=5/12`, `mean_d=-0.018` —
<!-- claims63: c1=const; c2=const -->
**not-detected** under the `>=11/12+mean` rule (the uncertain prediction
above is BORNE OUT as the neutral/mixed outcome, not the degradation one):
the no-bias-correction shape does not degrade held-out loss at this
operating point. This is an honest, real, non-vacuous non-detection
(genuinely perturbed hardware legs, correctly measured, correctly failing to
reach the threshold) — never suppressed or silently retried with a larger
dose, per this file's own "Pass criterion" section above.

**GATED reading (CONTRACT.md amendment 2026-08-29e, D*): INVALID.** Under
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=absdiff(paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'), #/mutant_dose_ladder/red_proof/0/clean_pair_count); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
the D*-decomposed learning-happened premise, 2 of `M_nobc`'s 12 mutant legs
— seeds 9 and 12 — show ASCENDING probes against `M_nobc`'s own declared
DESCENT direction (`RED_PROOF_EXPECTED_TRAIN_DIRECTION`'s own `9b3c824d…`
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
entry), so `clean_pair_count=10 ≠ 12` and the gated column reads INVALID
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
(10/12 clean pairs, correctness-of-measurement problem, never silently
rescaled to whatever count ran clean). This is a SEPARATE reading from the
RAW not-detected number above, at a different layer — both are current
truth at their own layers (CONTRACT.md's own RECONCILIATION clause); under
D* the `M_nobc` column remains INVALID — recorded, not rescued. Acceptance
<!-- claims63: c1=const -->
5's discharge schedules `redproof-signflip-v2` only (see "Scheduling
`M_signflip_v2`" below); `M_nobc`'s own committed INVALID record stands as
evidence, not as a second column to re-run.

### `M_signflip` — inverted-sign update (gradient ascent) — SUPERSEDED, honest inert-on-GPU record

**Definition:** `AdamThetaUpdate::cpu_fwd`'s final line is changed from
`theta[it] = theta_scaled - adj_scaled` to
`theta[it] = theta_scaled + adj_scaled` — the fused update applied with the
sign of the adjusted-gradient term inverted. Every other computation
(moment EMAs, bias correction, `denom`, `adjusted_grad`) is byte-identical
to the correct kernel; only the final combine's sign flips. Models the
sign-error regression class: a kernel that adds instead of subtracts the
update (e.g. a `-=`/`+=` typo, or a sign lost in a refactor of the
theta-update expression).

**Predicted direction: DEGRADATION, with CERTAINTY.** Mechanism: for any
nonzero gradient, `adjusted_grad` points in the direction that DECREASES
loss (that is what a gradient step is); adding it instead of subtracting it
is gradient ASCENT on that same direction — theta is driven to INCREASE
loss every single step, compounding for the length of the run. This
prediction requires no secant extrapolation from the measured lr data at
all (unlike `M_nobc`) — it follows directly from what a gradient step IS,
independent of this fixture's particular loss-surface geometry near
`lr=2e-4`. `M_signflip` is therefore the guaranteed RED-proof member of
this pair: if `M_nobc` reads neutral-or-improving (plausible per its own
uncertain prediction above), `M_signflip` still discharges acceptance 5's
"mutant column proven RED" on its own.

<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=zerocount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean'); c3=paircount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
**Measured (12-leg GPU, a100, `redproof-signflip`): INERT — 12/12 legs
bit-identical to the clean fused column. This patch is retired; see
`M_signflip_v2` below.** Committed artifact:
`docs/plans/63-how-well/measurements/red-proof/` at dc1cfc3b (the same
<!-- claims63: c1=ledger -->
36-raw-leg + gated-merge artifact `M_nobc`'s own "Measured" record above
cites; these v1 `signflip` legs are committed there as evidence only, never
scheduled through the merger). The lead's own bit-identity check on the raw legs
caught this before a false "gradient ascent doesn't degrade" conclusion
could form from an apparent not-detected/neutral sign-test result. Root
cause: this patch edits `AdamThetaUpdate::cpu_fwd`'s Rust body only, but
the campaign's fused arm dispatches the CUDA kernel on a100 —
`AdamThetaUpdate::cuda_fwd` calls `crate::cuda::adamw_step::
theta_update_cuda_fwd`, which runs its own compiled `cuda/adamw_step.cu`
PTX, never `cpu_fwd`'s Rust body. The CPU-side demonstration above
<!-- claims63: c1=code('M_signflip', 1); c2=code('M_signflip', 5) -->
(`3.464057e-3`..`1.7320165e-2` L2 divergence) is real and correctly shows
`cpu_fwd` diverging from the oracle — the demonstration procedure itself
worked exactly as designed — but it demonstrates a perturbation that never
reaches the arm the 12-leg GPU gate actually exercises. This is the
opposite failure mode from a premise violation (the legs were premise-CLEAN
— dispatch/admission fields read identical to a clean fused leg, exactly as
"no dispatch/identity surface touched" promised above) — the arithmetic
perturbation itself simply never executed on the measured arm.

**The lesson, stated plainly (current-truth discipline): a kernel mutant
patch must perturb a DISPATCH-INVARIANT site, or be verified per-arm
before scheduling.** `AdamThetaUpdate`'s two forward methods
(`cpu_fwd`/`cuda_fwd`) are separate implementations behind one trait, each
compiled into its own arm's code path — editing one never touches the
other. The `(1+eps)` lr-scale family (and `M_nobc`, which edits
`scale_m`/`scale_v` in `adamw_step_fused_t` itself, upstream of BOTH
`cpu_fwd`/`cuda_fwd`) already proved the SAFE pattern on hardware: perturb
a plain Rust scalar/struct field constructed in `adamw_step_fused_t`
BEFORE the `InplaceOp3::cpu_fwd`/`cuda_fwd` split, so both arms read the
SAME already-perturbed value — never a per-backend `cpu_fwd`/`cuda_fwd`
body edit without an explicit, separate check (or, absent CUDA hardware to
check against, a same-shape design note citing the precedent) that the
CUDA arm's own independent implementation carries the equivalent change.
`M_signflip_v2` (below) applies this lesson: it moves the sign flip to
`adamw_step_fused_t`'s own `lr` scalar, the exact site the `(1+eps)`
family already proved reaches CUDA on a100.

**CPU demonstration (not committed, per the dose ladder's own procedure
above):** the same temporary
`mutant_dose_demonstration_diverges_from_the_correct_oracle` test, run once
per mutant (patch applied, test added, `cargo test -p jammi-kernels --lib
mutant_dose_demonstration -- --nocapture`, both patch and test reverted via
`git checkout --` before moving to the next mutant), on the SAME fixed
4-element input the dose ladder used (`theta=[0.5,-1.25,3.0,0.0]`,
`g=[0.1,-0.2,0.05,0.0]`, `beta1=0.9, beta2=0.999, lr=1e-3,
weight_decay=0.01, eps=1e-8`), 5 consecutive steps:

```
M_nobc:
step=1 l2_divergence=3.7451058e-3
step=2 l2_divergence=9.373536e-3
step=3 l2_divergence=1.6215533e-2
step=4 l2_divergence=2.3908453e-2
step=5 l2_divergence=3.221707e-2

M_signflip:
step=1 l2_divergence=3.464057e-3
step=2 l2_divergence=6.928114e-3
step=3 l2_divergence=1.0392154e-2
step=4 l2_divergence=1.38561595e-2
step=5 l2_divergence=1.7320165e-2
```

Both mutants are real, non-trivial, growing perturbations against the
file's own `eager_step` oracle (neither a no-op nor a copy of the other's
constant) — `M_nobc`'s divergence is dominated by the early-step bias-
correction blowup (largest relative effect at small `t`, per its own
definition above); `M_signflip`'s divergence is dominated by the
compounding sign error (every step moves theta the wrong way, so the
divergence from the correct trajectory grows every step regardless of
`t`). Neither ratio is claimed to `x`-scale linearly the way the signed
`(1+eps)` family's dose-to-dose ratios do — these are NOT members of that
one-parameter family, so no such linear relationship is predicted or
claimed here.

### `M_signflip_v2` — inverted-sign update at a dispatch-invariant site (replaces `M_signflip`)

**Definition:** `adamw_step_fused_t` negates its own local `lr` scalar
(`let lr_signflip = -params.lr;`) and passes `lr_signflip` — not
`params.lr` — into `AdamThetaUpdate::new(...)`. `AdamThetaUpdate::new`
<!-- claims63: c1=const -->
bakes this value into BOTH `one_minus_lr_lambda` (`1.0 - lr*weight_decay`,
computed once at construction) and its own stored `lr` field, and BOTH
`AdamThetaUpdate::cpu_fwd` and `AdamThetaUpdate::cuda_fwd` read those SAME
struct fields — this is the identical shape (a plain Rust scalar perturbed
in `adamw_step_fused_t`, upstream of the CPU/CUDA fork) the `(1+eps)`
lr-scale family already used, and that family's own scheduled doses
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/n_neg; c2=#/mutant_dose_ladder/doses/0/gate_seed_count -->
(`eps-0.50`/`eps-0.10`) were measured firing on the a100 GPU legs (11/12
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/p_value -->
concordance, `p=0.00635`) — i.e. this exact site is PROVEN dispatch-
invariant on hardware, not merely argued from source structure. Unlike
`M_signflip` v1, this patch touches ZERO lines inside either `cpu_fwd` or
`cuda_fwd` — the sign flip happens entirely in `adamw_step_fused_t`,
before the `InplaceOp3` call.

**Predicted direction: DEGRADATION, with CERTAINTY — same mechanism as
`M_signflip` v1.** Negating `lr` negates the update the SAME way `-=`
<!-- claims63: c1=const -->
becoming `+=` did (`adj_scaled = adjusted_grad * lr_signflip + 0.0f32`
inside `cpu_fwd`/the CUDA kernel is now the correct term with the WRONG
sign folded in upstream, rather than the sign flipped at the final
combine) — gradient ASCENT on `adjusted_grad`'s direction, every step,
compounding for the length of the run. This mutant ALSO flips the sign of
<!-- claims63: c1=const -->
the `weight_decay` contribution (`one_minus_lr_lambda = 1.0 -
lr_signflip*weight_decay = 1.0 + params.lr*weight_decay`, since
`lr_signflip` is negative) — theta's own magnitude grows slightly from the
decay term too, rather than shrinking; this is a strictly ADDITIONAL
degradation pressure in the same direction (more, not less, certain to
degrade), not a competing effect, so the CERTAINTY prediction is
unaffected. This prediction, like v1's, requires no secant extrapolation —
it follows directly from what a gradient step IS.

**CPU demonstration** (same fixed 4-element input, `mutant_dose_
demonstration_diverges_from_the_correct_oracle`, patch applied, test added,
run, both reverted):

```
M_signflip_v2:
step=1 l2_divergence=3.519166e-3
step=2 l2_divergence=7.038332e-3
step=3 l2_divergence=1.0557481e-2
step=4 l2_divergence=1.4076664e-2
step=5 l2_divergence=1.7595848e-2
```

Close to (not identical to) `M_signflip` v1's own CPU numbers
<!-- claims63: c1=code('M_signflip', 1); c2=code('M_signflip', 5) -->
(`3.464057e-3`..`1.7320165e-2`) — the small difference is exactly the
additional weight-decay-sign effect described above (v1 only flipped the
`adjusted_grad` combine; v2 flips `lr` itself, which ALSO flips
`one_minus_lr_lambda`'s sign contribution) — both real, growing, non-trivial
divergences confirming a genuine gradient-ascent perturbation on the CPU
arm; the fix over v1 is that this same Rust-level scalar change is what
also reaches the CUDA arm (per the `(1+eps)` family's own hardware
precedent), not a claim that the CPU numbers themselves are new evidence
beyond v1's.

**Measured (12-leg GPU, a100, `redproof-signflip-v2`, D*-gated): RED —
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/n_pos; c2=#/mutant_dose_ladder/red_proof/0/clean_pair_count -->
12/12.** Committed artifact:
`docs/plans/63-how-well/measurements/red-proof/dstar/` at 82253c1b (the
D*-gated re-merge CONTRACT.md amendment 2026-08-29e pre-registered, run
against the SAME committed raw legs already cited above — nothing
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c2=#/mutant_dose_ladder/red_proof/0/n_pos -->
re-run). All four pre-registered predictions confirmed to the bit: 12/12
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=count(#/mutant_dose_ladder/doses/0/mutant_seeds, 'len') -->
clean pairs (all 12 legs ascend against `RED_PROOF_EXPECTED_TRAIN_
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/n_pos; c2=#/mutant_dose_ladder/red_proof/0/n_neg -->
DIRECTION`'s own entry, init anchors bit-identical); `n_pos=12, n_neg=0`,
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=numer(4096, #/mutant_dose_ladder/red_proof/0/p_value); c2=denom(2, #/mutant_dose_ladder/red_proof/0/p_value); c3=numer(2048, #/mutant_dose_ladder/red_proof/0/p_value); c4=denom(1, #/mutant_dose_ladder/red_proof/0/p_value); c5=#/mutant_dose_ladder/red_proof/0/p_value -->
`detected=RED`, two-sided `p = 2/4096 = 1/2048 = 0.00048828125` exact;
<!-- claims63: c1=ledger -->
`red_proof_verdict = PROVEN`; merge exit 0 (PROVEN contributes nothing to
the exit code). Acceptance 5's "mutant column proven RED" is DISCHARGED
at `M = M_signflip_v2`, per the amendment's honesty rider: this mutant is
a catastrophic degradation (the detector's sensitivity ceiling, not a
finding about the corridor between it and `M_nobc`'s own undetected
result above, which remains unresolved and is not claimed).

**Patch sha256s** (both against base `e340391c`, verified `git apply
--check` clean at that sha, verified apply -> `cargo build -p jammi-kernels`
(exit 0) -> `git checkout --` revert, independently, one mutant at a time;
`M_signflip_v2.patch` against base `74fd69ef` — `adamw_step.rs` is
byte-identical to `e340391c`'s copy, same blob `58a1418c`):

- `M_nobc.patch` — sha256
  `9b3c824dc041899c12c0e2d44d12a3ac8c7b86076ffc778638108925ba51bf4e`
  — measured NOT-DETECTED (see "Measured" above); still a candidate for
  re-dosing at a larger magnitude if a stronger RED-proof degradation
  demonstration is later needed, but that is not scheduled here.
- `M_signflip.patch` — sha256 `fb2bd11935e9a08e8a1197aa3a84535660119823aabb421105e389a388f6e5e4`
<!-- claims63: c1=zerocount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
  — **RETIRED, measured INERT on GPU (12/12 bit-identical)**; kept
  patch-file-only for the record, never scheduled to run again.
- `M_signflip_v2.patch` — sha256
  `c81d0ed59d45761bbd6487dbb23c5aaae22f30739c0e2e613d96c4901ad9b202` —
  **scheduled** (replaces `M_signflip` as the guaranteed RED-proof
  degradation candidate).

**Run procedure:** identical to the dose ladder's own on-pod procedure
above (scratch worktree at the recorded base sha, `git apply --check` then
`git apply`, build with the campaign's exact feature list, run the SAME
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/gate_seed_count -->
12-seed `run_leg` vector substituted into the fused arm, merge against the
SAME campaign `alloff` legs, stamp `--mutant-id`/`--mutant-base-sha`/
`--mutant-patch-sha256`, tear down after) — with dose labels
`redproof-nobc` and `redproof-signflip-v2` (NOT `redproof-signflip`, which
names the retired, measured-inert v1 patch) in place of an `epsNN` label,
per the RED-proof label class below.

**RED-proof label class (`ci/scripts/perf/ab_merge.py`, unit 63, CONTRACT.md
addendum 2026-08-29c's own dated postscript): the RED-proof verdict is a
first-class merger output in the SAME invocation and the SAME artifact as
the primary decision and the eps-family dose ladder — never a separate,
exit-1-expected invocation.** A `dose_label` carrying the literal prefix
`redproof-` (e.g. `redproof-nobc`, `redproof-signflip`) is a RED-PROOF
column:

- It participates fully in `build_mutant_dose_column` exactly like any
  eps-labeled column (premises, partner premises, identity, and the
  `detected`/`n_pos`/`n_neg`/`mean_d`/`p_value`/`clean_pair_count`
  computation) — unchanged.
- It is partitioned OUT of the eps-family scans
  (`mutant_dose_ladder_sensitivity`, `mutant_dose_ladder_two_sided_
  falsification`, `mutant_dose_ladder_anomalies`, and the duplicate-EPS arm
  of `mutant_dose_ladder_reject_duplicate_doses`) by
  `partition_red_proof_dose_columns`, called BEFORE those scans ever see
  the assembled `dose_columns` list — a partition on the label PREFIX,
  never a widening of `_dose_label_eps`'s own strict eps-only domain (an
  eps-labeled column keeps its existing strict validation untouched; a
  RED-proof label is never asked to satisfy it, and never silently
  admitted as though it could).
- It remains fully subject to the duplicate-LABEL and duplicate-PATCH_SHA
  arms of `mutant_dose_ladder_reject_duplicate_doses`, which run over the
  FULL `dose_columns` set (a RED-proof column citing the same
  `patch_sha256` as a co-scheduled eps column, or repeating a literal
  label, is refused exactly like any two eps columns would be).
- A `dose_label` that is exactly the bare prefix (`"redproof-"`, no mutant
  name after it) is refused loudly by `partition_red_proof_dose_columns`
  (`RedProofLabelError`), never silently accepted as an anonymous
  RED-proof column.

**Merged artifact fields:** `mutant_dose_ladder.red_proof` — one
`{dose_label, patch_sha256, detected, n_pos, n_neg, mean_d, p_value,
clean_pair_count}` entry per RED-proof column supplied to that invocation
(`build_red_proof_summary`), in the SAME `finetune_run_ab_report.json` the
primary decision and any co-scheduled eps dose ladder land in. And
`mutant_dose_ladder.red_proof_verdict`: the literal string `"PROVEN"` iff
at least one RED-proof column's own `detected` reads the literal string
`"RED"` (degradation-concordant — acceptance 5's own discharge condition);
otherwise `"NOT_PROVEN"` followed by every RED-proof column's own
`dose_label=detected` pair. A RED-proof column reading
`"RED_FOR_INVESTIGATION"` is recorded AS-IS in its own `detected` field (an
anomaly: this mutant is EXPECTED to degrade per this file's own prediction
above, so an improvement-concordant detection here is itself a finding to
investigate, never a second way to reach `"PROVEN"`).

**Exit-code semantics:** `red_proof_verdict == "PROVEN"` contributes
nothing to the merge's own exit code — it is the EXPECTED outcome for a
RED-proof column, unlike `dose_anomalies`. `red_proof_verdict` starting
with `"NOT_PROVEN"` contributes a non-zero exit code, named in the merge's
own stderr and in the artifact itself — acceptance 5's own "mutant column
proven RED" undischarged by every scheduled RED-proof column is a failure
of this run's own purpose, never silently passed through as green. An
`INVALID` RED-proof column (a correctness-of-measurement problem — the
same carve-out every dose column gets) is caught by the SAME
`invalid_doses` check every column in `doses[]` already goes through,
non-zero exit exactly as everywhere else in this module.

**Measured record (12-leg GPU, a100), current-truth discipline:** committed
<!-- claims63: c1=ledger -->
artifact `docs/plans/63-how-well/measurements/red-proof/` at dc1cfc3b (36
raw legs + the gated merge artifact). `--mutant-legs
redproof-nobc:<M_nobc sha256>:<seeds>` and `--mutant-legs
redproof-signflip-v2:<M_signflip_v2 sha256>:<seeds>` were run against the
SAME `ab_merge.py finetune-run` invocation the campaign's primary decision
uses (v1's `redproof-signflip` legs are committed in this same artifact as
evidence only — see `M_signflip` v1's own "Measured"/"lesson" notes above —
never scheduled through the merger; its label is retired). `redproof-nobc`'s
<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=poscount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c3=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c4=meand('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
RAW 12-pair concordance reads `n_pos=5/12`, `mean_d=-0.018` (see `M_nobc`'s
own "Measured" note above) — but the GATED column's own `detected` field
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c3=#/mutant_dose_ladder/red_proof/0/n_pos; c4=#/mutant_dose_ladder/red_proof/0/n_neg; c5=#/mutant_dose_ladder/red_proof/0/mean_d -->
reads `INVALID` (10/12 clean pairs, `n_pos=3`, `n_neg=7`, `mean_d=-0.058`,
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/p_value; c2=absdiff(paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'), #/mutant_dose_ladder/red_proof/0/clean_pair_count) -->
`p=0.34`; 2 legs — seeds 9 and 12 — already failed the (pre-D*)
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=absdiff(paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'), #/mutant_dose_ladder/red_proof/0/clean_pair_count) -->
learning-happened premise on this committed artifact, the SAME 2 legs D*'s
own `train_direction` premise names explicitly; see `M_nobc`'s own "GATED
reading" note above). Neither column discharged acceptance 5 (the committed
artifact's own `red_proof_verdict` reads `NOT_PROVEN (redproof-nobc=INVALID,
redproof-signflip-v2=INVALID)`). This record STANDS as-is, PRE-amendment,
measured before the learning-happened premise was decomposed into
`training_effective`/`train_direction` — it is not relabeled or overwritten
by the D*-gated re-merge below. The D*-gated re-merge of `redproof-
signflip-v2` ONLY (CONTRACT.md amendment 2026-08-29e, `M_nobc`'s own
committed INVALID record above stands as evidence, not as a second column
to re-run) is a SEPARATE, later-committed artifact
(`measurements/red-proof/dstar/` at 82253c1b) that supersedes this one
column's `NOT_PROVEN (..., redproof-signflip-v2=INVALID)` reading with
`RED`/`PROVEN` — see the "Measured" record in the `M_signflip_v2` section
below for the full current-truth discharge.

**Scheduling `M_signflip_v2` (replaces `redproof-signflip`):** pass
`--mutant-legs redproof-signflip-v2:<M_signflip_v2 sha256>:<seeds>` (never
reuse the retired `redproof-signflip` label — a repeated literal label is
refused by `mutant_dose_ladder_reject_duplicate_doses` if it were ever
supplied twice, and reusing it for a DIFFERENT patch would violate "one
dose, one label" even if the merger's own sha check did not already catch
it) to the SAME `ab_merge.py finetune-run` invocation the campaign's
primary decision (and, if co-scheduled, the eps-family dose ladder and
`redproof-nobc`) already runs — the legs themselves stamped exactly per the
on-pod procedure above (`--mutant-id`/`--mutant-base-sha`/
`--mutant-patch-sha256`, `dose_label = redproof-signflip-v2`). Read the
RED-proof verdict directly off `mutant_dose_ladder.red_proof_verdict`
(authoritative in EVERY state) and `.red_proof[]` (the per-column detail,
populated whenever RED-proof evaluation actually ran) in that invocation's
own artifact — never off a separate invocation's exit code, and never off
`sensitivity`/`two_sided_falsification`/`dose_anomalies`, which remain
scoped to the eps family only and are unaffected by a co-scheduled
RED-proof column (and vice versa). CONTRACT.md postscript 2026-08-29d's own
REFUSED-BUT-SCHEDULED state (a RED-proof-labeled column WAS supplied but the
dose set was refused before RED-proof evaluation ever ran, e.g. a
duplicate-patch_sha256 refusal) is the ONE case where `.red_proof[]` stays
`[]` while `red_proof_verdict` already carries an explicit
`"NOT_PROVEN (dose set refused before RED-proof evaluation: ...)"` string —
`red_proof_verdict` is read FIRST and is sufficient on its own to know
acceptance 5 is undischarged; `.red_proof[]` being empty in this one state
is never itself a second, independently-checked signal (never treat an
empty `.red_proof[]` as "nothing scheduled" without first checking whether
`red_proof_verdict` is non-null). Given `M_signflip_v2`'s dispatch-invariant
site (proven on
hardware by the `(1+eps)` family) and its certainty prediction, this run
was PREDICTED to read `redproof-signflip-v2: RED` and discharge acceptance
<!-- claims63: c1=const -->
5's "mutant column proven RED" — per this file's own family F/K discipline,
that was reported as a prediction to be measured, not assumed, and it now
HAS been measured: `redproof-signflip-v2: RED`, `red_proof_verdict =
PROVEN`, discharging acceptance 5 at `M = M_signflip_v2` (see the
"Measured" record above and `docs/plans/63-how-well/measurements/red-proof/
dstar/` at 82253c1b).

## Files

- `M1.patch` — the retired mutant's committed unified diff (patch-file-only;
  never applied to tree state on this branch). Kept for the record; see
  `measurements/campaign-v1/mutant-m1/` for its measured non-detection.
- `M_eps_0.02.patch`, `M_eps_0.10.patch` — falsified pre-spend (positive-eps
  direction predicted IMPROVEMENT, not degradation); committed unified
  diffs against `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` kept as the
  falsification record, patch-file-only, NOT scheduled to run.
- `M_eps_0.50.patch` — the two-sided falsification cell for the
  positive-eps improvement prediction; committed unified diff against
  `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a`; **scheduled**.
- `M_eps_-0.10.patch`, `M_eps_-0.50.patch` — the signed family's
  degradation-direction doses (CONTRACT.md addendum 2026-08-29c); committed
  unified diffs against `cba0b835` (byte-identical base file to
  `ca559b4f16cd1129a2f95ccdd82288b3418e0d0a` for `adamw_step.rs`);
  **scheduled**.
- All five `M_eps_*.patch` files are patch-file-only — never applied to
  tree state on this branch.
- `M_nobc.patch` — bias correction removed entirely (RED-proof pair,
  outside the lr-scale family); committed unified diff against `e340391c`;
  sha256 `9b3c824dc041899c12c0e2d44d12a3ac8c7b86076ffc778638108925ba51bf4e`;
  patch-file-only; **measured NOT-DETECTED (raw)** on 12-leg GPU
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c3=meand('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
  (`redproof-nobc`, `n_pos=5/12`, `mean_d=-0.018`; GATED column reads
  INVALID under CONTRACT amendment 2026-08-29e's own D* premise — see
  "RED-proof mutants" above for the full record, and the committed artifact
  at `docs/plans/63-how-well/measurements/red-proof/` (dc1cfc3b)).
- `M_signflip.patch` — inverted-sign update inside `AdamThetaUpdate::
  cpu_fwd` only (RED-proof pair, outside the lr-scale family); committed
  unified diff against `e340391c`; sha256
  `fb2bd11935e9a08e8a1197aa3a84535660119823aabb421105e389a388f6e5e4`;
  patch-file-only; **RETIRED — measured INERT on 12-leg GPU**
<!-- claims63: c1=zerocount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/signflip__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
  (`redproof-signflip`, 12/12 legs bit-identical to the clean fused column:
  the campaign's fused arm dispatches CUDA on a100, and this patch only
  edits the CPU-only `cpu_fwd` body — see "RED-proof mutants" above for the
  full record and the dispatch-invariant-site lesson; committed artifact
  `docs/plans/63-how-well/measurements/red-proof/` at dc1cfc3b). Kept,
  patch-file-only, for the record; never scheduled to run again.
- `M_signflip_v2.patch` — inverted-sign update at the dispatch-invariant
  `adamw_step_fused_t` `lr` scalar (replaces `M_signflip`; RED-proof pair,
  outside the lr-scale family); committed unified diff against `74fd69ef`
  (`adamw_step.rs` byte-identical to its `e340391c` copy, blob `58a1418c`);
  sha256 `c81d0ed59d45761bbd6487dbb23c5aaae22f30739c0e2e613d96c4901ad9b202`;
  patch-file-only; predicted DEGRADATION with certainty on BOTH CPU and
  CUDA arms (see "RED-proof mutants" above) — the guaranteed RED-proof
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
  member of this pair, **measured RED, 12/12, `red_proof_verdict=PROVEN`
  (D*-gated, `measurements/red-proof/dstar/` at 82253c1b) — discharges
  acceptance 5**.
- `README.md` — this file.
