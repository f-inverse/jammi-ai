# Campaign v1 evidence (unit 63, H5) — status: INVALID, committed as-measured

Run: a100 (NVIDIA A100-SXM4-80GB, RunPod), producer `ci/scripts/perf/finetune_run_ab.sh`
from the git-pinned checkout `7cca4c31` (merger/gate logic byte-identical through
`4257cde6` — docs-only delta), output dir `20260829T055912Z`. Checkpoint:
ModernBERT-large (config `55dca680…`, weights `44510fec…`, tokenizer `9fd55248…`
sha prefixes, identical on both fleet pods). Objective: MNRL, per the step-0 probe
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/probe/p4-alloff-triplet.json; c1=rel(#/tiers/finetune_run/tie_fraction, '>', 0.5); c2=const -->
ruling (`probe/` — triplet is INADMISSIBLE: tie_fraction 0.65625 > the 0.5 cap).

## What this artifact is

<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
The FIRST full execution of the pre-registered protocol: 12 seeds x {fused, alloff}
x {r1, r2} + lr=0 controls x2 seeds, merged by `ab_merge.py`'s finetune-run mode
(`finetune_run_ab_report.json` / `finetune_run_ab_table.txt`; per-leg reports under
`raw/`).

<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/clean_seed_count; c2=#/decision/gate_seed_count -->
- `status: INVALID` — clean_seed_count 11 != 12 under the pre-registered
  never-rescale rule. Sole premise failure: seed 4's alloff leg,
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v1/raw/seed4__alloff__r1.json#/tiers/finetune_run/learning_happened_delta -->
  `learning_happened_delta = -0.1125` (its held-out trajectory spiked at epoch 1,
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/per_seed/4/trajectory/alloff/0/held_out_mean; c2=#/per_seed/4/trajectory/alloff/1/held_out_mean; c3=#/per_seed/4/trajectory/alloff/2/held_out_mean; c4=docs/plans/63-how-well/measurements/campaign-v1/raw/seed4__fused__r1.json#/tiers/finetune_run/learning_happened_delta -->
  3.2276 -> 3.4211, and recovered to 3.2447; its fused twin cleared at +0.4585;
  both bit-identical across r1/r2).
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/determinism_floor/max_delta -->
- Determinism floor: exactly 0.0 (every r1/r2 pair bit-identical).
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/lr0_control/per_seed/1/per_arm/fused/learning_happened_delta -->
- lr=0 controls: delta exactly 0.0, both arms, both seeds — the floor bites.
- Diagnostic d-column (NOT a verdict; the run is INVALID): computed over the
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/clean_seed_count -->
  11 CLEAN (premise-passing) seeds — seed 4's alloff leg is the one premise
  failure named above, so its pair is excluded from this diagnostic too —
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/n_pos; c2=#/decision/n_neg; c3=#/decision/clean_seed_count; c4=#/decision/mean_d; c5=#/decision/p_value -->
  n_pos=3, n_neg=8 of 11; mean_d=-0.0238, p=0.2266, both also over the same
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/clean_seed_count -->
  11 — the fused arm TRENDS BETTER on held-out loss; no degradation signal.
  **Correction, 2026-08-29c** (docs-ci, unit-63 round-7 audit advisory (a)):
<!-- claims63: c1=hist; c2=hist; c3=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json#/decision/clean_seed_count -->
  an earlier revision of this line stated `n_neg=8/12`, mixing the 11-seed
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/gate_seed_count -->
  sign-count denominator with the pre-registered 12-seed gate count; the
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/finetune_run_ab_report.json; c1=#/decision/clean_seed_count -->
  diagnostic d-column was always computed over the 11 clean seeds only.
  CONTRACT.md amendment 2026-08-29b item 2(ii) carried the identical
  mixed-denominator wording — see that amendment's own 2026-08-29c
  postscript for the correction of record there.

## Why it is INVALID, root-caused (pressure-tested)

The premise leg's implementation is buggy, not the rule: `finetune_run.rs` takes the
probe baseline AFTER the first resume-cycle has already trained epoch 0, so
`learning_happened_delta` measures epoch-1..final, excluding the epoch that carries
the largest learning signal, while its field doc claims "over the run" — and the
endpoint choice was never pre-registered (the "CONTRACT H4" citation at its
definition site names a contract string that does not exist). See CONTRACT.md
amendment 2026-08-29b for the correction plan and the pre-published predictions.

## mutant-m1/ — sensitivity column, measured UNDETECTED

Mutant M1 (adamw bias-correction off-by-one; patch sha `68d2023b…`, base
<!-- claims63: c1=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean') -->
`fc6bd655`, applied+built RC:0) ran 12 fused-arm legs. Result: mutant-vs-fused
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c3=meand('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__fused__r1.json', 'held_out_example_mean'); c4=negcount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c5=paircount('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c6=meand('docs/plans/63-how-well/measurements/campaign-v1/mutant-m1/mutant-seed*.json', 'docs/plans/63-how-well/measurements/campaign-v1/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
n_pos=8/12 mean -0.0035; mutant-vs-alloff n_neg=8/12 mean -0.0236 — NOT RED.
Post-hoc analysis (pressure-tester): M1's net update multiplier is a sign-flipping
<!-- claims63: c1=ledger; c2=ledger -->
early transient (0.744 at t=1 -> 1.009 at t=50), not a sustained one-direction dose,
and the pairing used (mutant-vs-fused) does not mirror the gate's own statistic.
M1 stands as an honestly-recorded non-detection, NOT a sensitivity bound. The
corrected dose-ladder design is in amendment 2026-08-29b.

<!-- claims63: c1=const -->
## probe/ — H5 step-0 (4 legs, a100)

<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/probe/p1-alloff-mnrl-r1.json; c1=absdiff(#/tiers/finetune_run/held_out_example_mean, docs/plans/63-how-well/measurements/campaign-v1/probe/p2-alloff-mnrl-r2.json#/tiers/finetune_run/held_out_example_mean) -->
p1/p2: alloff-mnrl same-seed repeat — bit-identical (determinism floor 0).
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/probe/p3-fused-mnrl.json; c1=#/tiers/finetune_run/attention_block_flash_fused_dispatches; c2=#/tiers/finetune_run/attention_block_flash_declined_dispatches; c3=#/tiers/finetune_run/adamw_fused_dispatches -->
p3: fused-mnrl — flash fused 3276 / declined 0 / adamw fused 26208: the measurand
fires and matches the merger's arm proofs. p4: alloff-triplet — tie_fraction
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v1/probe/p4-alloff-triplet.json; c1=rel(#/tiers/finetune_run/tie_fraction, '>', 0.5); c2=const -->
0.65625 > 0.5 cap: triplet inadmissible; objective ruling = MNRL.
