# Campaign v1 evidence (unit 63, H5) — status: INVALID, committed as-measured

Run: a100 (NVIDIA A100-SXM4-80GB, RunPod), producer `ci/scripts/perf/finetune_run_ab.sh`
from the git-pinned checkout `7cca4c31` (merger/gate logic byte-identical through
`4257cde6` — docs-only delta), output dir `20260829T055912Z`. Checkpoint:
ModernBERT-large (config `55dca680…`, weights `44510fec…`, tokenizer `9fd55248…`
sha prefixes, identical on both fleet pods). Objective: MNRL, per the step-0 probe
ruling (`probe/` — triplet is INADMISSIBLE: tie_fraction 0.65625 > the 0.5 cap).

## What this artifact is

The FIRST full execution of the pre-registered protocol: 12 seeds x {fused, alloff}
x {r1, r2} + lr=0 controls x2 seeds, merged by `ab_merge.py`'s finetune-run mode
(`finetune_run_ab_report.json` / `finetune_run_ab_table.txt`; per-leg reports under
`raw/`).

- `status: INVALID` — clean_seed_count 11 != 12 under the pre-registered
  never-rescale rule. Sole premise failure: seed 4's alloff leg,
  `learning_happened_delta = -0.1125` (its held-out trajectory spiked at epoch 1,
  3.2276 -> 3.4211, and recovered to 3.2447; its fused twin cleared at +0.4585;
  both bit-identical across r1/r2).
- Determinism floor: exactly 0.0 (every r1/r2 pair bit-identical).
- lr=0 controls: delta exactly 0.0, both arms, both seeds — the floor bites.
- Diagnostic d-column (NOT a verdict; the run is INVALID): computed over the
  11 CLEAN (premise-passing) seeds — seed 4's alloff leg is the one premise
  failure named above, so its pair is excluded from this diagnostic too —
  n_pos=3, n_neg=8 of 11; mean_d=-0.0238, p=0.2266, both also over the same
  11 — the fused arm TRENDS BETTER on held-out loss; no degradation signal.
  **Correction, 2026-08-29c** (docs-ci, unit-63 round-7 audit advisory (a)):
  an earlier revision of this line stated `n_neg=8/12`, mixing the 11-seed
  sign-count denominator with the pre-registered 12-seed gate count; the
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
`fc6bd655`, applied+built RC:0) ran 12 fused-arm legs. Result: mutant-vs-fused
n_pos=8/12 mean -0.0035; mutant-vs-alloff n_neg=8/12 mean -0.0236 — NOT RED.
Post-hoc analysis (pressure-tester): M1's net update multiplier is a sign-flipping
early transient (0.744 at t=1 -> 1.009 at t=50), not a sustained one-direction dose,
and the pairing used (mutant-vs-fused) does not mirror the gate's own statistic.
M1 stands as an honestly-recorded non-detection, NOT a sensitivity bound. The
corrected dose-ladder design is in amendment 2026-08-29b.

## probe/ — H5 step-0 (4 legs, a100)

p1/p2: alloff-mnrl same-seed repeat — bit-identical (determinism floor 0).
p3: fused-mnrl — flash fused 3276 / declined 0 / adamw fused 26208: the measurand
fires and matches the merger's arm proofs. p4: alloff-triplet — tie_fraction
0.65625 > 0.5 cap: triplet inadmissible; objective ruling = MNRL.
