# RED-proof mutant columns (unit 63, acceptance 5 "mutant column proven RED")

<!-- claims63: c1=ledger; c2=ledger; c3=ledger -->
36 raw legs (3 mutants x 12 seeds, mutant-in-fused-arm, a100), merged against
campaign-v2's alloff legs by the round-13-audited gate (`ab_merge.py` at f2265452,
<!-- claims63: c1=ledger; c2=ledger -->
invocation: `finetune-run raw/ out/ 1..12 1,2 --mutant-legs redproof-nobc:9b3c824d…
--mutant-legs redproof-signflip-v2:c81d0ed5…`; v1 `signflip` legs committed as
<!-- claims63: c1=ledger -->
evidence only, not scheduled — its label is retired, see below). Merge exit code 1
(`red_proof_verdict` NOT_PROVEN — see finding 3).

| column | base sha | patch sha256 (prefix) | raw d-concordance vs alloff | gated column reads |
|---|---|---|---|---|
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json; c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c6=#/mutant_dose_ladder/red_proof/0/n_pos; c7=#/mutant_dose_ladder/red_proof/0/n_neg; c8=#/mutant_dose_ladder/red_proof/0/p_value -->
| `redproof-nobc` (v1 legs) | 4d1398a0 | 9b3c824d | n_pos=5/12, mean_d=-0.0183 | INVALID (2 premise-failing pairs; 10 clean: n_pos=3, n_neg=7, p=0.34) |
<!-- claims63: c1=ledger; c2=ledger -->
| `signflip` (v1, RETIRED, never scheduled) | 4d1398a0 | (see mutants/README.md) | 12/12 held-out values bit-identical to clean fused r1 | n/a — inert, retired |
<!-- claims63: c1=ledger; c2=ledger; c3=ledger; c4=ledger; c5=ledger; c6=ledger; c7=ledger; c8=ledger; c9=ledger; c10=ledger; c11=ledger -->
| `redproof-signflip-v2` | 8f06a42c | c81d0ed5 | 12/12 degradation-concordant, effects to +19.2 (sign test would read p = 2/4096 = 1/2048, the exact two-sided tail — NEVER 1/4096) | INVALID (12/12 legs fail the learning-happened premise) |

Findings, honestly stated:

<!-- claims63: c1=ledger; c2=ledger -->
1. `M_nobc` (no bias correction) genuinely perturbs (0/12 legs bit-identical to
   clean fused) but is NOT DETECTED at this operating point: raw concordance
<!-- claims63: c1=poscount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c3=meand('docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json', 'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
   5/12 positive, mean_d=-0.0183 — the uncertain prediction's neutral outcome.
<!-- claims63: c1=ledger; c2=ledger -->
2. `M_signflip` v1 was INERT ON GPU: 12/12 legs' `held_out_example_mean`
   bit-identical to the clean fused column (`campaign-v2/raw/seedN__fused__r1.json`)
   — the patch edited `AdamThetaUpdate::cpu_fwd`, which the a100 legs never
   execute (they dispatch `theta_update_cuda_fwd`'s own PTX). This is the sole
   and sufficient basis for retiring the `redproof-signflip` label; the
   dispatch-invariant lesson and the v2 redesign follow from it
   (mutants/README.md).
3. `M_signflip_v2` IS dispatch-effective (train_probe_series climbs, e.g. seed 1:
<!-- claims63: c1=docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed1.json#/tiers/finetune_run/train_probe_series/0; c2=docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed1.json#/tiers/finetune_run/train_probe_series/3 -->
   3.32 -> 20.25 — gradient ascent operating on GPU) and maximally detected in
<!-- claims63: c1=ledger; c2=ledger -->
   the raw data (12/12 degradation-concordant), BUT every mutant leg fails the
<!-- claims63: c1=const -->
   learning-happened premise (`series[0] - series[-1]` must clear floor 0.0;
   gradient ascent makes it negative BY DESIGN), so the gated column reads
   INVALID and `red_proof_verdict` reads NOT_PROVEN. The round-13 labeling
   fixes reported this state explicitly (per-column `detected`, named exit
   cause) rather than as an unexplained contradiction. STRUCTURAL FINDING,
   recorded before any amendment: the learning-happened premise (built for the
   primary C16 attribution question) refuses exactly the strongest true
   positives of the RED-proof detection question. Resolution was
   pressure-tested, not hand-decided, and now landed: CONTRACT.md amendment
   2026-08-29e decomposed the premise into `training_effective` +
   `train_direction`, and the D*-gated re-merge (`dstar/`) DISCHARGES
   acceptance 5's "mutant column proven RED" at `M = M_signflip_v2` — see
   `dstar/README.md`. This PRE-amendment gated reading (`INVALID`) still
   stands as-is below, current truth at the layer it was measured at; it is
   not relabeled.

The raw-concordance numbers above are derived from the committed legs in `raw/`
paired with `../campaign-v2/raw/seedN__alloff__{r1,r2}.json` exactly as the
merger pairs them; the gated readings are the committed
`finetune_run_ab_report.json`'s own `mutant_dose_ladder.red_proof[]` entries.

**D* re-merge DISCHARGED (CONTRACT.md amendment 2026-08-29e):** finding 3's
own INVALID gated reading above is the PRE-amendment record, measured before
the learning-happened premise was decomposed into `training_effective`/
`train_direction`; it STANDS as-is, never relabeled. The amendment's
pre-registered re-merge of `redproof-signflip-v2` ONLY (`M_nobc`'s own
committed INVALID record above stands as evidence, not as a second column
to re-run; see CONTRACT.md's own "FINAL SCHEDULING + PRE-REGISTERED
PREDICTIONS" for the full, pre-registered prediction set) has now run and is
committed at `dstar/` (`dstar/README.md`, artifact at 82253c1b): all four
pre-registered predictions confirmed to the bit, `redproof-signflip-v2`
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/n_pos; c2=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c3=#/mutant_dose_ladder/red_proof/0/n_neg -->
reads `detected=RED` (`n_pos=12/12`, `n_neg=0`, two-sided
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=numer(4096, #/mutant_dose_ladder/red_proof/0/p_value); c2=denom(2, #/mutant_dose_ladder/red_proof/0/p_value); c3=numer(2048, #/mutant_dose_ladder/red_proof/0/p_value); c4=denom(1, #/mutant_dose_ladder/red_proof/0/p_value); c5=ledger -->
`p=2/4096=1/2048` exact), `red_proof_verdict=PROVEN`, merge exit 0.
Acceptance 5's "mutant column proven RED" is DISCHARGED at
`M = M_signflip_v2`, per the amendment's honesty rider (M is a catastrophic
mutant, the detector's sensitivity ceiling — the corridor between
`M_nobc` and `M_signflip_v2` remains unresolved and is not claimed). None
of the raw numbers or gated readings recorded in THIS file are altered by
the discharge — the D*-gated re-merge is a separate, later-committed
artifact, per this file's own append-only, current-truth discipline.
