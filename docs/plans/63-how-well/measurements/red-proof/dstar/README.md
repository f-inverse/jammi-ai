# D*-gated RED-proof discharge merge (CONTRACT.md amendment 2026-08-29e)

The re-merge the amendment pre-registered: `redproof-signflip-v2` only, same
committed raw legs as `../raw/` (nothing re-run), merged by the D* merger
(`ab_merge.py` at b5989399: learning-happened decomposed into
`training_effective` + `train_direction`, direction read from the committed
`RED_PROOF_EXPECTED_TRAIN_DIRECTION` table keyed on patch_sha256, plus
`init_anchor_equality`).

All four pre-registered predictions confirmed to the bit:
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/clean_pair_count; c2=#/mutant_dose_ladder/red_proof/0/n_pos; c3=count(#/mutant_dose_ladder/doses/0/mutant_seeds, 'len') -->
(i) 12/12 clean pairs (all 12 legs ascend, matching the `ascent` declaration
    quoted from pre-spend 8f06a42c; init anchors bit-identical);
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/n_pos; c2=#/mutant_dose_ladder/red_proof/0/n_neg; c3=numer(4096, #/mutant_dose_ladder/red_proof/0/p_value); c4=denom(2, #/mutant_dose_ladder/red_proof/0/p_value); c5=numer(2048, #/mutant_dose_ladder/red_proof/0/p_value); c6=denom(1, #/mutant_dose_ladder/red_proof/0/p_value) -->
(ii) n_pos=12, n_neg=0, detected RED, two-sided p = 2/4096 = 1/2048
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/red_proof/0/p_value -->
     = 0.00048828125 exactly;
(iii) `red_proof_verdict` = PROVEN;
<!-- claims63: c1=ledger -->
(iv) merge exit 0 (primary decision GREEN unchanged; PROVEN contributes
     nothing to the exit code).

Acceptance 5's "mutant column proven RED (degradation)" is DISCHARGED at
M = M_signflip_v2, per the amendment's honesty rider: M is a catastrophic
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=ledger; c2=ledger; c3=absdiff(#/mutant_dose_ladder/doses/0/gate_seed_count, 1); c4=#/mutant_dose_ladder/doses/0/gate_seed_count -->
mutant (held-out ~3.3 -> ~20 on 11 of 12 seeds), the detector's sensitivity
ceiling; the corridor between M_nobc (undetected) and M_signflip_v2 remains
unresolved and is not claimed. Per-seed record, every number read from the
committed artifact (round-17 discipline: mechanical facts only, no
inferential characterization): `mutant_dose_ladder.doses[0].per_seed` d_i,
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/per_seed/2/d_i; c2=min(#/mutant_dose_ladder/doses/0/per_seed, '2'); c3=max(#/mutant_dose_ladder/doses/0/per_seed, '2'); c4=absdiff(#/mutant_dose_ladder/doses/0/gate_seed_count, 1) -->
sorted: +0.17900 (seed 2), then +11.19540 .. +20.08700 (the other 11
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=max(#/d_values) -->
seeds). The primary A/B's own `d_values` read max |d_i| = 0.15239 and
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/cross_seed_spread -->
`cross_seed_spread` = 0.08265. The only comparisons this record asserts:
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=rel(#/mutant_dose_ladder/doses/0/per_seed/2/d_i, '>', max(#/d_values)); c2=max(#/d_values); c3=#/mutant_dose_ladder/doses/0/per_seed/2/d_i; c4=#/mutant_dose_ladder/doses/0/clean_pair_count -->
0.17900 > 0.15239, and 0.17900 is the smallest of the 12 mutant values.
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/threshold; c2=#/mutant_dose_ladder/doses/0/gate_seed_count -->
The pre-registered rule is conjunctive — "RED iff >=11 of 12 d_i share a
sign AND the mean agrees in sign" (CONTRACT.md, Frame) — and both legs are
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=rel(#/mutant_dose_ladder/doses/0/n_pos, '>=', 11); c2=#/mutant_dose_ladder/doses/0/threshold; c3=rel(#/mutant_dose_ladder/doses/0/mean_d, '>', 0); c4=const -->
read from `doses[0]`: `n_pos` = 12 >= 11, and `mean_d` = +15.96714 > 0
<!-- claims63: default=docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/per_seed/2/d_i -->
(seed 2 enters that mean at its own +0.17900). The verdict above is
unaffected. Whether seed 2's magnitude exceeds what the primary noise band
could produce is NOT adjudicated by this record and no claim about it is
made.
