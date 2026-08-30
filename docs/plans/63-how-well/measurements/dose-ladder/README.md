# Signed dose ladder (unit 63, amendment 2026-08-29b item 3 / addendum 2026-08-29c)

<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=ledger; c2=count(#/mutant_dose_ladder/doses, 'len'); c3=count(#/mutant_dose_ladder/doses/0/mutant_seeds, 'len') -->
36 legs (3 doses x 12 seeds, mutant-in-fused-arm, a100, base 494fb3e3, per-dose patch
applied to a scratch worktree, legs stamped via --mutant-id/--mutant-base-sha/
--mutant-patch-sha256), merged against campaign-v2's alloff legs by the round-12-audited
<!-- claims63: c1=ledger -->
gate. Merge exit code 1 (dose_anomalies non-empty — by design).

<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/gate_seed_count -->
| dose | detected | n_neg/12 | mean_d | p |
|---|---|---|---|---|
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/n_neg; c2=#/mutant_dose_ladder/doses/0/mean_d; c3=#/mutant_dose_ladder/doses/0/p_value -->
| eps-0.50 | RED_FOR_INVESTIGATION | 11 | -0.0709 | 0.00635 |
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/1/n_neg; c2=#/mutant_dose_ladder/doses/1/mean_d; c3=#/mutant_dose_ladder/doses/1/p_value -->
| eps-0.10 | RED_FOR_INVESTIGATION | 11 | -0.0434 | 0.00635 |
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/2/n_neg; c2=#/mutant_dose_ladder/doses/2/mean_d; c3=#/mutant_dose_ladder/doses/2/p_value -->
| eps0.50  | not-detected          |  9 | -0.0399 | 0.146   |

Findings, honestly stated:
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=#/mutant_dose_ladder/doses/0/n_neg; c2=#/mutant_dose_ladder/doses/0/gate_seed_count; c3=numer(2048, #/mutant_dose_ladder/doses/0/p_value); c4=denom(13, #/mutant_dose_ladder/doses/0/p_value); c5=#/mutant_dose_ladder/doses/0/p_value -->
1. THE DETECTOR WORKS: 11/12 sign-concordance fired at p = 13/2048 = 0.00635 < alpha2
<!-- claims63: default=docs/plans/63-how-well/measurements/dose-ladder/finetune_run_ab_report.json; c1=const; c2=abs(#/mutant_dose_ladder/doses/1/mean_d); c3=abs(#/mutant_dose_ladder/doses/0/mean_d) -->
   (0.0064) on a real |mean| ~ 0.04-0.07 effect — the instrument detects sign-consistent
   shifts at this magnitude. The demonstration landed in the IMPROVEMENT direction.
2. THE SECANT PREDICTION IS REFUTED (the two-sided falsification cell working): lr
   DEFLATION significantly IMPROVES held-out loss (both negative doses), i.e. the
   campaign lr (2e-4) sits above the local optimum for this fixture. Recorded as
   dose_anomalies; a genuine training-configuration finding, out of scope for C16's
   fused-vs-alloff question (both arms share the lr).
3. sensitivity: None — no degradation-RED among the (1+eps) family: the lr-scale knob
   cannot produce a degradation demonstration near this operating point. Acceptance 5's
   "mutant column proven RED" is therefore discharged by a separate RED-proof mutant
   outside the lr-scale family (see mutants/README.md), with the (1+eps) ladder standing
   as the two-sided sensitivity map.
