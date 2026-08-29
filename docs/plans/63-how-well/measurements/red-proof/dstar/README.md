# D*-gated RED-proof discharge merge (CONTRACT.md amendment 2026-08-29e)

The re-merge the amendment pre-registered: `redproof-signflip-v2` only, same
committed raw legs as `../raw/` (nothing re-run), merged by the D* merger
(`ab_merge.py` at b5989399: learning-happened decomposed into
`training_effective` + `train_direction`, direction read from the committed
`RED_PROOF_EXPECTED_TRAIN_DIRECTION` table keyed on patch_sha256, plus
`init_anchor_equality`).

All four pre-registered predictions confirmed to the bit:
(i) 12/12 clean pairs (all 12 legs ascend, matching the `ascent` declaration
    quoted from pre-spend 8f06a42c; init anchors bit-identical);
(ii) n_pos=12, n_neg=0, detected RED, two-sided p = 2/4096 = 1/2048
     = 0.00048828125 exactly;
(iii) `red_proof_verdict` = PROVEN;
(iv) merge exit 0 (primary decision GREEN unchanged; PROVEN contributes
     nothing to the exit code).

Acceptance 5's "mutant column proven RED (degradation)" is DISCHARGED at
M = M_signflip_v2, per the amendment's honesty rider: M is a catastrophic
mutant (held-out ~3.3 -> ~20), the detector's sensitivity ceiling; the
corridor between M_nobc (undetected) and M_signflip_v2 remains unresolved
and is not claimed.
