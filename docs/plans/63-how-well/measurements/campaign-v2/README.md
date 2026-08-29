# Campaign v2 (unit 63, H5) — status: GREEN, both pre-published predictions confirmed

Run: a100, producer at git 6f5874d4 (merger gate logic identical through 9b98e0bd),
output 20260829T111951Z, ModernBERT-large, MNRL (probe-ruled), corrected init-anchored
train_probe_series per CONTRACT amendment 2026-08-29b.

Verdict: `status: GREEN` — clean_seed_count 12/12 (the corrected probe clears seed 4:
it trained, turbulently, deterministically); n_pos=4 / n_neg=8 of 12; mean_d = -0.020080;
p = 0.3876953125 (exact 1588/4096); alpha2 = 0.0064; concordant_direction none.
Determinism floor exactly 0.0 (every r1/r2 bit-identical); lr0 controls fail the floor at
exactly 0.0 (bite intact); no premise failures; cross-seed identity homogeneous.

Prediction verification (published in amendment 2026-08-29b BEFORE this run):
(i) per-seed d-column reproduces v1 BIT-IDENTICALLY — verified: all 24 main-leg
    held_out_example_mean values byte-equal to campaign-v1's (lead-run comparison,
    checked=24 mismatches=0);
(ii) conditional on (i), verdict GREEN — confirmed exactly (v1's full-12 diagnostic
    column: mean -0.02008, 8/12 negative).

Reading: the fused cascade does not degrade learning on this instrument; the fused arm
trends (non-significantly) BETTER. C16's lift condition is measured, GREEN, under the
pre-registered rule.
