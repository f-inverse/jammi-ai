# Superseded

`2026-08-25-flash-arm-encoder-oracle-4ad2cab-a100-sxm4.json` was found BLOCKING by the phase-4
re-audit (ledger rows 207, 214, 215) — see `crates/jammi-encoders/src/modernbert.rs`'s own
"Phase-4 re-audit close-out" section comment (currently around lines 3365-3419) for the full
derivation this file summarizes:

- Its `block1_flash_arm_encoder_level_oracle.main_oracle_green_measured` gradient numbers
  (`grad_err_flash: 34924.0`, `grad_err_block: 27795.0` at `b8_s512`, and the `b1_s128` pair) were
  VACUOUS: the loss they were measured under, `sum(l2_normalize(mean_pool(hidden))^2)`, is
  identically `batch` after `pool_and_normalize`'s own unit-norm, so `dL/d(theta) == 0` at every
  dtype, for every arm, always — the auditor showed a real K-unrotated kernel mutant IMPROVED that
  ratio and PASSED the grad leg on a zero-signal denominator.
- Its `bound_derivation` sentence — "Auditor's own hand-run oracle on this checkpoint measured
  err(flash,f32)=5.71e-2 vs err(block,f32)=9.11e-2 -- flash already closer to truth" — is WRONG and
  unreproducible from this repo (no committed script produces it). The committed GREEN run under
  THIS SAME artifact's own `main_oracle_green_measured` already shows the opposite sign on the
  pooled leg (`0.18823 > 0.17482` at `b8_s512`, `0.16754 > 0.15673` at `b1_s128` — flash further
  from the f32 reference, not closer); the sentence was never corrected to match the numbers sitting
  next to it. `modernbert.rs`'s current section comment deletes this claim rather than repeating it.

Per this directory's own README ("a re-proof of the same tip on another box is a new file, never
an overwrite"), this file is kept as-is (`status` flipped to `SUPERSEDED` only) rather than edited.
The replacement measurement — a fixed, seed-keyed non-uniform cotangent (`loss = (pooled *
dy).sum()`, never a unit-norm-vacuous loss), a cosine-distance grad metric measured at the LAST
layer's `Wqkv` LoRA B (not layer 0, where 28 backward matmuls of ordinary bf16 rounding noise
dominate any real signal), and an 8-fixed-seed sweep (`FLASH_ORACLE_SWEEP_SEEDS`) reused identically
by the healthy oracle and every RED control, replacing the single-seed `K = 1.5` bound with
seed-derived `FLASH_ORACLE_K_MEAN_POOLED` / `FLASH_ORACLE_K_MAX_POOLED` /
`FLASH_ORACLE_K_MEAN_GRAD` / `FLASH_ORACLE_K_MAX_GRAD` constants — is
`2026-08-25-flash-arm-encoder-oracle-2aa1551-a100-sxm4.json` in this directory, produced by the
SAME test (`modernbert::tests::flash_arm_encoder_level_three_way_oracle_dense_cuda_bf16`), and is
what `crates/jammi-encoders/src/modernbert.rs` implements today.
