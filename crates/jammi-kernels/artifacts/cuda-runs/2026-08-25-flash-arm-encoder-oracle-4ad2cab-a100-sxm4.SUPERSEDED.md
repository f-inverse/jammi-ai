# Superseded

`2026-08-25-flash-arm-encoder-oracle-4ad2cab-a100-sxm4.json` was found BLOCKING by a subsequent
phase-4 re-audit (ledger rows 207/214/215), then again by a THIRD audit block (ledger row 245) on
the design that replaced it:

- Its `block1_flash_arm_encoder_level_oracle.main_oracle_green_measured` gradient numbers
  (`grad_err_flash`/`grad_err_block`) were VACUOUS: the loss they were measured under,
  `sum(l2_normalize(mean_pool(hidden))^2)`, is identically `batch` after `pool_and_normalize`'s own
  unit-norm, so `dL/d(theta) == 0` at every dtype, for every arm, always — the auditor showed a real
  K-unrotated kernel mutant IMPROVED that number and PASSED the grad leg.
- Its `bound_derivation` sentence — "the auditor's own hand-run oracle measured
  err(flash,f32)=5.71e-2 vs err(block,f32)=9.11e-2 — flash already closer to truth" — was WRONG and
  unreproducible from this repo (no committed script produced it); the round that superseded this
  artifact deleted that sentence from `crates/jammi-encoders/src/modernbert.rs`'s own section
  comment rather than repeating it.
- The round-6 fix that followed this artifact (single-last-layer gradient, an 8-seed
  `err(other)/err(block)` MEAN+MAX ratio bound) was ITSELF found BLOCKING a third time: the grad
  leg sat on the model's GLOBAL last layer only, structurally blind to a backward-only defect
  confined to the 18 LOCAL (windowed) layers (a bwd-only window drop at the flash op's own `cfg`
  left every number bit-identical), and the 8-seed MAX bound false-REDded on fresh seeds outside
  the fitting set (7.18 pooled / 16.08 grad vs a 7.0 grad ceiling).

Per this directory's own README ("a re-proof of the same tip on another box is a new file, never
an overwrite"), this file is kept as-is (`status` flipped to `SUPERSEDED` only) rather than edited.
The replacement metric — the FULL concatenated gradient of every trainable LoRA `Var` in the model
(224 tensors, every layer including the windowed ones), `cos(flash, f32) >= FLOOR` AND `cos(flash,
f32) >= cos(block, f32)`, at `b4_s128`/`b8_s512`, with four RED controls (K-unrotated,
backward-only window drop, `dv` zeroed in the op's own `bwd`, `window: None` at the production call
site) — is `2026-08-25-flash-arm-grad-cosine-<sha7>-a100c-sxm4.json`, produced by
`crates/jammi-encoders/src/modernbert.rs`'s `flash_arm_grad_cosine_dense_cuda_bf16` and its sibling
RED-control tests on `perf/p6-fa2-dense`.
