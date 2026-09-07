# Provenance — `profile_421_clip_vision_a1/kernels.json`

Cut from the REAL `clip-vision-A1` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(`--task image_embedding`, F32, wire default, the contract's `A1` decision
leg for the `clip-vision` tower — the FIRST real `clip-vision` export this
crate has ever attributed; pass 1 declared `clip-vision`'s constants
(`width=768, heads=12, seq=50`) UNVERIFIED, "no `clip-vision` leg has been
pulled yet"):

```
scratchpad/pod421-run2/legs/clip-vision-A1/census.json
```

Same box/tooling as the `clip-text` fixtures. `clip-vision-A1`'s leg
parameters (its own real `manifest.json`): `batch=8` (-> `rows=24`),
`fusible_site_census.lora_sites_wrapped=48` (-> `layers=12`, IDENTICAL
site count to `clip-text` — both towers wrap all 4 sites/layer x 12
layers), `layer_norms=26` (ONE more than `clip-text`'s `25` — OpenCLIP's
ViT has an extra pre-transformer LayerNorm `clip-text` does not; this
module's `C-LN` rule matches BY NAME regardless of count, so the
discrepancy needs no special-casing). `max_seq_length` in the manifest is
`77` (the driver's shared CLI default) but IS NOT USED for this tower —
`derive_signatures` uses the DECLARED constant `seq=50` for `clip-vision`
(49 patches at `224/32=7` per side, `+1` CLS), which this fixture's
`layer_norm_fwd_f32_biased` row (`grid=[1200,1,1] = rows*seq = 24*50`)
CONFIRMS against a real export for the first time.

## What this fixture is FOR: cross-tower generalization + `C-PATCH-EMBED`

Every rule this crate derived from `clip-text` is re-checked here against a
SECOND tower with different `width`/`heads`/`seq` (`768`/`12`/`50` vs
`512`/`8`/`77`) — a rule that only happened to fit `clip-text`'s specific
numbers would fail here:

- `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` at `grid=[1200,1,1]`
  -> `C-LN` (name-only, unchanged); `usigmoid_f32`/`affine_f32`/`bmul_f32`
  at `grid=[3600,1,1]` (`= rows*seq*mlp_width = 1200*3072`) -> `C-GELU`,
  the SAME rule at vision's own `mlp` tier, not `clip-text`'s `3696`.
  `badd_f32` at the IDENTICAL `grid=[3600,1,1]` -> `BIAS/RESIDUAL-MLP`
  (pass 3, finding 2 — `badd_*`'s own tier-suffixed name-class bucket,
  replacing pass 2's coarser `ELEMENTWISE-MLP`), not `C-GELU` — the same
  exclusion pass 1 first documented for `clip-text`, reproduced on a
  second tower's own numbers.
- `fast_max_f32`/`fast_sum_f32` at `grid=[14400,1,1]`
  (`= rows*heads*seq = 24*12*50`) -> `C-ATTN-clip-vision`; a THIRD
  `fast_sum_f32` at `grid=[1,1,1]` -> `LOSS/REDUCE` (matches neither
  `14400` nor `ln_row_count=1200`).
- `badd_f32` at `grid=[704,1,1]` (`704*1024=720,896` covers the attention
  tensor's `attn_shape_elements = rows*heads*seq*seq = 24*12*50*50 =
  720,000` elements) -> `C-ATTN-clip-vision` — the SAME formula as
  `clip-text`'s `grid=1112` rule, at a DIFFERENT concrete number, proving
  the rule generalizes by formula rather than being a hand-copied grid
  value from one tower.
- `badd_f32` at `grid=[900,1,1]` (`out` tier, `rows*seq*width=1200*768`)
  -> `BIAS/RESIDUAL-OUT`; at `grid=[2700,1,1]` (`qkv` tier,
  `rows*seq*3*width`) -> `BIAS/RESIDUAL-QKV` (pass 3, finding 2 — the two
  tier-suffixed name-class buckets, replacing pass 2's `ELEMENTWISE-OUT`/
  `ELEMENTWISE-QKV`).
- `magma_sgemmEx_kernel` at `grid=[1,2,288]` — a THIRD gemm library
  entirely (neither `ampere_sgemm_*` nor `ampere_bf16_*gemm*`), observed
  ONLY on `clip-vision`, never `clip-text` — carries `288 = rows*heads`
  (vision's OWN `attn_batch_count`) at grid POSITION 2 (pass 3, finding 1 —
  the position every real match uses) -> `C-ATTN-clip-vision`, via the SAME
  name-independent batched-grid rule `Kernel2` uses on the BF16
  `clip-text-A2` fixture; `ampere_sgemm_128x32_tn` at `grid=[24,38,1]` (no
  `288` at position 2) -> `BASE-GEMM`.
- `im2col_f32` at `grid=[3528,1,1]` -> `C-PATCH-EMBED` — the ViT patch
  embedding's conv-as-matmul unfold step, observed ONLY on `clip-vision`
  (never `clip-text`, which has no convolutional stem at all).
- `urecip_f32` at `grid=[1,1,1]` (matching NEITHER `ln_row_count=1200` nor
  `attn_softmax_rows`/`attn_batch_count`) falls through `C-LN`'s
  name-restricted eager rule and lands `OPTIMIZER` via the parameter-scale
  catch-all (`total_threads=1,024` sits inside the `width=768` tier's
  `[768, 1792)` block-rounding range) — module doc's own documented,
  deliberately-not-guessed-a-specific-role case (plausibly the embedding
  L2-normalize step, but a single negligible row is not independent
  evidence for a dedicated new rule, so it lands the generic small-scalar
  bucket instead).
- `adamw_moment_update_f32` -> `OPTIMIZER`; `cast_f32_f32`/
  `scaled_cast_add_f32_f32` -> `CAST`; `dropout_fwd_f32` -> `DROPOUT`;
  `Kernel2` at `grid=[6,1,18]` (no `288`, no tier match) stays
  `UNATTRIBUTED` — all unchanged rules, re-confirmed on this tower's own
  numbers.

Only `badd_f32`'s own `launches_per_step` at `grid=[900,1,1]` is asserted
as a literal value anywhere (`test_badd_ladder_launches_per_step_by_tower`,
reading this leg's own point on the cross-leg launch-count ladder straight
off this committed fixture); no other row's timing fields are asserted as
literal values anywhere in the test suite — only which chain each lands
in.
