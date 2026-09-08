# Provenance — `profile_421_clip_text_d1/kernels.json`

Cut from the REAL `clip-text-D1` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(F32, `JAMMI_KERNELS_DISABLE=lora_linear_fused,layer_norm_fused` per
contract §D3's per-tower D1 set — CLIP has no `gelu_erf_fused` admit site,
so it is not named for this tower):

```
scratchpad/pod421-run2/legs/clip-text-D1/census.json
```

Same box/tooling as the other `profile_421_clip_text_*` fixtures.
`clip-text-D1`'s leg parameters: `batch=8` (-> `rows=24`),
`max_seq_length=77`, `dtype="f32"`, `kernels_disabled=
["lora_linear_fused", "layer_norm_fused"]`, same
`fusible_site_census.lora_sites_wrapped=48` as every other `clip-text` leg
(-> `layers=12`) so the declared shapes (`ln_row_count=rows*seq=1848`,
`attn_softmax_rows=14784`, `attn_batch_count=192`) are unchanged.

## What this fixture is FOR: the eager LayerNorm rule and D-leg tile variants

With `layer_norm_fused` disabled, `layer_norm_fwd_f32_biased`/
`layer_norm_bwd_dx_f32` do not appear anywhere in the real `D1` export
(`unmatched_disables()` would have refused the leg otherwise) — this
fixture does not need a negative-absence row for that (it is a property of
the WHOLE real census, not a single row); instead it carries the EAGER
LayerNorm's own replacement kernels, at the NEW declared row count
`ln_row_count = rows*seq = 1,848` (distinct from `attn_softmax_rows=14,784`
by construction):

- `fast_sum_f32` at `grid=[1848,1,1]` (`block=[512,1,1]`, the per-row
  mean/var reduction) -> `C-LN`. A SECOND `fast_sum_f32` row at
  `grid=[14784,1,1]` (`block=[128,1,1]`) still matches the SOFTMAX row
  count -> `C-ATTN-clip-text` (D1 still has attention, only LoRA/LN are
  disabled); a THIRD `fast_sum_f32` at `grid=[1,1,1]` (`block=[1024,1,1]`,
  matching NEITHER declared row count) -> `LOSS/REDUCE`. All three
  disambiguated purely by grid, same kernel name.
- `usqrt_f32`/`urecip_f32` at `grid=[2,1,1]` (`block=[1024,1,1]`,
  `total_threads=2048` covering the `1,848`-element flat row-scalar buffer)
  -> `C-LN` — the eager `sqrt(var+eps)` and `1/std` steps
  (`launches_per_step=25` on the real export, matching this leg's own
  witnessed `fusible_site_census.layer_norms=25`, corroborating but NOT
  asserted as a literal in any test here).
- `fast_max_f32` at `grid=[14784,1,1]` -> `C-ATTN-clip-text` (unchanged
  from `clip-text-A1`; attention itself is not disabled on `D1`).
- `ampere_sgemm_128x64_nt` at `grid=[4,32,5]` and
  `ampere_sgemm_32x32_sliced1x4_nt` at `grid=[16,16,2]` are TWO tile
  variants that do NOT appear anywhere in `clip-text-A1`'s vocabulary —
  cuBLAS picked different algorithms once the eager LoRA/LN arithmetic
  changed the surrounding shapes/strides slightly. Neither carries `192`
  (`rows*heads`) in its grid -> both `BASE-GEMM`, via `GEMM_FAMILY_NAME_RE`
  alone (no new name needs to be, or is, hand-added to `KNOWN_KERNEL_NAMES`
  for either — this is the D-leg tile-variant rule the contract asks for).
  A THIRD row, `ampere_sgemm_128x128_nn` at `grid=[1,1,192]`, IS in
  `clip-text-A1`'s vocabulary and IS carries `192` -> `C-ATTN-clip-text`,
  unchanged, confirming the same relational rule applies identically on a
  D-leg's shifted tile selection.
- `badd_f32` at `grid=[924,1,1]` (`launches_per_step=874` on the real
  export, vs `337` on `clip-text-A1` — eager LoRA composition adds extra
  calls at the SAME shape, not a new one) -> `BIAS/RESIDUAL-OUT` (the
  tier-suffixed name-class bucket `badd_*` gets).
- `usqr_f32` at `grid=[924,1,1]` (`launches_per_step=25`, matching
  `layer_norms` exactly — plausibly the eager LN's own `(x-mean)^2` step)
  -> `ELEMENTWISE-OTHER-OUT`, NOT `C-LN`: this row sits at the `out`
  ACTIVATION-tier shape (`out_shape_elements=946,176`), not at
  `ln_row_count=1,848` — `LN_EAGER_EXTENDED_KERNEL_NAMES`'s own shape gate
  only fires at the LATTER shape, so a launch-count coincidence alone
  (`25` matching `layer_norms`) is never sufficient by itself;
  `clip-vision-d1`'s own fixture shows the SAME name (`usqr_f32`) DOES
  land `C-LN` on that tower, at `grid=[2,1,1]` (`ln_row_count` for vision)
  — the rule is grounded in SHAPE, not name, and generalizes per-tower
  exactly where the real export shows it firing.
- `adamw_moment_update_f32` at `grid=[4,1,1]` -> `OPTIMIZER`, unchanged.
- `cutlass_80_simt_sgemm_128x32_8x5_nt_align1` at `grid=[8,2,28]`
  (`threads=57,344`) — the real name `kernel_census.py`'s demangled-name
  keying produces here (module doc, "Kernel identity"; the raw export's
  `shortName` collapses this to the generic `Kernel2` template-wrapper
  name) — matches `GEMM_FAMILY_NAME_RE` and lands `BASE-GEMM`;
  `clip-text-d2`'s own fixture carries the IDENTICAL row on an INDEPENDENT
  eager leg, corroborating.

None of the 13 rows' timing fields are asserted as literal values anywhere
in the test suite — only which chain each lands in.
