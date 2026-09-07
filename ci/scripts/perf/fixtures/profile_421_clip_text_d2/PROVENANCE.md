# Provenance — `profile_421_clip_text_d2/kernels.json`

Cut from the REAL `clip-text-D2` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(F32, `JAMMI_KERNELS_DISABLE=lora_linear_fused` ONLY — `layer_norm_fused`
stays ADMITTED on `D2`, per contract §D3: "D2 isolates the
`lora_linear_fused` kernel's realized gain"):

```
scratchpad/pod421-run2/legs/clip-text-D2/census.json
```

Same box/tooling, same declared shapes, as `profile_421_clip_text_d1`
(`rows=24, seq=77, ln_row_count=1848, attn_softmax_rows=14784,
attn_batch_count=192`) — this fixture is `clip-text-D1`'s TWIN, cut so the
two can drive a hermetic D1-vs-D2 DIFFERENTIAL test: "toggling only
`layer_norm_fused` must move mass into `C-LN`, and the tier buckets must
move by less than `C-LN` does".

## What this fixture is FOR: the D1-vs-D2 differential (LN fused, this leg)

- `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` at `grid=[1848,1,1]`
  -> `C-LN` by NAME (unconditional) — `layer_norm_fused` IS admitted on
  this leg, so these rows exist (they do NOT exist anywhere in `D1`'s own
  export — `unmatched_disables()` would have refused `D1` otherwise).
- `usqrt_f32` at `grid=[1,1,1]` (`total_threads=1024`, NOT covering
  `ln_row_count=1848`) does NOT match `LN_EAGER_EXTENDED_KERNEL_NAMES`'s
  shape gate even though `bsub_f32` is separately cut at the ATTENTION
  shape here (`grid=[1112,1,1]`, unrelated) — neither row is anywhere near
  `clip-text-D1`'s own `grid=[2,1,1]` eager-LN rows, because this leg's
  `layer_norm_fused` is FUSED: the eager mean/var/std/reciprocal sequence
  simply does not run, so no row at that shape exists to (mis)classify.
  This is the fixture's own negative evidence for the `ln_disabled` gate:
  even if a `bsub`/`usqr`/`usqrt`/`urecip` row happened to sit at
  `ln_row_count` on an LN-FUSED leg, `ln_disabled=False`
  here would keep it OUT of `C-LN` — this fixture simply does not exercise
  that hypothetical (no such row exists in the real export), but the
  companion `LnEagerGateUnitTests` exercise it directly on synthetic rows.
- `badd_f32` at `grid=[924,1,1]` (`launches_per_step` on the real export is
  LOWER than `clip-text-D1`'s own count at the identical shape — eager
  LoRA composition on `D1` adds extra `badd_f32` calls at the SAME shape;
  `D2` still has LoRA disabled too, so both legs' counts here are eager-
  LoRA counts, not the `A1` fused count) -> `BIAS/RESIDUAL-OUT`, same
  bucket `D1` uses for its own `badd_f32` row at this shape.
- `ampere_sgemm_128x64_nt`/`ampere_sgemm_32x32_sliced1x4_nt` (D-leg tile
  variants, no `192` in grid) -> `BASE-GEMM`; `ampere_sgemm_128x128_nn` at
  `grid=[1,1,192]` -> `C-ATTN-clip-text` (unchanged from `D1`/`A1`).
- `cutlass_80_simt_sgemm_128x32_8x5_nt_align1` at `grid=[8,2,28]` — the
  IDENTICAL row `D1`'s own fixture carries, matching `GEMM_FAMILY_NAME_RE`
  and landing `BASE-GEMM` on BOTH legs: two INDEPENDENT eager-mode legs
  producing the SAME cuBLAS tile selection is the corroborating evidence.
- `fast_sum_f32`/`fast_max_f32` at `grid=[14784,1,1]` -> `C-ATTN-clip-text`
  (softmax rows, unchanged); `fast_sum_f32` at `grid=[1,1,1]` ->
  `LOSS/REDUCE`.
- `adamw_moment_update_f32` -> `OPTIMIZER`, unchanged.

None of the 13 rows' timing fields are asserted as literal expected values
anywhere in the test suite — only which chain each lands in, and (via the
paired `D1`/`D2` differential test) the DIRECTION `C-LN`'s own busy moves
when `layer_norm_fused` toggles.
