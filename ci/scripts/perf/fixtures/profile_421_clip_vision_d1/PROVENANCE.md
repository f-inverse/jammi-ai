# Provenance — `profile_421_clip_vision_d1/kernels.json`

Cut from the REAL `clip-vision-D1` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(F32, `JAMMI_KERNELS_DISABLE=lora_linear_fused,layer_norm_fused` — the
per-tower D1 set, contract §D3):

```
scratchpad/pod421-run2/legs/clip-vision-D1/census.json
```

Same box/tooling as `profile_421_clip_vision_a1`. Leg parameters: `batch=8`
(-> `rows=24`), `dtype="f32"`, `fusible_site_census.lora_sites_wrapped=48`
(-> `layers=12`) — the SAME declared shapes as `clip-vision-a1`/`a2`
(`ln_row_count=rows*seq=1200`, `attn_softmax_rows=14400`,
`attn_batch_count=288`, `attn_shape_elements=720,000`,
`out_shape_elements=921,600`). This is the SECOND tower's D1 fixture (the
first, `clip-text-d1`, only carries `usqrt`/`urecip`/`bsub` at
`ln_row_count`, never `usqr`) — this leg's own real export ALSO carries
`usqr_f32` at `grid=[2,1,1]` (`ln_row_count` shape), which is the ONLY
real evidence across all eight pulled CLIP legs for the contract's own
`usqr` naming in `LN_EAGER_EXTENDED_KERNEL_NAMES`.

## What this fixture is FOR: `usqr` at the LN row count, and the D1-vs-D2 pair

- `usqr_f32`/`usqrt_f32`/`urecip_f32`/`bsub_f32`, ALL FOUR at
  `grid=[2,1,1],block=[1024,1,1]` (`total_threads=2048` covering
  `ln_row_count=1200`) -> `C-LN` (the eager LN's own mean/var/std/reciprocal
  steps; `ln_disabled=True` on this leg, gating the rule on).
  `launches_per_step` on the real export is `23`-`25` for all four,
  matching this leg's own `fusible_site_census.layer_norms=26` (`clip-
  vision`'s own LN-instance count, close but not asserted literally here).
- `fast_sum_f32` at `grid=[1200,1,1],block=[1024,1,1]` -> `C-LN` (the
  ROW-COUNT-gated reduction rule: `_is_row_count_grid` matches
  `ln_row_count` exactly, distinct from the SAME kernel's `grid=[14400,
  1,1]` row (softmax rows -> `C-ATTN-clip-vision`) and its `grid=[1,1,1]`
  row (-> `LOSS/REDUCE`) — three rows, one name, disambiguated purely by
  grid, same discipline `clip-text-d1`'s fixture already established.
- `ampere_sgemm_128x64_nt` (no `288` in grid) -> `BASE-GEMM`;
  `ampere_sgemm_128x128_nt` at `grid=[1,1,288]` and `magma_sgemmEx_kernel`'s
  own full demangled signature at `grid=[1,2,288]` (a THIRD GEMM library)
  -> `C-ATTN-clip-vision` (the batched-attention grid rule, gated on
  `grid[2]==288` — fires identically for a named cuBLAS tile and a MAGMA
  kernel alike).
- `cutlass_80_simt_sgemm_32x128_8x5_nt_align1` at `grid=[6,1,18]`
  (`dim[1]=1`) matches `GEMM_FAMILY_NAME_RE` and lands `BASE-GEMM`
  regardless of its own degenerate grid dimension — a SECOND tower's
  evidence corroborating `clip-vision-A1`'s own row at the IDENTICAL grid.
- `badd_f32` at `grid=[900,1,1]` -> `BIAS/RESIDUAL-OUT` (`out_shape_elements
  =900*1024`, real evidence: eager LoRA's extra composition adds inflate
  `launches_per_step` here relative to `clip-vision-a1`'s own fused count,
  same pattern `clip-text-d1`'s own fixture already documents).
- `adamw_moment_update_f32` -> `OPTIMIZER`, unchanged.
- `affine_f32` at `grid=[900,1,1]` (`out_shape_elements`) so that
  `Ln1VsD2DifferentialTests` has a REAL `ELEMENTWISE-OTHER-OUT` row
  present on BOTH this leg and its `D2` twin, at DIFFERENT busy (more
  launches here than on `D2`) — the eager-LN-affine "leak" this module's
  own docstring section "D1-vs-D2 differential: the measured split"
  describes (a real, non-zero `D1 > D2` delta the differential test
  asserts directly, byte-for-byte from the same real export).

Only `badd_f32`'s own `launches_per_step` at `grid=[900,1,1]` is asserted
as a literal value anywhere (`test_badd_ladder_launches_per_step_by_tower`,
reading this leg's own point on the cross-leg launch-count ladder straight
off this committed fixture); no other row's timing fields are asserted as
literal expected values anywhere in the test suite.
