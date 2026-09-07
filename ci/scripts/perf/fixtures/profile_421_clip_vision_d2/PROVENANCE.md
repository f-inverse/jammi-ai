# Provenance — `profile_421_clip_vision_d2/kernels.json`

Cut from the REAL `clip-vision-D2` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(F32, `JAMMI_KERNELS_DISABLE=lora_linear_fused` ONLY — `layer_norm_fused`
stays ADMITTED, contract §D3):

```
scratchpad/pod421-run2/legs/clip-vision-D2/census.json
```

`clip-vision-D1`'s TWIN — same declared shapes (`ln_row_count=1200`,
`attn_softmax_rows=14400`, `attn_batch_count=288`), cut to drive the same
D1-vs-D2 differential test as `clip-text-d1`/`clip-text-d2` (pass 3,
adversarial-audit finding 2), on the SECOND tower.

## What this fixture is FOR: LN fused (this leg) vs LN eager (`clip-vision-d1`)

- `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` at `grid=[1200,1,1]`
  -> `C-LN` by NAME. NEITHER `usqr_f32` NOR `bsub_f32`/`usqrt_f32`/
  `urecip_f32` appears anywhere in this leg's real export at
  `grid=[2,1,1]` (the `clip-vision-d1` shape) — `layer_norm_fused` is
  admitted here, so the eager mean/var/std/reciprocal sequence never runs.
  `usqrt_f32`/`urecip_f32` DO still appear, but only at `grid=[1,1,1]`
  (`total_threads=1024`, NOT covering `ln_row_count=1200`) — these fall
  through to the parameter-scale `OPTIMIZER` catch-all instead (the SAME
  shape/reasoning `clip-vision-a1`'s own fixture documents for its stray
  `urecip_f32`).
- `usqr_f32` at `grid=[704,1,1]` (`attn_shape_elements=720,000`, the SAME
  shape `clip-vision-d1`'s own `usqr_f32[704,1,1]` row sits at) ->
  `C-ATTN-clip-vision` via the generic attention-shape fallthrough — a row
  the LN toggle does NOT move (present, and classified identically, on
  BOTH legs of the pair — the differential test's own "not everything
  moves" control).
- `badd_f32` at `grid=[900,1,1]` — `launches_per_step=633` here, `865` on
  `clip-vision-d1` (adversarial re-audit, finding 2: NOT run-to-run count
  noise — this is the SAME deterministic `337(A1) -> 633(D2) -> 874(D1)`
  ladder `clip-text` shows identically, evidenced on the real, un-cut
  `clip-vision-A1` export too; module doc, "D1-vs-D2 differential: the
  measured split") -> `BIAS/RESIDUAL-OUT`.
- `ampere_sgemm_128x64_nt`/`ampere_sgemm_128x128_nt`/`magma_sgemmEx_kernel`
  -> `BASE-GEMM`/`C-ATTN-clip-vision` exactly as on `clip-vision-d1`
  (unaffected by the LN toggle, another "does not move" control).
- `Kernel2` at `grid=[6,1,18]` — the IDENTICAL anonymous grid
  `clip-vision-d1`'s own fixture carries, staying `UNATTRIBUTED` on BOTH
  legs (a `dim[1]=1` grid never qualifies for the anonymous-GEMM-tile
  rule, LN toggle or not).
- `adamw_moment_update_f32` -> `OPTIMIZER`, unchanged.
- `affine_f32` at `grid=[900,1,1]` -> ADDED post-audit (finding 2's own
  re-audit), the SAME row `clip-vision-d1`'s own PROVENANCE.md describes
  — present on BOTH legs at DIFFERENT busy (`309.4us`/`48` launches here
  vs `438.6us`/`71` launches on `D1`), giving `Ln1VsD2DifferentialTests` a
  real `ELEMENTWISE-OTHER-OUT` delta to assert directly.

None of the 15 rows' timing fields are asserted as literal expected values
anywhere in the test suite — only which chain each lands in, and (via the
paired `clip-vision-d1`/`clip-vision-d2` differential test) the DIRECTION
`C-LN`'s own busy moves when `layer_norm_fused` toggles.
