# Provenance — `profile_421_htsat_d1/kernels.json`

Cut from the REAL `htsat-D1` leg's nsys census export, pulled by the lead
from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(`--task audio_embedding`, F32, `kernels_disabled=["lora_linear_fused",
"layer_norm_fused", "gelu_erf_fused"]` — `htsat`'s own `D1` disable set,
`profile_421_attribute.D1_DISABLE_SET["htsat"]`):

```
scratchpad/pod421-run2/legs/htsat-D1/census.json
```

Real, byte-for-byte cut of 8 `by_kernel_and_grid` rows, same discipline as
every other fixture in this directory (see `profile_421_htsat_a1/
PROVENANCE.md` for the per-stage signature table this fixture's rows are
checked against).

**The attention/LN row-reduction collision, resolved by BLOCK SIZE (module
doc, `_is_attn_softmax_reduction_grid`):** this leg's own `fast_sum_f32`
carries TWO rows at grid `[98304,1,1]` on the real, un-cut export — one at
`block=[128,1,1]` (`launches_per_step=16`, ABSENT from `htsat-A1`, where
LN is fused) and one at `block=[64,1,1]` (`launches_per_step=18`, PRESENT
on `htsat-A1` too, unchanged by the LN toggle). This fixture's own
`[98304,...]` row is the `block=128` one (the FIRST row nsys reports at
this grid — verified against the real, un-cut file) — the eager LN's own
per-row mean/var reduction for stage 0 (`ln_row_count(0) =
rows*tokens_0 = 24*4096 = 98,304`), coinciding numerically (but NOT by
block) with stage 2's own attention-softmax row count
(`attn_softmax_rows(2) = rows*heads_2*tokens_2 = 24*16*256 = 98,304`).
`block[0]==128` is the smallest power of 2 `>= dim_0=96`
(`2**7=128 > 96 >= 2**6=64`) — a genuine per-row reduction over `dim_s`
elements naturally rounds its own block size up to the next power of 2,
exactly mirroring `layer_norm_fwd_f32_biased`'s own real block sizes on
`htsat-A1` (`256`, a fixed choice for the FUSED kernel, unrelated to this
per-stage rounding — the two LN implementations, fused vs eager, are
different kernels entirely and are not expected to share a block-sizing
convention). This fixture's `fast_sum_f32 grid=[393216,...] block=[64,...]`
row is the SAME attention-softmax reduction `htsat-A1` shows (stage 0,
`attn_softmax_rows(0) = rows*heads_0*tokens_0 = 24*4*4096 = 393,216`,
`block[0]==64==window_size**2`), present and IDENTICAL on both legs
(unaffected by the LN/GELU/LoRA toggle) — the honest negative control that
this fixture's own attention-reduction row is NOT swept into `C-LN` by the
`ln_disabled` gate.

`ugelu_erf_f32`/`uerf_f32`/`uneg_f32` at `grid=[36864,...]` (all three,
`launches_per_step=2` each) are candle's own eager `Tensor::gelu_erf()`
three-kernel decomposition — the D1 twin of `htsat-A1`'s single
`gelu_erf_fwd_f32` kernel at the IDENTICAL grid (stage 0's own
`mlp_shape_elements`), evidencing `GELU_ERF_HTSAT_EAGER_KERNEL_NAMES`
(module doc, "`C-GELU-HTSAT`").

None of these rows' timing fields are asserted as literal expected values
in `test_profile_421_attribute.py` — only structural invariants (which
chain a row lands in) are asserted, same convention as every other
fixture.
