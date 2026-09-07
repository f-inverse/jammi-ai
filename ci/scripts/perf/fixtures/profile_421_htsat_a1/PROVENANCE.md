# Provenance — `profile_421_htsat_a1/kernels.json`

Cut from the REAL `htsat-A1` leg's nsys census export, pulled by the lead
from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(`ci/scripts/perf/profile_421_legs.sh`, `--task audio_embedding`, F32, wire
default, `kernels_disabled=[]` — every fusible seam admitted, the
contract's `A1` decision leg for the `htsat` tower):

```
scratchpad/pod421-run2/legs/htsat-A1/census.json
```

(`nsys 2025.3.2.474-253236389321v0`, `NVIDIA A100-SXM4-80GB`). That file is
NOT git-tracked. The cut below was made by a short `python3 -c` script
reading the pulled file directly and writing out ONLY the 21
`by_kernel_and_grid` rows this crate's HTSAT tests need, byte-for-byte
from the real export, plus the leg's own top-level
`gpu_kernel_us_per_step`/`wall_s_per_step`.

`htsat-A1`'s leg parameters this fixture's tests derive signatures from
(the real, pulled `manifest.json` for this leg): `batch=8` (-> `rows=24`),
`fusible_site_census={"lora_sites_wrapped": 77, "layer_norms": 29,
"gelu_seam_calls_per_forward": 12}`. Combined with `profile_421_attribute.
py`'s own declared HTSAT constants (`HTSAT_DEPTHS=(2,2,6,2)`,
`HTSAT_HEADS=(4,8,16,32)`, `HTSAT_WINDOW_SIZE=8`,
`HTSAT_FINAL_STAGE_DIM=768`, `HTSAT_SPEC_SIZE=256`, `HTSAT_PATCH_SIZE=4`
— read from the public `laion/clap-htsat-fused` config, per
`crates/jammi-encoders/src/htsat_audio.rs`'s own `hidden_size ==
patch_embeds_hidden_size << (num_stages-1)` convention), the four
per-stage signatures `derive_htsat_signatures` derives are:

| stage | dim | tokens | windows | heads | attn_batch_count | attn_shape_elements | mlp_shape_elements | out_shape_elements |
|---|---|---|---|---|---|---|---|---|
| 0 | 96  | 4096 | 64 | 4  | 6144 | 25,165,824 | 37,748,736 | 9,437,184 |
| 1 | 192 | 1024 | 16 | 8  | 3072 | 12,582,912 | 18,874,368 | 4,718,592 |
| 2 | 384 | 256  | 4  | 16 | 1536 | 6,291,456  | 9,437,184  | 2,359,296 |
| 3 | 768 | 64   | 1  | 32 | 768  | 3,145,728  | 4,718,592  | 1,179,648 |

Every element count above was cross-checked against the REAL
`by_kernel_and_grid` rows before a single classification rule was written
(module doc, "HTSAT" in `profile_421_attribute.py`) — e.g.
`gelu_erf_fwd_f32`'s four real grids (`36864`/`9216`/`18432`/`4608`,
`launches_per_step` `2`/`6`/`2`/`2`) are EXACTLY `ceil(mlp_shape_elements(
stage)/1024)` for stages `0`/`2`/`1`/`3` respectively, and the launch
counts EXACTLY equal `HTSAT_DEPTHS[stage]` (`2,6,2,2` in stage order) —
`gelu_erf_fused` is called once per Swin block, never guessed.

**Known ambiguity, stated honestly (NOT resolved by this fixture or the
classifier):** `out_shape_elements(stage) == mlp_shape_elements(stage+2)`
is a MATHEMATICAL IDENTITY of this architecture for `stage in {0, 1}`
(tokens quarter and dim doubles each stage, so
`tokens_s * dim_s == (tokens_s/16) * (4*dim_s)`) — grid `9216` is BOTH
`out_shape_elements(stage=0)` AND `mlp_shape_elements(stage=2)`; grid
`4608` is BOTH `out_shape_elements(stage=1)` AND
`mlp_shape_elements(stage=3)`. `classify_htsat_kernel` checks the `out`
tier before the `mlp` tier (module doc), so a row at one of these TWO
grids always resolves `BIAS/RESIDUAL-OUT`/`ELEMENTWISE-OTHER-OUT`/
`PERMUTE/RESHAPE`, never the MLP-tier bucket — this is a DELIBERATE,
DOCUMENTED priority choice, not a verified disambiguation (the real
`badd_f32 grid=[9216,...]` row could plausibly be stage 2's own MLP bias
rather than stage 0's residual — this module does not have a shape-only
way to tell the two apart and does not guess). This fixture therefore
avoids `9216`/`4608` for its own `BIAS/RESIDUAL-OUT`/
`ELEMENTWISE-OTHER` test rows (using the UNAMBIGUOUS `2304`/`1152` grids,
stages 2/3's own `out` tier, which have no stage-4/5 MLP to collide with)
and instead documents the collision here rather than papering over it with
a cherry-picked row.

`ampere_sgemm_128x128_nt grid=[1,1,6144]` and
`magma_sgemmEx_kernel grid=[1,2,1536]` are TWO different GEMM libraries
selected for TWO different stages (0 and 2 respectively), both carrying
`grid[2] == attn_batch_count(stage)` exactly — the SAME name-independent,
grid-position-2 relational rule CLIP's `C-ATTN-<tower>` uses, evidenced
generalizing to a third stage geometry. `fast_max_f32 grid=[393216,...]
block=[64,...]` is the window-attention softmax's own reduction (stage 0,
`attn_softmax_rows(0) = rows*heads_0*tokens_0 = 24*4*4096 = 393,216`,
`block[0]==64==window_size**2`, module doc,
"`_is_attn_softmax_reduction_grid`").

None of the 21 rows' `us_per_step`/`launches_per_step`/`share` fields are
asserted as literal expected values anywhere in
`test_profile_421_attribute.py` — only structural invariants (which chain
a row lands in) are asserted, same convention as every CLIP fixture.
