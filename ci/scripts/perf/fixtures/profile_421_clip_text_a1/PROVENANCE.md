# Provenance — `profile_421_clip_text_a1/kernels.json`

Cut from the REAL `clip-text-A1` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(`ci/scripts/perf/profile_421_legs.sh`, `--task text_embedding`, F32, wire
default, the contract's `A1` decision leg):

```
scratchpad/pod421-run2/legs/clip-text-A1/census.json
```

(`nsys 2025.3.2.474-253236389321v0`, `NVIDIA A100-SXM4-80GB`). That file is
NOT git-tracked (it lives in a scratch pull directory, not a branch), so it
could not be `git show`n the way `p6_fa2_dense_raw_runs/`'s fixtures were —
the cut below was made with a short `python3 -c` script reading the pulled
file directly and writing out ONLY the `by_kernel_and_grid` rows this
crate's tests need, byte-for-byte from the real export (no hand-edited
field), plus the leg's own top-level `gpu_kernel_us_per_step` /
`wall_s_per_step`:

```python
rows = json.load(open(".../pod421-run2/legs/clip-text-A1/census.json"))["by_kernel_and_grid"]
wanted = [
    ("layer_norm_fwd_f32_biased", [1848, 1, 1]),
    ("layer_norm_bwd_dx_f32", [1848, 1, 1]),
    ("usigmoid_f32", [3696, 1, 1]),
    ("affine_f32", [3696, 1, 1]),
    ("bmul_f32", [3696, 1, 1]),
    ("badd_f32", [3696, 1, 1]),
    ("fast_max_f32", [14784, 1, 1]),
    ("fast_sum_f32", [14784, 1, 1]),
    ("ampere_sgemm_128x128_nt", [1, 1, 192]),
    ("ampere_sgemm_128x128_tn", [1, 1, 192]),
    ("ampere_sgemm_128x128_nn", [1, 1, 192]),
    ("ampere_sgemm_128x64_nn", [4, 29, 7]),
    ("dropout_fwd_f32", [3696, 1, 1]),
    ("adamw_moment_update_f32", [4, 1, 1]),
    ("bmul_f32", [1112, 1, 1]),
    # The ONE real `ucopy_f32` row at the `out` tier shape, for the
    # `outside_signature_plausibly_attention` field's own "a present (not
    # absent) PERMUTE/RESHAPE bucket" test — see below.
    ("ucopy_f32", [924, 1, 1]),
    # The ONE real `badd_f32` row at the `out` tier shape, needed by
    # `test_badd_ladder_launches_per_step_by_tower`
    # (`Ln1VsD2DifferentialTests`) to assert this leg's own point on the
    # cross-leg launch-count ladder at that grid.
    ("badd_f32", [924, 1, 1]),
]
```

`clip-text-A1`'s leg parameters this fixture's tests derive signatures from
(`crates/jammi-bench`'s own `manifest.json` for this leg, also real):
`batch=8` (→ `rows=24`), `max_seq_length=77`,
`fusible_site_census={"lora_sites_wrapped": 48, "layer_norms": 25,
"gelu_seam_calls_per_forward": 0}` (→ `layers=48/4=12`). Combined with this
module's declared constants for the stock `laion2b_s34b_b79k` ViT-B-32 text
tower (`width=512, heads=8, mlp_ratio=4`), the two declared shapes this cut
exercises are:

- GELU shape `[rows*S, 4*width] = [1848, 2048]`, `3,784,704` elements
  (`3696 * 1024` — the `usigmoid_f32`/`affine_f32`/`bmul_f32`/`badd_f32` rows
  at `grid=[3696,1,1]` all sit exactly here).
- Attention softmax reduction row count `rows*heads*S = 24*8*77 = 14,784`
  (the `fast_max_f32`/`fast_sum_f32` rows at `grid=[14784,1,1]`).
- Attention batched-matmul count `rows*heads = 192` (the three
  `ampere_sgemm_128x128_{nt,tn,nn}` rows at `grid=[1,1,192]`).

Rows chosen to be NEGATIVE controls for the GELU shape gate: `badd_f32`
at the GELU shape (lands OUT of `C-GELU` — see `profile_421_attribute.py`'s
module doc, "Deliberately EXCLUDED"; it instead lands the tier-suffixed
`BIAS/RESIDUAL-MLP` bucket, since a bias-add is intrinsically tied to its
site's own width, never the activation itself); `ampere_sgemm_128x64_nn`
at `grid=[4,29,7]` (no `192` in any dimension) -> `BASE-GEMM`;
`dropout_fwd_f32` -> `DROPOUT`; `adamw_moment_update_f32` -> `OPTIMIZER`.
The SECOND `bmul_f32` row at `grid=[1112,1,1]` (`1112*1024=1,138,688`
elements — NOT the GELU shape) FALLS THROUGH the GELU-shape gate (still
excluded from `C-GELU`) into the attention tensor's own elementwise tier
(`attn_shape_elements() = rows*heads*seq*seq = 1,138,368`, which
`1,138,688` covers) -> `C-ATTN-clip-text`, per `classify_kernel`'s
documented "fall through" behavior. The `ucopy_f32` row (`grid=[924,1,1]`,
the `out` tier shape) is real and byte-for-byte from the same export:
`ucopy_f32`/`copy2d_f32` get their OWN `PERMUTE/RESHAPE` bucket (module
doc, "split by kernel NAME-CLASS"), and this row lets the fixture exercise
`outside_signature_plausibly_attention`'s mirroring of a NON-`"absent"`
`PERMUTE/RESHAPE` entry directly (the BF16 `clip-text-A2` fixture's own
rows never include a `ucopy`/`copy2d` name, so that fixture alone could
not exercise this). The `badd_f32` row at that same
`grid=[924,1,1]` shape is likewise real and byte-for-byte from the same
export, added so this leg's own point on the `badd_f32` launch-count
ladder (module doc, "D1-vs-D2 differential") is present here alongside
the `D1`/`D2` legs' own points.

Only `badd_f32`'s own `launches_per_step` at `grid=[924,1,1]` is asserted
as a literal value anywhere (`test_badd_ladder_launches_per_step_by_tower`,
reading this leg's own point on the cross-leg launch-count ladder straight
off this committed fixture); no other row's `us_per_step`/
`launches_per_step`/`share` field is asserted as a literal expected value
anywhere in `test_profile_421_attribute.py` — otherwise only structural
invariants (which chain a row lands in, that shares sum `<= 1`, that every
declared chain is present) are asserted, per this crate's "never
transcribe a timing number into a test" convention. The timing fields are
kept in the fixture only because they are what a real export actually
contains (never deleted to make the fixture "look" hermetic).
