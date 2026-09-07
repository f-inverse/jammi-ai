# Provenance — `profile_421_clip_text_a2/kernels.json`

Cut from the REAL `clip-text-A2` leg's nsys census export, pulled by the
lead from pod `p421` run 2 at `git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`
(`ci/scripts/perf/profile_421_legs.sh`, `--task text_embedding`, **BF16**,
wire default LoRA/rank, the contract's `A2` decision leg):

```
scratchpad/pod421-run2/legs/clip-text-A2/census.json
```

Same box/tooling as `profile_421_clip_text_a1` (`nsys 2025.3.2.474-253236389321v0`,
`NVIDIA A100-SXM4-80GB`). Cut with a short `python3 -c` script reading the
pulled file directly and writing out ONLY the 22 `by_kernel_and_grid` rows
this crate's tests need, byte-for-byte from the real export.

`clip-text-A2`'s leg parameters (its own real `manifest.json`): `batch=8`
(-> `rows=24`), `max_seq_length=77`, `dtype="bf16"`,
`fusible_site_census={"lora_sites_wrapped": 48, "layer_norms": 25,
"gelu_seam_calls_per_forward": 0}` — IDENTICAL signature values to
`clip-text-A1` (same tower, same wire defaults; only the dtype differs),
so this fixture exercises the SAME declared shapes
(`gelu_shape_elements=1848*2048`, `attn_softmax_rows=14784`,
`attn_batch_count=192`, `attn_shape_elements=1112*1024`-range,
`ln_row_count=1848`) against a DISJOINT, BF16-suffixed kernel vocabulary.

## What this fixture is FOR: the twin rule and the batched-grid rule, BF16

- `layer_norm_fwd_bf16_biased`/`layer_norm_bwd_dx_bf16` -> `C-LN` by name
  (the exact bf16 twins of pass 1's f32 names — `LN_KERNEL_NAMES` lists
  both explicitly, no regex needed since there are only two).
- `usigmoid_bf16` -> `C-GELU` (exclusive, any grid); `affine_bf16`/
  `bmul_bf16` -> `C-GELU` ONLY at `grid=[3696,1,1]` (the `mlp` shape tier,
  identical to `clip-text-A1`'s since `rows`/`seq`/`width` are unchanged by
  dtype). A SECOND `affine_bf16` row at `grid=[1112,1,1]` (the attention
  tensor's own shape) is included as a NEGATIVE control for the GELU-shape
  gate that is also a POSITIVE control for the attention elementwise tier:
  it must land `C-ATTN-clip-text`, not `C-GELU` and not `UNATTRIBUTED`
  (module doc, "fall through" — this is the exact case that changed
  behavior from pass 1, where the equivalent f32 row was asserted `None`).
- `fast_max_bf16`/`fast_sum_bf16` at `grid=[14784,1,1]` -> `C-ATTN-clip-text`
  (softmax reduction rows, bf16 twin of pass 1's rule); a THIRD
  `fast_sum_bf16` row at `grid=[8,1,1]` (neither the softmax row count nor
  `ln_row_count=1848`) must land `LOSS/REDUCE` — the new fallback pass 1
  did not have (pass 1 would have called this `UNATTRIBUTED`).
- `badd_bf16` at `grid=[1112,1,1]` (the attention tensor shape) ->
  `C-ATTN-clip-text`; at `grid=[924,1,1]` (the `out` activation tier) ->
  `BIAS/RESIDUAL-OUT` (pass 3, finding 2 — the tier-suffixed name-class
  bucket `badd_*` gets, replacing pass 2's coarse `ELEMENTWISE-OUT`).
- `Kernel2` (an ANONYMOUS/unsymbolized CUDA launch — no name at all) at
  `grid=[2,1,192]` -> `C-ATTN-clip-text` via the batched-grid RELATIONAL
  rule alone (`192 = rows*heads` sits at grid POSITION 2 — pass 3, finding
  1 — the rule does not care that the kernel has no name); a SECOND
  `Kernel2` row at `grid=[16,1,10]` (no `192` at position 2, and `dim[1]=1`
  fails the anonymous-GEMM-tile test too) stays `UNATTRIBUTED` AND is now
  (pass 3, finding 3 — `Kernel2` is REMOVED from `KNOWN_KERNEL_NAMES`)
  flagged UNKNOWN at `share_gpu_busy=1.40%`, exceeding
  `UNKNOWN_KERNEL_SHARE_LIMIT` and making this ENTIRE FIXTURE'S leg
  `INVALID` when read through `attribute_leg` — this fixture is the
  committed, small-scale reproduction of the real `clip-text-A2` leg's own
  INVALIDATION (module doc, "Consequence, measured"); `AttributeCensusBf16
  FixtureTests` therefore no longer asserts `unknown == []` for this
  fixture (pass 2 did; pass 3's own `UnknownKernelGateOnRealFixtureTests`
  asserts the INVALIDATING reason instead).
- `cast_bf16_f32`/`cast_scale_bf16_f32` -> `CAST` by name
  (`"cast" in name.lower()`); `cast_add_bf16` -> `CAST` too, EVEN THOUGH it
  has no f32 twin in `KNOWN_KERNEL_NAMES` (`BF16_ONLY_KERNEL_NAMES`, hand
  admitted from this same real export); `cast_u8_bf16` at
  `grid=[1112,1,1]` (the attention tensor's OWN shape) -> `C-ATTN-clip-text`
  now, NOT `CAST` (pass 3, finding 1 corrects the priority: a cast-named
  row is shape-gated to the attention chain FIRST; a cast row at any OTHER
  shape still lands the generic `CAST` bucket — see the `A1` fixture's own
  `cast_u8_f32` row at the SAME shape for the f32 analog).
- `adamw_moment_update_f32` (an F32-NAMED kernel, even on this BF16 leg —
  the optimizer state itself stays F32 master weights) -> `OPTIMIZER` by
  name; `dropout_fwd_f32` (also F32-named on a BF16 leg) -> `DROPOUT` by
  name. Both are evidence that these two buckets key off the OPERATION
  name, never the leg's own dtype.
- `gather_u32_bf16` -> `EMBED/GATHER`; `is_u32_bf16` at `grid=[924,1,1]`
  (the `out` activation tier shape) -> `EMBED/GATHER`, NOT
  `BIAS/RESIDUAL-OUT`/`ELEMENTWISE-OTHER-OUT` — another priority-order
  fixture row (u32-typed name checks run before the generic activation-tier
  fallback).
- `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn` at
  `grid=[4,15,1]` (no `192` at position 2) -> `BASE-GEMM`, via
  `GEMM_NAME_RE` for CLASSIFICATION (unchanged from pass 2). For ADMISSION
  (`is_known_kernel_name`), pass 3 (finding 3) REMOVES the `"gemm"`-
  substring shortcut: this exact 32-character tile-variant string is now
  hand-listed literally in `KNOWN_KERNEL_NAMES` (one of eight bf16 GEMM
  tile names actually observed across `clip-text-A2`/`clip-vision-A2`),
  never admitted by pattern alone.

None of the 22 rows' `us_per_step`/`launches_per_step`/`share` fields are
asserted as literal expected values anywhere in the test suite — only
structural invariants (which chain a row lands in, that shares partition
`gpu_kernel_us_per_step`, that every declared chain is present-or-absent)
are asserted.
