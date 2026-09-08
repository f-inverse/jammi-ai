#!/usr/bin/env python3
"""Issue #421 tower-profile chain ATTRIBUTION (post-export unit; CONTRACT
`scratchpad/contract-421-profile.md` v2.5 `### Attribution` / `§D3`).

`profile_421_merge.py` decides whether a leg is a DATUM at all (the
positive-proof equation, checkpoint identity, the wall/front/busy
decomposition). This module answers a DIFFERENT question over the SAME
`$OUT_DIR/legs` tree: of one leg's own `gpu_kernel_us_per_step`, how much
belongs to which named chain, and how much is `UNATTRIBUTED`? `main` now
REQUIRES `--merge-json` (the merge's own report) — a leg's `decision_grade`
is a conjunction of what THIS module measures and what the merge already
validated (see `leg_decision_grade`'s doc for the exact split), so this
module refuses to run against a legs tree the named merge report does not
describe (see `main`).

## `C-ATTN`/`C-LN`/`C-GELU` — established by NAME/shape rules

Read off the real `clip-text-A1` (F32) export: `layer_norm_fwd_f32_biased`/
`layer_norm_bwd_dx_f32` -> `C-LN` by NAME at any grid; `usigmoid_f32` ->
`C-GELU` by NAME at any grid; `affine_f32`/`bmul_f32` -> `C-GELU` ONLY at
the declared `quick_gelu` tensor shape `[rows*S, 4*width]` (a `badd_f32`
row at the IDENTICAL shape is deliberately EXCLUDED — it is `c_fc`'s own
Linear bias, not the activation); `fast_max_f32`/`fast_sum_f32` ->
`C-ATTN-<tower>` ONLY at the declared softmax reduction row count
`rows*heads*S` (`grid[1]==grid[2]==1`); any GEMM-family kernel (module doc,
"GEMM-family kernel identity" below) -> `C-ATTN-<tower>` when its grid
carries `rows*heads` as one of its three dimensions (a RELATIONAL rule, not
a name list) else `BASE-GEMM`. `C-LORA` has NO literal GPU kernel name
anywhere in this export — the contract's own method for it is never
by-name: the `A1 - D2` busy/wall delta. `KNOWN_KERNEL_NAMES` gates: a
kernel name outside the validated vocabulary above 1% of a leg's busy makes
the WHOLE LEG `INVALID` rather than silently folding into `UNATTRIBUTED`.

The name/shape rules above, alone, leave roughly 85% of `clip-text-A1`'s
busy `UNATTRIBUTED`, so the validity gate's own bound (`UNATTRIBUTED <= 5%
of gpu_busy`) needs the additional named buckets below to be satisfiable
at all.

## Named-bucket extension (§Attribution/§D3)

Read off the real `clip-text-A2` (BF16), `clip-text-D1`, `clip-text-D2`,
`clip-vision-A1`/`A2`/`D1`/`D2` exports (pod `p421` run 2, tip
`c1b0b0ba`) and names a set of EVIDENCE-BACKED buckets that absorb the
overwhelming majority of what the name/shape rules above leave
`UNATTRIBUTED`.

## Priority-fix rules, plus the HTSAT mapping

Every rule below is grounded in a specific `(kernel, grid)` row from a real
export — see each subsection. The F32-only caveat states its own priority
order explicitly; the D1-vs-D2 differential's own split is MEASURED, never
carried as a literal expectation; `share_of_baseline_wall`'s own
denominator convention is pinned; every fixture row is copied byte-exact
from a real export; and every evidence paragraph cites a subsection that
still exists — plus the HTSAT kernel-name<->shape mapping itself (see
"HTSAT" below), read off the real `htsat-{A1,A2,D1,D2}` exports.

### `C-ATTN-<tower>`: the pre-registered signature does not move

The signature stays EXACTLY what the contract pre-registered: kernels at
the `[24, h, S, S]` element count (scores/probs elementwise, softmax
fwd/bwd, dropout on probs, the cast/compare rows at that shape) plus the
batched GEMMs whose grid carries `rows*heads`. Two rules below apply, both
evidenced against the real exports:

1. **`cast`-named rows are shape-gated, not name-exclusive.**
   `cast_u8_f32`/`cast_u8_bf16` at the attention tensor's own element count
   (evidenced: `clip-text-A1` `cast_u8_f32 grid=[1112,1,1]`, the exact
   attention-shape grid every other attention elementwise name in this
   module shares) belong to `C-ATTN-<tower>`, not the generic `CAST`
   bucket — `classify_kernel` checks the attention shape FIRST for any
   cast-named row; a cast row at any OTHER shape lands the generic `CAST`
   bucket.
2. **The batched-GEMM grid rule is gated on grid POSITION 2**, the
   position every real match uses (`clip-text-A2`'s
   `cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_{nn,nt,tn}_align1` rows
   at `grid=[2,1,192]`/`[4,1,192]`; `clip-vision-A1`'s
   `magma_sgemmEx_kernel<...>` at `grid=[1,2,288]`; `clip-text-A1`'s
   `ampere_sgemm_*` attention GEMMs at `grid=[1,1,192]`), not "anywhere in
   the 3-tuple" — a plain grid-membership test would falsely match a
   hypothetical non-attention row carrying `192`/`288` in `grid[0]` (a real
   value for some base-projection tile grids at other shapes);
   `classify_kernel` checks `grid[2] == attn_batch_count()` exactly, and
   the test suite carries a NEGATIVE CONTROL — a non-attention row with
   `192`/`288` in `grid[0]` (not `grid[2]`) — proving the membership test
   does NOT fire there.

**`ucopy_*`/`copy2d_*` (permute/reshape copies) do NOT join `C-ATTN`,
named or by grid** — they get their OWN bucket, `PERMUTE/RESHAPE` (see
below), never falling through into whichever activation-tier bucket their
element count happens to match. This module additionally carries a
leg-level diagnostic field, `outside_signature_plausibly_attention`
(`{busy_us, share_wall}`, mirroring that bucket's own numbers), so the
artifact can STATE the knife edge honestly without moving the verdict:
`C-ATTN-clip-text`'s own signature `s_wall` on the decision leg
`clip-text-A1` is measured at `9.87%` — just BELOW the two-sided rule's
`10%` `ACTIVATE` threshold, landing squarely in the 5-10% `UNRESOLVED`
band (its own `UNATTRIBUTED` combined share also exceeds the `5%` DECLINE
bound, so neither side of the rule fires). `PERMUTE/RESHAPE`'s own
`share_wall` on that SAME leg is `1.38%` — if the rule counted permutes as
attention-adjacent, `9.87% + 1.38% = 11.25%` would clear `ACTIVATE`. The
rule as written does NOT authorize counting it, and this module writes
NOTHING that resolves the question either way: **the verdict this leg
lands on stays exactly what `decide_candidate_port` computes from the
SIGNATURE alone (`UNRESOLVED`) — the `9.87%`-vs-`11.25%` knife edge is
STATED via `outside_signature_plausibly_attention`, never adjudicated.**

### The four activation-tensor width tiers, split by kernel NAME-CLASS

Every OpenCLIP block has exactly four LoRA-wrappable sites
(`in_proj`/`out_proj`/`c_fc`/`c_proj`, `crate::open_clip_block`), whose
OUTPUT tensors sit at four distinct, non-colliding element counts:
`qkv = rows*seq*3*width` (`in_proj`'s combined Q/K/V output), `out =
rows*seq*width` (`out_proj`/`c_proj`'s output and the residual stream),
`mlp = rows*seq*mlp_width` (`c_fc`'s output, the same shape `C-GELU`
already keys off). Every non-`C-GELU` elementwise name at these three
shapes is split by kernel NAME-CLASS, never lumped into one bucket per
tier — a single "elementwise tier" bucket would conflate a Linear's own
bias-add with an unrelated permute/reshape copy with (on D1 legs) the
eager LayerNorm's own normalize step:

- `badd_f32`/`badd_bf16` (evidenced: `clip-text-A1` has exactly three
  `badd_f32` grids at the `qkv`/`out`/`mlp` element counts — the sites'
  own bias adds and the residual stream's adds) -> `BIAS/RESIDUAL-QKV` /
  `BIAS/RESIDUAL-OUT` / `BIAS/RESIDUAL-MLP` (tier-suffixed, since a bias
  add is intrinsically tied to its site's own width).
- `ucopy_f32`/`copy2d_f32` (and bf16 twins) at any of the three tier
  shapes -> ONE bucket, `PERMUTE/RESHAPE` (NOT tier-suffixed — a
  permute/reshape copy is not a per-site op the way a bias-add is; it is
  the same reshape/transpose family regardless of which activation width
  it happens to run over).
- Every OTHER name at a tier shape (`affine_f32`/`bsub_f32`/`usqr_f32`/
  `const_set_f32`/... on an A/D2 leg, where `layer_norm_fused` is FUSED)
  -> `ELEMENTWISE-OTHER-<tier>`. This module does NOT guess a mechanism
  role for these (dropout's own inverse-probability scale? a LoRA
  composition add sharing the site's own width, per `§D3`'s scope note
  that `lora_epilogue_fused` fuses on every leg?) without a dedicated
  ablation isolating them — an honestly-labeled catch-all, not a
  transcribed hypothesis.
- `bmul`/`bsub`/`usqr`/`usqrt`/`urecip` (and bf16 twins) AT THE **LN ROW
  COUNT** (`rows*seq`, distinct from every tier shape above) ON A LEG
  WHERE `layer_norm_fused` IS DISABLED (D1; read off the leg's own
  `kernels_disabled`, equivalently the merge's validated
  `kernels_disabled_expected` for a VALID leg — see `attribute_leg`) ->
  `C-LN` (the eager normalize's own mean/var/std/reciprocal steps).
  Evidenced on `clip-text-D1`: `bsub_f32`/`usqr_f32`/`usqrt_f32`/
  `urecip_f32` at `grid=[2,1,1],block=[1024,1,1]` (`total_threads=2048`
  covering `ln_row_count=1848`), ABSENT from `clip-text-A1`/`clip-text-D2`
  (LN fused on both) entirely; `clip-vision-D1` shows the identical
  four-name set at the analogous shape, a SECOND tower confirming the
  rule by formula, not a copied grid number. `bmul` is included in the
  rule (the contract's own naming) even though no real D1 leg pulled so
  far shows it AT this exact shape — the rule fires on a match, never on
  a name alone, so this costs nothing when it does not fire. On an
  A/D2 leg (LN fused), these same five names at the LN row count do NOT
  auto-route to `C-LN` — the eager LN chain literally cannot exist while
  the fused kernel is admitted (`unmatched_disables()` would refuse the
  leg otherwise) — so they fall through to the ordinary tier/param-scale
  checks like any other name.

A dedicated D1-vs-D2 differential test (both real fixture cuts, same
tower) asserts the DIRECTION/ORDERING this rule predicts — toggling only
`layer_norm_fused` must move mass INTO `C-LN` and must move the tier
buckets by LESS than `C-LN` moves — never a literal delta.

### `BASE-GEMM`: GEMM-family kernel identity, by NAME

`kernel_census.py` keys every row on the CUDA export's own DEMANGLED
kernel name (module doc, its own "Kernel identity" paragraph) — a cutlass
GEMM tile instantiation's demangled name always carries its own tile/stage
identity (e.g. `cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nt_align1`,
`cutlass_80_simt_sgemm_32x128_8x5_nt_align1`), never the generic
`Kernel2` template-wrapper name a shorter symbol table would give it.
`GEMM_FAMILY_NAME_RE` recognizes every GEMM-library naming convention this
module has actually observed across all twelve real CLIP/HTSAT legs pulled
from pod `p421` run 2 (tip `c1b0b0ba`):

- `cutlass_\\d+_(simt|tensorop|wmma_tensorop)_\\w*gemm\\w*` — every cutlass
  tile instantiation, e.g. `cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_
  {nn,nt,tn}_align1` (`clip-text-A2`, `grid=[2,1,192]`),
  `cutlass_80_simt_sgemm_32x128_8x5_nt_align1` (`clip-text-A2`,
  `grid=[16,1,10]`), `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_
  {32x1,64x1}_{nn,nt,tn}_align{1,2,8}` (`htsat-A2`, `clip-vision-A2`).
- `ampere_\\w*gemm\\w*` — every `ampere_sgemm_*`/`ampere_bf16_s16816gemm_*`
  tile variant pulled across the eight real CLIP legs and `htsat-A2`.
- `magma_\\w*gemm\\w*` — `magma_sgemmEx_kernel<float, float, float,
  (bool)0, (bool)0, ...>(...)`'s own template signature (`clip-vision-A1`,
  `htsat-A1`), matched on the `sgemm` substring its template arguments
  surround.
- `\\w*splitKreduce\\w*` — `cublasLt::splitKreduce_kernel<...>(...)`'s own
  split-K reduction accessory launches (`clip-text-A1`'s `BASE-GEMM` grid
  reconciliation, below); its own demangled name carries a full template
  signature, never the bare `splitKreduce_kernel` identifier alone, so
  this is a SUBSTRING match, not an exact one.

A row whose name matches `GEMM_FAMILY_NAME_RE` and does NOT carry the
attention batch count at `grid[2]` is `BASE-GEMM`; one that DOES carry it
is `C-ATTN-<tower>` (the batched-GEMM grid rule above — the SAME
grid-position-2 check every real match uses, checked BEFORE the GEMM-name
branch so a GEMM-family kernel at the attention grid still lands
`C-ATTN-<tower>`, never `BASE-GEMM`). **`GRAD-BOOKKEEPING`'s own
parameter-scale rule (module doc, "`GRAD-BOOKKEEPING`") is gated on 1-D
launch geometry** (`grid[1]==1 and grid[2]==1`, i.e. `grid=[ceil(N/b),1,1]`
— a genuine elementwise/bookkeeping launch never tiles a second or third
grid dimension): evidenced by `clip-text-A2`'s
`cutlass_80_simt_sgemm_32x128_8x5_nt_align1 grid=[4,1,24]`, whose
`total_threads=12,288` EXACTLY matches `LORA_RANK*3*width` (a param-scale
element count) yet is plainly a 3-D tiled GEMM launch, not a 1-D
bookkeeping op — an ungated parameter-scale rule would sweep it into the
parameter-scale bucket by shape coincidence; the GEMM-family name check
above (which runs first) already routes it to `BASE-GEMM` regardless, and
the `grid[1]==1 and grid[2]==1` gate additionally keeps any NON-GEMM name
at this same element count out of `GRAD-BOOKKEEPING` unless its own launch
geometry is genuinely 1-D.

**A name outside `GEMM_FAMILY_NAME_RE` that no other rule classifies
COUNTS toward the 1%-of-busy unknown-kernel gate** — `is_known_kernel_name`
admits a name via `GEMM_FAMILY_NAME_RE` (any GEMM-family instantiation,
regardless of its own specific tile/stage suffix), the explicit
`KNOWN_KERNEL_NAMES`/`BF16_ONLY_KERNEL_NAMES` ground truth (every
non-GEMM name this module has hand-verified against a real export), or the
BF16-twin rule; a name matching none of these is unknown, full stop, no
matter how GEMM-shaped it merely looks. Measured on the real, regenerated
`clip-text-A2` census: `cutlass_80_simt_sgemm_32x128_8x5_nt_align1
grid=[16,1,10]` (`share_gpu_busy=1.40%`) and the SAME kernel at
`grid=[4,1,24]` (`share_gpu_busy=1.37%`) both match `GEMM_FAMILY_NAME_RE`
and land `BASE-GEMM` cleanly — no unknown-name finding, no leg
invalidation; the equivalent `clip-vision-A2` row
(`grid=[6,1,18]`, `share_gpu_busy=1.43%`) resolves identically. Both A2
legs are therefore decision-grade candidates alongside their F32 twins
(see `decide_candidate_port`'s own per-leg U/grade output).

### Realized-gain chains are NOT chain-partition members

`C-LORA`'s `A1-D2` delta is never attached directly into the A-leg's own
`chains` dict, alongside the partition-member buckets — a busy/wall
DELTA between two different leg configurations is not a share of any one
leg's own busy, and (since the fused configuration is usually cheaper)
the delta is frequently NEGATIVE, which would show up as a negative
`share_gpu_busy` in the very column this module's own tests assert is
`>= 0` for every partition member. `C-LORA` and `C-LN`'s
realized-fused-vs-eager numbers live OUTSIDE every leg's `chains` dict
entirely, in a SEPARATE top-level `realized_gains` list
(`compute_realized_gains`)
— one entry per `(chain, tower)`, carrying `busy_delta_us_per_step`
(`eager - fused`, so POSITIVE when eager is slower, as every real leg
pulled so far shows), `wall_delta_s_per_step` (sourced from the NAMED
`--merge-json` report's own `per_step.wall_s_per_step` per leg — "the run
reports' `train_run_wall_s` differenced exactly as the merge does",
contract; never re-derived from this module's own census-only `wall_s`),
`share_of_baseline_wall`, `fused_twin_wall_s`, `share_of_fused_twin_wall`,
and a `direction` string stating which leg was slower. `C-LORA` = `A1` vs
`D2`; `C-LN` = `D2` vs `D1` (D1 additionally disables LoRA, so `D1-D2`
isolates LN alone on CLIP — §D3).

**Denominator convention, pinned once here:**
`share_of_baseline_wall` ALWAYS divides the wall delta by the TOWER'S OWN
SHIPPED `A1` wall (`baseline_wall_s`/`baseline_leg_id`), regardless of
which two legs the pair actually compares — the shipped configuration is
always `A1` (wire-default LoRA, every fusible seam admitted), never the
pair's own "fused" leg (`C-LN`'s pair is `D1` vs `D2`; `D2` is NOT what
ships, it still has LoRA disabled). `fused_twin_wall_s`/
`share_of_fused_twin_wall` are EMITTED SEPARATELY and divide by the pair's
own fused leg's wall instead (for `C-LORA` this equals `A1`, so the two
share numbers COINCIDE; for `C-LN` it is `D2`'s own wall, the number pass
2 originally mislabeled `share_of_baseline_wall`). Both `None` if the
relevant wall could not be read from `--merge-json`. `C-GELU-HTSAT`'s
realized gain (`D1` vs `D2` on `htsat`, since `htsat`'s own `D1` disables
`layer_norm_fused`+`gelu_erf_fused` together — module doc, "HTSAT") reuses
this exact same `REALIZED_GAIN_SPECS`/`compute_realized_gains` machinery;
`C-LN`+`C-GELU-HTSAT`'s joint `D1`-vs-`D2` delta on `htsat` is reported
under the SAME `C-LN` chain-name entry (both are disabled together on that
tower's `D1`, so this module cannot isolate them from a single leg pair —
stated honestly, not force-split).

### `decision_grade` now REQUIRES the merge's own verdict

Contract, Validity gate, verbatim: "kernel table present; counter
equations hold; `--expect-kernels-disabled` satisfied on D legs;
UNATTRIBUTED <= 5% of gpu_busy (C-ATTN-HTSAT is attributed, so this is
satisfiable)." `profile_421_merge.py` enforces "kernel table present",
"counter equations hold", and "`--expect-kernels-disabled` satisfied on D
legs" (plus checkpoint identity and the wall/front/busy decomposition,
which the contract's method section separately requires) — THIS module
enforces "UNATTRIBUTED <= 5% of gpu_busy" and nothing else. Neither module
alone can certify the full gate, so `leg_decision_grade` now takes the
corresponding row from a NAMED `--merge-json` report and requires ALL of:
this module's own `verdict == VALID` (which folds in `census_ok`, since
`attribute_leg` refuses a leg whose manifest reports `census_ok is not
True`), this module's own `UNATTRIBUTED share_gpu_busy <= 5%`, AND the
merge row's `verdict == VALID`. `main` REFUSES (`exit 1`) if `--merge-json`
is missing, unreadable, or does not name EVERY leg found under
`--legs-dir` — `--legs-dir`'s own leg set must be a SUBSET of the merge
report's (a leg with no corresponding merge row can never be certified,
the unsafe case this refuses); the merge report MAY additionally cover
legs this module does not attribute at all (e.g. an HTSAT leg, out of
this pass's own scope — module doc, "HTSAT") without tripping the refusal.

### `limits`: the report's own recorded validity-gate bounds

`build_report`'s own top-level `limits` block (`unattributed_decision_
grade_limit`, `unknown_kernel_share_limit`) records THIS module's live
validity-gate constants (`UNATTRIBUTED_DECISION_GRADE_LIMIT`,
`UNKNOWN_KERNEL_SHARE_LIMIT`) AT THE TIME this report was built — never
re-derived downstream, never parsed back out of a leg's own
`decision_grade_reason` string. A downstream reader
(`profile_421_artifact.py`) that needs to know exactly what bound an
ALREADY-MEASURED run was judged against reads it from HERE, by name,
refusing if the key is absent, rather than parsing a leg's own English
`decision_grade_reason` (a leg that failed for an unrelated reason records
no such string at all — there would be nothing to parse) or trusting the
LIVE import directly (which could have moved since this run was
measured). `SCHEMA_VERSION` bumped `3 -> 4` for this new top-level key.

### Numbers in this doc are re-derived, never transcribed

Every number in this docstring is RE-DERIVED from the real census exports,
never carried over as a literal expectation in any test. Measured on the
real, full (not fixture-cut) `clip-text-A1` census: `BASE-GEMM = 35.08%`
of busy (the chain total, NOT one row); the single NAMED row
`ampere_sgemm_128x64_nn grid=[4,29,7]` is `8.08%` of busy alone — these are
two DIFFERENT numbers, never conflated. On `clip-text-A2`, `BASE-GEMM =
15.48%` (`~15.5%`) while `C-ATTN-clip-text = 19.41%` is the LARGEST bucket
on that leg (BF16 halves every elementwise op's byte traffic, shrinking
`BASE-GEMM`'s share relative to attention's own fixed kernel-launch
overhead) — read GEMM-family names by `GEMM_FAMILY_NAME_RE` (module doc,
"`BASE-GEMM`: GEMM-family kernel identity, by NAME"), so this leg is
decision-grade, both shares feed `decide_candidate_port` directly. See
`main`'s own documented run, pasted in the hand-off, for the exact
`UNATTRIBUTED`/`BASE-GEMM` share on EVERY one of the eight real CLIP legs.

**BASE-GEMM grid -> layer reconciliation, `clip-text-A1`:** `BASE-GEMM`'s
18 distinct `(kernel, grid)` rows carry `580` launches/step (`652` total
GEMM-family launches across 21 rows, minus the `72` launches on the three
rows carrying the attention batch count at `grid[2]=192`). The naive
expectation from the declared architecture (4 LoRA-wrappable sites x 12
layers x 3 logical GEMMs per site — forward, grad-input backward,
grad-weight backward — for the BASE Linear weight alone) is `4*12*3 =
144` logical matmuls; `580` does NOT reconcile to `144` (nor to any small
integer multiple of it) and this module does not force a fit. The most
likely accounting for the gap, NOT independently verified per-row here:
the four sites have four DIFFERENT `(K, N)` weight shapes, so cuBLAS's
own heuristic selects a DIFFERENT tile-variant kernel (a different
`(kernel, grid)` row) per site x direction, multiplying one "logical
matmul" into several distinct rows; `splitKreduce_kernel`'s own 5 rows
are accessory launches for certain large-K algorithms, not separate
logical GEMMs; and the wire-default rank-8 LoRA composition
(`lora_epilogue_fused` fuses the ADD, not necessarily the underlying rank-8
matmuls themselves) plausibly still dispatches its own small GEMMs through
cuBLAS. This module states the gap rather than resolving it.

**`eq_f32`'s mechanism is a HYPOTHESIS, not an asserted mask-compare
role**: `clip-vision-A1` carries the IDENTICAL `eq_f32` row (same shape,
same order of magnitude) as `clip-text-A1`, yet
OpenCLIP vision attention has no learned/data-dependent additive mask at
all (the encoder-side cascade synthesizes an all-zero padding mask for
vision — contract, "candidate ports"); an op that is genuinely mask-only
could not appear identically on a tower with no mask to compute. `eq_f32`
is therefore stated as a HYPOTHESIS — most plausibly part of the
softmax backward's own routing (e.g. a numerically-stable-softmax
backward's own equality/selection step), not a mask compare — and is
NOT asserted as mask-related anywhere in this module or its tests.
`cast_u8_f32`'s own role is plausible mask-adjacent on `clip-text` alone,
but this module does not assert that either — both names are classified
by SHAPE, never by an asserted mechanism.

### Evidence paragraphs every still-live code comment cites by name

**`GRAD-BOOKKEEPING` (distinct from `OPTIMIZER`):** `adamw_moment_update_
f32`/`adamw_theta_update_f32` (any grid) are `OPTIMIZER` by NAME —
`adamw_step_fused` is already admitted on
every A leg, so a kernel launched under that name IS the optimizer, full
stop. Measured on the real, full `clip-text-A1` census: THIRTY-TWO other
elementwise rows (`badd_f32`/`bmul_f32`/`bsub_f32`/`usqr_f32`/
`cast_u8_f32`/... at the three tiny grids `4`/`16`/`12`, i.e.
`total_threads` `4096`/`16384`/`12288` — EXACTLY `LORA_RANK*width`,
`LORA_RANK*mlp_width`, `LORA_RANK*3*width` for `LORA_RANK=8`) carry
`3.80` percentage points of that leg's busy — a MATERIAL share the
name/shape rules alone (module doc, "`C-ATTN`/`C-LN`/`C-GELU` — established
by NAME/shape rules") leave `UNATTRIBUTED`. These are NOT the optimizer (`adamw_step_fused`
already covers that by name) — they sit at the trainable PARAMETER
tensors' own element counts (LoRA A/B matrices, Linear-bias-scale
gradient/weight-decay/grad-norm-clipping bookkeeping), a fundamentally
DIFFERENT role from an activation-tensor elementwise op. This module
names this family `GRAD-BOOKKEEPING` (not `OPTIMIZER`) and does not
further split "gradient zeroing" from "weight decay" from "grad-norm
clip" from a single leg's census (all three are plausible roles at this
IDENTICAL element count) — an honestly-labeled catch-all, not a
transcribed mechanism, same discipline as `ELEMENTWISE-OTHER-<tier>`.

**`C-ATTN-<tower>` attention-elementwise, by SHAPE:** `badd_f32`/
`bdiv_f32`/`bmul_f32`/`bsub_f32`/`cast_u8_f32`/`eq_f32`/`uexp_f32`/
`usqr_f32`/`affine_f32` (and `*_bf16` twins) ALSO launch at the declared
attention tensor's own element count, `attn = rows*heads*seq*seq`
(`= 1,138,368` on `clip-text-A1`, `rows=24,heads=8,seq=77`; `= 720,000` on
`clip-vision-A1`, `heads=12,seq=50`) — the additive mask, the
`1/sqrt(head_dim)` scale, the boolean-mask cast/compare feeding it (a
HYPOTHESIS — see "`eq_f32`'s mechanism" above), and the backward pass's
own `exp`/`square` terms. Evidence: on `clip-text-A1`, all nine names
share a row at `grid=1112` (`1112*1024=1,138,688`, the smallest multiple
of the `1024` block covering `1,138,368` elements); on `clip-vision-A1`
the same nine names share a row at `grid=704` (`704*1024=720,896`,
covering `720,000`) — a SECOND tower confirming the rule generalizes by
FORMULA (`rows*heads*seq^2`), not a hand-copied grid number.

**`C-PATCH-EMBED` (`clip-vision`/`htsat` only): the conv-as-matmul
patchify stem.** `im2col_f32`/`im2col_bf16` — OpenCLIP's ViT patch
embedding and HTSAT's own mel-spectrogram-to-patch unfold are both a
strided `Conv2d`, implemented as `im2col` (unfold) followed by a GEMM; the
GEMM half is already `BASE-GEMM` (or `C-ATTN` if its grid happens to carry
the batch count, which it does not here) via `GEMM_FAMILY_NAME_RE`, and
`im2col_f32` itself gets its own small bucket by name. Never appears
on `clip-text` (no convolution there).

**`LOSS/REDUCE`: `fast_sum`/`fast_max` at every OTHER row count.**
Contract: "`fast_sum`/`fast_max` NOT at the softmax row count" is its own
named bucket, not `UNATTRIBUTED`. After the softmax-row and LN-row checks
both fail, `fast_sum_f32 grid=[1,1,1]` (`launches_per_step=96` on
`clip-text-A1`, `1.61%` of busy) is the largest single row this bucket
absorbs — the triplet loss's own margin/logit reductions and/or
gradient-norm bookkeeping the optimizer needs; this module does not, and
cannot from a single leg's census, further split "loss" from "grad-norm"
reductions sharing the identical degenerate `grid=[1,1,1]` shape, so both
are named `LOSS/REDUCE` together, honestly, rather than guessed apart.

**`CAST`: any kernel name containing `"cast"`, at any shape OTHER than the
attention tensor's own** (the attention-shape check runs FIRST for any
cast-named row — module doc, "`C-ATTN`/`C-LN`/`C-GELU`", rule 1).
`cast_f32_f32`/`cast_u8_f32`/`cast_bf16_f32`/`cast_f32_bf16`/
`cast_add_bf16`/`cast_scale_bf16_f32`/`cast_u8_bf16`/
`scaled_cast_add_f32_f32`/`scaled_cast_add_bf16_f32` — matched by NAME,
any OTHER grid, any tower. `scaled_cast_add_f32_f32`'s own grid pattern is
suggestive of a LoRA-composition-adjacent op — grids vary by leg and by
tower (the rule is name-based, not grid-based) — but this module does not
claim that role without a dedicated ablation.

**Declared per-tower architecture constants (`TOWER_ARCH`).** `width`,
`heads`, and `mlp_ratio` for `clip-text`/`clip-vision` are DECLARED, not
witnessed — the stock `laion2b_s34b_b79k` ViT-B-32 checkpoint's own public
config (`width=512/heads=8` text, `width=768/heads=12` vision,
`mlp_ratio=4` both — OpenCLIP's own `visual.transformer.width`-style
fields). `declared_seq=50` for `clip-vision` (the ViT-B-32 patch grid,
`(224/32)**2 + 1` CLS token) is likewise declared; `clip-text`'s own `seq`
is WITNESSED instead (`manifest.max_seq_length`), since the text tower's
sequence length is a wire/task parameter, not a checkpoint constant. Every
`Signature.source` string in this module's own JSON output distinguishes
`witnessed:...` from `declared_constant:...` so a consumer never confuses
the two provenances.

**`EMBED/GATHER`: the `u32`-typed index/token-id kernels, by name.**
`gather_u32_f32`/`sa_u32_f32`/`ucopy_u32`/`fast_argmax_u32`/`is_u32_f32`
(and bf16 twins where observed) are the only kernels operating on `u32`
(token-id / index) data in this export — embedding-table gather, its
scatter-add gradient, and the EOT-position equality check. All negligible
(`<0.1%` combined on every CLIP leg pulled) but unambiguous by
dtype-in-name.

**Memcpy/memset: reported, never folded into the busy partition.**
Contract §D2: "nsys memcpy rows reported separately from gpu_busy." The
census's own `memcpy_per_step`/`memset_per_step` top-level fields (counts
and `us`, NOT part of `by_kernel_and_grid` or `gpu_kernel_us_per_step`)
are copied onto this module's leg row verbatim as `memcpy_memset` for
visibility (`attribute_leg`) — they never enter any chain's
`share_gpu_busy` denominator.

### D1-vs-D2 differential: the measured split

The module doc ("split by kernel NAME-CLASS", above) already states the
DIRECTION rule (`ln_disabled` gates `LN_EAGER_EXTENDED_KERNEL_NAMES` at
`ln_row_count`). By-name `C-LN` moves in that direction on the toggle but
does not capture the whole `D1-D2` busy delta on either CLIP tower —
`ELEMENTWISE-OTHER-OUT` and `BIAS/RESIDUAL-OUT` are the two buckets the
remainder leaks into (mechanism below). The realized `C-LN` gain this
module reports (`compute_realized_gains`) is deliberately the
WHOLE-CHAIN `D1-D2` busy/wall delta, not a re-derivation from the by-name
`C-LN` bucket alone — the gain number is correct even though the by-name
attribution under-covers the mechanism.

**Mechanism hypothesis, evidenced on both towers:** eager LayerNorm's own
final affine step (`gamma * x_hat + beta`) is NOT a distinct named kernel
— it lands as an ordinary `badd_f32` launch at the `out` activation
tier's FULL width (`rows*seq*width`), the SAME grid the site's own
Linear-bias-add and the residual-stream's add already share. `badd_f32`'s
own `launches_per_step` at that grid forms a three-point ladder as fusion
is progressively disabled: `337` launches on `A1` (LoRA+LN both fused) ->
`633` on `D2` (LoRA disabled, LN still fused) -> `874` on `D1` (both
disabled) on `clip-text` (`grid=[924,...]`), and `337` -> `633` -> `865`
on `clip-vision` (`grid=[900,...]`) — read off the real fixture cuts and
asserted by `test_badd_ladder_launches_per_step_by_tower`. The `D1` leg's
own eager LN affine step is the extra launches landing in
`BIAS/RESIDUAL-OUT` on each tower, and eager LN's remaining
mean/var/normalize steps (excluding the affine) land in
`ELEMENTWISE-OTHER-OUT` (the true majority of the leaked delta on both
towers). Neither leak bucket's own share is asserted as a "less than"
bound against `C-LN`'s share anywhere in this module's tests (see
`Ln1VsD2DifferentialTests`) — the test states what THIS section states:
`C-LN` moves INTO on the toggle, and the two leak buckets are NAMED,
never silently smaller.

## HTSAT (mapping read off the real `htsat-{A1,A2,D1,D2}` exports)

Pod `p421` run 2 additionally pulled all four real HTSAT legs (`htsat-A1`,
`htsat-A2`, `htsat-D1`, `htsat-D2`) — the SAME `A1`/`A2`/`D1`/`D2` legs
table CLIP uses, contract §D3. `derive_htsat_signatures` derives FOUR
per-stage signatures from `manifest.batch` (witnessed) plus the public
`laion/clap-htsat-fused` config's own declared Swin geometry
(`HTSAT_DEPTHS=(2,2,6,2)`, `HTSAT_HEADS=(4,8,16,32)`, `HTSAT_WINDOW_SIZE=8`,
`HTSAT_FINAL_STAGE_DIM=768`, `HTSAT_SPEC_SIZE=256`, `HTSAT_PATCH_SIZE=4`),
cross-checked against the real export BEFORE a single classification rule
was written (this module's own §D3 discipline, applied to itself — see
`fixtures/profile_421_htsat_a1/PROVENANCE.md` for the full per-stage
table and the evidence behind every element count).

`classify_htsat_kernel` is a SEPARATE function from `classify_kernel`
(not a branch inside it) because HTSAT's signature is PER-STAGE (four
distinct widths/token/head counts) rather than CLIP's one declared shape
per tier:

- **`C-ATTN-HTSAT`**: the batched-window-attention GEMMs' grid carries
  `grid[2] == rows*windows_s*heads_s` for SOME stage `s` (the SAME
  name-independent, grid-position-2 rule CLIP's `C-ATTN-<tower>` uses) —
  evidenced on TWO different GEMM libraries at TWO different stages on
  `htsat-A1` (`ampere_sgemm_128x128_nt grid=[1,1,6144]`, stage 0;
  `magma_sgemmEx_kernel grid=[1,2,1536]`, stage 2). The per-window
  attention score/prob tensor's own elementwise ops (`badd`/`bdiv`/
  `bmul`/`bsub`/`eq`/`uexp`/`usqr`/`cast_u8` — cast rows are shape-gated
  here FIRST too, same priority as CLIP) sit at
  `attn_batch_count(s) * (window_size**2)**2` elements (module doc,
  `HtsatStageSignature.attn_shape_elements`) — evidenced on `htsat-A1`:
  `badd_f32 grid=[24576,...]` (stage 0) and `eq_f32 grid=[6144,...]`
  (stage 2). The softmax reduction (`fast_max`/`fast_sum`) launches
  `block=[64,1,1]` EXACTLY (`window_size**2`) at
  `rows*heads_s*tokens_s` rows for SOME stage — verified per stage on
  `htsat-A1`: the COMBINED `fast_sum_f32`+`fast_max_f32` launches at each
  stage's own row count (`393216`/`196608`/`98304`/`49152`) are `8`/`8`/
  `24`/`8` — EXACTLY `4*depth_s` for `depths=[2,2,6,2]`, a clean per-block
  reconciliation (one softmax needs a max-reduction call and a
  sum-reduction call per block, times two — forward and backward) — OUT
  OF TIER for the fixed-head-dim port scope (contract), still a NAMED,
  MEASURED chain.
- **`C-GELU-HTSAT`**: `gelu_erf_fwd_f32`/`gelu_erf_bwd_dx_f32` by NAME,
  any grid — the FUSED kernel (evidenced on `htsat-A1`: launches sum to
  `12` across the four stage shapes, EXACTLY `Σdepths=12`, the config's
  own `gelu_seam_calls_per_forward`). The EAGER (`D1`) twin is candle's
  own THREE-kernel `gelu_erf` decomposition (`ugelu_erf_f32`/`uerf_f32`/
  `uneg_f32`), shape-gated at `mlp_shape_elements(s) = rows*tokens_s*4*
  dim_s` for SOME stage — evidenced on `htsat-D1`: all three names share
  the IDENTICAL four grids/launch-counts as `htsat-A1`'s fused kernel.
- **`C-LN`**: `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` by name
  (shared CLIP/HTSAT constant, `LN_KERNEL_NAMES`), any grid. The eager
  (`ln_disabled`) reduction reuses `LN_EAGER_EXTENDED_KERNEL_NAMES` at
  `ln_row_count(s) = rows*tokens_s` for SOME stage — with ONE real
  collision this module resolves by BLOCK SIZE, not priority order: `ln_
  row_count(stage=0) == attn_softmax_rows(stage=2) == 98,304` on this
  checkpoint's declared constants, but the attention reduction ALWAYS
  launches `block=[64,...]` while the eager LN reduction launches the
  smallest power of 2 `>= dim_s` instead (`128` for `dim_0=96` — evidenced
  on `htsat-D1`: `fast_sum_f32 grid=[98304,...]` carries BOTH a
  `block=[64,...]` row, present on `htsat-A1` too and unaffected by the
  toggle, AND a `block=[128,...]` row, ABSENT from `htsat-A1` entirely).
  `_is_attn_softmax_reduction_grid` is checked FIRST (module doc) so this
  collision resolves correctly regardless of check order.
- **`C-LORA`**: the SAME `A1`-vs-`D2` busy/wall delta
  `compute_realized_gains` already computes for CLIP — no HTSAT-specific
  code, `by_tower_role` keys on `tower` generically.
- **`BASE-GEMM`/`OPTIMIZER`/`DROPOUT`/`CAST`/`EMBED/GATHER`/
  `LOSS/REDUCE`/`C-PATCH-EMBED`/`PERMUTE/RESHAPE`/`BIAS/RESIDUAL-<tier>`/
  `ELEMENTWISE-OTHER-<tier>`**: the SAME by-name/by-shape rules CLIP uses,
  reused verbatim (`GEMM_FAMILY_NAME_RE`, `EMBED_GATHER_KERNEL_NAMES`,
  `PATCH_EMBED_KERNEL_NAMES`, ...) — HTSAT has no combined-QKV tensor
  (`query`/`key`/`value`/
  `attention_output`/`intermediate_dense`/`output_dense` are all separate
  LoRA sites at the SAME `out`/`mlp` widths CLIP's `out`/`mlp` tiers
  already key off), so only TWO tiers apply, not three.
- **Known, DOCUMENTED, UNRESOLVED collision**: `out_shape_elements(s) ==
  mlp_shape_elements(s+2)` for `s in {0, 1}` is a MATHEMATICAL IDENTITY of
  this architecture (tokens quarter and dim doubles each stage) — grids
  `9216`/`4608` cannot be told apart by shape alone.
  `classify_htsat_kernel` checks the `out` tier before `mlp`
  (deterministic, documented priority), so a row at either grid ALWAYS
  resolves to the `out`-tier bucket; this module does not claim that
  resolution is correct, only that it is DETERMINISTIC and STATED (see
  `fixtures/profile_421_htsat_a1/PROVENANCE.md`, "Known ambiguity"). Every
  HTSAT leg's row also carries `ambiguous_out_mlp_collision` (`attribute_
  leg`, beside `chains`) — the PURELY diagnostic `{busy_us, share_gpu_busy,
  grids}` sum of every row landing at one of these two colliding element
  counts, read straight off the census, so a consumer can see how much
  mass sits at the ambiguous shape without this module moving any chain's
  own busy to guess at it.
- **`WINDOWING`/`FRONT-FUSION`: still UNDECLARED.** A window-partition/
  reverse/roll copy sits at the IDENTICAL element count as a generic
  residual-stream `PERMUTE/RESHAPE` copy (both are `rows*tokens_s*dim_s`)
  — this module has no shape-only way to tell them apart. The one
  plausible front-fusion-activation candidate observed (`urelu_f32`, rare,
  `1-3` launches/step, NOT scaled by `depths`) cannot be distinguished
  from the audio embedding's own projection-head ReLU (the config's
  `hidden_act="gelu"` for the Swin MLP but a separate, undocumented act
  for the projection head — the config's own "projection act relu" note
  explains why `gelu_seam_calls_per_forward` exactly equals `Σdepths`
  with no extra contribution from the projection head) without a
  dedicated ablation. Both stay UNDECLARED (module doc, "declaring a
  chain with no matching rule is dishonest") rather than guessed.
- **The front end is NOT in `by_kernel_and_grid`/`gpu_kernel_us_per_step`
  at all.** `manifest.media_front_end_wall_s` (`n`/`m`, differenced the
  same way `wall_s_per_step` is) gives `front_s_per_step ≈ 1.251 s` on
  `htsat-A1` — `≈80.7%` of that leg's own `wall_s_per_step` (`1.550 s`) —
  the mel-spectrogram extraction and AFF front-end fusion run OUTSIDE the
  profiled training-step kernel window entirely. Stated for context; this
  module never folds it into any chain's `share_gpu_busy`/`share_wall`.
- **`decision_grade`, reported honestly, not massaged**: on the real
  `htsat-A1`/`A2`/`D1`/`D2` legs (see the hand-off's own pasted `main()`
  run), `UNATTRIBUTED` is `5.76%`/`7.07%`/`4.00%`/`4.53%` of busy — `D1`
  and `D2` clear the `<=5%` decision-grade bound, `A1`/`A2` do NOT. HTSAT
  has no candidate port (contract), so this affects only the
  REALIZED-GAIN reporting's own `decision_grade`/`decision_grade_reason`
  fields, never a candidate-port verdict — `compute_realized_gains` itself
  only requires `verdict == VALID` (not `decision_grade`), so the
  `C-LORA`/`C-LN` numbers above are still reported for `htsat` even though
  `A1`/`A2` are not decision-grade.

Run:
  python3 ci/scripts/perf/profile_421_attribute.py --legs-dir .profile-421-legs/<ts>/legs \\
      --merge-json .profile-421-legs/<ts>/merge.json \\
      --out .profile-421-legs/<ts>/attribution.json
Hermetic self-tests: `python3 ci/scripts/perf/test_profile_421_attribute.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 4

VERDICT_VALID = "VALID"
VERDICT_INVALID = "INVALID"

# The merge script's own verdict string (`profile_421_merge.VERDICT_VALID`)
# — kept as an independent literal here (this module does not import the
# merge script) because they happen to share the exact same convention;
# `test_merge_verdict_literal_matches_the_merge_script` pins the two files
# never silently drift apart.
MERGE_VERDICT_VALID = "VALID"

# --- Chains established by NAME/shape rules (module doc, "`C-ATTN`/`C-LN`/
#     `C-GELU`") -------------------------------------------------------------
CHAIN_LORA = "C-LORA"
CHAIN_LN = "C-LN"
CHAIN_GELU = "C-GELU"
CHAIN_UNATTRIBUTED = "UNATTRIBUTED"

# --- Named-bucket extension (module doc, "Named-bucket extension") ---------
CHAIN_BASE_GEMM = "BASE-GEMM"
CHAIN_DROPOUT = "DROPOUT"
CHAIN_CAST = "CAST"
CHAIN_OPTIMIZER = "OPTIMIZER"
CHAIN_EMBED_GATHER = "EMBED/GATHER"
CHAIN_LOSS_REDUCE = "LOSS/REDUCE"
CHAIN_PATCH_EMBED = "C-PATCH-EMBED"

# --- Name-class-split buckets (module doc, "The four activation-tensor
#     width tiers") -----------------------------------------------------------
CHAIN_PERMUTE_RESHAPE = "PERMUTE/RESHAPE"
CHAIN_BIAS_RESIDUAL_QKV = "BIAS/RESIDUAL-QKV"
CHAIN_BIAS_RESIDUAL_OUT = "BIAS/RESIDUAL-OUT"
CHAIN_BIAS_RESIDUAL_MLP = "BIAS/RESIDUAL-MLP"
CHAIN_ELEMENTWISE_OTHER_QKV = "ELEMENTWISE-OTHER-QKV"
CHAIN_ELEMENTWISE_OTHER_OUT = "ELEMENTWISE-OTHER-OUT"
CHAIN_ELEMENTWISE_OTHER_MLP = "ELEMENTWISE-OTHER-MLP"

# Parameter-scale 1-D bookkeeping launches on the trainable PARAMETER
# tensors themselves (LoRA A/B, weight decay, grad-norm clipping) — NOT the
# optimizer (`adamw_*` by name is the ONLY thing `OPTIMIZER` names). See
# module doc, "`GRAD-BOOKKEEPING`".
CHAIN_GRAD_BOOKKEEPING = "GRAD-BOOKKEEPING"

# --- HTSAT-only, declared but unverified (module doc, "HTSAT") -------------
CHAIN_GELU_HTSAT = "C-GELU-HTSAT"
CHAIN_WINDOWING = "WINDOWING"
CHAIN_FRONT_FUSION = "FRONT-FUSION"

# Chain buckets attached to every CLIP leg's `chains` dict (present-or-absent
# — see `attribute_census`). `C-LORA` is deliberately ABSENT from this list:
# it is a realized-gain NUMBER (`compute_realized_gains`), never a
# chain-partition member — see the module doc, "Realized-gain chains are NOT
# chain-partition members".
CLIP_ALWAYS_DECLARED_CHAINS: tuple[str, ...] = (
    CHAIN_LN,
    CHAIN_GELU,
    CHAIN_BASE_GEMM,
    CHAIN_BIAS_RESIDUAL_QKV,
    CHAIN_BIAS_RESIDUAL_OUT,
    CHAIN_BIAS_RESIDUAL_MLP,
    CHAIN_ELEMENTWISE_OTHER_QKV,
    CHAIN_ELEMENTWISE_OTHER_OUT,
    CHAIN_ELEMENTWISE_OTHER_MLP,
    CHAIN_PERMUTE_RESHAPE,
    CHAIN_DROPOUT,
    CHAIN_CAST,
    CHAIN_OPTIMIZER,
    CHAIN_GRAD_BOOKKEEPING,
    CHAIN_EMBED_GATHER,
    CHAIN_LOSS_REDUCE,
)

# `tier ("qkv"|"out"|"mlp") -> (BIAS/RESIDUAL chain, ELEMENTWISE-OTHER chain)`
# — the tier-suffixed half of the name-class split (module doc).
BIAS_RESIDUAL_CHAIN_FOR_TIER: dict[str, str] = {
    "qkv": CHAIN_BIAS_RESIDUAL_QKV,
    "out": CHAIN_BIAS_RESIDUAL_OUT,
    "mlp": CHAIN_BIAS_RESIDUAL_MLP,
}
ELEMENTWISE_OTHER_CHAIN_FOR_TIER: dict[str, str] = {
    "qkv": CHAIN_ELEMENTWISE_OTHER_QKV,
    "out": CHAIN_ELEMENTWISE_OTHER_OUT,
    "mlp": CHAIN_ELEMENTWISE_OTHER_MLP,
}


def chain_attn(tower: str) -> str:
    """`C-ATTN-<tower>`, the one chain name that IS tower-suffixed (the
    contract names it `C-ATTN-<tower>` explicitly, unlike `C-LN`/`C-GELU`,
    because HTSAT's attention tier is out of the fixed-head-dim port scope
    but is still a NAMED, MEASURED chain per tower)."""
    return f"C-ATTN-{tower}"


def declared_chains_for_tower(tower: str) -> tuple[str, ...]:
    """Every chain this module ALWAYS reports a `measured`-or-`absent`
    entry for, given `tower` — see `attribute_census`'s "every declared
    chain present or explicitly absent" invariant."""
    if tower == "clip-vision":
        return CLIP_ALWAYS_DECLARED_CHAINS + (CHAIN_PATCH_EMBED, chain_attn(tower))
    if tower == "clip-text":
        return CLIP_ALWAYS_DECLARED_CHAINS + (chain_attn(tower),)
    if tower == "htsat":
        # Every bucket this module has an EVIDENCED rule for on the real
        # `htsat-{A1,A2,D1}` exports (module doc, "HTSAT"). `WINDOWING`/
        # `FRONT-FUSION` have NO reliable shape/name discriminator yet (a
        # window-partition/reverse copy sits at the IDENTICAL element count
        # as a generic residual-stream permute, and the one plausible
        # front-fusion-activation candidate, `urelu_f32`, cannot be
        # distinguished from the audio projection head's own ReLU without a
        # dedicated ablation) and are deliberately NOT declared here —
        # declaring a chain with no matching rule is dishonest.
        # `CHAIN_GRAD_BOOKKEEPING`
        # is likewise NOT declared for `htsat`: this module has no evidenced
        # LoRA-parameter-scale rule for HTSAT's nine distinct target-module
        # weight shapes (unlike CLIP's four uniform sites), so declaring it
        # would report "absent" forever, indistinguishable from "not
        # implemented".
        return (
            CHAIN_GELU_HTSAT,
            chain_attn(tower),
            CHAIN_LN,
            CHAIN_BASE_GEMM,
            CHAIN_BIAS_RESIDUAL_OUT,
            CHAIN_BIAS_RESIDUAL_MLP,
            CHAIN_ELEMENTWISE_OTHER_OUT,
            CHAIN_ELEMENTWISE_OTHER_MLP,
            CHAIN_PERMUTE_RESHAPE,
            CHAIN_DROPOUT,
            CHAIN_CAST,
            CHAIN_OPTIMIZER,
            CHAIN_EMBED_GATHER,
            CHAIN_LOSS_REDUCE,
            CHAIN_PATCH_EMBED,
        )
    raise SignatureError(f"no declared chain set for tower {tower!r}")


# The exact NON-GEMM kernel-name vocabulary observed across the CLIP and
# HTSAT legs pulled from pod `p421` run 2 at `c1b0b0ba`
# (`clip-text-{A1,A2,D1,D2}`, `clip-vision-{A1,A2,D1,D2}`,
# `htsat-{A1,A2,D1,D2}`). Every GEMM-library name (cutlass/ampere/magma/
# split-K-reduction) is admitted instead via `GEMM_FAMILY_NAME_RE` (below,
# `is_known_kernel_name`) — it is deliberately NOT hand-listed here; see
# that regex's own evidence paragraph in the module doc, "`BASE-GEMM`:
# GEMM-family kernel identity, by NAME".
KNOWN_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "badd_f32",
        "bmul_f32",
        "dropout_fwd_f32",
        "fast_sum_f32",
        "affine_f32",
        "bsub_f32",
        "bdiv_f32",
        "ucopy_f32",
        "copy2d_f32",
        "scaled_cast_add_f32_f32",
        "cast_f32_f32",
        "layer_norm_bwd_dx_f32",
        "layer_norm_fwd_f32_biased",
        "eq_f32",
        "usqr_f32",
        "adamw_moment_update_f32",
        "fast_max_f32",
        "usigmoid_f32",
        "adamw_theta_update_f32",
        "const_set_f32",
        "cast_u8_f32",
        "uexp_f32",
        "bmaximum_f32",
        "bminimum_f32",
        "usqrt_f32",
        "urecip_f32",
        "is_u32_f32",
        "fast_argmax_u32",
        "ucopy_u32",
        "gather_u32_f32",
        "sa_u32_f32",
        # clip-vision-only, evidenced on `clip-vision-A1`
        "im2col_f32",
        # HTSAT, hand-admitted from the real `htsat-{A1,A2,D1}` exports
        # (pod `p421` run 2), same discipline as every entry above.
        "gelu_erf_fwd_f32",
        "gelu_erf_bwd_dx_f32",
        # The eager (D1) twin of `gelu_erf_fused`'s own three-kernel
        # decomposition — `ugelu_erf_f32`/`uerf_f32`/`uneg_f32` launch at the
        # IDENTICAL grids/launch-counts on `htsat-D1` (module doc,
        # "`C-GELU-HTSAT`").
        "ugelu_erf_f32",
        "uerf_f32",
        "uneg_f32",
        "urelu_f32",
        "ge_f32",
    }
)

# Genuinely BF16-only kernel names with NO f32 twin in `KNOWN_KERNEL_NAMES`
# (`name.replace("bf16", "f32")` does not land on a known name) — hand
# admitted from the real `clip-text-A2` export, same discipline as every
# other entry in `KNOWN_KERNEL_NAMES` above, kept in a separate constant
# only so the twin-rule's own doc/tests can show what it does NOT cover.
BF16_ONLY_KERNEL_NAMES: frozenset[str] = frozenset({"cast_add_bf16", "cast_scale_bf16_f32"})

# GEMM-family kernel identity, by NAME — see the module doc's own evidence
# paragraph ("`BASE-GEMM`: GEMM-family kernel identity, by NAME") for the
# real `(kernel, grid)` row behind every alternative below. Used BOTH for
# CLASSIFICATION (`classify_kernel`'s `BASE-GEMM`-by-name rule, feeding the
# grid-position-2 `C-ATTN-<tower>` check ahead of it) and for ADMISSION
# (`is_known_kernel_name`, below) — a single source of truth for "is this
# row GEMM-family" rather than two independently-maintained lists that
# could drift apart.
GEMM_FAMILY_NAME_RE = re.compile(
    r"cutlass_\d+_(simt|tensorop|wmma_tensorop)_\w*gemm\w*"
    r"|ampere_\w*gemm\w*"
    r"|magma_\w*gemm\w*"
    r"|\w*splitKreduce\w*"
)


def is_known_kernel_name(name: str) -> bool:
    """`True` iff `name` is admitted ground truth for the INVALID-by-
    unknown-kernel gate (`UNKNOWN_KERNEL_SHARE_LIMIT`) — NOT the same
    question as "does this row land in a named chain" (`classify_kernel`
    already resolves every `GEMM_FAMILY_NAME_RE` match into a chain before
    this gate is ever consulted — module doc). THREE admission paths:
    (1) `GEMM_FAMILY_NAME_RE` (any GEMM-library instantiation, regardless
    of its own specific tile/stage suffix); (2) the explicit
    `KNOWN_KERNEL_NAMES`/`BF16_ONLY_KERNEL_NAMES` ground truth (every
    NON-GEMM name this module has hand-verified against a real export);
    (3) a BF16 elementwise name whose f32 TWIN (`"bf16"` -> `"f32"`) is
    already known (the twin RULE the contract asks for, rather than
    hand-listing every observed `*_bf16` name). A name matching none of
    these three is unknown, full stop, even if it looks GEMM-shaped by
    some OTHER convention this module has not actually observed and
    evidenced."""
    if GEMM_FAMILY_NAME_RE.search(name):
        return True
    if name in KNOWN_KERNEL_NAMES or name in BF16_ONLY_KERNEL_NAMES:
        return True
    if "bf16" in name and name.replace("bf16", "f32") in KNOWN_KERNEL_NAMES:
        return True
    return False


# A leg-invalidating "we cannot classify this kernel" finding requires the
# unknown name to carry MORE than this share of `gpu_kernel_us_per_step` —
# the contract's own validity gate bounds UNATTRIBUTED at 5%; a name outside
# `is_known_kernel_name` above 1% of busy is material enough that folding it
# into UNATTRIBUTED (rather than refusing) could silently mask a real
# attribution gap.
UNKNOWN_KERNEL_SHARE_LIMIT = 0.01

# Contract: "Validity gate: ... UNATTRIBUTED <= 5% of gpu_busy" — the bound
# `leg_decision_grade` checks against a VALID leg's own UNATTRIBUTED share.
UNATTRIBUTED_DECISION_GRADE_LIMIT = 0.05

# Exclusive-by-name kernels: matched at ANY grid, independent of shape.
LN_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "layer_norm_fwd_f32_biased",
        "layer_norm_bwd_dx_f32",
        "layer_norm_fwd_bf16_biased",
        "layer_norm_bwd_dx_bf16",
    }
)
GELU_EXCLUSIVE_KERNEL_NAMES: frozenset[str] = frozenset({"usigmoid_f32", "usigmoid_bf16"})

# Shape-gated kernels: matched only at the declared GELU element-count shape
# (`TowerSignatures.gelu_shape_elements`, i.e. the `mlp` activation tier).
GELU_SHAPE_KERNEL_NAMES: frozenset[str] = frozenset({"affine_f32", "bmul_f32", "affine_bf16", "bmul_bf16"})

# HTSAT's own `gelu_erf_fused` seam (module doc, "`C-GELU-HTSAT`"): the
# FUSED kernel's own names, matched by NAME at any grid, unconditionally —
# evidenced on `htsat-A1`/`A2` (`kernels_disabled=[]`, `gelu_erf_fused`
# admitted): `gelu_erf_fwd_f32`/`gelu_erf_bwd_dx_f32` launch EXACTLY 12
# times each per step (`launches_per_step` sums to `2+6+2+2=12` across the
# four stage shapes, matching `Σdepths=12` — the config's own
# `gelu_seam_calls_per_forward`), never appearing on `htsat-D1` at all
# (`unmatched_disables()` would refuse the leg otherwise).
GELU_ERF_HTSAT_EXCLUSIVE_KERNEL_NAMES: frozenset[str] = frozenset(
    {"gelu_erf_fwd_f32", "gelu_erf_bwd_dx_f32", "gelu_erf_fwd_bf16", "gelu_erf_bwd_dx_bf16"}
)

# HTSAT's EAGER twin of `gelu_erf_fused` (D1 only) — candle's own
# `Tensor::gelu_erf()` decomposes into THREE elementwise kernels
# (`ugelu_erf_f32`/`uerf_f32`/`uneg_f32`) that launch at the IDENTICAL four
# per-stage grids and IDENTICAL launch counts (evidenced on `htsat-D1`:
# all three names share `launches_per_step` of `2/6/2/2` at the four
# `mlp_shape_elements(stage)` grids `36864/9216/18432/4608`) — shape-gated
# like `GELU_SHAPE_KERNEL_NAMES`, never matched by name alone (candle's
# elementwise op names are generic enough that a name-only match would be
# too permissive without the declared HTSAT MLP-tier shape backing it).
GELU_ERF_HTSAT_EAGER_KERNEL_NAMES: frozenset[str] = frozenset({"ugelu_erf_f32", "uerf_f32", "uneg_f32"})

# `fast_sum`/`fast_max`: matched against TWO declared row counts (softmax
# reduction rows -> C-ATTN; LN row count -> C-LN, eager only); every OTHER
# row count is `LOSS/REDUCE` (module doc).
ROW_REDUCTION_KERNEL_NAMES: frozenset[str] = frozenset(
    {"fast_max_f32", "fast_sum_f32", "fast_max_bf16", "fast_sum_bf16"}
)

# The eager LayerNorm's own flat-elementwise mean/var/std/reciprocal steps
# (module doc, "split by kernel NAME-CLASS") — matched at `total_threads`
# covering `ln_row_count`, ONLY when `layer_norm_fused` is disabled for this
# leg (`ln_disabled` — see `classify_kernel`). `bmul`/`bsub`/`usqr` join
# `usqrt`/`urecip` here (evidenced on `clip-text-D1`/`clip-vision-D1` for
# `bsub`/`usqr`; `bmul` is in the contract's own naming and included even
# though no real leg has shown it AT this exact shape yet — matching costs
# nothing when it never fires); every name in this set is GATED on
# `ln_disabled` explicitly — an A/D2 leg (LN fused) never produces a row at
# this shape at all, so the gate costs nothing when it never fires, but
# stays explicit rather than relying on that absence.
LN_EAGER_EXTENDED_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "bmul_f32",
        "bsub_f32",
        "usqr_f32",
        "usqrt_f32",
        "urecip_f32",
        "bmul_bf16",
        "bsub_bf16",
        "usqr_bf16",
        "usqrt_bf16",
        "urecip_bf16",
    }
)

# `EMBED/GATHER`: u32-typed index/token-id bookkeeping, by name, any grid.
EMBED_GATHER_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "gather_u32_f32",
        "gather_u32_bf16",
        "sa_u32_f32",
        "sa_u32_bf16",
        "ucopy_u32",
        "fast_argmax_u32",
        "is_u32_f32",
        "is_u32_bf16",
    }
)

# `clip-vision`-only conv-as-matmul unfold, by name (module doc,
# "`C-PATCH-EMBED`"). Never appears on `clip-text`.
PATCH_EMBED_KERNEL_NAMES: frozenset[str] = frozenset({"im2col_f32", "im2col_bf16"})

# `PERMUTE/RESHAPE`: permute/reshape copies at any of the three activation
# tiers, by name — module doc, "split by kernel NAME-CLASS". Deliberately
# NOT tier-suffixed (a copy/permute is not a per-site op) and deliberately
# NOT eligible for `C-ATTN-<tower>` (module doc, "the pre-registered
# signature does not move").
PERMUTE_RESHAPE_KERNEL_NAMES: frozenset[str] = frozenset(
    {"ucopy_f32", "copy2d_f32", "ucopy_bf16", "copy2d_bf16"}
)

# `BIAS/RESIDUAL-<tier>`: a site's own bias-add / the residual stream's add,
# by name, tier-suffixed — module doc, "split by kernel NAME-CLASS".
BIAS_RESIDUAL_KERNEL_NAMES: frozenset[str] = frozenset({"badd_f32", "badd_bf16"})

# Declared architecture constants for the two OpenCLIP towers (see this
# module's doc, "Declared per-tower architecture constants" below).
TOWER_ARCH: dict[str, dict[str, object]] = {
    "clip-text": {"sites_per_layer": 4, "width": 512, "heads": 8, "mlp_ratio": 4, "declared_seq": None},
    "clip-vision": {"sites_per_layer": 4, "width": 768, "heads": 12, "mlp_ratio": 4, "declared_seq": 50},
}

# Wire-default LoRA rank (contract, "Wire defaults: rank 8, alpha 16, ...") —
# used ONLY to identify parameter-scale (not activation-scale) elementwise
# ops for the `GRAD-BOOKKEEPING` bucket (module doc). Never used to derive
# an activation-tensor shape.
LORA_RANK = 8

# The `--task` value each tower's `finetune-run` manifest carries — used
# only for a friendlier reason string, never for signature derivation.
TOWER_FAMILY_UNKNOWN = "unknown"

# `(chain, eager role, fused role)` triples for `compute_realized_gains`
# (module doc, "Realized-gain chains are NOT chain-partition members").
# `C-LORA` = `A1`(fused) vs `D2`(eager, LoRA disabled); `C-LN` = `D2`(fused,
# LN still admitted) vs `D1`(eager, LN additionally disabled) — contract
# §D3: "D1 - D2 = C-LN on CLIP".
REALIZED_GAIN_SPECS: tuple[tuple[str, str, str], ...] = (
    (CHAIN_LORA, "D2", "A"),
    (CHAIN_LN, "D1", "D2"),
)


@dataclass
class Signature:
    """One derived scalar with its provenance, so a consumer of the JSON
    output can tell a WITNESSED number from a DECLARED constant without
    re-deriving it."""

    value: int
    source: str


@dataclass
class TowerSignatures:
    rows: Signature
    seq: Signature
    width: Signature
    heads: Signature
    mlp_width: Signature
    layers: Signature

    def gelu_shape_elements(self) -> int:
        return self.rows.value * self.seq.value * self.mlp_width.value

    def attn_softmax_rows(self) -> int:
        return self.rows.value * self.heads.value * self.seq.value

    def attn_batch_count(self) -> int:
        return self.rows.value * self.heads.value

    def attn_shape_elements(self) -> int:
        """The attention score/prob tensor's own element count,
        `rows*heads*seq*seq` — module doc, "`C-ATTN-<tower>`
        attention-elementwise, by SHAPE"."""
        return self.rows.value * self.heads.value * self.seq.value * self.seq.value

    def qkv_shape_elements(self) -> int:
        """`in_proj`'s combined Q/K/V output tensor's element count."""
        return self.rows.value * self.seq.value * 3 * self.width.value

    def out_shape_elements(self) -> int:
        """`out_proj`/`c_proj`'s output (and the residual stream's own
        width) element count."""
        return self.rows.value * self.seq.value * self.width.value

    def ln_row_count(self) -> int:
        """The eager LayerNorm's own per-row reduction/elementwise count,
        `rows*seq` — module doc, "split by kernel NAME-CLASS". Distinct
        from `attn_softmax_rows` (`rows*heads*seq`) by construction
        (`heads>1` for both towers), so no collision with the attention
        reduction rule."""
        return self.rows.value * self.seq.value

    def param_scale_elements(self) -> frozenset[int]:
        """Element counts of the trainable PARAMETER tensors themselves
        (LoRA A/B matrices, Linear biases, LN gamma/beta) at the wire-
        default rank — module doc, "`GRAD-BOOKKEEPING`". Distinct from
        every activation-tier element count above for this checkpoint's
        declared constants (checked in this module's own tests)."""
        width = self.width.value
        mlp_width = self.mlp_width.value
        return frozenset(
            {
                width,
                3 * width,
                mlp_width,
                LORA_RANK * width,
                LORA_RANK * 3 * width,
                LORA_RANK * mlp_width,
            }
        )

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            "rows": {"value": self.rows.value, "source": self.rows.source},
            "seq": {"value": self.seq.value, "source": self.seq.source},
            "width": {"value": self.width.value, "source": self.width.source},
            "heads": {"value": self.heads.value, "source": self.heads.source},
            "mlp_width": {"value": self.mlp_width.value, "source": self.mlp_width.source},
            "layers": {"value": self.layers.value, "source": self.layers.source},
        }


class SignatureError(Exception):
    """This leg's manifest/census does not carry enough to derive
    signatures for its declared tower — carries the message that becomes
    this leg's INVALID reason."""


# HTSAT's declared Swin geometry (module doc, "HTSAT") — read from the
# public `laion/clap-htsat-fused` config's own `audio_config`
# (`depths=[2,2,6,2]`, `num_attention_heads=[4,8,16,32]`, `window_size=8`,
# `hidden_size=768` — the FINAL stage's width; `crates/jammi-encoders/src/
# htsat_audio.rs` documents `hidden_size ==
# patch_embeds_hidden_size << (num_stages-1)`, so the first stage's width
# is `768 >> 3 == 96` and each later stage DOUBLES it, evidenced against
# the real `htsat-A1` export below), `spec_size=256`, `patch_size=4` (a
# `4x4`-stride unfold, `img_size/patch_size == 64` patches per side at
# stage 0). Every element count `derive_signatures`/`classify_htsat_kernel`
# use is RE-DERIVED from these constants, never a hand-copied grid number.
HTSAT_DEPTHS: tuple[int, ...] = (2, 2, 6, 2)
HTSAT_HEADS: tuple[int, ...] = (4, 8, 16, 32)
HTSAT_WINDOW_SIZE = 8
HTSAT_FINAL_STAGE_DIM = 768
HTSAT_SPEC_SIZE = 256
HTSAT_PATCH_SIZE = 4


@dataclass
class HtsatStageSignature:
    """One Swin stage's own geometry, all four fields DERIVED from
    `HTSAT_DEPTHS`/`HTSAT_HEADS`/`HTSAT_WINDOW_SIZE`/`HTSAT_FINAL_STAGE_DIM`
    (module doc) — never a hand-copied per-stage literal."""

    stage: int
    depth: int
    heads: int
    dim: int
    tokens: int
    windows: int

    def attn_batch_count(self, rows: int) -> int:
        """`rows * windows * heads` — the batched-window-attention GEMMs'
        own `grid[2]` (module doc, "`C-ATTN-HTSAT`")."""
        return rows * self.windows * self.heads

    def attn_shape_elements(self, rows: int) -> int:
        """The per-window attention score/prob tensor's own element count.
        Each window holds `window_size**2` tokens, and the score matrix is
        `[tokens_per_window, tokens_per_window]` (every token in the
        window attends to every other), so the per-(batch, window, head)
        element count is `(window_size**2)**2`, not `window_size**2` —
        `attn_batch_count(rows) * (window_size**2)**2`. Evidenced on
        `htsat-A1`: `badd_f32 grid=[24576,...]` (`total_threads=
        25,165,824`) is EXACTLY `attn_batch_count(stage=0)=6144 * 4096`
        (module doc, "HTSAT")."""
        ws_squared = HTSAT_WINDOW_SIZE * HTSAT_WINDOW_SIZE
        return self.attn_batch_count(rows) * ws_squared * ws_squared

    def attn_softmax_rows(self, rows: int) -> int:
        """`rows * heads * tokens` — the softmax reduction's own row count
        (one row per query position per head; each row reduces over
        exactly `window_size**2` keys) — module doc, "HTSAT: the
        attention/LN row-reduction collision"."""
        return rows * self.heads * self.tokens

    def mlp_shape_elements(self, rows: int, mlp_ratio: int = 4) -> int:
        """`rows * tokens * mlp_ratio * dim` — the per-stage MLP
        activation's own element count, the same shape `C-GELU-HTSAT`'s
        eager twin keys off (module doc)."""
        return rows * self.tokens * mlp_ratio * self.dim

    def out_shape_elements(self, rows: int) -> int:
        """`rows * tokens * dim` — the per-stage residual-stream / Linear-
        output element count (module doc)."""
        return rows * self.tokens * self.dim

    def ln_row_count(self, rows: int) -> int:
        """`rows * tokens` — the eager LayerNorm's own per-row
        reduction/elementwise count for this stage (module doc)."""
        return rows * self.tokens


@dataclass
class HtsatSignatures:
    rows: Signature
    stages: tuple[HtsatStageSignature, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": {"value": self.rows.value, "source": self.rows.source},
            "stages": [
                {
                    "stage": s.stage,
                    "depth": s.depth,
                    "heads": s.heads,
                    "dim": s.dim,
                    "tokens": s.tokens,
                    "windows": s.windows,
                }
                for s in self.stages
            ],
        }


def derive_htsat_signatures(manifest: dict) -> HtsatSignatures:
    """Derive `HtsatSignatures` from `manifest.json`'s witnessed `batch`
    plus the declared Swin geometry (module doc). Every per-stage element
    count this returns was cross-checked against the real `htsat-A1`/`D1`
    census exports (module doc, "HTSAT") before a single classification
    rule was written — same §D3 discipline used for CLIP. Also
    cross-checks the WITNESSED `manifest.fusible_site_census.n.gelu_seam_
    calls_per_forward` against the DECLARED `sum(HTSAT_DEPTHS)` (module
    doc: "`gelu_erf_fused` is called once per Swin block, never guessed")
    and raises `SignatureError` naming both values on a mismatch — a leg
    whose checkpoint does not match the declared Swin depth would
    otherwise silently derive wrong per-stage element counts."""
    batch = manifest.get("batch")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise SignatureError(f"manifest.batch is not a positive int ({batch!r})")
    rows = Signature(3 * batch, "witnessed:manifest.batch*3 (--objective triplet, rows=3B)")

    # Cross-check the WITNESSED `gelu_seam_calls_per_forward` against the
    # DECLARED Swin depth (module doc, "HTSAT": "`gelu_erf_fused` is
    # called once per Swin block, never guessed") — a mismatch means this
    # leg's own checkpoint does not match the declared
    # `HTSAT_DEPTHS`/`laion/clap-htsat-fused` geometry this module assumes,
    # and every per-stage element count below would be silently wrong.
    fusible = manifest.get("fusible_site_census")
    site_n = fusible.get("n") if isinstance(fusible, dict) else None
    gelu_seam_calls = site_n.get("gelu_seam_calls_per_forward") if isinstance(site_n, dict) else None
    expected_gelu_seam_calls = sum(HTSAT_DEPTHS)
    if gelu_seam_calls != expected_gelu_seam_calls:
        raise SignatureError(
            "manifest.fusible_site_census.n.gelu_seam_calls_per_forward="
            f"{gelu_seam_calls!r} does not equal sum(HTSAT_DEPTHS)={expected_gelu_seam_calls!r}"
        )

    num_stages = len(HTSAT_DEPTHS)
    if len(HTSAT_HEADS) != num_stages:
        raise SignatureError("HTSAT_DEPTHS/HTSAT_HEADS length mismatch")
    patch_embeds_dim = HTSAT_FINAL_STAGE_DIM >> (num_stages - 1)
    tokens_stage0 = (HTSAT_SPEC_SIZE // HTSAT_PATCH_SIZE) ** 2

    stages = []
    for s, (depth, heads) in enumerate(zip(HTSAT_DEPTHS, HTSAT_HEADS)):
        dim = patch_embeds_dim << s
        tokens = tokens_stage0 // (4**s)
        windows = tokens // (HTSAT_WINDOW_SIZE * HTSAT_WINDOW_SIZE)
        stages.append(
            HtsatStageSignature(stage=s, depth=depth, heads=heads, dim=dim, tokens=tokens, windows=windows)
        )
    return HtsatSignatures(rows=rows, stages=tuple(stages))


def derive_signatures(tower: str, manifest: dict) -> TowerSignatures | HtsatSignatures:
    """Derive this leg's declared/witnessed signatures for `tower` — a
    `TowerSignatures` for `clip-text`/`clip-vision` (from `manifest.json`'s
    witnessed fields plus `TOWER_ARCH`'s declared constants, module doc) or
    an `HtsatSignatures` for `htsat` (from `derive_htsat_signatures`).
    Raises `SignatureError` for an unrecognised tower or a manifest missing
    a field this tower needs witnessed."""
    if tower == "htsat":
        return derive_htsat_signatures(manifest)

    arch = TOWER_ARCH.get(tower)
    if arch is None:
        raise SignatureError(f"no declared architecture constants for tower {tower!r}")

    batch = manifest.get("batch")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise SignatureError(f"manifest.batch is not a positive int ({batch!r})")
    rows = Signature(3 * batch, "witnessed:manifest.batch*3 (--objective triplet, rows=3B)")

    declared_seq = arch["declared_seq"]
    if declared_seq is None:
        seq_value = manifest.get("max_seq_length")
        if not isinstance(seq_value, int) or isinstance(seq_value, bool) or seq_value <= 0:
            raise SignatureError(f"manifest.max_seq_length is not a positive int ({seq_value!r})")
        seq = Signature(seq_value, "witnessed:manifest.max_seq_length")
    else:
        seq = Signature(int(declared_seq), f"declared_constant:{tower}")

    census = manifest.get("fusible_site_census")
    if not isinstance(census, dict):
        raise SignatureError("manifest.fusible_site_census is missing or not an object")
    # The two runs (`n`/`m`) of a leg agree by the merge script's own check;
    # this reader takes `n`'s (present on every VALID leg — see
    # `profile_421_merge.py`'s cross-run equality check) and does not
    # re-verify agreement here (that is `profile_421_merge`'s job, not this
    # module's).
    site_census = census.get("n") if "n" in census else census
    if not isinstance(site_census, dict):
        raise SignatureError("manifest.fusible_site_census.n is missing or not an object")
    sites_wrapped = site_census.get("lora_sites_wrapped")
    if not isinstance(sites_wrapped, int) or isinstance(sites_wrapped, bool) or sites_wrapped < 0:
        raise SignatureError(f"fusible_site_census.lora_sites_wrapped is not a valid int ({sites_wrapped!r})")
    sites_per_layer = int(arch["sites_per_layer"])
    if sites_wrapped % sites_per_layer != 0:
        raise SignatureError(
            f"lora_sites_wrapped={sites_wrapped} is not a multiple of {tower}'s "
            f"sites_per_layer={sites_per_layer}"
        )
    layers = Signature(
        sites_wrapped // sites_per_layer,
        f"witnessed:fusible_site_census.lora_sites_wrapped/{sites_per_layer}",
    )

    width = Signature(int(arch["width"]), f"declared_constant:{tower}")
    heads = Signature(int(arch["heads"]), f"declared_constant:{tower}")
    mlp_width = Signature(
        int(arch["width"]) * int(arch["mlp_ratio"]),
        f"derived:declared_constant:{tower}.width*mlp_ratio",
    )
    return TowerSignatures(rows=rows, seq=seq, width=width, heads=heads, mlp_width=mlp_width, layers=layers)


def htsat_stage_signatures() -> list[dict[str, int]]:
    """The four Swin stages' declared window/head geometry for HTSAT, read
    from the public `laion/clap-htsat-fused` config's own `audio_config`
    (`HTSAT_DEPTHS`/`HTSAT_HEADS`/`HTSAT_WINDOW_SIZE` — module doc). Kept as
    a light, name-only view for callers that only need `(depth, heads,
    window_size)` per stage (`derive_htsat_signatures` is the full
    element-count derivation `classify_htsat_kernel` actually uses, and IS
    now cross-checked against the real `htsat-A1`/`htsat-D1` exports —
    module doc, "HTSAT")."""
    return [
        {"stage": i, "depth": d, "heads": h, "window_size": HTSAT_WINDOW_SIZE}
        for i, (d, h) in enumerate(zip(HTSAT_DEPTHS, HTSAT_HEADS))
    ]


def _entry_total_threads(entry: dict) -> int:
    grid = entry["grid"]
    block = entry["block"]
    return grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]


def _entry_block_size(entry: dict) -> int:
    block = entry["block"]
    return block[0] * block[1] * block[2]


def _in_element_range(entry: dict, elements: int) -> bool:
    """`True` iff `entry`'s total launched thread count covers exactly
    `elements` (allowing for the block-size rounding a real launch grid
    always does — `elements <= total_threads < elements + block_size`,
    the same tolerance every element-count-based shape match in this
    module uses)."""
    total = _entry_total_threads(entry)
    block_size = _entry_block_size(entry)
    return elements <= total < elements + block_size


def _is_row_count_grid(entry: dict, count: int) -> bool:
    """`True` iff `entry`'s grid is a "one block per row" launch over
    exactly `count` rows — `grid[0] == count`, `grid[1] == grid[2] == 1`
    (block size is irrelevant to this shape, the same row-count-only test
    the softmax-reduction rule uses)."""
    grid = entry["grid"]
    return grid[0] == count and grid[1] == 1 and grid[2] == 1


def _is_one_d_launch_grid(grid: list[int]) -> bool:
    """`True` iff `grid` is a genuine 1-D launch (`grid=[ceil(N/b),1,1]`) —
    module doc, "`OPTIMIZER`'s own parameter-scale rule ... is now gated on
    1-D launch geometry"."""
    return grid[1] == 1 and grid[2] == 1


def _is_attn_softmax_reduction_grid(entry: dict, count: int) -> bool:
    """`True` iff `entry` is a "one block per row" launch over exactly
    `count` rows AND its own `block[0] == 64` — HTSAT's window-attention
    softmax reduces over exactly `window_size**2 == 64` keys per row, so
    its own reduction kernel always launches `block=[64,1,1]` (evidenced
    identically on `htsat-A1`/`htsat-D1`, all four stages). This
    discriminates the attention reduction from HTSAT's EAGER LayerNorm's
    OWN per-row reduction, which can coincide at the IDENTICAL `grid[0]`
    for one stage pair (`ln_row_count(stage=0) ==
    attn_softmax_rows(stage=2) == 98304` on this checkpoint's declared
    constants) but always launches a DIFFERENT block size (the smallest
    power of 2 `>= dim_s`, i.e. `128/256/512/1024` — evidenced on
    `htsat-D1`) — module doc, "HTSAT: the attention/LN row-reduction
    collision"."""
    return _is_row_count_grid(entry, count) and entry["block"][0] == 64


def _activation_tier(entry: dict, sig: TowerSignatures) -> str | None:
    """`"qkv"|"out"|"mlp"|None` — which of the three activation-tensor
    width tiers `entry`'s total launched thread count covers, checked in
    this fixed order (module doc, "The four activation-tensor width
    tiers"). The three element counts are evidenced disjoint for both
    towers' declared constants (this module's own tests)."""
    if _in_element_range(entry, sig.qkv_shape_elements()):
        return "qkv"
    if _in_element_range(entry, sig.out_shape_elements()):
        return "out"
    if _in_element_range(entry, sig.gelu_shape_elements()):
        return "mlp"
    return None


def classify_htsat_kernel(entry: dict, hsig: HtsatSignatures, ln_disabled: bool = False) -> str | None:
    """`classify_kernel`'s HTSAT counterpart (module doc, "HTSAT") — a
    SEPARATE function, not a branch inside `classify_kernel`'s CLIP logic,
    because HTSAT's signature is PER-STAGE (four distinct widths/token
    counts/head counts) rather than CLIP's single declared shape per tier.
    Every element-count/grid rule below was read off the real
    `htsat-A1`/`htsat-D1` exports before being written (same §D3
    discipline as every CLIP rule). `ln_disabled` gates the eager LN
    reduction/elementwise rule exactly like the CLIP path."""
    name = entry["kernel"]
    grid = entry["grid"]
    rows = hsig.rows.value
    tower = "htsat"

    def _any_stage(pred) -> bool:
        return any(pred(stage) for stage in hsig.stages)

    # --- `cast`-named kernels: shape-gated to C-ATTN-HTSAT first (same
    #     priority as the CLIP path — module doc, "`C-ATTN`/`C-LN`/`C-GELU`",
    #     rule 1), else CAST.
    if "cast" in name.lower():
        if _any_stage(lambda s: _in_element_range(entry, s.attn_shape_elements(rows))):
            return chain_attn(tower)
        return CHAIN_CAST

    # --- Name-only chains, independent of shape (shared with CLIP).
    if name.startswith("adamw_"):
        return CHAIN_OPTIMIZER
    if name in EMBED_GATHER_KERNEL_NAMES:
        return CHAIN_EMBED_GATHER
    if name.startswith("dropout_fwd"):
        return CHAIN_DROPOUT
    if name in LN_KERNEL_NAMES:
        return CHAIN_LN
    if name in PATCH_EMBED_KERNEL_NAMES:
        return CHAIN_PATCH_EMBED
    if name in GELU_ERF_HTSAT_EXCLUSIVE_KERNEL_NAMES:
        return CHAIN_GELU_HTSAT

    # --- Row-count-gated reductions: the attention softmax's OWN row
    #     reduction always launches `block=[64,1,1]` (module doc,
    #     "`_is_attn_softmax_reduction_grid`") — checked FIRST so it wins
    #     the one real grid[0] collision with a LN row count on this
    #     checkpoint's declared constants. The eager LN reduction (any
    #     OTHER block size, `ln_disabled` only) is checked next; every
    #     other row count is `LOSS/REDUCE`.
    if name in ROW_REDUCTION_KERNEL_NAMES:
        if _any_stage(lambda s: _is_attn_softmax_reduction_grid(entry, s.attn_softmax_rows(rows))):
            return chain_attn(tower)
        if ln_disabled and _any_stage(lambda s: _is_row_count_grid(entry, s.ln_row_count(rows))):
            return CHAIN_LN
        return CHAIN_LOSS_REDUCE

    chain: str | None = None

    # --- Eager LN's own flat mean/var/std/reciprocal steps, gated on
    #     `ln_disabled` (mirrors the CLIP rule; HTSAT's own name set is
    #     narrower — no `bmul`/`urecip` row observed at this shape yet).
    if ln_disabled and name in LN_EAGER_EXTENDED_KERNEL_NAMES and _any_stage(
        lambda s: _in_element_range(entry, s.ln_row_count(rows))
    ):
        chain = CHAIN_LN

    # --- `C-GELU-HTSAT`'s eager (D1) twin: candle's own three-kernel
    #     `gelu_erf` decomposition, shape-gated at the per-stage MLP tier
    #     (module doc, "`GELU_ERF_HTSAT_EAGER_KERNEL_NAMES`").
    if chain is None and name in GELU_ERF_HTSAT_EAGER_KERNEL_NAMES and _any_stage(
        lambda s: _in_element_range(entry, s.mlp_shape_elements(rows))
    ):
        chain = CHAIN_GELU_HTSAT

    # --- Batched-window-attention grid signature: relational, gated on
    #     grid POSITION 2 carrying `rows*windows*heads` for ANY stage
    #     (module doc, "`C-ATTN-HTSAT`").
    if chain is None and _any_stage(lambda s: grid[2] == s.attn_batch_count(rows)):
        chain = chain_attn(tower)

    # --- GEMM-family by name (not carrying an attention batch count).
    if chain is None and GEMM_FAMILY_NAME_RE.search(name):
        chain = CHAIN_BASE_GEMM

    # --- The attention score/prob tensor's own elementwise ops, ANY stage.
    if chain is None and _any_stage(lambda s: _in_element_range(entry, s.attn_shape_elements(rows))):
        chain = chain_attn(tower)

    # --- The two activation-tier buckets HTSAT reuses from CLIP (`out`/
    #     `mlp` — HTSAT's query/key/value/attention_output sites share the
    #     `out` width; there is no combined-QKV tensor to key a `qkv` tier
    #     off), split by kernel NAME-CLASS exactly like the CLIP rule.
    if chain is None:
        tier: str | None = None
        if _any_stage(lambda s: _in_element_range(entry, s.out_shape_elements(rows))):
            tier = "out"
        elif _any_stage(lambda s: _in_element_range(entry, s.mlp_shape_elements(rows))):
            tier = "mlp"
        if tier is not None:
            if name in PERMUTE_RESHAPE_KERNEL_NAMES:
                chain = CHAIN_PERMUTE_RESHAPE
            elif name in BIAS_RESIDUAL_KERNEL_NAMES:
                chain = BIAS_RESIDUAL_CHAIN_FOR_TIER[tier]
            else:
                chain = ELEMENTWISE_OTHER_CHAIN_FOR_TIER[tier]

    return chain


def classify_kernel(
    entry: dict, sig: TowerSignatures, tower: str, ln_disabled: bool = False
) -> str | None:
    """`chain name` this `(kernel, grid, block)` row belongs to, or `None`
    (UNATTRIBUTED — including "unknown kernel name", handled separately by
    the caller so it can also feed the INVALID gate). `ln_disabled` is
    `True` only for a leg whose own `kernels_disabled` names
    `layer_norm_fused` (D1 — see `attribute_leg`); it gates
    `LN_EAGER_EXTENDED_KERNEL_NAMES` (module doc, "split by kernel
    NAME-CLASS") and nothing else. See module doc for the evidence behind
    every branch. Checks run in priority order; an earlier match wins, but
    several branches deliberately FALL THROUGH (rather than returning
    early) when their specific shape does not match, so a kernel whose name
    is shape-gated for one chain can still land in a DIFFERENT chain at a
    DIFFERENT shape instead of falling all the way through to
    `UNATTRIBUTED`."""
    if tower == "htsat":
        return classify_htsat_kernel(entry, sig, ln_disabled=ln_disabled)

    name = entry["kernel"]
    grid = entry["grid"]

    # --- `cast`-named kernels: shape-gated to C-ATTN first, the generic
    #     CAST bucket otherwise. A cast row at any OTHER shape is
    #     unambiguous — no other computation in this model both casts AND
    #     sits at the attention tensor's own element count.
    if "cast" in name.lower():
        if _in_element_range(entry, sig.attn_shape_elements()):
            return chain_attn(tower)
        return CHAIN_CAST

    # --- Name-only chains, independent of shape -----------------------
    if name.startswith("adamw_"):
        return CHAIN_OPTIMIZER
    if name in EMBED_GATHER_KERNEL_NAMES:
        return CHAIN_EMBED_GATHER
    if name.startswith("dropout_fwd"):
        return CHAIN_DROPOUT
    if name in LN_KERNEL_NAMES:
        return CHAIN_LN
    if tower == "clip-vision" and name in PATCH_EMBED_KERNEL_NAMES:
        return CHAIN_PATCH_EMBED

    # --- Row-count-gated reductions: softmax rows -> ATTN; LN rows ->
    #     C-LN (eager only); every other row count -> LOSS/REDUCE.
    if name in ROW_REDUCTION_KERNEL_NAMES:
        if _is_row_count_grid(entry, sig.attn_softmax_rows()):
            return chain_attn(tower)
        if _is_row_count_grid(entry, sig.ln_row_count()):
            return CHAIN_LN
        return CHAIN_LOSS_REDUCE

    # --- Eager LN's own flat mean/var/std/reciprocal steps, GATED on
    #     `ln_disabled` (module doc, "split by kernel NAME-CLASS"). Falls
    #     through (does not return None) when the shape or the gate does
    #     not match.
    chain: str | None = None
    if (
        ln_disabled
        and name in LN_EAGER_EXTENDED_KERNEL_NAMES
        and _in_element_range(entry, sig.ln_row_count())
    ):
        chain = CHAIN_LN

    # --- GELU: exclusive by name; shape-gated for the shared elementwise
    #     names (falls through on a shape miss, same reasoning as above).
    if chain is None and name in GELU_EXCLUSIVE_KERNEL_NAMES:
        chain = CHAIN_GELU
    if chain is None and name in GELU_SHAPE_KERNEL_NAMES and _in_element_range(entry, sig.gelu_shape_elements()):
        chain = CHAIN_GELU

    # --- Batched-attention grid signature: relational, NAME-INDEPENDENT,
    #     gated on grid POSITION 2 exactly (module doc, "`C-ATTN`/`C-LN`/
    #     `C-GELU`", rule 2). Catches any GEMM-family kernel by grid alone.
    if chain is None and grid[2] == sig.attn_batch_count():
        chain = chain_attn(tower)

    # --- GEMM-family by name (not carrying the attention batch count).
    if chain is None and GEMM_FAMILY_NAME_RE.search(name):
        chain = CHAIN_BASE_GEMM

    # --- Attention tensor's own elementwise ops (mask add, scale, cast,
    #     backward exp/square, ...) — module doc, "`C-ATTN-<tower>`
    #     attention-elementwise, by SHAPE".
    if chain is None and _in_element_range(entry, sig.attn_shape_elements()):
        chain = chain_attn(tower)

    # --- Parameter-scale bookkeeping (module doc, "`GRAD-BOOKKEEPING`"),
    #     GATED on 1-D launch geometry — NOT `OPTIMIZER` (`OPTIMIZER` is
    #     `adamw_*` by name only).
    if chain is None and _is_one_d_launch_grid(grid):
        for elements in sig.param_scale_elements():
            if _in_element_range(entry, elements):
                chain = CHAIN_GRAD_BOOKKEEPING
                break

    # --- The three activation-tier buckets, split by kernel NAME-CLASS
    #     (module doc, "The four activation-tensor width tiers"):
    #     permute/reshape copies get their OWN bucket (not tier-suffixed);
    #     bias/residual adds get a tier-suffixed bucket; every other name
    #     at a tier shape is the honestly-labeled `ELEMENTWISE-OTHER-<tier>`
    #     catch-all.
    if chain is None:
        tier = _activation_tier(entry, sig)
        if tier is not None:
            if name in PERMUTE_RESHAPE_KERNEL_NAMES:
                chain = CHAIN_PERMUTE_RESHAPE
            elif name in BIAS_RESIDUAL_KERNEL_NAMES:
                chain = BIAS_RESIDUAL_CHAIN_FOR_TIER[tier]
            else:
                chain = ELEMENTWISE_OTHER_CHAIN_FOR_TIER[tier]

    return chain


@dataclass
class ChainResult:
    status: str  # "measured" | "absent"
    gpu_busy_us: float | None = None
    share_gpu_busy: float | None = None
    share_wall: float | None = None
    kernel_names: list[str] = field(default_factory=list)
    note: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"status": self.status}
        if self.gpu_busy_us is not None:
            out["gpu_busy_us"] = self.gpu_busy_us
        if self.share_gpu_busy is not None:
            out["share_gpu_busy"] = self.share_gpu_busy
        if self.share_wall is not None:
            out["share_wall"] = self.share_wall
        if self.kernel_names:
            out["kernel_names"] = sorted(set(self.kernel_names))
        if self.note:
            out["note"] = self.note
        return out


def attribute_census(
    census: dict, sig: TowerSignatures, tower: str, ln_disabled: bool = False
) -> tuple[dict[str, ChainResult], list[dict[str, object]], list[str]]:
    """Classify every `by_kernel_and_grid` row into a chain bucket.

    Returns `(chains, unknown_kernel_rows, invalid_reasons)`. `chains`
    always carries an entry for every `declared_chains_for_tower(tower)`
    name (per-test invariant: "every declared chain present or explicitly
    absent") plus `UNATTRIBUTED` — `status="absent"` means literally zero
    rows matched, distinct from `status="measured"` with `gpu_busy_us=0.0`.
    `C-LORA` is NEVER added here (module doc, "Realized-gain chains are not
    chain-partition members") — see `compute_realized_gains`.
    """
    rows = census.get("by_kernel_and_grid")
    if not isinstance(rows, list):
        raise SignatureError("census.by_kernel_and_grid is missing or not a list")
    total_busy_us = census.get("gpu_kernel_us_per_step")
    if not isinstance(total_busy_us, (int, float)) or isinstance(total_busy_us, bool):
        raise SignatureError(f"census.gpu_kernel_us_per_step is not a number ({total_busy_us!r})")
    wall_s = census.get("wall_s_per_step")
    if not isinstance(wall_s, (int, float)) or isinstance(wall_s, bool):
        raise SignatureError(f"census.wall_s_per_step is not a number ({wall_s!r})")

    declared = declared_chains_for_tower(tower)
    buckets: dict[str, float] = {name: 0.0 for name in declared}
    buckets[CHAIN_UNATTRIBUTED] = 0.0
    matched_kernels: dict[str, list[str]] = {k: [] for k in buckets}
    unknown_rows: list[dict[str, object]] = []
    invalid_reasons: list[str] = []

    for entry in rows:
        name = entry.get("kernel")
        us_per_step = entry.get("us_per_step")
        if not isinstance(name, str) or not isinstance(us_per_step, (int, float)) or isinstance(
            us_per_step, bool
        ):
            invalid_reasons.append(f"by_kernel_and_grid row is malformed: {entry!r}")
            continue
        # `classify_kernel` runs REGARDLESS of `is_known_kernel_name`: an
        # unknown-name row is classified by `classify_kernel`'s own
        # by-name/by-shape/grid-position rules FIRST, exactly like a known
        # row — "unknown by name" and "unattributed by rule" are different
        # questions. Only a row `classify_kernel` still could not place
        # (chain is `None`, i.e. UNATTRIBUTED) is THEN checked against
        # `is_known_kernel_name` for the leg-invalidating gate.
        chain = classify_kernel(entry, sig, tower, ln_disabled=ln_disabled)
        if chain is None and not is_known_kernel_name(name):
            share = (us_per_step / total_busy_us) if total_busy_us else 0.0
            unknown_rows.append({"kernel": name, "grid": entry.get("grid"), "share_gpu_busy": share})
            if share > UNKNOWN_KERNEL_SHARE_LIMIT:
                invalid_reasons.append(
                    f"kernel {name!r} (grid={entry.get('grid')!r}) is not a known kernel name "
                    f"and carries share_gpu_busy={share:.4f} > {UNKNOWN_KERNEL_SHARE_LIMIT} — "
                    "the mapping cannot classify it and refuses to guess"
                )
        target = chain if chain is not None else CHAIN_UNATTRIBUTED
        buckets[target] += us_per_step
        matched_kernels[target].append(name)

    chains: dict[str, ChainResult] = {}
    for chain_name, busy in buckets.items():
        names = matched_kernels[chain_name]
        if chain_name != CHAIN_UNATTRIBUTED and not names:
            chains[chain_name] = ChainResult(status="absent")
            continue
        share_busy = (busy / total_busy_us) if total_busy_us else None
        share_wall = (busy / 1e6 / wall_s) if wall_s else None
        chains[chain_name] = ChainResult(
            status="measured",
            gpu_busy_us=busy,
            share_gpu_busy=share_busy,
            share_wall=share_wall,
            kernel_names=names,
        )
    return chains, unknown_rows, invalid_reasons


def outside_signature_plausibly_attention(chains: dict[str, ChainResult]) -> dict[str, float]:
    """`{busy_us, share_wall}` mirroring the `PERMUTE/RESHAPE` chain's own
    numbers (`0.0` if absent) — module doc, "the pre-registered signature
    does not move": permute/reshape copies never join `C-ATTN-<tower>`, but
    this field lets the artifact STATE how much busy/wall sits just outside
    the signature at a shape that is plausibly attention-adjacent (a
    multi-head reshape/transpose), without moving any verdict."""
    entry = chains.get(CHAIN_PERMUTE_RESHAPE)
    if entry is None or entry.gpu_busy_us is None:
        return {"busy_us": 0.0, "share_wall": 0.0}
    return {"busy_us": entry.gpu_busy_us, "share_wall": entry.share_wall or 0.0}


_AMBIGUOUS_OUT_MLP_CHAINS = (CHAIN_BIAS_RESIDUAL_OUT, CHAIN_ELEMENTWISE_OTHER_OUT, CHAIN_PERMUTE_RESHAPE)


def htsat_ambiguous_out_mlp_collision(
    census: dict, hsig: HtsatSignatures, ln_disabled: bool = False
) -> dict[str, object]:
    """`{busy_us, share_gpu_busy, grids}` for the KNOWN, stated ambiguity
    (module doc, "HTSAT", "Known ambiguity"; `fixtures/profile_421_htsat_
    a1/PROVENANCE.md`): `out_shape_elements(stage)` for stage `0`/`1` is a
    MATHEMATICAL IDENTITY with `mlp_shape_elements(stage+2)`, so a row that
    `classify_htsat_kernel` places in an `out`-tier bucket (`BIAS/
    RESIDUAL-OUT`, `ELEMENTWISE-OTHER-OUT`, or `PERMUTE/RESHAPE`) AT one of
    these two element counts might ACTUALLY be that stage's own
    residual/bias/reshape, or the later stage's own MLP-tier op that
    happens to share the count — this module cannot tell from shape alone
    and does not guess. A row a NAME rule already resolves unambiguously
    (LN, GELU, dropout, cast, ...) is NOT counted here even if its grid
    happens to coincide — the ambiguity is specific to the tier-fallback
    classification, never to every kernel that happens to share a grid.
    This field never moves any chain's own `gpu_busy_us` — it is a PURELY
    diagnostic sum, read straight off `census`, so a consumer can see how
    much of the `out`-tier bucket's own mass sits at an admittedly-
    ambiguous shape. `grids` is the SET of `grid[0]` values the matching
    rows actually carried (sorted ascending), never a hand-copied
    literal."""
    rows = hsig.rows.value
    ambiguous_elements = [s.out_shape_elements(rows) for s in hsig.stages if s.stage in (0, 1)]
    total_busy_us = census.get("gpu_kernel_us_per_step")
    busy_us = 0.0
    grids: set[int] = set()
    for entry in census.get("by_kernel_and_grid", []) or []:
        us_per_step = entry.get("us_per_step")
        if not isinstance(us_per_step, (int, float)) or isinstance(us_per_step, bool):
            continue
        if not any(_in_element_range(entry, elements) for elements in ambiguous_elements):
            continue
        if classify_htsat_kernel(entry, hsig, ln_disabled=ln_disabled) not in _AMBIGUOUS_OUT_MLP_CHAINS:
            continue
        busy_us += us_per_step
        grids.add(entry["grid"][0])
    share_gpu_busy = (busy_us / total_busy_us) if total_busy_us else 0.0
    return {"busy_us": busy_us, "share_gpu_busy": share_gpu_busy, "grids": sorted(grids)}


# Per-tower `D1` disable set (contract §D3: "D1 is PER TOWER" — CLIP has no
# `gelu_erf_fused` admit site, naming it would make `unmatched_disables()`
# refuse the leg).
D1_DISABLE_SET: dict[str, frozenset[str]] = {
    "clip-text": frozenset({"lora_linear_fused", "layer_norm_fused"}),
    "clip-vision": frozenset({"lora_linear_fused", "layer_norm_fused"}),
    "htsat": frozenset({"lora_linear_fused", "layer_norm_fused", "gelu_erf_fused"}),
}
D2_DISABLE_SET: frozenset[str] = frozenset({"lora_linear_fused"})


def leg_role(kernels_disabled: list[str], tower: str) -> str:
    """`"A" | "D1" | "D2" | "other"` from a leg's OWN recorded
    `kernels_disabled` list (never from its id string)."""
    disabled = frozenset(kernels_disabled)
    if not disabled:
        return "A"
    if disabled == D2_DISABLE_SET:
        return "D2"
    if disabled == D1_DISABLE_SET.get(tower):
        return "D1"
    return "other"


def leg_decision_grade(row: dict, merge_row: dict | None) -> tuple[bool, str | None]:
    """`(decision_grade, reason)` for one leg's already-built attribution
    row. Contract, Validity gate, verbatim: "kernel table present; counter
    equations hold; `--expect-kernels-disabled` satisfied on D legs;
    UNATTRIBUTED <= 5% of gpu_busy (C-ATTN-HTSAT is attributed, so this is
    satisfiable)." `profile_421_merge.py` enforces "kernel table present",
    "counter equations hold", and "`--expect-kernels-disabled` satisfied on
    D legs" (plus checkpoint identity and the wall/front/busy decomposition
    the contract's Method section separately requires) — THIS module
    enforces "UNATTRIBUTED <= 5% of gpu_busy" and nothing else (module doc,
    "`decision_grade` now REQUIRES the merge's own verdict"). `reason` is
    `None` iff `decision_grade` is `True`. `merge_row` is the corresponding
    row from a `--merge-json` report (`None` if this leg has no such row —
    treated as INVALID-by-merge, never assumed clean)."""
    if row.get("verdict") != VERDICT_VALID:
        # NAME the reason, don't just point at it — `decide_candidate_
        # port`'s own F32-only caveat needs the actual text, not a pointer
        # a reader has to go dig up.
        reasons = row.get("reasons")
        detail = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) and reasons else "no reasons recorded"
        return False, f"leg is INVALID: {detail}"
    if merge_row is None:
        return False, "no corresponding row in --merge-json — the merge cannot certify this leg"
    if merge_row.get("verdict") != MERGE_VERDICT_VALID:
        return False, f"the merge's own verdict is {merge_row.get('verdict')!r}, not VALID"
    chains = row.get("chains")
    if not isinstance(chains, dict):
        return False, "no chains computed for this leg"
    unattributed = chains.get(CHAIN_UNATTRIBUTED)
    if not isinstance(unattributed, dict):
        return False, "no UNATTRIBUTED entry computed for this leg"
    share = unattributed.get("share_gpu_busy")
    if not isinstance(share, (int, float)) or isinstance(share, bool):
        return False, "UNATTRIBUTED share_gpu_busy is not a number"
    if share != share or share in (float("inf"), float("-inf")):  # NaN/inf guard (family F:
        # `NaN > 0.05` is `False`, so a naive threshold check would let a
        # diverged/non-finite share silently pass as decision_grade=True).
        return False, f"UNATTRIBUTED share_gpu_busy is non-finite ({share!r})"
    if share > UNATTRIBUTED_DECISION_GRADE_LIMIT:
        return False, (
            f"UNATTRIBUTED share_gpu_busy={share:.4f} > {UNATTRIBUTED_DECISION_GRADE_LIMIT}"
        )
    return True, None


def _merge_wall_s_per_step(merge_row: dict | None) -> float | None:
    """The merge's own `per_step.wall_s_per_step` for one leg (module doc,
    "wall_delta_s_per_step ... sourced from the NAMED --merge-json
    report"), or `None` if this leg has no VALID per-step decomposition
    there."""
    if not isinstance(merge_row, dict):
        return None
    per_step = merge_row.get("per_step")
    if not isinstance(per_step, dict):
        return None
    wall = per_step.get("wall_s_per_step")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool):
        return None
    if wall != wall or wall in (float("inf"), float("-inf")):
        return None
    return float(wall)


def _realized_gain_direction(chain_name: str, eager_id: str, fused_id: str, busy_delta: float) -> str:
    if busy_delta >= 0:
        return f"eager twin ({eager_id}) slower by {busy_delta:.1f} us/step busy than the fused leg ({fused_id})"
    return (
        f"eager twin ({eager_id}) FASTER by {abs(busy_delta):.1f} us/step busy than the fused leg "
        f"({fused_id}) — unexpected direction, reported as measured, never corrected"
    )


def compute_realized_gains(legs: list[dict], merge_by_leg_id: dict[str, dict]) -> list[dict]:
    """`C-LORA`/`C-LN`'s realized fused-vs-eager numbers, ONE entry per
    `(chain, tower)` — module doc, "Realized-gain chains are NOT
    chain-partition members". Requires BOTH legs of the pair to be F32 and
    VALID (this module's own verdict; `A1`/`D2` are F32-only per the
    contract's legs table). Never mutates any leg's own `chains` dict.
    `legs` must still carry the internal `_role`/`_gpu_busy_us_per_step`
    fields (called BEFORE `build_report` strips them)."""
    by_tower_role: dict[tuple[str, str], dict] = {}
    for leg in legs:
        tower = leg.get("tower")
        role = leg.get("_role")
        if not isinstance(tower, str) or role not in ("A", "D1", "D2"):
            continue
        if leg.get("dtype") != "f32" or leg.get("verdict") != VERDICT_VALID:
            continue
        by_tower_role[(tower, role)] = leg

    gains: list[dict] = []
    for chain_name, eager_role, fused_role in REALIZED_GAIN_SPECS:
        towers = sorted({t for (t, r) in by_tower_role if r in (eager_role, fused_role)})
        for tower in towers:
            eager = by_tower_role.get((tower, eager_role))
            fused = by_tower_role.get((tower, fused_role))
            if eager is None or fused is None:
                continue
            eager_busy = eager.get("_gpu_busy_us_per_step")
            fused_busy = fused.get("_gpu_busy_us_per_step")
            if not isinstance(eager_busy, (int, float)) or not isinstance(fused_busy, (int, float)):
                continue
            busy_delta = eager_busy - fused_busy
            eager_wall = _merge_wall_s_per_step(merge_by_leg_id.get(eager["leg_id"]))
            fused_wall = _merge_wall_s_per_step(merge_by_leg_id.get(fused["leg_id"]))
            wall_delta = (eager_wall - fused_wall) if (eager_wall is not None and fused_wall is not None) else None
            # `share_of_baseline_wall` ALWAYS divides by the tower's SHIPPED
            # `A1` wall — the fused LEG in a `(eager_role, fused_role)` pair
            # is not always `A1` (`C-LN`'s own pair is `D1` vs `D2`; `D2` is
            # not what ships). `fused_twin_wall_s`/`share_of_fused_twin_wall`
            # are emitted SEPARATELY and divide by the pair's own fused leg
            # instead — see the module doc's "Denominator convention"
            # paragraph.
            baseline_leg = by_tower_role.get((tower, "A"))
            baseline_wall = (
                _merge_wall_s_per_step(merge_by_leg_id.get(baseline_leg["leg_id"]))
                if baseline_leg is not None
                else None
            )
            share_of_baseline_wall = (
                (wall_delta / baseline_wall) if (wall_delta is not None and baseline_wall) else None
            )
            share_of_fused_twin_wall = (
                (wall_delta / fused_wall) if (wall_delta is not None and fused_wall) else None
            )
            gain: dict[str, object] = {
                "chain": chain_name,
                "tower": tower,
                "eager_leg_id": eager["leg_id"],
                "fused_leg_id": fused["leg_id"],
                "busy_delta_us_per_step": busy_delta,
                "wall_delta_s_per_step": wall_delta,
                "baseline_leg_id": baseline_leg["leg_id"] if baseline_leg is not None else None,
                "baseline_wall_s": baseline_wall,
                "share_of_baseline_wall": share_of_baseline_wall,
                "fused_twin_wall_s": fused_wall,
                "share_of_fused_twin_wall": share_of_fused_twin_wall,
                "direction": _realized_gain_direction(chain_name, eager["leg_id"], fused["leg_id"], busy_delta),
            }
            if tower == "htsat" and chain_name == CHAIN_LN:
                # `htsat`'s own `D1` disables `layer_norm_fused` AND
                # `gelu_erf_fused` TOGETHER (module doc, "HTSAT") — this
                # `D1-D2` delta cannot isolate `C-LN` from `C-GELU-HTSAT`
                # from a single leg pair, so it is reported as their JOINT
                # gain rather than force-split under one name alone.
                gain["note"] = (
                    "htsat's D1 disables layer_norm_fused and gelu_erf_fused together — this "
                    "delta is the JOINT C-LN + C-GELU-HTSAT realized gain, not C-LN alone; see "
                    "each leg's own by-name chain shares (chains.C-LN / chains.C-GELU-HTSAT) for "
                    "how the two buckets split within a single leg's busy"
                )
            gains.append(gain)
    return gains


# --- The two-sided decision rule (contract "Decision rule") -----------------

ACTIVATE_WALL_THRESHOLD = 0.10
DECLINE_COMBINED_THRESHOLD = 0.05

CANDIDATE_CHAIN_TOWERS: tuple[str, ...] = ("clip-text", "clip-vision")


def candidate_ports_for_tower(tower: str) -> dict[str, str]:
    return {f"C-ATTN-{tower}": chain_attn(tower), f"C-MLP-{tower}": CHAIN_GELU}


def _leg_chain_shares(leg: dict, chain_key: str) -> tuple[float, float, float, float]:
    """`(s_wall, s_busy, u_wall, u_busy)` for one leg row, `0.0` for any
    missing/absent/non-finite entry (contract F: a `NaN`/absent share must
    never silently pass a threshold check the way `NaN > c` would)."""
    chains = leg.get("chains") or {}
    chain = chains.get(chain_key) or {}
    unattributed = chains.get(CHAIN_UNATTRIBUTED) or {}

    def _finite(value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 0.0
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf guard
            return 0.0
        return float(value)

    return (
        _finite(chain.get("share_wall")),
        _finite(chain.get("share_gpu_busy")),
        _finite(unattributed.get("share_wall")),
        _finite(unattributed.get("share_gpu_busy")),
    )


def decide_candidate_port(
    port_name: str, chain_key: str, tower: str, legs: list[dict], merge_by_leg_id: dict[str, dict] | None = None
) -> dict:
    """The contract's two-sided rule for ONE candidate port. `legs` must
    still carry the internal `_role`/`_gpu_busy_us_per_step` fields (i.e.
    called BEFORE `build_report` strips them). `merge_by_leg_id` defaults
    to `{}` (every leg then fails `leg_decision_grade`'s merge check,
    reading UNRESOLVED honestly rather than silently accepting an
    un-certified leg — callers outside `build_report`, e.g. unit tests
    exercising the rule's own arithmetic on synthetic shares, must pass a
    real mapping to reach ACTIVATE/DECLINE)."""
    merge_by_leg_id = merge_by_leg_id if merge_by_leg_id is not None else {}
    # Do NOT pre-filter on `verdict == VALID` here — an INVALID `A2` leg
    # (e.g. one the 1%-unknown-kernel gate invalidates) must still be
    # FOUND here so `leg_decision_grade` can name its actual reason below,
    # rather than this leg silently looking "absent" the same way a truly
    # missing leg would.
    a_legs = [leg for leg in legs if leg.get("tower") == tower and leg.get("_role") == "A"]
    a1 = next((leg for leg in a_legs if leg.get("dtype") == "f32"), None)
    a2 = next((leg for leg in a_legs if leg.get("dtype") == "bf16"), None)

    a1_grade, a1_reason = (
        leg_decision_grade(a1, merge_by_leg_id.get(a1["leg_id"])) if a1 is not None else (False, "no A1 (F32) leg present")
    )
    if not a1_grade:
        return {
            "port": port_name,
            "tower": tower,
            "chain": chain_key,
            "verdict": "UNRESOLVED",
            "reason": f"A1 not decision-grade: {a1_reason}",
        }

    a2_grade, a2_reason = (
        leg_decision_grade(a2, merge_by_leg_id.get(a2["leg_id"])) if a2 is not None else (False, "no A2 (BF16) leg present")
    )
    # The F32-only caveat fires whenever A2 is not decision-grade for ANY
    # reason — present but INVALID (named via `a2_reason`), present but
    # missing a `--merge-json` row, or simply ABSENT (`a2 is None`) —
    # never only the "present but INVALID" case.
    f32_only_note = ""
    if not a2_grade:
        f32_only_note = f" (F32-only: A2 not decision-grade — {a2_reason})"

    decision_grade_legs = [a1] + ([a2] if a2 is not None and a2_grade else [])

    for leg in decision_grade_legs:
        s_wall, _s_busy, _u_wall, _u_busy = _leg_chain_shares(leg, chain_key)
        if s_wall >= ACTIVATE_WALL_THRESHOLD:
            return {
                "port": port_name,
                "tower": tower,
                "chain": chain_key,
                "verdict": "ACTIVATE",
                "reason": f"{leg['leg_id']}: s_wall={s_wall:.4f} >= {ACTIVATE_WALL_THRESHOLD}" + f32_only_note,
            }

    declines = True
    decline_detail = []
    for leg in decision_grade_legs:
        s_wall, s_busy, u_wall, u_busy = _leg_chain_shares(leg, chain_key)
        wall_combined = s_wall + u_wall
        busy_combined = s_busy + u_busy
        ok = wall_combined < DECLINE_COMBINED_THRESHOLD and busy_combined < DECLINE_COMBINED_THRESHOLD
        decline_detail.append(
            f"{leg['leg_id']}: s_wall+U_wall={wall_combined:.4f}, s_busy+U_busy={busy_combined:.4f}"
        )
        if not ok:
            declines = False

    if declines:
        return {
            "port": port_name,
            "tower": tower,
            "chain": chain_key,
            "verdict": "DECLINE",
            "reason": "; ".join(decline_detail) + f32_only_note,
        }
    return {
        "port": port_name,
        "tower": tower,
        "chain": chain_key,
        "verdict": "UNRESOLVED",
        "reason": "neither ACTIVATE (s_wall>=10% on any decision-grade leg) nor DECLINE "
        "(combined share <5% on every decision-grade leg) — " + "; ".join(decline_detail) + f32_only_note,
    }


def decide_all_candidates(legs: list[dict], merge_by_leg_id: dict[str, dict] | None = None) -> list[dict]:
    """Every candidate port's decision, across every tower this module
    knows candidate ports for. Called BEFORE `build_report` strips the
    internal `_role`/`_gpu_busy_us_per_step` fields."""
    decisions: list[dict] = []
    for tower in CANDIDATE_CHAIN_TOWERS:
        for port_name, chain_key in candidate_ports_for_tower(tower).items():
            decisions.append(decide_candidate_port(port_name, chain_key, tower, legs, merge_by_leg_id))
    return decisions


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def attribute_leg(leg_dir: Path) -> dict:
    """One leg's attribution row. NEVER raises for a leg-level problem —
    every failure becomes this row's `verdict="INVALID"` + `reasons`, the
    same discipline `profile_421_merge.merge_leg` uses, so one bad leg
    cannot discard the others. `decision_grade` here is computed WITHOUT a
    merge row (`None`) — `build_report`/`main` recompute it once a
    `--merge-json` is available (module doc, "`decision_grade` now
    REQUIRES the merge's own verdict")."""
    leg_id = leg_dir.name
    row: dict[str, object] = {
        "leg_id": leg_id,
        "tower": None,
        "dtype": None,
        "verdict": VERDICT_INVALID,
        "reasons": [],
        "signatures": None,
        "chains": None,
        "unknown_kernels": [],
        "decision_grade": False,
        "decision_grade_reason": None,
    }
    reasons: list[str] = []
    try:
        manifest = _load_json(leg_dir / "manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        row["reasons"] = [f"{leg_id}: manifest.json could not be read: {exc}"]
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row

    tower = manifest.get("tower")
    dtype = manifest.get("dtype")
    row["tower"] = tower
    row["dtype"] = dtype
    kernels_disabled = manifest.get("kernels_disabled")
    if not isinstance(kernels_disabled, list):
        row["reasons"] = [f"{leg_id}: manifest.kernels_disabled is not a list"]
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row
    kernels_disabled_str = [str(k) for k in kernels_disabled]
    row["_role"] = leg_role(kernels_disabled_str, str(tower))
    # `ln_disabled`: this leg's OWN recorded `kernels_disabled` names
    # `layer_norm_fused` — for a VALID leg this equals the merge's
    # validated `kernels_disabled_expected` (contract §D6: the merge
    # refuses a D leg whose requested set differs from the declared one),
    # so reading it here (rather than importing the merge's own report) is
    # equivalent for every leg this module ever calls `decision_grade=True`
    # on — module doc, "split by kernel NAME-CLASS".
    ln_disabled = "layer_norm_fused" in kernels_disabled_str

    if manifest.get("census_ok") is not True or manifest.get("census_exit") != 0:
        reasons.append(
            f"{leg_id}: manifest reports census_ok={manifest.get('census_ok')!r} "
            f"census_exit={manifest.get('census_exit')!r} — no trustworthy census to attribute"
        )

    census_path = leg_dir / "census.json"
    try:
        census = _load_json(census_path)
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"{leg_id}: census.json could not be read: {exc}")
        row["reasons"] = reasons
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row

    if census.get("excluded_from_chain_attribution") is True:
        reasons.append(
            f"{leg_id}: census.excluded_from_chain_attribution is true — this leg's signatures "
            "are not fixed-shape and cannot be attributed by this module (kernel_census.py's own "
            "E1 exclusion)"
        )
        row["reasons"] = reasons
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row

    if not isinstance(tower, str):
        reasons.append(f"{leg_id}: manifest.tower is not a string ({tower!r})")
        row["reasons"] = reasons
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row

    try:
        sig = derive_signatures(tower, manifest)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row
    row["signatures"] = sig.as_dict()

    try:
        chains, unknown, classify_reasons = attribute_census(census, sig, tower, ln_disabled=ln_disabled)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        # NAME the reason inline rather than pointing at `row["reasons"]`
        # — the F32-only caveat `decide_candidate_port` builds needs the
        # actual text.
        row["decision_grade_reason"] = f"leg is INVALID: {'; '.join(str(r) for r in row['reasons'])}"
        return row
    reasons.extend(classify_reasons)
    row["unknown_kernels"] = unknown
    row["chains"] = {name: result.as_dict() for name, result in chains.items()}
    row["outside_signature_plausibly_attention"] = outside_signature_plausibly_attention(chains)
    if tower == "htsat" and isinstance(sig, HtsatSignatures):
        row["ambiguous_out_mlp_collision"] = htsat_ambiguous_out_mlp_collision(
            census, sig, ln_disabled=ln_disabled
        )
    row["_gpu_busy_us_per_step"] = census.get("gpu_kernel_us_per_step")
    row["_wall_s_per_step"] = census.get("wall_s_per_step")
    row["gpu_busy_us_per_step"] = census.get("gpu_kernel_us_per_step")
    row["wall_s_per_step"] = census.get("wall_s_per_step")
    memcpy = census.get("memcpy_per_step")
    memset = census.get("memset_per_step")
    if isinstance(memcpy, dict) or isinstance(memset, dict):
        # Reported for visibility ONLY — never folded into any chain's
        # `share_gpu_busy` denominator (contract §D2: "nsys memcpy rows
        # reported separately from gpu_busy").
        row["memcpy_memset"] = {"memcpy_per_step": memcpy, "memset_per_step": memset}

    row["reasons"] = reasons
    row["verdict"] = VERDICT_INVALID if reasons else VERDICT_VALID
    # `decision_grade` here has no merge row (`None`) — always `False` with
    # a merge-shaped reason unless the caller (`build_report`) recomputes it
    # against a real `--merge-json`. Kept so a direct `attribute_leg` call
    # (e.g. this module's own tests) still returns a fully-shaped row.
    grade, grade_reason = leg_decision_grade(row, None)
    row["decision_grade"] = grade
    row["decision_grade_reason"] = grade_reason
    return row


def _leg_dirs(legs_dir: Path) -> list[Path]:
    """Every directory under `legs_dir` this module treats as a leg, EXCEPT
    the `p2-bf16` driver's own non-leg scratch directory (a known,
    legitimate sibling — the P2b BF16 corpus, never a
    `profile_421_legs.sh` leg). A directory with NO `manifest.json` is
    included here (never silently dropped), so `attribute_leg` (which
    already turns "manifest could not be read" into an `INVALID` row with
    a reason) surfaces it as a visible INVALID leg instead of a silent gap
    in the reported leg count."""
    return sorted(d for d in legs_dir.iterdir() if d.is_dir() and d.name != "p2-bf16")


def _merge_legs_by_id(merge_report: dict) -> dict[str, dict]:
    legs = merge_report.get("legs")
    if not isinstance(legs, list):
        return {}
    return {row["leg_id"]: row for row in legs if isinstance(row, dict) and isinstance(row.get("leg_id"), str)}


def build_report(legs_dir: Path, merge_report: dict) -> dict:
    """`merge_report` is the parsed `--merge-json` (module doc,
    "`decision_grade` now REQUIRES the merge's own verdict"). Every leg's
    `decision_grade` and every candidate port's decision are computed
    against it; `realized_gains` sources its wall deltas from it."""
    legs = [attribute_leg(d) for d in _leg_dirs(legs_dir)]
    merge_by_leg_id = _merge_legs_by_id(merge_report)
    for row in legs:
        grade, grade_reason = leg_decision_grade(row, merge_by_leg_id.get(row["leg_id"]))
        row["decision_grade"] = grade
        row["decision_grade_reason"] = grade_reason
    candidate_decisions = decide_all_candidates(legs, merge_by_leg_id)
    realized_gains = compute_realized_gains(legs, merge_by_leg_id)
    # Strip the leading-underscore working fields — internal to this
    # module's cross-leg pass, not part of the published schema.
    for row in legs:
        row.pop("_role", None)
        row.pop("_gpu_busy_us_per_step", None)
        row.pop("_wall_s_per_step", None)
    report = {
        "tool": "profile_421_attribute",
        "schema": SCHEMA_VERSION,
        "legs_dir": str(legs_dir),
        # This module's OWN live validity-gate constants, recorded at
        # report-build time (module doc, "`limits`: the report's own
        # recorded validity-gate bounds") — a downstream reader trusts
        # THIS, never a reason string, never the live import.
        "limits": {
            "unattributed_decision_grade_limit": UNATTRIBUTED_DECISION_GRADE_LIMIT,
            "unknown_kernel_share_limit": UNKNOWN_KERNEL_SHARE_LIMIT,
        },
        "legs": legs,
        "candidate_decisions": candidate_decisions,
        "realized_gains": realized_gains,
        "summary": {
            "legs_total": len(legs),
            "legs_valid": sum(1 for row in legs if row["verdict"] == VERDICT_VALID),
            "legs_invalid": sum(1 for row in legs if row["verdict"] == VERDICT_INVALID),
            "legs_decision_grade": sum(1 for row in legs if row["decision_grade"]),
        },
    }
    return report


def format_table(report: dict) -> str:
    """A human-readable per-(tower, dtype, leg, chain) share table — this is
    what a caller pastes into a PR/hand-off, never re-derived from the JSON
    by eye."""
    lines: list[str] = []
    header = (
        f"{'leg_id':<16} {'tower':<12} {'dtype':<5} {'verdict':<8} {'grade':<6} "
        f"{'chain':<24} {'status':<10} {'share_busy':>10} {'share_wall':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report.get("legs", []):
        chains = row.get("chains") or {}
        grade = "yes" if row.get("decision_grade") else "no"
        if not chains:
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {grade:<6} {'(none)':<24} {'':<10} {'':>10} {'':>10}"
            )
            continue
        for chain_name in sorted(chains):
            entry = chains[chain_name]
            share_busy = entry.get("share_gpu_busy")
            share_wall = entry.get("share_wall")
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {grade:<6} {chain_name:<24} {entry.get('status', ''):<10} "
                f"{(f'{share_busy:.4f}' if isinstance(share_busy, (int, float)) else ''):>10} "
                f"{(f'{share_wall:.4f}' if isinstance(share_wall, (int, float)) else ''):>10}"
            )
    lines.append("")
    lines.append("realized gains:")
    for gain in report.get("realized_gains", []):
        lines.append(
            f"  {gain['chain']:<10} {gain['tower']:<12} {gain['direction']}"
        )
    lines.append("")
    lines.append("candidate port decisions:")
    for decision in report.get("candidate_decisions", []):
        lines.append(f"  {decision['port']:<20} {decision['verdict']:<12} {decision['reason']}")
    return "\n".join(lines)


_PROVENANCE_LEG_ID_RE = re.compile(r"pod421-run2/legs/([A-Za-z0-9_-]+)/census\.json")

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def verify_fixtures(fixtures_dir: Path, legs_dir: Path) -> list[str]:
    """Every committed fixture's `kernels.json` rows must match its OWN
    `PROVENANCE.md`-named source
    leg's real `census.json` byte-for-byte, whenever that real leg is
    present under `legs_dir` — CI has no pod pull, so a fixture whose named
    leg is absent is SKIPPED cleanly (never a failure), but this function
    is meant to be run BY HAND against a real pulled `legs/` tree before a
    fixture change ships (see the hand-off's own pasted run). Returns a
    list of issue strings; empty means every checkable fixture matched
    exactly (or none could be checked)."""
    issues: list[str] = []
    dirs = sorted(
        d
        for d in fixtures_dir.iterdir()
        if d.is_dir() and (d / "kernels.json").is_file() and (d / "PROVENANCE.md").is_file()
    )
    if not dirs:
        return [f"no fixture directories found under {fixtures_dir}"]
    for fixture_dir in dirs:
        provenance = (fixture_dir / "PROVENANCE.md").read_text(encoding="utf-8")
        match = _PROVENANCE_LEG_ID_RE.search(provenance)
        if match is None:
            issues.append(f"{fixture_dir.name}: PROVENANCE.md names no pod421-run2/legs/<leg>/census.json source")
            continue
        leg_id = match.group(1)
        real_census_path = legs_dir / leg_id / "census.json"
        if not real_census_path.is_file():
            # Real leg not pulled in this environment — skip cleanly.
            continue
        try:
            fixture = json.loads((fixture_dir / "kernels.json").read_text(encoding="utf-8"))
            real = json.loads(real_census_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{fixture_dir.name}: could not read fixture or real census: {exc}")
            continue
        for top_key in ("gpu_kernel_us_per_step", "wall_s_per_step"):
            if top_key in fixture and fixture.get(top_key) != real.get(top_key):
                issues.append(
                    f"{fixture_dir.name}: top-level {top_key!r} = {fixture.get(top_key)!r}, "
                    f"real {leg_id} has {real.get(top_key)!r}"
                )
        real_rows = {
            (row["kernel"], tuple(row["grid"]), tuple(row["block"])): row for row in real.get("by_kernel_and_grid", [])
        }
        for row in fixture.get("by_kernel_and_grid", []):
            key = (row["kernel"], tuple(row["grid"]), tuple(row["block"]))
            real_row = real_rows.get(key)
            if real_row is None:
                issues.append(f"{fixture_dir.name}: row {key} has no match in real {leg_id}'s census.json")
            elif real_row != row:
                issues.append(
                    f"{fixture_dir.name}: row {key} does not match real {leg_id} byte-for-byte "
                    f"(fixture={row!r}, real={real_row!r})"
                )
    return issues


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_attribute.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", help="a profile_421_legs.sh $OUT_DIR/legs to attribute")
    ap.add_argument(
        "--merge-json",
        help="the output of profile_421_merge.py over the SAME legs-dir; required (module doc) "
        "unless --verify-fixtures is given",
    )
    ap.add_argument("--out", help="write the attribution report here (default: stdout)")
    ap.add_argument(
        "--no-table",
        action="store_true",
        help="suppress the human-readable table on stderr (JSON output is unaffected)",
    )
    ap.add_argument(
        "--verify-fixtures",
        metavar="LEGS_DIR",
        help="verify every committed fixture's kernels.json against its PROVENANCE-named source "
        "leg's real census.json under LEGS_DIR (skips a fixture cleanly if its leg is absent); "
        "runs INSTEAD of the normal --legs-dir/--merge-json attribution and exits before them "
        "(module doc, `verify_fixtures`)",
    )
    args = ap.parse_args(argv)

    if args.verify_fixtures:
        issues = verify_fixtures(FIXTURES_DIR, Path(args.verify_fixtures))
        if issues:
            for issue in issues:
                print(f"::error::profile_421_attribute --verify-fixtures: {issue}", file=sys.stderr)
            return 1
        print("profile_421_attribute --verify-fixtures: every checkable fixture matched byte-for-byte", file=sys.stderr)
        return 0

    if not args.legs_dir:
        print("::error::profile_421_attribute: --legs-dir is required (unless --verify-fixtures)", file=sys.stderr)
        return 1
    if not args.merge_json:
        print("::error::profile_421_attribute: --merge-json is required (unless --verify-fixtures)", file=sys.stderr)
        return 1

    legs_dir = Path(args.legs_dir)
    if not legs_dir.is_dir():
        print(f"::error::profile_421_attribute: {legs_dir} is not a directory", file=sys.stderr)
        return 1

    merge_path = Path(args.merge_json)
    try:
        merge_report = _load_json(merge_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"::error::profile_421_attribute: could not read --merge-json {merge_path}: {exc}", file=sys.stderr)
        return 1

    # `--legs-dir`'s own leg set must be a SUBSET of `--merge-json`'s (every
    # leg this module is about to attribute must have a corresponding merge
    # row to certify its `decision_grade` against) — the merge report is
    # allowed to name MORE legs than this run attributes (e.g. an HTSAT leg
    # this module cannot classify yet still needs a merge row of its own,
    # for a DIFFERENT tower's report to consume), so this is `<=`, not `==`.
    # A leg present in `--legs-dir` with NO merge row at all is the unsafe
    # case this refuses (module doc, "`decision_grade` now REQUIRES the
    # merge's own verdict").
    legs_dir_names = {d.name for d in _leg_dirs(legs_dir)}
    merge_leg_names = set(_merge_legs_by_id(merge_report))
    missing_from_merge = legs_dir_names - merge_leg_names
    if missing_from_merge:
        print(
            "::error::profile_421_attribute: --merge-json names a different legs set than --legs-dir "
            f"-- legs present under --legs-dir with no --merge-json row: {sorted(missing_from_merge)}",
            file=sys.stderr,
        )
        return 1

    report = build_report(legs_dir, merge_report)
    payload = json.dumps(report, indent=1, sort_keys=False)
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::profile_421_attribute: could not write {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)

    for row in report["legs"]:
        if row["verdict"] != VERDICT_VALID:
            print(f"::warning::leg {row['leg_id']}: {row['verdict']}", file=sys.stderr)
            for reason in row["reasons"]:
                print(f"  - {reason}", file=sys.stderr)
        elif not row["decision_grade"]:
            print(
                f"::warning::leg {row['leg_id']}: not decision_grade: {row['decision_grade_reason']}",
                file=sys.stderr,
            )
    if not args.no_table:
        print(format_table(report), file=sys.stderr)
    summary = report["summary"]
    print(
        f"profile_421_attribute: {summary['legs_valid']}/{summary['legs_total']} legs VALID, "
        f"{summary['legs_decision_grade']}/{summary['legs_total']} decision_grade",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
