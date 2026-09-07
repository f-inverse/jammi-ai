#!/usr/bin/env python3
"""Issue #421 tower-profile chain ATTRIBUTION (post-export unit; CONTRACT
`scratchpad/contract-421-profile.md` v2.5 `### Attribution` / `§D3`).

`profile_421_merge.py` decides whether a leg is a DATUM at all (the
positive-proof equation, checkpoint identity, the wall/front/busy
decomposition). This module answers a DIFFERENT question over the SAME
`$OUT_DIR/legs` tree: of one leg's own `gpu_kernel_us_per_step`, how much
belongs to which named chain, and how much is `UNATTRIBUTED`?

## Pass 1 (v1; `clip-text-A1` only) — kept as ground truth, not re-derived

Pass 1 read the mapping off the real `clip-text-A1` (F32) export and
established: `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` -> `C-LN`
by NAME at any grid; `usigmoid_f32` -> `C-GELU` by NAME at any grid;
`affine_f32`/`bmul_f32` -> `C-GELU` ONLY at the declared `quick_gelu` tensor
shape `[rows*S, 4*width]` (a `badd_f32` row at the IDENTICAL shape is
deliberately EXCLUDED — it is `c_fc`'s own Linear bias, not the activation);
`fast_max_f32`/`fast_sum_f32` -> `C-ATTN-<tower>` ONLY at the declared
softmax reduction row count `rows*heads*S` (`grid[1]==grid[2]==1`);
`ampere_sgemm_*` -> `C-ATTN-<tower>` when its grid carries `rows*heads` as
one of its three dimensions (a RELATIONAL rule, not a name list) else
UNATTRIBUTED (a Linear/LoRA projection's plain 2-D matmul). `C-LORA` has NO
literal GPU kernel name anywhere in this export — the contract's own method
for it is never by-name: the `A1 - D2` busy/wall delta, computed
opportunistically by `attach_lora_via_d2_delta` whenever both legs are
present and individually VALID. `KNOWN_KERNEL_NAMES` gates: a kernel name
outside the validated vocabulary above 1% of a leg's busy makes the WHOLE
LEG `INVALID` rather than silently folding into `UNATTRIBUTED` — the
discipline this module has followed since pass 1 and extends, never
loosens, below.

Pass 1 left roughly 85% of `clip-text-A1`'s busy `UNATTRIBUTED` — every
`ampere_sgemm_*` row NOT carrying the attention batch count, every
`badd_f32`/`bmul_f32`/... row outside the one declared GELU shape,
`adamw_*`, `dropout_fwd_*`, cast/copy kernels, `splitKreduce_kernel`, and
the handful of anonymous `Kernel2` rows. The validity gate's own bound
(`UNATTRIBUTED <= 5% of gpu_busy`) is unsatisfiable at 85% by construction,
so every leg was UNRESOLVED before a single candidate share was even read.

## Pass 2 (v2.5 §Attribution/§D3 — named-bucket extension)

Trigger: the real `clip-text-A2` (BF16), `clip-text-D1`, `clip-text-D2`, and
`clip-vision-A1`/`clip-vision-A2` exports landed (pod `p421` run 2, tip
`c1b0b0ba`; `scratchpad/pod421-run2/legs/<leg>/census.json`). This pass
reads OFF THOSE exports — never guessed, never transcribed as a literal
timing expectation — a set of NAMED, EVIDENCE-BACKED buckets that absorb
the overwhelming majority of what pass 1 left `UNATTRIBUTED`, so the
validity gate becomes evaluable rather than vacuously false everywhere.
Every new bucket below is grounded in a specific `(kernel, grid)` row from a
specific real export; the numbers cited are DESCRIPTIVE evidence for why a
rule exists, never asserted as literal values in this module's tests (same
discipline pass 1 established).

### The four activation-tensor width tiers (per CLIP tower)

Every OpenCLIP block has exactly four LoRA-wrappable sites
(`in_proj`/`out_proj`/`c_fc`/`c_proj`, `crate::open_clip_block`), whose
OUTPUT tensors sit at four distinct, non-colliding element counts derived
from this module's already-declared `rows`/`seq`/`width`/`mlp_width`
signatures:

- `qkv = rows*seq*3*width` (`in_proj`'s combined Q/K/V output)
- `out = rows*seq*width` (`out_proj`/`c_proj`'s output, and the residual
  stream itself, which is the same width)
- `mlp = rows*seq*mlp_width` (`c_fc`'s output, the same shape `C-GELU`
  already keys off)

Evidence on `clip-text-A1` (`rows=24, seq=77, width=512, mlp_width=2048`):
`badd_f32` launches at EXACTLY three grids matching these three tiers
(`grid=2772` -> `qkv=2,838,528` elements; `grid=924` -> `out=946,176`;
`grid=3696` -> `mlp=3,784,704`, the same grid `C-GELU`'s `affine_f32`/
`bmul_f32` already claim) — these are the sites' own bias adds, the
residual stream's adds, and (per contract, since `lora_epilogue_fused`
already fuses the LoRA composition on every leg per §D3's own scope note)
possibly a LoRA composition add too; a SINGLE leg's census cannot separate
"bias" from "residual" from "LoRA compose" at IDENTICAL shape any more than
pass 1 could separate GELU's bias from its activation, so this module names
the TIER, not the sub-role: `ELEMENTWISE-QKV`/`ELEMENTWISE-OUT`/
`ELEMENTWISE-MLP`. `bmul_f32`/`bsub_f32`/`affine_f32`/`copy2d_f32`/
`ucopy_f32`/`const_set_f32` at these SAME three shapes (evidenced the same
way, same three grids on `clip-text-A1`) land in the same three tiers.
`dropout_fwd_f32`/`dropout_fwd_bf16` land in their own `DROPOUT` bucket
by NAME (unambiguous — no other computation in this model is dropout),
regardless of which tier's shape a given dropout call sits at (both the
`out` tier at `grid=3696,block=256` and the `mlp` tier at
`grid=14784,block=256` are dropout on `clip-text-A1`).

### `C-ATTN-<tower>` extended: the attention SCORE/PROB tensor's own elementwise ops

`badd_f32`/`bdiv_f32`/`bmul_f32`/`bsub_f32`/`cast_u8_f32`/`eq_f32`/
`uexp_f32`/`usqr_f32`/`affine_f32` (and their `*_bf16` twins) ALSO launch at
the declared attention tensor's own element count,
`attn = rows*heads*seq*seq` (`= 1,138,368` on `clip-text-A1`,
`rows=24,heads=8,seq=77`; `= 720,000` on `clip-vision-A1`,
`heads=12,seq=50`) — the additive mask, the `1/sqrt(head_dim)` scale
(`bdiv_f32`/`affine_f32`), the boolean-mask cast/compare feeding it
(`cast_u8_f32`/`eq_f32`), and the backward pass's own `exp`/`square` terms.
Evidence: on `clip-text-A1`, `badd_f32`/`bdiv_f32`/`bmul_f32`/`bsub_f32`/
`cast_u8_f32`/`eq_f32`/`uexp_f32`/`usqr_f32`/`affine_f32` ALL have a row at
`grid=1112` (`1112*1024=1,138,688`, the smallest multiple of the `1024`
block covering `1,138,368` elements) and NOWHERE else does that row's
kernel name launch at that exact grid; on `clip-vision-A1` the same nine
names have a row at `grid=704` (`704*1024=720,896`, covering
`720,000` elements) — a second tower confirming the rule generalizes by
FORMULA (`rows*heads*seq^2`), not by a hand-copied grid number. A single
generic `total_threads in [attn, attn+block)` match (shared with the
tier-matching code the QKV/OUT/MLP buckets above use) covers all nine names
at once, rather than nine separate hand-verified rules.

### `C-ATTN-<tower>` extended: the batched-matmul grid signature, name-independent

Pass 1's `ampere_sgemm_*`-carries-`rows*heads` rule is a RELATIONAL rule on
the GRID, stated once so it "generalizes to whatever specific
`ampere_sgemm_*` tile variant a different shape/dtype/arch selects" (pass
1's own doc). `clip-text-A2` (BF16) and `clip-vision-A1` are exactly that
test: `clip-text-A2`'s bf16 GEMM family
(`ampere_bf16_s16816gemm_bf16_*_ldg8_f2f_stages_*_{nn,tn}`, EIGHT distinct
tile-variant names, none matching any f32 name) and `clip-vision-A1`'s
`magma_sgemmEx_kernel` (a THIRD gemm library entirely, `grid=[1,2,288]`,
`288 = rows*heads` for vision's `heads=12`) both carry the attention batch
count in their grid exactly where the rule predicts, with completely
different literal names. This module therefore matches the batch-count
grid rule FIRST, by grid alone, BEFORE checking whether a name looks like a
GEMM at all (`attn_batch_count() in entry["grid"]"`) — this is deliberately
name-INDEPENDENT (contract: "cuBLAS heuristic picks — same
BASE-GEMM/C-ATTN rules by grid"), so it also catches the anonymous
`Kernel2` rows `clip-text-A2` emits at `grid=[2,1,192]`/`[4,1,192]`/
`[12,1,192]` (three DIFFERENT tile shapes at the identical attention batch
count, alongside the correctly-classified bf16 GEMM family in the SAME
leg — nsys simply failed to symbolize these particular bf16 kernel
launches; their grid is exactly as diagnostic as a named kernel's).

### `BASE-GEMM`: every other GEMM-family kernel

Any kernel name containing `"gemm"` (case-insensitive — covers
`ampere_sgemm_*`, `ampere_bf16_*gemm*`, `magma_sgemmEx_kernel`, and any
future cuBLAS/CUTLASS/MAGMA tile-variant name a different shape/dtype/arch
selects) that does NOT carry the attention batch count in its grid is the
towers' own base/LoRA Linear projection matmul — `BASE-GEMM`. Evidence: on
`clip-text-A1`, `ampere_sgemm_128x64_nn` at `grid=[4,29,7]` (no `192`
anywhere) is `35.08%` of busy alone — the single largest bucket on every
CLIP leg pulled so far (`33-46%` across the five). `splitKreduce_kernel`
(cuBLAS's split-K reduction accessory kernel, launched only alongside
certain large-K GEMM algorithms) is likewise `BASE-GEMM` by name,
unconditionally — it never carries a batch dimension of its own (its grid
is the reduction's internal tiling), so it cannot be routed through the
grid rule; the large-K GEMMs it accompanies are the base projections
(`c_fc`: `K=512`, `c_proj`: `K=2048`), never the small `head_dim=64`
attention matmuls.

### `C-LN` extended: the EAGER LayerNorm's own kernels, by shape (D1 evidence)

`clip-text-D1` disables `layer_norm_fused` (contract §D3: D1's per-tower
disable set). `layer_norm_fwd_f32_biased`/`layer_norm_bwd_dx_f32` correctly
disappear from its `by_kernel_and_grid` list (`unmatched_disables()`
would have refused the leg otherwise). In their place, an EAGER
mean/var/normalize sequence appears at a NEW declared row count,
`ln_rows = rows*seq` (`= 1,848` on `clip-text-A1`/`D1`, DISTINCT from the
attention softmax row count `14,784` and every width tier above, so no
collision): `fast_sum_f32` at `grid=[1848,1,1]` (`launches_per_step=98`,
the per-row mean/var reduction; `block=512` here vs `128` for the attention
reduction — irrelevant to the grid-row-count match, same as pass 1's
existing rule ignores block for the softmax reduction); `usqrt_f32`/
`urecip_f32` at `total_threads` covering exactly `1,848` elements
(`grid=[2,1,1], block=1024` -> `2,048 >= 1,848`, ONE call per row-buffer,
`launches_per_step=25` on `D1` matching the leg's own witnessed
`fusible_site_census.layer_norms=25` exactly) — the eager `sqrt(var+eps)`
and `1/std` steps. `clip-vision-A1` (LN FUSED, not eager) shows a single,
negligible `urecip_f32` call (`share=0.0000%`) at `grid=[1,1,1]`
(`total_threads=1,024`, matching NEITHER `ln_rows=1,200` NOR `rows=24`
cleanly) — the shape gate correctly excludes it from `C-LN` (this module
does not guess it into `LOSS/REDUCE` without a row-count match the way
`fast_sum`/`fast_max` get). It DOES fall into the `OPTIMIZER`
parameter-scale catch-all below (`1,024` sits inside the `width=768`
tier's `[768, 1792)` block-rounding range) — a small, honestly-labeled
bucket rather than a guessed "L2-normalize" role.

### `LOSS/REDUCE`: `fast_sum`/`fast_max` at every OTHER row count

Contract: "`fast_sum`/`fast_max` NOT at the softmax row count" is its own
named bucket, not `UNATTRIBUTED`. After the softmax-row and LN-row checks
above both fail, `fast_sum_f32 grid=[1,1,1]` (`launches_per_step=96` on
`clip-text-A1`, `1.61%` of busy) is the largest single row this bucket
absorbs — the triplet loss's own margin/logit reductions and/or the
gradient-norm bookkeeping the optimizer needs; this module does not, and
cannot from a single leg's census, further split "loss" from "grad-norm"
reductions sharing the identical degenerate `grid=[1,1,1]` shape, so both
are named `LOSS/REDUCE` together, honestly, rather than guessed apart.

### `CAST`: any kernel name containing `"cast"`

`cast_f32_f32`/`cast_u8_f32`/`cast_bf16_f32`/`cast_f32_bf16`/
`cast_add_bf16`/`cast_scale_bf16_f32`/`cast_u8_bf16`/
`scaled_cast_add_f32_f32`/`scaled_cast_add_bf16_f32` — matched by NAME,
any grid, any tower. (`scaled_cast_add_f32_f32`'s own grid pattern is
suggestive of a LoRA-composition-adjacent op — it launches at EXACTLY the
three activation-tier grids above, one call per LoRA-wrappable site width,
on every leg pulled so far — but this module does not claim that role
without a dedicated ablation; `CAST` is where the contract's own bucket
list puts every cast-named kernel, so that is where it stays.)

### `OPTIMIZER` extended: AdamW by name, PLUS parameter-scale bookkeeping by shape

`adamw_moment_update_f32`/`adamw_theta_update_f32` (any grid) were already
`KNOWN_KERNEL_NAMES` in pass 1 but UNATTRIBUTED; they are `OPTIMIZER` by
name now. Their own three grids (`4`, `16`, `12` on `clip-text-A1`) are
EXACTLY the wire-default LoRA rank (`8`) times each of the three width
tiers (`8*512=4096`, `8*2048=16384`, `8*1536=12288` — the LoRA A/B matrices'
OWN element counts, not the activation tensors' — `rank*width`,
`rank*mlp_width`, `rank*3*width`). A dozen OTHER elementwise kernels
(`badd_f32`/`bmul_f32`/`bsub_f32`/`usqr_f32`/`cast_u8_f32`/... at these same
three tiny grids, plus the LN gamma/beta and Linear bias vectors' own width
(`512`/`1536`/`2048`, which fit in ONE or TWO `1024`-thread blocks) sit at
the IDENTICAL element counts — gradient/weight-decay/moment bookkeeping on
the trainable PARAMETER tensors themselves, as opposed to the ACTIVATION
tensors the tiers above key off. This module names this whole family
`OPTIMIZER` too (a parameter-scale elementwise op is optimizer/gradient
bookkeeping, not a forward-pass activation op, by construction — the two
scales never collide for this checkpoint's declared architecture
constants, checked once in this module's own tests) rather than leaving
every one of these dozen small-grid kernels an individually-unexplained
`UNATTRIBUTED` row.

### `EMBED/GATHER`: the `u32`-typed index/token-id kernels, by name

`gather_u32_f32`/`sa_u32_f32`/`ucopy_u32`/`fast_argmax_u32`/`is_u32_f32`
(and bf16 twins where observed) are the only kernels operating on `u32`
(token-id / index) data in this export — embedding-table gather, its
scatter-add gradient, and the EOT-position equality check. All negligible
(`<0.1%` combined on every leg pulled) but unambiguous by dtype-in-name.

### `C-PATCH-EMBED` (`clip-vision` only): the conv-as-matmul patchify stem

`im2col_f32`/`im2col_bf16` appears ONLY on `clip-vision` legs (never
`clip-text`) — OpenCLIP's ViT patch embedding is a strided `Conv2d`,
implemented as `im2col` (unfold) followed by a GEMM; the GEMM half is
already `BASE-GEMM` (or `C-ATTN` if its grid happened to carry the batch
count, which it does not here) via the generic gemm-name rule above, and
`im2col_f32` itself — the unfold, not a matmul — gets its own small,
vision-exclusive bucket by name.

### Memcpy/memset: reported, never folded into the busy partition

Contract §D2: "nsys memcpy rows reported separately from gpu_busy." The
census's own `memcpy_per_step`/`memset_per_step` top-level fields (counts
and `us`, NOT part of `by_kernel_and_grid` or `gpu_kernel_us_per_step`) are
copied onto this module's leg row verbatim as `memcpy_memset` for
visibility — they never enter any chain's `share_gpu_busy` denominator.

### `decision_grade` and the validity gate

Contract: "Validity gate: ... UNATTRIBUTED <= 5% of gpu_busy." Each leg row
now carries `decision_grade: bool` and, when `False`, `decision_grade_reason`
(an INVALID leg, or a VALID leg whose `UNATTRIBUTED` `share_gpu_busy`
still exceeds `5%`). Measured on the five real legs pulled so far (see this
module's own `main()` run, pasted in the hand-off): every one of
`clip-text-A1/A2/D1/D2` and `clip-vision-A1` is `decision_grade=True` —
`UNATTRIBUTED` share is `1.7-5.0%` of busy across all five (BF16's
`clip-text-A2` is the tightest, historically the least-verified
vocabulary), comfortably inside the gate once the buckets above are named.

### The two-sided decision rule (contract "Decision rule")

`decide_candidate_chains` implements the contract's two-sided rule
PER `(chain, tower)` for the four candidate ports only
(`C-ATTN-clip-text`, `C-ATTN-clip-vision`, `C-MLP-clip-text`,
`C-MLP-clip-vision` — `C-MLP-<tower>` reads the `C-GELU` chain's own
numbers, per contract: "C-MLP-CLIP-text/vision = the C-GELU quick_gelu
bucket"): `ACTIVATE` iff `s_wall >= 10%` on >= 1 decision-grade A leg;
`DECLINE` iff `s_wall + U_wall < 5%` AND `s_busy + U_busy < 5%` on EVERY
decision-grade A leg; `UNRESOLVED` otherwise (A1 not decision-grade ->
UNRESOLVED; A2 present but not decision-grade -> the DECLINE/ACTIVATE
reading is F32-only and the reason says so explicitly). The verdict
STRING is written by this function from the numbers — nothing calling it
may hand-edit a verdict.

`C-LORA`/`C-LN`/`C-GELU-HTSAT` are realized-gain chains: this module
reports their NUMBERS (already computed as ordinary chain entries / the
`A1-D2` delta) and never runs the two-sided rule over them.
`C-ATTN-HTSAT` gets a number and the contract's own standing "out of tier"
note, once an HTSAT leg exists to attribute (see below).

## HTSAT: declared, UNVERIFIED (no HTSAT leg has landed as of this pass)

`gelu_erf_fused` (by name, any grid — after P1-a's seam fix, HTSAT's MLP
now admits under the same seam BERT/DistilBERT do) maps to `C-GELU-HTSAT`,
a realized-gain chain (contract: "HTSAT D1 also disables `gelu_erf_fused`
... C-LN + C-GELU on HTSAT" — its number comes from the same `D1-D2` cross
check `attach_lora_via_d2_delta` already computes as an advisory).
`WINDOWING` (Swin window partition/roll ops) and `FRONT-FUSION` (the AFF
block's conv/batchnorm) are declared bucket NAMES here for forward
compatibility but have NO kernel-name/shape mapping yet — no HTSAT export
exists to read one off (§D3's own convention: mapping is read off the
FIRST real export of that tower, never guessed ahead of it). HTSAT's
per-window attention signature (`[B*nW, heads, ws^2, ws^2]`) is likewise
declared from the public `laion/clap-htsat-fused` config
(`depths=[2,2,6,2]`, `num_heads=[4,8,16,32]`, `window_size=8`,
`hidden_size=96`) and unverified.

Run:
  python3 ci/scripts/perf/profile_421_attribute.py --legs-dir .profile-421-legs/<ts>/legs \\
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

SCHEMA_VERSION = 2

VERDICT_VALID = "VALID"
VERDICT_INVALID = "INVALID"

# --- Original (pass 1) chain names -----------------------------------------
CHAIN_LORA = "C-LORA"
CHAIN_LN = "C-LN"
CHAIN_GELU = "C-GELU"
CHAIN_UNATTRIBUTED = "UNATTRIBUTED"

# --- Pass-2 named buckets ----------------------------------------------------
CHAIN_BASE_GEMM = "BASE-GEMM"
CHAIN_ELEMENTWISE_QKV = "ELEMENTWISE-QKV"
CHAIN_ELEMENTWISE_OUT = "ELEMENTWISE-OUT"
CHAIN_ELEMENTWISE_MLP = "ELEMENTWISE-MLP"
CHAIN_DROPOUT = "DROPOUT"
CHAIN_CAST = "CAST"
CHAIN_OPTIMIZER = "OPTIMIZER"
CHAIN_EMBED_GATHER = "EMBED/GATHER"
CHAIN_LOSS_REDUCE = "LOSS/REDUCE"
CHAIN_PATCH_EMBED = "C-PATCH-EMBED"

# --- HTSAT-only, declared but unverified (module doc, "HTSAT") -------------
CHAIN_GELU_HTSAT = "C-GELU-HTSAT"
CHAIN_WINDOWING = "WINDOWING"
CHAIN_FRONT_FUSION = "FRONT-FUSION"

# Chain buckets attached to every CLIP leg's `chains` dict (present-or-absent,
# same discipline pass 1 established for LN/GELU/ATTN — see `attribute_census`).
CLIP_ALWAYS_DECLARED_CHAINS: tuple[str, ...] = (
    CHAIN_LN,
    CHAIN_GELU,
    CHAIN_BASE_GEMM,
    CHAIN_ELEMENTWISE_QKV,
    CHAIN_ELEMENTWISE_OUT,
    CHAIN_ELEMENTWISE_MLP,
    CHAIN_DROPOUT,
    CHAIN_CAST,
    CHAIN_OPTIMIZER,
    CHAIN_EMBED_GATHER,
    CHAIN_LOSS_REDUCE,
)


def chain_attn(tower: str) -> str:
    """`C-ATTN-<tower>`, the one chain name that IS tower-suffixed (the
    contract names it `C-ATTN-<tower>` explicitly, unlike `C-LORA`/`C-LN`/
    `C-GELU`, because HTSAT's attention tier is out of the fixed-head-dim
    port scope but is still a NAMED, MEASURED chain per tower)."""
    return f"C-ATTN-{tower}"


def declared_chains_for_tower(tower: str) -> tuple[str, ...]:
    """Every chain this module ALWAYS reports a `measured`-or-`absent`
    entry for, given `tower` — see `attribute_census`'s "every declared
    chain present or explicitly absent" invariant."""
    if tower == "clip-vision":
        return CLIP_ALWAYS_DECLARED_CHAINS + (CHAIN_PATCH_EMBED, chain_attn(tower))
    if tower == "clip-text":
        return CLIP_ALWAYS_DECLARED_CHAINS + (chain_attn(tower),)
    # HTSAT: only the two name-only chains this pass can classify without a
    # real export (module doc, "HTSAT"); `WINDOWING`/`FRONT-FUSION` have no
    # mapping yet and are deliberately NOT declared here (declaring a chain
    # with no matching rule would silently report it "absent" forever,
    # which is indistinguishable from "not yet implemented" — dishonest).
    return (CHAIN_GELU_HTSAT, chain_attn(tower))


# The exact kernel-name vocabulary observed across the real `clip-text-A1`
# (F32, pass 1), `clip-text-D1`/`clip-text-D2` (F32, eager-LoRA/LN tile
# variants), and `clip-vision-A1` (F32) exports pulled from pod `p421` run 2
# at `c1b0b0ba`. See `is_known_kernel_name` for how BF16 names (a twin-name
# RULE, not a hand-list) and any GEMM-family name (a name-pattern rule) are
# additionally admitted without being enumerated here.
KNOWN_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "badd_f32",
        "ampere_sgemm_32x128_tn",
        "ampere_sgemm_32x128_nt",
        "ampere_sgemm_128x64_nn",
        "ampere_sgemm_128x64_nt",
        "bmul_f32",
        "ampere_sgemm_128x32_tn",
        "ampere_sgemm_128x32_nn",
        "dropout_fwd_f32",
        "fast_sum_f32",
        "affine_f32",
        "Kernel2",
        "bsub_f32",
        "bdiv_f32",
        "ucopy_f32",
        "ampere_sgemm_32x32_sliced1x4_nn",
        "ampere_sgemm_32x32_sliced1x4_nt",
        "ampere_sgemm_32x32_sliced1x4_tn",
        "ampere_sgemm_32x128_nn",
        "copy2d_f32",
        "ampere_sgemm_128x128_nt",
        "ampere_sgemm_128x128_nn",
        "ampere_sgemm_128x128_tn",
        "ampere_sgemm_64x32_sliced1x4_nn",
        "ampere_sgemm_64x32_sliced1x4_nt",
        "scaled_cast_add_f32_f32",
        "splitKreduce_kernel",
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
        "magma_sgemmEx_kernel",
    }
)

# Genuinely BF16-only kernel names with NO f32 twin in `KNOWN_KERNEL_NAMES`
# (`name.replace("bf16", "f32")` does not land on a known name) — hand
# admitted from the real `clip-text-A2` export, same discipline as every
# other entry in `KNOWN_KERNEL_NAMES` above, kept in a separate constant
# only so the twin-rule's own doc/tests can show what it does NOT cover.
BF16_ONLY_KERNEL_NAMES: frozenset[str] = frozenset({"cast_add_bf16", "cast_scale_bf16_f32"})

# Any kernel name containing this (case-insensitive) is GEMM-family
# (cuBLAS/CUTLASS/MAGMA), regardless of dtype or tile-variant string — see
# module doc, "`BASE-GEMM`: every other GEMM-family kernel".
GEMM_NAME_RE = re.compile(r"gemm", re.IGNORECASE)


def is_known_kernel_name(name: str) -> bool:
    """`True` iff `name` is admitted ground truth for the INVALID-by-
    unknown-kernel gate (`UNKNOWN_KERNEL_SHARE_LIMIT`) — NOT the same
    question as "does this row land in a named chain": `Kernel2` is KNOWN
    (pass 1 hand-verified it exists and is unclassifiable) but usually
    lands in `UNATTRIBUTED`. Three admission paths, in order: (1) the
    explicit `KNOWN_KERNEL_NAMES`/`BF16_ONLY_KERNEL_NAMES` ground truth;
    (2) any GEMM-family name (the tile-variant explosion is a cuBLAS
    heuristic, not a stable vocabulary — module doc); (3) a BF16 elementwise
    name whose f32 TWIN (`"bf16"` -> `"f32"`) is already known (the twin
    RULE the contract asks for, rather than hand-listing every observed
    `*_bf16` name)."""
    if name in KNOWN_KERNEL_NAMES or name in BF16_ONLY_KERNEL_NAMES:
        return True
    if GEMM_NAME_RE.search(name):
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

# `fast_sum`/`fast_max`: matched against TWO declared row counts (softmax
# reduction rows -> C-ATTN; LN row count -> C-LN, eager only); every OTHER
# row count is `LOSS/REDUCE` (module doc).
ROW_REDUCTION_KERNEL_NAMES: frozenset[str] = frozenset(
    {"fast_max_f32", "fast_sum_f32", "fast_max_bf16", "fast_sum_bf16"}
)

# The eager LayerNorm's own flat-elementwise std/reciprocal step (module
# doc, "C-LN extended"). Matched at `total_threads` covering `ln_row_count`.
LN_ROW_FLAT_KERNEL_NAMES: frozenset[str] = frozenset({"usqrt_f32", "urecip_f32", "usqrt_bf16", "urecip_bf16"})

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

# Declared architecture constants for the two OpenCLIP towers (see this
# module's doc, "Declared per-tower architecture constants" below).
TOWER_ARCH: dict[str, dict[str, object]] = {
    "clip-text": {"sites_per_layer": 4, "width": 512, "heads": 8, "mlp_ratio": 4, "declared_seq": None},
    "clip-vision": {"sites_per_layer": 4, "width": 768, "heads": 12, "mlp_ratio": 4, "declared_seq": 50},
}

# Wire-default LoRA rank (contract, "Wire defaults: rank 8, alpha 16, ...") —
# used ONLY to identify parameter-scale (not activation-scale) elementwise
# ops for the `OPTIMIZER` bucket (module doc). Never used to derive an
# activation-tensor shape.
LORA_RANK = 8

# The `--task` value each tower's `finetune-run` manifest carries — used
# only for a friendlier reason string, never for signature derivation.
TOWER_FAMILY_UNKNOWN = "unknown"


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
        `rows*heads*seq*seq` — module doc, "the attention SCORE/PROB
        tensor's own elementwise ops"."""
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
        `rows*seq` — module doc, "C-LN extended". Distinct from
        `attn_softmax_rows` (`rows*heads*seq`) by construction (`heads>1`
        for both towers), so no collision with the attention reduction
        rule."""
        return self.rows.value * self.seq.value

    def param_scale_elements(self) -> frozenset[int]:
        """Element counts of the trainable PARAMETER tensors themselves
        (LoRA A/B matrices, Linear biases, LN gamma/beta) at the wire-
        default rank — module doc, "`OPTIMIZER` extended". Distinct from
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


def derive_signatures(tower: str, manifest: dict) -> TowerSignatures:
    """Derive `TowerSignatures` for `tower` from its leg's `manifest.json`
    (witnessed fields) plus `TOWER_ARCH`'s declared constants (see module
    doc). Raises `SignatureError` for an unrecognised tower or a manifest
    missing a field this tower needs witnessed."""
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
    (`depths=[2,2,6,2]`, `num_attention_heads=[4,8,16,32]`, `window_size=8`,
    `hidden_size=768`, `hidden_act="gelu"` — fetched and cross-checked
    against the values below at pass-2 time; `hidden_act="gelu"` also
    confirms `crates/jammi-encoders/src/htsat_audio.rs`'s own load-time
    refusal of anything else, §D3). NOT cross-checked against a real
    KERNEL export — no HTSAT `by_kernel_and_grid` leg has landed yet.
    Returned for documentation/forward-compatibility; `derive_signatures`
    does not yet support `tower="htsat"` (raises `SignatureError`) — the
    kernel-name<->shape MAPPING is read off the first real HTSAT export,
    never guessed ahead of it (this module's own §D3 discipline, applied
    to itself)."""
    depths = [2, 2, 6, 2]
    num_heads = [4, 8, 16, 32]
    window_size = 8
    return [
        {"stage": i, "depth": d, "heads": h, "window_size": window_size}
        for i, (d, h) in enumerate(zip(depths, num_heads))
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
    the same tolerance pass 1's GELU-shape match used)."""
    total = _entry_total_threads(entry)
    block_size = _entry_block_size(entry)
    return elements <= total < elements + block_size


def _is_row_count_grid(entry: dict, count: int) -> bool:
    """`True` iff `entry`'s grid is a "one block per row" launch over
    exactly `count` rows — `grid[0] == count`, `grid[1] == grid[2] == 1`
    (block size is irrelevant to this shape, same as pass 1's softmax-
    reduction rule)."""
    grid = entry["grid"]
    return grid[0] == count and grid[1] == 1 and grid[2] == 1


def classify_kernel(entry: dict, sig: TowerSignatures, tower: str) -> str | None:
    """`chain name` this `(kernel, grid, block)` row belongs to, or `None`
    (UNATTRIBUTED — including "unknown kernel name", handled separately by
    the caller so it can also feed the INVALID gate). See module doc for
    the evidence behind every branch. Checks run in priority order; an
    earlier match wins, but several branches deliberately FALL THROUGH
    (rather than returning early) when their specific shape does not match,
    so a kernel whose name is shape-gated for one chain (e.g. `bmul_f32`
    for `C-GELU`) can still land in a DIFFERENT chain at a DIFFERENT shape
    (e.g. `C-ATTN-<tower>`'s own elementwise tier) instead of falling all
    the way through to `UNATTRIBUTED`."""
    name = entry["kernel"]
    grid = entry["grid"]

    # --- Name-only chains, independent of shape -----------------------
    if "cast" in name.lower():
        return CHAIN_CAST
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

    # --- Eager LN's own flat std/reciprocal step (module doc). Falls
    #     through (does not return None) when the shape does not match,
    #     so e.g. a stray `usqrt_f32` at parameter scale can still land in
    #     `OPTIMIZER` below.
    chain: str | None = None
    if name in LN_ROW_FLAT_KERNEL_NAMES and _in_element_range(entry, sig.ln_row_count()):
        chain = CHAIN_LN

    # --- GELU: exclusive by name; shape-gated for the shared elementwise
    #     names (falls through on a shape miss, same reasoning as above).
    if chain is None and name in GELU_EXCLUSIVE_KERNEL_NAMES:
        chain = CHAIN_GELU
    if chain is None and name in GELU_SHAPE_KERNEL_NAMES and _in_element_range(entry, sig.gelu_shape_elements()):
        chain = CHAIN_GELU

    # --- Batched-attention grid signature: relational, NAME-INDEPENDENT
    #     (module doc, "the batched-matmul grid signature") — catches any
    #     GEMM-family kernel AND anonymous kernels alike.
    if chain is None and sig.attn_batch_count() in grid:
        chain = chain_attn(tower)

    # --- GEMM-family by name (not carrying the attention batch count).
    if chain is None and GEMM_NAME_RE.search(name):
        chain = CHAIN_BASE_GEMM
    if chain is None and name == "splitKreduce_kernel":
        chain = CHAIN_BASE_GEMM

    # --- Attention tensor's own elementwise ops (mask add, scale, cast,
    #     backward exp/square, ...) — module doc, "the attention SCORE/PROB
    #     tensor's own elementwise ops".
    if chain is None and _in_element_range(entry, sig.attn_shape_elements()):
        chain = chain_attn(tower)

    # --- Parameter-scale bookkeeping (module doc, "OPTIMIZER extended").
    if chain is None:
        for elements in sig.param_scale_elements():
            if _in_element_range(entry, elements):
                chain = CHAIN_OPTIMIZER
                break

    # --- The three activation-tier elementwise buckets.
    if chain is None and _in_element_range(entry, sig.qkv_shape_elements()):
        chain = CHAIN_ELEMENTWISE_QKV
    if chain is None and _in_element_range(entry, sig.out_shape_elements()):
        chain = CHAIN_ELEMENTWISE_OUT
    if chain is None and _in_element_range(entry, sig.gelu_shape_elements()):
        chain = CHAIN_ELEMENTWISE_MLP

    return chain


@dataclass
class ChainResult:
    status: str  # "measured" | "absent" | "requires_d2_delta" | "measured_via_d2_delta"
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
    census: dict, sig: TowerSignatures, tower: str
) -> tuple[dict[str, ChainResult], list[dict[str, object]], list[str]]:
    """Classify every `by_kernel_and_grid` row into a chain bucket.

    Returns `(chains, unknown_kernel_rows, invalid_reasons)`. `chains`
    always carries an entry for every `declared_chains_for_tower(tower)`
    name (per-test invariant: "every declared chain present or explicitly
    absent") plus `UNATTRIBUTED` — `status="absent"` means literally zero
    rows matched, distinct from `status="measured"` with `gpu_busy_us=0.0`.
    `C-LORA` is NOT added here (it needs a cross-leg D2 delta — see
    `attach_lora_via_d2_delta`).
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
        if not is_known_kernel_name(name):
            share = (us_per_step / total_busy_us) if total_busy_us else 0.0
            unknown_rows.append(
                {"kernel": name, "grid": entry.get("grid"), "share_gpu_busy": share}
            )
            if share > UNKNOWN_KERNEL_SHARE_LIMIT:
                invalid_reasons.append(
                    f"kernel {name!r} (grid={entry.get('grid')!r}) is not a known kernel name "
                    f"and carries share_gpu_busy={share:.4f} > {UNKNOWN_KERNEL_SHARE_LIMIT} — "
                    "the mapping cannot classify it and refuses to guess"
                )
            buckets[CHAIN_UNATTRIBUTED] += us_per_step
            matched_kernels[CHAIN_UNATTRIBUTED].append(name)
            continue
        chain = classify_kernel(entry, sig, tower)
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


def leg_decision_grade(row: dict) -> tuple[bool, str | None]:
    """`(decision_grade, reason)` for one leg's already-built attribution
    row (contract: "Validity gate: ... UNATTRIBUTED <= 5% of gpu_busy").
    `reason` is `None` iff `decision_grade` is `True`."""
    if row.get("verdict") != VERDICT_VALID:
        return False, "leg is INVALID (see this leg's own `reasons`)"
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


def attach_lora_via_d2_delta(leg_rows: list[dict]) -> None:
    """Mutates `leg_rows` IN PLACE: for every tower with a VALID F32 `A`
    leg (`A1`; `D2` is F32-only per the contract's legs table) AND a VALID
    `D2` leg, sets that A-leg's `chains["C-LORA"]` to the measured
    `A.gpu_busy_us_per_step - D2.gpu_busy_us_per_step` delta (contract:
    "A1 - D2 is that one kernel"). Every OTHER leg (including `D2` itself,
    `A2`/BF16, and any tower missing its `D2` twin) gets the declared-absent
    `status: "requires_d2_delta"` placeholder — never a guess.

    Also attaches the ADVISORY `d1_minus_d2_busy_us_advisory` cross-check
    (contract: "D1 - D2 = C-LN on CLIP, C-LN + C-GELU on HTSAT") to the
    A-leg's row when both `D1` and `D2` are present and VALID for that
    tower.
    """
    by_tower: dict[str, dict[str, dict]] = {}
    for row in leg_rows:
        tower = row.get("tower")
        if not isinstance(tower, str):
            continue
        role = row.get("_role")
        if row.get("dtype") != "f32" or role not in ("A", "D1", "D2"):
            continue
        if row.get("verdict") != VERDICT_VALID:
            continue
        by_tower.setdefault(tower, {})[role] = row

    for row in leg_rows:
        chains = row.get("chains")
        if not isinstance(chains, dict):
            continue
        role = row.get("_role")
        # `C-LORA` is a per-TOWER quantity attributed to the A leg alone (it
        # is literally "the A leg's busy minus its D2 twin's") — a D1/D2/
        # other-role leg's OWN `chains` dict never gets this key at all,
        # measured or placeholder: D2 IS the reference subtracted, not a
        # consumer of its own delta.
        if role != "A":
            continue
        tower = row.get("tower")
        legs_for_tower = by_tower.get(tower, {}) if isinstance(tower, str) else {}
        a_leg = legs_for_tower.get("A")
        d2_leg = legs_for_tower.get("D2")
        if a_leg is not None and d2_leg is not None and row is a_leg:
            delta = a_leg["_gpu_busy_us_per_step"] - d2_leg["_gpu_busy_us_per_step"]
            wall_s = a_leg.get("_wall_s_per_step")
            chains[CHAIN_LORA] = ChainResult(
                status="measured_via_d2_delta",
                gpu_busy_us=delta,
                share_gpu_busy=(delta / a_leg["_gpu_busy_us_per_step"])
                if a_leg["_gpu_busy_us_per_step"]
                else None,
                share_wall=(delta / 1e6 / wall_s) if wall_s else None,
                note="A1.gpu_busy_us_per_step - D2.gpu_busy_us_per_step",
            ).as_dict()
            d1_leg = legs_for_tower.get("D1")
            if d1_leg is not None:
                row["d1_minus_d2_busy_us_advisory"] = (
                    d1_leg["_gpu_busy_us_per_step"] - d2_leg["_gpu_busy_us_per_step"]
                )
        else:
            chains[CHAIN_LORA] = ChainResult(
                status="requires_d2_delta",
                note="no literal GPU kernel realizes lora_linear_fused by name; this chain is "
                "measured only via the tower's A1-D2 delta (contract Attribution/§D3), and the "
                "matching D2 leg is not present/VALID for this leg's (tower, dtype)",
            ).as_dict()


# --- The two-sided decision rule (contract "Decision rule") -----------------

ACTIVATE_WALL_THRESHOLD = 0.10
DECLINE_COMBINED_THRESHOLD = 0.05

# `(port name) -> (tower, chain key that port's numbers come from)`. Built
# by `candidate_ports_for_tower` per tower — `C-MLP-<tower>` reads the
# `C-GELU` chain's OWN numbers (contract: "C-MLP-CLIP-text/vision = the
# C-GELU quick_gelu bucket"), never a separately-computed value.
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


def decide_candidate_port(port_name: str, chain_key: str, tower: str, legs: list[dict]) -> dict:
    """The contract's two-sided rule for ONE candidate port. `legs` must
    still carry the internal `_role`/`_gpu_busy_us_per_step` fields (i.e.
    called BEFORE `build_report` strips them, same requirement
    `attach_lora_via_d2_delta` has)."""
    a_legs = [
        leg
        for leg in legs
        if leg.get("tower") == tower and leg.get("_role") == "A" and leg.get("verdict") == VERDICT_VALID
    ]
    a1 = next((leg for leg in a_legs if leg.get("dtype") == "f32"), None)
    a2 = next((leg for leg in a_legs if leg.get("dtype") == "bf16"), None)

    a1_grade, a1_reason = leg_decision_grade(a1) if a1 is not None else (False, "no A1 (F32) leg present")
    if not a1_grade:
        return {
            "port": port_name,
            "tower": tower,
            "chain": chain_key,
            "verdict": "UNRESOLVED",
            "reason": f"A1 not decision-grade: {a1_reason}",
        }

    a2_grade, a2_reason = leg_decision_grade(a2) if a2 is not None else (False, "no A2 (BF16) leg present")
    f32_only_note = ""
    if a2 is not None and not a2_grade:
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


def decide_all_candidates(legs: list[dict]) -> list[dict]:
    """Every candidate port's decision, across every tower this module
    knows candidate ports for. Called BEFORE `build_report` strips the
    internal `_role`/`_gpu_busy_us_per_step` fields."""
    decisions: list[dict] = []
    for tower in CANDIDATE_CHAIN_TOWERS:
        for port_name, chain_key in candidate_ports_for_tower(tower).items():
            decisions.append(decide_candidate_port(port_name, chain_key, tower, legs))
    return decisions


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def attribute_leg(leg_dir: Path) -> dict:
    """One leg's attribution row. NEVER raises for a leg-level problem —
    every failure becomes this row's `verdict="INVALID"` + `reasons`, the
    same discipline `profile_421_merge.merge_leg` uses, so one bad leg
    cannot discard the others."""
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
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row

    tower = manifest.get("tower")
    dtype = manifest.get("dtype")
    row["tower"] = tower
    row["dtype"] = dtype
    kernels_disabled = manifest.get("kernels_disabled")
    if not isinstance(kernels_disabled, list):
        row["reasons"] = [f"{leg_id}: manifest.kernels_disabled is not a list"]
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row
    row["_role"] = leg_role([str(k) for k in kernels_disabled], str(tower))

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
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row

    if census.get("excluded_from_chain_attribution") is True:
        reasons.append(
            f"{leg_id}: census.excluded_from_chain_attribution is true — this leg's signatures "
            "are not fixed-shape and cannot be attributed by this module (kernel_census.py's own "
            "E1 exclusion)"
        )
        row["reasons"] = reasons
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row

    if not isinstance(tower, str):
        reasons.append(f"{leg_id}: manifest.tower is not a string ({tower!r})")
        row["reasons"] = reasons
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row

    try:
        sig = derive_signatures(tower, manifest)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row
    row["signatures"] = sig.as_dict()

    try:
        chains, unknown, classify_reasons = attribute_census(census, sig, tower)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row
    reasons.extend(classify_reasons)
    row["unknown_kernels"] = unknown
    row["chains"] = {name: result.as_dict() for name, result in chains.items()}
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
    grade, grade_reason = leg_decision_grade(row)
    row["decision_grade"] = grade
    row["decision_grade_reason"] = grade_reason
    return row


def _leg_dirs(legs_dir: Path) -> list[Path]:
    return sorted(
        d
        for d in legs_dir.iterdir()
        if d.is_dir() and d.name != "p2-bf16" and (d / "manifest.json").is_file()
    )


def build_report(legs_dir: Path) -> dict:
    legs = [attribute_leg(d) for d in _leg_dirs(legs_dir)]
    attach_lora_via_d2_delta(legs)
    # Recompute decision_grade for A legs now that `C-LORA` may have been
    # attached (it never changes the UNATTRIBUTED share, so this is a
    # no-op in practice, but keeps `decision_grade` a function of the
    # FINAL published `chains` dict rather than an intermediate one).
    for row in legs:
        grade, grade_reason = leg_decision_grade(row)
        row["decision_grade"] = grade
        row["decision_grade_reason"] = grade_reason
    candidate_decisions = decide_all_candidates(legs)
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
        "legs": legs,
        "candidate_decisions": candidate_decisions,
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
        f"{'chain':<20} {'status':<22} {'share_busy':>10} {'share_wall':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report.get("legs", []):
        chains = row.get("chains") or {}
        grade = "yes" if row.get("decision_grade") else "no"
        if not chains:
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {grade:<6} {'(none)':<20} {'':<22} {'':>10} {'':>10}"
            )
            continue
        for chain_name in sorted(chains):
            entry = chains[chain_name]
            share_busy = entry.get("share_gpu_busy")
            share_wall = entry.get("share_wall")
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {grade:<6} {chain_name:<20} {entry.get('status', ''):<22} "
                f"{(f'{share_busy:.4f}' if isinstance(share_busy, (int, float)) else ''):>10} "
                f"{(f'{share_wall:.4f}' if isinstance(share_wall, (int, float)) else ''):>10}"
            )
    lines.append("")
    lines.append("candidate port decisions:")
    for decision in report.get("candidate_decisions", []):
        lines.append(f"  {decision['port']:<20} {decision['verdict']:<12} {decision['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_attribute.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", required=True, help="a profile_421_legs.sh $OUT_DIR/legs to attribute")
    ap.add_argument("--out", help="write the attribution report here (default: stdout)")
    ap.add_argument(
        "--no-table",
        action="store_true",
        help="suppress the human-readable table on stderr (JSON output is unaffected)",
    )
    args = ap.parse_args(argv)

    legs_dir = Path(args.legs_dir)
    if not legs_dir.is_dir():
        print(f"::error::profile_421_attribute: {legs_dir} is not a directory", file=sys.stderr)
        return 1

    report = build_report(legs_dir)
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
