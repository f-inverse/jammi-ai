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
for it is never by-name: the `A1 - D2` busy/wall delta. `KNOWN_KERNEL_NAMES`
gates: a kernel name outside the validated vocabulary above 1% of a leg's
busy makes the WHOLE LEG `INVALID` rather than silently folding into
`UNATTRIBUTED` — the discipline this module has followed since pass 1 and
extends, never loosens, below.

Pass 1 left roughly 85% of `clip-text-A1`'s busy `UNATTRIBUTED`, so the
validity gate's own bound (`UNATTRIBUTED <= 5% of gpu_busy`) was
unsatisfiable by construction.

## Pass 2 (v2.5 §Attribution/§D3 — named-bucket extension)

Read the mapping off the real `clip-text-A2` (BF16), `clip-text-D1`,
`clip-text-D2`, `clip-vision-A1`/`A2`/`D1`/`D2` exports (pod `p421` run 2,
tip `c1b0b0ba`) and named a set of EVIDENCE-BACKED buckets that absorb the
overwhelming majority of what pass 1 left `UNATTRIBUTED`.

## Pass 3 (this pass — adversarial-audit fold, 7 findings)

Pass 2's own priority order over-attributed two RELATIONAL rules and
under-split the elementwise tiers; this pass's every change is grounded in
a specific `(kernel, grid)` row from a real export, same discipline as
pass 1/2 — see each subsection.

### `C-ATTN-<tower>`: the pre-registered signature does not move

The signature stays EXACTLY what the contract pre-registered: kernels at
the `[24, h, S, S]` element count (scores/probs elementwise, softmax
fwd/bwd, dropout on probs, the cast/compare rows at that shape) plus the
batched GEMMs whose grid carries `rows*heads`. Two priority bugs are fixed,
both evidenced, neither moving the signature itself:

1. **`cast`-named rows are shape-gated, not name-exclusive.** Pass 2's
   `CAST` check returned immediately on `"cast" in name.lower()`, before
   the attention-shape check ever ran — so `cast_u8_f32`/`cast_u8_bf16` at
   the attention tensor's own element count (evidenced: `clip-text-A1`
   `cast_u8_f32 grid=[1112,1,1]`, the exact attention-shape grid every
   other attention elementwise name in this module shares) were
   misclassified `CAST` instead of `C-ATTN-<tower>`. `classify_kernel` now
   checks the attention shape FIRST for any cast-named row; a cast row at
   any OTHER shape still lands the generic `CAST` bucket, unchanged.
2. **The batched-GEMM grid rule is gated on grid POSITION 2**, the
   position every real match uses (`clip-text-A2`'s anonymous `Kernel2`
   rows at `grid=[2,1,192]`/`[4,1,192]`/`[12,1,192]`; `clip-vision-A1`'s
   `magma_sgemmEx_kernel` at `grid=[1,2,288]`; `clip-text-A1`'s
   `ampere_sgemm_*` attention GEMMs at `grid=[1,1,192]`), not "anywhere in
   the 3-tuple" — pass 2's `attn_batch_count() in grid` membership test
   would have falsely matched a hypothetical non-attention row carrying
   `192`/`288` in `grid[0]` (a real value for some base-projection tile
   grids at other shapes); `classify_kernel` now checks `grid[2] ==
   attn_batch_count()` exactly, and the test suite carries a NEGATIVE
   CONTROL — a non-attention row with `192`/`288` in `grid[0]` (not
   `grid[2]`) — proving the membership test does NOT fire there.

**`ucopy_*`/`copy2d_*` (permute/reshape copies) do NOT join `C-ATTN`,
named or by grid** — pass 2 let them fall through into whichever
activation-tier bucket their element count matched (`ELEMENTWISE-<tier>`);
this pass gives them their OWN bucket, `PERMUTE/RESHAPE` (see below), and
adds a leg-level diagnostic field, `outside_signature_plausibly_attention`
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
already keys off). Pass 2 lumped every non-`C-GELU` elementwise name at
these three shapes into one bucket per tier (`ELEMENTWISE-<tier>`); this
pass splits by kernel NAME-CLASS instead, since a single "elementwise
tier" bucket conflated a Linear's own bias-add with an unrelated
permute/reshape copy with (on D1 legs) the eager LayerNorm's own
normalize step:

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

### `BASE-GEMM`: every other GEMM-family kernel, plus anonymous grid-family rules

Any kernel name containing `"gemm"` (case-insensitive) that does NOT carry
the attention batch count at `grid[2]` is `BASE-GEMM` (unchanged from pass
2); `splitKreduce_kernel` is `BASE-GEMM` unconditionally (unchanged).
**`OPTIMIZER`'s own parameter-scale rule (below) is now gated on 1-D
launch geometry** (`grid[1]==1 and grid[2]==1`, i.e. `grid=[ceil(N/b),1,1]`
— a genuine elementwise/bookkeeping launch never tiles a second or third
grid dimension): evidenced by `clip-text-A2`'s anonymous `Kernel2 grid=
[4,1,24]`, whose `total_threads=12,288` EXACTLY matches
`LORA_RANK*3*width` (a param-scale element count) yet is plainly a 3-D
tiled launch, not a 1-D bookkeeping op — pass 2's ungated rule would have
(and, before this fold, silently did) swept it into `OPTIMIZER` by shape
coincidence, hiding a row this module cannot actually explain.

**Anonymous kernels (`Kernel2`, and any future unsymbolized launch) are
classified ONLY by grid-family rules**, in this order: (1) the batched-
attention grid rule above (`grid[2] == attn_batch_count()`); (2) otherwise
`BASE-GEMM` ONLY if the grid is a genuine 3-D tile grid — EVERY one of its
three dimensions `> 1` (a plausible M-tile x N-tile x batch/split-K launch
never degenerates to size `1` in any dimension; a `1` in some position is
a 1-D-ish bookkeeping/reduction launch, `OPTIMIZER`'s own domain, not a
tiled matmul). Evidenced BOTH ways on `clip-text-D1`/`D2`: `Kernel2
grid=[8,2,28]` (every dimension `>1`) -> `BASE-GEMM` on both legs
(cuBLAS picked a different, unsymbolized algorithm once the eager
LoRA/LN arithmetic shifted the surrounding shapes/strides — the SAME
anonymous grid on two independent legs, corroborating); `clip-text-A2`'s
`Kernel2 grid=[16,1,10]`/`[4,1,24]`/`[12,1,8]`/`[128,2,1]` (each has a `1`
in some position) do NOT classify here and remain `UNATTRIBUTED`.

**An anonymous or unknown-name row that no rule classifies COUNTS toward
the 1%-of-busy unknown-kernel gate** — `KNOWN_KERNEL_NAMES` no longer
admits a name via the `"gemm"` substring, and the literal `Kernel2` is
REMOVED from it: the gate now applies to EXACT names only (plus the BF16-
twin rule, unchanged). Every real GEMM-family tile-variant name this
module has actually observed (all `ampere_sgemm_*`/`ampere_bf16_*gemm*`/
`magma_sgemmEx_kernel` variants pulled across all eight real CLIP legs) is
hand-added to `KNOWN_KERNEL_NAMES` by its exact literal string — the
GEMM-name-implies-known SHORTCUT is gone, but nothing that was actually
observed and real stops being admitted. Consequence, measured: on
`clip-text-A2`, `Kernel2 grid=[16,1,10]` (`share_gpu_busy=1.400%`) and
`Kernel2 grid=[4,1,24]` (`share_gpu_busy=1.367%`) both exceed
`UNKNOWN_KERNEL_SHARE_LIMIT` and INVALIDATE that leg; on `clip-vision-A2`,
`Kernel2 grid=[6,1,18]` (`share_gpu_busy=1.428%`) does the same. Both A2
legs are therefore NOT decision-grade — per the contract, the candidate-
port decisions for BOTH `clip-text` and `clip-vision` are F32-only, and
this module's own `decide_candidate_port` says so in the reason string.

### Realized-gain chains are NOT chain-partition members

Pass 2 attached `C-LORA`'s `A1-D2` delta directly into the A-leg's own
`chains` dict, alongside every partition-member bucket — a busy/wall
DELTA between two different leg configurations is not a share of any one
leg's own busy, and (since the fused configuration is usually cheaper)
the delta is frequently NEGATIVE, which would have shown up as a negative
`share_gpu_busy` in the very column this module's own tests assert is
`>= 0` for every partition member. This pass moves `C-LORA` and `C-LN`'s
realized-fused-vs-eager numbers OUT of every leg's `chains` dict entirely,
into a SEPARATE top-level `realized_gains` list (`compute_realized_gains`)
— one entry per `(chain, tower)`, carrying `busy_delta_us_per_step`
(`eager - fused`, so POSITIVE when eager is slower, as every real leg
pulled so far shows), `wall_delta_s_per_step` (sourced from the NAMED
`--merge-json` report's own `per_step.wall_s_per_step` per leg — "the run
reports' `train_run_wall_s` differenced exactly as the merge does",
contract; never re-derived from this module's own census-only `wall_s`),
`share_of_baseline_wall` (the delta as a fraction of the FUSED leg's own
wall — the shipped configuration is the baseline), and a `direction`
string stating which leg was slower. `C-LORA` = `A1` vs `D2`; `C-LN` =
`D2` vs `D1` (D1 additionally disables LoRA, so `D1-D2` isolates LN alone
on CLIP — §D3). `C-GELU-HTSAT`'s realized gain is deferred to the HTSAT
pass (no HTSAT leg is attributed by this module yet).

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

### Prose corrections (adversarial-audit finding 6)

Every number in this docstring is RE-DERIVED from the real census exports
at this tip, never carried over as a literal expectation in any test.
Measured on the real, full (not fixture-cut) `clip-text-A1` census:
`BASE-GEMM = 35.08%` of busy (the chain total, NOT one row); the single
NAMED row `ampere_sgemm_128x64_nn grid=[4,29,7]` is `8.08%` of busy alone
(pass 2's docstring conflated these two numbers — this is the correction).
On `clip-text-A2`, `BASE-GEMM = 15.48%` (`~15.5%`) while `C-ATTN-clip-text
= 19.41%` is the LARGEST bucket on that leg (BF16 halves every elementwise
op's byte traffic, shrinking `BASE-GEMM`'s share relative to attention's
own fixed kernel-launch overhead) — descriptive only: `clip-text-A2` is
itself `INVALID` at this tip (finding 3's own two anonymous-`Kernel2`
unknown-name rows), so neither share feeds a decision. Pass 2's `UNATTRIBUTED` claim
("`1.7-5.0%` across all five") and `BASE-GEMM` claim ("`33-46%` across the
five") are WRONG under this pass's rules and are not restated as ranges
here — see `main`'s own documented run, pasted in the hand-off, for the
exact `UNATTRIBUTED`/`BASE-GEMM` share on EVERY one of the eight real CLIP
legs at THIS tip.

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

**`eq_f32`'s mechanism is a HYPOTHESIS, corrected**: pass 2's doc labelled
`cast_u8_f32`/`eq_f32` together as "the boolean-mask cast/compare feeding"
the additive attention mask. `clip-vision-A1` carries the IDENTICAL
`eq_f32` row (same shape, same order of magnitude) as `clip-text-A1`, yet
OpenCLIP vision attention has no learned/data-dependent additive mask at
all (the encoder-side cascade synthesizes an all-zero padding mask for
vision — contract, "candidate ports"); an op that is genuinely mask-only
could not appear identically on a tower with no mask to compute. `eq_f32`
is therefore corrected to a HYPOTHESIS — most plausibly part of the
softmax backward's own routing (e.g. a numerically-stable-softmax
backward's own equality/selection step), not a mask compare — and is
NOT asserted as mask-related anywhere in this module or its tests.
`cast_u8_f32`'s own role is UNCHANGED by this correction (it is plausible
mask-adjacent on `clip-text` alone, but this module does not assert that
either — both names are classified by SHAPE, never by an asserted
mechanism).

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

SCHEMA_VERSION = 3

VERDICT_VALID = "VALID"
VERDICT_INVALID = "INVALID"

# The merge script's own verdict string (`profile_421_merge.VERDICT_VALID`)
# — kept as an independent literal here (this module does not import the
# merge script) because they happen to share the exact same convention;
# `test_merge_verdict_literal_matches_the_merge_script` pins the two files
# never silently drift apart.
MERGE_VERDICT_VALID = "VALID"

# --- Original (pass 1) chain names -----------------------------------------
CHAIN_LORA = "C-LORA"
CHAIN_LN = "C-LN"
CHAIN_GELU = "C-GELU"
CHAIN_UNATTRIBUTED = "UNATTRIBUTED"

# --- Pass-2 named buckets ----------------------------------------------------
CHAIN_BASE_GEMM = "BASE-GEMM"
CHAIN_DROPOUT = "DROPOUT"
CHAIN_CAST = "CAST"
CHAIN_OPTIMIZER = "OPTIMIZER"
CHAIN_EMBED_GATHER = "EMBED/GATHER"
CHAIN_LOSS_REDUCE = "LOSS/REDUCE"
CHAIN_PATCH_EMBED = "C-PATCH-EMBED"

# --- Pass-3 named buckets (adversarial-audit fold, findings 1/2) -----------
CHAIN_PERMUTE_RESHAPE = "PERMUTE/RESHAPE"
CHAIN_BIAS_RESIDUAL_QKV = "BIAS/RESIDUAL-QKV"
CHAIN_BIAS_RESIDUAL_OUT = "BIAS/RESIDUAL-OUT"
CHAIN_BIAS_RESIDUAL_MLP = "BIAS/RESIDUAL-MLP"
CHAIN_ELEMENTWISE_OTHER_QKV = "ELEMENTWISE-OTHER-QKV"
CHAIN_ELEMENTWISE_OTHER_OUT = "ELEMENTWISE-OTHER-OUT"
CHAIN_ELEMENTWISE_OTHER_MLP = "ELEMENTWISE-OTHER-MLP"

# --- HTSAT-only, declared but unverified (module doc, "HTSAT") -------------
CHAIN_GELU_HTSAT = "C-GELU-HTSAT"
CHAIN_WINDOWING = "WINDOWING"
CHAIN_FRONT_FUSION = "FRONT-FUSION"

# Chain buckets attached to every CLIP leg's `chains` dict (present-or-absent,
# same discipline pass 1 established for LN/GELU/ATTN — see `attribute_census`).
# `C-LORA` is deliberately ABSENT from this list (pass 3, finding 4): it is a
# realized-gain NUMBER (`compute_realized_gains`), never a chain-partition
# member — see the module doc, "Realized-gain chains are NOT chain-partition
# members".
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
    # HTSAT: only the two name-only chains this pass can classify without a
    # real export (module doc, "HTSAT"); `WINDOWING`/`FRONT-FUSION` have no
    # mapping yet and are deliberately NOT declared here (declaring a chain
    # with no matching rule would silently report it "absent" forever,
    # which is indistinguishable from "not yet implemented" — dishonest).
    return (CHAIN_GELU_HTSAT, chain_attn(tower))


# The exact kernel-name vocabulary observed across all eight real CLIP legs
# pulled from pod `p421` run 2 at `c1b0b0ba` (`clip-text-{A1,A2,D1,D2}`,
# `clip-vision-{A1,A2,D1,D2}`). Pass 3 (adversarial-audit finding 3) ADDS the
# eight `ampere_bf16_s16816gemm_*` tile-variant names actually observed on
# `clip-text-A2`/`clip-vision-A2` (previously admitted only via the removed
# `"gemm"`-substring shortcut in `is_known_kernel_name` — see below) and
# REMOVES the literal `Kernel2`: an anonymous kernel is admitted ONLY by
# `classify_kernel`'s own grid-family rules now (module doc), never by a
# blanket "we've seen an anonymous kernel before" entry in this vocabulary.
KNOWN_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "badd_f32",
        "ampere_sgemm_32x128_tn",
        "ampere_sgemm_32x128_nt",
        "ampere_sgemm_128x64_nn",
        "ampere_sgemm_128x64_nt",
        "ampere_sgemm_128x64_tn",
        "bmul_f32",
        "ampere_sgemm_128x32_tn",
        "ampere_sgemm_128x32_nn",
        "ampere_sgemm_128x32_sliced1x4_nt",
        "dropout_fwd_f32",
        "fast_sum_f32",
        "affine_f32",
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
        # BF16 GEMM tile variants, evidenced on `clip-text-A2`/`clip-vision-A2`
        # (all EIGHT distinct tile-variant names observed across both legs —
        # `is_known_kernel_name` no longer admits these via the `"gemm"`
        # substring, so they are hand-listed here, same discipline as every
        # other GEMM name above).
        "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn",
        "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn",
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nn",
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn",
        "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn",
        "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn",
        "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn",
        "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn",
    }
)

# Genuinely BF16-only kernel names with NO f32 twin in `KNOWN_KERNEL_NAMES`
# (`name.replace("bf16", "f32")` does not land on a known name) — hand
# admitted from the real `clip-text-A2` export, same discipline as every
# other entry in `KNOWN_KERNEL_NAMES` above, kept in a separate constant
# only so the twin-rule's own doc/tests can show what it does NOT cover.
BF16_ONLY_KERNEL_NAMES: frozenset[str] = frozenset({"cast_add_bf16", "cast_scale_bf16_f32"})

# Any kernel name containing this (case-insensitive) is GEMM-family
# (cuBLAS/CUTLASS/MAGMA) for CLASSIFICATION purposes (`classify_kernel`'s
# own `BASE-GEMM`-by-name rule) — this is UNCHANGED from pass 2. It is no
# longer used by `is_known_kernel_name` (pass 3, finding 3): admission into
# the validated vocabulary is by EXACT name only now, never a substring.
GEMM_NAME_RE = re.compile(r"gemm", re.IGNORECASE)


def is_known_kernel_name(name: str) -> bool:
    """`True` iff `name` is admitted ground truth for the INVALID-by-
    unknown-kernel gate (`UNKNOWN_KERNEL_SHARE_LIMIT`) — NOT the same
    question as "does this row land in a named chain" (`classify_kernel`
    has its own, separate, grid-family rules for anonymous kernels — module
    doc). Pass 3 (adversarial-audit finding 3) narrows this to TWO admission
    paths, exact names only: (1) the explicit `KNOWN_KERNEL_NAMES`/
    `BF16_ONLY_KERNEL_NAMES` ground truth; (2) a BF16 elementwise name whose
    f32 TWIN (`"bf16"` -> `"f32"`) is already known (the twin RULE the
    contract asks for, rather than hand-listing every observed `*_bf16`
    name). The pass-2 `"gemm"`-substring shortcut and the literal `Kernel2`
    entry are BOTH removed: a name this module has not actually observed
    and hand-added is unknown, full stop, even if it looks GEMM-shaped or
    is anonymous — `classify_kernel`'s grid-family rules are the ONLY path
    an anonymous kernel can be CLASSIFIED through, and are entirely
    independent of this admission question."""
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

# `fast_sum`/`fast_max`: matched against TWO declared row counts (softmax
# reduction rows -> C-ATTN; LN row count -> C-LN, eager only); every OTHER
# row count is `LOSS/REDUCE` (module doc).
ROW_REDUCTION_KERNEL_NAMES: frozenset[str] = frozenset(
    {"fast_max_f32", "fast_sum_f32", "fast_max_bf16", "fast_sum_bf16"}
)

# The eager LayerNorm's own flat-elementwise mean/var/std/reciprocal steps
# (module doc, "split by kernel NAME-CLASS") — matched at `total_threads`
# covering `ln_row_count`, ONLY when `layer_norm_fused` is disabled for this
# leg (`ln_disabled` — see `classify_kernel`). Pass 3 (finding 2) EXTENDS
# pass 2's `usqrt`/`urecip`-only set with `bmul`/`bsub`/`usqr` (evidenced on
# `clip-text-D1`/`clip-vision-D1` for `bsub`/`usqr`; `bmul` is in the
# contract's own naming and included even though no real leg has shown it AT
# this exact shape yet — matching costs nothing when it never fires) and
# GATES every name in this set on `ln_disabled` (pass 2 gated NONE of them,
# which was harmless only because no A/D2 leg happens to produce a row at
# this shape — pass 3 makes the gate explicit rather than accidental).
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
# ops for the `OPTIMIZER` bucket (module doc). Never used to derive an
# activation-tensor shape.
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
        `rows*seq` — module doc, "split by kernel NAME-CLASS". Distinct
        from `attn_softmax_rows` (`rows*heads*seq`) by construction
        (`heads>1` for both towers), so no collision with the attention
        reduction rule."""
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
    `hidden_size=768`, `hidden_act="gelu"`). NOT cross-checked against a
    real KERNEL export — no HTSAT `by_kernel_and_grid` leg is attributed by
    this module yet (pass 3 stays scoped to the eight real CLIP legs).
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


def _is_one_d_launch_grid(grid: list[int]) -> bool:
    """`True` iff `grid` is a genuine 1-D launch (`grid=[ceil(N/b),1,1]`) —
    module doc, "`OPTIMIZER`'s own parameter-scale rule ... is now gated on
    1-D launch geometry"."""
    return grid[1] == 1 and grid[2] == 1


def _is_plausible_gemm_tile_grid(grid: list[int]) -> bool:
    """`True` iff every one of `grid`'s three dimensions is `> 1` — module
    doc, "Anonymous kernels ... classified ONLY by grid-family rules": a
    genuine M-tile x N-tile x batch/split-K GEMM launch never degenerates
    to size `1` in any dimension; a `1` in some position is a 1-D-ish
    bookkeeping/reduction launch instead. Evidenced BOTH ways on
    `clip-text-D1`/`D2` (module doc)."""
    return grid[0] > 1 and grid[1] > 1 and grid[2] > 1


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
    name = entry["kernel"]
    grid = entry["grid"]

    # --- `cast`-named kernels: shape-gated to C-ATTN first (pass 3, finding
    #     1), the generic CAST bucket otherwise. A cast row at any OTHER
    #     shape is unambiguous — no other computation in this model both
    #     casts AND sits at the attention tensor's own element count.
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
    #     `ln_disabled` (pass 3, finding 2 — module doc). Falls through
    #     (does not return None) when the shape or the gate does not match.
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
    #     gated on grid POSITION 2 (pass 3, finding 1 — module doc). Catches
    #     any GEMM-family kernel AND anonymous kernels alike.
    if chain is None and grid[2] == sig.attn_batch_count():
        chain = chain_attn(tower)

    # --- GEMM-family by name (not carrying the attention batch count).
    if chain is None and GEMM_NAME_RE.search(name):
        chain = CHAIN_BASE_GEMM
    if chain is None and name == "splitKreduce_kernel":
        chain = CHAIN_BASE_GEMM
    # --- Anonymous-kernel-only grid-family fallback (pass 3, finding 3 —
    #     module doc): a name that does NOT look GEMM-shaped by name can
    #     still be BASE-GEMM if its own launch grid is a genuine 3-D tile
    #     grid (every dimension `> 1`).
    if chain is None and not GEMM_NAME_RE.search(name) and _is_plausible_gemm_tile_grid(grid):
        chain = CHAIN_BASE_GEMM

    # --- Attention tensor's own elementwise ops (mask add, scale, cast,
    #     backward exp/square, ...) — module doc, "the attention SCORE/PROB
    #     tensor's own elementwise ops".
    if chain is None and _in_element_range(entry, sig.attn_shape_elements()):
        chain = chain_attn(tower)

    # --- Parameter-scale bookkeeping (module doc, "OPTIMIZER extended"),
    #     GATED on 1-D launch geometry (pass 3, finding 3).
    if chain is None and _is_one_d_launch_grid(grid):
        for elements in sig.param_scale_elements():
            if _in_element_range(entry, elements):
                chain = CHAIN_OPTIMIZER
                break

    # --- The three activation-tier buckets, split by kernel NAME-CLASS
    #     (pass 3, finding 2 — module doc): permute/reshape copies get their
    #     OWN bucket (not tier-suffixed); bias/residual adds get a
    #     tier-suffixed bucket; every other name at a tier shape is the
    #     honestly-labeled `ELEMENTWISE-OTHER-<tier>` catch-all.
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
        # `classify_kernel` runs REGARDLESS of `is_known_kernel_name` (pass
        # 3, finding 3): an anonymous/unknown-name row is classified by
        # `classify_kernel`'s own grid-family rules FIRST, exactly like a
        # named row — "unknown by name" and "unattributed by rule" are
        # different questions. Only a row `classify_kernel` still could not
        # place (chain is `None`, i.e. UNATTRIBUTED) is THEN checked against
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
        return False, "leg is INVALID (see this leg's own `reasons`)"
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
            share = (wall_delta / fused_wall) if (wall_delta is not None and fused_wall) else None
            gains.append(
                {
                    "chain": chain_name,
                    "tower": tower,
                    "eager_leg_id": eager["leg_id"],
                    "fused_leg_id": fused["leg_id"],
                    "busy_delta_us_per_step": busy_delta,
                    "wall_delta_s_per_step": wall_delta,
                    "share_of_baseline_wall": share,
                    "direction": _realized_gain_direction(chain_name, eager["leg_id"], fused["leg_id"], busy_delta),
                }
            )
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
    a_legs = [
        leg
        for leg in legs
        if leg.get("tower") == tower and leg.get("_role") == "A" and leg.get("verdict") == VERDICT_VALID
    ]
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
    kernels_disabled_str = [str(k) for k in kernels_disabled]
    row["_role"] = leg_role(kernels_disabled_str, str(tower))
    # `ln_disabled`: this leg's OWN recorded `kernels_disabled` names
    # `layer_norm_fused` — for a VALID leg this equals the merge's
    # validated `kernels_disabled_expected` (contract §D6 finding 2: the
    # merge refuses a D leg whose requested set differs from the declared
    # one), so reading it here (rather than importing the merge's own
    # report) is equivalent for every leg this module ever calls
    # `decision_grade=True` on — module doc, "split by kernel NAME-CLASS".
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
        chains, unknown, classify_reasons = attribute_census(census, sig, tower, ln_disabled=ln_disabled)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        row["decision_grade_reason"] = "leg is INVALID (see this leg's own `reasons`)"
        return row
    reasons.extend(classify_reasons)
    row["unknown_kernels"] = unknown
    row["chains"] = {name: result.as_dict() for name, result in chains.items()}
    row["outside_signature_plausibly_attention"] = outside_signature_plausibly_attention(chains)
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
    return sorted(
        d
        for d in legs_dir.iterdir()
        if d.is_dir() and d.name != "p2-bf16" and (d / "manifest.json").is_file()
    )


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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_attribute.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", required=True, help="a profile_421_legs.sh $OUT_DIR/legs to attribute")
    ap.add_argument(
        "--merge-json",
        required=True,
        help="the output of profile_421_merge.py over the SAME legs-dir; required (module doc)",
    )
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
