#!/usr/bin/env python3
"""Issue #421 tower-profile chain ATTRIBUTION (post-export unit; CONTRACT
`scratchpad/contract-421-profile.md` v2.5 `### Attribution` / `§D3`).

`profile_421_merge.py` decides whether a leg is a DATUM at all (the
positive-proof equation, checkpoint identity, the wall/front/busy
decomposition). This module answers a DIFFERENT question over the SAME
`$OUT_DIR/legs` tree: of one leg's own `gpu_kernel_us_per_step`, how much
belongs to which named chain (`C-LORA`, `C-LN`, `C-GELU`, `C-ATTN-<tower>`),
and how much is `UNATTRIBUTED`?

## Why this is a SEPARATE post-export step (contract §D3)

The chain SIGNATURES' shapes are pre-declared in the contract (`rows = 3B =
24`, `S = 77` text / `50` vision, HTSAT per-window `[B*nW, heads, ws^2,
ws^2]`) — those are fixed before any measurement. What is NOT pre-declared,
and cannot be, is which literal GPU kernel NAME realizes which chain: nsys
reports candle's backend kernel names (`badd_f32`, `ampere_sgemm_128x128_nt`,
`layer_norm_fwd_f32_biased`, ...), not `jammi_kernels::admission`'s op keys
(`lora_linear_fused`, `layer_norm_fused`, `gelu_erf_fused`). That mapping is
read off the FIRST real export, with tests, before any attribution is read —
this module and its committed fixture
(`fixtures/profile_421_clip_text_a1/kernels.json`, cut from the real
`clip-text-A1` leg pulled at `c1b0b0ba` on pod `p421` run 2) are that
reading.

## What the real `clip-text-A1` export showed, and the decisions it grounds

The full `by_kernel_name` list (44 distinct kernel names, F32 CLIP-text,
`--arm fused`, wire-default shape) contains **no kernel literally named**
`lora_linear_fused`, `layer_norm_fused`, or `quick_gelu` — those are
`jammi_kernels::admission` op-key names, not nsys kernel names. The realized
GPU work for each admitted op key is spread across candle's generic backend
kernels (`ampere_sgemm_*`, `badd_f32`, `bmul_f32`, ...), most of which are
ALSO used by unrelated math elsewhere in the same forward/backward pass. Six
kernel names ARE unambiguous by construction, and this module's mapping
rests entirely on them plus one relational (not by-name) rule for attention's
batched matmuls:

1. **`layer_norm_fwd_f32_biased` / `layer_norm_bwd_dx_f32` → `C-LN`, by NAME
   alone, any grid.** These are the house LayerNorm CustomOp's own forward
   and backward kernels (`crates/jammi-encoders/src/layer_norm.rs`) — no
   other computation in this model is named `layer_norm_*`. Evidence: on
   `clip-text-A1` these launch 25/step and 24/step respectively, matching
   the leg's own witnessed `fusible_site_census.layer_norms = 25` (2 per
   block x 12 layers + `ln_final`) within one dispatch (`ln_final`'s
   backward composes slightly differently — irrelevant to the NAME match).

2. **`usigmoid_f32` → `C-GELU`, by NAME alone, any grid.** `quick_gelu(x) =
   x * sigmoid(1.702 * x)` (`crates/jammi-encoders/src/activations.rs`,
   `crate::clip_text`'s module doc) is the ONLY place this model computes a
   sigmoid; `usigmoid_f32` is therefore exclusive to the GELU chain by
   construction, independent of shape.

3. **`affine_f32` / `bmul_f32` at the declared GELU shape
   `[rows*S, 4*width]` → `C-GELU`.** `quick_gelu`'s remaining two elementwise
   ops are the `*1.702` pre-scale (`affine_f32`: `y = a*x + b`) and the final
   `x * sigmoid(...)` product (`bmul_f32`). Both names are used ELSEWHERE at
   OTHER shapes (LoRA-dropout scaling, generic backward products), so they
   are matched ONLY when a `(kernel, grid, block)` row's total thread count
   falls in `[elements, elements + block_size)` for `elements = rows * S *
   4 * width` (the MLP hidden tensor's element count — `4*width` is
   OpenCLIP's fixed MLP ratio, `crate::clip_text`/`crate::open_clip_vision`
   module docs). Evidence on `clip-text-A1` (`rows=24, S=77, width=512` →
   `elements = 3,784,704`): `usigmoid_f32` at this exact shape launches
   12/step (`= layers`, one quick_gelu call per block); `affine_f32`
   launches 36/step and `bmul_f32` 60/step at this shape (more than 12,
   consistent with backward passes reusing the same shape — NOT re-derived
   here, only the forward-pass semantic role is claimed).

   **Deliberately EXCLUDED at this same shape: `badd_f32`.** A bias-add at
   `[rows*S, 4*width]` is `c_fc`'s own Linear bias (or its LoRA-composed
   epilogue), not the activation function — attributing it to `C-GELU`
   would silently fold a Linear layer's cost into the activation chain. It
   is left `UNATTRIBUTED` (see "What stays UNATTRIBUTED" below): this
   module has no shape-only way to further split "the base Linear's bias"
   from "the LoRA epilogue's add" from a single leg's census.

4. **`fast_max_f32` / `fast_sum_f32` at the declared per-row softmax
   reduction count `rows*heads*S` → `C-ATTN-<tower>`.** Softmax over the
   last `S` axis of the `[rows, heads, S, S]` scores tensor needs a
   per-row max (numerical stability) and a per-row sum (normalization);
   the row count is `rows*heads*S`. Matched on `grid[0] == rows*heads*S`
   AND `grid[1] == grid[2] == 1` (one thread block per reduced row — the
   block's own size is irrelevant to this count). Evidence on
   `clip-text-A1` (`rows=24, heads=8, S=77` → `14,784` rows): EXACTLY one
   `fast_max_f32` row sits at `grid=[14784,1,1]` (12 launches/step — one
   softmax max per block's forward); a `fast_sum_f32` row sits at the same
   grid (36 launches/step — forward sum plus backward reduction passes).
   OTHER `fast_max_f32`/`fast_sum_f32` rows at different grids are left
   `UNATTRIBUTED` (their row count does not match any declared chain, and
   guessing a role for them — L2-normalize, gradient-norm clipping,
   EOT-pool reductions — is exactly the transcription this module refuses
   to do without independent evidence).

5. **`ampere_sgemm_*` kernels whose grid carries `rows*heads` as one of its
   three dimensions → `C-ATTN-<tower>`.** The batched per-(row,head)
   matmuls attention needs (`Q @ K^T`, `probs @ V`, and their backward
   transposes) batch by `rows*heads`; the base/LoRA Linear projections'
   matmuls do not (they are 2-D, `[rows*S, width] x [width, width']`).
   Evidence on `clip-text-A1` (`rows*heads = 192`): exactly the three
   `ampere_sgemm_128x128_{nt,tn,nn}` rows carry `grid=[1,1,192]` (24
   launches/step each — forward + the two backward transposes), and no
   OTHER `ampere_sgemm_*` row anywhere in the export carries `192` in any
   grid dimension. This is a RELATIONAL rule (a grid-dimension match, not a
   kernel name), stated once here rather than enumerating three kernel
   names, so it generalizes to whatever specific `ampere_sgemm_*` tile
   variant a different shape/dtype/arch selects.

## What stays UNATTRIBUTED (by design, not by omission)

`C-LORA` has **no literal GPU kernel name** in this export at all — the
`lora_linear_fused` admission counter is realized through the SAME
`ampere_sgemm_*`/`badd_f32` kernels the base Linear layers use, and a
single leg's census cannot tell "the base projection's sgemm" from "the
LoRA A/B sgemm" apart by name or grid alone (both are plain 2-D matmuls at
`[rows*S, width] x [width, rank-or-width]`, and a rank-8 matmul's grid tile
selection is a cuBLAS heuristic, not a stable signature). The contract's
OWN method for `C-LORA` is therefore never by-name: it is the `D2` delta
(`A1.gpu_busy_us_per_step − D2.gpu_busy_us_per_step`, `D2` being the twin
leg with ONLY `lora_linear_fused` forced eager). `attribute_report` computes
this whenever both a tower's A-leg (`A1`, F32 — `D2` is F32-only per the
contract's legs table) and its `D2` leg are present and individually VALID
under the SAME checks `profile_421_merge.py` applies (checkpoint identity,
matching `fusible_site_census`); otherwise `C-LORA` is reported
`status: "requires_d2_delta"` with `null` numbers — a declared-absent chain,
never a guessed one. The `D1 − D2` cross-check (contract: "isolates C-LN on
CLIP, C-LN + C-GELU on HTSAT") is likewise computed opportunistically as an
ADVISORY field (`d1_minus_d2_busy_us`) when both legs are present, to
sanity-check this module's by-name `C-LN` (+`C-GELU` on HTSAT) sum against
the contract's own independently-declared method — never a gate.

`badd_f32`'s many OTHER grids (bias adds / residual adds outside the GELU
shape, at `[rows*S, width]` and elsewhere), `bmul_f32`/`affine_f32`/
`fast_max_f32`/`fast_sum_f32` at grids OTHER than the two declared shapes
above, `dropout_fwd_f32`, `splitKreduce_kernel`, `adamw_*`, cast/copy
kernels, and every generic `ampere_sgemm_*` row that does not carry
`rows*heads` in its grid, all fall into `UNATTRIBUTED`.

## The INVALID-by-unknown-kernel gate

`KNOWN_KERNEL_NAMES` is the exact 44-name vocabulary observed in the real
`clip-text-A1` (F32) export — this module's ONLY validated ground truth.
Any `by_kernel_and_grid` row whose kernel name is NOT in that set, and whose
own `share` of `gpu_kernel_us_per_step` exceeds `UNKNOWN_KERNEL_SHARE_LIMIT`
(1%; the validity gate's own `UNATTRIBUTED <= 5%` bound is meaningless if a
one-name gap could silently eat several points of it), makes the WHOLE LEG
`INVALID` — never silently folded into `UNATTRIBUTED`. This is intentional
and EXPECTED to fire for every leg this vocabulary has not yet been
validated against: CLIP-vision (a different conv/patch-embed kernel set),
HTSAT (Swin window kernels), any BF16 leg (a disjoint half-precision kernel
vocabulary), and D1/D2 (same tower, same dtype, so likely a SUBSET of the A1
vocabulary — expected to pass once pulled). Extending `KNOWN_KERNEL_NAMES`
from that leg's own real export (with its own doc-comment evidence, same as
this module's) is the correct fix when that happens — never widening the
threshold or guessing a role for the new name.

## Declared per-tower architecture constants

`rows` (`= 3 * batch`, the triplet-encoding rule pinned contract-wide) and
`seq` (`max_seq_length` for text) are WITNESSED off the leg's own
`manifest.json`. `layers` is WITNESSED off the leg's own
`fusible_site_census.lora_sites_wrapped` (`= 4 * layers` for every OpenCLIP
tower — `in_proj`/`out_proj`/`c_fc`/`c_proj`, `crate::open_clip_block`).
`width`/`heads`/`mlp_ratio` have NO witness anywhere in a `finetune-run`
report (the checkpoint's `open_clip_config.json` is not read back out onto
the wire) and are DECLARED CONSTANTS for the stock, pinned checkpoint
(`laion2b_s34b_b79k` ViT-B-32 — `crate::clip_text::ClipTextConfig`/
`crate::open_clip_vision::OpenClipVisionConfig`'s own doc-comments and
`config_from_open_clip_json`/`config_heads_default_from_width` unit tests
name these exact numbers): text `width=512, heads=8`; vision `width=768,
heads=12`; both `mlp_ratio=4`. Vision's `seq=50` (49 patches at
`224/32=7` per side, `7*7=49`, `+1` CLS token) is likewise a DECLARED
constant — no `clip-vision` leg has been pulled yet, so this has not been
cross-checked against a real export the way `clip-text`'s constants were
(§D3's own convention: "shapes pre-declared ... mapping post-export"); when
a `clip-vision` leg lands, `KNOWN_KERNEL_NAMES` and this constant should be
re-verified the same way `clip-text`'s were (§ above).

HTSAT's per-window attention signature (`[B*nW, heads, ws^2, ws^2]`) is
DECLARED here from the public `laion/clap-htsat-fused` HTSAT-tiny config
(`depths=[2,2,6,2]`, `num_heads=[4,8,16,32]`, `window_size=8`,
`crates/jammi-encoders/src/htsat_audio.rs`'s own module doc) — UNVERIFIED
against any real export (no HTSAT leg has been pulled). `htsat_signatures`
below derives the four stages' window/head geometry from these declared
constants; its kernel-name mapping is not exercised by this module's tests
(no committed HTSAT fixture exists yet) beyond the generic reduction/matmul
rules above, which are tower-agnostic.

Run:
  python3 ci/scripts/perf/profile_421_attribute.py --legs-dir .profile-421-legs/<ts>/legs \\
      --out .profile-421-legs/<ts>/attribution.json
Hermetic self-tests: `python3 ci/scripts/perf/test_profile_421_attribute.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

VERDICT_VALID = "VALID"
VERDICT_INVALID = "INVALID"

CHAIN_LORA = "C-LORA"
CHAIN_LN = "C-LN"
CHAIN_GELU = "C-GELU"
CHAIN_UNATTRIBUTED = "UNATTRIBUTED"


def chain_attn(tower: str) -> str:
    """`C-ATTN-<tower>`, the one chain name that IS tower-suffixed (the
    contract names it `C-ATTN-<tower>` explicitly, unlike `C-LORA`/`C-LN`/
    `C-GELU`, because HTSAT's attention tier is out of the fixed-head-dim
    port scope but is still a NAMED, MEASURED chain per tower)."""
    return f"C-ATTN-{tower}"


# The exact 44 kernel names observed in the real `clip-text-A1` (F32) export
# pulled from pod `p421` run 2 at `c1b0b0ba`
# (`scratchpad/pod421-run2/legs/clip-text-A1/census.json`). See this
# module's own doc for why a name OUTSIDE this set is a leg-INVALIDATING
# finding, never a silent UNATTRIBUTED fold.
KNOWN_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "badd_f32",
        "ampere_sgemm_32x128_tn",
        "ampere_sgemm_128x64_nn",
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
        "ampere_sgemm_32x128_nn",
        "copy2d_f32",
        "ampere_sgemm_128x128_nt",
        "ampere_sgemm_64x32_sliced1x4_nn",
        "scaled_cast_add_f32_f32",
        "ampere_sgemm_128x128_tn",
        "splitKreduce_kernel",
        "ampere_sgemm_128x128_nn",
        "cast_f32_f32",
        "layer_norm_bwd_dx_f32",
        "eq_f32",
        "usqr_f32",
        "adamw_moment_update_f32",
        "fast_max_f32",
        "ampere_sgemm_128x64_tn",
        "layer_norm_fwd_f32_biased",
        "usigmoid_f32",
        "adamw_theta_update_f32",
        "const_set_f32",
        "cast_u8_f32",
        "uexp_f32",
        "bmaximum_f32",
        "bminimum_f32",
        "usqrt_f32",
        "is_u32_f32",
        "ampere_sgemm_32x32_sliced1x4_tn",
        "fast_argmax_u32",
        "ucopy_u32",
        "gather_u32_f32",
        "sa_u32_f32",
    }
)

# A leg-invalidating "we cannot classify this kernel" finding requires the
# unknown name to carry MORE than this share of `gpu_kernel_us_per_step` —
# the contract's own validity gate bounds UNATTRIBUTED at 5%; a name outside
# `KNOWN_KERNEL_NAMES` above 1% of busy is material enough that folding it
# into UNATTRIBUTED (rather than refusing) could silently mask a real
# attribution gap.
UNKNOWN_KERNEL_SHARE_LIMIT = 0.01

# Exclusive-by-name kernels: matched at ANY grid, independent of shape.
LN_KERNEL_NAMES: frozenset[str] = frozenset({"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"})
GELU_EXCLUSIVE_KERNEL_NAMES: frozenset[str] = frozenset({"usigmoid_f32"})

# Shape-gated kernels: matched only at the declared GELU element-count shape.
GELU_SHAPE_KERNEL_NAMES: frozenset[str] = frozenset({"affine_f32", "bmul_f32"})

# Shape-gated (row-count) kernels: matched only at the declared attention
# softmax reduction row count.
ATTN_REDUCTION_KERNEL_NAMES: frozenset[str] = frozenset({"fast_max_f32", "fast_sum_f32"})

ATTN_SGEMM_PREFIX = "ampere_sgemm"

# Declared architecture constants for the two OpenCLIP towers (see this
# module's doc, "Declared per-tower architecture constants"). `sites_per_
# layer` is witnessed elsewhere (`fusible_site_census.lora_sites_wrapped`)
# but stated here as the divisor `layers` is derived by.
TOWER_ARCH: dict[str, dict[str, object]] = {
    "clip-text": {"sites_per_layer": 4, "width": 512, "heads": 8, "mlp_ratio": 4, "declared_seq": None},
    "clip-vision": {"sites_per_layer": 4, "width": 768, "heads": 12, "mlp_ratio": 4, "declared_seq": 50},
}

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
    """The four Swin stages' declared window/head geometry for HTSAT, from
    the public `laion/clap-htsat-fused` HTSAT-tiny config (module doc). NOT
    cross-checked against a real export — no HTSAT leg has been pulled yet.
    Returned for documentation/forward-compatibility; no HTSAT census
    currently feeds through the by-name mapping below (the generic
    reduction/matmul rules are tower-agnostic and would apply once a real
    HTSAT `by_kernel_and_grid` export lands and this list is verified the
    same way `clip-text`'s constants were)."""
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


def classify_kernel(entry: dict, sig: TowerSignatures, tower: str) -> str | None:
    """`chain name` this `(kernel, grid, block)` row belongs to, or `None`
    (UNATTRIBUTED — including "unknown kernel name", handled separately by
    the caller so it can also feed the INVALID gate). See module doc for
    the evidence behind every branch."""
    name = entry["kernel"]
    if name in LN_KERNEL_NAMES:
        return CHAIN_LN
    if name in GELU_EXCLUSIVE_KERNEL_NAMES:
        return CHAIN_GELU
    if name in GELU_SHAPE_KERNEL_NAMES:
        elements = sig.gelu_shape_elements()
        total = _entry_total_threads(entry)
        block_size = _entry_block_size(entry)
        if elements <= total < elements + block_size:
            return CHAIN_GELU
        return None
    if name in ATTN_REDUCTION_KERNEL_NAMES:
        grid = entry["grid"]
        if grid[1] == 1 and grid[2] == 1 and grid[0] == sig.attn_softmax_rows():
            return chain_attn(tower)
        return None
    if name.startswith(ATTN_SGEMM_PREFIX):
        if sig.attn_batch_count() in entry["grid"]:
            return chain_attn(tower)
        return None
    return None


@dataclass
class ChainResult:
    status: str  # "measured" | "absent" | "requires_d2_delta"
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
    always carries an entry for `C-LN`, `C-GELU`, `C-ATTN-<tower>` and
    `UNATTRIBUTED` (per-test invariant: "every declared chain present or
    explicitly absent") — `status="absent"` means literally zero rows
    matched, distinct from `status="measured"` with `gpu_busy_us=0.0`.
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

    buckets: dict[str, float] = {
        CHAIN_LN: 0.0,
        CHAIN_GELU: 0.0,
        chain_attn(tower): 0.0,
        CHAIN_UNATTRIBUTED: 0.0,
    }
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
        if name not in KNOWN_KERNEL_NAMES:
            share = (us_per_step / total_busy_us) if total_busy_us else 0.0
            unknown_rows.append(
                {"kernel": name, "grid": entry.get("grid"), "share_gpu_busy": share}
            )
            if share > UNKNOWN_KERNEL_SHARE_LIMIT:
                invalid_reasons.append(
                    f"kernel {name!r} (grid={entry.get('grid')!r}) is not in KNOWN_KERNEL_NAMES "
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


def attach_lora_via_d2_delta(leg_rows: list[dict]) -> None:
    """Mutates `leg_rows` IN PLACE: for every tower with a VALID F32 `A`
    leg (`A1`; `D2` is F32-only per the contract's legs table) AND a VALID
    `D2` leg, sets that A-leg's `chains["C-LORA"]` to the measured
    `A.gpu_busy_us_per_step - D2.gpu_busy_us_per_step` delta (contract:
    "A1 - D2 is that one kernel"). Every OTHER leg (including `D2` itself,
    `A2`/BF16, and any tower missing its `D2` twin) gets the declared-absent
    `status: "requires_d2_delta"` placeholder — never a guess.

    Also attaches the ADVISORY `d1_minus_d2_busy_us` cross-check (contract:
    "D1 - D2 = C-LN on CLIP, C-LN + C-GELU on HTSAT") to the A-leg's row
    when both `D1` and `D2` are present and VALID for that tower.
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
    }
    reasons: list[str] = []
    try:
        manifest = _load_json(leg_dir / "manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        row["reasons"] = [f"{leg_id}: manifest.json could not be read: {exc}"]
        return row

    tower = manifest.get("tower")
    dtype = manifest.get("dtype")
    row["tower"] = tower
    row["dtype"] = dtype
    kernels_disabled = manifest.get("kernels_disabled")
    if not isinstance(kernels_disabled, list):
        row["reasons"] = [f"{leg_id}: manifest.kernels_disabled is not a list"]
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
        return row

    if census.get("excluded_from_chain_attribution") is True:
        reasons.append(
            f"{leg_id}: census.excluded_from_chain_attribution is true — this leg's signatures "
            "are not fixed-shape and cannot be attributed by this module (kernel_census.py's own "
            "E1 exclusion)"
        )
        row["reasons"] = reasons
        return row

    if not isinstance(tower, str):
        reasons.append(f"{leg_id}: manifest.tower is not a string ({tower!r})")
        row["reasons"] = reasons
        return row

    try:
        sig = derive_signatures(tower, manifest)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        return row
    row["signatures"] = sig.as_dict()

    try:
        chains, unknown, classify_reasons = attribute_census(census, sig, tower)
    except SignatureError as exc:
        reasons.append(f"{leg_id}: {exc}")
        row["reasons"] = reasons
        return row
    reasons.extend(classify_reasons)
    row["unknown_kernels"] = unknown
    row["chains"] = {name: result.as_dict() for name, result in chains.items()}
    row["_gpu_busy_us_per_step"] = census.get("gpu_kernel_us_per_step")
    row["_wall_s_per_step"] = census.get("wall_s_per_step")
    row["gpu_busy_us_per_step"] = census.get("gpu_kernel_us_per_step")
    row["wall_s_per_step"] = census.get("wall_s_per_step")

    row["reasons"] = reasons
    row["verdict"] = VERDICT_INVALID if reasons else VERDICT_VALID
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
        "summary": {
            "legs_total": len(legs),
            "legs_valid": sum(1 for row in legs if row["verdict"] == VERDICT_VALID),
            "legs_invalid": sum(1 for row in legs if row["verdict"] == VERDICT_INVALID),
        },
    }
    return report


def format_table(report: dict) -> str:
    """A human-readable per-(tower, dtype, leg, chain) share table — this is
    what a caller pastes into a PR/hand-off, never re-derived from the JSON
    by eye."""
    lines: list[str] = []
    header = (
        f"{'leg_id':<16} {'tower':<12} {'dtype':<5} {'verdict':<8} "
        f"{'chain':<16} {'status':<20} {'share_busy':>10} {'share_wall':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report.get("legs", []):
        chains = row.get("chains") or {}
        if not chains:
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {'(none)':<16} {'':<20} {'':>10} {'':>10}"
            )
            continue
        for chain_name in sorted(chains):
            entry = chains[chain_name]
            share_busy = entry.get("share_gpu_busy")
            share_wall = entry.get("share_wall")
            lines.append(
                f"{row['leg_id']:<16} {str(row.get('tower')):<12} {str(row.get('dtype')):<5} "
                f"{row['verdict']:<8} {chain_name:<16} {entry.get('status', ''):<20} "
                f"{(f'{share_busy:.4f}' if isinstance(share_busy, (int, float)) else ''):>10} "
                f"{(f'{share_wall:.4f}' if isinstance(share_wall, (int, float)) else ''):>10}"
            )
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
    if not args.no_table:
        print(format_table(report), file=sys.stderr)
    summary = report["summary"]
    print(
        f"profile_421_attribute: {summary['legs_valid']}/{summary['legs_total']} legs VALID",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
