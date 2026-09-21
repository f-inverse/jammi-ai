#!/usr/bin/env python3
"""Merge + table stage for `ci/scripts/perf/finetune_ab.sh`'s A/B sweep.

An importable module rather than an inline heredoc so `test_ab_merge.py` can
drive the real entry point (`main`, exactly what `finetune_ab.sh` invokes)
against fixture directories shaped like `run_leg`'s own
`.exit`/`.json`/`.stderr` output — `AB_DRY_RUN=1` alone only exercises the
DRY_RUN arm, never a real report shape.

Never imported by any Cargo crate, never a jammi-bench dependency — a
CI-adjacent script the sweep alone runs.

## Determinant table

`leg_premise_violations` certifies that a config's jammi and torch legs ran
under the SAME premise before their ratio/loss numbers are treated as
comparable. Every field either producer's finetune-step report emits,
classified — mirrors `grad_oracle.rs`'s own determinant table
(`crates/jammi-bench/src/grad_oracle.rs`'s module doc) for the OTHER
jammi-vs-torch comparator this repo carries:

| field | class | jammi emit site | torch emit site |
|---|---|---|---|
| `seed` | identity | `report.rs:FinetuneStepTier::seed` field; `seed: params.seed,` (`finetune_step.rs:997`) | `"seed": args.seed,` (`torch_finetune_step.py:1260`) |
| `batch` | identity | `batch: params.batch,` (`finetune_step.rs:1002`) | `"batch": args.batch,` (`torch_finetune_step.py:1278`) |
| `seq` | identity | `seq: params.seq,` (`finetune_step.rs:1003`) | `"seq": args.seq,` (`torch_finetune_step.py:1279`) |
| `lora_rank` | identity | `lora_rank: params.lora_rank,` (`finetune_step.rs:1004`) | `"lora_rank": args.lora_rank,` (`torch_finetune_step.py:1280`) |
| `lora_alpha` | identity | `lora_alpha: params.lora_alpha,` (`finetune_step.rs:1005`) | `"lora_alpha": args.lora_alpha,` (`torch_finetune_step.py:1253`) |
| `lora_dropout` | identity | `lora_dropout: params.lora_dropout` (`finetune_step.rs:1006`) | `"lora_dropout": args.lora_dropout,` (`torch_finetune_step.py:1281`) |
| `margin` | identity, but jammi HARDCODES `0.3` (no `--margin` CLI flag — the call site's own literal, `let loss = triplet_loss(&a, &p, &n, 0.3)?;` (`finetune_step.rs:613`)) | `margin: 0.3,` (`finetune_step.rs:1013`) | `"margin": args.margin,` (`torch_finetune_step.py:1262`) — `--margin` default `0.3` |
| `target_modules` | identity | `target_modules: params.target_modules.clone(),` (`finetune_step.rs:1014`) | `"target_modules": [` (`torch_finetune_step.py:1284`) |
| `batched_forward` | identity | `batched_forward: params.batched_forward,` (`finetune_step.rs:1015`) | `"batched_forward": args.batched_forward,` (`torch_finetune_step.py:1287`) |
| `backbone_dtype` | identity | `backbone_dtype: format!("{:?}", params.backbone_dtype)` (`finetune_step.rs:998`) | `"backbone_dtype": args.dtype,` (`torch_finetune_step.py:1269`) |
| `steps_measured` | identity — two legs measured at a DIFFERENT step count (e.g. a mismatched `--steps`/`--warmup` override) would otherwise merge to a "clean" ratio | `steps_measured: times.len(),` (`finetune_step.rs:1033`) | `"steps_measured": len(times),` (`torch_finetune_step.py:1320`) |
| `checkpoint_config_sha256` | identity — the same base-checkpoint CONTENT identity `grad_oracle.rs`'s tier carries | `let (checkpoint_config_sha256, _config_len) =` (`finetune_step.rs:771`), via the SHARED streaming `pub(crate) fn sha256_and_len` (`finetune_step.rs:1158`) | `checkpoint_identity_fields = checkpoint_identity(args.model_dir)` (`torch_finetune_step.py:1130`) |
| `checkpoint_weights_sha256` | identity | `let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =` (`finetune_step.rs:773`) | `"checkpoint_weights_sha256": weights_sha256,` (`torch_finetune_step.py:685`) |
| `checkpoint_weights_size_bytes` | identity | `checkpoint_weights_size_bytes) =` (`finetune_step.rs:773`) — same call as the row above, its second return value | `"checkpoint_weights_size_bytes": weights_len,` (`torch_finetune_step.py:686`) |
| `row_lengths` | identity — the per-row token lengths the padded fixture fed the encoder, requested or the dense-leg default `[seq; batch]`; NO canonicalizer, per-row order is load-bearing (`[3, 6]` != `[6, 3]`), compared directly, never hashed | `row_lengths: params` (`finetune_step.rs:1027`), dense fallback `vec![params.seq; params.batch]` (`finetune_step.rs:1030`) | `"row_lengths": args.row_lengths` (`torch_finetune_step.py:1295`), dense fallback `else [args.seq] * args.batch,` (`torch_finetune_step.py:1297`) |
| `max_grad_norm` | identity — `null` (clip OFF) or the positive finite bound the PRODUCTION `clip_gradients` ran with; a clip-on leg and a clip-off leg compute a different step. `null` is a VALUE for this field (`identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`), never folded into MISSING | `max_grad_norm: params.max_grad_norm,` (`finetune_step.rs`'s tier literal) | `"max_grad_norm": args.max_grad_norm,` in the `finetune_step` block (`torch_finetune_step.py`) |
| `attention_arm` | identity — the attention REFERENCE CLASS the leg was ASKED to run, `"eager"` or `"fused"`; jammi's is the operator's `JAMMI_KERNELS_DISABLE` request (an attention base in `kernels_disabled_requested` ⇒ eager), NEVER the counters (a by-design domain decline is a measurement, not a premise) — see `identity_fields.FINETUNE_IDENTITY_FIELDS`'s own entry | `attention_arm: attention_arm(&kernels_disabled_requested).to_string()` (`finetune_step.rs`'s tier literal) | `"attention_arm": attention_arm_of(resolved_attn_implementation)` in the `finetune_step` block (`torch_finetune_step.py`) |
| `warmup` | identity — changes what `clip_invocations` counts (pre-step + warmup + measured) | `warmup: params.warmup,` (`finetune_step.rs`'s tier literal) | `"warmup": args.warmup,` in the `args` block (`torch_finetune_step.py`) — an `_TORCH_ARGS_LEVEL_FIELDS` member |
| `clip_invocations` | measurement — the COUNTED number of times the production clip ran this process (pre-step + warmup + measured, every `step_once`), the fact behind a clip-on row rather than a log line; recorded in `leg_provenance` per leg AND cross-checked against `max_grad_norm` by `clip_fact_violations` (clip requested ⇒ `> 0`; not requested ⇒ `== 0`) | `clip_invocations:` (`finetune_step.rs`'s tier literal, a `CLIP_INVOCATIONS` before/after delta) | `"clip_invocations": clip_counter["clip_invocations"]` (`torch_finetune_step.py`) |
| `attn_requested` / `attn_implementation` | provenance — the RAW torch attention string (`--attn` as requested, and what HF resolved it to); the CLASS it implies is compared via `attention_arm` above, the raw string itself is recorded in `leg_provenance`, never compared (see `grad_oracle.rs`'s own table for the fuller rationale) | n/a | `"attn_requested": args.attn,` (`torch_finetune_step.py:1258`) in the `args` block; `attn_implementation` is the sibling `"attn_implementation": resolved_attn_implementation` field further down in the `finetune_step` block |
| `kernels_disabled_requested` / `kernels_disabled_fired` | provenance — torch has no equivalent env var; recorded in `leg_provenance`, never compared | `let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();` (`finetune_step.rs:982`) | n/a |
| `ln`/`rope`/`softmax`/`geglu`/`gelu`/`lora_epilogue`/`lora_linear`/`attention_block` `_fused_dispatches`/`_eager_dispatches` (16 fields) | measurement — this IS the fused-dispatch proof `fused_proof`/`dispatch_pairs` gate on, and `leg_provenance` additionally records the raw counters per config. `gelu` (`OPTIONAL_NON_CASCADE_PAIRS`) is unconditionally SERIALIZED by `FinetuneStepTier`/`FinetuneRunTier` alike, but its `fused > 0` bar is architecture-conditional — the dense erf-GELU seam (`gelu_erf_fused`, BERT's/DistilBERT's FFN) is never reached by ModernBERT's GeGLU MLP (already covered by its own `geglu` pair), so a ModernBERT leg legitimately reads `gelu_{fused,eager} == (0, 0)` forever; admission is by THIS RUN'S OWN COUNTERS (tensor state), never a model-identity branch — see that set's own module-level doc | `finetune_step.rs`'s own `*_fused_dispatches`/`*_eager_dispatches` fields | n/a |
| `attention_block_flash_fused_dispatches` / `attention_block_flash_declined_dispatches` (`report.rs`'s `FinetuneStepTier` fields) | measurement — a CASCADE-shaped pair (`CASCADE_BASES`): no `_eager_dispatches` sibling, its fallback counter is named `_declined_dispatches` instead; absorbs `attention_block` (`ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`), which in turn already absorbs `rope`/`softmax` — one chain, not a second mechanism. Every current `finetune-step`/`finetune-run` leg carries both keys. A fixture predating the multi-tensor AdamW counters (no `adamw_{fused,eager}_dispatches` keys, e.g. `fixtures/p6_fa2_dense_raw_runs/s128_flash_{on,off}_1.json`) fails `fused_proof` outright: `adamw` is a `REQUIRED_PAIRS` member and an absent required base fails the WHOLE leg. `CascadePairFixtureTests::test_real_flash_{on,off}_fixture_no_longer_keyerrors_but_predates_adamw` (`test_ab_merge.py`) pin both fixtures reading INVALID | `report.rs`'s `FinetuneStepTier::attention_block_flash_fused_dispatches`/`::attention_block_flash_declined_dispatches` fields | n/a |
| `flash_compiled` | provenance — recorded in `leg_provenance` as `jammi_flash_compiled`, never compared; distinguishes "this build cannot run flash at all" from "flash was compiled in but declined/disabled this run", and backs `fused_proof`'s own flash-disable-consistency check (see that function's doc) | `report.rs`'s `FinetuneStepTier::flash_compiled` field, same branch as above | n/a |
| `losses` / `loss_first` / `loss_last` | measurement — `loss_final_ratio` is printed for visibility, never gated (see that field's own note in `build_report`) | `finetune_step.rs`'s own fields | `torch_finetune_step.py`'s own fields |
| `s_per_step_p50` / `triplets_per_s` / VRAM fields | measurement — the actual perf numbers this sweep exists to produce | `finetune_step.rs`'s own fields | `torch_finetune_step.py`'s own fields |
| `model_dir` | provenance (a path string, not compared — superseded by the checksum fields above) | `FinetuneStepParams::model_dir` (not itself emitted on the tier) | `torch_finetune_step.py`'s own `args["model_dir"]` |
| `device` / `device_name` | provenance | `finetune_step.rs`'s own fields | `torch_finetune_step.py`'s own `provenance` block |

`identity_fields.FINETUNE_IDENTITY_FIELDS` (imported below, never redeclared
here) is the tuple that actually encodes the **identity** rows above — the
single source of truth `leg_identity_fields`/`leg_premise_violations`
iterate, and the SAME declaration `report.rs`'s and `test_ab_merge.py`'s
producer-emit pins read.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identity_fields import (  # noqa: E402
    FINETUNE_IDENTITY_FIELDS,
    FINETUNE_NULL_IS_A_VALUE_FIELDS,
    canonicalize_identity_field,
)

LEGS = ["jammi-eager", "jammi-fused", "torch-eager", "torch-sdpa"]

# ORDER-BALANCED BAR LEGS (finetune_ab.sh's own A,B,B,A protocol — see that
# script's header's "ORDER-BALANCED BAR LEGS" section): the two legs the
# throughput bar actually gates on, `jammi-fused` (A) and `torch-sdpa`
# (B), run TWICE per config in the fixed order A,B,B,A — the SAME
# drift-cancellation shape `gpu_inference_ab.py`'s own `LEG_ORDER`/
# `ADJACENT_PAIRS` document ("What actually cancels, and what does not").
# `jammi-fused`/`torch-sdpa` (already in `LEGS` above) ARE the first ("1")
# run of each; `BAR_SECOND_RUN_LEGS` names the SECOND ("2") run's own raw
# leg files — deliberately NOT folded into `LEGS` itself: every function
# keyed off `LEGS` (`leg_premise_violations`, `leg_provenance`,
# `fused_proof`'s own caller, the primary per-leg table rows) sees only the
# four primary legs, and a `raw_dir` carrying only those four still merges
# — `load_leg` reads `MISSING` for an absent
# `<slug>__jammi-fused-2`/`<slug>__torch-sdpa-2` pair, which
# `bar_pair_ratio`/`build_report` below treat as "second run not
# available", falling back to the single-pair ratio. `BAR_SECOND_RUN_LEGS`
# is keyed by the FIRST run's own leg name (the natural "which pair is this
# the repeat of" lookup both call sites below need).
BAR_SECOND_RUN_LEGS = {"jammi-fused": "jammi-fused-2", "torch-sdpa": "torch-sdpa-2"}

# The file `finetune_ab.sh` `touch`es under `raw_dir`, BEFORE any leg runs, on
# EVERY invocation (that script always runs the full A,B,B,A protocol —
# see its own header). The SAME filename, read here. Presence means "this
# raw_dir's operator promised all four bar legs" — under this marker:
#   * a genuinely MISSING second-run leg (the file never written at all)
#     is an INCOMPLETE SWEEP (INVALID, a named reason via
#     `two_run_missing_leg_reason`);
#   * an OK-outcome leg whose own report still carries a falsy/missing
#     `triplets_per_s` is ALSO refused — BOTH pair ratios must resolve,
#     never silently handed back from whichever ONE pair happened to
#     produce a usable number (see `build_report`'s own `elif bar_ratio is
#     None or (two_run_mode and ...)` branch);
# neither is silently degraded to the single-pair estimator the way an
# absent marker (a `raw_dir` hand-built without it) is. A DRY_RUN
# second-run leg (DRY_RUN is a deliberate, ANNOUNCED "nothing ran for real"
# mode `finetune_ab.sh`'s own `AB_DRY_RUN=1` writes uniformly across every
# leg, never an incompleteness signal — making it INVALID here would make
# every dry-run smoke-test read INVALID unconditionally) reads
# `N/A (dry-run)` instead — `any_dry_run` is
# checked FIRST in `build_report`'s own verdict chain, before either of
# the two checks above ever runs. Kept unprefixed (no leading `.`) so
# `ls`/a human browsing `raw_dir` sees it; `config_slugs()` below never
# matches it (it carries no `.exit` suffix and no `__` separator).
TWO_RUN_PROTOCOL_MARKER = "TWO_RUN_PROTOCOL_MARKER"


def two_run_protocol_active(raw_dir):
    """`True` iff `finetune_ab.sh`'s own `TWO_RUN_PROTOCOL_MARKER` file is
    present under `raw_dir` — see that constant's own doc. A pure
    filesystem check, read ONCE per `build_report` call (never per-config
    — the marker is a property of the WHOLE sweep/raw_dir, not of any one
    config within it).
    """
    return os.path.isfile(os.path.join(raw_dir, TWO_RUN_PROTOCOL_MARKER))

# --------------------------------------------------------------------------- #
# Generic leg-premise-refusal core — `leg_identity_fields`/
# `leg_premise_violations` below are the finetune-step-SPECIFIC callers
# (report shape, torch args-level field placement, `_MISSING`-folding
# doctrine); `generic_leg_identity_fields`/`generic_leg_premise_violations`
# are the SAME two-step shape (fold ABSENT-or-null into `_MISSING`, then
# compare after `canonicalize_identity_field`) factored out over an
# arbitrary `fields` tuple and two ALREADY-FLATTENED `{field: value}` dicts,
# so another producer (`encode_ab.sh`) reuses the identical premise-refusal
# logic against `identity_fields.ENCODE_IDENTITY_FIELDS` rather than
# hand-rolling a second, independently-drifting comparator.
# --------------------------------------------------------------------------- #
def generic_leg_identity_fields(block, fields, null_is_value_fields=frozenset()):
    """Read `fields` off `block` (a FLAT dict — the caller resolves WHERE
    each field actually lives on its own report shape before calling this;
    `encode_ab.sh`'s merge step reads directly off `report["tiers"]
    ["encode_step"]`, which already carries every `ENCODE_IDENTITY_FIELDS`
    entry at one level, so no per-field placement map is needed there).

    Returns `{field: value_or_MISSING}` — `_MISSING` (never `None`) marks a
    field genuinely ABSENT from `block` OR present with a JSON `null` value,
    UNLESS `field` is a `null_is_value_fields` member (mirrors
    `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`'s own doctrine: for
    those fields a present `null` IS the stated premise, not an inability to
    state one). No `ENCODE_IDENTITY_FIELDS` entry is a `null_is_value_fields`
    member today (every one is `Nullable::NonNull` on `EncodeStepTier`), so
    encode callers pass the default empty set.
    """
    fields_out = {}
    for field in fields:
        if field not in block:
            fields_out[field] = _MISSING
            continue
        value = block[field]
        if value is None and field not in null_is_value_fields:
            value = _MISSING
        fields_out[field] = value
    return fields_out


def generic_leg_premise_violations(fields, fields_a, fields_b, label_a="a", label_b="b"):
    """The SAME leg-premise-refusal shape `leg_premise_violations` applies to
    finetune-step's `FINETUNE_IDENTITY_FIELDS`, generalized over an
    arbitrary `fields` tuple and two `generic_leg_identity_fields`-shaped
    `{field: value_or_MISSING}` dicts: a field missing (or present-but-null,
    already folded to `_MISSING` by the caller) from EITHER side is a
    refusal (cannot verify the two legs share a premise); a field present
    on both but differing after `canonicalize_identity_field` (the SAME
    shared canonicalizer table `leg_premise_violations`/
    `compare_grad_oracle.py` both already use) is also a refusal. Returns a
    list of strings, empty when the two legs' premises agree on every
    named field.
    """
    violations = []
    for field in fields:
        va = fields_a.get(field, _MISSING)
        vb = fields_b.get(field, _MISSING)
        missing_sides = []
        if va is _MISSING:
            missing_sides.append(label_a)
        if vb is _MISSING:
            missing_sides.append(label_b)
        if missing_sides:
            violations.append(
                f"leg-identity field {field!r} missing from {missing_sides} leg's record -- cannot "
                "verify the two legs of this config ran under the same premise"
            )
            continue
        ca = canonicalize_identity_field(field, va)
        cb = canonicalize_identity_field(field, vb)
        if ca != cb:
            violations.append(f"leg-identity field {field!r} differs: {label_a}={ca!r} {label_b}={cb!r}")
    return violations


# The premise-identity check: identity is a checked record in the merged
# artifact, never assumed from `finetune_ab.sh`'s matched CLI flags across
# its `run_jammi_leg`/`run_torch_leg` call sites. Shares
# `identity_fields.canonicalize_identity_field` with
# `compare_grad_oracle.py`'s OWN identity check (one definition) rather
# than a second, independently-drifting copy of the SAME
# `backbone_dtype`/`target_modules` representational gaps.
#
# `seed`/`lora_alpha`/`margin` live in a DIFFERENT place on each producer
# (jammi's own `finetune_step.rs`'s `FinetuneStepTier` fields sit directly
# in the `finetune_step` block this module already reads via
# `finetune_block`; torch's sit one level UP, in `report["args"]` --
# `torch_finetune_step.py`'s own report literal never duplicates them into
# the `finetune_step` sub-block) -- `leg_identity_fields` below reads each
# from its OWN real location per leg (`_TORCH_ARGS_LEVEL_FIELDS`), never
# assumes a shared schema.
#
# `lora_alpha`/`margin`: torch emits both under `args`; jammi's
# `FinetuneStepTier` emits them on the tier. `margin` has no jammi CLI flag
# -- this tier hardcodes `0.3`, matching torch's own `--margin` default,
# see `FinetuneStepTier::margin`'s own field doc.
#
# `steps_measured`: two legs run at `--steps 20`/`--warmup 5` vs
# `--steps 5`/`--warmup 5` (e.g. `finetune_ab.sh` invoked with mismatched
# per-leg overrides, or a leg re-run by hand) would otherwise merge to a
# "clean" ratio and PASS verdict. It is a genuine per-run fact recorded on
# BOTH sides (`FinetuneStepTier::steps_measured`, `torch_finetune_step.py`'s
# own `"steps_measured": len(times)`), which `main()`'s `steps`/`warmup` CLI
# args cannot stand in for (those describe what THIS MERGE INVOCATION was
# told, not what either leg actually measured).
#
# `checkpoint_config_sha256`/`checkpoint_weights_sha256`/
# `checkpoint_weights_size_bytes`: the SAME base-checkpoint content-identity
# fields `grad_oracle.rs`'s determinant table covers, a checked record in
# place of an implicit "the operator passed the same --model-dir path"
# assumption.
#
# The field set itself lives ONLY in `identity_fields.FINETUNE_IDENTITY_FIELDS`
# (imported above); this module never redeclares it, so a field added to the
# shared tuple is refused here generically the moment either producer fails
# to emit it.

# torch keeps these fields one level UP from the `finetune_step` sub-block
# (`report["args"][field]`) rather than inside it — see
# `FINETUNE_IDENTITY_FIELDS`'s own doc. Every OTHER field (including the
# checkpoint-identity ones, which torch emits directly inside the
# `finetune_step` block, matching jammi's own placement) lives at the SAME
# level `finetune_block` already reads for both producers.
_TORCH_ARGS_LEVEL_FIELDS = frozenset({"seed", "lora_alpha", "margin", "warmup"})

# The DECLARED classification `fused_proof` checks a dispatch-counter pair
# against. A blanket "(fused, eager) == (0, 0) is always fine" rule would let
# a report where every real fused site read (0, 0) and only ONE unrelated
# pair read a positive fused count print `fused_proof YES`; silently
# skipping a base ENTIRELY ABSENT from the schema (a renamed / deleted /
# feature-gated-off field) would hide exactly the regression this proof
# exists to catch. The invariant: EVERY base that `dispatch_pairs` discovers
# in a real report is in EXACTLY ONE of the sets below (`ALL_BASES` is
# their union); a discovered base outside `ALL_BASES` is a schema-drift
# ERROR (`dispatch_pairs` raises, handled per-leg-loud/whole-merge-safe by
# `build_report`), never a silent exemption. ABSENCE from the report is a
# hard fail for every member of the "must be present" sets.
#
#   * REQUIRED_PAIRS — no fused block in this crate absorbs these; each
#     MUST be PRESENT and show its own `fused > 0` (and, like every pair,
#     `eager == 0`).
#       - `ln`: dispatches inside every layer's own norm call, never folded
#         into a whole-attention or whole-MLP kernel, and
#         `finetune_step.rs`'s own counter-delta test asserts its
#         (fused+eager) total is nonzero on every run. The fused LayerNorm
#         covers both the bias-free (ModernBERT) and the biased
#         (BERT-family) form through the SAME `layer_norm_fused` admit key,
#         so every architecture satisfies this pair.
#       - `geglu`: same reasoning as `ln` — `ModernBertMlp::forward`'s
#         training arm calls `geglu_apply_training` unconditionally for
#         every layer's MLP (see that function's own doc); its admission
#         domain (F32/BF16, contiguous, nonzero-even last dim) holds for
#         every real ModernBERT MLP shape, so nothing legitimately
#         absorbs or exempts it the way `attention_block` absorbs
#         `rope`/`softmax`. A deleted/feature-gated-off fused MLP reading
#         `geglu = (0, 0)` therefore fails the proof.
#       - `adamw`: `AdamW::step`'s per-`Var` dispatch to
#         `adamw_step_fused_t` (`report.rs`'s `adamw_fused_dispatches`
#         field doc) — same reasoning class as `ln`/`geglu`: its admission
#         domain is device/dtype/contiguity/shape agreement across
#         `theta`/`m`/`v`/`grad`, which holds unconditionally for every real
#         training run (all four are the SAME `Var`'s own state) — nothing
#         legitimately absorbs or exempts it, and no fused block in this
#         crate folds it into a wider kernel. Pinned against the committed
#         real artifact
#         `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4-raw-runs/
#         a100b/b8_s512_fused.r2.json.raw` (`TestRealAdamwArtifactFixtures`
#         in `test_ab_merge.py`).
#   * ABSORBABLE_BY_ATTENTION_BLOCK_FLASH — `attention_block` MUST be
#     PRESENT; may read `(0, 0)` IFF `attention_block_flash`'s OWN `fused`
#     count is `> 0` THIS run: when the FlashAttention-2 dense cascade fires
#     for a layer, that layer's `attention_block` `admit` call is never
#     reached at all (an early return — see `report.rs`'s
#     `attention_block_flash_fused_dispatches` field doc on that branch),
#     the exact same "one call site, mutually exclusive arms" shape
#     `rope`/`softmax`'s own absorption below documents one level down.
#     `by_base.get("attention_block_flash", (0, 0))[0]` defaults to `0` when
#     the key is entirely ABSENT from the report — so on such a report this
#     absorption condition is never satisfied and `attention_block` needs
#     its OWN `fused > 0`, exactly as a `REQUIRED_PAIRS` member would.
#       - A checkpoint whose `head_dim != 64` legitimately falls back to
#         eager here (`report.rs`'s `attention_block_eager_dispatches`
#         field doc) — that is ALREADY caught by rule 1 below (an
#         unaccounted-for fallback anywhere is a hard fail), so requiring
#         `fused > 0` (absent flash absorption) here for the cases rule 1
#         does not already reject adds detection without changing
#         behaviour on that documented domain-refusal case.
#   * ABSORBABLE_BY_ATTENTION_BLOCK — `rope`/`softmax` MUST be PRESENT; may
#     read `(0, 0)` IFF `attention_block`'s OWN `fused` count is `> 0`, OR
#     `attention_block_flash`'s OWN `fused` count is `> 0` (the SAME chain
#     extended one level, not a parallel mechanism), this run:
#     `ModernBertAttention::forward_training_attention`'s BLOCK-fused arm is
#     the whole RoPE+QKᵀ+mask+softmax+PV chain as one op and never calls
#     `rope_apply`/`softmax_apply_training` at all (see that method's own
#     doc), so their independent admission call sites are simply never
#     reached — and the FLASH arm is a further whole-attention alternative
#     to that SAME call site, so it never reaches them either. When NEITHER
#     whole-attention arm goes fused (the eager attention composition ran
#     instead), that composition DOES call
#     `rope_apply`/`softmax_apply_training` — each independently
#     admission-gated — so they must clear the same `fused > 0` bar a
#     required pair does.
#   * LORA_SITE_EXCLUSIVE_GROUP — `lora_epilogue`/`lora_linear` MUST both be
#     PRESENT, and are genuinely exclusive with EACH OTHER, not with a
#     third pair: every training-arm LoRA-adapted forward routes through
#     EXACTLY ONE of these two call sites
#     (`jammi_lora::lora_linear::lora_linear_fused_counters`'s own doc —
#     `lora_epilogue` reads `(0, 0)`, superseded by the fused whole-site
#     kernel `lora_linear` reports). So only the GROUP's sum needs a
#     `fused > 0` proof, never each member alone.
#   * CASCADE_BASES — see that set's own doc below. Its one member,
#     `attention_block_flash`, is deliberately NOT a member of any of the
#     "must be present" sets above: it is a genuinely OPTIONAL arm (nothing
#     requires a build to have compiled/used it) — its ONLY role in
#     `ALL_BASES` is to be a recognized (not schema-drift) base when
#     `dispatch_pairs` discovers it in a report, so its own
#     `fused`/`declined` counts are available for rule 1 (the
#     declined-count hard-fail-unless-requested check) and for
#     `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`/`ABSORBABLE_BY_ATTENTION_BLOCK`'s
#     absorption conditions above.
#   * OPTIONAL_NON_CASCADE_PAIRS — see that set's own doc below. Its one
#     member, `gelu`, is an ORDINARY fused/eager pair (a plain
#     `_eager_dispatches` fallback, not a `CASCADE_BASES` shape) that is
#     likewise NOT a member of any "must be present" set above — not
#     because the KEYS may be absent (`FinetuneStepTier`/`FinetuneRunTier`
#     both unconditionally serialize this pair), but because its
#     `fused > 0` half is genuinely architecture-conditional rather than
#     universal the way a `REQUIRED_PAIRS` base's is. Rule 1 (the
#     fallback-count hard-fail) still applies to it exactly like any
#     ordinary pair — a live `gelu_eager_dispatches > 0` is never exempted.
REQUIRED_PAIRS = frozenset({"ln", "geglu", "adamw"})
ABSORBABLE_BY_ATTENTION_BLOCK = frozenset({"rope", "softmax"})
ABSORBABLE_BY_ATTENTION_BLOCK_FLASH = frozenset({"attention_block"})
LORA_SITE_EXCLUSIVE_GROUP = frozenset({"lora_epilogue", "lora_linear"})

# CASCADE_BASES — a dispatch pair whose fallback counter is named
# `<base>_declined_dispatches` instead of `<base>_eager_dispatches` (see
# `_fallback_key`): there is no eager COMPOSITION this arm falls back to
# internally the way an ordinary pair's eager composition IS that pair's own
# fallback — on a domain/capability miss the caller falls through to a
# WHOLLY SEPARATE arm's own pair instead (`attention_block_flash` declining
# falls through to `attention_block`, one level up the SAME absorption chain
# `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` documents). Looking for a nonexistent
# `attention_block_flash_eager_dispatches` sibling would raise `KeyError`
# on every leg carrying the pair (e.g.
# `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/
# s128_flash_on_1.json`), and `build_report`'s per-leg `try`/`except` would
# then mark EVERY leg of EVERY config `INVALID`.
#
# Every current `finetune-step` leg (and, via `FinetuneRunTier`'s mirror of
# the same pair, every `finetune-run` leg) carries both keys
# (`report.rs`'s `attention_block_flash_{fused,declined}_dispatches`). A
# fixture predating the pair (absent both keys) is equally handled —
# `dispatch_pairs` simply never discovers this base on such a report.
CASCADE_BASES = frozenset({"attention_block_flash"})

# OPTIONAL_NON_CASCADE_PAIRS — an ORDINARY fused/eager pair (a plain
# `_eager_dispatches` fallback via `_fallback_key`, never the `_declined_
# dispatches` shape `CASCADE_BASES` documents) whose PRESENCE is
# unconditional (`FinetuneStepTier`/`FinetuneRunTier` both always serialize
# `gelu_fused_dispatches`/`gelu_eager_dispatches`) but whose `fused > 0`
# requirement is genuinely ARCHITECTURE-CONDITIONAL, unlike an ordinary
# `REQUIRED_PAIRS` base's universal one: `gelu_erf_fused` (the dense
# erf-GELU activation admit key, `jammi-encoders/src/activations.rs`) is
# BERT's/DistilBERT's dense-GELU FFN seam only — `ModernBertMlp::forward`'s
# GeGLU MLP (already covered by its own `geglu` REQUIRED_PAIRS member)
# never routes through it at all, so a real ModernBERT leg legitimately
# reads `gelu_{fused,eager} == (0, 0)` FOREVER, never a regression to
# investigate. Unlike `ABSORBABLE_BY_ATTENTION_BLOCK`/
# `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`, there is no absorbing counter this
# pair's `(0, 0)` is conditioned on — admission here is by THIS RUN'S OWN
# COUNTERS (tensor state: what actually dispatched), never a model-identity
# branch (no `if backbone == "modernbert"` — this module reads no such
# field and must not grow one). A BERT-family leg (dense GELU-erf FFN)
# reads a LIVE pair here and is held to exactly the bar an ordinary pair
# is: rule 1 below still hard-fails any nonzero `gelu_eager_dispatches` on
# ANY architecture — a real fallback is never exempted just because this
# base's `fused > 0` isn't mandatory. Like `CASCADE_BASES`, the KEYS
# themselves being entirely absent from a report/golden fixture that
# predates the pair is ALSO tolerated — `dispatch_pairs`
# simply never discovers this base on such a report, same mechanism.
OPTIONAL_NON_CASCADE_PAIRS = frozenset({"gelu"})

ALL_BASES = (
    REQUIRED_PAIRS
    | ABSORBABLE_BY_ATTENTION_BLOCK
    | ABSORBABLE_BY_ATTENTION_BLOCK_FLASH
    | LORA_SITE_EXCLUSIVE_GROUP
    | CASCADE_BASES
    | OPTIONAL_NON_CASCADE_PAIRS
)
# An explicit `if`/`raise`, never a bare `assert` -- `assert` is stripped
# entirely under `python -O`, which would silently disable this load-bearing
# pairwise-disjointness guard (every other classification in this module
# assumes exactly-one-class-per-base) in exactly the deployment shape that
# removes the safety net without removing the code path it protects.
if (
    len(REQUIRED_PAIRS)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK_FLASH)
    + len(LORA_SITE_EXCLUSIVE_GROUP)
    + len(CASCADE_BASES)
    + len(OPTIONAL_NON_CASCADE_PAIRS)
    != len(ALL_BASES)
):
    raise AssertionError(
        "REQUIRED_PAIRS / ABSORBABLE_BY_ATTENTION_BLOCK / "
        "ABSORBABLE_BY_ATTENTION_BLOCK_FLASH / LORA_SITE_EXCLUSIVE_GROUP / "
        "CASCADE_BASES / OPTIONAL_NON_CASCADE_PAIRS must be pairwise disjoint -- "
        "every base gets exactly ONE class"
    )

# bf16's ULP near a loss value around 0.30: 7 explicit mantissa bits,
# exponent bucket [0.25, 0.5) => 2^-9. Every real sweep leg runs
# --backbone-dtype/--dtype bf16 (see `run_jammi_leg`/`run_torch_leg` in
# `finetune_ab.sh`), so this is the resolution `loss_first`/`loss_last`
# entries actually carry — see `finetune_step.rs`'s `losses` field doc /
# `torch_finetune_step.py`'s `loss_note` for the same figure stated next to
# the field itself.
BF16_LOSS_ULP_NEAR_0P3 = 2.0**-9  # ~0.001953125


def load_leg(raw_dir, config_slug, leg):
    """Read one `run_leg`-produced `.exit`/`.json`/`.stderr` triple and
    classify its outcome. Never raises: a MISSING/FAIL/OOM/DRY_RUN leg is a
    normal row, not a script error.
    """
    base = os.path.join(raw_dir, f"{config_slug}__{leg}")
    exit_path, out_path, err_path = base + ".exit", base + ".json", base + ".stderr"
    if not os.path.exists(exit_path):
        return {"outcome": "MISSING", "err_tail": "", "report": None}

    with open(exit_path) as fh:
        exit_code = fh.read().strip()
    err_tail = ""
    if os.path.exists(err_path):
        with open(err_path, errors="replace") as fh:
            err_lines = fh.read().splitlines()
        err_tail = "\n".join(err_lines[-5:])

    report = None
    try:
        with open(out_path) as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError):
        report = None

    if report is not None and (report.get("tool") == "dry-run" or report.get("ab_dry_run") is True):
        return {"outcome": "DRY_RUN", "err_tail": "", "report": None}

    if exit_code != "0" or report is None:
        low = err_tail.lower()
        oom_markers = ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed", "outofmemoryerror")
        outcome = "OOM" if any(m in low for m in oom_markers) else "FAIL"
        return {"outcome": outcome, "err_tail": err_tail, "report": None}

    return {"outcome": "OK", "err_tail": "", "report": report}


def finetune_block(report, leg):
    return report["tiers"]["finetune_step"] if leg.startswith("jammi") else report["finetune_step"]


_MISSING = object()  # sentinel -- see leg_identity_fields's own doc


def leg_identity_fields(report, leg):
    """This leg's `FINETUNE_IDENTITY_FIELDS` values, read from their REAL
    location on the RAW report (never `metrics()`'s already-narrowed dict,
    which drops `lora_dropout`/`target_modules`/`seed`/etc. entirely).
    `_TORCH_ARGS_LEVEL_FIELDS` names the fields whose location differs by
    producer -- see `FINETUNE_IDENTITY_FIELDS`'s own doc.

    Returns a `{field: value_or_MISSING}` dict — `_MISSING` (a private
    sentinel, never `None`) marks a field ABSENT *or present-but-null* on
    this report, so `leg_premise_violations` treats BOTH the same way (a
    genuinely-absent key and a present-but-`None` value are the SAME
    "cannot verify this premise determinant" state). A present-but-`None`
    value is reachable from a real producer, not only a fixture: `serde_json`
    serializes a NaN/inf `f64` as JSON `null`, so a NaN `lora_alpha` on
    jammi's side arrives as `null` (the same reason
    `compare_grad_oracle.py` folds present-but-null the same way).

    EXCEPT for `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS` members
    (`max_grad_norm`): there a present `null` IS the premise ("clip OFF") —
    both producers refuse a non-finite value before running, so NaN can
    never reach the report and `null` has exactly one meaning. An ABSENT
    key is still `_MISSING` for those fields (a producer that does not emit
    the field cannot state its premise).
    """
    fs = finetune_block(report, leg)
    fields = {}
    for field in FINETUNE_IDENTITY_FIELDS:
        if field in _TORCH_ARGS_LEVEL_FIELDS and not leg.startswith("jammi"):
            args = report.get("args")
            block = args if isinstance(args, dict) else {}
        else:
            block = fs
        if field not in block:
            fields[field] = _MISSING
            continue
        value = block[field]
        if value is None and field not in FINETUNE_NULL_IS_A_VALUE_FIELDS:
            value = _MISSING
        fields[field] = value
    return fields


def leg_premise_violations(jammi_fields, torch_fields):
    """Per-config leg-premise check: BOTH legs' records must carry every
    `FINETUNE_IDENTITY_FIELDS` entry (present on both, equal after
    `canonicalize_identity_field` — the SAME canonicalizer table
    `compare_grad_oracle.py` uses for its own identity fields, imported
    from `identity_fields.py`), mirroring that module's own
    `_premise_violations` shape: presence checked EXPLICITLY (a field
    absent OR present-but-null from BOTH sides must not silently compare
    `None == None` and pass — `leg_identity_fields` already folds
    present-but-null into `_MISSING` before this function ever sees it, so
    both shapes land on the SAME branch here), never inferred from a bare
    `==`.
    """
    violations = []
    for field in FINETUNE_IDENTITY_FIELDS:
        ja = jammi_fields.get(field, _MISSING)
        jb = torch_fields.get(field, _MISSING)
        missing_sides = []
        if ja is _MISSING:
            missing_sides.append("jammi")
        if jb is _MISSING:
            missing_sides.append("torch")
        if missing_sides:
            violations.append(
                f"leg-identity field {field!r} missing from {missing_sides} leg's record -- cannot "
                "verify the two legs of this config ran under the same premise"
            )
            continue
        va = canonicalize_identity_field(field, ja)
        vb = canonicalize_identity_field(field, jb)
        if va != vb:
            violations.append(f"leg-identity field {field!r} differs: jammi={va!r} torch={vb!r}")
    return violations


def clip_fact_violations(report, leg):
    """Per-leg COUNTED-FACT check for the clip row: a
    leg's `max_grad_norm` states what was REQUESTED; its `clip_invocations`
    is what the producer COUNTED the production clip actually doing
    (jammi: a `CLIP_INVOCATIONS` before/after delta around `run()`'s
    pre-step + loop; torch: `clip_counter` bumped at every
    `clip_grad_norm_` call). The two must agree in kind — requested ⇒
    counted `> 0`; not requested ⇒ counted `== 0` — or the row is
    claiming a step it did not run. A leg that carries neither key is left
    to `leg_premise_violations`'
    own MISSING refusal on `max_grad_norm`; a leg that carries
    `max_grad_norm` but no `clip_invocations` is refused HERE (a clip claim
    with no counted fact behind it). Returns a list of strings, empty when
    consistent.
    """
    fs = finetune_block(report, leg)
    if "max_grad_norm" not in fs:
        return []
    requested = fs["max_grad_norm"]
    if "clip_invocations" not in fs or fs["clip_invocations"] is None:
        return [
            f"{leg}: max_grad_norm={requested!r} is stated but `clip_invocations` (the counted "
            "fact behind a clip row) is absent from this leg's record"
        ]
    counted = fs["clip_invocations"]
    if not isinstance(counted, int) or isinstance(counted, bool) or counted < 0:
        return [f"{leg}: clip_invocations must be a non-negative integer, got {counted!r}"]
    if requested is not None and counted == 0:
        return [f"{leg}: max_grad_norm={requested!r} was requested but clip_invocations == 0 (the clip never ran)"]
    if requested is None and counted > 0:
        return [f"{leg}: max_grad_norm is null (clip off) but clip_invocations == {counted} (the clip ran anyway)"]
    return []


def leg_provenance(report, leg):
    """PROVENANCE (recorded, never compared — see `grad_oracle.rs`'s module
    doc's determinant table for the same identity/provenance/measurement
    split applied to this OTHER cross-producer comparator): torch's
    `attn_requested`/`attn_implementation` pair, jammi's dispatch counters
    (including a `CASCADE_BASES` member's `_declined_dispatches` counter —
    `jammi_dispatch_counters` keys off EITHER fallback-counter suffix, not
    just `_eager_dispatches`, so a cascade pair's raw counts are recorded
    exactly like every other pair's), jammi's resolved
    `JAMMI_KERNELS_DISABLE` state (`kernels_disabled_requested`/
    `kernels_disabled_fired`), and `flash_compiled` — deliberately NOT
    `FINETUNE_IDENTITY_FIELDS` members (torch has no equivalent env var or
    build-capability flag to compare against), recorded here purely so a human
    reading the merged JSON can see which arm jammi's OWN leg measured. `None`
    for the fields the OTHER producer has no equivalent for (never fabricated).
    """
    fs = finetune_block(report, leg)
    if leg.startswith("jammi"):
        return {
            "torch_attn_requested": None,
            "torch_attn_implementation": None,
            "jammi_dispatch_counters": {
                k: v
                for k, v in fs.items()
                if k.endswith("_fused_dispatches") or k.endswith("_eager_dispatches") or k.endswith("_declined_dispatches")
            },
            "jammi_kernels_disabled_requested": fs.get("kernels_disabled_requested"),
            "jammi_kernels_disabled_fired": fs.get("kernels_disabled_fired"),
            "jammi_flash_compiled": fs.get("flash_compiled"),
            # The counted fact behind the clip row, next
            # to the dispatch counters it is the sibling of. Cross-checked
            # against `max_grad_norm` by `clip_fact_violations`.
            "jammi_clip_invocations": fs.get("clip_invocations"),
            "torch_clip_invocations": None,
        }
    args = report.get("args") if isinstance(report.get("args"), dict) else {}
    return {
        "torch_attn_requested": args.get("attn_requested"),
        "torch_attn_implementation": fs.get("attn_implementation"),
        "jammi_dispatch_counters": None,
        "jammi_kernels_disabled_requested": None,
        "jammi_kernels_disabled_fired": None,
        "jammi_flash_compiled": None,
        "jammi_clip_invocations": None,
        "torch_clip_invocations": fs.get("clip_invocations"),
    }


def _fallback_key(base):
    """The sibling counter key for a `<base>_fused_dispatches` field —
    `<base>_declined_dispatches` for a `CASCADE_BASES` member (see that
    set's own doc: a cascade pair has no eager COMPOSITION to fall back to
    internally, only a domain/capability DECLINE that falls through to a
    wholly separate arm's own pair), `<base>_eager_dispatches` for every
    other (ordinary) pair.
    """
    return f"{base}_declined_dispatches" if base in CASCADE_BASES else f"{base}_eager_dispatches"


def dispatch_pairs(fs):
    """Every `(base, fused_key, fallback_key)` positive-proof pair PRESENT
    in this report's `finetune_step` block, discovered from the JSON keys
    themselves rather than a hardcoded name list — a hardcoded ln/rope/
    softmax trio would silently stop catching a NEW fused op (geglu,
    lora_epilogue, lora_linear, attention_block, attention_block_flash, and
    whatever lands next) the day it is added to `finetune_step.rs`'s
    `FinetuneStepTier` without this script being updated in lockstep. Every
    key ending in `_fused_dispatches` names a pair; its sibling is the same
    base's fallback key (`_fallback_key`) — either `_eager_dispatches`
    (`finetune_step.rs`'s own struct guarantees this always exists
    alongside an ORDINARY fused counter — every such pair is added as a
    pair, never solo) or, for a `CASCADE_BASES` member,
    `_declined_dispatches` (see that set's own doc). The returned tuple's third
    element is that fallback count regardless of which key produced it —
    `fused_proof`'s rule 1 (below) treats BOTH shapes uniformly (a real,
    non-deliberate fallback anywhere is a hard fail), so `dispatch_pairs`
    itself does not need to distinguish them past this point.

    SCHEMA STRICTNESS: this function stays LOUD (raises `KeyError`) on a
    solo counter — a fused key with no fallback sibling is a genuine schema
    bug (a struct field added without its pair), never a config this script
    should silently skip. The SAME loudness covers a base `fused_proof`'s
    classification tables (`REQUIRED_PAIRS` /
    `ABSORBABLE_BY_ATTENTION_BLOCK` / `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` /
    `LORA_SITE_EXCLUSIVE_GROUP` / `CASCADE_BASES` / `OPTIONAL_NON_CASCADE_PAIRS`,
    whose union is `ALL_BASES`) do not know about: a NEW fused kernel landing in
    `finetune_step.rs` without this module's classification tables being
    updated in lockstep is exactly the same class of schema drift as a
    solo counter — `fused_proof` would otherwise silently never require
    anything of it. `metrics()`'s two `.get()` reads for
    `loss_first`/`loss_last` are the OPPOSITE choice, deliberately: those
    two fields are optional/best-effort table decoration (absence changes
    nothing this proof depends on), while a dispatch pair — and its
    classification — is STRUCTURAL to `fused_proof`'s entire claim. The
    exception is caught per leg by `build_report`'s `try`/`except`, so one
    bad leg's solo-counter (or unclassified-base) `KeyError` never discards
    the merged table for every other config.
    """
    pairs = []
    for key in fs:
        if not key.endswith("_fused_dispatches"):
            continue
        base = key[: -len("_fused_dispatches")]
        fallback_key = _fallback_key(base)
        if fallback_key not in fs:
            raise KeyError(
                f"'{key}' has no matching '{fallback_key}' in the report — "
                "finetune_step.rs's fused/eager (or, for a CASCADE_BASES "
                "member, fused/declined) counters are supposed to always "
                "come in pairs; a solo counter is a schema bug, not a "
                "config this script should silently skip."
            )
        if base not in ALL_BASES:
            raise KeyError(
                f"dispatch-pair base {base!r} (from {key!r}) is not classified in ALL_BASES "
                f"({sorted(ALL_BASES)!r}) — a NEW fused kernel landed in finetune_step.rs "
                "without fused_proof's REQUIRED_PAIRS / ABSORBABLE_BY_ATTENTION_BLOCK / "
                "ABSORBABLE_BY_ATTENTION_BLOCK_FLASH / LORA_SITE_EXCLUSIVE_GROUP / "
                "CASCADE_BASES / OPTIONAL_NON_CASCADE_PAIRS tables being updated to cover it. "
                "This is a schema-drift bug, not a base this script should silently leave "
                "unchecked (see the module-level classification tables' own doc)."
            )
        pairs.append((base, fs[key], fs[fallback_key]))
    return pairs


def metrics(entry, leg):
    """Extract this leg's table/proof metrics from its raw report. Returns
    `None` when the leg itself did not produce a usable report (see
    `load_leg`); raises (never silently drops a field) when the report WAS
    produced but a STRUCTURAL piece — a dispatch pair — is malformed (see
    `dispatch_pairs`'s own doc for why that is the loud half of this
    module's schema-strictness split).
    """
    if entry["outcome"] != "OK":
        return None
    fs = finetune_block(entry["report"], leg)
    m = {
        "s_per_step_p50": fs["s_per_step_p50"]["value"],
        "triplets_per_s": fs["triplets_per_s"]["value"],
        "loss_first": fs.get("loss_first"),
        "loss_last": fs.get("loss_last"),
    }
    if leg.startswith("jammi"):
        m["vram_delta_bytes"] = fs["peak_vram_bytes"]["value"]
        m["vram_absolute_bytes"] = None
        m["dispatch_pairs"] = dispatch_pairs(fs)
        # `fused_proof`'s own flash-disable-consistency check (see that
        # function's doc) reads these off `m` rather than the raw `fs` a
        # second time, keeping `fused_proof`'s `m`-only signature. `.get()`
        # (never `[...]`): a report may lack these keys, and this must not
        # raise for that case.
        m["flash_compiled"] = fs.get("flash_compiled")
        m["kernels_disabled_requested"] = fs.get("kernels_disabled_requested")
        m["kernels_disabled_fired"] = fs.get("kernels_disabled_fired")
    else:
        m["vram_delta_bytes"] = fs["peak_vram_delta_bytes"]["value"]
        m["vram_absolute_bytes"] = fs["peak_vram_absolute_bytes"]["value"]
    return m


def fused_proof(m):
    """See the module-level `REQUIRED_PAIRS`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
    `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`/`LORA_SITE_EXCLUSIVE_GROUP`/
    `CASCADE_BASES`/`OPTIONAL_NON_CASCADE_PAIRS` (union `ALL_BASES`) doc for
    the classification this checks each pair against. Returns
    `True`/`False`/`None` (no `dispatch_pairs` at all — not a jammi leg, or
    the leg itself did not run) or a `str` (the flash-disable-consistency
    check below errored; `build_report` treats a
    `str` return the same as `False`, see its own `proof is False or
    isinstance(proof, str)` branch). Raises (via `dispatch_pairs`, which
    `metrics()` already calls before this function ever sees `m` — see that
    function's own doc) if `m["dispatch_pairs"]` would ever contain a base
    outside `ALL_BASES`; `fused_proof` itself never receives an unclassified
    base to begin with.

    Rules, in order — EVERY base in `ALL_BASES` (not just `REQUIRED_PAIRS`)
    must be PRESENT in this report's pairs; absence is a hard fail for
    every classified base, never a silently-granted exemption, EXCEPT
    `CASCADE_BASES` members (genuinely OPTIONAL PRESENCE, see that set's own
    doc) and `OPTIONAL_NON_CASCADE_PAIRS` members (PRESENCE is unconditional,
    but `fused > 0` is architecture-conditional rather than
    universal — see that set's own doc; a `(0, 0)` reading there is never
    itself a failure, so this function runs no explicit "must be present"
    loop for it at all, unlike every set above):
      0. `flash_compiled is False` AND
         `kernels_disabled_requested` names `attention_block_flash` is a
         hard, unconditional fail — a disable request naming an op this
         BUILD never compiled in cannot possibly have exercised anything;
         the leg's own build configuration already contradicts its own
         disable request, before a single dispatch pair is even inspected.
      1. ANY pair with a fallback count (`eager`, or a `CASCADE_BASES`
         member's `declined`) `> 0` is a hard, unconditional fail — an
         admitted call site that actually fell back, on ANY pair, in ANY
         group — UNLESS that pair is a `CASCADE_BASES` member AND its base
         appears in BOTH `kernels_disabled_requested` AND
         `kernels_disabled_fired` on this SAME leg: a DELIBERATE,
         self-describing disable request (the reference/block-arm leg of a
         flash-vs-block A/B, `JAMMI_KERNELS_DISABLE=attention_block_flash`)
         is not a silent fallback — it is the transparently-requested and
         transparently-recorded way this crate forces the non-flash arm,
         and the reference leg's OWN `attention_block` pair still has to
         independently clear rule 2.5 below on its own `fused > 0`, so
         nothing here grants it a free pass on the thing that actually
         matters. An UNREQUESTED decline (a genuine domain/capability
         miss — real padding, wrong arch, `flash-attn` not compiled) stays
         a hard fail exactly like an ordinary silent eager fallback
         (`report.rs`'s own `attention_block_flash_declined_dispatches`
         field doc: "`declined > 0` on any bench leg -> INVALID").
         `flash_compiled is False` is deliberately NOT exempted here:
         `fused_proof` is shared by every sweep, and a build fact that
         voids ONE comparison's premise (the `train-run` ladder's fused
         rung IS the flash cascade) belongs in that comparison's own premise
         (`crates/jammi-bench/src/ladder/premise.rs`'s `FusedDispatch`), never
         a silent, generic exemption inside this dispatch-classification
         primitive.
      2. Every `REQUIRED_PAIRS` base must be PRESENT in this report's pairs
         (a required pair vanishing from the JSON entirely — the field
         renamed, deleted, or feature-gated off — is exactly the schema
         regression this proof exists to catch, never silently excluded)
         AND show `fused > 0`.
      2.5. Every `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` member (today, only
         `attention_block`) must be PRESENT (same "absence is a fail" rule
         as step 2), and may read `(0, 0)` ONLY when
         `attention_block_flash`'s own `fused` count is `> 0` in this SAME
         report (defaulting to `0` when that base is entirely absent, so
         this reduces to "must independently clear `fused > 0`" there);
         otherwise it must independently clear `fused > 0`.
      3. Every `ABSORBABLE_BY_ATTENTION_BLOCK` member must be PRESENT (same
         "absence is a fail" rule as step 2), and may read
         `(0, 0)` ONLY when `attention_block`'s own `fused` count is `> 0`
         OR `attention_block_flash`'s own `fused`
         count is `> 0`, in this SAME report; otherwise it must
         independently clear `fused > 0`.
      4. Every `LORA_SITE_EXCLUSIVE_GROUP` member must be PRESENT (same
         rule again), and the GROUP is then checked AS A GROUP: the SUM of
         their `fused` counts must be `> 0` (whichever member actually
         carries this run's dispatch — see the group's own doc).
      5. Overall: at least one pair ANYWHERE in the report must show
         `fused > 0` — a report where every single pair reads `(0, 0)`
         (e.g. a schema regression that dropped every counter, or a
         flash-arm leg where the cascade itself never fired AND the block
         arm it would otherwise have absorbed into also never fired) is
         NOT vacuously `True`. Steps 2/2.5/3/4 already make this true
         whenever `REQUIRED_PAIRS` is non-empty, but this stays a distinct,
         independently-stated check so the property holds even if
         `REQUIRED_PAIRS` were ever emptied.
    """
    if m is None:
        return None

    # Rule 0 — see this function's own doc. Checked BEFORE `dispatch_pairs`
    # is even inspected: a build/disable-request contradiction invalidates
    # the leg regardless of what its counters happen to read.
    if m.get("flash_compiled") is False and "attention_block_flash" in (m.get("kernels_disabled_requested") or []):
        return (
            "flash_compiled=False but kernels_disabled_requested names "
            "'attention_block_flash' — a disable request against an op "
            "this build never compiled in cannot have exercised anything; "
            "this leg's own build configuration contradicts its own "
            "disable request"
        )

    pairs = m.get("dispatch_pairs")
    if not pairs:
        return False
    by_base = {base: (fused, fallback) for base, fused, fallback in pairs}

    kernels_disabled_requested = set(m.get("kernels_disabled_requested") or [])
    kernels_disabled_fired = set(m.get("kernels_disabled_fired") or [])
    for base, (_fused, fallback) in by_base.items():
        if fallback <= 0:
            continue
        if base in CASCADE_BASES and base in kernels_disabled_requested and base in kernels_disabled_fired:
            continue  # rule 1: deliberate, self-describing disable request — not a silent fallback
        return False

    for base in REQUIRED_PAIRS:
        if base not in by_base:
            return False
        fused, _fallback = by_base[base]
        if fused == 0:
            return False

    attention_block_flash_fused = by_base.get("attention_block_flash", (0, 0))[0]
    for base in ABSORBABLE_BY_ATTENTION_BLOCK_FLASH:
        if base not in by_base:
            return False  # absence is a schema regression, never silently excluded
        fused, _fallback = by_base[base]
        if fused == 0 and attention_block_flash_fused == 0:
            return False

    attention_block_fused = by_base.get("attention_block", (0, 0))[0]
    attention_ran = attention_block_fused > 0 or attention_block_flash_fused > 0
    for base in ABSORBABLE_BY_ATTENTION_BLOCK:
        if base not in by_base:
            return False  # absence is a schema regression, never silently excluded
        fused, _fallback = by_base[base]
        if fused == 0 and not attention_ran:
            return False

    for base in LORA_SITE_EXCLUSIVE_GROUP:
        if base not in by_base:
            return False  # absence is a schema regression, never silently excluded
    lora_group_fused = sum(by_base[base][0] for base in LORA_SITE_EXCLUSIVE_GROUP)
    if lora_group_fused == 0:
        return False

    return any(fused > 0 for fused, _fallback in by_base.values())


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_loss(v):
    """`loss_first`/`loss_last` are bf16-sourced on every real sweep
    leg (ULP ~0.00195 near 0.30 — see `BF16_LOSS_ULP_NEAR_0P3`). `fmt`'s
    default 4 decimal digits (resolution 0.0001) implies precision the
    dtype does not carry; 3 decimals (resolution 0.001) is still finer
    than the ULP without implying a 4th significant digit exists.
    """
    return "n/a" if v is None else f"{v:.3f}"


def fmt_bytes(v):
    return "n/a" if v is None else f"{int(v):,}"


def bar_second_run_metrics(raw_dir, slug):
    """Load + extract metrics for the order-balanced bar legs' SECOND run
    (`BAR_SECOND_RUN_LEGS`: `"jammi-fused-2"`/`"torch-sdpa-2"`) — returns
    `(entries, metrics_by_leg, merge_errors_by_leg)`, mirroring the SAME
    load/try-except shape `build_report`'s own primary per-leg loop already
    uses for `LEGS` (never a second, differently-shaped error-handling
    path — the "LOUD, per-leg, never fatal to the rest of the merge"
    discipline applies here identically). A leg entirely ABSENT from
    `raw_dir` (a sweep that ran only the four primary legs), or
    `AB_DRY_RUN`'s own DRY_RUN outcome, reads `outcome="MISSING"`/`"DRY_RUN"`
    — `metrics()` then returns `None` for it, and the CALLER
    (`build_report`) falls back to the single-pair ratio, exactly as if the
    second-run legs did not exist.
    """
    entries = {}
    metrics_by_leg = {}
    errors_by_leg = {}
    for leg in BAR_SECOND_RUN_LEGS.values():
        entry = load_leg(raw_dir, slug, leg)
        entries[leg] = entry
        try:
            metrics_by_leg[leg] = metrics(entry, leg)
            errors_by_leg[leg] = None
        except Exception as exc:  # noqa: BLE001 -- LOUD, per-leg,
            # never silent, never fatal to the rest of the merge.
            metrics_by_leg[leg] = None
            errors_by_leg[leg] = f"{type(exc).__name__}: {exc}"
    return entries, metrics_by_leg, errors_by_leg


def bar_pair_ratio(fused_m, sdpa_m):
    """`triplets_per_s` ratio for ONE bar pair (jammi-fused-shaped metrics
    over torch-sdpa-shaped metrics) — the SAME expression `build_report`'s
    own pair-1 `ratio` uses, factored out so pair 1 and pair 2
    (the A,B,B,A protocol's second run) compute it identically rather than
    two independently-drifting copies of the same division. `None` when
    either leg did not produce usable metrics, or torch's own throughput
    read a falsy (zero/`None`) value — never a `ZeroDivisionError`.
    """
    return (
        fused_m["triplets_per_s"] / sdpa_m["triplets_per_s"]
        if (fused_m and sdpa_m and sdpa_m["triplets_per_s"])
        else None
    )


def bar_ratio_classification(pair1_ratio, pair2_ratio, pass_ratio):
    """The order-balanced A,B,B,A bar-ratio classification (finetune_ab.sh
    header's "ORDER-BALANCED BAR LEGS"): given the two adjacent-pair
    ratios (pair 1 = jammi-fused/torch-sdpa, pair 2 =
    jammi-fused-2/torch-sdpa-2 — both `jammi-fused-shaped/torch-sdpa-shaped`,
    see `bar_pair_ratio`), returns `(bar_ratio, indeterminate, detail)`.

    NEVER RAISES for ANY combination of `None`s — `bar_pair_ratio` itself
    reads `None` for a leg that OOM'd/FAILED (not just MISSING), and the
    call site in `build_report` is bare, never wrapped in the per-leg
    try/except `metrics()`/`dispatch_pairs()` get, so a raise here (e.g. a
    bare `min()` when the FIRST run's torch-sdpa OOM'd while a clean second
    run exists) would crash the ENTIRE merge, not just this config's row.
    This function is the one place that guarantee is enforced:

      * BOTH `None` (neither run produced a usable pair — no data at all):
        `bar_ratio = None`, `indeterminate = False` — `build_report`'s own
        `elif bar_ratio is None` branch renders this as its "no ratio"
        FAIL.
      * EXACTLY ONE `None` (the other run's own OOM/FAIL/MISSING legs are
        ALREADY classified by `build_report`'s `torch_fits`/
        `jammi_fused_fits` — see those variables' own doc; both cover both
        runs — before this value is ever consulted for a
        verdict; this function's OWN job is only to return a well-defined,
        non-crashing number here, never to re-derive that classification):
        `bar_ratio` is whichever pair IS available, `indeterminate =
        False` — the single-pair degrade, symmetric in EITHER direction.
      * BOTH present: `bar_ratio = min(pair1_ratio, pair2_ratio)` — the
        estimator LEAST FAVOURABLE to jammi (the SAME "ratio uses the min
        of two torch runs" convention `docs/maintainer/
        fine-tune-performance-guide.md`'s own stacked-sweep artifact
        caveat names for its own two-torch-run sweep, applied
        here to this producer's own two torch-sdpa repeats). `indeterminate`
        is `True` when the two pair ratios STRADDLE `pass_ratio` (one at
        or above, one below — genuinely conflicting classifications) OR
        their spread exceeds `bar_ratio`'s own distance from `pass_ratio`
        (`|pair1_ratio - pair2_ratio| > |bar_ratio - pass_ratio|` — even
        when both land on the same side, a spread that large means the
        combined estimate is not resolved with enough confidence relative
        to how close it sits to the bar to trust either classification).
        Straddling always implies the spread condition too (if one ratio
        is `>= pass_ratio` and the other, which equals `bar_ratio` since
        it is the smaller, is `< pass_ratio`, then `spread >= pass_ratio -
        bar_ratio == margin`) — both are checked explicitly anyway, for
        the boundary-equality edge case and for readability at the call
        site.
    """
    if pair1_ratio is None and pair2_ratio is None:
        return None, False, None
    if pair1_ratio is None:
        return pair2_ratio, False, None
    if pair2_ratio is None:
        return pair1_ratio, False, None
    bar = min(pair1_ratio, pair2_ratio)
    margin = abs(bar - pass_ratio)
    spread = abs(pair1_ratio - pair2_ratio)
    straddle = (pair1_ratio >= pass_ratio) != (pair2_ratio >= pass_ratio)
    indeterminate = straddle or spread > margin
    detail = (
        f"pair1(jammi-fused/torch-sdpa)={pair1_ratio:.3f} "
        f"pair2(jammi-fused-2/torch-sdpa-2)={pair2_ratio:.3f} "
        f"spread={spread:.3f} bar-distance-from-{pass_ratio}={margin:.3f}"
    )
    return bar, indeterminate, detail


def config_slugs(raw_dir):
    slugs = set()
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            if name.endswith(".exit") and "__" in name:
                slugs.add(name.split("__", 1)[0])
    return sorted(slugs)


def build_report(raw_dir, steps, warmup, pass_ratio, torch_lora_init="peft"):
    """The merge stage itself: read every leg under `raw_dir`, extract
    metrics, compute the fused-dispatch proof / throughput ratio / loss
    ratio / verdict per config, and render both the merged JSON dict and
    the printed table string. Returns `(merged, table)`, or `(None, None)`
    if `raw_dir` has no leg output at all (an empty sweep — the caller
    treats this as a hard failure).

    A merge-stage error on ONE leg (`metrics()`/`dispatch_pairs()`
    raising — a solo dispatch counter, a missing report key) is caught
    HERE, per leg, so it produces a LOUD per-row error (visible in both the
    table and the JSON, under that leg's `outcome`/this config's
    `jammi_fused_dispatch_proof`) instead of discarding every OTHER
    config's row too — `dispatch_pairs` still raises; only WHERE it is
    caught is chosen here.
    """
    slugs = config_slugs(raw_dir)
    if not slugs:
        return None, None

    # Read ONCE, applies to every config in this
    # `raw_dir` — see `TWO_RUN_PROTOCOL_MARKER`'s own doc.
    two_run_mode = two_run_protocol_active(raw_dir)

    merged = {
        "steps": steps,
        "warmup": warmup,
        "pass_ratio_bar": pass_ratio,
        "two_run_protocol": two_run_mode,
        "lora_init": {
            "torch": torch_lora_init,
            "jammi": "jammi (LoraInitMode::ZerosB; not configurable via finetune-step's CLI)",
            "note": "A loss-trajectory-equivalence comparison additionally requires "
            "torch_lora_init == 'jammi' (torch_finetune_step.py's --lora-init jammi re-draws "
            "A from jammi's own bound) — a throughput-only sweep (this script's default, "
            "'peft') does not need matched init at all. See torch_finetune_step.py's "
            "'LoRA INIT IS NOT A MATCH BY DEFAULT' section.",
        },
        "configs": {},
    }
    table_rows = []
    summary_rows = []

    for slug in slugs:
        entries = {leg: load_leg(raw_dir, slug, leg) for leg in LEGS}
        leg_metrics = {}
        leg_merge_errors = {}
        for leg in LEGS:
            try:
                leg_metrics[leg] = metrics(entries[leg], leg)
                leg_merge_errors[leg] = None
            except Exception as exc:  # noqa: BLE001 -- LOUD, per-leg,
                # never silent, never fatal to the rest of the merge; see
                # this function's own doc and `dispatch_pairs`'s.
                leg_metrics[leg] = None
                leg_merge_errors[leg] = f"{type(exc).__name__}: {exc}"

        # ORDER-BALANCED A,B,B,A bar legs (finetune_ab.sh header) — the
        # SECOND run of the bar pair (see `bar_second_run_metrics`'s own doc
        # for how a `raw_dir` without these two legs merges).
        second_run_entries, second_run_metrics, second_run_errors = bar_second_run_metrics(raw_dir, slug)

        if leg_merge_errors["jammi-fused"] is not None:
            proof = f"ERROR: {leg_merge_errors['jammi-fused']}"
        else:
            proof = fused_proof(leg_metrics["jammi-fused"])

        # ORDER-BALANCED A,B,B,A bar legs' SECOND run: the bar ratio
        # consumes BOTH pair legs (`bar_pair_ratio`'s pair-2 half below), so
        # `jammi-fused-2` must clear the SAME `fused_proof` positive-proof
        # channel `jammi-fused` does — an unproven leg feeding the
        # pre-registered throughput endpoint is exactly the class
        # `fused_proof` exists to catch, and it does not stop mattering
        # because the leg happens to be the SECOND run rather than the
        # first. `fused_proof(None)` is `None` (never `False`/`str`), so
        # this is safe to compute unconditionally: a MISSING (no second
        # run at all), FAILED, or OOM'd `jammi-fused-2`
        # never spuriously invalidates the config through this channel —
        # only a REPORT that was actually produced (`OK`, or a merge-stage
        # schema error on one that tried to be) can.
        jammi_fused_2_leg = BAR_SECOND_RUN_LEGS["jammi-fused"]
        if second_run_errors[jammi_fused_2_leg] is not None:
            proof2 = f"ERROR: {second_run_errors[jammi_fused_2_leg]}"
        else:
            proof2 = fused_proof(second_run_metrics[jammi_fused_2_leg])

        # LEG-PREMISE CHECK: compares the jammi-fused
        # leg's record (the one this sweep's own ratio/proof are computed
        # from) against the torch-sdpa leg's. The `jammi-eager`/`torch-eager`
        # FALLBACKS below are used for PROVENANCE only — see the
        # `leg_premise_not_comparable` note further down — the two legs of
        # ONE config are supposed to have run under the IDENTICAL
        # seed/batch/seq/dtype/dropout/lora premise (`finetune_ab.sh`'s
        # matched-flags convention), and this is where that premise is
        # CHECKED rather than merely assumed. `None`
        # (never an empty list) when neither side has an OK leg to compare
        # -- an EMPTY list asserts "checked, no violations found", which
        # would be false when there was nothing to check at all.
        jammi_premise_leg = "jammi-fused" if entries["jammi-fused"]["outcome"] == "OK" else (
            "jammi-eager" if entries["jammi-eager"]["outcome"] == "OK" else None
        )
        torch_premise_leg = "torch-sdpa" if entries["torch-sdpa"]["outcome"] == "OK" else (
            "torch-eager" if entries["torch-eager"]["outcome"] == "OK" else None
        )
        leg_premise_violations_list = None
        jammi_provenance = None
        torch_provenance = None
        if jammi_premise_leg is not None:
            jammi_provenance = leg_provenance(entries[jammi_premise_leg]["report"], jammi_premise_leg)
        if torch_premise_leg is not None:
            torch_provenance = leg_provenance(entries[torch_premise_leg]["report"], torch_premise_leg)
        # A leg that is only a FALLBACK
        # (torch-sdpa OOM'd → torch-eager; jammi-fused failed → jammi-eager)
        # is the OTHER attention reference class, so its `attention_arm`
        # can never match the preferred leg's — refusing that as an identity
        # mismatch would turn a documented NON-gating outcome (an OOM row)
        # into `INVALID` + exit 1. The row is "not comparable" instead: the
        # identity check is SKIPPED (`leg_premise_violations` stays `None`,
        # never an empty "checked, clean" list), the reason is recorded
        # (`leg_premise_not_comparable`), and the ratio/verdict logic below
        # handles the missing preferred leg.
        leg_premise_not_comparable = None
        if jammi_premise_leg is not None and torch_premise_leg is not None:
            fallbacks = [
                f"{leg} is a fallback for {preferred} ({entries[preferred]['outcome']})"
                for leg, preferred in ((jammi_premise_leg, "jammi-fused"), (torch_premise_leg, "torch-sdpa"))
                if leg != preferred
            ]
            if fallbacks:
                leg_premise_not_comparable = (
                    "identity check skipped — " + "; ".join(fallbacks) + " — the two legs are "
                    "different attention reference classes by construction, not a premise mismatch"
                )
            else:
                jammi_id_fields = leg_identity_fields(entries[jammi_premise_leg]["report"], jammi_premise_leg)
                torch_id_fields = leg_identity_fields(entries[torch_premise_leg]["report"], torch_premise_leg)
                leg_premise_violations_list = leg_premise_violations(jammi_id_fields, torch_id_fields)
                # The clip row's stated `max_grad_norm` must
                # be backed by its own counted `clip_invocations`, per leg, on
                # the SAME two legs the premise check compared.
                for leg in (jammi_premise_leg, torch_premise_leg):
                    leg_premise_violations_list.extend(clip_fact_violations(entries[leg]["report"], leg))

        # SECOND-RUN leg-premise + provenance — the SAME two checks the
        # primary pair gets, reused verbatim (`leg_identity_fields`/
        # `leg_premise_violations`/`clip_fact_violations`/`leg_provenance`
        # are already leg-name-generic, see their own docs), applied to
        # `jammi-fused-2`/`torch-sdpa-2` WHEN PRESENT. Unlike the primary
        # pair, there is no eager-leg FALLBACK to fall back to here
        # (`jammi-eager`/`torch-eager` are single, non-repeated context
        # legs — see `finetune_ab.sh`'s header): the second-run premise
        # check is therefore only MEANINGFUL (and only run) when BOTH
        # `jammi-fused-2` AND `torch-sdpa-2` themselves read `OK` — a
        # single-run `raw_dir` (both `MISSING`) or a second-run OOM/FAIL on
        # either side degrades to "not checked" (`None`, never an empty
        # "checked, clean" list), the SAME shape `leg_premise_violations_list`
        # itself uses when there is nothing to compare — `bar_pair_ratio`'s
        # own `pair2_ratio` already reads `None` in that case too, so the
        # verdict already degrades to the single-pair estimator without
        # this check's help; this check's OWN job is only to refuse when a
        # SECOND-RUN report was actually produced and disagrees.
        torch_sdpa_2_leg = BAR_SECOND_RUN_LEGS["torch-sdpa"]
        second_run_premise_violations_list = None
        # NOTE (not a gate): when NEITHER second-run leg is `OK`
        # (a single-run `raw_dir`, a fully-dry-run sweep, or both OOM'd/FAILED
        # independently), BOTH entries below stay `None`/`None` in the
        # rendered JSON — this is the ORDINARY "nothing to record"
        # rendering `leg_provenance` already gives every OTHER absent leg
        # (see that function's own doc), never itself a distinct signal a
        # human reader or another check should treat as meaningful beyond
        # "the second run did not produce a report" — the actual
        # measurement-completeness signal for THAT case lives in
        # `bar_second_run_legs[<leg>]["outcome"]` (`MISSING`/`FAIL`/`OOM`)
        # and, under `two_run_mode`, `two_run_missing_leg_reason` — never
        # in this dict reading `None`.
        second_run_provenance = {jammi_fused_2_leg: None, torch_sdpa_2_leg: None}
        if second_run_entries[jammi_fused_2_leg]["outcome"] == "OK":
            second_run_provenance[jammi_fused_2_leg] = leg_provenance(
                second_run_entries[jammi_fused_2_leg]["report"], jammi_fused_2_leg
            )
        if second_run_entries[torch_sdpa_2_leg]["outcome"] == "OK":
            second_run_provenance[torch_sdpa_2_leg] = leg_provenance(
                second_run_entries[torch_sdpa_2_leg]["report"], torch_sdpa_2_leg
            )
        if (
            second_run_entries[jammi_fused_2_leg]["outcome"] == "OK"
            and second_run_entries[torch_sdpa_2_leg]["outcome"] == "OK"
        ):
            jammi_id_fields2 = leg_identity_fields(second_run_entries[jammi_fused_2_leg]["report"], jammi_fused_2_leg)
            torch_id_fields2 = leg_identity_fields(second_run_entries[torch_sdpa_2_leg]["report"], torch_sdpa_2_leg)
            second_run_premise_violations_list = leg_premise_violations(jammi_id_fields2, torch_id_fields2)
            for leg in (jammi_fused_2_leg, torch_sdpa_2_leg):
                second_run_premise_violations_list.extend(
                    clip_fact_violations(second_run_entries[leg]["report"], leg)
                )

        # CROSS-RUN premise: the SAME-run checks
        # above (`leg_premise_violations_list` for jammi-fused/torch-sdpa,
        # `second_run_premise_violations_list` for jammi-fused-2/
        # torch-sdpa-2) never compare ACROSS the two runs at all -- a
        # config where run 1 used `seed=7` and run 2 used a DIFFERENT
        # seed (or `seq`, or any other identity field) would pass BOTH
        # same-run checks cleanly while the bar ratio silently averages
        # two genuinely different measurements together. Checked
        # independently: jammi-fused vs jammi-fused-2, and torch-sdpa vs
        # torch-sdpa-2, each only when BOTH sides read `OK`. Reuses
        # `leg_identity_fields` (already leg-name-generic, correctly
        # resolving torch's own args-level field split for either torch
        # leg name) to extract each leg's fields, then
        # `generic_leg_premise_violations` (custom `label_a`/`label_b`,
        # unlike `leg_premise_violations`'s own hardcoded "jammi="/
        # "torch=" prose, which would mislabel a jammi-vs-jammi or
        # torch-vs-torch pair) to diff them -- the SAME `_MISSING`
        # sentinel and `canonicalize_identity_field` table both paths
        # share, never a third, independently-drifting comparator.
        # Tri-state: `None` means "not checked at all" (neither
        # sub-comparison below had both its legs `OK`); an empty `[]`
        # means "checked -- at least one sub-comparison RAN -- and found
        # NO drift"; a non-empty list means "checked and found drift".
        # `(cross_run_premise_violations_list or []) + v` is the operative
        # line in EACH branch below: the FIRST branch that actually runs
        # flips the sentinel from `None` to a real (possibly empty) list
        # UNCONDITIONALLY, not merely when `v` is truthy -- otherwise
        # "checked, clean" and "never checked" would BOTH read `None`, and
        # this field could never state a positive "the cross-run premise
        # was verified" fact. (`ci/artifacts/finetune-ab-runs/
        # 2026-08-30-full-sweep-acce7b3d-a100-pcie/finetune_ab_report.json`
        # reads `null` here throughout; see that artifact's own README.)
        cross_run_premise_violations_list = None
        if entries["jammi-fused"]["outcome"] == "OK" and second_run_entries[jammi_fused_2_leg]["outcome"] == "OK":
            jammi_run1_fields = leg_identity_fields(entries["jammi-fused"]["report"], "jammi-fused")
            jammi_run2_fields = leg_identity_fields(second_run_entries[jammi_fused_2_leg]["report"], jammi_fused_2_leg)
            v = generic_leg_premise_violations(
                FINETUNE_IDENTITY_FIELDS, jammi_run1_fields, jammi_run2_fields,
                label_a="jammi-fused", label_b=jammi_fused_2_leg,
            )
            cross_run_premise_violations_list = (cross_run_premise_violations_list or []) + v
        if entries["torch-sdpa"]["outcome"] == "OK" and second_run_entries[torch_sdpa_2_leg]["outcome"] == "OK":
            torch_run1_fields = leg_identity_fields(entries["torch-sdpa"]["report"], "torch-sdpa")
            torch_run2_fields = leg_identity_fields(second_run_entries[torch_sdpa_2_leg]["report"], torch_sdpa_2_leg)
            v = generic_leg_premise_violations(
                FINETUNE_IDENTITY_FIELDS, torch_run1_fields, torch_run2_fields,
                label_a="torch-sdpa", label_b=torch_sdpa_2_leg,
            )
            cross_run_premise_violations_list = (cross_run_premise_violations_list or []) + v

        for leg in LEGS:
            err_tail = entries[leg]["err_tail"]
            if leg_merge_errors[leg] is not None:
                err_tail = (err_tail + "\n" if err_tail else "") + f"[merge-stage] {leg_merge_errors[leg]}"
            # The negative control's own two provenance facts, surfaced
            # on the jammi-eager row specifically (never every row — these
            # two fields are `None` on every other leg, see `leg_provenance`)
            # so a human reading the table sees, next to the row it
            # describes, whether the requested disable list actually fired.
            if leg == "jammi-eager" and entries[leg]["outcome"] == "OK" and leg_merge_errors[leg] is None:
                prov = leg_provenance(entries[leg]["report"], leg)
                kd_lines = (
                    f"kernels_disabled_requested={prov['jammi_kernels_disabled_requested']}\n"
                    f"kernels_disabled_fired={prov['jammi_kernels_disabled_fired']}"
                )
                err_tail = (err_tail + "\n" if err_tail else "") + kd_lines
            table_rows.append(
                (
                    slug,
                    leg,
                    entries[leg]["outcome"],
                    leg_metrics[leg],
                    proof if leg == "jammi-fused" else None,
                    err_tail,
                )
            )

        # ORDER-BALANCED A,B,B,A bar legs' own SECOND run — supplementary
        # table rows. `jammi-fused-2`'s OWN `fused_proof` (`proof2`, above)
        # surfaces in the SAME `fused_proof` column the primary
        # `jammi-fused` row uses — this pair leg carries the SAME positive-
        # proof discipline, so its own column reads the same way. Omitted
        # when the second run never ran at all (`MISSING` — a single-run
        # `raw_dir`/`AB_DRY_RUN`'s own placeholder legs), so a single-run
        # table carries no "MISSING" clutter rows.
        for leg in BAR_SECOND_RUN_LEGS.values():
            if second_run_entries[leg]["outcome"] == "MISSING":
                continue
            err_tail = second_run_entries[leg]["err_tail"]
            if second_run_errors[leg] is not None:
                err_tail = (err_tail + "\n" if err_tail else "") + f"[merge-stage] {second_run_errors[leg]}"
            table_rows.append(
                (
                    slug,
                    leg,
                    second_run_entries[leg]["outcome"],
                    second_run_metrics[leg],
                    proof2 if leg == jammi_fused_2_leg else None,
                    err_tail,
                )
            )

        fused_m, sdpa_m = leg_metrics["jammi-fused"], leg_metrics["torch-sdpa"]
        ratio = bar_pair_ratio(fused_m, sdpa_m)

        # `pair2_ratio` feeds `bar_ratio_classification` below.
        pair2_ratio = bar_pair_ratio(
            second_run_metrics[BAR_SECOND_RUN_LEGS["jammi-fused"]],
            second_run_metrics[BAR_SECOND_RUN_LEGS["torch-sdpa"]],
        )
        bar_ratio, bar_indeterminate, bar_indeterminate_detail = bar_ratio_classification(
            ratio, pair2_ratio, pass_ratio
        )

        # loss_final_ratio: jammi-fused's loss_last over torch-sdpa's
        # loss_last. SAME DATA, COST FIXTURE -- NOT A QUALITY RESULT (per
        # finetune_step.rs's own module doc's "Honesty about what is
        # measured", and torch_finetune_step.py's "LOSS TRAJECTORY"
        # section): the two stacks run different attention-kernel
        # arithmetic and different LoRA init distributions unless
        # torch_lora_init == "jammi", so a ratio far from 1.0 does NOT mean
        # either stack is wrong -- it means the loss values are not
        # comparable under these settings. Printed anyway so a large
        # divergence is VISIBLE to a human reader, never asserted against a
        # bar.
        loss_ratio = None
        if (
            fused_m
            and sdpa_m
            and fused_m.get("loss_last") is not None
            and sdpa_m.get("loss_last") is not None
            and sdpa_m["loss_last"] != 0.0
        ):
            loss_ratio = fused_m["loss_last"] / sdpa_m["loss_last"]

        # A DRY_RUN outcome on EITHER run of EITHER bar leg (not just
        # the primary `LEGS` four) is the SAME benign "nothing ran
        # for real" case -- so `AB_DRY_RUN=1` reads
        # `N/A (dry-run)` regardless of which run a stub leg happens to be.
        any_dry_run = any(entries[leg]["outcome"] == "DRY_RUN" for leg in LEGS) or any(
            second_run_entries[leg]["outcome"] == "DRY_RUN" for leg in BAR_SECOND_RUN_LEGS.values()
        )
        torch_fits = entries["torch-sdpa"]["outcome"] == "OK"
        jammi_fused_fits = entries["jammi-fused"]["outcome"] == "OK"

        # Under `two_run_mode`, `jammi-fused-2` gets the SAME OOM/no-OOM clause
        # handling as `jammi-fused` (folded into `jammi_fused_fits`
        # itself, never a parallel mechanism), and `torch-sdpa-2` is
        # `torch_fits`'s own counterpart -- a bar leg that fit on ITS
        # first run but not its second is treated exactly as "did not
        # fit" for the whole config, the same conservative posture a
        # single OOM'd run already takes. `two_run_missing_leg_reason`
        # names the STRICTER failure this marker adds beyond that: a
        # second-run leg that is not merely FAIL/OOM (a real, attempted
        # measurement outcome) but genuinely `MISSING` (never attempted at
        # all, despite the marker's own promise that it would be) is an
        # INCOMPLETE SWEEP, not a legitimate "didn't fit" — surfaced as an
        # INVALID override below, never silently folded into the ordinary
        # "N/A (bar does not apply)"/"FAIL (OOM where torch fits)" prose
        # those two booleans alone would otherwise produce.
        two_run_missing_leg_reason = None
        if two_run_mode:
            if second_run_entries[torch_sdpa_2_leg]["outcome"] != "OK":
                torch_fits = False
            if second_run_entries[jammi_fused_2_leg]["outcome"] != "OK":
                jammi_fused_fits = False
            missing_legs = [
                leg
                for leg in BAR_SECOND_RUN_LEGS.values()
                if second_run_entries[leg]["outcome"] == "MISSING"
            ]
            if missing_legs:
                two_run_missing_leg_reason = (
                    f"two_run protocol marker present ({TWO_RUN_PROTOCOL_MARKER}) but "
                    f"{', '.join(missing_legs)} never ran (MISSING) -- the sweep is incomplete, "
                    "not merely a config that did not fit"
                )

        # The bar is "no OOM where torch fits" -- it binds ONLY when
        # torch-sdpa itself succeeded (BOTH runs of it, under
        # `two_run_mode`). If torch-sdpa didn't fit, there is no baseline
        # to hold jammi-fused to and the bar does not apply -- that is NOT
        # the same thing as jammi failing, and must not print as FAIL.
        if any_dry_run:
            verdict = "N/A (dry-run)"
        elif not torch_fits:
            if two_run_mode:
                verdict = (
                    f"N/A (torch-sdpa itself did not fit -- torch-sdpa={entries['torch-sdpa']['outcome']} "
                    f"torch-sdpa-2={second_run_entries[torch_sdpa_2_leg]['outcome']} -- bar does not apply)"
                )
            else:
                verdict = f"N/A (torch-sdpa itself did not fit: {entries['torch-sdpa']['outcome']} — bar does not apply)"
        elif not jammi_fused_fits:
            if two_run_mode:
                verdict = (
                    f"FAIL (OOM where torch fits: jammi-fused={entries['jammi-fused']['outcome']} "
                    f"jammi-fused-2={second_run_entries[jammi_fused_2_leg]['outcome']})"
                )
            else:
                verdict = f"FAIL (OOM where torch fits: jammi-fused {entries['jammi-fused']['outcome']})"
        elif bar_ratio is None or (two_run_mode and (ratio is None or pair2_ratio is None)):
            # The OUTCOME-only
            # `torch_fits`/`jammi_fused_fits` checks above cannot see a
            # DATA-quality gap — a leg reading `OK` whose own report still
            # carries a falsy/missing `triplets_per_s` (`bar_pair_ratio`
            # then reads `None` for THAT pair specifically). Under
            # `two_run_mode` the bar ratio consumes BOTH pair ratios (see
            # `bar_ratio_classification`'s own doc), so EITHER one reading
            # `None` here — even though `bar_ratio_classification` itself
            # would gracefully hand back the OTHER, still-valid pair
            # (correct for the single-run case, where there IS no other
            # pair to cross-check against) — must refuse the WHOLE config
            # instead of silently computing a verdict off exactly ONE of the
            # two required measurements. Without the marker `two_run_mode`
            # is `False`, so this condition reduces to `bar_ratio is None`
            # (both pairs None, or no second run at all).
            verdict = "FAIL (no ratio: triplets_per_s missing on an OK leg — investigate)"
        elif bar_indeterminate:
            # See `bar_ratio_classification`'s own doc: the two A,B,B,A pair
            # ratios straddle `pass_ratio`, or their spread exceeds the
            # combined estimate's own distance from it — never PASS/FAIL.
            verdict = f"{FINETUNE_AB_VERDICT_INDETERMINATE} ({bar_indeterminate_detail})"
        elif bar_ratio < pass_ratio:
            verdict = f"FAIL (ratio {bar_ratio:.3f} < {pass_ratio})"
        else:
            verdict = f"PASS (ratio {bar_ratio:.3f})"

        # A failed/errored `fused_proof` REPLACES the ratio-based verdict
        # rather than annotating it: a config whose jammi-fused leg silently
        # fell back to EAGER kernels (the exact regression `fused_proof`
        # exists to catch) must never print `PASS (ratio 0.95x) [WARN: ...]`
        # that `main()`'s exit code and a human skimming for "FAIL" both
        # miss. This is a DIFFERENT class of problem than the ratio-based
        # PASS/FAIL bar this crate deliberately RECORDS, never GATES, across
        # a heterogeneous fleet (see `finetune_ab.sh`'s own "script's own
        # exit code reflects whether the sweep RAN, not whether every
        # [config] passed" doctrine): a ratio below bar is a real,
        # machine-dependent PERFORMANCE observation; a failed fused_proof
        # means the MEASUREMENT ITSELF is not known to have exercised the
        # code path it claims to -- the ratio computed above could belong to
        # a DIFFERENT kernel composition entirely, making the PASS/FAIL
        # classification meaningless rather than merely unfavorable.
        # `main()` treats ANY `INVALID` verdict as a hard sweep failure
        # (non-zero exit) -- the one carve-out from the record-don't-gate
        # doctrine, because this is a correctness-of-measurement question,
        # not a perf-number question.
        if proof is False or isinstance(proof, str):
            reason = (
                f"errored: {proof}" if isinstance(proof, str)
                else "checked and FAILED — see fused_proof column for the classification"
            )
            verdict = (
                f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} (fused-dispatch proof {reason} — this leg's "
                f"PASS/FAIL classification cannot be trusted; the ratio-based verdict this would "
                f"otherwise have been is discarded, not merely annotated)"
            )

        # Same carve-out `proof is False`'s own INVALID branch above takes
        # from this crate's record-don't-gate doctrine: a leg-premise
        # mismatch (or absence) is a correctness-of-MEASUREMENT problem —
        # the ratio/loss numbers computed above may not even describe the
        # SAME configuration on both sides — so it REPLACES (never merely
        # annotates) whatever verdict was computed, same mechanism `main()`
        # already gates its own exit code on. Checked independently of, and
        # in addition to, the fused-dispatch proof above -- either alone can
        # invalidate this config's verdict.
        if leg_premise_violations_list:
            verdict = (
                f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} (leg premise mismatch: "
                f"{'; '.join(leg_premise_violations_list)} — the {jammi_premise_leg}/"
                f"{torch_premise_leg} legs of this config did not run under the same "
                "seed/batch/seq/dtype/dropout/lora premise; the ratio-based verdict this would "
                "otherwise have been is discarded, not merely annotated)"
            )

        # SECOND-RUN carve-outs — the SAME two "identity-completeness"
        # refusals the primary pair gets, applied to `jammi-fused-2`/
        # `torch-sdpa-2` (see the block above these were computed in for
        # the full rationale: the bar ratio consumes BOTH pair legs, so an
        # unproven or premise-mismatched SECOND run is exactly as
        # untrustworthy as an unproven or premise-mismatched FIRST one).
        # Checked independently of, and in addition to, the primary-pair
        # carve-outs above — any ONE of the four can invalidate this
        # config; `None`/`False`-shaped "second run absent or not
        # attempted" never does (see `proof2`'s and
        # `second_run_premise_violations_list`'s own docs).
        if proof2 is False or isinstance(proof2, str):
            reason2 = (
                f"errored: {proof2}" if isinstance(proof2, str)
                else "checked and FAILED — see fused_proof column for the classification"
            )
            verdict = (
                f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} (second-run ({jammi_fused_2_leg}) fused-dispatch "
                f"proof {reason2} — this leg's PASS/FAIL classification cannot be trusted; the "
                f"ratio-based verdict this would otherwise have been is discarded, not merely annotated)"
            )

        if second_run_premise_violations_list:
            verdict = (
                f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} (second-run leg premise mismatch: "
                f"{'; '.join(second_run_premise_violations_list)} — the {jammi_fused_2_leg}/"
                f"{torch_sdpa_2_leg} legs of this config did not run under the same "
                "seed/batch/seq/dtype/dropout/lora premise; the ratio-based verdict this would "
                "otherwise have been is discarded, not merely annotated)"
            )

        # Cross-run premise drift invalidates the config
        # exactly like a same-run mismatch does (see the computation's own
        # doc above for why this is a DIFFERENT check than either
        # same-run one).
        if cross_run_premise_violations_list:
            verdict = (
                f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} (cross-run leg premise mismatch: "
                f"{'; '.join(cross_run_premise_violations_list)} — the first and second runs of "
                "the bar pair did not run under the same seed/batch/seq/dtype/dropout/lora "
                "premise; the ratio-based verdict this would otherwise have been is discarded, "
                "not merely annotated)"
            )

        # The STRONGEST of the carve-outs above: a genuinely
        # INCOMPLETE sweep (the marker promised all four bar legs, one
        # never ran at all) is not even a "didn't fit"/"OOM" measurement,
        # so it REPLACES whatever verdict any of the checks above produced
        # (deliberately last, so it always wins when it fires).
        if two_run_missing_leg_reason is not None:
            verdict = f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} ({two_run_missing_leg_reason})"

        summary_rows.append((slug, ratio, pair2_ratio, bar_ratio, loss_ratio, verdict))
        merged["configs"][slug] = {
            "legs": {leg: {"outcome": entries[leg]["outcome"], "metrics": leg_metrics[leg]} for leg in LEGS},
            # Order-balanced A,B,B,A bar legs' own SECOND run, keyed by the
            # raw leg name (`"jammi-fused-2"`/`"torch-sdpa-2"`),
            # never folded into `"legs"` above (which stays keyed by `LEGS`
            # only, so every reader of `merged["configs"][slug]["legs"]` sees
            # exactly `LEGS`'s own four names).
            "bar_second_run_legs": {
                leg: {
                    "outcome": second_run_entries[leg]["outcome"],
                    "metrics": second_run_metrics[leg],
                    "provenance": second_run_provenance[leg],
                }
                for leg in BAR_SECOND_RUN_LEGS.values()
            },
            "jammi_fused_dispatch_proof": proof,
            # `jammi-fused-2`'s OWN `fused_proof` result — identity-
            # completeness (the bar ratio consumes both pair legs, so both
            # must carry the same positive-proof discipline). `None` when
            # the second run never ran/never produced a report (see
            # `proof2`'s own doc).
            "jammi_fused_dispatch_proof_second_run": proof2,
            "leg_premise_violations": leg_premise_violations_list,
            "leg_premise_checked_legs": (
                {"jammi": jammi_premise_leg, "torch": torch_premise_leg}
                if leg_premise_violations_list is not None
                else None
            ),
            "leg_premise_not_comparable": leg_premise_not_comparable,
            # SECOND-RUN leg-premise check — `None` (never an empty
            # "checked, clean" list) when the second run's own two legs
            # did not BOTH read `OK` (see `second_run_premise_violations_list`'s
            # own doc).
            "leg_premise_violations_second_run": second_run_premise_violations_list,
            "leg_premise_checked_legs_second_run": (
                {"jammi": jammi_fused_2_leg, "torch": torch_sdpa_2_leg}
                if second_run_premise_violations_list is not None
                else None
            ),
            # Cross-RUN premise (jammi-fused vs jammi-fused-2,
            # torch-sdpa vs torch-sdpa-2), independent of the two SAME-run
            # checks above. TRI-STATE (see `cross_run_premise_violations_list`'s
            # own doc, just above where this value is computed):
            #   * `None`  -- UNCHECKED: neither cross-run sub-comparison
            #     (jammi-fused vs jammi-fused-2, torch-sdpa vs
            #     torch-sdpa-2) had both its own legs read `OK` -- a
            #     single-run `raw_dir` (no second run at all) reads
            #     this, always.
            #   * `[]`    -- CHECKED, CLEAN: at least one sub-comparison
            #     ran and found no drift.
            #   * `[...]` -- CHECKED, VIOLATIONS: at least one
            #     sub-comparison ran and found drift (the strings name the
            #     field and the two legs' differing values).
            "leg_premise_violations_cross_run": cross_run_premise_violations_list,
            # `None` unless `two_run_protocol` (top-level) is `True`
            # AND at least one second-run bar leg genuinely never ran
            # (`MISSING`, not merely FAIL/OOM) — see `TWO_RUN_PROTOCOL_MARKER`'s
            # own doc.
            "two_run_missing_leg_reason": two_run_missing_leg_reason,
            "provenance": {"jammi": jammi_provenance, "torch": torch_provenance},
            "ratio_jammi_fused_over_torch_sdpa": ratio,
            # The A,B,B,A protocol's own two pair ratios + the MIN-of-two,
            # least-favourable-to-jammi bar ratio the verdict above is
            # actually classified against — see `bar_ratio_classification`'s
            # own doc. `pair2_ratio`/`bar_indeterminate*` are `None`/`False`
            # when the second run is unavailable (a single-run `raw_dir`), in
            # which case `bar_ratio == ratio_jammi_fused_over_torch_sdpa`
            # (the single-pair behaviour).
            "bar_pair_ratios": {
                "pair1_jammi_fused_over_torch_sdpa": ratio,
                "pair2_jammi_fused_2_over_torch_sdpa_2": pair2_ratio,
            },
            "bar_ratio_min_of_two_least_favourable_to_jammi": bar_ratio,
            "bar_ratio_indeterminate": bar_indeterminate,
            "loss_final_ratio_jammi_fused_over_torch_sdpa": loss_ratio,
            "loss_final_ratio_note": "same data, cost fixture -- NOT a quality result "
            "(see finetune_step.rs's module doc / torch_finetune_step.py's LOSS "
            "TRAJECTORY section: different attention-kernel arithmetic and "
            "reduction order between the two stacks makes a loss VALUE comparison "
            "meaningless even given identical synthetic input ids, unless "
            "torch_lora_init == 'jammi'). Printed so a divergence is visible, never "
            "gated. loss values carry only bf16's ULP (~0.00195 near 0.30) of real "
            "precision -- see BF16_LOSS_ULP_NEAR_0P3.",
            "verdict": verdict,
        }

    lines = [
        "# finetune A/B -- jammi eager vs jammi fused vs torch eager vs torch sdpa",
        f"# steps={steps} warmup={warmup} pass_bar={pass_ratio}x torch-sdpa triplets/s, no OOM where torch fits",
        f"# torch --lora-init={torch_lora_init}; jammi always uses its own ZerosB init -- loss_final_ratio "
        "is only a loss-TRAJECTORY-equivalence signal when torch_lora_init == 'jammi'.",
        "# loss-trajectory equivalence (jammi-fused vs jammi-eager, real trainer, >=5 seeds) is a SEPARATE check -- not measured here.",
        "# loss_first->loss_last and loss_final_ratio below: SAME DATA, COST FIXTURE -- NOT A QUALITY RESULT. "
        "Values are bf16-sourced (ULP ~0.00195 near 0.30) -- printed to 3 decimals, never gated.",
        # `<14` guarantees at least one space after every leg name this
        # module emits (`jammi-fused-2`/`torch-sdpa-2` are 13/12
        # characters); `<13` would run the `outcome` column's text directly
        # into a 13-char name.
        f"{'config':<16}{'leg':<14}{'outcome':<9}{'s/step_p50':<12}{'triplets/s':<12}"
        f"{'vram_delta(comparable)':<24}{'vram_absolute(torch only)':<27}{'fused_proof':<28}{'loss_first->last':<24}",
    ]
    for slug, leg, outcome, m, proof_val, err_tail in table_rows:
        p50 = fmt(m["s_per_step_p50"]) if m else "n/a"
        tps = fmt(m["triplets_per_s"]) if m else "n/a"
        vd = fmt_bytes(m["vram_delta_bytes"]) if m else "n/a"
        va = fmt_bytes(m["vram_absolute_bytes"]) if m else "n/a"
        if proof_val is None:
            proof_s = "n/a"
        elif isinstance(proof_val, str):
            proof_s = proof_val[:26]
        else:
            proof_s = "YES" if proof_val else "NO"
        loss_s = (
            "n/a"
            if not m or m.get("loss_first") is None or m.get("loss_last") is None
            else f"{fmt_loss(m['loss_first'])}->{fmt_loss(m['loss_last'])}"
        )
        lines.append(
            f"{slug:<16}{leg:<14}{outcome:<9}{p50:<12}{tps:<12}{vd:<24}{va:<27}{proof_s:<28}{loss_s:<24}"
        )
        if outcome not in ("OK", "DRY_RUN") and err_tail:
            last = err_tail.splitlines()[-1][:120] if err_tail.splitlines() else ""
            lines.append(f"    -> {last}")
        elif err_tail and "kernels_disabled_requested=" in err_tail:
            # The negative control: the jammi-eager row's own
            # kernels_disabled_requested/_fired lines (appended to
            # `err_tail` above even on an OK outcome) — printed IN FULL
            # (both lines, no truncation): a fixed, small, non-adversarial
            # op-key list, never user-controlled arbitrary-length text the
            # way a stderr tail is.
            for kd_line in err_tail.splitlines():
                if kd_line.startswith("kernels_disabled_"):
                    lines.append(f"    -> {kd_line}")
        elif err_tail and "[merge-stage]" in err_tail:
            last = err_tail.splitlines()[-1][:120]
            lines.append(f"    -> {last}")

    lines.append("")
    lines.append(
        f"{'config':<16}{'pair1(fused/sdpa)':<19}{'pair2(fused2/sdpa2)':<21}{'bar_ratio(min)':<16}"
        f"{'loss_final_ratio(fused/sdpa,NOT-quality)':<42}{'verdict':<60}"
    )
    for slug, ratio, pair2_ratio, bar_ratio, loss_ratio, verdict in summary_rows:
        ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
        pair2_s = "n/a" if pair2_ratio is None else f"{pair2_ratio:.3f}"
        bar_s = "n/a" if bar_ratio is None else f"{bar_ratio:.3f}"
        loss_ratio_s = "n/a" if loss_ratio is None else f"{loss_ratio:.4f}"
        lines.append(f"{slug:<16}{ratio_s:<19}{pair2_s:<21}{bar_s:<16}{loss_ratio_s:<42}{verdict:<60}")

    table = "\n".join(lines)
    return merged, table


# `build_report`'s own per-config `verdict` string's `"INVALID"` prefix (the
# fused-dispatch-proof-failed / leg-premise-mismatch carve-out from this
# crate's record-don't-gate doctrine -- see that function's own comment),
# produced there and consumed by `main()`'s own `.startswith(...)` check.
# Named ONCE here; both sites read this constant, never a re-typed literal.
FINETUNE_AB_VERDICT_INVALID_PREFIX = "INVALID"

# The order-balanced A,B,B,A bar legs' own THIRD classification (finetune_ab.sh's
# header, "ORDER-BALANCED BAR LEGS") -- deliberately NOT a `FINETUNE_AB_VERDICT_
# INVALID_PREFIX`-shaped carve-out: an INDETERMINATE config is not a
# correctness-of-MEASUREMENT problem (both bar-pair ratios are real,
# individually trustworthy numbers -- `fused_proof`/`leg_premise_violations`
# already gate that separately and still take precedence, see `build_report`'s
# own verdict-computation comment), it is a genuine "this repeat pair disagrees
# with itself too much, relative to how close the combined estimate sits to the
# bar, to trust a PASS or FAIL classification" recording -- `main()` does NOT
# gate its exit code on this string the way it does on `FINETUNE_AB_VERDICT_
# INVALID_PREFIX` (it does not start with "INVALID"), matching the
# record-don't-gate doctrine every OTHER ratio-based verdict here follows.
FINETUNE_AB_VERDICT_INDETERMINATE = "INDETERMINATE"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) < 5:
        print(
            "usage: ab_merge.py RAW_DIR OUT_DIR STEPS WARMUP PASS_RATIO [TORCH_LORA_INIT]",
            file=sys.stderr,
        )
        return 2
    raw_dir, out_dir, steps, warmup, pass_ratio_s = argv[:5]
    torch_lora_init = argv[5] if len(argv) > 5 else "peft"
    pass_ratio = float(pass_ratio_s)

    merged, table = build_report(raw_dir, steps, warmup, pass_ratio, torch_lora_init)
    if merged is None:
        print(f"finetune_ab: FAIL — no leg output found under {raw_dir}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "finetune_ab_report.json"), "w") as fh:
        json.dump(merged, fh, indent=2)
    print(table)
    with open(os.path.join(out_dir, "finetune_ab_table.txt"), "w") as fh:
        fh.write(table + "\n")

    # The ONE carve-out from
    # this crate's own record-don't-gate doctrine (see `finetune_ab.sh`'s
    # module doc and `build_report`'s own verdict-computation comment) --
    # an `INVALID` verdict (a failed/errored `fused_proof`) is a
    # correctness-of-MEASUREMENT problem, not a machine-dependent
    # performance number, so it is the one thing this sweep's own exit code
    # DOES gate on. An ordinary ratio-based `FAIL` row stays
    # record-only.
    invalid_slugs = [
        slug
        for slug, cfg in merged["configs"].items()
        if str(cfg.get("verdict", "")).startswith(FINETUNE_AB_VERDICT_INVALID_PREFIX)
    ]
    if invalid_slugs:
        print(
            f"finetune_ab: FAIL — {len(invalid_slugs)} config(s) have an INVALID verdict "
            f"(fused-dispatch proof failed or errored, see the table above): {invalid_slugs}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
