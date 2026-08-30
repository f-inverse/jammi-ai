#!/usr/bin/env python3
"""Merge + table stage for `ci/scripts/perf/finetune_ab.sh`'s #352 A/B sweep.

Extracted out of that script's own inline heredoc (B3: an inline heredoc has
ZERO automated coverage — `AB_DRY_RUN=1` only exercises the DRY_RUN arm,
never a real leg, so `fused_proof`/`dispatch_pairs`/the merge loop never saw
a real report shape in CI) into this importable module specifically so
`test_ab_merge.py` in this same directory can drive the REAL entry point
(`main`, exactly what `finetune_ab.sh` invokes) against fixture directories
shaped like `run_leg`'s own `.exit`/`.json`/`.stderr` output, never a
hand-rolled call to `fused_proof()` with literal tuples standing in for a
report.

Never imported by any Cargo crate, never a jammi-bench dependency — a
CI-adjacent script the sweep alone runs, same footing `finetune_ab.sh`
itself already has.

## Determinant table (round-4 audit fold-in on PR #372)

`leg_premise_violations` certifies that a config's jammi and torch legs ran
under the SAME premise before their ratio/loss numbers are treated as
comparable. Every field either producer's finetune-step report emits,
classified — mirrors `grad_oracle.rs`'s own determinant table
(`crates/jammi-bench/src/grad_oracle.rs`'s module doc) for the OTHER
jammi-vs-torch comparator this repo carries:

| field | class | jammi emit site | torch emit site |
|---|---|---|---|
| `seed` | identity | `report.rs:FinetuneStepTier::seed` field; `seed: params.seed,` (`finetune_step.rs:936`) | `"seed": args.seed,` (`torch_finetune_step.py:1260`) |
| `batch` | identity | `batch: params.batch,` (`finetune_step.rs:941`) | `"batch": args.batch,` (`torch_finetune_step.py:1278`) |
| `seq` | identity | `seq: params.seq,` (`finetune_step.rs:942`) | `"seq": args.seq,` (`torch_finetune_step.py:1279`) |
| `lora_rank` | identity | `lora_rank: params.lora_rank,` (`finetune_step.rs:943`) | `"lora_rank": args.lora_rank,` (`torch_finetune_step.py:1280`) |
| `lora_alpha` | identity — input was already threaded through `FinetuneStepParams::lora_alpha`, just never emitted before this round | `lora_alpha: params.lora_alpha,` (`finetune_step.rs:944`) | `"lora_alpha": args.lora_alpha,` (`torch_finetune_step.py:1253`) |
| `lora_dropout` | identity | `lora_dropout: params.lora_dropout` (`finetune_step.rs:945`) | `"lora_dropout": args.lora_dropout,` (`torch_finetune_step.py:1281`) |
| `margin` | identity, but jammi HARDCODES `0.3` (no `--margin` CLI flag — the call site's own literal, `let loss = triplet_loss(&a, &p, &n, 0.3)?;` (`finetune_step.rs:602`)) | `margin: 0.3,` (`finetune_step.rs:952`) | `"margin": args.margin,` (`torch_finetune_step.py:1262`) — `--margin` default `0.3` |
| `target_modules` | identity | `target_modules: params.target_modules.clone(),` (`finetune_step.rs:953`) | `"target_modules": [` (`torch_finetune_step.py:1284`) |
| `batched_forward` | identity | `batched_forward: params.batched_forward,` (`finetune_step.rs:954`) | `"batched_forward": args.batched_forward,` (`torch_finetune_step.py:1287`) |
| `backbone_dtype` | identity | `backbone_dtype: format!("{:?}", params.backbone_dtype)` (`finetune_step.rs:937`) | `"backbone_dtype": args.dtype,` (`torch_finetune_step.py:1269`) |
| `steps_measured` | identity — the reachable divergence this table used to miss entirely: two legs measured at a DIFFERENT step count (e.g. a mismatched `--steps`/`--warmup` override) still merged to a "clean" ratio before this field was compared | `steps_measured: times.len(),` (`finetune_step.rs:972`) | `"steps_measured": len(times),` (`torch_finetune_step.py:1320`) |
| `checkpoint_config_sha256` | identity — same base-checkpoint CONTENT identity `grad_oracle.rs`'s tier already carries, added to THIS tier too | `let (checkpoint_config_sha256, _config_len) =` (`finetune_step.rs:717`), via the SHARED streaming `pub(crate) fn sha256_and_len` (`finetune_step.rs:1091`) | `checkpoint_identity_fields = checkpoint_identity(args.model_dir)` (`torch_finetune_step.py:1130`) |
| `checkpoint_weights_sha256` | identity | `let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =` (`finetune_step.rs:719`) | `"checkpoint_weights_sha256": weights_sha256,` (`torch_finetune_step.py:685`) |
| `checkpoint_weights_size_bytes` | identity | `checkpoint_weights_size_bytes) =` (`finetune_step.rs:719`) — same call as the row above, its second return value | `"checkpoint_weights_size_bytes": weights_len,` (`torch_finetune_step.py:686`) |
| `row_lengths` | identity (K7 audit: 17 -> 18 fields) — the per-row token lengths the padded fixture fed the encoder, requested or the dense-leg default `[seq; batch]`; NO canonicalizer, per-row order is load-bearing (`[3, 6]` != `[6, 3]`), compared directly, never hashed | `row_lengths: params` (`finetune_step.rs:966`), dense fallback `vec![params.seq; params.batch]` (`finetune_step.rs:969`) | `"row_lengths": args.row_lengths` (`torch_finetune_step.py:1295`), dense fallback `else [args.seq] * args.batch,` (`torch_finetune_step.py:1297`) |
| `max_grad_norm` | identity (PR #381 audit B1) — `null` (clip OFF) or the positive finite bound the PRODUCTION `clip_gradients` ran with; a clip-on leg and a clip-off leg compute a different step. `null` is a VALUE for this field (`identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`), never folded into MISSING | `max_grad_norm: params.max_grad_norm,` (`finetune_step.rs`'s tier literal) | `"max_grad_norm": args.max_grad_norm,` in the `finetune_step` block (`torch_finetune_step.py`) |
| `attention_arm` | identity (PR #381 audit B1 class probe) — the attention REFERENCE CLASS the leg was ASKED to run, `"eager"` or `"fused"`; jammi's is the operator's `JAMMI_KERNELS_DISABLE` request (an attention base in `kernels_disabled_requested` ⇒ eager), NEVER the counters (a by-design domain decline is a measurement, not a premise) — see `identity_fields.FINETUNE_IDENTITY_FIELDS`'s own entry | `attention_arm: attention_arm(&kernels_disabled_requested).to_string()` (`finetune_step.rs`'s tier literal) | `"attention_arm": attention_arm_of(resolved_attn_implementation)` in the `finetune_step` block (`torch_finetune_step.py`) |
| `warmup` | identity (PR #381 re-audit) — changes what `clip_invocations` counts (pre-step + warmup + measured) | `warmup: params.warmup,` (`finetune_step.rs`'s tier literal) | `"warmup": args.warmup,` in the `args` block (`torch_finetune_step.py`) — an `_TORCH_ARGS_LEVEL_FIELDS` member |
| `clip_invocations` | measurement (PR #381 audit B2) — the COUNTED number of times the production clip ran this process (pre-step + warmup + measured, every `step_once`), the fact behind a clip-on row rather than a log line; recorded in `leg_provenance` per leg AND cross-checked against `max_grad_norm` by `clip_fact_violations` (clip requested ⇒ `> 0`; not requested ⇒ `== 0`) | `clip_invocations:` (`finetune_step.rs`'s tier literal, a `CLIP_INVOCATIONS` before/after delta) | `"clip_invocations": clip_counter["clip_invocations"]` (`torch_finetune_step.py`) |
| `attn_requested` / `attn_implementation` | provenance — the RAW torch attention string (`--attn` as requested, and what HF resolved it to); the CLASS it implies is compared via `attention_arm` above, the raw string itself is recorded in `leg_provenance`, never compared (see `grad_oracle.rs`'s own table for the fuller rationale) | n/a | `"attn_requested": args.attn,` (`torch_finetune_step.py:1258`) in the `args` block; `attn_implementation` is the sibling `"attn_implementation": resolved_attn_implementation` field further down in the `finetune_step` block |
| `kernels_disabled_requested` / `kernels_disabled_fired` | provenance (K-aux, landed on `main` at `c0f0e98`) — torch has no equivalent env var; recorded in `leg_provenance`, never compared | `let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();` (`finetune_step.rs:921`) | n/a |
| `ln`/`rope`/`softmax`/`geglu`/`lora_epilogue`/`lora_linear`/`attention_block` `_fused_dispatches`/`_eager_dispatches` (14 fields) | measurement — this IS the fused-dispatch proof `fused_proof`/`dispatch_pairs` gate on, and `leg_provenance` additionally records the raw counters per config | `finetune_step.rs`'s own `*_fused_dispatches`/`*_eager_dispatches` fields | n/a |
| `attention_block_flash_fused_dispatches` / `attention_block_flash_declined_dispatches` (P6 Stage B FA2 fold-in, since merged to `main` — `pub attention_block_flash_fused_dispatches: u64,` (`report.rs:1670`), `pub attention_block_flash_declined_dispatches: u64,` (`report.rs:1678`)) | measurement — a CASCADE-shaped pair (`CASCADE_BASES`): no `_eager_dispatches` sibling, its fallback counter is named `_declined_dispatches` instead; absorbs `attention_block` (`ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`), which in turn already absorbs `rope`/`softmax` — one chain, not a second mechanism. Every current `finetune-step`/`finetune-run` leg carries both keys. unit-63 round-4 audit advisory (A1), CURRENT TRUTH (this row's own earlier "a pre-fold-in fixture predating this pair still works" claim was false on the `adamw` axis for the two committed P6 fixtures, `fixtures/p6_fa2_dense_raw_runs/s128_flash_{on,off}_1.json`): `adamw` is a `REQUIRED_PAIRS` member, and `REQUIRED_PAIRS`'s own absent-base rule is a hard fail for the WHOLE leg, not merely for the missing base's own classification — a pre-fold-in fixture predating the multi-tensor AdamW commit (carrying no `adamw_{fused,eager}_dispatches` keys at all, the SAME schema gap the cascade pair itself has for an even older fixture) therefore fails `fused_proof` outright, INVALID, never "still works". `CascadePairFixtureTests::test_real_flash_{on,off}_fixture_no_longer_keyerrors_but_predates_adamw` (`test_ab_merge.py`) are the honest, pinned record of this — both fixtures correctly read INVALID, not "clean, this base merely undiscovered" | `report.rs`'s `FinetuneStepTier::attention_block_flash_fused_dispatches`/`::attention_block_flash_declined_dispatches` fields | n/a |
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
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identity_fields import (  # noqa: E402
    FINETUNE_IDENTITY_FIELDS,
    FINETUNE_NULL_IS_A_VALUE_FIELDS,
    FINETUNE_RUN_IDENTITY_FIELDS,
    FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS,
    canonicalize_identity_field,
)

LEGS = ["jammi-eager", "jammi-fused", "torch-eager", "torch-sdpa"]

# ORDER-BALANCED BAR LEGS (finetune_ab.sh's own A,B,B,A protocol — see that
# script's header's "ORDER-BALANCED BAR LEGS" section): the two legs the
# #352 throughput bar actually gates on, `jammi-fused` (A) and `torch-sdpa`
# (B), run TWICE per config in the fixed order A,B,B,A — the SAME
# drift-cancellation shape `gpu_inference_ab.py`'s own `LEG_ORDER`/
# `ADJACENT_PAIRS` document ("What actually cancels, and what does not").
# `jammi-fused`/`torch-sdpa` (already in `LEGS` above) ARE the first ("1")
# run of each; `BAR_SECOND_RUN_LEGS` names the SECOND ("2") run's own raw
# leg files — additive, deliberately NOT folded into `LEGS` itself: every
# OTHER function keyed off `LEGS` (`leg_premise_violations`,
# `leg_provenance`, `fused_proof`'s own caller, the primary per-leg table
# rows) stays byte-for-byte unchanged, so a `raw_dir` from BEFORE this
# fold-in (carrying only the original four legs) still merges exactly as
# it always did — `load_leg` reads `MISSING` for an absent
# `<slug>__jammi-fused-2`/`<slug>__torch-sdpa-2` pair, which
# `bar_pair_ratio`/`build_report` below already treat as "second run not
# available", falling back to the single-pair ratio this module has always
# computed. `BAR_SECOND_RUN_LEGS` is keyed by the FIRST run's own leg name
# (the natural "which pair is this the repeat of" lookup both call sites
# below need).
BAR_SECOND_RUN_LEGS = {"jammi-fused": "jammi-fused-2", "torch-sdpa": "torch-sdpa-2"}

# F2 (adversarial audit — "make the header's promise true"): the file
# `finetune_ab.sh` `touch`es under `raw_dir`, BEFORE any leg runs, on
# EVERY invocation (that script always runs the full A,B,B,A protocol —
# see its own header). The SAME filename, read here. Presence means "this
# raw_dir's operator promised all four bar legs" — a MISSING/DRY_RUN
# second-run leg under this marker is therefore an INCOMPLETE SWEEP
# (INVALID, a named reason), never silently degraded to the single-pair
# estimator the way an absent marker (a genuinely legacy `raw_dir`,
# predating this fold-in, or one hand-built without it) still is. Kept
# unprefixed (no leading `.`) so `ls`/a human browsing `raw_dir` sees it;
# `config_slugs()` below never matches it (it carries no `.exit` suffix
# and no `__` separator).
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
# Generic leg-premise-refusal core (unit-62 E6) — `leg_identity_fields`/
# `leg_premise_violations` below are the finetune-step-SPECIFIC callers
# (report shape, torch args-level field placement, `_MISSING`-folding
# doctrine); `generic_leg_identity_fields`/`generic_leg_premise_violations`
# are the SAME two-step shape (fold ABSENT-or-null into `_MISSING`, then
# compare after `canonicalize_identity_field`) factored out over an
# arbitrary `fields` tuple and two ALREADY-FLATTENED `{field: value}` dicts,
# so a NEW producer (`encode_ab.sh`, this unit) reuses the identical
# premise-refusal logic against `identity_fields.ENCODE_IDENTITY_FIELDS`
# rather than hand-rolling a second, independently-drifting comparator.
# `leg_identity_fields`/`leg_premise_violations` are UNCHANGED by this
# addition (no weakening of the finetune-step check either function backs) —
# this is purely additive shared machinery a new caller can build on.
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


# The premise-identity check (fold-in, this round: the adjacent probe found
# this module carried NO premise-identity check at all -- identity was
# "by construction" of `finetune_ab.sh`'s own matched CLI flags across its
# `run_jammi_leg`/`run_torch_leg` call sites only, i.e. an assumption, never
# a checked record in the merged artifact). Shares
# `identity_fields.canonicalize_identity_field` with
# `compare_grad_oracle.py`'s OWN identity check (one definition, per the
# lead's own fold-in instruction) rather than a second, independently-
# drifting copy of the SAME `backbone_dtype`/`target_modules` representational
# gaps.
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
# `lora_alpha`/`margin` (round-4 audit fold-in on PR #372): torch has always
# emitted both under `args`; jammi's `FinetuneStepTier` did not carry either
# field at all until this round (`lora_alpha` was already a CLI input,
# `FinetuneStepParams::lora_alpha`, just never emitted on the report;
# `margin` has no jammi CLI flag at all -- this tier hardcodes `0.3`,
# matching torch's own `--margin` default, see `FinetuneStepTier::margin`'s
# own field doc).
#
# `steps_measured` (round-4 audit fold-in): reachable divergence this table
# used to miss entirely -- two legs run at `--steps 20`/`--warmup 5` vs
# `--steps 5`/`--warmup 5` (e.g. `finetune_ab.sh` invoked with mismatched
# per-leg overrides, or a leg re-run by hand) still merge to a "clean" ratio
# and PASS verdict; `steps_measured` is recorded on BOTH sides already
# (`FinetuneStepTier::steps_measured`, `torch_finetune_step.py`'s own
# `"steps_measured": len(times)`) and is a genuine per-run fact, not
# metadata `main()`'s `steps`/`warmup` CLI args alone can stand in for
# (those describe what THIS MERGE INVOCATION was told, not what either leg
# actually measured).
#
# `checkpoint_config_sha256`/`checkpoint_weights_sha256`/
# `checkpoint_weights_size_bytes` (round-4 audit fold-in): the SAME
# base-checkpoint content-identity fields `grad_oracle.rs`'s determinant
# table already covers, now also on the finetune-step tier (both
# producers), replacing an implicit "the operator passed the same
# --model-dir path" assumption with a checked record.
#
# `max_grad_norm` / `attention_arm` (PR #381 audit B1 + the lead's class
# probe): the tuple this module used to hand-keep here lacked BOTH — see
# `identity_fields.FINETUNE_IDENTITY_FIELDS`'s own per-field entries. The
# declaration now lives ONLY there (imported above); this module never
# redeclares the set, so a field added to the shared tuple is refused here
# generically the moment either producer fails to emit it.

# torch keeps these fields one level UP from the `finetune_step` sub-block
# (`report["args"][field]`) rather than inside it — see
# `FINETUNE_IDENTITY_FIELDS`'s own doc. Every OTHER field (including the 3
# new checkpoint-identity ones, which torch now emits directly inside the
# `finetune_step` block, matching jammi's own placement) lives at the SAME
# level `finetune_block` already reads for both producers.
_TORCH_ARGS_LEVEL_FIELDS = frozenset({"seed", "lora_alpha", "margin", "warmup"})

# B2 — the DECLARED classification `fused_proof` checks a dispatch-counter
# pair against, replacing the old blanket "(fused, eager) == (0, 0) is
# always fine, for every pair" rule (which made a report where every real
# fused site read (0, 0) and only ONE unrelated pair read a positive fused
# count print `fused_proof YES` — a net loss of detection versus the
# pre-generalization check, which positively required ln/rope/softmax each
# `fused > 0`).
#
# F5 (PR #372 audit round): the FIRST generalization of this table fixed
# the "(0, 0) silently excluded" bug for `ln` only, and in doing so
# introduced the SAME bug with the polarity flipped for every OTHER base:
# `rope`/`softmax` being ENTIRELY ABSENT from a report's schema (a real
# field renamed/deleted/feature-gated-off regression, not merely reading
# `(0, 0)`) was silently `continue`d past rather than failed, and any base
# not named in ANY of the three sets below (`geglu` — required by nothing,
# despite `finetune_ab.sh`'s own header claiming otherwise) was never
# checked AT ALL, present or absent, zero or nonzero. `fused_proof([('ln',
# 9, 0)])` — EVERY other pair entirely missing from the report — used to
# return `True`. The invariant this table now enforces: EVERY base that
# `dispatch_pairs` discovers in a real report must be in EXACTLY ONE of the
# three sets below (`ALL_BASES` is their union); a discovered base outside
# `ALL_BASES` is a schema-drift ERROR (`dispatch_pairs` raises, same B6
# per-leg-loud/whole-merge-safe handling `build_report` already gives a
# solo counter), never a silent exemption. Within each set, ABSENCE from
# the report is now ALSO a hard fail for every member (not just the
# `REQUIRED_PAIRS` ones) — a classified base that vanishes from the schema
# is exactly the regression this proof exists to catch.
#
#   * REQUIRED_PAIRS — no fused block in this crate absorbs these; each
#     MUST be PRESENT and show its own `fused > 0` (and, like every pair,
#     `eager == 0`).
#       - `ln`: dispatches inside every layer's own norm call, never folded
#         into a whole-attention or whole-MLP kernel, and
#         `finetune_step.rs`'s own counter-delta test already asserts its
#         (fused+eager) total is nonzero on every run.
#       - `geglu`: same reasoning as `ln` — `ModernBertMlp::forward`'s
#         training arm calls `geglu_apply_training` unconditionally for
#         every layer's MLP (see that function's own doc); its admission
#         domain (F32/BF16, contiguous, nonzero-even last dim) holds for
#         every real ModernBERT MLP shape, so nothing legitimately
#         absorbs or exempts it the way `attention_block` absorbs
#         `rope`/`softmax`. This closes F4/F5's own reproduction (a
#         "deleted/feature-gated-off fused MLP" reading `geglu = (0, 0)`
#         used to still print `fused_proof YES`).
#       - `adamw` (unit-63 round-3 audit, block 1): `AdamW::step`'s
#         per-`Var` dispatch to `adamw_step_fused_t`
#         (`report.rs`'s `adamw_fused_dispatches` field doc,
#         `adamw_fused_dispatches: adamw_dispatch_after` (`finetune_step.rs:1018`))
#         — same reasoning class as `ln`/
#         `geglu` again: its admission domain is device/dtype/contiguity/
#         shape agreement across `theta`/`m`/`v`/`grad`, which holds
#         unconditionally for every real training run (all four are the
#         SAME `Var`'s own state, so they always already agree on device,
#         dtype, contiguity, and shape) — nothing legitimately absorbs or
#         exempts it, and no fused block in this crate folds it into a
#         wider kernel the way `attention_block` folds `rope`/`softmax`.
#         `dispatch_pairs` raised `KeyError('adamw')` on EVERY real leg of
#         both the `finetune-step` and `finetune-run` tiers before this
#         fix (`adamw` was never added to `ALL_BASES` when the multi-
#         tensor AdamW commit landed) — reproduced against the committed
#         real artifact
#         `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4-raw-runs/
#         a100b/b8_s512_fused.r2.json.raw` (`TestRealAdamwArtifactFixtures`
#         in `test_ab_merge.py`).
#       - `attention_block` used to live in this set too. P6 Stage B FA2
#         fold-in (below): moved OUT, since a third training-attention arm
#         now gives it a legitimate absorption path of its own, the same
#         shape `rope`/`softmax` already had one level down —
#         see `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`.
#   * ABSORBABLE_BY_ATTENTION_BLOCK_FLASH (P6 Stage B FA2 fold-in — a
#     docs-ci co-sign of `origin/perf/p6-fa2-dense` @ `5886c6b`, NOT on
#     `main` as of this table) — `attention_block` MUST be PRESENT; may
#     read `(0, 0)` IFF `attention_block_flash`'s OWN `fused` count is `> 0`
#     THIS run: when the FlashAttention-2 dense cascade fires for a layer,
#     that layer's `attention_block` `admit` call is never reached at all
#     (an early return — see `report.rs`'s
#     `attention_block_flash_fused_dispatches` field doc on that branch),
#     the exact same "one call site, mutually exclusive arms" shape
#     `rope`/`softmax`'s own absorption below already documents one level
#     down. `by_base.get("attention_block_flash", (0, 0))[0]` defaults to
#     `0` when the key is entirely ABSENT from the report (every report
#     `main`'s own binary produces today) — so on such a report this
#     absorption condition is trivially never satisfied and
#     `attention_block` falls back to needing its OWN `fused > 0`, i.e.
#     EXACTLY the `REQUIRED_PAIRS` behaviour it used to get directly,
#     unchanged.
#       - A checkpoint whose `head_dim != 64` legitimately falls back to
#         eager here (`report.rs`'s `attention_block_eager_dispatches`
#         field doc) — that is ALREADY caught by rule 1 below (an
#         unaccounted-for fallback anywhere is a hard fail), so requiring
#         `fused > 0` (absent flash absorption) here for the cases rule 1
#         does not already reject adds detection without changing
#         behaviour on that documented domain-refusal case.
#   * ABSORBABLE_BY_ATTENTION_BLOCK — `rope`/`softmax` MUST be PRESENT; may
#     read `(0, 0)` IFF `attention_block`'s OWN `fused` count is `> 0`, OR
#     (P6 Stage B FA2 fold-in, extending this SAME chain rather than a
#     parallel mechanism) `attention_block_flash`'s OWN `fused` count is
#     `> 0`, this run: `ModernBertAttention::forward_training_attention`'s
#     BLOCK-fused arm is the whole RoPE+QKᵀ+mask+softmax+PV chain as one op
#     and never calls `rope_apply`/`softmax_apply_training` at all (see
#     that method's own doc), so their independent admission call sites are
#     simply never reached — and the FLASH arm is a further whole-attention
#     alternative to that SAME call site, so it never reaches
#     `rope_apply`/`softmax_apply_training` either, for the identical
#     reason one level up the chain. When NEITHER whole-attention arm goes
#     fused (the eager attention composition ran instead), that
#     composition DOES call `rope_apply`/`softmax_apply_training` — each
#     independently admission-gated — so they must clear the same
#     `fused > 0` bar a required pair does.
#   * LORA_SITE_EXCLUSIVE_GROUP — `lora_epilogue`/`lora_linear` MUST both be
#     PRESENT, and are genuinely exclusive with EACH OTHER, not with a
#     third pair: every training-arm LoRA-adapted forward routes through
#     EXACTLY ONE of these two call sites
#     (`jammi_lora::lora_linear::lora_linear_fused_counters`'s own doc —
#     today `lora_epilogue` is PERMANENTLY `(0, 0)`, superseded by the
#     fused whole-site kernel `lora_linear` now reports). So only the
#     GROUP's sum needs a `fused > 0` proof, never each member alone.
#   * CASCADE_BASES (P6 Stage B FA2 fold-in) — see that set's own doc below.
#     Its one member today, `attention_block_flash`, is deliberately NOT a
#     member of any of the three "must be present" sets above: it is a
#     genuinely OPTIONAL arm (absent entirely on every report `main`'s own
#     binary produces today, and even on a build that DOES carry the field,
#     nothing about this crate requires that build to have compiled/used
#     it) — its ONLY role in `ALL_BASES` is to be a recognized (not
#     schema-drift) base when `dispatch_pairs` discovers it in a report
#     that DOES carry it, so its own `fused`/`declined` counts are
#     available for rule 1 (the declined-count hard-fail-unless-requested
#     check) and for `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`/
#     `ABSORBABLE_BY_ATTENTION_BLOCK`'s absorption conditions above.
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
# `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` documents). Reproduction (docs-ci
# co-sign of `origin/perf/p6-fa2-dense` @ `5886c6b`): running `main`'s own
# `dispatch_pairs` (before this fix) against that branch's own committed
# `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/
# s128_flash_on_1.json` raises `KeyError` looking for a nonexistent
# `attention_block_flash_eager_dispatches` sibling, and `build_report`'s
# per-leg `try`/`except` (see that function's own doc) would then mark
# EVERY leg of EVERY config `INVALID` — silently voiding the flash-vs-block
# A/B this tool exists to judge, not merely failing loudly on the one
# genuinely new pair kind.
#
# STALE PREMISE, corrected (unit-63 round-3 audit advisory): the P6 Stage B
# FA2 fold-in landed on `main`.
#   * `pub attention_block_flash_fused_dispatches: u64,` (`report.rs:1670`)
#   * `pub attention_block_flash_declined_dispatches: u64,` (`report.rs:1678`)
#   * `attention_block_flash_fused_dispatches: attention_block_flash_dispatch_after` (`finetune_step.rs:1031`)
#   * `attention_block_flash_declined_dispatches: attention_block_flash_dispatch_after` (`finetune_step.rs:1034`)
# All four populate on every real `finetune-step` (and, via
# `FinetuneRunTier`'s own mirror of the same pair, `finetune-run`) leg —
# this is no longer a "not yet on main" base a fresh
# report might lack; every current leg carries it. `CASCADE_BASES` stays
# `{"attention_block_flash"}` for the reason below (a genuinely OPTIONAL
# proof participant, not a doubt about whether the FIELD is present), and a
# pre-fold-in fixture predating this pair (still absent the two keys
# entirely) remains equally well-handled — `dispatch_pairs` simply never
# discovers this base on such a report, same as always.
CASCADE_BASES = frozenset({"attention_block_flash"})

ALL_BASES = (
    REQUIRED_PAIRS
    | ABSORBABLE_BY_ATTENTION_BLOCK
    | ABSORBABLE_BY_ATTENTION_BLOCK_FLASH
    | LORA_SITE_EXCLUSIVE_GROUP
    | CASCADE_BASES
)
# Unit-63 round-16 audit advisory 1: an explicit `if`/`raise`, never a bare
# `assert` -- `assert` is stripped entirely under `python -O`, which would
# silently disable this load-bearing pairwise-disjointness guard (every
# other classification in this module assumes exactly-one-class-per-base)
# in exactly the deployment shape that removes the safety net without
# removing the code path it protects.
if (
    len(REQUIRED_PAIRS)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK_FLASH)
    + len(LORA_SITE_EXCLUSIVE_GROUP)
    + len(CASCADE_BASES)
    != len(ALL_BASES)
):
    raise AssertionError(
        "REQUIRED_PAIRS / ABSORBABLE_BY_ATTENTION_BLOCK / "
        "ABSORBABLE_BY_ATTENTION_BLOCK_FLASH / LORA_SITE_EXCLUSIVE_GROUP / "
        "CASCADE_BASES must be pairwise disjoint -- every base gets exactly ONE class"
    )

# B5 — bf16's ULP near a loss value around 0.30: 7 explicit mantissa bits,
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
    "cannot verify this premise determinant" state — round-4 audit fold-in
    on PR #372: an earlier draft of this function's own doc claimed a
    present-but-`None` value was something "no real producer ever emits,
    but a fixture could" — that is FALSE for the SAME reason
    `compare_grad_oracle.py`'s own fix this round applies: `serde_json`
    serializes a NaN/inf `f64` as JSON `null`, so a NaN `lora_alpha` on
    jammi's side is reachable, not hypothetical).

    EXCEPT for `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS` members
    (`max_grad_norm`): there a present `null` IS the premise ("clip OFF") —
    both producers refuse a non-finite value before running, so NaN can
    never reach the report and `null` has exactly one meaning. An ABSENT
    key is still `_MISSING` for those fields (a producer built before the
    field existed cannot state its premise).
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
    """Per-leg COUNTED-FACT check for the clip row (PR #381 audit B2): a
    leg's `max_grad_norm` states what was REQUESTED; its `clip_invocations`
    is what the producer COUNTED the production clip actually doing
    (jammi: a `CLIP_INVOCATIONS` before/after delta around `run()`'s
    pre-step + loop; torch: `clip_counter` bumped at every
    `clip_grad_norm_` call). The two must agree in kind — requested ⇒
    counted `> 0`; not requested ⇒ counted `== 0` — or the row is
    claiming a step it did not run. A leg that carries neither key (a
    producer built before both existed) is left to `leg_premise_violations`'
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
    (including, P6 Stage B FA2 fold-in, a `CASCADE_BASES` member's
    `_declined_dispatches` counter — `jammi_dispatch_counters` keys off
    EITHER fallback-counter suffix, not just `_eager_dispatches`, so a
    cascade pair's raw counts are recorded exactly like every other pair's),
    (K-aux, landed on `main` at `c0f0e98`) jammi's resolved
    `JAMMI_KERNELS_DISABLE` state (`kernels_disabled_requested`/
    `kernels_disabled_fired`), and (P6 Stage B FA2 fold-in) `flash_compiled`
    — deliberately NOT `FINETUNE_IDENTITY_FIELDS` members (torch has no
    equivalent env var or build-capability flag to compare against),
    recorded here purely so a human reading the merged JSON can see which
    arm jammi's OWN leg measured. `None` for the fields the OTHER producer
    has no equivalent for (never fabricated).
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
            # PR #381 audit B2: the counted fact behind the clip row, next
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
    other (ordinary) pair — unchanged from before the cascade-pair fold-in.
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
    `_declined_dispatches` (P6 Stage B FA2 fold-in — see that set's own
    doc). The returned tuple's third element is that fallback count
    regardless of which key produced it — `fused_proof`'s rule 1 (below)
    treats BOTH shapes uniformly (a real, non-deliberate fallback anywhere
    is a hard fail), so `dispatch_pairs` itself does not need to distinguish
    them past this point.

    B6 SCHEMA STRICTNESS: this function stays LOUD (raises `KeyError`) on a
    solo counter — a fused key with no fallback sibling is a genuine schema
    bug (a struct field added without its pair), never a config this script
    should silently skip. F5 extends the SAME loudness to a base
    `fused_proof`'s classification tables (`REQUIRED_PAIRS` /
    `ABSORBABLE_BY_ATTENTION_BLOCK` / `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` /
    `LORA_SITE_EXCLUSIVE_GROUP` / `CASCADE_BASES`, whose union is
    `ALL_BASES`) do not know about: a NEW fused kernel landing in
    `finetune_step.rs` without this module's classification tables being
    updated in lockstep is exactly the same class of schema drift as a
    solo counter — `fused_proof` would otherwise silently never require
    anything of it (the F5 bug this closes). `metrics()`'s two `.get()`
    reads for `loss_first`/`loss_last` are the OPPOSITE choice,
    deliberately: those two fields are optional/best-effort table
    decoration (present since the loss-trajectory unit landed, absent on
    an older report schema, and absence there changes nothing this proof
    depends on), while a dispatch pair — and its classification — is
    STRUCTURAL to `fused_proof`'s entire claim. The two windows are
    intentionally different; what changed in this revision is only WHERE
    this exception is caught — see `build_report`'s per-leg `try`/`except`,
    which stops one bad leg's solo-counter (or now, unclassified-base)
    `KeyError` from discarding the merged table for every other config
    (previously this raise propagated all the way to the top-level script
    and aborted the entire merge).

    Reproduction of the bug this fixes (P6 Stage B FA2, docs-ci co-sign of
    `origin/perf/p6-fa2-dense` @ `5886c6b`): BEFORE `_fallback_key`/
    `CASCADE_BASES` existed, this function always looked for
    `<base>_eager_dispatches` — a report carrying
    `attention_block_flash_fused_dispatches` (that branch's own
    `report.rs`'s `FinetuneStepTier`, not yet on `main`) has no
    `attention_block_flash_eager_dispatches` key at all (that arm's
    fallback counter is named `_declined_dispatches` instead), so this
    raised `KeyError` on every real leg from that branch's committed
    `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/*.json`
    fixtures.
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
                "CASCADE_BASES tables being updated to cover it. This is a "
                "schema-drift bug, not a base this script should silently leave unchecked "
                "(see F5's own fix note on the module-level classification tables)."
            )
        pairs.append((base, fs[key], fs[fallback_key]))
    return pairs


def metrics(entry, leg):
    """Extract this leg's table/proof metrics from its raw report. Returns
    `None` when the leg itself did not produce a usable report (see
    `load_leg`); raises (never silently drops a field) when the report WAS
    produced but a STRUCTURAL piece — a dispatch pair — is malformed (see
    `dispatch_pairs`'s own doc for why that is the loud half of this
    module's B6 schema-strictness split).
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
        # P6 Stage B FA2 fold-in: `fused_proof`'s own flash-disable-
        # consistency check (see that function's doc) reads these off `m`
        # rather than the raw `fs` a second time, keeping `fused_proof`'s
        # existing `m`-only signature. `.get()` (never `[...]`): both keys
        # are entirely ABSENT on every report `main`'s own binary produces
        # today, and this must not raise for that case.
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
    `CASCADE_BASES` (union `ALL_BASES`) doc for the classification this
    checks each pair against. Returns `True`/`False`/`None` (no
    `dispatch_pairs` at all — not a jammi leg, or the leg itself did not
    run) or a `str` (P6 Stage B FA2 fold-in — the flash-disable-consistency
    check below errored; `build_report` treats a `str` return the same as
    `False`, see its own `proof is False or isinstance(proof, str)` branch).
    Raises (via `dispatch_pairs`, which `metrics()` already calls before
    this function ever sees `m` — see that function's own doc) if
    `m["dispatch_pairs"]` would ever contain a base outside `ALL_BASES`;
    `fused_proof` itself never receives an unclassified base to begin with.

    Rules, in order — EVERY base in `ALL_BASES` (not just `REQUIRED_PAIRS`)
    must be PRESENT in this report's pairs; absence is a hard fail for
    every classified base, never a silently-granted exemption (F5: the
    pre-fix code granted this exemption to every base except `ln`), EXCEPT
    `CASCADE_BASES` members, which are genuinely OPTIONAL (see that set's
    own doc):
      0. (P6 Stage B FA2 fold-in) `flash_compiled is False` AND
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
         a hard fail exactly like an ordinary silent eager fallback always
         has (`report.rs`'s own `attention_block_flash_declined_dispatches`
         field doc, contract v5 §3.8: "`declined > 0` on any bench leg ->
         INVALID"). Unit-63 round-3 audit, coordinator correction: an
         EARLIER draft of this rule additionally exempted
         `flash_compiled is False` (a build-capability-miss carve-out) —
         reverted. `fused_proof` is shared by `finetune-step`'s own
         campaigns; a build fact that makes a WHOLE campaign's premise
         null (CONTRACT 63 Frame pre-registers the flash cascade as the
         finetune-run how-well A/B's own differential) belongs in that
         CAMPAIGN's own premise check
         (`finetune_run_dispatch_proof_violations`'s `arm == "fused"`
         branch), never a silent, generic exemption inside the SHARED
         dispatch-classification primitive every campaign reuses.
      2. Every `REQUIRED_PAIRS` base must be PRESENT in this report's pairs
         (a required pair vanishing from the JSON entirely — the field
         renamed, deleted, or feature-gated off — is exactly the schema
         regression this proof exists to catch, never silently excluded)
         AND show `fused > 0`.
      2.5. Every `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH` member (today, only
         `attention_block`) must be PRESENT (same "absence is a fail" rule
         as step 2), and may read `(0, 0)` ONLY when
         `attention_block_flash`'s own `fused` count is `> 0` in this SAME
         report (defaulting to `0` when that base is entirely absent —
         i.e. every report `main`'s own binary produces today — so this
         reduces to "must independently clear `fused > 0`" there, unchanged
         from before this fold-in); otherwise it must independently clear
         `fused > 0`.
      3. Every `ABSORBABLE_BY_ATTENTION_BLOCK` member must be PRESENT (same
         "absence is a fail" rule as step 2 — F5's fix), and may read
         `(0, 0)` ONLY when `attention_block`'s own `fused` count is `> 0`
         OR (P6 Stage B FA2 fold-in) `attention_block_flash`'s own `fused`
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
            return False  # F5: absence is a schema regression, never silently excluded
        fused, _fallback = by_base[base]
        if fused == 0 and attention_block_flash_fused == 0:
            return False

    attention_block_fused = by_base.get("attention_block", (0, 0))[0]
    attention_ran = attention_block_fused > 0 or attention_block_flash_fused > 0
    for base in ABSORBABLE_BY_ATTENTION_BLOCK:
        if base not in by_base:
            return False  # F5: absence is a schema regression, never silently excluded
        fused, _fallback = by_base[base]
        if fused == 0 and not attention_ran:
            return False

    for base in LORA_SITE_EXCLUSIVE_GROUP:
        if base not in by_base:
            return False  # F5: absence is a schema regression, never silently excluded
    lora_group_fused = sum(by_base[base][0] for base in LORA_SITE_EXCLUSIVE_GROUP)
    if lora_group_fused == 0:
        return False

    return any(fused > 0 for fused, _fallback in by_base.values())


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_loss(v):
    """B5: `loss_first`/`loss_last` are bf16-sourced on every real sweep
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
    path — B6's "LOUD, per-leg, never fatal to the rest of the merge"
    discipline applies here identically). A leg entirely ABSENT from
    `raw_dir` (an older sweep predating this fold-in, or `AB_DRY_RUN`'s own
    DRY_RUN outcome) reads `outcome="MISSING"`/`"DRY_RUN"` — `metrics()`
    then returns `None` for it, and the CALLER (`build_report`) falls back
    to the single-pair ratio this module has always computed, exactly as
    if this function's own second-run legs did not exist. This is what
    makes the A,B,B,A fold-in additive: a `raw_dir` from before it exists
    merges byte-for-byte as it always did.
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
        except Exception as exc:  # noqa: BLE001 -- B6: LOUD, per-leg,
            # never silent, never fatal to the rest of the merge.
            metrics_by_leg[leg] = None
            errors_by_leg[leg] = f"{type(exc).__name__}: {exc}"
    return entries, metrics_by_leg, errors_by_leg


def bar_pair_ratio(fused_m, sdpa_m):
    """`triplets_per_s` ratio for ONE bar pair (jammi-fused-shaped metrics
    over torch-sdpa-shaped metrics) — the SAME expression `build_report`'s
    own pair-1 `ratio` has always used, factored out so pair 1 and pair 2
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

    F1 (adversarial audit): NEVER RAISES for ANY combination of `None`s —
    `bar_pair_ratio` itself reads `None` for a leg that OOM'd/FAILED (not
    just MISSING), so a bare `min(pair1_ratio, pair2_ratio)` guarded only
    against `pair2_ratio is None` (an earlier version of this function)
    crashed the ENTIRE merge — not just this one config's row — the
    moment the FIRST run's own torch-sdpa OOM'd while a clean second run
    existed (`pair1_ratio is None`, `pair2_ratio` a real float): the call
    site in `build_report` is bare, never wrapped in the per-leg
    try/except B6 already gives `metrics()`/`dispatch_pairs()`. This
    function is the one place that guarantee is enforced instead:

      * BOTH `None` (neither run produced a usable pair — no data at all):
        `bar_ratio = None`, `indeterminate = False` — `build_report`'s own
        `elif bar_ratio is None` branch already renders this as its
        existing "no ratio" FAIL, unchanged.
      * EXACTLY ONE `None` (the other run's own OOM/FAIL/MISSING legs are
        ALREADY classified by `build_report`'s `torch_fits`/
        `jammi_fused_fits` — see those variables' own doc, extended by F2
        to cover both runs — before this value is ever consulted for a
        verdict; this function's OWN job is only to return a well-defined,
        non-crashing number here, never to re-derive that classification):
        `bar_ratio` is whichever pair IS available, `indeterminate =
        False` — the single-pair degrade this module has always had,
        symmetric in EITHER direction now, not just "pair2 missing".
      * BOTH present: `bar_ratio = min(pair1_ratio, pair2_ratio)` — the
        estimator LEAST FAVOURABLE to jammi (the SAME "ratio uses the min
        of two torch runs" convention `docs/maintainer/
        fine-tune-performance-guide.md`'s own stacked-sweep artifact
        caveat already names for its own two-torch-run campaign, applied
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
    treats this as a hard failure, unchanged from before this file's
    extraction).

    B6: a merge-stage error on ONE leg (`metrics()`/`dispatch_pairs()`
    raising — a solo dispatch counter, a missing report key) is caught
    HERE, per leg, so it produces a LOUD per-row error (visible in both the
    table and the JSON, under that leg's `outcome`/this config's
    `jammi_fused_dispatch_proof`) instead of discarding every OTHER
    config's row too. This is a change in WHERE the exception is caught,
    never in whether `dispatch_pairs` raises at all.
    """
    slugs = config_slugs(raw_dir)
    if not slugs:
        return None, None

    # F2 (adversarial audit): read ONCE, applies to every config in this
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
            "note": "B4: a loss-trajectory-equivalence comparison additionally requires "
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
            except Exception as exc:  # noqa: BLE001 -- B6: LOUD, per-leg,
                # never silent, never fatal to the rest of the merge; see
                # this function's own doc and `dispatch_pairs`'s.
                leg_metrics[leg] = None
                leg_merge_errors[leg] = f"{type(exc).__name__}: {exc}"

        # ORDER-BALANCED A,B,B,A bar legs (finetune_ab.sh header) — the
        # SECOND run of the bar pair, additive (see `bar_second_run_metrics`'s
        # own doc for why an older `raw_dir` without these two legs merges
        # unchanged).
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
        # run at all — backward compat), FAILED, or OOM'd `jammi-fused-2`
        # never spuriously invalidates the config through this channel —
        # only a REPORT that was actually produced (`OK`, or a merge-stage
        # schema error on one that tried to be) can.
        jammi_fused_2_leg = BAR_SECOND_RUN_LEGS["jammi-fused"]
        if second_run_errors[jammi_fused_2_leg] is not None:
            proof2 = f"ERROR: {second_run_errors[jammi_fused_2_leg]}"
        else:
            proof2 = fused_proof(second_run_metrics[jammi_fused_2_leg])

        # LEG-PREMISE CHECK (fold-in, this round): compares the jammi-fused
        # leg's record (the one this sweep's own ratio/proof are computed
        # from) against the torch-sdpa leg's. The `jammi-eager`/`torch-eager`
        # FALLBACKS below are used for PROVENANCE only — see the
        # `leg_premise_not_comparable` note further down — the two legs of
        # ONE config are supposed to have run under the IDENTICAL
        # seed/batch/seq/dtype/dropout/lora premise (`finetune_ab.sh`'s
        # matched-flags convention), and this is the first place that
        # premise is actually CHECKED rather than merely assumed. `None`
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
        # PR #381 re-audit (class-A, face A2): a leg that is only a FALLBACK
        # (torch-sdpa OOM'd → torch-eager; jammi-fused failed → jammi-eager)
        # is the OTHER attention reference class, so its `attention_arm`
        # can never match the preferred leg's — refusing that as an identity
        # mismatch would turn a documented NON-gating outcome (an OOM row)
        # into `INVALID` + exit 1. The row is "not comparable" instead: the
        # identity check is SKIPPED (`leg_premise_violations` stays `None`,
        # never an empty "checked, clean" list), the reason is recorded
        # (`leg_premise_not_comparable`), and the ratio/verdict logic below
        # handles the missing preferred leg exactly as before.
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
                # PR #381 audit B2: the clip row's stated `max_grad_norm` must
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
        # `jammi-fused-2` AND `torch-sdpa-2` themselves read `OK` — an
        # older `raw_dir` (both `MISSING`) or a second-run OOM/FAIL on
        # either side degrades to "not checked" (`None`, never an empty
        # "checked, clean" list), the SAME shape `leg_premise_violations_list`
        # itself uses when there is nothing to compare — `bar_pair_ratio`'s
        # own `pair2_ratio` already reads `None` in that case too, so the
        # verdict already degrades to the single-pair estimator without
        # this check's help; this check's OWN job is only to refuse when a
        # SECOND-RUN report was actually produced and disagrees.
        torch_sdpa_2_leg = BAR_SECOND_RUN_LEGS["torch-sdpa"]
        second_run_premise_violations_list = None
        # NOTE (advisory, not a gate): when NEITHER second-run leg is `OK`
        # (an older `raw_dir`, a fully-dry-run sweep, or both OOM'd/FAILED
        # independently), BOTH entries below stay `None`/`None` in the
        # rendered JSON — this is the ORDINARY "nothing to record"
        # rendering `leg_provenance` already gives every OTHER absent leg
        # (see that function's own doc), never itself a distinct signal a
        # human reader or a future check should treat as meaningful beyond
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

        # F3 (adversarial audit — cross-RUN premise): the SAME-run checks
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
        cross_run_premise_violations_list = None
        if entries["jammi-fused"]["outcome"] == "OK" and second_run_entries[jammi_fused_2_leg]["outcome"] == "OK":
            jammi_run1_fields = leg_identity_fields(entries["jammi-fused"]["report"], "jammi-fused")
            jammi_run2_fields = leg_identity_fields(second_run_entries[jammi_fused_2_leg]["report"], jammi_fused_2_leg)
            v = generic_leg_premise_violations(
                FINETUNE_IDENTITY_FIELDS, jammi_run1_fields, jammi_run2_fields,
                label_a="jammi-fused", label_b=jammi_fused_2_leg,
            )
            if v:
                cross_run_premise_violations_list = list(v)
        if entries["torch-sdpa"]["outcome"] == "OK" and second_run_entries[torch_sdpa_2_leg]["outcome"] == "OK":
            torch_run1_fields = leg_identity_fields(entries["torch-sdpa"]["report"], "torch-sdpa")
            torch_run2_fields = leg_identity_fields(second_run_entries[torch_sdpa_2_leg]["report"], torch_sdpa_2_leg)
            v = generic_leg_premise_violations(
                FINETUNE_IDENTITY_FIELDS, torch_run1_fields, torch_run2_fields,
                label_a="torch-sdpa", label_b=torch_sdpa_2_leg,
            )
            if v:
                cross_run_premise_violations_list = (cross_run_premise_violations_list or []) + v

        for leg in LEGS:
            err_tail = entries[leg]["err_tail"]
            if leg_merge_errors[leg] is not None:
                err_tail = (err_tail + "\n" if err_tail else "") + f"[merge-stage] {leg_merge_errors[leg]}"
            # A: the negative control's own two provenance facts, surfaced
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
        # now surfaces in the SAME `fused_proof` column the primary
        # `jammi-fused` row uses — this pair leg carries the SAME positive-
        # proof discipline, so its own column reads the same way. Omitted
        # when the second run never ran at all (`MISSING` — an older
        # `raw_dir`/`AB_DRY_RUN`'s own placeholder legs), so an old
        # fixture's table renders exactly as it always did, with no new
        # "MISSING" clutter rows.
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

        # F2: a DRY_RUN outcome on EITHER run of EITHER bar leg (not just
        # the primary `LEGS` four) is still the SAME benign "nothing ran
        # for real" case -- extended here so `AB_DRY_RUN=1` reads
        # `N/A (dry-run)` regardless of which run a stub leg happens to be.
        any_dry_run = any(entries[leg]["outcome"] == "DRY_RUN" for leg in LEGS) or any(
            second_run_entries[leg]["outcome"] == "DRY_RUN" for leg in BAR_SECOND_RUN_LEGS.values()
        )
        torch_fits = entries["torch-sdpa"]["outcome"] == "OK"
        jammi_fused_fits = entries["jammi-fused"]["outcome"] == "OK"

        # F2 (adversarial audit — "make the header's promise true"): under
        # `two_run_mode`, `jammi-fused-2` gets the SAME OOM/no-OOM clause
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

        # The #352 bar is "no OOM where torch fits" -- it binds ONLY when
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
        elif bar_ratio is None:
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

        # Advisory (iv), round-2 audit fix on PR #372: a failed/errored
        # `fused_proof` used to only APPEND a cosmetic "[WARN: ...]" suffix
        # to whatever ratio-based verdict was already computed above -- so a
        # config whose jammi-fused leg silently fell back to EAGER kernels
        # (the exact regression `fused_proof` exists to catch) could still
        # print `PASS (ratio 0.95x) [WARN: ...]`, and nothing downstream
        # (`main()`'s exit code, a human skimming the table for the string
        # "FAIL") ever noticed. This is a DIFFERENT class of problem than
        # the ratio-based PASS/FAIL bar this crate deliberately RECORDS,
        # never GATES, across a heterogeneous fleet (see
        # `finetune_ab.sh`'s own "script's own exit code reflects whether
        # the sweep RAN, not whether every [config] passed" doctrine): a
        # ratio below bar is a real, machine-dependent PERFORMANCE
        # observation; a failed fused_proof means the MEASUREMENT ITSELF is
        # not known to have exercised the code path it claims to -- the
        # ratio computed above could belong to a DIFFERENT kernel
        # composition entirely, making the PASS/FAIL classification
        # meaningless rather than merely unfavorable. INVALID therefore
        # REPLACES (never just annotates) whatever ratio-based verdict was
        # computed, and `main()` treats ANY `INVALID` verdict as a hard
        # sweep failure (non-zero exit) -- the one carve-out from the
        # record-don't-gate doctrine this crate makes, because this is a
        # correctness-of-measurement question, not a perf-number question.
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

        # F3 override — cross-run premise drift invalidates the config
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

        # F2 override — the STRONGEST of the carve-outs above: a genuinely
        # INCOMPLETE sweep (the marker promised all four bar legs, one
        # never ran at all) is not even a "didn't fit"/"OOM" measurement,
        # so it REPLACES whatever verdict any of the checks above produced
        # (deliberately last, so it always wins when it fires).
        if two_run_missing_leg_reason is not None:
            verdict = f"{FINETUNE_AB_VERDICT_INVALID_PREFIX} ({two_run_missing_leg_reason})"

        summary_rows.append((slug, ratio, pair2_ratio, bar_ratio, loss_ratio, verdict))
        merged["configs"][slug] = {
            "legs": {leg: {"outcome": entries[leg]["outcome"], "metrics": leg_metrics[leg]} for leg in LEGS},
            # Order-balanced A,B,B,A bar legs' own SECOND run — additive,
            # keyed by the raw leg name (`"jammi-fused-2"`/`"torch-sdpa-2"`),
            # never folded into `"legs"` above (which stays keyed by `LEGS`
            # only, so nothing reading `merged["configs"][slug]["legs"]`
            # against `LEGS`'s own four names needs to change).
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
            # F3 — cross-RUN premise (jammi-fused vs jammi-fused-2,
            # torch-sdpa vs torch-sdpa-2), independent of the two SAME-run
            # checks above. `None` (never an empty list) when neither
            # cross-run pair had both sides `OK` to compare.
            "leg_premise_violations_cross_run": cross_run_premise_violations_list,
            # F2 — `None` unless `two_run_protocol` (top-level) is `True`
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
            # when the second run is unavailable (an older `raw_dir`), in
            # which case `bar_ratio == ratio_jammi_fused_over_torch_sdpa`
            # (the single-pair behaviour this module has always had).
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
        # Advisory (adversarial audit): `<13` left ZERO trailing space
        # after a 13-char leg name (`jammi-fused-2`, `jammi-fused-2` being
        # exactly 13 characters), running the `outcome` column's text
        # directly into it with no separator at all. `<14` guarantees at
        # least one space after every leg name this module currently
        # emits (`jammi-fused-2`/`torch-sdpa-2` are 13/12 characters).
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
            # A's negative control: the jammi-eager row's own
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


# ============================================================================
# unit 63 H4b — finetune-run A/B merger (docs-ci domain).
#
# Merge + sign-test stage for `ci/scripts/perf/finetune_run_ab.sh`'s
# (seed, arm, repeat) legs -- NOT `finetune_ab.sh`-shaped (no torch reference,
# no checkout-switching, no throughput ratio): this producer drives ONE
# jammi binary over `{fused, alloff}` x N pre-registered seeds x `{r1, r2}`
# same-seed repeats, against the committed `cookbook/fixtures/
# finetune_heldout/` fixture (CONTRACT H3). The comparison here is
# LEARNING OUTCOME (held-out example-mean loss), never throughput -- the
# exact two-sided sign test over `d_i = fused - alloff` per seed is the
# PRIMARY decision statistic (CONTRACT Frame / C16), computed INTO this
# merged artifact rather than left to a human to re-derive from raw legs.
#
# `sign_test` below is a Python u128-equivalent mirror of
# `jammi_numerics::stats::sign_test::sign_test` (branch `numerics/
# 63-sign-test`, not yet on this tool's own base -- mirrored from that
# module's own doc/tests, since Python's `int` is already arbitrary-
# precision, no `checked_mul`/overflow-refusal machinery is needed to match
# its EXACT-INTEGER discipline): the SAME multiplicative binomial-row
# recurrence (`C(n,i) = C(n,i-1) * (n-i+1) // i`, exact integer division at
# every step), the SAME `2 * P(X >= max(n_pos, n_neg))` capped at `1.0`,
# ties excluded from `n` but reported on `SignTestResult.ties`, never
# silently dropped, and the SAME three typed refusals (n=0 empty input,
# n=0-via-all-ties with a distinct message, NaN differences) -- see
# `sign_test`'s own doc for the field-by-field mirror.
# ============================================================================

FINETUNE_RUN_ARMS = ("fused", "alloff")
FINETUNE_RUN_REPEATS = ("r1", "r2")

# Premise legs (CONTRACT Frame / H4 / PLAN.md v2 delta 8, conjunctive with
# the cross-arm identity check below -- ALL must hold for a seed's `d_i` to
# be trusted):
#
#   * admission_is_dense — PRE-REGISTERED `False` (v2 delta 8): the
#     committed arxiv fixture's variable-length pairs take the PADDED
#     transport, never the dense branch T6's own 1.55x figure was measured
#     on; a leg reading `True` is out of the scoped verdict, not merely a
#     surprising number.
#   * learning_happened_delta — must clear a strictly-positive floor: the
#     CONTRACT's own RED control ("lr=0 arm x2 seeds fails learning-
#     happened") is the falsification target this floor exists to catch --
#     a leg whose train-side probe shows no observed learning cannot have
#     its held-out movement attributed to training at all.
#
#     CONTRACT amendment 2026-08-29b (probe bug fix): this is no longer a
#     pre-derived scalar the producer computes -- `finetune_run.rs`'s old
#     `learning_happened_delta` took its baseline AFTER the first
#     resume-cycle had already trained epoch 0 (measuring epoch-1..final,
#     excluding the largest learning epoch), and its endpoint choice was
#     never pre-registered. The producer now emits a RAW per-epoch
#     `train_probe_series: Vec<f64>` -- index 0 the UNTRAINED-init probe
#     (LoRA init is ZerosB, so an lr=0 leg still reads exactly 0.0 there),
#     one entry per epoch thereafter, the last entry the final probe -- and
#     THIS MERGER derives the premise: `series[0] - series[-1] > floor`
#     (strict; "the rule lives where rules live", never in the producer).
#     See `finetune_run_probe_series_delta` for the derivation and its three
#     typed refusals (missing, too-short, non-finite), and the v1-era
#     scalar-only carve-out (a leg carrying the OLD `learning_happened_delta`
#     field with no `train_probe_series` is a producer-version mismatch,
#     INVALID outright -- historical v1 artifacts, including the committed
#     campaign-v1 evidence, are never silently re-adjudicated under this
#     corrected rule).
#   * tie_fraction — must stay under a cap: C16's own hinge-saturation
#     finding ("the bench's hinge saturates to loss_last == 0.0 on both
#     arms at the shapes tried") is exactly a saturated-ties failure mode;
#     MNRL (the amendment's default objective) keeps this leg as a cheap,
#     normally-trivially-clearing invariant rather than a live risk, but the
#     CONTRACT keeps it conjunctive regardless of which objective a run
#     selected. 0.5 is chosen deliberately loose (an ordinary held-out
#     batch's example-level ties are rare for either objective) and
#     deliberately tight enough to catch the C16 saturation shape
#     (tie_fraction -> 1.0) long before it reaches that ceiling.
FINETUNE_RUN_EXPECTED_ADMISSION_IS_DENSE = False
# esc-045 (the "metric is the defect" saga, docs/maintainer/fine-tune-
# performance-guide.md section 6): a floor is a PRE-REGISTERED number, never
# retroactively adjusted to make a measured run pass. 0.0 stays exactly
# 0.0 -- amendment 2026-08-29b fixes the INSTRUMENT feeding this floor
# (the probe's own baseline), never the floor value itself.
FINETUNE_RUN_LEARNING_HAPPENED_FLOOR = 0.0
FINETUNE_RUN_TIE_FRACTION_CAP = 0.5

# CONTRACT amendment 2026-08-29e (D*, post-RED-proof measurement): the
# learning-happened premise is DECOMPOSED into two conjunctive premises --
# `training_effective` (`|series[0] - series[-1]| > FLOOR`: the optimizer
# demonstrably moved the model, direction-agnostic) and `train_direction`
# (`sign(series[0] - series[-1])` must match the leg's own DECLARED
# direction). Because `FINETUNE_RUN_LEARNING_HAPPENED_FLOOR == 0.0`,
# `d > f <=> (|d| > f AND d > f)` for any `f >= 0` -- so for every leg whose
# declared direction is DESCENT (every non-RED-proof leg: primary A/B both
# arms, lr0 control, the alloff partner of a RED-proof column, and a
# non-RED-proof mutant column's own fused-shaped leg), this decomposition is
# BEHAVIOR-IDENTICAL to the pre-amendment single check `delta >
# FINETUNE_RUN_LEARNING_HAPPENED_FLOOR` -- same firing condition, same set of
# excluded legs, only the diagnostic MESSAGE differs (training_effective vs.
# train_direction names which half failed). A RED-proof column's own MUTANT
# leg is the one place `train_direction` can read `ASCENT` instead (see
# `RED_PROOF_EXPECTED_TRAIN_DIRECTION` below) -- gradient ascent makes
# `delta` negative BY DESIGN, so the pre-amendment descent-only check refused
# exactly the strongest true positives of the RED-proof detection question
# (the basis this amendment fixes; see docs/plans/63-how-well/CONTRACT.md's
# own "Amendment 2026-08-29e" section for the full pressure-tested rationale
# and falsifiers).
FINETUNE_RUN_TRAIN_DIRECTION_DESCENT = "descent"
FINETUNE_RUN_TRAIN_DIRECTION_ASCENT = "ascent"
# unit-63 round-7 audit advisory (d): amendment 2026-08-29b item 4's boundary
# constraint ("decaying LR schedules stay disabled for this tier until the
# resume-cycle LR-horizon defect ... is fixed; the campaign's constant/
# 0-warmup setting is unaffected") had NO mechanical enforcement anywhere in
# this merger -- `schedule` is a `FINETUNE_RUN_IDENTITY_FIELDS` member (so a
# fused/alloff MISMATCH was already caught), but a leg that ran a DECAYING
# schedule on BOTH arms identically would clear that identity check
# unchallenged. `finetune_run_named_arm_premise_violations` below now checks
# the POSITIVE fact the boundary constraint depends on.
FINETUNE_RUN_EXPECTED_SCHEDULE = "constant"

# The pre-registered decision rule (CONTRACT Frame / PLAN.md v2 delta 3/4,
# unit-63 audit finding 1 -- `build_finetune_run_report` used to compute the
# sign test and then hardcode `status = "GREEN"`, never actually applying
# this rule): N=12 seeds x 2 arms is the ONE cell this rule is pre-registered
# for; a premise-clean seed count other than exactly
# `FINETUNE_RUN_GATE_SEED_COUNT` is never rescaled to fit -- it is INVALID,
# full stop. `FINETUNE_RUN_DECISION_THRESHOLD` (11 of 12) is compared
# directly against `sign_test`'s own `n_pos`/`n_neg` (NOT its `n`, which
# excludes ties -- the threshold is "how many of the 12 shared this sign",
# not "how many of the non-tied pairs did"). `FINETUNE_RUN_ALPHA2` is the
# pre-registered significance level for the (12, 11) exact-tail cell
# (`SignTestMirrorTests.test_golden_n12_k11_pinned_cell` pins the exact
# p_value, 13/2048; alpha2=0.0064 is the rounded pre-registered level
# recorded BESIDE that computed p_value, never substituted for it -- the
# concordance-threshold count is the operative rule, not a p < alpha2 gate).
FINETUNE_RUN_GATE_SEED_COUNT = 12
FINETUNE_RUN_DECISION_THRESHOLD = 11
FINETUNE_RUN_ALPHA2 = 0.0064
FINETUNE_RUN_DECISION_RULE_TEXT = (
    "Pre-registered decision rule (CONTRACT 63 Frame / PLAN.md v2 delta 3-4): over "
    "N=12 premise-clean seeds' d_i = fused.held_out_example_mean - "
    "alloff.held_out_example_mean, RED iff (n_pos >= 11 of 12 OR n_neg >= 11 of 12) AND "
    "sign(mean(d_i)) agrees with that concordant direction (two-sided; alpha2=0.0064 is "
    "the pre-registered level for the (12, 11) cell, recorded beside the computed exact-tail "
    "p_value, never substituted for it). d_i > 0 dominant (fused worse) is "
    "degradation-concordant -> status RED; d_i < 0 dominant (fused better) is "
    "improvement-concordant -> status RED_FOR_INVESTIGATION (anomalous improvement is "
    "investigated, never silently celebrated); neither threshold met, or the threshold is "
    "met but the mean's sign disagrees, -> status GREEN. A premise-clean seed count other "
    "than exactly 12 -> status INVALID naming the count (the rule is pre-registered FOR 12 "
    "seeds; never rescaled silently)."
)

# Unit-63 round-16 audit (docs-ci, correcting round-15's own overclaim): the
# finetune-run merger's own COMPLETE `status` vocabulary -- every value
# `build_finetune_run_report`'s own status fold (immediately below) can set.
# Round-15 named this tuple ONCE but the fold itself kept RE-TYPING the six
# literal strings rather than reading from these names, so a producer-only
# seventh status (a new fold branch assigning some new literal never added
# here) would drift SILENTLY -- the fold would happily set it, nothing here
# would notice, and only `runpod_gpu_howwell.sh`'s own case block's `*)`
# catch-all (a warning, not a gate) would ever see it. Two independent fixes,
# mirroring the `DOSE_LADDER_EXIT_CAUSE_NAMES` pattern below (constant +
# runtime equality check, never a "verified by reading" comment alone):
#   1. the fold now ASSIGNS `status` FROM the single-value constants below
#      (`FINETUNE_RUN_STATUS_INVALID` etc.), never a re-typed literal --
#      behavior-identical (same strings), but a fold edit that doesn't touch
#      this module is now structurally impossible without a NameError.
#   2. the fold's own runtime guard (immediately after `status` is set,
#      below) raises `AssertionError` if the computed `status` is not a
#      member of `FINETUNE_RUN_STATUSES` -- the producer-side belt this
#      class needs: a NEW status value assigned via any future fold branch,
#      even one that (bug) hand-types a literal instead of using a named
#      constant, is caught immediately, at the point of production, never
#      silently forwarded into the artifact.
# Partitioned the way both consumers actually split it: `FINETUNE_RUN_GATING_
# STATUSES` is the exit-code-forcing subset (a correctness-of-measurement
# problem, INVALID, or a fired decision rule, RED / RED_FOR_INVESTIGATION --
# this module's own exit fold below asserts against this exact tuple, never a
# re-typed literal), `FINETUNE_RUN_GREEN_STATUS` is the single ordinary-pass
# value, and `FINETUNE_RUN_RECORD_ONLY_STATUSES` is the never-gates subset (a
# leg that never ran, or a dry run -- never itself a merge failure).
# `ShellStatusCaseArmsBoundToFinetuneRunStatusesTests`
# (`test_howwell_dose_ladder_cause.py`) parses `runpod_gpu_howwell.sh`'s own
# case-arm patterns for `$STATUS` and asserts they exactly cover
# `FINETUNE_RUN_STATUSES` with no extras and no gaps -- a status added to
# `FINETUNE_RUN_STATUSES` (correctly, via the constants below) without a
# matching shell arm (or vice versa) is a RED test there. Combined with the
# fold's own runtime guard, BOTH drift directions are now caught: a status
# added to the constants without a shell arm (shell-side test, above) AND a
# status the fold assigns without adding it to the constants (this fold's own
# runtime guard, immediately below) -- never silent drift, never merely "one
# capability enumerated by hand in two modules with no mechanical oracle".
# That shell-side test's own docstring names what ITS OWN binding does NOT
# cover: `$STATUS` consumed by anything OTHER than that one case block; the
# producer-side runtime guard below is the separate mechanism that covers the
# fold's own literal-drift direction, not that test.
FINETUNE_RUN_STATUS_INVALID = "INVALID"
FINETUNE_RUN_STATUS_RED = "RED"
FINETUNE_RUN_STATUS_RED_FOR_INVESTIGATION = "RED_FOR_INVESTIGATION"
FINETUNE_RUN_STATUS_GREEN = "GREEN"
FINETUNE_RUN_STATUS_DRY_RUN = "DRY_RUN"
FINETUNE_RUN_STATUS_INCOMPLETE = "INCOMPLETE"
FINETUNE_RUN_GATING_STATUSES = (
    FINETUNE_RUN_STATUS_INVALID,
    FINETUNE_RUN_STATUS_RED,
    FINETUNE_RUN_STATUS_RED_FOR_INVESTIGATION,
)
FINETUNE_RUN_GREEN_STATUS = FINETUNE_RUN_STATUS_GREEN
FINETUNE_RUN_RECORD_ONLY_STATUSES = (FINETUNE_RUN_STATUS_DRY_RUN, FINETUNE_RUN_STATUS_INCOMPLETE)
FINETUNE_RUN_STATUSES = (
    FINETUNE_RUN_GATING_STATUSES + (FINETUNE_RUN_GREEN_STATUS,) + FINETUNE_RUN_RECORD_ONLY_STATUSES
)

# The lr=0 RED control (CONTRACT Frame: "RED control: lr=0 arm x2 seeds fails
# learning-happened", unit-63 audit advisory (b)): a SEPARATE leg set, run at
# `--lr 0` for both arms at a small, fixed number of control seeds, tagged
# with this `repeat` label (never `r1`/`r2`) so `load_finetune_run_leg` can
# read them without ever being mistaken for -- or counted into -- the A/B
# set's own d_values/sign test. The calibration bite this exists to catch:
# `FINETUNE_RUN_LEARNING_HAPPENED_FLOOR = 0.0` is only a meaningful premise
# leg if a genuinely-non-learning run (lr=0, no parameter update possible)
# actually FAILS it -- a control leg that PASSES (its own
# `learning_happened_delta` clears the floor despite lr=0) is a finding
# against the floor's own validity, not a training result.
FINETUNE_RUN_LR0_REPEAT = "lr0"


class SignTestError(ValueError):
    """Raised by `sign_test` for the three typed refusals its Rust twin
    (`jammi_numerics::stats::sign_test::sign_test`) also raises: n=0 (empty
    input), n=0-via-all-ties (a distinct message from the empty-input case),
    and a NaN difference (no sign under IEEE-754). Never raised for
    ±inf (a well-defined, non-zero, non-tie sign) or for an `n` too large to
    represent -- Python's `int` is already arbitrary-precision, so the
    `u128`-overflow refusal the Rust side needs (`n` in the ~125-127 range)
    has no Python analogue; this mirror never refuses on `n` size.
    """


def sign_test(diffs):
    """Exact two-sided sign test over paired differences `diffs[i] = a_i -
    b_i` -- see this module section's own doc for the field-by-field mirror
    of `jammi_numerics::stats::sign_test::sign_test`. Returns
    `{"n", "n_pos", "n_neg", "ties", "p_value"}` (the SAME five fields
    `SignTestResult` carries); raises `SignTestError` for the three typed
    refusals.
    """
    if len(diffs) == 0:
        raise SignTestError("sign test requires at least one paired difference (n=0)")
    if any(d != d for d in diffs):  # NaN != NaN under IEEE-754 -- no math.isnan import needed
        raise SignTestError("sign test requires non-NaN differences (NaN has no sign)")

    n_pos = n_neg = ties = 0
    # Fixed left-to-right fold order over the caller-supplied sequence,
    # mirroring the Rust side's own pinned-order doc -- the classification
    # of any single `d` does not depend on any other, so this does not
    # affect the resulting counts, but it is pinned explicitly rather than
    # left to an unspecified builtin partition.
    for d in diffs:
        if d == 0.0:
            ties += 1
        elif d > 0.0:
            n_pos += 1
        else:
            n_neg += 1

    n = n_pos + n_neg
    if n == 0:
        raise SignTestError(
            f"sign test requires at least one non-tied pair; all {ties} difference(s) were exact ties"
        )

    t = max(n_pos, n_neg)
    p_value = _sign_test_exact_two_sided_tail(n, t)
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "ties": ties, "p_value": p_value}


def _sign_test_binomial_row(n):
    """Row `n` of Pascal's triangle via the SAME multiplicative recurrence
    `C(n,i) = C(n,i-1) * (n-i+1) / i` the Rust `binomial_row` uses -- exact
    integer division at every step (a standard combinatorial identity: the
    product is always evenly divisible by `i`), never a float/`lgamma`
    approximation. Python's `int` needs no `checked_mul`/overflow-refusal
    machinery here (see `SignTestError`'s own doc).
    """
    row = [1]  # C(n, 0)
    for i in range(1, n + 1):
        row.append(row[-1] * (n - i + 1) // i)
    return row


def _sign_test_exact_two_sided_tail(n, t):
    """`2 * P(X >= t)` under `X ~ Binomial(n, 0.5)`, capped at `1.0` --
    computed as an exact ratio of Python `int`s (the u128-equivalent: both
    numerator and denominator are exact by construction, division to
    `float` deferred to the single final step, matching the Rust side's own
    "division deferred to the very last step" discipline) rather than a
    floating-point CDF evaluation.
    """
    row = _sign_test_binomial_row(n)
    denom = 1 << n
    tail_sum = sum(row[t : n + 1])
    numerator = tail_sum * 2
    if numerator < denom:
        return numerator / denom
    return 1.0


def finetune_run_block(report):
    return report["tiers"]["finetune_run"]


def finetune_run_leg_identity(tier):
    """This leg's `FINETUNE_RUN_IDENTITY_FIELDS` values, reusing the SAME
    generic premise-refusal core `encode_ab.sh`'s own merge step already
    builds on (`generic_leg_identity_fields`) -- `margin`/`temperature`/
    `max_grad_norm`/`warmup`/`row_lengths` fold a present `null` in as the
    stated VALUE (per `FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS`), every other
    field folds a present `null` into MISSING.
    """
    return generic_leg_identity_fields(tier, FINETUNE_RUN_IDENTITY_FIELDS, FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS)


def finetune_run_probe_series_delta(label, tier):
    """Derive the learning-happened premise's delta from the producer's RAW
    `train_probe_series` (CONTRACT amendment 2026-08-29b) -- never trust a
    pre-derived scalar; the rule (`series[0] - series[-1]`) lives HERE, in
    the merger, never in the producer. `label` prefixes every returned
    message (a seed/arm/pool-qualified string the caller already has, e.g.
    `"seed 4 alloff"` or `"lr0 seed 101 fused"`).

    Returns `(violations, delta, series)`:
      * `violations` non-empty, `delta is None` -- one of five typed
        refusals fired (a leg is NEVER assumed-good on any of these):
          1. V1-ERA SCALAR-ONLY: `tier` carries the OLD
             `learning_happened_delta` field (non-null) but no
             `train_probe_series` at all -- a producer-version mismatch.
             This leg was emitted by the pre-fix instrument (whose baseline
             excluded epoch 0's own learning, see the module-level premise
             doc above) and is NEVER silently re-adjudicated under the
             corrected rule -- historical v1 artifacts (including the
             committed campaign-v1 evidence) stay exactly as measured and
             exactly as INVALID as they were recorded.
          2. MISSING: no `train_probe_series` (and no v1 scalar either) --
             an even older producer, or a malformed report.
          3. SHORT: fewer than 2 entries -- the rule needs both the
             untrained-init probe (index 0) and at least one epoch's own
             probe to subtract.
          4. NON-FINITE: any entry is not a finite real number (NaN, +-inf,
             or a non-numeric JSON value) -- `series[0] - series[-1]` has no
             well-defined premise verdict over a series that is not
             entirely real-valued.
          5. LENGTH-VS-EPOCHS MISMATCH (unit-63 round-7 audit finding 3):
             `len(series) != tier["epochs"] + 1` -- index 0 is the
             untrained-init probe PLUS one entry per epoch, so a
             premise-clean series must carry exactly `epochs + 1` entries.
             The SHORT check above (refusal 3) only catches a series with
             fewer than 2 entries; a series whose length equals `epochs`
             itself (never `epochs + 1`) is the v1 probe bug's EXACT shape
             (the pre-fix producer's baseline excluded the init point) and
             can still clear refusal 3 whenever `epochs >= 2` -- e.g.
             `epochs=3` with a 3-entry (not 4-entry) series is not
             "SHORT" (it has more than 2 entries) but is not
             init-anchored either, and was previously silently
             adjudicated as though it were. A series that is instead too
             LONG (`len(series) > epochs + 1`, a truncation/duplication
             producer bug in the other direction) is refused identically.
      * `violations` empty, `delta` the float `series[0] - series[-1]`,
        `series` the raw list itself (recorded by the caller into
        `premise_failure_diagnostic`/`per_arm` regardless of whether the
        floor check that consumes `delta` passes or fails).
    """
    v1_scalar = tier.get("learning_happened_delta")
    series = tier.get("train_probe_series")
    if series is None or not isinstance(series, list):
        if v1_scalar is not None:
            return (
                [
                    f"{label}: carries the v1-era scalar 'learning_happened_delta'={v1_scalar!r} "
                    "with no 'train_probe_series' -- producer-version mismatch (CONTRACT amendment "
                    "2026-08-29b): this leg was emitted by a pre-fix producer whose probe baseline "
                    "excluded epoch 0's own learning; it is INVALID outright, never silently "
                    "re-adjudicated under the corrected series-derived rule"
                ],
                None,
                None,
            )
        return (
            [
                f"{label}: 'train_probe_series' is missing (or not a list) -- the learning-happened "
                "premise is derived from this raw per-epoch series (CONTRACT amendment 2026-08-29b), "
                "never assumed present"
            ],
            None,
            None,
        )
    if len(series) < 2:
        return (
            [
                f"{label}: train_probe_series has {len(series)} "
                f"entr{'y' if len(series) == 1 else 'ies'}, need at least 2 (index 0 = the "
                "untrained-init probe, one entry per epoch, the last entry the final probe)"
            ],
            None,
            series,
        )
    for entry in series:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or entry != entry or math.isinf(entry):
            return (
                [f"{label}: train_probe_series contains a non-finite or non-numeric entry ({entry!r})"],
                None,
                series,
            )
    # unit-63 round-7 audit finding 3 -- see refusal 5 in this function's
    # own doc: a leg's length-`epochs`-shaped (never `epochs+1`-shaped)
    # series is silently adjudicated by refusals 1-4 alone whenever
    # `epochs >= 2`, which is exactly the v1 probe bug's own shape (the
    # baseline excluded the init point). `epochs` is only checked when it
    # is itself a genuine (non-bool) int -- a leg missing `epochs` entirely
    # is an even older producer, caught elsewhere (`FINETUNE_RUN_IDENTITY_FIELDS`
    # already requires it), never fabricated here.
    epochs = tier.get("epochs")
    if isinstance(epochs, int) and not isinstance(epochs, bool):
        expected_len = epochs + 1
        if len(series) != expected_len:
            return (
                [
                    f"{label}: train_probe_series has {len(series)} entries but this leg's own "
                    f"epochs={epochs} (expected exactly epochs+1={expected_len}) -- series is not "
                    "init-anchored or is truncated -- a producer-version mismatch"
                ],
                None,
                series,
            )
    return [], series[0] - series[-1], series


def finetune_run_named_arm_premise_violations(arm, tier, expected_train_direction=FINETUNE_RUN_TRAIN_DIRECTION_DESCENT):
    """`finetune_run_arm_premise_violations`'s own STRUCTURED core -- returns
    `[(premise_name, message), ...]` for the CONTRACT Frame's three
    conjunctive premise legs (`admission_is_dense`, `learning_happened`,
    `tie_fraction`) PLUS the `schedule` boundary constraint (unit-63 round-7
    audit advisory (d)), so a caller building `premise_failure_diagnostic`
    (amendment 2026-08-29b item 1(c)) can name WHICH premise leg(s) failed on
    a given leg, not merely that leg's flattened message strings.
    `finetune_run_arm_premise_violations` below is a thin wrapper over this
    for the existing flat `leg_premise_violations` field every other check
    in this module already appends to.

    `expected_train_direction` (CONTRACT amendment 2026-08-29e, D*): the
    `learning_happened` premise is internally decomposed into
    `training_effective` (`|delta| > FLOOR`) and `train_direction`
    (`sign(delta)` matches this parameter) -- both are STILL reported under
    the single `"learning_happened"` premise name (never a new name), so
    every EXISTING caller's `failing_premises` list is unaffected; only the
    diagnostic message text differs between the two failure shapes. Defaults
    to `FINETUNE_RUN_TRAIN_DIRECTION_DESCENT`, exactly today's behaviour for
    every non-RED-proof call site (primary A/B, lr0 control, alloff partner,
    a non-RED-proof mutant column's own fused-shaped leg) -- with
    `FINETUNE_RUN_LEARNING_HAPPENED_FLOOR == 0.0`, `d > f <=> (|d| > f AND
    d > f)`, so the decomposition is BEHAVIOR-IDENTICAL to the pre-amendment
    single check at this default. Only `finetune_run_mutant_column_violations`
    ever passes `FINETUNE_RUN_TRAIN_DIRECTION_ASCENT`, and only for a
    RED-proof column's own mutant leg whose `patch_sha256` the committed
    `RED_PROOF_EXPECTED_TRAIN_DIRECTION` table maps to `"ascent"`.
    """
    named = []
    schedule = tier.get("schedule")
    if schedule != FINETUNE_RUN_EXPECTED_SCHEDULE:
        named.append((
            "schedule",
            f"{arm}: schedule={schedule!r}, expected {FINETUNE_RUN_EXPECTED_SCHEDULE!r} -- "
            "CONTRACT amendment 2026-08-29b item 4: decaying LR schedules stay disabled for this "
            "tier until the resume-cycle LR-horizon defect (total_steps recomputed per cycle) is "
            "fixed; `schedule` is already recorded on the tier, so this boundary constraint is "
            "mechanically enforced here rather than left to operator discipline alone",
        ))
    is_dense = tier.get("admission_is_dense")
    if is_dense != FINETUNE_RUN_EXPECTED_ADMISSION_IS_DENSE:
        named.append((
            "admission_is_dense",
            f"{arm}: admission_is_dense={is_dense!r}, expected "
            f"{FINETUNE_RUN_EXPECTED_ADMISSION_IS_DENSE!r} -- CONTRACT H4/v2 delta 8 "
            "pre-registers the PADDED transport for the committed arxiv fixture "
            "(variable-length pairs); a dense leg falls outside the scoped verdict",
        ))
    series_violations, delta, _series = finetune_run_probe_series_delta(arm, tier)
    for msg in series_violations:
        named.append(("learning_happened", msg))
    if not series_violations:
        # CONTRACT amendment 2026-08-29e (D*): training_effective, then
        # train_direction -- see this function's own `expected_train_direction`
        # doc above for the `d > f <=> (|d| > f AND d > f)` equivalence this
        # decomposition preserves at the default (descent) direction.
        training_effective = abs(delta) > FINETUNE_RUN_LEARNING_HAPPENED_FLOOR
        if not training_effective:
            named.append((
                "learning_happened",
                f"{arm}: learning_happened_delta={delta!r} (derived: train_probe_series[0] - "
                f"train_probe_series[-1]) does not clear training_effective's floor "
                f"(|delta| > {FINETUNE_RUN_LEARNING_HAPPENED_FLOOR}) -- the train-side probe "
                "shows no observed learning, so this leg's held-out movement cannot be "
                "attributed to training (the CONTRACT's own RED-control precedent: an lr=0 arm "
                "fails exactly this leg)",
            ))
        else:
            actual_direction = (
                FINETUNE_RUN_TRAIN_DIRECTION_DESCENT if delta > 0.0 else FINETUNE_RUN_TRAIN_DIRECTION_ASCENT
            )
            if actual_direction != expected_train_direction:
                named.append((
                    "learning_happened",
                    f"{arm}: learning_happened_delta={delta!r} (derived: train_probe_series[0] - "
                    f"train_probe_series[-1]) clears training_effective's floor but its own "
                    f"train_direction={actual_direction!r} does not match this leg's declared "
                    f"direction={expected_train_direction!r} -- CONTRACT amendment 2026-08-29e "
                    "(D*): a leg's held-out movement is only attributable to training when the "
                    "train-side probe itself moved in the direction this leg was declared to move "
                    "in",
                ))
    tie = tier.get("tie_fraction")
    if not isinstance(tie, (int, float)) or isinstance(tie, bool) or not (tie < FINETUNE_RUN_TIE_FRACTION_CAP):
        named.append((
            "tie_fraction",
            f"{arm}: tie_fraction={tie!r} is not below the cap ({FINETUNE_RUN_TIE_FRACTION_CAP}) -- "
            "C16's own hinge-saturation warning (a near-saturated tie fraction means the "
            "held-out loss is not discriminating between examples)",
        ))
    return named


def finetune_run_arm_premise_violations(arm, tier, expected_train_direction=FINETUNE_RUN_TRAIN_DIRECTION_DESCENT):
    """Per-arm conjunctive premise legs (CONTRACT Frame / H4 -- see this
    module section's own doc for each leg's rationale): `admission_is_dense`
    matches the pre-registered `False`, `learning_happened_delta` (derived
    from `train_probe_series`, amendment 2026-08-29b, decomposed into
    `training_effective`/`train_direction` by amendment 2026-08-29e) clears
    its floor in the declared direction, `tie_fraction` stays under its cap.
    Independent of, and IN ADDITION TO, the cross-arm identity check
    (`generic_leg_premise_violations` over `FINETUNE_RUN_IDENTITY_FIELDS`) --
    this checks facts about ONE leg's own report, never a comparison between
    two legs. Returns a list of strings, empty when every leg clears --
    `finetune_run_named_arm_premise_violations` is the structured
    (premise-name-tagged) form this wraps; `expected_train_direction` is
    threaded straight through to it (see that function's own doc).
    """
    return [msg for _name, msg in finetune_run_named_arm_premise_violations(arm, tier, expected_train_direction)]


# ALLOFF_DISABLED_OP_BASES (unit-63 round-3 audit, class-fix discovery,
# docs-ci): `finetune_run_ab.sh`'s own documented `alloff` convention
# (`JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused` -- see
# that script's own comment and `main.rs`'s `FinetuneRunArgs::arm` doc, "the
# caller is responsible for setting [this] itself before invoking this
# binary for the alloff arm") disables EXACTLY these two ops -- `alloff`
# names the flash-cascade/multi-tensor-AdamW reference arm, never a blanket
# "every fused kernel off" run. `fixtures/finetune_run_golden/
# modernbert_alloff.json` (a REAL, producer-anchored alloff leg) reads
# `ln`/`rope`/`softmax`/`geglu`/`lora_linear` all FUSED (nonzero) on this
# arm -- the class-defect this table replaces (`finetune_run_dispatch_proof_violations`
# used to require EVERY dispatch pair to read `fused == 0` for `alloff`,
# which that real leg would have failed on every field above, marking every
# real campaign `alloff` leg INVALID the day this producer went live; caught
# only once this suite's own fixtures were derived from that real leg
# instead of a hand-typed "everything is eager" assumption). Maps the
# `JAMMI_KERNELS_DISABLE` op key to the `dispatch_pairs` base name it
# governs -- `"attention_block_flash"` names itself (a `CASCADE_BASES`
# member, so its own fallback is `declined`, not `eager`); `"adamw_step_fused"`
# governs the ordinary `"adamw"` pair.
ALLOFF_DISABLED_OP_BASES = {
    "attention_block_flash": "attention_block_flash",
    "adamw_step_fused": "adamw",
}


def finetune_run_dispatch_proof_violations(arm, tier):
    """Unit-63 adversarial-audit finding 2's merger half: the finetune-run
    tier now ALSO emits finetune-step's exact `*_fused_dispatches`/
    `*_eager_dispatches` counter pairs -- verbatim field names, the
    `attention_block` pair included -- from a concurrent bench dispatch.
    This reuses `dispatch_pairs`/`fused_proof` for the shared "every ordinary
    pair independently proves itself" machinery (never a second,
    independently-drifting hand-rolled dispatch-classification mechanism --
    this module's own B2 note), but the finetune-run tier's own CAMPAIGN
    PREMISE -- CONTRACT 63 Frame pre-registers the arms as "fused cascade vs
    ALLOFF=attention_block_flash,adamw_step_fused", i.e. the A/B's own
    differential IS whether the flash cascade (and the fused AdamW kernel)
    fired -- is checked HERE, per arm, never folded into the shared
    `fused_proof` primitive `finetune-step`'s own campaigns also reuse
    (coordinator correction, unit-63 round-3 audit: an earlier draft of this
    round put a build-capability-miss exemption directly into `fused_proof`
    itself, which would have legitimized a `finetune-run` `fused` leg built
    WITHOUT `flash-attn` compiled in -- a leg that can never exercise the
    pre-registered differential at all, making the experiment null).

    `arm` (`"fused"` or `"alloff"`) states what this leg CLAIMS to have run;
    this checks the COUNTED FACT behind that claim, mirroring
    `clip_fact_violations`'s own "a claim with no counted fact behind it is
    refused" shape, applied to dispatches instead of the clip counter:

      0. ARM-AGNOSTIC MERGER CONSISTENCY PREMISE (unit-63 round-4 audit
         F-1, checked on EVERY leg before either arm's own branch below):
         `flash_capability_gates` DomainMisses the flash cascade whenever
         `dtype != DType::BF16` (`jammi-encoders/src/modernbert.rs`'s own
         `dtype_is_bf16` gate) -- a leg reporting
         `attention_block_flash_fused_dispatches > 0` while its own
         `backbone_dtype` is not `bf16` is claiming a dispatch its own
         declared premise forbids; the report is internally contradictory
         and untrusted outright, never resolved by trusting one of the two
         fields over the other. This makes the unemittable state (nonzero
         flash-fused counters at a non-bf16 dtype) unrepresentable at the
         merger, on top of `finetune_run_ab.sh` now always passing
         `--backbone-dtype bf16` at the producer (belt-and-braces: a
         hand-run leg that skips the script, or a future producer
         regression, still cannot pass this check).
      * `arm == "fused"`:
          1. PREMISE: `backbone_dtype` must canonicalize to `bf16`
             (unit-63 round-4 audit F-1) -- independent of check 0 above,
             which only fires when the counters HAPPEN to claim a positive
             flash dispatch. CONTRACT 63 Frame pre-registers the flash
             cascade, itself BF16-only, as this arm's own admitted branch;
             a `fused` leg declared at any other dtype cannot exercise the
             pre-registered differential at all, regardless of what its
             OTHER counters read (e.g. the block arm's own absorption
             silently picking up the slack while flash itself never
             fires) -- an INVALID premise, checked before `flash_compiled`
             so a runtime dtype mismatch is never misreported as a
             compile-time one.
          2. PREMISE: `flash_compiled` must be `True`. CONTRACT 63 Frame
             pre-registers the flash cascade as this arm's own admitted
             branch; a build that never compiled it in cannot possibly
             exercise the pre-registered differential, regardless of what
             its OTHER dispatch counters read -- an INVALID premise, not a
             leg whose classification can be trusted at all
             (`finetune_run_ab.sh` builds `--features
             cuda,jammi-encoders/flash-attn` for exactly this reason).
          3. `fused_proof` (the SAME gate a finetune-step `jammi-fused` leg
             is held to, UNCHANGED -- see that function's own doc) must
             return `True`: every ordinary `REQUIRED_PAIRS` base (`ln`,
             `geglu`, `adamw`) independently clears `fused > 0`/`eager == 0`,
             and the CASCADE absorption chain (`attention_block_flash`
             absorbing `attention_block`, which in turn absorbs
             `rope`/`softmax`) is internally consistent.
          4. THE PRE-REGISTERED BRANCH ITSELF:
             `attention_block_flash_fused_dispatches > 0`. `fused_proof`'s
             own absorption rule (3.5) is satisfied whenever EITHER the
             flash cascade OR the block arm independently fires -- correct
             for `finetune-step`'s own flash-vs-block A/B, where either arm
             is a legitimate leg -- but THIS arm specifically claims to be
             running the flash-cascade branch, so it must be the one that
             actually fired, not merely "the cascade's absorption chain
             holds because the block arm picked up the slack instead."
      * `arm == "alloff"` (unit-63 round-3 audit, block 4 + the class-fix
        discovery below, + the coordinator's cascade-absorption correction):
        this leg's OWN `kernels_disabled_requested` names the op(s) it
        actually claims to have disabled -- ONLY the `ALLOFF_DISABLED_OP_BASES`
        members it names (`attention_block_flash`, `adamw`) must show
        `fused == 0` AND a POSITIVE counted fallback (`eager`/`declined`
        `> 0` -- the mirror-image of the "fused == 0 alone" check the
        pre-fix code ran: an all-zero pair reading is no counted fact
        behind the "disabled" classification, only a claim, exactly like
        `clip_fact_violations`'s own "a clip claim with no counted fact
        behind it" refusal). Every OTHER dispatch pair is UNCHECKED here --
        `alloff` never claimed anything about them, so a real leg's own
        `ln`/`rope`/`softmax`/`geglu`/`lora_linear` staying fused is not
        this arm's business (`fixtures/finetune_run_golden/
        modernbert_alloff.json`, a REAL leg, shows exactly this). SEPARATELY
        (the positive training-path proof for this arm): `attention_block`
        must show `fused > 0` -- `attention_block` is NOT itself named in
        the alloff disable list, only `attention_block_flash` is, so on a
        real (`head_dim == 64`) checkpoint it remains an ACTIVE, undisabled
        fused kernel that the disabled flash cascade must fall through to;
        the fallback ENGAGING is proven by `attention_block`'s own FUSED
        count (never its eager one -- an eager reading there would mean the
        block arm's OWN admission declined too, a different, unrelated
        failure mode this proof does not paper over).

    A leg with NO dispatch-counter fields at all (an older producer build
    predating this emission) is ALSO a violation -- `dispatch_pairs`
    returning empty is never silently ASSUMED to mean "ran the claimed arm
    cleanly"; a classification with no counted fact behind it is
    untrusted -- discarded, never merely annotated, mirroring the existing
    finetune-step tier's own semantics for this same proof. A malformed
    pair (`dispatch_pairs` raising -- a solo counter, or a base outside
    `ALL_BASES`) is caught HERE and reported the same way, never left to
    propagate and void the whole merge the way an uncaught raise would.

    Returns a list of violation strings, empty when this leg's claimed arm
    matches its counted dispatch facts.
    """
    try:
        pairs = dispatch_pairs(tier)
    except KeyError as exc:
        return [f"{arm}: dispatch-pair schema error -- {exc}"]
    if not pairs:
        return [
            f"{arm}: no *_fused_dispatches/*_eager_dispatches counter fields present on this "
            "leg's report at all -- an older producer build predating this emission cannot be "
            "assumed to have run the classified arm; the leg's own claim is untrusted, not "
            "merely unannotated"
        ]
    by_base = {base: (fused, fallback) for base, fused, fallback in pairs}

    # unit-63 round-4 audit F-1 (merger consistency premise, arm-agnostic --
    # checked before either arm's own branch): `flash_capability_gates`
    # DomainMisses the whole flash cascade whenever `dtype != DType::BF16`
    # (`jammi-encoders/src/modernbert.rs`'s own `dtype_is_bf16` gate) -- a
    # leg cannot have counted a positive `attention_block_flash_fused_
    # dispatches` unless the backbone it ran actually admitted BF16. A
    # report claiming BOTH a positive flash-fused count AND a non-bf16
    # `backbone_dtype` is not a leg whose classification can be trusted at
    # all -- the counters and the declared premise cannot both be true, so
    # this makes the unemittable-state class itself unrepresentable, rather
    # than trusting whichever one of the two a downstream check happens to
    # read first.
    flash_fused_claimed, _flash_declined_claimed = by_base.get("attention_block_flash", (0, 0))
    declared_dtype = canonicalize_identity_field("backbone_dtype", tier.get("backbone_dtype"))
    if flash_fused_claimed > 0 and declared_dtype != "bf16":
        return [
            f"{arm}: attention_block_flash_fused_dispatches={flash_fused_claimed!r} but "
            f"backbone_dtype={tier.get('backbone_dtype')!r} -- counters claim a dispatch the "
            "declared dtype forbids (flash_capability_gates admits only BF16, "
            "modernbert.rs's own dtype_is_bf16 gate); this leg's own report is internally "
            "contradictory, not merely unclassifiable"
        ]

    if arm == "fused":
        # A FUSED leg's own defining premise (unit-63 round-4 audit F-1,
        # independent of the counter-vs-dtype contradiction check above,
        # which only fires when the counters HAPPEN to claim a positive
        # dispatch): CONTRACT 63 Frame pre-registers the flash cascade as
        # this arm's own admitted branch, and that branch is BF16-only
        # (`flash_capability_gates`'s `dtype_is_bf16` gate) -- a `fused`
        # leg declaring any OTHER `backbone_dtype` cannot possibly exercise
        # the pre-registered differential, regardless of what its counters
        # happen to read (e.g. the block arm's own absorption silently
        # picking up the slack while flash itself never fires) -- an
        # INVALID premise outright, checked before `flash_compiled` so a
        # build/runtime mismatch is never misreported as a compile-time one.
        if declared_dtype != "bf16":
            return [
                f"fused: backbone_dtype={tier.get('backbone_dtype')!r} -- CONTRACT 63 Frame "
                "pre-registers the flash cascade as this arm's own admitted branch, and "
                "flash_capability_gates admits BF16 only (modernbert.rs's own dtype_is_bf16 "
                "gate); a 'fused' leg run at any other dtype cannot exercise the pre-registered "
                "differential at all -- an INVALID premise, not a leg whose classification can "
                "be trusted"
            ]
        flash_compiled = tier.get("flash_compiled")
        if flash_compiled is not True:
            return [
                f"fused: flash_compiled={flash_compiled!r} -- CONTRACT 63 Frame pre-registers "
                "the flash cascade as this arm's own admitted branch ('fused cascade vs "
                "ALLOFF=attention_block_flash,adamw_step_fused'); a build that cannot compile it "
                "in (flash_compiled is not true) can never exercise the pre-registered "
                "differential -- an INVALID premise, regardless of what its dispatch counters "
                "read"
            ]
        m = {
            "dispatch_pairs": pairs,
            "flash_compiled": flash_compiled,
            "kernels_disabled_requested": tier.get("kernels_disabled_requested"),
            "kernels_disabled_fired": tier.get("kernels_disabled_fired"),
        }
        proof = fused_proof(m)
        if proof is not True:
            return [
                f"fused: fused-dispatch proof failed or errored ({proof!r}) -- this leg's "
                "'fused' classification is untrusted, discarded, not merely annotated"
            ]
        flash_fused, _flash_declined = by_base.get("attention_block_flash", (0, 0))
        if flash_fused <= 0:
            return [
                f"fused: attention_block_flash_fused_dispatches={flash_fused!r} -- CONTRACT 63 "
                "Frame pre-registers the flash cascade as this arm's own admitted branch; a "
                "'fused' leg that never actually dispatched it (the block arm's own absorption "
                "picking up the slack instead) is not the pre-registered experiment this arm "
                "claims to run"
            ]
        return []
    if arm == "alloff":
        violations = []

        # The disabled-op positive proof (class-fix discovery, see
        # `ALLOFF_DISABLED_OP_BASES`'s own doc): scoped to exactly the ops
        # THIS leg's own `kernels_disabled_requested` names, never a blanket
        # "every pair must be zero" claim.
        requested = set(tier.get("kernels_disabled_requested") or [])
        disabled_bases = {base for op, base in ALLOFF_DISABLED_OP_BASES.items() if op in requested}
        if not disabled_bases:
            violations.append(
                f"alloff: kernels_disabled_requested={sorted(requested)!r} names none of this "
                f"tier's known disable-op keys {sorted(ALLOFF_DISABLED_OP_BASES)!r} -- an alloff "
                "leg that disabled nothing recognizable cannot prove anything about the ALLOFF "
                "reference"
            )
        for base in sorted(disabled_bases):
            if base not in by_base:
                violations.append(
                    f"alloff: {base!r} (named in kernels_disabled_requested) has no dispatch-pair "
                    "counters on this leg's report at all"
                )
                continue
            fused, fallback = by_base[base]
            if fused > 0:
                violations.append(
                    f"alloff: {base} shows {fused} fused dispatch(es) despite its disabling op "
                    "being named in kernels_disabled_requested -- this leg did not actually run "
                    "with it disabled; this leg's 'alloff' classification is untrusted, discarded, "
                    "not merely annotated"
                )
            elif fallback <= 0:
                violations.append(
                    f"alloff: {base} shows fused=0 but its own fallback counter is also 0 -- no "
                    "counted fact behind the 'disabled' classification, only a claim"
                )

        # The positive training-path proof for this arm (coordinator
        # correction, replacing the eager-based shape an earlier draft of
        # this round used): `attention_block` is NOT itself named in the
        # alloff disable list -- only `attention_block_flash` is -- so the
        # disabled flash cascade must fall through to `attention_block`'s
        # own, still-ACTIVE fused kernel.
        ab_fused, _ab_eager = by_base.get("attention_block", (0, 0))
        if ab_fused <= 0:
            violations.append(
                f"alloff: attention_block_fused_dispatches={ab_fused!r} -- the disabled flash "
                "cascade must fall through to attention_block's own fused kernel (the positive "
                "training-path proof for this arm, per CASCADE_BASES' own absorption semantics); "
                "an all-zero reading is no counted fact behind the 'alloff' classification, only "
                "a claim"
            )
        return violations
    return [f"{arm}: unrecognized finetune-run arm -- cannot apply the dispatch-proof gate to it"]


def finetune_run_cross_seed_homogeneity_violations(leg_identities, lr0_labels=frozenset()):
    """Cross-seed leg-premise check (unit-63 adversarial-audit finding 3):
    every OTHER identity check in this section compares WITHIN one seed
    only -- `generic_leg_premise_violations` over `FINETUNE_RUN_IDENTITY_FIELDS`
    (called per-seed, fused-vs-alloff below) and
    `finetune_run_lr0_control_seed_violations` (also per-seed) never compare
    seed N's own premise against seed M's.

    Empirical reproduction that motivated this check: 6 seeds run against
    ONE held-out fixture text, plus 6 seeds run against a DIFFERENT one --
    each seed's own fused/alloff pair internally agreed with itself (so the
    existing per-seed check saw nothing wrong), yet the merge still read
    GREEN, silently averaging two DIFFERENT premises' `d_i` values into one
    sign test as though they were twelve draws from the same experiment.

    `leg_identities` is `[(label, fields), ...]` -- `label` a human-readable
    string identifying the leg (seed/arm/repeat), `fields` the
    `finetune_run_leg_identity`-shaped `{field: value_or_MISSING}` dict for
    THAT leg. Every `FINETUNE_RUN_IDENTITY_FIELDS` entry EXCEPT `seed`
    itself (expected, by construction, to differ across legs -- it is the
    axis the sweep varies over) must canonicalize (the SAME
    `canonicalize_identity_field` table every other identity comparison in
    this module shares) to the SAME value across EVERY leg entering the
    decision -- every main-arm r1/r2 leg across every seed, PLUS every
    lr0-control leg (the caller assembles this list; see
    `build_finetune_run_report`'s own wiring). A leg where the field folds
    to `_MISSING` (absent, or present-but-null on a non-`null-is-a-value`
    field) is its own distinct group -- never silently treated as agreeing
    with a present value, nor with a DIFFERENT absent leg's own reason for
    being absent.

    `lr0_labels` (unit-63 round-3 audit block 3, optional, default empty --
    the labels in `leg_identities` naming an lr0-control leg; see
    `build_finetune_run_report`'s own wiring) is the SECOND field-level
    exception this function carries, alongside `seed` itself: `lr` is
    IDENTITY FIELD #17 on `FINETUNE_RUN_IDENTITY_FIELDS`, and the lr=0 RED
    control's own defining premise (CONTRACT H4 advisory (b)) is that its
    legs run at `--lr 0` BY CONSTRUCTION, while every main A/B leg runs at
    the sweep's real (nonzero) `--lr`. Comparing `lr` across the FULL
    combined pool the way every OTHER field is compared would make ANY
    nonempty `lr0_seeds` list unconditionally INVALID (the control's own
    legs always diverge from the main legs' `lr`), silently defeating the
    very control CONTRACT Frame's RED-control precedent asks this project to
    run. The exception is narrow and NAMED here (never a blanket "skip lr
    checking"): `lr` is compared WITHIN the main pool (every main leg must
    still agree with every other main leg) and WITHIN the lr0-control pool
    (every lr0-control leg must still agree with every other lr0-control
    leg) SEPARATELY -- only the CROSS-group comparison is dropped, and ONLY
    for `lr`. Every other `FINETUNE_RUN_IDENTITY_FIELDS` entry (including
    the checkpoint/target_modules/schedule/... fields an lr0-control leg is
    still required to match on) is compared across the FULL combined pool
    exactly as before -- an lr0-control leg diverging on any NON-`lr` field
    still collapses this check.

    Returns a list of violation strings, one per divergent field (or, for
    `lr`, one per divergent pool), each naming the field and, for every
    distinct value found, the labels of the legs that reported it -- never a
    silent drop of which legs disagreed. Homogeneous input (including fewer
    than two legs, which cannot disagree) returns `[]`.
    """
    violations = []
    if len(leg_identities) < 2:
        return violations
    for field in FINETUNE_RUN_IDENTITY_FIELDS:
        if field == "seed":
            continue  # the swept axis -- expected to differ, never compared
        if field == "lr":
            # The lr0-control exception -- see this function's own doc.
            # Checked within each pool SEPARATELY; the cross-group
            # divergence this control's own premise REQUIRES is never
            # itself flagged.
            pools = (
                ("main", [(l, f) for l, f in leg_identities if l not in lr0_labels]),
                ("lr0-control", [(l, f) for l, f in leg_identities if l in lr0_labels]),
            )
            for pool_name, pool in pools:
                if len(pool) < 2:
                    continue
                groups = {}
                for label, fields in pool:
                    value = fields.get(field, _MISSING)
                    display = "<absent-or-null>" if value is _MISSING else canonicalize_identity_field(field, value)
                    key = repr(display)
                    entry = groups.setdefault(key, (display, []))
                    entry[1].append(label)
                if len(groups) > 1:
                    parts = "; ".join(f"{display!r} on {labels}" for display, labels in groups.values())
                    violations.append(
                        f"cross-seed leg-identity field 'lr' diverges within the {pool_name} pool "
                        "(unit-63 round-3 audit block 3: an lr0-control-vs-main divergence in "
                        "'lr' is that control's own defining premise and is never compared across "
                        f"that boundary, but every leg WITHIN the {pool_name} pool must still "
                        f"agree): {parts}"
                    )
            continue
        groups = {}  # repr(display_value) -> (display_value, [labels])
        for label, fields in leg_identities:
            value = fields.get(field, _MISSING)
            display = "<absent-or-null>" if value is _MISSING else canonicalize_identity_field(field, value)
            key = repr(display)
            entry = groups.setdefault(key, (display, []))
            entry[1].append(label)
        if len(groups) > 1:
            parts = "; ".join(f"{display!r} on {labels}" for display, labels in groups.values())
            violations.append(
                f"cross-seed leg-identity field {field!r} diverges across legs entering the "
                "decision (unit-63 audit finding 3 -- every leg's premise for this field must "
                f"be the SAME, not merely fused/alloff agreement within one seed): {parts}"
            )
    return violations


def load_finetune_run_leg(raw_dir, seed, arm, repeat):
    """Read one `finetune_run_ab.sh`-produced `.exit`/`.json`/`.stderr`
    triple, named `seed{seed}__{arm}__{repeat}` (mirrors `finetune_ab.sh`'s
    own `{config_slug}__{leg}` naming, with `seed{seed}` as this producer's
    own config axis). Never raises: a MISSING/FAIL/DRY_RUN leg is a normal
    row, not a script error -- same discipline as `load_leg` above.
    """
    base = os.path.join(raw_dir, f"seed{seed}__{arm}__{repeat}")
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
        return {"outcome": "FAIL", "err_tail": err_tail, "report": None}

    return {"outcome": "OK", "err_tail": "", "report": report}


def finetune_run_lr0_control_seed_violations(raw_dir, seed):
    """The lr=0 RED control's own per-seed check (unit-63 audit advisory
    (b)): reads BOTH arms' `FINETUNE_RUN_LR0_REPEAT`-tagged leg for `seed`
    (never `r1`/`r2` -- these never enter `load_finetune_run_leg`'s normal
    `FINETUNE_RUN_REPEATS` iteration, so they can never leak into the A/B
    set's own `d_values`/sign test) and asserts:

      1. (unit-63 round-4 audit F-2) THE CONTROL'S OWN DEFINING FACT: each
         OK leg's reported `lr` field must equal `0.0` EXACTLY. The lr
         exception in `finetune_run_cross_seed_homogeneity_violations`
         (unit-63 round-3 audit block 3) removed the CROSS-GROUP `lr`
         comparison between the main pool and the lr0-control pool -- by
         design, that is the control's own defining premise, never itself
         a divergence to flag -- but nothing UNTIL this check asserted the
         POSITIVE fact the whole control depends on: that the leg tagged as
         an lr0-control leg actually ran at `lr == 0.0`. A control whose
         own `--lr` flag silently reverted to the CLI default (or was
         mis-plumbed to some other nonzero value) would validate the
         FLOOR=0.0 ruling against a leg that never tested it at all --
         "a control that never ran at lr=0 validates the floor silently".
      2. `learning_happened_delta` (amendment 2026-08-29b: DERIVED from the
         leg's own raw `train_probe_series`, via `finetune_run_probe_series_delta`
         -- never a pre-derived scalar, never assumed-good) does NOT clear
         `FINETUNE_RUN_LEARNING_HAPPENED_FLOOR` -- the calibration bite for
         the FLOOR=0.0 ruling: a passing lr=0 leg (learning "happened" with
         no possible parameter update) is a finding against the floor
         itself, not a training result. A leg whose series is missing,
         too-short, non-finite, or v1-scalar-only (see that function's own
         doc) is ALSO a violation here -- an unresolvable premise leaves the
         floor just as unvalidated as a leg that outright passed it.

    All checks are independent and conjunctive -- a leg can fail any subset
    of them. A leg that is `MISSING`/`FAIL` (never ran, or ran and errored)
    is ALSO recorded as a violation -- an absent control leaves the floor
    unvalidated, which this calibration check exists specifically to never
    pass over silently; a `DRY_RUN` leg is the one carve-out (never itself a
    finding, same doctrine every other `*_DRY_RUN` leg in this module
    already gets).

    Returns `(violations, per_arm, identities)` where `per_arm` records each
    arm's raw outcome and (when OK) its derived `learning_happened_delta`,
    `lr`, raw `train_probe_series`, and `failing_premises` (the structured
    names `premise_failure_diagnostic` reads, amendment 2026-08-29b item
    1(c)) -- never silently dropped even when clean -- and `identities` is
    `[(label, fields), ...]` (unit-63 audit finding 3) for every OK leg
    here, in the exact shape `finetune_run_cross_seed_homogeneity_violations`
    consumes, so the caller can fold the lr0 control's own legs into that
    check alongside the main A/B seeds without a second,
    independently-drifting loading path.
    """
    violations = []
    per_arm = {}
    identities = []
    for arm in FINETUNE_RUN_ARMS:
        leg = load_finetune_run_leg(raw_dir, seed, arm, FINETUNE_RUN_LR0_REPEAT)
        outcome = leg["outcome"]
        if outcome == "DRY_RUN":
            per_arm[arm] = {
                "outcome": outcome,
                "learning_happened_delta": None,
                "lr": None,
                "train_probe_series": None,
                "failing_premises": [],
            }
            continue
        if outcome != "OK":
            per_arm[arm] = {
                "outcome": outcome,
                "learning_happened_delta": None,
                "lr": None,
                "train_probe_series": None,
                "failing_premises": ["leg_outcome"],
            }
            violations.append(
                f"lr0 control seed {seed} {arm}: leg outcome={outcome!r} (not OK) -- an "
                "absent/failed lr=0 control leg leaves the FLOOR=0.0 premise unvalidated "
                "(CONTRACT H4 advisory (b))"
            )
            continue
        tier = finetune_run_block(leg["report"])
        lr = tier.get("lr")
        label = f"lr0 seed {seed} {arm}"
        series_violations, delta, series = finetune_run_probe_series_delta(label, tier)
        failing_premises = []
        identities.append((label, finetune_run_leg_identity(tier)))
        # unit-63 round-4 audit F-2: the control's own defining fact -- see
        # this function's own doc, point 1.
        if not (isinstance(lr, (int, float)) and not isinstance(lr, bool) and lr == 0.0):
            failing_premises.append("lr_nonzero")
            violations.append(
                f"lr0 control seed {seed} {arm}: reported lr={lr!r}, not exactly 0.0 -- a "
                "control that never ran at lr=0 validates the floor silently (unit-63 round-4 "
                "audit F-2; CONTRACT H4 advisory (b)'s own 'lr=0 arm x2 seeds' precondition)"
            )
        if series_violations:
            failing_premises.append("learning_happened")
            violations += series_violations  # already labeled ("lr0 seed {seed} {arm}: ...")
        elif delta > FINETUNE_RUN_LEARNING_HAPPENED_FLOOR:
            failing_premises.append("learning_happened")
            violations.append(
                f"lr0 control seed {seed} {arm}: learning_happened_delta={delta!r} "
                f"unexpectedly CLEARS the floor ({FINETUNE_RUN_LEARNING_HAPPENED_FLOOR}) under "
                "lr=0 -- a passing lr=0 control leg is a finding against the "
                "FLOOR=0.0 premise-leg ruling itself, not a training result "
                "(CONTRACT H4 advisory (b))"
            )
        per_arm[arm] = {
            "outcome": outcome,
            "learning_happened_delta": delta,
            "lr": lr,
            "train_probe_series": series,
            "failing_premises": failing_premises,
        }
    return violations, per_arm, identities


def build_finetune_run_report(raw_dir, seeds, lr0_seeds=(), allow_missing_lr0_control=False):
    """The finetune-run merge stage (unit 63 H4b): reads every
    `(seed, arm, repeat)` leg `finetune_run_ab.sh` wrote under `raw_dir`,
    computes the exact sign test over `d_i = fused.held_out_example_mean -
    alloff.held_out_example_mean` (the `r1` legs only -- the primary,
    un-repeated measurement each seed contributes to the decision; `r2` legs
    feed the determinism floor below, never the sign test itself), the
    conjunctive leg-premise refusal (cross-arm identity match via
    `generic_leg_premise_violations`, PLUS each arm's own
    `finetune_run_arm_premise_violations`), and the `r1`/`r2` same-seed
    determinism-floor delta (CONTRACT H4/PLAN.md v2 delta 6: MEASURED AND
    REPORTED always, RED only if it exceeds the cross-seed spread of `d_i`
    -- "no CUDA bitwise contract exists -- the measurement must be able to
    report what it measures").

    A seed with ANY premise violation still has its raw legs, `d_i` (when
    computable), and violations RECORDED in `per_seed` -- never silently
    dropped -- but its `d_i` is EXCLUDED from `d_values`/the sign test
    (an untrusted premise means the measurement itself is not known to
    describe the same comparison the other seeds' do), and the overall
    `status` becomes `INVALID` (the same "a correctness-of-measurement
    problem REPLACES, never merely annotates, the verdict" carve-out
    `build_report`'s own `INVALID` branch already establishes for
    `finetune_ab.sh`).

    `lr0_seeds` (unit-63 audit advisory (b), optional, default empty) names
    the lr=0 RED control's own seeds -- read via
    `finetune_run_lr0_control_seed_violations`, NEVER folded into `seeds`,
    `d_values`, or the sign test; a violation there (a missing/failed
    control leg, or one that passes learning-happened under lr=0) also
    collapses `status` to `INVALID`, alongside the premise/determinism-floor
    carve-outs above.

    `allow_missing_lr0_control` (unit-63 round-3 audit block 5, default
    `False`): CONTRACT Frame's own RED control ("lr=0 arm x2 seeds fails
    learning-happened") is PRE-REGISTERED, not optional -- an empty
    `lr0_seeds` used to be silently treated as a no-op (`gpu-howwell.yml`'s
    own `|| ''` collapsed "no input exists" and "an operator explicitly
    opted out" to the same untagged empty string on a `workflow_dispatch`
    run where the operator cleared the field -- unit-63 round-4 audit
    advisory (A2) corrects an earlier draft of this comment, which claimed
    this ALSO dropped the control on every label-triggered run: a
    `pull_request: types: [labeled]` trigger carries no workflow_dispatch
    inputs at all, so that path already refused loudly on the (also
    unset) `model_dir` input before ever reaching the `lr0_seeds` coalesce
    -- a run that could never start in the first place, not a silent
    drop). This function now REFUSES (`status` collapses to `INVALID`,
    naming the reason in `lr0_control.violations`) when `lr0_seeds` is
    empty UNLESS the caller passes `allow_missing_lr0_control=True`
    (`main`'s own `--allow-missing-lr0-control` CLI flag;
    `finetune_run_ab.sh` passes it only when
    `FINETUNE_RUN_AB_ALLOW_NO_LR0=1`) -- the skip is then a VISIBLE, RECORDED
    act (`lr0_control.allow_missing_lr0_control: true` in the merged
    artifact), never an unstated default.

    The DECISION RULE itself (unit-63 audit finding 1 --
    `FINETUNE_RUN_DECISION_RULE_TEXT`'s own doc has the exact predicate) is
    applied here, never left as a computed-but-unused `sign_test` result:
    a premise-clean seed count other than `FINETUNE_RUN_GATE_SEED_COUNT`
    (12) is INVALID; otherwise RED (degradation-concordant, fused worse),
    RED_FOR_INVESTIGATION (improvement-concordant, fused better -- anomalous
    improvement is investigated, never silently celebrated), or GREEN.

    Unit-63 audit finding 3: on top of the per-seed cross-ARM identity check
    above, EVERY OK leg entering the decision here -- every main-arm r1/r2
    leg across every seed in `seeds`, plus every lr0-control leg across
    `lr0_seeds` -- is also fed to
    `finetune_run_cross_seed_homogeneity_violations`, which requires every
    `FINETUNE_RUN_IDENTITY_FIELDS` entry except `seed` itself to agree
    across ALL of them, never just fused-vs-alloff within one seed. A
    divergence there ALSO collapses `status` to `INVALID`, alongside the
    other correctness-of-measurement carve-outs above.

    Returns `(merged, table)`, or `(None, None)` if not one `seed` produced
    any leg output at all (an empty sweep).
    """
    seeds = list(seeds)
    if not seeds:
        return None, None

    per_seed = {}
    determinism_deltas = []  # [(seed, arm, |r1-r2|)] for every OK r1/r2 pair
    any_leg_found = False
    any_dry_run_leg = False
    # unit-63 audit finding 3 -- [(label, fields), ...] for every OK leg
    # entering the decision, across every seed and repeat; extended with the
    # lr0-control legs below, then fed to
    # `finetune_run_cross_seed_homogeneity_violations` as one flat pool.
    cross_seed_leg_identities = []
    # amendment 2026-08-29b item 1(c) -- `premise_failure_diagnostic`'s own
    # accumulator: one entry per (seed, arm) leg that failed ANY CONTRACT
    # Frame premise leg (`admission_is_dense`/`learning_happened`/
    # `tie_fraction`), main-pool and lr0-control legs alike. ALWAYS present
    # in the merged artifact, even empty -- never conditionally omitted.
    premise_failure_entries = []

    for seed in seeds:
        entries = {
            (arm, repeat): load_finetune_run_leg(raw_dir, seed, arm, repeat)
            for arm in FINETUNE_RUN_ARMS
            for repeat in FINETUNE_RUN_REPEATS
        }
        # unit-63 audit finding 2's merger half -- every OK leg's own
        # dispatch-proof gate (see `finetune_run_dispatch_proof_violations`'s
        # own doc), accumulated per seed alongside its cross-seed identity
        # record; folded into `violations` below so an untrusted
        # classification EXCLUDES this seed's d_i from the decision, the
        # same "discarded, not merely annotated" carve-out every other leg
        # check in this section already gets.
        dispatch_proof_violations_for_seed = []
        for (arm, repeat), entry in entries.items():
            if entry["outcome"] == "OK":
                tier = finetune_run_block(entry["report"])
                cross_seed_leg_identities.append(
                    (f"seed {seed} {arm} {repeat}", finetune_run_leg_identity(tier))
                )
                dispatch_proof_violations_for_seed += [
                    f"seed {seed} {arm} {repeat}: {v}"
                    for v in finetune_run_dispatch_proof_violations(arm, tier)
                ]
        if any(e["outcome"] != "MISSING" for e in entries.values()):
            any_leg_found = True
        if any(e["outcome"] == "DRY_RUN" for e in entries.values()):
            any_dry_run_leg = True

        seed_out = {
            "legs": {
                f"{arm}_{repeat}": entries[(arm, repeat)]["outcome"]
                for arm in FINETUNE_RUN_ARMS
                for repeat in FINETUNE_RUN_REPEATS
            }
        }

        fused_r1, alloff_r1 = entries[("fused", "r1")], entries[("alloff", "r1")]
        violations = list(dispatch_proof_violations_for_seed)
        d_i = None
        trajectory = {}
        if fused_r1["outcome"] == "OK" and alloff_r1["outcome"] == "OK":
            fused_tier = finetune_run_block(fused_r1["report"])
            alloff_tier = finetune_run_block(alloff_r1["report"])

            fused_id = finetune_run_leg_identity(fused_tier)
            alloff_id = finetune_run_leg_identity(alloff_tier)
            violations += generic_leg_premise_violations(
                FINETUNE_RUN_IDENTITY_FIELDS, fused_id, alloff_id, "fused", "alloff"
            )
            fused_named_violations = finetune_run_named_arm_premise_violations("fused", fused_tier)
            alloff_named_violations = finetune_run_named_arm_premise_violations("alloff", alloff_tier)
            violations += [msg for _name, msg in fused_named_violations]
            violations += [msg for _name, msg in alloff_named_violations]
            # amendment 2026-08-29b item 1(c) -- see `premise_failure_entries`'s
            # own doc above: recorded per (seed, arm) leg, never per-seed
            # only, so the diagnostic can distinguish WHICH arm's premise
            # failed.
            for arm_label, arm_tier, named_violations in (
                ("fused", fused_tier, fused_named_violations),
                ("alloff", alloff_tier, alloff_named_violations),
            ):
                if named_violations:
                    premise_failure_entries.append({
                        "label": f"seed {seed} {arm_label} r1",
                        "pool": "main",
                        "seed": seed,
                        "failing_premises": sorted({name for name, _msg in named_violations}),
                        "train_probe_series": arm_tier.get("train_probe_series"),
                    })

            d_i = fused_tier["held_out_example_mean"] - alloff_tier["held_out_example_mean"]
            trajectory = {
                "fused": fused_tier.get("trajectory"),
                "alloff": alloff_tier.get("trajectory"),
            }
        elif fused_r1["outcome"] == "DRY_RUN" or alloff_r1["outcome"] == "DRY_RUN":
            # A dry-run leg never claims a real number and is never itself
            # a finding -- see `finetune_run_ab.sh`'s own `*_DRY_RUN`
            # contract (mirrors `finetune_ab.sh`'s `any_dry_run` carve-out
            # for its own ratio-based verdict). No violation recorded; this
            # seed simply contributes no `d_i`.
            pass
        else:
            not_ok = [f"{arm}/r1" for arm, e in (("fused", fused_r1), ("alloff", alloff_r1)) if e["outcome"] != "OK"]
            violations.append(f"seed {seed}: r1 leg(s) not OK for {not_ok} -- cannot compute d_i or check premise")

        for arm in FINETUNE_RUN_ARMS:
            r1e, r2e = entries[(arm, "r1")], entries[(arm, "r2")]
            if r1e["outcome"] == "OK" and r2e["outcome"] == "OK":
                r1_mean = finetune_run_block(r1e["report"])["held_out_example_mean"]
                r2_mean = finetune_run_block(r2e["report"])["held_out_example_mean"]
                delta = abs(r1_mean - r2_mean)
                determinism_deltas.append((seed, arm, delta))
                seed_out.setdefault("r1_r2_delta", {})[arm] = delta

        seed_out["leg_premise_violations"] = violations
        seed_out["d_i"] = d_i
        seed_out["trajectory"] = trajectory
        per_seed[seed] = seed_out

    if not any_leg_found:
        return None, None

    # The lr=0 RED control (unit-63 audit advisory (b)): a SEPARATE seed
    # axis, read via `FINETUNE_RUN_LR0_REPEAT`-tagged legs only, never folded
    # into `seeds`/`d_values`/the sign test above.
    lr0_seeds = list(lr0_seeds)
    lr0_control_violations = []
    lr0_control_per_seed = {}
    # unit-63 round-3 audit block 5 -- see this function's own doc: an empty
    # `lr0_seeds` is a REFUSAL (INVALID) unless the caller explicitly opted
    # out. Recorded here, in `lr0_control_violations`, so it collapses
    # `status` the SAME way every other lr0-control finding does (this
    # function's status computation already ORs `lr0_control_violations`
    # into the INVALID branch -- no separate flag needed there).
    if not lr0_seeds and not allow_missing_lr0_control:
        lr0_control_violations.append(
            "lr0_seeds is empty and allow_missing_lr0_control was not set -- CONTRACT Frame's own "
            "RED control ('lr=0 arm x2 seeds fails learning-happened') is pre-registered, not "
            "optional; skipping it silently would leave the learning-happened floor "
            "(FINETUNE_RUN_LEARNING_HAPPENED_FLOOR=0.0) unvalidated on this run. Pass "
            "--allow-missing-lr0-control (finetune_run_ab.sh's own FINETUNE_RUN_AB_ALLOW_NO_LR0=1) "
            "to record a deliberate, visible opt-out instead."
        )
    # unit-63 round-3 audit block 3 -- labels (from `identities` below)
    # naming an lr0-control leg, threaded into
    # `finetune_run_cross_seed_homogeneity_violations` so its own `lr`
    # exception (see that function's own doc) knows which legs in the
    # combined pool are the control's, never inferred from label spelling.
    lr0_labels = set()
    for lr0_seed in lr0_seeds:
        seed_violations, per_arm, identities = finetune_run_lr0_control_seed_violations(raw_dir, lr0_seed)
        lr0_control_violations += seed_violations
        lr0_control_per_seed[lr0_seed] = {"per_arm": per_arm, "violations": seed_violations}
        cross_seed_leg_identities += identities
        lr0_labels.update(label for label, _fields in identities)
        # amendment 2026-08-29b item 1(c) -- lr0-control legs feed the SAME
        # diagnostic accumulator as the main pool (tagged pool="lr0-control"),
        # never a second, separately-shaped block.
        for arm_label, arm_rec in per_arm.items():
            if arm_rec.get("failing_premises"):
                premise_failure_entries.append({
                    "label": f"lr0 seed {lr0_seed} {arm_label}",
                    "pool": "lr0-control",
                    "seed": lr0_seed,
                    "failing_premises": list(arm_rec["failing_premises"]),
                    "train_probe_series": arm_rec.get("train_probe_series"),
                })

    # unit-63 audit finding 3 -- see this function's own doc; every OK leg
    # gathered above (main A/B seeds' r1/r2, plus the lr0 control's own
    # legs) must agree on every FINETUNE_RUN_IDENTITY_FIELDS entry except
    # `seed` itself (and, per block 3's own `lr` exception, `lr` is compared
    # within each of the main/lr0-control pools separately, never across
    # that boundary).
    cross_seed_violations = finetune_run_cross_seed_homogeneity_violations(cross_seed_leg_identities, lr0_labels=lr0_labels)

    # Cross-seed spread: population stdev of every seed's `d_i` that could
    # be COMPUTED at all (even one from a premise-violating seed -- the
    # spread describes how much seeds naturally disagree, which is a fact
    # about the measurement independent of whether THIS merge trusts it for
    # the sign test). Falls back to 0.0 with fewer than 2 such seeds
    # (nothing to spread; the determinism floor then RED's on ANY nonzero
    # r1/r2 delta, the honest floor of "no spread to compare against").
    all_d = [per_seed[s]["d_i"] for s in per_seed if per_seed[s]["d_i"] is not None]
    cross_seed_spread = statistics.pstdev(all_d) if len(all_d) >= 2 else 0.0

    determinism_floor_findings = []
    max_determinism_delta = 0.0
    for seed, arm, delta in determinism_deltas:
        max_determinism_delta = max(max_determinism_delta, delta)
        if delta > cross_seed_spread:
            determinism_floor_findings.append(
                f"seed {seed} {arm}: r1/r2 delta {delta} exceeds the cross-seed spread "
                f"{cross_seed_spread} (CONTRACT H4/PLAN.md v2 delta 6: reported and RED)"
            )

    d_values = {s: per_seed[s]["d_i"] for s in per_seed if per_seed[s]["d_i"] is not None and not per_seed[s]["leg_premise_violations"]}

    sign_result = None
    sign_error = None
    if d_values:
        try:
            sign_result = sign_test(list(d_values.values()))
        except SignTestError as exc:
            sign_error = str(exc)

    any_premise_violation = any(per_seed[s]["leg_premise_violations"] for s in per_seed)

    # The decision rule itself (unit-63 audit finding 1): applied here, on
    # top of the sign test `sign_result` already computed above, never left
    # as a computed-but-unused diagnostic. `clean_seed_count` is the count
    # the pre-registered rule is actually FOR -- every seed contributing a
    # premise-clean `d_i` to `d_values` -- deliberately distinct from
    # `sign_result["n"]` (which excludes exact ties).
    clean_seed_count = len(d_values)
    decision = None
    wrong_seed_count = False
    if sign_result is not None:
        n_pos, n_neg = sign_result["n_pos"], sign_result["n_neg"]
        mean_d = statistics.mean(d_values.values())
        pos_dominant = n_pos >= FINETUNE_RUN_DECISION_THRESHOLD
        neg_dominant = n_neg >= FINETUNE_RUN_DECISION_THRESHOLD
        # `pos_dominant`/`neg_dominant` can never BOTH hold at once here:
        # n_pos + n_neg <= clean_seed_count, and 2 * FINETUNE_RUN_DECISION_THRESHOLD
        # (22) exceeds FINETUNE_RUN_GATE_SEED_COUNT (12) -- the AND below is
        # therefore never ambiguous about WHICH direction it is checking.
        if pos_dominant and mean_d > 0.0:
            concordant_direction = "degradation"  # d_i > 0 dominant: fused WORSE than alloff
        elif neg_dominant and mean_d < 0.0:
            concordant_direction = "improvement"  # d_i < 0 dominant: fused BETTER than alloff
        else:
            concordant_direction = "none"
        decision = {
            "d_i": list(d_values.values()),
            "clean_seed_count": clean_seed_count,
            "gate_seed_count": FINETUNE_RUN_GATE_SEED_COUNT,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "ties": sign_result["ties"],
            "mean_d": mean_d,
            "threshold": FINETUNE_RUN_DECISION_THRESHOLD,
            "concordant_direction": concordant_direction,
            "alpha2": FINETUNE_RUN_ALPHA2,
            "p_value": sign_result["p_value"],
            "rule": FINETUNE_RUN_DECISION_RULE_TEXT,
        }
        wrong_seed_count = clean_seed_count != FINETUNE_RUN_GATE_SEED_COUNT

    if (
        any_premise_violation
        or determinism_floor_findings
        or sign_error
        or lr0_control_violations
        or wrong_seed_count
        or cross_seed_violations
    ):
        status = FINETUNE_RUN_STATUS_INVALID
    elif sign_result is None:
        # `any_dry_run_leg` (never a premise violation, see the DRY_RUN
        # branch above) is distinguished from a genuine INCOMPLETE (a real
        # FAIL/MISSING leg with no violation recorded, e.g. an r1-only
        # partial run) -- a dry run must never itself fail this merge's own
        # exit code (finetune_ab.sh's own `any_dry_run` -> "N/A (dry-run)"
        # carve-out, never INVALID/FAIL), while a genuinely incomplete real
        # sweep still reads INCOMPLETE.
        status = FINETUNE_RUN_STATUS_DRY_RUN if any_dry_run_leg else FINETUNE_RUN_STATUS_INCOMPLETE
    elif decision["concordant_direction"] == "degradation":
        status = FINETUNE_RUN_STATUS_RED
    elif decision["concordant_direction"] == "improvement":
        status = FINETUNE_RUN_STATUS_RED_FOR_INVESTIGATION
    else:
        status = FINETUNE_RUN_STATUS_GREEN

    # Unit-63 round-16 audit (identity-completeness): the producer-side
    # runtime belt -- `status` above is now always assigned FROM the named
    # `FINETUNE_RUN_STATUS_*` constants (never a re-typed literal), but this
    # explicit membership check is what actually MAKES a future fold branch
    # that reintroduces a hand-typed literal (or a genuinely new status not
    # yet added to `FINETUNE_RUN_STATUSES`) fail LOUDLY, at the point of
    # production, rather than silently flowing into the artifact and only
    # ever being caught downstream by `runpod_gpu_howwell.sh`'s own
    # catch-all `*)` warning arm (which does not gate). Mirrors
    # `DOSE_LADDER_EXIT_CAUSE_NAMES`'s own runtime equality check in `main()`
    # below: an explicit `if`/`raise`, never a bare `assert` (stripped under
    # `python -O`).
    if status not in FINETUNE_RUN_STATUSES:
        raise AssertionError(
            f"build_finetune_run_report computed status={status!r}, not a member of the "
            f"committed FINETUNE_RUN_STATUSES={FINETUNE_RUN_STATUSES} -- a new status must be "
            "added to that tuple (and to runpod_gpu_howwell.sh's own case arms) in the SAME "
            "change that introduces it here, never assigned as a bare literal"
        )

    # amendment 2026-08-29b item 1(c): `premise_failure_diagnostic` is
    # ALWAYS present in the merged artifact (even when `premise_failure_entries`
    # is empty) -- non-parameterised (no threshold this block itself takes),
    # explicitly non-decisional (it can never promote an INVALID verdict to
    # GREEN; `status` above is computed entirely without reading this
    # block), and carries NO operator override anywhere: no
    # `--allow-premise-failure` flag, no waived-seed list, no rescale switch.
    premise_failure_diagnostic = {
        "failed_seeds": sorted({entry["seed"] for entry in premise_failure_entries}, key=str),
        "failing_legs": premise_failure_entries,
        "note": (
            "Non-parameterised, explicitly non-decisional (CONTRACT amendment 2026-08-29b item "
            "1(c)): names which CONTRACT Frame premise leg(s) failed on which leg, with that "
            "leg's raw train_probe_series, for investigation only. This block can NEVER promote "
            "an INVALID verdict to GREEN -- the 'status' field above is computed independently "
            "of it -- and the merger accepts NO operator override for a premise failure anywhere "
            "(no --allow-premise-failure, no waived-seed list, no rescale switch)."
        ),
    }

    merged = {
        "seeds": seeds,
        "arms": list(FINETUNE_RUN_ARMS),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "d_values": {str(s): d for s, d in d_values.items()},
        "premise_failure_diagnostic": premise_failure_diagnostic,
        "cross_seed_spread": cross_seed_spread,
        "determinism_floor": {
            "deltas": [{"seed": s, "arm": a, "delta": d} for s, a, d in determinism_deltas],
            "max_delta": max_determinism_delta,
            "cross_seed_spread": cross_seed_spread,
            "findings": determinism_floor_findings,
            "note": (
                "r1/r2 same-seed repeat delta MEASURED AND REPORTED as the determinism floor "
                "-- RED only if it exceeds the cross-seed spread of d_i (CONTRACT H4/PLAN.md "
                "v2 delta 6: 'no CUDA bitwise contract exists -- the measurement must be able "
                "to report what it measures')."
            ),
        },
        "sign_test": sign_result,
        "sign_test_error": sign_error,
        "decision": decision,
        "wrong_seed_count": wrong_seed_count,
        # unit-63 audit finding 3 -- see `finetune_run_cross_seed_homogeneity_violations`'s
        # own doc. Never a silent drop even when clean (empty list).
        "cross_seed_identity_violations": cross_seed_violations,
        "lr0_control": {
            "seeds": lr0_seeds,
            "per_seed": {str(s): v for s, v in lr0_control_per_seed.items()},
            "violations": lr0_control_violations,
            # unit-63 round-3 audit block 5: the skip is now a VISIBLE,
            # RECORDED act, never an unstated default -- see this function's
            # own doc.
            "allow_missing_lr0_control": allow_missing_lr0_control,
            "note": (
                "lr=0 RED control (CONTRACT Frame / audit advisory (b)): separate seeds, "
                "never counted into the A/B set's own d_values/sign test above. A clean "
                "control leg FAILS learning-happened (its own learning_happened_delta does "
                "not clear FINETUNE_RUN_LEARNING_HAPPENED_FLOOR); a passing leg is recorded "
                "as a violation against the FLOOR=0.0 premise-leg ruling itself. An empty "
                "'seeds' list is itself a refusal (INVALID) unless 'allow_missing_lr0_control' "
                "is true (block 5) -- the pre-registered control is not silently optional."
            ),
        },
        "status": status,
    }

    lines = [
        "# finetune-run A/B -- fused cascade vs ALLOFF, exact two-sided sign test over d_i "
        "= fused - alloff (CONTRACT Frame / C16)",
        f"{'seed':<8}{'fused_r1':<10}{'alloff_r1':<10}{'d_i':<14}{'premise':<12}",
    ]
    for seed in seeds:
        so = per_seed[seed]
        d_i = so["d_i"]
        d_s = "n/a" if d_i is None else f"{d_i:.6f}"
        premise_s = "clean" if not so["leg_premise_violations"] else f"VIOLATED ({len(so['leg_premise_violations'])})"
        lines.append(f"{seed:<8}{so['legs']['fused_r1']:<10}{so['legs']['alloff_r1']:<10}{d_s:<14}{premise_s:<12}")
    lines.append("")
    if sign_error is not None:
        lines.append(f"sign_test: ERROR -- {sign_error}")
    elif sign_result is not None:
        lines.append(
            f"sign_test: n={sign_result['n']} n_pos={sign_result['n_pos']} "
            f"n_neg={sign_result['n_neg']} ties={sign_result['ties']} "
            f"p_value={sign_result['p_value']}"
        )
    else:
        lines.append("sign_test: n/a -- no seed produced a premise-clean d_i")
    lines.append(
        f"determinism_floor: max_r1_r2_delta={max_determinism_delta} "
        f"cross_seed_spread={cross_seed_spread} findings={len(determinism_floor_findings)}"
    )
    if decision is not None:
        lines.append(
            f"decision: clean_seed_count={decision['clean_seed_count']}/{decision['gate_seed_count']} "
            f"threshold={decision['threshold']} concordant_direction={decision['concordant_direction']} "
            f"mean_d={decision['mean_d']:.6f} alpha2={decision['alpha2']} p_value={decision['p_value']}"
        )
    else:
        lines.append("decision: n/a -- no sign_test result to decide over")
    if lr0_seeds:
        lines.append(
            f"lr0_control: seeds={lr0_seeds} violations={len(lr0_control_violations)}"
        )
    if cross_seed_violations:
        lines.append(f"cross_seed_identity_violations: {len(cross_seed_violations)}")
        for v in cross_seed_violations:
            lines.append(f"  - {v}")
    lines.append(f"status: {status}")
    table = "\n".join(lines)
    return merged, table


# ============================================================================
# unit 63 amendment 2026-08-29b item 3 -- mutant dose-ladder merge mode.
#
# CONTRACT amendment 2026-08-29b item 3 pre-registers a one-parameter,
# monotone, SUSTAINED mutant dose family (the fused AdamW update scaled by
# `(1+eps)`; see `docs/plans/63-how-well/mutants/README.md` for the mutant's
# own patch/hash/on-pod-procedure doc), replacing M1's "sensitivity bound"
# claim (mutants/README.md's own post-hoc finding: M1 is a sign-flipping
# early transient, a NON-DETECTION, never a bound). CONTRACT.md addendum
# 2026-08-29c SIGNS the family: the pre-spend prediction table (also in
# mutants/README.md) falsified the original positive-only ladder's direction
# (predicted IMPROVEMENT, not degradation) before any spend, so the
# SCHEDULED ladder is `eps in {-0.50, -0.10, +0.50}` -- `eps in {0.02, 0.10}`
# stay committed as the falsified-but-recorded doses, never scheduled. Each
# dose is run as its OWN column of `fused`-arm legs (the mutant substituted
# INTO the fused arm, mutants/README.md's own on-pod procedure step 4/6) and
# merged, HERE, against the campaign's ALREADY-RUN `alloff` r1 legs -- the
# SAME alloff legs the main A/B decision consumed, never a second,
# independently-run alloff pool -- under the EXACT SAME `>=11/12` threshold
# and mean-sign rule the primary decision rule uses
# (`FINETUNE_RUN_DECISION_THRESHOLD`/`FINETUNE_RUN_GATE_SEED_COUNT` --
# reused by reference below, never re-declared, so the two rules cannot
# independently drift). Mutant legs NEVER enter the primary A/B set: this
# entire mechanism reads its own `mutant-<dose_label>`-tagged leg files
# (`mutant_leg_repeat_tag`), a `repeat` value that can never collide with
# `r1`/`r2` (the main pool) or `FINETUNE_RUN_LR0_REPEAT` (the lr=0 control)
# by construction -- the SAME file-naming isolation `FINETUNE_RUN_LR0_REPEAT`
# already gives the lr0 control, not a second bespoke mechanism.
# ============================================================================

# Reused BY REFERENCE from the primary decision rule -- see that rule's own
# doc (`FINETUNE_RUN_DECISION_RULE_TEXT`). A dose column is judged under the
# IDENTICAL numeric cell the main campaign is (CONTRACT amendment
# 2026-08-29b item 3: "under the SAME >=11/12+mean rule").
MUTANT_DECISION_THRESHOLD = FINETUNE_RUN_DECISION_THRESHOLD
MUTANT_GATE_SEED_COUNT = FINETUNE_RUN_GATE_SEED_COUNT

# Unit-63 round-16 audit (identity-completeness, sibling of the
# FINETUNE_RUN_STATUS_* fix above): the dose-column `detected` vocabulary --
# every value `build_mutant_dose_column` (below) can set -- was hand-typed at
# every production AND consumption site (this module's own assignment and
# comparison sites, PLUS `howwell_dose_ladder_cause.py`'s own re-typed
# `"INVALID"` literal) with no named source. Named ONCE here; every site in
# THIS module that assigns or compares a dose column's `detected` field reads
# these constants, never a re-typed literal, and `howwell_dose_ladder_cause.py`
# imports `MUTANT_DOSE_DETECTED_INVALID` from here directly (exactly as it
# already imports `DOSE_LADDER_EXIT_CAUSE_NAMES`), so the two can never
# independently drift on what `"INVALID"` (or any other member) means. This
# does NOT carry a runtime membership guard the way `FINETUNE_RUN_STATUS_*`
# does above -- `build_mutant_dose_column`'s own fold is a strict linear
# INVALID -> RED/RED_FOR_INVESTIGATION -> not-detected precedence (each
# branch's own `detected != MUTANT_DOSE_DETECTED_INVALID` guard already makes
# the four-way exclusivity structural, not merely conventional the way the
# six-way `FINETUNE_RUN_STATUS_*` if/elif chain was) -- stated only that far,
# not further.
MUTANT_DOSE_DETECTED_NOT_DETECTED = "not-detected"
MUTANT_DOSE_DETECTED_INVALID = "INVALID"
MUTANT_DOSE_DETECTED_RED = "RED"
MUTANT_DOSE_DETECTED_RED_FOR_INVESTIGATION = "RED_FOR_INVESTIGATION"
MUTANT_DOSE_DETECTED_VALUES = (
    MUTANT_DOSE_DETECTED_NOT_DETECTED,
    MUTANT_DOSE_DETECTED_INVALID,
    MUTANT_DOSE_DETECTED_RED,
    MUTANT_DOSE_DETECTED_RED_FOR_INVESTIGATION,
)

# Unit-63 round-16 audit (identity-completeness, same sibling class): the
# RED-proof verdict's own `"NOT_PROVEN"` prefix (`build_red_proof_summary`,
# below) is re-typed at `howwell_dose_ladder_cause.py`'s own
# `red_proof_verdict.startswith("NOT_PROVEN")` check -- named ONCE here,
# imported there directly.
RED_PROOF_VERDICT_PROVEN = "PROVEN"
RED_PROOF_VERDICT_NOT_PROVEN_PREFIX = "NOT_PROVEN"

# Unit-63 round-17 audit advisory (class sibling of the two named-constant
# fixes above): `build_report`'s own per-config `verdict` string's
# `"INVALID"` prefix (the fused-dispatch-proof-failed / leg-premise-mismatch
# carve-out from this crate's record-don't-gate doctrine -- see that
# function's own comment) was hand-typed at both places that produce it and
# re-typed at `main()`'s own `.startswith("INVALID")` consumption of it.
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

# unit-63 round-8 audit finding 3 (round-9 audit finding 2 makes the
# domain ASYMMETRIC -- a single `abs(eps) > MAX` check is not this
# family's real shape): the sane domain for this family's own SIGNED
# `eps` (CONTRACT.md addendum 2026-08-29c: the update-scale multiplier is
# `(1+eps)`) is `eps in (MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND,
# -MUTANT_DOSE_LADDER_MIN_ABS_EPS] union [MUTANT_DOSE_LADDER_MIN_ABS_EPS,
# MUTANT_DOSE_LADDER_MAX_EPS]`:
#   - `eps <= MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND` (`-1.0`) is
#     refused, EXCLUSIVE of the bound itself: at exactly that bound the
#     update-scale multiplier `(1+eps)` is exactly zero (a zero-update
#     leg -- this constant's own prior doc already named it "not a member
#     of this monotone family"), and past it the multiplier is NEGATIVE (a
#     sign flip, a different failure shape entirely). Unit-63 round-10
#     audit advisory (b): named here rather than left as a bare `-1.0`
#     literal scattered across every call site, so the `(1+eps)==0`
#     rationale has exactly one place to live. A single symmetric
#     `abs(eps) > MUTANT_DOSE_LADDER_MAX_EPS` check let this bound through
#     (`abs(-1.0) == 1.0`, not `> 1.0`) -- the round-9 audit demonstrated it
#     reported as the Acceptance-5-discharging degradation bound for a
#     zero-update leg.
#   - `eps > MUTANT_DOSE_LADDER_MAX_EPS` (`1.0`) is refused as the
#     family-sanity cap: the scheduled ladder never exceeds `|eps| = 0.50`;
#     nothing past `1.0` has ever been a scheduled dose. Unit-63 round-10
#     audit advisory (b): renamed from `..._MAX_ABS_EPS` -- despite the old
#     name, every call site compares the SIGNED `value` directly
#     (`value > MUTANT_DOSE_LADDER_MAX_EPS`), never `abs(value)`; this is a
#     one-sided ceiling on the POSITIVE branch only, not a magnitude cap,
#     and the old `_ABS_` name claimed a symmetry this domain does not have.
#   - `0.0 < abs(eps) < MUTANT_DOSE_LADDER_MIN_ABS_EPS` is refused. Unit-63
#     round-10 audit advisory (a): `MUTANT_DOSE_LADDER_MIN_ABS_EPS` (0.01)
#     is NOT "the smallest ever-scheduled dose" -- that is `|eps| = 0.10`
#     (the scheduled ladder's own floor). `MUTANT_DOSE_LADDER_MIN_ABS_EPS`
#     is a sanity floor deliberately set BELOW the schedule, so that a
#     genuine sub-schedule diagnostic dose (e.g. `eps=0.02`, the
#     falsified-but-recorded dose above) is still admitted, while a
#     manufactured `eps=1e-9` sitting at the bottom of a straddle is
#     refused rather than silently accepted as a real, schedulable dose.
#     (This one IS a magnitude/`abs()` comparison -- unlike the two bounds
#     above, both signed branches share this same floor.)
MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND = -1.0
MUTANT_DOSE_LADDER_MAX_EPS = 1.0
MUTANT_DOSE_LADDER_MIN_ABS_EPS = 0.01

# CONTRACT.md addendum 2026-08-29c's RED-proof label class (mutants/README.md's
# "RED-proof mutants" section, unit 63 -- `M_nobc`/`M_signflip`): a dose
# column OUTSIDE the (1+eps) lr-scale family this module's eps-family scans
# (`mutant_dose_ladder_sensitivity`, `_two_sided_falsification`,
# `_anomalies`, and the duplicate-EPS arm of
# `mutant_dose_ladder_reject_duplicate_doses`) measure -- a `dose_label`
# carrying this literal prefix can never parse as a signed `eps`
# (`_dose_label_eps` would raise on it, by design), so it is partitioned OUT
# of those scans before they ever see it, never fed through a widened
# `_dose_label_eps`. It still participates fully in `build_mutant_dose_column`
# (premises, partner premises, identity, `detected` computation) and in the
# duplicate-LABEL / duplicate-PATCH_SHA arms of
# `mutant_dose_ladder_reject_duplicate_doses` -- only the duplicate-EPS arm
# and the three eps-only findings are excluded.
RED_PROOF_LABEL_PREFIX = "redproof-"


def is_red_proof_dose_label(dose_label):
    """True iff `dose_label` carries the literal RED-PROOF prefix
    (`RED_PROOF_LABEL_PREFIX`) -- e.g. `"redproof-nobc"`, `"redproof-signflip"`
    (mutants/README.md's own `M_nobc`/`M_signflip` pair). This is a pure
    string-prefix test, never an `_dose_label_eps` parse attempt -- a
    RED-proof label is never expected to parse as a signed eps, by
    construction.
    """
    return dose_label.startswith(RED_PROOF_LABEL_PREFIX)


# CONTRACT amendment 2026-08-29e (D*): the committed, merger-side table a
# RED-proof column's own MUTANT leg's `train_direction` premise (see
# `finetune_run_named_arm_premise_violations`'s own doc) is looked up from --
# keyed on the FULL, case-folded (lowercased, stripped) `patch_sha256`,
# NEVER on the operator-supplied `dose_label` (mutants/README.md's own
# on-pod procedure already doubly-anchors this sha: the CLI `--mutant-legs`
# spec and the producer's own stamped `mutant_patch_sha256` must already
# agree, per `finetune_run_mutant_column_violations`'s own labeling-error
# check below). There is NO CLI surface for this table -- no
# operator-override channel exists, by design; a RED-proof column whose own
# `patch_sha256` is absent here is REFUSED (INVALID), never defaulted to
# either direction (see `red_proof_expected_train_direction`'s own doc).
# `c81d0ed5...` transcribes the `M_signflip_v2` prediction committed at
# 8f06a42c BEFORE its own legs ran ("gradient ASCENT on `adjusted_grad`'s
# direction, every step, compounding for the length of the run",
# mutants/README.md); `9b3c824d...` transcribes `M_nobc`'s own committed
# uncertain-but-still-learning prediction (mutants/README.md's own
# `M_nobc`/`M_signflip_v2` patch sha256 records).
RED_PROOF_EXPECTED_TRAIN_DIRECTION = {
    "c81d0ed59d45761bbd6487dbb23c5aaae22f30739c0e2e613d96c4901ad9b202": FINETUNE_RUN_TRAIN_DIRECTION_ASCENT,
    "9b3c824dc041899c12c0e2d44d12a3ac8c7b86076ffc778638108925ba51bf4e": FINETUNE_RUN_TRAIN_DIRECTION_DESCENT,
}


def red_proof_expected_train_direction(dose_label, patch_sha256):
    """This dose column's own MUTANT leg's declared `train_direction`
    (CONTRACT amendment 2026-08-29e, D*): `FINETUNE_RUN_TRAIN_DIRECTION_DESCENT`
    for every NON-RED-proof column (identical to today's behaviour -- see
    `finetune_run_named_arm_premise_violations`'s own doc for the `d > f <=>
    (|d| > f AND d > f)` equivalence this preserves), or the
    `RED_PROOF_EXPECTED_TRAIN_DIRECTION` table entry keyed on `patch_sha256`
    (case-folded, stripped, the SAME normalization every other sha
    comparison in this module already applies) for a RED-proof-labeled
    column.

    Returns `(direction, violation)`: `violation` is `None` when a direction
    was resolved (`direction` is then a real `FINETUNE_RUN_TRAIN_DIRECTION_*`
    value); when a RED-proof column's own `patch_sha256` is absent from the
    table, `direction` is `None` and `violation` names the refusal -- NEVER
    defaulted to either direction, and never silently accepted as
    descent-shaped, since a false "descent" default would apply the wrong
    (and generally unsatisfiable) direction check to a mutant this table
    simply has no record for.
    """
    if not is_red_proof_dose_label(dose_label):
        return FINETUNE_RUN_TRAIN_DIRECTION_DESCENT, None
    key = str(patch_sha256 or "").strip().lower()
    direction = RED_PROOF_EXPECTED_TRAIN_DIRECTION.get(key)
    if direction is None:
        # unit-63 round-15 audit advisory 5: unprefixed here, matching every
        # pre-existing leg-violation message in this module (e.g. the
        # `mutant_id`/`mutant_patch_sha256` messages just below this
        # function's own call site) -- `build_mutant_dose_column` prefixes
        # every entry in `leg_violations` with `f"{dose_label} seed {seed}:
        # "` exactly once; a self-prefixed message here would be prefixed
        # AGAIN there, doubling `dose_label` in the committed artifact.
        return None, (
            "RED-proof column's own patch_sha256="
            f"{patch_sha256!r} is not present "
            "in the committed RED_PROOF_EXPECTED_TRAIN_DIRECTION table -- CONTRACT amendment "
            "2026-08-29e (D*): a RED-proof column's declared train_direction is looked up from "
            "this merger-side table keyed on the FULL patch sha, never defaulted to either "
            "direction and never read from the operator-supplied dose_label"
        )
    return direction, None


def mutant_leg_repeat_tag(dose_label):
    """The `repeat` slot a mutant leg's `.exit`/`.json`/`.stderr` triple is
    filed under (`load_finetune_run_leg(raw_dir, seed, "fused",
    mutant_leg_repeat_tag(dose_label))`) -- always `"mutant-" + dose_label`,
    which can NEVER equal `"r1"`/`"r2"` (`FINETUNE_RUN_REPEATS`) or
    `FINETUNE_RUN_LR0_REPEAT` (`"lr0"`) for ANY `dose_label` string,
    including an empty one (`"mutant-"` itself is still distinct from all
    three). This is the SAME structural leakage guard
    `FINETUNE_RUN_LR0_REPEAT` already gives the lr=0 control -- a mutant leg
    literally cannot be read by the main pool's or the lr0 control's own
    loaders, which only ever request `"r1"`/`"r2"`/`"lr0"` verbatim.
    """
    return f"mutant-{dose_label}"


def finetune_run_mutant_column_violations(dose_label, patch_sha256, tier):
    """A mutant leg's own premise (amendment 2026-08-29b item 3): it claims
    to be a fused-arm leg with the mutant patch substituted in, so it is
    held to EXACTLY the checks a clean `fused` leg already is
    (`finetune_run_dispatch_proof_violations`'s `arm == "fused"` branch,
    `finetune_run_named_arm_premise_violations`) -- mutants/README.md's own
    "What M1 does NOT touch" section is precisely the claim that a mutant
    leg's premise/dispatch fields read IDENTICAL to a clean fused leg; only
    the DECISION statistic (held-out loss) is supposed to differ, EXCEPT the
    `learning_happened` premise's own `train_direction` half, which CAN
    legitimately read `ascent` for a RED-proof column (CONTRACT amendment
    2026-08-29e, D* -- see `red_proof_expected_train_direction`'s own doc).
    PLUS the
    mutant's own recorded provenance (unit-63 round-7 audit finding 1: these
    three fields are producer-stamped on `FinetuneRunTier` --
    `mutant_id`/`mutant_base_sha`/`mutant_patch_sha256`, serde-skipped when
    `None` -- via the on-pod procedure's own `--mutant-id`/
    `--mutant-base-sha`/`--mutant-patch-sha256` CLI flags, mutants/README.md's
    own on-pod procedure step 6; never hand-edited into the artifact) must be
    present, and this leg's own `mutant_patch_sha256` must equal the dose
    column's caller-supplied `patch_sha256` -- a leg claiming a different
    patch than the dose it was invoked under is a labeling error, never
    silently trusted.

    Unit-63 round-8 audit finding 4 (merger half): each of the three fields
    above, and the caller-supplied `patch_sha256`, is stripped before the
    emptiness/equality checks -- a whitespace-only value (`" "`) is exactly
    as absent as `""`/`None` (`:2759`'s pre-fix bare `if not tier.get(field)`
    passed it straight through), and the sha comparison itself is done on
    the STRIPPED values on both sides so a leg or caller value that differs
    only by incidental leading/trailing whitespace is never reported as a
    labeling-error mismatch. The producer side stamps already-trimmed
    values (a concurrent bench-dispatch fix); this check does not rely on
    that and re-trims independently, on both sides, every time.

    Unit-63 round-10 audit F2: the sha comparison is also done CASE-FOLDED
    on both sides -- sha hex is case-insensitive by domain, and the
    producer now lowercases its own stamped `mutant_patch_sha256`
    (finetune_run.rs's own round-9 advisory (b) fix) while the CLI's own
    `--mutant-legs` spec is only ever stripped, never case-normalized,
    before it reaches here. A case-sensitive compare of a canonicalized-
    lowercase leg value against an uppercase-hex caller spec would report a
    factually false "labeling error" for a pair that names the exact same
    patch -- refused here by folding case at the ONE place this comparison
    is actually made, so it holds regardless of which side (if either)
    happens to have normalized its own case beforehand.

    CONTRACT amendment 2026-08-29e (D*): this leg's own `train_direction`
    (part of `finetune_run_named_arm_premise_violations`'s decomposed
    `learning_happened` premise) is DESCENT for every non-RED-proof column,
    or the `RED_PROOF_EXPECTED_TRAIN_DIRECTION` table entry keyed on
    `patch_sha256` for a RED-proof-labeled one (`red_proof_expected_train_
    direction`) -- a RED-proof column whose own `patch_sha256` is absent
    from that table is a violation here too, named by that function's own
    refusal message, never silently defaulted.
    """
    violations = list(finetune_run_dispatch_proof_violations("fused", tier))
    expected_direction, direction_violation = red_proof_expected_train_direction(dose_label, patch_sha256)
    if direction_violation is not None:
        violations.append(direction_violation)
        # A missing table entry has no direction to check against -- the
        # violation above already invalidates this leg (and, since
        # `patch_sha256` is a per-COLUMN value, every seed sharing this dose
        # column), so the fused-shaped premise check below still runs at the
        # DEFAULT descent direction rather than being skipped outright --
        # never silently dropping the OTHER premise legs (admission_is_dense/
        # tie_fraction/schedule/training_effective) just because
        # train_direction itself could not be resolved.
        expected_direction = FINETUNE_RUN_TRAIN_DIRECTION_DESCENT
    violations += finetune_run_arm_premise_violations("fused", tier, expected_direction)
    for field in ("mutant_id", "mutant_base_sha", "mutant_patch_sha256"):
        value = tier.get(field)
        if not value or not str(value).strip():
            violations.append(
                f"mutant leg's own {field!r} is missing/empty -- mutants/README.md's own "
                "recorded fields (mutant_id, mutant_base_sha, mutant_patch_sha256) must be "
                "present so this leg is attributable to a specific, auditable mutant patch"
            )
    leg_patch_sha256 = tier.get("mutant_patch_sha256")
    if (
        leg_patch_sha256 is not None
        and str(leg_patch_sha256).strip()
        and str(leg_patch_sha256).strip().lower() != str(patch_sha256).strip().lower()
    ):
        violations.append(
            f"leg's own mutant_patch_sha256={leg_patch_sha256!r} does not match this dose "
            f"column's caller-supplied patch_sha256={patch_sha256!r} -- a labeling error, never "
            "silently trusted"
        )
    return violations


def build_mutant_dose_column(raw_dir, dose_label, patch_sha256, mutant_seeds):
    """One dose column (amendment 2026-08-29b item 3): reads each of
    `mutant_seeds`' own `mutant_leg_repeat_tag(dose_label)`-tagged `fused`
    leg, checks it via `finetune_run_mutant_column_violations`, cross-checks
    it against the SAME-SEED campaign `alloff` `r1` leg (loaded fresh from
    `raw_dir` here -- the campaign's ALREADY-RUN leg, never re-run, never a
    second alloff pool) via `generic_leg_premise_violations` over
    `FINETUNE_RUN_IDENTITY_FIELDS` (mirrors the main decision's own
    fused-vs-alloff identity check) AND `finetune_run_named_arm_premise_violations`
    (unit-63 round-7 audit finding 2: the alloff PARTNER's own conjunctive
    premise legs -- `admission_is_dense`/`learning_happened`/`tie_fraction`
    -- are checked here too, not just the mutant side; a premise-failing
    alloff partner excludes the PAIR -- the alloff partner's OWN premises are
    UNCHANGED by amendment 2026-08-29e, always checked at the default
    (descent) direction), PLUS (CONTRACT amendment 2026-08-29e, D*, RED-proof
    columns only) `init_anchor_equality` -- the mutant leg's own
    `train_probe_series[0]` must equal this SAME alloff partner's
    `train_probe_series[0]` exactly -- and, for every premise-clean pair,
    computes `d_i = mutant.held_out_example_mean - alloff.held_out_example_mean`
    -- mutant vs alloff is THE GATE'S OWN STATISTIC (amendment 2026-08-29b
    item 3: "mutant-vs-fused is explicitly NOT the sensitivity claim").

    `detected` is `"RED"` iff the SAME `>=11/12` threshold
    (`MUTANT_DECISION_THRESHOLD`/`MUTANT_GATE_SEED_COUNT`) is met in the
    DEGRADATION direction specifically (mutant worse than alloff, `d_i > 0`
    dominant, `mean_d > 0`); `"RED_FOR_INVESTIGATION"` iff the SAME
    threshold is met in the OPPOSITE, IMPROVEMENT-concordant direction
    instead (`d_i < 0` dominant, `n_neg >= MUTANT_DECISION_THRESHOLD`,
    `mean_d < 0`) -- unit-63 round-8 audit finding 2: this mirrors
    `build_finetune_run_report`'s own main-path `concordant_direction ==
    "improvement"` -> `status = "RED_FOR_INVESTIGATION"` branch; a
    two-sided (falsification-cell) POSITIVE-eps dose that lands here has
    CONFIRMED its own held-out-improvement prediction (see
    `mutant_dose_ladder_two_sided_falsification`'s own doc), never
    collapsed into `"not-detected"` the way it silently used to be before
    this fix (a column with no state for this arm cannot report the
    confirming outcome CONTRACT.md addendum 2026-08-29c's own +0.50 cell
    requires). A dose meeting NEITHER threshold at all (mutants/README.md's
    own M1 finding: a sign-flipping early transient reads 8/12, well under
    11) is `"not-detected"`; a dose whose premise-clean pair count is not
    exactly `MUTANT_GATE_SEED_COUNT`, or whose sign test itself refuses
    (all-tie / empty), is `"INVALID"` -- the SAME correctness-of-measurement
    carve-out every other verdict in this module gets, never silently
    rescaled to whatever count happened to run clean. `n_pos`/`n_neg` can
    never both dominate at once (`2 * MUTANT_DECISION_THRESHOLD` (22)
    exceeds `MUTANT_GATE_SEED_COUNT` (12), mirroring the main decision's own
    `pos_dominant`/`neg_dominant` mutual-exclusion comment), so the two
    branches below are never ambiguous about which direction fired.

    Returns a dict: `{dose_label, patch_sha256, mutant_seeds, per_seed,
    clean_pair_count, gate_seed_count, threshold, n_pos, n_neg, ties,
    mean_d, p_value, sign_test_error, detected, violations}` -- `violations`
    a flat list covering EVERY per-leg/labeling problem found (never
    silently dropped), `per_seed` recording each mutant seed's own
    outcome/d_i/violations even when excluded from the sign test.
    """
    per_seed = {}
    violations = []
    d_values = {}
    for seed in mutant_seeds:
        leg = load_finetune_run_leg(raw_dir, seed, "fused", mutant_leg_repeat_tag(dose_label))
        outcome = leg["outcome"]
        if outcome != "OK":
            per_seed[seed] = {"outcome": outcome, "d_i": None, "violations": []}
            if outcome != "DRY_RUN":
                violations.append(f"{dose_label} seed {seed}: mutant leg outcome={outcome!r} (not OK)")
            continue
        alloff_leg = load_finetune_run_leg(raw_dir, seed, "alloff", "r1")
        if alloff_leg["outcome"] != "OK":
            per_seed[seed] = {"outcome": outcome, "d_i": None, "violations": []}
            violations.append(
                f"{dose_label} seed {seed}: no OK campaign alloff r1 leg for this seed to merge "
                "against -- the dose column reuses the SAME alloff legs the main campaign already "
                "ran, never a second alloff run"
            )
            continue
        tier = finetune_run_block(leg["report"])
        alloff_tier = finetune_run_block(alloff_leg["report"])
        leg_violations = finetune_run_mutant_column_violations(dose_label, patch_sha256, tier)
        leg_violations += generic_leg_premise_violations(
            FINETUNE_RUN_IDENTITY_FIELDS,
            finetune_run_leg_identity(tier),
            finetune_run_leg_identity(alloff_tier),
            "mutant",
            "alloff",
        )
        # unit-63 round-7 audit finding 2: `finetune_run_mutant_column_violations`
        # (above) already premise-checks the mutant (fused-shaped) side via
        # `finetune_run_arm_premise_violations("fused", tier)` -- but the
        # REUSED alloff PARTNER never got the same check here, unlike the
        # main pool (`build_finetune_run_report` premise-checks BOTH
        # `fused_tier` and `alloff_tier` via `finetune_run_named_arm_premise_violations`).
        # A premise-failing alloff leg (live-demonstrated by the real
        # campaign-v1 seed-4 alloff leg, `learning_happened_delta=-0.1125` --
        # see measurements/campaign-v1/README.md) is EXCLUDED from the main
        # gate's own clean-pair count, yet used to silently count as a clean
        # partner in every dose column that reused it. Checking the SAME
        # named premise here, and excluding the PAIR (never merely the
        # fused/mutant side) on a partner failure, makes "the SAME alloff
        # legs under the SAME rule" (amendment 2026-08-29b item 3) literally
        # true.
        alloff_named_violations = finetune_run_named_arm_premise_violations("alloff", alloff_tier)
        leg_violations += [msg for _name, msg in alloff_named_violations]
        # CONTRACT amendment 2026-08-29e (D*), new compensating premise
        # `init_anchor_equality`: a RED-proof mutant leg's own
        # `train_probe_series[0]` (the untrained-init probe) must equal its
        # alloff PARTNER's `train_probe_series[0]` EXACTLY -- a measured
        # same-starting-model guarantee (the amendment's own basis: bit-
        # identical 3.3236749470233917 across every previously committed leg
        # and every signflip_v2 leg). Scoped to RED-proof-labeled columns
        # only (never an eps-family dose) -- checked here, never inside
        # `finetune_run_mutant_column_violations`, since only this function
        # has BOTH tiers in hand.
        if is_red_proof_dose_label(dose_label):
            mutant_series = tier.get("train_probe_series")
            alloff_series = alloff_tier.get("train_probe_series")
            mutant_init = mutant_series[0] if isinstance(mutant_series, list) and mutant_series else None
            alloff_init = alloff_series[0] if isinstance(alloff_series, list) and alloff_series else None
            if mutant_init is None or alloff_init is None or mutant_init != alloff_init:
                # unit-63 round-15 audit advisory 5: unprefixed here (no
                # `dose_label`/`seed` self-prefix) -- this list feeds
                # `leg_violations`, which the caller below (`violations +=
                # [f"{dose_label} seed {seed}: {v}" for v in
                # leg_violations]`) prefixes with `dose_label`/`seed`
                # exactly once; a self-prefixed message here doubled both in
                # the committed artifact pre-fix.
                leg_violations.append(
                    "RED-proof mutant leg's train_probe_series[0]="
                    f"{mutant_init!r} does not exactly equal its alloff partner's "
                    f"train_probe_series[0]={alloff_init!r} -- CONTRACT amendment 2026-08-29e "
                    "(D*) init_anchor_equality: a RED-proof mutant leg must start from the SAME "
                    "untrained-init probe as its alloff partner"
                )
        d_i = None
        mutant_mean = tier.get("held_out_example_mean")
        alloff_mean = alloff_tier.get("held_out_example_mean")
        if (
            not leg_violations
            and isinstance(mutant_mean, (int, float))
            and not isinstance(mutant_mean, bool)
            and isinstance(alloff_mean, (int, float))
            and not isinstance(alloff_mean, bool)
        ):
            d_i = mutant_mean - alloff_mean
            d_values[seed] = d_i
        per_seed[seed] = {"outcome": outcome, "d_i": d_i, "violations": leg_violations}
        violations += [f"{dose_label} seed {seed}: {v}" for v in leg_violations]

    sign_result = None
    sign_error = None
    if d_values:
        try:
            sign_result = sign_test(list(d_values.values()))
        except SignTestError as exc:
            sign_error = str(exc)

    clean_pair_count = len(d_values)
    n_pos = n_neg = ties = 0
    mean_d = None
    p_value = None
    detected = MUTANT_DOSE_DETECTED_NOT_DETECTED
    if sign_error is not None:
        detected = MUTANT_DOSE_DETECTED_INVALID
        violations.append(f"{dose_label}: sign test refused -- {sign_error}")
    elif clean_pair_count != MUTANT_GATE_SEED_COUNT:
        detected = MUTANT_DOSE_DETECTED_INVALID
        violations.append(
            f"{dose_label}: {clean_pair_count} premise-clean mutant/alloff pair(s), expected "
            f"exactly {MUTANT_GATE_SEED_COUNT} -- never silently rescaled to whatever count ran "
            "clean"
        )
    if sign_result is not None:
        n_pos, n_neg, ties, p_value = (
            sign_result["n_pos"],
            sign_result["n_neg"],
            sign_result["ties"],
            sign_result["p_value"],
        )
        mean_d = statistics.mean(d_values.values())
        if detected != MUTANT_DOSE_DETECTED_INVALID and n_pos >= MUTANT_DECISION_THRESHOLD and mean_d > 0.0:
            detected = MUTANT_DOSE_DETECTED_RED
        elif detected != MUTANT_DOSE_DETECTED_INVALID and n_neg >= MUTANT_DECISION_THRESHOLD and mean_d < 0.0:
            # unit-63 round-8 audit finding 2: the improvement-concordant
            # arm the two-sided-falsification cell needs to be able to
            # report -- mirrors the main decision's own
            # `concordant_direction == "improvement"` branch (:2565).
            detected = MUTANT_DOSE_DETECTED_RED_FOR_INVESTIGATION

    return {
        "dose_label": dose_label,
        "patch_sha256": patch_sha256,
        "mutant_seeds": list(mutant_seeds),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "clean_pair_count": clean_pair_count,
        "gate_seed_count": MUTANT_GATE_SEED_COUNT,
        "threshold": MUTANT_DECISION_THRESHOLD,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "ties": ties,
        "mean_d": mean_d,
        "p_value": p_value,
        "sign_test_error": sign_error,
        "detected": detected,
        "violations": violations,
    }


class MutantDoseLadderSensitivityError(ValueError):
    """Raised by `_dose_label_eps` (and therefore by
    `mutant_dose_ladder_sensitivity`/`mutant_dose_ladder_two_sided_falsification`)
    when a dose column's own `dose_label` does not parse as a SIGNED `eps`
    value -- see `_dose_label_eps`'s own doc. A label the sensitivity
    computation cannot place in either the degradation (negative-eps) or
    improvement (positive-eps) branch must never be silently dropped or
    misclassified; `main` catches this and reports it as a dose-ladder
    refusal, the same correctness-of-measurement carve-out every other
    typed refusal in this module gets.
    """


class RedProofLabelError(ValueError):
    """Raised by `partition_red_proof_dose_columns` for a `dose_label`
    carrying the RED-PROOF prefix (`RED_PROOF_LABEL_PREFIX`, `"redproof-"`)
    that names no real mutant after it -- TWO arms, both refused loudly here,
    never silently accepted as an anonymous RED-proof column:

      1. the BARE prefix (`dose_label == RED_PROOF_LABEL_PREFIX` exactly,
         `"redproof-"` with nothing after it at all).
      2. a WHITESPACE-ONLY name after the prefix (unit-63 round-13 audit
         advisory (c): `"redproof- "` / `"redproof-  "` reads as non-empty
         under a bare `==` check against arm 1 alone, so it passed this same
         edge undetected pre-fix and only failed loudly downstream, once
         some later consumer treated the whitespace itself as an opaque
         mutant name).

    A RED-proof column identifies a specific named mutant outside the
    (1+eps) lr-scale family (mutants/README.md's own "RED-proof mutants"
    section, e.g. `redproof-nobc`, `redproof-signflip`) -- neither arm above
    names one. `main` catches this alongside `MutantDoseLadderSensitivityError`
    and folds it into the same `sensitivity_error`/non-zero-exit refusal
    path, the same correctness-of-measurement carve-out every other typed
    refusal in this module gets.
    """


def _dose_label_eps(dose_label):
    """Parse a dose column's own SIGNED `eps` value from its `dose_label`
    (CONTRACT.md addendum 2026-08-29c's own convention -- mutants/README.md's
    on-pod procedure step 6: `dose_label = eps-0.50` / `eps-0.10` / `eps0.50`,
    i.e. the literal string `"eps"` immediately followed by a float literal,
    no separator, negative doses spelled with a bare `-`). Raises
    `MutantDoseLadderSensitivityError` (never silently returns a sentinel or
    guesses a sign) when `dose_label` does not start with `"eps"`, or the
    remainder does not parse as a float.

    Unit-63 round-8 audit finding 3: parsing successfully is not enough --
    the consumers (`mutant_dose_ladder_sensitivity`,
    `mutant_dose_ladder_two_sided_falsification`) partition purely on
    `eps < 0.0`/`eps > 0.0`, so a value that is neither (`nan`, `0.0`,
    `-0.0`) silently agrees with BOTH predicates' negation and vanishes
    from EVERY finding with a clean exit -- exactly the "never silently
    dropped" this module's own doc promises, and never delivers, for that
    one shape. The same applies to a non-finite magnitude (`inf`/`-inf`,
    which trivially satisfies `> 0.0`/`< 0.0` but is not a real dose at
    all) and to a value outside this family's own sane, ASYMMETRIC domain
    (`eps <= MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND`, `eps >
    MUTANT_DOSE_LADDER_MAX_EPS`, or `0.0 < abs(eps) <
    MUTANT_DOSE_LADDER_MIN_ABS_EPS` -- unit-63 round-9 audit finding 2, see
    those constants' own doc). Every one of these is
    refused here, loudly, by the SAME exception every other unparseable
    label raises -- never a silent pass-through to a partition predicate
    that was never designed to reject them.

    Unit-63 round-9 audit advisory (a): `dose_label` is parsed EXACTLY as
    it will be used for raw leg file lookup (`mutant_leg_repeat_tag`
    tags the leg file with the literal `dose_label` string, byte-for-byte)
    -- `float()` is more permissive than that file name ever is, silently
    accepting leading/trailing/embedded whitespace (`float(" 0.5")`) and
    an explicit `+` sign (`float("+0.5")`) that a raw file name lookup
    would never see the same way (e.g. a `+` sitting at a shell/URL
    boundary where it is conventionally decoded as a space). A `dose_label`
    whose eps substring contains either shape is refused here rather than
    silently parsed to a value that could diverge from the on-disk name.
    """
    prefix = "eps"
    if not dose_label.startswith(prefix):
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} does not start with {prefix!r} -- cannot parse a signed "
            "eps value from it (addendum 2026-08-29c's own convention: dose_label = 'eps' + a "
            "float literal, e.g. 'eps-0.50' / 'eps0.50')"
        )
    rest = dose_label[len(prefix):]
    if any(c.isspace() for c in rest) or "+" in rest:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r}'s eps substring {rest!r} contains whitespace or an "
            "explicit '+' sign -- refused rather than parsed by a more permissive float(), since "
            "this label is used VERBATIM for raw leg file lookup (mutant_leg_repeat_tag) and "
            "either shape could silently diverge from the on-disk file name (e.g. a URL-encoding "
            "boundary treating '+' as a space)"
        )
    try:
        value = float(rest)
    except ValueError as exc:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r}'s remainder {rest!r} (after stripping the 'eps' prefix) "
            "does not parse as a float -- cannot derive this dose's signed eps value"
        ) from exc
    if not math.isfinite(value):
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} parses to a non-finite eps value ({value!r}) -- nan/inf "
            "is not a member of either the degradation (eps < 0.0) or improvement (eps > 0.0) "
            "branch and must never be silently dropped from both findings"
        )
    if value == 0.0:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} parses to a zero eps value ({value!r}, positive or "
            "negative zero) -- a zero dose is not a member of either the degradation (eps < 0.0) "
            "or improvement (eps > 0.0) branch and must never be silently dropped from both "
            "findings"
        )
    # unit-63 round-9 audit finding 2: the domain is ASYMMETRIC, never a
    # single `abs(value) > MAX` check -- that check alone would let
    # `eps == MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND` through
    # (`abs(-1.0) == MUTANT_DOSE_LADDER_MAX_EPS`, not `>`), a zero-update
    # leg (multiplier `(1+eps) == 0`) this family's own doc already
    # declares out of family. Refused EXCLUSIVE of the bound itself:
    # `eps == MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND` is refused,
    # `eps == -0.99` is not.
    if value <= MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} parses to eps={value!r} (<= "
            f"{MUTANT_DOSE_LADDER_NEG_EPS_EXCLUSIVE_BOUND}) -- at this magnitude the update-scale "
            "multiplier (1+eps) is zero or negative, a different failure shape entirely and never "
            "a member of this monotone silent-(in/de)flation family, even though a single "
            "symmetric |eps| cap would have let it through"
        )
    if value > MUTANT_DOSE_LADDER_MAX_EPS:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} parses to eps={value!r}, which exceeds this family's own "
            f"sane domain (eps <= {MUTANT_DOSE_LADDER_MAX_EPS}) -- never silently accepted as "
            "though it were a scheduled dose"
        )
    if abs(value) < MUTANT_DOSE_LADDER_MIN_ABS_EPS:
        raise MutantDoseLadderSensitivityError(
            f"dose_label {dose_label!r} parses to eps={value!r}, whose magnitude is below this "
            f"family's own sanity floor (|eps| >= {MUTANT_DOSE_LADDER_MIN_ABS_EPS}), set "
            "deliberately BELOW the smallest ever-SCHEDULED dose (|eps| = 0.10) -- this refuses a "
            "manufactured near-zero eps while still admitting a genuine sub-schedule diagnostic "
            "dose, never silently accepted as though it were a real, schedulable dose"
        )
    return value


def mutant_dose_ladder_reject_duplicate_doses(dose_columns, eps_dose_columns=None):
    """Unit-63 round-10 audit F1: numeric label aliasing defeats
    `mutant_dose_ladder_sensitivity`'s own injectivity assumption (see that
    function's own doc -- dose columns are sorted by `abs(eps)`, and Python's
    STABLE sort then breaks a tie between two equal-magnitude entries by the
    caller-supplied order, "never the caller-supplied order" is a lie the
    moment two distinct literal labels parse to the SAME eps). `eps-0.1` /
    `eps-0.100` / `eps-.10` / `eps-1e-1` all parse to the identical
    `eps=-0.1` float while each filing a DISTINCT leg file
    (`mutant_leg_repeat_tag` tags a leg by the literal `dose_label` string,
    byte-for-byte) -- a stable sort tiebreak on caller order can then emit a
    zero-width "straddle" between two doses that are, by every measurement
    this module makes, THE SAME DOSE, not an adjacent pair.

    Refuses (raises `MutantDoseLadderSensitivityError`, never silently picks
    one of the aliases and drops the rest) over the FULL, already-assembled
    `dose_columns` list when:
      - two entries share the exact same literal `dose_label` string
        (checked FIRST, independent of parseability -- two identically-
        spelled labels are refused even if that label fails to parse as an
        eps value at all), or
      - two entries parse (via `_dose_label_eps`) to the SAME eps value
        under two DIFFERENT literal labels -- named in the refusal by BOTH
        labels and the shared eps: one dose, one label; a same-dose
        disagreement between two legs is a determinism question, never a
        sensitivity interval, or
      - two entries carry the SAME (case-folded, stripped) `patch_sha256`
        under two DIFFERENT literal labels -- unit-63 round-11 audit block:
        the strongest identity key of all, since `mutants/README.md` records
        one DISTINCT patch sha PER dose (the eps is baked into the patch
        itself, never a caller-supplied overlay on a shared patch), so two
        columns citing the same patch are the same mutant measured twice
        regardless of what eps their labels claim. Checked LAST (after the
        label/eps arms above, which already catch a literal- or
        eps-aliased pair without needing to consult the sha at all), and
        skipped for any column whose own `patch_sha256` is missing or
        empty after stripping -- an unset sha is never treated as though it
        aliased another unset sha. Named in the refusal by BOTH labels and
        the shared sha: one patch, one dose; two columns citing the same
        patch are the same mutant measured twice, and their disagreement is
        a determinism question, never a sensitivity interval.

    Called once, over the whole supplied set, at the CLI's own
    `--mutant-legs` assembly -- the same input edge every other dose-ladder
    guard in this module already lives at, and BEFORE
    `mutant_dose_ladder_sensitivity`/`_two_sided_falsification`/`_anomalies`
    ever see the list, so a straddle/anomaly finding can never be computed
    over an aliased set in the first place.

    `eps_dose_columns` (unit 63, RED-proof label class,
    `RED_PROOF_LABEL_PREFIX`): the duplicate-LABEL and duplicate-PATCH_SHA
    arms above run over the FULL `dose_columns` (a RED-proof column is
    subject to both, exactly like any eps-family column), but the
    duplicate-EPS arm calls `_dose_label_eps` on every column it scans --
    which raises, BY DESIGN, on a RED-proof label (it is not a member of
    the signed-eps family and is never expected to parse as one). Passing
    the caller's own already-partitioned eps-only subset here (defaulting
    to `dose_columns` itself when omitted, preserving this function's prior
    behaviour when no RED-proof column is present) scopes that one arm to
    the eps family only, per this unit's own instruction: partition on the
    label prefix BEFORE the scan, never widen `_dose_label_eps`'s own
    strict domain to admit a RED-proof label.
    """
    if eps_dose_columns is None:
        eps_dose_columns = dose_columns
    seen_labels = set()
    for col in dose_columns:
        label = col["dose_label"]
        if label in seen_labels:
            raise MutantDoseLadderSensitivityError(
                f"dose_label {label!r} is supplied more than once in --mutant-legs -- one dose, "
                "one label; a repeated literal label can never name two distinct doses"
            )
        seen_labels.add(label)
    seen_eps = {}
    for col in eps_dose_columns:
        label = col["dose_label"]
        eps = _dose_label_eps(label)
        if eps in seen_eps:
            other_label = seen_eps[eps]
            raise MutantDoseLadderSensitivityError(
                f"dose labels {other_label!r} and {label!r} both parse to the same eps={eps!r} "
                "-- two dose labels resolve to the same eps: one dose, one label; a same-dose "
                "disagreement is a determinism question, never a sensitivity interval"
            )
        seen_eps[eps] = label
    # unit-63 round-11 audit block: the strongest identity key of all --
    # mutants/README.md records one DISTINCT patch sha PER dose (the eps is
    # baked into the patch itself), so two columns naming the same
    # (case-folded, stripped) patch_sha256 are the same mutant measured
    # twice no matter what their labels/parsed epsilons claim. Checked LAST
    # (the label/eps arms above already catch a literal- or eps-aliased
    # pair without ever consulting the sha), and skipped for a column whose
    # own patch_sha256 is missing or empty after stripping -- an unset sha
    # is never treated as though it aliased another unset sha.
    seen_patch_shas = {}
    for col in dose_columns:
        label = col["dose_label"]
        patch_sha256 = str(col.get("patch_sha256") or "").strip().lower()
        if not patch_sha256:
            continue
        if patch_sha256 in seen_patch_shas:
            other_label = seen_patch_shas[patch_sha256]
            raise MutantDoseLadderSensitivityError(
                f"dose labels {other_label!r} and {label!r} both cite the same "
                f"patch_sha256={patch_sha256!r} -- one patch, one dose: two columns citing the "
                "same patch are the same mutant measured twice; their disagreement is a "
                "determinism question, never a sensitivity interval"
            )
        seen_patch_shas[patch_sha256] = label


def mutant_dose_ladder_sensitivity(dose_columns):
    """The reported sensitivity statement (CONTRACT.md addendum 2026-08-29c:
    the signed ladder, `eps in {-0.50, -0.10, +0.50}` -- see
    `docs/plans/63-how-well/mutants/README.md`'s own "signed dose family"
    section): "the adjacent-dose pair straddling detection", SCOPED TO THE
    DEGRADATION-DIRECTION (negative-eps) BRANCH ONLY, ordered by
    `abs(eps)` WITHIN that branch -- never the caller-supplied order and
    never every dose regardless of sign.

    Unit-63 round-7 audit finding 4: the pre-addendum version of this
    function straddled over the CALLER-SUPPLIED dose order, which was a safe
    assumption only while the ladder itself was scheduled ascending in
    detection strength by construction (a positive-only, ascending-magnitude
    ladder). Addendum 2026-08-29c's SIGNED, scheduled-ascending-`eps` ladder
    (`-0.50` run BEFORE `-0.10` BEFORE `+0.50`) breaks that assumption two
    ways: (a) a detection at `-0.50` (the LARGEST-magnitude degradation dose,
    run FIRST) would make the first-adjacent-transition scan see
    `(RED, not-detected, RED)` in caller order and return `None` for a real
    straddle that exists between `-0.50` and `-0.10` when reordered by
    magnitude; (b) a cross-sign `(-0.10 not-detected, +0.50 RED)` adjacent
    pair in caller order is NOT a degradation-direction straddle at all --
    `+0.50` reading RED is the two-sided-falsification finding (see
    `mutant_dose_ladder_two_sided_falsification`), and reporting it as
    "sensitivity" would misrepresent a POSITIVE-eps (inflation-direction)
    degradation detection (unit-63 round-9 audit finding 1: `"RED"` is
    ALWAYS the degradation-concordant arm, never an "improvement-direction
    detection" regardless of eps sign) as though it belonged to the
    NEGATIVE-eps (deflation-direction) degradation-bound family this
    statistic actually measures.

    Each dose's SIGNED eps is parsed from its own `dose_label` via
    `_dose_label_eps` (raises `MutantDoseLadderSensitivityError`, never
    silently skipped, when a label fails to parse). Returns
    `{"lower": lower_dose_label, "higher": higher_dose_label}` for the first
    adjacent (not-detected, RED) transition found when the negative-eps
    (`eps < 0.0`) subset of `dose_columns` is sorted by `abs(eps)` ascending,
    or `None` if no such transition exists in that degradation-only,
    magnitude-ordered subset (every negative dose RED, every negative dose
    not-detected/INVALID, the transition runs the OTHER way, a negative
    dose reads `RED_FOR_INVESTIGATION` -- unit-63 round-9 audit finding 3:
    an ANOMALY, an improvement detected under deflation, reported
    separately by `mutant_dose_ladder_anomalies` and never a `"RED"`
    straddle member here -- or there are fewer than two negative-eps doses
    to straddle at all). A positive-eps dose is NEVER a member of this
    subset, regardless of its own `detected` value.
    """
    negative = sorted(
        (col for col in dose_columns if _dose_label_eps(col["dose_label"]) < 0.0),
        key=lambda col: abs(_dose_label_eps(col["dose_label"])),
    )
    for lower, higher in zip(negative, negative[1:]):
        if lower["detected"] == MUTANT_DOSE_DETECTED_NOT_DETECTED and higher["detected"] == MUTANT_DOSE_DETECTED_RED:
            return {"lower": lower["dose_label"], "higher": higher["dose_label"]}
    return None


def mutant_dose_ladder_anomalies(dose_columns):
    """Unit-63 round-9 audit finding 3: a NEGATIVE-eps dose (silent lr
    DEFLATION, this family's DEGRADATION-PREDICTED branch) reading
    `"RED_FOR_INVESTIGATION"` is itself an ANOMALY -- a real, gate-detected
    IMPROVEMENT under deflation, the opposite of that branch's own
    predicted direction. `mutant_dose_ladder_sensitivity` correctly never
    treats it as a `"RED"` straddle member (see that function's own doc),
    but "never a straddle member" is not the same as "reported somewhere":
    before this fix, a negative-eps `RED_FOR_INVESTIGATION` column silently
    vanished -- `sensitivity` reads `None`, `sensitivity_error` reads
    `None`, and the merge's own exit code stayed 0 -- exactly the class of
    silent pass-through the primary decision rule's own
    `RED_FOR_INVESTIGATION` gate exists to prevent (`main`'s own
    `fr_merged["status"] in FINETUNE_RUN_GATING_STATUSES` check): "anomalous
    improvement is investigated, never silently celebrated".

    Returns the list of `{"dose_label", "eps", "detected", "finding"}`
    entries for every NEGATIVE-eps (`eps < 0.0`) dose column whose
    `detected` is `"RED_FOR_INVESTIGATION"`, in `dose_columns`' own order,
    with `finding` set to the literal string `"anomalous improvement under
    deflation (eps < 0)"`. A POSITIVE-eps dose reading `RED_FOR_INVESTIGATION`
    is NEVER a member of this list -- that is the ORDINARY, PREDICTED
    two-sided-falsification confirming arm (see
    `mutant_dose_ladder_two_sided_falsification`), not an anomaly. Empty
    when no negative-eps dose reads `RED_FOR_INVESTIGATION`. Each dose's
    SIGNED eps is parsed via `_dose_label_eps` (same refusal behaviour as
    the other two dose-ladder findings).
    """
    out = []
    for col in dose_columns:
        eps = _dose_label_eps(col["dose_label"])
        if eps < 0.0 and col["detected"] == MUTANT_DOSE_DETECTED_RED_FOR_INVESTIGATION:
            out.append({
                "dose_label": col["dose_label"],
                "eps": eps,
                "detected": col["detected"],
                "finding": "anomalous improvement under deflation (eps < 0)",
            })
    return out


def mutant_dose_ladder_two_sided_falsification(dose_columns):
    """The two-sided-falsification finding CONTRACT.md addendum 2026-08-29c
    names for a POSITIVE-eps ("improvement-direction") dose: `+0.50` is
    "retained deliberately as the two-sided falsification cell for the
    improvement prediction itself" (mutants/README.md's own "signed dose
    family" section). The prediction under test (Step 2/3 of that same
    README) is HELD-OUT IMPROVEMENT at positive eps; `build_mutant_dose_column`'s
    own `detected` names which arm a positive-eps dose actually landed in --
    unit-63 round-8 audit finding 1 corrects this function's own prior
    (inverted) polarity claim:

    - `detected == "RED"` is the DEGRADATION-concordant arm (mutant worse
      than alloff, `mean_d > 0.0`). A positive-eps dose reading `"RED"`
      REFUTES the improvement prediction -- the secant extrapolation was
      wrong over this range, since more effective lr made held-out loss
      WORSE, not better. This is never a "confirmation".
    - `detected == "RED_FOR_INVESTIGATION"` is the IMPROVEMENT-concordant
      arm (`mean_d < 0.0`, unit-63 round-8 audit finding 2's new state on
      this column, mirroring the main decision's own
      `concordant_direction == "improvement"` branch). A positive-eps dose
      reading `"RED_FOR_INVESTIGATION"` CONFIRMS the improvement
      prediction -- a real, gate-detectable improvement in the predicted
      direction.

    Neither arm is folded into `mutant_dose_ladder_sensitivity` (the
    degradation-direction, negative-eps-only statistic) -- see that
    function's own doc for why a cross-sign or cross-arm detection there
    would misrepresent it.

    Each dose's SIGNED eps is parsed via `_dose_label_eps` (same refusal
    behaviour as `mutant_dose_ladder_sensitivity`). Returns the list of
    `{"dose_label", "eps", "detected", "finding"}` entries for every
    positive-eps (`eps > 0.0`) dose column whose `detected` is `"RED"` or
    `"RED_FOR_INVESTIGATION"`, in `dose_columns`' own order, with `finding`
    set to the literal string `"secant refuted (degradation at +eps)"` for
    the `"RED"` arm and `"secant confirmed (improvement at +eps)"` for the
    `"RED_FOR_INVESTIGATION"` arm; empty when neither arm fired at any
    positive-eps dose -- the ORDINARY, not-yet-refuted case (the prediction
    surviving is not itself a "confirmation": that only happens when the
    `RED_FOR_INVESTIGATION` arm actually fires).
    """
    out = []
    for col in dose_columns:
        eps = _dose_label_eps(col["dose_label"])
        if eps > 0.0 and col["detected"] == MUTANT_DOSE_DETECTED_RED:
            out.append({
                "dose_label": col["dose_label"],
                "eps": eps,
                "detected": col["detected"],
                "finding": "secant refuted (degradation at +eps)",
            })
        elif eps > 0.0 and col["detected"] == MUTANT_DOSE_DETECTED_RED_FOR_INVESTIGATION:
            out.append({
                "dose_label": col["dose_label"],
                "eps": eps,
                "detected": col["detected"],
                "finding": "secant confirmed (improvement at +eps)",
            })
    return out


def partition_red_proof_dose_columns(dose_columns):
    """Splits the FULL, already-built `dose_columns` list (every column
    `build_mutant_dose_column` produced, regardless of label shape) into
    `(eps_dose_columns, red_proof_dose_columns)` by the `dose_label`'s own
    PREFIX (`RED_PROOF_LABEL_PREFIX`) -- BEFORE any of the eps-family scans
    (`mutant_dose_ladder_sensitivity`, `_two_sided_falsification`,
    `_anomalies`, and the duplicate-EPS arm of
    `mutant_dose_ladder_reject_duplicate_doses`) ever see the list. This is
    the split unit 63's own RED-proof label class requires: a partition on
    the label prefix, never a widening of `_dose_label_eps`'s own strict
    eps-only domain -- an eps-family label keeps its existing strict
    validation untouched, and a RED-proof label is never asked to satisfy
    it.

    Raises `RedProofLabelError` for a bare-prefix label (`dose_label ==
    RED_PROOF_LABEL_PREFIX`, i.e. `"redproof-"` with an empty mutant name
    after it) OR a whitespace-only mutant name after the prefix (unit-63
    round-13 audit advisory (c): `"redproof- "` / `"redproof-  "` reads as
    non-empty by `==`, so it passed this same edge undetected under the
    bare-prefix check alone and only failed loudly downstream, once some
    later consumer treated the whitespace as an opaque mutant name) --
    refused loudly here, at the same input edge every other dose-ladder
    label guard in this module already lives at, never silently accepted as
    an anonymous RED-proof column.

    `eps_dose_columns`/`red_proof_dose_columns` each preserve `dose_columns`'
    own relative order.
    """
    eps_dose_columns = []
    red_proof_dose_columns = []
    for col in dose_columns:
        label = col["dose_label"]
        if is_red_proof_dose_label(label):
            if label == RED_PROOF_LABEL_PREFIX or label[len(RED_PROOF_LABEL_PREFIX):].strip() == "":
                raise RedProofLabelError(
                    f"dose_label {label!r} is the bare RED-PROOF prefix ({RED_PROOF_LABEL_PREFIX!r}) "
                    "or a whitespace-only mutant name after it, with no mutant name after it -- a "
                    "RED-proof column must name a specific mutant (e.g. 'redproof-nobc', "
                    "'redproof-signflip'), never an anonymous prefix or a whitespace-only name"
                )
            red_proof_dose_columns.append(col)
        else:
            eps_dose_columns.append(col)
    return eps_dose_columns, red_proof_dose_columns


def build_red_proof_summary(red_proof_dose_columns):
    """The first-class RED-proof merger output (unit 63): the honest
    alternative to reading a verdict out of a separate, exit-1-expected
    invocation (mutants/README.md's own retired "minimal labeling
    convention" section) -- `M_nobc`/`M_signflip` (CONTRACT.md addendum
    2026-08-29c) are OUTSIDE the (1+eps) lr-scale family this module's
    eps-family scans measure, but their own measured verdict
    (`build_mutant_dose_column`'s own `detected`/`n_pos`/`n_neg`/`mean_d`/
    `p_value`/`clean_pair_count`) is computed identically to any other dose
    column and is reported here as a real, first-class field in THIS
    merge's own artifact.

    Returns `(red_proof, red_proof_verdict)`:
      - `red_proof`: one `{dose_label, patch_sha256, detected, n_pos, n_neg,
        mean_d, p_value, clean_pair_count}` entry per RED-proof column, in
        `red_proof_dose_columns`' own order -- the exact subset of fields
        `build_mutant_dose_column` already computed, never a re-derivation.
      - `red_proof_verdict`: the literal string `"PROVEN"` iff at least one
        RED-proof column's own `detected` reads the literal string `"RED"`
        (degradation-concordant: the mutant EXPECTED to degrade actually
        measured worse than alloff at the pre-registered threshold) --
        acceptance 5's "mutant column proven RED" is discharged. A
        RED-proof column reading `"RED_FOR_INVESTIGATION"` is recorded
        AS-IS in its own `detected` field (an anomaly: this mutant is
        EXPECTED to degrade with certainty or high confidence per
        mutants/README.md's own prediction, so an improvement-concordant
        detection here is itself a finding to investigate, never a second
        way to discharge acceptance 5) -- it never counts toward
        `"PROVEN"` on its own. Otherwise (no column reads `"RED"`) the
        literal string `"NOT_PROVEN"` followed by a parenthesized
        `dose_label=detected` listing for every RED-proof column, so
        acceptance 5's own undischarged state is legible directly off this
        one string, never buried in a JSON field alone.
    """
    red_proof = [
        {
            "dose_label": col["dose_label"],
            "patch_sha256": col["patch_sha256"],
            "detected": col["detected"],
            "n_pos": col["n_pos"],
            "n_neg": col["n_neg"],
            "mean_d": col["mean_d"],
            "p_value": col["p_value"],
            "clean_pair_count": col["clean_pair_count"],
        }
        for col in red_proof_dose_columns
    ]
    if any(col["detected"] == MUTANT_DOSE_DETECTED_RED for col in red_proof_dose_columns):
        red_proof_verdict = RED_PROOF_VERDICT_PROVEN
    else:
        listing = ", ".join(f"{col['dose_label']}={col['detected']}" for col in red_proof_dose_columns)
        red_proof_verdict = f"{RED_PROOF_VERDICT_NOT_PROVEN_PREFIX} ({listing})"
    return red_proof, red_proof_verdict


# Unit-63 round-14 audit F6: the EXACT set of dose-ladder cause names this
# module's own `main()` ties to a non-zero `exit_code` in its `finetune-run`
# branch (see the `dose_ladder_causes` list built there) -- the SAME source
# `howwell_dose_ladder_cause.py`'s own namer (`_ALL_CAUSE_NAMES`) is tested
# against, so the two can never drift apart the way a hand-duplicated list on
# each side could: `main()`'s own exit fold is now DATA over exactly this
# set (a fold arm cannot exist without a name here), and
# `test_howwell_dose_ladder_cause.py`'s cross-module test imports BOTH this
# constant and the namer's own checked-cause set and asserts equality -- a
# fifth cause added to one side without the other is a RED test, never
# silent drift (the prior state this round's audit found: the namer's own
# comment CLAIMED "can never drift apart" with nothing mechanical enforcing
# it).
DOSE_LADDER_EXIT_CAUSE_NAMES = (
    "sensitivity_error",
    "invalid dose column",
    "dose_anomalies",
    "red_proof_verdict",
)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    # unit 63 H4b: `finetune_run_ab.sh`'s own invocation shape --
    # `ab_merge.py finetune-run RAW_DIR OUT_DIR SEED1,SEED2,...` -- kept
    # entirely distinct from the positional `finetune_ab.sh` contract below
    # (a leading literal `"finetune-run"` can never collide with that
    # contract's own first positional arg, `RAW_DIR`, since no real
    # directory path is spelled exactly that way).
    if argv and argv[0] == "finetune-run":
        rest = argv[1:]
        # amendment 2026-08-29b item 3 -- REPEATABLE flag (like lr0's own
        # positional, but a dose ladder is N columns, not one list), scanned
        # out of `rest` BEFORE the positional contract below so its own
        # placement never shifts SEEDS/LR0_SEEDS/--allow-missing-lr0-control.
        # Each occurrence: `--mutant-legs DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,...`
        # (mutants/README.md's own per-leg recorded fields -- the dose label
        # + the mutant patch sha256 the leg's producer recorded).
        mutant_leg_specs = []
        filtered = []
        i = 0
        while i < len(rest):
            if rest[i] == "--mutant-legs":
                if i + 1 >= len(rest):
                    print(
                        "usage: ab_merge.py finetune-run ... --mutant-legs "
                        "DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,...",
                        file=sys.stderr,
                    )
                    return 2
                mutant_leg_specs.append(rest[i + 1])
                i += 2
                continue
            filtered.append(rest[i])
            i += 1
        rest = filtered
        # unit-63 round-3 audit block 5: a flag, not a positional -- scanned
        # out of `rest` wherever it appears so it never shifts the existing
        # positional contract (`finetune_run_ab.sh` appends it, if at all,
        # after LR0_SEEDS, but this does not depend on that ordering).
        allow_missing_lr0_control = "--allow-missing-lr0-control" in rest
        rest = [a for a in rest if a != "--allow-missing-lr0-control"]
        if len(rest) < 3:
            print(
                "usage: ab_merge.py finetune-run RAW_DIR OUT_DIR SEED1,SEED2,... "
                "[LR0_SEED1,LR0_SEED2,...] [--allow-missing-lr0-control] "
                "[--mutant-legs DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,... ...]",
                file=sys.stderr,
            )
            return 2
        fr_raw_dir, fr_out_dir, seeds_s = rest[:3]
        seeds = [s for s in seeds_s.split(",") if s]
        # unit-63 audit advisory (b): the lr=0 RED control's own seed list --
        # OPTIONAL, and, unlike `seeds` above, never itself entering the A/B
        # set's own d_values/sign test (see `finetune_run_lr0_control_seed_violations`).
        # unit-63 round-3 audit block 5: an EMPTY list here is now itself a
        # refusal (INVALID) unless `allow_missing_lr0_control` above was
        # passed -- see `build_finetune_run_report`'s own doc.
        lr0_seeds_s = rest[3] if len(rest) > 3 else ""
        lr0_seeds = [s for s in lr0_seeds_s.split(",") if s]

        fr_merged, fr_table = build_finetune_run_report(
            fr_raw_dir, seeds, lr0_seeds=lr0_seeds, allow_missing_lr0_control=allow_missing_lr0_control
        )
        if fr_merged is None:
            print(f"finetune_run_ab: FAIL — no leg output found under {fr_raw_dir}", file=sys.stderr)
            return 1

        # amendment 2026-08-29b item 3 -- the dose ladder, computed BEFORE
        # the artifact is written so it lands in the SAME
        # `finetune_run_ab_report.json`/`.._table.txt` pair as the primary
        # decision, never a second, separately-discoverable file.
        exit_code = 0
        if mutant_leg_specs:
            dose_columns = []
            for spec in mutant_leg_specs:
                parts = spec.split(":", 2)
                if len(parts) != 3:
                    print(
                        f"usage: --mutant-legs DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,... (got {spec!r})",
                        file=sys.stderr,
                    )
                    return 2
                dose_label, patch_sha256, mutant_seeds_s = parts
                # unit-63 round-8 audit finding 4 (merger half): the
                # caller-supplied sha is stripped here too, at the SAME
                # point every other producer-stamped-field trim happens --
                # a whitespace-only `--mutant-legs` sha must compare as
                # empty, never as some never-matching opaque string.
                # unit-63 round-10 audit F2: also lowercased here -- sha hex
                # is case-insensitive by domain, and the producer now
                # canonicalizes its own stamped sha to lowercase, so the
                # caller-supplied spec is folded to the SAME case rather
                # than relying solely on the comparison site
                # (finetune_run_mutant_column_violations, which case-folds
                # independently too, so this holds either way).
                patch_sha256 = patch_sha256.strip().lower()
                mutant_seeds = [s for s in mutant_seeds_s.split(",") if s]
                dose_columns.append(build_mutant_dose_column(fr_raw_dir, dose_label, patch_sha256, mutant_seeds))
            # unit-63 round-7 audit finding 4 / CONTRACT.md addendum
            # 2026-08-29c: sensitivity is scoped to the degradation-direction
            # (negative-eps) branch, magnitude-ordered, never the
            # caller-supplied order -- a signed dose_label that fails to
            # parse is a dose-ladder refusal (never a script crash), reported
            # here and folded into the exit code like every other
            # correctness-of-measurement problem this merge gates on.
            sensitivity_error = None
            eps_dose_columns = dose_columns
            red_proof_dose_columns = []
            red_proof = []
            red_proof_verdict = None
            try:
                # unit 63, RED-proof label class (RED_PROOF_LABEL_PREFIX):
                # partitioned OUT of the eps-family set BEFORE any of the
                # scans below ever see it -- a RED-proof label can never
                # parse as a signed eps and must never be fed through a
                # widened `_dose_label_eps`. Raises `RedProofLabelError` for
                # a bare-prefix label, folded into the same refusal path as
                # `MutantDoseLadderSensitivityError` below.
                eps_dose_columns, red_proof_dose_columns = partition_red_proof_dose_columns(dose_columns)
                # unit-63 round-10 audit F1 -- refused BEFORE the straddle/
                # falsification/anomaly scans ever see the (possibly
                # aliased) set, over the FULL assembled `dose_columns` for
                # the duplicate-LABEL/PATCH_SHA arms (a RED-proof column is
                # subject to both, exactly like any eps-family column), and
                # over `eps_dose_columns` only for the duplicate-EPS arm
                # (see this function's own doc).
                mutant_dose_ladder_reject_duplicate_doses(dose_columns, eps_dose_columns)
                sensitivity = mutant_dose_ladder_sensitivity(eps_dose_columns)
                two_sided_falsification = mutant_dose_ladder_two_sided_falsification(eps_dose_columns)
                dose_anomalies = mutant_dose_ladder_anomalies(eps_dose_columns)
                if red_proof_dose_columns:
                    red_proof, red_proof_verdict = build_red_proof_summary(red_proof_dose_columns)
            except (MutantDoseLadderSensitivityError, RedProofLabelError) as exc:
                sensitivity = None
                two_sided_falsification = []
                dose_anomalies = []
                sensitivity_error = str(exc)
                # unit-63 round-13 audit F2: a refusal here must not leave
                # `red_proof_verdict` byte-identical to "no RED-proof column
                # was ever scheduled" (the None it was initialized to above)
                # when a RED-proof-labeled column WAS actually present in
                # the supplied dose set -- CONTRACT acceptance 5's own field
                # is read directly off this string. Test the RAW
                # `dose_columns` labels here, never `red_proof_dose_columns`
                # alone: `partition_red_proof_dose_columns` itself may be
                # the raiser (a bare-prefix or whitespace-only mutant name),
                # in which case `red_proof_dose_columns` never got assigned
                # and stays the pre-try `[]` even though a RED-proof label
                # was present in `dose_columns`. When no RED-proof label was
                # ever scheduled, `red_proof_verdict` stays `None` exactly
                # as before -- nothing to report.
                if any(is_red_proof_dose_label(col["dose_label"]) for col in dose_columns):
                    red_proof_verdict = (
                        f"{RED_PROOF_VERDICT_NOT_PROVEN_PREFIX} (dose set refused before "
                        f"RED-proof evaluation: {exc})"
                    )
            fr_merged["mutant_dose_ladder"] = {
                "doses": dose_columns,
                "sensitivity": sensitivity,
                "sensitivity_error": sensitivity_error,
                "two_sided_falsification": two_sided_falsification,
                "dose_anomalies": dose_anomalies,
                "red_proof": red_proof,
                "red_proof_verdict": red_proof_verdict,
                "note": (
                    "CONTRACT amendment 2026-08-29b item 3, signed per addendum 2026-08-29c: each "
                    "dose column merges the mutant, substituted into the fused arm, against the "
                    "SAME campaign alloff legs (never re-run) under the SAME >=11/12+mean rule the "
                    "primary decision uses. Mutant legs never enter the primary A/B set "
                    "(mutant_leg_repeat_tag's own file-naming isolation). 'sensitivity' names the "
                    "adjacent-dose pair straddling detection WITHIN the degradation-direction "
                    "(negative-eps) branch only, magnitude-ordered, or null when no such transition "
                    "exists there; a positive-eps dose reading RED or RED_FOR_INVESTIGATION is "
                    "reported separately under 'two_sided_falsification' (RED refutes the held-out-"
                    "improvement prediction, RED_FOR_INVESTIGATION confirms it), never folded into "
                    "'sensitivity'; a NEGATIVE-eps dose reading RED_FOR_INVESTIGATION (an anomalous "
                    "improvement detected under deflation, unit-63 round-9 audit finding 3) is "
                    "reported under 'dose_anomalies' instead and gates this merge's own exit code "
                    "exactly as the primary decision's own RED_FOR_INVESTIGATION state does -- "
                    "investigated, never silently celebrated; 'sensitivity_error' names ANY of "
                    "this family's own refusal classes over the supplied dose set -- a dose_label "
                    "that failed to parse as a signed eps value, a parsed eps outside this "
                    "family's own domain, or a duplicate identity across two columns (the same "
                    "literal dose_label, the same parsed eps under two different labels, or the "
                    "same patch_sha256 under two different labels, unit-63 round-11 audit block) "
                    "-- never silently ignored. 'red_proof'/'red_proof_verdict' (unit 63, the "
                    "RED-proof label class, RED_PROOF_LABEL_PREFIX='redproof-') report every dose "
                    "column whose label carries that prefix -- OUTSIDE the (1+eps) lr-scale family "
                    "and excluded from 'sensitivity'/'two_sided_falsification'/'dose_anomalies' and "
                    "the duplicate-EPS arm above (still subject to the duplicate-LABEL/PATCH_SHA "
                    "arms), but computed in 'doses' exactly like any other column. "
                    "'red_proof_verdict' is 'PROVEN' iff at least one RED-proof column's own "
                    "'detected' reads 'RED' (acceptance 5's own discharge condition); otherwise "
                    "'NOT_PROVEN' naming each RED-proof column's own 'detected' -- a RED-proof "
                    "column reading RED_FOR_INVESTIGATION is recorded as-is (an anomaly for a "
                    "mutant EXPECTED to degrade), never a second way to discharge PROVEN."
                ),
            }
            dose_lines = ["", "# mutant dose ladder (amendment 2026-08-29b item 3; addendum 2026-08-29c signs it)"]
            for col in dose_columns:
                dose_lines.append(
                    f"  dose={col['dose_label']:<12} detected={col['detected']:<12} "
                    f"n_pos={col['n_pos']} n_neg={col['n_neg']} mean_d={col['mean_d']} "
                    f"p_value={col['p_value']} clean_pairs={col['clean_pair_count']}/{col['gate_seed_count']}"
                )
            if sensitivity_error is not None:
                dose_lines.append(f"  sensitivity: REFUSED -- {sensitivity_error}")
            else:
                dose_lines.append(f"  sensitivity: {sensitivity}")
            if two_sided_falsification:
                dose_lines.append(f"  two_sided_falsification: {two_sided_falsification}")
            if dose_anomalies:
                dose_lines.append(f"  dose_anomalies: {dose_anomalies}")
            if red_proof_dose_columns or red_proof_verdict is not None:
                # unit-63 round-13 audit F2: `red_proof_verdict` can be set
                # (the refused-but-scheduled state) even when
                # `red_proof_dose_columns` itself stayed empty (the raiser
                # was `partition_red_proof_dose_columns` itself) -- the
                # human-readable table line must stay in sync with the
                # artifact's own field rather than silently omitting it.
                dose_lines.append(f"  red_proof: {red_proof}")
                dose_lines.append(f"  red_proof_verdict: {red_proof_verdict}")
            fr_table = fr_table + "\n" + "\n".join(dose_lines)
            invalid_doses = [
                c["dose_label"] for c in dose_columns if c["detected"] == MUTANT_DOSE_DETECTED_INVALID
            ]
            anomalous_doses = [a["dose_label"] for a in dose_anomalies]
            # unit-63 round-14 audit F6: the dose-ladder exit fold is DATA,
            # never four independently-hand-maintained `if` blocks -- each
            # `(cause_name, triggered, message)` entry here is the ONE
            # source of truth both this loop and (via
            # `DOSE_LADDER_EXIT_CAUSE_NAMES`, asserted below) `howwell_
            # dose_ladder_cause.py`'s own namer are tested against
            # (`DoseLadderCauseNamesBoundToAbMergeExitFoldTests` in
            # `test_howwell_dose_ladder_cause.py`) -- a fifth cause added
            # here without a matching namer check is now a RED test, never
            # silent drift. Order mirrors this fold's own historical
            # ordering (sensitivity_error, invalid dose column,
            # dose_anomalies, red_proof_verdict); the namer's own set
            # equality check is order-independent.
            dose_ladder_causes = [
                (
                    "sensitivity_error",
                    sensitivity_error is not None,
                    f"finetune_run_ab mutant-dose-ladder: FAIL — {sensitivity_error}",
                ),
                (
                    "invalid dose column",
                    bool(invalid_doses),
                    f"finetune_run_ab mutant-dose-ladder: FAIL — dose column(s) {invalid_doses} are "
                    "INVALID (correctness-of-measurement problem, see the table above)",
                ),
                (
                    # unit-63 round-9 audit finding 3: mirrors the primary
                    # decision rule's own RED_FOR_INVESTIGATION gate (below)
                    # -- an anomalous improvement under deflation is
                    # investigated, never silently celebrated, even though
                    # it can never be a 'sensitivity' straddle member.
                    "dose_anomalies",
                    bool(dose_anomalies),
                    f"finetune_run_ab mutant-dose-ladder: FAIL — dose column(s) {anomalous_doses} "
                    "read RED_FOR_INVESTIGATION at a NEGATIVE eps (anomalous improvement under "
                    "deflation, see 'dose_anomalies' above) -- investigated, never silently "
                    "celebrated",
                ),
                (
                    # unit 63: NOT_PROVEN is a failure of THIS run's own
                    # purpose (a RED-proof column exists precisely to
                    # discharge acceptance 5's "mutant column proven RED"
                    # outside the (1+eps) family) -- named here exactly as
                    # it is recorded in the artifact's own
                    # 'red_proof_verdict', never silently passed through as
                    # green. PROVEN contributes nothing to `exit_code` here
                    # -- it is the EXPECTED outcome, unlike 'dose_anomalies'
                    # above.
                    "red_proof_verdict",
                    red_proof_verdict is not None
                    and red_proof_verdict.startswith(RED_PROOF_VERDICT_NOT_PROVEN_PREFIX),
                    f"finetune_run_ab mutant-dose-ladder: FAIL — red_proof_verdict={red_proof_verdict!r} "
                    "-- acceptance 5's 'mutant column proven RED' is undischarged by every scheduled "
                    "RED-proof column",
                ),
            ]
            # unit-63 round-15 audit advisory 4: an explicit `if`/`raise`,
            # never a bare `assert` -- `assert` is stripped entirely under
            # `python -O`, which would silently disable this runtime binding
            # to `DOSE_LADDER_EXIT_CAUSE_NAMES` in exactly the deployment
            # shape (`-O`) that removes the safety net without removing the
            # code path it protects. `howwell_dose_ladder_cause.py`'s own
            # test-side binding (`DoseLadderCauseNamesBoundToAbMergeExitFoldTests`
            # in `test_howwell_dose_ladder_cause.py`) remains the PRIMARY
            # enforcement (it runs on every commit, `-O` or not); this
            # runtime check is the defense-in-depth belt for the one process
            # that actually folds `dose_ladder_causes` at merge time.
            _dose_ladder_cause_names = {name for name, _triggered, _message in dose_ladder_causes}
            if _dose_ladder_cause_names != set(DOSE_LADDER_EXIT_CAUSE_NAMES):
                raise AssertionError(
                    "dose_ladder_causes drifted from the committed DOSE_LADDER_EXIT_CAUSE_NAMES set "
                    f"(dose_ladder_causes names={sorted(_dose_ladder_cause_names)}, "
                    f"DOSE_LADDER_EXIT_CAUSE_NAMES={sorted(DOSE_LADDER_EXIT_CAUSE_NAMES)})"
                )
            for _cause_name, triggered, message in dose_ladder_causes:
                if triggered:
                    print(message, file=sys.stderr)
                    exit_code = 1

        os.makedirs(fr_out_dir, exist_ok=True)
        with open(os.path.join(fr_out_dir, "finetune_run_ab_report.json"), "w") as fh:
            json.dump(fr_merged, fh, indent=2)
        print(fr_table)
        with open(os.path.join(fr_out_dir, "finetune_run_ab_table.txt"), "w") as fh:
            fh.write(fr_table + "\n")

        # unit-63 audit finding 1: the decision rule's own three non-GREEN
        # outcomes -- INVALID (a correctness-of-measurement problem: a leg
        # premise violation, a determinism-floor breach, a sign-test
        # refusal, a wrong premise-clean seed count, or an lr=0 control
        # violation -- the SAME carve-out `build_report`'s own `INVALID`
        # branch already makes for `finetune_ab.sh`), RED (degradation-
        # concordant: the pre-registered rule fired against fused), and
        # RED_FOR_INVESTIGATION (improvement-concordant: anomalous
        # improvement, investigated, never silently celebrated) -- are ALL
        # the things this merge's own exit code gates on now; only a plain
        # `FAIL`/`INCOMPLETE`/`DRY_RUN` leg (never ran, or a dry run) stays
        # record-only. Unit-63 round-15 audit: this gating check reads
        # `FINETUNE_RUN_GATING_STATUSES` (the module-level constant, see its
        # own doc above) rather than a re-typed literal tuple, so this fold
        # and `runpod_gpu_howwell.sh`'s own `case "$STATUS"` arms can never
        # independently drift from the same committed status vocabulary.
        if fr_merged["status"] in FINETUNE_RUN_GATING_STATUSES:
            print(
                f"finetune_run_ab: FAIL — status={fr_merged['status']} — see the table above "
                "(CONTRACT 63 Frame: the pre-registered decision rule; INVALID names a "
                "correctness-of-measurement problem, RED/RED_FOR_INVESTIGATION name a fired "
                "decision rule, never silently passed through as green)",
                file=sys.stderr,
            )
            exit_code = 1
        return exit_code

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

    # Advisory (iv), round-2 audit fix on PR #372: the ONE carve-out from
    # this crate's own record-don't-gate doctrine (see `finetune_ab.sh`'s
    # module doc and `build_report`'s own verdict-computation comment) --
    # an `INVALID` verdict (a failed/errored `fused_proof`) is a
    # correctness-of-MEASUREMENT problem, not a machine-dependent
    # performance number, so it is the one thing this sweep's own exit code
    # DOES gate on. An ordinary ratio-based `FAIL` row remains
    # record-only, unchanged.
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
