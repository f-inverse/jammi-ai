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
| `seed` | identity | `report.rs:FinetuneStepTier::seed` field; `seed: params.seed,` (`finetune_step.rs:783`) | `"seed": args.seed,` (`torch_finetune_step.py:1116`) |
| `batch` | identity | `batch: params.batch,` (`finetune_step.rs:788`) | `"batch": args.batch,` (`torch_finetune_step.py:1104`) |
| `seq` | identity | `seq: params.seq,` (`finetune_step.rs:789`) | `"seq": args.seq,` (`torch_finetune_step.py:1105`) |
| `lora_rank` | identity | `lora_rank: params.lora_rank,` (`finetune_step.rs:790`) | `"lora_rank": args.lora_rank,` (`torch_finetune_step.py:1108`) |
| `lora_alpha` | identity — input was already threaded through `FinetuneStepParams::lora_alpha`, just never emitted before this round | `lora_alpha: params.lora_alpha,` (`finetune_step.rs:791`) | `"lora_alpha": args.lora_alpha,` (`torch_finetune_step.py:1109`) |
| `lora_dropout` | identity | `lora_dropout: params.lora_dropout` (`finetune_step.rs:792`) | `"lora_dropout": args.lora_dropout,` (`torch_finetune_step.py:1110`) |
| `margin` | identity, but jammi HARDCODES `0.3` (no `--margin` CLI flag — the call site's own literal, `let loss = triplet_loss(&a, &p, &n, 0.3)?;` (`finetune_step.rs:469`)) | `margin: 0.3,` (`finetune_step.rs:799`) | `"margin": args.margin,` (`torch_finetune_step.py:1118`) — `--margin` default `0.3` |
| `target_modules` | identity | `target_modules: params.target_modules.clone(),` (`finetune_step.rs:800`) | `"target_modules": [` (`torch_finetune_step.py:1140`) |
| `batched_forward` | identity | `batched_forward: params.batched_forward,` (`finetune_step.rs:801`) | `"batched_forward": args.batched_forward,` (`torch_finetune_step.py:1117`) |
| `backbone_dtype` | identity | `backbone_dtype: format!("{:?}", params.backbone_dtype)` (`finetune_step.rs:784`) | `"backbone_dtype": args.dtype,` (`torch_finetune_step.py:1125`) |
| `steps_measured` | identity — the reachable divergence this table used to miss entirely: two legs measured at a DIFFERENT step count (e.g. a mismatched `--steps`/`--warmup` override) still merged to a "clean" ratio before this field was compared | `steps_measured: times.len(),` (`finetune_step.rs:805`) | `"steps_measured": len(times),` (`torch_finetune_step.py:1166`) |
| `checkpoint_config_sha256` | identity — same base-checkpoint CONTENT identity `grad_oracle.rs`'s tier already carries, added to THIS tier too | `let (checkpoint_config_sha256, _config_len) =` (`finetune_step.rs:577`), via the SHARED streaming `pub(crate) fn sha256_and_len` (`finetune_step.rs:906`) | `checkpoint_identity_fields = checkpoint_identity(args.model_dir)` (`torch_finetune_step.py:996`) |
| `checkpoint_weights_sha256` | identity | `let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =` (`finetune_step.rs:579`) | `"checkpoint_weights_sha256": weights_sha256,` (`torch_finetune_step.py:551`) |
| `checkpoint_weights_size_bytes` | identity | `checkpoint_weights_size_bytes) =` (`finetune_step.rs:579`) — same call as the row above, its second return value | `"checkpoint_weights_size_bytes": weights_len,` (`torch_finetune_step.py:552`) |
| `max_grad_norm` | identity (PR #381 audit B1) — `null` (clip OFF) or the positive finite bound the PRODUCTION `clip_gradients` ran with; a clip-on leg and a clip-off leg compute a different step. `null` is a VALUE for this field (`identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`), never folded into MISSING | `max_grad_norm: params.max_grad_norm,` (`finetune_step.rs`'s tier literal) | `"max_grad_norm": args.max_grad_norm,` in the `finetune_step` block (`torch_finetune_step.py`) |
| `attention_arm` | identity (PR #381 audit B1 class probe) — the attention REFERENCE CLASS the leg was ASKED to run, `"eager"` or `"fused"`; jammi's is the operator's `JAMMI_KERNELS_DISABLE` request (an attention base in `kernels_disabled_requested` ⇒ eager), NEVER the counters (a by-design domain decline is a measurement, not a premise) — see `identity_fields.FINETUNE_IDENTITY_FIELDS`'s own entry | `attention_arm: attention_arm(&kernels_disabled_requested).to_string()` (`finetune_step.rs`'s tier literal) | `"attention_arm": attention_arm_of(resolved_attn_implementation)` in the `finetune_step` block (`torch_finetune_step.py`) |
| `warmup` | identity (PR #381 re-audit) — changes what `clip_invocations` counts (pre-step + warmup + measured) | `warmup: params.warmup,` (`finetune_step.rs`'s tier literal) | `"warmup": args.warmup,` in the `args` block (`torch_finetune_step.py`) — an `_TORCH_ARGS_LEVEL_FIELDS` member |
| `clip_invocations` | measurement (PR #381 audit B2) — the COUNTED number of times the production clip ran this process (pre-step + warmup + measured, every `step_once`), the fact behind a clip-on row rather than a log line; recorded in `leg_provenance` per leg AND cross-checked against `max_grad_norm` by `clip_fact_violations` (clip requested ⇒ `> 0`; not requested ⇒ `== 0`) | `clip_invocations:` (`finetune_step.rs`'s tier literal, a `CLIP_INVOCATIONS` before/after delta) | `"clip_invocations": clip_counter["clip_invocations"]` (`torch_finetune_step.py`) |
| `attn_requested` / `attn_implementation` | provenance — the RAW torch attention string (`--attn` as requested, and what HF resolved it to); the CLASS it implies is compared via `attention_arm` above, the raw string itself is recorded in `leg_provenance`, never compared (see `grad_oracle.rs`'s own table for the fuller rationale) | n/a | `"attn_requested": args.attn,` (`torch_finetune_step.py:1114`) in the `args` block; `attn_implementation` is the sibling `"attn_implementation": resolved_attn_implementation` field further down in the `finetune_step` block |
| `kernels_disabled_requested` / `kernels_disabled_fired` | provenance (K-aux, landed on `main` at `c0f0e98`) — torch has no equivalent env var; recorded in `leg_provenance`, never compared | `let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();` (`finetune_step.rs:768`) | n/a |
| `ln`/`rope`/`softmax`/`geglu`/`lora_epilogue`/`lora_linear`/`attention_block` `_fused_dispatches`/`_eager_dispatches` (14 fields) | measurement — this IS the fused-dispatch proof `fused_proof`/`dispatch_pairs` gate on, and `leg_provenance` additionally records the raw counters per config | `finetune_step.rs`'s own `*_fused_dispatches`/`*_eager_dispatches` fields | n/a |
| `attention_block_flash_fused_dispatches` / `attention_block_flash_declined_dispatches` (P6 Stage B FA2 fold-in — a docs-ci co-sign of `origin/perf/p6-fa2-dense` @ `5886c6b`, NOT on `main` as of this table) | measurement — a CASCADE-shaped pair (`CASCADE_BASES`): no `_eager_dispatches` sibling, its fallback counter is named `_declined_dispatches` instead; absorbs `attention_block` (`ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`), which in turn already absorbs `rope`/`softmax` — one chain, not a second mechanism. A report from `main`'s own binary today carries neither key at all; `dispatch_pairs`/`fused_proof` behave byte-for-byte as before such a report | `report.rs`'s `FinetuneStepTier::attention_block_flash_fused_dispatches`/`::attention_block_flash_declined_dispatches` fields on that branch (not yet in this crate's own `finetune_step.rs`) | n/a |
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
REQUIRED_PAIRS = frozenset({"ln", "geglu"})
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
# `attention_block_flash` is NOT on `main` yet (this table's own `main` at
# the time of writing has zero `flash` references anywhere in
# `crates/jammi-bench/src/`) — a report from `main`'s own binary carries
# NEITHER `attention_block_flash_fused_dispatches` NOR
# `..._declined_dispatches` at all, so `dispatch_pairs` never discovers this
# base on such a report and every fixture in `test_ab_merge.py` that
# predates this change stays green, unmodified, byte-for-byte.
CASCADE_BASES = frozenset({"attention_block_flash"})

ALL_BASES = (
    REQUIRED_PAIRS
    | ABSORBABLE_BY_ATTENTION_BLOCK
    | ABSORBABLE_BY_ATTENTION_BLOCK_FLASH
    | LORA_SITE_EXCLUSIVE_GROUP
    | CASCADE_BASES
)
assert (
    len(REQUIRED_PAIRS)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK)
    + len(ABSORBABLE_BY_ATTENTION_BLOCK_FLASH)
    + len(LORA_SITE_EXCLUSIVE_GROUP)
    + len(CASCADE_BASES)
    == len(ALL_BASES)
), (
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
         INVALID").
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

    merged = {
        "steps": steps,
        "warmup": warmup,
        "pass_ratio_bar": pass_ratio,
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

        if leg_merge_errors["jammi-fused"] is not None:
            proof = f"ERROR: {leg_merge_errors['jammi-fused']}"
        else:
            proof = fused_proof(leg_metrics["jammi-fused"])

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

        for leg in LEGS:
            err_tail = entries[leg]["err_tail"]
            if leg_merge_errors[leg] is not None:
                err_tail = (err_tail + "\n" if err_tail else "") + f"[merge-stage] {leg_merge_errors[leg]}"
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

        fused_m, sdpa_m = leg_metrics["jammi-fused"], leg_metrics["torch-sdpa"]
        ratio = (
            fused_m["triplets_per_s"] / sdpa_m["triplets_per_s"]
            if (fused_m and sdpa_m and sdpa_m["triplets_per_s"])
            else None
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

        any_dry_run = any(entries[leg]["outcome"] == "DRY_RUN" for leg in LEGS)
        torch_fits = entries["torch-sdpa"]["outcome"] == "OK"
        jammi_fused_fits = entries["jammi-fused"]["outcome"] == "OK"

        # The #352 bar is "no OOM where torch fits" -- it binds ONLY when
        # torch-sdpa itself succeeded. If torch-sdpa didn't fit, there is
        # no baseline to hold jammi-fused to and the bar does not apply --
        # that is NOT the same thing as jammi failing, and must not print
        # as FAIL.
        if any_dry_run:
            verdict = "N/A (dry-run)"
        elif not torch_fits:
            verdict = f"N/A (torch-sdpa itself did not fit: {entries['torch-sdpa']['outcome']} — bar does not apply)"
        elif not jammi_fused_fits:
            verdict = f"FAIL (OOM where torch fits: jammi-fused {entries['jammi-fused']['outcome']})"
        elif ratio is None:
            verdict = "FAIL (no ratio: triplets_per_s missing on an OK leg — investigate)"
        elif ratio < pass_ratio:
            verdict = f"FAIL (ratio {ratio:.3f} < {pass_ratio})"
        else:
            verdict = f"PASS (ratio {ratio:.3f})"

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
                f"INVALID (fused-dispatch proof {reason} — this leg's PASS/FAIL classification "
                f"cannot be trusted; the ratio-based verdict this would otherwise have been is "
                f"discarded, not merely annotated)"
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
                f"INVALID (leg premise mismatch: {'; '.join(leg_premise_violations_list)} — the "
                f"{jammi_premise_leg}/{torch_premise_leg} legs of this config did not run under the "
                "same seed/batch/seq/dtype/dropout/lora premise; the ratio-based verdict this would "
                "otherwise have been is discarded, not merely annotated)"
            )

        summary_rows.append((slug, ratio, loss_ratio, verdict))
        merged["configs"][slug] = {
            "legs": {leg: {"outcome": entries[leg]["outcome"], "metrics": leg_metrics[leg]} for leg in LEGS},
            "jammi_fused_dispatch_proof": proof,
            "leg_premise_violations": leg_premise_violations_list,
            "leg_premise_checked_legs": (
                {"jammi": jammi_premise_leg, "torch": torch_premise_leg}
                if leg_premise_violations_list is not None
                else None
            ),
            "leg_premise_not_comparable": leg_premise_not_comparable,
            "provenance": {"jammi": jammi_provenance, "torch": torch_provenance},
            "ratio_jammi_fused_over_torch_sdpa": ratio,
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
        f"{'config':<16}{'leg':<13}{'outcome':<9}{'s/step_p50':<12}{'triplets/s':<12}"
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
            f"{slug:<16}{leg:<13}{outcome:<9}{p50:<12}{tps:<12}{vd:<24}{va:<27}{proof_s:<28}{loss_s:<24}"
        )
        if outcome not in ("OK", "DRY_RUN") and err_tail:
            last = err_tail.splitlines()[-1][:120] if err_tail.splitlines() else ""
            lines.append(f"    -> {last}")
        elif err_tail and "[merge-stage]" in err_tail:
            last = err_tail.splitlines()[-1][:120]
            lines.append(f"    -> {last}")

    lines.append("")
    lines.append(
        f"{'config':<16}{'ratio(fused/sdpa)':<20}{'loss_final_ratio(fused/sdpa,NOT-quality)':<42}{'verdict':<60}"
    )
    for slug, ratio, loss_ratio, verdict in summary_rows:
        ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
        loss_ratio_s = "n/a" if loss_ratio is None else f"{loss_ratio:.4f}"
        lines.append(f"{slug:<16}{ratio_s:<20}{loss_ratio_s:<42}{verdict:<60}")

    table = "\n".join(lines)
    return merged, table


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

    # Advisory (iv), round-2 audit fix on PR #372: the ONE carve-out from
    # this crate's own record-don't-gate doctrine (see `finetune_ab.sh`'s
    # module doc and `build_report`'s own verdict-computation comment) --
    # an `INVALID` verdict (a failed/errored `fused_proof`) is a
    # correctness-of-MEASUREMENT problem, not a machine-dependent
    # performance number, so it is the one thing this sweep's own exit code
    # DOES gate on. An ordinary ratio-based `FAIL` row remains
    # record-only, unchanged.
    invalid_slugs = [
        slug for slug, cfg in merged["configs"].items() if str(cfg.get("verdict", "")).startswith("INVALID")
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
