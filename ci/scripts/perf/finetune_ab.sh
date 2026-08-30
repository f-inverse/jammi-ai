#!/usr/bin/env bash
# The #352 A/B: jammi eager vs jammi fused vs the PyTorch/PEFT reference —
# runs ON THE POD, invoked either via
#   ci/scripts/gpu-dev.sh run <session> bash ci/scripts/perf/finetune_ab.sh
# or directly over ssh once the checkout is on the pod. NOT a CI job (no
# GPU on the CI image).
#
# #352 has two clauses, and this producer discharges only the FIRST:
#   * throughput + no-OOM (the ratio/PASS/FAIL/INDETERMINATE bar this
#     script's own table computes, against a synthetic cost-fixture step —
#     see "Honesty about what is measured" below).
#   * loss-TRAJECTORY equivalence (jammi-fused vs jammi-eager, a REAL
#     trainer, >= 5 seeds) is discharged SEPARATELY, by the pre-registered
#     real-trainer instrument at `docs/plans/63-how-well/measurements/
#     campaign-v2` (`finetune_run_ab.sh` + `ab_merge.py`'s
#     `finetune-run` mode) — never by this script, which never runs a real
#     trainer or a held-out eval.
#
# ## ONE binary, no git-ref switching (this script no longer switches refs)
#
# Every leg below — jammi-eager INCLUDED — runs off the SAME tip binary,
# built ONCE at the start (`build_binary`). jammi-eager is NOT "the
# pre-fusion commit": there is no separate build, no separate ref, and no
# `cargo clean -p jammi-kernels` between legs. It is the tip binary with
# every fused op named in `JAMMI_EAGER_DISABLE_OP_KEYS` (below) forced
# eager via `JAMMI_KERNELS_DISABLE`, under `JAMMI_KERNELS_STRICT=1` as a
# negative control (disable wins over Strict — see that constant's own
# doc). A prior version of this script resolved a separate "eager base"
# commit by grepping commit subjects and rebuilt jammi-kernels between two
# checkouts; that design is gone. Report/table prose calls this leg
# "tip binary, fused ops forced eager" — NEVER "the pre-fusion commit".
#
# What it does, for each of {b8 s128, b8 s512, b16 s128} x {dropout 0,
# dropout 0.05} (6 configs), SIX legs total:
#   1. jammi eager   — tip binary, fused ops forced eager (see above).
#      `JAMMI_KERNELS_STRICT=1` PLUS `--expect-kernels-disabled
#      $JAMMI_EAGER_DISABLE_OP_KEYS` — the negative control: disable wins
#      over Strict (`crates/jammi-kernels/src/admission.rs:60-62`), and
#      `--expect-kernels-disabled` hard-errors before a single step runs if
#      `JAMMI_KERNELS_DISABLE` was dropped, mistyped, or not forwarded to
#      this process — `params.expect_kernels_disabled` (`finetune_step.rs:692-699`)
#      checks it FIRST, before any device/checkpoint/tensor work — so a
#      silently-clean env var can never masquerade as a real eager leg.
#      `kernels_disabled_requested`/`kernels_disabled_fired` are surfaced on
#      this leg's own row in `ab_merge.py`'s printed table.
#   2. jammi fused   — tip binary, JAMMI_KERNELS_STRICT=1, no
#      JAMMI_KERNELS_DISABLE (an admission failure on any fused op ERRORS
#      instead of silently falling back — see
#      jammi-encoders/src/layer_norm.rs's `admission_mode`), so the run
#      cannot pass on a silent eager fallback. Run TWICE per config, in an
#      order-balanced A,B,B,A interleaving with torch-sdpa — see
#      "ORDER-BALANCED BAR LEGS" below.
#   3. torch eager   — crates/jammi-bench/reference/torch_finetune_step.py
#      --attn eager --lora-init peft --dtype bf16 (jammi eager's semantic
#      twin: no fused attention kernel). Single leg (a context leg, not
#      part of the bar ratio).
#   4. torch sdpa    — the same script --attn sdpa (torch's best-case
#      number; what the #352 throughput ratio is measured against). Run
#      TWICE per config, order-balanced against jammi-fused — see below.
# Emits one merged JSON report + a printed table: s/step p50, triplets/s,
# peak VRAM (delta, comparable across stacks, and absolute where torch has
# it — see "VRAM columns" below), EVERY fused-kernel dispatch counter PAIR
# present in the jammi-fused report (the positive-proof channel `ab_merge.py`'s
# `fused_proof` computes: `eager == 0` on EVERY pair, ALWAYS, no exceptions —
# but `fused > 0` is NOT a flat per-pair rule; it is the classification
# `ab_merge.py`'s `REQUIRED_PAIRS`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
# `LORA_SITE_EXCLUSIVE_GROUP` (their union, `ALL_BASES`) declares and that
# module's own doc is the authority on — restated here only for a reader of
# THIS script, not duplicated logic:
#     * ln, geglu, adamw            — each MUST independently show fused > 0.
#     * rope, softmax                — MUST be present; may read (0, 0) ONLY
#                                       when attention_block's OWN fused count
#                                       is > 0 this run (its fused arm folds
#                                       RoPE+softmax into itself and never
#                                       calls their independent sites at all);
#                                       otherwise each independently needs
#                                       fused > 0.
#     * lora_epilogue, lora_linear   — MUST both be present; mutually
#                                       EXCLUSIVE call sites for the same LoRA
#                                       adapted forward, so only their SUM
#                                       needs fused > 0 — either one alone may
#                                       legitimately read (0, 0).
#     * attention_block (P6 Stage B FA2 fold-in) — MUST be present; may read
#                                       (0, 0) ONLY when attention_block_flash's
#                                       own fused count is > 0 this run (the
#                                       flash arm subsumes the fused attention
#                                       block, which subsumes rope/softmax —
#                                       the SAME absorption chain above,
#                                       extended, never a parallel rule);
#                                       otherwise it must independently clear
#                                       fused > 0.
#     * attention_block_flash        — CASCADE-shaped: OPTIONAL (may be
#                                       entirely absent from the report's
#                                       schema, unlike every base above); its
#                                       fallback counter is
#                                       `_declined_dispatches`, not
#                                       `_eager_dispatches`; a nonzero
#                                       `declined` count is a hard fail
#                                       UNLESS `kernels_disabled_requested`
#                                       AND `kernels_disabled_fired` BOTH
#                                       name it — see `ab_merge.py`'s own
#                                       `CASCADE_BASES`/
#                                       `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`
#                                       doc for the full rule table.
# Every classified base (present in `ALL_BASES`) must be PRESENT in the
# report at all — an ABSENT pair (the field renamed, deleted, or
# feature-gated off) is a hard fail for every one of them, never silently
# excluded. A pair discovered in the report but NOT in `ALL_BASES` (a NEW
# fused kernel this script's classification tables were never updated for)
# is a loud per-config schema-drift ERROR, not a silently-ignored base — see
# `fused_proof`/`dispatch_pairs` in the merge stage for the full rule table
# and "whatever lands next" being read off the report's own keys rather than
# a hardcoded name list), each leg's per-step loss_first->loss_last and a
# loss_final_ratio jammi-fused/torch-sdpa column (SAME DATA, COST FIXTURE —
# NOT A QUALITY RESULT, printed only so a large divergence is visible — see
# "loss_final_ratio" below), the ratio jammi-fused/torch-sdpa, and a
# PASS/FAIL/INDETERMINATE against #352's bar (>= 0.9x torch-sdpa throughput
# at matched batch/seq, no OOM on a config torch itself completed). Like
# every jammi-bench tier, this RECORDS — it does not gate the process exit
# code on a config missing the bar; a FAIL or INDETERMINATE row is data for
# a human to read, not an infrastructure failure (see finetune_step.rs's own
# module doc). The script's own exit code reflects whether the sweep RAN,
# not whether every config passed — WITH ONE CARVE-OUT (advisory iv,
# round-2 audit fix on PR #372): a config whose `fused_proof` check FAILED
# or ERRORED reads `INVALID`, not `FAIL`/`PASS`/`INDETERMINATE`, and
# `ab_merge.py`'s own exit code DOES go non-zero on an `INVALID` config — a
# failed proof means the fused kernels may not have actually dispatched at
# all, so the ratio-based classification above it is not trustworthy; that
# is a correctness-of-measurement question, not a machine-dependent
# performance number, and is the one thing this doctrine gates on.
#
# ## ORDER-BALANCED BAR LEGS: A, B, B, A (jammi-fused, torch-sdpa,
# ## torch-sdpa, jammi-fused — never A, A, B, B)
#
# Only the two legs the #352 throughput bar actually gates on
# (jammi-fused == "A", torch-sdpa == "B") run this way — torch-eager and
# jammi-eager stay single legs (context, never part of the bar ratio).
# Mirrors `gpu_inference_ab.sh`'s own documented drift rationale (that
# script's module doc's "What actually cancels, and what does not"
# section): placing the two B-role legs symmetrically between the two
# A-role legs cancels a first-order MULTIPLICATIVE clock/thermal drift
# trend's first-order term under EITHER adjacent-pair averaging or a naive
# mean(B)/mean(A) estimator, under this exact order.
#
# `ab_merge.py` computes TWO adjacent-pair ratios — pair 1 =
# jammi-fused(A1)/torch-sdpa(B1), pair 2 = torch-sdpa-2(B2)/jammi-fused-2(A2)
# read as jammi-fused-2/torch-sdpa-2 — and the BAR ratio is the MIN of the
# two: the estimator LEAST FAVOURABLE to jammi (the same "ratio uses the
# min of two torch runs" convention `docs/maintainer/
# fine-tune-performance-guide.md`'s own stacked-sweep artifact caveat
# already names — this producer applies the identical discipline to its
# own two torch-sdpa runs). When the two pair ratios straddle the 0.9 bar
# (one at-or-above, one below) — or their spread exceeds the bar ratio's
# own distance from 0.9 — the config reports `INDETERMINATE`, never
# `PASS`/`FAIL`: the two repeats disagree too much, relative to how close
# the combined estimate sits to the bar, to trust either classification.
# Both pair ratios are always printed so a human can see why.
#
# IDENTITY-COMPLETENESS: the bar ratio consumes BOTH pair legs, so BOTH
# must carry the SAME measurement discipline. `jammi-fused-2` clears
# `fused_proof` exactly like `jammi-fused` does (an unproven leg feeding
# the pre-registered throughput endpoint is exactly the class `fused_proof`
# exists to catch, regardless of which run it is); `jammi-fused-2`/
# `torch-sdpa-2` are leg-premise-checked against each other exactly like
# `jammi-fused`/`torch-sdpa` are. A failure on EITHER pair — first run or
# second — invalidates the WHOLE config (`INVALID`, `ab_merge.py`'s own
# exit-code carve-out), never silently discarded from just the ratio that
# happened to notice it. `--expect-kernels-disabled`'s SAME hard check
# (see the jammi-eager leg above) also protects `jammi-fused`/
# `jammi-fused-2` implicitly: neither ever sets `JAMMI_KERNELS_DISABLE` at
# all, so `fused_proof` is the only channel that can catch a fused op
# silently falling back on either run.
#
# NOT covered here: loss-TRAJECTORY equivalence between jammi-fused and
# jammi-eager (the #352 quality constraint) is a REAL-TRAINER check over
# >= 5 seeds reusing C0's distributional oracle machinery — a different,
# slower harness than this one-step-timing sweep (see the top of this
# header — `docs/plans/63-how-well/measurements/campaign-v2`). Run it
# separately. The loss_first/loss_last/loss_final_ratio columns THIS
# script prints are a different, weaker thing: one synthetic-data
# cost-fixture step count from `finetune-step`/`torch_finetune_step.py`
# itself, printed for visibility, never a substitute for that real-trainer
# check.
#
# loss_final_ratio (jammi-fused loss_last / torch-sdpa loss_last): SAME
# DATA (both stacks feed the identical synthetic token ids — the two
# scripts' `synthetic_ids` are a literal LCG port of each other; see
# torch_finetune_step.py's module doc), but NOT a quality result: the two
# stacks run different attention-kernel arithmetic and, at this script's
# `--lora-init peft` default, different LoRA init distributions, so a ratio
# far from 1.0 does not indicate either stack regressed — it indicates the
# comparison's preconditions (matched init, matched attention arithmetic)
# were not met, which is exactly why this rides as a printed reference, not
# a gated bar.
#
# ## Build: --features cuda,jammi-encoders/flash-attn (sm_80/86/89/90-only)
#
# `--features cuda` ALONE cannot produce even one VALID config: this
# checkpoint shape's own fused-attention path prefers the FlashAttention-2
# dense cascade (`attention_block_flash`), and with `flash-attn` not
# compiled in, EVERY jammi-fused leg's `attention_block_flash` admit call
# DECLINES — a genuine domain/capability miss recorded as
# `attention_block_flash_declined_dispatches > 0`, which `ab_merge.py`'s
# `fused_proof` rule 1 hard-fails on (an unaccounted-for, unrequested
# decline on ANY pair, in ANY group, is a hard fail) — every jammi-fused
# row in the sweep reads `INVALID`, never PASS/FAIL, until the flash
# feature is compiled in. This build therefore always turns on
# `--features cuda,jammi-encoders/flash-attn` — the SAME convention
# `finetune_run_ab.sh:305`/`fa2_ab.sh:7`/`clip_artifact_producer.sh`'s own
# flash build already use, never a second, independently-drifting
# feature-list spelling.
#
# On THIS script's own path, that refusal is `fused_proof` rule 1's own
# hard fail via `attention_block_flash_declined_dispatches > 0`
# (`report.rs`'s field doc: "declined > 0 on any bench leg -> INVALID") —
# a CapabilityMiss from `flash_capability_gates`
# (`crates/jammi-encoders/src/modernbert.rs`'s `PredicateOutcome::
# CapabilityMiss`) surfacing as a declined dispatch, NOT the SEPARATE
# `flash_compiled` guard `finetune_run_ab.sh`'s own campaign premise check
# (`finetune_run_dispatch_proof_violations`'s `arm == "fused"` branch)
# gates on — the two paths reach a related conclusion (a flash-less build
# cannot certify this sweep) through genuinely different mechanisms; never
# conflate them.
#
# This build is `sm_80`/`sm_86`/`sm_89`/`sm_90`-ONLY by construction — the
# compiled `-gencode` arch set `jammi-kernels/build.rs`'s own
# `GENCODE_ARCHES` const fixes at compile time. An unlisted arch (anything
# outside Ampere/Ada/Hopper's `sm_80/86/89/90`) is a CapabilityMiss the
# SAME way an uncompiled `flash-attn` feature is — declined counters, then
# an `INVALID` `fused_proof` verdict — the correct failure mode: this
# sweep refuses to CLAIM a fused-throughput number on hardware it was
# never validated against, rather than silently falling back to a
# comparison that never exercised the code path it claims to measure.
#
# Torch env: `uv venv "$TORCH_VENV"` (default crates/jammi-bench/reference/
# README.md's own `.venv-torch-ref` convention, resolved under the repo
# root) then `uv pip install --python "$TORCH_VENV/bin/python3" torch
# "transformers>=4.48" peft`. Tolerates an existing venv: if the interpreter
# is already there AND can `import torch, transformers, peft`, it is reused
# rather than reprovisioned (each `uv pip install` re-downloads real GPU
# wheels, and that cost belongs on the first pod session of the day, not
# every invocation of this script). torch/transformers/peft are ORACLE
# dependencies per crates/jammi-bench/reference/README.md's own B2 section —
# never a Cargo dependency, never installed by any CI job, never vendored;
# this script's `uv venv`/`uv pip install` calls are the only place they are
# installed, and only on a pod, only into a venv this script owns.
#
# VRAM columns (binds C8 contract section 2's rules — read
# torch_finetune_step.py's own module doc / crates/jammi-bench/reference/
# README.md for the full derivation; not re-derived here, only the two
# columns this table prints):
#   * "vram_delta(comparable)" — jammi's `peak_vram_bytes` next to torch's
#     `peak_vram_delta_bytes`: same concept (device-memory growth over a
#     baseline read after model+optimizer are resident), the pair this
#     table's numbers are meant to be read against each other.
#   * "vram_absolute(torch only)" — torch's `peak_vram_absolute_bytes`
#     (raw bytes live at the peak, no baseline subtraction). jammi has no
#     analogous field; jammi rows print "n/a" here, not a zero.
#
# Every LoRA-shaped and dtype flag is passed EXPLICITLY and identically to
# both stacks (r=16, alpha=32, dropout as swept, targets Wqkv,Wo,Wi, bf16),
# matching the C8 contract's section 1 spec — this deliberately overrides
# jammi-bench's own CLI defaults (rank 8 / alpha 16), which differ from the
# torch reference script's C8-contract defaults (rank 16 / alpha 32) on
# purpose (see torch_finetune_step.py's --lora-rank/--lora-alpha help text).
# --lora-init defaults to `peft` (AB_TORCH_LORA_INIT, below) — right for
# throughput rows, where the adapter's initial values do not matter. Set
# AB_TORCH_LORA_INIT=jammi to sweep the loss-TRAJECTORY-equivalence
# precondition instead (torch_finetune_step.py re-draws every lora_A matrix
# from jammi's own bound — see that script's own LoRA-init section). jammi's
# own leg always uses its LoraInitMode::ZerosB init; it has no CLI knob for
# this today, so it is not itself switchable — the merged report's
# top-level "lora_init" block records which value each side actually used.
#
# Env vars:
#   MODEL_DIR            checkpoint dir (config.json + weights) BOTH stacks
#                         load from. Required unless AB_DRY_RUN=1.
#   JAMMI_MODEL_DIR       overrides MODEL_DIR for the jammi legs only.
#   TORCH_MODEL_DIR       overrides MODEL_DIR for the torch legs only.
#   AB_CUDA_ORDINAL       CUDA device ordinal both stacks target (default 0).
#   AB_STEPS / AB_WARMUP  measured / warmup step counts (default 20 / 5,
#                         matching both CLIs' own defaults).
#   AB_SEED               synthetic-data + init seed (default 42).
#   AB_PASS_RATIO         the #352 throughput bar (default 0.9).
#   AB_TORCH_LORA_INIT    torch_finetune_step.py's --lora-init for BOTH
#                         torch legs (default "peft" — throughput rows;
#                         "jammi" is required before a loss_final_ratio
#                         column means anything as a trajectory-equivalence
#                         signal — see torch_finetune_step.py's own LoRA-init
#                         section). Must be "peft" or "jammi".
#   AB_OUT_DIR            where the merged report + table land (default
#                         "<repo>/.ab-report/<UTC timestamp>").
#   TORCH_VENV            torch venv path (default "<repo>/.venv-torch-ref").
#   AB_DRY_RUN=1          print every command this script would run (cargo,
#                         uv, the bench binary, the torch script) instead of
#                         executing it, and write a `{"tool":"dry-run",...}`
#                         stub per leg so the merge/table stage still runs
#                         end-to-end against real (if fabricated-empty)
#                         files. Never mutates the checkout, never touches
#                         the network, never claims a real number — every
#                         dry-run row prints outcome DRY_RUN, never OK/FAIL/
#                         OOM.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

# The named constant this script's jammi-eager leg disables — EXACTLY the
# eight LIVE, STANDALONE `admit()`/`op_disabled()` op keys this crate's
# fused finetune-step call graph actually reaches on a real training step
# (confirmed at this contract's tip: `layer_norm_fused`
# `crates/jammi-encoders/src/layer_norm.rs:197`, `geglu_fused`
# `crates/jammi-encoders/src/modernbert.rs:1862`, `attention_block_flash`
# `crates/jammi-encoders/src/modernbert.rs:2523` (`op_disabled`, the
# cascade's own capability gate), `attention_block_fused`
# `crates/jammi-encoders/src/modernbert.rs:1270`, `rope_fused`
# `crates/jammi-encoders/src/modernbert.rs:566`, `softmax_last_dim_fused`
# `crates/jammi-encoders/src/modernbert.rs:1791`, `lora_linear_fused`
# `crates/jammi-lora/src/lora_linear.rs:722`, `adamw_step_fused`
# `crates/jammi-ai/src/fine_tune/adamw.rs:257`).
#
#   * NOT `all` — `crates/jammi-kernels/src/admission.rs:169-177`
#     disclaims it as whole-registry evidence: `unmatched_disables()`
#     coming back empty for `JAMMI_KERNELS_DISABLE=all` proves only that AT
#     LEAST ONE op reached `admit` and was forced eager, never that EVERY
#     registered op was — exactly the wrong guarantee for a leg whose whole
#     job is to certify every fused op ran eager.
#   * NOT the report's `..._fused_dispatches`/`..._eager_dispatches`
#     COUNTER base-names (`ln`, `rope`, `softmax`, `geglu`,
#     `attention_block`, `lora_epilogue`, `lora_linear`, `adamw`) — a
#     DIFFERENT vocabulary from the `admit`/`op_disabled` OP KEYS this
#     env var actually consumes (e.g. the counter base is `ln`, the admit
#     op key is `layer_norm_fused`; the counter base is `attention_block`,
#     the admit op key for its own site is `attention_block_fused`).
#     Naming a counter base directly in `JAMMI_KERNELS_DISABLE` is a
#     silent no-op — it never matches any real `admit`/`op_disabled` call,
#     so `unmatched_disables()` (or, on this leg,
#     `--expect-kernels-disabled`'s own hard check) refuses the run rather
#     than accepting a name that never fired.
#   * NOT `lora_epilogue`/`lora_dropout`/`cast_scale_bf16_f32`/
#     `cast_add_bf16` — `crates/jammi-kernels/src/admission.rs:99-126`:
#     `lora_epilogue`/`lora_dropout` are REGISTERED but PERMANENTLY DEAD
#     (their stand-alone call sites were superseded by the fused LoRA
#     site's single `CustomOp3`, which never calls `admit` for either name
#     — always reads `{fused: 0, eager: 0}`); `cast_scale_bf16_f32`/
#     `cast_add_bf16` are reachable ONLY as a SUBSUMED op inside
#     `lora_linear_fused`'s own admitted branch, never named directly by a
#     caller that wants the whole LoRA site eager. Naming any of the four
#     directly ABORTS the run — a real, present-in-the-registry op name
#     that nonetheless never reaches `admit`/`op_disabled` is exactly as
#     unmatched as a typo
#     (`crates/jammi-bench/tests/finetune_step_kernel_disable.rs:113`'s
#     `kernel_disable_of_a_registered_but_dead_op_name_invalidates_the_run`
#     proves this against the real CLI).
JAMMI_EAGER_DISABLE_OP_KEYS="layer_norm_fused,geglu_fused,attention_block_flash,attention_block_fused,rope_fused,softmax_last_dim_fused,lora_linear_fused,adamw_step_fused"

AB_DRY_RUN="${AB_DRY_RUN:-0}"
AB_CUDA_ORDINAL="${AB_CUDA_ORDINAL:-0}"
AB_STEPS="${AB_STEPS:-20}"
AB_WARMUP="${AB_WARMUP:-5}"
AB_SEED="${AB_SEED:-42}"
AB_PASS_RATIO="${AB_PASS_RATIO:-0.9}"
AB_TORCH_LORA_INIT="${AB_TORCH_LORA_INIT:-peft}"
case "$AB_TORCH_LORA_INIT" in
  peft|jammi) ;;
  *)
    echo "::error::AB_TORCH_LORA_INIT must be 'peft' or 'jammi', got '${AB_TORCH_LORA_INIT}'."
    exit 2
    ;;
esac
TORCH_VENV="${TORCH_VENV:-$REPO_ROOT/.venv-torch-ref}"

MODEL_DIR="${MODEL_DIR:-}"
JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-$MODEL_DIR}"
TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-$MODEL_DIR}"

if [ -z "$JAMMI_MODEL_DIR" ] || [ -z "$TORCH_MODEL_DIR" ]; then
  if [ "$AB_DRY_RUN" = "1" ]; then
    JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER}"
    TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER}"
    echo "::warning::AB_DRY_RUN=1 and MODEL_DIR unset — printed commands use a placeholder path; nothing is read from it."
  else
    echo "::error::MODEL_DIR (or JAMMI_MODEL_DIR/TORCH_MODEL_DIR) must name a checkpoint directory."
    exit 2
  fi
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${AB_OUT_DIR:-$REPO_ROOT/.ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

CONFIGS=("8:128" "8:512" "16:128")
DROPOUTS=("0" "0.05")

# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

# A state-changing command (cargo/uv). Always echoes what it would run;
# under AB_DRY_RUN it never executes. Returns the real exit status otherwise.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# One measurement leg (a jammi or torch invocation). NEVER aborts the
# sweep: OOM or any other failure is recorded as this row's outcome, not
# propagated as a script error — the whole point of an A/B sweep is that
# one config OOM-ing tells you something; it must not hide the other five.
run_leg() {
  local config_slug="$1" leg="$2"
  shift 2
  local -a cmd=("$@")
  local out_file="$RAW_DIR/${config_slug}__${leg}.json"
  local err_file="$RAW_DIR/${config_slug}__${leg}.stderr"
  local exit_file="$RAW_DIR/${config_slug}__${leg}.exit"

  printf -- '--- %s / %s: ' "$config_slug" "$leg"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"config":"%s","leg":"%s"}\n' \
      "$config_slug" "$leg" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${config_slug}/${leg} FAILED (exit ${rc}) — recorded as a row outcome; sweep continues."
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

slug_for() {
  local batch="$1" seq="$2" dropout="$3"
  local dslug
  case "$dropout" in
    0) dslug="d0" ;;
    *) dslug="d${dropout//./p}" ;;
  esac
  printf 'b%s-s%s-%s' "$batch" "$seq" "$dslug"
}

# One jammi leg. `disable_ops` (optional, arg 8): when non-empty, names the
# `JAMMI_KERNELS_DISABLE` op-key list AND the `--expect-kernels-disabled`
# hard check (see header) — the jammi-eager leg's own call shape. Every
# jammi leg runs `JAMMI_KERNELS_STRICT=1`: an eligible-but-failed fused op
# ERRORS instead of falling back, so no leg can silently "pass" on eager
# numbers wearing a fused label
# (jammi-encoders/src/layer_norm.rs::admission_mode) — disable wins over
# Strict (admission.rs:60-62), which is exactly the deliberate,
# self-describing negative control `disable_ops` exists to run, never a
# silent Strict-mode bypass.
run_jammi_leg() {
  local leg="$1" binary="$2" model_dir="$3" batch="$4" seq="$5" dropout="$6" config_slug="$7"
  local disable_ops="${8:-}"
  local -a cmd=(
    "$binary" finetune-step
    --model-dir "$model_dir"
    --batch "$batch" --seq "$seq"
    --steps "$AB_STEPS" --warmup "$AB_WARMUP"
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$dropout"
    --target-modules "Wqkv,Wo,Wi"
    --backbone-dtype bf16
    --cuda "$AB_CUDA_ORDINAL" --seed "$AB_SEED"
    --batched-forward true
  )
  if [ -n "$disable_ops" ]; then
    cmd+=(--expect-kernels-disabled "$disable_ops")
    JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$disable_ops" run_leg "$config_slug" "$leg" "${cmd[@]}"
  else
    JAMMI_KERNELS_STRICT=1 run_leg "$config_slug" "$leg" "${cmd[@]}"
  fi
}

run_torch_leg() {
  local attn="$1" leg="$2" python_bin="$3" ref_script="$4" model_dir="$5" batch="$6" seq="$7" dropout="$8" config_slug="$9"
  local -a cmd=(
    "$python_bin" "$ref_script"
    --model-dir "$model_dir"
    --batch "$batch" --seq "$seq"
    --steps "$AB_STEPS" --warmup "$AB_WARMUP"
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$dropout"
    --target-modules "Wqkv,Wo,Wi"
    --dtype bf16 --attn "$attn" --lora-init "$AB_TORCH_LORA_INIT"
    --cuda "$AB_CUDA_ORDINAL" --seed "$AB_SEED"
  )
  run_leg "$config_slug" "$leg" "${cmd[@]}"
}

# Provision (or reuse) the torch reference venv — crates/jammi-bench/
# reference/README.md's own `.venv-torch-ref` convention.
setup_torch_venv() {
  local py="$TORCH_VENV/bin/python3"
  if [ -x "$py" ] && [ "$AB_DRY_RUN" != "1" ] && "$py" -c 'import torch, transformers, peft' >/dev/null 2>&1; then
    echo "torch venv at $TORCH_VENV already has torch+transformers+peft — reusing."
    return 0
  fi
  if [ -x "$py" ] && [ "$AB_DRY_RUN" = "1" ]; then
    echo "torch venv at $TORCH_VENV exists — [dry-run] would verify torch+transformers+peft import before reprovisioning."
  fi
  echo "provisioning torch venv at $TORCH_VENV"
  run_cmd uv venv "$TORCH_VENV" || { echo "::error::uv venv failed"; exit 1; }
  run_cmd uv pip install --python "$py" torch "transformers>=4.48" peft \
    || { echo "::error::uv pip install (torch/transformers/peft) failed"; exit 1; }
}

# Build ONCE, at the very start — no ref-switching, no in-script checkout,
# no jammi-kernels clean-on-switch (A/C: this script no longer has more
# than one build). `--features cuda,jammi-encoders/flash-attn` — see
# header for why `cuda` alone cannot produce even one VALID config.
build_binary() {
  echo "=== building jammi-bench (--features cuda,jammi-encoders/flash-attn) ==="
  run_cmd cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda,jammi-encoders/flash-attn failed"; exit 1; }
  check_bin_provenance "$JAMMI_BIN"
}

# --- provenance cross-check (unification contract C5.1), same shape as
# stacked_sweep.sh/clip_artifact_producer.sh/fa2_ab.sh: called immediately
# after the one build above, BEFORE any leg runs. Refuses if the
# jammi-bench binary's own baked identity does not match the sha ACTUALLY
# checked out (`git rev-parse HEAD`). `unknown`/a `-dirty` suffix can
# never equal a resolved 40-hex sha, so a single string-equality check
# catches mismatch/unknown/dirty uniformly; an empty reading is ALSO a
# refusal, never silently skipped -- never a leg silently marked GREEN off
# a binary that was not built cleanly at HEAD.
check_bin_provenance() {
  local bin="$1"
  if [ "$AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  local sha sha_re='^[0-9a-fA-F]{40}$'
  sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if ! [[ "$sha" =~ $sha_re ]]; then
    echo "::error::HEAD did not resolve to a 40-hex commit ('$sha') -- refusing" >&2
    exit 1
  fi
  local bin_prov_json bin_prov_sha
  bin_prov_json="$("$bin" provenance 2>&1)" || { echo "::error::'$bin provenance' failed: $bin_prov_json" >&2; exit 1; }
  bin_prov_sha="$(printf '%s' "$bin_prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$bin provenance' output: $bin_prov_json" >&2; exit 1; }
  if [ -z "$bin_prov_sha" ] || [ "$bin_prov_sha" != "$sha" ]; then
    echo "::error::'$bin provenance' reports build_sha=$bin_prov_sha, but this checkout is at sha=$sha -- refusing before any leg runs off this binary." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------- #
# build + sweep (ONE binary, no ref-switching)
# ---------------------------------------------------------------------- #

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
JAMMI_BIN="$TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_finetune_step.py"
TORCH_PY="$TORCH_VENV/bin/python3"

echo "=== tip binary: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown) ==="

build_binary
setup_torch_venv

for cfg in "${CONFIGS[@]}"; do
  BATCH="${cfg%%:*}"; SEQ="${cfg##*:}"
  for DROPOUT in "${DROPOUTS[@]}"; do
    SLUG="$(slug_for "$BATCH" "$SEQ" "$DROPOUT")"

    # Context legs (single, never part of the bar ratio).
    run_jammi_leg jammi-eager "$JAMMI_BIN" "$JAMMI_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG" "$JAMMI_EAGER_DISABLE_OP_KEYS"
    run_torch_leg eager torch-eager "$TORCH_PY" "$REF_SCRIPT" "$TORCH_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"

    # Order-balanced bar legs: A, B, B, A (jammi-fused, torch-sdpa,
    # torch-sdpa, jammi-fused) — see header's "ORDER-BALANCED BAR LEGS".
    run_jammi_leg jammi-fused "$JAMMI_BIN" "$JAMMI_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG" ""
    run_torch_leg sdpa torch-sdpa "$TORCH_PY" "$REF_SCRIPT" "$TORCH_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
    run_torch_leg sdpa torch-sdpa-2 "$TORCH_PY" "$REF_SCRIPT" "$TORCH_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
    run_jammi_leg jammi-fused-2 "$JAMMI_BIN" "$JAMMI_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG" ""
  done
done

# ---------------------------------------------------------------------- #
# merge + table
# ---------------------------------------------------------------------- #
# B3: this used to be an inline heredoc with zero automated coverage
# (AB_DRY_RUN=1 only ever exercised the DRY_RUN arm). It is now
# ci/scripts/perf/ab_merge.py, an importable module `ci/scripts/perf/
# test_ab_merge.py` drives directly against fixture leg directories — this
# call is exactly the "real entry point" that test suite exercises.
python3 "$DIR/ab_merge.py" "$RAW_DIR" "$OUT_DIR" "$AB_STEPS" "$AB_WARMUP" "$AB_PASS_RATIO" "$AB_TORCH_LORA_INIT"
PY_RC=$?

echo
echo "=== merged report + table: ${OUT_DIR} ==="
exit "$PY_RC"
