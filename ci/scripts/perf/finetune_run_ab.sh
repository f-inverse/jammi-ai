#!/usr/bin/env bash
# The how-well producer: runs the legs of the kernel edge of the `train-run`
# ladder -- `resident-reference` (the flash cascade and fused AdamW disabled)
# against `resident` (the fused kernels) -- over the committed
# `cookbook/fixtures/finetune_heldout/` held-out fixture, then hands the legs
# to `jammi-bench ladder train-run`. One `jammi-bench finetune-run` leg per
# (rung, seed, take): `r1`/`r2` same-seed repeats, and the `lr0` control.
# This script runs legs and decides nothing; every premise, the identity
# check, the paired sign test, the equivalence test and the mutant columns
# are the ladder's (`crates/jammi-bench/src/ladder/`).
#
# NOT `stacked_sweep.sh`-shaped for its measured legs: no cookbook book
# stack, no server. Every input a MEASURED leg reads is a committed repo
# path (the fixture under `cookbook/fixtures/finetune_heldout/`, a local
# `--model-dir` checkpoint the operator already has on-box); no leg itself
# builds the cookbook corpus, starts a `jammi-server`, or touches the
# network. The ONE exception is the PRE-RUN provisioning step below,
# which runs strictly BEFORE any measured
# leg and is never counted as one: it may invoke the book-side
# `cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs`
# (network-backed, checksum-gated) to (re)populate `train_pairs.jsonl`, then
# ALWAYS byte-verifies it against the committed `train_ids_sha256.json`
# before letting any leg proceed.
#
# HELD-OUT FIXTURE LAYOUT (cookbook/fixtures/finetune_heldout/):
#   heldout_ids.txt      the committed held-out id list -- what
#                         `heldout_ids_sha256` hashes.
#   heldout_pairs.jsonl  the FULL held-out pair text (committed).
#   train_ids_sha256.json ids + a per-pair SHA-256 for the 1372 TRAIN-side
#                         pairs -- deliberately NOT full text (repo-size
#                         discipline, that directory's own README.md "Why
#                         train text isn't committed" section). This means
#                         `--train-jsonl` (a required `jammi-bench
#                         finetune-run` flag) has no committed source of its
#                         own text in this checkout.
#
# PRE-RUN PROVISIONING: before any measured
# leg, if `$TRAIN_JSONL` (default `$REPO_ROOT/cookbook/fixtures/
# finetune_heldout/train_pairs.jsonl`, gitignored -- never committed) is
# absent, this script invokes the book-side producer's own
# `--emit-train-pairs` mode (network-backed, checksum-gated; reuses the
# exact `mine_pairs()`/`_text()` code path `--check` already re-derives
# against) to write it. Then -- REGARDLESS of whether the file was just
# emitted or was already present on this pod from a prior run -- this
# script ALWAYS byte-verifies every pair against the committed
# `train_ids_sha256.json` via the standalone
# `ci/scripts/perf/verify_train_pairs.py` (sha256 per pair id, exact count
# 1372, no extras/duplicates), refusing loudly with the first mismatching
# id on any divergence, before a single leg runs. A pre-existing
# `train_pairs.jsonl` is never trusted on name alone: a stale or
# hand-edited file left over from an earlier checkout fails this exactly
# like a corrupted fresh fetch would.
#
# Batch size: 32 (`cookbook/fixtures/finetune_heldout/README.md` -- the
# chapter-config value every real `db.fine_tune(...)` call over this exact
# pair set already uses; 128 held-out pairs is a multiple of both 32 and the engine's own
# unset-default 8, so this pick does not change the fixture's own held-out
# count, only which "N held-out = k batches" framing is reported).
#
# Objective: MNRL by default -- override with FINETUNE_RUN_AB_OBJECTIVE=
# triplet. The objective is chosen by a dynamic-range probe outside this
# script; this script just runs whichever one it is told, over both arms, at
# every pre-registered seed.
#
# Seeds: the pre-registered 12-seed gate set (N=12 seeds x 2 arms), 1..12
# by default -- override with FINETUNE_RUN_AB_SEEDS (a
# comma-separated list, no spaces).
#
# Env vars:
#   MODEL_DIR                 checkpoint dir (config.json + model.safetensors
#                              + tokenizer.json). Required unless
#                              FINETUNE_RUN_AB_DRY_RUN=1.
#   FINETUNE_RUN_AB_SEEDS      comma-separated seed list (default: 1..12,
#                              the pre-registered gate set).
#   FINETUNE_RUN_AB_OBJECTIVE  "mnrl" or "triplet" (default: mnrl).
#   FINETUNE_RUN_AB_EPOCHS     epochs per leg (default: 3).
#   FINETUNE_RUN_AB_BATCH      batch size (default: 32 -- see "Batch size"
#                              above).
#   FINETUNE_RUN_AB_LR         --lr passthrough for the main A/B legs
#                              (default: unset, so the CLI's own default
#                              (2e-4, main.rs's `FinetuneRunArgs::lr`) is
#                              used).
#   FINETUNE_RUN_AB_LR0_SEEDS  comma-separated seed list for the lr=0 RED
#                              control (an lr=0 arm over >= 2 seeds must fail
#                              learning-happened); default
#                              empty = skipped). Each seed here runs BOTH
#                              rungs at --lr 0 as the `lr0` take -- a
#                              control leg, never a measured repeat: the
#                              ladder checks each one ran at lr=0 and FAILS
#                              learning-happened, and never counts it into
#                              the paired statistic.
#   FINETUNE_RUN_AB_ALLOW_NO_LR0
#                              Default "0": the kernel edge declares the
#                              lr=0 control at two seeds, and the ladder
#                              REFUSES (INVALID) an edge whose control is
#                              missing. Set to "1" to pass the ladder's
#                              `--waive-control` flag instead, recording a
#                              DELIBERATE, visible opt-out in the verdict
#                              (`control.waived`) rather than an unstated
#                              default.
#   FINETUNE_RUN_AB_MUTANT_LEGS
#                              OPTIONAL, ';'-separated list of
#                              'LABEL:PATCH_SHA256' specs, forwarded as one
#                              '--mutant SPEC' per entry to the ladder. PURE
#                              pass-through: this script never runs a mutant
#                              leg itself (a mutant is a patched
#                              jammi-kernels build, run from a scratch
#                              worktree); this variable only tells the ladder
#                              which already-produced
#                              'mutant-<LABEL>__seed<N>__r1.json' legs under
#                              THIS run's own $RAW_DIR are columns. Default
#                              empty = no mutant columns.
#   FINETUNE_RUN_AB_BACKBONE_DTYPE
#                              --backbone-dtype passthrough for EVERY leg
#                              `run_leg` runs -- both A/B arms, every seed,
#                              AND the lr=0 RED control below (default
#                              "bf16"). `backbone_dtype` is an identity
#                              field of `FinetuneRunTier` -- the ladder
#                              requires every leg of the comparison to
#                              report the SAME value, so this is read ONCE
#                              here and forwarded unconditionally from the
#                              one `run_leg` both loops (main sweep, lr=0
#                              control) share, never overridden per-arm/
#                              per-loop. `main.rs`'s own CLI accepts "f32",
#                              "f16", or "bf16" (`FinetuneRunArgs::
#                              backbone_dtype`'s match arms) -- this script
#                              does not itself re-validate the value, it
#                              relies on the binary's own refusal of an
#                              unrecognized spelling.
#   FINETUNE_RUN_AB_CUDA       CUDA ordinal (default: 0). Unset
#                              FINETUNE_RUN_AB_CPU=1 to omit --cuda entirely
#                              (the CPU-hermetic smoke path finetune-run's
#                              own CLI doc names) -- never both.
#   TRAIN_JSONL / HELDOUT_IDS / HELDOUT_JSONL
#                              override the committed-fixture paths (see
#                              "HELD-OUT FIXTURE LAYOUT" / "PRE-RUN
#                              PROVISIONING" above -- TRAIN_JSONL's default
#                              is auto-provisioned + byte-verified before any
#                              leg runs, never committed itself).
#   FINETUNE_RUN_AB_OUT_DIR    where the raw legs + ladder verdict land
#                              (default "<repo>/.finetune-run-ab-report/
#                              <UTC timestamp>").
#   FINETUNE_RUN_AB_PROVISION_PYTHON
#                              the python interpreter invoked for the ONE
#                              provisioning step above (`derive_heldout_
#                              fixture.py --emit-train-pairs`) -- default
#                              "python3" (a bare checkout's system
#                              interpreter, which is enough when
#                              `train_pairs.jsonl` is already pre-staged, so
#                              this step never actually runs). A pod driver
#                              (e.g. `ci/scripts/runpod_gpu_howwell.sh`) that
#                              provisions a dedicated venv for
#                              `jammi_cookbook`/numpy/pyarrow/requests (a
#                              bare pod has no pip on PATH and this script's
#                              own producer binary
#                              build/run never needs any of those packages)
#                              points this at that venv's own interpreter
#                              instead; every OTHER step in this script
#                              (verification, the cargo build, every measured
#                              leg) stays on plain "python3"/the system
#                              toolchain -- MEASURED legs are deliberately
#                              venv-free, only this one pre-run provisioning
#                              call is not.
#   FINETUNE_RUN_AB_DRY_RUN=1  print every command this script would run,
#                              the ladder invocation included, instead of
#                              executing it. Never mutates the checkout,
#                              never touches the network, writes no leg.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

FINETUNE_RUN_AB_DRY_RUN="${FINETUNE_RUN_AB_DRY_RUN:-0}"
FINETUNE_RUN_AB_SEEDS="${FINETUNE_RUN_AB_SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12}"
FINETUNE_RUN_AB_OBJECTIVE="${FINETUNE_RUN_AB_OBJECTIVE:-mnrl}"
case "$FINETUNE_RUN_AB_OBJECTIVE" in
  mnrl|triplet) ;;
  *)
    echo "::error::FINETUNE_RUN_AB_OBJECTIVE must be 'mnrl' or 'triplet', got '${FINETUNE_RUN_AB_OBJECTIVE}'." >&2
    exit 2
    ;;
esac
FINETUNE_RUN_AB_EPOCHS="${FINETUNE_RUN_AB_EPOCHS:-3}"
FINETUNE_RUN_AB_BATCH="${FINETUNE_RUN_AB_BATCH:-32}"
# --lr passthrough. Unset means "omit --lr entirely", i.e. the CLI's own default (2e-4) -- never fabricate
# a value here that main.rs's own `#[arg(long, default_value_t = 2e-4)]`
# already owns.
FINETUNE_RUN_AB_LR="${FINETUNE_RUN_AB_LR:-}"
# lr=0 RED control seeds -- comma-separated,
# default empty (skipped). NEVER added to FINETUNE_RUN_AB_SEEDS/the main
# sweep loop below; run through their own dedicated loop as the `lr0` take.
FINETUNE_RUN_AB_LR0_SEEDS="${FINETUNE_RUN_AB_LR0_SEEDS:-}"
# --backbone-dtype passthrough for EVERY leg (see env-var doc above).
FINETUNE_RUN_AB_BACKBONE_DTYPE="${FINETUNE_RUN_AB_BACKBONE_DTYPE:-bf16}"
FINETUNE_RUN_AB_CUDA="${FINETUNE_RUN_AB_CUDA:-0}"
FINETUNE_RUN_AB_CPU="${FINETUNE_RUN_AB_CPU:-0}"
# Interpreter for the one provisioning step -- see the env-var doc above.
FINETUNE_RUN_AB_PROVISION_PYTHON="${FINETUNE_RUN_AB_PROVISION_PYTHON:-python3}"

FIXTURE_DIR="$REPO_ROOT/cookbook/fixtures/finetune_heldout"
TRAIN_JSONL="${TRAIN_JSONL:-$FIXTURE_DIR/train_pairs.jsonl}"
HELDOUT_IDS="${HELDOUT_IDS:-$FIXTURE_DIR/heldout_ids.txt}"
HELDOUT_JSONL="${HELDOUT_JSONL:-$FIXTURE_DIR/heldout_pairs.jsonl}"

MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ]; then
  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    MODEL_DIR="/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER"
    echo "::warning::FINETUNE_RUN_AB_DRY_RUN=1 and MODEL_DIR unset — printed commands use a placeholder path; nothing is read from it."
  else
    echo "::error::MODEL_DIR must name a checkpoint directory (config.json + model.safetensors + tokenizer.json)." >&2
    exit 2
  fi
fi

# Refuse loudly, before any leg runs, if the fixture's real held-out files
# are absent -- a real run over a missing/renamed fixture must not silently
# produce a stub-shaped FAIL row indistinguishable from a real training
# failure.
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  for f in "$HELDOUT_IDS" "$HELDOUT_JSONL"; do
    if [ ! -f "$f" ]; then
      echo "::error::committed fixture file not found: $f (cookbook/fixtures/finetune_heldout/) — refusing before any leg runs." >&2
      exit 1
    fi
  done

  # --- PRE-RUN provisioning -- see module
  # doc "PRE-RUN PROVISIONING" above. Outside every measured leg: this runs
  # once, before the sweep loop, never inside run_leg. Emit is SKIPPED
  # whenever `$TRAIN_JSONL` is already present (an operator/pod driver may
  # pre-stage it) -- byte-verification below still ALWAYS runs regardless.
  #
  # Invoked from `cookbook/book` as cwd, per that
  # directory's own fixture README ("cd cookbook/book && python scripts/
  # derive_heldout_fixture.py ...") -- `derive_heldout_fixture.py` itself
  # resolves every path it reads/writes off `__file__`, never cwd, so this
  # is the DOCUMENTED invocation convention, not a functional requirement of
  # that script; `$FINETUNE_RUN_AB_PROVISION_PYTHON` (default "python3") is
  # this call's own interpreter knob -- a bare checkout's system Python
  # cannot `import jammi_cookbook`/numpy, so a pod driver that provisions a
  # dedicated venv for this ONE step points this env var at that venv's
  # interpreter instead (see the env-var's own doc above).
  if [ ! -f "$TRAIN_JSONL" ]; then
    echo "::notice::$TRAIN_JSONL not found -- provisioning via 'derive_heldout_fixture.py --emit-train-pairs' (network-backed, checksum-gated fetch of train text; outside measured legs)."
    (cd "$REPO_ROOT/cookbook/book" && "$FINETUNE_RUN_AB_PROVISION_PYTHON" scripts/derive_heldout_fixture.py --emit-train-pairs) \
      || { echo "::error::train-pairs provisioning failed (cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs, invoked from cookbook/book via \$FINETUNE_RUN_AB_PROVISION_PYTHON='$FINETUNE_RUN_AB_PROVISION_PYTHON') — refusing before any leg runs." >&2; exit 1; }
  else
    echo "::notice::$TRAIN_JSONL already present -- skipping the emit step (pre-staged); byte-verification below still always runs."
  fi
  # ALWAYS byte-verify -- whether train_pairs.jsonl was just emitted above or
  # was already present on this pod from a prior run. A stale/hand-edited
  # file must fail exactly like a corrupted fresh fetch would; this is the
  # ONE reviewable unit both this producer and any other caller share
  # (ci/scripts/perf/verify_train_pairs.py), never a second hand-rolled
  # comparator.
  python3 "$DIR/verify_train_pairs.py" --pairs "$TRAIN_JSONL" \
    || { echo "::error::$TRAIN_JSONL failed byte-verification against cookbook/fixtures/finetune_heldout/train_ids_sha256.json — refusing before any leg runs." >&2; exit 1; }
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${FINETUNE_RUN_AB_OUT_DIR:-$REPO_ROOT/.finetune-run-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"

# --- state-changing command wrapper (same shape as finetune_ab.sh/
# encode_ab.sh's own run_cmd): always echoes what it would run; under
# FINETUNE_RUN_AB_DRY_RUN never executes.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# The arms are pre-registered as "fused cascade vs ALLOFF=
# attention_block_flash,adamw_step_fused" -- the A/B's own differential IS
# the flash cascade. Building WITHOUT flash-attn makes
# attention_block_flash unable to dispatch in EITHER arm, nulling the
# experiment -- mirrors
# stacked_sweep.sh's own flash-A/B build feature list exactly
# (`--features cuda,jammi-encoders/flash-attn`), never a second,
# independently-drifting feature-list spelling.
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  run_cmd cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda,jammi-encoders/flash-attn failed" >&2; exit 1; }
fi

# --- provenance cross-check, same shape as
# finetune_ab.sh/encode_ab.sh/stacked_sweep.sh/
# clip_artifact_producer.sh: refuse BEFORE any leg runs if the binary's own
# baked identity does not match the sha this checkout is actually at.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BIN" provenance 2>&1)" || { echo "::error::'$BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# --- one measurement leg, filed as `<rung>__seed<N>__<take>.json` -- the
# ladder's leg name. NEVER aborts the sweep: a failed leg leaves its stdout,
# stderr and exit code beside where its `.json` would be and no `.json`, so
# the ladder names it as a missing leg rather than this script discarding
# every other seed's row.
#
# `arm` selects BOTH the CLI's own `--arm` flag (recorded on the report,
# report.rs's own PROVENANCE_FIELDS) AND, for the `alloff` arm
# only, the `JAMMI_KERNELS_DISABLE` env var this binary's own CLI doc names
# as the CALLER's responsibility ("the caller is responsible for setting
# JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused itself
# before invoking this binary for the alloff arm" -- main.rs's own
# `FinetuneRunArgs::arm` doc).
#
# `lr_override` (5th, optional): when non-empty, forwarded as `--lr`
# (main.rs's own CLI flag) -- the lr=0 RED control loop below passes
# `"0"` explicitly; the main A/B loop passes `$FINETUNE_RUN_AB_LR`, which is
# empty by default (omit --lr entirely, i.e. the CLI's own 2e-4 default).
run_leg() {
  local seed="$1" arm="$2" repeat="$3" work_dir="$4" lr_override="${5:-}"
  local rung="resident"
  if [ "$arm" = "alloff" ]; then
    rung="resident-reference"
  fi
  local leg="$RAW_DIR/${rung}__seed${seed}__${repeat}"
  local out_file="$leg.stdout" err_file="$leg.stderr" exit_file="$leg.exit"

  local -a cmd=(
    "$BIN" finetune-run
    --model-dir "$MODEL_DIR"
    --arm "$arm"
    --train-jsonl "$TRAIN_JSONL"
    --heldout-ids "$HELDOUT_IDS"
    --heldout-jsonl "$HELDOUT_JSONL"
    --seed "$seed"
    --epochs "$FINETUNE_RUN_AB_EPOCHS"
    --batch "$FINETUNE_RUN_AB_BATCH"
    --objective "$FINETUNE_RUN_AB_OBJECTIVE"
    # Early stopping DISABLED both arms -- the "never
    # stops before the pre-registered epoch budget" idiom, so a seed's
    # trajectory is never truncated by an early-stopping decision the sign
    # test would then have to account for.
    --early-stopping-patience 10000
    # `main.rs`'s own `--backbone-dtype` default
    # is `f32` (`FinetuneRunArgs::backbone_dtype`'s own `#[arg(long,
    # default_value = "f32")]`), and `flash_capability_gates` DomainMisses
    # the whole flash cascade whenever `dtype` is neither `DType::BF16` nor
    # `DType::F16` (`jammi-encoders/src/modernbert.rs`'s
    # `dtype_is_bf16_or_f16` gate — `f32` is outside that admitted set, and
    # this script's own default, `bf16`, is inside it). The flash cascade
    # is the `fused` arm's pre-registered admitted branch, so an unset
    # `--backbone-dtype` (silently f32) makes `attention_block_flash`
    # unable to fire on EITHER arm's real leg -- the same null differential
    # the flash-attn build feature above prevents. `backbone_dtype` is
    # also an identity field of `FinetuneRunTier` -- the ladder
    # requires every leg (both rungs, every seed, INCLUDING the lr=0
    # control below) to report the SAME value, so this is passed
    # unconditionally here in the one `run_leg` both loops share, never
    # only on the `fused` arm. Value comes from
    # `$FINETUNE_RUN_AB_BACKBONE_DTYPE` (default "bf16" -- see that
    # env-var's own doc above).
    --backbone-dtype "$FINETUNE_RUN_AB_BACKBONE_DTYPE"
    --work-dir "$work_dir"
  )
  if [ -n "$lr_override" ]; then
    cmd+=(--lr "$lr_override")
  fi
  if [ "$FINETUNE_RUN_AB_CPU" != "1" ]; then
    cmd+=(--cuda "$FINETUNE_RUN_AB_CUDA")
  fi

  printf -- '--- seed%s/%s/%s: ' "$seed" "$arm" "$repeat"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    return 0
  fi

  local rc=0
  if [ "$arm" = "alloff" ]; then
    JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  else
    "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  fi
  echo "$rc" > "$exit_file"
  if [ "$rc" -eq 0 ]; then
    mv "$out_file" "$leg.json"
  fi
  if [ "$rc" -ne 0 ]; then
    echo "::warning::seed${seed}/${arm}/${repeat} FAILED (exit ${rc}) — recorded as a leg outcome; sweep continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

IFS=',' read -r -a SEEDS <<< "$FINETUNE_RUN_AB_SEEDS"

for seed in "${SEEDS[@]}"; do
  for arm in fused alloff; do
    for repeat in r1 r2; do
      work_dir="$OUT_DIR/work/seed${seed}__${arm}__${repeat}"
      mkdir -p "$work_dir"
      run_leg "$seed" "$arm" "$repeat" "$work_dir" "$FINETUNE_RUN_AB_LR"
    done
  done
done

# --- lr=0 RED control legs:
# both rungs, at --lr 0, filed as the `lr0` take -- a control, never a
# measured repeat, so the ladder never counts one into the paired statistic.
# Skipped entirely when FINETUNE_RUN_AB_LR0_SEEDS is unset; the ladder then
# refuses the edge unless the control is waived (see below).
if [ -n "$FINETUNE_RUN_AB_LR0_SEEDS" ]; then
  IFS=',' read -r -a LR0_SEEDS <<< "$FINETUNE_RUN_AB_LR0_SEEDS"
  for seed in "${LR0_SEEDS[@]}"; do
    for arm in fused alloff; do
      work_dir="$OUT_DIR/work/seed${seed}__${arm}__lr0"
      mkdir -p "$work_dir"
      run_leg "$seed" "$arm" "lr0" "$work_dir" "0"
    done
  done
fi

# --- compare: the kernel edge of the `train-run` ladder over this run's legs.
# Outcome axis only: a `finetune-run` leg carries the held-out loss, not a
# per-iteration time series. The ladder's verdict (`ladder_verdict.json`,
# `ladder_table.txt`) is this run's decision, and its exit code this
# script's: non-zero on a refusal (INVALID) or a fired decision rule (RED,
# RED_FOR_INVESTIGATION).
#
# The kernel edge declares the lr=0 control; an edge whose control is missing
# is refused unless FINETUNE_RUN_AB_ALLOW_NO_LR0=1 forwards `--waive-control`
# -- a deliberate, visible opt-out recorded in the verdict, never a silent
# default.
FINETUNE_RUN_AB_ALLOW_NO_LR0="${FINETUNE_RUN_AB_ALLOW_NO_LR0:-0}"
LADDER_ARGS=(ladder train-run "$RAW_DIR" --from resident-reference --to resident --axes outcome --out "$OUT_DIR")
if [ "$FINETUNE_RUN_AB_ALLOW_NO_LR0" = "1" ]; then
  LADDER_ARGS+=(--waive-control)
fi
# Mutant legs are produced OUTSIDE this script (a patched jammi-kernels build
# in a scratch worktree) and filed under THIS run's own $RAW_DIR as
# `mutant-<LABEL>__seed<N>__r1.json`; each ';'-separated 'LABEL:PATCH_SHA256'
# spec here names one column for the ladder to judge, by the kernel edge's own
# rules, against this run's `resident-reference` legs. Default empty = none.
FINETUNE_RUN_AB_MUTANT_LEGS="${FINETUNE_RUN_AB_MUTANT_LEGS:-}"
if [ -n "$FINETUNE_RUN_AB_MUTANT_LEGS" ]; then
  IFS=';' read -r -a MUTANT_LEG_SPECS <<< "$FINETUNE_RUN_AB_MUTANT_LEGS"
  for spec in "${MUTANT_LEG_SPECS[@]}"; do
    [ -n "$spec" ] && LADDER_ARGS+=(--mutant "$spec")
  done
fi
run_cmd "$BIN" "${LADDER_ARGS[@]}"
LADDER_RC=$?

echo
echo "=== raw legs + ladder verdict: ${OUT_DIR} ==="
exit "$LADDER_RC"
