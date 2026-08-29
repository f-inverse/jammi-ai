#!/usr/bin/env bash
# The unit-63 how-well producer (H4b, docs-ci domain): drives
# `jammi-bench finetune-run` (CONTRACT H4) over the committed
# `cookbook/fixtures/finetune_heldout/` held-out fixture (CONTRACT H3), one
# leg per (seed, arm, repeat) — `{fused, alloff}` arms, `{r1, r2}` same-seed
# repeats — against the SAME committed fixture and objective every leg of a
# run shares.
#
# NOT `stacked_sweep.sh`-shaped for its measured legs: no cookbook book
# stack, no server. Every input a MEASURED leg reads is a committed repo
# path (the fixture under `cookbook/fixtures/finetune_heldout/`, a local
# `--model-dir` checkpoint the operator already has on-box); no leg itself
# builds the cookbook corpus, starts a `jammi-server`, or touches the
# network. The ONE exception is the PRE-RUN provisioning step below
# (CONTRACT amendment 2026-08-28b), which runs strictly BEFORE any measured
# leg and is never counted as one: it may invoke the book-side
# `cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs`
# (network-backed, checksum-gated) to (re)populate `train_pairs.jsonl`, then
# ALWAYS byte-verifies it against the committed `train_ids_sha256.json`
# before letting any leg proceed.
#
# HELD-OUT FIXTURE LAYOUT (cookbook/fixtures/finetune_heldout/, CONTRACT H3):
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
# PRE-RUN PROVISIONING (CONTRACT amendment 2026-08-28b): before any measured
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
# Batch size: 32 (`cookbook/fixtures/finetune_heldout/README.md`'s own
# "this agent's pick, pending lead confirmation" -- the chapter-config value
# every real `db.fine_tune(...)` call over this exact pair set already
# uses; 128 held-out pairs is a multiple of both 32 and the engine's own
# unset-default 8, so this pick does not change the fixture's own held-out
# count, only which "N held-out = k batches" framing is reported).
#
# Objective: MNRL by default (CONTRACT amendment 2026-08-28's own default-
# on-ties-or-ambiguity rule) -- override with FINETUNE_RUN_AB_OBJECTIVE=
# triplet. H5's own dynamic-range probe (step 0) is what actually PICKS the
# objective for the real campaign; this script just runs whichever one it
# is told, over both arms, at every pre-registered seed.
#
# Seeds: the pre-registered 12-seed gate set (CONTRACT Frame: "N=12 seeds x
# 2 arms"), 1..12 by default -- override with FINETUNE_RUN_AB_SEEDS (a
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
#                              used -- this script previously exposed no
#                              --lr passthrough at all, unit-63 audit
#                              advisory (b)).
#   FINETUNE_RUN_AB_LR0_SEEDS  comma-separated seed list for the lr=0 RED
#                              control (CONTRACT Frame: "RED control: lr=0
#                              arm x2 seeds fails learning-happened"; default
#                              empty = skipped). Each seed here runs BOTH
#                              arms at --lr 0, tagged with ab_merge.py's own
#                              `FINETUNE_RUN_LR0_REPEAT` label -- NEVER
#                              folded into FINETUNE_RUN_AB_SEEDS/the A/B set
#                              (ab_merge.py's `finetune-run` mode reads these
#                              via a separate positional arg and checks each
#                              one FAILS learning-happened; a control seed
#                              value need not, and by default does not,
#                              collide with the gate/off-sample seed
#                              namespaces).
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
#   FINETUNE_RUN_AB_OUT_DIR    where the raw legs + merged report land
#                              (default "<repo>/.finetune-run-ab-report/
#                              <UTC timestamp>").
#   FINETUNE_RUN_AB_DRY_RUN=1  print every command this script would run
#                              instead of executing it, and write a
#                              `{"tool":"dry-run",...}` stub per leg so the
#                              merge stage still runs end-to-end against
#                              real (if fabricated-empty) files. Never
#                              mutates the checkout, never touches the
#                              network, never claims a real number -- same
#                              contract `finetune_ab.sh`/`encode_ab.sh`'s
#                              own `*_DRY_RUN` knobs already carry.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

# require-env panic-not-skip (this repo's own `JAMMI_REQUIRE_CUDA` idiom --
# see crates/jammi-ai/src/fine_tune/optimizer.rs, crates/jammi-kernels/
# tests/cuda_parity.rs, ci/scripts/perf/clip_artifact_producer.sh's own
# `export ... JAMMI_REQUIRE_CUDA=1`): a real (non-dry-run) run of this
# producer is ALWAYS a GPU run -- there is no CPU-hermetic mode this
# producer itself opts into by default (FINETUNE_RUN_AB_CPU=1 is an
# explicit opt-OUT an operator can still set for local wiring smoke-tests).
# Exporting this unconditionally makes a missing CUDA device on a real
# invocation a hard failure inside the jammi-bench binary/its own test
# surfaces, never a silent skip that would let a "green" run report numbers
# that were actually computed on CPU.
export JAMMI_REQUIRE_CUDA=1

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
# --lr passthrough (unit-63 audit advisory (b): the CLI has always had this
# flag, main.rs:141 -- this script simply never forwarded it). Unset means
# "omit --lr entirely", i.e. the CLI's own default (2e-4) -- never fabricate
# a value here that main.rs's own `#[arg(long, default_value_t = 2e-4)]`
# already owns.
FINETUNE_RUN_AB_LR="${FINETUNE_RUN_AB_LR:-}"
# lr=0 RED control seeds (CONTRACT Frame; advisory (b)) -- comma-separated,
# default empty (skipped). NEVER added to FINETUNE_RUN_AB_SEEDS/the main
# sweep loop below; run through their own dedicated loop, tagged with
# ab_merge.py's own FINETUNE_RUN_LR0_REPEAT label.
FINETUNE_RUN_AB_LR0_SEEDS="${FINETUNE_RUN_AB_LR0_SEEDS:-}"
FINETUNE_RUN_AB_CUDA="${FINETUNE_RUN_AB_CUDA:-0}"
FINETUNE_RUN_AB_CPU="${FINETUNE_RUN_AB_CPU:-0}"

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
      echo "::error::committed fixture file not found: $f (cookbook/fixtures/finetune_heldout/, CONTRACT H3) — refusing before any leg runs." >&2
      exit 1
    fi
  done

  # --- PRE-RUN provisioning (CONTRACT amendment 2026-08-28b) -- see module
  # doc "PRE-RUN PROVISIONING" above. Outside every measured leg: this runs
  # once, before the sweep loop, never inside run_leg.
  if [ ! -f "$TRAIN_JSONL" ]; then
    echo "::notice::$TRAIN_JSONL not found -- provisioning via 'derive_heldout_fixture.py --emit-train-pairs' (network-backed, checksum-gated fetch of train text; outside measured legs)."
    python3 "$REPO_ROOT/cookbook/book/scripts/derive_heldout_fixture.py" --emit-train-pairs \
      || { echo "::error::train-pairs provisioning failed (cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs) — refusing before any leg runs." >&2; exit 1; }
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

if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda failed" >&2; exit 1; }
fi

# --- provenance cross-check (unification contract C5.1), same shape as
# fa2_ab.sh/finetune_ab.sh/encode_ab.sh/stacked_sweep.sh/
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

# --- one measurement leg (mirrors finetune_ab.sh/encode_ab.sh's own
# run_leg: NEVER aborts the sweep -- a leg failure is recorded as this
# leg's own outcome, so one seed's OOM/refusal does not discard every other
# seed's row).
#
# `arm` selects BOTH the CLI's own `--arm` flag (recorded on the report,
# CONTRACT H4/report.rs's own PROVENANCE_FIELDS) AND, for the `alloff` arm
# only, the `JAMMI_KERNELS_DISABLE` env var this binary's own CLI doc names
# as the CALLER's responsibility ("the caller is responsible for setting
# JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused itself
# before invoking this binary for the alloff arm" -- main.rs's own
# `FinetuneRunArgs::arm` doc).
#
# `lr_override` (5th, optional): when non-empty, forwarded as `--lr`
# (main.rs:141's own CLI flag) -- the lr=0 RED control loop below passes
# `"0"` explicitly; the main A/B loop passes `$FINETUNE_RUN_AB_LR`, which is
# empty by default (omit --lr entirely, i.e. the CLI's own 2e-4 default).
run_leg() {
  local seed="$1" arm="$2" repeat="$3" work_dir="$4" lr_override="${5:-}"
  local out_file="$RAW_DIR/seed${seed}__${arm}__${repeat}.json"
  local err_file="$RAW_DIR/seed${seed}__${arm}__${repeat}.stderr"
  local exit_file="$RAW_DIR/seed${seed}__${arm}__${repeat}.exit"

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
    # CONTRACT Frame: early stopping DISABLED both arms -- the "never
    # stops before the pre-registered epoch budget" idiom, so a seed's
    # trajectory is never truncated by an early-stopping decision the sign
    # test would then have to account for.
    --early-stopping-patience 10000
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
    printf '{"tool":"dry-run","ab_dry_run":true,"seed":%s,"arm":"%s","repeat":"%s"}\n' \
      "$seed" "$arm" "$repeat" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  if [ "$arm" = "alloff" ]; then
    JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  else
    "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  fi
  echo "$rc" > "$exit_file"
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

# --- lr=0 RED control legs (CONTRACT Frame; unit-63 audit advisory (b)):
# both arms, at --lr 0, tagged with ab_merge.py's own FINETUNE_RUN_LR0_REPEAT
# label ("lr0") -- a DISTINCT repeat token from r1/r2, so these legs are
# never picked up by the main sweep's own r1/r2 loader and never enter the
# A/B set. Skipped entirely (no legs, no wiring cost) when
# FINETUNE_RUN_AB_LR0_SEEDS is unset -- an operator opts in explicitly per
# H5 campaign step 3.
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

# --- merge: sign test + conjunctive leg-premise refusal + determinism-
# floor reporting + the lr=0 control's own learning-happened check, computed
# INTO the merged artifact by ab_merge.py's own `finetune-run` mode (unit 63
# H4b) -- reusing the same generic leg-premise-refusal core `encode_ab.sh`'s
# merge step already builds on, never a second, hand-rolled comparator.
python3 "$DIR/ab_merge.py" finetune-run "$RAW_DIR" "$OUT_DIR" "$FINETUNE_RUN_AB_SEEDS" "$FINETUNE_RUN_AB_LR0_SEEDS"
PY_RC=$?

echo
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$PY_RC"
