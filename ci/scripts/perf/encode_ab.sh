#!/usr/bin/env bash
# The encode-surface producer: jammi's `encode-step` against the PyTorch
# reference (`crates/jammi-bench/reference/torch_encode.py`), on ONE box, over
# ONE corpus, in one run — rows in a table → persisted embeddings, both sides.
#
# FOUR ARMS, each run twice, as a palindrome (`encode_ab.py`'s module doc has
# why the order cancels a drifting box out of every pair of arms at once):
#
#   jammi-p1 jammi-pN torch-corpus torch-sorted | torch-sorted-2 torch-corpus-2 jammi-pN-2 jammi-p1-2
#
#   jammi-p1 / jammi-pN   `jammi-bench encode-step` at `[inference] partitions`
#                         1 and N — the same serve, serial and fanned out, so
#                         the sweep's fitted fixed and per-row cost are known
#                         at both.
#   torch-corpus          the reference forwarding the rows in the engine's own
#                         order with eager attention — the semantic twin.
#   torch-sorted          the reference forwarding them longest-first with
#                         SDPA, as `sentence-transformers`' `encode()` does —
#                         the bar a user would hold the engine to.
#
# The first jammi-p1 leg leaves the corpus it served and the vectors it
# persisted in its exchange directory; every torch leg reads THAT corpus and
# reports the per-row cosine of its own vectors against THOSE vectors.
#
# The merge (`encode_ab.py`) refuses legs that did not measure the same thing
# (`INVALID`) or that embedded different things (`INVALID_MEASUREMENT`), and
# otherwise RECORDS every ratio by the estimator least favourable to jammi.
# It never gates on a ratio.
#
# Not a CI job: it builds and runs a real release binary and needs a torch
# venv. Invoked on a pod or a dev box:
#   ci/scripts/perf/encode_ab.sh
#
# Env vars:
#   ENCODE_AB_OUT_DIR       where the merged report + raw legs land (default
#                           "<repo>/.encode-ab-report/<UTC timestamp>").
#   ENCODE_AB_MODEL_DIR     a local checkpoint directory both stacks load
#                           (config.json, model.safetensors, tokenizer.json,
#                           optionally 1_Pooling/config.json). Unset: the jammi
#                           legs serve their compiled-in fixture, leave it in
#                           the exchange directory, and the torch legs load it
#                           from there — the same bytes either way.
#   ENCODE_AB_ROWS          the sweep (default "16,1024,16384").
#   ENCODE_AB_PARTITIONS    N for the jammi-pN arm (default 4).
#   ENCODE_AB_BATCH_SIZE    rows per forward, both stacks (default 32).
#   ENCODE_AB_DTYPE         f32 | bf16 | f16, both stacks (default f32): the
#                           jammi legs' `--compute-precision`, the torch legs'
#                           `--dtype`. The merge refuses a leg whose model
#                           RESOLVED another.
#   ENCODE_AB_TORCH_ANN_INDEX
#                           1 (default): the torch legs also build and save the
#                           ANN graph the engine's sink builds, so both spans
#                           close on the same work (needs `usearch` in the torch
#                           venv). 0: they stop at the Parquet file, and their
#                           reports say so.
#   ENCODE_AB_WARMUP / ENCODE_AB_ITERS
#                           warm and measured serves per point (defaults 2, 10).
#   ENCODE_AB_CUDA_ORDINAL  optional CUDA device ordinal (unset = CPU; when set,
#                           every leg runs on it and the build adds
#                           `--features cuda`).
#   ENCODE_AB_PASS_RATIO    the rows/s parity bar verdicts are recorded against
#                           (default 0.9, `finetune_ab.sh`'s).
#   ENCODE_AB_COSINE_FLOOR  the per-row cosine a torch leg must reach against
#                           the jammi vectors (default 0.99).
#   ENCODE_AB_DRY_RUN=1     print every command instead of executing it, and
#                           write a `{"tool":"dry-run",...}` stub per leg so the
#                           merge stage still runs end-to-end. Never builds,
#                           never touches the network, never claims a number.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

ENCODE_AB_DRY_RUN="${ENCODE_AB_DRY_RUN:-0}"
ENCODE_AB_MODEL_DIR="${ENCODE_AB_MODEL_DIR:-}"
ENCODE_AB_ROWS="${ENCODE_AB_ROWS:-16,1024,16384}"
ENCODE_AB_PARTITIONS="${ENCODE_AB_PARTITIONS:-4}"
ENCODE_AB_BATCH_SIZE="${ENCODE_AB_BATCH_SIZE:-32}"
ENCODE_AB_DTYPE="${ENCODE_AB_DTYPE:-f32}"
ENCODE_AB_TORCH_ANN_INDEX="${ENCODE_AB_TORCH_ANN_INDEX:-1}"
ENCODE_AB_WARMUP="${ENCODE_AB_WARMUP:-2}"
ENCODE_AB_ITERS="${ENCODE_AB_ITERS:-10}"
ENCODE_AB_CUDA_ORDINAL="${ENCODE_AB_CUDA_ORDINAL:-}"
ENCODE_AB_PASS_RATIO="${ENCODE_AB_PASS_RATIO:-0.9}"
ENCODE_AB_COSINE_FLOOR="${ENCODE_AB_COSINE_FLOOR:-0.99}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ENCODE_AB_OUT_DIR:-$REPO_ROOT/.encode-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_encode.py"
# The torch venv and its default are resolved in one place, torch_venv.py.
TORCH_PY="$(python3 "$DIR/torch_venv.py" --path)/bin/python3"
# The exchange directory the torch legs read: the FIRST jammi-p1 leg's.
EXCHANGE_DIR="$RAW_DIR/jammi-p1.exchange"

# What the merge stage needs to know about this run, passed as files (the
# convention `gpu_inference_ab.sh`'s `mode` marker uses).
printf '%s\n' "$ENCODE_AB_PARTITIONS" > "$RAW_DIR/partitions"
printf '%s\n' "$ENCODE_AB_TORCH_ANN_INDEX" > "$RAW_DIR/torch_ann_index"
printf '%s\n' "$ENCODE_AB_PASS_RATIO" > "$RAW_DIR/pass_ratio"
printf '%s\n' "$ENCODE_AB_COSINE_FLOOR" > "$RAW_DIR/cosine_floor"

# --- state-changing command wrapper (same shape as finetune_ab.sh's
# run_cmd): always echoes what it would run; under ENCODE_AB_DRY_RUN never
# executes it.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$ENCODE_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

if [ "$ENCODE_AB_DRY_RUN" != "1" ]; then
  if [ -n "$ENCODE_AB_CUDA_ORDINAL" ]; then
    # A CUDA ordinal was requested: pull in the engine's CUDA backend —
    # without it `--cuda` has no device to select.
    run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench --features cuda failed" >&2; exit 1; }
  else
    run_cmd cargo build --release -p jammi-bench --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench failed" >&2; exit 1; }
  fi
fi

# --- provenance cross-check, same shape as
# finetune_ab.sh/stacked_sweep.sh/clip_artifact_producer.sh:
# refuse BEFORE any leg runs if the binary's own baked identity does not
# match the sha this checkout is actually at.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$ENCODE_AB_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BIN" provenance 2>&1)" || { echo "::error::'$BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# --- one measurement leg (mirrors finetune_ab.sh's run_leg: NEVER aborts
# the sweep -- a leg failure is recorded as this leg's own outcome).
# run_leg LEG CMD...
run_leg() {
  local leg="$1"; shift
  local out_file="$RAW_DIR/${leg}.json"
  local err_file="$RAW_DIR/${leg}.stderr"
  local exit_file="$RAW_DIR/${leg}.exit"

  printf -- '--- %s: ' "$leg"
  printf '%q ' "$@"
  printf '\n'

  if [ "$ENCODE_AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"leg":"%s"}\n' "$leg" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "$@" > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${leg} FAILED (exit ${rc}) -- recorded as a leg outcome; sweep continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

# run_jammi_leg LEG PARTITIONS
run_jammi_leg() {
  local leg="$1" partitions="$2"
  local -a cmd=("$BIN" encode-step
    --rows "$ENCODE_AB_ROWS" --partitions "$partitions" --batch-size "$ENCODE_AB_BATCH_SIZE"
    --compute-precision "$ENCODE_AB_DTYPE"
    --warmup "$ENCODE_AB_WARMUP" --iters "$ENCODE_AB_ITERS"
    --exchange-dir "$RAW_DIR/${leg}.exchange")
  # `--model-dir`/`--cuda` are OMITTED entirely when unset, so the hermetic
  # default (the compiled-in fixture on `Device::Cpu`) is the flagless run.
  [ -n "$ENCODE_AB_MODEL_DIR" ] && cmd+=(--model-dir "$ENCODE_AB_MODEL_DIR")
  [ -n "$ENCODE_AB_CUDA_ORDINAL" ] && cmd+=(--cuda "$ENCODE_AB_CUDA_ORDINAL")
  run_leg "$leg" "${cmd[@]}"
}

# run_torch_leg LEG ORDER ATTN
run_torch_leg() {
  local leg="$1" order="$2" attn="$3"
  local -a cmd=("$TORCH_PY" "$REF_SCRIPT"
    --model-dir "${ENCODE_AB_MODEL_DIR:-$EXCHANGE_DIR/model}" --exchange-dir "$EXCHANGE_DIR" --out-dir "$RAW_DIR/${leg}.out"
    --rows "$ENCODE_AB_ROWS" --batch-size "$ENCODE_AB_BATCH_SIZE" --dtype "$ENCODE_AB_DTYPE"
    --warmup "$ENCODE_AB_WARMUP" --iters "$ENCODE_AB_ITERS" --order "$order" --attn "$attn")
  [ "$ENCODE_AB_TORCH_ANN_INDEX" = "1" ] && cmd+=(--ann-index)
  [ -n "$ENCODE_AB_CUDA_ORDINAL" ] && cmd+=(--cuda "$ENCODE_AB_CUDA_ORDINAL")
  mkdir -p "$RAW_DIR/${leg}.out"
  run_leg "$leg" "${cmd[@]}"
}

# The palindrome — `encode_ab.py`'s LEG_ORDER, in order.
run_jammi_leg jammi-p1 1
run_jammi_leg jammi-pN "$ENCODE_AB_PARTITIONS"
run_torch_leg torch-corpus corpus eager
run_torch_leg torch-sorted length-sorted sdpa
run_torch_leg torch-sorted-2 length-sorted sdpa
run_torch_leg torch-corpus-2 corpus eager
run_jammi_leg jammi-pN-2 "$ENCODE_AB_PARTITIONS"
run_jammi_leg jammi-p1-2 1

python3 "$DIR/encode_ab.py" "$RAW_DIR" "$OUT_DIR" "$SHA"
PY_RC=$?

echo
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$PY_RC"
