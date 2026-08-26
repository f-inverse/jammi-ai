#!/usr/bin/env bash
# The tracked producer for `crates/jammi-kernels/artifacts/cuda-runs/
# *-grad-ablation-*.json` — a manual, GPU-required pod-lane script (no CI
# lane has a GPU, docs/maintainer/cuda-kernel-guide.md §5), run by hand on a
# rented CUDA box against a real checkpoint. Builds `jammi-bench --release
# --features cuda` once, then runs `grad-oracle --ablate-each-op` for every
# (batch, seq, lora_init) triple named on the command line — each such call
# ALREADY runs the internal 3-seed cascade (`--seeds`, default
# `42,43,44`) and aggregates it — writing one AblationReport JSON per leg
# under OUT_DIR.
#
# Usage:
#   MODEL_DIR=/root/checkpoints/ModernBERT-large \
#   run_grad_oracle_ablation.sh OUT_DIR \
#       "b4-s128-peft-step1:4:128:peft-step1" \
#       "b8-s128-peft-step1:8:128:peft-step1" \
#       "b4-s128-gaussian:4:128:gaussian" \
#       "b8-s128-gaussian:8:128:gaussian"
#
# Each leg argument is "<label>:<batch>:<seq>:<lora_init>"; writes
# OUT_DIR/<label>.json. `SEEDS` (env var, default `42,43,44`) is forwarded
# to every leg's own `--seeds`. See `crates/jammi-bench/reference/README.md`'s
# "--ablate-each-op — the per-op gradient-ablation gate (pod lane)" section
# for the full attach-to-PR checklist this script's output feeds.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: MODEL_DIR=<dir> $0 OUT_DIR LABEL:BATCH:SEQ:LORA_INIT [LABEL:BATCH:SEQ:LORA_INIT...]" >&2
  exit 2
fi

: "${MODEL_DIR:?MODEL_DIR env var (checkpoint dir: config.json + model.safetensors) is required}"
SEEDS="${SEEDS:-42,43,44}"
OUT_DIR="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.cargo/bin:$PATH"
cargo build -p jammi-bench --release --features cuda --manifest-path "$REPO_ROOT/Cargo.toml"

BIN="$REPO_ROOT/target/release/jammi-bench"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  BIN="$CARGO_TARGET_DIR/release/jammi-bench"
fi

COSINE_FLOOR="${COSINE_FLOOR:-0.7}"

# Parsed (label, batch, seq, lora_init) tuples, one per leg -- tracked in
# parallel arrays (bash has no clean nested-map type) so the comparator/gate
# pass below can group legs by (lora_init, shape) WITHOUT the caller having
# to separately name which two legs are "the pair" — this script's own job
# (per this file's doc, and round-7 audit item 5: "one recorded invocation
# chains the tool, the comparator, and the gate").
labels=()
batches=()
seqs=()
lora_inits=()

for leg in "$@"; do
  label="${leg%%:*}"
  rest="${leg#*:}"
  batch="${rest%%:*}"
  rest="${rest#*:}"
  seq="${rest%%:*}"
  lora_init="${rest#*:}"
  echo "== leg ${label}: batch=${batch} seq=${seq} lora_init=${lora_init} seeds=${SEEDS} =="
  "$BIN" grad-oracle \
    --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" \
    --lora-rank 16 --lora-alpha 32 --target-modules Wqkv,Wo,Wi \
    --backbone-dtype bf16 --cuda 0 \
    --lora-init "$lora_init" \
    --seeds "$SEEDS" \
    --ablate-each-op \
    --out "$OUT_DIR/${label}.json"
  labels+=("$label")
  batches+=("$batch")
  seqs+=("$seq")
  lora_inits+=("$lora_init")
done

echo "== compare_grad_oracle.py --ablation (--cosine-floor ${COSINE_FLOOR}) per leg =="
for label in "${labels[@]}"; do
  python3 "$REPO_ROOT/ci/scripts/perf/compare_grad_oracle.py" --ablation \
    "$OUT_DIR/${label}.json" --cosine-floor "$COSINE_FLOOR" \
    --out "$OUT_DIR/${label}-ablation-summary.json" \
    | tee "$OUT_DIR/${label}-ablation-summary.log"
done

echo "== check_fused_op_gradient_parity.py per (lora_init, both shapes present) pair =="
distinct_inits="$(printf '%s\n' "${lora_inits[@]}" | sort -u)"
while IFS= read -r li; do
  [ -z "$li" ] && continue
  b4_label=""
  b8_label=""
  for i in "${!labels[@]}"; do
    if [ "${lora_inits[$i]}" = "$li" ] && [ "${batches[$i]}" = "4" ] && [ "${seqs[$i]}" = "128" ]; then
      b4_label="${labels[$i]}"
    fi
    if [ "${lora_inits[$i]}" = "$li" ] && [ "${batches[$i]}" = "8" ] && [ "${seqs[$i]}" = "128" ]; then
      b8_label="${labels[$i]}"
    fi
  done
  if [ -n "$b4_label" ] && [ -n "$b8_label" ]; then
    echo "-- gate for lora_init=${li}: ${b4_label} + ${b8_label} --"
    set +e
    python3 "$REPO_ROOT/ci/scripts/perf/check_fused_op_gradient_parity.py" \
      "$OUT_DIR/${b4_label}.json" "$OUT_DIR/${b8_label}.json" \
      | tee "$OUT_DIR/gate-${li}.log"
    echo "${PIPESTATUS[0]}" > "$OUT_DIR/gate-${li}-exit-code.txt"
    set -e
  else
    echo "-- lora_init=${li}: both b4-s128 and b8-s128 not present, gate skipped for this init --"
  fi
done <<< "$distinct_inits"
