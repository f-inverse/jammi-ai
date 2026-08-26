#!/usr/bin/env bash
# The tracked producer for `crates/jammi-kernels/artifacts/cuda-runs/
# *-grad-ablation-*.json` — a manual, GPU-required pod-lane script (no CI
# lane has a GPU, docs/maintainer/cuda-kernel-guide.md §5), run by hand on a
# rented CUDA box against a real checkpoint. Builds `jammi-bench --release
# --features cuda` once, then runs `grad-oracle --ablate-each-op` for every
# (batch, seq, seed) triple named on the command line, writing one
# AblationReport JSON per leg under OUT_DIR.
#
# Usage:
#   MODEL_DIR=/root/checkpoints/ModernBERT-large \
#   run_grad_oracle_ablation.sh OUT_DIR "b4-s128-seed42:4:128:42" "b8-s128:8:128:42"
#
# Each leg argument is "<label>:<batch>:<seq>:<seed>"; writes
# OUT_DIR/<label>.json. See `crates/jammi-bench/reference/README.md`'s
# "--ablate-each-op — the per-op gradient-ablation gate (pod lane)" section
# for the full attach-to-PR checklist this script's output feeds.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: MODEL_DIR=<dir> $0 OUT_DIR LABEL:BATCH:SEQ:SEED [LABEL:BATCH:SEQ:SEED...]" >&2
  exit 2
fi

: "${MODEL_DIR:?MODEL_DIR env var (checkpoint dir: config.json + model.safetensors) is required}"
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

for leg in "$@"; do
  label="${leg%%:*}"
  rest="${leg#*:}"
  batch="${rest%%:*}"
  rest="${rest#*:}"
  seq="${rest%%:*}"
  seed="${rest#*:}"
  echo "== leg ${label}: batch=${batch} seq=${seq} seed=${seed} =="
  "$BIN" grad-oracle \
    --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank 16 --lora-alpha 32 --target-modules Wqkv,Wo,Wi \
    --backbone-dtype bf16 --cuda 0 \
    --ablate-each-op \
    --out "$OUT_DIR/${label}.json"
done
