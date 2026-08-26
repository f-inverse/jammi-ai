#!/usr/bin/env bash
# esc-045 control (a): the NULL-BAND measurement `run_esc045_torch_column.sh`
# itself does not take. That script's own `full_tensor_cosine(arm_bf16,
# jammi_f32_truth)` statistic ("paired-table.txt"'s `torch_bf16`/`eager`
# columns) is read ONCE per (config, op_point, arm) and compared ACROSS arms
# (block/flash/eager vs torch_bf16) to get esc-045's 6/6 sign. Before that
# cross-arm sign means anything, the metric's own SAME-ARM replicate spread
# must be measured: run the IDENTICAL bf16 leg (same seed, same shared LoRA
# weights file, same batch, same config -- every `RUN_IDENTITY_FIELDS` entry
# `compare_grad_oracle.py`/`grad_oracle.rs`'s own determinant table names)
# TWICE, as two independent process invocations, and see whether the two
# replicates' `full_tensor_cosine` vs the SAME jammi_f32_truth dump differ at
# all -- and if so, whether that same-arm paired difference also lands on a
# consistent sign across the 6 b4-s128 operating points row 11/s4:42 cover.
#
# REPLICATE DESIGN (see ci/scripts/perf/analyze_esc045_null_band.py's module
# doc for the full write-up this script's own comment summarizes): the ONLY
# honest nuisance to vary between replicate A and replicate B of one arm at
# one operating point is RUN-TO-RUN NONDETERMINISM of that arm's OWN bf16
# kernel execution (GPU reduction-order / algorithm-selection variability) --
# not a different LoRA-init draw, not a different data seed. Both of those
# are pinned IDENTICAL between the real block/torch arms being compared in
# esc-045's own table (`grad_oracle.rs`'s "Weight interchange format" section:
# both arms load the SAME `--lora-weights-in` safetensors file and the SAME
# `synthetic_ids(.., seed + i, ..)` batch), so a replicate that varied init or
# data would inject a nuisance the real between-arm comparison never carries
# -- an apples-to-oranges null band, not this control's null band. Only the
# BF16 leg is replicated; the F32-truth dump is generated ONCE per (label,
# op_point) and reused for both replicate A and B's cosine -- f32 accumulation
# order is not the noise source under test (torch's OWN bf16-vs-f32 spread,
# the thing esc-045's whole table is about, lives entirely in the bf16 leg).
#
# Usage (mirrors run_esc045_torch_column.sh's own usage):
#   MODEL_DIR=/root/checkpoints/ModernBERT-large \
#   TORCH_PY=/root/jammi-ai/.venv-torch-ref/bin/python3 \
#   run_esc045_null_band.sh OUT_DIR "b4-s128-seed42:4:128:42" [...]
#
# Writes raw JSON dumps under OUT_DIR/raw/<label>__<op>__<arm>.json, with the
# two bf16 arms suffixed _repA/_repB. analyze_esc045_null_band.py OUT_DIR
# reads exactly this layout.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: MODEL_DIR=<dir> TORCH_PY=<python3> $0 OUT_DIR LABEL:BATCH:SEQ:SEED [...]" >&2
  exit 2
fi

: "${MODEL_DIR:?MODEL_DIR env var (checkpoint dir: config.json + model.safetensors) is required}"
: "${TORCH_PY:?TORCH_PY env var (the torch-ref venv python3 binary) is required}"
OUT_DIR="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

export PATH="$HOME/.cargo/bin:$PATH"
if [ "${SKIP_CARGO_BUILD:-0}" != "1" ]; then
  cargo build -p jammi-bench --release --features cuda --manifest-path "$REPO_ROOT/Cargo.toml"
fi

BIN="$REPO_ROOT/target/release/jammi-bench"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  BIN="$CARGO_TARGET_DIR/release/jammi-bench"
fi
TORCH_ORACLE="$REPO_ROOT/crates/jammi-bench/reference/torch_grad_oracle.py"

LORA_RANK=16
LORA_ALPHA=32
TARGET_MODULES="Wqkv,Wo,Wi"

for leg in "$@"; do
  label="${leg%%:*}"
  rest="${leg#*:}"
  batch="${rest%%:*}"
  rest="${rest#*:}"
  seq="${rest%%:*}"
  seed="${rest#*:}"

  for op in gaussian poststep; do
    echo "== leg ${label} ${op}: batch=${batch} seq=${seq} seed=${seed} =="
    shared="$RAW_DIR/${label}__${op}__nullband_shared.safetensors"

    if [ "$op" = "gaussian" ]; then
      lora_init_flag="gaussian"
    else
      lora_init_flag="peft-step1"
    fi

    # One jammi F32 arm generates the shared weights file AND doubles as
    # the truth dump both replicates compare against -- same convention
    # run_esc045_torch_column.sh's own gaussian/poststep legs use.
    "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
      --batch "$batch" --seq "$seq" --seed "$seed" \
      --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
      --backbone-dtype f32 --cuda 0 --lora-init "$lora_init_flag" \
      --lora-weights-out "$shared" \
      --out "$RAW_DIR/${label}__${op}__jammi_f32_truth.json"

    # jammi eager bf16, TWO independent process invocations -- identical
    # CLI args, identical shared weights file, identical batch (same seed).
    for rep in repA repB; do
      JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=all "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
        --batch "$batch" --seq "$seq" --seed "$seed" \
        --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
        --backbone-dtype bf16 --cuda 0 \
        --lora-weights-in "$shared" \
        --out "$RAW_DIR/${label}__${op}__jammi_bf16_eager_${rep}.json"
    done

    # torch bf16, TWO independent process invocations -- same discipline.
    for rep in repA repB; do
      "$TORCH_PY" "$TORCH_ORACLE" --model-dir "$MODEL_DIR" \
        --batch "$batch" --seq "$seq" --seed "$seed" \
        --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
        --dtype bf16 --attn eager --cuda 0 \
        --lora-weights-in "$shared" \
        --out "$RAW_DIR/${label}__${op}__torch_bf16_${rep}.json"
    done
  done
done

echo "wrote raw dumps under $RAW_DIR"
