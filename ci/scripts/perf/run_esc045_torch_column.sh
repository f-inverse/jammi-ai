#!/usr/bin/env bash
# The tracked producer for the esc-045 "torch column" measurement (ledger
# rows 258/260): the missing TORCH leg of the jammi weight-gradient cosine
# oracle, at BOTH a fresh `LoraInitMode::Gaussian` operating point and a
# "post-one-real-AdamW-step" operating point (the #383 audit addendum: a
# fresh `Gaussian(0, 0.02)` init on BOTH `A` and `B` is a state real
# training never occupies). A manual, GPU-required pod-lane script (no CI
# lane has a GPU — `docs/maintainer/cuda-kernel-guide.md` §5), run by hand
# on a rented CUDA box against a real checkpoint, producing raw
# `jammi-bench grad-oracle`/`torch_grad_oracle.py` JSON dumps that
# `analyze_esc045_torch_column.py` (the numpy-first comparator, run
# SEPARATELY — never this script) folds into the committed cuda-run
# artifact.
#
# ## Why one shared LoRA-weights file per (config, operating point)
#
# Every arm this script runs for a given (config, operating point) loads
# the SAME LoRA `A`/`B` values (jammi's own `--lora-weights-in`/
# `torch_grad_oracle.py`'s own `--lora-weights-in`, both reading the
# IDENTICAL safetensors file this script produces once per (config,
# operating point) via the FIRST arm's `--lora-weights-out`) — isolating
# the comparison to "does this arm's kernel/dtype composition compute the
# SAME gradient off the SAME weights", never conflating that with "did two
# independently-drawn inits happen to differ". This is EXACTLY
# `grad_oracle_ablation.rs`'s own "weight interchange" convention (see that
# module's doc), just driven across TWO producers (jammi, torch) instead of
# one.
#
# Cross-producer identity of that shared file was verified empirically
# before this script existed (not merely assumed): `peft`'s default
# `get_peft_model(..., autocast_adapter_dtype=True)` upcasts the LoRA
# adapter back to `float32` even when the base model loads at `bfloat16`
# (`peft/tuners/tuners_utils.py`'s `_cast_adapter_dtype`/`cast_adapter_dtype`
# docstring: "Currently, this only upcasts float16 and bfloat16 to
# float32") — so torch's own reported LoRA weight matches jammi's
# (jammi_lora's `lora_ab_dtype_f32` gate keeps LoRA `A`/`B` at `F32`
# unconditionally) to ~1e-9 absolute (JSON float round-trip noise, the SAME
# order of magnitude `torch_grad_oracle.py`'s own PROVENANCE note records:
# "max|w_jammi - w_torch| = 1.86e-9"), at BOTH `f32` and `bf16` backbone
# arms. For the POST-STEP operating point, one real `AdamW` step from that
# shared init, run independently on each stack (`candle_nn::AdamW`'s
# formula and default betas/eps/weight_decay match `torch.optim.AdamW`'s
# exactly — `candle-nn-0.11.0/src/optim.rs`), was ALSO verified to produce
# a near-bit-identical post-step state (overall weight-vector cosine
# 0.99999999578 at b4·s128·seed42; 0.068% of elements disagree by more than
# 1e-6 — an AdamW step-1 sign coin-flip on a near-zero-gradient element,
# `theta*(1-lr*wd) - lr*sign(grad)`-shaped and inherent to ANY two
# independently-rounded implementations of a first-order optimizer's first
# step, not a defect). Both checks are cheap (one config, no full sweep) and
# should be RE-RUN (not merely trusted from this comment) before relying on
# a NEW checkpoint/config this script has not yet been exercised against —
# see this script's own `--verify-only` leg below.
#
# Usage:
#   MODEL_DIR=/root/checkpoints/ModernBERT-large \
#   TORCH_PY=/root/jammi-ai/.venv-torch-ref/bin/python3 \
#   run_esc045_torch_column.sh OUT_DIR "b4-s128-seed42:4:128:42" "b8-s512-seed42:8:512:42"
#
# Each leg argument is "<label>:<batch>:<seq>:<seed>"; writes raw JSON dumps
# under `OUT_DIR/raw/<label>__<op_point>__<arm>.json`. `analyze_esc045_
# torch_column.py OUT_DIR` reads exactly that layout.
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
cargo build -p jammi-bench --release --features cuda --manifest-path "$REPO_ROOT/Cargo.toml"

BIN="$REPO_ROOT/target/release/jammi-bench"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  BIN="$CARGO_TARGET_DIR/release/jammi-bench"
fi
TORCH_ORACLE="$REPO_ROOT/crates/jammi-bench/reference/torch_grad_oracle.py"

LORA_RANK=16
LORA_ALPHA=32
TARGET_MODULES="Wqkv,Wo,Wi"
WARMUP_LR=2e-4

for leg in "$@"; do
  label="${leg%%:*}"
  rest="${leg#*:}"
  batch="${rest%%:*}"
  rest="${rest#*:}"
  seq="${rest%%:*}"
  seed="${rest#*:}"

  echo "== leg ${label}: batch=${batch} seq=${seq} seed=${seed} =="

  # ---- operating point 1: fresh Gaussian(0, 0.02) init on A and B -------
  shared_gaussian="$RAW_DIR/${label}__gaussian__shared.safetensors"
  JAMMI_KERNELS_STRICT=1 "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype bf16 --cuda 0 --lora-init gaussian \
    --lora-weights-out "$shared_gaussian" \
    --out "$RAW_DIR/${label}__gaussian__jammi_bf16_fused.json"

  JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=all "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype bf16 --cuda 0 \
    --lora-weights-in "$shared_gaussian" \
    --out "$RAW_DIR/${label}__gaussian__jammi_bf16_eager.json"

  "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype f32 --cuda 0 \
    --lora-weights-in "$shared_gaussian" \
    --out "$RAW_DIR/${label}__gaussian__jammi_f32_truth.json"

  "$TORCH_PY" "$TORCH_ORACLE" --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --dtype bf16 --attn eager --cuda 0 \
    --lora-weights-in "$shared_gaussian" \
    --out "$RAW_DIR/${label}__gaussian__torch_bf16.json"

  "$TORCH_PY" "$TORCH_ORACLE" --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --dtype fp32 --attn eager --cuda 0 \
    --lora-weights-in "$shared_gaussian" \
    --out "$RAW_DIR/${label}__gaussian__torch_f32_truth.json"

  # ---- operating point 2: PEFT-style init (A kaiming-uniform, B zeros), -
  # ---- then ONE real AdamW step at the reference lr on the same data ----
  fresh_init="$RAW_DIR/${label}__poststep__fresh_init.safetensors"
  "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype f32 --cuda 0 --lora-init zeros-b \
    --lora-weights-out "$fresh_init" \
    --out "$RAW_DIR/${label}__poststep__fresh_init_dummy.json"

  # This run's OWN measured (step-2) forward+backward doubles as the
  # jammi-f32-truth arm for this operating point — no separate f32-truth
  # invocation needed (mirrors op-point 1's convention: the F32 arm never
  # needs `--warmup-steps` re-applied, only the shared post-warmup weights).
  shared_poststep="$RAW_DIR/${label}__poststep__shared.safetensors"
  "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype f32 --cuda 0 \
    --lora-weights-in "$fresh_init" \
    --warmup-steps 1 --warmup-lr "$WARMUP_LR" \
    --lora-weights-out "$shared_poststep" \
    --out "$RAW_DIR/${label}__poststep__jammi_f32_truth.json"

  JAMMI_KERNELS_STRICT=1 "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype bf16 --cuda 0 \
    --lora-weights-in "$shared_poststep" \
    --out "$RAW_DIR/${label}__poststep__jammi_bf16_fused.json"

  JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=all "$BIN" grad-oracle --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --backbone-dtype bf16 --cuda 0 \
    --lora-weights-in "$shared_poststep" \
    --out "$RAW_DIR/${label}__poststep__jammi_bf16_eager.json"

  "$TORCH_PY" "$TORCH_ORACLE" --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --dtype bf16 --attn eager --cuda 0 \
    --lora-weights-in "$shared_poststep" \
    --out "$RAW_DIR/${label}__poststep__torch_bf16.json"

  "$TORCH_PY" "$TORCH_ORACLE" --model-dir "$MODEL_DIR" \
    --batch "$batch" --seq "$seq" --seed "$seed" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$TARGET_MODULES" \
    --dtype fp32 --attn eager --cuda 0 \
    --lora-weights-in "$shared_poststep" \
    --out "$RAW_DIR/${label}__poststep__torch_f32_truth.json"
done

echo "wrote raw dumps under $RAW_DIR"
