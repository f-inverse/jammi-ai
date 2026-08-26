#!/bin/bash
# esc-045 round 7 (GH #374) E7 bisection driver.
#
# One BLOCK binary (target-r6-base's `jammi-bench` release build — no
# `flash-attn` feature, so `attention_block_flash` never admits and every
# forward here dispatches `attention_block_fused` as the production
# default arm), one `JAMMI_KERNELS_DISABLE=<key>` per invocation
# (`JAMMI_KERNELS_STRICT=1` throughout — a genuine domain/capability
# admission failure at these shapes is a hard error, not a silent
# eager fallback). Runs `jammi-bench grad-oracle` once per key against
# the SAME Gaussian shared LoRA weights (`--lora-weights-in`) round 6
# seeded, so every leg trains from IDENTICAL A/B values -- the same
# premise `ci/scripts/perf/compare_grad_oracle.py`'s comparator
# requires.
#
# Usage: esc045_r7_bisect_sweep.sh [batch] [seq] [seed] [tag]
#
# Every dump records `kernels_disabled_requested`/`kernels_disabled_fired`
# (`grad_oracle.rs`'s own unconditional provenance pair) -- compare those
# two lists per run to confirm `requested == fired` (or, for a key with no
# live call site under this arm -- `rope_fused`/`softmax_last_dim_fused`
# when `attention_block_fused` is on, `lora_epilogue`/`lora_dropout`,
# permanently dead per `lora_linear.rs`'s own doc -- that the mismatch is
# the EXPECTED "requested but never reached" case, not a silently dropped
# env var).
set -u
BIN=${JAMMI_BENCH_BIN:-/root/wt-esc045-r6/target-r6-base/release/jammi-bench}
MODEL=${JAMMI_MODEL_DIR:-/root/checkpoints/ModernBERT-large}
LORA_IN=${JAMMI_LORA_WEIGHTS_IN:-/root/esc045-r6/lora/shared_gaussian_seed42.safetensors}
D=${JAMMI_DUMP_DIR:-/root/esc045-r6/dumps/r7}
LOG=${JAMMI_SWEEP_LOG:-/root/esc045-r6/r7_e7_sweep.log}
BATCH=${1:-4}
SEQ=${2:-128}
SEED=${3:-42}
TAG=${4:-b${BATCH}s${SEQ}_seed${SEED}}

mkdir -p "$D"
: > "$LOG"

run_one() {
  name=$1
  disable=$2
  out="$D/jammi_r7_${name}_${TAG}.json"
  if [ -n "$disable" ]; then
    JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$disable" "$BIN" grad-oracle \
      --model-dir "$MODEL" --batch "$BATCH" --seq "$SEQ" --backbone-dtype bf16 --cuda 0 \
      --seed "$SEED" --lora-weights-in "$LORA_IN" --out "$out" >>"$LOG" 2>&1
  else
    JAMMI_KERNELS_STRICT=1 "$BIN" grad-oracle \
      --model-dir "$MODEL" --batch "$BATCH" --seq "$SEQ" --backbone-dtype bf16 --cuda 0 \
      --seed "$SEED" --lora-weights-in "$LORA_IN" --out "$out" >>"$LOG" 2>&1
  fi
  ec=$?
  echo "RUN name=$name disable=[$disable] exit=$ec out=$out" | tee -a "$LOG"
}

# The full registered-key lattice (E7): baseline (nothing disabled -- the
# PRODUCTION default, all-fused arm), each two-arm-predicate key alone,
# the cast_scale/cast_add pair together (both live INSIDE
# `LowRankResidualLinear::bwd`'s own internal admission, independent of
# `lora_linear_fused` itself), the two dead-path controls, then the
# `all`-eager ceiling.
run_one baseline ""
run_one lora_linear_fused lora_linear_fused
run_one geglu_fused geglu_fused
run_one layer_norm_fused layer_norm_fused
run_one attention_block_fused attention_block_fused
run_one rope_fused rope_fused
run_one softmax_last_dim_fused softmax_last_dim_fused
run_one cast_pair "cast_scale_bf16_f32,cast_add_bf16"
run_one lora_epilogue lora_epilogue
run_one lora_dropout lora_dropout
run_one all_eager all

echo DONE >>"$LOG"
