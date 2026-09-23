#!/usr/bin/env bash
# The train-step ladder on a GPU: `torch` -> `fused`, one unit per shape,
# judged by `jammi-bench ladder train-step`. Runs on a pod; not a CI job.
#
# Every jammi leg runs off ONE binary built here, with nothing requested off:
# each leg's dispatch counters are what the ladder's rung premise reads to
# prove the fused arm and the flash cascade.
#
# Legs of one shape run in a balanced order — fused, torch, torch, fused — so
# a drift over the session lands on both sides of every ratio. Each rung runs
# twice per shape; the repeats are what the ladder's noise band is measured
# from, and a cost inside that band is reported as indistinguishable from 1
# whatever its point value.
#
# Env vars:
#   MODEL_DIR             checkpoint dir both stacks load from (required
#                         unless FINETUNE_STEP_AB_DRY_RUN=1)
#   JAMMI_MODEL_DIR / TORCH_MODEL_DIR   override MODEL_DIR per stack
#   FINETUNE_STEP_AB_SHAPES   `batch:seq:dropout` triples, comma-separated
#                         (default "8:128:0,8:512:0,16:128:0,8:128:0.05,8:512:0.05,16:128:0.05")
#   FINETUNE_STEP_AB_STEPS / _WARMUP   measured / warmup steps (default 30 / 10)
#   FINETUNE_STEP_AB_SEED     synthetic-data and init seed (default 42)
#   FINETUNE_STEP_AB_CUDA     CUDA ordinal both stacks target (default 0)
#   FINETUNE_STEP_AB_OUT_DIR  where legs and the verdict land
#                         (default "<repo>/.finetune-step-ab-report/<UTC timestamp>")
#   FINETUNE_STEP_AB_DRY_RUN=1   print every command instead of running it
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

FINETUNE_STEP_AB_DRY_RUN="${FINETUNE_STEP_AB_DRY_RUN:-0}"
FINETUNE_STEP_AB_SHAPES="${FINETUNE_STEP_AB_SHAPES:-8:128:0,8:512:0,16:128:0,8:128:0.05,8:512:0.05,16:128:0.05}"
FINETUNE_STEP_AB_STEPS="${FINETUNE_STEP_AB_STEPS:-30}"
FINETUNE_STEP_AB_WARMUP="${FINETUNE_STEP_AB_WARMUP:-10}"
FINETUNE_STEP_AB_SEED="${FINETUNE_STEP_AB_SEED:-42}"
FINETUNE_STEP_AB_CUDA="${FINETUNE_STEP_AB_CUDA:-0}"
TARGET_MODULES="Wqkv,Wo,Wi"

MODEL_DIR="${MODEL_DIR:-}"
JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-$MODEL_DIR}"
TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-$MODEL_DIR}"
if [ -z "$JAMMI_MODEL_DIR" ] || [ -z "$TORCH_MODEL_DIR" ]; then
  if [ "$FINETUNE_STEP_AB_DRY_RUN" = "1" ]; then
    JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-/checkpoints/DRY-RUN-PLACEHOLDER}"
    TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-/checkpoints/DRY-RUN-PLACEHOLDER}"
  else
    echo "::error::MODEL_DIR (or JAMMI_MODEL_DIR/TORCH_MODEL_DIR) must name a checkpoint directory." >&2
    exit 2
  fi
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${FINETUNE_STEP_AB_OUT_DIR:-$REPO_ROOT/.finetune-step-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_finetune_step.py"
# The torch venv and its default are resolved in one place, torch_venv.py.
TORCH_PY="$(python3 "$DIR/torch_venv.py" --path)/bin/python3"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$FINETUNE_STEP_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

if [ "$FINETUNE_STEP_AB_DRY_RUN" != "1" ]; then
  run_cmd cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda,jammi-encoders/flash-attn failed" >&2; exit 1; }
  run_cmd python3 "$DIR/torch_venv.py" --provision \
    || { echo "::error::torch venv provisioning failed (ci/scripts/perf/torch_venv.py --provision)" >&2; exit 1; }
  for ckpt in "$JAMMI_MODEL_DIR" "$TORCH_MODEL_DIR"; do
    python3 "$DIR/checkpoint_files.py" "$ckpt" \
      || { echo "::error::refusing before any leg: $ckpt is not a whole checkpoint." >&2; exit 1; }
  done
  SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  BIN_PROV_SHA="$("$BIN" provenance | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])')" \
    || { echo "::error::'$BIN provenance' failed" >&2; exit 1; }
  if [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this checkout is at $SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# One leg: `<rung>__<unit>__<take>.json`. A failed run leaves its exit code
# and stderr beside the leg it did not write; the ladder refuses the unit.
run_leg() {
  local rung="$1" unit="$2" take="$3"
  shift 3
  local leg="$RAW_DIR/${rung}__${unit}__${take}"
  printf -- '--- %s: ' "$(basename "$leg")"
  printf '%q ' "$@"
  printf '\n'
  if [ "$FINETUNE_STEP_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  local rc=0
  "$@" > "$leg.stdout" 2> "$leg.stderr" || rc=$?
  echo "$rc" > "$leg.exit"
  if [ "$rc" -eq 0 ]; then
    mv "$leg.stdout" "$leg.json"
  else
    echo "::warning::$(basename "$leg") FAILED (exit ${rc}) -- recorded; the sweep continues." >&2
    tail -n 5 "$leg.stderr" 2>/dev/null || true
  fi
  return 0
}

fused_leg() { # $1=unit $2=take $3=batch $4=seq $5=dropout
  JAMMI_KERNELS_STRICT=1 run_leg fused "$1" "$2" \
    "$BIN" finetune-step \
    --model-dir "$JAMMI_MODEL_DIR" \
    --batch "$3" --seq "$4" \
    --steps "$FINETUNE_STEP_AB_STEPS" --warmup "$FINETUNE_STEP_AB_WARMUP" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$5" \
    --target-modules "$TARGET_MODULES" \
    --backbone-dtype bf16 \
    --cuda "$FINETUNE_STEP_AB_CUDA" --seed "$FINETUNE_STEP_AB_SEED" \
    --batched-forward true \
    --expect-kernels-disabled ""
}
torch_leg() { # $1=unit $2=take $3=batch $4=seq $5=dropout
  run_leg torch "$1" "$2" \
    "$TORCH_PY" "$REF_SCRIPT" \
    --model-dir "$TORCH_MODEL_DIR" \
    --batch "$3" --seq "$4" \
    --steps "$FINETUNE_STEP_AB_STEPS" --warmup "$FINETUNE_STEP_AB_WARMUP" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$5" \
    --target-modules "$TARGET_MODULES" \
    --dtype bf16 --attn sdpa --lora-init peft \
    --cuda "$FINETUNE_STEP_AB_CUDA" --seed "$FINETUNE_STEP_AB_SEED"
}

IFS=',' read -r -a SHAPES <<< "$FINETUNE_STEP_AB_SHAPES"
for shape in "${SHAPES[@]}"; do
  IFS=':' read -r BATCH SEQ DROPOUT <<< "$shape"
  UNIT="b${BATCH}s${SEQ}d${DROPOUT//./p}"
  fused_leg "$UNIT" r1 "$BATCH" "$SEQ" "$DROPOUT"
  torch_leg "$UNIT" r1 "$BATCH" "$SEQ" "$DROPOUT"
  torch_leg "$UNIT" r2 "$BATCH" "$SEQ" "$DROPOUT"
  fused_leg "$UNIT" r2 "$BATCH" "$SEQ" "$DROPOUT"
done

run_cmd "$BIN" ladder train-step "$RAW_DIR" --axes speed,space --out "$OUT_DIR"
LADDER_RC=$?
echo
echo "=== raw legs + ladder verdict: ${OUT_DIR} ==="
exit "$LADDER_RC"
