#!/usr/bin/env bash
# The `encode` ladder's producer run: every rung's legs, on ONE box, over ONE
# corpus, in one run — then the comparator. This script RUNS legs and decides
# nothing; every ratio, budget and verdict is `jammi-bench ladder encode`'s.
#
# THE LEGS. Two producers, four arms, each arm run once per take, in a
# palindrome over the arms so a box that drifts over the run (a thermal or
# clock trend) moves every arm's mean alike and cancels out of any two arms'
# ratio:
#
#   jammi (direct,plan,plan-partitioned)  torch-plan  torch-sorted  | reversed …
#
#   jammi          `jammi-bench encode-step --rung direct --rung plan --rung
#                  plan-partitioned`: the three engine rungs INTERLEAVED in one
#                  process per unit (the legs an edge's speed is read from),
#                  then each rung again ALONE (`--rung <one>`, filed as
#                  `a<take>` beside the interleaved `r<take>` legs: the legs
#                  its space is read from, a shared process's high-water marks
#                  belonging to no one rung).
#   torch-plan     `torch_encode.py --order plan --attn eager`: the reference
#                  forwarding the chunks the engine's plan cuts — the
#                  semantic twin, the `torch` rung.
#   torch-sorted   `torch_encode.py --order length-sorted --attn sdpa`: the
#                  reference forwarding them longest-first, as
#                  `sentence-transformers`' `encode()` does — the bar a user
#                  holds the engine to. Filed under the `torch` rung too, in
#                  its own legs directory, so the comparator sees each order as
#                  its own run.
#
# The first jammi run leaves the corpus it served (and, without a checkpoint,
# the fixture it served) in its exchange directory; every torch leg reads THAT
# corpus and loads THAT checkpoint. Every leg, jammi and torch, carries its
# per-iteration time series; every leg's device memory is read by the ONE
# external sampler (`jammi-bench sample-device`) wrapped around its process.
#
# Not a CI job: it builds and runs a real release binary and needs a torch
# venv. Invoked on a pod or a dev box:
#   ci/scripts/perf/encode_ab.sh
#
# Env vars:
#   ENCODE_AB_OUT_DIR       where the legs and the verdict land (default
#                           "<repo>/.encode-ab-report/<UTC timestamp>").
#   ENCODE_AB_MODEL_DIR     a local checkpoint directory both stacks load
#                           (config.json, model.safetensors, tokenizer.json,
#                           optionally 1_Pooling/config.json). Unset: the jammi
#                           legs serve their compiled-in fixture, leave it in
#                           the exchange directory, and the torch legs load it
#                           from there — the same bytes either way.
#   ENCODE_AB_ROWS          the sweep (default "16,1024,16384").
#   ENCODE_AB_TAKE          the takes each unit is measured as, comma-separated;
#                           unset, the producers' own default — the fewest
#                           the ladder measures a rung against itself with.
#   ENCODE_AB_PARTITIONS    N for the plan-partitioned rung (default 4).
#   ENCODE_AB_BATCH_SIZE / ENCODE_AB_BATCH_TOKENS
#                           the chunk budget every rung's forwards are cut
#                           under — `[inference] batch_size` rows and
#                           `batch_tokens` padded tokens (defaults 32, 16384).
#   ENCODE_AB_DTYPE         f32 | bf16 | f16, both stacks (default f32): the
#                           jammi legs' `--compute-precision`, the torch legs'
#                           `--dtype`.
#   ENCODE_AB_TORCH_ANN_INDEX
#                           1 (default): the torch legs also build and save the
#                           ANN graph the engine's sink builds, so both spans
#                           close on the same work (needs `usearch` in the torch
#                           venv). 0: they stop at the Parquet file, and their
#                           legs say so.
#   ENCODE_AB_RUNGS         the engine rungs, comma-separated (default
#                           `direct,plan,plan-partitioned`); the plane's
#                           `placed` and `shape-d` join them when named —
#                           built with the plane, over the pinned Postgres
#                           catalog and S3-class store (`pg_test_catalog.sh`,
#                           `s3_test_store.sh`) this run starts, with a
#                           `jammi-server` fleet built beside the bench.
#   ENCODE_AB_ITERS         serves per rung, every one timed and filed (default
#                           64; even — the rungs are interleaved — and at least
#                           the ladder's minimum run, 32: the ladder cuts each
#                           run's initial transient itself).
#   ENCODE_AB_CUDA_ORDINAL  optional CUDA device ordinal (unset = CPU; when set,
#                           every leg runs on it and the build is the fused GPU
#                           stack, `cuda,jammi-encoders/flash-attn` — the same
#                           features every GPU producer builds, so a serve is
#                           measured on the arms a deployment admits).
#   ENCODE_AB_DRY_RUN=1     print every command instead of executing it. Never
#                           builds, never touches the network, never claims a
#                           number.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

ENCODE_AB_DRY_RUN="${ENCODE_AB_DRY_RUN:-0}"
ENCODE_AB_MODEL_DIR="${ENCODE_AB_MODEL_DIR:-}"
ENCODE_AB_ROWS="${ENCODE_AB_ROWS:-16,1024,16384}"
ENCODE_AB_TAKE="${ENCODE_AB_TAKE:-}"
ENCODE_AB_PARTITIONS="${ENCODE_AB_PARTITIONS:-4}"
ENCODE_AB_BATCH_SIZE="${ENCODE_AB_BATCH_SIZE:-32}"
ENCODE_AB_BATCH_TOKENS="${ENCODE_AB_BATCH_TOKENS:-16384}"
ENCODE_AB_DTYPE="${ENCODE_AB_DTYPE:-f32}"
ENCODE_AB_TORCH_ANN_INDEX="${ENCODE_AB_TORCH_ANN_INDEX:-1}"
ENCODE_AB_ITERS="${ENCODE_AB_ITERS:-64}"
ENCODE_AB_RUNGS="${ENCODE_AB_RUNGS:-direct,plan,plan-partitioned}"
IFS=',' read -r -a RUNGS <<< "$ENCODE_AB_RUNGS"
FLEET=0
for rung in "${RUNGS[@]}"; do
  case "$rung" in
    direct|plan|plan-partitioned) ;;
    placed|shape-d) FLEET=1 ;;
    *) echo "::error::ENCODE_AB_RUNGS names '$rung'; the engine rungs are direct, plan, plan-partitioned, placed, shape-d." >&2; exit 2 ;;
  esac
done
ENCODE_AB_CUDA_ORDINAL="${ENCODE_AB_CUDA_ORDINAL:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ENCODE_AB_OUT_DIR:-$REPO_ROOT/.encode-ab-report/$TS}"
mkdir -p "$OUT_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_encode.py"
# The torch venv and its default are resolved in one place, torch_venv.py.
TORCH_PY="$(python3 "$DIR/torch_venv.py" --path)/bin/python3"
# The corpus (and fixture) every leg reads: what the first jammi run served.
EXCHANGE_DIR="$OUT_DIR/exchange"
# One legs directory per comparator run: the engine's rungs beside the torch
# rung in its semantic order, and beside it in its length-sorted order.
LEGS_PLAN="$OUT_DIR/legs-plan"
LEGS_SORTED="$OUT_DIR/legs-sorted"

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

# A CUDA ordinal pulls in the engine's CUDA backend — without it `--cuda`
# has no device to select; a plane rung pulls in the plane, and the fleet
# its jobs run on is built with the same kernels and the S3 driver.
BENCH_FEATURES=()
SERVER_FEATURES=(storage-s3)
[ -n "$ENCODE_AB_CUDA_ORDINAL" ] && BENCH_FEATURES+=(cuda jammi-encoders/flash-attn) && SERVER_FEATURES+=(cuda flash-attn)
[ "$FLEET" = 1 ] && BENCH_FEATURES+=(plane)
SERVER_BIN="$TARGET_DIR/release/jammi-server"
if [ "$ENCODE_AB_DRY_RUN" != "1" ]; then
  features="$(IFS=','; echo "${BENCH_FEATURES[*]}")"
  run_cmd cargo build --release -p jammi-bench ${features:+--features "$features"} --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench ${features:+--features $features} failed" >&2; exit 1; }
  if [ "$FLEET" = 1 ]; then
    run_cmd cargo build --release -p jammi-server --bin jammi-server --features "$(IFS=','; echo "${SERVER_FEATURES[*]}")" --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-server failed" >&2; exit 1; }
  fi
fi

# A plane rung's catalog and store: the pinned Postgres and S3-class store,
# started for this run and stopped with it.
if [ "$FLEET" = 1 ] && [ "$ENCODE_AB_DRY_RUN" != "1" ]; then
  source "$DIR/plane_backends.sh"
  plane_backends_up "$OUT_DIR/plane"
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

# --- one producer invocation (mirrors finetune_ab.sh's run_leg: NEVER
# aborts the run -- a failure is recorded as that invocation's own outcome,
# and the comparator refuses the legs it left missing).
# run_legs LABEL CMD...
run_legs() {
  local label="$1"; shift
  local log="$OUT_DIR/${label}.log"
  local exit_file="$OUT_DIR/${label}.exit"

  printf -- '--- %s: ' "$label"
  printf '%q ' "$@"
  printf '\n'

  if [ "$ENCODE_AB_DRY_RUN" = "1" ]; then
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "$@" > "$log" 2>&1 || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${label} FAILED (exit ${rc}) -- recorded; the run continues." >&2
    tail -n 5 "$log" 2>/dev/null || true
  fi
  return 0
}

# run_jammi_legs LABEL LEGS_DIR RUNG...
run_jammi_legs() {
  local label="$1" legs_dir="$2"; shift 2
  local -a cmd=("$BIN" encode-step --task embed
    --rows "$ENCODE_AB_ROWS"
    --partitions "$ENCODE_AB_PARTITIONS"
    --batch-size "$ENCODE_AB_BATCH_SIZE" --batch-tokens "$ENCODE_AB_BATCH_TOKENS"
    --compute-precision "$ENCODE_AB_DTYPE"
    --iters "$ENCODE_AB_ITERS"
    --exchange-dir "$EXCHANGE_DIR" --legs-dir "$legs_dir")
  [ -n "$ENCODE_AB_TAKE" ] && cmd+=(--take "$ENCODE_AB_TAKE")
  local rung
  for rung in "$@"; do cmd+=(--rung "$rung"); done
  [ "$FLEET" = 1 ] && cmd+=(--server-bin "$SERVER_BIN")
  # `--model-dir`/`--cuda` are OMITTED entirely when unset, so the hermetic
  # default (the compiled-in fixture on `Device::Cpu`) is the flagless run.
  [ -n "$ENCODE_AB_MODEL_DIR" ] && cmd+=(--model-dir "$ENCODE_AB_MODEL_DIR")
  [ -n "$ENCODE_AB_CUDA_ORDINAL" ] && cmd+=(--cuda "$ENCODE_AB_CUDA_ORDINAL")
  run_legs "$label" "${cmd[@]}"
}

# run_torch_legs LABEL LEGS_DIR ORDER ATTN
run_torch_legs() {
  local label="$1" legs_dir="$2" order="$3" attn="$4"
  local -a cmd=("$TORCH_PY" "$REF_SCRIPT"
    --model-dir "${ENCODE_AB_MODEL_DIR:-$EXCHANGE_DIR/model}" --exchange-dir "$EXCHANGE_DIR"
    --out-dir "$OUT_DIR/${label}.out" --legs-dir "$legs_dir" --sampler-bin "$BIN"
    --rows "$ENCODE_AB_ROWS"
    --batch-size "$ENCODE_AB_BATCH_SIZE" --batch-tokens "$ENCODE_AB_BATCH_TOKENS"
    --dtype "$ENCODE_AB_DTYPE" --iters "$ENCODE_AB_ITERS"
    --order "$order" --attn "$attn")
  [ "$ENCODE_AB_TORCH_ANN_INDEX" = "1" ] && cmd+=(--ann-index)
  [ -n "$ENCODE_AB_TAKE" ] && cmd+=(--take "$ENCODE_AB_TAKE")
  [ -n "$ENCODE_AB_CUDA_ORDINAL" ] && cmd+=(--cuda "$ENCODE_AB_CUDA_ORDINAL")
  mkdir -p "$OUT_DIR/${label}.out"
  run_legs "$label" "${cmd[@]}"
}

# The palindrome over the arms. The interleaved jammi run is first so its
# exchange directory exists for every torch leg; the solo runs close it.
run_jammi_legs jammi-interleaved "$LEGS_PLAN" "${RUNGS[@]}"
run_torch_legs torch-plan "$LEGS_PLAN" plan eager
run_torch_legs torch-sorted "$LEGS_SORTED" length-sorted sdpa
for rung in "${RUNGS[@]}"; do
  run_jammi_legs "jammi-$rung" "$LEGS_PLAN" "$rung"
done

# The sorted comparison sees the same engine legs beside the other torch order.
if [ "$ENCODE_AB_DRY_RUN" != "1" ]; then
  mkdir -p "$LEGS_SORTED"
  cp -R "$LEGS_PLAN"/. "$LEGS_SORTED"/ 2>/dev/null || true
fi

rc=0
run_cmd "$BIN" ladder encode "$LEGS_PLAN" --out "$OUT_DIR/verdict-plan" || rc=$?
run_cmd "$BIN" ladder encode "$LEGS_SORTED" --out "$OUT_DIR/verdict-sorted" || rc=$?

echo
echo "=== legs + verdicts: ${OUT_DIR} ==="
exit "$rc"
