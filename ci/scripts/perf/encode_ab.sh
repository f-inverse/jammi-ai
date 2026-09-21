#!/usr/bin/env bash
# The encode-step producer: runs
# `jammi-bench encode-step` TWICE (replicate legs r1/r2) and asserts, via a
# leg-premise-refusal check, that the two legs agree on every
# `EncodePayload::IDENTITY_FIELDS` entry before their measured
# numbers (`embed_rows_per_s`/`embed_serve_ms`) are treated as "the same
# measurement" -- reusing `ci/scripts/perf/ab_merge.py`'s
# `generic_leg_identity_fields`/`generic_leg_premise_violations` (the SAME
# shared premise-refusal core `leg_premise_violations`/
# `compare_grad_oracle.py`'s own identity check build on), never a
# second, independently-drifting comparator.
#
# WHY TWO REPLICATE LEGS, NOT A JAMMI-VS-TORCH A/B: unlike
# `finetune_ab.sh`, there is no torch twin for the encode
# surface (eval is single-arm) and no forced-attention-arm A/B either (the
# fused arms are training-only by design, `attention_arm` is constant on
# this surface and FORBIDDEN from identity). So the one meaningful A/B this
# producer runs is a same-binary, same-premise REPRODUCIBILITY check: two
# independent invocations must agree on every identity field (the complete
# output-affecting parameter set for this surface), an "r1 vs r2"
# replicate convention.
#
# `jammi-bench encode-step` takes ONE flag, `--cuda <ordinal>` (omit for
# CPU -- `EncodeStepParams::gpu_device` defaults to `CPU_HERMETIC_DEVICE`,
# `main.rs`'s CI-hermetic const), the SAME `Option<usize>` convention
# `finetune_ab.sh` already threads through its own
# `--cuda "$AB_CUDA_ORDINAL"`. This script mirrors that convention via
# `ENCODE_AB_CUDA_ORDINAL` (below): UNSET keeps the CPU-hermetic default
# path byte-for-byte unchanged (no `--cuda` flag, no `cuda` cargo feature);
# SET threads `--cuda "$ENCODE_AB_CUDA_ORDINAL"` into both legs and builds
# jammi-bench with `--features cuda` -- the SAME `cuda` cargo feature
# `finetune_ab.sh`'s own `build_binary` always turns on for its GPU legs
# (that script's build additionally turns on `jammi-encoders/flash-attn`,
# which this CPU/encode-only surface has no use for -- `cuda` alone is
# already everything `--cuda` needs here) -- so the engine's CUDA backend
# is actually compiled in.
#
# Not a CI job (no GPU strictly required -- `encode-step` is CPU-hermetic by
# default -- but this DOES build+run a real jammi-bench release binary, the
# same "not free enough for every PR" reasoning `finetune_ab.sh`'s own
# header states). Invoked either via a pod/dev session or directly once a
# checkout has cargo available:
#   ci/scripts/perf/encode_ab.sh
#
# Env vars:
#   ENCODE_AB_OUT_DIR      where the merged report + raw legs land (default
#                           "<repo>/.encode-ab-report/<UTC timestamp>").
#   ENCODE_AB_CUDA_ORDINAL optional CUDA device ordinal (unset = CPU-
#                           hermetic default, unchanged; when set, both
#                           legs run `--cuda "$ENCODE_AB_CUDA_ORDINAL"` and
#                           the build step adds `--features cuda`).
#   ENCODE_AB_DRY_RUN=1    print every command this script would run instead
#                           of executing it, and write a
#                           `{"tool":"dry-run",...}` stub per leg so the
#                           merge stage still runs end-to-end against real
#                           (if fabricated-empty) files. Never mutates the
#                           checkout, never touches the network, never
#                           claims a real number.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

ENCODE_AB_DRY_RUN="${ENCODE_AB_DRY_RUN:-0}"
ENCODE_AB_CUDA_ORDINAL="${ENCODE_AB_CUDA_ORDINAL:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ENCODE_AB_OUT_DIR:-$REPO_ROOT/.encode-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"

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
    # A CUDA ordinal was requested: pull in the engine's CUDA backend, the
    # SAME `cuda` cargo feature `finetune_ab.sh`'s own `build_binary`
    # always turns on for its GPU legs -- without it `--cuda` has no
    # device to select.
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
run_leg() {
  local leg="$1"
  local out_file="$RAW_DIR/${leg}.json"
  local err_file="$RAW_DIR/${leg}.stderr"
  local exit_file="$RAW_DIR/${leg}.exit"

  # `--cuda` is OMITTED entirely when ENCODE_AB_CUDA_ORDINAL is unset, so the
  # CPU-hermetic default path (`CPU_HERMETIC_DEVICE`, main.rs's own default)
  # is byte-for-byte unchanged -- an explicit `--cuda 0` and "no flag at all"
  # are not the same premise to pin two replicate legs' identity against.
  local -a cmd=("$BIN" encode-step)
  if [ -n "$ENCODE_AB_CUDA_ORDINAL" ]; then
    cmd+=(--cuda "$ENCODE_AB_CUDA_ORDINAL")
  fi

  printf -- '--- %s: ' "$leg"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$ENCODE_AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"leg":"%s"}\n' "$leg" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${leg} FAILED (exit ${rc}) -- recorded as a leg outcome; sweep continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

run_leg r1
run_leg r2

# --- the verdict: the two replicates are the `direct` rung of the `encode`
# ladder on both sides of a revision edge with one build on every side — the
# instrument's own null. The ladder refuses a disagreement on any identity
# field and judges the second replicate's cost against the rung's own noise
# band.
for leg in r1 r2; do
  [ -f "$RAW_DIR/$leg.json" ] || continue
  cp "$RAW_DIR/$leg.json" "$RAW_DIR/direct@base__rows8__$leg.json"
  cp "$RAW_DIR/$leg.json" "$RAW_DIR/direct@revised__rows8__$leg.json"
done
run_cmd "$BIN" ladder encode "$RAW_DIR" --revision direct --axes outcome,speed,space --out "$OUT_DIR"
PY_RC=$?

echo
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$PY_RC"
