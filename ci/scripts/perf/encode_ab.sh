#!/usr/bin/env bash
# The unit-62 encode-step producer (E6, docs-ci domain): runs
# `jammi-bench encode-step` TWICE (replicate legs r1/r2) and asserts, via a
# leg-premise-refusal check, that the two legs agree on every
# `identity_fields.ENCODE_IDENTITY_FIELDS` entry before their measured
# numbers (`embed_rows_per_s`/`embed_serve_ms`) are treated as "the same
# measurement" -- reusing `ci/scripts/perf/ab_merge.py`'s
# `generic_leg_identity_fields`/`generic_leg_premise_violations` (the SAME
# shared premise-refusal core `leg_premise_violations`/
# `compare_grad_oracle.py`'s own identity check already build on), never a
# second, independently-drifting comparator.
#
# WHY TWO REPLICATE LEGS, NOT A JAMMI-VS-TORCH A/B: unlike
# `finetune_ab.sh`/`fa2_ab.sh`, there is no torch twin for the encode
# surface today (unit-62 PLAN.md v2 OQ4 ruling: "torch_encode.py NOT now --
# C16-style front-door record; eval is single-arm") and no forced-attention-
# arm A/B either (CONTRACT.md's Frame: "NO forced-arm encode A/B" -- the
# fused arms are training-only by design, `attention_arm` is constant on
# this surface and FORBIDDEN from identity). So the one meaningful A/B this
# producer runs is a same-binary, same-premise REPRODUCIBILITY check: two
# independent invocations must agree on every identity field (the complete
# output-affecting parameter set for this surface), the same "r1 vs r2"
# replicate convention `fa2_ab.sh` already uses for its own timing legs.
#
# `jammi-bench encode-step` takes ONE flag, `--cuda <ordinal>` (omit for
# CPU -- `EncodeStepParams::gpu_device` defaults to `CPU_HERMETIC_DEVICE`,
# `main.rs`'s CI-hermetic const), the SAME `Option<usize>` convention
# `finetune_ab.sh`/`fa2_ab.sh` already thread through their own
# `--cuda "$AB_CUDA_ORDINAL"`. This script mirrors that convention via
# `ENCODE_AB_CUDA_ORDINAL` (below): UNSET keeps the CPU-hermetic default
# path byte-for-byte unchanged (no `--cuda` flag, no `cuda` cargo feature);
# SET threads `--cuda "$ENCODE_AB_CUDA_ORDINAL"` into both legs and builds
# jammi-bench with `--features cuda` (the SAME feature `finetune_ab.sh`'s
# own `checkout_and_build` always turns on for its GPU-only legs) so the
# engine's CUDA backend is actually compiled in.
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
    # SAME `cuda` cargo feature `finetune_ab.sh`'s own `checkout_and_build`
    # always turns on for its GPU-only legs -- without it `--cuda` has no
    # device to select.
    run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench --features cuda failed" >&2; exit 1; }
  else
    run_cmd cargo build --release -p jammi-bench --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench failed" >&2; exit 1; }
  fi
fi

# --- provenance cross-check (unification contract C5.1), same shape as
# fa2_ab.sh/finetune_ab.sh/stacked_sweep.sh/clip_artifact_producer.sh:
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

# --- merge: leg-premise refusal (ENCODE_IDENTITY_FIELDS) reusing
# ab_merge.py's generic core, then record both legs' identity+provenance
# blocks (EncodeStepTier::IDENTITY_FIELDS + ::PROVENANCE_FIELDS,
# CONTRACT.md's disjoint E3 shape) into one merged, push-stamp-friendly
# JSON -- schema_version/git_sha/box/producer/status, the SAME shape
# `check_cuda_run_artifacts.py`'s schema expects of a committed cuda-run
# artifact (this script itself does not commit anything; the pod evidence
# train, CONTRACT.md Step 5, folds a real run's output into a committed
# artifact via that gate's own schema).
python3 - "$RAW_DIR" "$OUT_DIR" "$SHA" "$DIR" <<'PYEOF'
import json
import os
import sys

# argv[4] (PERF_DIR) is $DIR from the calling shell -- this script's OWN
# directory (ci/scripts/perf), passed explicitly rather than derived from
# `__file__` (meaningless for a heredoc piped over stdin via `python3 -`).
RAW_DIR, OUT_DIR, SHA, PERF_DIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, os.path.abspath(PERF_DIR))
import ab_merge  # noqa: E402
from identity_fields import ENCODE_IDENTITY_FIELDS  # noqa: E402

LEGS = ["r1", "r2"]


def load_leg(name):
    exit_path = os.path.join(RAW_DIR, f"{name}.exit")
    out_path = os.path.join(RAW_DIR, f"{name}.json")
    if not os.path.exists(exit_path):
        return {"outcome": "MISSING", "report": None}
    with open(exit_path) as fh:
        exit_code = fh.read().strip()
    try:
        with open(out_path) as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError):
        report = None
    if report is not None and report.get("ab_dry_run") is True:
        return {"outcome": "DRY_RUN", "report": None}
    if exit_code != "0" or report is None:
        return {"outcome": "FAIL", "report": None}
    return {"outcome": "OK", "report": report}


entries = {leg: load_leg(leg) for leg in LEGS}
legs_out = {}
identity_by_leg = {}
for leg, entry in entries.items():
    legs_out[leg] = {"outcome": entry["outcome"]}
    if entry["outcome"] != "OK":
        continue
    tier = entry["report"].get("tiers", {}).get("encode_step") or {}
    prov = entry["report"].get("provenance") or {}
    legs_out[leg]["identity"] = {k: tier.get(k) for k in ENCODE_IDENTITY_FIELDS}
    legs_out[leg]["provenance"] = {
        k: tier.get(k)
        for k in ("device_name", "kernels_disabled_requested", "kernels_disabled_fired", "flash_compiled", "build_features", "chunk_size", "attention_arm")
    }
    legs_out[leg]["provenance"]["build_sha"] = prov.get("build_sha")
    legs_out[leg]["measurements"] = {
        "embed_rows_per_s": tier.get("embed_rows_per_s"),
        "embed_serve_ms": tier.get("embed_serve_ms"),
    }
    identity_by_leg[leg] = ab_merge.generic_leg_identity_fields(tier, ENCODE_IDENTITY_FIELDS)

leg_premise_violations = []
ok_legs = list(identity_by_leg.keys())
if len(ok_legs) == 2:
    leg_premise_violations = ab_merge.generic_leg_premise_violations(
        ENCODE_IDENTITY_FIELDS, identity_by_leg[ok_legs[0]], identity_by_leg[ok_legs[1]], ok_legs[0], ok_legs[1]
    )
elif len(ok_legs) < 2:
    leg_premise_violations = [f"only {len(ok_legs)} of {len(LEGS)} legs produced an OK report -- cannot check leg premise"]

status = "PREMISE_MISMATCH" if leg_premise_violations else ("GREEN" if len(ok_legs) == len(LEGS) else "INCOMPLETE")

merged = {
    "schema_version": 1,
    "git_sha": SHA,
    "box": os.uname().nodename if hasattr(os, "uname") else "unknown",
    "producer": {
        "path": "ci/scripts/perf/encode_ab.sh",
        "kind": "script",
        "invocation": "ci/scripts/perf/encode_ab.sh",
        "gating": "none",
    },
    "status": status,
    "identity_fields": list(ENCODE_IDENTITY_FIELDS),
    "leg_premise_violations": leg_premise_violations,
    "legs": legs_out,
}

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "encode_ab_report.json")
with open(out_path, "w") as fh:
    json.dump(merged, fh, indent=2)

print(f"=== merged report: {out_path} ===")
print(f"status={status} leg_premise_violations={leg_premise_violations}")
sys.exit(1 if leg_premise_violations else 0)
PYEOF
PY_RC=$?

echo
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$PY_RC"
