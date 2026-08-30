#!/usr/bin/env bash
# gpu_inference_ab.sh -- issue #335's within-run GPU perf A/B producer:
# builds parent-HEAD and the PR change as two FULL, SIMULTANEOUSLY-RESIDENT
# clones, runs `jammi-bench gpu-inference-scale` on the SAME rented pod, back
# to back, in an order-balanced A,B,B,A interleaving, and merges the four
# legs through `gpu_inference_ab.py` (this directory) -- the SAME
# `generic_leg_identity_fields`/`generic_leg_premise_violations` refusal core
# `encode_ab.sh` already builds on.
#
# RECORDING-ONLY (v1): this producer, and the comparator it drives, NEVER
# fail a run over the measured ratio -- issue #335's own exit criterion
# forbids enforcement before multi-pod/both-device-model validation. The
# only hard (nonzero, non-75) refusal here is a CORRECTNESS-of-measurement
# problem: an identity/premise mismatch between legs, or a binary whose own
# `provenance` does not match the clone it was built from.
#
# ## WHY TWO FULL CLONES, NOT ONE CHECKOUT SWITCHING REFS
#
# `ModelInferenceSpec::embed_model_dir`/`::classifier_model_dir`
# (`crates/jammi-bench/src/model_inference.rs:165-175`) bake
# `env!("CARGO_MANIFEST_DIR")` -- a COMPILE-TIME constant -- into the fixture
# path each binary reads its committed `config.json`/`model.safetensors`/
# `tokenizer.json` from, joined against a RELATIVE `../../cookbook/
# fixtures/...` that resolves through whatever is CURRENTLY on disk at that
# absolute path when the binary actually RUNS, not when it was compiled.
# `gpu_inference.rs::run` now also HASHES those same files
# (`GpuInferenceTier::embed_checkpoint_*_sha256`/`infer_checkpoint_*_sha256`,
# issue #335's own D4 identity contract) at RUN time, off whatever bytes sit
# at that path at that moment.
#
# A single checkout that builds parent, `git checkout`s the PR ref IN PLACE,
# then builds the PR binary would therefore be UNSOUND the instant the
# parent binary is later RUN: its own baked fixture path now resolves
# through whatever the checkout was switched to LAST (the PR's tree), not
# the tree parent was built against -- silently measuring the wrong
# checkpoint bytes under the parent binary's own label, or (if the PR
# genuinely changed the fixture) producing a checksum on the parent leg that
# does not match what the parent binary was actually compiled to serve. Two
# clones, each holding its OWN ref checked out for its OWN binary's entire
# lifetime, is the only sound shape here -- never a single repo whose ref
# moves between builds.
#
# ## Parent = merge-base(origin/main, HEAD), NEVER HEAD^
#
# `HEAD^` is "the commit immediately before this one" -- for a PR with more
# than one commit (or a squash-merge target whose HEAD already IS the merge
# commit), that is not "main before this PR's changes landed", it is an
# arbitrary interior commit of the PR's own history (or, for a merge commit,
# an UNRELATED parent branch entirely). `git merge-base origin/main HEAD` is
# the actual common ancestor -- the real baseline this PR diverged from --
# regardless of how many commits the PR carries or whether HEAD is itself a
# merge commit.
#
# ## Order-balanced legs: A, B, B, A (never A, A, B, B)
#
# See `gpu_inference_ab.py`'s own module doc for the full rationale (a
# first-order linear clock/thermal drift trend cancels when the two
# adjacent-pair ratios straddle the run in opposite physical order).
#
# ## `--aa-null`: the D6 empirical-null instrument
#
# `GPU_INFERENCE_AB_AA_NULL=1` builds the PARENT sha TWICE, from TWO
# independent clones (this script's normal clone-a, plus a THIRD clone
# playing clone-b's role) -- comparing a sha against itself, built and run
# independently, so the resulting ratio distribution is pure build+
# measurement+pod noise, never a real code difference. This is the
# instrument that will eventually populate
# `gpu_inference_ab.py::PLACEHOLDER_ADVISORY_BAND` with a real,
# derived-from-evidence band (D6) -- until that artifact exists and a real
# band is derived from it, the placeholder band stays exactly that, a
# placeholder. Output lands under `ci/artifacts/gpu-perf-aa-null/` (staged
# here; a human still decides which run(s) get committed as the campaign's
# own evidence, the same convention `runpod_gpu_howwell.sh`'s own artifact
# pull follows).
#
# ## Exit codes
#   0  -- GREEN (see `gpu_inference_ab.py`'s own exit-code doc: recorded
#         regardless of the ratio's own value).
#   1  -- a REAL correctness-of-measurement refusal: an identity mismatch
#         between legs (`gpu_inference_ab.py` status INVALID), the PR/
#         comparison clone's build FAILED, or a binary's own `provenance`
#         does not match the clone it was supposedly built from.
#   75 -- neutral "nothing to compare": the PARENT clone's build failed (a
#         broken baseline is not a code regression this A/B can attribute
#         to the PR), HEAD already equals origin/main's merge-base (no
#         PR-side commits at all), both binaries report the SAME
#         `build_sha` outside `--aa-null` mode, the GPU was busy, or fewer
#         than all four legs produced an `OK` report.
#
# Env vars:
#   GPU_INFERENCE_AB_WORK_DIR       where the two/three clones + their own
#                                   CARGO_TARGET_DIRs live (default a
#                                   sibling of this checkout,
#                                   "../gpu-perf-ab-<UTC timestamp>").
#   GPU_INFERENCE_AB_OUT_DIR        where the merged report + raw legs land
#                                   (default "<repo>/.gpu-inference-ab-report/
#                                   <UTC timestamp>").
#   GPU_INFERENCE_AB_AA_NULL=1      the D6 instrument (see above).
#   GPU_INFERENCE_AB_SKIP_GPU_CHECK=1  skip the nvidia-smi idle check
#                                   (CPU/dry-run smoke test only).
#   GPU_INFERENCE_AB_DRY_RUN=1      print every command this script would
#                                   run instead of executing it; writes
#                                   `{"tool":"dry-run",...}` stub files per
#                                   leg so the merge stage still runs
#                                   end-to-end against real (if
#                                   fabricated-empty) files. Never clones,
#                                   never builds, never touches the GPU or
#                                   the network; never claims a real number.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

GPU_INFERENCE_AB_DRY_RUN="${GPU_INFERENCE_AB_DRY_RUN:-0}"
GPU_INFERENCE_AB_AA_NULL="${GPU_INFERENCE_AB_AA_NULL:-0}"
GPU_INFERENCE_AB_SKIP_GPU_CHECK="${GPU_INFERENCE_AB_SKIP_GPU_CHECK:-0}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="${GPU_INFERENCE_AB_WORK_DIR:-$(dirname "$REPO_ROOT")/gpu-perf-ab-$TS}"
OUT_DIR="${GPU_INFERENCE_AB_OUT_DIR:-$REPO_ROOT/.gpu-inference-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

CLONE_A="$WORK_DIR/clone-a"
CLONE_B="$WORK_DIR/clone-b"
TARGET_A="$WORK_DIR/target-a"
TARGET_B="$WORK_DIR/target-b"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# --- GPU must be idle before the first build even starts (stacked_sweep.sh
# precedent) -- a busy GPU makes every timing this script would eventually
# produce meaningless before a single clone is even made. ---
if [ "$GPU_INFERENCE_AB_SKIP_GPU_CHECK" != "1" ] && [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "::error::'nvidia-smi --query-compute-apps' failed (rc=$RC): $BUSY -- refusing to proceed without a confirmed-idle GPU. Set GPU_INFERENCE_AB_SKIP_GPU_CHECK=1 only for a CPU/dry-run smoke test." >&2
    exit 1
  fi
  if [ -n "$BUSY" ]; then
    echo "::error::GPU is not idle -- nvidia-smi reports compute processes:" >&2
    echo "$BUSY" >&2
    exit 1
  fi
fi

# --- ensure this checkout carries enough history for a real merge-base ---
if [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  if [ "$(git -C "$REPO_ROOT" rev-parse --is-shallow-repository 2>&1)" = "true" ]; then
    run_cmd git -C "$REPO_ROOT" fetch --unshallow --quiet origin \
      || { echo "::error::'git fetch --unshallow' failed -- cannot compute a real merge-base off a shallow checkout." >&2; exit 2; }
  fi
  run_cmd git -C "$REPO_ROOT" fetch --quiet origin main \
    || echo "::warning::'git fetch origin main' failed -- using whatever local origin/main ref this checkout already has." >&2
fi

PR_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>&1)" || { echo "::error::'git rev-parse HEAD' failed: $PR_SHA" >&2; exit 2; }
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$PR_SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$PR_SHA') -- refusing" >&2
  exit 2
fi
if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
  PARENT_SHA="0000000000000000000000000000000000000a"
else
  PARENT_SHA="$(git -C "$REPO_ROOT" merge-base origin/main HEAD 2>&1)" \
    || { echo "::error::'git merge-base origin/main HEAD' failed: $PARENT_SHA -- is origin/main fetched?" >&2; exit 2; }
fi

if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
  # The D6 instrument: the "b" role is a THIRD clone of the SAME parent sha,
  # never the PR sha -- see this script's own header.
  B_SHA="$PARENT_SHA"
else
  B_SHA="$PR_SHA"
  if [ "$PARENT_SHA" = "$PR_SHA" ]; then
    echo "::warning::HEAD IS origin/main's merge-base (no PR-side commit) -- nothing to compare; neutral exit 75." >&2
    exit 75
  fi
fi

mkdir -p "$WORK_DIR"

# --- TWO SIMULTANEOUSLY-RESIDENT clones, checked out BEFORE any build ---
clone_and_checkout() {
  local clone="$1" sha="$2" label="$3"
  run_cmd git clone --no-hardlinks --quiet "$REPO_ROOT" "$clone" \
    || { echo "::error::cloning $label ($REPO_ROOT -> $clone) failed" >&2; return 1; }
  run_cmd git -C "$clone" checkout --quiet --detach "$sha" \
    || { echo "::error::checking out $label sha $sha in $clone failed" >&2; return 1; }
  # jammi-kernels/build.rs panics loudly the moment a flash-attn build
  # reaches it with no CUTLASS submodule checked out (runpod_gpu_howwell.sh's
  # own note) -- a local clone of a checkout that DID init the submodule
  # still carries no submodule content of its own until this runs.
  run_cmd git -C "$clone" submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
    || { echo "::error::submodule init for $label ($clone) failed" >&2; return 1; }
  return 0
}

clone_and_checkout "$CLONE_A" "$PARENT_SHA" "a (parent)" || exit 2
clone_and_checkout "$CLONE_B" "$B_SHA" "b ($( [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ] && echo 'aa-null parent' || echo pr ))" || exit 2

# --- both binaries built FULLY, up front, before any measurement ---
build_clone() {
  local clone="$1" target="$2"
  CARGO_TARGET_DIR="$target" run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$clone/Cargo.toml"
}

if ! build_clone "$CLONE_A" "$TARGET_A"; then
  echo "::warning::parent clone build FAILED -- a broken baseline is not a code regression this A/B can attribute to the PR; neutral exit 75." >&2
  exit 75
fi
if ! build_clone "$CLONE_B" "$TARGET_B"; then
  echo "::warning::comparison clone build FAILED -- treated the same as a parent build failure (this producer is recording-only in v1: a build that cannot even run is 'nothing to compare', never a perf FAIL); neutral exit 75." >&2
  exit 75
fi

BIN_A="$TARGET_A/release/jammi-bench"
BIN_B="$TARGET_B/release/jammi-bench"

# --- per-binary provenance cross-check (C5.1 shape, cf. encode_ab.sh) ---
check_provenance() {
  local bin="$1" clone="$2" label="$3"
  local expect_sha prov_json prov_sha
  expect_sha="$(git -C "$clone" rev-parse HEAD)"
  prov_json="$("$bin" provenance 2>&1)" || { echo "::error::'$bin provenance' ($label) failed: $prov_json" >&2; return 1; }
  prov_sha="$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$bin provenance' ($label): $prov_json" >&2; return 1; }
  if [ -z "$prov_sha" ] || [ "$prov_sha" != "$expect_sha" ]; then
    echo "::error::$label binary's provenance build_sha=$prov_sha but its own clone's HEAD=$expect_sha -- refusing before any leg." >&2
    return 1
  fi
  printf '%s' "$prov_sha"
}

if [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  A_PROV_SHA="$(check_provenance "$BIN_A" "$CLONE_A" a)" || exit 1
  B_PROV_SHA="$(check_provenance "$BIN_B" "$CLONE_B" b)" || exit 1
  if [ "$GPU_INFERENCE_AB_AA_NULL" != "1" ] && [ "$A_PROV_SHA" = "$B_PROV_SHA" ]; then
    echo "::warning::both binaries report the SAME build_sha ($A_PROV_SHA) -- nothing to compare outside --aa-null mode; neutral exit 75." >&2
    exit 75
  fi
else
  A_PROV_SHA="$PARENT_SHA"
  B_PROV_SHA="$B_SHA"
fi

# --- one leg. NEVER aborts the sweep -- a leg failure is recorded as this
# leg's own outcome (its .exit file + stderr), same discipline
# stacked_sweep.sh/encode_ab.sh's own run_leg already follow. ---
run_leg() {
  local name="$1" bin="$2"
  local out_file="$RAW_DIR/${name}.json"
  local err_file="$RAW_DIR/${name}.stderr"
  local exit_file="$RAW_DIR/${name}.exit"

  printf -- '--- %s: ' "$name"
  printf '%q ' "$bin" gpu-inference-scale
  printf '\n'

  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"leg":"%s"}\n' "$name" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "$bin" gpu-inference-scale > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${name} FAILED (exit ${rc}) -- recorded as this leg's own outcome; run continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

# --- Order-balanced A, B, B, A -- NEVER A, A, B, B (see this script's own
# header for why the order itself is load-bearing). ---
run_leg a1 "$BIN_A"
run_leg b1 "$BIN_B"
run_leg b2 "$BIN_B"
run_leg a2 "$BIN_A"

# --- merge: gpu_inference_ab.py's own leg-premise refusal + primary-endpoint
# ratio + advisory classification. ---
python3 "$DIR/gpu_inference_ab.py" "$RAW_DIR" "$OUT_DIR" "$A_PROV_SHA" "$B_PROV_SHA"
MERGE_RC=$?

# --- --aa-null: stage the merged artifact for eventual commit under
# ci/artifacts/gpu-perf-aa-null/ (D6's own evidence path) -- a human still
# decides which run(s) get committed, the same convention
# runpod_gpu_howwell.sh's own artifact pull follows. ---
if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
  AA_NULL_DIR="$REPO_ROOT/ci/artifacts/gpu-perf-aa-null"
  mkdir -p "$AA_NULL_DIR"
  if [ -f "$OUT_DIR/gpu_inference_ab_report.json" ]; then
    cp "$OUT_DIR/gpu_inference_ab_report.json" "$AA_NULL_DIR/aa-null-$TS.json"
    echo "=== --aa-null artifact staged (not committed): $AA_NULL_DIR/aa-null-$TS.json ===" >&2
  fi
fi

echo
echo "=== clones: a=$CLONE_A ($PARENT_SHA) b=$CLONE_B ($B_SHA) ==="
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$MERGE_RC"
