#!/usr/bin/env bash
# A revision edge of the `encode` ladder on a GPU: the `direct` rung built
# from the parent commit against the same rung built from this checkout,
# judged by `jammi-bench ladder encode --revision direct`. Runs on a pod;
# not a CI job.
#
# Three builds from three independent clones: the parent (`base`), the
# parent again (`rebuilt` — the edge's own A/A null, because two builds of
# one sha differ by more than one build's repeats do), and this checkout
# (`revised`). Each runs `encode-step --cuda` twice, interleaved so a drift
# over the session lands on every side: base, rebuilt, revised, revised,
# rebuilt, base. The ladder judges the revised cost against the wider of the
# repeat noise band and the A/A band; a cost outside it in the slow direction
# fails, in the fast direction is routed to investigation.
#
# `GPU_INFERENCE_AB_AA_NULL=1` builds the parent a third time in place of this
# checkout, so every side is the same revision: the reading is then the
# instrument's own null.
#
# Exit codes: 0 the ladder's verdict is GREEN; 1 the verdict is not (a
# refusal, a failed hard rule, or an improvement to investigate); 2 a usage
# or git error; 75 nothing to compare (no idle GPU, no parent-side build,
# HEAD is its own merge-base).
#
# Env vars:
#   GPU_INFERENCE_AB_AA_NULL=1      every side the parent sha
#   GPU_INFERENCE_AB_CUDA           CUDA ordinal (default 0)
#   GPU_INFERENCE_AB_WORK_DIR       where clones and build trees live
#                                   (default "<repo>/../.gpu-inference-ab-work")
#   GPU_INFERENCE_AB_OUT_DIR        where legs and the verdict land
#                                   (default "<repo>/.gpu-inference-ab-report/<UTC timestamp>")
#   GPU_INFERENCE_AB_SKIP_GPU_CHECK=1   skip the idle-GPU check (smoke tests)
#   GPU_INFERENCE_AB_DRY_RUN=1      print every command instead of running it
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

GPU_INFERENCE_AB_AA_NULL="${GPU_INFERENCE_AB_AA_NULL:-0}"
GPU_INFERENCE_AB_CUDA="${GPU_INFERENCE_AB_CUDA:-0}"
GPU_INFERENCE_AB_DRY_RUN="${GPU_INFERENCE_AB_DRY_RUN:-0}"
GPU_INFERENCE_AB_SKIP_GPU_CHECK="${GPU_INFERENCE_AB_SKIP_GPU_CHECK:-0}"
WORK_DIR="${GPU_INFERENCE_AB_WORK_DIR:-$(cd "$REPO_ROOT/.." && pwd)/.gpu-inference-ab-work}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${GPU_INFERENCE_AB_OUT_DIR:-$REPO_ROOT/.gpu-inference-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

if [ "$GPU_INFERENCE_AB_SKIP_GPU_CHECK" != "1" ] && [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)" \
    || { echo "::error::'nvidia-smi --query-compute-apps' failed: $BUSY" >&2; exit 2; }
  if [ -n "$BUSY" ]; then
    echo "::warning::GPU is not idle -- nothing to compare safely now: $BUSY" >&2
    exit 75
  fi
fi

# The parent: the merge-base of this checkout with main. A shallow checkout
# is deepened first; without an `origin/main` there is no parent to compare.
SHA_RE='^[0-9a-fA-F]{40}$'
if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
  PARENT_SHA="a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"
  HEAD_SHA="b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1"
else
  if [ "$(git -C "$REPO_ROOT" rev-parse --is-shallow-repository)" = "true" ]; then
    git -C "$REPO_ROOT" fetch --unshallow --quiet origin \
      || { echo "::error::'git fetch --unshallow' failed -- no merge-base off a shallow checkout" >&2; exit 2; }
  fi
  git -C "$REPO_ROOT" fetch --quiet origin +refs/heads/main:refs/remotes/origin/main \
    || echo "::warning::fetching origin/main failed -- the merge-base below decides" >&2
  HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  PARENT_SHA="$(git -C "$REPO_ROOT" merge-base origin/main HEAD)" \
    || { echo "::error::'git merge-base origin/main HEAD' failed" >&2; exit 2; }
fi
[[ "$HEAD_SHA" =~ $SHA_RE && "$PARENT_SHA" =~ $SHA_RE ]] \
  || { echo "::error::HEAD ($HEAD_SHA) or its merge-base ($PARENT_SHA) is not a commit" >&2; exit 2; }
if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
  REVISED_SHA="$PARENT_SHA"
elif [ "$PARENT_SHA" = "$HEAD_SHA" ]; then
  echo "::warning::HEAD is its own merge-base with origin/main -- nothing to compare; exit 75." >&2
  exit 75
else
  REVISED_SHA="$HEAD_SHA"
fi

# One clone and one build per side, each in its own tree: a checkout whose
# ref moves between builds is unsound the moment an earlier binary is run
# with its fixture paths resolving through the later tree.
build_side() { # $1=side $2=sha
  local clone="$WORK_DIR/clone-$1" target="$WORK_DIR/target-$1"
  run_cmd git clone --no-hardlinks --quiet --filter=blob:none "file://$REPO_ROOT" "$clone" || return 2
  run_cmd git -C "$clone" checkout --quiet --detach "$2" || return 2
  run_cmd git -C "$clone" submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass || return 2
  CARGO_TARGET_DIR="$target" run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$clone/Cargo.toml" || return 1
  if [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
    local prov
    prov="$("$target/release/jammi-bench" provenance | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])')" || return 1
    [ "$prov" = "$2" ] || { echo "::error::side $1: the binary reports build_sha=$prov, its clone is at $2" >&2; return 1; }
  fi
}

mkdir -p "$WORK_DIR"
build_side base "$PARENT_SHA" || { echo "::warning::the parent side did not build -- nothing to compare; exit 75." >&2; exit 75; }
build_side rebuilt "$PARENT_SHA" || { echo "::warning::the parent's second build failed -- no A/A null; exit 75." >&2; exit 75; }
build_side revised "$REVISED_SHA"
case $? in
  0) ;;
  1) if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then exit 75; fi
     echo "::error::this checkout did not build -- the revision's own failure; exit 1." >&2; exit 1 ;;
  *) exit 2 ;;
esac

# One leg: `direct@<side>__<unit>__<take>.json`; the unit is the corpus size
# served. `encode-step` files its leg as `direct__rows256__r1.json` in the
# side's own legs directory, and the leg is refiled under the side tag the
# revision edge reads.
run_leg() { # $1=side $2=take
  local bin="$WORK_DIR/target-$1/release/jammi-bench"
  local leg="$RAW_DIR/direct@$1__rows256__$2"
  local side_legs="$RAW_DIR/side-$1-$2"
  printf -- '--- %s: %q encode-step --cuda %s --rung direct --rows 256 --legs-dir %q\n' "$(basename "$leg")" "$bin" "$GPU_INFERENCE_AB_CUDA" "$side_legs"
  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  local rc=0
  "$bin" encode-step --cuda "$GPU_INFERENCE_AB_CUDA" --rung direct --rows 256 --legs-dir "$side_legs" > "$leg.stdout" 2> "$leg.stderr" || rc=$?
  echo "$rc" > "$leg.exit"
  if [ "$rc" -eq 0 ] && [ -f "$side_legs/direct__rows256__r1.json" ]; then
    mv "$side_legs/direct__rows256__r1.json" "$leg.json"
  else
    echo "::warning::$(basename "$leg") FAILED (exit ${rc}) -- recorded; the ladder refuses the unit." >&2
    tail -n 5 "$leg.stderr" 2>/dev/null || true
  fi
  return 0
}

run_leg base r1
run_leg rebuilt r1
run_leg revised r1
run_leg revised r2
run_leg rebuilt r2
run_leg base r2

run_cmd "$WORK_DIR/target-base/release/jammi-bench" ladder encode "$RAW_DIR" --revision direct --axes outcome,speed,space --out "$OUT_DIR"
LADDER_RC=$?
echo
echo "=== sides: base=$PARENT_SHA rebuilt=$PARENT_SHA revised=$REVISED_SHA ==="
echo "=== raw legs + ladder verdict: ${OUT_DIR} ==="
exit "$LADDER_RC"
