#!/usr/bin/env bash
# The CPU ladders' producer run — `propagate`, `structure`, `graph-sample`,
# `predictor-train-run`: one workload's legs on ONE box, in one run, then the
# comparator. This script RUNS legs and decides nothing; every ratio, budget
# and verdict is `jammi-bench ladder <workload>`'s.
#
# THE LEGS. Per unit, every rung once per take, in a palindrome over the
# rungs so a drift over the session lands on both sides of every ratio:
#
#   engine rungs r1 · torch rungs r1 · torch rungs r2 (reversed) · engine rungs r2 (reversed)
#
# Each leg is a process of its own (the producer's `--take` names its one
# point), so no leg inherits another's resident high-water mark. The engine's
# first take of a unit runs first: it leaves the inputs its twin reads
# (`<legs>/input/…`) and names the unit.
#
#   workload             engine rungs (default)        torch rungs (the twin)
#   propagate            plan, plan-partitioned        torch (exact), torch-geometric (PyG)
#   structure            plan, plan-partitioned        torch
#   graph-sample         sampler                       torch (PyG's node2vec walker)
#   predictor-train-run  in-process                    torch
#
# The plane's rungs join when named: `placed` for the three graph workloads,
# `placed` and `shape-d` for the predictor. They build the bench with the
# plane and a `jammi-server` fleet beside it, over the pinned Postgres catalog
# and S3-class store this run starts (`plane_backends.sh`).
#
# The predictor twin trains with the knobs the engine's leg of the same seed
# states (architecture, epochs, learning rate, clip, heads, layers), so the
# two stacks train one configuration by construction.
#
# Env vars:
#   CPU_AB_WORKLOAD   propagate | structure | graph-sample | predictor-train-run
#                     (required)
#   CPU_AB_UNITS      the sweep, comma-separated: node counts for propagate
#                     and structure (default 2048,8192,32768), nodes per
#                     community for graph-sample (default 64,256,1024: 4,224 /
#                     16,896 / 67,584 edges), seeds for the predictor
#                     (default 1..12, the seeds its learning rule is stated for)
#   CPU_AB_RUNGS      the engine rungs, comma-separated, in ladder order
#                     (defaults above); the verdict spans torch to the last
#   CPU_AB_CPUS       a `taskset -c` CPU list every leg — engine and twin
#                     alike — is pinned to (default: unpinned)
#   CPU_AB_OUT_DIR    where legs and the verdict land
#                     (default "<repo>/.cpu-ladders-ab-report/<workload>/<UTC timestamp>")
#   CPU_AB_DRY_RUN=1  print every command instead of running it
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

CPU_AB_DRY_RUN="${CPU_AB_DRY_RUN:-0}"
CPU_AB_WORKLOAD="${CPU_AB_WORKLOAD:-}"
CPU_AB_CPUS="${CPU_AB_CPUS:-}"
REFERENCE="$REPO_ROOT/crates/jammi-bench/reference"

case "$CPU_AB_WORKLOAD" in
  propagate|structure)
    DEFAULT_UNITS="2048,8192,32768"; DEFAULT_RUNGS="plan,plan-partitioned"
    LADDER_RUNGS=(plan plan-partitioned placed) ;;
  graph-sample)
    DEFAULT_UNITS="64,256,1024"; DEFAULT_RUNGS="sampler"
    LADDER_RUNGS=(sampler) ;;
  predictor-train-run)
    DEFAULT_UNITS="$(seq -s, 1 12)"; DEFAULT_RUNGS="in-process"
    LADDER_RUNGS=(in-process placed shape-d) ;;
  *)
    echo "::error::CPU_AB_WORKLOAD must be propagate, structure, graph-sample or predictor-train-run (got '$CPU_AB_WORKLOAD')." >&2
    exit 2 ;;
esac
CPU_AB_UNITS="${CPU_AB_UNITS:-$DEFAULT_UNITS}"
CPU_AB_RUNGS="${CPU_AB_RUNGS:-$DEFAULT_RUNGS}"
IFS=',' read -r -a UNITS <<< "$CPU_AB_UNITS"
IFS=',' read -r -a RUNGS <<< "$CPU_AB_RUNGS"

FLEET=0
for rung in "${RUNGS[@]}"; do
  [[ " ${LADDER_RUNGS[*]} " == *" $rung "* ]] \
    || { echo "::error::CPU_AB_RUNGS names '$rung'; the $CPU_AB_WORKLOAD engine rungs are ${LADDER_RUNGS[*]}." >&2; exit 2; }
  case "$rung" in placed|shape-d) FLEET=1 ;; esac
done
TOP_RUNG="${RUNGS[${#RUNGS[@]}-1]}"

case "$CPU_AB_WORKLOAD" in
  propagate) TORCH_RUNGS=(torch torch-geometric) ;;
  *) TORCH_RUNGS=(torch) ;;
esac

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${CPU_AB_OUT_DIR:-$REPO_ROOT/.cpu-ladders-ab-report/$CPU_AB_WORKLOAD/$TS}"
LEGS="$OUT_DIR/legs"
LOGS="$OUT_DIR/logs"
mkdir -p "$LEGS" "$LOGS"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"
SERVER_BIN="$TARGET_DIR/release/jammi-server"
# The torch venv and its default are resolved in one place, torch_venv.py.
TORCH_PY="$(python3 "$DIR/torch_venv.py" --path)/bin/python3"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  [ "$CPU_AB_DRY_RUN" = "1" ] || "$@"
}

if [ "$CPU_AB_DRY_RUN" != "1" ]; then
  [ -z "$CPU_AB_CPUS" ] || command -v taskset >/dev/null \
    || { echo "::error::CPU_AB_CPUS is set but taskset is not on PATH." >&2; exit 1; }
  # Each shape is its own literal invocation, as the guards read it.
  if [ "$FLEET" = 1 ]; then
    run_cmd cargo build --release -p jammi-bench --features plane --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench --features plane failed" >&2; exit 1; }
    run_cmd cargo build --release -p jammi-server --bin jammi-server --features storage-s3 --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-server failed" >&2; exit 1; }
  else
    run_cmd cargo build --release -p jammi-bench --manifest-path "$REPO_ROOT/Cargo.toml" \
      || { echo "::error::cargo build -p jammi-bench failed" >&2; exit 1; }
  fi
  run_cmd python3 "$DIR/torch_venv.py" --provision-graph \
    || { echo "::error::torch venv provisioning failed (ci/scripts/perf/torch_venv.py --provision-graph)" >&2; exit 1; }
  SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  BIN_PROV_SHA="$("$BIN" provenance | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])')" \
    || { echo "::error::'$BIN provenance' failed" >&2; exit 1; }
  if [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this checkout is at $SHA -- refusing before any leg." >&2
    exit 1
  fi
  if [ "$FLEET" = 1 ]; then
    source "$DIR/plane_backends.sh"
    plane_backends_up "$OUT_DIR/plane"
  fi
fi

PIN=()
[ -n "$CPU_AB_CPUS" ] && PIN=(taskset -c "$CPU_AB_CPUS")

# One leg: its producer files it under `$LEGS` and prints the file names;
# the names land in `$LOGS/<label>.stdout`. A failed leg leaves its exit code
# and stderr; the ladder refuses the unit it is missing from.
run_leg() {
  local label="$1"; shift
  printf -- '--- %s: ' "$label"
  printf '%q ' ${PIN[@]+"${PIN[@]}"} "$@"
  printf '\n'
  [ "$CPU_AB_DRY_RUN" = "1" ] && return 0
  local rc=0
  ${PIN[@]+"${PIN[@]}"} "$@" > "$LOGS/$label.stdout" 2> "$LOGS/$label.stderr" || rc=$?
  echo "$rc" > "$LOGS/$label.exit"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::$label FAILED (exit ${rc}) -- recorded; the run continues." >&2
    tail -n 5 "$LOGS/$label.stderr" 2>/dev/null || true
  fi
  return 0
}

# The unit a leg run filed, off the leg file name `<rung>__<unit>__<take>.json`
# its producer printed; in a dry run, the unit's sweep value stands in.
filed_unit() {
  local label="$1" point="$2"
  if [ "$CPU_AB_DRY_RUN" = "1" ]; then echo "unit-of-$point"; return; fi
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0].split("__")[1])' "$LOGS/$label.stdout" 2>/dev/null
}

# graph-sample's graphs: the committed synthetic shape at each size.
graph_dir() { echo "$OUT_DIR/graphs/nodes-per$1"; }
if [ "$CPU_AB_WORKLOAD" = graph-sample ]; then
  for point in "${UNITS[@]}"; do
    run_cmd "$BIN" graph-fixture --nodes-per "$point" --out "$(graph_dir "$point")" \
      || { echo "::error::graph-fixture --nodes-per $point failed" >&2; exit 1; }
  done
fi

engine_leg() { # $1=point $2=rung $3=take
  local point="$1" rung="$2" take="$3" label="${2}__${1}__r${3}"
  local -a cmd=("$BIN" "$CPU_AB_WORKLOAD" --legs-dir "$LEGS" --take "$take")
  case "$CPU_AB_WORKLOAD" in
    propagate|structure) cmd+=(--nodes "$point" --rung "$rung") ;;
    graph-sample) cmd+=(--graph "$(graph_dir "$point")") ;;
    predictor-train-run) cmd+=(--seeds "$point" --rung "$rung") ;;
  esac
  case "$rung" in placed|shape-d) cmd+=(--server-bin "$SERVER_BIN") ;; esac
  run_leg "$label" "${cmd[@]}"
  LAST_LABEL="$label"
}

# The predictor's training knobs, as flags, off the engine leg of the unit.
predictor_knobs() {
  local unit="$1"
  if [ "$CPU_AB_DRY_RUN" = "1" ]; then
    printf -- '--%s of-engine-leg ' arch epochs learning-rate grad-clip num-heads num-layers; echo; return
  fi
  python3 - "$LEGS" "$unit" <<'PY'
import json, sys
from pathlib import Path
legs, unit = Path(sys.argv[1]), sys.argv[2]
leg = next(p for p in sorted(legs.glob(f"*__{unit}__r*.json")) if not p.name.startswith("torch__"))
knobs = json.loads(leg.read_text())["tiers"]["predictor_train_run"]
print(f"--arch {knobs['architecture']} --epochs {knobs['epochs']} --learning-rate {knobs['lr']} "
      f"--grad-clip {knobs['grad_clip']} --num-heads {knobs['num_heads']} --num-layers {knobs['num_layers']}")
PY
}

torch_leg() { # $1=point $2=unit $3=rung $4=take
  local point="$1" unit="$2" rung="$3" take="$4" label="${3}__${1}__r${4}"
  local -a cmd=("$TORCH_PY")
  case "$CPU_AB_WORKLOAD" in
    propagate)
      local impl=exact; [ "$rung" = torch-geometric ] && impl=pyg
      cmd+=("$REFERENCE/torch_propagate.py" --legs-dir "$LEGS" --unit "$unit" --impl "$impl") ;;
    structure)
      cmd+=("$REFERENCE/torch_structure.py" --legs-dir "$LEGS" --unit "$unit") ;;
    graph-sample)
      cmd+=("$REFERENCE/torch_graph_sample.py" --legs-dir "$LEGS" --graph "$(graph_dir "$point")") ;;
    predictor-train-run)
      local -a knobs
      read -r -a knobs <<< "$(predictor_knobs "$unit")"
      cmd+=("$REFERENCE/torch_context_predictor.py" --legs-dir "$LEGS" --seeds "$point" "${knobs[@]}") ;;
  esac
  run_leg "$label" "${cmd[@]}" --take "$take"
}

reversed() { local i; for ((i = $# ; i > 0 ; i--)); do echo "${!i}"; done; }

for point in "${UNITS[@]}"; do
  unit=""
  for rung in "${RUNGS[@]}"; do
    engine_leg "$point" "$rung" 1
    [ -n "$unit" ] || unit="$(filed_unit "$LAST_LABEL" "$point")"
  done
  if [ -z "$unit" ]; then
    echo "::warning::no engine leg of $point was filed; its torch legs cannot run and the ladder refuses the unit." >&2
  else
    for rung in "${TORCH_RUNGS[@]}"; do torch_leg "$point" "$unit" "$rung" 1; done
    for rung in $(reversed "${TORCH_RUNGS[@]}"); do torch_leg "$point" "$unit" "$rung" 2; done
  fi
  for rung in $(reversed "${RUNGS[@]}"); do engine_leg "$point" "$rung" 2; done
done

LADDER_ARGS=(ladder "$CPU_AB_WORKLOAD" "$LEGS" --from torch --to "$TOP_RUNG" --out "$OUT_DIR")
[ "$CPU_AB_WORKLOAD" = graph-sample ] && LADDER_ARGS+=(--law-dir "$LEGS/law")
run_cmd "$BIN" "${LADDER_ARGS[@]}"
LADDER_RC=$?
echo
echo "=== legs + ladder verdict: ${OUT_DIR} ==="
exit "$LADDER_RC"
