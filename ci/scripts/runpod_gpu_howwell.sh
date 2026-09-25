#!/usr/bin/env bash
# how-well GPU driver: rents a real A100 (sm_80), clones the
# checkout at GIT_REF onto it, and runs `ci/scripts/perf/finetune_run_ab.sh`
# remotely — the producer of the `train-run` ladder's torch edge, executed
# against a live checkpoint. Shared deploy/run/teardown
# lives in runpod_lib.sh, the same machinery `runpod_gpu_prove.sh` already
# uses (rp_sweep/rp_init/rp_deploy_live_a100/rp_run_remote) — this script is
# that one's how-well sibling, not a second orchestration mechanism.
#
# Off the merge path (see gpu-howwell.yml's own doc): this driver is invoked
# ONLY by that workflow's workflow_dispatch / `run-howwell` PR-label triggers,
# never a schedule.
#
# Exit 0 = the ladder's status was GREEN; non-zero = status is RED,
# RED_FOR_INVESTIGATION, or INVALID (the pre-registered decision rule fired,
# or a correctness-of-measurement problem was found — see the "ladder status"
# log line below for WHICH one and its named causes, rather than a bare exit
# code an operator has to cross-reference against the pulled artifact);
# 75 = no A100 capacity (neutral skip, the SAME convention runpod_gpu_prove.sh
# uses).
#
# Env vars (forwarded verbatim into the remote leg — see
# finetune_run_ab.sh's own doc for each one's meaning/default):
#   GIT_REPO / GIT_REF          what to clone (defaults mirror
#                                runpod_gpu_prove.sh's own).
#   HOWWELL_MODEL_DIR            REQUIRED — the pod's own provisioned
#                                checkpoint directory (ModernBERT-large
#                                primary — this driver does not provision
#                                the checkpoint itself, only forwards the
#                                path an operator/pod-seed step already
#                                placed there).
#   HOWWELL_SEEDS                forwarded as FINETUNE_RUN_AB_SEEDS
#                                (default: the pre-registered 12-seed gate
#                                set, finetune_run_ab.sh's own default).
#   HOWWELL_OBJECTIVE             forwarded as FINETUNE_RUN_AB_OBJECTIVE.
#                                REQUIRED -- no default (the choice must be
#                                made deliberately on every dispatch); this
#                                script refuses loudly if unset.
#   HOWWELL_LR0_SEEDS             forwarded as FINETUNE_RUN_AB_LR0_SEEDS
#                                (default: empty — the lr=0 RED control is
#                                opt-in).
#   HOWWELL_PODS                  pods to spread HOWWELL_SEEDS over, one share
#                                each, round-robin (default 1: one pod runs
#                                every seed and judges its own legs). Above 1,
#                                each share runs on its own pod at once, the
#                                legs are pulled and merged, and the first
#                                share's pod judges the merged legs before it
#                                is torn down; a control seed's legs run on
#                                the pod that trains that seed.
#   HOWWELL_DRY_RUN=1             print the shares and the judge command; rent
#                                nothing.
#   HOWWELL_ARTIFACT_DIR          where the ladder verdict/table is pulled
#                                back to once the remote run finishes
#                                (default: "<repo>/.gpu-pull/how-well" —
#                                this mirrors gpu-dev.sh's own `pull` subcommand's rsync
#                                invocation rather than inventing a second
#                                retrieval mechanism. `.gpu-pull/` is
#                                already gitignored — a human commits the
#                                specific run(s) that matter as evidence,
#                                this driver
#                                only makes them retrievable).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

HOWWELL_MODEL_DIR="${HOWWELL_MODEL_DIR:-}"
if [ -z "$HOWWELL_MODEL_DIR" ]; then
  echo "::error::HOWWELL_MODEL_DIR must name the pod's own provisioned checkpoint directory." >&2
  exit 2
fi
HOWWELL_SEEDS="${HOWWELL_SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12}"
# No silent default -- gpu-howwell.yml's own workflow_dispatch `objective`
# input is REQUIRED for exactly this reason, refusing before this script is
# even invoked; a `mnrl` fallback here would only be reachable via a DIRECT
# invocation of this script that bypasses that workflow, silently choosing
# an objective on the caller's behalf rather than making the choice
# deliberate on every dispatch).
HOWWELL_OBJECTIVE="${HOWWELL_OBJECTIVE:-}"
if [ -z "$HOWWELL_OBJECTIVE" ]; then
  echo "::error::HOWWELL_OBJECTIVE must be set to 'mnrl' or 'triplet' -- no default (the objective is chosen deliberately on every dispatch); gpu-howwell.yml's workflow_dispatch 'objective' input already enforces this on the merge path, but a direct invocation of this script must refuse just as loudly." >&2
  exit 2
fi
HOWWELL_LR0_SEEDS="${HOWWELL_LR0_SEEDS:-}"
HOWWELL_ARTIFACT_DIR="${HOWWELL_ARTIFACT_DIR:-${REPO_ROOT}/.gpu-pull/how-well}"

# Sweep before renting anything — same orphan-bounding reasoning
# runpod_gpu_prove.sh's own header states (this workflow ALSO sets
# cancel-in-progress: false at the job level, so a cancelled run never
# orphans a pod, but the sweep stays cheap insurance regardless).
HOWWELL_PODS="${HOWWELL_PODS:-1}"
case "$HOWWELL_PODS" in ''|*[!0-9]*|0) echo "::error::HOWWELL_PODS must be a positive integer (got '${HOWWELL_PODS}')." >&2; exit 2 ;; esac
IFS=',' read -r -a ALL_SEEDS <<< "$HOWWELL_SEEDS"
if [ "$HOWWELL_PODS" -gt "${#ALL_SEEDS[@]}" ]; then
  echo "::error::HOWWELL_PODS=${HOWWELL_PODS} exceeds the ${#ALL_SEEDS[@]} seeds to run -- a pod with no seed rents for nothing." >&2
  exit 2
fi
HOWWELL_DRY_RUN="${HOWWELL_DRY_RUN:-0}"

# The seeds pod `i` runs (1-based), round-robin over HOWWELL_SEEDS, and the
# lr=0 control seeds among them -- a control leg is paired with its own seed's
# adapter, so a control seed's pod is the pod that trains that seed.
share_seeds() { local i="$1" out=() k; for k in "${!ALL_SEEDS[@]}"; do [ $(( (k % HOWWELL_PODS) + 1 )) -eq "$i" ] && out+=("${ALL_SEEDS[$k]}"); done; (IFS=','; echo "${out[*]}"); }
share_lr0() { local seeds=",$1," out=() s want; IFS=',' read -r -a want <<< "$HOWWELL_LR0_SEEDS"; for s in "${want[@]}"; do [ -n "$s" ] && case "$seeds" in *",$s,"*) out+=("$s") ;; esac; done; (IFS=','; echo "${out[*]}"); }

# One pod's share: rent, clone, provision, run the producer over `seeds` with
# `lr0` as its control seeds, pull the raw legs to `pull_dir`. The first share
# (`lane_tests`=1) first runs the torch-host lane's tests on its pod — the
# twins and the producers exercised for real, with cargo and the torch venv —
# so a broken twin fails before a single leg is paid for. The pod is torn
# down on this shell's exit (rp_cleanup). Exit 75 = no capacity.
howwell_share() {
  local seeds="$1" lr0="$2" pull_dir="$3" lane_tests="${4:-0}" allow_no_lr0=1 rc
  [ -n "$lr0" ] && allow_no_lr0=0
  if [ "$HOWWELL_DRY_RUN" = "1" ]; then
    echo "--- share: seeds=${seeds} lr0=${lr0:-none} lane_tests=${lane_tests} -> ${pull_dir}"
    return 0
  fi
  rp_init
  echo "=== provisioning a live A100 (sm_80) for seeds ${seeds} ==="
  rp_deploy_live_a100 || return $?
  echo "=== running finetune_run_ab.sh on ${RP_HOST}:${RP_PORT} for seeds ${seeds} ==="
  rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
# the depth-1 clone above carries no submodule content; jammi-kernels/build.rs
# panics loudly ("CUTLASS submodule is not checked out") the moment a
# jammi-encoders/flash-attn build reaches it, so init it explicitly before
# any build step runs.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass

echo "::group::torch venv (torch_venv.py --provision)"
python3 ci/scripts/perf/torch_venv.py --provision \
  || { echo "::error::the torch venv could not be provisioned on this pod -- refusing before any leg." >&2; exit 1; }
echo "::endgroup::"

if [ "${lane_tests}" = "1" ]; then
  echo "::group::torch-host lane tests (the twins and the producers, on this pod, before any leg)"
  TORCH_VENV=/root/jammi-ai/.venv-torch-ref python3 ci/scripts/run_script_tests.py --lane torch-host \
    || { echo "::error::the torch-host lane's tests failed on this pod -- a twin or a producer is broken; no leg is produced." >&2; exit 1; }
  echo "::endgroup::"
fi

echo "::group::train-run, torch edge (finetune_run_ab.sh) seeds=${seeds} lr0=${lr0}"
MODEL_DIR="${HOWWELL_MODEL_DIR}" \
  FINETUNE_RUN_AB_SEEDS="${seeds}" \
  FINETUNE_RUN_AB_OBJECTIVE="${HOWWELL_OBJECTIVE}" \
  FINETUNE_RUN_AB_LR0_SEEDS="${lr0}" FINETUNE_RUN_AB_ALLOW_NO_LR0="${allow_no_lr0}" \
  FINETUNE_RUN_AB_LORA_DROPOUT=0 \
  bash ci/scripts/perf/finetune_run_ab.sh
rc=\$?
echo "::endgroup::"
echo "HOWWELL_EXIT=\${rc}"; exit \$rc
REMOTE
  rc=$?
  echo "=== share ${seeds}: producer exit=${rc} ==="
  mkdir -p "$pull_dir"
  if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
    rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" --exclude 'work/' \
      "root@${RP_HOST}:/root/jammi-ai/.finetune-run-ab-report/" "${pull_dir}/" \
      && echo "=== pulled share ${seeds} (ladder verdict + table + raw/ legs, work/ excluded) -> ${pull_dir} ===" \
      || echo "::warning::share ${seeds}: artifact pull failed -- its legs are unrecoverable for this invocation."
  fi
  return "$rc"
}

# The merged legs of every share, judged once: on the first share's pod (still
# alive here), so the runner needs no toolchain of its own.
howwell_judge_merged() {
  local merged="$HOWWELL_ARTIFACT_DIR/merged" report d
  mkdir -p "$merged/raw/natural"
  for d in "$HOWWELL_ARTIFACT_DIR"/share-*/; do
    report="$(find "$d" -maxdepth 2 -type d -name raw 2>/dev/null | head -1)"; [ -n "$report" ] || continue
    cp "$report"/resident__*.json "$report"/torch__*.json "$merged/raw/" 2>/dev/null
    cp "$report"/natural/torch__*.json "$merged/raw/natural/" 2>/dev/null
  done
  echo "=== merged $(ls "$merged/raw"/*.json 2>/dev/null | wc -l | tr -d ' ') legs from ${HOWWELL_PODS} shares ==="
  if [ "$HOWWELL_DRY_RUN" = "1" ]; then
    echo "--- judge: jammi-bench ladder train-run merged/raw --from torch --to resident --axes outcome --out merged"
    return 0
  fi
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" "$merged/raw/" "root@${RP_HOST}:/root/merged/raw/" || { echo "::error::could not stage the merged legs on the judging pod." >&2; return 1; }
  rp_run_remote <<REMOTE
cd /root/jammi-ai && ./target/release/jammi-bench ladder train-run /root/merged/raw --from torch --to resident --axes outcome --out /root/merged
REMOTE
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" "root@${RP_HOST}:/root/merged/ladder_verdict.json" "root@${RP_HOST}:/root/merged/ladder_table.txt" "$merged/" \
    || echo "::warning::the merged verdict could not be pulled."
}

[ "$HOWWELL_DRY_RUN" = "1" ] || rp_sweep
if [ "$HOWWELL_PODS" -eq 1 ]; then
  howwell_share "$HOWWELL_SEEDS" "$HOWWELL_LR0_SEEDS" "$HOWWELL_ARTIFACT_DIR" 1
  rc=$?
  VERDICT_DIR="$HOWWELL_ARTIFACT_DIR"
else
  # Shares 2..N rent and run in their own subshells, each torn down on its own
  # exit; RunPod refuses a burst of simultaneous deploys, so the starts are
  # staggered. Share 1 runs here, and its pod then judges the merged legs.
  mkdir -p "$HOWWELL_ARTIFACT_DIR"
  pids=()
  for i in $(seq 2 "$HOWWELL_PODS"); do
    ( howwell_share "$(share_seeds "$i")" "$(share_lr0 "$(share_seeds "$i")")" "$HOWWELL_ARTIFACT_DIR/share-$i" ) > "$HOWWELL_ARTIFACT_DIR/share-$i.log" 2>&1 &
    pids+=($!)
    [ "$HOWWELL_DRY_RUN" = "1" ] || sleep 20
  done
  howwell_share "$(share_seeds 1)" "$(share_lr0 "$(share_seeds 1)")" "$HOWWELL_ARTIFACT_DIR/share-1" 1
  rc=$?
  for pid in "${pids[@]}"; do wait "$pid" || echo "::warning::a share's pod exited non-zero (see its share-*.log; its legs, if pulled, still enter the merge)."; done
  for i in $(seq 2 "$HOWWELL_PODS"); do tail -n 3 "$HOWWELL_ARTIFACT_DIR/share-$i.log" 2>/dev/null | sed "s/^/[share-$i] /"; done
  howwell_judge_merged || rc=1
  VERDICT_DIR="$HOWWELL_ARTIFACT_DIR/merged"
fi
echo "=== how-well exit=${rc} ==="

# --- surface the ladder's status by NAME, with every cause it names: the
# verdict's `status` already folds the torch edge, every refusal and the
# mutant columns, and `causes` says which. Defensive: if the exit code somehow
# read 0 despite a non-GREEN status, force non-zero here rather than let a
# mismatch pass silently.
VERDICT_JSON="$(find "$VERDICT_DIR" -maxdepth 1 -name ladder_verdict.json 2>/dev/null | head -1)"
if [ -n "$VERDICT_JSON" ] && [ -f "$VERDICT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$VERDICT_JSON" 2>/dev/null || echo "UNKNOWN")"
  CAUSES="$(python3 -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["causes"]))' "$VERDICT_JSON" 2>/dev/null || echo "unknown (could not read ${VERDICT_JSON})")"
  echo "=== ladder status: ${STATUS} (${VERDICT_JSON}) ==="
  if [ "$STATUS" != "GREEN" ]; then
    echo "::error::how-well status=${STATUS} -- ${CAUSES}"
    if [ "$rc" -eq 0 ]; then
      echo "::error::exit was 0 but the ladder's status=${STATUS} is non-GREEN -- forcing a non-zero exit."
      rc=1
    fi
  elif [ "$rc" -ne 0 ]; then
    echo "::error::how-well status=GREEN but exit=${rc} -- the failure is outside the ladder (a build, a provisioning step, or a pod)."
  fi
elif [ "$HOWWELL_DRY_RUN" != "1" ]; then
  echo "::warning::no ladder_verdict.json found under ${VERDICT_DIR} -- cannot name the status."
fi

exit "$rc"
