#!/usr/bin/env bash
# gpu-perf-ab GPU driver: rents a real A100 (sm_80), clones the checkout at
# GIT_REF onto it, and runs `ci/scripts/perf/gpu_inference_ab.sh` remotely —
# the producer of the `encode` ladder's revision edge (this checkout against
# its merge-base with main, judged by `jammi-bench ladder encode --revision
# direct`). Shared deploy/run/teardown lives in runpod_lib.sh, the same
# machinery `runpod_gpu_prove.sh`/`runpod_gpu_howwell.sh` use.
#
# Off the merge path (see gpu-perf-ab.yml): invoked only by that workflow's
# workflow_dispatch / `run-gpu-perf-ab` PR-label triggers, never a schedule.
#
# Exit 0 = the ladder's status was GREEN; 1 = the status is RED (the revised
# build is slower than the base by more than the rung's own noise band),
# RED_FOR_INVESTIGATION (faster by more than the band — investigated, never
# assumed favourable) or INVALID (a refusal: an identity disagreement, a
# missing or unreadable leg, a non-stationary series); 2 = a usage/infra
# error; 75 = nothing to compare (no capacity, no idle GPU, no parent-side
# build, HEAD is its own merge-base).
#
# Env vars:
#   RUNPOD_API_KEY, GIT_REPO, GIT_REF   as the other drivers
#   GPU_PERF_AB_AA_NULL=1               every side the parent sha: the
#                                       instrument's own null
#   GPU_PERF_AB_ARTIFACT_DIR            where the verdict and legs are pulled
#                                       (default .gpu-pull/gpu-perf-ab)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# Three clones and three release build trees beside the checkout.
RP_DISK_GB="${RP_DISK_GB:-90}"
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-}"
if [ -z "$GIT_REF" ]; then
  echo "::error::GIT_REF must be set explicitly (a branch name or a commit sha)." >&2
  exit 2
fi

GPU_PERF_AB_AA_NULL="${GPU_PERF_AB_AA_NULL:-0}"
GPU_PERF_AB_ARTIFACT_DIR="${GPU_PERF_AB_ARTIFACT_DIR:-${REPO_ROOT}/.gpu-pull/gpu-perf-ab}"

rp_sweep

rp_init
echo "=== provisioning a live A100 (sm_80) for the gpu-perf-ab producer ==="
rp_deploy_live_a100
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "::warning::gpu-perf-ab FAILED for capacity — no A100 on RunPod (SUPPLY_CONSTRAINT); neutral exit 75."
  exit 75
fi

echo "=== running gpu_inference_ab.sh on ${RP_HOST}:${RP_PORT} (GPU_PERF_AB_AA_NULL=${GPU_PERF_AB_AA_NULL}) ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"

echo "::group::disk space pre-flight"
GPU_PERF_AB_MIN_FREE_GB=40
AVAIL_GB="\$(df -BG / | awk 'NR==2 {gsub(/G/,"",\$4); print \$4}')"
if [ -z "\$AVAIL_GB" ] || ! [[ "\$AVAIL_GB" =~ ^[0-9]+\$ ]]; then
  echo "::warning::could not parse available disk space from 'df -BG /' (got '\$AVAIL_GB') -- skipping the pre-flight check."
elif [ "\$AVAIL_GB" -lt "\$GPU_PERF_AB_MIN_FREE_GB" ]; then
  echo "::error::only \${AVAIL_GB}GB free on / but three source trees and three build trees need an estimated \${GPU_PERF_AB_MIN_FREE_GB}GB -- refusing before any clone; exit 75."
  exit 75
fi
echo "\${AVAIL_GB}GB free, >= \${GPU_PERF_AB_MIN_FREE_GB}GB needed -- proceeding."
echo "::endgroup::"

cd /root && rm -rf jammi-ai
$(cat "$DIR/runpod_clone_checkout.sh")

runpod_perf_ab_clone_and_checkout /root/jammi-ai "${GIT_REPO}" "${GIT_REF}" main \
  || { echo "::error::clone/checkout/wrong-tree-verification failed -- see the log above"; exit 2; }

echo "::group::gpu-perf-ab revision edge (gpu_inference_ab.sh)"
GPU_INFERENCE_AB_AA_NULL="${GPU_PERF_AB_AA_NULL}" bash ci/scripts/perf/gpu_inference_ab.sh
rc=\$?
echo "::endgroup::"
echo "GPU_PERF_AB_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== gpu-perf-ab exit=${rc} ==="

# The producer's own `$OUT_DIR` holds `ladder_verdict.json`, `ladder_table.txt`
# and `raw/` (one leg per side and repeat); the clones and build trees live
# beside the checkout and are never pulled.
mkdir -p "$GPU_PERF_AB_ARTIFACT_DIR"
if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    "root@${RP_HOST}:/root/jammi-ai/.gpu-inference-ab-report/" "${GPU_PERF_AB_ARTIFACT_DIR}/" \
    && echo "=== pulled gpu-perf-ab artifact (ladder verdict + table + raw/ legs) -> ${GPU_PERF_AB_ARTIFACT_DIR} ===" \
    || echo "::warning::gpu-perf-ab artifact pull failed -- ${rc} above is still authoritative; the pod is torn down on exit, so this evidence is now unrecoverable."
else
  echo "::warning::no live pod (RP_HOST/RP_PORT unset) -- skipping artifact pull."
fi

# Surface the ladder's status by name, with every cause it names; a remote
# exit of 0 with a non-GREEN status is forced non-zero rather than trusted.
VERDICT_JSON="$(find "$GPU_PERF_AB_ARTIFACT_DIR" -name ladder_verdict.json 2>/dev/null | sort | tail -1)"
if [ -n "$VERDICT_JSON" ] && [ -f "$VERDICT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$VERDICT_JSON" 2>/dev/null || echo "UNKNOWN")"
  CAUSES="$(python3 -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["causes"]))' "$VERDICT_JSON" 2>/dev/null || echo "unknown (could not read ${VERDICT_JSON})")"
  echo "=== ladder status: ${STATUS} (${VERDICT_JSON}) ==="
  if [ "$STATUS" != "GREEN" ]; then
    echo "::error::gpu-perf-ab status=${STATUS} -- ${CAUSES}"
    if [ "$rc" -eq 0 ]; then
      echo "::error::remote exit was 0 but the ladder's status=${STATUS} is non-GREEN -- forcing a non-zero exit."
      rc=1
    fi
  elif [ "$rc" -ne 0 ]; then
    echo "::error::gpu-perf-ab status=GREEN but exit=${rc} -- the failure is outside the ladder (a build, a clone, or the pod)."
  fi
elif [ "$rc" -eq 0 ]; then
  echo "::error::the run exited 0 but no ladder_verdict.json was pulled -- nothing was judged; forcing a non-zero exit."
  rc=1
else
  echo "::warning::no ladder_verdict.json found under ${GPU_PERF_AB_ARTIFACT_DIR} -- cannot name the status."
fi

exit "$rc"
