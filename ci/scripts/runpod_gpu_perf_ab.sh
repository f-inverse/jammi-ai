#!/usr/bin/env bash
# gpu-perf-ab GPU driver (issue #335): rents a real A100 (sm_80), clones the
# checkout at GIT_REF onto it, and runs `ci/scripts/perf/gpu_inference_ab.sh`
# remotely — the SAME producer this repo's own tests
# (`ci/scripts/perf/test_gpu_inference_ab.py`) drive against fixture leg
# directories, now actually executed on live hardware. Shared deploy/run/
# teardown lives in runpod_lib.sh, the same machinery `runpod_gpu_prove.sh`/
# `runpod_gpu_howwell.sh` already use (rp_sweep/rp_init/rp_deploy_live_a100/
# rp_run_remote) — this script is that family's gpu-perf-ab sibling, not a
# second orchestration mechanism.
#
# Off the merge path (see gpu-perf-ab.yml's own doc): this driver is invoked
# ONLY by that workflow's workflow_dispatch / `run-gpu-perf-ab` PR-label
# triggers, never a schedule.
#
# RECORDING-ONLY (v1): `gpu_inference_ab.py`'s own exit codes (see that
# module's doc) are the SAME ones this script propagates — 0 = GREEN
# (recorded regardless of the ratio), 1 = a real premise/provenance
# refusal, 75 = neutral "nothing to compare" (no capacity, a build failure,
# or fewer than four OK legs). This driver treats RunPod capacity misses
# (rp_deploy_live_a100 failing) with the SAME 75 convention
# runpod_gpu_prove.sh/runpod_gpu_howwell.sh already use.
#
# Env vars:
#   GIT_REPO / GIT_REF        what to clone (defaults mirror
#                             runpod_gpu_howwell.sh's own).
#   GPU_PERF_AB_AA_NULL=1     forwarded as GPU_INFERENCE_AB_AA_NULL — the D6
#                             empirical-null instrument (see
#                             gpu_inference_ab.sh's own doc). Default unset
#                             (0): the normal parent-vs-PR A/B.
#   GPU_PERF_AB_ARTIFACT_DIR  where the merged report/table is pulled back
#                             to once the remote run finishes (default:
#                             "<repo>/.gpu-pull/gpu-perf-ab" — mirrors
#                             runpod_gpu_howwell.sh's own `.gpu-pull/`
#                             convention).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

GPU_PERF_AB_AA_NULL="${GPU_PERF_AB_AA_NULL:-0}"
GPU_PERF_AB_ARTIFACT_DIR="${GPU_PERF_AB_ARTIFACT_DIR:-${REPO_ROOT}/.gpu-pull/gpu-perf-ab}"

# Sweep before renting anything — same orphan-bounding reasoning
# runpod_gpu_prove.sh/runpod_gpu_howwell.sh's own headers state.
rp_sweep

rp_init
echo "=== provisioning a live A100 (sm_80) for the gpu-perf-ab producer ==="
rp_deploy_live_a100
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "::warning::gpu-perf-ab FAILED for capacity — no A100 on RunPod (SUPPLY_CONSTRAINT); neutral, mirrors gpu_inference_ab.sh's own exit-75 'nothing to compare' convention." >&2
  exit 75
fi

echo "=== running gpu_inference_ab.sh on ${RP_HOST}:${RP_PORT} (GPU_PERF_AB_AA_NULL=${GPU_PERF_AB_AA_NULL}) ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
export JAMMI_REQUIRE_CUDA=1
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
# gpu_inference_ab.sh clones THIS checkout twice (or three times under
# --aa-null) into fresh sibling directories and builds each independently
# (see that script's own doc for why single-repo checkout-phasing is
# unsound here) — each of ITS OWN clones inits the CUTLASS submodule
# itself, so this outer clone needs no submodule init of its own.

echo "::group::gpu-perf-ab A/B (gpu_inference_ab.sh)"
GPU_INFERENCE_AB_AA_NULL="${GPU_PERF_AB_AA_NULL}" \
  bash ci/scripts/perf/gpu_inference_ab.sh
rc=\$?
echo "::endgroup::"
echo "GPU_PERF_AB_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== gpu-perf-ab A/B exit=${rc} ==="

# --- merged-artifact retrieval, unconditional (best-effort) regardless of
# ${rc} — a REFUSED (exit 1) run's own artifact is exactly the evidence an
# operator needs to see WHY, not less so than a GREEN (exit 0) one's.
# Mirrors runpod_gpu_howwell.sh's own rsync-based pull exactly. ---
mkdir -p "$GPU_PERF_AB_ARTIFACT_DIR"
if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
  # gpu_inference_ab.sh's own clones/CARGO_TARGET_DIRs live OUTSIDE this
  # checkout (a sibling of /root/jammi-ai, see that script's own
  # GPU_INFERENCE_AB_WORK_DIR default) -- `.gpu-inference-ab-report/` here
  # only ever holds the merged report + raw per-leg JSON/stderr, never a
  # clone or a build tree, so no exclude filter is needed on this path.
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    "root@${RP_HOST}:/root/jammi-ai/.gpu-inference-ab-report/" "${GPU_PERF_AB_ARTIFACT_DIR}/" \
    && echo "=== pulled merged gpu-perf-ab artifact -> ${GPU_PERF_AB_ARTIFACT_DIR} ===" \
    || echo "::warning::merged gpu-perf-ab artifact pull failed -- ${rc} above is still authoritative; the pod is torn down on this script's own exit, so this evidence is now unrecoverable for this invocation."
else
  echo "::warning::no live pod (RP_HOST/RP_PORT unset) -- skipping artifact pull."
fi

# --- surface the merged status BY NAME (mirrors runpod_gpu_howwell.sh's own
# idiom: an operator reading the log should never have to cross-reference a
# bare exit code against the pulled artifact). ---
REPORT_JSON="$(find "$GPU_PERF_AB_ARTIFACT_DIR" -name gpu_inference_ab_report.json 2>/dev/null | sort | tail -1)"
if [ -n "$REPORT_JSON" ] && [ -f "$REPORT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$REPORT_JSON" 2>/dev/null || echo "UNKNOWN")"
  echo "=== merged status: ${STATUS} (${REPORT_JSON}) ==="
else
  echo "::warning::no gpu_inference_ab_report.json found under ${GPU_PERF_AB_ARTIFACT_DIR} -- cannot name the merged status."
fi

exit "$rc"
