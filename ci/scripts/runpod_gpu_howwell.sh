#!/usr/bin/env bash
# how-well GPU driver (unit 63 H4/H4b): rents a real A100 (sm_80), clones the
# checkout at GIT_REF onto it, and runs `ci/scripts/perf/finetune_run_ab.sh`
# remotely — the SAME producer this repo's own tests
# (`ci/scripts/perf/test_ab_merge.py`) drive against fixture leg directories,
# now actually executed against a live checkpoint. Shared deploy/run/teardown
# lives in runpod_lib.sh, the same machinery `runpod_gpu_prove.sh` already
# uses (rp_sweep/rp_init/rp_deploy_live_a100/rp_run_remote) — this script is
# that one's how-well sibling, not a second orchestration mechanism.
#
# Off the merge path (see gpu-howwell.yml's own doc): this driver is invoked
# ONLY by that workflow's workflow_dispatch / `run-howwell` PR-label triggers,
# never a schedule (CONTRACT H4: "NO schedule in v1").
#
# Exit 0 = the merge's own status was not INVALID (see
# finetune_run_ab.sh/ab_merge.py's own record-don't-gate doctrine — a plain
# FAIL/INCOMPLETE leg is recorded, never fatal; only INVALID is); 75 = no A100
# capacity (neutral skip, the SAME convention runpod_gpu_prove.sh uses).
#
# Env vars (forwarded verbatim into the remote leg — see
# finetune_run_ab.sh's own doc for each one's meaning/default):
#   GIT_REPO / GIT_REF          what to clone (defaults mirror
#                                runpod_gpu_prove.sh's own).
#   HOWWELL_MODEL_DIR            REQUIRED — the pod's own provisioned
#                                checkpoint directory (ModernBERT-large
#                                primary, CONTRACT H5's own checkpoint
#                                choice — this driver does not provision
#                                the checkpoint itself, only forwards the
#                                path an operator/pod-seed step already
#                                placed there).
#   HOWWELL_SEEDS                forwarded as FINETUNE_RUN_AB_SEEDS
#                                (default: the pre-registered 12-seed gate
#                                set, finetune_run_ab.sh's own default).
#   HOWWELL_OBJECTIVE             forwarded as FINETUNE_RUN_AB_OBJECTIVE
#                                (default: mnrl).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
HOWWELL_OBJECTIVE="${HOWWELL_OBJECTIVE:-mnrl}"

# Sweep before renting anything — same orphan-bounding reasoning
# runpod_gpu_prove.sh's own header states (this workflow ALSO sets
# cancel-in-progress: false at the job level, mirroring that lane's own
# $187 lesson, but the sweep stays cheap insurance regardless).
rp_sweep

rp_init
echo "=== provisioning a live A100 (sm_80) for the how-well producer ==="
rp_deploy_live_a100 || exit $?

echo "=== running finetune_run_ab.sh on ${RP_HOST}:${RP_PORT} ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
export JAMMI_REQUIRE_CUDA=1
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
echo "::group::how-well A/B (finetune_run_ab.sh)"
MODEL_DIR="${HOWWELL_MODEL_DIR}" \
  FINETUNE_RUN_AB_SEEDS="${HOWWELL_SEEDS}" \
  FINETUNE_RUN_AB_OBJECTIVE="${HOWWELL_OBJECTIVE}" \
  bash ci/scripts/perf/finetune_run_ab.sh
rc=\$?
echo "::endgroup::"
echo "HOWWELL_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== how-well A/B exit=${rc} ==="
exit "$rc"
