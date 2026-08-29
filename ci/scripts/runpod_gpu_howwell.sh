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
# Exit 0 = the merge's own status was GREEN (a plain FAIL/INCOMPLETE/DRY_RUN
# leg is recorded, never fatal, per finetune_run_ab.sh/ab_merge.py's own
# record-don't-gate doctrine); non-zero = status is RED, RED_FOR_INVESTIGATION,
# or INVALID (unit-63 audit finding 1: the pre-registered decision rule fired,
# or a correctness-of-measurement problem was found — see the "merged status"
# log line below for WHICH one, named explicitly rather than left as a bare
# exit code an operator has to cross-reference against the pulled artifact);
# 75 = no A100 capacity (neutral skip, the SAME convention runpod_gpu_prove.sh
# uses).
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
#   HOWWELL_LR0_SEEDS             forwarded as FINETUNE_RUN_AB_LR0_SEEDS
#                                (default: empty — the lr=0 RED control is
#                                opt-in per H5 campaign step 3).
#   HOWWELL_ARTIFACT_DIR          where the merged report/table is pulled
#                                back to once the remote run finishes
#                                (default: "<repo>/.gpu-pull/how-well" —
#                                unit-63 audit advisory (c): the artifact
#                                previously never left the pod; this mirrors
#                                gpu-dev.sh's own `pull` subcommand's rsync
#                                invocation rather than inventing a second
#                                retrieval mechanism. `.gpu-pull/` is
#                                already gitignored — a human commits the
#                                specific run(s) that matter for the
#                                campaign's own evidence record, this driver
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
HOWWELL_OBJECTIVE="${HOWWELL_OBJECTIVE:-mnrl}"
HOWWELL_LR0_SEEDS="${HOWWELL_LR0_SEEDS:-}"
HOWWELL_ARTIFACT_DIR="${HOWWELL_ARTIFACT_DIR:-${REPO_ROOT}/.gpu-pull/how-well}"

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
  FINETUNE_RUN_AB_LR0_SEEDS="${HOWWELL_LR0_SEEDS}" \
  bash ci/scripts/perf/finetune_run_ab.sh
rc=\$?
echo "::endgroup::"
echo "HOWWELL_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== how-well A/B exit=${rc} ==="

# --- merged-artifact retrieval (unit-63 audit advisory (c): the artifact
# never otherwise left the pod — this driver is the ONE place still able to
# reach it, since the EXIT trap (rp_cleanup, installed by rp_init) tears the
# pod down once THIS script itself exits). Mirrors gpu-dev.sh's own `pull`
# subcommand's rsync invocation verbatim rather than inventing a second
# retrieval mechanism. Best-effort and unconditional (pulled regardless of
# ${rc} — a RED/RED_FOR_INVESTIGATION/INVALID run's own artifact is exactly
# the evidence this campaign needs to keep, not less so than a GREEN one's).
mkdir -p "$HOWWELL_ARTIFACT_DIR"
if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    "root@${RP_HOST}:/root/jammi-ai/.finetune-run-ab-report/" "${HOWWELL_ARTIFACT_DIR}/" \
    && echo "=== pulled merged how-well artifact -> ${HOWWELL_ARTIFACT_DIR} ===" \
    || echo "::warning::merged how-well artifact pull failed -- ${rc} above is still authoritative; the pod is torn down on this script's own exit, so this evidence is now unrecoverable for this invocation."
else
  echo "::warning::no live pod (RP_HOST/RP_PORT unset) -- skipping artifact pull."
fi

# --- surface the merged status by NAME (unit-63 audit finding 1: "must exit
# non-zero with the status named", not a bare exit code an operator has to
# cross-reference against the pulled artifact to identify). Defensive: if
# the remote's own exit code somehow read 0 despite a non-GREEN status (it
# should not, per ab_merge.py's own finetune-run exit-code branch), force
# non-zero here rather than let a mismatch pass silently.
REPORT_JSON="$(find "$HOWWELL_ARTIFACT_DIR" -name finetune_run_ab_report.json 2>/dev/null | sort | tail -1)"
if [ -n "$REPORT_JSON" ] && [ -f "$REPORT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$REPORT_JSON" 2>/dev/null || echo "UNKNOWN")"
  echo "=== merged status: ${STATUS} (${REPORT_JSON}) ==="
  case "$STATUS" in
    RED|RED_FOR_INVESTIGATION|INVALID)
      echo "::error::how-well status=${STATUS} -- non-GREEN (CONTRACT 63 Frame's pre-registered decision rule, or a correctness-of-measurement problem)."
      if [ "$rc" -eq 0 ]; then
        echo "::error::remote exit was 0 but merged status=${STATUS} is non-GREEN -- forcing a non-zero exit."
        rc=1
      fi
      ;;
    GREEN|DRY_RUN|INCOMPLETE) : ;;
    *) echo "::warning::how-well status=${STATUS} unrecognised." ;;
  esac
else
  echo "::warning::no finetune_run_ab_report.json found under ${HOWWELL_ARTIFACT_DIR} -- cannot name the merged status."
fi

exit "$rc"
