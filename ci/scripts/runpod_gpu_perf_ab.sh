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
# module's doc) are the SAME ones this script propagates — 0 = report
# written, merge status GREEN; 1 = a real correctness-of-measurement
# refusal (identity/provenance mismatch, or a PR-side build failure — the
# PR's own problem); 2 = a usage/infra error (bad args, a clone/checkout/
# fetch failure); 75 = neutral "nothing to compare safely right now" (no
# RunPod capacity, a GPU-busy pod, a PARENT-side build failure, an
# `origin/main` refresh-fetch failure, `merge-base == HEAD`, or fewer than
# four `OK` legs) — see `gpu_inference_ab.sh`'s own header for the full,
# reconciled table this driver's exit code is drawn from verbatim. This
# driver treats RunPod capacity misses (`rp_deploy_live_a100` failing) with
# the SAME 75 convention runpod_gpu_prove.sh/runpod_gpu_howwell.sh already
# use.
#
# Env vars:
#   GIT_REPO   what to clone.
#   GIT_REF    what to check out — a BRANCH NAME or a commit sha, REQUIRED
#              (no silent default): `git clone` (below) never passes this
#              to `-b` (which REJECTS an arbitrary sha, only accepting a
#              branch/tag name — round-1 adversarial audit B2's own
#              advisory), it clones the whole repo first and `git
#              checkout`s this value afterward, which accepts either shape
#              uniformly. Refuses loudly if unset rather than silently
#              defaulting to a sha-shaped `GITHUB_SHA` value that would
#              have been rejected under the OLD `clone -b` shape (the
#              caller must state a real branch/sha deliberately;
#              gpu-perf-ab.yml's own `GIT_REF` env always sets this
#              explicitly to `github.head_ref || github.ref_name`, a
#              branch name).
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
# round-1 adversarial audit B2 advisory: no silent sha-shaped default (the
# old `${GITHUB_SHA:-main}` fallback would have fed a raw commit sha into
# `clone -b`, which REJECTS anything that is not a branch/tag name) — GIT_REF
# is REQUIRED, and this script refuses loudly rather than guessing.
GIT_REF="${GIT_REF:-}"
if [ -z "$GIT_REF" ]; then
  echo "::error::GIT_REF must be set explicitly (a branch name or a commit sha) — no silent default; gpu-perf-ab.yml's own driver always sets it." >&2
  exit 2
fi

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
# round-1 adversarial audit B2 (the empirically-proven bug: "merge-base
# exits 128"): a FULL, non-single-branch clone, THEN an explicit checkout —
# never \`git clone --depth 1 -b "\${GIT_REF}"\`. That old shape was BOTH
# (a) a single-branch SHALLOW clone, whose own remote config scopes its
# default fetch refspec to ONE branch alone, so \`origin/main\` never
# existed as a local tracking ref at all once \`gpu_inference_ab.sh\` (run
# next, against THIS checkout) tried \`git merge-base origin/main HEAD\`
# against it -- empirically \`fatal: ... unknown revision\`, exit 128; and
# (b) fed straight into \`-b\`, which REJECTS an arbitrary commit sha (only
# accepts a branch/tag name) -- a GIT_REF that happened to be a raw sha
# (the old \${GITHUB_SHA:-main} fallback's own shape) would have failed
# this clone outright. A full clone + a separate \`checkout\` accepts
# EITHER shape uniformly and always creates \`origin/main\` from the
# initial clone onward (no single-branch restriction narrows the remote's
# own default fetch refspec), fixing both (a) and (b) in one change.
git clone --quiet "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
git checkout --quiet "${GIT_REF}" 2>&1 | tail -1
# gpu_inference_ab.sh clones THIS checkout TWICE (clone-a, clone-b — the
# SAME two clones every invocation makes, in either mode; --aa-null changes
# only which sha clone-b checks out, never the clone COUNT — see that
# script's own doc) into fresh sibling directories and builds each
# independently (see that script's own doc for why single-repo
# checkout-phasing is unsound here) — each of ITS OWN clones inits the
# CUTLASS submodule itself, so this outer clone needs no submodule init of
# its own.

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
