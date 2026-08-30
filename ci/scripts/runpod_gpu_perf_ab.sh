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
# TWO MODES, propagated verbatim (issue #335's final unit — see
# `gpu_inference_ab.sh`'s own "TWO MODES" doc for the full rationale):
# non-enforcing (the default, `GPU_PERF_AB_ENFORCE` unset/`0`) stays
# recording-only; enforcing (`GPU_PERF_AB_ENFORCE=1`, mutually exclusive
# with `GPU_PERF_AB_AA_NULL=1`) opts this run into also refusing a
# GREEN-premise run whose ratio falls outside the pre-registered advisory
# band, on EITHER side (a NULL band, never a "faster is always fine" one —
# see `gpu_inference_ab.py`'s own doc for why a too-fast ratio is
# AMBIGUOUS, not favorable by default). `gpu_inference_ab.py`'s own exit
# codes (see that module's doc) are the SAME ones this script propagates —
# 0 = report written, merge status GREEN (and, under enforcing mode, `mode
# == "ab"` AND the ratio landed inside the pre-registered band); 1 = EITHER
# a real correctness-of-measurement refusal (an identity/provenance
# mismatch, an INVALID_MEASUREMENT, a GENUINE (parsed) recorded-order
# violation, or a `b`-role runtime failure with the producer's own `mode`
# marker CONFIRMING `ab` — round-3 adversarial audit B2 correction: an
# earlier version of this doc claimed this arm was always "the PR's own
# problem, never the parent's", which overclaimed a confirmation this
# driver does not always have -- under `--aa-null` there is no PR to blame
# at all, and an unconfirmed `mode` never escalates either) OR, under
# enforcing mode ONLY, one of THREE direction-honest enforcement refusals
# (`status` stays GREEN in every one): `enforce_verdict=PERF_REGRESSION`
# (ratio above the upper edge, a real slowdown signal),
# `enforce_verdict=OUTSIDE_BAND_FAST` (ratio below the lower edge, an
# AMBIGUOUS signal — a genuine improvement or a broken/short-circuited leg,
# never assumed favorable), or `enforce_verdict=ENFORCE_INVALID_MODE`
# (enforcement requested but `mode != "ab"`) — the merged report's own
# `status`/`enforce_verdict` fields are what distinguish all of these, never
# the bare exit code alone; 2 = a usage/infra error (bad args, a
# clone/checkout/wrong-tree/fetch failure, OR
# `GPU_PERF_AB_AA_NULL=1`-together-with-`GPU_PERF_AB_ENFORCE=1`, refused at
# the producer's own edge before anything is rented); 75 = neutral "nothing
# to compare safely right now" (no RunPod capacity, insufficient free disk
# on the pod — this driver's own pre-flight `df` check, round-2 adversarial
# audit F6 — a GPU-busy pod, a PARENT-side build failure, a `MISSING`/
# `DRY_RUN` leg of either role, a `b`-role runtime failure under
# `--aa-null`/an unconfirmed mode, an unreadable/unparseable recorded
# timestamp (round-3 adversarial audit B3), an `origin/main` refresh-fetch
# failure, `merge-base == HEAD`, or fewer than four `OK` legs) — see
# `gpu_inference_ab.sh`'s own header for the full, reconciled table this
# driver's exit code is drawn from verbatim. This driver treats RunPod
# capacity misses (`rp_deploy_live_a100` failing) with the SAME 75
# convention runpod_gpu_prove.sh/runpod_gpu_howwell.sh already use.
#
# GUARD (round-4 delta-audit advisory (3)): when `GPU_PERF_AB_ENFORCE=1`
# was requested, this driver ALSO asserts the pulled report's own
# `enforce_verdict` actually reflects that (a `GREEN`-status report reading
# `NOT_ENFORCED`, or carrying no `enforce_verdict` at all, means the
# `GPU_INFERENCE_AB_ENFORCE` env var was silently DROPPED somewhere between
# this driver and the remote comparator — see the "surface the merged
# status BY NAME" block below) — forcing a nonzero exit even if the
# underlying `$rc` was `0`, so a dropped env var fails the workflow loudly
# rather than reporting a silent, unearned green.
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
#   GPU_PERF_AB_ENFORCE=1     forwarded as GPU_INFERENCE_AB_ENFORCE — issue
#                             #335's final unit, the enforcement flip (see
#                             gpu_inference_ab.sh's own "TWO MODES" doc).
#                             Default unset (0): non-enforcing, recording-
#                             only — gpu-perf-ab.yml's own label-triggered
#                             PR path never sets this; only an explicit
#                             workflow_dispatch with enforce: true does.
#   GPU_PERF_AB_ARTIFACT_DIR  where the merged report/table is pulled back
#                             to once the remote run finishes (default:
#                             "<repo>/.gpu-pull/gpu-perf-ab" — mirrors
#                             runpod_gpu_howwell.sh's own `.gpu-pull/`
#                             convention).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# round-2 adversarial audit F6: RP_DISK_GB, set BEFORE sourcing
# runpod_lib.sh (its own `RP_DISK_GB="${RP_DISK_GB:-60}"` default runs at
# SOURCE time, so this assignment must precede that line to take effect).
# Sized per runpod_lib.sh's own rule of thumb (that file's lines 56-72):
# `>= 25 (base) + S_src + S_seed + N*S_clone`. This pod hosts THREE full
# source trees at once (the outer bootstrap checkout at /root/jammi-ai,
# plus gpu_inference_ab.sh's own clone-a and clone-b) and TWO independent
# release+cuda build trees (target-a, target-b) -- exactly the "3+ trees"
# case docs/maintainer/dev-gpu.md's own citable measured numbers
# (S_src ~= 3.6 GB, S_seed ~= 7.8 GB, S_clone ~= 8.1 GB) already name:
# "a pod hosting 3+ trees sizes up (RP_DISK_GB=70+)". Using that
# already-reviewed number directly rather than re-deriving a bespoke one.
RP_DISK_GB="${RP_DISK_GB:-70}"
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
GPU_PERF_AB_ENFORCE="${GPU_PERF_AB_ENFORCE:-0}"
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

echo "=== running gpu_inference_ab.sh on ${RP_HOST}:${RP_PORT} (GPU_PERF_AB_AA_NULL=${GPU_PERF_AB_AA_NULL} GPU_PERF_AB_ENFORCE=${GPU_PERF_AB_ENFORCE}) ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
export JAMMI_REQUIRE_CUDA=1
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"

echo "::group::disk space pre-flight"
# round-2 adversarial audit F6: this workload's OWN incremental need beyond
# whatever the pod image already consumes -- TWO more full source trees
# (gpu_inference_ab.sh's own clone-a, clone-b) plus TWO independent
# release+cuda build trees (target-a, target-b), 2*(S_src + S_seed) ~=
# 2*(3.6+7.8) GB ~= 22.8 GB per docs/maintainer/dev-gpu.md's own measured
# numbers (see this driver's own RP_DISK_GB comment for the citation),
# rounded up to 30 GB for headroom. A pre-flight refusal here is strictly
# cheaper than discovering "no space left on device" partway through a
# multi-GB clone or build.
GPU_PERF_AB_MIN_FREE_GB=30
AVAIL_GB="\$(df -BG / | awk 'NR==2 {gsub(/G/,"",\$4); print \$4}')"
if [ -z "\$AVAIL_GB" ] || ! [[ "\$AVAIL_GB" =~ ^[0-9]+\$ ]]; then
  echo "::warning::could not parse available disk space from 'df -BG /' (got '\$AVAIL_GB') -- skipping the pre-flight check (best-effort only)."
elif [ "\$AVAIL_GB" -lt "\$GPU_PERF_AB_MIN_FREE_GB" ]; then
  echo "::error::only \${AVAIL_GB}GB free on / but this workload needs an estimated \${GPU_PERF_AB_MIN_FREE_GB}GB (two more source trees + two build trees) -- refusing before any clone/build starts; neutral exit 75."
  exit 75
fi
echo "\${AVAIL_GB}GB free, >= \${GPU_PERF_AB_MIN_FREE_GB}GB needed -- proceeding."
echo "::endgroup::"

cd /root && rm -rf jammi-ai
# round-2 adversarial audit F1: the clone+checkout+wrong-tree-verification
# block lives in runpod_clone_checkout.sh (never embedded inline here) --
# inlined VERBATIM below so this pod runs the EXACT same, independently
# hermetic-tested code (see that file's own doc for the full rationale,
# including round-1 adversarial audit B2's "merge-base exits 128" bug this
# clone shape already fixes: a FULL, non-single-branch, blobless partial
# clone, then a separate checkout -- never \`git clone --depth 1 -b
# "\${GIT_REF}"\`).
$(cat "$DIR/runpod_clone_checkout.sh")

runpod_perf_ab_clone_and_checkout /root/jammi-ai "${GIT_REPO}" "${GIT_REF}" main \
  || { echo "::error::clone/checkout/wrong-tree-verification failed -- see the log above"; exit 2; }
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
GPU_INFERENCE_AB_ENFORCE="${GPU_PERF_AB_ENFORCE}" \
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
# bare exit code against the pulled artifact). Also surfaces
# `enforce_verdict` (issue #335's final unit) -- with TWO modes now sharing
# exit 1, printing `status` alone is no longer enough to tell "a correctness
# refusal" apart from "a perf-magnitude refusal" without opening the JSON. ---
REPORT_JSON="$(find "$GPU_PERF_AB_ARTIFACT_DIR" -name gpu_inference_ab_report.json 2>/dev/null | sort | tail -1)"
if [ -n "$REPORT_JSON" ] && [ -f "$REPORT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$REPORT_JSON" 2>/dev/null || echo "UNKNOWN")"
  ENFORCE_VERDICT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("enforce_verdict", "n/a"))' "$REPORT_JSON" 2>/dev/null || echo "UNKNOWN")"
  echo "=== merged status: ${STATUS} enforce_verdict: ${ENFORCE_VERDICT} (${REPORT_JSON}) ==="

  # round-4 delta-audit advisory (3): enforcement was explicitly requested
  # for THIS invocation but the pulled, GREEN-status report shows it was
  # NOT applied (either no enforce_verdict field at all, or the
  # NOT_ENFORCED value gpu_inference_ab.py::build_report only ever writes
  # when its own `enforce` marker read False) -- the ONLY way that can
  # happen on a GREEN report is a dropped GPU_INFERENCE_AB_ENFORCE env var
  # somewhere between this driver and the remote comparator. A dropped
  # request must fail the workflow, never silently pass as if
  # non-enforcing had been asked for on purpose.
  if [ "$GPU_PERF_AB_ENFORCE" = "1" ] && [ "$STATUS" = "GREEN" ] \
     && { [ "$ENFORCE_VERDICT" = "NOT_ENFORCED" ] || [ "$ENFORCE_VERDICT" = "n/a" ] || [ "$ENFORCE_VERDICT" = "UNKNOWN" ]; }; then
    echo "::error::GPU_PERF_AB_ENFORCE=1 was requested for this run, but the pulled report's own status=GREEN carries enforce_verdict='${ENFORCE_VERDICT}' -- enforcement was silently NOT applied (a dropped GPU_INFERENCE_AB_ENFORCE env var somewhere in the remote pipeline is the likely cause). Forcing failure rather than trusting rc=${rc} alone -- an operator's explicit enforcement request must never report a silent, unearned green." >&2
    rc=1
  fi
else
  echo "::warning::no gpu_inference_ab_report.json found under ${GPU_PERF_AB_ARTIFACT_DIR} -- cannot name the merged status."
  # Force failure ONLY over a would-be-GREEN exit (the silent-unearned-green
  # case this guard exists for). A run that already failed keeps ITS OWN exit
  # code -- the header's exit lattice (75 = capacity/pre-flight, 2 = usage or
  # the aa_null+enforce guard) stays authoritative, never rewritten into a 1
  # that gpu-perf-ab.yml would mis-annotate as an enforcement refusal
  # (mirrors runpod_gpu_howwell.sh's own rc -eq 0 guard on its rewrite).
  if [ "$GPU_PERF_AB_ENFORCE" = "1" ] && [ "$rc" -eq 0 ]; then
    echo "::error::GPU_PERF_AB_ENFORCE=1 was requested and the run exited 0, but no merged report was pulled -- cannot confirm enforcement was applied; forcing failure rather than reporting an unconfirmable green." >&2
    rc=1
  elif [ "$GPU_PERF_AB_ENFORCE" = "1" ]; then
    echo "::warning::enforcement was requested but the run already failed (rc=${rc}) with no report to confirm against -- keeping the run's own exit code, which the header's exit lattice names." >&2
  fi
fi

exit "$rc"
