#!/usr/bin/env bash
# GPU gang pod leg: build jammi from source on ONE RunPod pod holding TWO
# A100s and run the distributed fine-tune (gang) tests on it — the only
# place a real multi-device NCCL gang is exercised before release. Shared
# deploy/run/teardown lives in runpod_lib.sh; this driver is to the gang
# lane what runpod_gpu_prove.sh is to the prove lane, and deliberately
# shares its shapes (group markers, the verdict function, the rc contract)
# rather than inventing a second dialect.
#
# WHAT IT RENTS: one pod, `gpuCount: 2` (RP_GPU_COUNT below), from the
# `a100` candidate list. With a count > 1 `rp_deploy_arch` tries every SXM4
# candidate before any PCIe one — spike S4 rented a 2-GPU A100-SXM4-80GB
# SECURE pod at $3.18/h while the PCIe pool returned zero 2-GPU capacity.
#
# COST BOUND (human-approved). Two bounds, each stated with
# the mechanism that enforces it — none of this is prose:
#
#   (i) TERMINATE-SUCCEEDS (the ordinary path). rp_deploy_live walks the
#       `a100` candidate list (4 entries) and terminates a pod that is not
#       SSH-reachable within RP_SSH_WAIT_SECS before trying the next, so the
#       search itself can bill up to `candidates x RP_SSH_WAIT_SECS`. This
#       lane pins that wait to 300s below (the library default is 600s), and
#       gpu-gang.yml pins MAX_ATTEMPTS to "1" so there is exactly one such
#       search per run. Then the winning pod bills to RP_TTL_HOURS=1, baked
#       into the pod's OWN entrypoint (runpod_lib.sh's `_rp_deploy_payload`
#       watchdog), so it self-terminates even if this runner is SIGKILLed:
#         4 x 300 s x $3.18/h + 1 h x $3.18/h = $1.06 + $3.18 = $4.24.
#   (ii) SWEEP-ONLY (the worst path). If EVERY rp_terminate call fails — the
#       case runpod_lib.sh's own header opens with — nothing is torn down
#       early and each pod bills to its baked-in TTL. The remaining
#       enforcers are that TTL and gpu-reap.yml's rp_sweep:
#         (4 + 1) x 1 h x $3.18/h = $15.90.
#
# $3.18/h is what spike S4 measured for a SECURE 2-GPU A100-SXM4-80GB pod.
# The COMMUNITY 2-GPU rate is UNMEASURED — nothing has priced one — so
# neither figure is stated for a COMMUNITY landing.
# ci/scripts/test_gpu_gang_lane.sh re-derives (i) from the candidate list,
# the two values below and the workflow's MAX_ATTEMPTS, and fails if this
# figure and the mechanism disagree.
#
# The shared RP_TIMEOUT default (runpod_lib.sh's own 3000s = 50m) sits
# inside the TTL hour so the driver's own budget cut lands first with a
# legible group name. NOT ESTABLISHED on this base: whether a cold
# `cuda,flash-attn` build of the gang target plus the gang tests fits
# inside that hour — no run has measured it. A 124 (budget cut) or a pod
# that vanishes mid-run is therefore a COST DECISION for a human (raise
# RP_TTL_HOURS/RP_TIMEOUT deliberately, re-approving the bound), never
# something this script raises on its own.
#
# WHAT IT PROVES: the gang tests in `jammi-ai`'s `gpu_capability` target,
# selected by the `gang_` name filter named here (GANG_TEST_FILTER) so the
# filter lives in exactly one place. A name filter matching zero tests
# exits 0 with "running 0 tests ... test result: ok" — a false green. This
# driver reads
# that case as a FAILURE with its own named reason (the same never-vacuous
# rule runpod_gpu_prove.sh's `capability-surface-proof` group applies, and
# the same doctrine `check_gpu_prove_once.py` states for the prove lane):
# a leg with no test is a leg with no proof.
#
# THE ARTIFACT: the gang tests write their evidence JSON (world, collective,
# per-rank device, the same-seed digest pair, the measured per-step loss
# delta, the leg's own verdict -- exactly `pass` or `fail`, a `fail`
# carrying its own `reason` and its numbers AS MEASURED -- and the
# PRE-REGISTERED epsilon with its derivation) into
# `$JAMMI_GANG_ARTIFACT_DIR` on the pod. This driver pulls that directory
# back before the EXIT trap tears the pod down — the pod is the only place
# it exists, and it is unrecoverable afterwards. The committed form of that
# file is schema-gated by `ci/scripts/check_cuda_run_artifacts.py`'s `gang`
# kind (rule (k)); a human reviews the pulled artifact and commits it under
# `crates/jammi-kernels/artifacts/cuda-runs/`.
#
# TRIGGERS: `.github/workflows/gpu-gang.yml` only — the `run-gang` PR label
# and manual dispatch. Never `push:`, never
# `workflow_call:`, and nothing may `uses:` that workflow: a paid leg is
# never in the critical path of an automated publisher (gpu-prove.yml's own
# header states the doctrine; `check_gpu_prove_once.py` pins both lanes by
# name).
#
# Exit 0 = every gating group passed; 75 = no 2-GPU capacity (neutral
# provider condition, still RED at the workflow level — a leg with no
# capacity proved nothing); 76 = the inactivity watchdog killed a hang; 77 =
# wrong tree (the pod's own PROVE_SHA disagreed with PROVE_EXPECT_SHA); 97 =
# the rented pod is not the device this leg asked for (wrong arch, or fewer
# GPUs than RP_GPU_COUNT); 124 = budget cut with a gating group unresolved.
# A nonzero exit that is none of the above, with the suite's own groups
# otherwise green, is this driver's OWN post-run check refusing the leg:
# rsync's own exit code when the artifact pull fails (a suite that passed
# but left no retrievable evidence proves nothing reviewable). The
# id-secrecy scan that backstops the NCCL id's out-of-band crossing ships
# with the cluster leg (docs/plans/67-distributed-training/UNITS.md § U7b acceptance (id-secrecy)),
# beside the crossing it protects — this driver mints/ships no id today.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The pod's own hard deadline == this lane's approved cost ceiling (see the
# header). An explicit RP_TTL_HOURS from the caller always wins, so raising
# the ceiling is a deliberate, visible act.
RP_TTL_HOURS="${RP_TTL_HOURS:-1}"
# 1 pod x 2 GPU. This is the ONE caller that moves runpod_lib.sh's own
# `gpuCount` default off 1 (test_pod_substrate.sh's `(ab/gpuCount D5)` leg
# derives that set by scanning every tracked ci/scripts + .github/workflows
# file), so no other lane's deploy payload changes.
export RP_GPU_COUNT="${RP_GPU_COUNT:-2}"
# The deploy search's own cost term (bound (i) in the header): every
# candidate this lane tries and abandons bills for up to this long at the
# 2-GPU rate, so 300s here instead of runpod_lib.sh's 600s default halves
# that term. NOT ESTABLISHED: whether a 2-GPU SXM4 pod routinely reaches
# sshd inside 300s — S4 rented one but did not record the time to first
# SSH. A pod slower than this is terminated and the next candidate tried,
# which costs a candidate rather than money; if the lane starts exhausting
# its list on healthy capacity, this value is the first thing to look at.
# Exported BEFORE the source below because runpod_lib.sh validates it at
# source time.
export RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-300}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# The remote job's budget is runpod_lib.sh's OWN shared default (its
# `${RP_TIMEOUT:-3000}` at the ssh call — 3000s = 50m), deliberately NOT
# re-declared here: `check_gpu_prove_timings.py`'s R1 allows exactly two
# committed setter sites in this repo, and a third would be a hidden second
# source of truth for a value one place already owns. That default happens
# to be what this lane wants anyway — 50m sits inside the 1h pod deadline
# above, so a budget cut is reported by this driver, with the cut group
# named, instead of the pod disappearing under an in-flight ssh session. An
# operator raising the ceiling raises both in the environment, together.

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

# sm_80 is the gang leg's device: the plan's own 2xA100 pod-leg oracles, and
# the only arch S4 measured multi-GPU capacity for. NATIVE_COMPUTE_CAP
# overrides the CI image's baked CUDA_COMPUTE_CAP for the same #434 reason
# runpod_gpu_prove.sh states: candle-kernels builds single-arch SASS from
# that env var.
GANG_DEPLOY_ARCH=a100
NATIVE_COMPUTE_CAP=80

# The ONE place the gang test-name filter lives. The gang tests are named
# `gang_*` in `jammi-ai`'s `gpu_capability` target; a rename moves this line
# and nothing else.
GANG_TEST_FILTER="${GANG_TEST_FILTER:-gang_}"

# Where the gang tests write their evidence JSON on the pod, and where this
# driver pulls it to locally. The remote path is passed to the tests as
# JAMMI_GANG_ARTIFACT_DIR — the ONE contract between this driver and the
# gang test code. The NCCL id (128 opaque bytes minted by rank 0, nccl.rs's own
# "opaque secret ... travel to the peers out of band") rides NO path under
# either directory: it is a capability, never evidence. The id-secrecy scan
# that backstops that contract ships with the cluster leg
# (docs/plans/67-distributed-training/UNITS.md § U7b acceptance (id-secrecy)), beside the crossing
# it protects; this driver mints/ships no id today.
GANG_REMOTE_ARTIFACT_DIR="/root/jammi-ai/.gang-artifact"
GANG_ARTIFACT_DIR="${GANG_ARTIFACT_DIR:-.gpu-pull/gpu-gang}"

# The gating groups this driver's own pass/fail rule reads `PROVE_GROUP_RC`
# markers for — `::group::` names in the heredoc below, verbatim. Unlike the
# prove lane there is no non-gating member: this lane rents two GPUs to run
# exactly one thing, so every group gates. Declared BEFORE the
# sourced-execution guard so a fixture can `source` this file and see it.
GANG_GROUPS=(gang-build gang-proof)

# F13's shared zero-test tripwire text (runpod_lib.sh's
# `_rp_zero_test_tripwire_lines`), computed here — BEFORE the
# sourced-execution guard below, same reason GANG_GROUPS is — so a fixture
# that merely `source`s this file (never executes it, never rents a pod)
# can still read the exact text the real remote heredoc splices in via
# `${zero_test_tripwire}`, below.
zero_test_tripwire="$(_rp_zero_test_tripwire_lines grc '$gang_log' "${GANG_TEST_FILTER}")"

# Verdict rule, mirroring `rp_prove_verdict` (runpod_gpu_prove.sh) minus the
# bench-cut exception this lane has no non-gating group to need:
#
#   - a ZERO ssh status may still be turned into a FAILURE by a GANG_GROUPS
#     member whose marker is missing or non-zero (never the reverse — a bare
#     zero is not trusted against its own markers);
#   - an in-suite exit (a `PROVE_EXIT=` line reached) is returned VERBATIM;
#   - a cut/hang with no `PROVE_EXIT=` (76/124) is returned as-is: with every
#     group gating, there is no case where a cut leaves a complete proof.
#
# A plain function (not inlined after the heredoc) so a fixture can drive it
# against hand-built log files without renting a pod.
rp_gang_verdict() {
  local raw_rc="$1" log="$2"
  declare -A grc_map=()
  local line
  # `|| [ -n "$line" ]`: an abrupt ssh cut leaves an unterminated final
  # line, which a bare `while read` loop drops -- `rp_prove_verdict`
  # carries the identical guard. `rp_parse_prove_marker` (runpod_lib.sh) is
  # the ONE shared grammar both drivers and the live-stream bookkeeping use.
  while IFS= read -r line || [ -n "$line" ]; do
    if rp_parse_prove_marker "$line"; then
      grc_map["$RP_PARSED_MARKER_NAME"]="$RP_PARSED_MARKER_RC"
    fi
  done < "$log"

  local has_prove_exit=0
  if grep -q '^PROVE_EXIT=' "$log" 2>/dev/null; then # tripwire-ok: grep's own stderr on a missing/unreadable log is not evidence; the `has_prove_exit=0` default is the fail-closed reading, and an unreadable log independently loses every group marker above, which the all_pass rule below then reports by name.
    has_prove_exit=1
  fi

  local all_pass=1
  local missing_or_failed=()
  local g v
  for g in "${GANG_GROUPS[@]}"; do
    v="${grc_map[$g]:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -ne 0 ]; then
      all_pass=0
      missing_or_failed+=("${g}=${v:-<missing>}")
    fi
  done

  local rc="$raw_rc"
  if [ "$raw_rc" -eq 0 ]; then
    if [ "$all_pass" -ne 1 ]; then
      echo "::error::GPU gang: ssh exited 0 but GANG_GROUPS member(s) missing or non-zero: ${missing_or_failed[*]:-<none>} — PROVE_EXIT disagrees with its own markers" >&2
      rc=1
    fi
  elif [ "$has_prove_exit" -eq 1 ]; then
    rc="$raw_rc"
  else
    if [ "$all_pass" -ne 1 ]; then
      echo "::error::GPU gang: cut/hang (raw rc=${raw_rc}) with group(s) unresolved: ${missing_or_failed[*]:-<none>}" >&2
    fi
    rc="$raw_rc"
  fi
  return "$rc"
}

# Everything below runs only when this file is EXECUTED, never when it is
# `source`d (the same guard runpod_gpu_prove.sh uses, so a fixture can reach
# `rp_gang_verdict`/`GANG_GROUPS` without renting a pod or making any network
# call merely by sourcing).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

# Sweep before renting anything: a superseded/killed runner never runs its
# EXIT trap, and the pod it had just rented is orphaned until something
# sweeps it. Two GPUs orphan at twice the rate one does.
rp_sweep

rp_init
echo "=== provisioning a ${RP_GPU_COUNT}-GPU ${GANG_DEPLOY_ARCH} pod (SXM4 first at this count) ==="
rp_deploy_arch "$GANG_DEPLOY_ARCH" || exit $?

echo "=== running the gang suite on ${RP_HOST}:${RP_PORT} ==="
LOG="$(mktemp)"
# RP_WATCH_POLL_S: fixture/diagnostic-only escape hatch for
# rp_run_remote_watched's poll interval (default 5s), exactly as the prove
# lane uses it; a real leg never sets it.
rp_run_remote_watched "" "${RP_WATCH_POLL_S:-5}" <<REMOTE | tee "$LOG"
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (ledger row 17: no cross-target-dir reuse, ~+33% wall on this image)
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
export JAMMI_GANG_ARTIFACT_DIR=${GANG_REMOTE_ARTIFACT_DIR}
# This is the one place a single-visible-device host must hard-fail rather
# than skip: JAMMI_REQUIRE_CUDA_GANG is exported by no other driver in this
# tree, so a silent skip here would let this paid two-GPU leg report a
# false green with no gang ever proven. The plain JAMMI_REQUIRE_CUDA half of
# this guard is not unique to this leg — it is exported by four other
# drivers too. gang_nccl.rs's own serial_cuda_device_or_require /
# second_cuda_device_or_require read these two atoms and panic when the
# matching var is set and this leg's own device acquisition fails; the
# single-GPU prove lane never sets either, so a one-device host there still
# skips with its reason.
export JAMMI_REQUIRE_CUDA=1
export JAMMI_REQUIRE_CUDA_GANG=1
# CUBLAS_WORKSPACE_CONFIG is deliberately NOT pinned here: spike S5 measured
# it as a kernel-SELECTION input that must merely be CONSISTENT across the
# ranks of one gang (which it is — one pod, one environment, ranks spawned
# from one process tree), not a value this lane owns. Pinning it would make
# every gang digest incomparable with every non-gang run of the same code.
echo "::group::device"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
echo "CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>}"
# Two hard assertions before anything is built, both of which mean "the pod
# is not what this leg asked for", and both of which would otherwise be
# discovered as a confusing test failure an hour later:
#   1. the device count MUST equal the requested gpuCount — a 2-rank gang on
#      a 1-GPU pod is not the topology this leg exists to prove;
#   2. nvidia-smi's own compute_cap MUST match this leg's NATIVE_COMPUTE_CAP
#      override (issue #434's failure mode, one layer earlier).
gpu_seen="\$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)"
if [ "\${gpu_seen}" != "${RP_GPU_COUNT}" ]; then
  echo "::error::device count mismatch: nvidia-smi reports \${gpu_seen} GPU(s) but this leg rented ${RP_GPU_COUNT} -- a gang cannot be proven on a pod that does not hold the ranks, refusing to build"
  exit 97
fi
compute_cap_raw="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
compute_cap_norm="\${compute_cap_raw//./}"
if [ "\${compute_cap_norm}" != "\${CUDA_COMPUTE_CAP:-}" ]; then
  echo "::error::compute_cap mismatch: nvidia-smi reports compute_cap=\${compute_cap_raw} (normalized \${compute_cap_norm}) but CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>} -- rented device does not match this leg's requested arch, refusing to build"
  exit 97
fi
echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
echo "PROVE_SHA=\$(git rev-parse HEAD)"
rc=0
# The vendored FlashAttention-2 build needs the CUTLASS submodule; a shallow
# clone does not fetch submodules.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
  || { echo "::error::CUTLASS submodule init failed (network/remote unreachable?) — refusing to attempt the flash-attn build" >&2; exit 1; }

# gang-build: compile the gang test binary. Separated from the proof group
# so a compile failure is never read as a failed gang, and so the (long)
# build has its own inactivity-watchdog group name in the log.
echo "::group::gang-build"
grc=0
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=gang-build rc=\${grc}"
echo "::endgroup::"

# gang-proof: the gang tests themselves, on both devices, single-threaded
# (two ranks already hold both GPUs; a second concurrent test would contend
# for them). NEVER-VACUOUS: a name filter matching zero tests exits 0 with
# "running 0 tests"; this group refuses to read that as a pass, and equally
# refuses a run that wrote no artifact.
echo "::group::gang-proof"
grc=0
mkdir -p "\${JAMMI_GANG_ARTIFACT_DIR}"
gang_log=/tmp/gang_proof.log
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability ${GANG_TEST_FILTER} -- --nocapture --test-threads=1 2>&1 | tee "\$gang_log"
grc=\${PIPESTATUS[0]}
${zero_test_tripwire}
if [ "\$grc" -eq 0 ] && [ -z "\$(ls -A "\${JAMMI_GANG_ARTIFACT_DIR}" 2>/dev/null)" ]; then # tripwire-ok: ls's stderr on a missing dir is not evidence; an empty result is exactly the "no artifact written" case this arm reports by name on the next line.
  echo "::error::the gang tests passed but wrote NO artifact to \${JAMMI_GANG_ARTIFACT_DIR} — a gang leg with no recorded world/collective/device/digest-pair/delta/epsilon evidence proves nothing that can be reviewed or committed" >&2
  grc=1
fi
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=gang-proof rc=\${grc}"
echo "::endgroup::"

echo "PROVE_EXIT=\${rc}"; exit \$rc
REMOTE
raw_rc="${PIPESTATUS[0]}"

rp_gang_verdict "$raw_rc" "$LOG"
rc=$?

# --- artifact retrieval, before the EXIT trap (rp_cleanup, installed by
# rp_init) tears the pod down. UNCONDITIONAL: a failed run's own artifact is
# exactly the evidence a non-reproducible gang needs to leave behind, and
# this driver is the last thing able to reach it. Mirrors
# runpod_gpu_howwell.sh's own rsync invocation rather than inventing a
# second retrieval mechanism. FATAL, not best-effort: a pull failure joins
# this leg's own rc exactly the way raw_rc already does above, never a
# second, independent exit path — a suite that passed but left no
# retrievable evidence has proven nothing that can be reviewed or committed.
mkdir -p "$GANG_ARTIFACT_DIR"
if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
  if rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    "root@${RP_HOST}:${GANG_REMOTE_ARTIFACT_DIR}/" "${GANG_ARTIFACT_DIR}/"; then
    echo "=== pulled the gang artifact -> ${GANG_ARTIFACT_DIR} ==="
  else
    pull_rc=$?
    echo "::error::gang artifact pull failed (rsync rc=${pull_rc}) -- a leg that cannot retrieve its own evidence has proven nothing reviewable or committable, and the pod is torn down on this script's own exit, so this evidence is now unrecoverable for this invocation." >&2
    [ "$rc" -eq 0 ] && rc="$pull_rc"
  fi
else
  echo "::warning::no live pod (RP_HOST/RP_PORT unset) -- skipping the artifact pull."
fi

# The NCCL id (128 opaque bytes minted by rank 0) crosses hosts ONLY
# hex-encoded; it must never reach a committed artifact or a CI log -- it
# is the capability to join this gang, not evidence of one; the driver
# never sees the id. This lane mints/ships no id today. The id file's own
# committed contract, fixed here BEFORE that mechanism exists, is that it
# rides OUTSIDE ${GANG_REMOTE_ARTIFACT_DIR}/${GANG_ARTIFACT_DIR} -- never
# inside the directory this driver pulls back and a human later commits.
# The scan that backstops this contract against a future mistake ships
# with the cluster leg (docs/plans/67-distributed-training/UNITS.md § U7b acceptance (id-secrecy)),
# beside the id-ship crossing it protects.
# --- end artifact retrieval ---

rm -f "$LOG"

echo "=== GPU gang leg exit=${rc} (raw=${raw_rc}) ==="
exit "$rc"

fi # end sourced-execution guard
