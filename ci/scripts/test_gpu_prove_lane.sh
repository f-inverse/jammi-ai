#!/usr/bin/env bash
# GPU-prove-lane fixture suite (esc-080/esc-082/esc-083). Mocks-only, no
# network, no GPU, no RunPod account: drives the REAL `rp_run_remote_watched`
# (runpod_lib.sh) and the REAL `rp_prove_verdict`/`PROVE_GROUPS`
# (runpod_gpu_prove.sh, `source`d rather than executed — see that file's own
# "sourced-execution guard" comment, which skips the deploy-a-pod flow when
# sourced) against a STUBBED `ssh`.
#
# Cases (contract esc-080..083, D6):
#   F1  inactivity kill -> 76, named group, `[name:rc…]` list.
#   F2  budget cut (ssh 124, no PROVE_EXIT) -> 124, `[name:rc…]` list;
#       N=0 groups -> `[]`.
#   F3  in-suite exits pass through VERBATIM: 255; in-suite 124 WITH
#       `PROVE_EXIT=124` (no BUDGET line); in-suite 76; an early `exit 97`
#       with no markers at all.
#   F4  bench is non-gating: bench exit 1 with every proof group rc=0 -> 0 +
#       `BENCH_EXIT=1`; a served-proof failure followed by a HUNG bench ->
#       FAIL; a served-proof failure with a normal exit -> FAIL; a cut
#       INSIDE a proof group -> FAIL; a cut inside bench with every proof
#       group already rc=0 -> 0 + `::warning::`; a marker written after the
#       final poll tick still credits.
#   F7  `rp_cleanup` (the `trap ... EXIT` runpod_lib.sh installs at source
#       time) fires on the 76 path and on the exit-0 bench-cut path.
#   Plus: the `PROVE_GROUPS` closure fixture (every `::group::` name in the
#   script minus `device`/`bench` equals `PROVE_GROUPS`, exactly);
#   partial-line-at-poll-boundary (a marker split across two 5s polls is
#   still parsed); `PROVE_EXIT` disagreeing with its own markers -> FAIL.
#
# F5 (`check_flash_attn_closure.py --self-test`) and F6
# (`check_gpu_prove_timings.py --self-test`) are NOT duplicated here — they
# are already required, standalone steps in `ci.yml`'s guard matrix; this
# suite owns only the driver/watchdog mechanism itself.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
PROVE_SH="$DIR/runpod_gpu_prove.sh"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

# --------------------------------------------------------------------------
# Source the real driver (no network call merely from sourcing: the
# sourced-execution guard skips rp_sweep/rp_init/rp_deploy_arch/the heredoc).
# --------------------------------------------------------------------------
export RUNPOD_API_KEY="test-dummy-key"
export GPU_PROVE_ARCH="sm_80"
# shellcheck source=ci/scripts/runpod_gpu_prove.sh
source "$PROVE_SH"

if declare -f rp_prove_verdict >/dev/null && declare -f rp_run_remote_watched >/dev/null; then
  ok "sourcing runpod_gpu_prove.sh (no network) defines rp_prove_verdict and (via runpod_lib.sh) rp_run_remote_watched"
else
  bad "sourcing runpod_gpu_prove.sh did not define the expected functions"
fi

# --------------------------------------------------------------------------
# PROVE_GROUPS closure: the script's own `::group::` names, minus device and
# bench, must equal PROVE_GROUPS exactly.
# --------------------------------------------------------------------------
mapfile -t script_groups < <(grep -oE '::group::[a-z0-9-]+' "$PROVE_SH" | sed 's/::group:://' | sort -u)
mapfile -t declared_groups < <(printf '%s\n' "${PROVE_GROUPS[@]}" | sort -u)
non_members=()
for g in "${script_groups[@]}"; do
  [ "$g" = "device" ] && continue
  [ "$g" = "bench" ] && continue
  non_members+=("$g")
done
mapfile -t non_members_sorted < <(printf '%s\n' "${non_members[@]}" | sort -u)
if [ "${#non_members_sorted[@]}" -eq "${#declared_groups[@]}" ] && [ "$(printf '%s\n' "${non_members_sorted[@]}")" = "$(printf '%s\n' "${declared_groups[@]}")" ]; then
  ok "PROVE_GROUPS closure: {::group:: names} - {device, bench} == PROVE_GROUPS exactly (${#declared_groups[@]} members)"
else
  bad "PROVE_GROUPS closure mismatch: script non-member groups=[${non_members_sorted[*]}] vs PROVE_GROUPS=[${declared_groups[*]}]"
fi

# --------------------------------------------------------------------------
# helper: build a log fixture from a marker spec, run rp_prove_verdict, and
# report the resulting rc plus stderr.
# --------------------------------------------------------------------------
all_six_pass_log() {
  local log="$1"
  {
    for g in "${PROVE_GROUPS[@]}"; do
      echo "PROVE_GROUP_RC name=${g} rc=0"
    done
  } > "$log"
}

# ============================================================================
# F1: rp_run_remote_watched itself — inactivity kill -> 76, named group,
# [name:rc...] list (real function, stubbed ssh).
# ============================================================================
ssh() {
  cat >/dev/null
  echo "::group::served-client-server-proof"
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  sleep 30
  return 0
}
export -f ssh
RP_SSHO=(); RP_PORT=22; RP_HOST=localhost
# A fixture-local threshold passed as a plain function ARGUMENT (never a
# committed `RP_INACTIVITY=<n>` assignment, which check_gpu_prove_timings.
# py's R1 setter-predicate scan would -- correctly -- flag as a second
# source of truth for the real default) -- see rp_run_remote_watched's own
# `$1` doc.
out="$(rp_run_remote_watched 3 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] \
  && echo "$out" | grep -q 'NO PROGRESS' \
  && echo "$out" | grep -q 'group "served-client-server-proof"' \
  && echo "$out" | grep -q 'groups: \[capability-surface-build:0\]'; then
  ok "F1: inactivity kill -> 76, names the open group, lists [name:rc...]"
else
  bad "F1: expected 76 + named group + list; got rc=$rc out=$out"
fi

# ============================================================================
# F2: budget cut (ssh 124, no PROVE_EXIT) -> 124 with list; N=0 -> [].
# ============================================================================
ssh() {
  cat >/dev/null
  echo "::group::kernels-cuda"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'BUDGET' && echo "$out" | grep -q 'groups: \[kernels-default:0\]'; then
  ok "F2: budget cut -> 124, lists [name:rc...]"
else
  bad "F2: expected 124 + list; got rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "no groups at all"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'groups: \[\]'; then
  ok "F2: N=0 groups renders as []"
else
  bad "F2: expected groups: []; got rc=$rc out=$out"
fi

# ============================================================================
# F3: in-suite exits pass through VERBATIM.
# ============================================================================
ssh() { cat >/dev/null; return 255; }
export -f ssh
rp_run_remote_watched <<< "noop" >/dev/null 2>&1
[ $? -eq 255 ] && ok "F3: ssh 255 passes through verbatim" || bad "F3: expected 255"

ssh() {
  cat >/dev/null
  echo "::group::bench"
  echo "PROVE_EXIT=124"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && ! echo "$out" | grep -q 'BUDGET'; then
  ok "F3: in-suite 124 (PROVE_EXIT=124 present) -> 124 verbatim, no BUDGET line"
else
  bad "F3: expected 124 with no BUDGET line; got rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "PROVE_EXIT=76"
  return 76
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && ! echo "$out" | grep -q 'NO PROGRESS'; then
  ok "F3: in-suite 76 (PROVE_EXIT=76 present) passes through unchanged, no watchdog diagnostic"
else
  bad "F3: expected in-suite 76 unchanged; got rc=$rc out=$out"
fi

ssh() { cat >/dev/null; return 97; }
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 97 ] && ! echo "$out" | grep -qE 'NO PROGRESS|BUDGET'; then
  ok "F3: an early exit 97 with no markers passes through verbatim"
else
  bad "F3: expected 97 verbatim with no diagnostic; got rc=$rc out=$out"
fi

# ============================================================================
# F4: bench is non-gating (driven through rp_prove_verdict directly).
# ============================================================================
log="$SANDBOX/f4a.log"
{
  for g in "${PROVE_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
  echo "BENCH_EXIT=1"
  echo "PROVE_GROUP_RC name=bench rc=1"
  echo "PROVE_EXIT=0"
} > "$log"
rp_prove_verdict 0 "$log"
rc=$?
[ "$rc" -eq 0 ] && ok "F4: bench exit 1 with every proof group rc=0 -> 0 (BENCH_EXIT=1 is non-gating)" \
  || bad "F4: expected 0, got $rc"

log="$SANDBOX/f4b.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  echo "PROVE_GROUP_RC name=capability-surface-proof rc=0"
  echo "PROVE_GROUP_RC name=served-client-server-proof rc=1"
  echo "PROVE_GROUP_RC name=engine-core-sweep rc=0"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  echo "PROVE_GROUP_RC name=kernels-cuda rc=0"
  # bench then hangs -- no BENCH_EXIT, no PROVE_EXIT line at all.
} > "$log"
rp_prove_verdict 76 "$log"   # the watchdog would have killed the hung bench
rc=$?
[ "$rc" -eq 76 ] && ok "F4: served-proof failure (rc=1) then a hung bench -> FAIL (not the bench exception, since a REAL proof group failed)" \
  || bad "F4: expected 76 (FAIL), got $rc"

log="$SANDBOX/f4c.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  echo "PROVE_GROUP_RC name=capability-surface-proof rc=0"
  echo "PROVE_GROUP_RC name=served-client-server-proof rc=1"
  echo "PROVE_GROUP_RC name=engine-core-sweep rc=0"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  echo "PROVE_GROUP_RC name=kernels-cuda rc=0"
  echo "BENCH_EXIT=0"
  echo "PROVE_GROUP_RC name=bench rc=0"
  echo "PROVE_EXIT=1"
} > "$log"
rp_prove_verdict 1 "$log"
rc=$?
[ "$rc" -eq 1 ] && ok "F4: served-proof failure (rc=1) with a NORMAL exit -> FAIL, verbatim" \
  || bad "F4: expected 1, got $rc"

log="$SANDBOX/f4d.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  # cut happens INSIDE capability-surface-proof -- its own marker never lands.
} > "$log"
rp_prove_verdict 124 "$log"
rc=$?
[ "$rc" -eq 124 ] && ok "F4: a cut INSIDE a proof group (marker missing) -> FAIL" \
  || bad "F4: expected 124, got $rc"

log="$SANDBOX/f4e.log"
all_six_pass_log "$log"
# cut happens inside bench -- no BENCH_EXIT, no PROVE_EXIT.
rp_prove_verdict 76 "$log"
rc=$?
out="$(rp_prove_verdict 76 "$log" 2>&1 >/dev/null)"
if [ "$rc" -eq 0 ] && echo "$out" | grep -q '::warning::'; then
  ok "F4: a cut INSIDE bench with every proof group already rc=0 -> 0 + ::warning::"
else
  bad "F4: expected 0 + warning, got rc=$rc out=$out"
fi

# A marker written after the final poll tick still credits: rp_prove_verdict
# only ever reads the FINAL drained log file, so any marker present in it —
# regardless of when it landed — is honored. Simulate by writing the LAST
# group's marker only after constructing the rest of the log (order in the
# file, not wall-clock time, is what a static read sees; the wall-clock
# guarantee itself is exercised by rp_run_remote_watched's own final-drain
# step, proven directly below).
log="$SANDBOX/f4f.log"
: > "$log"
for g in "${PROVE_GROUPS[@]:0:5}"; do echo "PROVE_GROUP_RC name=${g} rc=0" >> "$log"; done
echo "PROVE_GROUP_RC name=${PROVE_GROUPS[5]} rc=0" >> "$log"   # the "late" marker
echo "PROVE_EXIT=0" >> "$log"
rp_prove_verdict 0 "$log"
[ $? -eq 0 ] && ok "F4: a marker landing last in the log still credits" || bad "F4: late marker did not credit"

# ============================================================================
# partial-line-at-poll-boundary: a marker split across two 5s polls is still
# reassembled by rp_run_remote_watched's own carry logic.
# ============================================================================
ssh() {
  cat >/dev/null
  printf 'PROVE_GROUP_RC name=engine'
  sleep 6
  printf -- '-core-sweep rc=0\n'
  sleep 30
  return 0
}
export -f ssh
out="$(rp_run_remote_watched 6 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && echo "$out" | grep -q 'groups: \[engine-core-sweep:0\]'; then
  ok "partial-line-at-poll-boundary: a marker split across two polls is reassembled correctly"
else
  bad "partial-line-at-poll-boundary: expected engine-core-sweep:0 in the group list; got rc=$rc out=$out"
fi

# ============================================================================
# PROVE_EXIT disagreeing with its own markers -> FAIL (never trusted blindly).
# ============================================================================
log="$SANDBOX/disagree.log"
all_six_pass_log "$log"
# Corrupt ONE marker after the fact so the file disagrees with a PROVE_EXIT=0.
sed -i.bak 's/name=kernels-cuda rc=0/name=kernels-cuda rc=1/' "$log"
echo "PROVE_EXIT=0" >> "$log"
rp_prove_verdict 0 "$log"
rc=$?
[ "$rc" -ne 0 ] && ok "PROVE_EXIT disagreeing with its own markers -> FAIL (rc=$rc)" \
  || bad "expected a nonzero rc when PROVE_EXIT=0 disagrees with a failed marker"

# ============================================================================
# F7: rp_cleanup (runpod_lib.sh's own `trap ... EXIT`, installed at source
# time) fires on the 76 path and on the exit-0 bench-cut path. Run as
# SEPARATE bash processes (each a fresh trap-armed process) so the OS-level
# EXIT-trap guarantee is what is actually being exercised, not just a
# function call in this suite's own process.
# ============================================================================
f7_case() {
  # $1 = "watchdog" | "bench-cut"
  local mode="$1"
  local out
  out="$(RUNPOD_API_KEY=test-dummy-key bash -c '
    set -uo pipefail
    DIR="'"$DIR"'"
    source "$DIR/runpod_lib.sh"
    rp_init
    echo "RP_WORK=$RP_WORK"
    if [ "'"$mode"'" = "watchdog" ]; then
      exit 76
    else
      echo "::warning::bench cut/hung after every proof group already passed (simulated)"
      exit 0
    fi
  ' 2>&1)"
  local work_dir
  work_dir="$(echo "$out" | grep '^RP_WORK=' | head -1 | cut -d= -f2-)"
  if [ -n "$work_dir" ] && [ ! -d "$work_dir" ]; then
    ok "F7 ($mode): rp_cleanup fired -- the temp RP_WORK dir ($work_dir) was removed on exit"
  else
    bad "F7 ($mode): rp_cleanup did not appear to fire (RP_WORK=$work_dir still present, or unreported); out=$out"
  fi
}
f7_case "watchdog"
f7_case "bench-cut"

echo
echo "gpu-prove-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
