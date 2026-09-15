#!/usr/bin/env bash
# GPU-cluster-lane fixture suite (plan #500 U7b, commit c3). Mocks-only: no
# network, no GPU, no RunPod account, no cluster. Shaped on
# `test_gpu_gang_lane.sh` (which owns the same job for the pod leg) and
# driving the REAL objects rather than paraphrases of them:
# `runpod_gpu_cluster.sh` is `source`d, not executed — its own
# sourced-execution guard skips the create/wait/heredoc/pull flow when
# sourced — so `rp_cluster_verdict`, `rp_cluster_rank_verdict`,
# `CLUSTER_GROUPS`, `_rpc_*` and the lane's own env pins are the committed
# ones.
#
# Cases:
#   G0  sourcing the driver invokes no curl/ssh/scp/rsync/runpodctl — the
#       "mocks-only" claim measured through a PATH shim, not asserted in a
#       comment.
#   G1  `rp_cluster_rank_verdict` (per rank) and `rp_cluster_verdict`
#       (combining both ranks) over every rc arm.
#   G2  the CLUSTER_GROUPS closure: every `::group::` name in the driver
#       minus `device` equals CLUSTER_GROUPS exactly.
#   G3  the shared zero-test tripwire (F13): the SAME `_rp_zero_test_
#       tripwire_lines` text `runpod_gpu_gang.sh` shares, spliced into
#       `_rpc_remote_script`'s own per-rank text.
#   F2/F3  the launch-time read-back refusal (`_rpc_check_readback`): args
#       mismatch, ssh.direct null with no overlay ip, ssh.direct null with a
#       usable overlay ip (still OK — the no-public-port fallback), a
#       parse failure.
#   F11 the 128-byte id-file-ready gate (`_rpc_id_file_ready`): 127/128/129
#       bytes, and a missing file.
#   A5  the `ens1` log assertion (`_rpc_ens1_seen`).
#   G5  the post-run artifact retrieval: a failed pull JOINS the leg's own
#       rc (never a silently-warned second exit path).
#   F5  the id-secrecy scan invocation (`_rpc_run_id_secrecy_scan`) against
#       fixtures: a planted raw/hex/base64 id -> FAIL; an archive under the
#       pulled dir -> 2; clean -> pass, staging deleted (--delete-staging).
#   G4  the cost bound is what the MECHANISM produces: $1.908/GPU/h x 2 GPUs
#       = $3.816/h; (i) = RP_TTL_HOURS x rate; (ii) = (RP_TTL_HOURS + 6) x
#       rate (gpu-reap.yml's 6-hourly sweep).
#   G7  NO `schedule:` key exists anywhere in the committed `gpu-cluster.yml`.
#
# Run: bash ci/scripts/test_gpu_cluster_lane.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
CLUSTER_SH="$DIR/runpod_gpu_cluster.sh"
CLUSTER_YML="$REPO_ROOT/.github/workflows/gpu-cluster.yml"
DEV_GPU_MD="$REPO_ROOT/docs/maintainer/dev-gpu.md"
PROVE_ONCE_PY="$DIR/check_gpu_prove_once.py"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

SANDBOX="$(mktemp -d)"
NETPROBE_LOG="$SANDBOX/netprobe.log"
trap 'rm -rf "$SANDBOX"' EXIT

# --------------------------------------------------------------------------
# Source the real driver. Sourcing makes NO network call (the guard skips
# the create/wait/heredoc/pull orchestration); runpod_lib.sh validates its
# own env at source time, hence the dummy key.
# --------------------------------------------------------------------------
export RUNPOD_API_KEY="test-dummy-key"

NETPROBE_BIN="$SANDBOX/netprobe-bin"
mkdir -p "$NETPROBE_BIN"
for tool in curl ssh scp rsync runpodctl; do
  cat >"$NETPROBE_BIN/$tool" <<PROBE
#!/usr/bin/env bash
printf '%s %s\n' "$tool" "\$*" >>"$NETPROBE_LOG"
exit 1
PROBE
  chmod +x "$NETPROBE_BIN/$tool"
done
: >"$NETPROBE_LOG"
PATH="$NETPROBE_BIN:$PATH"

# shellcheck source=ci/scripts/runpod_gpu_cluster.sh
source "$CLUSTER_SH"

netprobe_calls="$(grep -c . "$NETPROBE_LOG" 2>/dev/null || true)" # tripwire-ok: grep -c on an EMPTY log legitimately exits 1; zero IS the pass condition, asserted on the next line.
if [ "${netprobe_calls:-0}" -eq 0 ]; then
  ok "G0: sourcing runpod_gpu_cluster.sh invoked NO curl/ssh/scp/rsync/runpodctl (counted through a PATH shim, not assumed) — the sourced-execution guard really does cover the create/wait/pull path"
else
  bad "G0: sourcing runpod_gpu_cluster.sh invoked ${netprobe_calls} external call(s): $(tr '\n' '; ' <"$NETPROBE_LOG") — the sourced-execution guard no longer covers the rent path, and this suite would spend money"
fi

if declare -f rp_cluster_rank_verdict >/dev/null \
  && declare -f rp_cluster_verdict >/dev/null \
  && declare -f _rpc_remote_script >/dev/null \
  && declare -f _rpc_check_readback >/dev/null \
  && declare -f _rpc_id_file_ready >/dev/null \
  && declare -f _rpc_ens1_seen >/dev/null \
  && declare -f _rpc_run_id_secrecy_scan >/dev/null \
  && declare -f _rpc_assemble_gang_artifact >/dev/null \
  && declare -f _rpc_self_remove_status >/dev/null \
  && declare -f _rpc_cleanup_cluster >/dev/null; then
  ok "G0: sourcing runpod_gpu_cluster.sh (no network) defines every _rpc_* helper and both verdict functions"
else
  bad "G0: sourcing runpod_gpu_cluster.sh did not define the expected functions"
fi

# ============================================================================
# F2/F3(c): _rpc_cleanup_cluster — chains rp_cleanup (F2) and joins a failed
# rp_cluster_delete into the exit status (F3c, naming the cluster LEAKED).
# Drives the REAL function (moved out of the executed-only guard specifically
# so it is sourceable), mocking only _rp_rest/rp_cluster_delete/rp_cleanup.
# ============================================================================
run_cleanup_in_subshell() { # $1=cluster_id (or "" for none) $2=pending_rc
  # Runs in a real SUBSHELL PROCESS (not `( ... )` capture alone) so
  # `_rpc_cleanup_cluster`'s own `exit "$rc"` terminates THAT process, never
  # this suite's own shell — exactly how a real EXIT trap fires.
  bash -c '
    source "'"$CLUSTER_SH"'"
    cluster_id="$1"
    rp_cleanup_called_marker="$2"
    _rp_rest() { printf "%s\n%s" "$MOCK_SELF_REMOVE_STATUS" "{}"; }
    rp_cluster_delete() { [ "$MOCK_DELETE_OK" = "1" ]; }
    rp_cleanup() { : > "$rp_cleanup_called_marker"; }
    ( exit "$3" )   # sets $? to the PENDING exit status _rpc_cleanup_cluster reads first.
    _rpc_cleanup_cluster
  ' _ "$1" "$SANDBOX/rp-cleanup-called" "$2"
}

# (a) self-remove already "ok" (404) -> rp_cleanup chained, exit stays the
# pending rc, rp_cluster_delete never called.
rm -f "$SANDBOX/rp-cleanup-called"
MOCK_SELF_REMOVE_STATUS="404" MOCK_DELETE_OK="1" \
  run_cleanup_in_subshell "cl-test-1" 0 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "F2: self-remove ok (404) -> pending rc (0) preserved AND rp_cleanup still chained (rm -rf \$RP_WORK ran)"
else
  bad "F2: self-remove-ok case: expected rc=0 and rp_cleanup called (rc=$rc, marker=$([ -f "$SANDBOX/rp-cleanup-called" ] && echo yes || echo no))"
fi

# (b) self-remove refused, delete SUCCEEDS -> rp_cleanup still chained, exit
# stays the pending rc (a successful driver-initiated delete is not itself a
# failure).
rm -f "$SANDBOX/rp-cleanup-called"
MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="1" \
  run_cleanup_in_subshell "cl-test-2" 0 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "F2: self-remove refused, delete succeeds -> pending rc (0) preserved AND rp_cleanup chained"
else
  bad "F2: self-remove-refused-delete-ok case: expected rc=0 and rp_cleanup called (rc=$rc)"
fi

# (c) self-remove refused, delete FAILS -> F3(c): the cluster is LEAKED —
# the exit status is joined to non-zero even though the pending rc was 0 —
# and rp_cleanup is STILL chained (F2 holds even on this arm).
rm -f "$SANDBOX/rp-cleanup-called"
out="$(MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="0" \
  run_cleanup_in_subshell "cl-test-3" 0 2>&1)"
rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "LEAKED cluster cl-test-3"; then
  ok "F3(c): a failed rp_cluster_delete on exit is LEAKED, naming the cluster id, and joins the exit status non-zero"
else
  bad "F3(c): expected a non-zero exit naming 'LEAKED cluster cl-test-3' (rc=$rc): $out"
fi
if [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "F2: rp_cleanup is STILL chained even when the cluster delete itself failed"
else
  bad "F2: rp_cleanup must be chained on EVERY arm, including a failed cluster delete"
fi

# (d) a REAL failure already pending (rc=76) with a failed delete -> the
# ORIGINAL failure code is preserved, never overwritten by the generic "1"
# the delete failure alone would have joined.
rm -f "$SANDBOX/rp-cleanup-called"
out="$(MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="0" \
  run_cleanup_in_subshell "cl-test-4" 76 2>&1)"
rc=$?
if [ "$rc" -eq 76 ]; then
  ok "F3(c): a pre-existing failure (76) is preserved verbatim, not overwritten by the delete-failure join"
else
  bad "F3(c): expected the pre-existing rc=76 to survive a failed delete (got rc=$rc): $out"
fi

# (e) no cluster_id at all (cleanup fires before create ever ran) -> no
# self-remove check, no delete attempt, rp_cleanup still chained, pending
# rc preserved.
rm -f "$SANDBOX/rp-cleanup-called"
run_cleanup_in_subshell "" 75 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 75 ] && [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "F2: no cluster_id at all -> rp_cleanup still chained, pending rc (75) preserved, no delete attempted"
else
  bad "F2: empty-cluster_id case: expected rc=75 and rp_cleanup called (rc=$rc)"
fi

# ============================================================================
# G2: CLUSTER_GROUPS closure — {::group:: names} - {device} == CLUSTER_GROUPS.
# ============================================================================
mapfile -t script_groups < <(grep -oE '::group::[a-z0-9-]+' "$CLUSTER_SH" | sed 's/::group:://' | sort -u)
mapfile -t declared_groups < <(printf '%s\n' "${CLUSTER_GROUPS[@]}" | sort -u)
non_members=()
for g in "${script_groups[@]}"; do
  [ "$g" = "device" ] && continue
  non_members+=("$g")
done
mapfile -t non_members_sorted < <(printf '%s\n' "${non_members[@]}" | sort -u)
if [ "${#non_members_sorted[@]}" -eq "${#declared_groups[@]}" ] \
  && [ "$(printf '%s\n' "${non_members_sorted[@]}")" = "$(printf '%s\n' "${declared_groups[@]}")" ]; then
  ok "G2: CLUSTER_GROUPS closure: {::group:: names} - {device} == CLUSTER_GROUPS exactly (${#declared_groups[@]} members)"
else
  bad "G2: CLUSTER_GROUPS closure mismatch: script non-member groups=[${non_members_sorted[*]}] vs CLUSTER_GROUPS=[${declared_groups[*]}]"
fi

# ============================================================================
# G1: rp_cluster_rank_verdict (per rank) over every rc arm.
# ============================================================================
all_pass_log() {
  local log="$1"
  {
    for g in "${CLUSTER_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
    echo "PROVE_EXIT=0"
  } > "$log"
}

drive_rank() { # $1=raw_rc $2=log -> prints "rc|stderr"
  local out rc
  out="$(rp_cluster_rank_verdict "$1" "$2" 2>&1 >/dev/null)"
  rp_cluster_rank_verdict "$1" "$2" >/dev/null 2>&1
  rc=$?
  printf '%s|%s' "$rc" "$out"
}

log="$SANDBOX/g1-clean.log"
all_pass_log "$log"
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 0 ] && [ -z "$err" ]; then
  ok "G1: every gating group rc=0 with ssh 0 -> 0, no diagnostic"
else
  bad "G1: expected a silent 0; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-missing.log"
{
  echo "PROVE_GROUP_RC name=cluster-build rc=0"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"cluster-proof=<missing>"* ]]; then
  ok "G1: ssh 0 with a MISSING group marker -> 1, naming the group as <missing>"
else
  bad "G1: expected 1 + cluster-proof=<missing>; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-nonzero.log"
{
  echo "PROVE_GROUP_RC name=cluster-build rc=0"
  echo "PROVE_GROUP_RC name=cluster-proof rc=3"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"cluster-proof=3"* ]]; then
  ok "G1: ssh 0 with a NON-ZERO group marker -> 1 (a bare zero is never trusted against its own markers)"
else
  bad "G1: expected 1 + cluster-proof=3; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-hang.log"
echo "PROVE_GROUP_RC name=cluster-build rc=0" > "$log"
res="$(drive_rank 76 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 76 ] && [[ "$err" == *"cut/hang (raw rc=76)"* ]] && [[ "$err" == *"cluster-proof=<missing>"* ]]; then
  ok "G1: inactivity kill (76, no PROVE_EXIT) -> 76, naming the unresolved group"
else
  bad "G1: expected 76 + cut/hang naming cluster-proof; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-cut.log"
echo "PROVE_GROUP_RC name=cluster-build rc=0" > "$log"
res="$(drive_rank 124 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 124 ] && [[ "$err" == *"cut/hang (raw rc=124)"* ]]; then
  ok "G1: budget cut (124, no PROVE_EXIT) -> 124, naming the unresolved group"
else
  bad "G1: expected 124 + cut/hang; got rc=$rc err=$err"
fi

for insuite in 1 97 77 255; do
  log="$SANDBOX/g1-insuite-${insuite}.log"
  {
    for g in "${CLUSTER_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
    echo "PROVE_EXIT=${insuite}"
  } > "$log"
  res="$(drive_rank "$insuite" "$log")"; rc="${res%%|*}"
  if [ "$rc" -eq "$insuite" ]; then
    ok "G1: an in-suite exit ${insuite} (PROVE_EXIT present) is returned VERBATIM"
  else
    bad "G1: expected in-suite exit ${insuite} returned verbatim; got rc=$rc"
  fi
done

log="$SANDBOX/g1-unreadable.log"
# no such file
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"
if [ "$rc" -eq 1 ]; then
  ok "G1: an unreadable log (every marker lost) with ssh 0 -> 1, fail-closed"
else
  bad "G1: expected rc=1 on an unreadable log; got rc=$rc"
fi

# rp_cluster_verdict: combining both ranks.
if rp_cluster_verdict 0 0; then
  ok "G1: rp_cluster_verdict(0,0) -> 0 (both ranks clean)"
else
  bad "G1: rp_cluster_verdict(0,0) expected 0, got $?"
fi

rp_cluster_verdict 75 0; rc=$?
[ "$rc" -eq 75 ] && ok "G1: rp_cluster_verdict(75,0) -> 75 (a special code on EITHER rank wins over a plain 0)" \
  || bad "G1: rp_cluster_verdict(75,0) expected 75, got $rc"

rp_cluster_verdict 0 76; rc=$?
[ "$rc" -eq 76 ] && ok "G1: rp_cluster_verdict(0,76) -> 76 (the other rank's special code still wins)" \
  || bad "G1: rp_cluster_verdict(0,76) expected 76, got $rc"

rp_cluster_verdict 77 97; rc=$?
[ "$rc" -eq 77 ] && ok "G1: rp_cluster_verdict(77,97) -> 77 (priority order: 75,76,77,97,124)" \
  || bad "G1: rp_cluster_verdict(77,97) expected 77 (priority order), got $rc"

rp_cluster_verdict 1 0; rc=$?
[ "$rc" -eq 1 ] && ok "G1: rp_cluster_verdict(1,0) -> 1 (a plain nonzero on either rank still fails the leg)" \
  || bad "G1: rp_cluster_verdict(1,0) expected 1, got $rc"

# ============================================================================
# G3: the shared zero-test tripwire (F13) — the SAME text runpod_gpu_gang.sh
# shares, spliced into _rpc_remote_script's own per-rank text.
# ============================================================================
rank0_text="$(_rpc_remote_script 0)"
rank1_text="$(_rpc_remote_script 1)"
if [[ "$rank0_text" == *'if grep -q "running 0 tests" "$rank_log"; then'* ]] \
  && [[ "$rank1_text" == *'if grep -q "running 0 tests" "$rank_log"; then'* ]]; then
  ok "G3: both ranks' remote text carries the shared zero-test tripwire verbatim"
else
  bad "G3: the shared zero-test tripwire is missing from one or both ranks' remote text"
fi
if [[ "$rank0_text" == *"matched ZERO tests"* ]]; then
  ok "G3: the tripwire's own message names 'matched ZERO tests'"
else
  bad "G3: expected 'matched ZERO tests' in the rank's remote text"
fi
if [[ "$rank0_text" == *"JAMMI_GANG_TWO_HOSTS_RANK=0"* ]] && [[ "$rank1_text" == *"JAMMI_GANG_TWO_HOSTS_RANK=1"* ]]; then
  ok "G3: each rank's text pins its OWN JAMMI_GANG_TWO_HOSTS_RANK"
else
  bad "G3: rank env pins are wrong: rank0=${rank0_text}, rank1=${rank1_text}"
fi
if [[ "$rank0_text" == *"JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT}"* ]]; then
  ok "G3: world derives from RP_CLUSTER_POD_COUNT (${RP_CLUSTER_POD_COUNT}), never a second literal"
else
  bad "G3: expected JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT} in the rank text"
fi

# ============================================================================
# F2/F3: the launch-time read-back refusal (_rpc_check_readback).
# ============================================================================
SETUP_TEXT="the-shared-entrypoint-text"

readback_body() {
  local rank0_args="$1" rank0_ssh="$2" rank0_ip="$3" rank1_args="$4" rank1_ssh="$5" rank1_ip="$6"
  python3 -c '
import json, sys
r0a, r0s, r0i, r1a, r1s, r1i = sys.argv[1:7]
def pod(pid, rank, args, ssh_json, ip):
    p = {"id": pid, "args": args, "cluster": {"rank": rank, "ip": ip or None}}
    p["ssh"] = json.loads(ssh_json) if ssh_json else {}
    return p
body = {"pods": [
    pod("primary", 0, r0a, r0s, r0i),
    pod("member", 1, r1a, r1s, r1i),
]}
print(json.dumps(body))
' "$rank0_args" "$rank0_ssh" "$rank0_ip" "$rank1_args" "$rank1_ssh" "$rank1_ip"
}

body="$(readback_body "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"5.6.7.8","port":22}}' "10.0.0.3")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if printf '%s' "$out" | grep -q "^primary 0 READBACK_OK 10.0.0.2 1.2.3.4 22$" \
  && printf '%s' "$out" | grep -q "^member 1 READBACK_OK 10.0.0.3 5.6.7.8 22$"; then
  ok "F2/F3: both members with matching args + direct ssh -> READBACK_OK, host/port carried"
else
  bad "F2/F3: expected both READBACK_OK with host/port; got: $out"
fi

body="$(readback_body "bash -c 'not the setup'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"5.6.7.8","port":22}}' "10.0.0.3")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if printf '%s' "$out" | grep -q "READBACK_ARGS_MISMATCH"; then
  ok "F2: a member whose args do not echo the setup text -> READBACK_ARGS_MISMATCH"
else
  bad "F2: expected READBACK_ARGS_MISMATCH; got: $out"
fi

body="$(readback_body "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{}' "")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if printf '%s' "$out" | grep -q "READBACK_NO_SSH_PATH"; then
  ok "F3: a member with NO ssh.direct and NO overlay ip -> READBACK_NO_SSH_PATH"
else
  bad "F3: expected READBACK_NO_SSH_PATH; got: $out"
fi

body="$(readback_body "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{}' "10.0.0.3")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if printf '%s' "$out" | grep -q "^member 1 READBACK_OK 10.0.0.3 - -$"; then
  ok "F3: a member with NO ssh.direct but a usable overlay ip -> still READBACK_OK (the no-public-port fallback)"
else
  bad "F3: expected member READBACK_OK with dash host/port (proxy fallback); got: $out"
fi

out="$(_rpc_check_readback "not json at all" "$SETUP_TEXT" 2>/dev/null)"; rc=$?
if [ "$rc" -eq 2 ] && printf '%s' "$out" | grep -q "PARSE_ERROR"; then
  ok "F2/F3: an unparseable pods response -> rc=2, named PARSE_ERROR"
else
  bad "F2/F3: expected rc=2 + PARSE_ERROR on unparseable body; got rc=$rc out=$out"
fi

# ============================================================================
# F11: the 128-byte id-file-ready gate (_rpc_id_file_ready).
# ============================================================================
for size in 127 128 129; do
  f="$SANDBOX/id-${size}"
  head -c "$size" /dev/zero > "$f"
  if _rpc_id_file_ready "$f"; then got_ok=1; else got_ok=0; fi
  if [ "$size" -eq 128 ] && [ "$got_ok" -eq 1 ]; then
    ok "F11: a 128-byte id file is ready"
  elif [ "$size" -ne 128 ] && [ "$got_ok" -eq 0 ]; then
    ok "F11: a ${size}-byte id file is refused (not exactly 128)"
  else
    bad "F11: unexpected result for a ${size}-byte file (ready=${got_ok})"
  fi
done
if _rpc_id_file_ready "$SANDBOX/does-not-exist"; then
  bad "F11: a missing id file must never read as ready"
else
  ok "F11: a missing id file is refused"
fi

# ============================================================================
# A5: the ens1 log assertion (_rpc_ens1_seen).
# ============================================================================
good_log="$SANDBOX/ens1-good.log"
printf 'some noise\nNCCL INFO NET/Socket : Using [0]eth0:1.2.3.4<0> [1]ens1:10.0.0.2<0>\nmore noise\n' > "$good_log"
if _rpc_ens1_seen "$good_log"; then
  ok "A5: a log naming ens1 for the NCCL NET transport is recognized"
else
  bad "A5: expected _rpc_ens1_seen to recognize a genuine ens1 NET/Socket line"
fi

bad_log="$SANDBOX/ens1-missing.log"
printf 'some noise\nNCCL INFO NET/Socket : Using [0]eth0:1.2.3.4<0>\nmore noise\n' > "$bad_log"
if _rpc_ens1_seen "$bad_log"; then
  bad "A5: a log with NO ens1 NET/Socket line must never read as seen"
else
  ok "A5: a log with no ens1 NET line is refused"
fi

# ============================================================================
# G5: the post-run artifact retrieval — a failed pull JOINS the leg's rc.
# ============================================================================
mkdir -p "$SANDBOX/gpu-pull/gpu-cluster"
if grep -qF 'echo "::error::artifact pull failed (rsync rc=${pull_rc})' "$CLUSTER_SH"; then
  ok "G5: the driver's own pull-failure line names 'artifact pull failed (rsync rc=...)'"
else
  bad "G5: the driver's pull-failure diagnostic text is missing or changed shape"
fi
if grep -qF '[ "$rc" -eq 0 ] && rc="$pull_rc"' "$CLUSTER_SH"; then
  ok "G5: a failed pull JOINS the leg's own rc only when it was still 0 (never overwrites a real failure)"
else
  bad "G5: the pull-failure-joins-rc line is missing or changed shape"
fi

# ============================================================================
# F6(a): the run log lives INSIDE CLUSTER_ARTIFACT_DIR from its first byte
# (never a bare mktemp outside the uploaded/scanned directory), and both
# ranks' own logs are copied there too.
# ============================================================================
mkdir_line="$(grep -n '^mkdir -p "\$CLUSTER_ARTIFACT_DIR"$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
runlog_line="$(grep -n '^RUN_LOG="\$CLUSTER_ARTIFACT_DIR/run.log"$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
if [ -n "$mkdir_line" ] && [ -n "$runlog_line" ] && [ "$mkdir_line" -lt "$runlog_line" ]; then
  ok "F6(a): CLUSTER_ARTIFACT_DIR is created (line ${mkdir_line}) BEFORE RUN_LOG is assigned inside it (line ${runlog_line})"
else
  bad "F6(a): expected mkdir -p \$CLUSTER_ARTIFACT_DIR before RUN_LOG=\$CLUSTER_ARTIFACT_DIR/run.log; mkdir_line=${mkdir_line:-<none>} runlog_line=${runlog_line:-<none>}"
fi
if grep -qF 'cp -f "$rank0_log" "${CLUSTER_ARTIFACT_DIR}/rank0.log"' "$CLUSTER_SH" \
  && grep -qF 'cp -f "$rank1_log" "${CLUSTER_ARTIFACT_DIR}/rank1.log"' "$CLUSTER_SH"; then
  ok "F6(a): both ranks' own logs are copied into CLUSTER_ARTIFACT_DIR"
else
  bad "F6(a): expected both rank0_log/rank1_log to be copied into CLUSTER_ARTIFACT_DIR"
fi

# ============================================================================
# F5: the id-secrecy scan invocation, driven through the REAL scanner
# against fixtures built the same way _rpc_run_id_secrecy_scan expects.
# ============================================================================
scan_fixture_dir="$SANDBOX/scan"
mkdir -p "$scan_fixture_dir/staging" "$scan_fixture_dir/pulled"
ID_BYTES_PY="import sys; sys.stdout.buffer.write(bytes((i*7+1) % 256 for i in range(128)))"
python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/staging/nccl.id"
echo "clean log" > "$scan_fixture_dir/run.log"
echo '{"gang":{"leg":"cluster"}}' > "$scan_fixture_dir/assembled.json"

_rpc_run_id_secrecy_scan "$scan_fixture_dir/staging/nccl.id" "$scan_fixture_dir/pulled" \
  "$scan_fixture_dir/run.log" "$scan_fixture_dir/assembled.json"
rc=$?
if [ "$rc" -eq 0 ] && [ ! -f "$scan_fixture_dir/staging/nccl.id" ]; then
  ok "F5: a clean scan passes (rc=0) and deletes the staging copy (--delete-staging)"
else
  bad "F5: expected a clean scan to pass and delete staging; rc=$rc, staging exists=$([ -f "$scan_fixture_dir/staging/nccl.id" ] && echo yes || echo no)"
fi

python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/staging/nccl.id"
id_hex="$(python3 -c "$ID_BYTES_PY" | python3 -c 'import sys; print(sys.stdin.buffer.read().hex())')"
echo "leaked hex: ${id_hex}" > "$scan_fixture_dir/run.log"
_rpc_run_id_secrecy_scan "$scan_fixture_dir/staging/nccl.id" "$scan_fixture_dir/pulled" \
  "$scan_fixture_dir/run.log" "$scan_fixture_dir/assembled.json"
rc=$?
if [ "$rc" -eq 1 ]; then
  ok "F5: a hex-encoded id leaked into the run log -> FAIL (rc=1)"
else
  bad "F5: expected rc=1 on a hex leak in the run log; got rc=$rc"
fi
echo "clean log" > "$scan_fixture_dir/run.log"

python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/staging/nccl.id"
python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/pulled/bundle.tar.gz"
_rpc_run_id_secrecy_scan "$scan_fixture_dir/staging/nccl.id" "$scan_fixture_dir/pulled" \
  "$scan_fixture_dir/run.log" "$scan_fixture_dir/assembled.json"
rc=$?
if [ "$rc" -eq 2 ]; then
  ok "F5: an archive under the pulled dir -> rc=2 (refused, never opened)"
else
  bad "F5: expected rc=2 on an archive carrier; got rc=$rc"
fi
rm -f "$scan_fixture_dir/pulled/bundle.tar.gz"

# ============================================================================
# F1: rp_init precedes the FIRST rp_cluster_create in the executed block —
# a static line-number comparison over the committed driver text, never a
# behavioral probe (rp_init itself needs no network; ssh-keygen only).
# ============================================================================
rp_init_line="$(grep -n '^rp_init$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
rp_cluster_create_line="$(grep -n 'rp_cluster_create ' "$CLUSTER_SH" | grep -v '^\s*#' | head -1 | cut -d: -f1)"
if [ -n "$rp_init_line" ] && [ -n "$rp_cluster_create_line" ] && [ "$rp_init_line" -lt "$rp_cluster_create_line" ]; then
  ok "F1: rp_init (line ${rp_init_line}) precedes the first rp_cluster_create call (line ${rp_cluster_create_line})"
else
  bad "F1: expected rp_init before the first rp_cluster_create call; rp_init_line=${rp_init_line:-<none>} rp_cluster_create_line=${rp_cluster_create_line:-<none>}"
fi

# ============================================================================
# G4: the cost bound is what the MECHANISM produces.
# ============================================================================
CLUSTER_RATE_USD_PER_GPU_HOUR=1.908
# shellcheck disable=SC2034  # cost_rate is printed for a human reading the
# fixture's own stdout on failure; not read by any later line.
read -r cost_rate cost_i cost_ii < <(python3 - "$RP_TTL_HOURS" "$CLUSTER_RATE_USD_PER_GPU_HOUR" <<'PY'
import sys
ttl_h, per_gpu = int(sys.argv[1]), float(sys.argv[2])
rate = 2 * per_gpu
bound_i = ttl_h * rate
bound_ii = (ttl_h + 6) * rate
print(f"{rate:.3f} {bound_i:.2f} {bound_ii:.2f}")
PY
)

if [ "$RP_TTL_HOURS" = "1" ]; then
  ok "G4: the driver pins RP_TTL_HOURS=1 (the cluster's own entrypoint deadline)"
else
  bad "G4: RP_TTL_HOURS is '${RP_TTL_HOURS}', not 1 — the TTL term of both bounds moved"
fi

if [ "$RP_CLUSTER_POD_COUNT" = "2" ] && [ "$RP_CLUSTER_GPU_COUNT_PER_POD" = "1" ]; then
  ok "G4: the driver pins a 2x1 cluster shape — the shape the \$${CLUSTER_RATE_USD_PER_GPU_HOUR}/GPU/h rate was measured at"
else
  bad "G4: cluster shape is ${RP_CLUSTER_POD_COUNT}x${RP_CLUSTER_GPU_COUNT_PER_POD}, not 2x1 — the rate no longer prices this lane"
fi

for site in "$CLUSTER_YML" "$CLUSTER_SH" "$DEV_GPU_MD"; do
  rel="${site#"$REPO_ROOT"/}"
  if [ ! -f "$site" ]; then
    bad "G4: ${rel} does not exist"
    continue
  fi
  if grep -qF "\$${cost_i}" "$site"; then
    ok "G4: ${rel} states bound (i) as \$${cost_i} — the figure the mechanism produces"
  else
    bad "G4: ${rel} does not state bound (i) (\$${cost_i}) — its prose and the mechanism disagree"
  fi
  if grep -qF "\$${cost_ii}" "$site"; then
    ok "G4: ${rel} states bound (ii) as \$${cost_ii} (the sweep-only path)"
  else
    bad "G4: ${rel} does not state bound (ii) (\$${cost_ii})"
  fi
done

if grep -qE 'MAX_ATTEMPTS:[[:space:]]*"1"' "$CLUSTER_YML"; then
  ok "G4: gpu-cluster.yml pins MAX_ATTEMPTS to \"1\""
else
  bad "G4: gpu-cluster.yml does not pin MAX_ATTEMPTS to \"1\""
fi

# ============================================================================
# G7: NO schedule: key anywhere in the committed gpu-cluster.yml.
# ============================================================================
g7_out="$(python3 "$PROVE_ONCE_PY" --read-on-block "$CLUSTER_YML" 2>&1)"
g7_rc=$?
if [ "$g7_rc" -eq 0 ] && ! printf '%s\n' "$g7_out" | grep -qx "schedule"; then
  ok "G7: gpu-cluster.yml carries no schedule: key (read through the shared on: block reader)"
else
  bad "G7: expected gpu-cluster.yml to carry no schedule: key; got rc=${g7_rc} out=${g7_out}"
fi

# ============================================================================
# check_gpu_prove_once.py's own P7/P8 gates stay green over this lane
# (the real gate, not a paraphrase of it).
# ============================================================================
if python3 "$PROVE_ONCE_PY" >/dev/null 2>&1; then
  ok "check_gpu_prove_once.py is green over the cluster lane (P7/P8, live)"
else
  bad "check_gpu_prove_once.py FAILED with the cluster lane in the tree: $(python3 "$PROVE_ONCE_PY" 2>&1)"
fi

echo
echo "gpu-cluster-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
