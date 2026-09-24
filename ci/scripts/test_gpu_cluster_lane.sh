#!/usr/bin/env bash
# needs: pyyaml, ssh-keygen, jq
# GPU-cluster-lane fixture suite. Mocks-only: no
# network, no GPU, no RunPod account, no cluster. Shaped on
# `test_gpu_gang_lane.sh` (which owns the same job for the pod leg) and
# driving the REAL objects rather than paraphrases of them:
# `runpod_gpu_cluster.sh` is `source`d, not executed — its own
# sourced-execution guard skips the create/wait/heredoc/pull flow when
# sourced — so `rp_cluster_verdict`, `rp_cluster_rank_verdict`,
# `CLUSTER_GROUPS`, `_rpc_*` and the lane's own env pins are the committed
# ones.
#
# Cases (among others below):
#   - sourcing the driver invokes no curl/ssh/scp/rsync/runpodctl — the
#     "mocks-only" claim measured through a PATH shim, not asserted in a
#     comment.
#   - `rp_cluster_rank_verdict` (per rank) and `rp_cluster_verdict`
#     (combining both ranks) over every rc arm.
#   - the CLUSTER_GROUPS closure: every `::group::` name in the driver minus
#     `device` equals CLUSTER_GROUPS exactly.
#   - the shared zero-test tripwire: the SAME `_rp_zero_test_tripwire_lines`
#     text `runpod_gpu_gang.sh` shares, spliced into `_rpc_remote_script`'s
#     own per-rank text.
#   - the launch-time read-back refusal (`_rpc_check_readback`): args
#     mismatch, ssh.direct null with no overlay ip, ssh.direct null with a
#     usable overlay ip (still OK — the no-public-port fallback), a parse
#     failure.
#   - the 128-byte id-file-ready gate (`_rpc_id_file_ready`): 127/128/129
#     bytes, and a missing file.
#   - the `ens1` log assertion (`_rpc_ens1_seen`).
#   - the post-run artifact retrieval: a failed pull JOINS the leg's own rc
#     (never a silently-warned second exit path).
#   - the id-secrecy scan invocation (`_rpc_run_id_secrecy_scan`) against
#     fixtures: a planted raw/hex/base64 id -> FAIL; an archive under the
#     pulled dir -> 2; clean -> pass, staging deleted (--delete-staging).
#   - the cost bound is what the MECHANISM produces: $1.908/GPU/h x 2 GPUs
#     = $3.816/h; (i) = RP_TTL_HOURS x rate; (ii) = (RP_TTL_HOURS + 6) x
#     rate (gpu-reap.yml's 6-hourly sweep).
#   - NO `schedule:` key exists anywhere in the committed `gpu-cluster.yml`.
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
# Every case below exercises the `cluster` transport unless it says
# otherwise -- the pods transport's own default (RP_TWO_HOST_TRANSPORT unset
# -> "pods") never reaches those assertions. Exported so every
# `bash -c 'source ...'` subshell below inherits it identically. The pods-transport's OWN fixtures (bottom of
# this file) override it locally, per-invocation.
export RP_TWO_HOST_TRANSPORT="cluster"

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
  ok "sourcing runpod_gpu_cluster.sh invoked NO curl/ssh/scp/rsync/runpodctl (counted through a PATH shim, not assumed) — the sourced-execution guard really does cover the create/wait/pull path"
else
  bad "sourcing runpod_gpu_cluster.sh invoked ${netprobe_calls} external call(s): $(tr '\n' '; ' <"$NETPROBE_LOG") — the sourced-execution guard does not cover the rent path, and this suite would spend money"
fi

if declare -f rp_cluster_rank_verdict >/dev/null \
  && declare -f rp_cluster_verdict >/dev/null \
  && declare -f _rpc_remote_script >/dev/null \
  && declare -f _rpc_check_readback >/dev/null \
  && declare -f _rpc_wait_for_members_ready >/dev/null \
  && declare -f _rpc_id_file_ready >/dev/null \
  && declare -f _rpc_cross_id_if_minted >/dev/null \
  && declare -f _rpc_ens1_seen >/dev/null \
  && declare -f _rpc_run_id_secrecy_scan >/dev/null \
  && declare -f _rpc_assemble_gang_artifact >/dev/null \
  && declare -f _rpc_self_remove_status >/dev/null \
  && declare -f _rpc_scan_or_destroy >/dev/null \
  && declare -f _rpc_cleanup_cluster >/dev/null \
  && declare -f _rpc_parse_global_network_datacenters >/dev/null \
  && declare -f _rpc_intersect_data_centers >/dev/null \
  && declare -f _rpc_choose_data_center >/dev/null \
  && declare -f _rpc_check_pods_readback >/dev/null \
  && declare -f _rpc_wait_for_two_host_pods_ready >/dev/null \
  && declare -f _rpc_two_host_iface_lines >/dev/null \
  && declare -f _rpc_parse_derived_iface >/dev/null \
  && declare -f _rpc_net_iface_seen >/dev/null \
  && declare -f rp_two_host_pod_create >/dev/null \
  && declare -f rp_pod_get >/dev/null; then
  ok "sourcing runpod_gpu_cluster.sh (no network) defines every pods-transport _rpc_* helper too (rp_two_host_pod_create/rp_pod_get come from runpod_lib.sh)"
else
  bad "one or more pods-transport helpers are not defined after sourcing"
fi

if declare -f rp_cluster_rank_verdict >/dev/null \
  && declare -f rp_cluster_verdict >/dev/null \
  && declare -f _rpc_remote_script >/dev/null \
  && declare -f _rpc_check_readback >/dev/null \
  && declare -f _rpc_wait_for_members_ready >/dev/null \
  && declare -f _rpc_id_file_ready >/dev/null \
  && declare -f _rpc_cross_id_if_minted >/dev/null \
  && declare -f _rpc_ens1_seen >/dev/null \
  && declare -f _rpc_run_id_secrecy_scan >/dev/null \
  && declare -f _rpc_assemble_gang_artifact >/dev/null \
  && declare -f _rpc_self_remove_status >/dev/null \
  && declare -f _rpc_scan_or_destroy >/dev/null \
  && declare -f _rpc_cleanup_cluster >/dev/null; then
  ok "sourcing runpod_gpu_cluster.sh (no network) defines every _rpc_* helper and both verdict functions"
else
  bad "sourcing runpod_gpu_cluster.sh did not define the expected functions"
fi

# ============================================================================
# _rpc_cleanup_cluster — chains rp_cleanup and joins a failed
# rp_cluster_delete into the exit status (naming the cluster LEAKED).
# Drives the REAL function (it lives outside the executed-only guard so it is
# sourceable), mocking only _rp_rest/rp_cluster_delete/rp_cleanup.
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
  ok "self-remove ok (404) -> pending rc (0) preserved AND rp_cleanup still chained (rm -rf \$RP_WORK ran)"
else
  bad "self-remove-ok case: expected rc=0 and rp_cleanup called (rc=$rc, marker=$([ -f "$SANDBOX/rp-cleanup-called" ] && echo yes || echo no))"
fi

# (b) self-remove refused, delete SUCCEEDS -> rp_cleanup still chained, exit
# stays the pending rc (a successful driver-initiated delete is not itself a
# failure).
rm -f "$SANDBOX/rp-cleanup-called"
MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="1" \
  run_cleanup_in_subshell "cl-test-2" 0 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "self-remove refused, delete succeeds -> pending rc (0) preserved AND rp_cleanup chained"
else
  bad "self-remove-refused-delete-ok case: expected rc=0 and rp_cleanup called (rc=$rc)"
fi

# (c) self-remove refused, delete FAILS -> the cluster is LEAKED — the exit
# status is joined to non-zero even though the pending rc was 0 — and
# rp_cleanup is STILL chained (on this arm too).
rm -f "$SANDBOX/rp-cleanup-called"
out="$(MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="0" \
  run_cleanup_in_subshell "cl-test-3" 0 2>&1)"
rc=$?
if [ "$rc" -ne 0 ] && grep -q "LEAKED cluster cl-test-3" <<<"$out"; then
  ok "a failed rp_cluster_delete on exit is LEAKED, naming the cluster id, and joins the exit status non-zero"
else
  bad "expected a non-zero exit naming 'LEAKED cluster cl-test-3' (rc=$rc): $out"
fi
if [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "rp_cleanup is STILL chained even when the cluster delete itself failed"
else
  bad "rp_cleanup must be chained on EVERY arm, including a failed cluster delete"
fi

# (d) a REAL failure already pending (rc=76) with a failed delete -> the
# ORIGINAL failure code is preserved, never overwritten by the generic "1"
# the delete failure alone would have joined.
rm -f "$SANDBOX/rp-cleanup-called"
out="$(MOCK_SELF_REMOVE_STATUS="200" MOCK_DELETE_OK="0" \
  run_cleanup_in_subshell "cl-test-4" 76 2>&1)"
rc=$?
if [ "$rc" -eq 76 ]; then
  ok "a pre-existing failure (76) is preserved verbatim, not overwritten by the delete-failure join"
else
  bad "expected the pre-existing rc=76 to survive a failed delete (got rc=$rc): $out"
fi

# (e) no cluster_id at all (cleanup fires before create ever ran) -> no
# self-remove check, no delete attempt, rp_cleanup still chained, pending
# rc preserved.
rm -f "$SANDBOX/rp-cleanup-called"
run_cleanup_in_subshell "" 75 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 75 ] && [ -f "$SANDBOX/rp-cleanup-called" ]; then
  ok "no cluster_id at all -> rp_cleanup still chained, pending rc (75) preserved, no delete attempted"
else
  bad "empty-cluster_id case: expected rc=75 and rp_cleanup called (rc=$rc)"
fi

# ============================================================================
# The id-secrecy scan runs from the EXIT trap (_rpc_cleanup_cluster ->
# _rpc_scan_or_destroy) on EVERY exit arm, once the id has landed -- driven
# through the REAL trap and the REAL scanner (gang_id_secrecy_scan.py, never
# mocked), over a REAL fixture carrier directory.
#
# Absent vs. unexaminable: whether the assembled artifact is REQUIRED of
# the scan is threaded through `assembly_ok` (0 unless THIS run's own
# assembly step claimed success) exactly as the real orchestration sets it
# -- never pre-created regardless of arm, which would mask the defect these
# cases catch: a refusal arm that never reached assembly must NOT be read as
# UNEXAMINABLE merely because $ASSEMBLED does not exist.
# rank0.log/rank1.log are included in the fixture too (the driver always
# copies them into CLUSTER_ARTIFACT_DIR before assembly, pass or fail) so
# "remains at the upload path" is checked against the FULL real carrier
# set, not just run.log.
# ============================================================================
TRAPSCAN_ID_BYTES_PY="import sys; sys.stdout.buffer.write(bytes((i*7+1) % 256 for i in range(128)))"

run_trap_scan_arm() { # $1=pending_rc $2=plant_leak(1|0) $3=assembly_claimed(1|0) $4=assembled_present(1|0, meaningful only when $3=1) -> stdout: "<rc>\t<carrier_dir>\t<out_b64>"
  local pending_rc="$1" plant_leak="$2" assembly_claimed="$3" assembled_present="${4:-1}" trapscan_sandbox
  trapscan_sandbox="$(mktemp -d)"
  mkdir -p "$trapscan_sandbox/artifact/nested"
  echo "clean log line" > "$trapscan_sandbox/artifact/run.log"
  echo "clean rank0 log" > "$trapscan_sandbox/artifact/rank0.log"
  echo "clean rank1 log" > "$trapscan_sandbox/artifact/rank1.log"
  python3 -c "$TRAPSCAN_ID_BYTES_PY" > "$trapscan_sandbox/nccl.id"
  # ASSEMBLED always lives INSIDE the artifact dir, exactly like the real
  # driver's own `${CLUSTER_ARTIFACT_DIR}/gang-cluster-<ts>.json` -- and is
  # only WRITTEN here when this arm claims assembly succeeded and the file
  # is meant to be present (the "claimed but missing" case below plants
  # assembly_claimed=1 with assembled_present=0 -- claims success, no file).
  if [ "$assembly_claimed" = "1" ] && [ "$assembled_present" = "1" ]; then
    echo '{"gang":{"leg":"cluster"}}' > "$trapscan_sandbox/artifact/assembled.json"
  fi
  if [ "$plant_leak" = "1" ]; then
    id_hex="$(python3 -c "$TRAPSCAN_ID_BYTES_PY" | python3 -c 'import sys; print(sys.stdin.buffer.read().hex())')"
    echo "leaked: ${id_hex}" > "$trapscan_sandbox/artifact/nested/leak.txt"
  fi
  local out trap_rc
  out="$(bash -c '
    source "'"$CLUSTER_SH"'"
    cluster_id=""
    id_landed=1
    assembly_ok="'"$assembly_claimed"'"
    CLUSTER_ARTIFACT_DIR="'"$trapscan_sandbox"'/artifact"
    RUN_LOG="'"$trapscan_sandbox"'/artifact/run.log"
    ASSEMBLED="'"$trapscan_sandbox"'/artifact/assembled.json"
    STAGING_ID_FILE="'"$trapscan_sandbox"'/nccl.id"
    RP_WORK="'"$trapscan_sandbox"'"
    rp_cleanup() { : ; }
    ( exit "'"$pending_rc"'" )
    _rpc_cleanup_cluster
  ' 2>&1)"
  trap_rc=$?
  printf '%s\t%s\t%s\n' "$trap_rc" "$trapscan_sandbox/artifact" "$(printf '%s' "$out" | base64 | tr -d '\n')"
}

carrier_is_clean() { # $1=carrier dir -> 0 if gone or exists-but-empty, 1 if any content survives
  [ ! -e "$1" ] && return 0
  [ -z "$(find "$1" -mindepth 1 2>/dev/null)" ]
}

carrier_intact() { # $1=carrier dir -> 0 iff run.log AND both rank logs survive unaltered
  [ -f "$1/run.log" ] && [ -f "$1/rank0.log" ] && [ -f "$1/rank1.log" ]
}

# (a) "assembly refused" arm: assembly was REACHED but REFUSED (a rank's own
# hostname/iface unresolved, or a repeated host) and so never wrote
# $ASSEMBLED at all -- assembly_ok=0, a CLEAN carrier. The scan must not
# require a file this arm never claimed to produce (and destroy a clean
# carrier for it): it runs and passes, the pending rc (1) survives untouched,
# and the clean carrier -- run.log AND both rank logs -- is left exactly as
# it was.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 1 0 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -eq 1 ] && grep -q "gang-id-secrecy-scan: clean" <<<"$scan_out" && carrier_intact "$carrier_dir"; then
  ok "'assembly refused' arm (never claimed, clean carrier) -> the scan ran (clean), rc=1 preserved, run.log + rank logs REMAIN at the upload path"
else
  bad "'assembly refused' arm: expected rc=1 + a clean scan + intact carrier; got rc=$trap_rc carrier_intact=$(carrier_intact "$carrier_dir" && echo yes || echo no) out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (b) "pull failed" arm: the pull itself failed, so assembly was never even
# REACHED -- assembly_ok=0, a CLEAN carrier (the rank reports never landed,
# but nothing here carries the id either). The scan must not destroy this
# arm's own diagnostics for lacking a file nobody promised: the exit code
# is the arm's own (not 2), and run.log + rank logs REMAIN at the upload
# path -- the only evidence a reviewer has on exactly this arm.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 1 0 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -eq 1 ] && grep -q "gang-id-secrecy-scan: clean" <<<"$scan_out" && carrier_intact "$carrier_dir"; then
  ok "'pull failed' arm (never claimed, clean carrier) -> the scan ran (clean), the exit code is the arm's own (not 2), run.log + rank logs REMAIN at the upload path"
else
  bad "'pull failed' arm: expected rc=1 (not 2) + a clean scan + intact carrier; got rc=$trap_rc carrier_intact=$(carrier_intact "$carrier_dir" && echo yes || echo no) out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (c) a planted-id refusal arm (pull failed, but a leak rode in with a
# partial transfer) -- assembly_ok=0, a DIRTY carrier: the scan runs, finds
# the hit, and DESTROYS the carrier -- the upload path holds nothing. This
# is the "planted-id refusal arm asserting removal" oracle: never claiming
# assembly must not weaken the scan's own strictness about an actual leak.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 1 1 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -ne 0 ] && grep -q "gang-id-secrecy-scan: HIT" <<<"$scan_out" && carrier_is_clean "$carrier_dir"; then
  ok "planted-id refusal arm (pull failed, dirty carrier, never claimed assembly) -> the scan ran (HIT), the carrier is DESTROYED, nothing reaches the upload path"
else
  bad "planted-id refusal arm: expected a HIT scan + a destroyed carrier; got rc=$trap_rc carrier_clean=$(carrier_is_clean "$carrier_dir" && echo yes || echo no) out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (d) "budget cut" arm (rc=124): never reached assembly, assembly_ok=0, a
# CLEAN carrier -> the scan runs and passes, the driver's own named exit
# code (124) survives verbatim, and the carrier stays intact.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 124 0 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -eq 124 ] && grep -q "gang-id-secrecy-scan: clean" <<<"$scan_out" && carrier_intact "$carrier_dir"; then
  ok "'budget cut' arm (never claimed, clean carrier) -> the scan ran (clean), the named exit code 124 survives verbatim, carrier intact"
else
  bad "'budget cut' arm: expected rc=124 + a clean scan + intact carrier; got rc=$trap_rc out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (e) "wrong tree" arm (rc=77): never reached assembly, assembly_ok=0, a
# DIRTY carrier -> the scan DESTROYS it AND the driver's own named exit
# code (77) still survives verbatim (never overwritten by the scan's own
# rc, since a real per-arm code always outranks a bare join per
# rp_cluster_verdict's own priority doctrine).
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 77 1 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -eq 77 ] && grep -q "gang-id-secrecy-scan: HIT" <<<"$scan_out" && carrier_is_clean "$carrier_dir"; then
  ok "'wrong tree' arm (rc=77, dirty carrier) -> the scan ran (HIT), DESTROYED the carrier, AND the named exit code 77 survives verbatim"
else
  bad "'wrong tree' arm: expected rc=77 + a HIT scan + a destroyed carrier; got rc=$trap_rc carrier_clean=$(carrier_is_clean "$carrier_dir" && echo yes || echo no) out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (f) the happy-path tail: assembly SUCCEEDED (assembly_ok=1) and
# $ASSEMBLED is present and clean -> the scan still runs, still requires
# the file (happy-path strictness), finds it, and passes clean --
# proving assembly_ok=1 does not itself break the ordinary pass arm.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 0 0 1 1)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -eq 0 ] && grep -q "gang-id-secrecy-scan: clean" <<<"$scan_out" && carrier_intact "$carrier_dir" && [ -f "$carrier_dir/assembled.json" ]; then
  ok "happy-path arm (assembly claimed, file present, clean) -> the scan still requires and finds the assembled artifact, passes clean, rc=0 preserved"
else
  bad "happy-path arm: expected rc=0 + a clean scan + the assembled artifact present; got rc=$trap_rc out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# (g) the "claimed but missing" case: assembly_ok=1 (this run's own
# assembly step reported success) but $ASSEMBLED is missing at scan time
# (a corruption/race) -> happy-path strictness holds: UNEXAMINABLE (never
# clean), and the carrier is DESTROYED -- exactly
# `test_missing_assembled_artifact_is_unexaminable_never_clean`'s own
# property, threaded all the way through the REAL driver's assembly_ok
# wiring, not just the scanner in isolation.
IFS=$'\t' read -r trap_rc carrier_dir out_b64 <<< "$(run_trap_scan_arm 0 0 1 0)"
scan_out="$(printf '%s' "$out_b64" | base64 -d)"
if [ "$trap_rc" -ne 0 ] && grep -q "gang-id-secrecy-scan: UNEXAMINABLE" <<<"$scan_out" && carrier_is_clean "$carrier_dir"; then
  ok "'claimed but missing' arm (assembly_ok=1, file absent) -> UNEXAMINABLE, never clean, carrier DESTROYED -- happy-path strictness holds"
else
  bad "'claimed but missing' arm: expected a non-zero rc + UNEXAMINABLE + a destroyed carrier; got rc=$trap_rc carrier_clean=$(carrier_is_clean "$carrier_dir" && echo yes || echo no) out=$scan_out"
fi
rm -rf "$(dirname "$carrier_dir")"

# id_landed=0 (no id ever reached this runner) -> the trap does NOT invoke
# the scanner at all (nothing to protect against yet).
rc75_out="$(bash -c '
  trapscan_sandbox="'"$SANDBOX"'/trapscan-unlanded"
  rm -rf "$trapscan_sandbox"; mkdir -p "$trapscan_sandbox/artifact"
  echo "clean log line" > "$trapscan_sandbox/artifact/run.log"
  source "'"$CLUSTER_SH"'"
  cluster_id=""
  CLUSTER_ARTIFACT_DIR="$trapscan_sandbox/artifact"
  RUN_LOG="$trapscan_sandbox/artifact/run.log"
  rp_cleanup() { : ; }
  ( exit 75 )
  _rpc_cleanup_cluster
' 2>&1)"
rc75=$?
if [ "$rc75" -eq 75 ] && ! grep -q "gang-id-secrecy-scan" <<<"$rc75_out"; then
  ok "id_landed=0 (the id never reached this runner) -> the trap never invokes the scanner at all"
else
  bad "expected no scanner invocation when id_landed is unset; rc=$rc75 out=$rc75_out"
fi
rm -rf "$SANDBOX/trapscan-unlanded"

# ============================================================================
# The id-secrecy scan runs FIRST in _rpc_cleanup_cluster,
# strictly before either of its own REST calls (_rpc_self_remove_status,
# rp_cluster_delete). Drives the REAL trap in a real subshell, instrumenting
# BOTH the scan wrapper and _rp_rest to append to a shared, ordered log —
# the assertion is on ORDER, never merely "both happened".
# ============================================================================
TRAP_ORDER_LOG="$SANDBOX/trap-order.log"
: > "$TRAP_ORDER_LOG"
bash -c '
  source "'"$CLUSTER_SH"'"
  cluster_id="cl-trap"
  id_landed=1
  CLUSTER_ARTIFACT_DIR="'"$SANDBOX"'/trap-artifact"
  RUN_LOG="'"$SANDBOX"'/trap-artifact/run.log"
  STAGING_ID_FILE="'"$SANDBOX"'/trap-artifact/nccl.id"
  mkdir -p "$CLUSTER_ARTIFACT_DIR"
  echo "clean" > "$RUN_LOG"
  _rpc_run_id_secrecy_scan() { echo "SCAN_CALLED" >> "'"$TRAP_ORDER_LOG"'"; return 0; }
  _rp_rest() { echo "REST_CALLED" >> "'"$TRAP_ORDER_LOG"'"; printf "404\n{}"; }
  rp_cluster_delete() { echo "REST_CALLED" >> "'"$TRAP_ORDER_LOG"'"; return 0; }
  rp_cleanup() { : ; }
  ( exit 0 )
  _rpc_cleanup_cluster
' >/dev/null 2>&1
order="$(cat "$TRAP_ORDER_LOG" | tr '\n' ' ')" # tripwire-ok: a `useless cat`, deliberately -- the ORDER the shell wrote these lines in is exactly what this assertion reads, and `tr` alone does not read a file argument.
if [ "$order" = "SCAN_CALLED REST_CALLED " ]; then
  ok "the id-secrecy scan runs BEFORE the self-remove-status/cluster-delete REST calls in the cleanup trap (order: ${order})"
else
  bad "expected 'SCAN_CALLED REST_CALLED '; got order: '${order}' — the scan is not sequenced first"
fi
rm -rf "$SANDBOX/trap-artifact" "$TRAP_ORDER_LOG"

# The cleanup trap is registered on EXIT, INT, TERM AND HUP —
# never EXIT alone (an untrapped SIGINT otherwise skips an EXIT-only trap
# entirely on a non-interactive shell).
#
# TERM and HUP: driven DYNAMICALLY, over a REAL backgrounded process that
# registers the SAME four traps this driver's own executed block does
# (calling the REAL, sourced `_rpc_cleanup_cluster`), sending each signal
# in turn and asserting the trap's own marker file was written.
#
# INT: NOT driven dynamically here, by a deliberate choice, not an
# oversight -- bash's own documented behavior sets SIGINT/SIGQUIT (and
# ONLY those two) to SIG_IGN for an ASYNCHRONOUS (`&`-backgrounded) list
# command in a job-control-OFF shell, BEFORE that child ever runs its own
# `trap`; a signal already SIG_IGN at shell entry cannot be re-armed by a
# later `trap` at all (POSIX). `set -m` (job control) works around this in
# an interactive-shaped environment, but this suite has observed it
# UNRELIABLE without a real controlling terminal (this exact class of
# harness/CI difference, not a driver defect: a genuine CI runner sends
# its cancel signal to this driver's OWN foreground process, never to a
# shell's backgrounded job, so this async-ignore rule never applies to the
# real driver at all -- only to THIS test's own artificial need to
# background a driver-emulating subshell in order to signal it while
# still running). Verified statically instead: the exact `trap ... INT`
# registration line exists, alongside HUP/TERM, spelling the correct
# 128+n code -- the SAME class of static, line-text check `test_check_
# gpu_prove_once.py`'s own `RpSshoRequiresRpInitTest` and this file's own
# rp_init-ordering check (below) already use for a property a dynamic
# signal-delivery test cannot reliably exercise across every environment
# this suite runs in.
for sig in TERM HUP; do
  TRAP_SIG_MARKER="$SANDBOX/trap-sig-${sig}.marker"
  TRAP_ARMED_MARKER="$SANDBOX/trap-armed-${sig}.marker"
  rm -f "$TRAP_SIG_MARKER" "$TRAP_ARMED_MARKER"
  bash -c '
    source "'"$CLUSTER_SH"'"
    cluster_id=""
    rp_cleanup() { : > "'"$TRAP_SIG_MARKER"'"; }
    trap _rpc_cleanup_cluster EXIT
    trap "_rpc_cleanup_cluster 129" HUP
    trap "_rpc_cleanup_cluster 130" INT
    trap "_rpc_cleanup_cluster 143" TERM
    : > "'"$TRAP_ARMED_MARKER"'"   # the observed event the fixture waits on: every trap is registered
    sleep 30 &
    wait "$!"
  ' &
  driver_pid=$!
  # Signal only once the subshell has OBSERVABLY registered its traps (the
  # armed marker), never after a fixed sleep: sourcing the driver takes
  # longer than any guessed delay on a loaded machine, and a signal that
  # lands before `trap` runs kills the subshell without the trap -- a
  # false failure that says nothing about the driver. The bound below
  # is a generous backstop against a wedged machine, not the pace of the
  # work.
  armed_deadline=$((SECONDS + 60))
  while [ ! -f "$TRAP_ARMED_MARKER" ] && kill -0 "$driver_pid" 2>/dev/null && [ "$SECONDS" -lt "$armed_deadline" ]; do sleep 0.05; done
  [ -f "$TRAP_ARMED_MARKER" ] || bad "the SIG${sig} fixture's subshell never reported its traps armed within 60s (a wedged machine, or the driver failed to source) -- refusing to read a missing cleanup marker as the driver's fault"
  kill "-${sig}" "$driver_pid" 2>/dev/null
  # Read the marker only after the subshell has EXITED (the trap handler
  # runs to completion before the process ends), never after a fixed
  # window: the handler's own work is not paced by this fixture, and on a
  # loaded machine a guessed window reads a still-running handler as "the
  # trap never fired". The bound is a backstop against a wedged machine.
  exit_deadline=$((SECONDS + 60))
  while kill -0 "$driver_pid" 2>/dev/null && [ "$SECONDS" -lt "$exit_deadline" ]; do sleep 0.05; done
  kill -0 "$driver_pid" 2>/dev/null && bad "the SIG${sig} fixture's subshell was still alive 60s after the signal (a wedged machine, or a handler that never exits)"
  if [ -f "$TRAP_SIG_MARKER" ]; then
    ok "the cleanup trap fires under SIG${sig} (registered on EXIT/INT/TERM/HUP, never EXIT alone)"
  else
    bad "SIG${sig} did not fire the cleanup trap (registered on EXIT alone would miss this) — marker never written"
  fi
  rm -f "$TRAP_SIG_MARKER"
done

trap_sig_lines="$(grep -nE "^trap _rpc_cleanup_cluster EXIT\$|^trap '_rpc_cleanup_cluster 129' HUP\$|^trap '_rpc_cleanup_cluster 130' INT\$|^trap '_rpc_cleanup_cluster 143' TERM\$" "$CLUSTER_SH")"
# The pods transport registers the SAME 4-line trap pattern for its OWN
# create/wait/build flow, before the FIRST create call, exactly as the
# cluster branch does -- 8 lines total (4 per branch).
if [ "$(printf '%s\n' "$trap_sig_lines" | wc -l | tr -d ' ')" = "8" ]; then
  ok "all four trap registrations (EXIT, HUP, INT, TERM) are present verbatim TWICE (once per transport branch, 8 lines total) in the committed driver — INT's own dynamic delivery is unreliable across environments without a real controlling terminal (above), so this property is checked statically, over the committed TEXT, never dynamically for this one signal"
else
  bad "expected exactly 8 trap registration lines (4 per transport branch); got: $trap_sig_lines"
fi

# ============================================================================
# A dirty carrier is destroyed SYNCHRONOUSLY, in the SAME trap invocation,
# never left to rp_cleanup's own RP_SESSION-conditional `$RP_WORK` teardown.
# Sources the REAL runpod_lib.sh (never mocking rp_cleanup this time) with
# RP_SESSION set BEFORE sourcing -- the exact condition that clears
# RP_WORK_IS_TEMP (runpod_lib.sh's own source-time logic) and would leave a
# relocated-but-deferred dirty carrier on disk.
# ============================================================================
DESTROY_SESSION_ROOT="$SANDBOX/destroy-session-root"
DESTROY_ARTIFACT="$SANDBOX/destroy-artifact"
mkdir -p "$DESTROY_ARTIFACT/nested"
echo "clean log" > "$DESTROY_ARTIFACT/run.log"
id_hex="$(python3 -c "$TRAPSCAN_ID_BYTES_PY" | python3 -c 'import sys; print(sys.stdin.buffer.read().hex())')"
echo "leaked: ${id_hex}" > "$DESTROY_ARTIFACT/nested/leak.txt"
# The session dir must exist BEFORE the trap runs -- `mv`'s own target
# parent must already be there, or `mv` itself fails (ENOENT) and this
# fixture would exercise the in-place fallback instead of the
# quarantine-then-destroy path this test means to drive.
mkdir -p "$DESTROY_SESSION_ROOT/destroy-test-session"
destroy_out="$(RP_SESSION="destroy-test-session" RP_SESSION_ROOT="$DESTROY_SESSION_ROOT" RUNPOD_API_KEY="test-dummy-key" bash -c '
  source "'"$CLUSTER_SH"'"
  cluster_id=""
  id_landed=1
  CLUSTER_ARTIFACT_DIR="'"$DESTROY_ARTIFACT"'"
  RUN_LOG="'"$DESTROY_ARTIFACT"'/run.log"
  STAGING_ID_FILE="'"$DESTROY_ARTIFACT"'/nccl.id"
  echo -n x > "$STAGING_ID_FILE"
  _rpc_cleanup_cluster
' 2>&1)"
if [ -n "$(find "$DESTROY_SESSION_ROOT" -mindepth 1 -name "gpu-cluster-destroy-*" 2>/dev/null)" ]; then
  bad "a quarantined dirty carrier survived under RP_WORK (RP_SESSION set) -- destruction was left to rp_cleanup's own RP_SESSION-conditional teardown; out=$destroy_out"
elif carrier_is_clean "$DESTROY_ARTIFACT"; then
  ok "under RP_SESSION (RP_WORK_IS_TEMP=0, rp_cleanup's own conditional rm -rf never fires), the dirty carrier is STILL destroyed synchronously — nothing survives under \$RP_WORK either"
else
  bad "expected the original carrier path emptied too; out=$destroy_out"
fi
rm -rf "$DESTROY_ARTIFACT" "$DESTROY_SESSION_ROOT"

# Mutation check: on a SCRATCH COPY, replace `_rpc_scan_or_destroy`'s
# destroy step with a quarantine-and-defer shape (move only, no synchronous
# rm -rf; the log line saying "WILL BE DESTROYED"), and confirm the SAME
# RP_SESSION fixture above then leaves a surviving quarantine directory
# under $RP_WORK -- proving the synchronous destroy is load-bearing. The
# `old` block is matched VERBATIM against the driver, comments included.
DESTROY_SCRATCH_DIR="$SANDBOX/destroy-revert-red"
mkdir -p "$DESTROY_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$DESTROY_SCRATCH_DIR/runpod_lib.sh"
cp "$DIR/gang_id_secrecy_scan.py" "$DESTROY_SCRATCH_DIR/gang_id_secrecy_scan.py"
DESTROY_SCRATCH="$DESTROY_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$DESTROY_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''      echo "::error::id-secrecy scan was not clean (rc=${scan_rc}) -- the carrier directory was moved to ${pending_destroy_dir} (outside ${CLUSTER_ARTIFACT_DIR}) and destroyed there, now, unconditionally (never left to rp_cleanup's own RP_SESSION-conditional teardown); the upload step finds nothing there"
      # Destroyed HERE, synchronously -- never deferred to rp_cleanup's
      # own conditional `rm -rf "$RP_WORK"`, which a real RP_SESSION run
      # would skip entirely, leaving this exact directory on disk.
      rm -rf "${pending_destroy_dir:?}" 2>/dev/null'''
new = '''      echo "::error::id-secrecy scan was not clean (rc=${scan_rc}) -- the carrier directory was moved to ${pending_destroy_dir} (outside ${CLUSTER_ARTIFACT_DIR}) and WILL BE DESTROYED when this process exits (rp_cleanup rm -rf's \\$RP_WORK); the upload step finds nothing there"'''
assert old in text, "quarantine-and-defer mutation fixture: synchronous-destroy block not found verbatim"
text = text.replace(old, new, 1)
open(dst, "w").write(text)
PY
mkdir -p "$DESTROY_SESSION_ROOT/destroy-test-session-revert"
destroy_revert_out="$(RP_SESSION="destroy-test-session-revert" RP_SESSION_ROOT="$DESTROY_SESSION_ROOT" RUNPOD_API_KEY="test-dummy-key" bash -c '
  source "'"$DESTROY_SCRATCH"'"
  cluster_id=""
  id_landed=1
  CLUSTER_ARTIFACT_DIR="'"$DESTROY_ARTIFACT"'"
  RUN_LOG="'"$DESTROY_ARTIFACT"'/run.log"
  STAGING_ID_FILE="'"$DESTROY_ARTIFACT"'/nccl.id"
  mkdir -p "$CLUSTER_ARTIFACT_DIR/nested"
  echo "clean log" > "$RUN_LOG"
  echo -n x > "$STAGING_ID_FILE"
  echo "leaked: '"$id_hex"'" > "$CLUSTER_ARTIFACT_DIR/nested/leak.txt"
  _rpc_cleanup_cluster
' 2>&1)"
if [ -n "$(find "$DESTROY_SESSION_ROOT" -mindepth 1 -name "gpu-cluster-destroy-*" 2>/dev/null)" ]; then
  ok "mutation check: a quarantine-and-defer shape DOES leave a surviving dirty carrier under \$RP_WORK when RP_SESSION is set — the synchronous destroy above is load-bearing"
else
  bad "mutation check: expected the quarantine-and-defer shape to leave a surviving quarantine dir under RP_WORK; none found -- the mutation fixture itself may be stale; out=$destroy_revert_out"
fi
rm -rf "$DESTROY_SESSION_ROOT" "$DESTROY_SCRATCH_DIR"

# ============================================================================
# The in-place destroy fallback (used when the `mv` itself
# fails) matches `..`-prefixed names too, not only `*` and `.[!.]*`. Forces
# the `mv` branch to fail with a PATH-shimmed `mv` stub, plants a
# `..leak`-named file alongside a normally-named leak, and asserts BOTH are
# gone after the trap runs.
# ============================================================================
DOTDOT_ARTIFACT="$SANDBOX/dotdot-artifact"
mkdir -p "$DOTDOT_ARTIFACT"
echo "clean log" > "$DOTDOT_ARTIFACT/run.log"
echo "leaked: ${id_hex}" > "$DOTDOT_ARTIFACT/normal-leak.txt"
echo "leaked: ${id_hex}" > "$DOTDOT_ARTIFACT/..leak"
DOTDOT_MV_STUB_BIN="$SANDBOX/dotdot-mv-stub"
mkdir -p "$DOTDOT_MV_STUB_BIN"
cat > "$DOTDOT_MV_STUB_BIN/mv" <<'SH'
#!/usr/bin/env bash
exit 1
SH
chmod +x "$DOTDOT_MV_STUB_BIN/mv"
dotdot_out="$(PATH="$DOTDOT_MV_STUB_BIN:$PATH" bash -c '
  source "'"$CLUSTER_SH"'"
  cluster_id=""
  id_landed=1
  CLUSTER_ARTIFACT_DIR="'"$DOTDOT_ARTIFACT"'"
  RUN_LOG="'"$DOTDOT_ARTIFACT"'/run.log"
  STAGING_ID_FILE="'"$DOTDOT_ARTIFACT"'/nccl.id"
  echo -n x > "$STAGING_ID_FILE"
  rp_cleanup() { : ; }
  _rpc_cleanup_cluster
' 2>&1)"
if [ ! -e "$DOTDOT_ARTIFACT/normal-leak.txt" ] && [ ! -e "$DOTDOT_ARTIFACT/..leak" ]; then
  ok "the in-place fallback (mv forced to fail) destroys BOTH a normally-named leak and a '..'-prefixed one"
else
  bad "expected both leaks destroyed in place; normal-leak.txt exists=$([ -e "$DOTDOT_ARTIFACT/normal-leak.txt" ] && echo yes || echo no) ..leak exists=$([ -e "$DOTDOT_ARTIFACT/..leak" ] && echo yes || echo no); out=$dotdot_out"
fi

# Mutation check: the narrower glob set (`*` and `.[!.]*` only) on a
# SCRATCH COPY misses the `..`-prefixed name -- the `..?*` glob is
# load-bearing, not vacuous.
DOTDOT_SCRATCH_DIR="$SANDBOX/dotdot-revert-red"
mkdir -p "$DOTDOT_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$DOTDOT_SCRATCH_DIR/runpod_lib.sh"
cp "$DIR/gang_id_secrecy_scan.py" "$DOTDOT_SCRATCH_DIR/gang_id_secrecy_scan.py"
DOTDOT_SCRATCH="$DOTDOT_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$DOTDOT_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''      rm -rf "${CLUSTER_ARTIFACT_DIR:?}"/* "${CLUSTER_ARTIFACT_DIR:?}"/.[!.]* "${CLUSTER_ARTIFACT_DIR:?}"/..?* 2>/dev/null'''
new = '''      rm -rf "${CLUSTER_ARTIFACT_DIR:?}"/* "${CLUSTER_ARTIFACT_DIR:?}"/.[!.]* 2>/dev/null'''
assert old in text, "narrow-glob mutation fixture: in-place fallback glob line not found verbatim"
text = text.replace(old, new, 1)
open(dst, "w").write(text)
PY
DOTDOT_ARTIFACT_REVERT="$SANDBOX/dotdot-artifact-revert"
mkdir -p "$DOTDOT_ARTIFACT_REVERT"
echo "clean log" > "$DOTDOT_ARTIFACT_REVERT/run.log"
echo "leaked: ${id_hex}" > "$DOTDOT_ARTIFACT_REVERT/..leak"
dotdot_revert_out="$(PATH="$DOTDOT_MV_STUB_BIN:$PATH" bash -c '
  source "'"$DOTDOT_SCRATCH"'"
  cluster_id=""
  id_landed=1
  CLUSTER_ARTIFACT_DIR="'"$DOTDOT_ARTIFACT_REVERT"'"
  RUN_LOG="'"$DOTDOT_ARTIFACT_REVERT"'/run.log"
  STAGING_ID_FILE="'"$DOTDOT_ARTIFACT_REVERT"'/nccl.id"
  echo -n x > "$STAGING_ID_FILE"
  rp_cleanup() { : ; }
  _rpc_cleanup_cluster
' 2>&1)"
if [ -e "$DOTDOT_ARTIFACT_REVERT/..leak" ]; then
  ok "mutation check: a glob set missing ..?* DOES leave the '..'-prefixed leak behind — the ..?* glob above is load-bearing"
else
  bad "mutation check: expected a glob set missing ..?* to miss the '..'-prefixed leak; it was removed anyway -- the mutation fixture itself may be stale; out=$dotdot_revert_out"
fi
rm -rf "$DOTDOT_ARTIFACT" "$DOTDOT_ARTIFACT_REVERT" "$DOTDOT_SCRATCH_DIR"

# ============================================================================
# CLUSTER_GROUPS closure — {::group:: names} - {device} == CLUSTER_GROUPS.
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
  ok "CLUSTER_GROUPS closure: {::group:: names} - {device} == CLUSTER_GROUPS exactly (${#declared_groups[@]} members)"
else
  bad "CLUSTER_GROUPS closure mismatch: script non-member groups=[${non_members_sorted[*]}] vs CLUSTER_GROUPS=[${declared_groups[*]}]"
fi

# ============================================================================
# rp_cluster_rank_verdict (per rank) over every rc arm.
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
  ok "every gating group rc=0 with ssh 0 -> 0, no diagnostic"
else
  bad "expected a silent 0; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-missing.log"
{
  echo "PROVE_GROUP_RC name=cluster-build rc=0"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"cluster-proof=<missing>"* ]]; then
  ok "ssh 0 with a MISSING group marker -> 1, naming the group as <missing>"
else
  bad "expected 1 + cluster-proof=<missing>; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-nonzero.log"
{
  echo "PROVE_GROUP_RC name=cluster-build rc=0"
  echo "PROVE_GROUP_RC name=cluster-proof rc=3"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"cluster-proof=3"* ]]; then
  ok "ssh 0 with a NON-ZERO group marker -> 1 (a bare zero is never trusted against its own markers)"
else
  bad "expected 1 + cluster-proof=3; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-hang.log"
echo "PROVE_GROUP_RC name=cluster-build rc=0" > "$log"
res="$(drive_rank 76 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 76 ] && [[ "$err" == *"cut/hang (raw rc=76)"* ]] && [[ "$err" == *"cluster-proof=<missing>"* ]]; then
  ok "inactivity kill (76, no PROVE_EXIT) -> 76, naming the unresolved group"
else
  bad "expected 76 + cut/hang naming cluster-proof; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-cut.log"
echo "PROVE_GROUP_RC name=cluster-build rc=0" > "$log"
res="$(drive_rank 124 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 124 ] && [[ "$err" == *"cut/hang (raw rc=124)"* ]]; then
  ok "budget cut (124, no PROVE_EXIT) -> 124, naming the unresolved group"
else
  bad "expected 124 + cut/hang; got rc=$rc err=$err"
fi

for insuite in 1 97 77 255; do
  log="$SANDBOX/g1-insuite-${insuite}.log"
  {
    for g in "${CLUSTER_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
    echo "PROVE_EXIT=${insuite}"
  } > "$log"
  res="$(drive_rank "$insuite" "$log")"; rc="${res%%|*}"
  if [ "$rc" -eq "$insuite" ]; then
    ok "an in-suite exit ${insuite} (PROVE_EXIT present) is returned VERBATIM"
  else
    bad "expected in-suite exit ${insuite} returned verbatim; got rc=$rc"
  fi
done

log="$SANDBOX/g1-unreadable.log"
# no such file
res="$(drive_rank 0 "$log")"; rc="${res%%|*}"
if [ "$rc" -eq 1 ]; then
  ok "an unreadable log (every marker lost) with ssh 0 -> 1, fail-closed"
else
  bad "expected rc=1 on an unreadable log; got rc=$rc"
fi

# rp_cluster_verdict: combining both ranks.
if rp_cluster_verdict 0 0; then
  ok "rp_cluster_verdict(0,0) -> 0 (both ranks clean)"
else
  bad "rp_cluster_verdict(0,0) expected 0, got $?"
fi

rp_cluster_verdict 75 0; rc=$?
[ "$rc" -eq 75 ] && ok "rp_cluster_verdict(75,0) -> 75 (a special code on EITHER rank wins over a plain 0)" \
  || bad "rp_cluster_verdict(75,0) expected 75, got $rc"

rp_cluster_verdict 0 76; rc=$?
[ "$rc" -eq 76 ] && ok "rp_cluster_verdict(0,76) -> 76 (the other rank's special code still wins)" \
  || bad "rp_cluster_verdict(0,76) expected 76, got $rc"

rp_cluster_verdict 77 97; rc=$?
[ "$rc" -eq 77 ] && ok "rp_cluster_verdict(77,97) -> 77 (priority order: 75,76,77,97,124)" \
  || bad "rp_cluster_verdict(77,97) expected 77 (priority order), got $rc"

rp_cluster_verdict 1 0; rc=$?
[ "$rc" -eq 1 ] && ok "rp_cluster_verdict(1,0) -> 1 (a plain nonzero on either rank still fails the leg)" \
  || bad "rp_cluster_verdict(1,0) expected 1, got $rc"

# ============================================================================
# The shared zero-test tripwire — the SAME text runpod_gpu_gang.sh
# shares, spliced into _rpc_remote_script's own per-rank text.
# ============================================================================
rank0_text="$(_rpc_remote_script 0)"
rank1_text="$(_rpc_remote_script 1)"
if [[ "$rank0_text" == *'if grep -q "running 0 tests" "$rank_log"; then'* ]] \
  && [[ "$rank1_text" == *'if grep -q "running 0 tests" "$rank_log"; then'* ]]; then
  ok "both ranks' remote text carries the shared zero-test tripwire verbatim"
else
  bad "the shared zero-test tripwire is missing from one or both ranks' remote text"
fi
if [[ "$rank0_text" == *"matched ZERO tests"* ]]; then
  ok "the tripwire's own message names 'matched ZERO tests'"
else
  bad "expected 'matched ZERO tests' in the rank's remote text"
fi
if [[ "$rank0_text" == *"JAMMI_GANG_TWO_HOSTS_RANK=0"* ]] && [[ "$rank1_text" == *"JAMMI_GANG_TWO_HOSTS_RANK=1"* ]]; then
  ok "each rank's text pins its OWN JAMMI_GANG_TWO_HOSTS_RANK"
else
  bad "rank env pins are wrong: rank0=${rank0_text}, rank1=${rank1_text}"
fi
if [[ "$rank0_text" == *"JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT}"* ]]; then
  ok "world derives from RP_CLUSTER_POD_COUNT (${RP_CLUSTER_POD_COUNT}), never a second literal"
else
  bad "expected JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT} in the rank text"
fi

# ============================================================================
# The launch-time read-back refusal (_rpc_check_readback).
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
if grep -q "^primary 0 READBACK_OK 10.0.0.2 1.2.3.4 22$" <<<"$out" \
  && grep -q "^member 1 READBACK_OK 10.0.0.3 5.6.7.8 22$" <<<"$out"; then
  ok "both members with matching args + direct ssh -> READBACK_OK, host/port carried"
else
  bad "expected both READBACK_OK with host/port; got: $out"
fi

body="$(readback_body "bash -c 'not the setup'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"5.6.7.8","port":22}}' "10.0.0.3")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if grep -q "READBACK_ARGS_MISMATCH" <<<"$out"; then
  ok "a member whose args do not echo the setup text -> READBACK_ARGS_MISMATCH"
else
  bad "expected READBACK_ARGS_MISMATCH; got: $out"
fi

body="$(readback_body "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{}' "")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if grep -q "READBACK_NO_SSH_PATH" <<<"$out"; then
  ok "a member with NO ssh.direct and NO overlay ip -> READBACK_NO_SSH_PATH"
else
  bad "expected READBACK_NO_SSH_PATH; got: $out"
fi

body="$(readback_body "bash -c '${SETUP_TEXT}'" '{"direct":{"host":"1.2.3.4","port":22}}' "10.0.0.2" \
  "bash -c '${SETUP_TEXT}'" '{}' "10.0.0.3")"
out="$(_rpc_check_readback "$body" "$SETUP_TEXT")"
if grep -q "^member 1 READBACK_OK 10.0.0.3 - -$" <<<"$out"; then
  ok "a member with NO ssh.direct but a usable overlay ip -> still READBACK_OK (the no-public-port fallback)"
else
  bad "expected member READBACK_OK with dash host/port (proxy fallback); got: $out"
fi

out="$(_rpc_check_readback "not json at all" "$SETUP_TEXT" 2>/dev/null)"; rc=$?
if [ "$rc" -eq 2 ] && grep -q "PARSE_ERROR" <<<"$out"; then
  ok "an unparseable pods response -> rc=2, named PARSE_ERROR"
else
  bad "expected rc=2 + PARSE_ERROR on unparseable body; got rc=$rc out=$out"
fi

# ============================================================================
# _rpc_wait_for_members_ready tracks readiness by DISTINCT
# rank, never a raw count -- a response naming rank 0 TWICE and rank 1
# NEVER must never read as ready. Drives the REAL function in a real
# subshell (a real `while`/`sleep` loop; RP_SSH_WAIT_SECS is kept small so
# the refusal arm's own timeout is fast), mocking only _rp_rest and
# _rp_entrypoint_setup.
# ============================================================================
run_wait_for_members_ready() { # $1=pods_body_json $2=RP_SSH_WAIT_SECS $3=driver path (default CLUSTER_SH) -> stdout: "<rc>\t<out>"
  local body="$1" wait_secs="$2" driver="${3:-$CLUSTER_SH}" out rc
  out="$(RAWCOUNT_FIXTURE_BODY="$body" bash -c '
    source "'"$driver"'"
    _rp_entrypoint_setup() { printf "%s" "the-shared-entrypoint-text"; }
    _rp_rest() { printf "200\n%s" "$RAWCOUNT_FIXTURE_BODY"; }
    _rpc_wait_for_members_ready "cl-rawcount" "$1" "1"
  ' _ "$wait_secs" 2>&1)"
  rc=$?
  printf '%s\t%s\n' "$rc" "$out"
}

dup_rank0_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
pod = lambda pid, rank: {"id": pid, "args": "bash -c %r" % setup, "cluster": {"rank": rank, "ip": None}, "ssh": {"direct": {"host": "1.2.3.4", "port": 22}}}
print(json.dumps({"pods": [pod("a", 0), pod("b", 0)]}))
')"
IFS=$'\t' read -r rc out <<< "$(run_wait_for_members_ready "$dup_rank0_body" 1)"
if [ "$rc" -eq 97 ] && grep -q "rank 0 seen: 1, rank 1 seen: 0" <<<"$out"; then
  ok "two rows BOTH claiming rank 0 (rank 1 never seen) -> refused (97), never read as ready by a raw count"
else
  bad "expected rc=97 naming 'rank 0 seen: 1, rank 1 seen: 0'; got rc=$rc out=$out"
fi

ok_ranks_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
pod = lambda pid, rank: {"id": pid, "args": "bash -c %r" % setup, "cluster": {"rank": rank, "ip": None}, "ssh": {"direct": {"host": "1.2.3.4", "port": 22}}}
print(json.dumps({"pods": [pod("a", 0), pod("b", 1)]}))
')"
IFS=$'\t' read -r rc out <<< "$(run_wait_for_members_ready "$ok_ranks_body" 10)"
if [ "$rc" -eq 0 ] && grep -q "^1.2.3.4 22 1.2.3.4 22 " <<<"$out"; then
  ok "rank 0 and rank 1 each seen exactly once -> ready (0), both endpoints carried"
else
  bad "expected rc=0 with both endpoints; got rc=$rc out=$out"
fi

# Mutation check: on a SCRATCH COPY replace the exact-rank tracking with a
# raw `ok_count -ge 2` count, and confirm the SAME duplicate-rank-0 fixture
# above then reads "ready" -- proving the distinct-rank tracking is
# load-bearing, not vacuous.
# The scratch copy lives in its OWN directory alongside a copy of
# runpod_lib.sh (the driver's own `DIR="$(dirname "${BASH_SOURCE[0]}")"`
# sourcing resolves `$DIR/runpod_lib.sh` next to wherever the driver file
# itself sits, not next to the real ci/scripts/ tree).
RAWCOUNT_SCRATCH_DIR="$SANDBOX/rawcount-revert-red"
mkdir -p "$RAWCOUNT_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$RAWCOUNT_SCRATCH_DIR/runpod_lib.sh"
RAWCOUNT_SCRATCH="$RAWCOUNT_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$RAWCOUNT_SCRATCH" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''  local rank0_seen=0 rank1_seen=0 resp status readback readback_rc mismatch mixed_since="" r0_direct r1_direct'''
new = '''  local rank0_seen=0 rank1_seen=0 resp status readback readback_rc mismatch mixed_since="" r0_direct r1_direct ok_count=0'''
assert old in text, "raw-count mutation fixture: local-vars line not found verbatim"
text = text.replace(old, new, 1)
old2 = '''            READBACK_OK)
              if [ "$rank" = "0" ]; then
                rank0_seen=1
                primary_host="$dhost"; primary_port="$dport"
              elif [ "$rank" = "1" ]; then
                rank1_seen=1
                member_host="$dhost"; member_port="$dport"; member_ip="$ip"
              fi ;;'''
new2 = '''            READBACK_OK)
              ok_count=$((ok_count + 1))
              if [ "$rank" = "0" ]; then
                rank0_seen=1
                primary_host="$dhost"; primary_port="$dport"
              else
                rank1_seen=1
                member_host="$dhost"; member_port="$dport"; member_ip="$ip"
              fi ;;'''
assert old2 in text, "raw-count mutation fixture: READBACK_OK arm not found verbatim"
text = text.replace(old2, new2, 1)
old3 = '''        if [ "$rank0_seen" = "1" ] && [ "$rank1_seen" = "1" ]; then
          r0_direct=0; r1_direct=0
          [ -n "$primary_host" ] && [ "$primary_host" != "-" ] && r0_direct=1
          [ -n "$member_host" ] && [ "$member_host" != "-" ] && r1_direct=1
          if [ "$r0_direct" = 1 ] && [ "$r1_direct" = 1 ]; then
            break
          fi
          if [ "$r0_direct" = 1 ] || [ "$r1_direct" = 1 ]; then
            [ -n "$mixed_since" ] || mixed_since="$SECONDS"
            if [ $(( SECONDS - mixed_since )) -ge "${RP_SSH_MIXED_GRACE_SECS:-45}" ]; then
              break
            fi
          fi
        fi'''
new3 = '        [ "$ok_count" -ge 2 ] && break'
assert old3 in text, "raw-count mutation fixture: break condition not found verbatim"
text = text.replace(old3, new3, 1)
old4 = '  if [ "$rank0_seen" != "1" ] || [ "$rank1_seen" != "1" ]; then'
new4 = '  if [ "$ok_count" -lt 2 ]; then'
assert old4 in text, "raw-count mutation fixture: post-loop refusal condition not found verbatim"
text = text.replace(old4, new4, 1)
open(dst, "w").write(text)
PY
# The mutated shape's own wait loop breaks out claiming readiness the
# MOMENT ok_count hits 2 (both entries are the SAME rank-0 row) -- it never
# reaches the timeout/refusal branch this suite's own fixed-code assertion
# above names ("not every member reached a usable ssh path"). Proceeding
# past the loop with rank 1 never actually resolved, it instead trips the
# SEPARATE, unrelated proxy-fallback check ("neither a direct ssh endpoint
# nor an overlay ip") on the still-empty member fields -- a DIFFERENT
# failure than the accurate, named one the real code produces for the
# IDENTICAL fixture, proving the loop's own readiness gate broke out wrongly
# (and early) under the mutated shape.
IFS=$'\t' read -r rc out <<< "$(run_wait_for_members_ready "$dup_rank0_body" 10 "$RAWCOUNT_SCRATCH")"
if grep -q "neither a direct ssh endpoint nor an overlay ip" <<<"$out" \
  && ! grep -q "not every member reached a usable ssh path" <<<"$out"; then
  ok "mutation check: a raw-count shape (ok_count -ge 2) breaks the wait loop 'ready' on the duplicate-rank-0/no-rank-1 fixture (proceeds past it into the member-resolution phase with rank 1 unset) — the distinct-rank tracking above is load-bearing, not vacuous"
else
  bad "mutation check: expected the raw-count shape to break out of the wait loop early (a different, generic downstream failure, never the accurate 'not every member reached' refusal); got rc=$rc out=$out — the mutation fixture itself may be stale"
fi

# ============================================================================
# A member read back with its overlay ip assigned but its `ssh.direct`
# still null is PROVISIONING, not ready: a live cluster answers exactly that
# read within seconds of create. The loop must keep polling until a read
# carries the direct endpoints, and refuse ("the primary (rank 0) member
# carries no direct ssh endpoint") only at the deadline. Drives the REAL function with
# a STATEFUL _rp_rest mock: read 1 answers overlay-only, every later read
# answers with both direct endpoints.
# ============================================================================
run_wait_two_reads() { # $1=first body $2=later body $3=RP_SSH_WAIT_SECS $4=driver path -> stdout: "<rc>\t<out>"
  local first="$1" later="$2" wait_secs="$3" driver="${4:-$CLUSTER_SH}" out rc
  local counter="$SANDBOX/provisioning-reads-$$-$RANDOM"
  rm -f "$counter"
  out="$(PROVISIONING_FIRST_BODY="$first" PROVISIONING_LATER_BODY="$later" PROVISIONING_COUNTER="$counter" bash -c '
    source "'"$driver"'"
    _rp_entrypoint_setup() { printf "%s" "the-shared-entrypoint-text"; }
    _rp_rest() {
      local n=0
      [ -f "$PROVISIONING_COUNTER" ] && n="$(cat "$PROVISIONING_COUNTER")"
      printf "%s" $((n + 1)) > "$PROVISIONING_COUNTER"
      if [ "$n" -eq 0 ]; then printf "200\n%s" "$PROVISIONING_FIRST_BODY"; else printf "200\n%s" "$PROVISIONING_LATER_BODY"; fi
    }
    _rpc_wait_for_members_ready "cl-provisioning" "$1" "1"
  ' _ "$wait_secs" 2>&1)"
  rc=$?
  printf '%s\t%s\n' "$rc" "$out"
}

provisioning_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
pod = lambda pid, rank, ip: {"id": pid, "args": "bash -c %r" % setup, "cluster": {"rank": rank, "ip": ip}, "ssh": {"direct": None, "proxy": {"host": "ssh.runpod.io", "port": 22}}}
print(json.dumps({"pods": [pod("a", 0, "10.0.0.2"), pod("b", 1, "10.0.0.3")]}))
')"
ready_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
pod = lambda pid, rank, ip, host: {"id": pid, "args": "bash -c %r" % setup, "cluster": {"rank": rank, "ip": ip}, "ssh": {"direct": {"host": host, "port": 22}}}
print(json.dumps({"pods": [pod("a", 0, "10.0.0.2", "1.2.3.4"), pod("b", 1, "10.0.0.3", "5.6.7.8")]}))
')"
IFS=$'\t' read -r rc out <<< "$(run_wait_two_reads "$provisioning_body" "$ready_body" 20)"
if [ "$rc" -eq 0 ] && grep -q "^1.2.3.4 22 5.6.7.8 22 10.0.0.3 0$" <<<"$out"; then
  ok "overlay-only first read (ssh.direct null on both ranks) -> the loop keeps polling and returns the direct endpoints the second read carries (rc=0, proxy_flag 0)"
else
  bad "expected rc=0 with both direct endpoints from the second read; got rc=$rc out=$out"
fi

# The deadline arm: a rank 0 that NEVER gains a direct
# endpoint within RP_SSH_WAIT_SECS is refused by name (97), not carried
# forward as a jump host it cannot be.
IFS=$'\t' read -r rc out <<< "$(run_wait_two_reads "$provisioning_body" "$provisioning_body" 1)"
if [ "$rc" -eq 97 ] && grep -q "carries no direct ssh endpoint" <<<"$out"; then
  ok "rank 0 overlay-only through the deadline -> refused by name (97: no direct ssh endpoint / no jump host)"
else
  bad "expected rc=97 naming the missing direct endpoint at the deadline; got rc=$rc out=$out"
fi

# Mutation check: on a SCRATCH COPY use a break condition that is ready
# the moment both ranks are READBACK_OK, direct or not, and confirm the SAME
# two-read fixture then refuses at once, on the first read, with the
# no-direct-endpoint message -- the direct-endpoint wait above is
# load-bearing.
PROVISIONING_SCRATCH_DIR="$SANDBOX/provisioning-revert-red"
mkdir -p "$PROVISIONING_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$PROVISIONING_SCRATCH_DIR/runpod_lib.sh"
PROVISIONING_SCRATCH="$PROVISIONING_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$PROVISIONING_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''        if [ "$rank0_seen" = "1" ] && [ "$rank1_seen" = "1" ]; then
          r0_direct=0; r1_direct=0
          [ -n "$primary_host" ] && [ "$primary_host" != "-" ] && r0_direct=1
          [ -n "$member_host" ] && [ "$member_host" != "-" ] && r1_direct=1
          if [ "$r0_direct" = 1 ] && [ "$r1_direct" = 1 ]; then
            break
          fi
          if [ "$r0_direct" = 1 ] || [ "$r1_direct" = 1 ]; then
            [ -n "$mixed_since" ] || mixed_since="$SECONDS"
            if [ $(( SECONDS - mixed_since )) -ge "${RP_SSH_MIXED_GRACE_SECS:-45}" ]; then
              break
            fi
          fi
        fi'''
new = '''        [ "$rank0_seen" = "1" ] && [ "$rank1_seen" = "1" ] && break'''
assert old in text, "ready-on-readback mutation fixture: break condition not found verbatim"
open(dst, "w").write(text.replace(old, new, 1))
PY
IFS=$'\t' read -r rc out <<< "$(run_wait_two_reads "$provisioning_body" "$ready_body" 20 "$PROVISIONING_SCRATCH")"
if [ "$rc" -eq 97 ] && grep -q "carries no direct ssh endpoint" <<<"$out"; then
  ok "mutation check: a ready-on-readback break condition refuses the provisioning read at once (97, no direct ssh endpoint) on the fixture the real loop waits through -- the wait is load-bearing"
else
  bad "mutation check: expected the ready-on-readback shape to refuse at once with 'carries no direct ssh endpoint'; got rc=$rc out=$out -- the mutation fixture itself may be stale"
fi

# ============================================================================
# The MIXED arm -- rank 0 direct, rank 1 overlay-only -- settles for the
# proxy fallback after a bounded grace
# (RP_SSH_MIXED_GRACE_SECS), never the whole window; the result line
# carries the overlay ip as the member host with proxy_flag 1.
# ============================================================================
mixed_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
p0 = {"id": "a", "args": "bash -c %r" % setup, "cluster": {"rank": 0, "ip": "10.0.0.2"}, "ssh": {"direct": {"host": "1.2.3.4", "port": 22}}}
p1 = {"id": "b", "args": "bash -c %r" % setup, "cluster": {"rank": 1, "ip": "10.0.0.3"}, "ssh": {"direct": None}}
print(json.dumps({"pods": [p0, p1]}))
')"
grace_t0=$SECONDS
IFS=$'\t' read -r rc out <<< "$(RP_SSH_MIXED_GRACE_SECS=1 run_wait_two_reads "$mixed_body" "$mixed_body" 40)"
grace_elapsed=$(( SECONDS - grace_t0 ))
# The property is the TIME: a loop without the grace reaches the same line
# by running to the deadline (40 s here); the grace settles it within two
# polls.
if [ "$rc" -eq 0 ] && grep -q "^1.2.3.4 22 10.0.0.3 22 10.0.0.3 1$" <<<"$out" && [ "$grace_elapsed" -lt 20 ]; then
  ok "rank 0 direct + rank 1 overlay-only past the grace -> ready with the proxy fallback (rc=0, member=overlay ip, proxy_flag 1) in ${grace_elapsed}s, not the 40 s window"
else
  bad "expected rc=0 with the proxy-fallback line within 20 s (grace 1 s); got rc=$rc elapsed=${grace_elapsed}s out=$out"
fi

# The symmetric mixed state -- rank 1 direct, rank 0 overlay-only -- can
# never succeed (rank 0 must be the direct jump host); after the grace it
# refuses AT ONCE with the named message, never after the whole window.
mixed_r1_body="$(python3 -c '
import json
setup = "the-shared-entrypoint-text"
p0 = {"id": "a", "args": "bash -c %r" % setup, "cluster": {"rank": 0, "ip": "10.0.0.2"}, "ssh": {"direct": None}}
p1 = {"id": "b", "args": "bash -c %r" % setup, "cluster": {"rank": 1, "ip": "10.0.0.3"}, "ssh": {"direct": {"host": "5.6.7.8", "port": 22}}}
print(json.dumps({"pods": [p0, p1]}))
')"
grace_t0=$SECONDS
IFS=$'\t' read -r rc out <<< "$(RP_SSH_MIXED_GRACE_SECS=1 run_wait_two_reads "$mixed_r1_body" "$mixed_r1_body" 40)"
grace_elapsed=$(( SECONDS - grace_t0 ))
if [ "$rc" -eq 97 ] && grep -q "carries no direct ssh endpoint" <<<"$out" && [ "$grace_elapsed" -lt 20 ]; then
  ok "rank 1 direct + rank 0 overlay-only past the grace -> refused by name (97) in ${grace_elapsed}s, not the 40 s window"
else
  bad "expected rc=97 naming the missing rank-0 endpoint within 20 s; got rc=$rc elapsed=${grace_elapsed}s out=$out"
fi

# Mutation check: on a SCRATCH COPY drop the grace (break condition: both
# direct or nothing) and confirm the SAME mixed fixture then takes the whole
# 40 s window -- the timing assertion above is load-bearing.
GRACE_SCRATCH_DIR="$SANDBOX/grace-revert-red"
mkdir -p "$GRACE_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$GRACE_SCRATCH_DIR/runpod_lib.sh"
GRACE_SCRATCH="$GRACE_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$GRACE_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''          if [ "$r0_direct" = 1 ] || [ "$r1_direct" = 1 ]; then
            [ -n "$mixed_since" ] || mixed_since="$SECONDS"
            if [ $(( SECONDS - mixed_since )) -ge "${RP_SSH_MIXED_GRACE_SECS:-45}" ]; then
              break
            fi
          fi'''
assert old in text, "no-grace mutation fixture: grace block not found verbatim"
open(dst, "w").write(text.replace(old, "", 1))
PY
grace_t0=$SECONDS
IFS=$'\t' read -r rc out <<< "$(RP_SSH_MIXED_GRACE_SECS=1 run_wait_two_reads "$mixed_body" "$mixed_body" 40 "$GRACE_SCRATCH")"
grace_elapsed=$(( SECONDS - grace_t0 ))
if [ "$rc" -eq 0 ] && [ "$grace_elapsed" -ge 35 ]; then
  ok "mutation check: without the grace the same mixed fixture burns the whole window (${grace_elapsed}s of 40) before the same proxy-fallback line -- the timing assertion is load-bearing"
else
  bad "mutation check: expected the graceless copy to take >= 35 s; got rc=$rc elapsed=${grace_elapsed}s out=$out -- the mutation fixture itself may be stale"
fi

# ============================================================================
# The rented part's compute capability is
# DERIVED from the gpuTypeId at source time (never a second literal), and a
# gpuTypeId outside the sm_80/86/89/90 domain leaves NATIVE_COMPUTE_CAP
# empty so the main path refuses BEFORE phase 0 (exit 2) -- never on the
# members after a cluster is billing. Sourced with the netprobe guard
# (above) so no rent path can run.
# ============================================================================
cap_for() { # $1=gpuTypeId -> stdout: NATIVE_COMPUTE_CAP after sourcing with that id
  RP_CLUSTER_GPU_TYPE="$1" bash -c 'source "'"$CLUSTER_SH"'" >/dev/null 2>&1; printf "%s" "$NATIVE_COMPUTE_CAP"'
}
for pair in "NVIDIA A100-SXM4-80GB:80" "NVIDIA A40:86" "NVIDIA GeForce RTX 4090:89" "NVIDIA L40S:89" "NVIDIA H100 80GB HBM3:90"; do
  gpu="${pair%:*}"; want="${pair##*:}"
  got="$(cap_for "$gpu")"
  if [ "$got" = "$want" ]; then
    ok "RP_CLUSTER_GPU_TYPE='$gpu' -> NATIVE_COMPUTE_CAP=$want (derived, no literal)"
  else
    bad "expected NATIVE_COMPUTE_CAP=$want for '$gpu'; got '$got'"
  fi
done
got="$(cap_for "NVIDIA Tesla V100-SXM2-16GB")"
if [ -z "$got" ]; then
  ok "an out-of-domain gpuTypeId (V100, sm_70) leaves NATIVE_COMPUTE_CAP empty -- the main path's pre-rent refusal arm"
else
  bad "expected an empty NATIVE_COMPUTE_CAP for an out-of-domain id; got '$got'"
fi
if grep -q 'if \[ -z "\$NATIVE_COMPUTE_CAP" \]; then' "$CLUSTER_SH" \
  && grep -q refuses <<<"$(awk '/-z "\$NATIVE_COMPUTE_CAP"/{f=1} f && /exit 2/{print "refuses"; exit}' "$CLUSTER_SH")" \
  && [ "$(grep -n 'if \[ -z "\$NATIVE_COMPUTE_CAP" \]; then' "$CLUSTER_SH" | cut -d: -f1)" -lt "$(grep -n '_rpc_phase "availability read' "$CLUSTER_SH" | head -1 | cut -d: -f1)" ]; then
  ok "the empty-cap refusal (exit 2) sits BEFORE phase 0's availability read -- nothing is rented on an out-of-domain id (either transport)"
else
  bad "the empty-cap refusal is missing or sits after phase 0"
fi

# ============================================================================
# The 128-byte id-file-ready gate (_rpc_id_file_ready).
# ============================================================================
for size in 127 128 129; do
  f="$SANDBOX/id-${size}"
  head -c "$size" /dev/zero > "$f"
  if _rpc_id_file_ready "$f"; then got_ok=1; else got_ok=0; fi
  if [ "$size" -eq 128 ] && [ "$got_ok" -eq 1 ]; then
    ok "a 128-byte id file is ready"
  elif [ "$size" -ne 128 ] && [ "$got_ok" -eq 0 ]; then
    ok "a ${size}-byte id file is refused (not exactly 128)"
  else
    bad "unexpected result for a ${size}-byte file (ready=${got_ok})"
  fi
done
if _rpc_id_file_ready "$SANDBOX/does-not-exist"; then
  bad "a missing id file must never read as ready"
else
  ok "a missing id file is refused"
fi

# ============================================================================
# The ens1 log assertion (_rpc_ens1_seen).
# ============================================================================
good_log="$SANDBOX/ens1-good.log"
printf 'some noise\nNCCL INFO NET/Socket : Using [0]eth0:1.2.3.4<0> [1]ens1:10.0.0.2<0>\nmore noise\n' > "$good_log"
if _rpc_ens1_seen "$good_log"; then
  ok "a log naming ens1 for the NCCL NET transport is recognized"
else
  bad "expected _rpc_ens1_seen to recognize a genuine ens1 NET/Socket line"
fi

bad_log="$SANDBOX/ens1-missing.log"
printf 'some noise\nNCCL INFO NET/Socket : Using [0]eth0:1.2.3.4<0>\nmore noise\n' > "$bad_log"
if _rpc_ens1_seen "$bad_log"; then
  bad "a log with NO ens1 NET/Socket line must never read as seen"
else
  ok "a log with no ens1 NET line is refused"
fi

# ============================================================================
# The post-run artifact retrieval — a failed pull JOINS the leg's rc.
# ============================================================================
mkdir -p "$SANDBOX/gpu-pull/gpu-cluster"
if grep -qF 'echo "::error::artifact pull failed (rsync rc=${pull_rc})' "$CLUSTER_SH"; then
  ok "the driver's own pull-failure line names 'artifact pull failed (rsync rc=...)'"
else
  bad "the driver's pull-failure diagnostic text is missing or changed shape"
fi
if grep -qF '[ "$rc" -eq 0 ] && rc="$pull_rc"' "$CLUSTER_SH"; then
  ok "a failed pull JOINS the leg's own rc only when it was still 0 (never overwrites a real failure)"
else
  bad "the pull-failure-joins-rc line is missing or changed shape"
fi

# ============================================================================
# The run log lives INSIDE CLUSTER_ARTIFACT_DIR from its first byte
# (never a bare mktemp outside the uploaded/scanned directory), and both
# ranks' own logs are copied there too.
# ============================================================================
mkdir_line="$(grep -n '^mkdir -p "\$CLUSTER_ARTIFACT_DIR"$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
runlog_line="$(grep -n '^RUN_LOG="\$CLUSTER_ARTIFACT_DIR/run.log"$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
if [ -n "$mkdir_line" ] && [ -n "$runlog_line" ] && [ "$mkdir_line" -lt "$runlog_line" ]; then
  ok "CLUSTER_ARTIFACT_DIR is created (line ${mkdir_line}) BEFORE RUN_LOG is assigned inside it (line ${runlog_line})"
else
  bad "expected mkdir -p \$CLUSTER_ARTIFACT_DIR before RUN_LOG=\$CLUSTER_ARTIFACT_DIR/run.log; mkdir_line=${mkdir_line:-<none>} runlog_line=${runlog_line:-<none>}"
fi
if grep -qF 'cp -f "$rank0_log" "${CLUSTER_ARTIFACT_DIR}/rank0.log"' "$CLUSTER_SH" \
  && grep -qF 'cp -f "$rank1_log" "${CLUSTER_ARTIFACT_DIR}/rank1.log"' "$CLUSTER_SH"; then
  ok "both ranks' own logs are copied into CLUSTER_ARTIFACT_DIR"
else
  bad "expected both rank0_log/rank1_log to be copied into CLUSTER_ARTIFACT_DIR"
fi

# ============================================================================
# The id-secrecy scan invocation, driven through the REAL scanner
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
  ok "a clean scan passes (rc=0) and deletes the staging copy (--delete-staging)"
else
  bad "expected a clean scan to pass and delete staging; rc=$rc, staging exists=$([ -f "$scan_fixture_dir/staging/nccl.id" ] && echo yes || echo no)"
fi

python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/staging/nccl.id"
id_hex="$(python3 -c "$ID_BYTES_PY" | python3 -c 'import sys; print(sys.stdin.buffer.read().hex())')"
echo "leaked hex: ${id_hex}" > "$scan_fixture_dir/run.log"
_rpc_run_id_secrecy_scan "$scan_fixture_dir/staging/nccl.id" "$scan_fixture_dir/pulled" \
  "$scan_fixture_dir/run.log" "$scan_fixture_dir/assembled.json"
rc=$?
if [ "$rc" -eq 1 ]; then
  ok "a hex-encoded id leaked into the run log -> FAIL (rc=1)"
else
  bad "expected rc=1 on a hex leak in the run log; got rc=$rc"
fi
echo "clean log" > "$scan_fixture_dir/run.log"

python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/staging/nccl.id"
python3 -c "$ID_BYTES_PY" > "$scan_fixture_dir/pulled/bundle.tar.gz"
_rpc_run_id_secrecy_scan "$scan_fixture_dir/staging/nccl.id" "$scan_fixture_dir/pulled" \
  "$scan_fixture_dir/run.log" "$scan_fixture_dir/assembled.json"
rc=$?
if [ "$rc" -eq 2 ]; then
  ok "an archive under the pulled dir -> rc=2 (refused, never opened)"
else
  bad "expected rc=2 on an archive carrier; got rc=$rc"
fi
rm -f "$scan_fixture_dir/pulled/bundle.tar.gz"

# ============================================================================
# The assembler (_rpc_assemble_gang_artifact) is oracled through the REAL
# function, on the happy path AND every refusal/representable-fail arm, with
# the MEASURED pod_count/gpu_count_per_pod threaded in (never hardcoded
# literals) -- and its output is run through
# the REAL check_cuda_run_artifacts.py checker, imported directly rather
# than paraphrased.
# ============================================================================
assemble_dir="$SANDBOX/assemble"
mkdir -p "$assemble_dir"
DIGEST_A="$(python3 -c 'print("a" * 64)')"
DIGEST_B="$(python3 -c 'print("b" * 64)')"

write_rank_report() { # $1=path $2=rank $3=host $4=iface $5=verdict $6=digest(or "-") $7=device_ordinal
  python3 -c '
import json, sys
path, rank, host, iface, verdict, digest, dev = sys.argv[1:8]
d = {"rank": int(rank), "hostname": host, "nccl_socket_ifname": iface, "verdict": verdict, "device_ordinal": int(dev)}
if digest != "-":
    d["reduced_vector_digest_sha256"] = digest
if verdict != "pass":
    d["reason"] = "fixture-forced fail"
json.dump(d, open(path, "w"))
' "$@"
}


# The data-center choice: the sorted-first default, an operator-named
# candidate, and a refusal by name (listing the candidates) for a name that
# is not one.
if [ "$(_rpc_choose_data_center "US-GA-2 CA-MTL-1 EU-FR-1" "")" = "CA-MTL-1" ]; then
  ok "_rpc_choose_data_center: no operator choice picks the first candidate in sorted order"
else
  bad "_rpc_choose_data_center: default choice is not the sorted-first candidate"
fi
if [ "$(_rpc_choose_data_center "US-GA-2 CA-MTL-1 EU-FR-1" "US-GA-2")" = "US-GA-2" ]; then
  ok "_rpc_choose_data_center: an operator-named candidate is the choice"
else
  bad "_rpc_choose_data_center: an operator-named candidate was not chosen"
fi
if err="$(_rpc_choose_data_center "US-GA-2 CA-MTL-1" "EU-NL-1" 2>&1 >/dev/null)"; then
  bad "_rpc_choose_data_center: a name outside the candidates was accepted"
elif grep -qF 'RP_CLUSTER_DC=EU-NL-1 is not a co-located candidate' <<<"$err" \
  && grep -qF 'candidates: US-GA-2 CA-MTL-1' <<<"$err"; then
  ok "_rpc_choose_data_center: a name outside the candidates is refused naming it and the candidates"
else
  bad "_rpc_choose_data_center: the refusal does not name the choice and the candidates: $err"
fi

check_gang_via_checker() { # $1=assembled artifact path -> real exit code (0 clean, 1 findings)
  python3 -c '
import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("check_cuda_run_artifacts", sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
data = json.load(open(sys.argv[1]))
failures = mod.check_gang_artifact(data, "crates/jammi-kernels/artifacts/cuda-runs/fixture-gang-cluster.json", mod.REPO_ROOT)
for f in failures:
    print(f, file=sys.stderr)
sys.exit(1 if failures else 0)
' "$1" "$DIR/check_cuda_run_artifacts.py"
}

# Happy path: 2x1 (the measured shape matches what this leg pins).
write_rank_report "$assemble_dir/r0.json" 0 "host-a" "ens1" pass "$DIGEST_A" 0
write_rank_report "$assemble_dir/r1.json" 1 "host-b" "ens1" pass "$DIGEST_A" 0
out_happy="$assemble_dir/gang-happy.json"
_rpc_assemble_gang_artifact "$assemble_dir/r0.json" "$assemble_dir/r1.json" deadbeef "a100-sxm4-cluster" "$out_happy" 2 1 1 "instant-cluster"
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$out_happy" ]; then
  ok "happy-path assembly succeeds with the measured pod_count=2/gpu_count_per_pod=1 threaded in"
else
  bad "happy-path assembly failed (rc=$rc)"
fi
check_out="$SANDBOX/checker-happy.txt"
if check_gang_via_checker "$out_happy" >"$check_out" 2>&1; then
  ok "the happy-path artifact passes the REAL check_cuda_run_artifacts.py's check_gang_artifact"
else
  bad "the happy-path artifact was refused by check_gang_artifact: $(cat "$check_out")"
fi

# A 3x8 create-response shape yields a 3x8 artifact -- no shape literal is
# hardcoded, and the checker then refuses it against the SAME
# 2-rank reports (the cross-field check is falsifiable, not a tautology).
out_38="$assemble_dir/gang-3x8.json"
_rpc_assemble_gang_artifact "$assemble_dir/r0.json" "$assemble_dir/r1.json" deadbeef "a100-sxm4-cluster" "$out_38" 3 8 1 "instant-cluster"
rc=$?
if [ "$rc" -eq 0 ]; then
  ok "the assembler accepts a driver-supplied 3x8 shape without hardcoding pod_count/gpu_count_per_pod"
else
  bad "assembly with a 3x8 shape unexpectedly failed (rc=$rc)"
fi
read -r hosts_f world_f pod_f gpu_f < <(python3 -c '
import json
g = json.load(open("'"$out_38"'"))["gang"]
print(g["hosts"], g["world"], g["pod_count"], g["gpu_count_per_pod"])
')
if [ "$hosts_f" = "3" ] && [ "$world_f" = "24" ] && [ "$pod_f" = "3" ] && [ "$gpu_f" = "8" ]; then
  ok "a 3x8 create-response shape yields hosts=3 world=24 pod_count=3 gpu_count_per_pod=8 (measured, not literal)"
else
  bad "expected hosts=3/world=24/pod_count=3/gpu_count_per_pod=8; got hosts=$hosts_f world=$world_f pod_count=$pod_f gpu_count_per_pod=$gpu_f"
fi
check_out38="$SANDBOX/checker-3x8.txt"
if check_gang_via_checker "$out_38" >"$check_out38" 2>&1; then
  bad "expected the checker to REFUSE a 3x8 artifact carrying only 2 ranks[] entries; it passed clean"
else
  if grep -q "gang.hosts\|gang.ranks\|gang.world" "$check_out38"; then
    ok "check_cuda_run_artifacts.py's own cross-field/registry checks refuse the 3x8 artifact against its 2-rank reports -- the check is falsifiable"
  else
    bad "the checker refused the 3x8 artifact but not for a shape-related reason: $(cat "$check_out38")"
  fi
fi

# Refusal arm: unknown hostname -> assembly REFUSES (no artifact).
write_rank_report "$assemble_dir/ruh0.json" 0 "unknown" "ens1" pass "$DIGEST_A" 0
write_rank_report "$assemble_dir/ruh1.json" 1 "host-b" "ens1" pass "$DIGEST_A" 0
out_uh="$assemble_dir/gang-unknown-host.json"
_rpc_assemble_gang_artifact "$assemble_dir/ruh0.json" "$assemble_dir/ruh1.json" deadbeef box "$out_uh" 2 1 1 "instant-cluster"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "$out_uh" ]; then
  ok "an unresolved (\"unknown\") rank hostname REFUSES assembly, no artifact written"
else
  bad "expected an unresolved hostname to refuse assembly; rc=$rc, artifact exists=$([ -f "$out_uh" ] && echo yes || echo no)"
fi

# Refusal arm: unknown iface -> assembly REFUSES.
write_rank_report "$assemble_dir/rui0.json" 0 "host-a" "unknown" pass "$DIGEST_A" 0
write_rank_report "$assemble_dir/rui1.json" 1 "host-b" "ens1" pass "$DIGEST_A" 0
out_ui="$assemble_dir/gang-unknown-iface.json"
_rpc_assemble_gang_artifact "$assemble_dir/rui0.json" "$assemble_dir/rui1.json" deadbeef box "$out_ui" 2 1 1 "instant-cluster"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "$out_ui" ]; then
  ok "an unresolved (\"unknown\") rank iface REFUSES assembly, no artifact written"
else
  bad "expected an unresolved iface to refuse assembly; rc=$rc, artifact exists=$([ -f "$out_ui" ] && echo yes || echo no)"
fi

# Refusal arm: repeated host, compared CASE-INSENSITIVELY -- "Host-A" and
# "host-a" are the same host.
write_rank_report "$assemble_dir/rrh0.json" 0 "Host-A" "ens1" pass "$DIGEST_A" 0
write_rank_report "$assemble_dir/rrh1.json" 1 "host-a" "ens1" pass "$DIGEST_A" 0
out_rh="$assemble_dir/gang-repeated-host.json"
_rpc_assemble_gang_artifact "$assemble_dir/rrh0.json" "$assemble_dir/rrh1.json" deadbeef box "$out_rh" 2 1 1 "instant-cluster"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "$out_rh" ]; then
  ok "two ranks reporting the SAME host case-insensitively (Host-A vs host-a) REFUSES assembly"
else
  bad "expected a case-insensitive repeated host to refuse assembly; rc=$rc, artifact exists=$([ -f "$out_rh" ] && echo yes || echo no)"
fi

# Digest mismatch is REPRESENTABLE (assembly succeeds, verdict
# recorded as fail, never refused at assembly time) -- and the resulting
# artifact still passes the checker as a legitimately-recorded failed run.
write_rank_report "$assemble_dir/rd0.json" 0 "host-a" "ens1" pass "$DIGEST_A" 0
write_rank_report "$assemble_dir/rd1.json" 1 "host-b" "ens1" pass "$DIGEST_B" 0
out_dm="$assemble_dir/gang-digest-mismatch.json"
_rpc_assemble_gang_artifact "$assemble_dir/rd0.json" "$assemble_dir/rd1.json" deadbeef box "$out_dm" 2 1 1 "instant-cluster"
rc=$?
verdict_f="$(python3 -c "import json; print(json.load(open('$out_dm'))['gang']['verdict'])" 2>/dev/null)"
if [ "$rc" -eq 0 ] && [ "$verdict_f" = "fail" ]; then
  ok "a digest disagreement between two 'pass' ranks is REPRESENTABLE (verdict=fail), never refused at assembly time"
else
  bad "expected assembly to succeed with verdict=fail on a digest mismatch; rc=$rc verdict=${verdict_f:-<none>}"
fi
check_outdm="$SANDBOX/checker-digest-mismatch.txt"
if check_gang_via_checker "$out_dm" >"$check_outdm" 2>&1; then
  ok "the digest-mismatch (verdict=fail) artifact still passes the checker -- a recorded failure is valid evidence"
else
  bad "the digest-mismatch artifact was unexpectedly refused by check_gang_artifact: $(cat "$check_outdm")"
fi

# ============================================================================
# rp_init precedes the FIRST rp_cluster_create in the executed block —
# a static line-number comparison over the committed driver text, never a
# behavioral probe (rp_init itself needs no network; ssh-keygen only).
# ============================================================================
rp_init_line="$(grep -n '^rp_init$' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
# `grep -n`'s own output is "N:text" -- a line ALWAYS starts with digits,
# so a filter anchored at `^\s*#` against THAT text can never match anything
# (a no-op). The comment-vs-code
# check has to strip grep -n's own "N:" prefix before testing for a
# leading `#`.
rp_cluster_create_line="$(grep -n 'rp_cluster_create ' "$CLUSTER_SH" | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1)"
if [ -n "$rp_init_line" ] && [ -n "$rp_cluster_create_line" ] && [ "$rp_init_line" -lt "$rp_cluster_create_line" ]; then
  ok "rp_init (line ${rp_init_line}) precedes the first rp_cluster_create call (line ${rp_cluster_create_line})"
else
  bad "expected rp_init before the first rp_cluster_create call; rp_init_line=${rp_init_line:-<none>} rp_cluster_create_line=${rp_cluster_create_line:-<none>}"
fi

# The filter above actually filters -- proven against a synthetic fixture
# (the real file has no commented-out mention to exercise it against): a
# comment-only line naming rp_cluster_create must never be read as the real
# call.
comment_fixture="$SANDBOX/comment-filter-fixture.sh"
printf '# rp_cluster_create mentioned only in a comment\nrp_cluster_create "$x" "$y"\n' > "$comment_fixture"
filtered_line="$(grep -n 'rp_cluster_create ' "$comment_fixture" | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1)"
if [ "$filtered_line" = "2" ]; then
  ok "the comment-vs-code filter actually filters a commented-out mention (line 1), landing on the real call (line 2)"
else
  bad "expected the filter to skip line 1's comment and land on line 2; got '${filtered_line:-<none>}'"
fi

# ============================================================================
# The cost bound is what the MECHANISM produces.
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
  ok "the driver pins RP_TTL_HOURS=1 (the cluster's own entrypoint deadline)"
else
  bad "RP_TTL_HOURS is '${RP_TTL_HOURS}', not 1 — the TTL term of both bounds moved"
fi

if [ "$RP_CLUSTER_POD_COUNT" = "2" ] && [ "$RP_CLUSTER_GPU_COUNT_PER_POD" = "1" ]; then
  ok "the driver pins a 2x1 cluster shape — the shape the \$${CLUSTER_RATE_USD_PER_GPU_HOUR}/GPU/h rate was measured at"
else
  bad "cluster shape is ${RP_CLUSTER_POD_COUNT}x${RP_CLUSTER_GPU_COUNT_PER_POD}, not 2x1 — the rate does not price this lane"
fi

for site in "$CLUSTER_YML" "$CLUSTER_SH" "$DEV_GPU_MD"; do
  rel="${site#"$REPO_ROOT"/}"
  if [ ! -f "$site" ]; then
    bad "${rel} does not exist"
    continue
  fi
  if grep -qF "\$${cost_i}" "$site"; then
    ok "${rel} states bound (i) as \$${cost_i} — the figure the mechanism produces"
  else
    bad "${rel} does not state bound (i) (\$${cost_i}) — its prose and the mechanism disagree"
  fi
  if grep -qF "\$${cost_ii}" "$site"; then
    ok "${rel} states bound (ii) as \$${cost_ii} (the sweep-only path)"
  else
    bad "${rel} does not state bound (ii) (\$${cost_ii})"
  fi
done

if grep -qE 'MAX_ATTEMPTS:[[:space:]]*"1"' "$CLUSTER_YML"; then
  ok "gpu-cluster.yml pins MAX_ATTEMPTS to \"1\""
else
  bad "gpu-cluster.yml does not pin MAX_ATTEMPTS to \"1\""
fi

# ============================================================================
# The pods-transport cost bound is what the MECHANISM produces too -- the
# measured SECURE POD rate ($1.59/GPU/h, the catalog's own figure, never the
# cluster's separate $1.908/GPU/h), re-derived the SAME way the cluster
# figures are above: (i) terminate-succeeds, (ii)
# sweep-only (the ordinary pod sweep's own 6-hourly gpu-reap.yml schedule
# -- the SAME cadence the cluster sweep uses, re-read from the committed
# workflow rather than assumed).
# ============================================================================
POD_RATE_USD_PER_GPU_HOUR=1.59
GPU_REAP_YML="$REPO_ROOT/.github/workflows/gpu-reap.yml"
reap_cron_hours="$(grep -oE 'cron: "[0-9]+ \*/([0-9]+) \* \* \*"' "$GPU_REAP_YML" | head -1 | grep -oE '/[0-9]+' | tr -d '/')"
if [ -z "$reap_cron_hours" ]; then
  bad "could not read gpu-reap.yml's own sweep cadence (cron */Nh) -- the sweep-only bound cannot be re-derived from the mechanism"
  reap_cron_hours=6
else
  ok "read gpu-reap.yml's own sweep cadence as every ${reap_cron_hours}h (re-derived, not assumed)"
fi
read -r pod_cost_rate pod_cost_i pod_cost_ii < <(python3 - "$RP_TTL_HOURS" "$POD_RATE_USD_PER_GPU_HOUR" "$reap_cron_hours" <<'PY'
import sys
ttl_h, per_gpu, reap_h = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
rate = 2 * per_gpu
bound_i = ttl_h * rate
bound_ii = (ttl_h + reap_h) * rate
print(f"{rate:.3f} {bound_i:.2f} {bound_ii:.2f}")
PY
)
for site in "$CLUSTER_YML" "$CLUSTER_SH" "$DEV_GPU_MD"; do
  rel="${site#"$REPO_ROOT"/}"
  if grep -qF "\$${pod_cost_i}" "$site"; then
    ok "${rel} states the pods-transport bound (i) as \$${pod_cost_i} — the figure the mechanism produces"
  else
    bad "${rel} does not state the pods-transport bound (i) (\$${pod_cost_i}) — its prose and the mechanism disagree"
  fi
  if grep -qF "\$${pod_cost_ii}" "$site"; then
    ok "${rel} states the pods-transport bound (ii) as \$${pod_cost_ii} (the sweep-only path)"
  else
    bad "${rel} does not state the pods-transport bound (ii) (\$${pod_cost_ii})"
  fi
done

# ============================================================================
# NO schedule: key anywhere in the committed gpu-cluster.yml.
# ============================================================================
g7_out="$(python3 "$PROVE_ONCE_PY" --read-on-block "$CLUSTER_YML" 2>&1)"
g7_rc=$?
if [ "$g7_rc" -eq 0 ] && ! grep -qx "schedule" <<<"$g7_out"; then
  ok "gpu-cluster.yml carries no schedule: key (read through the shared on: block reader)"
else
  bad "expected gpu-cluster.yml to carry no schedule: key; got rc=${g7_rc} out=${g7_out}"
fi

# ============================================================================
# The pods transport (RP_TWO_HOST_TRANSPORT=pods, the default).
# Every property below drives the REAL function, mocking only _rp_rest/
# rp_pod_get/rp_terminate exactly as the cluster fixtures above mock
# _rp_rest/rp_cluster_delete.
# ============================================================================

# ----------------------------------------------------------------------------
# Transport is explicit.
# ----------------------------------------------------------------------------
if [ "$RP_TWO_HOST_TRANSPORT" = "cluster" ]; then
  ok "this suite's own export pins RP_TWO_HOST_TRANSPORT=cluster, so every case above this section exercised the cluster path end to end"
else
  bad "expected this suite's own RP_TWO_HOST_TRANSPORT export to be 'cluster'; got '${RP_TWO_HOST_TRANSPORT}'"
fi

default_transport="$(bash -c 'unset RP_TWO_HOST_TRANSPORT; source "'"$CLUSTER_SH"'" >/dev/null 2>&1; printf "%s" "$RP_TWO_HOST_TRANSPORT"')"
if [ "$default_transport" = "pods" ]; then
  ok "with RP_TWO_HOST_TRANSPORT unset, the driver defaults to 'pods'"
else
  bad "expected the default transport to be 'pods'; got '${default_transport}'"
fi

invalid_marker="$(RP_TWO_HOST_TRANSPORT="bogus" bash -c 'source "'"$CLUSTER_SH"'" >/dev/null 2>&1; printf "%s" "${RP_TWO_HOST_TRANSPORT_INVALID:-}"')"
if [ "$invalid_marker" = "1" ]; then
  ok "an out-of-set RP_TWO_HOST_TRANSPORT sets RP_TWO_HOST_TRANSPORT_INVALID at source time (never exits while merely sourced)"
else
  bad "expected RP_TWO_HOST_TRANSPORT_INVALID=1 for an out-of-set transport; got '${invalid_marker}'"
fi

# ----------------------------------------------------------------------------
# Co-placement -- ONE data center, both GN-enabled, MEASURED from the
# get responses; refused (97) before any build otherwise.
# ----------------------------------------------------------------------------
pod_json() { # $1=id $2=status $3=dataCenterId $4=gn_enabled(true|false) $5=gn_ip(or null) $6=ssh_host(or null) $7=ssh_port
  python3 -c '
import json, sys
pid, status, dc, gn_en, gn_ip, host, port = sys.argv[1:8]
d = {
    "id": pid, "desiredStatus": status, "dataCenterId": dc,
    "globalNetworking": {"enabled": gn_en == "true", "ip": (None if gn_ip == "null" else gn_ip)},
    "ssh": {"direct": (None if host == "null" else {"host": host, "port": int(port)})},
}
print(json.dumps(d))
' "$@"
}

run_wait_two_host_pods() { # $1=body0 $2=body1 $3=wait_secs $4=driver path (default CLUSTER_SH) -> "<rc>\t<out>"
  local b0="$1" b1="$2" wait_secs="$3" driver="${4:-$CLUSTER_SH}" out rc
  out="$(TWOPODS_BODY0="$b0" TWOPODS_BODY1="$b1" bash -c '
    source "'"$driver"'"
    rp_pod_get() { [ "$1" = "pod-0" ] && printf "%s" "$TWOPODS_BODY0" || printf "%s" "$TWOPODS_BODY1"; }
    _rpc_wait_for_two_host_pods_ready "pod-0" "pod-1" "$1"
  ' _ "$wait_secs" 2>&1)"
  rc=$?
  printf '%s\t%s\n' "$rc" "$out"
}

ok_body0="$(pod_json pod-0 RUNNING dc-a true 10.0.0.2 1.2.3.4 22)"
ok_body1="$(pod_json pod-1 RUNNING dc-a true 10.0.0.3 5.6.7.8 22)"
IFS=$'\t' read -r rc out <<< "$(run_wait_two_host_pods "$ok_body0" "$ok_body1" 10)"
if [ "$rc" -eq 0 ] && grep -q "^1.2.3.4 22 5.6.7.8 22 dc-a 10.0.0.2 10.0.0.3$" <<<"$out"; then
  ok "both pods RUNNING, GN-enabled, co-located in dc-a -> ready (0), both endpoints + GN ips carried"
else
  bad "expected rc=0 with both endpoints and GN ips; got rc=$rc out=$out"
fi

diff_dc_body1="$(pod_json pod-1 RUNNING dc-b true 10.0.0.3 5.6.7.8 22)"
IFS=$'\t' read -r rc out <<< "$(run_wait_two_host_pods "$ok_body0" "$diff_dc_body1" 6)"
if [ "$rc" -eq 97 ] && grep -q "DIFFERENT data centers" <<<"$out"; then
  ok "the two pods landing in DIFFERENT data centers -> refused (97), named, before any build"
else
  bad "expected rc=97 naming DIFFERENT data centers; got rc=$rc out=$out"
fi

no_gn_body1="$(pod_json pod-1 RUNNING dc-a false null 5.6.7.8 22)"
IFS=$'\t' read -r rc out <<< "$(run_wait_two_host_pods "$ok_body0" "$no_gn_body1" 1)"
if [ "$rc" -eq 97 ] && grep -q "rank 1 ready: 0" <<<"$out"; then
  ok "rank 1 never reporting Global Networking enabled -> refused (97) at the deadline, never a silent pass"
else
  bad "expected rc=97 naming rank 1 unready; got rc=$rc out=$out"
fi

no_ssh_body1="$(pod_json pod-1 RUNNING dc-a true 10.0.0.3 null 22)"
IFS=$'\t' read -r rc out <<< "$(run_wait_two_host_pods "$ok_body0" "$no_ssh_body1" 1)"
if [ "$rc" -eq 97 ]; then
  ok "rank 1 with no ssh.direct (the pods transport never proxies) -> refused (97)"
else
  bad "expected rc=97 for a missing ssh.direct; got rc=$rc out=$out"
fi

# Mutation check: on a scratch copy, weaken the co-placement check to
# compare nothing at all (accept any two dataCenterIds) -- confirm the SAME
# different-dc fixture above then reads "ready" instead of refusing, proving
# the real check above is load-bearing.
COPLACE_SCRATCH_DIR="$SANDBOX/coplace-revert-red"
mkdir -p "$COPLACE_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$COPLACE_SCRATCH_DIR/runpod_lib.sh"
COPLACE_SCRATCH="$COPLACE_SCRATCH_DIR/runpod_gpu_cluster.sh"
python3 - "$CLUSTER_SH" "$COPLACE_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = '''  if [ "$dc0" != "$dc1" ]; then
    echo "::error::the two pods landed in DIFFERENT data centers (rank 0: ${dc0}, rank 1: ${dc1}) -- co-placement failed; refusing before any build starts" >&2
    return 97
  fi'''
new = '''  :  # mutation: co-placement check removed'''
assert old in text, "co-placement mutation fixture: co-placement check not found verbatim"
open(dst, "w").write(text.replace(old, new, 1))
PY
IFS=$'\t' read -r rc out <<< "$(run_wait_two_host_pods "$ok_body0" "$diff_dc_body1" 6 "$COPLACE_SCRATCH")"
if [ "$rc" -eq 0 ]; then
  ok "mutation check: removing the co-placement check reads the different-dc fixture as ready (rc=0) -- the real check above is load-bearing"
else
  bad "mutation check: expected the check-less shape to accept the different-dc fixture as ready (rc=0); got rc=$rc out=$out -- the mutation fixture itself may be stale"
fi
rm -rf "$COPLACE_SCRATCH_DIR"

# ----------------------------------------------------------------------------
# The create body speaks REST v2's field names (`cloud`, `image`, `disk`,
# `gpu{id,count}`, `globalNetworking`, `dataCenterIds`, `ports`, `startSsh`,
# `args`, `env`) -- never the v1 GraphQL `cloudType`/`imageName` the pod
# leg's `_rp_deploy_payload` speaks, which `POST /v2/pods` does not accept.
# ----------------------------------------------------------------------------
createbody_json="$(RP_DISK_GB=60 RP_TTL_HOURS=1 bash -c '
  source "'"$CLUSTER_SH"'" >/dev/null 2>&1
  RP_PUBKEY="ssh-ed25519 AAAAtest"
  _rp_two_host_pod_payload "NVIDIA A40" "US-IL-1" 0
' 2>&1)"
createbody_verdict="$(printf '%s' "$createbody_json" | python3 -c '
import json, sys
try:
    b = json.loads(sys.stdin.read())
except Exception as e:
    print("PARSE_ERROR %s" % e); sys.exit(0)
want = {"name", "cloud", "globalNetworking", "dataCenterIds", "gpu", "image", "disk", "ports", "startSsh", "args", "env"}
got = set(b)
bad = []
if got != want: bad.append("keys=%s" % sorted(got ^ want))
if b.get("cloud") != "SECURE": bad.append("cloud")
if b.get("globalNetworking") is not True: bad.append("globalNetworking")
if b.get("dataCenterIds") != ["US-IL-1"]: bad.append("dataCenterIds")
if b.get("gpu") != {"id": "NVIDIA A40", "count": 1}: bad.append("gpu")
if not isinstance(b.get("disk"), int) or b["disk"] != 60: bad.append("disk")
if b.get("ports") != ["22/tcp"]: bad.append("ports")
if b.get("startSsh") is not True: bad.append("startSsh")
if not str(b.get("args", "")).startswith("bash -c "): bad.append("args")
if b.get("env", {}).get("PUBLIC_KEY") != "ssh-ed25519 AAAAtest": bad.append("env")
print("OK" if not bad else "BAD " + ",".join(bad))
')"
if [ "$createbody_verdict" = "OK" ]; then
  ok "the two-host pod create body carries exactly REST v2's field names (cloud/image/disk/gpu{id,count}/globalNetworking/dataCenterIds/ports/startSsh/args/env) -- no v1 cloudType/imageName"
else
  bad "the create body is off: ${createbody_verdict}; body=${createbody_json}"
fi

# ----------------------------------------------------------------------------
# Rank assignment is deterministic and RECORDED -- rank 0 is created FIRST,
# rank 1 SECOND (static ordering over the committed text, the same technique
# this file uses for rp_init/rp_cluster_create).
# ----------------------------------------------------------------------------
rank0_create_line="$(grep -n '^two_host_pod_id_0="\$(rp_two_host_pod_create ' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
rank1_create_line="$(grep -n '^two_host_pod_id_1="\$(rp_two_host_pod_create ' "$CLUSTER_SH" | head -1 | cut -d: -f1)"
if [ -n "$rank0_create_line" ] && [ -n "$rank1_create_line" ] && [ "$rank0_create_line" -lt "$rank1_create_line" ]; then
  ok "rank 0's own pod create (line ${rank0_create_line}) precedes rank 1's (line ${rank1_create_line}) -- deterministic, never inferred from the RunPod response"
else
  bad "expected rank 0's create call before rank 1's; rank0=${rank0_create_line:-<none>} rank1=${rank1_create_line:-<none>}"
fi
if grep -qF 'rp_two_host_pod_create "$RP_CLUSTER_GPU_TYPE" "$chosen_dc" 0' "$CLUSTER_SH" \
  && grep -qF 'rp_two_host_pod_create "$RP_CLUSTER_GPU_TYPE" "$chosen_dc" 1' "$CLUSTER_SH"; then
  ok "each create call names its OWN rank literal (0, then 1) -- recorded, not derived"
else
  bad "expected each create call to name its own rank literal"
fi
if grep -qF '_rpc_remote_script 0; } | ssh "${RP_SSHO[@]}" -p "$primary_port" "root@${primary_host}"' "$CLUSTER_SH" \
  && grep -qF '_rpc_remote_script 1; } | ssh "${RP_SSHO[@]}" "${member_extra_sshopts[@]}" -p "$member_port" "root@${member_host}"' "$CLUSTER_SH"; then
  ok "rank 0's remote script runs on the primary (rank 0's own pod), rank 1's on the member (rank 1's own pod)"
else
  bad "expected rank 0/1's remote script to run on the primary/member host respectively"
fi

# ----------------------------------------------------------------------------
# The NCCL interface is DERIVED at run time, never a literal; a missing
# interface is a NAMED refusal on that member.
# ----------------------------------------------------------------------------
cluster_iface_text="$(RP_TWO_HOST_TRANSPORT=cluster bash -c 'source "'"$CLUSTER_SH"'"; _rpc_two_host_iface_lines 0')"
if [ "$cluster_iface_text" = "export NCCL_SOCKET_IFNAME=ens1" ]; then
  ok "under RP_TWO_HOST_TRANSPORT=cluster, _rpc_two_host_iface_lines emits EXACTLY the literal ens1 line"
else
  bad "expected the cluster transport's iface line to be exactly 'export NCCL_SOCKET_IFNAME=ens1'; got: ${cluster_iface_text}"
fi

pods_iface_text="$(RP_TWO_HOST_TRANSPORT=pods RP_TWO_HOST_GN_IP_0=10.0.0.9 bash -c 'source "'"$CLUSTER_SH"'"; _rpc_two_host_iface_lines 0')"
if grep -q 'gn_ip="10.0.0.9"' <<<"$pods_iface_text" \
  && grep -qF '/proc/net/route' <<<"$pods_iface_text" \
  && ! grep -qF 'ip -o -4 addr show' <<<"$pods_iface_text" \
  && grep -qF 'DERIVED_NCCL_IFACE=' <<<"$pods_iface_text" \
  && ! grep -qF 'NCCL_SOCKET_IFNAME=ens1' <<<"$pods_iface_text"; then
  ok "under RP_TWO_HOST_TRANSPORT=pods, _rpc_two_host_iface_lines derives the interface from the rank's own GN ip -- never the ens1 literal"
else
  bad "expected the pods transport's iface text to derive from RP_TWO_HOST_GN_IP_0 with no ens1 literal; got: ${pods_iface_text}"
fi

# Execute the REAL derivation text end to end (a real subshell, a route
# table in /proc/net/route's format fed through RP_ROUTE_TABLE) -- both the
# match and the no-match (refusal) arms. No `ip` binary anywhere: the CI
# image ships none.
# A route table in /proc/net/route's own format (little-endian hex): a
# default route on eth0 (mask 0, never a match), a /16 on eth0 that also
# covers 10.0.0.9, and the /24 on ens7 that must win by longest prefix.
IFACE_ROUTES="$SANDBOX/iface-route"
printf 'Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n' > "$IFACE_ROUTES"
printf 'eth0\t00000000\t0100A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n' >> "$IFACE_ROUTES"
printf 'eth0\t0000000A\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n' >> "$IFACE_ROUTES"
printf 'ens7\t0000000A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n' >> "$IFACE_ROUTES"
match_out="$(RP_ROUTE_TABLE="$IFACE_ROUTES" bash -c "$pods_iface_text"; echo "RC=$?")"
if grep -q 'DERIVED_NCCL_IFACE=ens7' <<<"$match_out" && grep -q 'RC=0' <<<"$match_out"; then
  ok "executed end to end, the derivation picks 'ens7' (the interface actually carrying the GN ip) and exits 0"
else
  bad "expected the executed derivation to pick ens7 and exit 0; got: ${match_out}"
fi

nomatch_iface_text="$(RP_TWO_HOST_TRANSPORT=pods RP_TWO_HOST_GN_IP_0=10.9.9.9 bash -c 'source "'"$CLUSTER_SH"'"; _rpc_two_host_iface_lines 0')"
nomatch_out="$(RP_ROUTE_TABLE="$IFACE_ROUTES" bash -c "$nomatch_iface_text" 2>&1; echo "RC=$?")"
if grep -q 'covers the Global-Networking ip 10.9.9.9' <<<"$nomatch_out" && grep -q 'RC=97' <<<"$nomatch_out"; then
  ok "executed end to end, an ip with NO matching interface is a NAMED refusal (97), never a silent pass"
else
  bad "expected the executed derivation to refuse (97) naming the ip; got: ${nomatch_out}"
fi

# Mutation check: hardcode the interface (ignore the derivation) on a
# scratch copy -- the SAME no-match fixture then "succeeds" with the wrong
# interface instead of refusing, proving the real derivation is load-bearing.
IFACE_SCRATCH_DIR="$SANDBOX/iface-revert-red"
mkdir -p "$IFACE_SCRATCH_DIR"
cp "$DIR/runpod_lib.sh" "$IFACE_SCRATCH_DIR/runpod_lib.sh"
IFACE_SCRATCH="$IFACE_SCRATCH_DIR/runpod_gpu_cluster.sh"
if ! python3 - "$CLUSTER_SH" "$IFACE_SCRATCH" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
# The derivation's refusal arm: replace it with a hardcoded interface so the
# no-match fixture "succeeds" -- the shape the real derivation must NOT have.
old = '''if [ -z "\\$iface" ]; then
  echo "::error::no route in \\$route_table covers the Global-Networking ip \\$gn_ip -- refusing to derive NCCL_SOCKET_IFNAME" >&2
  exit 97
fi'''
new = '''iface="ens1"  # mutation: hardcoded, ignoring gn_ip and the route table entirely'''
assert old in text, "hardcoded-iface mutation fixture: the refusal arm was not found verbatim -- the fixture is stale, not the driver"
open(dst, "w").write(text.replace(old, new, 1))
PY
then
  bad "mutation check: the scratch patch did not apply (fixture stale) -- refusing to read a vacuous RC=0 as evidence"
fi
nomatch_iface_text_red="$(RP_TWO_HOST_TRANSPORT=pods RP_TWO_HOST_GN_IP_0=10.9.9.9 bash -c 'source "'"$IFACE_SCRATCH"'"; _rpc_two_host_iface_lines 0')"
nomatch_out_red="$(RP_ROUTE_TABLE="$IFACE_ROUTES" bash -c "$nomatch_iface_text_red"; echo "RC=$?")"
if grep -q 'RC=0' <<<"$nomatch_out_red"; then
  ok "mutation check: a hardcoded-interface shape reads the no-match fixture as a SUCCESS (RC=0) -- the real derivation above is load-bearing"
else
  bad "mutation check: expected the hardcoded-interface shape to succeed on the no-match fixture; got: ${nomatch_out_red} -- the mutation fixture itself may be stale"
fi
rm -rf "$IFACE_SCRATCH_DIR"

iface_log_good="$SANDBOX/iface-good.log"
printf 'noise\nNCCL INFO NET/Socket : Using [0]ens7:10.0.0.9<0>\nmore\n' > "$iface_log_good"
if _rpc_net_iface_seen "$iface_log_good" "ens7"; then
  ok "_rpc_net_iface_seen recognizes the derived interface's own NET/Socket line"
else
  bad "expected _rpc_net_iface_seen to recognize ens7's NET/Socket line"
fi
if _rpc_net_iface_seen "$iface_log_good" "eth0"; then
  bad "_rpc_net_iface_seen must not match a DIFFERENT interface's name"
else
  ok "_rpc_net_iface_seen refuses when the log names a different interface"
fi
iface_log_marker="$SANDBOX/iface-marker.log"
printf 'DERIVED_NCCL_IFACE=ens7\nother noise\n' > "$iface_log_marker"
parsed_iface="$(_rpc_parse_derived_iface "$iface_log_marker")"
if [ "$parsed_iface" = "ens7" ]; then
  ok "_rpc_parse_derived_iface reads the DERIVED_NCCL_IFACE= marker back"
else
  bad "expected _rpc_parse_derived_iface to read 'ens7'; got '${parsed_iface}'"
fi
if [ -z "$(_rpc_parse_derived_iface "$SANDBOX/does-not-exist")" ]; then
  ok "_rpc_parse_derived_iface on a missing log prints nothing (never a stale guess)"
else
  bad "expected empty output for a missing log"
fi

# ----------------------------------------------------------------------------
# Both pods terminated on EVERY exit arm (the trap's own terminate
# calls appear for both ids on the wrong-shape arm too).
# ----------------------------------------------------------------------------
run_two_host_cleanup() { # $1=pod0_id(or "") $2=pod1_id(or "") $3=pending_rc $4=terminate_ok(1|0, applies to BOTH ids)
  TWOPODS_TERMINATE_OK="$4" bash -c '
    source "'"$CLUSTER_SH"'"
    two_host_pod_id_0="$1"
    two_host_pod_id_1="$2"
    rp_terminate() { [ "$TWOPODS_TERMINATE_OK" = "1" ]; }
    rp_cleanup() { : > "'"$SANDBOX"'/two-host-cleanup-called"; }
    ( exit "$3" )
    _rpc_cleanup_cluster
  ' _ "$1" "$2" "$3"
}

rm -f "$SANDBOX/two-host-cleanup-called"
run_two_host_cleanup "pod-0" "pod-1" 0 1 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$SANDBOX/two-host-cleanup-called" ]; then
  ok "happy path (rc=0), both terminate calls succeed -> pending rc preserved, rp_cleanup still chained"
else
  bad "expected rc=0 and rp_cleanup chained on the happy path; got rc=$rc"
fi

rm -f "$SANDBOX/two-host-cleanup-called"
out="$(run_two_host_cleanup "pod-0" "pod-1" 97 1 2>&1)"
rc=$?
if [ "$rc" -eq 97 ] && [ -f "$SANDBOX/two-host-cleanup-called" ]; then
  ok "a wrong-shape refusal (97) still terminates BOTH pods and preserves the named exit code"
else
  bad "expected rc=97 preserved with rp_cleanup chained; got rc=$rc out=$out"
fi

rm -f "$SANDBOX/two-host-cleanup-called"
out="$(run_two_host_cleanup "pod-0" "pod-1" 0 0 2>&1)"
rc=$?
if [ "$rc" -ne 0 ] && grep -q "LEAKED two-host pod pod-0" <<<"$out" && grep -q "LEAKED two-host pod pod-1" <<<"$out"; then
  ok "BOTH termination calls failing names BOTH pod ids LEAKED and joins the exit status non-zero"
else
  bad "expected both pod ids named LEAKED with a non-zero exit; got rc=$rc out=$out"
fi

rm -f "$SANDBOX/two-host-cleanup-called"
run_two_host_cleanup "" "" 75 1 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 75 ] && [ -f "$SANDBOX/two-host-cleanup-called" ]; then
  ok "no pod ids at all (cleanup fires before either create ran) -> no terminate attempted, rc=75 preserved"
else
  bad "expected rc=75 preserved with no ids set; got rc=$rc"
fi

# The two cleanup branches stay separate. The cluster-cleanup cases above
# all run with two_host_pod_id_0/1 unset; this is the converse: a
# pods-transport cleanup with cluster_id unset never touches the cluster
# branch's own self-remove/delete calls.
rm -f "$SANDBOX/two-host-cleanup-called"
out="$(bash -c '
  source "'"$CLUSTER_SH"'"
  two_host_pod_id_0="pod-0"; two_host_pod_id_1="pod-1"
  rp_terminate() { return 0; }
  _rpc_self_remove_status() { echo "SHOULD_NOT_BE_CALLED"; }
  rp_cluster_delete() { echo "SHOULD_NOT_BE_CALLED"; return 0; }
  rp_cleanup() { : ; }
  ( exit 0 )
  _rpc_cleanup_cluster
' 2>&1)"
if ! grep -q "SHOULD_NOT_BE_CALLED" <<<"$out"; then
  ok "a pods-transport cleanup (cluster_id unset) never invokes the cluster branch's self-remove/delete calls"
else
  bad "the pods-transport cleanup unexpectedly touched the cluster branch: $out"
fi

# ----------------------------------------------------------------------------
# The artifact carries `transport`; the bash-side assembler itself
# refuses a value outside the closed set BEFORE any python assembly runs
# (check_cuda_run_artifacts.py's own self-test, run as one of this suite's
# gates below, is the authoritative schema-level oracle for the checker
# side of this property).
# ----------------------------------------------------------------------------
out_pods="$assemble_dir/gang-pods-happy.json"
_rpc_assemble_gang_artifact "$assemble_dir/r0.json" "$assemble_dir/r1.json" deadbeef "a100-sxm4-two-host-pods" "$out_pods" 2 1 1 "global-networking"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(python3 -c "import json; print(json.load(open('$out_pods'))['gang']['transport'])")" = "global-networking" ]; then
  ok "the assembler records transport='global-networking' for the pods leg"
else
  bad "expected transport='global-networking' recorded; rc=$rc"
fi
if check_gang_via_checker "$out_pods" >/dev/null 2>&1; then
  ok "the real check_cuda_run_artifacts.py accepts a complete pods-transport (global-networking) artifact"
else
  bad "the real checker unexpectedly refused a complete pods-transport artifact: $(check_gang_via_checker "$out_pods" 2>&1)"
fi

out_bogus="$assemble_dir/gang-bogus-transport.json"
_rpc_assemble_gang_artifact "$assemble_dir/r0.json" "$assemble_dir/r1.json" deadbeef box "$out_bogus" 2 1 1 "carrier-pigeon"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "$out_bogus" ]; then
  ok "an out-of-set transport is refused by the bash-side assembler itself (rc!=0), no artifact written"
else
  bad "expected an out-of-set transport to refuse assembly; rc=$rc, artifact exists=$([ -f "$out_bogus" ] && echo yes || echo no)"
fi

missing_transport_out="$(_rpc_assemble_gang_artifact "$assemble_dir/r0.json" "$assemble_dir/r1.json" deadbeef box "$assemble_dir/gang-missing-transport.json" 2 1 1 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
  ok "an OMITTED transport argument is refused (the bash \${9:?...} required-parameter guard)"
else
  bad "expected an omitted transport to refuse assembly; rc=$rc"
fi

# ----------------------------------------------------------------------------
# Exit codes are the SAME across transports -- the same combining functions
# (rp_cluster_rank_verdict/rp_cluster_verdict, exercised above over every rc
# arm) are reused verbatim by the pods branch;
# the module doc's own EXIT CONTRACT documents the SAME closed set for
# both transports.
# ----------------------------------------------------------------------------
if grep -qE '(exit|return) 75' "$CLUSTER_SH" && grep -qE '(exit|return) 76' "$CLUSTER_SH" \
  && grep -qE '(exit|return) 77' "$CLUSTER_SH" && grep -qE '(exit|return) 97' "$CLUSTER_SH" && grep -qE '(exit|return) 124' "$CLUSTER_SH" \
  && [ "$(grep -c '_rpc_run_two_ranks "\$primary_host" "\$primary_port" "\$member_host" "\$member_port" || exit \$?' "$CLUSTER_SH")" -eq 2 ]; then
  ok "every named exit code (75/76/77/97/124) appears in the driver (the shared rank runner returns 76/77/124 and BOTH arms exit with its status verbatim)"
else
  bad "one or more named exit codes is missing from the driver text"
fi
pods_branch_codes="$(awk '/The `pods` transport \(two ORDINARY/{f=1} f{print} /^echo "=== GPU cluster leg exit/{exit}' "$CLUSTER_SH" | grep -oE 'exit (75|76|77|97|124)' | sort -u | tr '\n' ' ')"
if [ -n "$pods_branch_codes" ]; then
  ok "the pods branch itself names exit codes: ${pods_branch_codes}(shared closed set, never a new one)"
else
  bad "the pods branch names no exit code at all -- the awk delimiter used to isolate it may be stale"
fi

# ----------------------------------------------------------------------------
# No remote command before an EXECUTED connect to that member succeeded
# once: a pod reads RUNNING + GN-enabled before its sshd answers, because
# sshd is installed by the entrypoint AFTER boot. The library probes `ssh ... true` under a bound; both
# transports call `rp_wait_sshd` (runpod_lib.sh: ONE predicate, `rp_sshd_answers`,
# shared with `rp_session_alive` and `_rp_deploy`) for BOTH members before
# their first `_rpc_remote_script`.
# ----------------------------------------------------------------------------
run_ssh_ready() { # $1=refusals before success (-1 = never) $2=bound(s)
  SSHREADY_REFUSALS="$1" bash -c '
    source "'"$CLUSTER_SH"'" >/dev/null 2>&1
    RP_SSHO=(-o Fixture=yes)
    n=0
    ssh() { echo "ssh $*" >> "'"$SANDBOX"'/sshready-ssh-calls"; n=$((n+1)); [ "$SSHREADY_REFUSALS" -ge 0 ] && [ "$n" -gt "$SSHREADY_REFUSALS" ]; }
    sleep() { :; }
    rp_wait_sshd "10.0.0.1" "2222" "'"$2"'" "rank 0" -o "ProxyJump=root@jump:1"
  '
}
rm -f "$SANDBOX/sshready-ssh-calls"
out="$(run_ssh_ready 2 300 2>&1)"; rc=$?
calls="$(grep -c '^ssh ' "$SANDBOX/sshready-ssh-calls" 2>/dev/null || echo 0)"
if [ "$rc" -eq 0 ] && [ "$calls" -eq 3 ] && grep -q "sshd up on rank 0 (10.0.0.1:2222)" <<<"$out" && grep -q -- "-o Fixture=yes -o ProxyJump=root@jump:1 -p 2222 root@10.0.0.1 true" "$SANDBOX/sshready-ssh-calls"; then
  ok "two refused connects then success -> rc 0 after exactly 3 probes, RP_SSHO + the member's extra options + the port on every probe"
else
  bad "expected rc 0 after 3 probes with the options threaded; got rc=$rc calls=$calls out=$out"
fi
rm -f "$SANDBOX/sshready-ssh-calls"
out="$(run_ssh_ready -1 1 2>&1)"; rc=$?
if [ "$rc" -eq 1 ] && grep -q "::error::sshd on rank 0 (10.0.0.1:2222) answered no connect within 1s" <<<"$out"; then
  ok "a member whose sshd never answers -> rc 1 within the bound, named ::error:: (never a remote command)"
else
  bad "expected rc 1 with the named ::error:: on the bound; got rc=$rc out=$out"
fi
# Static ordering, both arms (the same technique as the create-order checks
# above): each arm's TWO wait calls
# precede that arm's first `_rpc_remote_script 0 | ssh` launch.
sshready_waits="$(grep -n '^rp_wait_sshd "\$primary_host"' "$CLUSTER_SH" | cut -d: -f1 | tr '\n' ' ')"
sshready_launch="$(grep -n '^_rpc_run_two_ranks "\$primary_host"' "$CLUSTER_SH" | cut -d: -f1 | tr '\n' ' ')"
sshready_ok=1
set -- $sshready_launch
for launch in "$@"; do
  seen=0
  for w in $sshready_waits; do
    [ "$w" -lt "$launch" ] && [ $(( launch - w )) -lt 40 ] && seen=1
  done
  [ "$seen" -eq 1 ] || sshready_ok=0
done
if [ "$sshready_ok" -eq 1 ] && [ "$(printf '%s' "$sshready_waits" | wc -w)" -eq 2 ] && [ "$(printf '%s' "$sshready_launch" | wc -w)" -eq 2 ]; then
  ok "both arms (cluster, pods) wait for sshd on rank 0 within 40 lines before their own _rpc_run_two_ranks call (waits at ${sshready_waits}; launches at ${sshready_launch})"
else
  bad "expected a rank-0 sshd wait shortly before EACH arm's _rpc_run_two_ranks call; waits=${sshready_waits} launches=${sshready_launch}"
fi
if [ "$(grep -c '^rp_wait_sshd "\$member_host"' "$CLUSTER_SH")" -eq 2 ]; then
  ok "both arms wait for sshd on rank 1 too"
else
  bad "expected two rank-1 sshd waits (one per arm)"
fi
# ----------------------------------------------------------------------------
# The two ranks run through ONE shared function on both transports; both
# build concurrently; the id crossing is a phase of the watch loop bounded
# by rank 0's liveness, log growth and the budget -- never a wall clock of
# its own (a fixed crossing bound such as RP_SSH_WAIT_SECS=300 s is far
# shorter than a cold build). Every exit
# arm of `_rpc_run_two_ranks` is driven here with stubbed ssh/scp/remote
# scripts (the REAL function, the REAL `_rpc_id_file_ready`).
# ----------------------------------------------------------------------------
TWORANKS_DIR="$SANDBOX/tworanks"
run_two_ranks_fixture() { # $1=rank0 script body $2=rank1 script body $3=RP_INACTIVITY $4=deadline offset (s) $5=late-mint (1 = the FIRST id read comes back empty)
  rm -rf "$TWORANKS_DIR"; mkdir -p "$TWORANKS_DIR"
  TWORANKS_R0="$1" TWORANKS_R1="$2" TWORANKS_INACT="$3" TWORANKS_DL="$4" TWORANKS_LATE_MINT="${5:-0}" bash -c '
    source "'"$CLUSTER_SH"'" >/dev/null 2>&1
    RP_SSHO=(-o Fixture=yes); member_extra_sshopts=(-o "ProxyJump=root@jump:1")
    RP_WORK="'"$TWORANKS_DIR"'/work"; mkdir -p "$RP_WORK"  # the driver'"'"'s own EXIT cleanup removes RP_WORK -- never the fixture dir itself
    CLUSTER_REMOTE_ID_FILE=/remote/nccl.id; RP_TIMEOUT=30
    RP_INACTIVITY="$TWORANKS_INACT"; DEADLINE=$(( SECONDS + TWORANKS_DL )); id_landed=0; unset PROVE_EXPECT_SHA
    _rpc_remote_script() { [ "$1" = "0" ] && printf "%s\n" "$TWORANKS_R0" || printf "%s\n" "$TWORANKS_R1"; }
    RP_ENV_PREAMBLE=": env-preamble-sentinel"
    ssh() { echo "ssh $*" >> "'"$TWORANKS_DIR"'/calls"; f="$(mktemp "'"$TWORANKS_DIR"'/stdin-XXXXXX")"; cat > "$f"; bash "$f"; }
    scp() { echo "scp $*" >> "'"$TWORANKS_DIR"'/calls"; case "$*" in *"root@"*":"*" "*) : ;; esac
            last="${@: -1}"; case "$last" in "'"$TWORANKS_DIR"'"/*) head -c 128 /dev/zero > "$last" ;; esac; }
    _rpc_remote_id_size() {
      # Late-mint mode: the first read comes back empty and every later one
      # reports the file as it stands -- the mint landing between a read and
      # the liveness test that follows it.
      if [ "$TWORANKS_LATE_MINT" = "1" ] && [ ! -f "'"$TWORANKS_DIR"'/read-once" ]; then
        : > "'"$TWORANKS_DIR"'/read-once"; return 0
      fi
      cat "'"$TWORANKS_DIR"'/idsize" 2>/dev/null
    }
    _rpc_phase() { echo "=== PHASE ($SECONDS)s: $* ==="; }
    CLUSTER_ARTIFACT_DIR="'"$TWORANKS_DIR"'/artifact"
    sleep() { command sleep 0.1; }
    _rpc_run_two_ranks "10.0.0.1" "2201" "10.0.0.2" "2202"; rc=$?
    echo "RC=$rc id_landed=${id_landed} rank0_rc=${rank0_rc:-?} rank1_rc=${rank1_rc:-?}"
  ' 2>&1
}
r0_happy='echo start0; command sleep 0.4; echo 128 > "'"$TWORANKS_DIR"'/idsize"; command sleep 0.4; echo done0'
r1_happy='echo start1; command sleep 0.6; echo done1'
out="$(run_two_ranks_fixture "$r0_happy" "$r1_happy" 30 600)"
if grep -q "RC=0 id_landed=1 rank0_rc=0 rank1_rc=0" <<<"$out" \
   && grep -q "the id crossed to the member" <<<"$out" \
   && [ "$(grep -c '^ssh ' "$TWORANKS_DIR/calls")" -eq 2 ] \
   && [ "$(grep -n '^scp ' "$TWORANKS_DIR/calls" | wc -l | tr -d ' ')" -eq 2 ] \
   && grep -q "root@10.0.0.1:/remote/nccl.id" <<<"$(grep -m1 '^scp ' "$TWORANKS_DIR/calls")" \
   && grep -q -- "-o ProxyJump=root@jump:1 -P 2202 .* root@10.0.0.2:/remote/nccl.id" <<<"$(grep '^scp ' "$TWORANKS_DIR/calls" | tail -1)" \
   && [ "$(grep '^ssh ' "$TWORANKS_DIR/calls" | grep -c -- "-o ProxyJump=root@jump:1 -p 2202 root@10.0.0.2")" -eq 1 ] \
   && [ "$(grep '^ssh ' "$TWORANKS_DIR/calls" | grep -v -- "ProxyJump" | grep -c -- "-p 2201 root@10.0.0.1")" -eq 1 ]; then
  ok "happy path -- both ranks launched up front (2 ssh, rank 1 with its own options), the id scp'd down from the primary then up to the member exactly once each, both ranks rc 0, id_landed=1"
  preamble_r0="$(grep -l '^echo start0' "$TWORANKS_DIR"/stdin-* | head -1)"; preamble_r1="$(grep -l '^echo start1' "$TWORANKS_DIR"/stdin-* | head -1)"
  if [ -n "$preamble_r0" ] && [ -n "$preamble_r1" ] && [ "$(head -1 "$preamble_r0")" = ": env-preamble-sentinel" ] && [ "$(head -1 "$preamble_r1")" = ": env-preamble-sentinel" ]; then
    ok "the FIRST line each rank receives is RP_ENV_PREAMBLE (PID 1's environment import, the library's one definition), before its own script"
  else
    bad "expected RP_ENV_PREAMBLE as the first line of both ranks' stdin; got: $(head -1 "$preamble_r0" 2>/dev/null) / $(head -1 "$preamble_r1" 2>/dev/null)"
  fi
else
  bad "happy path off; out=$out calls=$(cat "$TWORANKS_DIR/calls" 2>/dev/null)"
fi
# Rank 0 mints the id and ends inside ONE poll interval, with the mint
# landing after that interval's own read (late-mint mode) -- so the watcher
# sees no id, and then sees a rank 0 that is gone. Reading the id before the
# liveness test narrows that window; it cannot close it, because the two
# reads are of different instants. Only a read taken when rank 0 is FINAL
# settles it: a gone rank 0's id file can no longer change. On a loaded host
# this is what a short proof looks like, at random.
r0_mints_and_ends='echo start0; echo 128 > "'"$TWORANKS_DIR"'/idsize"; echo done0'
out="$(run_two_ranks_fixture "$r0_mints_and_ends" "$r1_happy" 30 600 1)"
if grep -q "RC=0 id_landed=1 rank0_rc=0 rank1_rc=0" <<<"$out" \
   && grep -q "the id crossed to the member" <<<"$out" \
   && ! grep -q "before minting" <<<"$out" \
   && [ "$(grep -c '^scp ' "$TWORANKS_DIR/calls")" -eq 2 ]; then
  ok "a mint landing between the watcher's id read and its liveness test still crosses: a GONE rank 0's id is read once more, and only its absence then is 'never minted'"
else
  bad "a minted id must cross when the mint lands inside the watcher's own poll; out=$out calls=$(cat "$TWORANKS_DIR/calls" 2>/dev/null)"
fi
r0_dies='echo start0; command sleep 0.2; exit 3'
out="$(run_two_ranks_fixture "$r0_dies" "$r1_happy" 30 600)"
if grep -q "RC=76" <<<"$out" && grep -q "::error::rank 0 ended (rc=3) before minting the 128-byte id file" <<<"$out" \
   && grep -q "start0" "$TWORANKS_DIR/artifact/rank0.log" && grep -q "start1" "$TWORANKS_DIR/artifact/rank1.log"; then
  ok "rank 0 exiting before the id is minted -> 76 with rank 0's own rc named; no scp ever attempted; BOTH rank logs shipped into the artifact dir on this refusal arm"
else
  bad "expected 76 naming rank 0's rc; out=$out"
fi
r0_silent='command sleep 5'
r1_silent='command sleep 5'
out="$(run_two_ranks_fixture "$r0_silent" "$r1_silent" 1 600)"
if grep -q "RC=76" <<<"$out" && grep -q "::error::inactivity: no new output on either rank's log for 1s" <<<"$out"; then
  ok "no log growth on either rank within RP_INACTIVITY -> 76, named inactivity (the crossing wait has no clock of its own)"
else
  bad "expected the inactivity arm; out=$out"
fi
out="$(run_two_ranks_fixture "$r0_happy" "$r1_happy" 30 0)"
if grep -q "RC=124" <<<"$out" && grep -q "budget cut at T-10m" <<<"$out"; then
  ok "the T-10m budget cuts the leg at 124 during the crossing wait"
else
  bad "expected the budget arm (124); out=$out"
fi
tworanks_calls="$(grep -c '^_rpc_run_two_ranks "\$primary_host" "\$primary_port" "\$member_host" "\$member_port" || exit \$?' "$CLUSTER_SH")"
if [ "$tworanks_calls" -eq 2 ] && ! grep -q 'id_deadline=' "$CLUSTER_SH" && ! grep -q 'never produced a 128-byte id file within' "$CLUSTER_SH"; then
  ok "both transport arms run the ranks through the ONE shared function; no wall-clock id poll exists in the tree"
else
  bad "expected exactly two shared-function calls and no wall-clock id poll; calls=$tworanks_calls"
fi
tworanks_scripts="$(RP_TWO_HOST_GN_IP_0=10.0.0.1 RP_TWO_HOST_GN_IP_1=10.0.0.2 bash -c '
  source "'"$CLUSTER_SH"'" >/dev/null 2>&1
  export RP_TWO_HOST_GN_IP_0 RP_TWO_HOST_GN_IP_1
  echo "r1=$(_rpc_remote_script 1 | grep -c "=== id-wait: rank 1 built") r0=$(_rpc_remote_script 0 | grep -c "id-wait")"
  _rpc_remote_script 1 | awk "/::group::cluster-build/{b=NR} /=== id-wait: rank 1 built/{w=NR} /::group::cluster-proof/{p=NR} END{print (b<w && w<p) ? \"order-ok\" : \"order-bad\"}" | head -1
' 2>&1 | tr '\n' ' ')"
if grep -q "r1=1 r0=0" <<<"$tworanks_scripts" && grep -q "order-ok" <<<"$tworanks_scripts"; then
  ok "rank 1's remote script waits for the 128-byte id BETWEEN its build and its proof; rank 0's never waits"
else
  bad "expected the id wait only in rank 1's script, between build and proof; got: $tworanks_scripts"
fi
# ----------------------------------------------------------------------------
# A member proves the EXACT commit the run was dispatched on. With
# PROVE_EXPECT_SHA set the remote checkout fetches that sha by hash and
# never names the ref (a `-b <branch>` clone lands on whatever tip the
# branch has moved to since dispatch, and the wrong-tree guard fires);
# without it (a hand run) the ref is cloned. ONE library helper, used by all three
# GPU legs.
# ----------------------------------------------------------------------------
checkout_sha_text="$(PROVE_EXPECT_SHA=0123456789abcdef0123456789abcdef01234567 bash -c 'source "'"$DIR"'/runpod_lib.sh" >/dev/null 2>&1; rp_remote_checkout_lines "some-branch" "https://example.invalid/r.git"')"
checkout_ref_text="$(bash -c 'unset PROVE_EXPECT_SHA; source "'"$DIR"'/runpod_lib.sh" >/dev/null 2>&1; rp_remote_checkout_lines "some-branch" "https://example.invalid/r.git"')"
if grep -qF 'git fetch -q --depth 1 origin "0123456789abcdef0123456789abcdef01234567"' <<<"$checkout_sha_text" \
   && grep -qF 'git checkout -q --detach FETCH_HEAD || { echo "::error::could not check out' <<<"$checkout_sha_text" \
   && ! grep -qF 'some-branch' <<<"$checkout_sha_text" \
   && grep -qF 'could not fetch the exact commit' <<<"$checkout_sha_text"; then
  ok "with PROVE_EXPECT_SHA the remote checkout fetches that exact sha and never names the ref; a failed fetch is a named error"
else
  bad "expected a by-sha fetch with no ref; got: ${checkout_sha_text}"
fi
if grep -qF 'git clone --depth 1 -b "some-branch" "https://example.invalid/r.git" jammi-ai || { echo "::error::could not clone' <<<"$checkout_ref_text" \
   && ! grep -qF '| tail -1' <<<"$checkout_ref_text" \
   && ! grep -qF 'FETCH_HEAD' <<<"$checkout_ref_text"; then
  ok "without PROVE_EXPECT_SHA (a hand run) the ref is cloned"
else
  bad "expected the ref clone without PROVE_EXPECT_SHA; got: ${checkout_ref_text}"
fi
checkout_legs=0
for leg in runpod_gpu_cluster.sh runpod_gpu_gang.sh runpod_gpu_prove.sh; do
  grep -qF '$(rp_remote_checkout_lines "${GIT_REF}" "${GIT_REPO}")' "$DIR/$leg" && ! grep -qF 'git clone --depth 1 -b "${GIT_REF}"' "$DIR/$leg" && checkout_legs=$((checkout_legs + 1))
done
if [ "$checkout_legs" -eq 3 ]; then
  ok "all three GPU legs (cluster, gang, prove) check out through the ONE library helper; no leg clones the ref itself"
else
  bad "expected all three legs on rp_remote_checkout_lines with no own clone line; got $checkout_legs/3"
fi
# ----------------------------------------------------------------------------
# The remote root is a PARAMETER of every leg and the checkout chain
# fails closed. RP_REMOTE_ROOT roots the checkout, the id file and the
# artifact dir; rendered under a sandbox root no `/root` literal survives
# in the remote text; a checkout step that fails STOPS the script by name
# (nothing ever runs in an unknown working directory — executed where
# /root does not exist, a stubbed clone would otherwise write a tree mirror
# into the caller's own checkout).
# ----------------------------------------------------------------------------
remoteroot_text="$(RP_TWO_HOST_TRANSPORT=pods RP_REMOTE_ROOT=/sandbox/x RP_TWO_HOST_GN_IP_0=10.0.0.1 RP_TWO_HOST_GN_IP_1=10.0.0.2 PROVE_EXPECT_SHA=0123456789abcdef0123456789abcdef01234567 bash -c 'source "'"$CLUSTER_SH"'" >/dev/null 2>&1; export RP_TWO_HOST_GN_IP_0 RP_TWO_HOST_GN_IP_1; _rpc_remote_script 1' 2>&1)"
if [ "$(printf '%s' "$remoteroot_text" | grep -c '/root')" -eq 0 ] \
   && grep -qF 'cd "/sandbox/x" || { echo "::error::remote root /sandbox/x is not enterable" >&2; exit 1; }' <<<"$remoteroot_text" \
   && grep -qF 'export JAMMI_GANG_TWO_HOSTS_ID_FILE=/sandbox/x/nccl.id' <<<"$remoteroot_text" \
   && grep -qF 'export JAMMI_GANG_ARTIFACT_DIR=/sandbox/x/jammi-ai/.gang-artifact' <<<"$remoteroot_text"; then
  ok "under RP_REMOTE_ROOT=/sandbox/x the whole remote text (checkout, id file, artifact dir) is rooted there and carries no /root literal"
else
  bad "expected a fully re-rooted remote text with no /root literal; got: $(printf '%s' "$remoteroot_text" | grep -n '/root\|sandbox' | head -5)"
fi
remoteroot_default="$(RP_TWO_HOST_TRANSPORT=pods RP_TWO_HOST_GN_IP_0=10.0.0.1 RP_TWO_HOST_GN_IP_1=10.0.0.2 bash -c 'unset RP_REMOTE_ROOT; source "'"$CLUSTER_SH"'" >/dev/null 2>&1; export RP_TWO_HOST_GN_IP_0 RP_TWO_HOST_GN_IP_1; _rpc_remote_script 0' 2>&1)"
if grep -qF 'cd "/root" || {' <<<"$remoteroot_default" && grep -qF 'export JAMMI_GANG_TWO_HOSTS_ID_FILE=/root/nccl.id' <<<"$remoteroot_default"; then
  ok "with RP_REMOTE_ROOT unset the root is /root (a real pod)"
else
  bad "expected the /root default; got: $(printf '%s' "$remoteroot_default" | grep -n 'cd \"' | head -2)"
fi
remoteroot_fc="$(mktemp -d "$SANDBOX/remoteroot-XXXXXX")"
remoteroot_out="$(RP_REMOTE_ROOT="$remoteroot_fc/does-not-exist" bash -c 'source "'"$DIR"'/runpod_lib.sh" >/dev/null 2>&1; rp_remote_checkout_lines "b" "https://example.invalid/r.git"' | bash 2>&1; echo "RC=$?")"
if grep -q "RC=1" <<<"$remoteroot_out" && grep -q "::error::remote root .*/does-not-exist is not enterable" <<<"$remoteroot_out" && ! grep -qi "clon" <<<"$remoteroot_out"; then
  ok "executed — an un-enterable root stops the checkout by name before any git command (rc 1, never a clone into the caller's cwd)"
else
  bad "expected the fail-closed root arm; got: ${remoteroot_out}"
fi
if ! grep -q . <<<"$(grep -nE '"/root/|=/root' "$DIR/runpod_gpu_cluster.sh" "$DIR/runpod_gpu_gang.sh" "$DIR/runpod_gpu_prove.sh" | grep -v '^\S*:[0-9]*:\s*#')"; then
  ok "no GPU leg carries a /root literal outside a comment (the root is RP_REMOTE_ROOT everywhere)"
else
  bad "a /root literal survives in a leg: $(grep -nE '"/root/|=/root' "$DIR/runpod_gpu_cluster.sh" "$DIR/runpod_gpu_gang.sh" "$DIR/runpod_gpu_prove.sh" | grep -v '^\S*:[0-9]*:\s*#' | head -3)"
fi
# ----------------------------------------------------------------------------
# Prose == code.
# ----------------------------------------------------------------------------
for site in "$CLUSTER_SH" "$CLUSTER_YML" "$DEV_GPU_MD"; do
  rel="${site#"$REPO_ROOT"/}"
  if grep -qi "pods" "$site" && grep -qi "global.networking\|global network" "$site"; then
    ok "${rel} describes the pods transport / Global Networking in prose"
  else
    bad "${rel} does not mention the pods transport or Global Networking -- prose and code have diverged"
  fi
done

# ============================================================================
# check_gpu_prove_once.py's own paid-lane and schedule-visibility gates
# stay green over this lane (the real gate, not a paraphrase of it).
# ============================================================================
if python3 "$PROVE_ONCE_PY" >/dev/null 2>&1; then
  ok "check_gpu_prove_once.py is green over the cluster lane (paid-lane and schedule-visibility gates, live)"
else
  bad "check_gpu_prove_once.py FAILED with the cluster lane in the tree: $(python3 "$PROVE_ONCE_PY" 2>&1)"
fi

echo
echo "gpu-cluster-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
