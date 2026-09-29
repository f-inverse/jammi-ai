#!/usr/bin/env bash
# The compute plane's scaling test: does splitting one materialization across
# GPU hosts pay? The test, its workload and its pass bar are D15 of
# `docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md`, fixed before
# any leg of it ran; this driver rents the hosts, runs the legs, and hands
# them to the one judge (`jammi-bench ladder encode`) and to D15's reading of
# its verdicts (`plane_scaling_verdict.py`). It decides nothing itself.
#
# WHAT IT RENTS: five ordinary pods (`rp_fleet_pod_create`), one GPU each, of
# one type, co-located in one data center on Global Networking and measured
# to reach each other (`rp_fleet_routes`). Host 0 is the control host — the
# fleet's Postgres catalog, S3-class store, shape-d scheduler and query tier,
# and the direct session's process; hosts 1..4 are the compute tier. The
# candidates are every SCALING_GPU_TYPES type and co-located data center that
# clears SCALING_MIN_AVAILABILITY within SCALING_MAX_GPU_RATE.
#
# WHAT IT RUNS, per D15 (`plane_scaling_host.sh` holds each host's part):
#   the edge session on host 4 — `plan-partitioned` and `placed`, both takes,
#     over a catalog and store of host 4's own — while the fleet runs K = 1
#     and K = 2 on hosts 1 and 2;
#   the direct session on host 0 — `plan-partitioned` beside `shape-d`
#     through the query tier — once per take per fleet size, in a palindrome
#     over the fleet sizes (K1 r1, K2 r1, K4 r1, K4 r2, K2 r2, K1 r2) so a
#     drift over the run moves every size alike. A fleet shrinks by DRAIN
#     (SIGTERM), which takes an executor out of the live set at once; every
#     `shape-d` leg records the tier it served against, and a block whose legs
#     do not all record its K is refused at assembly.
#
# THE ARTIFACT: the legs and each K's verdict under SCALING_ARTIFACT_DIR, and
# D15's reading of them; a human reviews it and commits it under
# `ci/artifacts/parity-ladder-runs/`.
#
# Run by hand (the CI image runs it; macOS bash cannot):
#   RUNPOD_API_KEY=… GIT_REF=<pushed branch> PROVE_EXPECT_SHA=<its head> \
#     ci/dev.sh bash ci/scripts/runpod_plane_scaling.sh
#
# Exit 0 = D15 PASS; 1 = D15 FAIL; 2 = a session or the assembly failed, so
# nothing was decided; 75 = no co-located capacity for any candidate; 97 = a
# rented host is not the shape asked for.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-10}"
export RP_GPU_COUNT=1
export RP_DISK_GB="${RP_DISK_GB:-120}"
export RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-1200}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/f-inverse/jammi-ai.git}"
GIT_REF="${GIT_REF:?GIT_REF names the pushed branch the hosts check out}"
REMOTE_CHECKOUT_LINES="$(rp_remote_checkout_lines "${GIT_REF}" "${GIT_REPO}")"

# D15 pre-registered A100 hosts: another card changes compute against the
# network, the very ratio the test measures, so no other type is a candidate.
SCALING_GPU_TYPES="${SCALING_GPU_TYPES:-NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe}"
SCALING_MIN_AVAILABILITY="${SCALING_MIN_AVAILABILITY:-LOW}"
SCALING_MAX_GPU_RATE="${SCALING_MAX_GPU_RATE:-4.00}"
SCALING_DATA_CENTER="${SCALING_DATA_CENTER:-}"
SCALING_ARTIFACT_DIR="${SCALING_ARTIFACT_DIR:-$REPO_ROOT/.gpu-pull/plane-scaling}"
HOSTS=5
EDGE_HOST=4
REMOTE_DIR="${RP_REMOTE_ROOT}/plane-scaling"
HOST_SCRIPT="bash ci/scripts/plane_scaling_host.sh"
CHECKOUT_LINES="cd ${RP_REMOTE_ROOT}/jammi-ai || exit 1
echo \"PROVE_SHA=\$(git rev-parse HEAD)\"
export CARGO_TERM_COLOR=never CARGO_BUILD_RUSTC_WRAPPER="

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

rp_sweep
rp_init

avail_resp="$(_rp_rest GET "/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=1&cloud=SECURE")"
[ "$(printf '%s\n' "$avail_resp" | head -n1)" = "200" ] || { echo "::error::pod catalog read failed"; exit 75; }
gn_resp="$(_rp_rest GET /v2/catalog/datacenters)"
[ "$(printf '%s\n' "$gn_resp" | head -n1)" = "200" ] || { echo "::error::datacenters read failed"; exit 75; }
gn_dcs="$(printf '%s\n' "$gn_resp" | tail -n +2 | rp_fleet_global_network_datacenters)"
case "$gn_dcs" in PARSE_ERROR*) echo "::error::${gn_dcs}"; exit 75 ;; esac
candidates="$(rp_fleet_candidates "$(printf '%s\n' "$avail_resp" | tail -n +2)" "$gn_dcs" \
  "$SCALING_GPU_TYPES" "$SCALING_MIN_AVAILABILITY" "$SCALING_MAX_GPU_RATE" "$SCALING_DATA_CENTER")" || exit $?

declare -a POD HOST PORT GN
release_fleet() {
  local id
  for id in "${POD[@]:-}"; do
    [ -n "$id" ] || continue
    rp_terminate "$id" >/dev/null || echo "::warning::terminate ${id} refused -- its TTL and gpu-reap.yml's sweep remain the backstop"
  done
  POD=()
}
trap 'rc=$?; release_fleet; exit $rc' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

placed=0
while IFS='|' read -r GPU_TYPE GPU_RATE chosen_dc <&3; do
  echo "=== ${GPU_TYPE} at \$${GPU_RATE}/GPU/h x ${HOSTS} hosts in ${chosen_dc}, TTL ${RP_TTL_HOURS}h ==="
  POD=()
  for i in $(seq 0 $((HOSTS - 1))); do
    id="$(rp_fleet_pod_create "$GPU_TYPE" "$chosen_dc" "$i")" || { release_fleet; continue 2; }
    POD+=("$id")
  done
  ready="$(rp_fleet_wait_ready "$RP_SSH_WAIT_SECS" "${POD[@]}")" || { release_fleet; continue; }
  HOST=(); PORT=(); GN=()
  measured_dc=""
  triples=()
  while IFS=' ' read -r a b c; do
    if [ -z "$measured_dc" ]; then measured_dc="$a"; continue; fi
    HOST+=("$a"); PORT+=("$b"); GN+=("$c"); triples+=("$a" "$b" "$c")
  done <<< "$ready"
  echo "=== ${HOSTS} hosts RUNNING in ${measured_dc}: ${GN[*]} ==="
  ok=1
  for i in "${!HOST[@]}"; do rp_wait_sshd "${HOST[$i]}" "${PORT[$i]}" "$RP_SSH_WAIT_SECS" "host $i" || ok=0; done
  if [ "$ok" = 1 ] && rp_fleet_routes "${triples[@]}"; then placed=1; break; fi
  release_fleet
done 3<<< "$candidates"
[ "$placed" -eq 1 ] || { echo "::error::no candidate gave ${HOSTS} ready hosts that reach each other in one data center"; exit 75; }

# One remote script on host $1, under the inactivity watchdog; $2 bounds it.
on_host() {
  RP_HOST="${HOST[$1]}" RP_PORT="${PORT[$1]}" RP_TIMEOUT="${2:-3600}" rp_run_remote_watched ""
}

# ── host 0 builds; hosts 1..4 check out meanwhile ───────────────────────────
BUILD_LOG="$(mktemp)"
on_host 0 7200 > "$BUILD_LOG" 2>&1 <<REMOTE &
${REMOTE_CHECKOUT_LINES}
echo "PROVE_SHA=\$(git rev-parse HEAD)"
export CARGO_TERM_COLOR=never CARGO_BUILD_RUSTC_WRAPPER=
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass || exit 1
( ${HOST_SCRIPT} build ${REMOTE_DIR} ) & pid=\$!
while kill -0 \$pid 2>/dev/null; do sleep 60; echo "building: \$(date -u +%T)"; done
wait \$pid || exit 1
${HOST_SCRIPT} serve ${GN[0]} ${REMOTE_DIR} || exit 1
echo "PROVE_EXIT=0"
REMOTE
build_pid=$!
# Wait on every pid in turn; any non-zero status names its host.
await_all() { # $1=what $2..=pid:host
  local what="$1" failed=0 entry; shift
  for entry in "$@"; do
    wait "${entry%%:*}" || { echo "::error::host ${entry#*:}: ${what} failed"; failed=1; }
  done
  return "$failed"
}
pids=()
for i in 1 2 3 4; do
  on_host "$i" 1800 <<REMOTE &
${REMOTE_CHECKOUT_LINES}
echo "PROVE_SHA=\$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo "PROVE_EXIT=0"
REMOTE
  pids+=("$!:$i")
done
await_all "the build and serve" "${build_pid}:0" || { cat "$BUILD_LOG"; exit 2; }
await_all "the checkout" "${pids[@]}" || exit 2
cat "$BUILD_LOG"
server_sha="$(sed -n 's/^SERVER_SHA256=\([0-9a-f]\{64\}\)$/\1/p' "$BUILD_LOG" | tail -1)"
bench_sha="$(sed -n 's/^BENCH_SHA256=\([0-9a-f]\{64\}\)$/\1/p' "$BUILD_LOG" | tail -1)"
model_sha="$(sed -n 's/^MODEL_SHA256=\([0-9a-f]\{64\}\)$/\1/p' "$BUILD_LOG" | tail -1)"
[ -n "$server_sha" ] && [ -n "$bench_sha" ] && [ -n "$model_sha" ] || { echo "::error::host 0 reported no build digests"; exit 2; }
pids=()
for i in 1 2 3 4; do
  on_host "$i" 1800 <<REMOTE &
${CHECKOUT_LINES}
${HOST_SCRIPT} fetch ${GN[0]} ${REMOTE_DIR} ${server_sha} ${bench_sha} ${model_sha} || exit 1
echo "PROVE_EXIT=0"
REMOTE
  pids+=("$!:$i")
done
await_all "the fetch of host 0's build" "${pids[@]}" || exit 2

# ── the fleet's control plane, and the edge session on its own host ────────
on_host 0 900 <<REMOTE || { echo "::error::the scheduler or query tier did not start"; exit 2; }
${CHECKOUT_LINES}
${HOST_SCRIPT} role scheduler ${GN[0]} ${GN[0]} ${REMOTE_DIR} 0 || exit 1
${HOST_SCRIPT} role query ${GN[0]} ${GN[0]} ${REMOTE_DIR} 0 || exit 1
echo "PROVE_EXIT=0"
REMOTE
on_host "$EDGE_HOST" 300 <<REMOTE || { echo "::error::the edge session did not start"; exit 2; }
${CHECKOUT_LINES}
setsid nohup bash -c '${HOST_SCRIPT} edge ${REMOTE_DIR} ${REMOTE_DIR}/legs-edge 1 && ${HOST_SCRIPT} edge ${REMOTE_DIR} ${REMOTE_DIR}/legs-edge 2; echo "rc=\$?" > ${REMOTE_DIR}/edge.done' \
  > ${REMOTE_DIR}/edge.log 2>&1 < /dev/null &
echo "PROVE_EXIT=0"
REMOTE

compute() { # $1=start|stop $2=host
  on_host "$2" 900 <<REMOTE
${CHECKOUT_LINES}
$( [ "$1" = start ] && echo "${HOST_SCRIPT} role compute ${GN[$2]} ${GN[0]} ${REMOTE_DIR} 0" || echo "${HOST_SCRIPT} stop ${REMOTE_DIR} compute" ) || exit 1
echo "PROVE_EXIT=0"
REMOTE
}

block() { # $1=K $2=take
  on_host 0 10800 <<REMOTE
${CHECKOUT_LINES}
( ${HOST_SCRIPT} direct ${GN[0]} ${GN[0]} ${REMOTE_DIR} ${REMOTE_DIR}/legs-direct-K$1 $2 ) > ${REMOTE_DIR}/direct-K$1-r$2.log 2>&1 & pid=\$!
while kill -0 \$pid 2>/dev/null; do sleep 60; echo "K=$1 take $2: \$(ls ${REMOTE_DIR}/legs-direct-K$1 2>/dev/null | grep -c 'json\$') legs filed"; done
wait \$pid; rc=\$?
tail -20 ${REMOTE_DIR}/direct-K$1-r$2.log
echo "PROVE_EXIT=\$rc"; exit \$rc
REMOTE
}

await_edge() {
  local deadline=$((SECONDS + 4 * 3600)) done_line
  while [ "$SECONDS" -lt "$deadline" ]; do
    done_line="$(ssh "${RP_SSHO[@]}" -p "${PORT[$EDGE_HOST]}" "root@${HOST[$EDGE_HOST]}" "cat ${REMOTE_DIR}/edge.done 2>/dev/null")"
    case "$done_line" in
      rc=0) echo "=== the edge session finished ==="; return 0 ;;
      rc=*) echo "::error::the edge session failed (${done_line})"; return 1 ;;
    esac
    sleep 120
  done
  echo "::error::the edge session did not finish within 4 h"; return 1
}

rc=0
run() { "$@" || { echo "::error::$* failed"; rc=2; }; }
run compute start 1
[ "$rc" -eq 0 ] && run block 1 1
[ "$rc" -eq 0 ] && run compute start 2
[ "$rc" -eq 0 ] && run block 2 1
[ "$rc" -eq 0 ] && run await_edge
[ "$rc" -eq 0 ] && run compute start 3
[ "$rc" -eq 0 ] && run compute start 4
[ "$rc" -eq 0 ] && run block 4 1
[ "$rc" -eq 0 ] && run block 4 2
[ "$rc" -eq 0 ] && run compute stop 4
[ "$rc" -eq 0 ] && run compute stop 3
[ "$rc" -eq 0 ] && run block 2 2
[ "$rc" -eq 0 ] && run compute stop 2
[ "$rc" -eq 0 ] && run block 1 2

# ── pull everything, whatever happened: a failed run's logs are its evidence ─
mkdir -p "$SCALING_ARTIFACT_DIR"
pull() { # $1=host $2=remote path $3=local path
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${PORT[$1]}" "root@${HOST[$1]}:$2" "$3" \
    || echo "::warning::could not pull $2 from host $1"
}
for k in 1 2 4; do pull 0 "${REMOTE_DIR}/legs-direct-K${k}/" "$SCALING_ARTIFACT_DIR/direct-K${k}/"; done
pull "$EDGE_HOST" "${REMOTE_DIR}/legs-edge/" "$SCALING_ARTIFACT_DIR/edge/"
for i in $(seq 0 $((HOSTS - 1))); do
  mkdir -p "$SCALING_ARTIFACT_DIR/logs/host$i"
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${PORT[$i]}" --include '*.log' --include '*.env' --exclude '*' \
    "root@${HOST[$i]}:${REMOTE_DIR}/" "$SCALING_ARTIFACT_DIR/logs/host$i/" \
    || echo "::warning::could not pull host $i's logs"
done
printf '{"gpu_type": "%s", "data_center": "%s", "hosts": %d, "sha": "%s"}\n' \
  "$GPU_TYPE" "$measured_dc" "$HOSTS" "${PROVE_EXPECT_SHA:-$(git -C "$REPO_ROOT" rev-parse HEAD)}" \
  > "$SCALING_ARTIFACT_DIR/fleet.json"
[ "$rc" -eq 0 ] || exit "$rc"

# ── the judge, then D15's reading of it ─────────────────────────────────────
python3 "$DIR/plane_scaling_verdict.py" assemble "$SCALING_ARTIFACT_DIR" || exit 2
# A verdict that is not GREEN exits non-zero and is still written; D15's
# reading below is what decides, and it refuses a missing verdict.
for k in 1 2 4; do
  (cd "$REPO_ROOT" && cargo run -q -p jammi-bench -- ladder encode \
    "$SCALING_ARTIFACT_DIR/ladder-K${k}" --from plan-partitioned --to shape-d \
    --axes outcome,speed,shape --out "$SCALING_ARTIFACT_DIR/ladder-K${k}/verdict") \
    || echo "=== the K=${k} verdict is not GREEN (exit $?) ==="
done
python3 "$DIR/plane_scaling_verdict.py" decide "$SCALING_ARTIFACT_DIR" | tee "$SCALING_ARTIFACT_DIR/d15.txt"
exit "${PIPESTATUS[0]}"

fi # end sourced-execution guard
