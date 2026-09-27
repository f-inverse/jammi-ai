#!/usr/bin/env bash
# GPU topology lane: the fine-tune gang on real GPUs at EVERY topology the
# engine lays a job out as — the one place the collective's NCCL transport and
# the compute plane's multi-host gang run on real hardware before release.
# Shared rent/run/teardown lives in runpod_lib.sh (its fleet primitives); this
# driver shares the prove lane's shapes (group markers, the verdict function,
# the rc contract) rather than a dialect of its own.
#
# WHAT IT RENTS: a FLEET of two ordinary pods (`rp_fleet_pod_create`), each
# with `RP_GPU_COUNT=2` GPUs of ONE type, co-located in ONE data center and
# joined by RunPod's Global Networking. The candidates are every
# TOPOLOGY_GPU_TYPES type (the workflow's `gpu_types` input, in order) and
# co-located data center that clears TOPOLOGY_MIN_AVAILABILITY at a secure
# per-GPU rate — read from the live catalog and printed before anything is
# rented — within TOPOLOGY_MAX_GPU_RATE; the driver walks them until both
# hosts land in one place, releasing a first host whose partner cannot. Every type must map to a compute capability
# (`rp_compute_cap_for_gpu_type`): a type outside sm_80/86/89/90 is refused
# before renting.
#
# COST BOUND (human-approved), stated with the mechanisms that enforce it:
#
#   4 GPUs x rate x TTL, where rate <= TOPOLOGY_MAX_GPU_RATE (refused above it,
#   before any create) and TTL = RP_TTL_HOURS (baked into each pod's OWN
#   entrypoint watchdog, `_rp_entrypoint_setup`, so a SIGKILLed runner still
#   bills no longer than the TTL):
#     (i) terminate-succeeds: the EXIT trap terminates both pods:
#           4 x TOPOLOGY_MAX_GPU_RATE x RP_TTL_HOURS = 4 x $4.00 x 4 h = $64.00
#           (the ceiling; the driver prints the live rate it rents at)
#     (ii) sweep-only (every terminate fails): each pod bills to its TTL, then
#          gpu-reap.yml's 6-hourly `rp_sweep` (both pods carry the ordinary pod
#          name shape) is the backstop:
#           4 x TOPOLOGY_MAX_GPU_RATE x (RP_TTL_HOURS + 6) = 4 x $4.00 x 10 h = $160.00
#   `ci/scripts/test_gpu_topology_lane.sh` re-derives both figures from this
#   script's own defaults and fails when this header and the mechanism
#   disagree.
#
# WHAT IT PROVES, in one run:
#
#   one-host (host 0, both its GPUs):
#     `gang_nccl` (jammi-ai `gpu_capability`, `live-gpu-gang-tests`) — the
#       NCCL device primitive: every verb with the inline transport's bytes, an
#       abort ends a real NCCL wait, the bounded join forms or ends at its
#       deadline;
#     `gpu::topology` (jammi-server `it`, `live-gpu-gang-tests`) — a real
#       fine-tune through the worker at 1 x 1, 1 x 2 in one process and 2
#       processes x 1, under the inline and NCCL transports, a zero-row
#       remainder rank included: one adapter across them all.
#   multi-host (both hosts, all four GPUs): four `jammi-server` processes, one
#     per GPU, sharing a Postgres catalog and an S3-class result root on host 0
#     (the pinned `s3_test_store.sh` store), each advertising its gang
#     listener on its host's Global-Networking ip. The fleet program
#     (`gpu_topology_fleet.py`) submits the same world-4 fine-tune twice — the
#     fleet restarted between with `[worker] collective = "nccl"`, then
#     `"cpu"` — plus the single-rank reference, and this driver requires: every
#     job completed; the gang's ranks spanned both hosts, one process each;
#     the coordinator logged the NCCL transport for the first and the inline
#     one for the second; the two runs' loss curves and embeddings are
#     IDENTICAL; and both are within the trainer's pre-registered 1e-4 of the
#     single rank.
#
# THE ARTIFACT: `topology.json` (both cells' records, the device, the NCCL
# version) under TOPOLOGY_ARTIFACT_DIR, pulled before teardown; a human
# reviews it and commits it under `crates/jammi-kernels/artifacts/cuda-runs/`
# (`check_cuda_run_artifacts.py`'s `topology` kind). The NCCL communicator id
# rides only the gang's own `RunRank` link between the servers — never a file,
# a log line or this artifact.
#
# TRIGGERS: `.github/workflows/gpu-topology.yml` only — the `run-topology` PR
# label and manual dispatch. Never `push:`, never `workflow_call:`; nothing may
# `uses:` that workflow (`check_gpu_prove_once.py` P7/P8).
#
# Exit 0 = every gating group passed; 75 = no co-located capacity for any
# candidate (neutral provider condition, still RED at the workflow level); 76 =
# the inactivity watchdog killed a hang; 77 = wrong tree (a host's PROVE_SHA
# disagreed with PROVE_EXPECT_SHA); 97 = a rented host is not the shape asked
# for (GPU count, compute capability, Global Networking, co-placement); 124 =
# budget cut with a gating group unresolved.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The pods' own hard deadline == this lane's approved ceiling term. An explicit
# RP_TTL_HOURS from the caller always wins, so raising it is a visible act.
RP_TTL_HOURS="${RP_TTL_HOURS:-4}"
# Two GPUs per fleet host: host 0 runs the one-host cell on both, and the
# fleet runs one server per GPU.
export RP_GPU_COUNT="${RP_GPU_COUNT:-2}"
# A cold host pulls the multi-GB CI image before sshd starts: the fleet's
# readiness window is generous, and inside the TTL.
export RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-1200}"
# The remote budget per phase: a cold CUDA build of the server and both test
# targets is the long one.
export RP_TIMEOUT="${RP_TIMEOUT:-7200}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"
REMOTE_CHECKOUT_LINES="$(rp_remote_checkout_lines "${GIT_REF}" "${GIT_REPO}")"

# The candidate GPU types, in preference order (newline- or |-separated), the
# availability floor, and the per-GPU rate ceiling ($/h, secure cloud).
TOPOLOGY_GPU_TYPES="${TOPOLOGY_GPU_TYPES:-NVIDIA A100-SXM4-80GB|NVIDIA RTX A6000|NVIDIA A40|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL}"
TOPOLOGY_MIN_AVAILABILITY="${TOPOLOGY_MIN_AVAILABILITY:-LOW}"
TOPOLOGY_MAX_GPU_RATE="${TOPOLOGY_MAX_GPU_RATE:-4.00}"
TOPOLOGY_DATA_CENTER="${TOPOLOGY_DATA_CENTER:-}"

# The ONE place each proof's name filter lives.
ONE_HOST_DEVICE_FILTER="gang_nccl"
ONE_HOST_PRODUCT_FILTER="gpu::topology"

TOPOLOGY_ARTIFACT_DIR="${TOPOLOGY_ARTIFACT_DIR:-.gpu-pull/gpu-topology}"
TOPOLOGY_REMOTE_DIR="${RP_REMOTE_ROOT}/topology"

# The gating groups the verdict reads `PROVE_GROUP_RC` markers for, per host
# script — `::group::` names below, verbatim. Declared BEFORE the
# sourced-execution guard so a fixture can `source` this file and read them.
HOST0_GROUPS=(topology-build one-host-device one-host-product fleet-infra)
HOST1_GROUPS=(fleet-checkout)
FLEET_GROUPS=(fleet-nccl fleet-inline fleet-compare)

zero_test_device="$(_rp_zero_test_tripwire_lines grc '$device_log' "${ONE_HOST_DEVICE_FILTER}")"
zero_test_product="$(_rp_zero_test_tripwire_lines grc '$product_log' "${ONE_HOST_PRODUCT_FILTER}")"

# Verdict for one remote script: `rp_gang_verdict`'s rule (a zero ssh status
# may still become a FAILURE through a group whose marker is missing or
# non-zero; an in-suite exit is returned verbatim; a cut/hang is returned
# as-is). $1=raw ssh rc $2=log $3..=the gating groups.
rp_topology_verdict() {
  local raw_rc="$1" log="$2"; shift 2
  local -a groups=("$@")
  declare -A grc_map=()
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    if rp_parse_prove_marker "$line"; then
      grc_map["$RP_PARSED_MARKER_NAME"]="$RP_PARSED_MARKER_RC"
    fi
  done < "$log"
  local has_prove_exit=0
  if grep -q '^PROVE_EXIT=' "$log" 2>/dev/null; then # tripwire-ok: grep's own stderr on an unreadable log is not evidence; has_prove_exit=0 is the fail-closed reading, and an unreadable log also loses every group marker, which the all_pass rule then reports by name.
    has_prove_exit=1
  fi
  local all_pass=1 g v
  local -a missing_or_failed=()
  for g in "${groups[@]}"; do
    v="${grc_map[$g]:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -ne 0 ]; then
      all_pass=0
      missing_or_failed+=("${g}=${v:-<missing>}")
    fi
  done
  local rc="$raw_rc"
  if [ "$raw_rc" -eq 0 ] && [ "$all_pass" -ne 1 ]; then
    echo "::error::GPU topology: exited 0 but group(s) missing or non-zero: ${missing_or_failed[*]}" >&2
    rc=1
  elif [ "$raw_rc" -ne 0 ] && [ "$has_prove_exit" -eq 0 ] && [ "$all_pass" -ne 1 ]; then
    echo "::error::GPU topology: cut/hang (raw rc=${raw_rc}) with group(s) unresolved: ${missing_or_failed[*]}" >&2
  fi
  return "$rc"
}

# Every place this lane may rent its fleet, in preference order: one line
# `<gpuTypeId>|<rate>|<dataCenterId>` per candidate type (TOPOLOGY_GPU_TYPES
# order) and co-located Global-Networking data center (sorted) where the type
# clears TOPOLOGY_MIN_AVAILABILITY at RP_GPU_COUNT and its secure per-GPU rate
# clears TOPOLOGY_MAX_GPU_RATE — only TOPOLOGY_DATA_CENTER when one is named.
# An availability level is not a slot count (LOW may hold one 2-GPU pod), so
# the driver walks these until both hosts land in one place. $1=pod catalog
# body $2=Global-Networking data centers (space-separated). rc 75 when none
# qualifies; 2 when a candidate type maps to no compute capability.
rp_topology_candidates() {
  local catalog="$1" gn_dcs="$2" type dcs rate co dc found=0
  local IFS_SAVE="$IFS"
  IFS='|'
  # shellcheck disable=SC2086
  set -- $TOPOLOGY_GPU_TYPES
  IFS="$IFS_SAVE"
  for type in "$@"; do
    [ -n "$type" ] || continue
    rp_compute_cap_for_gpu_type "$type" >/dev/null || {
      echo "::error::candidate GPU type '${type}' maps to no compute capability (sm_80/86/89/90) -- refused" >&2
      return 2
    }
    dcs="$(printf '%s' "$catalog" | rp_fleet_pick_data_centers "$type" "$TOPOLOGY_MIN_AVAILABILITY")"
    case "$dcs" in PARSE_ERROR*) echo "::error::${dcs}" >&2; return 75 ;; esac
    co="$(rp_fleet_intersect "$dcs" "$gn_dcs")"
    [ -z "$TOPOLOGY_DATA_CENTER" ] || co="$(rp_fleet_intersect "$co" "$TOPOLOGY_DATA_CENTER")"
    [ -n "$co" ] || { echo "no co-located capacity for ${type}" >&2; continue; }
    rate="$(printf '%s' "$catalog" | python3 -c '
import json, sys
t = sys.argv[1]
d = json.load(sys.stdin)
e = next((g for g in d.get("gpus", []) if g.get("id") == t), {})
p = (e.get("price") or {}).get("secure")
print("" if p is None else p)
' "$type")"
    [ -n "$rate" ] || { echo "no secure rate listed for ${type}" >&2; continue; }
    if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$rate" "$TOPOLOGY_MAX_GPU_RATE"; then
      echo "${type} at \$${rate}/GPU/h exceeds the \$${TOPOLOGY_MAX_GPU_RATE} ceiling" >&2
      continue
    fi
    for dc in $co; do
      printf '%s|%s|%s\n' "$type" "$rate" "$dc"
      found=1
    done
  done
  [ "$found" -eq 1 ] || return 75
}

# Everything below runs only when this file is EXECUTED, never when it is
# `source`d (so a fixture reaches the pure functions above without renting).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

rp_sweep
rp_init

echo "=== availability read (product=POD, count=${RP_GPU_COUNT}, cloud=SECURE) ==="
avail_resp="$(_rp_rest GET "/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=${RP_GPU_COUNT}&cloud=SECURE")"
avail_status="$(printf '%s\n' "$avail_resp" | head -n1)"
avail_body="$(printf '%s\n' "$avail_resp" | tail -n +2)"
[ "$avail_status" = "200" ] || { echo "::error::pod catalog read failed (status ${avail_status})"; exit 75; }
gn_resp="$(_rp_rest GET /v2/catalog/datacenters)"
gn_status="$(printf '%s\n' "$gn_resp" | head -n1)"
[ "$gn_status" = "200" ] || { echo "::error::datacenters read failed (status ${gn_status})"; exit 75; }
gn_dcs="$(printf '%s\n' "$gn_resp" | tail -n +2 | rp_fleet_global_network_datacenters)"
case "$gn_dcs" in PARSE_ERROR*) echo "::error::${gn_dcs}"; exit 75 ;; esac
candidates="$(rp_topology_candidates "$avail_body" "$gn_dcs")" || exit $?

fleet_pod_0=""; fleet_pod_1=""
cleanup_fleet() {
  local rc="${1:-$?}" id
  for id in "$fleet_pod_0" "$fleet_pod_1"; do
    [ -n "$id" ] || continue
    rp_terminate "$id" >/dev/null || echo "::warning::terminate ${id} refused -- its TTL and gpu-reap.yml's sweep remain the backstop"
  done
  exit "$rc"
}
trap cleanup_fleet EXIT
trap 'cleanup_fleet 129' HUP
trap 'cleanup_fleet 130' INT
trap 'cleanup_fleet 143' TERM

# Both hosts in one place, or neither: a first host whose partner cannot be
# placed there is released before the next candidate is tried.
while IFS='|' read -r GPU_TYPE GPU_RATE chosen_dc; do
  echo "=== ${GPU_TYPE} at \$${GPU_RATE}/GPU/h x ${RP_GPU_COUNT} GPUs x 2 hosts in ${chosen_dc}: $(python3 -c "print(round(float('${GPU_RATE}')*${RP_GPU_COUNT}*2, 2))")/h, TTL ${RP_TTL_HOURS}h ==="
  fleet_pod_0="$(rp_fleet_pod_create "$GPU_TYPE" "$chosen_dc" 0)" || { fleet_pod_0=""; continue; }
  fleet_pod_1="$(rp_fleet_pod_create "$GPU_TYPE" "$chosen_dc" 1)" && break
  fleet_pod_1=""
  rp_terminate "$fleet_pod_0" >/dev/null || echo "::warning::terminate ${fleet_pod_0} refused -- its TTL and gpu-reap.yml's sweep remain the backstop"
  fleet_pod_0=""
done <<< "$candidates"
[ -n "$fleet_pod_1" ] || { echo "::error::no candidate placed both hosts in one data center (SUPPLY_CONSTRAINT)"; exit 75; }
NATIVE_COMPUTE_CAP="$(rp_compute_cap_for_gpu_type "$GPU_TYPE")"
echo "=== fleet pods ${fleet_pod_0} (host 0), ${fleet_pod_1} (host 1) ==="
ready="$(rp_fleet_wait_ready "$fleet_pod_0" "$fleet_pod_1" "$RP_SSH_WAIT_SECS")" || exit $?
IFS=' ' read -r HOST0 PORT0 HOST1 PORT1 measured_dc GN_IP0 GN_IP1 <<< "$ready"
echo "=== both hosts RUNNING in ${measured_dc}; Global-Networking ips ${GN_IP0}, ${GN_IP1} ==="
rp_wait_sshd "$HOST0" "$PORT0" "$RP_SSH_WAIT_SECS" "host 0" || exit 76
rp_wait_sshd "$HOST1" "$PORT1" "$RP_SSH_WAIT_SECS" "host 1" || exit 76

# One remote script on one host, under the inactivity watchdog. $1=host $2=port.
on_host() {
  RP_HOST="$1" RP_PORT="$2" rp_run_remote_watched ""
}

# The device assertions every host script opens with: the GPU count and the
# compute capability are what this lane rented.
device_lines() {
  cat <<DEVICE
export CARGO_TERM_COLOR=never CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
gpu_seen="\$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)"
[ "\$gpu_seen" = "${RP_GPU_COUNT}" ] || { echo "::error::nvidia-smi reports \$gpu_seen GPU(s), this lane rented ${RP_GPU_COUNT}"; exit 97; }
cap="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:].')"
[ "\$cap" = "${NATIVE_COMPUTE_CAP}" ] || { echo "::error::compute_cap \$cap, this lane rented sm_${NATIVE_COMPUTE_CAP}"; exit 97; }
DEVICE
}


# Every later remote session re-enters the checkout and names its tree, so
# the watched runner's wrong-tree rule holds for every session, not the first.
CHECKOUT_LINES="cd ${RP_REMOTE_ROOT}/jammi-ai || exit 1
echo \"PROVE_SHA=\$(git rev-parse HEAD)\""
HOST_SCRIPT="bash ci/scripts/gpu_topology_host.sh"

# One gating group around one command, as remote text. $1=group $2=command.
group_lines() {
  cat <<GROUP
echo "::group::$1"
grc=0
$2 || grc=\$?
echo "PROVE_GROUP_RC name=$1 rc=\${grc}"
echo "::endgroup::"
echo "PROVE_EXIT=\${grc}"; exit \$grc
GROUP
}

# ── host 0: build, the one-host cell, the fleet's catalog and store ─────────
HOST0_LOG="$(mktemp)"
on_host "$HOST0" "$PORT0" <<REMOTE | tee "$HOST0_LOG"
$(device_lines)
${REMOTE_CHECKOUT_LINES}
echo "PROVE_SHA=\$(git rev-parse HEAD)"
rc=0
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass || exit 1
mkdir -p ${TOPOLOGY_REMOTE_DIR}
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv > ${TOPOLOGY_REMOTE_DIR}/device.csv

echo "::group::topology-build"
grc=0
cargo build --release -p jammi-server --bin jammi-server --features cuda,flash-attn,mysql,postgres,storage-cloud || grc=\$?
cargo test -p jammi-ai --features cuda,live-gpu-gang-tests --test gpu_capability --no-run || grc=\$?
cargo test -p jammi-server --features cuda,live-gpu-gang-tests --test it --no-run || grc=\$?
[ "\$grc" -ne 0 ] || cp "\${CARGO_TARGET_DIR:-\$PWD/target}/release/jammi-server" ${TOPOLOGY_REMOTE_DIR}/jammi-server || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=topology-build rc=\${grc}"
echo "::endgroup::"

echo "::group::one-host-device"
grc=0
device_log=${TOPOLOGY_REMOTE_DIR}/one-host-device.log
cargo test -p jammi-ai --features cuda,live-gpu-gang-tests --test gpu_capability ${ONE_HOST_DEVICE_FILTER} -- --test-threads=1 2>&1 | tee "\$device_log"
grc=\${PIPESTATUS[0]}
${zero_test_device}
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=one-host-device rc=\${grc}"
echo "::endgroup::"

echo "::group::one-host-product"
grc=0
product_log=${TOPOLOGY_REMOTE_DIR}/one-host-product.log
cargo test -p jammi-server --features cuda,live-gpu-gang-tests --test it -- ${ONE_HOST_PRODUCT_FILTER} --test-threads=1 2>&1 | tee "\$product_log"
grc=\${PIPESTATUS[0]}
${zero_test_product}
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=one-host-product rc=\${grc}"
echo "::endgroup::"

echo "::group::fleet-infra"
grc=0
${HOST_SCRIPT} infra ${GN_IP0} ${TOPOLOGY_REMOTE_DIR} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=fleet-infra rc=\${grc}"
echo "::endgroup::"
echo "PROVE_EXIT=\${rc}"; exit \$rc
REMOTE
rp_topology_verdict "${PIPESTATUS[0]}" "$HOST0_LOG" "${HOST0_GROUPS[@]}"
rc=$?

# ── host 1: the checkout, and host 0's server binary ────────────────────────
if [ "$rc" -eq 0 ]; then
  HOST1_LOG="$(mktemp)"
  on_host "$HOST1" "$PORT1" <<REMOTE | tee "$HOST1_LOG"
$(device_lines)
${REMOTE_CHECKOUT_LINES}
echo "PROVE_SHA=\$(git rev-parse HEAD)"
$(group_lines fleet-checkout "mkdir -p ${TOPOLOGY_REMOTE_DIR}")
REMOTE
  rp_topology_verdict "${PIPESTATUS[0]}" "$HOST1_LOG" "${HOST1_GROUPS[@]}"
  rc=$?
fi
if [ "$rc" -eq 0 ]; then
  relay="$(mktemp -d)"
  if ! scp "${RP_SSHO[@]}" -P "$PORT0" "root@${HOST0}:${TOPOLOGY_REMOTE_DIR}/jammi-server" "$relay/" \
     || ! scp "${RP_SSHO[@]}" -P "$PORT1" "$relay/jammi-server" "root@${HOST1}:${TOPOLOGY_REMOTE_DIR}/jammi-server"; then
    echo "::error::could not relay host 0's jammi-server to host 1"
    rc=1
  fi
  rm -rf "$relay"
fi

# ── the fleet, once per transport ───────────────────────────────────────────
# Per phase: host 1's servers, then one host-0 session that starts its own,
# runs the fleet program and stops them; then host 1's are stopped. A server
# that never became ready fails that phase's group by name.
FLEET_LOG="$(mktemp)"
fleet_phase() {
  local phase="$1" group="fleet-${1/cpu/inline}" reference="" start_rc
  [ "$phase" = nccl ] && reference="--reference"
  on_host "$HOST1" "$PORT1" <<REMOTE
${CHECKOUT_LINES}
$(rp_fleet_iface_lines "$GN_IP1")
${HOST_SCRIPT} servers 1 ${GN_IP1} ${GN_IP0} ${phase} ${TOPOLOGY_REMOTE_DIR} ${RP_GPU_COUNT}
REMOTE
  start_rc=$?
  on_host "$HOST0" "$PORT0" <<REMOTE | tee -a "$FLEET_LOG"
${CHECKOUT_LINES}
$(rp_fleet_iface_lines "$GN_IP0")
echo "::group::${group}"
grc=${start_rc}
[ "\$grc" -ne 0 ] && echo "::error::host 1's ${phase} servers did not start (rc \$grc)"
[ "\$grc" -ne 0 ] || ${HOST_SCRIPT} servers 0 ${GN_IP0} ${GN_IP0} ${phase} ${TOPOLOGY_REMOTE_DIR} ${RP_GPU_COUNT} || grc=\$?
[ "\$grc" -ne 0 ] || ${TOPOLOGY_REMOTE_DIR}/venv/bin/python ci/scripts/gpu_topology_fleet.py \\
  --endpoint grpc://${GN_IP0}:7000 \\
  --pairs ${RP_REMOTE_ROOT}/jammi-ai/tests/fixtures/training_triplets.csv \\
  --model local:${RP_REMOTE_ROOT}/jammi-ai/cookbook/fixtures/tiny_bert \\
  --world-size $((RP_GPU_COUNT * 2)) --per-rank-batch 2 --expect-hosts 2 ${reference} \\
  > ${TOPOLOGY_REMOTE_DIR}/fleet-${phase}.json || grc=\$?
cat ${TOPOLOGY_REMOTE_DIR}/fleet-${phase}.json 2>/dev/null | head -c 4000
${HOST_SCRIPT} stop ${TOPOLOGY_REMOTE_DIR} ${phase}
echo "PROVE_GROUP_RC name=${group} rc=\${grc}"
echo "::endgroup::"
REMOTE
  on_host "$HOST1" "$PORT1" <<REMOTE
${CHECKOUT_LINES}
${HOST_SCRIPT} stop ${TOPOLOGY_REMOTE_DIR} ${phase}
REMOTE
}
if [ "$rc" -eq 0 ]; then
  fleet_phase nccl
  fleet_phase cpu
fi

# ── pull everything, then judge across the runs ─────────────────────────────
# Unconditional: a failed run's logs are the evidence it leaves behind, and
# this driver is the last thing able to reach them.
mkdir -p "$TOPOLOGY_ARTIFACT_DIR/host0" "$TOPOLOGY_ARTIFACT_DIR/host1"
for host in 0 1; do
  if [ "$host" = 0 ]; then h="$HOST0"; p="$PORT0"; else h="$HOST1"; p="$PORT1"; fi
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${p}" \
    --exclude venv --exclude jammi-server --exclude "pg-*" --exclude s3 --exclude 'artifacts-*' \
    "root@${h}:${TOPOLOGY_REMOTE_DIR}/" "$TOPOLOGY_ARTIFACT_DIR/host${host}/" \
    || { echo "::error::could not pull host ${host}'s evidence"; [ "$rc" -ne 0 ] || rc=1; }
done
if [ "$rc" -eq 0 ]; then
  {
    if python3 "$DIR/gpu_topology_assemble.py" "$TOPOLOGY_ARTIFACT_DIR" \
         --sha "${PROVE_EXPECT_SHA:-$(git rev-parse HEAD)}" --gpu-type "$GPU_TYPE" \
         --data-center "$measured_dc" --gpus-per-host "$RP_GPU_COUNT" \
         > "$TOPOLOGY_ARTIFACT_DIR/topology.json"; then
      echo "PROVE_GROUP_RC name=fleet-compare rc=0"
    else
      echo "PROVE_GROUP_RC name=fleet-compare rc=1"
    fi
    echo "PROVE_EXIT=0"
  } >> "$FLEET_LOG"
  cat "$TOPOLOGY_ARTIFACT_DIR/topology.json"
  rp_topology_verdict 0 "$FLEET_LOG" "${FLEET_GROUPS[@]}"
  rc=$?
fi

echo "=== GPU topology lane exit=${rc} ==="
exit "$rc"

fi # end sourced-execution guard
