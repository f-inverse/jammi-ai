#!/usr/bin/env bash
# One fleet host's part in the GPU topology lane's multi-host cell
# (`runpod_gpu_topology.sh`): the shared catalog and store, and this host's
# `jammi-server` processes — one per GPU, each a fleet member advertising its
# gang listener on the host's private-network ip. Run from the checkout root.
#
#   gpu_topology_host.sh infra <infra-ip> <dir>
#       On the first host: start one Postgres catalog per phase
#       (`pg_test_catalog.sh`, `nccl` on 5433 and `cpu` on 5434) and the
#       S3-class store (`s3_test_store.sh`) on <infra-ip>, and build the
#       Python client into <dir>/venv.
#   gpu_topology_host.sh servers <host> <ip> <infra-ip> <collective> <dir> <gpus>
#       Start <gpus> servers on this host (server g on GPU g, flight port
#       7000+g, health 7100+g, gang listener <ip>:7200+g) over the catalog
#       and result root of the phase named <collective> — each phase its own,
#       so a phase never sees the previous phase's instances or jobs — and
#       wait until every one is ready. <collective> is the `[worker] collective` value.
#       The binary is <dir>/jammi-server; logs are <dir>/<collective>-h<host>-g<g>.log.
#   gpu_topology_host.sh stop <dir> <collective>
#       Stop this host's servers of that phase.
set -euo pipefail

die() { echo "gpu_topology_host.sh: $*" >&2; exit 1; }

S3_PORT=9000
FLIGHT_BASE=7000
HEALTH_BASE=7100
PEER_BASE=7200
# A fixed, non-secret key: the server refuses to start without 32 hex bytes.
AUDIT_KEY="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# Each phase's catalog port.
pg_port() {
  case "$1" in
    nccl) echo 5433 ;;
    cpu) echo 5434 ;;
    *) die "no catalog for collective '$1'" ;;
  esac
}

infra() {
  local ip="$1" dir="$2" phase
  mkdir -p "$dir"
  for phase in nccl cpu; do
    bash ci/scripts/pg_test_catalog.sh start --host "$ip" --port "$(pg_port "$phase")" \
      --data "$dir/pg-${phase}" >/dev/null
  done
  bash ci/scripts/s3_test_store.sh start --addr "${ip}:${S3_PORT}" --data "$dir/s3" >/dev/null
  python3 -m venv "$dir/venv"
  "$dir/venv/bin/pip" install -q --upgrade pip
  "$dir/venv/bin/pip" install -q -e 'clients/python[dev]'
  ( . "$dir/venv/bin/activate" && make -C clients/python generate >/dev/null )
  echo "catalogs postgres://jammi@${ip}:{$(pg_port nccl),$(pg_port cpu)}/postgres; store http://${ip}:${S3_PORT}"
}

server_toml() {
  local g="$1" ip="$2" infra_ip="$3" collective="$4" dir="$5"
  cat <<TOML
artifact_dir = "${dir}/artifacts-${collective}-g${g}"

[gpu]
device = ${g}

[catalog.postgres]
url = "postgres://jammi@${infra_ip}:$(pg_port "$collective")/postgres"
pool_size = 8

[storage]
result_root = "s3://jammi-dist/topology-${collective}/"

[storage.cloud.s3]
region = "us-east-1"
endpoint = "http://${infra_ip}:${S3_PORT}"
allow_http = true

[worker]
enabled = true
collective = "${collective}"

[distributed]
max_world_size = 8

[server]
flight_listen = "${ip}:$((FLIGHT_BASE + g))"
health_listen = "${ip}:$((HEALTH_BASE + g))"
peer_bind = "${ip}:$((PEER_BASE + g))"
peer_advertise = "${ip}:$((PEER_BASE + g))"
services = []
TOML
}

servers() {
  local host="$1" ip="$2" infra_ip="$3" collective="$4" dir="$5" gpus="$6" g
  [ -x "$dir/jammi-server" ] || die "no server binary at $dir/jammi-server"
  for g in $(seq 0 $((gpus - 1))); do
    server_toml "$g" "$ip" "$infra_ip" "$collective" "$dir" > "$dir/${collective}-g${g}.toml"
    mkdir -p "$dir/artifacts-${collective}-g${g}"
    JAMMI_WORKER_ID="h${host}-gpu${g}" \
    JAMMI_AUDIT_MASTER_KEY="$AUDIT_KEY" \
    AWS_ACCESS_KEY_ID=jammi-test AWS_SECRET_ACCESS_KEY=jammi-test-secret AWS_REGION=us-east-1 \
    RUST_LOG="${RUST_LOG:-info}" NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
      setsid nohup "$dir/jammi-server" --config "$dir/${collective}-g${g}.toml" \
        > "$dir/${collective}-h${host}-g${g}.log" 2>&1 < /dev/null &
    echo "$!" > "$dir/${collective}-g${g}.pid"
  done
  local deadline=$((SECONDS + 300))
  for g in $(seq 0 $((gpus - 1))); do
    until curl -fs -o /dev/null "http://${ip}:$((HEALTH_BASE + g))/readyz"; do
      kill -0 "$(cat "$dir/${collective}-g${g}.pid")" 2>/dev/null \
        || { tail -50 "$dir/${collective}-h${host}-g${g}.log" >&2; die "server h${host}-gpu${g} exited before ready"; }
      [ "$SECONDS" -lt "$deadline" ] \
        || { tail -50 "$dir/${collective}-h${host}-g${g}.log" >&2; die "server h${host}-gpu${g} not ready within 300 s"; }
      sleep 1
    done
    echo "server h${host}-gpu${g} ready (${collective})"
  done
}

stop() {
  local dir="$1" collective="$2" pidfile pid
  for pidfile in "$dir/${collective}"-g*.pid; do
    [ -f "$pidfile" ] || continue
    pid="$(cat "$pidfile")"
    kill "$pid" 2>/dev/null || echo "server ${pid} had already exited"
    rm -f "$pidfile"
  done
}

[ "$#" -ge 1 ] || die "usage: infra IP DIR | servers HOST IP INFRA_IP COLLECTIVE DIR GPUS | stop DIR COLLECTIVE"
verb="$1"; shift
case "$verb" in
  infra) [ "$#" -eq 2 ] || die "infra IP DIR"; infra "$@" ;;
  servers) [ "$#" -eq 6 ] || die "servers HOST IP INFRA_IP COLLECTIVE DIR GPUS"; servers "$@" ;;
  stop) [ "$#" -eq 2 ] || die "stop DIR COLLECTIVE"; stop "$@" ;;
  *) die "unknown verb '$verb'" ;;
esac
