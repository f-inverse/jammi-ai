#!/usr/bin/env bash
# One fleet host's part in the compute plane's scaling test
# (`runpod_plane_scaling.sh`; the test and its pass bar are
# `docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md` D15). Host 0
# is the control host — the fleet's catalog, store, scheduler and query tier —
# and hosts 1..4 are its compute tier, one GPU each. Run from the checkout
# root.
#
#   plane_scaling_host.sh build <dir>
#       On host 0: build `jammi-server` and `jammi-bench` with the features
#       every GPU encode producer builds (`encode_ab.sh`), fetch the model into
#       `MODEL_DIR`, and put all three under <dir>/dist; print each one's
#       sha256 (`SERVER_SHA256=`, `BENCH_SHA256=`, `MODEL_SHA256=`).
#   plane_scaling_host.sh serve <ip> <dir>
#       On host 0: start the fleet's Postgres catalog and S3-class store on
#       <ip>, and serve <dir>/dist to the other hosts over HTTP on <ip>.
#   plane_scaling_host.sh fetch <infra-ip> <dir> <server-sha> <bench-sha> <model-sha>
#       On hosts 1..4: fetch the binaries and the model from host 0, each
#       refused unless its sha256 is the one host 0 reported.
#   plane_scaling_host.sh role <role> <ip> <infra-ip> <dir> <device>
#       Start one shape-d role (`scheduler`, `query` or `compute`) on this
#       host — the committed config, placed by `jammi-bench fleet-env` with the
#       test's inference shape — and wait until it is ready. <device> is the
#       fleet's CUDA ordinal, or -1 for a CPU fleet.
#   plane_scaling_host.sh stop <dir> <role>
#       Stop that role with SIGTERM — a DRAIN, whose executor leaves the
#       catalog's live set at once — and wait for it to exit.
#   plane_scaling_host.sh edge <dir> <out> <take>
#       The `plan-partitioned` and `placed` legs of one take, over a catalog
#       and store of this host's own, so no fleet executor is visible to them.
#   plane_scaling_host.sh direct <ip> <infra-ip> <dir> <out> <take>
#       The direct pair of one take: `plan-partitioned` in this process and
#       `shape-d` through the fleet's query tier, interleaved.
set -euo pipefail

die() { echo "plane_scaling_host.sh: $*" >&2; exit 1; }

# The workload D15 fixes.
MODEL_REPO="answerdotai/ModernBERT-large"
MODEL_DIR="/root/checkpoints/ModernBERT-large"
ROWS="16384,32768,65536"
PARTITIONS=8
BATCH_SIZE=32
BATCH_TOKENS=16384
PRECISION=f32
ITERS=32

PG_PORT=5433
S3_PORT=9000
DIST_PORT=7300
SCHEDULER_PORT=50050
PEER_PORT=7200
# A fixed, non-secret key: the server refuses to start without 32 hex bytes.
AUDIT_KEY="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

flight_port() { case "$1" in query) echo 8815 ;; scheduler) echo 8816 ;; compute) echo 8817 ;; *) die "no role '$1'" ;; esac; }
health_port() { case "$1" in query) echo 8080 ;; scheduler) echo 8081 ;; compute) echo 8082 ;; *) die "no role '$1'" ;; esac; }

sha() { sha256sum "$1" | awk '{print $1}'; }

# The fleet's backends, as the variables `DistributedBackends::from_env` reads.
backends_env() {
  local infra_ip="$1"
  bash ci/scripts/pg_test_catalog.sh env --host "$infra_ip" --port "$PG_PORT"
  bash ci/scripts/s3_test_store.sh env --addr "${infra_ip}:${S3_PORT}"
}

build() {
  local dir="$1" target
  mkdir -p "$dir/dist"
  target="${CARGO_TARGET_DIR:-$PWD/target}"
  cargo build --release -p jammi-server --bin jammi-server --features cuda,flash-attn,storage-s3
  cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn,plane
  cp "$target/release/jammi-server" "$target/release/jammi-bench" "$dir/dist/"
  python3 -m venv "$dir/hf"
  "$dir/hf/bin/pip" install -q huggingface_hub
  "$dir/hf/bin/python3" ci/scripts/perf/checkpoint_files.py --fetch "$MODEL_REPO" "$MODEL_DIR"
  # The repo snapshot carries ONNX exports and the download cache beside the
  # checkpoint; the candle loader reads neither, so neither crosses the fleet.
  tar -C "$(dirname "$MODEL_DIR")" --exclude="$(basename "$MODEL_DIR")/onnx" \
    --exclude="$(basename "$MODEL_DIR")/.cache" -cf "$dir/dist/model.tar" "$(basename "$MODEL_DIR")"
  echo "MODEL_TAR_BYTES=$(stat -c %s "$dir/dist/model.tar")"
  echo "SERVER_SHA256=$(sha "$dir/dist/jammi-server")"
  echo "BENCH_SHA256=$(sha "$dir/dist/jammi-bench")"
  echo "MODEL_SHA256=$(sha "$dir/dist/model.tar")"
}

serve() {
  local ip="$1" dir="$2"
  bash ci/scripts/pg_test_catalog.sh start --host "$ip" --port "$PG_PORT" --data "$dir/pg" >/dev/null
  bash ci/scripts/s3_test_store.sh start --addr "${ip}:${S3_PORT}" --data "$dir/s3" >/dev/null
  setsid nohup python3 -m http.server "$DIST_PORT" --bind "$ip" --directory "$dir/dist" \
    > "$dir/dist.log" 2>&1 < /dev/null &
  echo "catalog postgres://jammi@${ip}:${PG_PORT}/postgres; store http://${ip}:${S3_PORT}; dist http://${ip}:${DIST_PORT}"
}

fetch() {
  local infra_ip="$1" dir="$2" server_sha="$3" bench_sha="$4" model_sha="$5" name want
  mkdir -p "$dir/dist"
  for name in jammi-server jammi-bench model.tar; do
    case "$name" in jammi-server) want="$server_sha" ;; jammi-bench) want="$bench_sha" ;; *) want="$model_sha" ;; esac
    # A silent transfer outlives the driver's inactivity watchdog, so the
    # bytes landed are reported every minute until it ends.
    curl -fsS --retry 5 --retry-connrefused -o "$dir/dist/$name" "http://${infra_ip}:${DIST_PORT}/${name}" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      echo "fetching ${name}: $(stat -c %s "$dir/dist/$name" 2>/dev/null || echo 0) bytes at $(date -u +%T)"
    done
    wait "$pid" || die "could not fetch ${name} from ${infra_ip}:${DIST_PORT}"
    [ "$(sha "$dir/dist/$name")" = "$want" ] || die "the fetched ${name}'s sha256 is not host 0's ${want}"
  done
  chmod 0755 "$dir/dist/jammi-server" "$dir/dist/jammi-bench"
  mkdir -p "$(dirname "$MODEL_DIR")"
  tar -C "$(dirname "$MODEL_DIR")" -xf "$dir/dist/model.tar"
  echo "fetched the server, the bench and the model"
}

role() {
  local role="$1" ip="$2" infra_ip="$3" dir="$4" device="$5" bucket env_file
  [ -x "$dir/dist/jammi-server" ] || die "no server binary at $dir/dist/jammi-server"
  env_file="$dir/${role}.env"
  set -a
  eval "$(backends_env "$infra_ip")"
  set +a
  bucket="$JAMMI_TEST_S3_BUCKET"
  mkdir -p "$dir/artifacts-${role}"
  {
    backends_env "$infra_ip"
    "$dir/dist/jammi-bench" fleet-env --role "$role" \
      --result-root "s3://${bucket}/plane-scaling" --artifact-dir "$dir/artifacts-${role}" \
      --advertise-host "$ip" --scheduler-address "${infra_ip}:${SCHEDULER_PORT}" \
      $( [ "$device" -ge 0 ] && echo "--device $device" ) \
      --flight-port "$(flight_port "$role")" --health-port "$(health_port "$role")" \
      --peer-port "$PEER_PORT" --scheduler-port "$SCHEDULER_PORT" \
      --partitions "$PARTITIONS" --batch-size "$BATCH_SIZE" --batch-tokens "$BATCH_TOKENS" \
      --compute-precision "$PRECISION" | grep -v '^#'
    echo "JAMMI_WORKER_ID=${role}-${ip}"
    echo "JAMMI_AUDIT_MASTER_KEY=${AUDIT_KEY}"
    echo "RUST_LOG=${RUST_LOG:-info}"
  } > "$env_file"
  ( set -a; . "$env_file"; set +a
    exec setsid nohup "$dir/dist/jammi-server" \
      --config "deploy/kubernetes/overlays/shape-d/jammi-${role}.toml" \
      > "$dir/${role}.log" 2>&1 < /dev/null ) &
  echo "$!" > "$dir/${role}.pid"
  local deadline=$((SECONDS + 600))
  until curl -fs -o /dev/null "http://127.0.0.1:$(health_port "$role")/readyz"; do
    kill -0 "$(cat "$dir/${role}.pid")" 2>/dev/null \
      || { tail -50 "$dir/${role}.log" >&2; die "the ${role} role exited before it was ready"; }
    [ "$SECONDS" -lt "$deadline" ] || { tail -50 "$dir/${role}.log" >&2; die "the ${role} role was not ready within 600 s"; }
    sleep 1
  done
  echo "the ${role} role is ready on ${ip}"
}

stop() {
  local dir="$1" role="$2" pid deadline
  [ -f "$dir/${role}.pid" ] || { echo "no ${role} role is running here"; return 0; }
  pid="$(cat "$dir/${role}.pid")"
  kill -TERM "$pid" 2>/dev/null || { rm -f "$dir/${role}.pid"; echo "the ${role} role had already exited"; return 0; }
  deadline=$((SECONDS + 300))
  while kill -0 "$pid" 2>/dev/null; do
    [ "$SECONDS" -lt "$deadline" ] || die "the ${role} role did not exit within 300 s of its DRAIN"
    sleep 1
  done
  rm -f "$dir/${role}.pid"
  echo "the ${role} role stopped"
}

encode_step() {
  local dir="$1" out="$2" take="$3"; shift 3
  "$dir/dist/jammi-bench" encode-step --task embed --rows "$ROWS" --partitions "$PARTITIONS" \
    --batch-size "$BATCH_SIZE" --batch-tokens "$BATCH_TOKENS" --compute-precision "$PRECISION" \
    --iters "$ITERS" --take "$take" --model-dir "$MODEL_DIR" --cuda 0 \
    --exchange-dir "$out/exchange" --legs-dir "$out" "$@"
}

edge() {
  local dir="$1" out="$2" take="$3" rc=0
  mkdir -p "$out"
  bash ci/scripts/pg_test_catalog.sh start --data "$dir/edge-pg-${take}" >/dev/null
  bash ci/scripts/s3_test_store.sh start --addr "127.0.0.1:9100" --data "$dir/edge-s3-${take}" >/dev/null
  ( set -a
    eval "$(bash ci/scripts/pg_test_catalog.sh env)"
    eval "$(bash ci/scripts/s3_test_store.sh env --addr 127.0.0.1:9100)"
    set +a
    encode_step "$dir" "$out" "$take" --rung plan-partitioned --rung placed ) || rc=$?
  bash ci/scripts/s3_test_store.sh stop --data "$dir/edge-s3-${take}"
  bash ci/scripts/pg_test_catalog.sh stop --data "$dir/edge-pg-${take}"
  return "$rc"
}

direct() {
  local ip="$1" infra_ip="$2" dir="$3" out="$4" take="$5"
  mkdir -p "$out"
  set -a
  eval "$(backends_env "$infra_ip")"
  set +a
  encode_step "$dir" "$out" "$take" --rung plan-partitioned --rung shape-d \
    --query-addr "${ip}:$(flight_port query)"
}

[ "$#" -ge 1 ] || die "usage: build DIR | serve IP DIR | fetch INFRA_IP DIR SERVER_SHA BENCH_SHA MODEL_SHA | role ROLE IP INFRA_IP DIR DEVICE | stop DIR ROLE | edge DIR OUT TAKE | direct IP INFRA_IP DIR OUT TAKE"
verb="$1"; shift
case "$verb" in
  build) [ "$#" -eq 1 ] || die "build DIR"; build "$@" ;;
  serve) [ "$#" -eq 2 ] || die "serve IP DIR"; serve "$@" ;;
  fetch) [ "$#" -eq 5 ] || die "fetch INFRA_IP DIR SERVER_SHA BENCH_SHA MODEL_SHA"; fetch "$@" ;;
  role) [ "$#" -eq 5 ] || die "role ROLE IP INFRA_IP DIR DEVICE"; role "$@" ;;
  stop) [ "$#" -eq 2 ] || die "stop DIR ROLE"; stop "$@" ;;
  edge) [ "$#" -eq 3 ] || die "edge DIR OUT TAKE"; edge "$@" ;;
  direct) [ "$#" -eq 5 ] || die "direct IP INFRA_IP DIR OUT TAKE"; direct "$@" ;;
  *) die "unknown verb '$verb'" ;;
esac
