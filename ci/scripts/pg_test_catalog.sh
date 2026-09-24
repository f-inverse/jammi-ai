#!/usr/bin/env bash
# The Postgres catalog a live lane runs against where no Postgres service is
# provided — a GPU pod's distributed rungs — beside `s3_test_store.sh`, which
# provides the store the same way.
#
# The server is PostgreSQL 16 as zonky's embedded-postgres-binaries publish it
# (Apache-2.0 packaging of the PostgreSQL release,
# https://github.com/zonkyio/embedded-postgres-binaries): a relocatable
# `initdb`/`pg_ctl`/`postgres` build per architecture, pinned here by release
# and checksum. The pin is an immutable Maven Central artifact: no "latest"
# reshuffle can change the server under a lane.
#
#   pg_test_catalog.sh fetch
#       Download and verify this host's build into the cache dir
#       (`JAMMI_PG_CATALOG_DIR`, else `<CARGO_TARGET_DIR or ./target>/pg-test-catalog`);
#       print its `bin` directory. A second call is a checksum and a print.
#   pg_test_catalog.sh start [--port PORT] [--data DIR]
#       Initialise a cluster in DIR (a fresh temp dir by default), start it in
#       the background on 127.0.0.1:PORT (default 5433), wait until it accepts
#       connections, and leave it running. Run as root — a pod's user — the
#       server runs as the `jammi-pg` system user, created on first use:
#       Postgres refuses to run as root.
#   pg_test_catalog.sh stop [--data DIR]
#       Stop a cluster `start` left running.
#   pg_test_catalog.sh env [--port PORT]
#       Print the `JAMMI_TEST_PG_URL=…` line a lane exports.
#   pg_test_catalog.sh run [--port PORT] -- CMD…
#       `start`, run CMD with `env` in its environment, `stop`, exit as CMD did.
set -euo pipefail

VERSION="16.15.0"
# sha256 of `embedded-postgres-binaries-linux-<arch>-${VERSION}.jar`.
SHA256_amd64="653abc065c682b85d3da50168fb95dc524bd85426cdec9ce3695a550ef431df2"
SHA256_arm64v8="f846a9989d686b7977d6eca9bc9d8b2b69e3f70c7eb60e032197d9c574a6136c"
USER_NAME="jammi"
SYSTEM_USER="jammi-pg"

die() { echo "pg_test_catalog.sh: $*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else die "neither sha256sum nor shasum on PATH"; fi
}

# `<maven arch> <archive name inside the jar>`.
host_arch() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) echo "amd64 postgres-linux-x86_64.txz" ;;
    Linux-aarch64|Linux-arm64) echo "arm64v8 postgres-linux-arm_64.txz" ;;
    *) die "no pinned build for $(uname -s)/$(uname -m); the lanes run on Linux (the CI image)" ;;
  esac
}

cache_dir() { echo "${JAMMI_PG_CATALOG_DIR:-${CARGO_TARGET_DIR:-$PWD/target}/pg-test-catalog}"; }

fetch() {
  local arch archive dir jar expected
  read -r arch archive <<<"$(host_arch)"
  dir="$(cache_dir)/postgres-${VERSION}-${arch}"
  jar="embedded-postgres-binaries-linux-${arch}-${VERSION}.jar"
  expected="$(eval "echo \"\$SHA256_${arch}\"")"
  if [ ! -x "$dir/bin/postgres" ]; then
    mkdir -p "$dir"
    curl -fsSL "https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-linux-${arch}/${VERSION}/${jar}" -o "$dir/$jar"
    [ "$(sha256_of "$dir/$jar")" = "$expected" ] || die "checksum mismatch for $jar"
    unzip -o -q "$dir/$jar" "$archive" -d "$dir"
    tar -xJf "$dir/$archive" -C "$dir"
    rm -f "$dir/$jar" "$dir/$archive"
  fi
  # A server that is not the pinned build is a corrupt cache, never a catalog.
  grep -q "PostgreSQL) ${VERSION%.0}" <<<"$("$dir/bin/postgres" --version)" || die "$dir/bin/postgres is not PostgreSQL ${VERSION%.0}"
  echo "$dir/bin"
}

PORT="5433"
DATA=""
parse_opts() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --port) PORT="$2"; shift 2 ;;
      --data) DATA="$2"; shift 2 ;;
      --) shift; break ;;
      *) die "unknown option '$1'" ;;
    esac
  done
  REST=("$@")
}

# Run a server command as whoever may: the invoking user, or — as root — the
# system user the cluster belongs to.
as_owner() {
  if [ "$(id -u)" = 0 ]; then
    id "$SYSTEM_USER" >/dev/null 2>&1 || useradd --system --no-create-home --shell /sbin/nologin "$SYSTEM_USER"
    runuser -u "$SYSTEM_USER" -- "$@"
  else
    "$@"
  fi
}

start() {
  local bin
  bin="$(fetch)"
  DATA="${DATA:-$(mktemp -d "${TMPDIR:-/tmp}/pg-test-catalog.XXXXXX")}"
  mkdir -p "$DATA"
  [ "$(id -u)" != 0 ] || { as_owner true; chown -R "$SYSTEM_USER" "$DATA" "$(dirname "$bin")"; }
  as_owner "$bin/initdb" -D "$DATA/cluster" -U "$USER_NAME" --auth=trust --encoding=UTF8 --locale=C >"$DATA/.initdb.log" 2>&1 \
    || { cat "$DATA/.initdb.log" >&2; die "initdb failed"; }
  as_owner "$bin/pg_ctl" -D "$DATA/cluster" -l "$DATA/.log" -w -t 60 \
    -o "-h 127.0.0.1 -p ${PORT} -k $DATA/cluster -c max_connections=400 -c fsync=off" start >/dev/null \
    || { cat "$DATA/.log" >&2; die "the catalog did not start on 127.0.0.1:${PORT}"; }
  echo "$DATA"
}

stop() {
  [ -n "$DATA" ] || die "stop needs --data DIR"
  [ -d "$DATA/cluster" ] || die "no cluster under $DATA"
  as_owner "$(fetch)/pg_ctl" -D "$DATA/cluster" -m fast -w stop >/dev/null
}

env_lines() {
  printf 'JAMMI_TEST_PG_URL=postgres://%s@127.0.0.1:%s/postgres\n' "$USER_NAME" "$PORT"
}

run() {
  [ "${#REST[@]}" -gt 0 ] || die "run needs a command after --"
  DATA="$(start)"
  local rc=0
  env $(env_lines) "${REST[@]}" || rc=$?
  stop
  rm -rf "$DATA"
  exit "$rc"
}

[ "$#" -ge 1 ] || die "usage: fetch | start [--port P] [--data D] | stop --data D | env [--port P] | run [--port P] -- CMD…"
verb="$1"; shift
parse_opts "$@"
case "$verb" in
  fetch) fetch ;;
  start) start ;;
  stop) stop ;;
  env) env_lines ;;
  run) run ;;
  *) die "unknown verb '$verb'" ;;
esac
