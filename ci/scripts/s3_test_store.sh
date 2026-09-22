#!/usr/bin/env bash
# The S3-class store every live lane runs against — one definition, used by
# `ci/dev.sh --with s3`, `distributed.yml`, and a GPU pod alike.
#
# The store is versitygw (Apache-2.0, https://github.com/versity/versitygw):
# a stateless S3 gateway over a POSIX directory, one static binary per
# architecture, pinned here by release and checksum. A bucket is a directory
# under the data root and an object is a file, so a lane's artifacts can be
# read off the disk when a test fails. The pin is an immutable GitHub release
# asset: no "latest" reshuffle can change the binary under a lane.
#
#   s3_test_store.sh fetch
#       Download and verify this host's binary into the cache dir
#       (`JAMMI_S3_STORE_DIR`, else `<CARGO_TARGET_DIR or ./target>/s3-test-store`);
#       print its path. A second call is a checksum and a print.
#   s3_test_store.sh start [--addr HOST:PORT] [--data DIR]
#       Start the store in the background over DIR (a fresh temp dir by
#       default) with the lane's bucket, wait until it answers, and leave it
#       running (pid in `<data>/.pid`). Default address 127.0.0.1:9000.
#   s3_test_store.sh stop [--data DIR]
#       Stop a store `start` left running.
#   s3_test_store.sh env [--addr HOST:PORT]
#       Print `KEY=VALUE` lines: the endpoint, bucket and credentials a lane
#       exports (`eval "$(… env)"`, or `>> "$GITHUB_ENV"`).
#   s3_test_store.sh run [--addr HOST:PORT] -- CMD…
#       `start`, run CMD with `env` in its environment, `stop`, exit as CMD did.
set -euo pipefail

VERSION="1.8.0"
# sha256 of `versitygw_v${VERSION}_Linux_<arch>.tar.gz`, per release asset.
SHA256_x86_64="2ba2c734d10d2c4e651d03182cb4b246656bc735a2f282db7b0b73fba6073467"
SHA256_arm64="b34051d33f5a9c457f790896acb7bd7d7e15ad8d92efb70616b924f37e401910"
BUCKET="jammi-dist"
ACCESS_KEY="jammi-test"
SECRET_KEY="jammi-test-secret"
REGION="us-east-1"

die() { echo "s3_test_store.sh: $*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else die "neither sha256sum nor shasum on PATH"; fi
}

host_arch() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) echo x86_64 ;;
    Linux-aarch64|Linux-arm64) echo arm64 ;;
    *) die "no pinned store binary for $(uname -s)/$(uname -m); the lanes run on Linux (the CI image)" ;;
  esac
}

cache_dir() { echo "${JAMMI_S3_STORE_DIR:-${CARGO_TARGET_DIR:-$PWD/target}/s3-test-store}"; }

fetch() {
  local arch dir bin tarball expected
  arch="$(host_arch)"
  dir="$(cache_dir)/versitygw-${VERSION}-${arch}"
  bin="$dir/versitygw"
  expected="$(eval "echo \"\$SHA256_${arch}\"")"
  tarball="versitygw_v${VERSION}_Linux_${arch}.tar.gz"
  if [ ! -x "$bin" ]; then
    mkdir -p "$dir"
    curl -fsSL "https://github.com/versity/versitygw/releases/download/v${VERSION}/${tarball}" -o "$dir/$tarball"
    [ "$(sha256_of "$dir/$tarball")" = "$expected" ] || die "checksum mismatch for $tarball"
    tar -xzf "$dir/$tarball" -C "$dir" --strip-components=1 "versitygw_v${VERSION}_Linux_${arch}/versitygw"
    rm -f "$dir/$tarball"
    chmod 0755 "$bin"
  fi
  # A binary that is not the pinned build is a corrupt cache, never a store.
  "$bin" --version 2>/dev/null | grep -q "Version  : ${VERSION}" || die "$bin is not versitygw ${VERSION}"
  echo "$bin"
}

ADDR="127.0.0.1:9000"
DATA=""
parse_opts() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --addr) ADDR="$2"; shift 2 ;;
      --data) DATA="$2"; shift 2 ;;
      --) shift; break ;;
      *) die "unknown option '$1'" ;;
    esac
  done
  REST=("$@")
}

answers() { curl -s -o /dev/null -w '%{http_code}' "http://${ADDR}/" 2>/dev/null | grep -qE '^[0-9]{3}$'; }

start() {
  local bin
  bin="$(fetch)"
  DATA="${DATA:-$(mktemp -d "${TMPDIR:-/tmp}/s3-test-store.XXXXXX")}"
  mkdir -p "$DATA/$BUCKET" "$DATA/.iam"
  ! answers || die "something already answers at http://${ADDR}/"
  ROOT_ACCESS_KEY="$ACCESS_KEY" ROOT_SECRET_KEY="$SECRET_KEY" \
    nohup "$bin" --port "$ADDR" --region "$REGION" --iam-dir "$DATA/.iam" posix "$DATA" \
    > "$DATA/.log" 2>&1 < /dev/null &
  echo "$!" > "$DATA/.pid"
  local i
  for i in $(seq 1 60); do
    answers && break
    kill -0 "$(cat "$DATA/.pid")" 2>/dev/null || { cat "$DATA/.log" >&2; die "the store exited before answering"; }
    sleep 0.5
  done
  answers || { cat "$DATA/.log" >&2; die "the store did not answer at http://${ADDR}/ within 30 s"; }
  echo "$DATA"
}

stop() {
  [ -n "$DATA" ] || die "stop needs --data DIR"
  [ -f "$DATA/.pid" ] || die "no store pid under $DATA"
  kill "$(cat "$DATA/.pid")" 2>/dev/null || true
  rm -f "$DATA/.pid"
}

env_lines() {
  printf 'JAMMI_TEST_S3_ENDPOINT=http://%s\nJAMMI_TEST_S3_BUCKET=%s\nAWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nAWS_REGION=%s\n' \
    "$ADDR" "$BUCKET" "$ACCESS_KEY" "$SECRET_KEY" "$REGION"
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

[ "$#" -ge 1 ] || die "usage: fetch | start [--addr A] [--data D] | stop --data D | env [--addr A] | run [--addr A] -- CMD…"
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
