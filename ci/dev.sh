#!/usr/bin/env bash
# Run a command inside the CI image against this checkout — the same
# toolchain, linker and warnings posture CI uses, on any host with Docker.
#
#   ci/dev.sh                                  # interactive shell
#   ci/dev.sh cargo test -p jammi-db
#   ci/dev.sh cargo clippy --workspace --all-targets
#   ci/dev.sh --with pg cargo test -p jammi-db --features live-postgres-tests --test it
#   ci/dev.sh --with pg,s3 cargo test -p jammi-ballista --features live-distributed-tests --test distributed
#   ci/dev.sh --scratch rebase cargo test --workspace     # a target volume that dies with the command
#   ci/dev.sh --gc                                        # remove what earlier runs left behind
#
# Build output, the cargo registry and the sccache live in named Docker
# volumes, so they persist between runs and never mix with a host `target/`.
#
# A backend `--with` provides is the one the workflows declare, reachable
# under the same host name and passed through the same variables, and it
# belongs to THIS run alone: Postgres is a sidecar on the run's own network,
# removed when the run exits; the S3-class store runs inside the run's
# container from the pinned binary `ci/scripts/s3_test_store.sh` defines (the
# same definition `distributed.yml` and a GPU pod use) — two runs never share
# a catalog or a bucket, and nothing a run starts outlives it.
#
#   --with pg[,s3]         the backends a live lane needs (Postgres 16; the
#                          S3-class store holding the lane's bucket)
#   --scratch NAME         build into the volume jammi-dev-target-NAME instead
#                          of the shared one, and remove it on exit
#   --gc                   remove every jammi-dev-* container, network and
#                          volume that is not one of the kept caches, and every
#                          dangling image; print what remains
#   JAMMI_CI_IMAGE         image to use (default: the one the workflows use)
#   JAMMI_CI_PLATFORM      e.g. linux/amd64 to match the hosted runners exactly
#                          (emulated, slow, on an arm64 host); default is native
#   JAMMI_DEV_DOCKER_ARGS  extra `docker run` arguments (ports, env)
#   JAMMI_DEV_MIN_FREE_GIB free space the host and the Docker VM must both have
#                          before a run starts (default 20): a build that fills
#                          the VM's disk halts the VM, which is worse than a
#                          refusal naming the number
set -euo pipefail

image="${JAMMI_CI_IMAGE:-ghcr.io/f-inverse/jammi-ai-ci:latest}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="jammi-dev-$$"
min_free_gib="${JAMMI_DEV_MIN_FREE_GIB:-20}"

# The volumes every run shares: the caches that make a rebuild cheap. `--gc`
# keeps exactly these.
kept_volumes=(jammi-dev-target jammi-dev-cargo-registry jammi-dev-cargo-git jammi-dev-sccache)

# The sidecar the workflows declare, by the image the workflows pin.
pg_image="postgres:16"

with=()
scratch=""
store=""
gc=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --with) IFS=, read -r -a with <<<"$2"; shift 2 ;;
    --with=*) IFS=, read -r -a with <<<"${1#--with=}"; shift ;;
    --scratch) scratch="$2"; shift 2 ;;
    --scratch=*) scratch="${1#--scratch=}"; shift ;;
    --gc) gc=1; shift ;;
    --) shift; break ;;
    -*) echo "ci/dev.sh: unknown option '$1'" >&2; exit 64 ;;
    *) break ;;
  esac
done

# `docker <verb> <ids…>` over the ids on stdin, and nothing when there are
# none (BSD xargs would still invoke the verb once, and fail on it).
each() { local ids; ids="$(cat)"; [ -z "$ids" ] || echo "$ids" | xargs docker "$@" >/dev/null; }

# A run is orphaned when the ci/dev.sh process it is named after is gone;
# what a live run holds is its own, and `--gc` leaves it alone.
orphaned() { while read -r name; do kill -0 "$(echo "$name" | sed -E 's/^jammi-dev-([0-9]+).*/\1/')" 2>/dev/null || echo "$name"; done; }

if [ -n "$gc" ]; then
  kept="^($(IFS='|'; echo "${kept_volumes[*]}"))$"
  docker ps -a --format '{{.Names}}' --filter name='^jammi-dev-' | orphaned | each rm -fv
  docker network ls --format '{{.Name}}' --filter name='^jammi-dev-' | orphaned | each network rm
  docker volume ls -q --filter name='^jammi-dev-' | { grep -Ev "$kept" || true; } \
    | { grep -Fxv -f <(docker ps --format '{{.Mounts}}' | tr ',' '\n') || true; } | each volume rm
  docker image prune -f >/dev/null
  docker system df
  exit 0
fi

# A run that fills the Docker VM's disk halts the VM; refuse before that
# point, naming the number, on both sides of the VM boundary.
free_gib_host() { df -Pk / | awk 'NR==2 {printf "%d", $4 / 1048576}'; }
free_gib_vm() {
  docker run --rm -v jammi-dev-target:/cache/target "$image" df -Pk /cache/target \
    | awk 'NR==2 {printf "%d", $4 / 1048576}'
}
for side in host vm; do
  free="$("free_gib_$side")"
  if [ "$free" -lt "$min_free_gib" ]; then
    echo "ci/dev.sh: the $side has ${free} GiB free, under the ${min_free_gib} GiB a run needs — run \`ci/dev.sh --gc\`, or lower JAMMI_DEV_MIN_FREE_GIB" >&2
    exit 75
  fi
done

target_volume="jammi-dev-target${scratch:+-$scratch}"
args=(--rm --init --name "$run"
  -v "$repo:/work" -w /work
  -v "$target_volume:/cache/target"
  -v jammi-dev-cargo-registry:/usr/local/cargo/registry
  -v jammi-dev-cargo-git:/usr/local/cargo/git
  -v jammi-dev-sccache:/cache/sccache
  -e CARGO_TARGET_DIR=/cache/target
  -e SCCACHE_DIR=/cache/sccache
  -e CARGO_TERM_COLOR=always
  # The checkout is owned by the host user, not the container's.
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/work
)
# A linked worktree's `.git` is a file naming the main checkout's git
# directory by absolute host path; mount that directory at the same path so
# git — and every gate that lists tracked files — works inside the container.
git_common="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)"
case "$git_common" in
  "$repo"/*) ;;
  *) args+=(-v "$git_common:$git_common" -e GIT_CONFIG_COUNT=2
            -e GIT_CONFIG_KEY_1=safe.directory -e "GIT_CONFIG_VALUE_1=$git_common") ;;
esac
[ -n "${JAMMI_CI_PLATFORM:-}" ] && args+=(--platform "$JAMMI_CI_PLATFORM")
[ -t 0 ] && [ -t 1 ] && args+=(-it)
# shellcheck disable=SC2206  # deliberate word-splitting of the caller's extra args
[ -n "${JAMMI_DEV_DOCKER_ARGS:-}" ] && args+=(${JAMMI_DEV_DOCKER_ARGS})

# Everything this run starts is named after it and removed when it exits,
# whichever way it exits.
cleanup() {
  # `-v` takes the anonymous volume a sidecar's image declares (Postgres's
  # data directory) with the container; without it every run leaves one.
  docker ps -aq --filter "name=^${run}" | each rm -fv 2>/dev/null || true
  docker network rm "$run" >/dev/null 2>&1 || true
  if [ -n "$scratch" ]; then docker volume rm "$target_volume" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

if [ "${#with[@]}" -gt 0 ]; then
  docker network create "$run" >/dev/null
  args+=(--network "$run")
fi
for service in "${with[@]}"; do
  case "$service" in
    pg)
      docker run -d --name "$run-postgres" --network "$run" --network-alias postgres \
        -e POSTGRES_USER=jammi -e POSTGRES_PASSWORD=jammi -e POSTGRES_DB=jammi_test "$pg_image" >/dev/null
      until docker exec "$run-postgres" pg_isready -U jammi -d jammi_test >/dev/null 2>&1; do sleep 1; done
      args+=(-e JAMMI_TEST_PG_URL=postgres://jammi:jammi@postgres:5432/jammi_test)
      ;;
    s3)
      # Started inside the run's container (below), so the harness and the
      # processes it spawns reach it on localhost exactly as the workflow's
      # job does; the binary is cached in the target volume.
      store=1
      ;;
    *) echo "ci/dev.sh: unknown backend '$service' (pg, s3)" >&2; exit 64 ;;
  esac
done

# CI fails on warnings through a per-target variable, never a bare RUSTFLAGS
# (which would replace `.cargo/config.toml`'s linker and target-feature flags).
read -r -d '' entry <<'EOF' || true
triple="$(rustc -vV | sed -n 's/^host: //p' | tr '[:lower:]-' '[:upper:]_')"
export "CARGO_TARGET_${triple}_RUSTFLAGS=-D warnings"
if [ "$#" -eq 0 ]; then exec bash; else exec "$@"; fi
EOF

if [ -n "$store" ]; then
  [ "$#" -gt 0 ] || set -- bash
  set -- ci/scripts/s3_test_store.sh run -- "$@"
fi
docker run "${args[@]}" "$image" bash -c "$entry" dev "$@"
