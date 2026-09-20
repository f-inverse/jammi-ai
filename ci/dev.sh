#!/usr/bin/env bash
# Run a command inside the CI image against this checkout — the same
# toolchain, linker and warnings posture CI uses, on any host with Docker.
#
#   ci/dev.sh                                  # interactive shell
#   ci/dev.sh cargo test -p jammi-db
#   ci/dev.sh cargo clippy --workspace --all-targets
#   ci/dev.sh --with pg cargo test -p jammi-db --features live-postgres-tests --test it
#   ci/dev.sh --with pg,minio cargo test -p jammi-ballista --features live-distributed-tests --test distributed
#   ci/dev.sh --scratch rebase cargo test --workspace     # a target volume that dies with the command
#   ci/dev.sh --gc                                        # remove what earlier runs left behind
#
# Build output, the cargo registry and the sccache live in named Docker
# volumes, so they persist between runs and never mix with a host `target/`.
#
# A sidecar `--with` starts is the service the workflows declare (`ci.yml`'s
# and `distributed.yml`'s `services:`), reachable under the same host name
# and passed through the same variables, on a network and with a database
# that belong to THIS run alone and are removed when it exits — two runs never
# share a catalog, and nothing a run starts outlives it.
#
#   --with pg[,minio]      sidecars for this run (Postgres 16; MinIO holding
#                          the workflows' bucket)
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

# The services the workflows declare, by the images the workflows pin.
pg_image="postgres:16"
minio_image="quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z"
minio_bucket="jammi-dist"

with=()
scratch=""
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

if [ -n "$gc" ]; then
  kept="^($(IFS='|'; echo "${kept_volumes[*]}"))$"
  docker ps -aq --filter name='^jammi-dev-' | each rm -f
  docker network ls -q --filter name='^jammi-dev-' | each network rm
  docker volume ls -q --filter name='^jammi-dev-' | { grep -Ev "$kept" || true; } | each volume rm
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
  docker ps -aq --filter "name=^${run}" | each rm -f 2>/dev/null || true
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
    minio)
      docker run -d --name "$run-minio" --network "$run" --network-alias minio \
        -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin "$minio_image" \
        server /data --address :9000 >/dev/null
      until docker exec "$run-minio" mc alias set local http://127.0.0.1:9000 minioadmin minioadmin >/dev/null 2>&1; do sleep 1; done
      docker exec "$run-minio" mc mb --ignore-existing "local/$minio_bucket" >/dev/null
      args+=(-e JAMMI_TEST_S3_ENDPOINT=http://minio:9000 -e "JAMMI_TEST_S3_BUCKET=$minio_bucket"
             -e AWS_ACCESS_KEY_ID=minioadmin -e AWS_SECRET_ACCESS_KEY=minioadmin -e AWS_REGION=us-east-1)
      ;;
    *) echo "ci/dev.sh: unknown sidecar '$service' (pg, minio)" >&2; exit 64 ;;
  esac
done

# CI fails on warnings through a per-target variable, never a bare RUSTFLAGS
# (which would replace `.cargo/config.toml`'s linker and target-feature flags).
read -r -d '' entry <<'EOF' || true
triple="$(rustc -vV | sed -n 's/^host: //p' | tr '[:lower:]-' '[:upper:]_')"
export "CARGO_TARGET_${triple}_RUSTFLAGS=-D warnings"
if [ "$#" -eq 0 ]; then exec bash; else exec "$@"; fi
EOF

docker run "${args[@]}" "$image" bash -c "$entry" dev "$@"
