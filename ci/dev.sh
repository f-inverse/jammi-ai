#!/usr/bin/env bash
# Run a command inside the CI image against this checkout — the same
# toolchain, linker and warnings posture CI uses, on any host with Docker.
#
#   ci/dev.sh                                  # interactive shell
#   ci/dev.sh cargo test -p jammi-db
#   ci/dev.sh cargo clippy --workspace --all-targets
#
# Build output, the cargo registry and the sccache live in named Docker
# volumes, so they persist between runs and never mix with a host `target/`.
#
#   JAMMI_CI_IMAGE     image to use (default: the one the workflows use)
#   JAMMI_CI_PLATFORM  e.g. linux/amd64 to match the hosted runners exactly
#                      (emulated, slow, on an arm64 host); default is native
#   JAMMI_DEV_DOCKER_ARGS  extra `docker run` arguments (ports, networks, env)
set -euo pipefail

image="${JAMMI_CI_IMAGE:-ghcr.io/f-inverse/jammi-ai-ci:latest}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

args=(--rm --init
  -v "$repo:/work" -w /work
  -v jammi-dev-target:/cache/target
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

# CI fails on warnings through a per-target variable, never a bare RUSTFLAGS
# (which would replace `.cargo/config.toml`'s linker and target-feature flags).
read -r -d '' entry <<'EOF' || true
triple="$(rustc -vV | sed -n 's/^host: //p' | tr '[:lower:]-' '[:upper:]_')"
export "CARGO_TARGET_${triple}_RUSTFLAGS=-D warnings"
if [ "$#" -eq 0 ]; then exec bash; else exec "$@"; fi
EOF

exec docker run "${args[@]}" "$image" bash -c "$entry" dev "$@"
