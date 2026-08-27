#!/usr/bin/env bash
# Client-substrate dependency-boundary guards (wired from ci.yml's
# `test-clients` job).
#
# Two modes, one mechanism:
#
#   substrate — the candle-free client substrate: the wire transport
#     (`jammi-wire`), the control-plane client (`jammi-admin`), and the
#     data-plane client (`jammi-client`) must pull NO candle / hf-hub /
#     symphonia / tokenizers ML stack.
#   cli — the `jammi` CLI is a strict control-plane client: it depends on
#     `jammi-admin`, NOT `jammi-ai`, so it must stay candle-free AND carry no
#     `jammi-ai` edge at all.
#
# The workspace build feature-unifies `local` on via jammi-server /
# jammi-python, so `cargo tree` would falsely flag candle; the real
# per-package build set is what `cargo build -p X` compiles, so both guards
# inspect the compiler-artifact stream of the actual isolated build.
#
# Fail-closed on the build itself: the JSON stream goes to a file and cargo's
# own exit code is checked (via `set -e`) before any grep runs. A plain
# `cargo build --message-format=json | grep` pipeline has no pipefail, so a
# FAILED build could still exit 0 through grep's status and report the
# boundary green on a build that never completed.
#
# Usage: check_client_deps.sh substrate|cli
set -euo pipefail

mode="${1:?usage: check_client_deps.sh substrate|cli}"

# The forbidden embedded-engine ML stack, matched against the artifact names
# of the isolated build.
ML_DEP_RE='"name":"(candle[^"]*|hf-hub|hf_hub|symphonia|tokenizers)"'

case "$mode" in
  substrate)
    packages=(-p jammi-wire -p jammi-admin -p jammi-client)
    ml_error="the client substrate compiled an embedded-engine ML dep — the candle-free boundary regressed"
    ml_ok="wire / admin / client build set is candle-free"
    ;;
  cli)
    packages=(-p jammi-cli)
    ml_error="the jammi CLI compiled an embedded-engine ML dep — the strict-client boundary regressed"
    ml_ok="jammi-cli build set is candle-free"
    ;;
  *)
    echo "unknown mode: ${mode} (expected substrate|cli)" >&2
    exit 2
    ;;
esac

json="$(mktemp)"
trap 'rm -f "$json"' EXIT

# `set -e` aborts here on a failed build — cargo's exit code is the gate,
# never a downstream grep's.
cargo build "${packages[@]}" --message-format=json > "$json"

# Every `"name":"…"` token on the compiler-artifact lines — the per-package
# build set the isolated build above actually compiled.
artifacts="$(grep '"reason":"compiler-artifact"' "$json" | grep -oE '"name":"[^"]*"' || true)"

if printf '%s\n' "$artifacts" | grep -iE "$ML_DEP_RE"; then
  echo "::error::${ml_error}"
  exit 1
fi
echo "$ml_ok"

if [ "$mode" = "cli" ]; then
  if printf '%s\n' "$artifacts" | grep -iE '"name":"jammi[-_]ai"'; then
    echo "::error::the jammi CLI compiled jammi-ai — the strict-client boundary regressed"
    exit 1
  fi
  echo "jammi-cli build set carries no jammi-ai edge"
fi
