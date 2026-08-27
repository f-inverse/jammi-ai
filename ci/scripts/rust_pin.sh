#!/usr/bin/env bash
# Prints the pinned Rust channel from rust-toolchain.toml — the single source
# of truth for the toolchain pin. Every workflow that needs the pin reads it
# through this script rather than restating the parse (or the version), so the
# pin cannot drift between sites without anything noticing. Fails closed if
# the channel cannot be parsed.
#
# Usage: rust_pin.sh   (run from the repo root; prints the channel to stdout)
set -euo pipefail

channel="$(sed -n 's/^ *channel *= *"\([^"]*\)".*/\1/p' rust-toolchain.toml)"
test -n "$channel" || { echo "could not parse channel from rust-toolchain.toml" >&2; exit 1; }
printf '%s\n' "$channel"
