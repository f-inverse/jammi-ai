#!/usr/bin/env bash
# Extracts every `-C target-feature=<value>` entry from `.cargo/config.toml`'s
# `[target.<triple>]` `rustflags` array — the single source of truth for a
# target's ISA-floor flags, so `.github/actions/setup-rust-ci/action.yml`'s
# CI-side re-application (an exported RUSTFLAGS replaces, never merges with,
# config-file rustflags — see that file's own comment) never carries its own,
# second, driftable copy of the same literal (e.g. `+fp16`).
#
# Same precedent as `rust_pin.sh` reading `rust-toolchain.toml`: a pinned
# value lives in exactly one file, and CI reads it back instead of repeating
# it. Not a `check_*` script (nothing here asserts a repo invariant on every
# run — it is a value extractor a caller consumes), so it carries no
# `check_ci_guard_wiring.py` obligation, the same way `rust_pin.sh` carries
# none; its own `--self-test` below is not wired into any workflow's guard
# matrix for the same reason `rust_pin.sh` has no self-test wired at all —
# nothing in this repo requires a plain-named script's embedded self-test to
# be reachable from CI, only a `check_*`/`test_*`-named FILE.
#
# Usage:
#   rust_target_features.sh <target-triple>
#     Prints one `-C target-feature=<value>` per line to stdout for the
#     named target. Prints nothing (exits 0) if the target has no
#     `rustflags` array, or no `target-feature=` entries in it — an
#     unpinned target is not an error.
#   rust_target_features.sh --self-test
#     Asserts against the REAL `.cargo/config.toml` this repo ships (no
#     fixtures, no network) — the file's own drift is exactly what this
#     script exists to make impossible to introduce silently.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_TOML="$REPO_ROOT/.cargo/config.toml"

_extract() {
  local triple="$1"
  python3 - "$CONFIG_TOML" "$triple" << 'PYEOF'
import sys
import tomllib

config_path, triple = sys.argv[1], sys.argv[2]
with open(config_path, "rb") as f:
    config = tomllib.load(f)

target = config.get("target", {}).get(triple, {})
rustflags = target.get("rustflags", [])

# rustflags is a flat list, e.g. ["-C", "link-arg=-fuse-ld=mold", "-C",
# "target-feature=+fp16"] -- pair up "-C"/value entries and keep only the
# target-feature= ones, in the order they appear.
features = []
i = 0
while i < len(rustflags):
    if rustflags[i] == "-C" and i + 1 < len(rustflags) and rustflags[i + 1].startswith("target-feature="):
        features.append(rustflags[i + 1][len("target-feature="):])
        i += 2
    else:
        i += 1

for feature in features:
    print(feature)
PYEOF
}

_self_test() {
  local failures=0
  local got

  # Positive: aarch64-unknown-linux-gnu's rustflags carry +fp16 in the REAL
  # config file this repo ships -- the one fact this script exists to keep
  # true without a second, driftable copy.
  got="$(_extract aarch64-unknown-linux-gnu)"
  if [ "$got" = "+fp16" ]; then
    echo "self-test[aarch64-unknown-linux-gnu extracts +fp16]: OK"
  else
    echo "self-test[aarch64-unknown-linux-gnu extracts +fp16]: FAIL (got: '$got')" >&2
    failures=$((failures + 1))
  fi

  # Negative: x86_64-unknown-linux-gnu has no target-feature entries today --
  # absence is silence, never an error, and the mold-only stanza's OWN
  # entries must never leak through as a false feature.
  got="$(_extract x86_64-unknown-linux-gnu)"
  if [ -z "$got" ]; then
    echo "self-test[x86_64-unknown-linux-gnu extracts nothing]: OK"
  else
    echo "self-test[x86_64-unknown-linux-gnu extracts nothing]: FAIL (got: '$got')" >&2
    failures=$((failures + 1))
  fi

  # Negative: an unpinned/unknown triple is silence too, never a crash.
  got="$(_extract totally-unknown-target-triple)"
  if [ -z "$got" ]; then
    echo "self-test[unknown triple extracts nothing]: OK"
  else
    echo "self-test[unknown triple extracts nothing]: FAIL (got: '$got')" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "rust_target_features.sh --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "rust_target_features.sh --self-test: all 3 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

triple="${1:?usage: rust_target_features.sh <target-triple> | rust_target_features.sh --self-test}"
_extract "$triple"
