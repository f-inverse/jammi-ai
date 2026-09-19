#!/usr/bin/env bash
# A test that needs something the host may lack is compiled only under a
# `live-*` (or `unprivileged-tests`) feature and acquires it through
# `jammi-test-resources`, which returns the resource or fails naming it. There
# is no skip left for a require flag to turn into a failure, so a
# `JAMMI_REQUIRE_*` name under `crates/` is the signature of a test that reads
# green when its resource is missing.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if hits="$(git grep -n -E 'JAMMI_REQUIRE_[A-Z]' -- crates)"; then
  printf '%s\n' "$hits"
  echo "a JAMMI_REQUIRE_* flag gates a run-time skip: select the test with a feature and acquire its resource through jammi-test-resources" >&2
  exit 1
fi
