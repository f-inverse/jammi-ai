#!/usr/bin/env bash
# One leg of the distributed lane: the named tests from an archive one at a
# time (nextest's `distributed` profile), each spawning its own worker fleet
# against the shared catalog and store. Every filter must run at least one
# test: a filter that matches nothing would pass having run nothing.
#
#   bash ci/scripts/run_distributed_leg.sh ARCHIVE TEST...
set -euo pipefail

archive="${1:?ARCHIVE}"; shift
[ "$#" -gt 0 ] || { echo "::error::no tests named for $archive" >&2; exit 2; }

filter=""
for t in "$@"; do filter="${filter:+$filter | }test($t)"; done
cargo nextest run --profile distributed \
  --archive-file "$archive" \
  --workspace-remap "$GITHUB_WORKSPACE" --extract-to "$GITHUB_WORKSPACE" \
  -E "$filter"

junit=target/nextest/distributed/junit.xml
for t in "$@"; do
  grep -q "<testcase name=\"[^\"]*$t\"" "$junit" \
    || { echo "::error::filter '$t' ran no test" >&2; exit 1; }
done
