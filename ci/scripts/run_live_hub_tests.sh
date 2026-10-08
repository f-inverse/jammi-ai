#!/usr/bin/env bash
# The tests `live-hub-tests` adds, against the live model hub: the difference
# between the workspace's test list with and without the feature, run under
# nextest's `ci` profile. A feature that adds no tests is a failure, never an
# empty green run.
#
#   HF_TOKEN=... bash ci/scripts/run_live_hub_tests.sh
set -euo pipefail

list() { cargo nextest list --workspace --exclude jammi-python "$@" --message-format oneline --color never | sort; }
filter="$(comm -13 <(list) <(list --features live-hub-tests) \
  | awk '{ printf "%s(binary_id(%s) & test(=%s))", (NR > 1 ? " | " : ""), $1, $2 }')"
test -n "$filter" || { echo "::error::live-hub-tests added no tests"; exit 1; }
cargo nextest run --profile ci --workspace --exclude jammi-python --features live-hub-tests -E "$filter"
