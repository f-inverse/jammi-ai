#!/usr/bin/env bash
# A test that needs something the host may lack is compiled only under a
# `live-*` (or `unprivileged-tests`) feature and acquires it through
# `jammi-test-resources`, which returns the resource or fails naming it. There
# is no skip left for a `JAMMI_REQUIRE_*` flag to turn into a failure, so the
# name anywhere a test, lane or doc lives is the signature of a test that reads
# green when its resource is missing — or of a lane or doc describing one.
#
# Excluded: committed run evidence, which records the invocation that ran at
# its own commit, and the evidence checker that reads those records.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if hits="$(git grep -n -E 'JAMMI_REQUIRE_[A-Z]' -- crates ci .github docs README.md \
  ':!crates/jammi-kernels/artifacts' ':!ci/artifacts' ':!docs/plans' \
  ':!ci/scripts/check_cuda_run_artifacts.py' ':!ci/scripts/check_no_require_flags.sh')"; then
  printf '%s\n' "$hits"
  echo "a JAMMI_REQUIRE_* flag gates a run-time skip: select the test with a feature and acquire its resource through jammi-test-resources" >&2
  exit 1
fi
