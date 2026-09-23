#!/usr/bin/env bash
# A test either runs its assertions or is not selected. One that needs
# something the host may lack is selected by the lane that offers it — a
# `live-*` (or `unprivileged-tests`) cargo feature, a pytest marker, a guard's
# `needs` in `ci/guards.toml`, a script test's `# lane:` line — and acquires
# it or fails naming it.
# Two signatures of a test that instead reads green when its resource is
# missing, or of a lane or doc describing one:
#
#   - a `JAMMI_REQUIRE_*` flag: there is no skip left for one to turn into a
#     failure;
#   - a call into a test framework's run-time skip API.
#
# Excluded: committed run evidence, which records the invocation that ran at
# its own commit, and the evidence checker that reads those records.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
status=0
if hits="$(git grep -n -E 'JAMMI_REQUIRE_[A-Z]' -- crates ci .github docs README.md \
  ':!crates/jammi-kernels/artifacts' ':!ci/artifacts' ':!docs/plans' \
  ':!ci/scripts/check_cuda_run_artifacts.py' ':!ci/scripts/check_no_runtime_skips.sh')"; then
  printf '%s\n' "$hits"
  echo "a JAMMI_REQUIRE_* flag gates a run-time skip: select the test with a feature and acquire its resource through jammi-test-resources" >&2
  status=1
fi
if hits="$(git grep -n -E 'skipTest|skipUnless|skipIf|SkipTest|unittest\.skip|pytest\.skip|importorskip|mark\.skip' -- '*.py')"; then
  printf '%s\n' "$hits"
  echo "a run-time skip reads green when its resource is missing: select the test where the resource is (a marker, or a guard's needs and lane) and fail naming it inside" >&2
  status=1
fi
exit "$status"
