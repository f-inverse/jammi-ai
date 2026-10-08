#!/usr/bin/env bash
# The permission-fault tests, run as an unprivileged account. Tests of how
# the engine handles an unreadable or undeletable file need a process that
# mode bits apply to, and the CI image runs as root, so they are compiled only
# under the `unprivileged-tests` feature. For each test target below this
# builds the binary with and without the feature, takes exactly the tests the
# feature adds, and runs each one as `jammi-probe`. Each must report one
# passed test: a filter that matches nothing exits 0 having run nothing, and a
# run that added no tests at all is a failure, never a vacuous pass.
#
#   bash ci/scripts/run_permission_fault_tests.sh
#
# Runs as root (it creates the probe account); reads fixtures anywhere in the
# checkout and never writes to it.
set -euo pipefail

TARGETS=(
  "-p jammi-ai --lib"
  "-p jammi-ai --test it"
  "-p jammi-db --lib"
  "-p jammi-db --test it"
  "-p jammi-server --lib"
)

id jammi-probe >/dev/null 2>&1 || useradd --no-create-home --shell /usr/sbin/nologin jammi-probe
probe_home="$(mktemp -d)"
chown jammi-probe "$probe_home"
chmod -R o+rX "$(git rev-parse --show-toplevel)"

# The test executable `cargo test --no-run` built for the given target.
binary() {
  cargo test "$@" --no-run --message-format=json \
    | python3 -c 'import json, sys; [print(m["executable"]) for m in map(json.loads, sys.stdin) if m.get("reason") == "compiler-artifact" and m.get("executable")]' \
    | tail -n1
}
tests_of() { "$1" --list --format terse | sed -n 's/: test$//p' | sort; }

ran=0
for target in "${TARGETS[@]}"; do
  # shellcheck disable=SC2086 # each target is a word list by construction
  plain="$(binary $target)"
  # shellcheck disable=SC2086
  gated="$(binary $target --features unprivileged-tests)"
  while read -r name; do
    out="$(runuser -u jammi-probe -- env HOME="$probe_home" "$gated" "$name" --exact 2>&1)" \
      || { echo "$out"; echo "::error::$name failed as an unprivileged user"; exit 1; }
    grep -q 'test result: ok. 1 passed;' <<<"$out" \
      || { echo "$out"; echo "::error::$name did not run exactly one test"; exit 1; }
    echo "ok $name"
    ran=$((ran + 1))
  done < <(comm -13 <(tests_of "$plain") <(tests_of "$gated"))
done
test "$ran" -gt 0 || { echo "::error::unprivileged-tests added no tests"; exit 1; }
echo "$ran permission-fault tests ran unprivileged"
