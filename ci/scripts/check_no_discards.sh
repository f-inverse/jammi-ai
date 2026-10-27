#!/usr/bin/env bash
# No Rust code discards a result with `let _ = expr`. A result is handled as
# what it means: propagated, matched with each outcome named, or not produced.
# A value nothing reads is not bound, and a guard is held under a name for as
# long as it guards. The pattern inside a string literal (a scanner's fixture)
# is not code, and does not start its line.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if hits="$(git grep -n -E '^[[:space:]]*let _ =' -- '*.rs')"; then
  printf '%s\n' "$hits"
  echo "a \`let _ =\` discards what the code did not handle: propagate it, match each outcome, or do not produce it" >&2
  exit 1
fi
