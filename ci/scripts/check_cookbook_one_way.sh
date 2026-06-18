#!/usr/bin/env bash
# One-way guard for the in-repo book (cookbook/book/).
#
# The book consumes the engine; the engine never consumes the book. The engine
# crates legitimately depend on the engine-owned cookbook/fixtures/ (golden
# fixtures, audio preprocessors, e2e inputs) — that edge stays. But no crate may
# reference the book subtree at cookbook/book/: that would invert the dependency
# direction (engine depending on its own proof). This fails, naming the offender,
# if any crates/** file mentions cookbook/book.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if hits="$(grep -rn "cookbook/book" crates/ 2>/dev/null)"; then
  echo "::error::a crates/** file references the book subtree (cookbook/book/) — the engine must not depend on the book:"
  echo "$hits" | sed 's/^/    /'
  exit 1
fi

echo "one-way guard: clean — no crates/** reference to cookbook/book/."
exit 0
