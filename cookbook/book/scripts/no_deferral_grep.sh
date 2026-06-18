#!/usr/bin/env bash
# The no-deferral policy (README §2), enforced as a CI gate.
#
# Every decision is build or cut — nothing is left in limbo. This scans committed
# work (the shared lib and the executable chapters) for band-aid tell-signs and
# fails on any hit. A thing that cannot be built is CUT with a written rationale
# in EXECUTION-STATUS.md, never half-shipped behind one of these markers.
#
# Scope is deliberately the lib + chapters only: this script, CLAUDE.md, the
# README, and EXECUTION-STATUS.md legitimately *name* the forbidden tokens to
# define the policy, so they are not scanned.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Files in scope: the shared lib (.py) and every executable chapter source (.qmd
# / .ipynb) anywhere in the repo, including the root home page — but not build
# output, caches, or the venv.
mapfile -t FILES < <(
  {
    find jammi_cookbook -type f -name '*.py'
    find . -type f \( -name '*.qmd' -o -name '*.ipynb' \) \
      -not -path './_book/*' -not -path './.quarto/*' -not -path './.venv/*' \
      -not -path './.git/*'
  } 2>/dev/null
)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no-deferral: no lib/chapter files to scan yet — pass."
  exit 0
fi

# Forbidden tell-signs. Each is an extended-regex; word boundaries where it
# matters so prose like "fixmext" or "todofu" does not false-positive.
PATTERNS=(
  '\bTODO\b'
  '\bFIXME\b'
  'unimplemented!'
  'todo!\('
  'placeholder'
  '<TBD>'
  '\bTBD\b'
  'deferred'
  'v1 later'
  'for now'
)

status=0
for pat in "${PATTERNS[@]}"; do
  if matches="$(grep -rnE -- "$pat" "${FILES[@]}" 2>/dev/null)"; then
    echo "no-deferral: forbidden marker /$pat/ found:"
    echo "$matches" | sed 's/^/    /'
    status=1
  fi
done

# `# type: ignore` is only allowed when justified with an inline reason after it.
if matches="$(grep -rnE -- '# type: ignore([^[]|$)' "${FILES[@]}" 2>/dev/null | grep -vE '# type: ignore\[.+\]  # ')"; then
  if [ -n "$matches" ]; then
    echo "no-deferral: unjustified '# type: ignore' (use '# type: ignore[code]  # <reason>'):"
    echo "$matches" | sed 's/^/    /'
    status=1
  fi
fi

if [ "$status" -eq 0 ]; then
  echo "no-deferral: clean (${#FILES[@]} file(s) scanned)."
fi
exit "$status"
