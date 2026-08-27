#!/bin/sh
# lead-gate-stop.sh — SubagentStop (matcher: the verifier agent types, D1).
# Parses `last_assistant_message` for the `<verdict>...</verdict>` JSON block
# and appends the unit's verdict row (CONTRACT-v1.md §C3). A pure WRITER:
# never denies, always exits 0 — blocking a verifier's own Stop event would
# prevent the verdict this gate depends on from ever being recorded. Logic
# lives in lead-gate-lib.py.
#
# POSIX sh. Reads the SubagentStop payload on stdin, hands it to python3.

set -u

DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

if ! command -v python3 >/dev/null 2>&1; then
  # No interpreter: cannot write the verdict row. An unwritten verdict row
  # means the unit has no rows -> reads as clean, not as an open BLOCK. This
  # is the documented "hooks were off" tell (D9/§C5): a verifier verdict in
  # the transcript with no hook-written state row is visible, not silent.
  printf '%s\n' "lead-gate-stop: python3 not available — verdict row not written" >&2
  exit 0
fi

printf '%s' "$PAYLOAD" | python3 "$DIR/lead-gate-lib.py" stop
exit 0
