#!/bin/sh
# lead-gate-start.sh — SubagentStart. Writes the agent_id -> unit_branch
# binding (CONTRACT-v1.md §C2). A pure WRITER: it never denies a subagent
# from starting, even on internal error (blocking this event could prevent
# the very agent whose eventual verdict the gate depends on from ever
# running) — always exits 0. Logic lives in lead-gate-lib.py; see its module
# doc for the fail-closed doctrine this hook deliberately does NOT apply
# (that doctrine is `lead-gate-pre.sh`'s alone).
#
# POSIX sh. Reads the SubagentStart payload on stdin, hands it to python3.

set -u

DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

if ! command -v python3 >/dev/null 2>&1; then
  # No interpreter: cannot write the binding. This is a writer, not the
  # decider — do not block the subagent; the absence of a binding row is
  # itself visible (an UNBOUND agent gates every relay, §C2).
  printf '%s\n' "lead-gate-start: python3 not available — binding not written" >&2
  exit 0
fi

printf '%s' "$PAYLOAD" | python3 "$DIR/lead-gate-lib.py" start
exit 0
