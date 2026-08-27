#!/bin/sh
# lead-gate-pre.sh — PreToolUse (matcher: `Agent|Task`). THE DECIDER.
# Exit 0 = allow. Exit 2 + a stderr reason = deny (the model sees the
# reason). This is the swarm's first FAIL-CLOSED hook by design — unlike
# the three advisory hooks in this directory (`agent-routing-gate.sh`,
# `build-env-guard.sh`, `stop-gate.sh`), an internal error here DENIES, it
# does not silently allow (see `.claude/hooks/README.md`,
# `docs/plans/53-agentic-swarm/ARCHITECTURE.md` §7).
#
# `SendMessage` and `Bash` are OUT OF SCOPE by the v3 core cut — this hook
# does not gate them (the mechanical control for hook-file self-protection
# is `permissions.deny` in `.claude/settings.json`, plus the human-reviewed
# PR path).
#
# Logic lives in lead-gate-lib.py; this wrapper's ONLY job is: read stdin,
# hand off to python3, and map its exit code onto the two-value lattice
# {0, 2} — NEVER propagate python's raw exit code verbatim (v2 finding 3:
# a non-UTF-8 payload, a broken/old python3, or any interpreter-level
# failure that exits non-zero-non-two must still deny, not leak exit 1 —
# which Claude Code treats as non-blocking and would silently make this a
# norm again, not a gate).
#
# POSIX sh. No git subprocess (hot-path speed, §C5).

set -u

DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' "lead-gate-pre: python3 not available — failing closed (deny)" >&2
  exit 2
fi

printf '%s' "$PAYLOAD" | python3 "$DIR/lead-gate-lib.py" pre
rc=$?
case "$rc" in
  0) exit 0 ;;
  *) exit 2 ;;
esac
