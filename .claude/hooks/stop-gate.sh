#!/bin/sh
# stop-gate.sh — opt-in Stop hook. On a DIRTY tree, run the P0 static checks and
# block Stop (exit 2) only if a P0 gate fails. Clean tree or all-green -> exit 0.
#
# Family Q/T (LESSONS.md): don't let a session end on an unverified artifact. This
# is the local echo of the fail-closed CI P0s (bijection / constitution-anchors /
# doc-parity). It is opt-in (not armed by default — see README) and advisory in the
# sense that it only fires the checks the repo already trusts as fail-closed.
#
# Loop-safe: honors stop_hook_active so a blocked-then-resumed Stop does not recurse.
# POSIX sh. Only P0 gate FAILURES block; a missing gate script or a tooling error
# is treated as "not a P0 signal" and does not block (fail-open on infrastructure,
# fail-closed only on a real gate verdict).

set -u

PAYLOAD="$(cat 2>/dev/null || true)"

# --- loop guard: if we are already inside a stop-hook continuation, do nothing ---
is_true_stop_active() {
  if command -v jq >/dev/null 2>&1; then
    [ "$(printf '%s' "$PAYLOAD" | jq -r '.stop_hook_active // false' 2>/dev/null)" = "true" ] && return 0
  elif command -v python3 >/dev/null 2>&1; then
    [ "$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json
try: print(str(json.load(sys.stdin).get("stop_hook_active", False)).lower())
except Exception: print("false")' 2>/dev/null)" = "true" ] && return 0
  else
    # no parser: best-effort raw scan
    printf '%s' "$PAYLOAD" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true' && return 0
  fi
  return 1
}
is_true_stop_active && exit 0

# --- locate repo root (prefer git; fall back to CLAUDE_PROJECT_DIR / cwd) ---
ROOT="${CLAUDE_PROJECT_DIR:-}"
if command -v git >/dev/null 2>&1; then
  GITROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  [ -n "$GITROOT" ] && ROOT="$GITROOT"
fi
[ -n "$ROOT" ] || ROOT="$(pwd 2>/dev/null || printf '%s' "$PWD")"
cd "$ROOT" 2>/dev/null || exit 0

# --- dirty-tree check: nothing staged/unstaged/untracked -> nothing to gate ---
if command -v git >/dev/null 2>&1; then
  if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
    exit 0   # clean tree — nothing to verify
  fi
else
  exit 0     # no git — cannot tell dirty; do not block
fi

# --- run the P0 static gates that exist; collect only real gate FAILURES ---
FAILED=""
run_gate() {
  # $1 = script path (relative to ROOT); run only if present; capture verdict.
  script="$1"
  [ -f "$script" ] || return 0            # not yet wired in this tree -> skip
  case "$script" in
    *.py)  runner="python3" ;;
    *.sh)  runner="sh" ;;
    *)     runner="" ;;
  esac
  if [ -n "$runner" ]; then
    command -v "$runner" >/dev/null 2>&1 || return 0   # no interpreter -> can't judge
    out="$("$runner" "$script" 2>&1)"; rc=$?
  else
    out="$("$script" 2>&1)"; rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    FAILED="${FAILED}
  P0 FAIL: $script (exit $rc)
$(printf '%s' "$out" | sed 's/^/      /')"
  fi
}

run_gate "ci/scripts/check_swarm_bijection.py"
run_gate "ci/scripts/check_constitution_anchors.py"
run_gate "ci/scripts/check_doc_parity.py"

if [ -n "$FAILED" ]; then
  # exit 2 -> Stop is blocked; stderr is surfaced back to the model as the reason.
  printf '%s\n' "stop-gate: a P0 static check failed on the dirty tree — resolve before ending the session:${FAILED}" >&2
  exit 2
fi

exit 0
