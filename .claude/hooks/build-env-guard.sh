#!/bin/sh
# build-env-guard.sh — PreToolUse(Bash) advisory guard for the load-bearing build env.
#
# Family S (LESSONS.md): the build/host environment can silently invalidate
# correctness. This hook warns — it NEVER blocks. It always exits 0 and prints
# to stderr only. It is advisory discipline, not enforcement; the real teeth are
# per-agent worktree isolation and fail-closed CI. Wiring is opt-in (see README).
#
# Detection is by generic PATTERN, not by any hardcoded incident. It warns on:
#   1. RUSTFLAGS / RUSTC_WRAPPER override in the command   (sccache cache-key miss)
#   2. a cargo invocation with no unique CARGO_TARGET_DIR   (build-lock contention /
#      an auditor testing stale artifacts from a shared dir)
#   3. (best-effort) NVMe / working-disk pressure           (a full target volume)
#   4. (best-effort) maturin run with a PYTHONPATH that shadows across worktrees
#
# POSIX sh. Fail-open: any internal error still exits 0.

set -u

# ---- read the hook payload (fail-open on anything) -------------------------
PAYLOAD="$(cat 2>/dev/null || true)"

# Extract a top-level-ish JSON string field. Prefer jq, then python3, else empty.
# On any failure we return empty and simply emit fewer warnings — never block.
json_field() {
  # $1 = dotted path e.g. tool_input.command ; reads $PAYLOAD
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$PAYLOAD" | jq -r ".${1} // empty" 2>/dev/null && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$PAYLOAD" | python3 -c '
import sys, json
path = sys.argv[1].split(".")
try:
    obj = json.load(sys.stdin)
    for k in path:
        obj = obj.get(k) if isinstance(obj, dict) else None
    if isinstance(obj, str):
        sys.stdout.write(obj)
except Exception:
    pass
' "$1" 2>/dev/null && return 0
  fi
  return 0
}

TOOL="$(json_field tool_name)"
# Only Bash commands are in scope; anything else is a no-op pass.
case "$TOOL" in
  Bash|bash|"") : ;;   # "" tolerated: if we can't tell, fall through and best-effort scan
  *) exit 0 ;;
esac

CMD="$(json_field tool_input.command)"
# If we could not extract a command, fall back to scanning the raw payload so an
# advisory warning still has a chance to fire. (Warn-only: false positives are cheap.)
[ -n "$CMD" ] || CMD="$PAYLOAD"
[ -n "$CMD" ] || exit 0

warn() { printf '%s\n' "build-env-guard [advisory]: $1" >&2; }

# ---- 1. RUSTFLAGS / RUSTC_WRAPPER override ---------------------------------
# An assignment of either var in the command changes the sccache key -> full
# cache miss (~100 min of redundant compiles). The project sets these globally
# via .cargo/config.toml; a per-command override is the hazard.
if printf '%s' "$CMD" | grep -Eq '(^|[^A-Z_])(RUSTFLAGS|RUSTC_WRAPPER)[[:space:]]*=' ; then
  warn "command overrides RUSTFLAGS/RUSTC_WRAPPER — this changes the sccache cache key and forces a full cache-miss rebuild. Use the project build env (.cargo/config.toml); do not override the wrapper/flags."
fi

# ---- 2. cargo without a unique CARGO_TARGET_DIR ----------------------------
# A shared target dir serves stale artifacts across worktrees (an auditor then
# tests the wrong code). Each concurrent worktree needs its OWN target dir.
if printf '%s' "$CMD" | grep -Eq '(^|[^[:alnum:]_])cargo[[:space:]]+(build|test|check|clippy|nextest|run|bench|doc)' ; then
  # Satisfied if the command sets CARGO_TARGET_DIR inline, or it is already
  # exported in this env, or --target-dir is passed explicitly.
  if printf '%s' "$CMD" | grep -Eq 'CARGO_TARGET_DIR[[:space:]]*=|--target-dir' ; then
    : # explicit — good
  elif [ -n "${CARGO_TARGET_DIR:-}" ]; then
    : # inherited from env — assume the caller set a unique one
  else
    warn "cargo invoked with no unique CARGO_TARGET_DIR (neither inline, --target-dir, nor exported). Concurrent worktrees sharing one target dir cause build-lock contention and stale-artifact test runs. Set a per-worktree CARGO_TARGET_DIR."
  fi
fi

# ---- 3. NVMe / working-disk pressure (best-effort) -------------------------
# Only checks when the command actually builds; guarded so a missing df/awk is silent.
if printf '%s' "$CMD" | grep -Eq '(^|[^[:alnum:]_])(cargo|maturin)([[:space:]]|$)' ; then
  if command -v df >/dev/null 2>&1; then
    USE="$(df -P . 2>/dev/null | awk 'NR==2 {gsub(/%/,"",$5); print $5}' 2>/dev/null || true)"
    case "$USE" in
      ''|*[!0-9]*) : ;;                      # non-numeric / unavailable: skip
      *) [ "$USE" -ge 90 ] && warn "working-disk usage is ${USE}% — an NVMe/target volume near full can fail or corrupt a build. Free space or run cargo clean before building." ;;
    esac
  fi
fi

# ---- 4. maturin cross-worktree PYTHONPATH shadowing (best-effort) ----------
# A maturin build with a PYTHONPATH pointing outside the current tree can import
# another worktree's built extension, so tests exercise the wrong artifact.
if printf '%s' "$CMD" | grep -Eq '(^|[^[:alnum:]_])maturin([[:space:]]|$)' ; then
  PP=""
  # Prefer an inline PYTHONPATH= in the command, else the inherited env.
  INLINE_PP="$(printf '%s' "$CMD" | sed -n 's/.*PYTHONPATH[[:space:]]*=\([^[:space:];]*\).*/\1/p' 2>/dev/null | head -n1)"
  [ -n "$INLINE_PP" ] && PP="$INLINE_PP"
  [ -z "$PP" ] && PP="${PYTHONPATH:-}"
  if [ -n "$PP" ]; then
    CWD="$(pwd 2>/dev/null || printf '%s' "$PWD")"
    case ":$PP:" in
      *":$CWD"*|*":$CWD/"*|*":.:"*|*":.:"*) : ;;   # includes cwd — fine
      *) warn "maturin invoked with PYTHONPATH='$PP' that does not include the current tree — a cross-worktree PYTHONPATH can shadow this worktree's built extension so tests import the wrong artifact. Pin PYTHONPATH to this worktree." ;;
    esac
  fi
fi

exit 0
