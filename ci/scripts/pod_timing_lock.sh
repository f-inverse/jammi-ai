#!/usr/bin/env bash
# One flock seam for every pod-side thing that must not race a `run --timing`
# job, a seed build, or another producer over shared pod resources
# (CARGO_TARGET_DIR, the seed, nvcc). Replaces an earlier rename-based
# "steal" scheme that reproduced a double-acquire under a scheduling gap
# between "observe the incumbent is stale" and "rename it away" — see the
# module doc in gpu-dev.sh (M6) and ledger row 21. `flock` makes the KERNEL
# own liveness instead: the lock is a held file descriptor, so it dies the
# INSTANT the holding process exits or is killed — crash-safe, reboot-safe,
# no stale marker to observe, and nothing to steal because there is nothing
# left once the holder is gone.
#
# Usage:
#   pod_timing_lock.sh acquire -n         -- <cmd...>   # non-blocking (refuse now)
#   pod_timing_lock.sh acquire -w <SECS>  -- <cmd...>   # blocking up to SECS
#
# `-n` is what the tmux pane launcher builds directly into its own command
# line for `run --timing` and for the seed (`tmux new-session -d "flock -n -E
# 75 $LOCK bash <tree>/.jammi-job.sh"` — see gpu-dev.sh) rather than going
# through this script: acquiring the lock is then the FIRST thing that
# happens INSIDE the detached tmux pane's own process, so the lock's
# lifetime is tied to the pane's job, not to the (short-lived) launcher
# process that started tmux and returned immediately — a launcher-side flock
# would release the instant the launcher exits, seconds after the real job
# keeps running, and prove nothing.
#
# This script is for PRODUCERS that are NOT already running inside such a
# pane — pod_seed_target.sh invoked directly over SSH, ci/scripts/perf/
# pod_build_timings.sh's own driver — and therefore need to wait their turn
# rather than assume the lock is already held on their behalf. They pass
# `-w <SECS>`.
#
# Exit 75 = the lock was refused (another holder has it, or the wait timed
# out) — distinguishable from the wrapped command's own exit code, and equal
# to rp_deploy_live's "neutral capacity skip" code elsewhere in this tooling
# (not a coincidence: both mean "the resource was not available, try again",
# never "this is broken").
#
# The holder file (${LOCK}.holder) is written by tmp-then-rename UNDER the
# lock (never truncate-then-write in place): a reader racing the write can
# only ever see the OLD complete file or the NEW complete file, never a
# truncated/partial one — verified empirically (pt8/holder_race.py-style
# probe: truncate+write showed torn reads under a scheduling gap between
# the truncate and the write; tmp+rename showed zero).
#
# The holder file is REMOVED on release (round-4 audit A3): a witness file
# that is written once and never cleaned up answers "is this held, right
# now" correctly only until the FIRST release, then reads as "held" forever
# after — a downstream reader (pod_build_timings.sh's own LOCK_HELD check)
# reproduced exactly that: the prior run exits, the witness still reads
# true, and an outsider acquires the lock immediately despite it. Removal
# happens via a trap on EXIT/INT/TERM set INSIDE the flock-held child, after
# the holder file is written and BEFORE the wrapped command runs (so a
# crash/kill of the wrapped command still triggers cleanup) — the wrapped
# command is run as a normal child (never `exec`'d away), because `exec`
# REPLACES the process image and would discard the trap before it could
# ever fire on the wrapped command's own later exit. Readers ALSO
# cross-check the recorded `pid=` against a live process (`kill -0`),
# belt-and-suspenders against a removal that itself raced or failed.
set -uo pipefail

LOCK="${JAMMI_TIMING_LOCK:-/root/.jammi-timing.lock}"
HOLDER="${LOCK}.holder"

usage() {
  echo "usage: $(basename "$0") acquire (-n | -w SECS) -- <cmd...>" >&2
  exit 2
}

[ $# -ge 1 ] || usage
mode="$1"; shift
[ "$mode" = "acquire" ] || usage

flock_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -n) flock_args=(-n); shift ;;
    -w)
      [ $# -ge 2 ] || { echo "::error::-w needs a value (seconds)" >&2; exit 2; }
      case "$2" in ''|*[!0-9]*) echo "::error::-w must be a positive integer (got '$2')" >&2; exit 2 ;; esac
      flock_args=(-w "$2"); shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ ${#flock_args[@]} -gt 0 ] || usage
[ $# -ge 1 ] || { echo "::error::no command given after --" >&2; exit 2; }

command -v flock >/dev/null 2>&1 || {  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
  echo "::error::flock (util-linux) not found on PATH — the pod image is expected to carry it" >&2
  exit 2
}

# JAMMI_TIMING_LABEL: a human-readable "who holds this" string (e.g.
# "seed@a100", "pod_build_timings@fa2-int"). Falls back to the wrapped
# command line itself so a refusal always names SOMETHING useful even when
# the caller did not set it.
label="${JAMMI_TIMING_LABEL:-$*}"

# The bash -c body runs ONLY once flock has the lock (it is flock's own
# child), so the holder-file write below is genuinely "under the lock".
# Positional args after `_` become "$@" inside the -c script; the label and
# lock/holder paths are passed the same way rather than interpolated into
# the script text, so a label or path containing a shell metacharacter
# cannot break out of the quoting.
exec flock "${flock_args[@]}" -E 75 "$LOCK" bash -c '
  lock="$1"; holder="$2"; label="$3"; shift 3
  tmp="${holder}.tmp.$$"
  { printf "holder=%s\n" "$label"
    printf "pid=%s\n" "$$"
    printf "job=%s\n" "${JAMMI_TIMING_JOB:-}"
    printf "started=%s\n" "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" # tripwire-ok: unknown is a visible non-empty sentinel for the holder-file started= field; never a silent empty timestamp
  } > "$tmp" && mv -f "$tmp" "$holder"
  trap "rm -f \"$holder\"" EXIT INT TERM
  "$@"
  exit "$?"
' _ "$LOCK" "$HOLDER" "$label" "$@"
