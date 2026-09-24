#!/usr/bin/env bash
# No bash source tests a pipeline that ends in `grep -q`. `grep -q` exits at
# its first match; a writer with anything left to write then takes SIGPIPE,
# and under `pipefail` its 141 becomes the pipeline's status — a match read
# as no match. No output is small enough to be safe: bash's `printf` and
# `echo` write one line per write(2), so a match on any line but the last
# races the lines after it, and a stream past one pipe buffer loses every
# time.
#
# The form that cannot lose: `grep -q PATTERN <<<"$text"`, or
# `<<<"$(command)"` for a command's output — the shell reads the command to
# its end before grep starts, and grep has no writer to kill.
#
# Scope: bash sources — `*.sh` and workflow `run:` blocks. A Dockerfile
# `RUN` is POSIX sh without `pipefail`, where a pipeline's status is grep's
# own. Comment lines are prose, not pipelines.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

# A single `|` (never `||`), then grep, then its leading option words, one
# of which carries `q` or is `--quiet`.
PIPED_GREP_Q='(^|[^|])\|[[:space:]]*grep([[:space:]]+-[A-Za-z-]+)*[[:space:]]+(-[A-Za-z]*q[A-Za-z]*|--quiet)([[:space:]]|$)'
COMMENT_LINE='^[^:]+:[0-9]+:[[:space:]]*#'

rc=0
hits="$(git grep -n -E "$PIPED_GREP_Q" -- '*.sh' '.github/workflows/*.yml' \
  ':!ci/scripts/check_no_piped_grep_q.sh')" || rc=$?
case "$rc" in
  0) ;;
  1) exit 0 ;;
  *) echo "git grep failed (exit $rc): the tree was not scanned" >&2; exit 2 ;;
esac
rc=0
pipelines="$(grep -v -E "$COMMENT_LINE" <<<"$hits")" || rc=$?
case "$rc" in
  0)
    printf '%s\n' "$pipelines"
    echo "a pipeline into grep -q reads a match as no match when its writer takes SIGPIPE: use grep -q PATTERN <<<\"\$text\" (or <<<\"\$(command)\")" >&2
    exit 1
    ;;
  1) exit 0 ;;
  *) echo "grep failed (exit $rc) filtering comment lines" >&2; exit 2 ;;
esac
