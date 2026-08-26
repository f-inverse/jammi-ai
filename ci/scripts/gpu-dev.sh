#!/usr/bin/env bash
# GPU development on RunPod — one CLI over the shared pod primitive.
#
# Two lifetimes, one substrate:
#   * `shell` is a throwaway debug pod — it dies when you exit (the CI default).
#   * `up` starts a named session whose pod SURVIVES disconnect, so a fine-tune,
#     eval or bench keeps running after you close the terminal.
#
# Every pod carries an RP_TTL_HOURS deadline in its own entrypoint regardless of
# lifetime, and `reap` sweeps anything that outlives it.
#
# The pod itself is disposable (volumeInGb 0). Compilation state lives in a
# per-pod seed/clone build substrate (`target`, below) rather than an
# S3-backed compile cache — the cargo registry is deliberately NOT cached;
# fetching it measures 9s on a RunPod host. See `docs/maintainer/dev-gpu.md`.
#
# A pod can host more than one checkout — a "tree", a plain directory, never
# a git worktree (a worktree add fails on the checked-out ref, and a shared
# .git couples trees that must be able to diverge independently). `--tree
# <name>` selects one on run/push/pull/attach/logs/target; the default tree
# ("jammi-ai") is the historical single checkout at /root/jammi-ai that every
# session already has from bootstrap.
#
# Usage:
#   gpu-dev.sh shell   [arch] [--ref R] [--tree T]   throwaway shell; pod dies on exit
#   gpu-dev.sh up      [arch] [--ref R]              start a surviving session (name = arch)
#                      [--replace]                   replace an alias's existing record
#   gpu-dev.sh target  [session] <name>              clone the pod's build-substrate seed
#                      [--verify] [--with-cutlass]    into a new tree (`/root/trees/<name>`)
#   gpu-dev.sh attach  [session] [--tree T]           shell into a surviving session
#   gpu-dev.sh run     [session] [--tree T] <cmd...>  run <cmd> detached under tmux
#   gpu-dev.sh logs    [session] [--tree T]           tail the detached job's output
#   gpu-dev.sh push    [session] [--tree T]           rsync your working tree TO the pod
#   gpu-dev.sh pull    [session] [--tree T] <path>    rsync <path> back FROM the pod
#   gpu-dev.sh down    [session]                       terminate the pod, forget the session
#   gpu-dev.sh ls                                      list sessions
#
# arch: a100 (default) | l40s | h100 | a40 | l4
# --ref: branch, tag or commit the pod's checkout is placed on (default main).
#        Verified against the remote BEFORE a pod is rented.
# --tree: the checkout/tree a run/push/pull/attach/logs/target command acts
#        on (default "jammi-ai" — the bootstrap checkout at /root/jammi-ai).
#        Any other name is a plain directory at /root/trees/<name>, created
#        by `target` (see pod_target_clone.sh) — never git-cloned itself.
# --replace: `up` normally REFUSES to touch a session alias that already has a
#        recorded pod id — even one that failed to answer SSH — rather than
#        silently deploying a second pod under the same alias (that is how an
#        unrelated stale pod got terminated on 2026-08-25). --replace opts in
#        to overwriting the LOCAL record only; it does not terminate the old
#        pod (`down` it first if it should be).
# Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,
#      RP_TTL_HOURS (default 8 for `shell`; CI's own lanes use 3/8 — see
#      runpod_gpu_prove.sh), RP_DEV_TTL_HOURS (default 72 — the deadline `up`
#      alone uses when RP_TTL_HOURS is not set explicitly: a dev session is
#      meant to survive a workday, not die at the throwaway-pod default),
#      RP_DISK_GB (60), RP_VOLUME_GB (0). Disk sizing rule of thumb: roughly
#      25 GB base + 3 GB per concurrent agent target dir + 2 GB per
#      `cargo mutants` job — a mutation-testing session wants >= 120 GB.
#
# A running measurement is protected only by its own TTL — there is no way to
# pause the sweep for one pod without touching every other pod's deadline
# (RunPod's pod-edit mutation has no `name` field, so a marker-in-the-name
# scheme is not possible; the only pod-edit mutation that DOES exist would
# reconfigure and restart the container, which is worse than the problem it
# would solve). Rent with `RP_TTL_HOURS`/`RP_DEV_TTL_HOURS` set to at least
# the job's expected length instead of relying on `down` to arrive first.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "${HOME}/.config/runpod/key" 2>/dev/null || true)}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY or write it to ~/.config/runpod/key}"

usage() {
  cat <<'USAGE'
gpu-dev.sh — GPU development on RunPod

  shell   [arch] [--ref R] [--tree T]     throwaway shell; the pod dies when you exit
  up      [arch] [--ref R]                start a session whose pod SURVIVES disconnect
          [--replace]                     (default TTL 72h — see RP_DEV_TTL_HOURS below)
                                          refuses if the alias already has a recorded pod
                                          unless --replace is given (see --replace below)
  target  [session] <name>                clone the pod's build-substrate seed into a
          [--verify] [--with-cutlass]     new tree /root/trees/<name> (pod_target_clone.sh)
  attach  [session] [--tree T]            join a surviving session's running job
                                          (--shell for a plain prompt instead)
  run     [session] [--tree T] <cmd...>   run <cmd> detached under tmux, in <tree>
  logs    [session] [--tree T]            tail the detached job's output
  push    [session] [--tree T]            rsync your working tree TO the pod's <tree>
  pull    [session] [--tree T] <path>     rsync <path> back FROM the pod's <tree>
  down    [session]                       terminate the pod, forget the session
  ls                                      list sessions
  reap    [hours]             ACCOUNT-WIDE: terminate every orphaned jammi-gpu*
                              pod past its own deadline, not just one session's
                              ([hours] force-reaps EVERY such pod older than
                              that, regardless of session — not a per-pod verb)

A running measurement is protected only by its own TTL — there is no verb to
pause the sweep for a single pod (RunPod's pod-edit API has no rename/name
field; see the module header comment). Rent with RP_TTL_HOURS/RP_DEV_TTL_HOURS
set to at least the job's expected length.

arch: a100 (default) | l40s | h100 | a40 | l4
--ref R: branch, tag or commit for the pod's checkout (default main), checked
         against the remote before anything is rented. `up` does not move a live
         pod: `down` it first.
--tree T: which checkout on the pod a run/push/pull/attach/logs/target command
         acts on (default "jammi-ai", the bootstrap checkout at
         /root/jammi-ai). Any other name is a plain directory at
         /root/trees/<name> populated only by `target` — never a git worktree
         (a shared .git would couple trees that must diverge independently)
         and never git-cloned itself.
--replace: `up` normally REFUSES (exit 2) a session alias that already has a
         recorded pod id, even an unreachable one, rather than silently
         deploying a second pod under the same alias. --replace overwrites only
         the LOCAL record; it never terminates the old pod itself — `down` it
         first if it should be.
--verify (target only): after `target` clones and you have built once against
         the new tree, re-run with --verify and a `cargo build -v` log on
         stdin to assert no member unit reported Fresh (see pod_target_clone.sh).
--with-cutlass (target only): also provisions the CUTLASS submodule into the
         new tree (`git submodule update --init --depth 1`) — needed only for
         a tree that will build the `flash-attn` feature.
Sessions are named after the arch; RP_SESSION names one explicitly. On
down/attach/run/logs/push/pull/target, RP_SESSION and an explicit positional
session must AGREE — a differing pair REFUSES (exit 2, naming both) rather
than silently picking one. A session name may contain letters, digits, _, -,
and . anywhere but as the whole name (so bench.1 is fine; ., .., a name
containing /, or one starting with - are refused).
Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,
     RP_TTL_HOURS (default 8; explicit always wins), RP_DEV_TTL_HOURS (default
     72 — what `up` alone falls back to when RP_TTL_HOURS is not set),
     RP_DISK_GB (default 60), RP_VOLUME_GB (default 0).
     Disk sizing once the seed/clone substrate is in use: RP_DISK_GB >= 25
     (base) + S_src + S_seed + N*S_clone (one clone per tree this pod hosts);
     the S_src/S_seed/S_clone byte counts are MEASURED by
     ci/scripts/perf/pod_build_timings.sh (A2), not guessed — see
     docs/maintainer/dev-gpu.md, which marks them "pending" until that has
     run and its JSON is committed. Add 3 GB per OTHER concurrent agent target
     dir + 2 GB per `cargo mutants` job — a mutation-testing session wants
     >= 120 GB (RP_DISK_GB=150).
USAGE
  exit "${1:-2}"
}

CMD="${1:-}"; [ $# -gt 0 ] && shift
case "$CMD" in ""|-h|--help|help) usage 0 ;; esac

# `ls` and `reap` are account-level: no session, no arch, no pod. Resolve them
# before the session/arch argument parsing below, which does not apply to them.
case "$CMD" in
  ls)
    # shellcheck source=ci/scripts/runpod_lib.sh
    source "$DIR/runpod_lib.sh"
    rp_session_list
    exit 0
    ;;
  reap)
    # shellcheck source=ci/scripts/runpod_lib.sh
    source "$DIR/runpod_lib.sh"
    # No argument => judge each pod against the deadline in its own name.
    # Passing RP_TTL_HOURS here would impose THIS shell's limit on every pod.
    rp_sweep "${1:-}"
    exit $?
    ;;
esac

# Sessions are named after the arch, so the common case needs no bookkeeping:
# `up a100` … `attach a100` … `down a100`. RP_SESSION overrides for a second pod
# of the same arch.
#
# Only the two commands that BOOT a pod take options, and only they get a flag
# loop. `run` hands its whole tail to the pod as a command line, so parsing flags
# there would eat `cargo test -- --nocapture` alive.
REF=main
REF_EXPLICIT=0
REPLACE=0
RESEED=0
TREE=jammi-ai
TARGET_NAME=""
TARGET_VERIFY=0
TARGET_WITH_CUTLASS=0
case "$CMD" in
  shell|up)
    ARCH=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --ref)
          # A flag whose value is missing must not silently become a default:
          # `up --ref` would otherwise boot main and look like it honoured you.
          [ $# -ge 2 ] || { echo "::error::--ref needs a value (a branch, tag or commit)"; exit 2; }
          REF="$2"; REF_EXPLICIT=1; shift 2 ;;
        --ref=*)
          REF="${1#--ref=}"; REF_EXPLICIT=1; shift ;;
        # `--tree` only means something for `shell` (an interactive session
        # in a non-default checkout); `up` establishes the SESSION/pod itself
        # — trees are directories WITHIN an already-up pod, selected on the
        # verbs that act inside one (attach/run/logs/push/pull/target).
        --tree)
          [ "$CMD" = "shell" ] || { echo "::error::--tree applies only to 'shell'/attach/run/logs/push/pull/target, not '${CMD}'"; exit 2; }
          [ $# -ge 2 ] || { echo "::error::--tree needs a value"; exit 2; }
          TREE="$2"; shift 2 ;;
        --tree=*)
          [ "$CMD" = "shell" ] || { echo "::error::--tree applies only to 'shell'/attach/run/logs/push/pull/target, not '${CMD}'"; exit 2; }
          TREE="${1#--tree=}"; shift ;;
        # `--replace` only means something for `up` (it overwrites a session's
        # LOCAL record — see the `up` case below); `shell` has no session to
        # overwrite, so accepting it silently there would look like it did
        # something. Reject it explicitly instead of ignoring it.
        --replace)
          [ "$CMD" = "up" ] || { echo "::error::--replace applies only to 'up' ($CMD has no session to replace)"; exit 2; }
          REPLACE=1; shift ;;
        # Forces the seed build to rerun even if .jammi-seed-complete already
        # exists (pod_seed_target.sh's own default is otherwise a no-op —
        # see M1).
        --reseed) RESEED=1; shift ;;
        # Asking for help is not an error, and `up -h` is where someone reaches
        # when they have forgotten the flag they came to look up.
        -h|--help) usage 0 ;;
        # Any OTHER unrecognised option is a hard error, never a positional
        # argument. Absorbing one makes `up --ref x` die on "unknown arch
        # '--ref'" — a message naming a problem the user does not have.
        -*) echo "::error::unknown option '$1' for ${CMD}"; usage 2 ;;
        *)
          [ -z "$ARCH" ] || { echo "::error::${CMD}: unexpected argument '$1'"; usage 2; }
          ARCH="$1"; shift ;;
      esac
    done
    ARCH="${ARCH:-a100}"; SESSION="${RP_SESSION:-$ARCH}"
    ;;
  target)
    # target [session] <name> [--verify] [--with-cutlass]
    # `<name>` is the TREE being created/verified, never `--tree` — there is
    # nothing to select yet; this verb is what CREATES a tree.
    ARG=""
    case "${1:-}" in
      -*) : ;;
      *) [ $# -gt 0 ] && { ARG="$1"; shift; } ;;
    esac
    if [ -n "$ARG" ] && [ -n "${RP_SESSION:-}" ] && [ "$ARG" != "$RP_SESSION" ]; then
      echo "::error::conflicting session: positional argument '${ARG}' vs exported RP_SESSION='${RP_SESSION}' — they name different sessions"
      exit 2
    fi
    SESSION="${ARG:-${RP_SESSION:-a100}}"; ARCH=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --verify) TARGET_VERIFY=1; shift ;;
        --with-cutlass) TARGET_WITH_CUTLASS=1; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "::error::unknown option '$1' for target"; usage 2 ;;
        *)
          [ -z "$TARGET_NAME" ] || { echo "::error::target: unexpected argument '$1'"; usage 2; }
          TARGET_NAME="$1"; shift ;;
      esac
    done
    [ -n "$TARGET_NAME" ] || [ "$TARGET_VERIFY" = "1" ] || { echo "::error::target: need a tree name (or --verify against an existing one)"; usage 2; }
    ;;
  *)
    ARG=""
    case "${1:-}" in
      -*) : ;;
      *) [ $# -gt 0 ] && { ARG="$1"; shift; } ;;
    esac
    # Unlike shell|up above (where ARCH and RP_SESSION are genuinely
    # different axes — RP_SESSION deliberately overrides the session ALIAS
    # for a second pod of the same arch, per this file's own header doc),
    # here the positional IS the session name: `down`/`attach`/`run`/`logs`/
    # `push`/`pull` all take `[session]` directly. `${RP_SESSION:-${ARG:-a100}}`
    # let an exported RP_SESSION silently WIN over an explicit positional —
    # `RP_SESSION=a100 gpu-dev.sh down l40s` terminated pod-a100 and forgot
    # ITS record, never touching l40s at all, discarding the one argument
    # the caller typed on the command line. An explicit positional is never
    # discarded: when it conflicts with a differing exported RP_SESSION,
    # that is a real ambiguity (which one did the caller mean?) and this
    # refuses rather than silently picking either; when only one is set, it
    # wins; when neither is, the a100 default applies exactly as before.
    if [ -n "$ARG" ] && [ -n "${RP_SESSION:-}" ] && [ "$ARG" != "$RP_SESSION" ]; then
      echo "::error::conflicting session: positional argument '${ARG}' vs exported RP_SESSION='${RP_SESSION}' — they name different sessions"
      echo "::error::pick one: unset RP_SESSION to act on '${ARG}', or drop the positional to act on RP_SESSION='${RP_SESSION}'"
      exit 2
    fi
    SESSION="${ARG:-${RP_SESSION:-a100}}"; ARCH=""
    # `--tree`, when present, is a LEADING flag right after the (optional)
    # session positional — before `run`'s own command tail, which this loop
    # must never touch (`cargo test -- --tree-of-life` is someone's ACTUAL
    # command, not a flag for this script). Only attach/run/logs/push/pull
    # take it; `down` acts on the pod, not a tree within it.
    case "$CMD" in
      attach|run|logs|push|pull)
        case "${1:-}" in
          --tree)
            [ $# -ge 2 ] || { echo "::error::--tree needs a value"; exit 2; }
            TREE="$2"; shift 2 ;;
          --tree=*)
            TREE="${1#--tree=}"; shift ;;
        esac
        ;;
    esac
    ;;
esac

# `shell` is throwaway: no named session, so the EXIT trap wipes
# the temp dir and terminates the pod. RP_SESSION is force-cleared here, not
# merely left unset, because an EXPORTED RP_SESSION set earlier in the SAME
# shell (`export RP_SESSION=a100` — a one-off command-prefix assignment like
# `RP_SESSION=a100 gpu-dev.sh attach a100` does NOT persist past that single
# invocation, so it is not the precondition here) survives into this
# invocation's environment untouched otherwise — runpod_lib.sh's own default
# is only "${RP_SESSION:-}", which keeps whatever is already set. A `shell`
# that silently inherited a live session's name would then write the
# throwaway pod's coordinates over that session's own meta file
# (rp_session_save keys off RP_SESSION alone) and, on exit, terminate the
# throwaway pod under RP_POD_CREATED — leaving the REAL, still-running pod
# behind with no record pointing at it at all.
# RP_SESSION_VALIDATE_SESSION tells runpod_lib.sh's own rp_session_name_check
# gate (see its doc) that THIS invocation resolved a real, named session and
# must have it validated before RP_WORK is derived from it. Set only here —
# never for `shell` (deliberately anonymous) and never reached at all by
# `ls`/`reap` (account-level, they source runpod_lib.sh from their own early
# dispatch branch above, before this line ever runs) — so an unrelated
# exported RP_SESSION sitting in the caller's shell for some other purpose
# can never block a verb that was never going to consume it.
case "$CMD" in
  shell) RP_SESSION="" ;;
  *) export RP_SESSION="$SESSION"; RP_SESSION_VALIDATE_SESSION=1 ;;
esac

# `up` sessions persist past the terminal and are the ones the 2026-08-25
# incident hit: runpod_lib.sh's own RP_TTL_HOURS default (8h) is sized for a
# throwaway `shell`/CI pod, not a dev session someone is actively using —
# so a session rented at the default died at 8h no matter what. An explicit
# RP_TTL_HOURS from the caller always wins (checked BEFORE runpod_lib.sh
# applies its OWN "${RP_TTL_HOURS:-8}" default); only in its absence does `up`
# alone raise the floor to RP_DEV_TTL_HOURS. `shell` and every CI lane
# (runpod_gpu_prove.sh sets its own default before sourcing this file) are
# untouched.
if [ "$CMD" = "up" ] && [ -z "${RP_TTL_HOURS:-}" ]; then
  RP_DEV_TTL_HOURS="${RP_DEV_TTL_HOURS:-72}"
  # Validated HERE, under its OWN name, not left to runpod_lib.sh's shared
  # RP_TTL_HOURS check below: a bad RP_DEV_TTL_HOURS would otherwise be
  # assigned into RP_TTL_HOURS first and reported as "RP_TTL_HOURS must be
  # a positive integer" (or "> 0") — naming a variable the caller never set
  # as the thing that is wrong. Both the digit-shape check AND the >0 check
  # are mirrored here: "0" passes a digit-only pattern (it has no non-digit
  # character), so without the second check RP_DEV_TTL_HOURS=0 slips past
  # THIS validation and still gets misattributed by runpod_lib.sh's own
  # "RP_TTL_HOURS must be > 0" once it has already been assigned in.
  case "$RP_DEV_TTL_HOURS" in
    ''|*[!0-9]*) echo "::error::RP_DEV_TTL_HOURS must be a positive integer (got '${RP_DEV_TTL_HOURS}')"; exit 2 ;;
  esac
  [ "$RP_DEV_TTL_HOURS" -gt 0 ] || { echo "::error::RP_DEV_TTL_HOURS must be > 0"; exit 2; }
  RP_TTL_HOURS="$RP_DEV_TTL_HOURS"
  export RP_TTL_HOURS
fi

# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# Resolved once, here, so every verb below (attach/run/logs/push/pull/target)
# reads the SAME directories for the SAME --tree value — the one place
# TREE -> path resolution happens on the laptop side (rp_tree_dir/
# rp_target_dir also run on the pod, inside the remote heredocs below,
# computed identically since both sides apply the exact same function).
# TREE_DIR is the SOURCE checkout (rsync'd by push); TARGET_DIR is the
# build-substrate CLONE's own CARGO_TARGET_DIR — a deliberately DISJOINT
# directory (round-2 audit finding 1: conflating the two made `target`'s
# clone destination collide with `push --tree`'s rsync destination, so the
# first push after a `target` deleted the clone it had just made).
TREE_DIR="$(rp_tree_dir "$TREE")"
TARGET_DIR="$(rp_target_dir "$TREE")"
# `=`-anchored so a tmux SESSION lookup never prefix-matches another tree's
# session (`jammi-ai` would otherwise match a session literally named
# `jammi-ai-2`) — the fix for the shipped unanchored `-t jammi` bug (M6).
TMUX_SESSION="jammi-${TREE}"

# The ref rules live with the code that sends the ref to the pod, so they are
# applied here as soon as that code is available — before anything is rented.
case "$CMD" in
  shell|up) rp_ref_check "$REF" || exit 2 ;;
esac

# Interactive remote command: correct env, then either the running job's terminal
# or a plain shell in the checkout. Pass "job" to prefer the job when one exists.
# $2 = tree dir, $3 = tmux session name (both required; no bare defaults here
# so a caller can never silently land on the WRONG tree's job). Exports
# CARGO_TARGET_DIR at THIS tree's own build-substrate clone too (TARGET_DIR,
# resolved above), the same wiring `run`'s job wrapper uses (rp_job_wrapper_lines)
# — an interactive `cargo build` in a tree gets the seed's benefit exactly
# like a `run` job does.
rp_login_cmd() { # $1 = "job" to join a live tmux job, $2 = tree dir, $3 = tmux session
  local tree_dir="${2:?rp_login_cmd needs a tree dir}" tmux_sess="${3:?rp_login_cmd needs a tmux session name}"
  # Built from the SAME rp_job_wrapper_lines `run`'s own job wrapper uses
  # (round-3 audit Class B) — never a second hand-rolled copy of the
  # source-env/CARGO_TARGET_DIR/cd sequence that could silently drift from
  # it. The no-op job line (":") is dropped (an interactive shell has no
  # job to run); `cd` gets its own `2>/dev/null` so a not-yet-provisioned
  # tree lands an interactive login shell in cwd rather than aborting the
  # whole SSH session outright — a `run` job, by contrast, should hard-fail
  # on a missing tree, which is why the shared function itself has no such
  # suppression.
  local -a wlines=()
  while IFS= read -r wline; do wlines+=("$wline"); done < <(rp_job_wrapper_lines "$tree_dir" "$TARGET_DIR" ":")
  local pre="${wlines[0]}; ${wlines[1]}; ${wlines[2]} 2>/dev/null;"
  if [ "${1:-}" = "job" ]; then
    # Ctrl-C inside the job's pane signals the job itself, so say so before
    # handing over the keyboard — this is a terminal that can destroy work.
    printf '%s %s' "$pre" "if tmux has-session -t \"=${tmux_sess}\" 2>/dev/null; then
      echo \"=== joining the running job. Ctrl-B then D detaches. Ctrl-C KILLS the job. ===\";
      exec tmux attach -t \"=${tmux_sess}\"; fi; exec bash -i"
  else
    printf '%s %s' "$pre" 'exec bash -i'
  fi
}

require_pod() {
  rp_session_load && rp_session_alive && return 0
  echo "no live session '${SESSION}' — the pod may have been reaped."
  echo "start one with: $(basename "$0") up ${SESSION}"
  exit 1
}

# Bootstrap the pod, and decide what each outcome means. A pod on the wrong code
# answers the wrong question convincingly, so a failed bootstrap ends the
# session rather than handing you a shell on it.
#
# The one exception is an image that ships no git (rp_bootstrap's exit 3):
# reproducing the shipped RUNTIME image is a real use of a pod and that image has
# no toolchain, so there is no checkout to be had and nothing is being hidden —
# RP_REF stays empty and every report says `<none>`. It is still fatal when the
# caller NAMED a ref: that request cannot be honoured here, and a pod that
# quietly ignores it is the exact failure `--ref` exists to remove.
bootstrap_or_die() {
  rp_bootstrap "$REF"
  case "$?" in
    0) return 0 ;;
    3) [ "$REF_EXPLICIT" = 0 ] || {
         echo "::error::--ref ${REF} cannot be honoured: ${RP_IMAGE} ships no git"
         echo "::error::terminating pod (trap)"
         exit 1
       }
       echo "=== no checkout: this image ships no git, so the pod is on no ref ==="
       return 0 ;;
    *) echo "::error::bootstrap failed — terminating pod (trap)"; exit 1 ;;
  esac
}

# Kicks off the pod's build-substrate seed, DETACHED — never blocks
# `shell`/`up` (a cold seed is real compile minutes). Idempotent:
# pod_seed_target.sh itself no-ops when .jammi-seed-complete already exists
# (unless --reseed). Runs under the SAME lock-in-pane pattern as `run
# --timing` (M6): the flock acquisition is the FIRST thing the detached
# pane's own command does, so the lock's lifetime is the seed job's
# lifetime, not this short-lived launcher's.
start_seed_build() {
  local reseed_flag=""
  [ "$RESEED" = "1" ] && reseed_flag="--reseed"
  rp_run_remote <<EOF
set -uo pipefail
tmux kill-session -t "=jammi-seed" 2>/dev/null
tmux new-session -d -s "jammi-seed" "flock -n -E 75 /root/.jammi-timing.lock bash /root/jammi-ai/ci/scripts/pod_seed_target.sh --no-lock ${reseed_flag} > /root/.jammi-seed.log 2>&1"
tmux set-option -w -t "=jammi-seed:" remain-on-exit off
echo "seed build started detached (tmux session jammi-seed; log: /root/.jammi-seed.log)"
EOF
}

case "$CMD" in

  shell)
    # Before rp_init, before anything is rented: a bad ref costs a second here
    # and a GPU-hour plus a four-minute SSH wait on the far side of the deploy.
    rp_ref_precheck "$REF" || exit 2
    rp_init
    echo "=== provisioning ${ARCH} on ${REF} (image: ${RP_IMAGE}) ==="
    rp_deploy_arch "$ARCH" || exit $?
    echo "=== bootstrapping ==="
    bootstrap_or_die
    [ -n "$RP_REF" ] && start_seed_build
    echo "=== pod ${RP_POD_ID} on ${RP_HOST}:${RP_PORT} @ ${RP_REF:-<none>} (tree: ${TREE}) — it TERMINATES when you exit ==="
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd "" "$TREE_DIR" "$TMUX_SESSION")" || true
    echo "=== shell closed — terminating pod (trap) ==="
    ;;

  up)
    # rp_session_load below dot-sources ANY existing session's meta file,
    # which includes RP_TTL_HOURS/RP_IMAGE from whatever pod that alias last
    # recorded — a dead 8h session's meta would otherwise silently overwrite
    # the 72h dev default (or an explicit RP_TTL_HOURS/RP_IMAGE) THIS
    # invocation wants for its OWN new pod, and rp_session_save at the end
    # would then persist that stale value into the new pod's own record too
    # (demonstrated: --replace over an 8h dead session's meta deployed a
    # 'jammi-gpu-ttl8' pod even with the 72h dev default in effect).
    # Captured before rp_session_load can touch either variable, and restored
    # unconditionally below — covering both the fresh-session path (nothing
    # to restore FROM, a no-op) and --replace (the actual bug).
    UP_TTL_WANTED="$RP_TTL_HOURS"
    UP_IMAGE_WANTED="$RP_IMAGE"
    if rp_session_load; then
      if rp_session_alive; then
        # rp_keep FIRST: this pod belongs to a session that is already running,
        # and every exit below must leave it alone.
        rp_keep
        echo "session '${SESSION}' already live on ${RP_HOST}:${RP_PORT} (pod ${RP_POD_ID}, ref ${RP_REF:-<none>})"
        # `up` never moves a live pod. Reporting success on an ignored --ref
        # would leave you reading results from one ref while believing they
        # came from another — the exact failure this flag exists to remove.
        if [ "$REF_EXPLICIT" = 1 ] && [ "$REF" != "${RP_REF:-}" ]; then
          # `up` takes an ARCH, never a session name — they are only the same
          # string until RP_SESSION overrides it. Suggesting a bare `up ${ARCH}`
          # would then boot a DIFFERENT session than the `down` beside it, so the
          # override is carried explicitly whenever there is one.
          SESSION_ENV=""
          [ "$SESSION" = "$ARCH" ] || SESSION_ENV="RP_SESSION=${SESSION} "
          echo "::error::--ref ${REF} was IGNORED: the live pod is on '${RP_REF:-<none>}'"
          echo "::error::to boot on ${REF}: $(basename "$0") down ${SESSION} && ${SESSION_ENV}$(basename "$0") ${CMD} ${ARCH} --ref ${REF}"
          exit 1
        fi
        echo "attach with: $(basename "$0") attach ${SESSION}"
        exit 0
      elif [ "$REPLACE" != "1" ]; then
        # A recorded-but-unreachable pod is NOT proof the pod is gone — it can
        # be reaped, mid-reboot, or (the 2026-08-25 incident) a DIFFERENT
        # process's `up` on this same alias racing this one. Silently deploying
        # a second pod here overwrites the local record of whichever pod is
        # real, so the wrong one gets `down`ed later. Refuse instead: the
        # operator decides, with the recorded pod in front of them.
        # `up` takes an ARCH positionally, never a session name — they are
        # only the same string until RP_SESSION overrides it (same reasoning
        # as the ref-mismatch message above). Naming the SESSION here (as an
        # earlier version of this message did) reads back as a valid `up`
        # invocation only when SESSION happens to equal ARCH, and boots the
        # WRONG alias — silently, since `up bare-session-name` is parsed as
        # an arch — whenever RP_SESSION was used to override it.
        SESSION_ENV=""
        [ "$SESSION" = "$ARCH" ] || SESSION_ENV="RP_SESSION=${SESSION} "
        echo "::error::session '${SESSION}' already has a recorded pod (${RP_POD_ID}, ${RP_HOST:-?}:${RP_PORT:-?}) that did not answer SSH."
        echo "::error::it may be reaped, mid-reboot, or another process's — refusing to silently replace it."
        echo "::error::inspect it first: $(basename "$0") ls   (or the RunPod console)"
        echo "::error::once you are sure it should be replaced: ${SESSION_ENV}$(basename "$0") up ${ARCH} --replace"
        exit 2
      else
        echo "::warning::--replace: overwriting session '${SESSION}''s local record of pod ${RP_POD_ID} WITHOUT terminating it — run '$(basename "$0") down ${SESSION}' first if it should be terminated"
      fi
    fi
    # Restored regardless of which branch above ran (see the capture above
    # this case's rp_session_load): only the fresh-session and --replace
    # paths reach here, both of which are about to deploy a NEW pod and must
    # never inherit a stale session's recorded TTL/image.
    RP_TTL_HOURS="$UP_TTL_WANTED"
    RP_IMAGE="$UP_IMAGE_WANTED"
    rp_ref_precheck "$REF" || exit 2
    # A recorded-but-dead pod (reaped, or a host that died) must not be carried
    # into the deploy — the EXIT trap would act on a stale id.
    RP_POD_ID=""
    rp_init
    echo "=== provisioning ${ARCH} on ${REF} (image: ${RP_IMAGE}, TTL: ${RP_TTL_HOURS}h) ==="
    rp_deploy_arch "$ARCH" || exit $?
    # No separate arming step: the ${RP_TTL_HOURS}h deadline is baked into the
    # pod's entrypoint at deploy, so it is already running.
    echo "=== bootstrapping ==="
    bootstrap_or_die
    [ -n "$RP_REF" ] && start_seed_build
    rp_keep
    rp_ssh_config_sync
    echo
    echo "=== session '${SESSION}' up on ${RP_HOST}:${RP_PORT} @ ${RP_REF:-<none>} (pod ${RP_POD_ID}) ==="
    echo "    deadline: self-terminates in ${RP_TTL_HOURS}h unless you 'down' it first (RP_DEV_TTL_HOURS/RP_TTL_HOURS to rent longer up front)"
    echo "    ssh:     ssh -F ${RP_SSH_CONFIG} jammi-${SESSION}"
    echo "    attach:  $(basename "$0") attach ${SESSION}"
    echo "    run job: $(basename "$0") run ${SESSION} cargo test -p jammi-ai --features cuda,live-gpu-tests"
    echo "    STOP:    $(basename "$0") down ${SESSION}      # else it self-terminates in ${RP_TTL_HOURS}h"
    ;;

  attach)
    require_pod; rp_keep
    # Joins the running job's terminal when there is one — that is what the word
    # means everywhere else. `--shell` forces a plain prompt, for when you want to
    # poke around WHILE a job runs rather than take its keyboard.
    MODE=job; [ "${1:-}" = "--shell" ] && MODE=shell
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd "$MODE" "$TREE_DIR" "$TMUX_SESSION")" || true
    echo "=== detached; pod ${RP_POD_ID} still running (down with: $(basename "$0") down ${SESSION}) ==="
    ;;

  run)
    TIMING=0
    case "${1:-}" in --timing) TIMING=1; shift ;; esac
    [ $# -gt 0 ] || { echo "run: need a command"; exit 2; }
    require_pod; rp_keep
    JOB="$*"
    # Per-tree job script/log (round-4/5 finding: a global /root/job.sh +
    # /root/jammi.log meant `run --tree b` clobbered tree a's still-running
    # job) — `=`-anchored tmux target (M6), `remain-on-exit off` so a job
    # that exits leaves its output visible in `logs`/`attach` rather than the
    # pane vanishing before either can read it.
    #
    # `--timing`: the launcher's OWN command line acquires the shared timing
    # lock BEFORE the job script runs, INSIDE the detached pane (M6) — never
    # in the (short-lived) ssh invocation that starts tmux and returns
    # immediately, which would release the lock the instant IT exits, not
    # when the real job does. `-n -E 75`: refuse instantly rather than queue,
    # since a human `run --timing` conflicting with another timing-sensitive
    # run should be told NOW, not block silently.
    if [ "$TIMING" = "1" ]; then
      # Redirection (`>`), not a `| tee` pipe: a pipe would make the PANE's
      # own exit status `tee`'s (near-always 0), masking flock's own exit 75
      # refusal from anything reading the pane/log afterward. Output still
      # lands in .jammi.log; it is simply not tailed live via tee.
      LAUNCH="flock -n -E 75 /root/.jammi-timing.lock bash '${TREE_DIR}'/.jammi-job.sh > '${TREE_DIR}'/.jammi.log 2>&1"
    else
      LAUNCH="bash '${TREE_DIR}'/.jammi-job.sh 2>&1 | tee '${TREE_DIR}'/.jammi.log"
    fi
    rp_run_remote <<EOF
set -uo pipefail
cat > '${TREE_DIR}'/.jammi-job.sh <<'JOBEOF'
$(rp_job_wrapper_lines "$TREE_DIR" "$TARGET_DIR" "$JOB")
JOBEOF
tmux kill-session -t "=${TMUX_SESSION}" 2>/dev/null
tmux new-session -d -s "${TMUX_SESSION}" "${LAUNCH}"
tmux set-option -w -t "=${TMUX_SESSION}:" remain-on-exit off
echo "started (tree=${TREE}, timing-locked=${TIMING}): ${JOB}"
EOF
    echo "=== detached. follow with: $(basename "$0") logs ${SESSION} --tree ${TREE} ==="
    ;;

  logs)
    require_pod; rp_keep
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "tail -f -n 200 '${TREE_DIR}/.jammi.log'" || true
    ;;

  push)
    require_pod; rp_keep
    # Ship the working tree, including uncommitted work, but never the local
    # build output — target/ is host-arch and would poison the pod's. The
    # exclude set is defined ONCE, in pod_push_stamp.sh, so the real rsync
    # and the stamp's own manifest hash below can never drift apart. cutlass
    # (a submodule) is excluded here too — it is provisioned by `target
    # --with-cutlass` (never `push` — round-3 audit Class B: an earlier
    # comment here invented a `push --with-cutlass` flag that has never
    # existed), never pushed as plain files (an rsync --delete of it would
    # otherwise delete the pod's own checkout).
    EXCLUDE_ARGS=()
    while IFS= read -r pat; do
      [ -n "$pat" ] || continue
      EXCLUDE_ARGS+=(--exclude "$pat")
    done < <("$DIR/pod_push_stamp.sh" excludes)
    rsync -azc --no-times --delete "${EXCLUDE_ARGS[@]}" \
      -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "${REPO_ROOT}/" "root@${RP_HOST}:${TREE_DIR}/" \
      && echo "=== pushed $(basename "$REPO_ROOT") → pod (tree: ${TREE}) ==="
    rc=$?
    if [ "$rc" -eq 0 ]; then
      STAMP="$(mktemp)"
      "$DIR/pod_push_stamp.sh" compute "$REPO_ROOT" "$SESSION" > "$STAMP" \
        && rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
             "$STAMP" "root@${RP_HOST}:${TREE_DIR}/.jammi-push-stamp.json" \
        && echo "=== push-stamp written to ${TREE_DIR}/.jammi-push-stamp.json (iteration provenance only — a COMMITTED artifact still requires a pushed sha) ==="
      rm -f "$STAMP"
    fi
    ;;

  pull)
    require_pod; rp_keep
    REMOTE="${1:-}"
    [ -n "$REMOTE" ] || { echo "pull: need a remote path (e.g. pull ${SESSION} target/nextest)"; exit 2; }
    mkdir -p "${REPO_ROOT}/.gpu-pull"
    rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "root@${RP_HOST}:${TREE_DIR}/${REMOTE}" "${REPO_ROOT}/.gpu-pull/" \
      && echo "=== pulled ${REMOTE} (tree: ${TREE}) → .gpu-pull/ ==="
    ;;

  target)
    require_pod; rp_keep
    # NAME_TARGET_DIR is the CLONE destination (a CARGO_TARGET_DIR — build
    # OUTPUT), NAME_SOURCE_TREE_DIR is the tree's own SOURCE checkout — two
    # deliberately DIFFERENT directories (round-2 audit finding 1; see
    # rp_target_dir's own doc in runpod_lib.sh). `target` clones INTO the
    # former; cutlass (below) is provisioned into the LATTER, since it is
    # C++ source consumed by build.rs, not a build artifact.
    NAME_TARGET_DIR="$(rp_target_dir "$TARGET_NAME")"
    NAME_SOURCE_TREE_DIR="$(rp_tree_dir "$TARGET_NAME")"
    if [ "$TARGET_VERIFY" = "1" ]; then
      # Reads a `cargo build -v` log piped in on THIS invocation's own stdin
      # and forwards it to pod_target_clone.sh --verify running remotely —
      # ssh is never mocked in this tooling's own test suite, so the
      # hermetic coverage for --verify's actual PASS/FAIL logic lives in
      # test_pod_substrate.sh against the standalone script, not against
      # this remote invocation.
      cat | ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" \
        "bash /root/jammi-ai/ci/scripts/pod_target_clone.sh '' '${NAME_TARGET_DIR}' --verify"
      exit $?
    fi
    [ -n "$TARGET_NAME" ] || { echo "target: need a tree name"; exit 2; }
    rp_run_remote <<EOF
set -uo pipefail
bash /root/jammi-ai/ci/scripts/pod_target_clone.sh /root/.jammi-seed '${NAME_TARGET_DIR}'
EOF
    rc=$?
    if [ "$rc" -eq 0 ] && [ "$TARGET_WITH_CUTLASS" = "1" ]; then
      # `cp -a` from /root/jammi-ai's OWN initialised submodule — never
      # `git submodule update` INSIDE the tree (round-2 audit finding 1): a
      # tree populated by `push` (rsync, which excludes `.git` — see
      # pod_push_stamp.sh) carries no `.git` at all, so `git submodule`
      # there fails with "not a git repository" on every tree except the
      # default bootstrap checkout. /root/jammi-ai IS always a real git
      # clone (rp_bootstrap's own, untouched by push), so its submodule is
      # initialised there once — but round-3 audit N1: /root/jammi-ai's
      # CURRENT gitlink is not necessarily the commit the DESTINATION
      # tree's own ref actually needs (the gitlink has already moved once,
      # 0ee65de) — a tree on an FA2 branch pinning a DIFFERENT cutlass
      # commit than whatever /root/jammi-ai (usually main) happens to have
      # checked out would silently receive the WRONG headers. The tree's
      # own push stamp (pod_push_stamp.sh's cutlass_gitlink field, written
      # at push time from THAT tree's actual HEAD) is the source of truth:
      # verified via pod_push_cutlass_matches (the SAME script this
      # invocation's own hermetic tests exercise, never a second copy of
      # the comparison logic) against /root/jammi-ai's submodule AFTER
      # `submodule update`; on a mismatch, fetch+checkout the STAMPED
      # commit into /root/jammi-ai's own submodule (network — fails loudly
      # if unreachable) and re-verify before copying; refuses the copy on
      # any remaining mismatch, naming both shas.
      # round-4 audit A1: `git rev-parse HEAD:<gitlink-path>` reads the
      # SUPERPROJECT's own recorded pin for that path — a property of
      # /root/jammi-ai's OWN HEAD commit, entirely UNAFFECTED by whether
      # `submodule update` actually ran, or what the submodule's working
      # directory is actually checked out to. It is not a proxy for "what
      # commit does the submodule dir cp -a would copy actually hold" —
      # `git -C <submodule-dir> rev-parse HEAD` is that. Reproduced (the
      # auditor's own repro, confirmed against a real two-commit submodule
      # fixture — see this fix's own hermetic test): the superproject pin
      # stayed unchanged while the submodule HEAD differed, and `cp -a`
      # would have copied the WRONG commit's tree. This also means the
      # OLD remediation arm could never succeed: checking out a different
      # commit INSIDE the submodule cannot change what `HEAD:path` reports
      # in the superproject, so the old re-check after fetch+checkout
      # compared the exact same (always-passing-or-always-failing) pair
      # every time. `set -euo pipefail` (not `-uo pipefail`) so
      # `submodule update` failing aborts here rather than silently
      # continuing into a stale/absent submodule.
      rp_run_remote <<EOF
set -euo pipefail
git -C /root/jammi-ai submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass
[ -d '${NAME_SOURCE_TREE_DIR}' ] || { echo "::error::tree source dir '${NAME_SOURCE_TREE_DIR}' does not exist — push to it first (target --with-cutlass provisions cutlass INTO an existing tree, it does not create one)"; exit 1; }
CUTLASS_DIR=/root/jammi-ai/crates/jammi-kernels/third_party/cutlass
[ -d "\$CUTLASS_DIR/.git" ] || [ -f "\$CUTLASS_DIR/.git" ] || { echo "::error::\$CUTLASS_DIR has no .git after submodule update — deinitialised or never checked out; refusing the copy"; exit 1; }
STAMP='${NAME_SOURCE_TREE_DIR}/.jammi-push-stamp.json'
ACTUAL_SHA="\$(git -C "\$CUTLASS_DIR" rev-parse HEAD)"
bash /root/jammi-ai/ci/scripts/pod_push_stamp.sh cutlass-check "\$STAMP" "\$ACTUAL_SHA"
CHECK_RC=\$?
if [ "\$CHECK_RC" -eq 1 ]; then
  STAMP_SHA="\$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("cutlass_gitlink") or "")' "\$STAMP")"
  echo "attempting to fetch+checkout the stamp's pinned cutlass commit \$STAMP_SHA into the submodule at \$CUTLASS_DIR..."
  git -C "\$CUTLASS_DIR" fetch --depth 1 origin "\$STAMP_SHA" \
    && git -C "\$CUTLASS_DIR" checkout --quiet "\$STAMP_SHA" \
    || { echo "::error::could not fetch/checkout cutlass \$STAMP_SHA into \$CUTLASS_DIR (network unreachable?) — refusing the copy"; exit 1; }
  ACTUAL_SHA="\$(git -C "\$CUTLASS_DIR" rev-parse HEAD)"
  bash /root/jammi-ai/ci/scripts/pod_push_stamp.sh cutlass-check "\$STAMP" "\$ACTUAL_SHA" || { echo "::error::even after fetch+checkout, the SUBMODULE's own HEAD still does not match the stamp — refusing the copy"; exit 1; }
elif [ "\$CHECK_RC" -ne 0 ]; then
  exit 1
fi
mkdir -p '${NAME_SOURCE_TREE_DIR}/crates/jammi-kernels/third_party'
rm -rf '${NAME_SOURCE_TREE_DIR}/crates/jammi-kernels/third_party/cutlass'
cp -a "\$CUTLASS_DIR" '${NAME_SOURCE_TREE_DIR}/crates/jammi-kernels/third_party/cutlass'
# round-4 addendum: \$CUTLASS_DIR's own \`.git\` is a SUBMODULE GITLINK
# pointer file (not a full repo), and \`cp -a\` copies it verbatim into the
# destination tree — a plain directory tree that is not itself registered
# as owning that gitlink. \${NAME_SOURCE_TREE_DIR} is itself a real git
# checkout (rp_bootstrap's default tree, or a pushed tree whose OWN .git
# already exists); a second, foreign, un-registered .git nested inside it
# makes \`git status\`/\`git add\` run from that tree's root treat the path
# as an embedded-repository boundary it cannot resolve, failing fatally.
# The gitlink file is never needed in the copy (this path is deliberately
# NOT git-managed inside the destination tree at all — target --with-
# cutlass provisions it by \`cp -a\`, never by \`git submodule\` inside the
# tree, exactly because a pushed tree carries no .git of its own to attach
# a submodule to). Strip it and assert it is gone.
rm -rf '${NAME_SOURCE_TREE_DIR}/crates/jammi-kernels/third_party/cutlass/.git'
[ -e '${NAME_SOURCE_TREE_DIR}/crates/jammi-kernels/third_party/cutlass/.git' ] && { echo "::error::cutlass/.git still present in the destination tree after stripping — refusing to leave a foreign gitlink in a git-backed tree"; exit 1; }
true
EOF
      rc=$?
    fi
    echo "=== target '${TARGET_NAME}' — clone at ${NAME_TARGET_DIR}, cutlass ${TARGET_WITH_CUTLASS}: exit ${rc} ==="
    exit "$rc"
    ;;

  down)
    if rp_session_load; then
      # Never trust the locally-recorded id on its own: confirm it is BOTH
      # still present in the account AND named like one of this tooling's
      # own pods before terminating anything. This is what stops `down`
      # from ending a pod that a race with another `up` on the same alias
      # silently swapped in underneath this session's record (2026-08-25
      # incident) — a mismatch refuses rather than acts. The id is
      # authoritative (RunPod pod ids are globally unique); the TTL never
      # gates release — see rp_pod_verify's own doc in runpod_lib.sh for why
      # two earlier attempts to make the TTL part of this check were both
      # removed rather than patched again.
      rp_pod_verify "$RP_POD_ID" >/dev/null
      verify_rc=$?
      if [ "$verify_rc" -eq 0 ]; then
        rp_terminate "$RP_POD_ID"
        # rp_terminate's own result is thrown away by design (it also runs
        # as rp_cleanup's best-effort EXIT-trap teardown) — `down` is a
        # single deliberate action and demands independent confirmation
        # before it forgets the local record. A rejected podTerminate must
        # never both leak the pod AND destroy the only record pointing at
        # it.
        if rp_pod_gone "$RP_POD_ID"; then
          echo "=== terminated pod ${RP_POD_ID} (session '${SESSION}') ==="
          RP_POD_ID=""   # already gone; keep the EXIT trap from acting on it again
          rp_session_forget
          # After the session file is gone, so the dead host stops being offered.
          rp_ssh_config_sync
        else
          echo "::error::terminate not confirmed for pod ${RP_POD_ID} (session '${SESSION}') — it is still present in the account's pod list."
          echo "::error::the local session record is KEPT — retry: $(basename "$0") down ${SESSION}"
          exit 1
        fi
      elif [ "$verify_rc" -eq 3 ]; then
        # The recorded id is ABSENT from the account entirely — not an
        # ambiguity to refuse on. This is the ORDINARY shape of "this pod
        # already ended on its own" (its own in-pod deadline, or the
        # sweep) — the single most common way a session's pod goes away.
        # Nothing is left to release, so this is cleanup, not a refusal:
        # forget the record and say so plainly, rather than leaving it
        # stuck until an operator remembers `up --replace`.
        #
        # This shares rp_pod_verify's underlying assumption (see
        # rp_pod_gone's own doc in runpod_lib.sh, which states it directly):
        # the account's `myself{ pods{...} } }` response is COMPLETE, never
        # paginated. "Absent from the returned list" is only "absent from
        # the account" under that assumption, verified live on first use —
        # not merely inferred from the schema.
        echo "recorded pod ${RP_POD_ID} is gone from the account (deadline/sweep) — forgetting the record"
        RP_POD_ID=""   # already gone; keep the EXIT trap from acting on it again
        rp_session_forget
        rp_ssh_config_sync
      else
        # verify_rc is 1 (present, but under a name that is not this
        # tooling's shape) or 2 (could not query the account at all) — both
        # REAL ambiguities, unlike a confirmed-absent id above. The LOCAL
        # record is deliberately KEPT here, not forgotten: this is exactly
        # the case where a follow-up `up` on this alias most needs to still
        # see a recorded pod and refuse (or ask for --replace) rather than
        # silently deploying a THIRD pod on top of an already-confused
        # alias.
        echo "::error::refusing to terminate pod ${RP_POD_ID} for session '${SESSION}' — it did not verify against the account's live pod list (see above)."
        echo "::error::the local session record is KEPT (not forgotten), so this alias still refuses a plain 'up' rather than deploying on top of the ambiguity."
        # `reap` is ACCOUNT-WIDE — it judges EVERY jammi-gpu* pod against its
        # OWN deadline, never just this one (probed: `reap 1` terminated
        # unrelated 4h-old pods) — so it is never a per-pod remedy. There is
        # also no per-pod OVERRIDE remedy any more: an explicit
        # RP_TTL_HOURS on this invocation was tried and found INERT — the
        # exact TTL was never part of the check to begin with once
        # rp_pod_verify moved to id + name shape, so an override had nothing
        # left to override (round-3 audit, probe d2) — and was removed
        # rather than left as a promise that does nothing.
        echo "::error::inspect it: $(basename "$0") ls   /   the RunPod console"
        exit 1
      fi
    else
      echo "no recorded pod for session '${SESSION}'"
    fi
    ;;

  *) echo "unknown command: $CMD"; usage 2 ;;
esac
