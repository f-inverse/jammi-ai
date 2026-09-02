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
# `push`/`run`/`target` act on YOUR OWN checkout — REPO_ROOT below (what
# push rsyncs, and the checkout whose gpu-dev.sh copy you must be running)
# is resolved from THIS SCRIPT's own on-disk location, never from $PWD. On a
# laptop with more than one checkout of this repo (a multi-worktree swarm),
# always invoke the copy INSIDE the tree you mean to act on — these three
# verbs refuse (naming both paths) when $PWD's own git toplevel disagrees
# with that location; RP_ALLOW_ROOT_MISMATCH=1 overrides for deliberate
# cross-tree use.
#
# Usage:
#   gpu-dev.sh shell   [arch] [--ref R] [--tree T]   throwaway shell; pod dies on exit
#   gpu-dev.sh up      [arch] [--ref R]              start a surviving session (name = arch)
#                      [--replace]                   replace an alias's existing record
#   gpu-dev.sh target  [session] <name>              clone the pod's build-substrate seed
#                      [--verify] [--with-cutlass]    into a fresh CARGO_TARGET_DIR
#                      [--adopt]                       (`/root/target-<name>`) for a tree that
#                                                      already exists — `push` to <name> FIRST;
#                                                      --adopt instead stamps the marker on an
#                                                      already-WARM dir that carries none
#   gpu-dev.sh attach  [session] [--tree T]           shell into a surviving session
#   gpu-dev.sh run     [session] [--tree T] <cmd...>  run <cmd> detached under tmux — REFUSES
#                      [--wave W]                      (esc-077) unless <tree>'s CARGO_TARGET_DIR
#                                                      carries a `target`-stamped clone marker
#                                                      (`.jammi-clone-of-seed`); remedy in the
#                                                      error, or RP_ALLOW_COLD_TARGET=1 to force —
#                                                      ALSO refuses (one-pod-per-wave, WAVE-scoped)
#                                                      a job while a DIFFERENT wave's job is live on
#                                                      this pod; --wave/RP_WAVE (default: the tree
#                                                      name) lets a wave's own sub-units share one
#                                                      pod across trees; rent another pod or
#                                                      RP_ALLOW_CONCURRENT=1 for real cross-wave use
#   gpu-dev.sh logs    [session] [--tree T]           tail the detached job's output
#   gpu-dev.sh push    [session] [--tree T]           rsync YOUR OWN checkout (this script's
#                                                      own on-disk location, never $PWD) TO the pod
#   gpu-dev.sh pull    [session] [--tree T] <path>    rsync <path> back FROM the pod
#   gpu-dev.sh wait-seed [session] [--timeout SECS]   block until the pod's own build-substrate
#                                                      seed completes, fails, or the timeout
#                                                      elapses — never silently misreads an
#                                                      unreachable pod as "still building"
#   gpu-dev.sh wait-job [session] [--tree T]          block until <tree>'s detached `run` job
#                       [--timeout SECS]               ends, fails, or the timeout elapses
#   gpu-dev.sh down    [session]                       terminate the pod, forget the session
#   gpu-dev.sh ls                                      list sessions
#
# arch: a100 (default) | l40s | h100 | a40 | l4
# --ref: branch, tag or commit the pod's checkout is placed on (default main).
#        Verified against the remote BEFORE a pod is rented.
# --tree: the checkout/tree a run/push/pull/attach/logs/wait-job/target command
#        acts on (default "jammi-ai" — the bootstrap checkout at
#        /root/jammi-ai). Any other name is a plain directory at
#        /root/trees/<name>, populated ONLY by `push --tree <name>` (rsync) —
#        NEVER by `target`, which clones the build substrate into a wholly
#        disjoint CARGO_TARGET_DIR namespace (/root/target-<name>, see
#        rp_target_dir in runpod_lib.sh and pod_target_clone.sh) rather than
#        the tree itself, and never git-cloned either way. `target
#        --with-cutlass` therefore REFUSES against a tree that has not been
#        pushed yet (pod_provision_cutlass.sh's own error: "tree source dir
#        does not exist — push to it first") — it provisions cutlass INTO an
#        existing tree, it does not create one.
# push/run/target REFUSE (exit 2, naming both paths) when $PWD's own git
#        toplevel disagrees with REPO_ROOT (this script's own on-disk
#        location) — a multi-worktree laptop invoking one tree's script copy
#        from inside a DIFFERENT tree would otherwise silently act on the
#        WRONG one. RP_ALLOW_ROOT_MISMATCH=1 overrides for deliberate
#        cross-tree use.
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
#      RP_ALLOW_ROOT_MISMATCH (push/run/target only — see above).
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
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "${HOME}/.config/runpod/key" 2>/dev/null || true)}" # tripwire-ok: an absent key file is a valid, common state (no key configured yet) — the `:?` right below fails loudly and immediately if the result is still empty, never a silent pass
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY or write it to ~/.config/runpod/key}"

usage() {
  cat <<'USAGE'
gpu-dev.sh — GPU development on RunPod

  shell   [arch] [--ref R] [--tree T]     throwaway shell; the pod dies when you exit
  up      [arch] [--ref R]                start a session whose pod SURVIVES disconnect
          [--replace]                     (default TTL 72h — see RP_DEV_TTL_HOURS below)
                                          refuses if the alias already has a recorded pod
                                          unless --replace is given (see --replace below)
  target  [session] <name>                clone the pod's build-substrate seed into a fresh
          [--verify] [--with-cutlass]     CARGO_TARGET_DIR (/root/target-<name>) for a tree
          [--adopt]                       that already exists — push to <name> FIRST
                                          (pod_target_clone.sh). --adopt does NOT clone: it
                                          re-checks an EXISTING warm dir's own content and
                                          stamps the seed-clone marker on it — the remedy
                                          for run's UNMARKED_WARM refusal, which a clone
                                          cannot serve (it refuses an existing destination)
  attach  [session] [--tree T]            join a surviving session's running job
                                          (--shell for a plain prompt instead)
  run     [session] [--tree T] <cmd...>   run <cmd> detached under tmux, in <tree> — REFUSES
          [--wave W]                     (esc-077) unless <tree>'s CARGO_TARGET_DIR carries a
                                          `target`-stamped clone marker (else a silent cold
                                          full-workspace build); RP_ALLOW_COLD_TARGET=1 bypasses.
                                          ALSO refuses (one-pod-per-wave, WAVE-scoped) while a
                                          DIFFERENT wave's job is live on this pod (--wave/RP_WAVE
                                          defaults to the tree name — a wave's own sub-units may
                                          share one pod across trees under the SAME wave id);
                                          RP_ALLOW_CONCURRENT=1 bypasses for real cross-wave use
  logs    [session] [--tree T]            tail the detached job's output
  push    [session] [--tree T]            rsync YOUR OWN checkout (this script's own on-disk
                                          location, never $PWD) TO the pod's <tree>
  pull    [session] [--tree T] <path>     rsync <path> back FROM the pod's <tree>
  wait-seed [session] [--timeout SECS]    block until the pod's own build-substrate seed
                                          completes/fails/times out (never misreads an
                                          unreachable pod as "still building" — see below)
  wait-job  [session] [--tree T]          block until <tree>'s detached `run` job ends,
            [--timeout SECS]              fails (no evidence it ever ran), or times out
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
--tree T: which checkout on the pod a run/push/pull/attach/logs/wait-job/
         target command acts on (default "jammi-ai", the bootstrap checkout
         at /root/jammi-ai). Any other name is a plain directory at
         /root/trees/<name>, populated ONLY by `push --tree <name>` (rsync)
         — NEVER by `target`, which clones the build substrate into a wholly
         disjoint CARGO_TARGET_DIR namespace (/root/target-<name>, a build
         OUTPUT directory, not the tree/checkout itself) — never a git
         worktree (a shared .git would couple trees that must diverge
         independently) and never git-cloned either way.
push/run/target REFUSE (exit 2, naming both paths) when the current
         directory's own git toplevel disagrees with REPO_ROOT (this
         script's own on-disk location — a multi-worktree laptop invoking
         one tree's script copy from inside a DIFFERENT tree would
         otherwise silently push/run/target the WRONG one). Invoke that
         tree's OWN copy of this script instead; RP_ALLOW_ROOT_MISMATCH=1
         overrides for deliberate cross-tree use.
--replace: `up` normally REFUSES (exit 2) a session alias that already has a
         recorded pod id, even an unreachable one, rather than silently
         deploying a second pod under the same alias. --replace overwrites only
         the LOCAL record; it never terminates the old pod itself — `down` it
         first if it should be.
--verify (target only): after `target` clones and you have built once against
         the new tree, re-run with --verify and a `cargo build -v` log on
         stdin to assert no member unit reported Fresh (see pod_target_clone.sh).
--with-cutlass (target only): provisions the CUTLASS submodule into an
         ALREADY-PUSHED tree (`pod_provision_cutlass.sh`: a `cp -a` copy from
         /root/jammi-ai's own initialised submodule, never a bare `git
         submodule update` run inside the tree itself) — needed only for a
         tree that will build the `flash-attn` feature. `target` does not
         create the tree — push to <name> FIRST (`gpu-dev.sh push --tree
         <name>`); against a tree that has never been pushed,
         pod_provision_cutlass.sh REFUSES with "tree source dir does not
         exist — push to it first" rather than silently doing nothing.
--timeout SECS (wait-seed/wait-job only): overall wall-clock budget for the
         poll loop (default RP_WAIT_TIMEOUT_SECS, else 5400s/90min — the
         pod-build-guide's own "budget tens of minutes for a fresh Ampere pod
         compiling the full cuda+FA2 graph" note, with real headroom). An
         unreachable pod is NEVER read as "still running": RP_WAIT_MAX_
         TRANSPORT_FAILS (default 3) consecutive unreachable polls exits
         loudly instead (see wait-seed/wait-job below).
Sessions are named after the arch; RP_SESSION names one explicitly. On
down/attach/run/logs/push/pull/wait-seed/wait-job/target, RP_SESSION and an
explicit positional session must AGREE — a differing pair REFUSES (exit 2,
naming both) rather than silently picking one. A session name may contain
letters, digits, _, -, and . anywhere but as the whole name (so bench.1 is
fine; ., .., a name containing /, or one starting with - are refused).

wait-seed/wait-job poll the pod at RP_WAIT_INTERVAL_SECS (default 20s) up to
--timeout, and distinguish three outcomes: SUCCESS (exit 0 — wait-seed's
/root/.jammi-seed.jammi-seed-complete marker; wait-job's tmux job session has
ended AND the tree's .jammi.log exists — the job ran to completion, inspect
the log for its own pass/fail verdict), a NAMED FAILURE (exit 1 —
wait-seed's .jammi-seed-failed marker, or "no evidence this ever ran": no
marker/session and, for wait-job, no .jammi.log either), or a TRANSPORT
FAILURE (exit 2 — RP_WAIT_MAX_TRANSPORT_FAILS consecutive unreachable polls;
this means the pod could not be reached, never that the job/seed is still
running). A timeout with no verdict either way exits 3.
Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,
     RP_TTL_HOURS (default 8; explicit always wins), RP_DEV_TTL_HOURS (default
     72 — what `up` alone falls back to when RP_TTL_HOURS is not set),
     RP_DISK_GB (default 60), RP_VOLUME_GB (default 0),
     RP_WAIT_TIMEOUT_SECS (wait-seed/wait-job default, 5400 — overridden
     per-invocation by --timeout), RP_WAIT_INTERVAL_SECS (poll interval,
     default 20), RP_WAIT_MAX_TRANSPORT_FAILS (default 3).
     Disk sizing once the seed/clone substrate is in use: RP_DISK_GB >= 25
     (base) + S_src + S_seed + N*S_clone (one clone per tree this pod hosts);
     the S_src/S_seed/S_clone byte counts are MEASURED by
     ci/scripts/perf/pod_build_timings.sh (A2), not guessed — see
     docs/maintainer/dev-gpu.md, which cites the committed JSON under
     ci/artifacts/pod-build-timings/ (src/seed/clone ≈ 3.6/7.8/8.1 GB). Add 3 GB per OTHER concurrent agent target
     dir + 2 GB per `cargo mutants` job — a mutation-testing session wants
     >= 120 GB (RP_DISK_GB=150).
     RP_ALLOW_ROOT_MISMATCH (push/run/target only — see above).
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

# esc-056 (defect 2): REPO_ROOT above is derived from THIS SCRIPT'S OWN
# on-disk location, never from the caller's $PWD — a laptop with more than
# one checkout of this repo (a multi-worktree swarm: separate `git
# worktree` checkouts, each with its own copy of this very script) can
# invoke ONE tree's copy of gpu-dev.sh from INSIDE a DIFFERENT tree's
# directory, and push/run/target would then silently act on the SCRIPT's
# own tree (REPO_ROOT) — never the tree the caller believes they are
# standing in. Observed live: M1b's pod legs initially validated the WRONG
# tree via the main checkout's own script copy; the push-stamp's own
# laptop_head field was the only tell. Fail closed rather than silently
# doing the wrong thing: refuse when $PWD's own git toplevel names a
# DIFFERENT root than REPO_ROOT, naming both paths and the fix (invoke that
# tree's OWN copy of this script) — checked here, before require_pod/rp_init
# ever contact the pod, exactly like the other usage-shape checks in this
# file. A $PWD that is not itself a git checkout (git rev-parse fails, e.g.
# a plain non-worktree "tree" directory on the LAPTOP side) falls back to a
# raw string comparison against $PWD — still refuses on a genuine mismatch,
# never silently "passes" for lack of a git answer. RP_ALLOW_ROOT_MISMATCH=1
# opts into the mismatch deliberately (e.g. one tree's helper acting on
# another tree's already-up pod session on purpose).
case "$CMD" in
  push|run|target)
    if [ "${RP_ALLOW_ROOT_MISMATCH:-0}" != "1" ]; then
      CWD_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)" # tripwire-ok: a $PWD that is not itself a git checkout is a valid, common state (a plain non-worktree "tree" directory) — the raw-$PWD fallback right below is what actually handles it, never a silent pass
      CWD_ROOT="${CWD_ROOT:-$PWD}"
      if [ "$CWD_ROOT" != "$REPO_ROOT" ]; then
        echo "::error::this script's own location resolves to REPO_ROOT=${REPO_ROOT}, but the current directory resolves to ${CWD_ROOT} — '${CMD}' would silently act on ${REPO_ROOT}, NOT the tree you are standing in."
        echo "::error::invoke THAT tree's own copy instead: ${CWD_ROOT}/ci/scripts/gpu-dev.sh ${CMD} ..."
        echo "::error::deliberate cross-tree use: set RP_ALLOW_ROOT_MISMATCH=1 to override"
        exit 2
      fi
    fi
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
TARGET_ADOPT=0
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
    # target [session] <name> [--verify|--adopt] [--with-cutlass]
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
        --adopt) TARGET_ADOPT=1; shift ;;
        --with-cutlass) TARGET_WITH_CUTLASS=1; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "::error::unknown option '$1' for target"; usage 2 ;;
        *)
          [ -z "$TARGET_NAME" ] || { echo "::error::target: unexpected argument '$1'"; usage 2; }
          TARGET_NAME="$1"; shift ;;
      esac
    done
    [ -n "$TARGET_NAME" ] || [ "$TARGET_VERIFY" = "1" ] || { echo "::error::target: need a tree name (or --verify against an existing one)"; usage 2; }
    [ "$TARGET_VERIFY" = "1" ] && [ "$TARGET_ADOPT" = "1" ] && { echo "::error::target: --verify and --adopt are different operations; pass at most one"; usage 2; }
    ;;
  wait-seed|wait-job)
    # wait-seed [session] [--timeout SECS]
    # wait-job  [session] [--tree T] [--timeout SECS]
    # A dedicated flag loop (like `target` above), not the generic `*)`
    # branch below: --timeout is a new flag neither `up` nor the plain
    # session verbs have, and wait-job's own --tree may appear in either
    # order relative to it, unlike the single-leading-flag convention the
    # generic branch's own attach/run/logs/push/pull case relies on.
    ARG=""
    case "${1:-}" in
      -*) : ;;
      *) [ $# -gt 0 ] && { ARG="$1"; shift; } ;;
    esac
    if [ -n "$ARG" ] && [ -n "${RP_SESSION:-}" ] && [ "$ARG" != "$RP_SESSION" ]; then
      echo "::error::conflicting session: positional argument '${ARG}' vs exported RP_SESSION='${RP_SESSION}' — they name different sessions"
      echo "::error::pick one: unset RP_SESSION to act on '${ARG}', or drop the positional to act on RP_SESSION='${RP_SESSION}'"
      exit 2
    fi
    SESSION="${ARG:-${RP_SESSION:-a100}}"; ARCH=""
    WAIT_TIMEOUT=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --tree)
          [ "$CMD" = "wait-job" ] || { echo "::error::--tree applies only to 'wait-job', not '${CMD}'"; exit 2; }
          [ $# -ge 2 ] || { echo "::error::--tree needs a value"; exit 2; }
          TREE="$2"; shift 2 ;;
        --tree=*)
          [ "$CMD" = "wait-job" ] || { echo "::error::--tree applies only to 'wait-job', not '${CMD}'"; exit 2; }
          TREE="${1#--tree=}"; shift ;;
        --timeout)
          [ $# -ge 2 ] || { echo "::error::--timeout needs a value (seconds)"; exit 2; }
          WAIT_TIMEOUT="$2"; shift 2 ;;
        --timeout=*)
          WAIT_TIMEOUT="${1#--timeout=}"; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "::error::unknown option '$1' for ${CMD}"; usage 2 ;;
        *) echo "::error::${CMD}: unexpected argument '$1'"; usage 2 ;;
      esac
    done
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
    # `--tree`/`--wave`, when present, are LEADING flags right after the
    # (optional) session positional — before `run`'s own command tail,
    # which this loop must never touch (`cargo test -- --tree-of-life` is
    # someone's ACTUAL command, not a flag for this script). Only
    # attach/run/logs/push/pull take `--tree`; `down` acts on the pod, not
    # a tree within it. `--wave` (one-pod-per-wave, the WAVE identity a
    # concurrency claim is recorded/compared under — see
    # rp_concurrency_preflight_lines) is narrower still: only `run` (which
    # records the claim) and `push` (so an operator can set the SAME
    # `--wave`/RP_WAVE across both verbs without either one rejecting it —
    # `push` itself writes no claim; the claim exists only once a job
    # actually launches) accept it. A LOOP, not a one-shot `case` like the
    # old `--tree`-only form: the two flags may now appear in either order.
    case "$CMD" in
      attach|run|logs|push|pull)
        while :; do
          case "${1:-}" in
            --tree)
              [ $# -ge 2 ] || { echo "::error::--tree needs a value"; exit 2; }
              TREE="$2"; shift 2 ;;
            --tree=*)
              TREE="${1#--tree=}"; shift ;;
            --wave)
              case "$CMD" in
                run|push) : ;;
                *) echo "::error::--wave applies only to 'run'/'push', not '${CMD}'"; exit 2 ;;
              esac
              [ $# -ge 2 ] || { echo "::error::--wave needs a value"; exit 2; }
              RP_WAVE="$2"; shift 2 ;;
            --wave=*)
              case "$CMD" in
                run|push) : ;;
                *) echo "::error::--wave applies only to 'run'/'push', not '${CMD}'"; exit 2 ;;
              esac
              RP_WAVE="${1#--wave=}"; shift ;;
            *) break ;;
          esac
        done
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

# Resolves and validates the three wait-seed/wait-job knobs. $1 = --timeout's
# value (empty if not given — RP_WAIT_TIMEOUT_SECS, then the 5400s/90min
# default, apply in that order; 90 minutes budgets a fresh Ampere pod's full
# cuda+FA2 seed build with real headroom — pod-build-guide.md §4's own
# "budget tens of minutes" note). Sets WAIT_TIMEOUT_S/WAIT_INTERVAL_S/
# WAIT_MAX_FAIL as plain (non-`local`) variables, or exits 2 naming the bad
# value. Called BEFORE anything is rented/contacted (same discipline as the
# RP_DEV_TTL_HOURS validation above it) — a malformed --timeout is a usage
# error, not something to discover after require_pod has already run.
wait_resolve_knobs() {
  WAIT_TIMEOUT_S="${1:-${RP_WAIT_TIMEOUT_SECS:-5400}}"
  case "$WAIT_TIMEOUT_S" in
    ''|*[!0-9]*) echo "::error::--timeout/RP_WAIT_TIMEOUT_SECS must be a positive integer (seconds), got '${WAIT_TIMEOUT_S}'"; exit 2 ;;
  esac
  [ "$WAIT_TIMEOUT_S" -gt 0 ] || { echo "::error::--timeout/RP_WAIT_TIMEOUT_SECS must be > 0"; exit 2; }

  WAIT_INTERVAL_S="${RP_WAIT_INTERVAL_SECS:-20}"
  case "$WAIT_INTERVAL_S" in
    ''|*[!0-9]*) echo "::error::RP_WAIT_INTERVAL_SECS must be a positive integer (seconds), got '${WAIT_INTERVAL_S}'"; exit 2 ;;
  esac
  [ "$WAIT_INTERVAL_S" -gt 0 ] || { echo "::error::RP_WAIT_INTERVAL_SECS must be > 0"; exit 2; }

  WAIT_MAX_FAIL="${RP_WAIT_MAX_TRANSPORT_FAILS:-3}"
  case "$WAIT_MAX_FAIL" in
    ''|*[!0-9]*) echo "::error::RP_WAIT_MAX_TRANSPORT_FAILS must be a positive integer, got '${WAIT_MAX_FAIL}'"; exit 2 ;;
  esac
  [ "$WAIT_MAX_FAIL" -gt 0 ] || { echo "::error::RP_WAIT_MAX_TRANSPORT_FAILS must be > 0"; exit 2; }
}
case "$CMD" in
  wait-seed|wait-job) wait_resolve_knobs "$WAIT_TIMEOUT" ;;
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

# Validated BEFORE any path derives from it or any remote heredoc could
# embed it (round-N audit: the injection class `wait-job`'s own new heredoc
# site joins — see rp_tree_name_check's own doc). The default "jammi-ai"
# passes trivially; only a caller-supplied --tree can fail this.
RP_TREE_CHECK_VALUE="$TREE" rp_tree_name_check || exit 2
# `target`'s POSITIONAL name is a tree name too — it is what rp_target_dir/
# rp_tree_dir resolve for that verb (:1148), and both results go straight
# into a remote heredoc. It never passes through $TREE, so the check above
# does not cover it: `target a100 '<injection>'` reached the pod unchecked.
# Empty is the "no name given" state `target --verify` legitimately has, so
# only a non-empty value is checked here; the verb's own
# "need a tree name" refusal still handles the empty case.
[ -z "$TARGET_NAME" ] || { RP_TREE_CHECK_VALUE="$TARGET_NAME" rp_tree_name_check || exit 2; }

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
# One-pod-per-wave is WAVE-scoped, not tree-scoped (operator-directed
# refinement of the original tree-scoped gate, 8515cbb9): a single wave
# legitimately spans more than one tree (e.g. a CPU-build sub-unit and a
# GPU-test sub-unit sharing the same warm seed), and the old tree-scoped
# check tripped a wave against ITSELF the moment it used a second tree.
# WAVE defaults to the tree name — a caller who sets neither `--wave` nor
# RP_WAVE gets EXACTLY 8515cbb9's tree-scoped behavior (the regression this
# default is required to preserve; pinned by Group 10's own
# default-wave-is-tree-scoped test). Resolved here, for every verb that
# reaches this point (not just `run`), so `push --wave X` / `RP_WAVE=X
# gpu-dev.sh push ...` is accepted rather than erroring on an unrecognised
# concept — `push` itself writes no claim (only `run`'s job launch does;
# see rp_job_wrapper_with_marker_lines's own doc), so this is purely so an
# operator can set ONE `--wave`/RP_WAVE across a whole push-then-run
# sequence without either verb rejecting it.
WAVE="${RP_WAVE:-$TREE}"
# Same rule as --tree, for the same reason: WAVE is written into the pod's
# active-wave claim and compared as a literal inside the concurrency
# preflight's generated text, so an unvalidated value is remote shell text.
# The `${RP_WAVE:-$TREE}` default inherits an already-checked tree name;
# only a caller-supplied --wave/RP_WAVE can fail this.
RP_WAVE_CHECK_VALUE="$WAVE" rp_wave_name_check || exit 2

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
  local pre="${wlines[0]}; ${wlines[1]}; ${wlines[2]} 2>/dev/null;" # tripwire-ok: builds a REMOTE command-string template; the cd suppression is intentional (see the comment above -- a not-yet-provisioned tree must land an interactive shell, not abort the whole ssh session)
  if [ "${1:-}" = "job" ]; then
    # Ctrl-C inside the job's pane signals the job itself, so say so before
    # handing over the keyboard — this is a terminal that can destroy work.
    # tripwire-ok (has-session check below): a REMOTE command-string
    # template fragment -- "no such session" is a real, valid state
    # (checked explicitly by the if/then), never a silent pass.
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

# esc-056: a bootstrap failure ends this invocation before `rp_keep` is ever
# called, so the EXIT trap's own best-effort `rp_terminate` (rp_cleanup,
# runpod_lib.sh) is the ONLY thing standing between this pod and orphaned
# billing — and that call throws its own response away by design (it also
# runs on every ordinary exit, where a network hiccup must not turn a normal
# teardown into a hard failure). Never a swallowed exit: name the pod id and
# the session it is recorded under (the write-ahead record — see
# rp_deploy_live's own doc in runpod_lib.sh — already has it on disk the
# moment the pod exists, well before this can run) so a silently-failed trap
# termination is still recoverable by hand. `shell` never reaches here with a
# named session (RP_SESSION is force-cleared for it above), so this is
# effectively `up`-only.
report_pod_recorded() {
  [ -n "${RP_SESSION:-}" ] && [ -n "${RP_POD_ID:-}" ] || return 0
  echo "::error::pod ${RP_POD_ID} recorded under session '${RP_SESSION}'; run '$(basename "$0") down ${RP_SESSION}' to terminate"
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
         report_pod_recorded
         exit 1
       }
       echo "=== no checkout: this image ships no git, so the pod is on no ref ==="
       return 0 ;;
    *) echo "::error::bootstrap failed — terminating pod (trap)"; report_pod_recorded; exit 1 ;;
  esac
}

# Kicks off the pod's build-substrate seed, DETACHED — never blocks
# `shell`/`up` (a cold seed is real compile minutes). Idempotent:
# pod_seed_target.sh itself no-ops when .jammi-seed-complete already exists
# (unless --reseed). Runs under the SAME lock-in-pane pattern as `run
# --timing` (M6): the flock acquisition is the FIRST thing the detached
# pane's own command does, so the lock's lifetime is the seed job's
# lifetime, not this short-lived launcher's.
#
# Deliberately NOT given the `target` verb's own staged-caller-scripts fix
# (deployment-gap note there): the seed BUILDS a CARGO_TARGET_DIR for the
# pod's bootstrapped `/root/jammi-ai` checkout AT THE COMMIT that checkout
# is actually on — running THIS laptop's own (possibly different-commit)
# pod_seed_target.sh against that tree could seed against a workspace
# member set / lockfile the checkout does not have, which is a worse
# inconsistency than the one being avoided. No laptop-side preflight reads
# a marker THIS script's version-specific behavior would need to agree
# with (unlike pod_target_clone.sh's marker, which esc-077's
# rp_target_preflight_lines DOES check) — the seed's own completion marker
# (.jammi-seed-complete) is read by rp_seed_wait_script, whose OWN
# reasonable expectations (a JSON marker existing/not) have not changed
# across any version relevant here.
start_seed_build() {
  local reseed_flag=""
  [ "$RESEED" = "1" ] && reseed_flag="--reseed"
  rp_run_remote <<EOF
set -uo pipefail
tmux kill-session -t "=jammi-seed" 2>/dev/null # tripwire-ok: idempotent best-effort cleanup of a session that may legitimately not exist yet (first-ever seed build) — the new-session command right below is what actually matters and is unconditional
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
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd "" "$TREE_DIR" "$TMUX_SESSION")" || true # tripwire-ok: an interactive TTY session's own exit status is meaningless here (a user typing `exit`, a dropped connection, Ctrl-C — none are this command's own success/failure); the trap on the line below always terminates the pod regardless
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
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd "$MODE" "$TREE_DIR" "$TMUX_SESSION")" || true # tripwire-ok: same as the `shell` case above — an interactive TTY session's own exit status is not this command's success/failure signal
    echo "=== detached; pod ${RP_POD_ID} still running (down with: $(basename "$0") down ${SESSION}) ==="
    ;;

  run)
    TIMING=0
    case "${1:-}" in --timing) TIMING=1; shift ;; esac
    [ $# -gt 0 ] || { echo "run: need a command"; exit 2; }
    require_pod; rp_keep

    # esc-077: refuse a job whose CARGO_TARGET_DIR (TARGET_DIR, resolved at
    # :562 from --tree) never went through the seed-clone substrate
    # (pod_target_clone.sh, the `target` verb below) — a missing or unmarked
    # target dir means this job would silently pay a full COLD workspace
    # build (~20-40min) instead of the seeded clone's ~80s incremental
    # build, and nothing else at this launch choke point would notice
    # (esc-077's own observable: four wave contracts paid this cost
    # unnoticed). Checked on the POD's filesystem — TARGET_DIR is a remote
    # path — before anything is launched. RP_ALLOW_COLD_TARGET=1 is the
    # SOLE, explicit bypass; `ci/scripts/perf/pod_build_timings.sh`'s own
    # cold-build leg sets it deliberately (see that script's own comment).
    # Documented residual: this covers only a job launched THROUGH this
    # verb — a caller who `ssh`es in directly and runs `cargo` by hand
    # bypasses this wrapper entirely (stated in docs/maintainer/
    # pod-build-guide.md and this script's own usage header).
    if [ "${RP_ALLOW_COLD_TARGET:-0}" != "1" ]; then
      TARGET_PREFLIGHT_STATE="$(rp_run_remote <<EOF
set -uo pipefail
$(rp_target_preflight_lines "$TARGET_DIR")
EOF
)"
      case "$TARGET_PREFLIGHT_STATE" in
        *GPU_DEV_TARGET_STATE=OK*) : ;;
        *GPU_DEV_TARGET_STATE=MISSING*)
          echo "::error::run refused: CARGO_TARGET_DIR ${TARGET_DIR} does not exist on the pod — this job would pay a COLD full workspace build (~20-40min) instead of the seeded clone's ~80s incremental build." >&2
          echo "remedy: $(basename "$0") target ${SESSION} ${TREE} --with-cutlass   (or set RP_ALLOW_COLD_TARGET=1 to proceed cold deliberately)" >&2
          exit 1
          ;;
        *GPU_DEV_TARGET_STATE=UNMARKED_WARM*)
          # A dir that predates the marker scheme, or was built by any path
          # other than the `target` verb: it is genuinely WARM (it carries
          # this workspace's own member fingerprints), so the refusal is
          # about PROVENANCE, not a cold build — and the remedy has to be
          # one that can actually run, which a clone cannot: it refuses to
          # write over an existing destination.
          echo "::error::run refused: CARGO_TARGET_DIR ${TARGET_DIR} carries workspace-member build artifacts (it is WARM) but no seed-clone marker (.jammi-clone-of-seed) — it predates the marker scheme or was built outside the \`target\` verb, so its provenance is unverified. This is NOT a cold-build refusal." >&2
          echo "remedy: $(basename "$0") target ${SESSION} ${TREE} --adopt   (re-checks the dir's own content and stamps the marker; nothing is copied or deleted)" >&2
          echo "   or:  set RP_ALLOW_COLD_TARGET=1 to proceed without the marker" >&2
          exit 1
          ;;
        *GPU_DEV_TARGET_STATE=UNMARKED_COLD*)
          echo "::error::run refused: CARGO_TARGET_DIR ${TARGET_DIR} exists but contains no workspace-member build artifacts and no seed-clone marker (.jammi-clone-of-seed) — it is a COLD target dir, so this job would pay a full workspace build (~20-40min) instead of the seeded clone's ~80s incremental build." >&2
          echo "remedy: $(basename "$0") target ${SESSION} ${TREE} --with-cutlass   (or set RP_ALLOW_COLD_TARGET=1 to proceed cold deliberately)" >&2
          exit 1
          ;;
        *)
          echo "::error::run preflight could not determine ${TARGET_DIR}'s provisioning state (got: '${TARGET_PREFLIGHT_STATE}') — refusing to guess; set RP_ALLOW_COLD_TARGET=1 to bypass" >&2
          exit 1
          ;;
      esac
    fi

    # esc-077-class (one-pod-per-wave, WAVE-scoped): refuse a job when this
    # pod already has a LIVE job for a DIFFERENT WAVE — an operator kept
    # re-learning this norm from prose alone (the same class esc-077 fixed
    # for cold builds). Scoped to WAVE, not tree (operator-directed
    # refinement): a single wave spanning two trees (e.g. a CPU-build
    # sub-unit and a GPU-test sub-unit sharing the warm seed) is the
    # SANCTIONED shape and must proceed; two DIFFERENT waves' builds/tests
    # competing for the same pod's CPU/disk/nvcc still produce meaningless
    # timings at best and can corrupt a shared CARGO_TARGET_DIR at worst.
    # `jammi-seed` (the boot-time seed build) and this tree's OWN prior
    # session are excluded, and a live OTHER session with no readable
    # active-wave claim fails CLOSED (unknown wave) — see
    # rp_concurrency_preflight_lines's own doc for the full state table.
    # RP_ALLOW_CONCURRENT=1 is the SOLE, explicit CROSS-WAVE bypass, for
    # deliberate co-tenancy (e.g. a build-only job that genuinely doesn't
    # compete for the GPU).
    if [ "${RP_ALLOW_CONCURRENT:-0}" != "1" ]; then
      CONCURRENCY_PREFLIGHT_STATE="$(rp_run_remote <<EOF
set -uo pipefail
$(rp_concurrency_preflight_lines "$TMUX_SESSION" "$WAVE")
EOF
)"
      case "$CONCURRENCY_PREFLIGHT_STATE" in
        *GPU_DEV_CONCURRENCY_STATE=CLEAR*) : ;;
        *GPU_DEV_CONCURRENCY_STATE=BUSY:*)
          # `GPU_DEV_CONCURRENCY_STATE=BUSY:<owning-wave-or-UNKNOWN>:<other-session>`
          BUSY_LINE="$(printf '%s\n' "$CONCURRENCY_PREFLIGHT_STATE" | grep -o 'GPU_DEV_CONCURRENCY_STATE=BUSY:[^[:space:]]*' | head -1)"
          BUSY_REST="${BUSY_LINE#GPU_DEV_CONCURRENCY_STATE=BUSY:}"
          BUSY_WAVE="${BUSY_REST%%:*}"
          BUSY_SESSION="${BUSY_REST#*:}"
          if [ "$BUSY_WAVE" = "UNKNOWN" ]; then
            echo "::error::run refused: this pod already has a live job (tmux session ${BUSY_SESSION}) but its wave identity could not be determined (no readable claim in /root/.jammi-active-wave.d) — one-pod-per-wave fails CLOSED on an unknown wave." >&2
          else
            echo "::error::run refused: this pod already has a live job for wave '${BUSY_WAVE}' (tmux session ${BUSY_SESSION}) — one-pod-per-wave, wave-scoped. Pass --wave ${BUSY_WAVE} (or RP_WAVE=${BUSY_WAVE}) if this job is really part of that same wave." >&2
          fi
          echo "remedy: rent another pod: RP_SESSION=<alias> $(basename "$0") up <arch>   (or set RP_ALLOW_CONCURRENT=1 to co-tenant across waves deliberately, e.g. a build-only job)" >&2
          exit 1
          ;;
        *)
          echo "::error::run preflight could not determine the pod's job-concurrency state (got: '${CONCURRENCY_PREFLIGHT_STATE}') — refusing to guess; set RP_ALLOW_CONCURRENT=1 to bypass" >&2
          exit 1
          ;;
      esac
    fi

    JOB="$*"
    # A per-run token, generated HERE (locally, before anything is sent to
    # the pod) — carried into the completion marker purely for a human
    # reading it directly; wait-job's own correctness never depends on
    # matching it (see rp_job_wrapper_with_marker_lines's own doc for why
    # the remove-then-rewrite-under-session-liveness discipline already
    # suffices). `$$` + `$RANDOM` alongside the timestamp keeps two `run`s
    # issued in the same wall-clock second from sharing a token.
    RUN_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
    # Per-tree job script/log (round-4/5 finding: a global /root/job.sh +
    # /root/jammi.log meant `run --tree b` clobbered tree a's still-running
    # job) — `=`-anchored tmux target (M6), `remain-on-exit off` so a job
    # that exits leaves its output visible in `logs`/`attach` rather than the
    # pane vanishing before either can read it.
    #
    # `--timing`'s own flock acquisition now happens INSIDE the generated
    # `.jammi-job.sh` (rp_job_wrapper_with_marker_lines, runpod_lib.sh) —
    # still essentially the first thing the detached pane's own script does
    # (M6's own reasoning: never in the short-lived ssh invocation that
    # starts tmux and returns immediately, which would release the lock the
    # instant IT exits, not when the real job does) — rather than split
    # across an outer `flock -n -E 75 ... bash job.sh` command line, so a
    # lock refusal and the job's own real exit code can never collide on the
    # same literal value (round-N audit finding B3). LAUNCH is now the SAME
    # shape either way: redirection (`>`), not a `| tee` pipe, for BOTH —
    # the wrapper's own marker write is what wait-job reads now, so there is
    # no need to preserve `flock`'s outer exit code through the pane's own
    # command anymore either.
    if [ "$TIMING" = "1" ]; then
      LAUNCH="bash '${TREE_DIR}'/.jammi-job.sh > '${TREE_DIR}'/.jammi.log 2>&1"
    else
      LAUNCH="bash '${TREE_DIR}'/.jammi-job.sh 2>&1 | tee '${TREE_DIR}'/.jammi.log"
    fi
    rp_run_remote <<EOF
set -uo pipefail
cat > '${TREE_DIR}'/.jammi-job.sh <<'JOBEOF'
$(rp_job_wrapper_with_marker_lines "$TREE_DIR" "$TARGET_DIR" "$JOB" "$RUN_TOKEN" "$TIMING" "$WAVE" "$TREE")
JOBEOF
tmux kill-session -t "=${TMUX_SESSION}" 2>/dev/null # tripwire-ok: idempotent best-effort cleanup of a session that may legitimately not exist yet (first \`run\` for this tree/session) — the new-session command right below is unconditional; the backticks around \`run\` are ESCAPED because this heredoc's delimiter is UNQUOTED (it must expand \${TMUX_SESSION}/\${LAUNCH}) — an unescaped backtick pair in an unquoted heredoc body is LIVE command substitution, run on the LAPTOP, which printed 'run: command not found' and spliced the word away before the text ever reached the pod
tmux new-session -d -s "${TMUX_SESSION}" "${LAUNCH}"
tmux set-option -w -t "=${TMUX_SESSION}:" remain-on-exit off
echo "started (tree=${TREE}, timing-locked=${TIMING}, token=${RUN_TOKEN}): ${JOB}"
EOF
    echo "=== detached. follow with: $(basename "$0") logs ${SESSION} --tree ${TREE}"
    echo "=== or block for it: $(basename "$0") wait-job ${SESSION} --tree ${TREE} ==="
    ;;

  logs)
    require_pod; rp_keep
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "tail -f -n 200 '${TREE_DIR}/.jammi.log'" || true # tripwire-ok: `logs` is a live-follow (`tail -f`) the user Ctrl-C's out of intentionally — a non-zero exit here is the NORMAL way this command ends, not a failure to surface
    ;;

  wait-seed)
    # Blocks until the pod's own build-substrate seed reports a verdict — the
    # fail-open-watcher lesson: an operator's hand-rolled poll loop silently
    # idled forever whenever SSH dropped, because "no evidence yet" and
    # "could not check" read as the SAME thing to it. rp_wait_poll (see its
    # own doc in runpod_lib.sh) makes them different signals; this script's
    # only job is to name the three states the seed itself can be in.
    require_pod; rp_keep
    SEED_WAIT_SCRIPT="$(rp_seed_wait_script)"
    rp_wait_poll "seed(${SESSION})" "$SEED_WAIT_SCRIPT" "$WAIT_INTERVAL_S" "$WAIT_TIMEOUT_S" "$WAIT_MAX_FAIL"
    exit $?
    ;;

  wait-job)
    # Blocks until <tree>'s detached `run` job ends, reading the per-run
    # completion marker `run` itself writes (rp_job_wrapper_with_marker_lines,
    # runpod_lib.sh) — succeeding only on a real rc=0, naming a real job
    # failure and a refused-timing-lock (rc=75) distinctly, and reporting
    # "no evidence this job ever ran" when neither a live session nor a
    # marker exists (round-N audit finding B3 — see rp_job_wait_script's
    # own doc for why a bare "does a log file exist" check was not enough).
    require_pod; rp_keep
    JOB_WAIT_SCRIPT="$(rp_job_wait_script "$TREE_DIR" "$TMUX_SESSION" "$TREE")"
    rp_wait_poll "job(${SESSION}:${TREE})" "$JOB_WAIT_SCRIPT" "$WAIT_INTERVAL_S" "$WAIT_TIMEOUT_S" "$WAIT_MAX_FAIL"
    exit $?
    ;;

  push)
    require_pod; rp_keep
    # esc-056: rsync creates only the LAST path component of its own
    # destination — never a missing PARENT chain. On a fresh pod the very
    # first `push --tree <name>` for a name that has never been pushed
    # before failed outright ("mkdir ... No such file or directory") since
    # nothing else on the pod provisions /root/trees itself. A bounded,
    # idempotent remote `mkdir -p` on the tree's parent (rp_push_ensure_parent,
    # runpod_lib.sh — the SAME rp_run_remote primitive every other
    # pod-reaching verb here uses) runs BEFORE the rsync below, on every
    # push, not just a tree's first.
    rp_push_ensure_parent "$TREE_DIR" \
      || { echo "::error::could not provision ${TREE_DIR%/*} on the pod before pushing"; exit 1; }
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
    # round-6 audit item D (a real, class-shaped failure: "the scripts
    # assume a git state of the tree that a pushed/provisioned tree does
    # not have"): `-a` (archive) includes `-o -g` (preserve owner/group)
    # — meaningful only when the RECEIVING process can chown, which this
    # one can, since it connects as root@pod. Without `--no-owner
    # --no-group`, the pushed tree's files land OWNED BY THE LAPTOP
    # USER's uid (e.g. 501 for a typical macOS account), copied verbatim
    # from the local files' own metadata — NOT root, even though rsync
    # itself runs as root on the receiving end. `git`, run later inside
    # that SAME tree as root, then refuses with "fatal: detected dubious
    # ownership" (euid root != file owner uid 501) on both the tree and
    # its submodule. `--no-owner --no-group` makes the receiver leave
    # ownership at its own create-time default (root:root, since the
    # connection itself is root@pod), never copying the laptop's uid.
    rsync -azc --no-times --no-owner --no-group --delete "${EXCLUDE_ARGS[@]}" \
      -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "${REPO_ROOT}/" "root@${RP_HOST}:${TREE_DIR}/" \
      && echo "=== pushed $(basename "$REPO_ROOT") → pod (tree: ${TREE}) ==="
    rc=$?
    if [ "$rc" -eq 0 ]; then
      STAMP="$(mktemp)"
      "$DIR/pod_push_stamp.sh" compute "$REPO_ROOT" "$SESSION" > "$STAMP" \
        && rsync -az --no-owner --no-group -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
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
    # Deployment-gap fix (numerics r2 wave finding, folded into esc-077):
    # this verb used to run `/root/jammi-ai/ci/scripts/pod_target_clone.sh`
    # — the POD's own bootstrapped checkout, baked at boot time from
    # whatever `main` was THEN, which can predate (or simply differ from)
    # THIS laptop checkout by any number of commits. The esc-077 run-
    # preflight (rp_target_preflight_lines, THIS checkout's own copy,
    # always laptop-side) checks for a marker ONLY the matching version of
    # pod_target_clone.sh knows to stamp — a pod booted before esc-077
    # landed would clone successfully but stamp NOTHING, and the very next
    # `run` would then refuse a perfectly legitimate clone. Fixed by
    # STAGING this checkout's OWN copies of the pod-side scripts `target`
    # depends on (never executing the pod's bootstrapped copies for this
    # verb) — version consistency BY CONSTRUCTION, not by hoping the pod
    # tree happens to be fresh, and immune to the SAME class of drift no
    # matter which future commit adds the next pod-side behavior a local
    # preflight needs to agree with.
    #
    # Four files, not just pod_target_clone.sh itself: it `.`-sources
    # pod_seed_target.sh from ITS OWN directory (self-location via
    # `${BASH_SOURCE[0]}`), and pod_provision_cutlass.sh (this verb's
    # `--with-cutlass` arm) invokes pod_push_stamp.sh as a subprocess the
    # SAME way — piping either script over a bare `ssh ... bash -s` stdin
    # would break that self-location entirely (BASH_SOURCE[0] resolves to
    # something with no real sibling directory). Staging all four into ONE
    # fixed pod directory preserves both self-location relationships
    # exactly as if this checkout's own ci/scripts/ had been pushed.
    # `/root/.jammi-caller-scripts` is a single new path component directly
    # under `/root` (which always exists on a booted pod), so rsync creates
    # it unaided — no `rp_push_ensure_parent` needed (contrast `push`'s own
    # `/root/trees/<name>`, two levels deep).
    STAGE_DIR="/root/.jammi-caller-scripts"
    rsync -az --no-owner --no-group -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "$DIR/pod_target_clone.sh" "$DIR/pod_seed_target.sh" \
      "$DIR/pod_provision_cutlass.sh" "$DIR/pod_push_stamp.sh" \
      "root@${RP_HOST}:${STAGE_DIR}/" \
      || { echo "::error::target: failed to stage this checkout's own pod-side scripts to ${STAGE_DIR} on the pod"; exit 1; }
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
        "bash ${STAGE_DIR}/pod_target_clone.sh '' '${NAME_TARGET_DIR}' --verify"
      exit $?
    fi
    [ -n "$TARGET_NAME" ] || { echo "target: need a tree name"; exit 2; }
    # `--adopt` stamps the clone marker on an already-warm target dir that
    # carries none, after re-checking its content (pod_target_clone.sh's
    # own --adopt doc has the rule) — the executable remedy for `run`'s
    # UNMARKED_WARM refusal. Nothing is cloned, copied, or deleted; the
    # ONLY difference here is the flag, so both paths keep the identical
    # staging, cutlass, and reporting behavior.
    if [ "$TARGET_ADOPT" = "1" ]; then
      rp_run_remote <<EOF
set -uo pipefail
bash ${STAGE_DIR}/pod_target_clone.sh /root/.jammi-seed '${NAME_TARGET_DIR}' --adopt
EOF
    else
      rp_run_remote <<EOF
set -uo pipefail
bash ${STAGE_DIR}/pod_target_clone.sh /root/.jammi-seed '${NAME_TARGET_DIR}'
EOF
    fi
    rc=$?
    if [ "$rc" -eq 0 ] && [ "$TARGET_WITH_CUTLASS" = "1" ]; then
      # round-5 audit A1: this logic used to be inlined as heredoc TEXT
      # right here — the only coverage for it was two `grep`s against that
      # text (a proxy never run against a real instance; shellcheck cannot
      # parse a heredoc body either) — and that shape is exactly what let
      # a `set -e`-vs-bare-command regression (the mismatch-remediation
      # arm silently becoming dead code) ship undetected. Extracted into
      # ci/scripts/pod_provision_cutlass.sh, a real file this checkout's
      # own hermetic suite sources and runs against a genuine two-commit
      # submodule fixture — see that file's own module doc for the
      # mechanism (source of truth: the tree's own push stamp;
      # provisioning: `cp -a` from /root/jammi-ai's initialised submodule,
      # never `git submodule` inside the tree). `/root/jammi-ai` in the
      # ARGUMENT below (super-dir, the git checkout whose own initialised
      # cutlass submodule gets copied FROM) is unrelated to the STAGE_DIR
      # fix above — it names the real submodule content's location, never
      # a script version.
      rp_run_remote <<EOF
set -uo pipefail
bash ${STAGE_DIR}/pod_provision_cutlass.sh '${NAME_SOURCE_TREE_DIR}' /root/jammi-ai
EOF
      rc=$?
    fi
    if [ "$TARGET_ADOPT" = "1" ]; then
      echo "=== target '${TARGET_NAME}' — ADOPTED existing dir ${NAME_TARGET_DIR}, cutlass ${TARGET_WITH_CUTLASS}: exit ${rc} ==="
    else
      echo "=== target '${TARGET_NAME}' — clone at ${NAME_TARGET_DIR}, cutlass ${TARGET_WITH_CUTLASS}: exit ${rc} ==="
    fi
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
