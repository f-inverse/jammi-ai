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
# The pod itself is disposable (volumeInGb 0). The only thing that persists is
# the sccache compile cache in an S3-compatible object store — the cargo registry
# is deliberately NOT cached; fetching it measures 9s on a RunPod host.
# See `docs/maintainer/dev-gpu.md`.
#
# Usage:
#   gpu-dev.sh shell   [arch] [--ref R]     throwaway shell; pod dies on exit
#   gpu-dev.sh up      [arch] [--ref R]     start a surviving session (name = arch)
#                      [--replace]          replace an alias's existing record
#   gpu-dev.sh attach  [session]            shell into a surviving session
#   gpu-dev.sh run     [session] <cmd...>   run <cmd> detached under tmux
#   gpu-dev.sh logs    [session]            tail the detached job's output
#   gpu-dev.sh push    [session]            rsync your working tree TO the pod
#   gpu-dev.sh pull    [session] <path>     rsync <path> back FROM the pod
#   gpu-dev.sh down    [session]            terminate the pod, forget the session
#   gpu-dev.sh ls                           list sessions
#
# arch: a100 (default) | l40s | h100 | a40 | l4
# --ref: branch, tag or commit the pod's checkout is placed on (default main).
#        Verified against the remote BEFORE a pod is rented.
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

  shell   [arch] [--ref R]    throwaway shell; the pod dies when you exit
  up      [arch] [--ref R]    start a session whose pod SURVIVES disconnect
          [--replace]         (default TTL 72h — see RP_DEV_TTL_HOURS below)
                              refuses if the alias already has a recorded pod
                              unless --replace is given (see --replace below)
  attach  [session]           join a surviving session's running job
                              (--shell for a plain prompt instead)
  run     [session] <cmd...>  run <cmd> detached under tmux
  logs    [session]           tail the detached job's output
  push    [session]           rsync your working tree TO the pod
  pull    [session] <path>    rsync <path> back FROM the pod
  down    [session]           terminate the pod, forget the session
  ls                          list sessions
  reap    [hours]             terminate orphaned pods past their own deadline
                              ([hours] force-reaps everything older than that)

A running measurement is protected only by its own TTL — there is no verb to
pause the sweep for a single pod (RunPod's pod-edit API has no rename/name
field; see the module header comment). Rent with RP_TTL_HOURS/RP_DEV_TTL_HOURS
set to at least the job's expected length.

arch: a100 (default) | l40s | h100 | a40 | l4
--ref R: branch, tag or commit for the pod's checkout (default main), checked
         against the remote before anything is rented. `up` does not move a live
         pod: `down` it first.
--replace: `up` normally REFUSES (exit 2) a session alias that already has a
         recorded pod id, even an unreachable one, rather than silently
         deploying a second pod under the same alias. --replace overwrites only
         the LOCAL record; it never terminates the old pod itself — `down` it
         first if it should be.
Sessions are named after the arch; RP_SESSION overrides.
Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,
     RP_TTL_HOURS (default 8; explicit always wins), RP_DEV_TTL_HOURS (default
     72 — what `up` alone falls back to when RP_TTL_HOURS is not set),
     RP_DISK_GB (default 60), RP_VOLUME_GB (default 0).
     Disk sizing rule of thumb: roughly 25 GB base + 3 GB per concurrent agent
     target dir + 2 GB per `cargo mutants` job — a mutation-testing session
     wants >= 120 GB (RP_DISK_GB=150).
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
        # `--replace` only means something for `up` (it overwrites a session's
        # LOCAL record — see the `up` case below); `shell` has no session to
        # overwrite, so accepting it silently there would look like it did
        # something. Reject it explicitly instead of ignoring it.
        --replace)
          [ "$CMD" = "up" ] || { echo "::error::--replace applies only to 'up' ($CMD has no session to replace)"; exit 2; }
          REPLACE=1; shift ;;
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
  *)
    ARG="${1:-}"; [ $# -gt 0 ] && shift
    SESSION="${RP_SESSION:-${ARG:-a100}}"; ARCH=""
    ;;
esac

# `shell` is throwaway: no named session, so the EXIT trap wipes
# the temp dir and terminates the pod.
case "$CMD" in
  shell) : ;;
  *) export RP_SESSION="$SESSION" ;;
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
  RP_TTL_HOURS="${RP_DEV_TTL_HOURS:-72}"
  export RP_TTL_HOURS
fi

# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# The ref rules live with the code that sends the ref to the pod, so they are
# applied here as soon as that code is available — before anything is rented.
case "$CMD" in
  shell|up) rp_ref_check "$REF" || exit 2 ;;
esac

# Interactive remote command: correct env, then either the running job's terminal
# or a plain shell in the checkout. Pass "job" to prefer the job when one exists.
rp_login_cmd() { # $1 = "job" to join a live tmux job
  local pre='[ -f /root/.jammi_env ] && . /root/.jammi_env; cd /root/jammi-ai 2>/dev/null;'
  if [ "${1:-}" = "job" ]; then
    # Ctrl-C inside the job's pane signals the job itself, so say so before
    # handing over the keyboard — this is a terminal that can destroy work.
    printf '%s %s' "$pre" 'if tmux has-session -t jammi 2>/dev/null; then
      echo "=== joining the running job. Ctrl-B then D detaches. Ctrl-C KILLS the job. ===";
      exec tmux attach -t jammi; fi; exec bash -i'
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
    echo "=== pod ${RP_POD_ID} on ${RP_HOST}:${RP_PORT} @ ${RP_REF:-<none>} — it TERMINATES when you exit ==="
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd)" || true
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
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd "$MODE")" || true
    echo "=== detached; pod ${RP_POD_ID} still running (down with: $(basename "$0") down ${SESSION}) ==="
    ;;

  run)
    [ $# -gt 0 ] || { echo "run: need a command"; exit 2; }
    require_pod; rp_keep
    JOB="$*"
    rp_run_remote <<EOF
set -uo pipefail
cat > /root/job.sh <<'JOBEOF'
[ -f /root/.jammi_env ] && . /root/.jammi_env
cd /root/jammi-ai
${JOB}
JOBEOF
tmux kill-session -t jammi 2>/dev/null
tmux new-session -d -s jammi "bash /root/job.sh 2>&1 | tee /root/jammi.log"
echo "started: ${JOB}"
EOF
    echo "=== detached. follow with: $(basename "$0") logs ${SESSION} ==="
    ;;

  logs)
    require_pod; rp_keep
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "tail -f -n 200 /root/jammi.log" || true
    ;;

  push)
    require_pod; rp_keep
    # Ship the working tree, including uncommitted work, but never the local
    # build output — target/ is host-arch and would poison the pod's.
    rsync -az --delete --exclude '.git' --exclude 'target' --exclude '.venv' \
      -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "${REPO_ROOT}/" "root@${RP_HOST}:/root/jammi-ai/" \
      && echo "=== pushed $(basename "$REPO_ROOT") → pod ==="
    ;;

  pull)
    require_pod; rp_keep
    REMOTE="${1:-}"
    [ -n "$REMOTE" ] || { echo "pull: need a remote path (e.g. pull ${SESSION} target/nextest)"; exit 2; }
    mkdir -p "${REPO_ROOT}/.gpu-pull"
    rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
      "root@${RP_HOST}:/root/jammi-ai/${REMOTE}" "${REPO_ROOT}/.gpu-pull/" \
      && echo "=== pulled ${REMOTE} → .gpu-pull/ ==="
    ;;

  down)
    if rp_session_load; then
      # Never trust the locally-recorded id on its own: confirm it is BOTH
      # still present in the account AND still carries this session's OWN
      # name (its "<prefix>-ttl<H>") before terminating anything. This is
      # what stops `down` from ending a pod that a race with another `up` on
      # the same alias silently swapped in underneath this session's record
      # (2026-08-25 incident) — a mismatch refuses rather than acts.
      if rp_pod_verify "$RP_POD_ID" "${RP_TTL_HOURS:-8}" >/dev/null; then
        rp_terminate "$RP_POD_ID"
        echo "=== terminated pod ${RP_POD_ID} (session '${SESSION}') ==="
        RP_POD_ID=""   # already gone; keep the EXIT trap from acting on it again
        rp_session_forget
        # After the session file is gone, so the dead host stops being offered.
        rp_ssh_config_sync
      else
        # The LOCAL record is deliberately KEPT here, not forgotten: this is
        # exactly the ambiguous case (a mismatched or ghost id) where a
        # follow-up `up` on this alias most needs to still see a recorded
        # pod and refuse (or ask for --replace) rather than silently
        # deploying a THIRD pod on top of an already-confused alias.
        echo "::error::refusing to terminate pod ${RP_POD_ID} for session '${SESSION}' — it did not verify against the account's live pod list (see above)."
        echo "::error::the local session record is KEPT (not forgotten), so this alias still refuses a plain 'up' rather than deploying on top of the ambiguity."
        echo "::error::inspect it: $(basename "$0") ls   /   force-reap by hand: $(basename "$0") reap <hours>   /   the RunPod console"
        exit 1
      fi
    else
      echo "no recorded pod for session '${SESSION}'"
    fi
    ;;

  *) echo "unknown command: $CMD"; usage 2 ;;
esac
