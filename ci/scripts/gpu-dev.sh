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
#   gpu-dev.sh shell   [arch] [--ref R]    throwaway shell; pod dies on exit
#   gpu-dev.sh up      [arch] [--ref R]    start a surviving session (name = arch)
#   gpu-dev.sh attach  [session]           shell into a surviving session
#   gpu-dev.sh run     [session] <cmd...>  run <cmd> detached under tmux
#   gpu-dev.sh logs    [session]           tail the detached job's output
#   gpu-dev.sh push    [session]           rsync your working tree TO the pod
#   gpu-dev.sh pull    [session] <path>    rsync <path> back FROM the pod
#   gpu-dev.sh down    [session]           terminate the pod, forget the session
#   gpu-dev.sh ls                          list sessions
#
# arch: a100 (default) | l40s | h100 | a40 | l4
# --ref: branch, tag or commit the pod's checkout is placed on (default main).
#        Verified against the remote BEFORE a pod is rented.
# Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE, RP_TTL_HOURS (8),
#      RP_DISK_GB (60), RP_VOLUME_GB (0). Disk sizing rule of thumb: roughly
#      25 GB base + 3 GB per concurrent agent target dir + 2 GB per
#      `cargo mutants` job — a mutation-testing session wants >= 120 GB.
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

arch: a100 (default) | l40s | h100 | a40 | l4
--ref R: branch, tag or commit for the pod's checkout (default main), checked
         against the remote before anything is rented. `up` does not move a live
         pod: `down` it first.
Sessions are named after the arch; RP_SESSION overrides.
Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE, RP_TTL_HOURS (default 8),
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
    if rp_session_load && rp_session_alive; then
      # rp_keep FIRST: this pod belongs to a session that is already running, and
      # every exit below must leave it alone.
      rp_keep
      echo "session '${SESSION}' already live on ${RP_HOST}:${RP_PORT} (pod ${RP_POD_ID}, ref ${RP_REF:-<none>})"
      # `up` never moves a live pod. Reporting success on an ignored --ref would
      # leave you reading results from one ref while believing they came from
      # another — the exact failure this flag exists to remove.
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
    fi
    rp_ref_precheck "$REF" || exit 2
    # A recorded-but-dead pod (reaped, or a host that died) must not be carried
    # into the deploy — the EXIT trap would act on a stale id.
    RP_POD_ID=""
    rp_init
    echo "=== provisioning ${ARCH} on ${REF} (image: ${RP_IMAGE}) ==="
    rp_deploy_arch "$ARCH" || exit $?
    # No separate arming step: the ${RP_TTL_HOURS}h deadline is baked into the
    # pod's entrypoint at deploy, so it is already running.
    echo "=== bootstrapping ==="
    bootstrap_or_die
    rp_keep
    rp_ssh_config_sync
    echo
    echo "=== session '${SESSION}' up on ${RP_HOST}:${RP_PORT} @ ${RP_REF:-<none>} (pod ${RP_POD_ID}) ==="
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
      rp_terminate "$RP_POD_ID"
      echo "=== terminated pod ${RP_POD_ID} (session '${SESSION}') ==="
    else
      echo "no recorded pod for session '${SESSION}'"
    fi
    RP_POD_ID=""   # already gone; keep the EXIT trap from double-terminating
    rp_session_forget
    # After the session file is gone, so the dead host stops being offered.
    rp_ssh_config_sync
    ;;

  *) echo "unknown command: $CMD"; usage 2 ;;
esac
