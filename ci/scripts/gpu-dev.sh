#!/usr/bin/env bash
# GPU development on RunPod — one CLI over the shared pod primitive.
#
# Two lifetimes, one substrate:
#   * `shell` is a throwaway debug pod — it dies when you exit (the CI default).
#   * `up` starts a named session whose pod SURVIVES disconnect, so a fine-tune,
#     eval or bench keeps running after you close the terminal. Such a pod is
#     always armed with a TTL reaper; nothing else would ever stop the meter.
#
# The pod itself is disposable in both cases (volumeInGb 0). What persists lives
# in an S3-compatible object store — see `docs/maintainer/dev-gpu.md`.
#
# Usage:
#   gpu-dev.sh shell   [arch]              throwaway shell; pod dies on exit
#   gpu-dev.sh up      [arch]              start a surviving session (name = arch)
#   gpu-dev.sh attach  [session]           shell into a surviving session
#   gpu-dev.sh run     [session] <cmd...>  run <cmd> detached under tmux
#   gpu-dev.sh logs    [session]           tail the detached job's output
#   gpu-dev.sh push    [session]           rsync your working tree TO the pod
#   gpu-dev.sh pull    [session] <path>    rsync <path> back FROM the pod
#   gpu-dev.sh down    [session]           terminate the pod, forget the session
#   gpu-dev.sh ls                          list sessions
#   gpu-dev.sh prewarm [arch]              publish the cargo-registry prewarm
#
# arch: a100 (default) | l40s | h100 | a40 | l4
# Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE, RP_TTL_HOURS (8).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "${HOME}/.config/runpod/key" 2>/dev/null || true)}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY or write it to ~/.config/runpod/key}"

usage() {
  cat <<'USAGE'
gpu-dev.sh — GPU development on RunPod

  shell   [arch]              throwaway shell; the pod dies when you exit
  up      [arch]              start a session whose pod SURVIVES disconnect
  attach  [session]           shell into a surviving session
  run     [session] <cmd...>  run <cmd> detached under tmux
  logs    [session]           tail the detached job's output
  push    [session]           rsync your working tree TO the pod
  pull    [session] <path>    rsync <path> back FROM the pod
  down    [session]           terminate the pod, forget the session
  ls                          list sessions
  reap    [hours]             terminate orphaned pods past their own deadline
                              ([hours] force-reaps everything older than that)
  prewarm [arch]              publish the cargo-registry prewarm object

arch: a100 (default) | l40s | h100 | a40 | l4
Sessions are named after the arch; RP_SESSION overrides.
Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE, RP_TTL_HOURS (default 8).
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
    printf '%-16s %-20s %s\n' SESSION POD ARCH@HOST
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

ARG="${1:-}"; [ $# -gt 0 ] && shift

# Sessions are named after the arch, so the common case needs no bookkeeping:
# `up a100` … `attach a100` … `down a100`. RP_SESSION overrides for a second pod
# of the same arch.
case "$CMD" in
  shell|up|prewarm) ARCH="${ARG:-a100}"; SESSION="${RP_SESSION:-$ARCH}" ;;
  *)                SESSION="${RP_SESSION:-${ARG:-a100}}"; ARCH="" ;;
esac

# `shell` and `prewarm` are throwaway: no named session, so the EXIT trap wipes
# the temp dir and terminates the pod.
case "$CMD" in
  shell|prewarm) : ;;
  *) export RP_SESSION="$SESSION" ;;
esac

# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# Interactive remote command: correct env, then a shell in the checkout.
rp_login_cmd() {
  echo '[ -f /root/.jammi_env ] && . /root/.jammi_env; cd /root/jammi-ai 2>/dev/null; exec bash -i'
}

require_pod() {
  rp_session_load && rp_session_alive && return 0
  echo "no live session '${SESSION}' — the pod may have been reaped."
  echo "start one with: $(basename "$0") up ${SESSION}"
  exit 1
}

case "$CMD" in

  shell)
    rp_init
    echo "=== provisioning ${ARCH} (image: ${RP_IMAGE}) ==="
    rp_deploy_arch "$ARCH" || exit $?
    echo "=== bootstrapping ==="
    rp_bootstrap
    echo "=== pod ${RP_POD_ID} on ${RP_HOST}:${RP_PORT} — it TERMINATES when you exit ==="
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd)" || true
    echo "=== shell closed — terminating pod (trap) ==="
    ;;

  up)
    if rp_session_load && rp_session_alive; then
      echo "session '${SESSION}' already live on ${RP_HOST}:${RP_PORT} (pod ${RP_POD_ID})"
      echo "attach with: $(basename "$0") attach ${SESSION}"
      rp_keep; exit 0
    fi
    # A recorded-but-dead pod (reaped, or a host that died) must not be carried
    # into the deploy — the EXIT trap would act on a stale id.
    RP_POD_ID=""
    rp_init
    echo "=== provisioning ${ARCH} (image: ${RP_IMAGE}) ==="
    rp_deploy_arch "$ARCH" || exit $?
    # No separate arming step: the ${RP_TTL_HOURS}h deadline is baked into the
    # pod's entrypoint at deploy, so it is already running.
    echo "=== bootstrapping ==="
    rp_bootstrap || { echo "::error::bootstrap failed — terminating pod"; exit 1; }
    rp_keep
    echo
    echo "=== session '${SESSION}' up on ${RP_HOST}:${RP_PORT} (pod ${RP_POD_ID}) ==="
    echo "    attach:  $(basename "$0") attach ${SESSION}"
    echo "    run job: $(basename "$0") run ${SESSION} cargo test -p jammi-ai --features cuda,live-gpu-tests"
    echo "    STOP:    $(basename "$0") down ${SESSION}      # else the reaper does it in ${RP_TTL_HOURS}h"
    ;;

  attach)
    require_pod; rp_keep
    ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" "$(rp_login_cmd)" || true
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
    ;;

  prewarm)
    rp_s3_load || { echo "::error::no object store configured at ${RP_S3_CONF} — see docs/maintainer/dev-gpu.md"; exit 1; }
    rp_init
    echo "=== provisioning ${ARCH} to build the prewarm ==="
    rp_deploy_arch "$ARCH" || exit $?
    rp_bootstrap
    rp_prewarm_publish
    echo "=== done — terminating pod (trap) ==="
    ;;

  *) echo "unknown command: $CMD"; usage 2 ;;
esac
