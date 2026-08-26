#!/usr/bin/env bash
# Shared RunPod GPU primitive — one seam for every caller.
#
# A pod is described by two independent axes:
#
#   * lifetime — terminate-on-exit (the default) or survive-exit (RP_KEEP=1).
#                A surviving pod is what lets a fine-tune / eval / bench outlive
#                the SSH session.
#
# EVERY pod — CI, throwaway, or surviving — carries an RP_TTL_HOURS deadline
# baked into its entrypoint at deploy time, and the deadline is repeated in the
# pod's name so any sweeper can honour it. The EXIT trap is best-effort only: a
# SIGKILLed process (a cancelled GitHub run, a dropped laptop) never runs it, and
# a pod with no other deadline then bills until the account empties.
#
# The in-pod deadline self-terminates with `runpodctl remove pod` and is verified
# on hardware. It still needs the network at deadline time, so rp_sweep is
# load-bearing rather than belt-and-braces.
#   * state    — the pod is always disposable (volumeInGb 0). Durable state lives
#                in an S3-compatible object store that rp_bootstrap rehydrates.
#                A RunPod network volume is deliberately NEVER attached: an
#                attached volume is Secure-Cloud-only and pinned to a single
#                datacenter, which would delete both failover dimensions below —
#                and intermittent A100 supply is precisely why they exist.
#
# Deploys a live GPU pod with two failover dimensions (capacity + liveness), runs
# a remote job on it importing the container's real ENV, and terminates the pod
# on exit unless asked to keep it. Sourced by:
#   * runpod_gpu_prove.sh — build-from-source + run the gated GPU suites (CI)
#   * gpu-dev.sh          — the interactive / long-running dev CLI
#
# Requires env: RUNPOD_API_KEY.
# Optional env:
#   RP_IMAGE      pod image.
#   RP_TIMEOUT    seconds allowed for ONE rp_run_remote invocation (default
#                 3000). It bounds the SSH invocation, not whatever that
#                 invocation leaves behind: gpu-dev.sh's `run` uses it only to
#                 launch a detached tmux session, which daemonizes and outlives
#                 the timeout entirely. It is NOT a cost guard — RP_TTL_HOURS and
#                 rp_sweep are.
#   RP_SESSION    named session. Persists pod coordinates AND the SSH key under
#                 RP_SESSION_ROOT so a *different* terminal can reattach. Unset =
#                 throwaway temp dir, wiped on exit.
#   RP_KEEP       1 = leave the pod running when this process exits.
#   RP_TTL_HOURS  hard deadline baked into every pod at deploy (default 8 in
#                 this file). gpu-dev.sh's own `up` raises the default to
#                 RP_DEV_TTL_HOURS (72h) before sourcing this file, since a
#                 throwaway-pod default is too short for a dev session someone
#                 is actively using — an explicit RP_TTL_HOURS always wins.
#   RP_DISK_GB    container disk size in GB (default 60). Rule of thumb: roughly
#                 25 GB base + 3 GB per concurrent agent CARGO_TARGET_DIR + 2 GB
#                 per `cargo mutants -j N` job (COPY MODE makes one full
#                 workspace+target copy per job — standing clause 1 REQUIRES
#                 COPY MODE, never `--in-place`, so a shared target dir does not
#                 report mutated sources as "Fresh"). A mutation-testing session
#                 wants >= 120 GB.
#   RP_VOLUME_GB  attached volume size in GB (default 0). The pod is deliberately
#                 disposable (see "state" above) — leave this 0 unless a caller
#                 has a specific reason to attach one.
# Sets globals: RP_POD_ID, RP_HOST, RP_PORT, RP_REF. Installs an EXIT trap for
# teardown.

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (GitHub secret)}"
RP_IMAGE="${RP_IMAGE:-ghcr.io/f-inverse/jammi-ai-ci-cuda:latest}"
# Minimum NVIDIA driver major the CUDA build needs: the image ships CUDA 12.6 PTX
# that the deployment driver JIT-compiles at model load, so a pod below r560
# (< CUDA 12.6) cannot run it — the engine's own startup floor (#304) rejects it
# and every model load fails. RunPod's fleet is mixed, so pod selection fails
# over past an under-floor pod rather than deploying onto one that cannot run.
RP_MIN_DRIVER_MAJOR="${RP_MIN_DRIVER_MAJOR:-560}"
RP_SESSION="${RP_SESSION:-}"
RP_KEEP="${RP_KEEP:-0}"
RP_TTL_HOURS="${RP_TTL_HOURS:-8}"
# Every pod this tooling rents is named "<prefix>-ttl<H>". The deadline travels
# with the pod so a sweeper can honour each pod's OWN limit instead of imposing
# its own — otherwise a CI sweep (3h) reaps a developer's 8h session.
RP_POD_PREFIX="jammi-gpu"
# Validated here, not at use: RP_TTL_HOURS goes into arithmetic expansion AND the
# pod name. A non-integer would yield a malformed payload plus an unparseable
# name, and no script here sets -e, so it would fail silently in both places.
case "$RP_TTL_HOURS" in
  ''|*[!0-9]*) echo "::error::RP_TTL_HOURS must be a positive integer (got '${RP_TTL_HOURS}')" >&2; exit 2 ;;
esac
[ "$RP_TTL_HOURS" -gt 0 ] || { echo "::error::RP_TTL_HOURS must be > 0" >&2; exit 2; }
RP_DISK_GB="${RP_DISK_GB:-60}"
RP_VOLUME_GB="${RP_VOLUME_GB:-0}"
# Same validation shape as RP_TTL_HOURS above: both values go straight into a
# GraphQL payload as unquoted JSON numbers (see _rp_deploy_payload), so a
# non-integer here would send garbage to the API instead of failing loudly here.
case "$RP_DISK_GB" in
  ''|*[!0-9]*) echo "::error::RP_DISK_GB must be a positive integer (got '${RP_DISK_GB}')" >&2; exit 2 ;;
esac
[ "$RP_DISK_GB" -gt 0 ] || { echo "::error::RP_DISK_GB must be > 0" >&2; exit 2; }
case "$RP_VOLUME_GB" in
  ''|*[!0-9]*) echo "::error::RP_VOLUME_GB must be a non-negative integer (got '${RP_VOLUME_GB}')" >&2; exit 2 ;;
esac
RP_SESSION_ROOT="${RP_SESSION_ROOT:-${HOME}/.config/runpod/sessions}"
RP_S3_CONF="${RP_S3_CONF:-${HOME}/.config/runpod/s3}"
# An ssh config this tooling owns outright, so ~/.ssh/config is never rewritten.
RP_SSH_CONFIG="${RP_SSH_CONFIG:-${HOME}/.config/runpod/ssh_config}"
RP_REPO_URL="${RP_REPO_URL:-https://github.com/f-inverse/jammi-ai}"

# A named session must outlive the process; an anonymous one must not leak. The
# session dir is created lazily by the first writer, so read-only commands
# against a session that does not exist leave nothing behind.
if [ -n "$RP_SESSION" ]; then
  RP_WORK="${RP_SESSION_ROOT}/${RP_SESSION}"
  RP_WORK_IS_TEMP=0
else
  RP_WORK="$(mktemp -d)"
  RP_WORK_IS_TEMP=1
fi
RP_SSH_KEY="$RP_WORK/id_ed25519"
RP_META="$RP_WORK/meta"
# RP_POD_CREATED is 1 only once THIS invocation's own rp_deploy_live actually
# rented a pod — set the MOMENT a pod id comes back from the deploy mutation,
# not at SSH-up (which can be minutes later): a pod bills, and can be leaked
# by an EXIT trap firing during the reachability wait, from the instant it is
# rented, not from the instant it becomes reachable. rp_session_load — used
# by every read-only subcommand (attach/run/logs/push/pull/down) to recognize
# a pod someone else's invocation already rented — deliberately never sets
# it. rp_cleanup below gates termination-on-failure on this flag, not merely
# on "RP_POD_ID is non-empty": before this flag existed, a read-only
# subcommand against an unreachable session left RP_POD_ID set from the
# loaded session and exited 1, and the EXIT trap terminated a pod that
# invocation never rented — the incident this flag closes.
RP_POD_ID=""; RP_HOST=""; RP_PORT=""; RP_PUBKEY=""; RP_ARCH=""; RP_SSHO=(); RP_POD_CREATED=0
# The git ref the pod's checkout sits on. Two sites keep it honest, so recorded
# state never claims a ref the pod is not on: rp_bootstrap sets it only after the
# checkout succeeded, and rp_deploy_live clears it the moment pod identity
# changes — otherwise a dead session's ref survives into the new pod's record.
RP_REF=""

# An SSH login shell does NOT inherit the container's Dockerfile ENV (CC=gcc-13,
# PATH with cuda+mold+rust). Every remote job imports PID 1's real environment.
RP_ENV_PREAMBLE='while IFS= read -r -d "" __e; do export "$__e"; done < /proc/1/environ'

rp_gql() { curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$1"; }

rp_terminate() { # $1=podId
  rp_gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${1}\\\"}) }\"}" >/dev/null 2>&1
}

# Confirm a pod id is BOTH present in the account's live pod list AND carries
# a name this SESSION could plausibly own before any caller acts on it
# irreversibly. Never trusts a locally-recorded id on its own: the id could
# be stale (the account-side pod is already gone), or — the incident this
# closes — a DIFFERENT pod could now be recorded under this session's name
# (two processes racing `up` on the same alias). $1=podId $2=expected TTL
# hours, OR EMPTY.
#
# A non-empty $2 (the SESSION's own, as recorded at deploy time — never the
# caller's current RP_TTL_HOURS, which may differ and would defeat the whole
# check) requires an EXACT "<prefix>-ttl<H>" match. An EMPTY $2 means the
# caller genuinely does not know this session's own TTL — a meta file
# written before RP_TTL_HOURS was tracked in the session record has no such
# line, and `${RP_TTL_HOURS:-8}` cannot tell that apart from an EXPLICIT
# ttl8 record; confidently verifying against the wrong guessed number
# refused to release a real `jammi-gpu-ttl72` pod this tooling itself
# rented, which then billed its full 72h. With $2 empty this instead matches
# on the id plus the "<prefix>-ttl<digits>" NAME SHAPE alone — the id is the
# authoritative half of the check; the exact number only disambiguates
# between multiple pods this tooling could have rented under one alias, and
# an unknown number is not evidence the id is wrong.
#
# Prints the account's name for the id on success. Returns 1 (not found, or
# name does not match/does not have the shape) or 2 (could not query the
# account at all) — both are "do not act", never "assume safe".
#
# Piped input (the account's own pod list) and script source cannot both come
# from stdin — `python3 -` with a heredoc reads the heredoc AS THE PROGRAM,
# leaving nothing for the piped JSON to land on (shellcheck SC2259). This
# therefore uses `python3 -c '<script>' argv...`, the same shape
# `rp_deploy_live`'s own parser already uses, so the piped JSON reaches
# `sys.stdin` intact and dynamic values travel as argv, never interpolated
# into the script source.
rp_pod_verify() { # $1=podId $2=expectedTtlHours (empty = unknown; match by name SHAPE only)
  local id="$1" ttl="${2:-}"
  rp_gql '{"query":"query{ myself{ pods{ id name } } }"}' | python3 -c '
import sys, json, re
podid, ttl, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("could not parse RunPod response: %s" % e, file=sys.stderr)
    sys.exit(2)
if d.get("errors"):
    print(json.dumps(d["errors"])[:200], file=sys.stderr)
    sys.exit(2)
me = (d.get("data") or {}).get("myself")
if me is None or me.get("pods") is None:
    print("response contained no pod list", file=sys.stderr)
    sys.exit(2)
if ttl:
    want_desc = "%s-ttl%s" % (prefix, ttl)
    matches = lambda name: name == want_desc
else:
    want_desc = "%s-ttl<digits> (this session recorded no TTL -- matching by name shape only)" % prefix
    shape = re.compile(r"^%s-ttl[0-9]+$" % re.escape(prefix))
    matches = lambda name: shape.match(name) is not None
for p in me["pods"]:
    if p.get("id") == podid:
        name = p.get("name") or ""
        if matches(name):
            print(name)
            sys.exit(0)
        print("pod %s account name %s does not match %s -- refusing to act on it" % (podid, name, want_desc), file=sys.stderr)
        sys.exit(1)
print("pod %s is not in the account pod list" % podid, file=sys.stderr)
sys.exit(1)
' "$id" "$ttl" "$RP_POD_PREFIX"
}

rp_cleanup() {
  if [ -n "$RP_POD_ID" ]; then
    if [ "$RP_KEEP" = "1" ]; then
      echo "::notice::pod ${RP_POD_ID} left running (self-terminates in ≤${RP_TTL_HOURS}h)"
    elif [ "$RP_POD_CREATED" = "1" ]; then
      rp_terminate "$RP_POD_ID"
      echo "::notice::terminated RunPod pod ${RP_POD_ID}"
    else
      # This invocation loaded an EXISTING session's pod (attach/run/logs/
      # push/pull/down) rather than renting one itself, and is
      # exiting without having called rp_keep — e.g. require_pod failing
      # against an unreachable pod. That pod is not this invocation's to
      # terminate; only its own creator (an `up`/`shell` that actually
      # deployed it, or an explicit `down`) may end it.
      echo "::notice::pod ${RP_POD_ID} left running — this invocation did not create it; not terminating"
    fi
  fi
  [ "$RP_WORK_IS_TEMP" = "1" ] && rm -rf "$RP_WORK"
}
trap rp_cleanup EXIT

# Disarm teardown for this process. The pod survives; the reaper is the only
# thing that will stop it, so rp_reap must already be installed.
rp_keep() { RP_KEEP=1; }

# Generate (or reuse) the SSH keypair used to reach the pod. Reuse matters for a
# named session: regenerating would lock us out of a pod that is still running.
rp_init() {
  _rp_work_mkdir
  [ -f "$RP_SSH_KEY" ] || ssh-keygen -t ed25519 -N '' -f "$RP_SSH_KEY" -q
  RP_PUBKEY="$(cat "${RP_SSH_KEY}.pub")"
  RP_SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i "$RP_SSH_KEY")
}

# Create the work dir on first write. Split out so read-only commands against a
# nonexistent session do not litter RP_SESSION_ROOT with empty directories.
_rp_work_mkdir() { [ -d "$RP_WORK" ] || { mkdir -p "$RP_WORK" && chmod 700 "$RP_WORK"; }; }

rp_session_save() {
  [ -n "$RP_SESSION" ] || return 0
  _rp_work_mkdir
  # RP_TTL_HOURS travels with the session because it is baked into the pod's
  # OWN name at deploy ("<prefix>-ttl<H>") and cannot be re-derived from the
  # caller's current environment later — a `down` run with a different
  # RP_TTL_HOURS in its own shell must still reason about THIS pod's actual
  # name, not whatever the current invocation happens to default to.
  { echo "RP_POD_ID=$RP_POD_ID"; echo "RP_HOST=$RP_HOST"; echo "RP_PORT=$RP_PORT"
    echo "RP_ARCH=$RP_ARCH"; echo "RP_IMAGE=$RP_IMAGE"; echo "RP_REF=$RP_REF"
    echo "RP_TTL_HOURS=$RP_TTL_HOURS"; } > "$RP_META"
  chmod 600 "$RP_META"
}

# Restore pod coordinates for a session started by another process. Returns 1
# when the session has no recorded pod.
rp_session_load() {
  [ -f "$RP_META" ] || return 1
  # shellcheck disable=SC1090  # path is a runtime-selected session dir
  . "$RP_META"
  [ -n "${RP_POD_ID:-}" ] || return 1
  rp_init
}

# A recorded pod is not a live pod — the reaper may have collected it, or the
# host may have died. Callers must confirm before treating a session as usable.
rp_session_alive() {
  [ -n "$RP_POD_ID" ] && [ -n "$RP_HOST" ] && [ -n "$RP_PORT" ] || return 1
  ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" true 2>/dev/null
}

# The whole session table — header included, so the row format exists once. The
# column widths are derived from the rows rather than fixed: a ref is a branch
# name or a 40-character commit id, so every fixed width is eventually too narrow
# and a single over-long cell shifts every column after it out of alignment.
rp_session_list() {
  local rows="" d name w_s=7 w_p=3 w_r=3 f_s f_p f_r f_rest
  if [ -d "$RP_SESSION_ROOT" ]; then
    for d in "$RP_SESSION_ROOT"/*/; do
      [ -f "${d}meta" ] || continue
      name="$(basename "$d")"
      # Subshell: every meta file sets the same variable names. A session with no
      # recorded ref never completed a bootstrap; say so rather than imply main,
      # which is the guess this whole path exists to remove.
      # shellcheck disable=SC1090
      rows="${rows}$( . "${d}meta"
        printf '%s\t%s\t%s\t%s@%s:%s' "$name" "${RP_POD_ID:-?}" "${RP_REF:-<none>}" \
          "${RP_ARCH:-?}" "${RP_HOST:-?}" "${RP_PORT:-?}" )"$'\n'
    done
  fi
  while IFS=$'\t' read -r f_s f_p f_r f_rest; do
    [ -n "$f_s" ] || continue
    [ "${#f_s}" -gt "$w_s" ] && w_s="${#f_s}"
    [ "${#f_p}" -gt "$w_p" ] && w_p="${#f_p}"
    [ "${#f_r}" -gt "$w_r" ] && w_r="${#f_r}"
  done <<< "$rows"
  # `%-*s` takes each width as an argument, so the format string stays a literal
  # and the header cannot drift from the rows it labels.
  printf '%-*s  %-*s  %-*s  %s\n' "$w_s" SESSION "$w_p" POD "$w_r" REF ARCH@HOST
  while IFS=$'\t' read -r f_s f_p f_r f_rest; do
    [ -n "$f_s" ] || continue
    printf '%-*s  %-*s  %-*s  %s\n' "$w_s" "$f_s" "$w_p" "$f_p" "$w_r" "$f_r" "$f_rest"
  done <<< "$rows"
}

# Regenerate the ssh config THIS TOOLING OWNS, one `Host jammi-<session>` block
# per live session, derived from the session files.
#
# It deliberately never edits ~/.ssh/config. A bug in a rewrite there would break
# every other host the user has, and that file is not ours. Consumers opt in once
# — an `Include` line, or an editor's remote-SSH configFile setting — and keep
# control of their own config.
#
# Regenerating the whole file (rather than patching a block) makes it
# self-healing: a session directory that no longer exists simply stops appearing.
rp_ssh_config_sync() {
  local d name tmp
  mkdir -p "$(dirname "$RP_SSH_CONFIG")" || return 0
  tmp="${RP_SSH_CONFIG}.tmp.$$"
  {
    echo "# Generated by ci/scripts/gpu-dev.sh — do not edit; regenerated on up/down."
    echo "# Use it with:  ssh -F ~/.config/runpod/ssh_config jammi-<session>"
    echo "# or add to ~/.ssh/config:  Include ~/.config/runpod/ssh_config"
    echo
    for d in "$RP_SESSION_ROOT"/*/; do
      [ -f "${d}meta" ] || continue
      name="$(basename "$d")"
      # Subshell: each meta file sets the same variable names.
      # shellcheck disable=SC1090
      ( . "${d}meta"
        [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ] || exit 0
        echo "Host jammi-${name}"
        echo "    HostName ${RP_HOST}"
        echo "    Port ${RP_PORT}"
        echo "    User root"
        echo "    IdentityFile ${d}id_ed25519"
        echo "    IdentitiesOnly yes"
        # Pod host keys are new every time and the address is recycled across
        # tenants, so a known_hosts entry is guaranteed to conflict.
        echo "    StrictHostKeyChecking no"
        echo "    UserKnownHostsFile /dev/null"
        echo "    LogLevel ERROR"
        echo )
    done
  } > "$tmp" 2>/dev/null && mv "$tmp" "$RP_SSH_CONFIG" && chmod 600 "$RP_SSH_CONFIG"
}

# Forget a session's local state. The caller terminates the pod first.
rp_session_forget() {
  [ -n "$RP_SESSION" ] && [ -d "$RP_WORK" ] && rm -rf "$RP_WORK"
}

# Load the object-store config used for the build-substrate cache. Returns 1 when
# unconfigured — every caller treats that as "run cold", never as an error: the
# cache is an optimisation and correctness must not depend on it.
rp_s3_load() {
  [ -f "$RP_S3_CONF" ] || return 1
  # shellcheck disable=SC1090  # user-provided config outside the repo
  . "$RP_S3_CONF"
  [ -n "${RP_S3_ENDPOINT:-}" ] && [ -n "${RP_S3_BUCKET:-}" ] \
    && [ -n "${RP_S3_ACCESS_KEY_ID:-}" ] && [ -n "${RP_S3_SECRET_ACCESS_KEY:-}" ] || return 1
  # RunPod signs SigV4 against the datacenter as the region and rejects "auto"
  # outright ("the region 'auto' is wrong; expecting 'us-ne-1'"), which bricks
  # every cargo command on the pod since sccache wraps rustc globally. Derive it
  # from the endpoint so there is one source of truth; a non-RunPod endpoint
  # (R2 and friends) keeps "auto" unless the config states otherwise.
  if [ -z "${RP_S3_REGION:-}" ]; then
    case "$RP_S3_ENDPOINT" in
      *s3api-*.runpod.io*)
        RP_S3_REGION="${RP_S3_ENDPOINT#*s3api-}"
        RP_S3_REGION="${RP_S3_REGION%%.runpod.io*}" ;;
      *) RP_S3_REGION=auto ;;
    esac
  fi
}

_rp_deploy_payload() { # $1=cloudType $2=gpuTypeId
  python3 - "$1" "$2" "$RP_IMAGE" "$RP_PUBKEY" "$RP_TTL_HOURS" "$RP_POD_PREFIX" "$RP_DISK_GB" "$RP_VOLUME_GB" <<'PY'
import json, sys
cloud, gpu, image, pub, ttl_h, prefix, disk_gb, volume_gb = sys.argv[1:9]
ttl = int(ttl_h) * 3600
# The deadline is part of the pod's own entrypoint, so it exists from the moment
# the container starts. It CANNOT be installed over SSH after the fact: the
# window between "pod rented" and "SSH reachable" is minutes long, and a runner
# SIGKILLed inside it never runs its EXIT trap. That is exactly how an A100 was
# orphaned for seven days on 2026-07-24.
#
# `sleep` comes FIRST so the deadline is reached no matter what the network does;
# an install ahead of it can hang and leave the pod with no deadline at all. Every
# step after it is `timeout`-bounded so nothing can wedge the sequence.
#
# `runpodctl remove pod $RUNPOD_POD_ID` is the ONLY in-pod termination that works,
# and it is verified on real hardware: RunPod special-cases self-removal, so it
# succeeds in this custom image with no config file and no key of ours — even
# though `runpodctl config` fails and `runpodctl get pod` returns Unauthorized.
#
# There is deliberately no `kill 1` fallback. It was measured to be a no-op:
# PID 1 in a PID namespace ignores signals it has no handler for, including
# SIGKILL, so the pod kept RUNNING and kept billing at full rate. A fallback that
# cannot work is worse than none — it invites trusting a guard that does nothing.
# The retry loop replaces it: the only real failure mode left is no network at
# deadline time, and retrying costs nothing. rp_sweep remains the true backstop.
watchdog = ("( sleep %d; "
            "while :; do "
            "timeout 120 sh -c \"command -v runpodctl >/dev/null 2>&1 || "
            "curl -fsSL cli.runpod.net | bash\" >/dev/null 2>&1; "
            "timeout 60 runpodctl remove pod \"$RUNPOD_POD_ID\" >/dev/null 2>&1 && break; "
            "sleep 60; done ) & " % ttl)
# The watchdog is armed BEFORE anything else in the entrypoint. Any command
# placed ahead of it — `yum install` reaching the network, for instance — can
# hang and leave the pod running with no deadline, which is the failure this
# whole mechanism exists to prevent.
setup = (watchdog
         + "yum install -y openssh-server openssh-clients >/dev/null 2>&1; ssh-keygen -A; "
         "mkdir -p /root/.ssh; printf \"%s\\n\" \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
         "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
         "/usr/sbin/sshd -D")

# `setup` is wrapped in bash -c '...' below. A single quote anywhere inside it
# closes that wrapper early and hands the remainder — pipes, redirects,
# semicolons — to the OUTER shell. The entrypoint then dies on a syntax error and
# the pod boots, bills, and never becomes reachable: a silent, paid failure that
# looks exactly like a capacity problem. Quote with double quotes only.
assert "'" not in setup, "pod entrypoint must contain no single quotes (breaks bash -c wrapping)"
inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu,
       # The deadline travels in the name so any sweeper honours THIS pod's limit.
       "name": "%s-ttl%s" % (prefix, ttl_h),
       "imageName": image, "containerDiskInGb": int(disk_gb), "volumeInGb": int(volume_gb), "ports": "22/tcp",
       "dockerArgs": "bash -c '%s'" % setup, "env": [{"key": "PUBLIC_KEY", "value": pub}]}
print(json.dumps({"query": "mutation D($i: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$i){ id } }",
                  "variables": {"i": inp}}))
PY
}

# Deploy a live GPU pod, failing over across a candidate list of "CLOUD|GPU_TYPE"
# args (capacity) and terminating any pod that never becomes reachable
# (liveness). On success sets RP_POD_ID/RP_HOST/RP_PORT and returns 0. Returns 75
# (neutral skip) when no candidate yields a reachable pod — a provider condition,
# not a code failure.
rp_deploy_live() {
  # A new pod voids everything recorded about the PREVIOUS pod's contents, and
  # the ref is the only such axis. rp_session_save runs the moment SSH comes up —
  # minutes ahead of the checkout, and never at all if the bootstrap fails — so a
  # ref inherited from a dead session would be written against the new pod id and
  # `ls` would report a live pod as being on code it was never on. Cleared here,
  # at the one place pod identity changes, so no caller has to remember; the
  # `<none>` sentinel then means what it says: this pod has no checkout yet.
  RP_REF=""
  local supply_seen=0 combo cloud gpu R parsed code msg
  for combo in "$@"; do
    cloud="${combo%%|*}"; gpu="${combo##*|}"
    # Only SUPPLY_CONSTRAINT is a capacity condition worth failing over. Every
    # other refusal — INSUFFICIENT_BALANCE, a bad key, an unpullable image — is a
    # real fault, and reporting it as "no capacity" would return the neutral-skip
    # 75 and let the GPU gate pass while proving nothing.
    # Emitted as three LINES, not delimited fields: every plausible single-char
    # delimiter is either IFS whitespace (which collapses the empty id field on
    # an error, silently turning the error code into the pod id) or can occur in
    # the message text.
    parsed="$(rp_gql "$(_rp_deploy_payload "$cloud" "$gpu")" | python3 -c 'import sys,json
d=json.load(sys.stdin)
e=(d.get("errors") or [])
if e:
    x=e[0]
    print("")
    print((x.get("extensions") or {}).get("code") or "UNKNOWN")
    print(" ".join((x.get("message") or "").split()))
else:
    print((d.get("data",{}).get("podFindAndDeployOnDemand") or {}).get("id","") or "")
    print("NO_ID")
    print("")')"
    { read -r RP_POD_ID; read -r code; read -r msg; } <<< "$parsed"
    if [ -z "$RP_POD_ID" ]; then
      case "$code" in
        SUPPLY_CONSTRAINT|NO_ID) echo "  no capacity: ${cloud} / ${gpu}"; continue ;;
        *) echo "::error::RunPod refused the deploy (${code}): ${msg}"
           echo "::error::this is not a capacity condition — failing loudly rather than skipping"
           return 1 ;;
      esac
    fi
    supply_seen=1
    # THIS is the moment a pod exists and starts billing — not the SSH-up
    # point below, which can be minutes later. An EXIT trap firing anywhere
    # in the ≤4m reachability wait (Ctrl-C, a cancelled CI run's SIGTERM) must
    # still terminate a pod this invocation just rented, or it leaks for its
    # full deadline (72h under the `up` dev default) with rp_cleanup reporting
    # "did not create it; not terminating" — exactly backwards. Stays 1 for
    # the rest of this call even if this candidate is later torn down for
    # being unreachable/under-floor: any pod rented after it is equally this
    # invocation's own.
    RP_POD_CREATED=1
    echo "  deployed ${RP_POD_ID} on ${cloud} / ${gpu}; waiting for SSH (≤4m)..."
    RP_HOST=""; RP_PORT=""
    for _ in $(seq 1 24); do
      R="$(rp_gql "{\"query\":\"query{ pod(input:{podId:\\\"${RP_POD_ID}\\\"}){ runtime{ ports{ ip publicPort privatePort isIpPublic type } } } }\"}")"
      read -r RP_HOST RP_PORT < <(printf '%s' "$R" | python3 -c 'import sys,json
p=(json.load(sys.stdin).get("data",{}).get("pod") or {}).get("runtime") or {}
[print(x["ip"], x["publicPort"]) for x in (p.get("ports") or []) if x.get("privatePort")==22 and x.get("isIpPublic")]' | head -1)
      if [ -n "${RP_HOST:-}" ] && ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" true 2>/dev/null; then
        # Reachable — now gate on the driver floor. A pod below r560 cannot JIT
        # the image's CUDA 12.6 PTX, so it is unusable for this build; fail over
        # to the next candidate rather than run every test into the #304 floor.
        local drv drv_major
        drv="$(ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" \
          "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)"
        drv_major="${drv%%.*}"
        if [ -n "$drv_major" ] && [ "$drv_major" -ge "$RP_MIN_DRIVER_MAJOR" ] 2>/dev/null; then
          echo "  SSH up on ${RP_HOST}:${RP_PORT} (driver ${drv})"
          # RP_POD_CREATED was already set the moment RP_POD_ID was read from
          # the deploy response, above — not here, which is minutes later.
          rp_session_save; return 0
        fi
        echo "  pod ${RP_POD_ID} driver '${drv:-unknown}' is below the r${RP_MIN_DRIVER_MAJOR} floor; terminating and trying next candidate"
        rp_terminate "$RP_POD_ID"
        RP_POD_ID=""; break
      fi
      RP_HOST=""; sleep 10
    done
    if [ -n "$RP_POD_ID" ]; then
      echo "  pod ${RP_POD_ID} never became reachable; terminating and trying next candidate"
      rp_terminate "$RP_POD_ID"
      RP_POD_ID=""
    fi
  done
  if [ "$supply_seen" = "0" ]; then echo "::error::no GPU capacity on RunPod for the requested candidates (SUPPLY_CONSTRAINT); retry later"
  else echo "::error::GPU pod(s) deployed but none became reachable over SSH; retry later"; fi
  return 75
}

# arch → RunPod GPU-type candidates (SECURE then COMMUNITY), the one place the
# mapping lives. A100 is the #277 floor (sm_80); l40s/l4 are Ada (sm_89, fp8
# #308); h100 is Hopper (sm_90); a40 is Ampere-workstation (sm_86). (RunPod has
# no Tesla T4 — #306 is Ampere+.) Returns 2 on an unknown arch.
rp_deploy_arch() { # $1=arch
  local cand
  case "$1" in
    a100) cand=("SECURE|NVIDIA A100 80GB PCIe" "COMMUNITY|NVIDIA A100 80GB PCIe" "SECURE|NVIDIA A100-SXM4-80GB" "COMMUNITY|NVIDIA A100-SXM4-80GB") ;;
    l40s) cand=("SECURE|NVIDIA L40S" "COMMUNITY|NVIDIA L40S") ;;
    h100) cand=("SECURE|NVIDIA H100 80GB HBM3" "SECURE|NVIDIA H100 PCIe" "COMMUNITY|NVIDIA H100 80GB HBM3" "COMMUNITY|NVIDIA H100 PCIe") ;;
    a40)  cand=("SECURE|NVIDIA A40" "COMMUNITY|NVIDIA A40") ;;
    l4)   cand=("SECURE|NVIDIA L4" "COMMUNITY|NVIDIA L4") ;;
    *) echo "::error::unknown arch '$1' (want: a100|l40s|h100|a40|l4)"; return 2 ;;
  esac
  RP_ARCH="$1"
  rp_deploy_live "${cand[@]}"
}

# A100 (sm_80) — the arch that proves the compute_80 floor / #277.
rp_deploy_live_a100() { rp_deploy_arch a100; }

# Run a bash script (read from stdin) on the pod, with the container ENV imported
# first and a hard timeout. Returns the remote script's exit code.
#
# The timeout binds THIS invocation only. A script that daemonizes something —
# gpu-dev.sh's `run` starts a detached tmux session — returns immediately and
# leaves the real work running past any RP_TIMEOUT. The pod's own deadline is the
# thing that bounds that, which is why the deadline is armed in the entrypoint.
rp_run_remote() {
  { printf '%s\n' "$RP_ENV_PREAMBLE"; cat; } \
    | ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" "timeout ${RP_TIMEOUT:-3000} bash -s"
}

# Terminate orphaned pods. The in-pod deadline is the primary guard; this is the
# backstop for a pod whose container never got far enough to arm it, and the only
# thing that can clean up after a runner killed during the minutes-long gap
# between "pod rented" and "pod reachable".
#
# Only touches pods this tooling created (the RP_POD_PREFIX name): anything else
# in the account is somebody's deliberate work and is never in scope. A stopped
# jammi pod is always garbage — the tooling terminates, it never stops.
#
# With no argument each pod is judged against the deadline carried in its own
# name, so a sweeper never imposes its own limit on someone else's pod. An
# explicit hours argument is a force-reap override for the emergency case.
#
# Every ambiguity resolves toward terminating. A pod this tooling rented that we
# cannot reason about is far more likely to be a leak than healthy work, and the
# cost of a wrong sweep is one re-run; the cost of a wrong spare is $187.
rp_sweep() { # $1=optional override age in hours
  local override="${1:-}" out rc id age why n=0
  if [ -n "$override" ]; then
    case "$override" in
      ''|*[!0-9]*) echo "::error::reap: hours must be a positive integer (got '${override}')"; return 2 ;;
    esac
  fi
  out="$(rp_gql '{"query":"query{ myself{ pods{ id name desiredStatus createdAt runtime{ uptimeInSeconds } } } }"}' \
    | python3 -c "
import sys, json, datetime
override = '''$override'''.strip()
prefix = '$RP_POD_PREFIX'
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('could not parse RunPod response: %s' % e); sys.exit(3)
if d.get('errors'):
    print(json.dumps(d['errors'])[:200]); sys.exit(3)
me = (d.get('data') or {}).get('myself')
if me is None or me.get('pods') is None:
    print('response contained no pod list'); sys.exit(3)
now = datetime.datetime.now(datetime.timezone.utc)
for p in me['pods']:
    name = p.get('name') or ''
    if not name.startswith(prefix):
        continue
    # Age comes from createdAt, never from runtime.uptimeInSeconds. Measured on
    # live pods: uptime is null for the first minutes of a perfectly healthy pod,
    # so treating null as 'unreachable' terminates in-flight CI runs. createdAt is
    # also a true since-RENTAL clock, so a container restart cannot reset it —
    # which uptime can, letting a pod live forever in deadline-length increments.
    age = None
    ca = p.get('createdAt')
    if ca:
        try:
            age = int((now - datetime.datetime.fromisoformat(ca.replace('Z', '+00:00'))).total_seconds())
        except Exception:
            age = None
    if p.get('desiredStatus') != 'RUNNING':
        print(p['id'], age if age is not None else -1, 'not-running'); continue
    if age is None:
        # Cannot establish age. Killing on this basis is how healthy pods die, so
        # surface it instead and let an operator force-reap.
        print('UNAGEABLE', p['id'], name); continue
    if override:
        limit = int(override) * 3600
    else:
        tail = name[len(prefix):]
        if tail.startswith('-ttl') and tail[4:].isdigit():
            limit = int(tail[4:]) * 3600
        else:
            print(p['id'], age, 'unparseable-deadline'); continue
    if age > limit:
        print(p['id'], age, 'past-deadline-%ds' % limit)
")"
  rc=$?
  # A failed query must never read as "nothing to clean up" — this is the
  # independent backstop, and a silent green here is how a guard stops guarding.
  if [ "$rc" -ne 0 ]; then
    echo "::error::sweep could NOT enumerate pods; orphans may exist unseen: ${out}"
    return 1
  fi
  [ -n "$out" ] || { echo "sweep: queried OK — no orphaned ${RP_POD_PREFIX} pods"; return 0; }
  while read -r id age why; do
    [ -n "$id" ] || continue
    if [ "$id" = "UNAGEABLE" ]; then
      echo "::error::pod ${age} (${why}) has no usable createdAt — cannot judge its age; reap explicitly if it is an orphan"
      continue
    fi
    rp_terminate "$id"
    echo "::warning::swept pod ${id} (${why}, age ${age}s)"
    n=$(( n + 1 ))
  done <<< "$out"
  echo "sweep: terminated ${n} orphaned pod(s)"
}

# A ref travels to the pod inside a remote command line, so it is constrained to
# git's own refname charset before it can get there. Anything outside that charset
# — a quote, a semicolon, a backtick — closes the interpolation and hands the
# remainder to the pod's shell as commands.
#
# The empty string is rejected rather than defaulted: a caller that computed an
# empty ref asked for a specific thing and got nothing, and silently substituting
# `main` builds the wrong commit while reporting success. A leading `-` reads as
# an option to git, and `..` is a range, never a single ref.
rp_ref_check() { # $1=ref
  case "${1:-}" in
    '')   echo "::error::git ref is empty" >&2; return 2 ;;
    -*)   echo "::error::git ref must not start with '-' (got '${1}')" >&2; return 2 ;;
    *..*) echo "::error::git ref must not contain '..' (got '${1}')" >&2; return 2 ;;
    *[!A-Za-z0-9._/-]*)
          echo "::error::git ref may contain only [A-Za-z0-9._/-] (got '${1}')" >&2; return 2 ;;
  esac
}

# An abbreviated or full commit id: all hex, at least 7 characters. Such a ref is
# spelled exactly like a branch of the same name and a remote resolves names, not
# object ids, so it can never be pre-checked.
_rp_ref_is_objectish() { # $1=ref
  [ "${#1}" -ge 7 ] && [ "${#1}" -le 40 ] || return 1
  case "$1" in *[!0-9a-f]*) return 1 ;; esac
}

# Ask the remote whether a ref exists, in bounded time and without ever asking a
# human anything. This runs on every `up` and `shell`, and RP_REPO_URL is
# overridable, so a private URL has three ways to turn a one-second gate into an
# indefinite stall: HTTPS credentials prompted on the terminal, an ssh remote
# asking to trust a host key, and a server that accepts the connection and then
# sends nothing. The first two are refused outright; the third is bounded by
# git's own low-speed knobs rather than `timeout`, which is coreutils and absent
# on a BSD host — the gate must behave identically wherever a maintainer runs it.
_rp_ls_remote() { # $1=ref
  GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='ssh -oBatchMode=yes -oConnectTimeout=10' \
  git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 \
      ls-remote --exit-code --heads --tags "$RP_REPO_URL" "$1" >/dev/null 2>&1
}

# Prove a ref exists BEFORE a pod is rented. `git ls-remote` costs about a second
# and no money; learning the same fact from the pod costs a GPU plus the
# minutes-long wait for SSH. An unverifiable ref refuses to rent, in both
# directions: a ref that is absent and a remote that cannot be reached are
# different messages but the same answer, because renting on a guess is the
# expensive mistake.
#
# A commit id is the one ref that cannot be answered here and is therefore
# verified on the pod — the single case where the failure is paid for.
rp_ref_precheck() { # $1=ref
  local rc
  if _rp_ref_is_objectish "$1"; then
    echo "note: '${1}' looks like a commit id — it can only be verified on the pod"
    return 0
  fi
  _rp_ls_remote "$1"
  rc=$?
  case "$rc" in
    0) return 0 ;;
    2) echo "::error::'${1}' is not a branch or tag in ${RP_REPO_URL} — nothing was rented" >&2 ;;
    *) echo "::error::could not reach ${RP_REPO_URL} to verify '${1}' (git ls-remote exit ${rc})" >&2
       echo "::error::refusing to rent a pod for a ref that cannot be verified" >&2 ;;
  esac
  return 1
}

# Make the pod ready to build jammi: import the container ENV (already done by
# rp_run_remote), point sccache at the object store, and place the repo at $1
# (default: main).
#
# The cargo REGISTRY is deliberately not cached. It looks expensive — the CI image
# wipes /usr/local/cargo/registry, so every pod re-fetches the whole
# arrow/datafusion/candle tree — but measured on a RunPod host that fetch is 9s
# for 868 crates, because datacenter bandwidth makes it free. Restoring a 285MB
# tarball was slower than just fetching. The real cold cost is COMPILATION, which
# is what sccache addresses.
# An omitted ref means main; an empty one is a caller bug, not a default. The
# distinction is made on argument COUNT, because `${1:-main}` collapses the two
# and turns "the ref I computed is empty" into a silent, successful boot on main.
#
# Returns 0 (pod is on $1), 2 (the ref itself is malformed), 3 (the image ships
# no git, so the pod is on NO ref and RP_REF stays empty), or the remote script's
# code for a real failure.
rp_bootstrap() { # $1=git ref (optional; default main)
  local ref=main s3=0 rc
  [ $# -gt 0 ] && ref="$1"
  rp_ref_check "$ref" || return 2
  rp_s3_load && s3=1
  rp_run_remote <<EOF
set -uo pipefail
export CARGO_HOME="\${CARGO_HOME:-/usr/local/cargo}"

# Pod-side tools the dev loop needs and the CI image has no reason to carry:
# rsync for the working-tree sync, tmux for detached jobs that outlive the SSH
# session.
yum install -y rsync tmux >/dev/null 2>&1 || echo "warn: pod tool install failed"

# One re-sourceable file that makes any later shell correct. The container's
# Dockerfile ENV is captured here rather than re-derived from /proc/1/environ at
# each use site, so \`attach\` and detached \`run\` jobs cannot drift from it.
{ while IFS= read -r -d '' __e; do
    printf 'export %s=%q\n' "\${__e%%=*}" "\${__e#*=}"
  done < /proc/1/environ; } > /root/.jammi_env

# Make shells we do NOT launch correct too — plain ssh, an editor's remote server,
# a language server. An SSH session inherits none of the container's Dockerfile
# ENV, so without this cargo, nvcc and mold are absent from any session that did
# not come through gpu-dev.sh. Both files are needed: /etc/profile.d is read by
# LOGIN shells (which is how an editor server and its language server are
# started), .bashrc by interactive non-login ones.
echo '[ -f /root/.jammi_env ] && . /root/.jammi_env' > /etc/profile.d/jammi-env.sh
grep -q jammi_env /root/.bashrc 2>/dev/null \
  || echo '[ -f /root/.jammi_env ] && . /root/.jammi_env' >> /root/.bashrc

if [ "${s3}" = "1" ]; then
  export AWS_ACCESS_KEY_ID='${RP_S3_ACCESS_KEY_ID:-}'
  export AWS_SECRET_ACCESS_KEY='${RP_S3_SECRET_ACCESS_KEY:-}'
  export S3_ENDPOINT='${RP_S3_ENDPOINT:-}'
  export S3_BUCKET='${RP_S3_BUCKET:-}'

  # sccache reads the compile cache straight from the object store, so a pod in
  # any datacenter warms up. Wrapper is already set repo-wide in .cargo/config.toml.
  export SCCACHE_BUCKET="\$S3_BUCKET"
  export SCCACHE_ENDPOINT="\$S3_ENDPOINT"
  export SCCACHE_REGION='${RP_S3_REGION:-auto}'
  export SCCACHE_S3_USE_SSL=true
  export SCCACHE_S3_KEY_PREFIX=sccache

  # sccache wraps rustc for EVERY cargo invocation, so an object store it cannot
  # reach does not degrade the build — it stops the build dead, and the pod is
  # useless for anything. Prove the server starts against this bucket; if it does
  # not, drop the S3 backend and let sccache fall back to its local disk cache.
  # A cold pod that works beats a warm one that cannot compile.
  sccache --stop-server >/dev/null 2>&1
  if sccache --start-server >/dev/null 2>&1 && sccache --show-stats >/dev/null 2>&1; then
    { echo "export AWS_ACCESS_KEY_ID=\$(printf %q "\$AWS_ACCESS_KEY_ID")"
      echo "export AWS_SECRET_ACCESS_KEY=\$(printf %q "\$AWS_SECRET_ACCESS_KEY")"
      echo "export SCCACHE_BUCKET=\$(printf %q "\$SCCACHE_BUCKET")"
      echo "export SCCACHE_ENDPOINT=\$(printf %q "\$SCCACHE_ENDPOINT")"
      echo "export SCCACHE_REGION=\$(printf %q "\$SCCACHE_REGION")"
      echo "export SCCACHE_S3_USE_SSL=true"
      echo "export SCCACHE_S3_KEY_PREFIX=sccache"; } >> /root/.jammi_env
    echo "sccache: S3 backend live (region \$SCCACHE_REGION)"
  else
    echo "::warning::sccache could not use the S3 backend; falling back to local disk cache"
    sccache --stop-server >/dev/null 2>&1
    unset SCCACHE_BUCKET SCCACHE_ENDPOINT SCCACHE_REGION SCCACHE_S3_USE_SSL SCCACHE_S3_KEY_PREFIX
    sccache --start-server >/dev/null 2>&1
  fi
fi

# "This image CANNOT hold a checkout" and "the checkout failed" are different
# facts and only the second is a broken pod. Reproducing the shipped RUNTIME
# image is a real use of a GPU pod, and that image ships no toolchain and no git
# — there is nothing to fix and nothing to retry. Exit 3 reports a pod that is on
# NO ref; RP_REF stays empty so nothing ever claims otherwise, and the caller
# decides whether a refless pod is what was asked for.
if ! command -v git >/dev/null 2>&1; then
  echo "::notice::no git in ${RP_IMAGE} — this pod gets no checkout"
  exit 3
fi

if [ ! -d /root/jammi-ai/.git ]; then
  git clone --filter=blob:none "${RP_REPO_URL}" /root/jammi-ai || exit 1
fi
cd /root/jammi-ai || exit 1

# Every step below is checked. A pod that reports "bootstrap complete" while
# sitting on an older commit is worse than one that failed: the build, the test
# result and the benchmark are all real, and all answer a question about code
# nobody is looking at.
git fetch --all --tags --prune --quiet \
  || { echo "::error::git fetch failed — the pod cannot see the current refs"; exit 1; }
git checkout --quiet "${ref}" \
  || { echo "::error::ref '${ref}' not found in ${RP_REPO_URL}"; exit 1; }
# Only a branch tracks anything; a tag or a commit id is a detached HEAD where
# there is nothing to fast-forward to. A branch that will NOT fast-forward has
# been force-pushed, so the checkout is on a commit that no longer exists
# upstream — the one case this whole sequence exists to catch.
if git symbolic-ref -q HEAD >/dev/null; then
  git pull --quiet --ff-only \
    || { echo "::error::'${ref}' cannot fast-forward — force-pushed? the pod is on a stale commit"; exit 1; }
fi

echo "bootstrap complete: ${ref} @ \$(git rev-parse --short HEAD)"
EOF
  rc=$?
  [ "$rc" -eq 0 ] || return "$rc"
  # The ref is an identity axis of a session: nothing else records which code a
  # pod is running, so `ls` and a second terminal cannot otherwise tell. Written
  # only once the checkout has actually happened, and its failure is the caller's
  # failure — a session that cannot be recorded is a pod nobody can find again.
  RP_REF="$ref"
  rp_session_save
}
