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
#   RP_TIMEOUT    remote job timeout in seconds (default 3000).
#   RP_SESSION    named session. Persists pod coordinates AND the SSH key under
#                 RP_SESSION_ROOT so a *different* terminal can reattach. Unset =
#                 throwaway temp dir, wiped on exit.
#   RP_KEEP       1 = leave the pod running when this process exits.
#   RP_TTL_HOURS  hard deadline baked into every pod at deploy (default 8).
# Sets globals: RP_POD_ID, RP_HOST, RP_PORT. Installs an EXIT trap for teardown.

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
RP_SESSION_ROOT="${RP_SESSION_ROOT:-${HOME}/.config/runpod/sessions}"
RP_S3_CONF="${RP_S3_CONF:-${HOME}/.config/runpod/s3}"
RP_REPO_URL="${RP_REPO_URL:-https://github.com/f-inverse/jammi-ai}"
# s5cmd: the S3 client the pod uses to move the prewarm tarball. Pinned to a
# release binary for the same reason mold/protoc/sccache are in the CI image —
# an unpinned fetch makes an otherwise reproducible pod drift under you.
RP_S5CMD_VERSION="${RP_S5CMD_VERSION:-2.3.0}"

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
RP_POD_ID=""; RP_HOST=""; RP_PORT=""; RP_PUBKEY=""; RP_ARCH=""; RP_SSHO=()

# An SSH login shell does NOT inherit the container's Dockerfile ENV (CC=gcc-13,
# PATH with cuda+mold+rust). Every remote job imports PID 1's real environment.
RP_ENV_PREAMBLE='while IFS= read -r -d "" __e; do export "$__e"; done < /proc/1/environ'

rp_gql() { curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$1"; }

rp_terminate() { # $1=podId
  rp_gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${1}\\\"}) }\"}" >/dev/null 2>&1
}

rp_cleanup() {
  if [ -n "$RP_POD_ID" ]; then
    if [ "$RP_KEEP" = "1" ]; then
      echo "::notice::pod ${RP_POD_ID} left running (self-terminates in ≤${RP_TTL_HOURS}h)"
    else
      rp_terminate "$RP_POD_ID"
      echo "::notice::terminated RunPod pod ${RP_POD_ID}"
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
  { echo "RP_POD_ID=$RP_POD_ID"; echo "RP_HOST=$RP_HOST"; echo "RP_PORT=$RP_PORT"
    echo "RP_ARCH=$RP_ARCH"; echo "RP_IMAGE=$RP_IMAGE"; } > "$RP_META"
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

rp_session_list() {
  [ -d "$RP_SESSION_ROOT" ] || return 0
  local d name
  for d in "$RP_SESSION_ROOT"/*/; do
    [ -f "${d}meta" ] || continue
    name="$(basename "$d")"
    # shellcheck disable=SC1090
    ( . "${d}meta"; printf '%-16s %-20s %s@%s\n' "$name" "${RP_POD_ID}" "${RP_ARCH}" "${RP_HOST}:${RP_PORT}" )
  done
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
    && [ -n "${RP_S3_ACCESS_KEY_ID:-}" ] && [ -n "${RP_S3_SECRET_ACCESS_KEY:-}" ]
}

_rp_deploy_payload() { # $1=cloudType $2=gpuTypeId
  python3 - "$1" "$2" "$RP_IMAGE" "$RP_PUBKEY" "$RP_TTL_HOURS" "$RP_POD_PREFIX" <<'PY'
import json, sys
cloud, gpu, image, pub, ttl_h, prefix = sys.argv[1:7]
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
       "imageName": image, "containerDiskInGb": 60, "volumeInGb": 0, "ports": "22/tcp",
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
          echo "  SSH up on ${RP_HOST}:${RP_PORT} (driver ${drv})"; rp_session_save; return 0
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
  out="$(rp_gql '{"query":"query{ myself{ pods{ id name desiredStatus runtime{ uptimeInSeconds } } } }"}' \
    | python3 -c "
import sys, json
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
for p in me['pods']:
    name = p.get('name') or ''
    if not name.startswith(prefix):
        continue
    up = (p.get('runtime') or {}).get('uptimeInSeconds')
    if p.get('desiredStatus') != 'RUNNING':
        print(p['id'], up if up is not None else -1, 'not-running'); continue
    if up is None:
        # A pod we rented that reports no telemetry is unreachable, not young.
        print(p['id'], -1, 'no-telemetry'); continue
    if override:
        limit = int(override) * 3600
    else:
        tail = name[len(prefix):]
        if tail.startswith('-ttl') and tail[4:].isdigit():
            limit = int(tail[4:]) * 3600
        else:
            print(p['id'], up, 'unparseable-deadline'); continue
    if up > limit:
        print(p['id'], up, 'past-deadline-%ds' % limit)
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
    rp_terminate "$id"
    echo "::warning::swept pod ${id} (${why}, up ${age}s)"
    n=$(( n + 1 ))
  done <<< "$out"
  echo "sweep: terminated ${n} orphaned pod(s)"
}

# Make the pod ready to build jammi: import the container ENV (already done by
# rp_run_remote), rehydrate the cargo registry from the prewarm object, point
# sccache at the object store, and place the repo at $1 (default: main).
#
# The cargo registry is the expensive cold cost — the CI image deliberately wipes
# /usr/local/cargo/registry, so a fresh pod otherwise re-resolves the whole
# arrow/datafusion/candle tree before it compiles a line. It is restored as ONE
# tarball rather than synced: RunPod's S3 surface degrades badly past ~10k
# objects, and a cargo registry is far past that.
rp_bootstrap() { # $1=git ref (optional)
  local ref="${1:-main}" s3=0
  rp_s3_load && s3=1
  rp_run_remote <<EOF
set -uo pipefail
export CARGO_HOME="\${CARGO_HOME:-/usr/local/cargo}"

# Pod-side tools the dev loop needs and the CI image has no reason to carry:
# zstd backs the prewarm tarball, rsync the working-tree sync, tmux the detached
# jobs that outlive the SSH session.
yum install -y zstd rsync tmux >/dev/null 2>&1 || echo "warn: pod tool install failed"

# One re-sourceable file that makes any later shell correct. The container's
# Dockerfile ENV is captured here rather than re-derived from /proc/1/environ at
# each use site, so \`attach\` and detached \`run\` jobs cannot drift from it.
{ while IFS= read -r -d '' __e; do
    printf 'export %s=%q\n' "\${__e%%=*}" "\${__e#*=}"
  done < /proc/1/environ; } > /root/.jammi_env

if [ "${s3}" = "1" ]; then
  export AWS_ACCESS_KEY_ID='${RP_S3_ACCESS_KEY_ID:-}'
  export AWS_SECRET_ACCESS_KEY='${RP_S3_SECRET_ACCESS_KEY:-}'
  export S3_ENDPOINT='${RP_S3_ENDPOINT:-}'
  export S3_BUCKET='${RP_S3_BUCKET:-}'

  if ! command -v s5cmd >/dev/null 2>&1; then
    curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${RP_S5CMD_VERSION}/s5cmd_${RP_S5CMD_VERSION}_Linux-64bit.tar.gz" \
      | tar -xz -C /usr/local/bin s5cmd 2>/dev/null || echo "warn: s5cmd install failed — running cold"
  fi

  # sccache reads the compile cache straight from the object store, so a pod in
  # any datacenter warms up. Wrapper is already set repo-wide in .cargo/config.toml.
  export SCCACHE_BUCKET="\$S3_BUCKET"
  export SCCACHE_ENDPOINT="\$S3_ENDPOINT"
  export SCCACHE_REGION=auto
  export SCCACHE_S3_USE_SSL=true
  export SCCACHE_S3_KEY_PREFIX=sccache
  { echo "export AWS_ACCESS_KEY_ID=\$(printf %q "\$AWS_ACCESS_KEY_ID")"
    echo "export AWS_SECRET_ACCESS_KEY=\$(printf %q "\$AWS_SECRET_ACCESS_KEY")"
    echo "export SCCACHE_BUCKET=\$(printf %q "\$SCCACHE_BUCKET")"
    echo "export SCCACHE_ENDPOINT=\$(printf %q "\$SCCACHE_ENDPOINT")"
    echo "export SCCACHE_REGION=auto"
    echo "export SCCACHE_S3_USE_SSL=true"
    echo "export SCCACHE_S3_KEY_PREFIX=sccache"; } >> /root/.jammi_env
fi

if [ ! -d /root/jammi-ai/.git ]; then
  git clone --filter=blob:none "${RP_REPO_URL}" /root/jammi-ai || exit 1
fi
cd /root/jammi-ai && git fetch --all --quiet && git checkout --quiet "${ref}" && git pull --quiet --ff-only 2>/dev/null

# Restore the registry only when this exact Cargo.lock has a prewarm object.
if [ "${s3}" = "1" ] && command -v s5cmd >/dev/null 2>&1; then
  KEY="registry/\$(sha256sum Cargo.lock | cut -c1-16).tar.zst"
  if s5cmd --endpoint-url "\$S3_ENDPOINT" cat "s3://\$S3_BUCKET/\$KEY" 2>/dev/null \
       | tar -x --zstd -C "\$CARGO_HOME" 2>/dev/null; then
    echo "registry prewarm restored (\$KEY)"
  else
    echo "no registry prewarm for this Cargo.lock — cold fetch (run 'gpu-dev.sh prewarm' to publish one)"
  fi
fi
echo "bootstrap complete: \$(cd /root/jammi-ai && git rev-parse --short HEAD)"
EOF
}

# Build and publish the registry prewarm object for the pod's current Cargo.lock.
# Idempotent: overwrites the key for that lock hash.
rp_prewarm_publish() {
  rp_s3_load || { echo "::error::no object store configured (${RP_S3_CONF}); nothing to publish"; return 1; }
  rp_run_remote <<EOF
set -uo pipefail
[ -f /root/.jammi_env ] && . /root/.jammi_env
export CARGO_HOME="\${CARGO_HOME:-/usr/local/cargo}"
cd /root/jammi-ai || exit 1
cargo fetch --locked || exit 1
KEY="registry/\$(sha256sum Cargo.lock | cut -c1-16).tar.zst"
tar -c --zstd -C "\$CARGO_HOME" registry \
  | s5cmd --endpoint-url '${RP_S3_ENDPOINT}' pipe "s3://${RP_S3_BUCKET}/\$KEY" || exit 1
echo "published \$KEY"
EOF
}
