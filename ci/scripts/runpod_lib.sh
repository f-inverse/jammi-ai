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
#                in git (your working tree, via `push`/`--ref`); build-time
#                COMPILATION state — the expensive part — lives in a per-pod
#                seed/clone build substrate instead of an S3-backed compile
#                cache (see pod_seed_target.sh, pod_target_clone.sh and
#                docs/maintainer/dev-gpu.md — a measured S3-backed sccache gave
#                ZERO cross-target-dir reuse on this image and cost ~+33%
#                wall). A RunPod network volume is deliberately NEVER
#                attached: an attached volume is Secure-Cloud-only and pinned to
#                a single datacenter, which would delete both failover
#                dimensions below — and intermittent A100 supply is precisely
#                why they exist.
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
#   RP_INACTIVITY seconds of silent (no new output byte) remote stdout+stderr
#                 before `rp_run_remote_watched` kills the ssh
#                 session as a hang, rather than waiting out the full
#                 RP_TIMEOUT budget (default 900 -- derived from the largest
#                 healthy in-window silence in the committed prove timings,
#                 285.2s on sm_90, x R2's 3x margin, rounded up to the next 300s step;
#                 see the setter's own comment below for the full
#                 derivation). This is a DIFFERENT axis
#                 from RP_TIMEOUT: a hung leg (e.g. a genuinely stuck test)
#                 goes silent well before any wall-clock budget expires, and
#                 a bare `timeout` has no notion of "still running but dead"
#                 vs "still running and busy" -- only `rp_run_remote` (no
#                 watchdog) is unaffected by this variable.
#   RP_SESSION    named session. Persists pod coordinates AND the SSH key under
#                 RP_SESSION_ROOT so a *different* terminal can reattach. Unset =
#                 throwaway temp dir, wiped on exit.
#   RP_KEEP       1 = leave the pod running when this process exits.
#   RP_TTL_HOURS  hard deadline baked into every pod at deploy (default 8 in
#                 this file). gpu-dev.sh's own `up` raises the default to
#                 RP_DEV_TTL_HOURS (72h) before sourcing this file, since a
#                 throwaway-pod default is too short for a dev session someone
#                 is actively using — an explicit RP_TTL_HOURS always wins.
#   RP_DISK_GB    container disk size in GB (default 60). Rule of thumb once the
#                 seed/clone build substrate is in use (see pod_seed_target.sh):
#                 RP_DISK_GB >= 25 (base) + S_src (one `git` source tree) +
#                 S_seed (one seed CARGO_TARGET_DIR) + N * S_clone (one clone
#                 per concurrent tree this pod hosts). The
#                 S_src/S_seed/S_clone byte counts are MEASURED, not guessed —
#                 see ci/scripts/perf/pod_build_timings.sh, the
#                 producer (dev-gpu.md quotes its S values). Add 3 GB per OTHER concurrent agent
#                 CARGO_TARGET_DIR sharing this pod and 2 GB per `cargo mutants
#                 -j N` job (COPY MODE makes one full workspace+target copy per
#                 job — mutation testing runs in COPY MODE, never
#                 `--in-place`, so a shared target dir does not report mutated
#                 sources as "Fresh"). A mutation-testing session wants >= 120 GB.
#   RP_VOLUME_GB  attached volume size in GB (default 0). The pod is deliberately
#                 disposable (see "state" above) — leave this 0 unless a caller
#                 has a specific reason to attach one.
#   RP_GPU_COUNT  GPUs per pod (default 1). Every lane that leaves it alone
#                 deploys a single-GPU pod; the topology lane
#                 (runpod_gpu_topology.sh) sets 2 per fleet host. A count > 1
#                 also reorders the arch candidate list — see
#                 _rp_order_candidates_for_gpu_count.
#   RP_SSH_WAIT_SECS  wall-clock deadline on rp_deploy_live's SSH-reachability
#                 poll, in seconds (default 600). A cold host still pulling the
#                 multi-GB CUDA image can take minutes before sshd is even up;
#                 raise this rather than losing a healthy pod to the poll's own
#                 timeout.
# Sets globals: RP_POD_ID, RP_HOST, RP_PORT, RP_REF. Installs an EXIT trap for
# teardown.

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (GitHub secret)}"
# The tree's own CUDA CI image (`ci_image.py`): a pod runs the environment the
# tree under test defines. A workflow passes it after waiting for it to exist.
RP_IMAGE="${RP_IMAGE:-$(python3 "$(dirname "${BASH_SOURCE[0]}")/ci_image.py" refs | sed -n 's/^cuda=//p')}"
# Minimum NVIDIA driver major the CUDA build needs: the image ships CUDA 12.6 PTX
# that the deployment driver JIT-compiles at model load, so a pod below r560
# (< CUDA 12.6) cannot run it — the engine's own startup driver floor rejects it
# and every model load fails. RunPod's fleet is mixed, so pod selection fails
# over past an under-floor pod rather than deploying onto one that cannot run.
RP_MIN_DRIVER_MAJOR="${RP_MIN_DRIVER_MAJOR:-560}"
RP_SESSION="${RP_SESSION:-}"
RP_KEEP="${RP_KEEP:-0}"
RP_TTL_HOURS="${RP_TTL_HOURS:-8}"
# The wall-clock bound on ONE `_rp_rest` REST v2 call. Without it, a caller
# sequencing work behind the call -- e.g. a cleanup trap's own fleet
# teardown -- hangs indefinitely on a dropped
# connection. 30s is generous above every
# observed RunPod REST latency in this file's own probes; a transport that
# has not completed by then is exactly the "no response at all" case
# `_rp_rest`'s own doc already documents as a nonzero return, never a status.
RP_REST_MAX_TIME="${RP_REST_MAX_TIME:-30}"
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
# How many GPUs ONE pod is rented with. Default 1: every lane that does not
# set this deploys the single-GPU pod it always did (gpu-prove, gpu-perf-ab,
# gpu-dev, howwell all inherit it — `test_pod_substrate.sh`'s own
# `(ab/gpuCount D5)` leg derives that set by scanning every tracked
# ci/scripts + .github/workflows file for this variable, so a lane that
# starts overriding it is named there rather than silently changing its own
# payload). The topology lane (`runpod_gpu_topology.sh`) is the one lane that
# asks for more: 2 GPUs on each fleet host.
#
# Same validation shape as RP_TTL_HOURS/RP_DISK_GB above: it lands in the
# GraphQL payload as an unquoted JSON number and drives an arithmetic
# comparison in `_rp_order_candidates_for_gpu_count`, with no `-e` set
# anywhere in this file.
RP_GPU_COUNT="${RP_GPU_COUNT:-1}"
case "$RP_GPU_COUNT" in
  ''|*[!0-9]*) echo "::error::RP_GPU_COUNT must be a positive integer (got '${RP_GPU_COUNT}')" >&2; exit 2 ;;
esac
[ "${#RP_GPU_COUNT}" -le 2 ] || { echo "::error::RP_GPU_COUNT has too many digits (got '${RP_GPU_COUNT}')" >&2; exit 2; }
RP_GPU_COUNT=$((10#$RP_GPU_COUNT))
[ "$RP_GPU_COUNT" -gt 0 ] || { echo "::error::RP_GPU_COUNT must be > 0" >&2; exit 2; }
# Wall-clock deadline for rp_deploy_live's SSH-reachability poll. Default 600s:
# a cold image pull alone has measured ~2 minutes, and a healthy candidate has
# needed over 4 minutes end to end — a 4-minute budget terminates pods that are
# still becoming reachable, not only ones that never would.
#
# Validated here, not at use, same reasoning as RP_TTL_HOURS above: it drives
# arithmetic (the deadline computed in rp_deploy_live) with no -e set
# anywhere in this file, so a non-integer would otherwise fail silently
# rather than loudly at the one place its value is known. The digit-only
# pattern match alone is not sufficient — an unquoted leading-zero operand
# (`08`) is read by bash arithmetic in its DEFAULT (octal) base, and
# `08`/`09` are not even legal octal digits, so `$(( ))` on a raw `08` is a
# fatal "value too great for base" error, not a misread — the read below is
# forced to base 10 with an explicit `10#` prefix instead of trusting the
# shell's own default base. The length check runs BEFORE that arithmetic, on
# the string's length alone: 9 digits is generous headroom above 600 while
# ruling out a decimal string long enough to overflow bash's signed 64-bit
# arithmetic in the `SECONDS + RP_SSH_WAIT_SECS` deadline computation.
RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-600}"
case "$RP_SSH_WAIT_SECS" in
  ''|*[!0-9]*) echo "::error::RP_SSH_WAIT_SECS must be a positive integer (got '${RP_SSH_WAIT_SECS}')" >&2; exit 2 ;;
esac
[ "${#RP_SSH_WAIT_SECS}" -le 9 ] || { echo "::error::RP_SSH_WAIT_SECS has too many digits (got '${RP_SSH_WAIT_SECS}')" >&2; exit 2; }
RP_SSH_WAIT_SECS=$((10#$RP_SSH_WAIT_SECS))
[ "$RP_SSH_WAIT_SECS" -gt 0 ] || { echo "::error::RP_SSH_WAIT_SECS must be > 0" >&2; exit 2; }
# Inactivity watchdog threshold for `rp_run_remote_watched`: a
# silent-output span this long (seconds, no NEW bytes on the remote's
# stdout+stderr stream) is read as a genuine hang, not merely a slow
# command, and kills the ssh session rather than waiting out the full
# RP_TIMEOUT budget. Default 900s: three times the longest silence a healthy
# prove leg has shown (the repository clone, under five minutes on sm_90),
# rounded up to the next 300s step. Validated here, not at use, same
# reasoning as RP_SSH_WAIT_SECS above: it drives arithmetic with no `-e`
# set anywhere in this file.
RP_INACTIVITY="${RP_INACTIVITY:-900}"
case "$RP_INACTIVITY" in
  ''|*[!0-9]*) echo "::error::RP_INACTIVITY must be a positive integer (got '${RP_INACTIVITY}')" >&2; exit 2 ;;
esac
[ "${#RP_INACTIVITY}" -le 9 ] || { echo "::error::RP_INACTIVITY has too many digits (got '${RP_INACTIVITY}')" >&2; exit 2; }
RP_INACTIVITY=$((10#$RP_INACTIVITY))
[ "$RP_INACTIVITY" -gt 0 ] || { echo "::error::RP_INACTIVITY must be > 0" >&2; exit 2; }
RP_SESSION_ROOT="${RP_SESSION_ROOT:-${HOME}/.config/runpod/sessions}"
# An ssh config this tooling owns outright, so ~/.ssh/config is never rewritten.
RP_SSH_CONFIG="${RP_SSH_CONFIG:-${HOME}/.config/runpod/ssh_config}"
RP_REPO_URL="${RP_REPO_URL:-https://github.com/f-inverse/jammi-ai}"

# A named session becomes a directory under RP_SESSION_ROOT and, in
# rp_cleanup/rp_session_forget below, the target of an UNCONDITIONAL
# `rm -rf "$RP_WORK"`. RP_SESSION reaches here from gpu-dev.sh's own SESSION
# resolution (an arch name, a caller-supplied alias, or an exported
# environment variable) with no sanitization upstream, so a value containing
# a path separator or a `.`/`..` segment could point RP_WORK — and therefore
# that `rm -rf` — OUTSIDE RP_SESSION_ROOT entirely (`RP_SESSION=../../etc`).
#
# The ONE name rule every operator-supplied identifier in this tooling must
# satisfy — session, tree, and wave alike. All three end up in the same two
# dangerous positions: a LOCAL path (RP_WORK, and the unconditional
# `rm -rf "$RP_WORK"` in rp_cleanup/rp_session_forget below — `RP_SESSION=
# ../../etc` would point that removal outside RP_SESSION_ROOT entirely) and
# REMOTE shell text (a tree name reaches `run`'s `.jammi-job.sh` dispatch,
# `target --with-cutlass`, and `wait-job`'s check script; a wave and a tree
# reach the active-wave claim write), where a quote, a backtick, a `$(...)`,
# or a `%` in a printf format position is arbitrary code or a corrupted
# claim on the pod.
#
# Refused, in this order so each shape gets its OWN diagnosis:
#   empty / `.` / `..`   — resolve outside their own root (this function
#                          only ever runs for a NAMED session, see the
#                          RP_SESSION_VALIDATE_SESSION gate below, so an
#                          empty value here is a caller bug, never
#                          `shell`'s genuinely anonymous RP_SESSION="")
#   contains `/`         — a multi-segment path
#   leading `-`          — reads as an option to any tool the name is later
#                          passed positionally
#   anything outside `[A-Za-z0-9._-]` — the allowlist proper
#
# `.` is INSIDE the allowlist, deliberately: a dotted name (`bench.1`,
# `a100.2`, `fix.443`) is a legitimate, in-use shape, and a rule that
# refused it would refuse every verb against a session that had already
# rented a real pod — stranding that pod for its full deadline with no
# `down` able to reach it. `_` and `-` are in for the same reason (every
# arch/tree/wave name in use). Nothing outside those four classes has ever
# named a real session, tree, or wave in this repo, and each excluded
# character is exactly what makes the two dangerous positions dangerous.
# $1=the label to name in the error (e.g. RP_SESSION, --tree, --wave)
# $2=the value.
rp_name_allowlist_check() {
  local label="${1:?rp_name_allowlist_check needs a label}" value="${2-}"
  case "$value" in
    ''|.|..)
      echo "::error::${label} may not be empty, '.', or '..' (got '${value}')" >&2
      return 2 ;;
    */*)
      echo "::error::${label} may not contain '/' (got '${value}')" >&2
      return 2 ;;
    -*)
      echo "::error::${label} may not start with '-' (got '${value}')" >&2
      return 2 ;;
    *[!A-Za-z0-9._-]*)
      echo "::error::${label} may contain only letters, digits, '.', '_' and '-' (got '${value}') — it is embedded in remote shell text and in a printf argument" >&2
      return 2 ;;
  esac
}

rp_session_name_check() {
  rp_name_allowlist_check RP_SESSION "$RP_SESSION"
}
# Gated on RP_SESSION_VALIDATE_SESSION, set by gpu-dev.sh only for the verbs
# that actually RESOLVE a named session (up/attach/run/logs/push/pull/down)
# — never `ls`/`reap` (account-level; they never read RP_SESSION at all,
# and source this file from their OWN early dispatch branch before this
# variable would ever be set) and never `shell` (deliberately anonymous,
# RP_SESSION force-cleared to "" before sourcing). Without this gate, an
# UNRELATED exported RP_SESSION sitting in a maintainer's own shell for
# some other purpose would make `ls`/`reap` refuse outright even though
# neither verb consumes it.
if [ "${RP_SESSION_VALIDATE_SESSION:-0}" = "1" ]; then
  rp_session_name_check || exit $?
fi

# A named session must outlive the process; an anonymous one must not leak. The
# session dir is created lazily by the first writer, so read-only commands
# against a session that does not exist leave nothing behind.
if [ -n "$RP_SESSION" ]; then
  RP_WORK="${RP_SESSION_ROOT}/${RP_SESSION}"
  RP_WORK_IS_TEMP=0
else
  # An explicit path TEMPLATE ("$dir/prefix.XXXXXX"), not a bare `mktemp -d`,
  # so the base directory is resolved the SAME way on every `mktemp`
  # implementation this tooling runs under: GNU `mktemp -d` (Linux CI) reads
  # $TMPDIR itself, but BSD `mktemp -d` (macOS, a maintainer's own laptop)
  # does NOT — it ignores $TMPDIR entirely for a bare `-d` with no template
  # and always resolves under its own darwin temp root, so a bare call would
  # behave differently across the two platforms this tooling actually runs
  # on. Building the base path ourselves from "${TMPDIR:-/tmp}" removes that
  # divergence, defaults identically to the previous bare call (`/tmp` when
  # $TMPDIR is unset, which is the common case), and is what makes the
  # temp-dir-isolation regression test in test_gpu_dev_lifecycle.sh able to
  # observe which root a throwaway `shell` invocation's RP_WORK actually
  # resolved under.
  #
  # A failed `mktemp` (e.g. a nonexistent/unwritable $TMPDIR) must abort
  # here, not fall through with RP_WORK unset/empty: every later use of
  # RP_WORK (the ssh key path, the meta path, and eventually the pod
  # deploy) would then operate on a garbage or empty path instead of
  # failing loudly at the one place the real cause is known.
  RP_WORK="$(mktemp -d "${TMPDIR:-/tmp}/jammi-gpu-dev.XXXXXX")" \
    || { echo "::error::could not create a temp work dir under ${TMPDIR:-/tmp}" >&2; exit 2; }
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
# on "RP_POD_ID is non-empty": a read-only subcommand against an
# unreachable session leaves RP_POD_ID set from the loaded session and exits
# 1, and without this flag the EXIT trap would terminate a pod that
# invocation never rented.
RP_POD_ID=""; RP_HOST=""; RP_PORT=""; RP_PUBKEY=""; RP_ARCH=""; RP_SSHO=(); RP_POD_CREATED=0
# The git ref the pod's checkout sits on. Two sites keep it honest, so recorded
# state never claims a ref the pod is not on: rp_bootstrap sets it only after the
# checkout succeeded, and rp_deploy_live clears it the moment pod identity
# changes — otherwise a dead session's ref survives into the new pod's record.
RP_REF=""

# An SSH login shell does NOT inherit the container's Dockerfile ENV (CC=gcc-13,
# PATH with cuda+mold+rust). Every remote job imports PID 1's real environment.
RP_ENV_PREAMBLE='while IFS= read -r -d "" __e; do export "$__e"; done < /proc/1/environ'

# The directory a rented host's remote job works under. /root on a real pod;
# a lane suite that executes a leg's remote text points it at a sandbox and
# runs the text UNMODIFIED — never a string patch of the script under test.
RP_REMOTE_ROOT="${RP_REMOTE_ROOT:-/root}"

rp_gql() { curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$1"; }

# The REST v2 primitive (`https://api.runpod.io/v2/...`, Bearer auth) the
# fleet primitives below are built on, beside the pod surface's `rp_gql`.
#
# Body and HTTP status are captured SEPARATELY — the body via `-o` to a
# throwaway file, the status via `-w '%{http_code}'` into its OWN command
# substitution — never a single piped read. Under `set -o pipefail` (every
# caller of this file), `curl ... | python3 -c ...` would report CURL's own
# exit code whenever curl itself failed, even when it still delivered a
# complete, parseable body — the identical class rp_pod_verify's own doc
# (above) closes for the GraphQL side, reproduced here for REST.
#
# Prints "STATUS\n<body>" (status on line 1, the raw response body on every
# line after — a caller splits with `head -n1`/`tail -n +2`). Returns curl's
# own exit code: 0 means the REQUEST completed and SOME HTTP status came
# back (2xx, 4xx, 5xx — the caller's own per-arm lattice decides what that
# status means); nonzero means a TRANSPORT failure (no response at all — DNS,
# TLS, a dropped connection). A caller must never read a nonzero return here
# as "not found" or "refused"; those are STATUSES on a successful transport,
# never this function's own return code.
#
# Bounded by `RP_REST_MAX_TIME`: a caller that sequences its OWN cleanup
# behind a REST call here (e.g. a cleanup trap's own fleet teardown) has
# nothing else to time out a dropped connection. A transport that has not
# completed within the bound is exactly the "TRANSPORT failure" case above,
# never a status.
#
# $1=METHOD (GET/POST/PATCH/DELETE) $2=PATH (e.g. "/v2/pods", leading
# slash) $3=optional JSON BODY (POST/PATCH only).
_rp_rest() {
  local method="${1:?_rp_rest needs a METHOD}" path="${2:?_rp_rest needs a PATH}" body="${3-}"
  local body_file status rc
  body_file="$(mktemp "${TMPDIR:-/tmp}/jammi-rp-rest.XXXXXX")" \
    || { echo "::error::_rp_rest could not create a capture file" >&2; return 1; }
  if [ -n "$body" ]; then
    status="$(curl -s --max-time "$RP_REST_MAX_TIME" -o "$body_file" -w '%{http_code}' -X "$method" "https://api.runpod.io${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$body")"
  else
    status="$(curl -s --max-time "$RP_REST_MAX_TIME" -o "$body_file" -w '%{http_code}' -X "$method" "https://api.runpod.io${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}")"
  fi
  rc=$?
  printf '%s\n' "$status"
  cat "$body_file"
  rm -f "$body_file"
  return "$rc"
}

# $1=podId. Returns 0 when the mutation's response carries no `errors`.
# Returns 1 — a REFUSAL — when the body DOES carry `errors`, OR when
# the body could not be parsed as a JSON object AT ALL (a transport hiccup,
# an HTML error page, a truncated response): an unparseable body is a refusal
# too, never a silent "no errors" success — otherwise an HTML 502 page from
# podTerminate would be reported as a genuinely SWEPT pod. PRINTS the refusal
# reason to stdout on the refusal arm ONLY, so a caller that wants it
# (rp_sweep, below) can capture it via `$(rp_terminate "$id")` while every
# other caller (which never captures this function's stdout) is unaffected.
# `rp_cleanup`'s EXIT-trap teardown and the two capacity-failover call sites
# in `rp_deploy_live` never check the return code, so a network hiccup
# mid-exit is not fatal there; only rp_sweep captures and reports this
# function's stdout and rc, counting an unparseable body as a refused
# terminate.
rp_terminate() {
  local id="${1:?rp_terminate needs a podId}" body reason rc
  body="$(rp_gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${id}\\\"}) }\"}" 2>/dev/null)"
  reason="$(printf '%s' "$body" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("podTerminate response was unparseable")
    sys.exit(1)
if not isinstance(d, dict):
    print("podTerminate response was not a JSON object")
    sys.exit(1)
e = d.get("errors")
if e:
    print(" ".join((e[0].get("message") or "unknown").split())[:200])
    sys.exit(1)
sys.exit(0)
' 2>/dev/null)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '%s\n' "${reason:-unknown reason}"
    return 1
  fi
  return 0
}

# Confirm a pod id is BOTH present in the account's live pod list AND carries
# a name shaped like one of THIS TOOLING'S OWN pods ("<prefix>-ttl<digits>")
# before any caller acts on it irreversibly. Never trusts a locally-recorded
# id on its own: the id could be stale (the account-side pod is already
# gone), or a DIFFERENT pod could now be recorded under this session's name
# (two processes racing `up` on the same alias). $1=podId.
#
# The id is the AUTHORITATIVE half of this check; the exact TTL never gates
# release, and this function takes none. Matching an EXACT
# "<prefix>-ttl<H>" against a recorded TTL would make `down` refuse to
# release a real `jammi-gpu-ttl72` pod whose meta file carries no TTL,
# billing its full 72h — and an `RP_TTL_HOURS=<H>` override cannot rescue
# that, because `rp_session_load`'s meta dot-source runs before the override
# is read.
# RunPod pod ids are globally unique — if the recorded id names a REAL pod
# in the account, that pod IS the one with that id; there is no id-level
# ambiguity a TTL number could ever have resolved. The name-shape check
# alone already establishes "this is one of our own jammi-gpu* pods" (as
# opposed to some entirely unrelated pod in the account); the specific
# number never added a real safety margin once the id already matched.
#
# Prints the account's name for the id on success. Returns 1 (present, but
# the name is not this tooling's shape — a REAL ambiguity, "do not act, never
# assume safe"), 2 (could not query the account at all — same "do not act"),
# or 3 (id ABSENT from the account entirely). 3 is deliberately its OWN code,
# not folded into 1: an absent id is NOT an ambiguity to refuse on — it is
# the ordinary, expected shape of "this pod already ended on its own" (its
# in-pod deadline, or the sweep), the single most common way a session's
# pod goes away, and `down`'s caller treats it as a normal cleanup, not a
# refusal.
#
# Piped input (the account's own pod list) and script source cannot both come
# from stdin — `python3 -` with a heredoc reads the heredoc AS THE PROGRAM,
# leaving nothing for the piped JSON to land on (shellcheck SC2259). This
# therefore uses `python3 -c '<script>' argv...`, the same shape
# `rp_deploy_live`'s own parser already uses, so the piped JSON reaches
# `sys.stdin` intact and dynamic values travel as argv, never interpolated
# into the script source.
#
# The query is captured into `body` FIRST, with its own `||` gate, rather
# than piped straight into the parser (`rp_gql ... | python3 -c ...`). Every
# caller of this file runs under `set -o pipefail`, where a pipeline's exit
# status is the LAST command to fail non-zero — so a direct pipe would report
# rp_gql's (curl's) own exit code whenever curl itself failed, EVEN IF curl
# still delivered a complete, parseable body and the python parser reached a
# real verdict. That collides with this function's own return codes, which
# are not all equivalent: code 3 means "id confirmed ABSENT — ordinary
# cleanup, `down` forgets the record", not "could not query". A curl exit
# that happens to also be 3 would then read as a confirmed absence and make
# `down` forget a session record for a pod the account list still actually
# shows present — the opposite of "do not act, never assume safe" this
# function exists to enforce. Splitting the capture from the parse removes
# the collision entirely: a curl failure returns 2 ("could not query")
# unconditionally, and only a successful query ever reaches the parser, whose
# own exit code is then this function's exit code with nothing to alias.
rp_pod_verify() { # $1=podId
  local id="$1" body
  body="$(rp_gql '{"query":"query{ myself{ pods{ id name } } }"}')" || return 2
  printf '%s' "$body" | python3 -c '
import sys, json, re
podid, prefix = sys.argv[1], sys.argv[2]
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
shape = re.compile(r"^%s-ttl[0-9]+$" % re.escape(prefix))
for p in me["pods"]:
    if p.get("id") == podid:
        name = p.get("name") or ""
        if shape.match(name):
            print(name)
            sys.exit(0)
        print("pod %s account name %s is not this tooling shape %s-ttl<digits> -- refusing to act on it" % (podid, name, prefix), file=sys.stderr)
        sys.exit(1)
print("pod %s is not in the account pod list" % podid, file=sys.stderr)
sys.exit(3)
' "$id" "$RP_POD_PREFIX"
}

# Confirm a pod id is ABSENT from the account's live pod list — the only
# signal `down` trusts that a `rp_terminate` mutation actually took. Never
# assumed from rp_terminate's own return: that call throws its response away
# (`>/dev/null 2>&1`) by design, since it also runs as rp_cleanup's
# best-effort EXIT-trap teardown, where a network hiccup must not turn a
# normal shell exit into a hard failure. `down` is a single, deliberate,
# foreground action instead, and gets to demand confirmation before it
# forgets the local record — a rejected `podTerminate` must never both leak
# the pod AND destroy the only record pointing at it.
#
# $1=podId. Returns 0 (confirmed gone — absent from the account's pod list
# entirely) or 1 (still present, OR the query itself failed — both are "not
# confirmed", never "must have succeeded anyway").
#
# Assumes `myself{ pods{...} } }` returns the account's COMPLETE pod list in
# one response — RunPod's schema documents no pagination on this field, and
# it was verified live against a real account on first use (this tooling has
# never observed a truncated list). "Absent from the returned list" and
# "absent from the account" are the same fact only under that assumption; if
# RunPod ever pages this field, both this function's code-0 and
# `rp_pod_verify`'s code-3 would need to change together, since they read
# the account's pod list the same way for the same reason.
rp_pod_gone() { # $1=podId
  local id="$1" body
  # Same capture-then-parse shape as rp_pod_verify's own fix (see its doc): a
  # direct `rp_gql | python3` pipe would report curl's own exit code under
  # `set -o pipefail` whenever curl fails, even if the body it still
  # delivered was complete. This function only has two return codes (0
  # confirmed-gone, 1 everything else), so a curl failure landing on 1 was
  # already the correct answer either way — there was no code-3-shaped
  # collision to alias with, unlike rp_pod_verify. Applied anyway so this
  # function does not depend on pipefail's own last-failing-command rule to
  # get the right answer by coincidence, and so both functions read the
  # account the same way for the same reason.
  body="$(rp_gql '{"query":"query{ myself{ pods{ id } } }"}')" || return 1
  printf '%s' "$body" | python3 -c '
import sys, json
podid = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("could not parse RunPod response: %s" % e, file=sys.stderr)
    sys.exit(1)
if d.get("errors"):
    print(json.dumps(d["errors"])[:200], file=sys.stderr)
    sys.exit(1)
me = (d.get("data") or {}).get("myself")
if me is None or me.get("pods") is None:
    print("response contained no pod list", file=sys.stderr)
    sys.exit(1)
for p in me["pods"]:
    if p.get("id") == podid:
        print("pod %s is still present in the account pod list" % podid, file=sys.stderr)
        sys.exit(1)
sys.exit(0)
' "$id"
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
  # IdentitiesOnly=yes pins every connection to exactly $RP_SSH_KEY. Without
  # it, ssh offers every identity ssh-agent already holds BEFORE this key —
  # on macOS the agent auto-adds each session key it uses, so once it holds
  # more than a handful the reachability probe below can exhaust the
  # server's MaxAuthTries before $RP_SSH_KEY is ever tried, reading a
  # perfectly healthy, reachable pod as unreachable and terminating it.
  # Observed on a kept candidate: sshd was up and `-o IdentitiesOnly=yes`
  # connected cleanly while the agent held 12 identities.
  # -oServerAliveInterval=30 -oServerAliveCountMax=6 (attached
  # form) keep the client's NAT state alive across long, LEGITIMATELY
  # silent remote phases (clone/build) — a healthy session must not be torn
  # down by an idle-TCP window, and a genuinely dead connection is still
  # declared within ~3 minutes (surfacing as ssh's own exit 255, reported
  # verbatim by every caller). Keepalives do NOT mask hangs: the inactivity
  # watchdog (`rp_run_remote_watched`) measures remote OUTPUT bytes, never
  # TCP liveness, so a session that stays connected but produces nothing
  # still trips 76 on schedule. `rp_wait_poll` (below) PREPENDS its own
  # tighter probe options ahead of this array, so its liveness contract
  # (10s/3 tries — a probe wants to fail FAST, not survive a long silence)
  # is unaffected by this shared, looser session-liveness default.
  RP_SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o IdentitiesOnly=yes -oServerAliveInterval=30 -oServerAliveCountMax=6 -i "$RP_SSH_KEY")
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

# THE readiness predicate for a rented host: its sshd answers an executed
# connect. RunPod's `RUNNING` status and a mapped port say the CONTAINER is
# up; the image's entrypoint installs openssh-server after boot
# (`_rp_entrypoint_setup`), so the mapped port refuses connections for tens
# of seconds first (measured on a two-host Global-Networking pair: both pods
# RUNNING and GN-enabled at 224 s, the first ssh refused 0.1 s later). Every
# leg decides "usable" with this ONE probe -- never with the API status.
#   rp_sshd_answers <host> <port> [extra ssh options...]
rp_sshd_answers() {
  local host="${1:?rp_sshd_answers needs a host}" port="${2:?rp_sshd_answers needs a port}"
  shift 2
  ssh "${RP_SSHO[@]}" "$@" -p "$port" "root@${host}" true 2>/dev/null
}

# The bounded wait on that predicate, for a host whose endpoint is already
# known: probe every 5 s until sshd answers or `bound` seconds elapse.
# Prints one line on success; on the bound a named `::error::` and rc 1 --
# the caller never issues a remote command after rc 1.
#   rp_wait_sshd <host> <port> <bound-seconds> <label> [extra ssh options...]
rp_wait_sshd() {
  local host="${1:?rp_wait_sshd needs a host}" port="${2:?rp_wait_sshd needs a port}" \
        bound="${3:?rp_wait_sshd needs a bound in seconds}" label="${4:?rp_wait_sshd needs a label}"
  shift 4
  local deadline=$(( SECONDS + bound )) started=$SECONDS attempts=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    attempts=$(( attempts + 1 ))
    if rp_sshd_answers "$host" "$port" "$@"; then
      echo "sshd up on ${label} (${host}:${port}) after $(( SECONDS - started ))s, ${attempts} probe(s)"
      return 0
    fi
    sleep 5
  done
  echo "::error::sshd on ${label} (${host}:${port}) answered no connect within ${bound}s (${attempts} probes) -- the pod is RUNNING but its entrypoint has not started sshd; refusing to issue a remote command" >&2
  return 1
}

# A recorded pod is not a live pod — the reaper may have collected it, or the
# host may have died. Callers must confirm before treating a session as usable.
rp_session_alive() {
  [ -n "$RP_POD_ID" ] && [ -n "$RP_HOST" ] && [ -n "$RP_PORT" ] || return 1
  rp_sshd_answers "$RP_HOST" "$RP_PORT"
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

# A `--tree` value reaches gpu-dev.sh with no sanitization upstream, and —
# unlike RP_SESSION, which is used only for LOCAL file paths and ssh config
# blocks — a tree name (via TREE_DIR/TARGET_DIR/TMUX_SESSION, all derived
# from it) gets embedded UNQUOTED into several REMOTE heredoc scripts sent
# over ssh (`run`'s own `.jammi-job.sh` dispatch, `target --with-cutlass`,
# and `wait-job`'s own check script, gpu_dev_job_wait_script). A value
# containing a double-quote, backtick, `$(...)`, or a path separator could
# break out of that heredoc and inject commands into the remote shell; every
# new heredoc site that embeds a tree name joins the same exposure.
# Applies rp_name_allowlist_check, the SAME rule
# rp_session_name_check applies: a tree name is a directory-name-shaped
# string with the identical legal shapes a session name has, so the
# identical rule applies for the identical reason. Reads
# `$RP_TREE_CHECK_VALUE` (the same "check a global, not a parameter" shape
# rp_session_name_check itself uses, so a caller resolves it exactly the
# same way both times).
rp_tree_name_check() {
  rp_name_allowlist_check --tree "$RP_TREE_CHECK_VALUE"
}

# A wave id reaches the pod as TEXT in the active-wave claim
# (rp_job_wrapper_with_marker_lines) and as a literal in the concurrency
# preflight's own comparison (rp_concurrency_preflight_lines), so it carries
# the same exposure a tree name does and gets the same rule. Reads
# `$RP_WAVE_CHECK_VALUE`, matching rp_tree_name_check's own shape.
rp_wave_name_check() {
  rp_name_allowlist_check --wave "$RP_WAVE_CHECK_VALUE"
}

# Tree name -> plain SOURCE checkout directory on the pod. "jammi-ai" is the
# ONE default: the single-checkout location docs and scripts name directly
# (rp_bootstrap's own clone destination,
# below, is the other of the exactly two "/root/jammi-ai" literal sites this
# tooling permits — see test_pod_substrate.sh's grep gate). Any OTHER name
# is a caller-chosen additional tree — a plain directory under /root/trees,
# never a git worktree (a worktree add fails on the checked-out ref, and a
# shared .git couples trees that must be able to diverge). A tree is
# populated by `push` (rsync, excludes `.git`) — NEVER by cloning the
# build-substrate seed: the seed is a CARGO_TARGET_DIR (build OUTPUT), a
# wholly different directory namespace from a tree (SOURCE) — see
# rp_target_dir, immediately below. Conflating the two makes `target`'s own
# clone destination collide with `push --tree`'s rsync destination, so the
# first push after a `target` deletes the clone it had just made.
rp_tree_dir() { # $1=tree name (optional; default "jammi-ai")
  local t="${1:-jammi-ai}"
  if [ "$t" = "jammi-ai" ]; then echo "/root/jammi-ai"; else echo "/root/trees/${t}"; fi
}

# Tree name -> the CARGO_TARGET_DIR a `target` clone for that tree lives at
# — a build OUTPUT namespace (/root/target-<name>), deliberately disjoint
# from rp_tree_dir's SOURCE namespace (/root/trees/<name> or
# /root/jammi-ai) so `push --tree <name>`'s `rsync --delete` (which mirrors
# the SOURCE tree only) can never reach it, and `target`'s own clone can
# never be mistaken for — or overwritten by — a checkout. Applies the SAME
# "jammi-ai" naming convention as rp_tree_dir purely for symmetry (there is
# no special-cased literal here, unlike rp_tree_dir's default branch: every
# name, including "jammi-ai", maps the same way).
rp_target_dir() { # $1=tree name (optional; default "jammi-ai")
  local t="${1:-jammi-ai}"
  echo "/root/target-${t}"
}

# rsync creates only the LAST path component of its own destination — it
# never mkdir -p's a whole missing chain. Nothing in the pod bootstrap or
# the build-substrate seed provisions /root/trees itself (only
# /root/jammi-ai, the default tree, exists from bootstrap), so the very
# FIRST `push --tree <name>` against a name no session has ever pushed
# before would fail outright on a fresh pod: `rsync: mkdir "/root/trees/<name>"
# failed: No such file or directory (2)` — a "push first" flow gpu-dev.sh's own header
# doc and every recipe in dev-gpu-recipes.md document as the FIRST step for
# a new tree. rp_push_ensure_parent issues a tiny, bounded remote `mkdir
# -p` on the tree's PARENT directory (derived from tree_dir, never passed
# separately, so a caller can never name a parent that disagrees with the
# tree it is about to push into) over the SAME rp_run_remote primitive
# every other pod-reaching verb in this file uses — idempotent: `mkdir -p`
# against an already-existing parent (e.g. /root, the default "jammi-ai"
# tree's own parent, already present from bootstrap) is a silent no-op, so
# this runs unconditionally on every push, not just a tree's first one.
# $1=tree_dir (the FULL tree path, e.g. /root/trees/mytree or
# /root/jammi-ai).
rp_push_ensure_parent() {
  local tree_dir="${1:?rp_push_ensure_parent needs a tree dir}"
  local parent_dir="${tree_dir%/*}"
  [ -n "$parent_dir" ] || parent_dir="/"
  rp_run_remote <<EOF
set -uo pipefail
mkdir -p '${parent_dir}'
EOF
}

# The shell TEXT that classifies $1 (a CARGO_TARGET_DIR path, e.g.
# TARGET_DIR from rp_target_dir) — `.jammi-clone-of-seed` is the marker
# `pod_target_clone.sh` stamps on every successful clone or adoption. A
# plain function, not inlined by hand into gpu-dev.sh's `run` heredoc, so
# it is directly testable: source this file, call it with a real local
# directory, and run the printed text (`bash <
# <(rp_target_preflight_lines "$dir")`) against real fixture directories —
# no ssh, no pod, no mocking. Prints exactly one line,
# `GPU_DEV_TARGET_STATE=<MISSING|UNMARKED_COLD|UNMARKED_WARM|OK>`, so the
# caller's remote execution of this same text (over `rp_run_remote`) is
# trivially parsed by a `case` at the call site. $1=target_dir.
#
# The two UNMARKED states are DIFFERENT facts and get different diagnoses
# and different remedies at the call site. "No marker" does not imply "it
# was never provisioned via pod_target_clone.sh, so this job would pay a
# COLD full workspace build" — a target dir that predates the marker scheme
# (or was built by any path other than the `target` verb) is genuinely
# WARM, the job would NOT rebuild the workspace,
# and the offered remedy (`target ... --with-cutlass`) cannot even run,
# since pod_target_clone.sh refuses to clone over an existing destination.
# WARM is decided on the ONE piece of evidence that actually answers "would
# this build be incremental": workspace-member fingerprints under a cargo
# profile directory. It is a cheap, structural signal for the DIAGNOSIS
# only — `target --adopt`, the executable remedy, re-checks the same
# content with pod_seed_assert_member_free (the authoritative scan, off the
# real workspace member list) before it stamps anything.
rp_target_preflight_lines() {
  local target_dir="${1:?rp_target_preflight_lines needs a target dir}"
  cat <<EOF
if [ ! -d '${target_dir}' ]; then
  echo GPU_DEV_TARGET_STATE=MISSING
elif [ ! -f '${target_dir}/.jammi-clone-of-seed' ]; then
  if ls '${target_dir}/debug/.fingerprint' '${target_dir}/release/.fingerprint' 2>/dev/null | grep -c '^jammi' >/dev/null; then # tripwire-ok: '-c' (never '-q') so a huge fingerprint listing can't SIGPIPE the 'ls' writer on an early match -- PRE-EXISTING, fail-closed asymmetry, not this line's own claim: the real call site (gpu-dev.sh's \`rp_run_remote <<EOF / set -uo pipefail\` wrapper) runs this under pipefail, so when only ONE of debug/release exists, 'ls' itself exits nonzero for the missing side and poisons the pipeline regardless of what grep found -- a debug-only or release-only WARM dir reads UNMARKED_COLD, never the reverse (a genuinely cold dir never reads WARM); the safe direction (a needless full rebuild) is the one this asymmetry falls on
    echo GPU_DEV_TARGET_STATE=UNMARKED_WARM
  else
    echo GPU_DEV_TARGET_STATE=UNMARKED_COLD
  fi
else
  echo GPU_DEV_TARGET_STATE=OK
fi
EOF
}

# The pod-global one-pod-per-wave claim store, named ONCE here and
# interpolated into every generated remote script that touches it (the
# reader below, the writer in rp_job_wrapper_with_marker_lines).
#
# A DIRECTORY of per-holder claims, never one shared file: same-wave
# co-tenancy is the sanctioned shape (a wave's sub-units sharing one warm
# seed), so N holders are live at once and a single file can only ever
# record the last writer. With one file, holder B's launch overwrites
# holder A's claim and holder A's completion then deletes the claim B is
# still relying on — after which the pod, still genuinely busy, reads as
# "no claim" to the next `run`. One file per holder, keyed by the holder's
# TREE (the pod runs at most one job per tree: `run` kills any existing
# `jammi-<tree>` session before starting a new one, so a same-tree re-run
# refreshes its own claim in place rather than colliding), makes every
# holder independently visible and independently removable.
#
# Every write and every removal happens inside a flock on RP_CLAIM_LOCK, so
# a reader never observes a half-written claim and two launching holders
# never race on the store's creation. RP_CLAIM_LOCK is a separate lock from
# `/root/.jammi-timing.lock` (which serialises whole JOBS for `--timing`);
# this one is held only for the length of a single claim write or removal.
RP_CLAIM_DIR='/root/.jammi-active-wave.d'
RP_CLAIM_LOCK='/root/.jammi-active-wave.lock'

# The shell TEXT that checks whether this pod is genuinely BUSY with a
# DIFFERENT wave's job — the one-pod-per-wave rule, mechanized rather than
# left to prose. Scoped to WAVE rather than tree: a tree-scoped gate trips a
# single wave against ITSELF the moment it uses a second tree, although a
# wave's own jobs can safely share one warm seed. Two signals, tmux liveness
# PRIMARY:
#   1. Is any OTHER jammi-* tmux JOB session alive at all (`jammi-seed`, the
#      boot-time build-substrate seed, and this tree's OWN session are
#      excluded)? If not, CLEAR — regardless of what any claim file happens
#      to still say (a claim whose session already ended is definitionally
#      stale: the wrapper that wrote it, rp_job_wrapper_with_marker_lines,
#      removes it at BOTH of its own exit paths, so a leftover claim with
#      no live session can only mean the wrapper's own cleanup was
#      interrupted — SIGKILL/pod death — never a job that finished
#      normally. Failing OPEN on that staleness, rather than refusing
#      forever on an orphaned file, is the deliberate choice here: tmux's
#      own liveness check is what actually answers "is the pod busy right
#      now", and a stale claim answers nothing).
#   2. Only if another session IS alive: aggregate EVERY holder's claim in
#      RP_CLAIM_DIR, skipping this session's own and skipping any holder
#      whose own `jammi-<TREE>` session is no longer alive (per-holder
#      staleness, the same liveness rule signal 1 applies to the pod as a
#      whole). Same-wave co-tenancy is sanctioned, so ALL surviving holders
#      must name THIS wave for CLEAR; the FIRST holder naming a different
#      wave is reported BUSY, naming that holder's own wave and session
#      rather than an arbitrary "some other session". No surviving holder
#      at all (a job launched by tooling that predates the claim store, or
#      a session started outside `run` entirely) -> BUSY:UNKNOWN, failing
#      CLOSED on the ambiguity rather than guessing it is safe to share.
# $1=this tree's own tmux session name (e.g. "jammi-mywork", TMUX_SESSION
# at the call site) $2=this run's own wave id (WAVE at the call site,
# defaults to the tree name — see gpu-dev.sh's own WAVE resolution). A
# plain function, directly testable the same way rp_target_preflight_lines
# is (source this file, run the printed text against a faked `tmux` + a
# fixture claim store on PATH/disk — no ssh, no pod). Prints exactly one
# line:
#   GPU_DEV_CONCURRENCY_STATE=CLEAR
#   GPU_DEV_CONCURRENCY_STATE=BUSY:<owning-wave-or-UNKNOWN>:<other-session>
# NOTE on the heredoc body below: this is TEXT generated for REMOTE
# execution, not this file's own bash — every escaped `\$` is deliberate
# (it survives THIS function's own heredoc expansion unexpanded and
# resolves only when the REMOTE shell runs it, same pattern as
# rp_target_preflight_lines) — shellcheck's static parse of the heredoc
# body cannot know that and reads an escaped form used in a `-n`/`-z` test
# as a hardcoded non-empty literal (SC2157), hence the disable directives
# placed INSIDE the heredoc, immediately before each flagged line. No
# backticks appear anywhere in the heredoc body itself — an unquoted
# heredoc treats a literal backtick as old-style command substitution,
# which would corrupt the generated text.
#
# Both session exclusions are `grep -Fvx -e` — a LITERAL, whole-line,
# option-safe match — never a bare `grep -vx PATTERN`. A session name is a
# tree name (`jammi-<tree>`), and a tree name legitimately contains `.`
# (rp_tree_name_check's containment blacklist permits it, deliberately:
# `bench.1`-shaped names are real). Read as a BRE, `jammi-fix.443` also
# matches ANOTHER wave's `jammi-fixX443` — the own-session exclusion then
# deletes the very session it exists to detect and the gate reports CLEAR:
# a concurrency gate that fails OPEN. `*` mis-excludes a sibling the same
# way, and an unbalanced `[` makes grep exit 2 with NO output at all,
# emptying `other` and failing open for every session on the pod. `-e`
# additionally keeps a name that begins with `-` from being read as an
# option. This robustness is the FUNCTION's own, independent of the
# entrypoint allowlist that also refuses the exotic shapes.
rp_concurrency_preflight_lines() {
  local own_session="${1:?rp_concurrency_preflight_lines needs its own tmux session name}" \
        own_wave="${2:?rp_concurrency_preflight_lines needs its own wave id}"
  cat <<EOF
live="\$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^jammi-')" # tripwire-ok: tmux list-sessions exits non-zero with no server/sessions at all -- a real, expected, checked state (an empty \$live, which makes \$other empty and lands on the CLEAR branch below), never a silent pass on a real error this remote script would otherwise need to escalate
other="\$(printf '%s\\n' "\$live" | grep -Fvx -e 'jammi-seed' | grep -Fvx -e '${own_session}' | head -1)" # tripwire-ok: grep -v legitimately matches nothing when this tree's own session is the only one -- the empty \$other IS the CLEAR state, handled by the if/else right below
# shellcheck disable=SC2157
if [ -z "\$other" ]; then
  echo "GPU_DEV_CONCURRENCY_STATE=CLEAR"
else
  busy_wave=""
  busy_session=""
  holders=0
  for claim in ${RP_CLAIM_DIR}/*.claim; do
    [ -f "\$claim" ] || continue
    claim_wave="\$(grep '^WAVE=' "\$claim" 2>/dev/null | head -1 | cut -d= -f2-)" # tripwire-ok: an unreadable/malformed claim file is a real, checked state (the empty-field guard on the next line drops it, and a store with no readable holder at all lands on the fail-closed UNKNOWN branch below), never a silent pass
    claim_tree="\$(grep '^TREE=' "\$claim" 2>/dev/null | head -1 | cut -d= -f2-)" # tripwire-ok: same as claim_wave above -- both fields are required, and a claim missing either is dropped by the guard on the next line
    [ -n "\$claim_wave" ] && [ -n "\$claim_tree" ] || continue
    claim_session="jammi-\$claim_tree"
    [ "\$claim_session" = '${own_session}' ] && continue
    grep -Fxq -e "\$claim_session" <<<"\$live" || continue
    holders=\$((holders + 1))
    if [ "\$claim_wave" != '${own_wave}' ]; then
      busy_wave="\$claim_wave"
      busy_session="\$claim_session"
      break
    fi
  done
  if [ -n "\$busy_wave" ]; then
    echo "GPU_DEV_CONCURRENCY_STATE=BUSY:\$busy_wave:\$busy_session"
  elif [ "\$holders" -eq 0 ]; then
    echo "GPU_DEV_CONCURRENCY_STATE=BUSY:UNKNOWN:\$other"
  else
    echo "GPU_DEV_CONCURRENCY_STATE=CLEAR"
  fi
fi
EOF
}

# The env-source + CARGO_TARGET_DIR + cd preamble shared by every remote
# command that must run correctly as if it were an interactive shell in the
# tree (an SSH login shell does not inherit the container's Dockerfile ENV —
# see RP_ENV_PREAMBLE's own doc). Split out of rp_job_wrapper_lines so a
# caller that needs its OWN dispatch logic after the preamble
# (rp_job_wrapper_with_marker_lines, below) does not have to reconstruct it
# by hand — one definition, two consumers, never a second hand-copy that
# could drift. $1=tree_dir $2=target_dir.
rp_job_env_lines() {
  local tree_dir="${1:?rp_job_env_lines needs a tree dir}" target_dir="${2:?rp_job_env_lines needs a target dir}"
  printf '[ -f /root/.jammi_env ] && . /root/.jammi_env\n'
  printf "export CARGO_TARGET_DIR='%s'\n" "$target_dir"
  printf "cd '%s'\n" "$tree_dir"
  rp_job_build_sha_lines "$tree_dir"
}

# The pod-side default for `JAMMI_BUILD_SHA`, emitted as shell text into the
# job wrapper (rp_job_env_lines above) and evaluated ON THE POD, where the
# push stamp actually lives.
#
# WHY THIS EXISTS: `push` rsyncs the working tree WITHOUT `.git` (the exclude
# set is pod_push_stamp.sh's own `pod_push_excludes`), so a `cargo build` on a
# pushed tree has no repository for `crates/jammi-bench/build.rs`'s
# `git rev-parse` fallback to read and bakes `build_sha="unknown"`. Every
# producer that cross-checks provenance then refuses the binary's own output,
# which is correct — the binary genuinely could not say what it was built
# from — but it makes an otherwise valid measurement unrecordable unless the
# sha is typed by hand on every build. `<tree>/.jammi-push-stamp.json`
# already records exactly the missing
# fact (`laptop_head`, written by the same `push` that sent the bytes), so the
# default reads it from there.
#
# CLEAN-ONLY, and that restriction is the whole safety argument. `push` sends
# uncommitted work too, so `laptop_head` names the commit the pushed tree was
# BASED on, not necessarily the commit it IS. Naming a commit whose tree is
# not the built bytes is a FABRICATED provenance — strictly worse than
# "unknown", because it is a false claim a downstream reader cannot detect.
# So the stamp's own `porcelain_sha256`/`diff_head_sha256` must BOTH equal the
# digest of the empty string (a clean `git status --porcelain` and a clean
# `git diff HEAD` at push time — the emitted python computes that digest
# rather than hardcoding it, so the comparison cannot rot into a stale
# literal). Anything else leaves the variable UNSET, and build.rs resolves
# the sha itself: from the tree's own git when the tree is a checkout (a
# pod's bootstrap checkout moved to a commit with `git checkout` — `<sha>`,
# or `<sha>-dirty` when its tracked files differ), else "unknown", which the
# producer refuses loudly; the message says which.
#
# A caller-supplied `JAMMI_BUILD_SHA` always wins and is never overwritten —
# it is the manual override for the dirty-tree case.
#
# Cost: `cargo:rerun-if-env-changed=JAMMI_BUILD_SHA` means the first job after
# a push at a NEW commit re-runs jammi-bench's build script and relinks that
# one binary. That is the price of the binary knowing what it is.
#
# $1=tree_dir.
rp_job_build_sha_lines() {
  local tree_dir="${1:?rp_job_build_sha_lines needs a tree dir}"
  printf "__jammi_tree='%s'\n" "$tree_dir"
  printf "__jammi_push_stamp='%s/.jammi-push-stamp.json'\n" "$tree_dir"
  cat <<'BUILDSHAEOF'
if [ -z "${JAMMI_BUILD_SHA:-}" ]; then
  JAMMI_BUILD_SHA="$(python3 - "$__jammi_push_stamp" <<'JAMMIPUSHSTAMPEOF'
import hashlib, json, re, sys

try:
    stamp = json.load(open(sys.argv[1]))
except Exception:
    # No stamp, or an unreadable one: print nothing. The caller's own
    # empty-result branch says what that means; a traceback here would only
    # bury it.
    sys.exit(0)
head = stamp.get("laptop_head") or ""
clean = hashlib.sha256(b"").hexdigest()
if (
    re.fullmatch(r"[0-9a-f]{40}", head)
    and stamp.get("porcelain_sha256") == clean
    and stamp.get("diff_head_sha256") == clean
):
    print(head)
JAMMIPUSHSTAMPEOF
)"
  if [ -n "${JAMMI_BUILD_SHA:-}" ]; then
    export JAMMI_BUILD_SHA
    echo "jammi: JAMMI_BUILD_SHA=${JAMMI_BUILD_SHA} (from ${__jammi_push_stamp} — the pushed tree was CLEAN at that commit)" >&2
  else
    unset JAMMI_BUILD_SHA
    if [ ! -e "$__jammi_push_stamp" ] && git -C "$__jammi_tree" rev-parse -q --verify HEAD >/dev/null 2>&1; then
      echo "jammi: no push stamp — ${__jammi_tree} is a git checkout, so build.rs resolves build_sha from its HEAD ($(git -C "$__jammi_tree" rev-parse HEAD), '-dirty' if its tracked files differ)" >&2
    else
      echo "::warning::JAMMI_BUILD_SHA left UNSET — ${__jammi_push_stamp} is absent or unreadable with no git checkout to read, or records a push whose working tree was DIRTY (its bytes are not any one commit). A binary built here bakes build_sha=\"unknown\" (or the checkout's '-dirty' sha) and every provenance cross-check will refuse its output. Commit and re-push, or set JAMMI_BUILD_SHA=<the 40-hex tip the pushed tree actually is> yourself — never a sha these bytes are not." >&2
    fi
  fi
fi
unset __jammi_push_stamp __jammi_tree
BUILDSHAEOF
}

# Builds the per-tree job wrapper script body (`<tree>/.jammi-job.sh`,
# written by gpu-dev.sh's `run`): the env/cd preamble (rp_job_env_lines),
# then the caller's command. A plain function, not inlined into the remote
# heredoc, so it is directly testable (source this file, call it with
# fixture args) without a live pod. Used by `rp_login_cmd`'s interactive-job
# pane too (job=":", a no-op) — this function is deliberately NEVER given
# the completion-marker bookkeeping rp_job_wrapper_with_marker_lines carries
# below, since an interactive shell has no "run" for wait-job to identify.
# $1=tree_dir $2=target_dir $3=job command.
rp_job_wrapper_lines() {
  local tree_dir="${1:?rp_job_wrapper_lines needs a tree dir}" \
        target_dir="${2:?rp_job_wrapper_lines needs a target dir}" \
        job="${3:?rp_job_wrapper_lines needs a job command}"
  rp_job_env_lines "$tree_dir" "$target_dir"
  printf '%s\n' "$job"
}

# Wraps the SAME env/cd preamble with per-run completion-marker bookkeeping
# for gpu-dev.sh's `run`/`wait-job` pair:
# wait-job has no other way to know that a "no live session" state belongs
# to THIS invocation of `run` rather than an ARBITRARY earlier one — a
# flock-refused `run --timing`, or simply a stale `.jammi.log` left over
# from two runs ago, both read as false SUCCESS under a check that only
# asks "does a log file exist". `<tree>/.jammi.exit` is removed at the
# VERY START of this script (before the flock attempt, before the job ever
# runs) and written EXACTLY ONCE with the job's own real exit code — so a
# run that is CURRENTLY in flight (tmux session alive) is guaranteed to
# have NO marker (its own wrapper already removed it), and a marker that
# DOES exist once the session has ended can only be the most recent run
# that actually reached this script, never a stale leftover: no run since
# it wrote that marker has both started (which would have removed it) and
# also NOT yet finished (which would mean the session is still alive) —
# those two states are mutually exclusive by construction. wait-job checks
# session-liveness FIRST for exactly this reason (the same
# tmux-session-before-markers ordering wait-seed uses).
#
# `timing=1` puts the flock acquisition INSIDE this wrapper (fd 9, `flock
# -n 9`) rather than an outer `flock -n -E 75 ... bash job.sh` in the LAUNCH
# string: a plain bash `if`/`else` on the flock CALL's own exit status is
# unambiguous, where checking the OUTER command's exit code for the literal
# value 75 cannot tell "lock refused" apart from "the job itself happened
# to exit 75". The lock is held for the whole job's lifetime (acquired
# essentially first — only a harmless `rm -f` precedes it — released only
# when this whole script/fd closes).
#
# $1=tree_dir $2=target_dir $3=job $4=token (caller-generated, unique per
# `run` invocation — carried in the marker purely for a human reading it
# directly; wait-job itself never needs to know the expected value, since
# the remove-then-rewrite-under-session-liveness discipline above already
# rules out a stale marker on its own) $5=timing (0|1) $6=wave (RP_WAVE,
# defaults to the tree name at the gpu-dev.sh call site) $7=tree (the short
# tree name, e.g. "mywork" — distinct from $1 tree_dir, which is the full
# checkout path).
#
# Wave-scoped one-pod-per-wave claim (folded into this SAME start/end
# lifecycle `.jammi.exit` already uses, per that marker's own idiom): THIS
# holder's own claim file in RP_CLAIM_DIR (see that constant's doc for why
# the store is a directory of per-holder files and not one shared file) is
# WRITTEN before the job runs, so a concurrent `run` sees this holder the
# instant it launches, and REMOVED whenever this wrapper's execution ends —
# normal completion AND the lock-refused (timing) early exit both clear it,
# so a stale claim can only ever be "wrapper cleanup was interrupted"
# (SIGKILL/pod death), never "a job finished and cleanup was forgotten".
# Every write and removal runs inside a flock on RP_CLAIM_LOCK (fd 8).
#
# Under `timing=1` the claim write comes AFTER the timing lock is acquired,
# never before: a run whose `flock -n 9` is REFUSED never became a holder
# at all, and writing its claim first would let a refused run touch the
# store on behalf of a job that never ran. `rp_concurrency_preflight_lines`
# (above) reads the store and treats a holder with NO matching live tmux
# session as absent (fail OPEN on staleness) rather than refusing forever
# on an orphaned file — tmux liveness stays the PRIMARY signal.
rp_job_wrapper_with_marker_lines() {
  local tree_dir="${1:?rp_job_wrapper_with_marker_lines needs a tree dir}" \
        target_dir="${2:?rp_job_wrapper_with_marker_lines needs a target dir}" \
        job="${3:?rp_job_wrapper_with_marker_lines needs a job command}" \
        token="${4:?rp_job_wrapper_with_marker_lines needs a token}" \
        timing="${5:?rp_job_wrapper_with_marker_lines needs a timing flag}" \
        wave="${6:?rp_job_wrapper_with_marker_lines needs a wave id}" \
        tree_name="${7:?rp_job_wrapper_with_marker_lines needs a tree name}"
  printf "rm -f '%s/.jammi.exit'\n" "$tree_dir"
  # This holder's own claim file, and the write/removal statements that
  # maintain it — built ONCE here and emitted on every path below, so the
  # three sites can never drift apart. Both statements run inside a flock
  # on RP_CLAIM_LOCK (fd 8): a reader never sees a half-written claim, and
  # two holders launching at once never race on the store's creation.
  local claim_file="${RP_CLAIM_DIR}/${tree_name}.claim" claim_write claim_clear
  claim_write="$(printf "mkdir -p %s\n( flock 8; printf '%%s\\\\n' 'WAVE=%s' 'TREE=%s' 'TS=%s' > '%s' ) 8>%s" \
    "$RP_CLAIM_DIR" "$wave" "$tree_name" "$(date -u +%FT%TZ)" "$claim_file" "$RP_CLAIM_LOCK")"
  claim_clear="$(printf "( flock 8; rm -f '%s' ) 8>%s" "$claim_file" "$RP_CLAIM_LOCK")"
  if [ "$timing" = "1" ]; then
    # The timing lock FIRST, this holder's claim only once it is held: a
    # refused run never ran a job, so it must never appear in the store.
    # Its own claim is still cleared on the refusal path, which reaps a
    # claim an EARLIER, SIGKILLed run on this same tree may have left.
    cat <<FLOCKEOF
exec 9>/root/.jammi-timing.lock
if ! flock -n 9; then
  printf '{"token":"${token}","rc":75,"lock_refused":true}' > '${tree_dir}/.jammi.exit'
${claim_clear}
  exit 75
fi
FLOCKEOF
  fi
  printf '%s\n' "$claim_write"
  rp_job_env_lines "$tree_dir" "$target_dir"
  # `( job )` — a SUBSHELL, never the bare job text directly in this script
  # — so a job command that happens to invoke the shell's own `exit` builtin
  # at its top level (`gpu-dev.sh run a100 exit 1` is unusual but a caller
  # CAN type it — JOB is `"$*"` verbatim) terminates only the subshell, not
  # the WHOLE wrapper script; a bare `exit` here would otherwise skip every
  # line after it, including the marker write below, reproducing the exact
  # "no evidence" false state this wrapper exists to prevent. A normal job
  # (`cargo test ...`) behaves identically either way — it never calls
  # `exit` itself, it simply returns control with `$?` set.
  printf '( %s )\n' "$job"
  cat <<MARKEREOF
__jammi_job_rc=\$?
printf '{"token":"${token}","rc":'"\$__jammi_job_rc"',"lock_refused":false}' > '${tree_dir}/.jammi.exit'
${claim_clear}
exit "\$__jammi_job_rc"
MARKEREOF
}

# Builds wait-seed's remote check script (rp_wait_poll's own doc has the rc
# contract: 0 success / 1 named failure / 2 keep polling). A plain
# function, not inlined at gpu-dev.sh's `wait-seed` call site, so a test can
# call it directly (source this file, no live pod needed) with a SANDBOXED
# seed_dir_prefix and a throwaway tmux session name, run the returned text
# locally, and assert its rc against real fixture files — the CLI-level
# mocked-ssh tests alone cannot construct the state-lattice cases the
# liveness-first ordering depends on (both markers present at once; a
# marker alongside a LIVE session).
# $1=seed_dir_prefix (default /root/.jammi-seed — the SAME prefix
# pod_seed_target.sh's own JAMMI_SEED_DIR defaults to) $2=tmux session name
# for the seed build (default jammi-seed).
rp_seed_wait_script() {
  local prefix="${1:-/root/.jammi-seed}" tmux_sess="${2:-jammi-seed}"
  cat <<SCRIPT
if [ -f '${prefix}.jammi-seed-failed' ]; then
  echo "seed FAILED -- tail:"
  tail -n 20 '${prefix}.jammi-seed-failed'
  exit 1
fi
# tripwire-ok: REMOTE script text -- "no such session" is a real, valid
# state (checked explicitly by the if/then below), never a silent pass.
# Session-liveness checked BEFORE the completion marker, not after: a
# --reseed removes BOTH markers at rebuild start (pod_seed_target.sh), but
# the narrow window between "the detached tmux session starts" and "the
# script reaches that removal" can still show a STALE COMPLETE marker from
# the PREVIOUS build while THIS session is alive -- a live session is
# authoritative over any marker's content, always, since a marker cannot be
# trusted while whatever wrote it (or its successor) might still be
# running.
if tmux has-session -t "=${tmux_sess}" 2>/dev/null; then
  if [ -f '${prefix}.jammi-seed-complete' ]; then
    echo "seed still building (tmux session ${tmux_sess} alive; a COMPLETE marker from a PRIOR build is also present -- a live session always wins, never read as success mid-reseed)"
  else
    echo "seed still building (tmux session ${tmux_sess} alive)"
  fi
  exit 2
fi
if [ -f '${prefix}.jammi-seed-complete' ]; then
  echo "seed complete: \$(cat '${prefix}.jammi-seed-complete')"
  exit 0
fi
echo "no seed evidence: no completion/failure marker and no running tmux session '${tmux_sess}' -- did up/shell ever start one?"
exit 3
SCRIPT
}

# Builds wait-job's remote check script. Reads <tree>/.jammi.exit, the
# per-run completion marker rp_job_wrapper_with_marker_lines writes (above)
# -- NEVER .jammi.log's mere existence (a flock-refused `run --timing`, or a
# stale log left from an earlier run, both read as false SUCCESS under a
# content-free "does a log file exist" check). Session-liveness is checked FIRST, same ordering as
# rp_seed_wait_script above and for the identical reason: `run` removes
# .jammi.exit at the VERY START of its own wrapper, so a marker can only be
# read once the session that would have removed it has ended -- a marker
# present with no live session can therefore only be the most recent run
# that actually reached the wrapper, never a stale leftover.
# $1=tree_dir $2=tmux_session $3=tree_name (for messages only).
rp_job_wait_script() {
  local tree_dir="${1:?rp_job_wait_script needs a tree dir}" \
        tmux_sess="${2:?rp_job_wait_script needs a tmux session}" \
        tree_name="${3:?rp_job_wait_script needs a tree name}"
  cat <<SCRIPT
# tripwire-ok: REMOTE script text -- "no such session" is a real, valid
# state (checked explicitly by the if/then below), never a silent pass.
if tmux has-session -t "=${tmux_sess}" 2>/dev/null; then
  echo "job still running (tmux session ${tmux_sess} alive)"
  exit 2
fi
if [ -f '${tree_dir}/.jammi.exit' ]; then
  marker="\$(cat '${tree_dir}/.jammi.exit')"
  rc="\$(printf '%s' "\$marker" | sed -n 's/.*"rc":\([0-9-]*\).*/\1/p')"
  if [ -z "\$rc" ]; then
    echo "job completion marker at ${tree_dir}/.jammi.exit is malformed: \$marker"
    exit 1
  fi
  if grep -q '"lock_refused":true' <<<"\$marker"; then
    echo "job REFUSED: the shared pod-wide timing lock was already held (rc=75) -- \$marker"
    exit 1
  fi
  if [ "\$rc" = "0" ]; then
    echo "job finished successfully (rc=0) -- \$marker -- tail of ${tree_dir}/.jammi.log:"
    tail -n 20 '${tree_dir}/.jammi.log' 2>/dev/null
    exit 0
  fi
  echo "job FAILED (rc=\$rc) -- \$marker -- tail of ${tree_dir}/.jammi.log:"
  tail -n 20 '${tree_dir}/.jammi.log' 2>/dev/null
  exit 1
fi
echo "no job evidence for tree '${tree_name}': no live tmux session '${tmux_sess}' and no completion marker at ${tree_dir}/.jammi.exit -- run 'gpu-dev.sh run' first"
exit 3
SCRIPT
}

# The ONE entrypoint text every rented pod boots — the self-terminating
# watchdog + sshd string — factored out of `_rp_deploy_payload` (below) so the
# single-pod payload and the fleet's `_rp_fleet_pod_payload` share exactly ONE
# "kill this thing, then let me in" mechanism instead of two that could
# quietly drift apart.
# Prints the setup TEXT (no trailing newline — command substitution strips
# it, matching what the pre-factoring inline python variable held). $1=ttl
# hours (the deadline baked into the watchdog's own `sleep`).
#
# The deadline is part of the entrypoint, so it exists from the moment the
# container starts. It CANNOT be installed over SSH after the fact: the
# window between "pod rented" and "SSH reachable" is minutes long, and a
# runner SIGKILLed inside it never runs its EXIT trap. That is exactly how an
# A100 was orphaned for seven days on 2026-07-24.
#
# `sleep` comes FIRST so the deadline is reached no matter what the network
# does; an install ahead of it can hang and leave the pod with no deadline at
# all. Every step after it is `timeout`-bounded so nothing can wedge the
# sequence.
#
# `runpodctl remove pod $RUNPOD_POD_ID` is the ONLY in-pod termination that
# works, and it is verified on real hardware: RunPod special-cases
# self-removal, so it succeeds in this custom image with no config file and
# no key of ours — even though `runpodctl config` fails and `runpodctl get
# pod` returns Unauthorized.
#
# There is deliberately no `kill 1` fallback. It is measured to be a no-op:
# PID 1 in a PID namespace ignores signals it has no handler for, including
# SIGKILL, so the pod keeps RUNNING and keeps billing at full rate. A fallback
# that cannot work is worse than none — it invites trusting a guard that does
# nothing. The retry loop stands in its place: the only real failure mode is
# no network at deadline time, and retrying costs nothing. rp_sweep remains
# the true backstop.
_rp_entrypoint_setup() { # $1=ttl_hours
  python3 - "$1" <<'PY'
import sys
ttl_h = sys.argv[1]
ttl = int(ttl_h) * 3600
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
#
# The image ships sshd; an image without it installs it here, retrying while
# a fresh host's egress comes up, with yum's own errors in the pod log — a
# silenced failure left a pod restarting forever on a missing sshd with
# nothing to say why.
sshd = ("for try in 1 2 3 4 5 6; do "
        "[ -x /usr/sbin/sshd ] && break; "
        "yum install -y -q openssh-server openssh-clients && break; "
        "echo \"entrypoint: installing openssh-server failed (try $try of 6)\" >&2; "
        "sleep 10; done; ")
setup = (watchdog
         + sshd + "ssh-keygen -A; "
         "mkdir -p /root/.ssh; printf \"%s\\n\" \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
         "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
         "/usr/sbin/sshd -D")

# `setup` is wrapped in bash -c '...' by every caller. A single quote
# anywhere inside it closes that wrapper early and hands the remainder —
# pipes, redirects, semicolons — to the OUTER shell. The entrypoint then dies
# on a syntax error and the pod boots, bills, and never becomes reachable: a
# silent, paid failure that looks exactly like a capacity problem. Quote with
# double quotes only.
assert "'" not in setup, "pod entrypoint must contain no single quotes (breaks bash -c wrapping)"
print(setup)
PY
}

_rp_deploy_payload() { # $1=cloudType $2=gpuTypeId
  local setup
  setup="$(_rp_entrypoint_setup "$RP_TTL_HOURS")" || return 1
  python3 - "$1" "$2" "$RP_IMAGE" "$RP_PUBKEY" "$RP_TTL_HOURS" "$RP_POD_PREFIX" "$RP_DISK_GB" "$RP_VOLUME_GB" "$RP_GPU_COUNT" "$setup" <<'PY'
import json, sys
cloud, gpu, image, pub, ttl_h, prefix, disk_gb, volume_gb, gpu_count, setup = sys.argv[1:11]
# `gpuCount` is the caller's own RP_GPU_COUNT (default 1, validated as a
# positive integer at the top of this file), passed as an unquoted JSON
# number — the API rejects the string form.
inp = {"cloudType": cloud, "gpuCount": int(gpu_count), "gpuTypeId": gpu,
       # The deadline travels in the name so any sweeper honours THIS pod's limit.
       "name": "%s-ttl%s" % (prefix, ttl_h),
       "imageName": image, "containerDiskInGb": int(disk_gb), "volumeInGb": int(volume_gb), "ports": "22/tcp",
       "dockerArgs": "bash -c '%s'" % setup, "env": [{"key": "PUBLIC_KEY", "value": pub}]}
print(json.dumps({"query": "mutation D($i: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$i){ id } }",
                  "variables": {"i": inp}}))
PY
}

# The "zero tests matched" tripwire, shared by every leg whose remote text
# runs a filtered `cargo test` (`runpod_gpu_topology.sh`'s proof groups).
# A `cargo test ... <name-filter> ...` invocation whose own filter matches NO
# tests exits 0 with "running 0 tests ... test result: ok" printed to its
# log — a false green a leg with no proof must never read as a pass (the
# same never-vacuous doctrine `runpod_gpu_prove.sh`'s
# `capability-surface-proof` group and `check_gpu_prove_once.py` state for
# the prove lane). Factored out so this ONE check text exists once, never
# re-typed per leg — a caller drops it into its own remote heredoc via a
# bare `$(...)` command substitution (never `\$(...)`; see this function's
# own $2 doc for why).
#
# Prints shell TEXT (never a value) meant to be substituted, via a bare
# `$(...)`, directly after a `cargo test ... 2>&1 | tee "<log>"` line inside
# a caller's own (unquoted, `<<REMOTE`-style) heredoc. $1=the shell variable
# name already holding that group's own accumulated rc (e.g. `grc`,
# unquoted — no leading `$`, no quotes: it is spliced as-is into an
# assignment target). $2=the log file path exactly as it must appear in the
# FINAL remote text (e.g. the literal three characters `$log_var` when the
# remote script itself will read a shell variable — pass it SINGLE-QUOTED
# at the call site, `'$gang_log'`, never double-quoted or bare, so bash
# does not expand it while building the caller's OWN command-substitution
# argv; this function performs no further escaping of it). $3=a human label
# for the filter that matched (message text only).
_rp_zero_test_tripwire_lines() {
  local rc_var="${1:?_rp_zero_test_tripwire_lines needs the rc variable name}" \
        log_path="${2:?_rp_zero_test_tripwire_lines needs the log path text}" \
        filter_label="${3:?_rp_zero_test_tripwire_lines needs a filter label}"
  cat <<EOF
if grep -q "running 0 tests" "${log_path}"; then
  echo "::error::the test filter '${filter_label}' matched ZERO tests on this ref -- refusing to read a 0-test run as a proof"
  ${rc_var}=1
fi
EOF
}

# ═════════════════════════════════════════════════════════════════════════
# Fleet primitives (REST v2, `POST/GET /v2/pods`): ORDINARY pods rented
# individually into ONE data center and joined by RunPod's Global Networking —
# the multi-host topology `runpod_gpu_topology.sh` rents. A second renting
# shape beside the single-pod GraphQL payload (`_rp_deploy_payload`,
# `podFindAndDeployOnDemand`). Named identically to an ordinary single pod
# ("<RP_POD_PREFIX>-ttl<H>", no index suffix — the shape `rp_sweep`'s TTL-name
# parser recognizes, so every fleet pod falls under the ordinary pod sweep's
# backstop; the host index is tracked by this tooling's own local id mapping,
# never by name). The caller decides the data center (co-placement is the
# driver's).
# ═════════════════════════════════════════════════════════════════════════

# $1=gpuTypeId $2=dataCenterId $3=host index (name-shape validation only;
# the created pod's own id, not its name, is what the caller tracks). Prints
# the REST v2 `POST /v2/pods` request body, `RP_GPU_COUNT` GPUs per pod.
# Shares `_rp_entrypoint_setup` with the single-pod payload (the SAME
# watchdog+sshd text on every leg).
_rp_fleet_pod_payload() {
  local gpu="${1:?_rp_fleet_pod_payload needs a gpuTypeId}" dc="${2:?_rp_fleet_pod_payload needs a dataCenterId}" \
        index="${3:?_rp_fleet_pod_payload needs a host index}" setup name
  setup="$(_rp_entrypoint_setup "$RP_TTL_HOURS")" || return 1
  name="${RP_POD_PREFIX}-ttl${RP_TTL_HOURS}"
  rp_name_allowlist_check "fleet pod name (host ${index})" "$name" || return 2
  # REST v2 `POST /v2/pods` field names (docs.runpod.io/api-reference-v2/pods/
  # create-a-pod): `cloud` and `image` -- NOT the v1 GraphQL `cloudType`/
  # `imageName` `_rp_deploy_payload` speaks; `disk` is the container disk in GB.
  python3 - "$gpu" "$dc" "$RP_IMAGE" "$RP_PUBKEY" "$name" "$setup" "$RP_DISK_GB" "$RP_GPU_COUNT" <<'PY'
import json, sys
gpu, dc, image, pub, name, setup, disk_gb, gpu_count = sys.argv[1:9]
body = {
    "name": name,
    "cloud": "SECURE",
    "globalNetworking": True,
    "dataCenterIds": [dc],
    "gpu": {"id": gpu, "count": int(gpu_count)},
    "image": image,
    "disk": int(disk_gb),
    "ports": ["22/tcp"],
    "startSsh": True,
    "args": "bash -c '%s'" % setup,
    "env": {"PUBLIC_KEY": pub},
}
print(json.dumps(body))
PY
}

# POST /v2/pods — the RENTING ROOT for the fleet (`RENTING_ROOTS` in
# `check_gpu_prove_once.py`; its P7 closure/derivation scan is seeded from
# this name). $1=gpuTypeId $2=dataCenterId $3=host index. Prints the new
# pod's id on success; a 201 body with no id, or an unparseable one, is a
# named refusal.
rp_fleet_pod_create() {
  local gpu="${1:?rp_fleet_pod_create needs a gpuTypeId}" dc="${2:?rp_fleet_pod_create needs a dataCenterId}" \
        index="${3:?rp_fleet_pod_create needs a host index}" payload resp status body id prc
  payload="$(_rp_fleet_pod_payload "$gpu" "$dc" "$index")" \
    || { echo "::error::fleet pod create (host ${index}): could not build the request body" >&2; return 1; }
  resp="$(_rp_rest POST /v2/pods "$payload")" \
    || { echo "::error::fleet pod create (host ${index}): REST request failed (transport)" >&2; return 1; }
  status="$(printf '%s\n' "$resp" | head -n1)"
  body="$(printf '%s\n' "$resp" | tail -n +2)"
  case "$status" in
    201)
      id="$(printf '%s' "$body" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(2)
if not isinstance(d, dict):
    sys.exit(2)
i = d.get("id")
if not i:
    sys.exit(1)
print(i)
' 2>/dev/null)"
      prc=$?
      case "$prc" in
        0) printf '%s\n' "$id"; return 0 ;;
        1) echo "::error::fleet pod create (host ${index}): 201 but the response body carried no id: ${body}" >&2; return 1 ;;
        *) echo "::error::fleet pod create (host ${index}): 201 but the response body is unparseable: $(printf '%s' "$body" | head -c 300)" >&2; return 1 ;;
      esac ;;
    *)
      echo "::error::fleet pod create (host ${index}) refused (status ${status}): $(printf '%s' "$body" | head -c 300)" >&2
      return 1 ;;
  esac
}

# GET /v2/pods/{id}. Prints the raw Pod body on success (200 with the
# required `id` key present); a 200 without `id`, a non-200, or a transport
# failure is a named refusal. $1=podId.
rp_pod_get() {
  local id="${1:?rp_pod_get needs a pod id}" resp status body prc
  resp="$(_rp_rest GET "/v2/pods/${id}")" \
    || { echo "::error::pod get ${id}: REST request failed (transport)" >&2; return 1; }
  status="$(printf '%s\n' "$resp" | head -n1)"
  body="$(printf '%s\n' "$resp" | tail -n +2)"
  case "$status" in
    200)
      printf '%s' "$body" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(2)
if not isinstance(d, dict):
    sys.exit(2)
sys.exit(0 if d.get("id") else 1)
' 2>/dev/null
      prc=$?
      case "$prc" in
        0) printf '%s' "$body"; return 0 ;;
        1) echo "::error::pod get ${id}: 200 but the body is missing the required 'id' key: ${body}" >&2; return 1 ;;
        *) echo "::error::pod get ${id}: 200 but the body is unparseable: $(printf '%s' "$body" | head -c 300)" >&2; return 1 ;;
      esac ;;
    *)
      echo "::error::pod get ${id} refused (status ${status}): $(printf '%s' "$body" | head -c 300)" >&2
      return 1 ;;
  esac
}

# ── The fleet: pods co-located in one data center, joined by Global
# Networking — the multi-host topology `runpod_gpu_topology.sh` rents. ──────

# The rented part's compute capability, DERIVED from the gpuTypeId (never a
# second literal): a host exports it as `CUDA_COMPUTE_CAP` and refuses to build
# when `nvidia-smi`'s own `compute_cap` disagrees, so the SASS a lane proves is
# the SASS the rented silicon runs. The domain is the sm_XX set
# `runpod_gpu_prove.sh` names (sm_80/86/89/90); a gpuTypeId outside it is
# refused before anything is rented. `$1`=gpuTypeId -> stdout: the bare numeric
# cap; rc 1 when unknown.
rp_compute_cap_for_gpu_type() {
  case "${1:?rp_compute_cap_for_gpu_type needs a gpuTypeId}" in
    "NVIDIA A100-SXM4-80GB"|"NVIDIA A100 80GB PCIe"|"NVIDIA A100-PCIE-40GB") echo 80 ;; # sm_80 (Ampere floor)
    "NVIDIA A40"|"NVIDIA RTX A6000"|"NVIDIA RTX A5000") echo 86 ;;                       # sm_86 (Ampere workstation)
    "NVIDIA L4"|"NVIDIA L40S"|"NVIDIA L40"|"NVIDIA GeForce RTX 4090") echo 89 ;;         # sm_89 (Ada)
    "NVIDIA H100 80GB HBM3"|"NVIDIA H100 PCIe"|"NVIDIA H100 NVL"|"NVIDIA H200") echo 90 ;; # sm_90 (Hopper)
    *) return 1 ;;
  esac
}

# $1=min level $2=candidate level. 0 when candidate >= min in the ranking
# above; 1 when candidate is a recognized level below min; 2 when EITHER
# level is not one of the four documented spellings (a schema drift, never
# silently read as passing or failing).
rp_availability_at_least() {
  local min="${1:?rp_availability_at_least needs a min level}" have="${2:?rp_availability_at_least needs a candidate level}"
  local order="$RP_AVAILABILITY_ORDER" min_i=-1 have_i=-1 i=0 lvl
  for lvl in $order; do
    [ "$lvl" = "$min" ] && min_i=$i
    [ "$lvl" = "$have" ] && have_i=$i
    i=$((i + 1))
  done
  if [ "$min_i" -lt 0 ] || [ "$have_i" -lt 0 ]; then
    echo "::error::rp_availability_at_least: unrecognized AvailabilityLevel (min='${min}' have='${have}')" >&2
    return 2
  fi
  [ "$have_i" -ge "$min_i" ]
}

# The per-data-center availability read: $1=gpuTypeId $2=min level;
# the catalog response BODY (the `GET /v2/catalog/gpus?...` JSON) on stdin.
# Prints a SPACE-SEPARATED list of qualifying data center ids (possibly
# empty — "the gpu type was found, but no data center clears the floor",
# read by the caller as no-capacity, exit 75) on stdout. Returns 0 on ANY
# successful parse (including zero qualifying data centers, or the gpu type
# entirely absent from the catalog — both are "no capacity", never a hard
# error); 2 when the body itself could not be read as the documented shape
# (a genuinely different failure — the catalog endpoint itself is
# unreachable or its schema moved).
rp_fleet_pick_data_centers() {
  local gpu="${1:?rp_fleet_pick_data_centers needs a gpuTypeId}" min="${2:?rp_fleet_pick_data_centers needs a min level}"
  python3 -c '
import json, sys
gpu, min_level = sys.argv[1], sys.argv[2]
order = ["NONE", "LOW", "MEDIUM", "HIGH"]
try:
    min_i = order.index(min_level)
except ValueError:
    print("PARSE_ERROR: unrecognized min level %r" % min_level)
    sys.exit(2)
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE_ERROR: could not parse the catalog response: %s" % e)
    sys.exit(2)
gpus = d.get("gpus")
if gpus is None:
    print("PARSE_ERROR: catalog response carries no gpus key")
    sys.exit(2)
entry = next((g for g in gpus if g.get("id") == gpu), None)
if entry is None:
    print("")
    sys.exit(0)
out = []
for dc in (entry.get("dataCenters") or []):
    lvl = dc.get("availability")
    try:
        i = order.index(lvl)
    except ValueError:
        continue
    if i >= min_i:
        did = dc.get("id")
        if did:
            out.append(did)
print(" ".join(out))
' "$gpu" "$min"
}

# $1(stdin)=the raw `GET /v2/catalog/datacenters` response BODY. Prints a
# SPACE-SEPARATED, SORTED list of every data center id carrying
# `globalNetwork: true` — read LIVE at run time (never a hard-coded list;
# see the module doc for one snapshot of it). Accepts either a bare list or an object carrying a
# `dataCenters` key (RunPod's documented REST v2 shape has varied across
# endpoints — so both shapes are read rather than assumed). Returns 0 on any
# successful parse (including zero GN data centers — "no capacity", read by
# the caller as SUPPLY_CONSTRAINT); 2 when the body could not be read as
# either documented shape at all.
rp_fleet_global_network_datacenters() {
  python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE_ERROR: could not parse the datacenters response: %s" % e)
    sys.exit(2)
if isinstance(d, list):
    items = d
elif isinstance(d, dict) and isinstance(d.get("dataCenters"), list):
    items = d["dataCenters"]
else:
    print("PARSE_ERROR: datacenters response is neither a list nor an object carrying a dataCenters list")
    sys.exit(2)
out = []
for dc in items:
    if isinstance(dc, dict) and dc.get("globalNetwork") is True:
        did = dc.get("id")
        if did:
            out.append(did)
print(" ".join(sorted(out)))
'
}

# The co-placement intersection: $1=space-separated candidate list A
# $2=space-separated candidate list B. Prints the SORTED intersection,
# space-separated (possibly empty). Pure set arithmetic — no parsing, no
# network — so the two upstream reads (pod availability, Global-Networking
# data centers) stay independently testable and this join is tested on its
# own.
rp_fleet_intersect() {
  local a="${1:-}" b="${2:-}"
  python3 -c '
import sys
a = set(sys.argv[1].split())
b = set(sys.argv[2].split())
print(" ".join(sorted(a & b)))
' "$a" "$b"
}

# Every place a fleet may be rented, in preference order: one line
# `<gpuTypeId>|<rate>|<dataCenterId>` per candidate type (in `types` order)
# and co-located Global-Networking data center (sorted) where the type
# clears `min_availability` at RP_GPU_COUNT and its secure per-GPU rate
# clears `max_rate` — only `data_center` when one is named. An availability
# level is not a slot count (LOW may hold one pod), so a driver walks these
# until its whole fleet lands in one place. $1=pod catalog body $2=Global-
# Networking data centers (space-separated) $3=candidate types (`|`-
# separated) $4=min availability $5=max per-GPU rate $6=data center or "".
# rc 75 when none qualifies; 2 when a candidate type maps to no compute
# capability.
rp_fleet_candidates() {
  local catalog="$1" gn_dcs="$2" types="$3" min_availability="$4" max_rate="$5" data_center="${6:-}" \
        type dcs rate co dc found=0
  local IFS_SAVE="$IFS"
  IFS='|'
  # shellcheck disable=SC2086
  set -- $types
  IFS="$IFS_SAVE"
  for type in "$@"; do
    [ -n "$type" ] || continue
    rp_compute_cap_for_gpu_type "$type" >/dev/null || {
      echo "::error::candidate GPU type '${type}' maps to no compute capability (sm_80/86/89/90) -- refused" >&2
      return 2
    }
    dcs="$(printf '%s' "$catalog" | rp_fleet_pick_data_centers "$type" "$min_availability")"
    case "$dcs" in PARSE_ERROR*) echo "::error::${dcs}" >&2; return 75 ;; esac
    co="$(rp_fleet_intersect "$dcs" "$gn_dcs")"
    [ -z "$data_center" ] || co="$(rp_fleet_intersect "$co" "$data_center")"
    [ -n "$co" ] || { echo "no co-located capacity for ${type}" >&2; continue; }
    rate="$(printf '%s' "$catalog" | python3 -c '
import json, sys
t = sys.argv[1]
d = json.load(sys.stdin)
e = next((g for g in d.get("gpus", []) if g.get("id") == t), {})
p = (e.get("price") or {}).get("secure")
print("" if p is None else p)
' "$type")"
    [ -n "$rate" ] || { echo "no secure rate listed for ${type}" >&2; continue; }
    if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$rate" "$max_rate"; then
      echo "${type} at \$${rate}/GPU/h exceeds the \$${max_rate} ceiling" >&2
      continue
    fi
    for dc in $co; do
      printf '%s|%s|%s\n' "$type" "$rate" "$dc"
      found=1
    done
  done
  [ "$found" -eq 1 ] || return 75
}

# The fleet's readback parser over one `GET /v2/pods/{id}` body per host,
# in host order ($1 is host 0's). Prints ONE line per host (the second field
# is the host's index): `<podId> <index> <status> <dataCenterId>
# <gn_enabled 0|1> <gn_ip_or_dash> <ssh_host_or_dash> <ssh_port_or_dash>`.
# Returns 0 when EVERY body parses (even when a pod is not yet RUNNING or
# not yet GN-enabled — the caller's own poll loop reads the per-host
# fields); 2 when ANY body could not be read as the documented `Pod`
# object shape at all.
rp_fleet_pods_readback() {
  [ "$#" -ge 1 ] || { echo "PARSE_ERROR: rp_fleet_pods_readback needs one Pod body per host"; return 2; }
  python3 -c '
import json, sys

for index, body in enumerate(sys.argv[1:]):
    try:
        d = json.loads(body)
    except Exception as e:
        print("PARSE_ERROR: could not parse host %d Pod body: %s" % (index, e))
        sys.exit(2)
    if not isinstance(d, dict):
        print("PARSE_ERROR: host %d Pod body is not an object" % index)
        sys.exit(2)
    pid = d.get("id") or "?"
    status = d.get("desiredStatus") or d.get("status") or "?"
    dc = d.get("dataCenterId") or "-"
    gn = d.get("globalNetworking") if isinstance(d.get("globalNetworking"), dict) else {}
    gn_enabled = "1" if gn.get("enabled") is True else "0"
    gn_ip = gn.get("ip") or "-"
    ssh = d.get("ssh") if isinstance(d.get("ssh"), dict) else {}
    ssh_direct = ssh.get("direct") if isinstance(ssh.get("direct"), dict) else {}
    shost = ssh_direct.get("host") or "-"
    sport = ssh_direct.get("port")
    sport = str(sport) if sport is not None else "-"
    print("%s %d %s %s %s %s %s %s" % (pid, index, status, dc, gn_enabled, gn_ip, shost, sport))
' "$@"
}

# The fleet's reachability wait over its independently polled pods, in host
# order. Refuses (97) BEFORE any build starts when: a pod never reaches
# RUNNING within the window, a pod never reports `globalNetworking.enabled`
# with a GN ip, the pods' OWN `dataCenterId`s disagree (co-placement did
# not actually happen — measured, never trusted from the create-time
# request alone), or a pod carries no `ssh.direct` (`scp`/`rsync` need a
# direct endpoint on every host). $1=RP_SSH_WAIT_SECS $2..=pod ids. On
# success prints the data center on the first line, then ONE line per host
# in order: `<ssh_host> <ssh_port> <gn_ip>`, and returns 0. On any refusal,
# prints nothing to stdout, names the reason on stderr, returns 97.
rp_fleet_wait_ready() {
  local wait_secs="${1:?rp_fleet_wait_ready needs RP_SSH_WAIT_SECS}"; shift
  [ "$#" -ge 1 ] || { echo "::error::rp_fleet_wait_ready needs at least one pod id" >&2; return 97; }
  local -a pods=("$@") bodies=() ready_lines=()
  local deadline=$(( SECONDS + wait_secs )) readback="" id body ready _pid index status dc gn_en gn_ip shost sport
  while [ "$SECONDS" -lt "$deadline" ]; do
    bodies=()
    for id in "${pods[@]}"; do
      body="$(rp_pod_get "$id" 2>/dev/null)"
      [ -n "$body" ] || break
      bodies+=("$body")
    done
    if [ "${#bodies[@]}" -ne "${#pods[@]}" ] || ! readback="$(rp_fleet_pods_readback "${bodies[@]}")"; then
      sleep 5
      continue
    fi
    ready_lines=()
    while IFS=' ' read -r _pid index status dc gn_en gn_ip shost sport; do
      [ -n "$index" ] || continue
      if [ "$status" = "RUNNING" ] && [ "$gn_en" = "1" ] && [ "$gn_ip" != "-" ] && [ "$shost" != "-" ]; then
        ready_lines+=("${dc} ${shost} ${sport} ${gn_ip}")
      fi
    done <<< "$readback"
    [ "${#ready_lines[@]}" -eq "${#pods[@]}" ] && break
    sleep 5
  done
  if [ "${#ready_lines[@]}" -ne "${#pods[@]}" ]; then
    echo "::error::${#ready_lines[@]} of ${#pods[@]} fleet pods reached RUNNING with Global Networking enabled and a direct ssh endpoint within ${wait_secs}s; last readback (<pod> <host> <status> <dc> <gn> <gn_ip> <ssh_host> <ssh_port>): ${readback:-<none>}" >&2
    return 97
  fi
  local dcs
  dcs="$(printf '%s\n' "${ready_lines[@]}" | awk '{print $1}' | sort -u)"
  if [ "$(printf '%s\n' "$dcs" | grep -c .)" -ne 1 ]; then
    echo "::error::the fleet's pods landed in DIFFERENT data centers ($(printf '%s' "$dcs" | tr '\n' ' ')) -- co-placement failed; refusing before any build starts" >&2
    return 97
  fi
  printf '%s\n' "$dcs"
  for ready in "${ready_lines[@]}"; do
    printf '%s\n' "${ready#* }"
  done
  return 0
}

# Shell TEXT a fleet host runs to DERIVE its NCCL network interface at run
# time from its own Global-Networking ip ($1 — never a literal interface
# name): /proc/net/route lists every route as little-endian hex
# Destination/Mask per Iface; the interface whose route covers the ip by
# LONGEST prefix wins (the default route, mask 0, never matches); no match is
# a NAMED refusal (97). No iproute2 — the image ships none. Echoes
# `DERIVED_NCCL_IFACE=<iface>` and exports `NCCL_SOCKET_IFNAME`.
# `RP_ROUTE_TABLE` lets a fixture feed a table.
rp_fleet_iface_lines() {
  local gn_ip="${1:?rp_fleet_iface_lines needs a Global-Networking ip}"
  cat <<IFACE
gn_ip="${gn_ip}"
route_table="\${RP_ROUTE_TABLE:-/proc/net/route}"
IFS=. read -r _a _b _c _d <<< "\$gn_ip"
gn_le=\$(( (_d << 24) | (_c << 16) | (_b << 8) | _a ))
iface=""; best_mask=-1
while read -r _ifn _dest _gw _flags _ref _use _metric _mask _rest; do
  [ "\$_ifn" = "Iface" ] && continue
  [ -n "\$_mask" ] || continue
  _m=\$(( 16#\$_mask )); _dst=\$(( 16#\$_dest ))
  [ "\$_m" -eq 0 ] && continue
  if [ \$(( gn_le & _m )) -eq "\$_dst" ] && [ "\$_m" -gt "\$best_mask" ]; then
    best_mask=\$_m; iface="\$_ifn"
  fi
done < "\$route_table"
if [ -z "\$iface" ]; then
  echo "::error::no route in \$route_table covers the Global-Networking ip \$gn_ip -- refusing to derive NCCL_SOCKET_IFNAME" >&2
  exit 97
fi
echo "DERIVED_NCCL_IFACE=\$iface"
export NCCL_SOCKET_IFNAME="\$iface"
IFACE
}

# Whether a fleet's hosts reach each other on Global Networking — MEASURED,
# never trusted from the pods' own report: a data center can list Global
# Networking, give each pod an ip and a route, resolve `<pod>.runpod.internal`,
# and still carry no packet between two of its pods. Each host listens on its
# own Global-Networking ip (`RP_FLEET_ROUTE_PORT`), then each dials every
# other until it answers or `RP_FLEET_ROUTE_SECS` pass (the network
# converging after the pods start). Args: one `<ssh_host> <ssh_port> <gn_ip>`
# triple per host, in host order. 0 when every ordered pair connects; 97,
# naming the pair, otherwise.
RP_FLEET_ROUTE_PORT="${RP_FLEET_ROUTE_PORT:-7999}"
RP_FLEET_ROUTE_SECS="${RP_FLEET_ROUTE_SECS:-120}"
rp_fleet_routes() {
  [ "$#" -ge 6 ] && [ $(( $# % 3 )) -eq 0 ] \
    || { echo "::error::rp_fleet_routes needs a <ssh_host> <ssh_port> <gn_ip> triple for each of at least two hosts" >&2; return 97; }
  local -a hosts=() ports=() ips=()
  while [ "$#" -gt 0 ]; do hosts+=("$1"); ports+=("$2"); ips+=("$3"); shift 3; done
  local i j peers
  for i in "${!hosts[@]}"; do
    ssh "${RP_SSHO[@]}" -p "${ports[$i]}" "root@${hosts[$i]}" "timeout 60 bash -s" <<LISTEN || { echo "::error::host ${i}: could not start its route listener" >&2; return 97; }
setsid nohup timeout $((RP_FLEET_ROUTE_SECS * 3)) python3 -c 'import socket
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("${ips[$i]}", ${RP_FLEET_ROUTE_PORT})); s.listen(8)
while True: s.accept()[0].close()' >/dev/null 2>&1 < /dev/null &
LISTEN
  done
  for i in "${!hosts[@]}"; do
    peers=""
    for j in "${!ips[@]}"; do [ "$i" = "$j" ] || peers="${peers} ${ips[$j]}"; done
    ssh "${RP_SSHO[@]}" -p "${ports[$i]}" "root@${hosts[$i]}" "timeout $((RP_FLEET_ROUTE_SECS + 30)) bash -s" <<REACH || { echo "::error::host ${i} (${ips[$i]}) cannot reach every other host on ${RP_FLEET_ROUTE_PORT} over Global Networking within ${RP_FLEET_ROUTE_SECS}s -- this data center does not route between the fleet's hosts" >&2; return 97; }
deadline=\$((SECONDS + ${RP_FLEET_ROUTE_SECS}))
for peer in${peers}; do
  until timeout 5 bash -c "</dev/tcp/\${peer}/${RP_FLEET_ROUTE_PORT}" 2>/dev/null; do
    [ "\$SECONDS" -lt "\$deadline" ] || { echo "no route to \${peer}" >&2; exit 1; }
    sleep 3
  done
done
REACH
  done
  echo "=== the fleet's ${#hosts[@]} hosts reach each other over Global Networking (${ips[*]}) ==="
}

# The ONE `-ttl<H>` deadline-NAME parser rp_sweep reads — a `python3` SOURCE
# FRAGMENT, not a bash function, because the sweep does its age/deadline math
# in ONE python process over the account's full JSON list (never one
# subprocess per row); prepended (plain string concatenation) ahead of the
# sweep's own script. Defines `_rp_parse_ttl_seconds(prefix, name)`:
# the deadline in SECONDS carried by a `<prefix>-ttl<H>` name, or `None` when
# the name does not carry that shape (an operator's own pod, or a
# malformed name — "unparseable-deadline", never silently skipped).
_rp_ttl_parser_pysrc() {
  cat <<'PY'
def _rp_parse_ttl_seconds(prefix, name):
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    if tail.startswith('-ttl') and tail[4:].isdigit() and tail[4:] != '':
        return int(tail[4:]) * 3600
    return None
PY
}

# The ONE `force_hours` override validator rp_sweep reads: an all-digit,
# strictly-greater-than-zero hours count.
# "0"/"00" are deliberately REFUSED rather than accepted as a (vacuous)
# force-reap: both are all-digit (pass any digit-shape check alone), and a
# naive `if override:` in the python side reads the STRING "0" as truthy
# exactly like "8", giving `limit = 0` under which every RUNNING pod is
# already "past-deadline-0s" — `reap 0` would mass-sweep the whole account's
# pods instead of refusing. $1=override string (empty
# is NOT validated here — empty means "no override", the caller's own
# per-object deadline applies, handled by the caller before this is called).
# Prints nothing on success; on refusal prints the reason and returns 2.
_rp_validate_force_hours() {
  local override="${1?_rp_validate_force_hours needs an override string}"
  case "$override" in
    ''|*[!0-9]*) echo "::error::reap: hours must be a positive integer (got '${override}')"; return 2 ;;
  esac
  [ "$override" -gt 0 ] || { echo "::error::reap: hours must be > 0"; return 2; }
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
    echo "  deployed ${RP_POD_ID} on ${cloud} / ${gpu}; waiting for SSH (≤${RP_SSH_WAIT_SECS}s)..."
    RP_HOST=""; RP_PORT=""
    # WRITE-AHEAD: record the session — pod id, arch, a
    # host-unknown placeholder (RP_HOST/RP_PORT are still "" here) — THE
    # MOMENT the pod exists, before the reachability wait below. The wait can
    # run for minutes; a failure in that window (an external kill that
    # bypasses the EXIT trap, or a trap-time `rp_terminate` call that itself
    # silently fails — rp_cleanup throws that response away by design) must
    # not leave a running, billing pod recorded NOWHERE — invisible to
    # `ls`/`down`, caught only by `reap`'s own 72h-late sweep (observed: a
    # four-way parallel `up` can leave a pod running with no session record
    # after a post-create failure). rp_session_save is a no-op for a
    # `shell` (RP_SESSION is empty there — see gpu-dev.sh), so this only
    # takes effect for `up`. The identical call below (after SSH is up and
    # the driver floor is confirmed) then UPDATES this same record with the
    # real RP_HOST/RP_PORT — this is an update-in-place, not a second,
    # independent record, and does not change the `up --replace` contract
    # (that check runs in gpu-dev.sh BEFORE rp_deploy_arch/rp_deploy_live are
    # ever called, against whatever THIS invocation loaded at startup).
    rp_session_save
    # A wall-clock deadline, not a fixed iteration count: a hard-coded
    # budget is one no caller can raise, and a cold host still pulling the
    # multi-GB CUDA image can
    # take longer than that before sshd is even up (see RP_SSH_WAIT_SECS's own
    # doc at the top of this file). `SECONDS` is bash's own
    # elapsed-since-this-shell-started counter — no subprocess per check, unlike
    # `date +%s`.
    local _rp_deploy_deadline=$((SECONDS + RP_SSH_WAIT_SECS)) _rp_deploy_remaining
    while [ "$SECONDS" -lt "$_rp_deploy_deadline" ]; do
      R="$(rp_gql "{\"query\":\"query{ pod(input:{podId:\\\"${RP_POD_ID}\\\"}){ runtime{ ports{ ip publicPort privatePort isIpPublic type } } } }\"}")"
      read -r RP_HOST RP_PORT < <(printf '%s' "$R" | python3 -c 'import sys,json
p=(json.load(sys.stdin).get("data",{}).get("pod") or {}).get("runtime") or {}
[print(x["ip"], x["publicPort"]) for x in (p.get("ports") or []) if x.get("privatePort")==22 and x.get("isIpPublic")]' | head -1)
      if [ -n "${RP_HOST:-}" ] && rp_sshd_answers "$RP_HOST" "$RP_PORT"; then
        # Reachable — now gate on the driver floor. A pod below r560 cannot JIT
        # the image's CUDA 12.6 PTX, so it is unusable for this build; fail over
        # to the next candidate rather than run every test into the engine's
        # startup driver floor.
        local drv drv_major
        drv="$(ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" \
          "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)"
        drv_major="${drv%%.*}"
        if [ -n "$drv_major" ] && [ "$drv_major" -ge "$RP_MIN_DRIVER_MAJOR" ] 2>/dev/null; then
          echo "  SSH up on ${RP_HOST}:${RP_PORT} (driver ${drv})"
          # RP_POD_CREATED was already set the moment RP_POD_ID was read from
          # the deploy response, above — not here, which is minutes later.
          # This UPDATES the write-ahead record made right after that (see
          # above): same session file, now with the real RP_HOST/RP_PORT in
          # place of the host-unknown placeholder — never a second record.
          rp_session_save; return 0
        fi
        echo "  pod ${RP_POD_ID} driver '${drv:-unknown}' is below the r${RP_MIN_DRIVER_MAJOR} floor; terminating and trying next candidate"
        rp_terminate "$RP_POD_ID"
        RP_POD_ID=""; break
      fi
      RP_HOST=""
      # Sleep only up to what is left of the budget, never past it — a fixed
      # 10s sleep against a short RP_SSH_WAIT_SECS (a test's 2s deadline, or a
      # caller's own tight override) would otherwise overrun the deadline by
      # up to a full sleep cycle on every iteration.
      _rp_deploy_remaining=$((_rp_deploy_deadline - SECONDS))
      [ "$_rp_deploy_remaining" -gt 0 ] || break
      sleep "$(( _rp_deploy_remaining < 10 ? _rp_deploy_remaining : 10 ))"
    done
    if [ -n "$RP_POD_ID" ]; then
      echo "  pod ${RP_POD_ID} never became reachable within ${RP_SSH_WAIT_SECS}s; terminating and trying next candidate"
      rp_terminate "$RP_POD_ID"
      RP_POD_ID=""
    fi
  done
  if [ "$supply_seen" = "0" ]; then echo "::error::no GPU capacity on RunPod for the requested candidates (SUPPLY_CONSTRAINT); retry later"
  else echo "::error::GPU pod(s) deployed but none became reachable over SSH; retry later"; fi
  return 75
}

# arch → RunPod GPU-type candidates (SECURE then COMMUNITY), the one place the
# mapping lives. A100 is the compute-capability floor (sm_80); l40s/l4 are
# Ada (sm_89, fp8); h100 is Hopper (sm_90); a40 is Ampere-workstation
# (sm_86). (RunPod has no Tesla T4 — the floor is Ampere+.) Returns 2 on an
# unknown arch.
#
# Multi-GPU candidate ordering. Rewrites the CALLER's own `cand` array in
# place (bash locals are dynamically scoped, so `rp_deploy_arch`'s array is
# visible here) as a STABLE partition: every candidate whose GPU-type name
# carries `SXM` first, every other candidate after, each group keeping its
# own declared SECURE-before-COMMUNITY order. No candidate is dropped — a
# multi-GPU rental never narrows the capacity search, it only reorders it.
#
# WHY: a measured `gpuCount: 2` rental found the A100 PCIe pool returning
# zero 2-GPU capacity while `A100-SXM4-80GB` SECURE provisioned immediately
# at $3.18/h. That is a measurement about sm_80's pools at one point in
# time; nothing here establishes how any other arch's multi-GPU pools behave,
# and an arch whose candidate list carries no `SXM` spelling is simply left
# in its declared order.
#
# At the default count this returns immediately, so every single-GPU lane's
# provisioning order is exactly its declared order
# (`test_pod_substrate.sh`'s `(ab/gpuCount D3)` leg pins that, and a mutant
# of the guard below fails it).
_rp_order_candidates_for_gpu_count() {
  [ "$RP_GPU_COUNT" -gt 1 ] || return 0
  local combo ordered=()
  for combo in "${cand[@]}"; do
    case "${combo##*|}" in *SXM*) ordered+=("$combo") ;; esac
  done
  for combo in "${cand[@]}"; do
    case "${combo##*|}" in *SXM*) ;; *) ordered+=("$combo") ;; esac
  done
  cand=("${ordered[@]}")
}

rp_deploy_arch() { # $1=arch
  local cand
  case "$1" in
    # sm_80 has NO useful same-SASS sibling on RunPod (no A800/A30 in the
    # catalog; A100-SXM4-40GB is half-VRAM and community-only) — both 80GB
    # variants are already listed, so an sm_80 drought has no fallback lever.
    a100)    cand=("SECURE|NVIDIA A100 80GB PCIe" "COMMUNITY|NVIDIA A100 80GB PCIe" "SECURE|NVIDIA A100-SXM4-80GB" "COMMUNITY|NVIDIA A100-SXM4-80GB") ;;
    # sm_89, 48GB-class: RTX 6000 Ada is the equal-VRAM same-SASS sibling.
    l40s)    cand=("SECURE|NVIDIA L40S" "COMMUNITY|NVIDIA L40S" "SECURE|NVIDIA RTX 6000 Ada Generation" "COMMUNITY|NVIDIA RTX 6000 Ada Generation") ;;
    # sm_90 capacity fallbacks (the H100 SXM/PCIe pools run dry too): H100 NVL (94GB) and H200 (141GB) are
    # same-GH100 sm_90 SASS with >= VRAM, so both the prove proof and a
    # dev pod's memory envelope are preserved; H200 last (priciest).
    h100)    cand=("SECURE|NVIDIA H100 80GB HBM3" "SECURE|NVIDIA H100 PCIe" "COMMUNITY|NVIDIA H100 80GB HBM3" "COMMUNITY|NVIDIA H100 PCIe" "SECURE|NVIDIA H100 NVL" "COMMUNITY|NVIDIA H100 NVL" "SECURE|NVIDIA H200" "COMMUNITY|NVIDIA H200") ;;
    # sm_86 capacity fallbacks (the A40 pool runs dry): RTX A6000 (48GB,
    # ECC — closest A40 twin) then RTX 3090
    # (24GB, no ECC — same GA10x sm_86 SASS, identical correctness proof;
    # ordered last for the ECC difference, which affects fault tolerance,
    # never the computed values a parity/capability suite asserts).
    a40)     cand=("SECURE|NVIDIA A40" "COMMUNITY|NVIDIA A40" "SECURE|NVIDIA RTX A6000" "COMMUNITY|NVIDIA RTX A6000" "SECURE|NVIDIA GeForce RTX 3090" "COMMUNITY|NVIDIA GeForce RTX 3090") ;;
    l4)      cand=("SECURE|NVIDIA L4" "COMMUNITY|NVIDIA L4" "SECURE|NVIDIA GeForce RTX 4090" "COMMUNITY|NVIDIA GeForce RTX 4090") ;; # sm_89, 24GB-class: 4090 = equal-VRAM same-SASS sibling
    *) echo "::error::unknown arch '$1' (want: a100|l40s|h100|a40|l4)"; return 2 ;;
  esac
  RP_ARCH="$1"
  _rp_order_candidates_for_gpu_count
  rp_deploy_live "${cand[@]}"
}

# A100 (sm_80) — the arch that proves the compute_80 floor.
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

# Like `rp_run_remote`, plus an INACTIVITY watchdog: `$1` seconds
# (optional; defaults to the validated `RP_INACTIVITY` global -- every real
# caller omits `$1` and gets that default) of silent (no new output byte)
# remote stdout+stderr kills the ssh session as a hang, rather than waiting
# out the full RP_TIMEOUT budget -- the two axes are independent (a leg can
# be busy-but-slow, which only RP_TIMEOUT should catch, or silent-and-stuck,
# which this watchdog catches far earlier). `$1` exists so a FIXTURE can pass
# a short test-local threshold as a plain function argument rather than a
# second committed `RP_INACTIVITY=<n>` assignment beside the one default
# above -- see `test_gpu_prove_lane.sh`'s own use.
# File-backed streaming (never a `$(...)` capture, which would buffer the
# ENTIRE output in memory and print nothing until the process exits) so a
# caller sees the same bytes live, exactly as `rp_run_remote`'s direct pipe
# does.
#
# This is a GENERIC primitive: it knows the `::group::`/`PROVE_GROUP_RC
# name=<n> rc=<v>` marker SHAPE (to name a hung/cut group in its own
# diagnostic), but nothing about which group names are meaningful, which
# are gating, or the bench-cut exception -- deciding whether a leg PASSES
# from those markers is the CALLER's job (e.g. `runpod_gpu_prove.sh`'s own
# `PROVE_GROUPS` array and driver rule), never this shared function's.
#
# Returns:
#   * 76, after printing a "NO PROGRESS" diagnostic naming the last-opened
#     group and every `PROVE_GROUP_RC` marker seen so far, if RP_INACTIVITY
#     seconds pass with no new output byte. This is NOT the remote script's
#     own exit status -- it never got to choose one; the watchdog killed it.
#   * the remote script's own status, VERBATIM, whenever ssh itself exits on
#     its own (0, 1, 97, 255, an in-suite 124/76 the remote script chose to
#     exit with -- anything ssh reports as ITS OWN status) -- ALWAYS taken
#     from `wait $pid` after the FINAL drain to EOF, never guessed early. A
#     status-124 exit additionally gets a "BUDGET" diagnostic (same
#     group/marker naming), printed WITHOUT changing the exit code, but
#     ONLY when the drained output carries no `PROVE_EXIT=` line -- the 124
#     discriminator: an in-suite 124 (the remote script's OWN exit, with
#     `PROVE_EXIT=124` already printed) is a different case, indistinguishable
#     from a real budget cut by exit code alone, and gets no extra line.
# ONE marker grammar, one parser: `PROVE_GROUP_RC name=<n> rc=<v>` -- used by BOTH this file's own
# `rp_run_remote_watched` (live-stream bookkeeping, below) and
# `runpod_gpu_prove.sh`'s `rp_prove_verdict` (which reads an already-drained
# log file) -- never two independently-drifting copies of the same
# match+extract logic. `ci/scripts/prove_surface.py`'s `PROVE_GROUP_RC_RE`
# is the Python-side twin of this grammar; a cross-parser fixture in
# `test_gpu_prove_lane.sh` feeds the identical marker text to both this
# function and that regex and asserts identical (name, rc) extraction.
#
# `$1` = a candidate line (a whole logical line, never a fragment). Sets
# `RP_PARSED_MARKER_NAME`/`RP_PARSED_MARKER_RC` (plain globals -- bash has
# no return-by-value) and returns 0 on a match; UNSETS both and returns 1
# on a non-match, so a caller can never read a PRIOR call's stale values on
# a miss.
#
# Tightened to the EXACT shape `ci/scripts/prove_surface.py`'s
# `PROVE_GROUP_RC_RE` accepts: a bash regex
# (`[[ =~ ]]`), never substring slicing, so the two languages agree on every
# edge case a slicing-based extractor would silently mishandle --
# double-spaced fields, an empty name/rc, a non-numeric rc, two markers on
# one line (matches the FIRST only, exactly like Python's `.search()`),
# leading/trailing junk (matched anywhere in the line, no anchors), and a
# CRLF line ending (a trailing `\r` is stripped up front so it can never be
# folded into a captured value). `test_gpu_prove_lane.sh`'s cross-parser
# fixture feeds both parsers the identical set of inputs and asserts
# identical (name, rc) or identical NOMATCH.
rp_parse_prove_marker() {
  unset RP_PARSED_MARKER_NAME RP_PARSED_MARKER_RC
  local line="${1%$'\r'}"
  if [[ "$line" =~ PROVE_GROUP_RC\ name=([^[:space:]]+)\ rc=(-?[0-9]+) ]]; then
    RP_PARSED_MARKER_NAME="${BASH_REMATCH[1]}"
    RP_PARSED_MARKER_RC="${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

# The REMOTE checkout text every GPU leg ships to its host: the tree it
# proves is the exact commit the workflow ran at (`PROVE_EXPECT_SHA`,
# fetched by sha, depth 1 — a branch name is a moving target: a push that
# moves the branch between workflow start and clone trips the wrong-tree
# guard below). Without an expected sha (a hand
# run) the ref is cloned. Emits shell lines for the remote script's
# heredoc: `cd /root`, a fresh `jammi-ai` dir, and leaves the caller INSIDE
# it. The root is RP_REMOTE_ROOT (default /root) — a parameter, never a
# literal, so a fixture executes this text unmodified in a sandbox.
# Every leg then deepens the history WITHOUT blobs (commits and trees
# only, seconds): the artifact registry's ancestry rule (`git merge-base
# --is-ancestor`, check_cuda_run_artifacts.py rule (d)/(k)) answers "not an
# ancestor" for every commit but HEAD on a depth-1 history, which fails the
# pod-leg synthetic tests. `$1` = the ref (used only without
# PROVE_EXPECT_SHA), `$2` = the repo.
rp_remote_checkout_lines() {
  local ref="${1:?rp_remote_checkout_lines needs a ref}" repo="${2:?rp_remote_checkout_lines needs a repo url}"
  local root="${RP_REMOTE_ROOT:-/root}"
  # Every step is chained fail-closed: a checkout that cannot enter its root,
  # clone, fetch or check out STOPS the remote script by name. Nothing may run
  # in an unknown working directory (on a host with no /root, a stubbed clone
  # would otherwise write a tree mirror into the caller's own checkout).
  if [ -n "${PROVE_EXPECT_SHA:-}" ]; then
    cat <<LINES
cd "${root}" || { echo "::error::remote root ${root} is not enterable" >&2; exit 1; }
rm -rf jammi-ai && mkdir jammi-ai && cd jammi-ai || { echo "::error::could not create ${root}/jammi-ai" >&2; exit 1; }
git init -q && git remote add origin "${repo}" || { echo "::error::could not initialise the checkout" >&2; exit 1; }
git fetch -q --depth 1 origin "${PROVE_EXPECT_SHA}" \\
  || { echo "::error::could not fetch the exact commit ${PROVE_EXPECT_SHA} from ${repo}" >&2; exit 1; }
git checkout -q --detach FETCH_HEAD || { echo "::error::could not check out ${PROVE_EXPECT_SHA}" >&2; exit 1; }
git fetch -q --filter=blob:none --unshallow origin \\
  || { echo "::error::could not deepen the checkout (blobless --unshallow failed) -- the artifact registry's ancestry rule cannot run on a depth-1 history" >&2; exit 1; }
LINES
  else
    cat <<LINES
cd "${root}" || { echo "::error::remote root ${root} is not enterable" >&2; exit 1; }
rm -rf jammi-ai || exit 1
git clone --depth 1 -b "${ref}" "${repo}" jammi-ai || { echo "::error::could not clone ${ref} from ${repo}" >&2; exit 1; }
cd jammi-ai || exit 1
git fetch -q --filter=blob:none --unshallow origin \\
  || { echo "::error::could not deepen the checkout (blobless --unshallow failed) -- the artifact registry's ancestry rule cannot run on a depth-1 history" >&2; exit 1; }
LINES
  fi
}

# ONE grammar, hand-mirrored across languages -- bash cannot
# `import` `ci/scripts/prove_surface.py`'s `PROVE_SHA_RE`, so
# this function's `[0-9a-f]+`-after-`PROVE_SHA=`, first-match-only, no-
# anchors shape is a BY-HAND copy of it, the same discipline
# `rp_parse_prove_marker` above follows for `PROVE_GROUP_RC`.
# `test_gpu_prove_lane.sh`'s `xp_sha_div_check` cross-parser fixture is what
# actually pins the two mirrored grammars to agreement: it feeds both
# parsers the identical set of inputs and asserts identical (sha) or
# identical NOMATCH.
rp_parse_prove_sha() {
  unset RP_PARSED_PROVE_SHA
  local line="${1%$'\r'}"
  if [[ "$line" =~ PROVE_SHA=([0-9a-f]+) ]]; then
    RP_PARSED_PROVE_SHA="${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

rp_run_remote_watched() {
  local inactivity="${1:-$RP_INACTIVITY}"
  # `$2` = the poll interval in seconds (default 5; the real caller in
  # `runpod_gpu_prove.sh` always omits it and gets 5, except for its own
  # `RP_WATCH_POLL_S` escape hatch -- fixture/diagnostic-only, see that
  # script's own doc). `sleep` accepts a fractional argument on both GNU and
  # BSD coreutils, so a fixture can drive this well below 1s (0.2s) to keep
  # every timing-sensitive case's actual wall-clock cost, and its distance
  # from the poll boundary, small and deterministic -- a fixed 5s poll
  # combined with fixture thresholds only 1.5-2x its own size makes fixtures
  # flaky on a loaded host: detection latency is bounded by the poll
  # interval, so a threshold
  # smaller than (or close to) the poll makes the "5s tick" the true
  # deciding clock, not the declared threshold.
  local poll_interval="${2:-5}"
  local out; out="$(mktemp)"
  local preamble; preamble="$RP_ENV_PREAMBLE"
  { printf '%s\n' "$preamble"; cat; } \
    | ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" "timeout ${RP_TIMEOUT:-3000} bash -s" \
    > "$out" 2>&1 &
  local pid=$!
  local printed=0 last_growth=$SECONDS last_group="" parse_carry=""
  local group_names=() group_rcs_assoc_keys=() group_rcs_assoc_vals=()
  # When PROVE_EXPECT_SHA is set (a workflow
  # run, never a hand run), a disagreeing PROVE_SHA marker, or (when the
  # session's own rc is 0) an absent one, is a
  # wrong-tree failure -- the ref moved between run creation and clone, or
  # a tag was moved. `wrong_tree`/`wrong_tree_got` are set by
  # `_rrw_scan_new_text` below (a closure over these, same as
  # `last_group`/`group_names`); `prove_sha_seen` distinguishes "saw a
  # matching sha" from "never saw any sha at all" for the absence check at
  # normal exit.
  local wrong_tree=0 wrong_tree_got="" prove_sha_seen=0

  _rrw_group_list() {
    local i out_str=""
    for i in "${!group_names[@]}"; do
      [ -n "$out_str" ] && out_str+=","
      out_str+="${group_names[$i]}:${group_rcs_assoc_vals[$i]}"
    done
    printf '[%s]' "$out_str"
  }

  _rrw_record_rc() {
    # $1=name $2=rc -- already extracted by `rp_parse_prove_marker`, the ONE
    # shared marker parser (never re-parsed here).
    local gname="$1" grcv="$2"
    local i found=0
    for i in "${!group_names[@]}"; do
      if [ "${group_names[$i]}" = "$gname" ]; then
        group_rcs_assoc_vals[$i]="$grcv"
        found=1
        break
      fi
    done
    [ "$found" -eq 1 ] || { group_names+=("$gname"); group_rcs_assoc_vals+=("$grcv"); }
  }

  # Sets `_RRW_CHUNK` (a plain global, NOT a `$(...)`-captured return value
  # -- a caller doing `x="$(_rrw_read_chunk ...)"` would immediately lose
  # the fix below to ITS OWN command-substitution newline-stripping): `$(...)`
  # unconditionally strips ALL trailing newlines from whatever it captures,
  # which would make every chunk look "complete" to the partial-line carry
  # logic below even when the real bytes end mid-line -- append a sentinel
  # byte, capture, then strip exactly that sentinel, so any REAL trailing
  # newline(s) in the chunk survive intact.
  _rrw_read_chunk() {
    local from="$1"
    _RRW_CHUNK="$(tail -c "+${from}" "$out"; printf 'X')"
    _RRW_CHUNK="${_RRW_CHUNK%X}"
  }

  _rrw_scan_new_text() {
    # Parse only COMPLETE (newline-terminated) lines out of `parse_carry +
    # $1`; the trailing incomplete remainder (if `$1` does not itself end in
    # a newline) is carried forward to the NEXT poll tick rather than
    # mis-parsed as a whole line.
    local combined="${parse_carry}$1"
    parse_carry=""
    if [ "${combined: -1}" != $'\n' ]; then
      parse_carry="${combined##*$'\n'}"
      combined="${combined%"$parse_carry"}"
    fi
    [ -z "$combined" ] && return
    local line
    while IFS= read -r line; do
      case "$line" in
        *"::group::"*) last_group="${line#*::group::}" ;;
      esac
      if rp_parse_prove_marker "$line"; then
        _rrw_record_rc "$RP_PARSED_MARKER_NAME" "$RP_PARSED_MARKER_RC"
      fi
      if [ -n "${PROVE_EXPECT_SHA:-}" ] && rp_parse_prove_sha "$line"; then
        prove_sha_seen=1
        if [ "$RP_PARSED_PROVE_SHA" != "$PROVE_EXPECT_SHA" ]; then
          wrong_tree=1
          wrong_tree_got="$RP_PARSED_PROVE_SHA"
        fi
      fi
    done <<< "$combined"
  }

  # ONE shared final-flush used by BOTH terminal arms (the inactivity-kill
  # arm below AND the normal-exit arm after the main loop). Without the carry
  # flush on the kill arm, an unterminated final `::group::`/`PROVE_GROUP_RC`
  # landing right as the kill fires is silently dropped or mis-attributed
  # to whatever group was open BEFORE it (e.g.
  # `NO PROGRESS ... in group "kernels-default"` for a cut genuinely inside
  # `kernels-cuda`; `groups: []` for an unterminated final marker). Drains
  # any bytes written since the last read, THEN forces any still-carried
  # partial final line into the marker scan (a `PROVE_EXIT=<n>` or
  # `PROVE_GROUP_RC` line the remote never newline-terminated before dying
  # must still count). `_rrw_scan_new_text` ALREADY prepends `parse_carry`
  # to its own `$1` -- passing `$parse_carry` here TOO would double it,
  # corrupting the final unterminated line into e.g.
  # `cut group "kernels-cuda::group::kernels-cuda"`. Bare `$'\n'` supplies
  # only the missing terminator the carry itself lacks.
  _rrw_flush_carry() {
    local size; size=$(wc -c < "$out" 2>/dev/null || echo 0)
    if [ "$size" -gt "$printed" ]; then
      _rrw_read_chunk "$((printed + 1))"
      printf '%s' "$_RRW_CHUNK"
      _rrw_scan_new_text "$_RRW_CHUNK"
      printed=$size
    fi
    if [ -n "$parse_carry" ]; then
      _rrw_scan_new_text $'\n'
    fi
  }

  # Shared diagnostic -- STDERR, like the 76/124
  # arms, so it reaches the job log regardless of which terminal arm fires
  # it. `$1` is the observed sha, or empty for the absence case.
  _rrw_wrong_tree_diag() {
    echo "=== GPU prove: WRONG TREE expected=${PROVE_EXPECT_SHA} got=${1:-none} ===" >&2
  }

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$poll_interval"
    local size; size=$(wc -c < "$out" 2>/dev/null || echo 0)
    if [ "$size" -gt "$printed" ]; then
      _rrw_read_chunk "$((printed + 1))"
      printf '%s' "$_RRW_CHUNK"
      _rrw_scan_new_text "$_RRW_CHUNK"
      printed=$size
      last_growth=$SECONDS
      # Wrong-tree kill: checked EVERY tick right after a scan
      # sees new bytes, so a disagreeing PROVE_SHA= line is caught within
      # ONE poll tick of arriving -- never deferred to the inactivity arm or
      # the normal exit, which could be minutes away.
      if [ "$wrong_tree" = "1" ]; then
        kill -TERM "$pid" 2>/dev/null
        sleep 1
        kill -KILL "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        _rrw_flush_carry
        _rrw_wrong_tree_diag "$wrong_tree_got"
        rm -f "$out"
        return 77
      fi
    elif [ $((SECONDS - last_growth)) -ge "$inactivity" ]; then
      kill -TERM "$pid" 2>/dev/null
      sleep 1
      kill -KILL "$pid" 2>/dev/null
      # `wait "$pid"` reaps the ENTIRE backgrounded pipeline job (both the
      # `{ preamble; cat; }` head and ssh, the tail this PID names) --
      # verified empirically: a bash pipeline backgrounded as one job is
      # reaped as one unit, so a SECOND explicit `wait` on the head's own
      # recorded pid returns 127 ("no such job") here, a pure no-op.
      wait "$pid" 2>/dev/null
      _rrw_flush_carry
      echo "=== GPU prove: NO PROGRESS for ${inactivity}s in group \"${last_group}\"; groups: $(_rrw_group_list) ===" >&2
      rm -f "$out"
      return 76
    fi
  done
  wait "$pid"; local rc=$?
  # Final drain to EOF after `wait $pid`, BEFORE the 124-discriminator check
  # below reads the file -- the last group's marker and `PROVE_EXIT=` line
  # can land milliseconds before ssh's own exit, inside what would otherwise
  # be the NEXT poll window. The wrong-tree check below runs AFTER this
  # flush (a mismatching PROVE_SHA= landing only in the final flush must
  # still be caught). A MISMATCH wins regardless of `$rc` or
  # which markers landed -- a session that explicitly asserted the WRONG
  # identity proved nothing, whatever its own exit code claims. ABSENCE is
  # narrower: it wins ONLY when the session's own rc
  # is 0 (it claimed success without ever asserting identity); when rc is
  # non-zero the absence of a `PROVE_SHA=` marker is exactly what a
  # transport death (rc 255) or a genuine budget
  # cut (rc 124) looks like -- the session never got far enough to echo it
  # -- so it falls through UNCHANGED to the existing 124/255 handling below,
  # never relabeled as wrong-tree and never suppressing the BUDGET
  # diagnostic.
  _rrw_flush_carry
  if [ -n "${PROVE_EXPECT_SHA:-}" ]; then
    if [ "$wrong_tree" = "1" ]; then
      _rrw_wrong_tree_diag "$wrong_tree_got"
      rm -f "$out"
      return 77
    fi
    if [ "$prove_sha_seen" != "1" ] && [ "$rc" -eq 0 ]; then
      # Absence-with-a-claimed-success is a failure, same doctrine as
      # check_gpu_prove_once.py's P1 zero-producers rule: identity was
      # never asserted, so this leg proved nothing about the tree it ran on even though it reports success.
      _rrw_wrong_tree_diag ""
      rm -f "$out"
      return 77
    fi
  fi
  if [ "$rc" -eq 124 ] && ! grep -q '^PROVE_EXIT=' "$out"; then
    echo "=== GPU prove: BUDGET (RP_TIMEOUT=${RP_TIMEOUT:-3000}s) cut group \"${last_group}\"; groups: $(_rrw_group_list) ===" >&2
  fi
  rm -f "$out"
  return "$rc"
}

# Runs `$2...` (stdin inherited unchanged — a caller pipes into this
# function exactly as it would pipe into the command directly) in the
# BACKGROUND and kills it — TERM, then KILL a second later if TERM did not
# take — if it is still alive after `$1` seconds; portable pure-bash+kill,
# never the GNU-coreutils-only `timeout` binary (absent on a BSD host, per
# `_rp_ls_remote`'s own doc above — the identical constraint applies here).
# stdout+stderr are captured to a temp file and printed once the command
# (or the kill) has actually finished, so the caller's own `$(...)`
# capture sees the SAME merged-stream text either way. Returns the
# command's own exit code, or 124 (the same convention GNU `timeout` uses)
# if it had to be killed.
_rp_bounded_capture() {
  local bound="$1"; shift
  local outfile; outfile="$(mktemp)"
  "$@" > "$outfile" 2>&1 &
  local pid=$! dl=$((SECONDS + bound))
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$SECONDS" -ge "$dl" ]; then
      kill -TERM "$pid" 2>/dev/null
      sleep 1
      kill -KILL "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      cat "$outfile"; rm -f "$outfile"
      return 124
    fi
    sleep 0.2
  done
  wait "$pid"; local rc=$?
  cat "$outfile"; rm -f "$outfile"
  return "$rc"
}

# Polls a live pod at a fixed interval, via a caller-supplied REMOTE check
# script (bash source, delivered over stdin exactly like rp_run_remote —
# never interpolated into the ssh command line, so it can contain quotes
# freely), until the script reports a verdict or the wall-clock TIMEOUT
# elapses. The primitive behind gpu-dev.sh's `wait-seed`/`wait-job` verbs.
#
# The remote script's OWN exit code is the whole contract, and only four
# values are legitimate: 0 (success — the thing being waited on finished
# cleanly), 1 (a NAMED failure — a failure marker), 2 ("still running,
# nothing to report yet — keep polling"), 3 ("no evidence this ever ran":
# neither a live session nor a marker). A 3 is a real state a pod passes
# through right after `up`/`run` returns, before its detached session has
# started, so it is tolerated for RP_WAIT_GRACE_SECS (default 300) from the
# first poll and is a NAMED failure after that — a session that never
# appeared is one that never ran. Every OTHER exit code — ssh's own 255 on a
# refused/dropped connection, a timeout wrapper's 124, or literally anything
# else, since gpu-dev.sh never hands this function a script that exits any
# other way on purpose — is therefore unambiguous: the poll could not be
# answered.
#
# LOAD-BEARING: an unanswerable poll is NEVER treated as rc-2's "still
# running". A watcher that collapses "no evidence yet" and "could not check
# at all" into the same silent, keep-waiting branch idles forever — a
# dropped SSH connection reads back exactly like a healthy in-progress
# build. Here the two are different signals:
# rc 2 resets the transport-failure counter (a real, reachable "not done
# yet"); anything else increments it, and RP_WAIT_MAX_TRANSPORT_FAILS (a
# caller-supplied count, never a single blip — a healthy pod can drop one
# poll) consecutive transport failures exits LOUDLY, naming the transport
# failure by count and last exit code, rather than continuing to poll a pod
# that may no longer even exist.
#
# $1=label (for messages only) $2=remote check script $3=poll interval
# (seconds) $4=overall timeout (seconds) $5=max consecutive transport
# failures before giving up loudly.
# Returns 0 (success), 1 (named failure — see the remote script's own
# stdout/stderr for which), 2 (transport failure — too many consecutive
# unreachable polls), 3 (timed out with no verdict either way).
rp_wait_poll() {
  local label="$1" script="$2" interval="$3" timeout="$4" max_fail="$5"
  local deadline=$((SECONDS + timeout)) consec=0 out rc remaining
  local grace_deadline=$((SECONDS + ${RP_WAIT_GRACE_SECS:-300}))
  # RP_SSHO's own ConnectTimeout=10 bounds only the TCP handshake, and the
  # remote-side `timeout 20 bash -s` below bounds only a shell that has
  # already STARTED on the pod — neither one bounds the gap between them
  # (SSH protocol negotiation/auth, or a channel that goes silent after
  # connecting). A pod that accepts TCP but stops answering — the exact
  # OOM-thrashing-after-a-CUDA-build case this verb exists to watch for — or
  # an auth prompt (a passphrase-protected key, a fallback to interactive
  # password auth) hangs the `ssh` client itself, indefinitely, with `consec`
  # never incrementing and this loop's own deadline (below) never reached:
  # the exact silent-idle fail-open the verb exists to close. `-oBatchMode=yes` (the SAME pattern `_rp_ls_remote`'s own
  # GIT_SSH_COMMAND already uses) refuses to prompt at all, failing fast
  # instead of hanging on one; `-oServerAliveInterval=10
  # -oServerAliveCountMax=3` makes the CLIENT itself detect a connection
  # that has gone silent after connecting and give up within ~30s, rather
  # than waiting on channel data that may never arrive.
  #
  # ssh options are first-wins (verified with `ssh -G`).
  # `RP_SSHO` (see its own doc above) carries its own, LOOSER
  # ServerAliveInterval/CountMax (30s/6) for the session-liveness contract:
  # long silent phases like a clone/build must survive a NAT idle window;
  # the inactivity watchdog, not TCP, is what detects a genuine hang. This
  # probe's own, TIGHTER options are PREPENDED ahead of `"${RP_SSHO[@]}"`
  # below, so THIS invocation's 10s/3-try liveness contract wins by
  # ordering regardless of what the shared array carries — a probe wants to
  # fail FAST on a genuinely silent connection, a different, deliberately
  # TIGHTER contract than an interactive `attach`/`shell` session (or the
  # prove lane's own long clone/build phases) has any business inheriting.
  local -a wait_sshopts=(-oBatchMode=yes -oServerAliveInterval=10 -oServerAliveCountMax=3 "${RP_SSHO[@]}")
  # A SECOND, portable backstop UNDER the ssh-option hardening above (both
  # apply, neither alone): `_rp_bounded_capture`
  # runs the ssh invocation in the BACKGROUND and kills it if it exceeds
  # RP_WAIT_SSH_BOUND_SECS (default 60 — comfortably above ConnectTimeout
  # (10s) + the ServerAlive silence-detection window (~30s) + the remote
  # `timeout 20`, with real margin), using plain bash job control (`kill -0`
  # polling + `kill -TERM`/`-KILL`), never the GNU-coreutils-only `timeout`
  # binary this file's own `_rp_ls_remote` doc already notes is ABSENT on a
  # BSD host — the same portability constraint applies here. This is what
  # makes the hang path itself testable against a bare mock `ssh` stub that
  # simply sleeps (ignoring every ssh option, since it is not real openssh)
  # — real ssh's ServerAlive settings cannot be exercised by a stub, but
  # this wrapper's own kill still bounds it regardless of what the stub does.
  local -a wait_cmd=(ssh "${wait_sshopts[@]}" -p "$RP_PORT" "root@${RP_HOST}" "timeout 20 bash -s")
  echo "=== ${label}: polling every ${interval}s, up to ${timeout}s (pod ${RP_HOST:-?}:${RP_PORT:-?}) ==="
  while :; do
    # `2>&1`: MERGES, never discards — a transport failure's own stderr (ssh's
    # "Connection refused"/"Connection timed out") is exactly the evidence
    # the loud transport-failure message below needs to be more than a bare
    # exit code.
    out="$(printf '%s\n' "$script" | _rp_bounded_capture "${RP_WAIT_SSH_BOUND_SECS:-60}" "${wait_cmd[@]}" 2>&1)"
    rc=$?
    case "$rc" in
      0) echo "${out}"; echo "=== ${label}: SUCCESS ==="; return 0 ;;
      1) echo "::error::${label}: FAILURE — ${out}"; return 1 ;;
      2) consec=0; echo "${label}: still waiting — ${out}" ;;
      3)
        consec=0
        if [ "$SECONDS" -lt "$grace_deadline" ]; then
          echo "${label}: not started yet (within the ${RP_WAIT_GRACE_SECS:-300}s startup grace) — ${out}"
        else
          echo "::error::${label}: FAILURE — ${out}"
          return 1
        fi
        ;;
      *)
        consec=$((consec + 1))
        echo "::warning::${label}: poll unreachable (ssh/remote exit ${rc}) — ${consec}/${max_fail} consecutive — ${out}"
        if [ "$consec" -ge "$max_fail" ]; then
          echo "::error::${label}: TRANSPORT FAILURE — ${consec} consecutive unreachable polls (last exit ${rc})."
          echo "::error::${label}: this is NOT evidence it is still running — it means the pod could not be reached. Check it directly: gpu-dev.sh ls / ssh / the RunPod console."
          return 2
        fi
        ;;
    esac
    [ "$SECONDS" -lt "$deadline" ] || { echo "::error::${label}: timed out after ${timeout}s with no verdict"; return 3; }
    remaining=$((deadline - SECONDS))
    sleep "$(( remaining < interval ? remaining : interval ))"
  done
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
  local override="${1:-}" body out rc id age why n=0 refused=0 reason_out term_rc
  local -a refused_reasons=()
  if [ -n "$override" ]; then
    # See `_rp_validate_force_hours`'s own doc for why "0"/"00" refuse rather than sweeping, and why a leading zero
    # ("08") is accepted like any other digit string.
    _rp_validate_force_hours "$override" || return 2
  fi
  # Captured before parsing — the same capture-then-parse shape as
  # rp_pod_verify's own fix (see its doc): under `set -o pipefail`, a direct
  # `rp_gql | python3 ...` pipe reports CURL's own exit code whenever curl
  # fails, even if python still received (and successfully parsed) a
  # complete body. This function's own "rc -ne 0" check below already
  # treats every nonzero rc uniformly (return 1, "could not enumerate
  # pods"), so there was no distinct-code aliasing bug here the way there
  # was in rp_pod_verify's 0/1/2/3 codes — applied anyway so this function's
  # correctness does not depend on pipefail's last-failing-command rule
  # working out by coincidence.
  body="$(rp_gql '{"query":"query{ myself{ pods{ id name desiredStatus createdAt runtime{ uptimeInSeconds } } } }"}')" \
    || { echo "::error::sweep could NOT enumerate pods; orphans may exist unseen: RunPod query failed"; return 1; }
  out="$(printf '%s' "$body" | python3 -c "$(_rp_ttl_parser_pysrc)
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
if not isinstance(me['pods'], list):
    print('response contained no pod list'); sys.exit(3)
# Every read below is TOTAL: a row of the wrong shape is
# 'could not read the list' (exit 3), never an uncaught exception's exit 1.
try:
  now = datetime.datetime.now(datetime.timezone.utc)
  for p in me['pods']:
    if not isinstance(p, dict):
        raise ValueError('pod row is not an object: %r' % (p,))
    name = p.get('name') or ''
    if not isinstance(name, str) or not name.startswith(prefix):
        continue
    if not isinstance(p.get('id'), str) or not p.get('id'):
        raise ValueError('pod row %r carries no readable id' % (name,))
    # Age comes from createdAt, never from runtime.uptimeInSeconds. Measured on
    # live pods: uptime is null for the first minutes of a perfectly healthy pod,
    # so treating null as 'unreachable' terminates in-flight CI runs. createdAt is
    # also a true since-RENTAL clock, so a container restart cannot reset it —
    # which uptime can, letting a pod live forever in deadline-length increments.
    age = None
    ca = p.get('createdAt')
    if ca:
        try:
            age = int((now - datetime.datetime.fromisoformat(str(ca).replace('Z', '+00:00'))).total_seconds())
        except Exception:
            age = None
    if p.get('desiredStatus') != 'RUNNING':
        print(p['id'], age if age is not None else -1, 'not-running'); continue
    if age is None:
        # Cannot establish age. Killing on this basis is how healthy pods die, so
        # surface it instead and let an operator retire it by id.
        print('UNAGEABLE', p['id'], name); continue
    if override:
        limit = int(override) * 3600
    else:
        limit = _rp_parse_ttl_seconds(prefix, name)
        if limit is None:
            # A prefixed name with no parseable -ttl<H> cannot be judged
            # either — named, never terminated on a guess.
            print('UNPARSEABLE', p['id'], name); continue
    if age > limit:
        print(p['id'], age, 'past-deadline-%ds' % limit)
except Exception as e:
    print('could not read the pod list: %s' % e); sys.exit(3)
")"
  rc=$?
  # A failed query must never read as "nothing to clean up" — this is the
  # independent backstop, and a silent green here is how a guard stops guarding.
  if [ "$rc" -ne 0 ]; then
    echo "::error::sweep could NOT enumerate pods; orphans may exist unseen: ${out}"
    return 1
  fi
  [ -n "$out" ] || { echo "sweep: queried OK — no orphaned ${RP_POD_PREFIX} pods"; return 0; }
  local unjudged=()
  while read -r id age why; do
    [ -n "$id" ] || continue
    if [ "$id" = "UNAGEABLE" ] || [ "$id" = "UNPARSEABLE" ]; then
      # A pod this sweep cannot JUDGE (no usable createdAt, or a prefixed
      # name with no parseable -ttl<H>) is BILLING with no deadline this
      # sweep could establish — never "nothing to reap", never terminated on
      # a guess. It is named; the rest of the list is still judged and swept
      # THIS run (one unjudgeable pod must not shield a real orphan behind
      # it forever); the sweep exits 1 at the end so the reap cron reddens
      # until an operator retires it BY ID (`rp_terminate <id>`) — the
      # override cannot help, there is no age to apply it to.
      unjudged+=("pod ${age} (${why}): $([ "$id" = "UNAGEABLE" ] && echo 'no usable createdAt' || echo 'no parseable -ttl<H> in its name') — retire it by id with rp_terminate ${age} if it is an orphan")
      continue
    fi
    reason_out="$(rp_terminate "$id")"; term_rc=$?
    if [ "$term_rc" -eq 0 ]; then
      echo "::warning::swept pod ${id} (${why}, age ${age}s)"
      n=$(( n + 1 ))
    else
      echo "::warning::terminate refused: ${reason_out:-unknown reason} (pod ${id})"
      refused=$(( refused + 1 ))
      refused_reasons+=("${id}: ${reason_out:-unknown reason}")
    fi
  done <<< "$out"
  echo "sweep: terminated ${n} orphaned pod(s) (${refused} terminate(s) refused, ${#unjudged[@]} could not be judged)"
  if [ "${#unjudged[@]}" -gt 0 ]; then
    local u
    for u in "${unjudged[@]}"; do echo "::error::${u}"; done
  fi
  # An unexpected terminate refusal (auth/rate-limit/API error) is never folded
  # into a silent 0 here. A refused terminate leaves a pod BILLING with no
  # further sweep attempt this run, which reap's own "could not enumerate
  # must never read as nothing to clean up" doctrine treats identically to
  # an enumeration failure.
  if [ "$refused" -gt 0 ]; then
    echo "::error::sweep: ${refused} terminate(s) refused: ${refused_reasons[*]}"
    return 1
  fi
  [ "${#unjudged[@]}" -eq 0 ] || return 1
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
# rp_run_remote), turn the compile-wrapper off, and place the repo at $1
# (default: main).
#
# The cargo REGISTRY is deliberately not cached. It looks expensive — the CI image
# wipes /usr/local/cargo/registry, so every pod re-fetches the whole
# arrow/datafusion/candle tree — but measured on a RunPod host that fetch is 9s
# for 868 crates, because datacenter bandwidth makes it free. Restoring a 285MB
# tarball was slower than just fetching. The real cold cost is COMPILATION, which
# the pod-build-substrate (seed + clone; see docs/maintainer/dev-gpu.md) addresses.
#
# There is deliberately no S3-backed sccache here. Measured on a live pod
# (docs/maintainer/dev-gpu.md): sccache gives ZERO cross-target-dir cache
# reuse for rustc units on this image (every populate-then-reuse pair against
# a FRESH CARGO_TARGET_DIR re-misses everything sccache had just written)
# while adding ~+33% wall clock to every build that runs it. `CARGO_BUILD_RUSTC_WRAPPER=`
# below turns the wrapper off outright; `.cargo/config.toml`'s repo-wide
# `rustc-wrapper = "sccache"` default is untouched (a pod-local override, not a
# repo edit) and every OTHER (non-pod) build keeps using it.
#
# An omitted ref means main; an empty one is a caller bug, not a default. The
# distinction is made on argument COUNT, because `${1:-main}` collapses the two
# and turns "the ref I computed is empty" into a silent, successful boot on main.
#
# Returns 0 (pod is on $1), 2 (the ref itself is malformed), 3 (the image ships
# no git, so the pod is on NO ref and RP_REF stays empty), or the remote script's
# code for a real failure.
rp_bootstrap() { # $1=git ref (optional; default main)
  local ref=main rc
  [ $# -gt 0 ] && ref="$1"
  rp_ref_check "$ref" || return 2
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

# Wrapper-off (see rp_bootstrap's doc for the sccache measurement). Every shell
# that sources /root/.jammi_env — interactive, \`run\`'s detached tmux job, the
# seed build, a clone build — gets CARGO_BUILD_RUSTC_WRAPPER= (empty, which
# overrides \`.cargo/config.toml\`'s repo-wide \`rustc-wrapper = "sccache"\` for
# THIS shell only) and CARGO_INCREMENTAL=0 (the member-free seed's own
# precondition: an incremental dir surviving \`cargo clean\` is exactly the
# drift class that seed cleaning exists to remove — see pod_seed_target.sh).
{ echo 'export CARGO_BUILD_RUSTC_WRAPPER='
  echo 'export CARGO_INCREMENTAL=0'; } >> /root/.jammi_env

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

# One literal, one variable — bootstrap ALWAYS targets this exact path
# regardless of any --tree the caller later selects (it is the checkout
# every tree's own /root/.jammi-seed is eventually built FROM), so this is
# the tooling's second (of exactly two) "/root/jammi-ai" literal site rather
# than a call through rp_tree_dir: the value here can never legitimately
# change with a tree name, so routing it through that function would be
# indirection with no real degree of freedom behind it.
jammi_dir=/root/jammi-ai
if [ ! -d "\${jammi_dir}/.git" ]; then
  git clone --filter=blob:none "${RP_REPO_URL}" "\${jammi_dir}" || exit 1
fi
cd "\${jammi_dir}" || exit 1

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
