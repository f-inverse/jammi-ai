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
#                ZERO cross-target-dir reuse on this image and cost ~+33% wall,
#                ledger row 17). A RunPod network volume is deliberately NEVER
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
#                 per concurrent tree this pod hosts); add 1.2 * S_seed staging
#                 headroom once M7 (the cross-pod seed cache, phase 2, blocked
#                 on a user action — see docs/maintainer/dev-gpu.md) lands. The
#                 S_src/S_seed/S_clone byte counts are MEASURED, not guessed —
#                 see ci/scripts/perf/pod_build_timings.sh (A2) for the
#                 producer; the committed JSON under
#                 ci/artifacts/pod-build-timings/ is the citable record
#                 (dev-gpu.md quotes its S values and walls). Add 3 GB per OTHER concurrent agent
#                 CARGO_TARGET_DIR sharing this pod and 2 GB per `cargo mutants
#                 -j N` job (COPY MODE makes one full workspace+target copy per
#                 job — standing clause 1 REQUIRES COPY MODE, never
#                 `--in-place`, so a shared target dir does not report mutated
#                 sources as "Fresh"). A mutation-testing session wants >= 120 GB.
#   RP_VOLUME_GB  attached volume size in GB (default 0). The pod is deliberately
#                 disposable (see "state" above) — leave this 0 unless a caller
#                 has a specific reason to attach one.
#   RP_SSH_WAIT_SECS  wall-clock deadline on rp_deploy_live's SSH-reachability
#                 poll, in seconds (default 600). A cold host still pulling the
#                 multi-GB CUDA image can take minutes before sshd is even up;
#                 raise this rather than losing a healthy pod to the poll's own
#                 timeout.
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
# Wall-clock deadline for rp_deploy_live's SSH-reachability poll. Default 600s:
# a cold image pull alone has measured ~2 minutes, and a healthy candidate has
# needed over 4 minutes end to end (2026-08-26) — the previous fixed
# 24-iteration/10s-sleep budget (4m total) could terminate a pod that was
# still becoming reachable, not one that never would.
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
# A CONTAINMENT blacklist, not a character WHITELIST (round-3 audit on
# #388): the previous `[A-Za-z0-9_-]+` whitelist rejected every session
# name containing a `.` — including one gpu-dev.sh's OWN dispatch is happy
# to create, e.g. `RP_SESSION=bench.1 gpu-dev.sh up` — so EVERY verb
# refused against a session that had already rented a real pod, stranding
# it for its full deadline with no `down` able to reach it. Only the shapes
# that actually let RP_SESSION resolve outside RP_SESSION_ROOT are refused:
# empty, `.`, or `..` (this function only ever runs for a NAMED session —
# see the RP_SESSION_VALIDATE_SESSION gate below — so an empty value here
# is a caller bug, never `shell`'s own genuinely anonymous RP_SESSION="",
# which never reaches this check at all); anything containing a `/` (a
# multi-segment path); and a leading `-` (reads as an option to any tool
# RP_SESSION is later passed to positionally). Every OTHER shape — a dot
# anywhere but as the WHOLE name included — is a legitimate session name.
rp_session_name_check() {
  case "$RP_SESSION" in
    ''|.|..)
      echo "::error::RP_SESSION may not be empty, '.', or '..' (got '${RP_SESSION}')" >&2
      return 2 ;;
    */*)
      echo "::error::RP_SESSION may not contain '/' (got '${RP_SESSION}')" >&2
      return 2 ;;
    -*)
      echo "::error::RP_SESSION may not start with '-' (got '${RP_SESSION}')" >&2
      return 2 ;;
  esac
}
# Gated on RP_SESSION_VALIDATE_SESSION, set by gpu-dev.sh only for the verbs
# that actually RESOLVE a named session (up/attach/run/logs/push/pull/down)
# — never `ls`/`reap` (account-level; they never read RP_SESSION at all,
# and source this file from their OWN early dispatch branch before this
# variable would ever be set) and never `shell` (deliberately anonymous,
# RP_SESSION force-cleared to "" before sourcing). Without this gate, an
# UNRELATED exported RP_SESSION sitting in a maintainer's own shell for
# some other purpose made `ls`/`reap` refuse outright even though neither
# verb was ever going to consume it (round-3 audit finding, mechanism 2).
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
# a name shaped like one of THIS TOOLING'S OWN pods ("<prefix>-ttl<digits>")
# before any caller acts on it irreversibly. Never trusts a locally-recorded
# id on its own: the id could be stale (the account-side pod is already
# gone), or — the incident this closes — a DIFFERENT pod could now be
# recorded under this session's name (two processes racing `up` on the same
# alias). $1=podId.
#
# The id is the AUTHORITATIVE half of this check; the exact TTL never gates
# release, and this function no longer takes one. Two earlier versions tried
# anyway, and both were removed rather than patched further:
#   - v1 matched an EXACT "<prefix>-ttl<H>" against the session's own
#     recorded TTL. A meta file written before RP_TTL_HOURS was tracked in
#     the session record has no TTL to check, and verifying against a
#     guessed default made `down` refuse to release a real
#     `jammi-gpu-ttl72` pod this tooling itself rented, billing its full 72h
#     (round-2 audit, PR #387).
#   - v2 tried an explicit `RP_TTL_HOURS=<H>` override as the recovery path
#     for exactly that case, documented in every refusal message. It was
#     found INERT on every input: `rp_session_load`'s meta dot-source always
#     ran before the override was read, so it either clobbered an explicit
#     override (when the meta recorded a TTL) or forced it empty (when it
#     did not) — the promised remedy never once took effect (round-3 audit,
#     probe d2).
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
# refusal (round-4 audit advisory).
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
# the pod AND destroy the only record pointing at it (round-3 audit
# advisory, PR #387).
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
  # Confirmed 2026-08-26 on a kept candidate: sshd was up and
  # `-o IdentitiesOnly=yes` connected cleanly while the agent held 12
  # identities (ledger row 328).
  RP_SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o IdentitiesOnly=yes -i "$RP_SSH_KEY")
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

# A `--tree` value reaches gpu-dev.sh with no sanitization upstream, and —
# unlike RP_SESSION, which is used only for LOCAL file paths and ssh config
# blocks — a tree name (via TREE_DIR/TARGET_DIR/TMUX_SESSION, all derived
# from it) gets embedded UNQUOTED into several REMOTE heredoc scripts sent
# over ssh (`run`'s own `.jammi-job.sh` dispatch, `target --with-cutlass`,
# and now `wait-job`'s own check script, gpu_dev_job_wait_script). A value
# containing a double-quote, backtick, `$(...)`, or a path separator could
# break out of that heredoc and inject commands into the remote shell
# (round-N audit: "closing the class your new heredoc site joins" — the
# class was already reachable via `run`/`target`, `wait-job` is simply one
# more instance of it). Mirrors rp_session_name_check's OWN containment
# blacklist EXACTLY (never a whitelist — see that function's own doc for
# why): a tree name is a directory-name-shaped string with the identical
# legal shapes a session name has, so the identical rule applies for the
# identical reason. Reads `$RP_TREE_CHECK_VALUE` (the same "check a global,
# not a parameter" shape rp_session_name_check itself uses, so a caller
# resolves it exactly the same way both times).
rp_tree_name_check() {
  case "$RP_TREE_CHECK_VALUE" in
    ''|.|..)
      echo "::error::--tree may not be empty, '.', or '..' (got '${RP_TREE_CHECK_VALUE}')" >&2
      return 2 ;;
    */*)
      echo "::error::--tree may not contain '/' (got '${RP_TREE_CHECK_VALUE}')" >&2
      return 2 ;;
    -*)
      echo "::error::--tree may not start with '-' (got '${RP_TREE_CHECK_VALUE}')" >&2
      return 2 ;;
  esac
}

# Tree name -> plain SOURCE checkout directory on the pod. "jammi-ai" is the
# ONE default: the historical single-checkout location every existing
# doc/script still names directly (rp_bootstrap's own clone destination,
# below, is the other of the exactly two "/root/jammi-ai" literal sites this
# tooling permits — see test_pod_substrate.sh's grep gate). Any OTHER name
# is a caller-chosen additional tree — a plain directory under /root/trees,
# never a git worktree (a worktree add fails on the checked-out ref, and a
# shared .git couples trees that must be able to diverge — round-1
# pressure-test finding). A tree is populated by `push` (rsync, excludes
# `.git`) — NEVER by cloning the build-substrate seed: the seed is a
# CARGO_TARGET_DIR (build OUTPUT), a wholly different directory namespace
# from a tree (SOURCE) — see rp_target_dir, immediately below. Conflating
# the two made `target`'s own clone destination collide with `push --tree`'s
# rsync destination, so the first push after a `target` deleted the clone it
# had just made (round-2 audit finding 1).
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
# before failed outright on a fresh pod: `rsync: mkdir "/root/trees/<name>"
# failed: No such file or directory (2)` (esc-055, observed live on pod
# u4hfsqyu0i2qwa, 2026-08-28) — a "push first" flow gpu-dev.sh's own header
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
# for gpu-dev.sh's `run`/`wait-job` pair (round-N audit finding B3):
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
# session-liveness FIRST for exactly this reason (same defensive ordering
# as wait-seed's own tmux-session-before-markers fix for B2).
#
# `timing=1` moves the flock acquisition INSIDE this wrapper (fd 9, `flock
# -n 9`) rather than the outer `flock -n -E 75 ... bash job.sh` form `run
# --timing` used to build directly into its own LAUNCH string: a plain bash
# `if`/`else` on the flock CALL's own exit status is unambiguous, where
# checking the OUTER command's exit code for the literal value 75 could not
# tell "lock refused" apart from "the job itself happened to exit 75" (a
# real, if rare, collision the old outer-flock shape could not rule out).
# The lock is still held for the whole job's lifetime (acquired essentially
# first — only a harmless `rm -f` precedes it — released only when this
# whole script/fd closes), the SAME contract `run --timing` already had.
#
# $1=tree_dir $2=target_dir $3=job $4=token (caller-generated, unique per
# `run` invocation — carried in the marker purely for a human reading it
# directly; wait-job itself never needs to know the expected value, since
# the remove-then-rewrite-under-session-liveness discipline above already
# rules out a stale marker on its own) $5=timing (0|1).
rp_job_wrapper_with_marker_lines() {
  local tree_dir="${1:?rp_job_wrapper_with_marker_lines needs a tree dir}" \
        target_dir="${2:?rp_job_wrapper_with_marker_lines needs a target dir}" \
        job="${3:?rp_job_wrapper_with_marker_lines needs a job command}" \
        token="${4:?rp_job_wrapper_with_marker_lines needs a token}" \
        timing="${5:?rp_job_wrapper_with_marker_lines needs a timing flag}"
  printf "rm -f '%s/.jammi.exit'\n" "$tree_dir"
  if [ "$timing" = "1" ]; then
    cat <<FLOCKEOF
exec 9>/root/.jammi-timing.lock
if ! flock -n 9; then
  printf '{"token":"${token}","rc":75,"lock_refused":true}' > '${tree_dir}/.jammi.exit'
  exit 75
fi
FLOCKEOF
  fi
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
exit "\$__jammi_job_rc"
MARKEREOF
}

# Builds wait-seed's remote check script (rp_wait_poll's own doc has the rc
# contract: 0 success / 1 named failure / 2 keep polling). A plain
# function, not inlined at gpu-dev.sh's `wait-seed` call site, so a test can
# call it directly (source this file, no live pod needed) with a SANDBOXED
# seed_dir_prefix and a throwaway tmux session name, run the returned text
# locally, and assert its rc against real fixture files — the CLI-level
# mocked-ssh tests alone cannot construct the state-lattice cases round-N
# audit finding B2 depends on (both markers present at once; a marker
# alongside a LIVE session).
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
# Session-liveness checked BEFORE the completion marker, not after (B2): a
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
exit 1
SCRIPT
}

# Builds wait-job's remote check script. Reads <tree>/.jammi.exit, the
# per-run completion marker rp_job_wrapper_with_marker_lines writes (above)
# -- NEVER .jammi.log's mere existence (round-N audit finding B3: a
# flock-refused `run --timing`, or a stale log left from an earlier run,
# both read as false SUCCESS under a content-free "does a log file exist"
# check). Session-liveness is checked FIRST, same ordering as
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
  if printf '%s' "\$marker" | grep -q '"lock_refused":true'; then
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
exit 1
SCRIPT
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
    echo "  deployed ${RP_POD_ID} on ${cloud} / ${gpu}; waiting for SSH (≤${RP_SSH_WAIT_SECS}s)..."
    RP_HOST=""; RP_PORT=""
    # A wall-clock deadline, not a fixed iteration count: the old
    # 24-iteration/10s-sleep loop was a HARD-CODED 4-minute budget no caller
    # could raise, and a cold host still pulling the multi-GB CUDA image can
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
# The remote script's OWN exit code is the whole contract, and only three
# values are legitimate: 0 (success — the thing being waited on finished
# cleanly), 1 (a NAMED failure — a failure marker, or "no evidence this ever
# ran"), 2 ("still running, nothing to report yet — keep polling"). Every
# OTHER exit code — ssh's own 255 on a refused/dropped connection, a timeout
# wrapper's 124, or literally anything else, since gpu-dev.sh never hands
# this function a script that exits any other way on purpose — is therefore
# unambiguous: the poll could not be answered.
#
# LOAD-BEARING (the fail-open-watcher lesson this verb exists to close): an
# unanswerable poll is NEVER treated as rc-2's "still running". The
# hand-rolled watcher this replaces idled forever precisely because "no
# evidence yet" and "could not check at all" collapsed into the same silent,
# keep-waiting branch — a dropped SSH connection read back exactly like a
# healthy in-progress build, forever. Here the two are different signals:
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
  # RP_SSHO's own ConnectTimeout=10 bounds only the TCP handshake, and the
  # remote-side `timeout 20 bash -s` below bounds only a shell that has
  # already STARTED on the pod — neither one bounds the gap between them
  # (SSH protocol negotiation/auth, or a channel that goes silent after
  # connecting). A pod that accepts TCP but stops answering — the exact
  # OOM-thrashing-after-a-CUDA-build case this verb exists to watch for — or
  # an auth prompt (a passphrase-protected key, a fallback to interactive
  # password auth) hangs the `ssh` client itself, indefinitely, with `consec`
  # never incrementing and this loop's own deadline (below) never reached:
  # the exact silent-idle fail-open the verb claims to close (round-N audit
  # finding B1). `-oBatchMode=yes` (the SAME pattern `_rp_ls_remote`'s own
  # GIT_SSH_COMMAND already uses) refuses to prompt at all, failing fast
  # instead of hanging on one; `-oServerAliveInterval=10
  # -oServerAliveCountMax=3` makes the CLIENT itself detect a connection
  # that has gone silent after connecting and give up within ~30s, rather
  # than waiting on channel data that may never arrive. Scoped to THIS
  # ssh invocation only (a local array, never folded into the shared
  # RP_SSHO every OTHER call site also uses) — an interactive `attach`/
  # `shell` session has a different, deliberately looser liveness contract
  # this function has no business changing.
  local -a wait_sshopts=("${RP_SSHO[@]}" -oBatchMode=yes -oServerAliveInterval=10 -oServerAliveCountMax=3)
  # A SECOND, portable backstop UNDER the ssh-option hardening above (round-N
  # audit B1's "AND/OR" — this repo applies both): `_rp_bounded_capture`
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
  local override="${1:-}" body out rc id age why n=0
  if [ -n "$override" ]; then
    case "$override" in
      ''|*[!0-9]*) echo "::error::reap: hours must be a positive integer (got '${override}')"; return 2 ;;
    esac
    # "0" and "00" are all-digit (no non-digit character), so they pass the
    # shape check above untouched, and the python override BELOW treats any
    # non-empty override STRING as truthy — `if override:` is true for "0"
    # just as it is for "8" — giving `limit = 0`, under which every RUNNING
    # pod's age is "past-deadline-0s": `reap 0` mass-terminates the whole
    # account's jammi-gpu* fleet instead of refusing. Mirrors the
    # RP_DEV_TTL_HOURS >0 check in gpu-dev.sh.
    #
    # An all-digit override too large for `[ -gt ]`'s arithmetic (e.g. forty
    # 9s) still refuses — but not with a message naming the real cause. The
    # `test` builtin errors out on it ("integer expected" / "value too large
    # for defined data type", bash-version-dependent) with a non-zero status,
    # which `||` catches the same as an ordinary `false`, so the operator
    # sees this function's own "hours must be > 0" text layered under bash's
    # raw builtin error — accurate in effect (refused, exit 2) but not in
    # wording (an overflow is not "not greater than 0"). Left as-is rather
    # than special-cased: the shape check above already bounds the digit
    # count in every REALISTIC input (an operator fat-fingering a TTL), and
    # a wrong wording on a refusal is not the failure mode this guard exists
    # to prevent — a wrong wording on an ACCEPTED input would be.
    #
    # A leading zero ("08") is deliberately NOT a shape-check rejection: it
    # is all-digit like any other override, `[ "08" -gt 0 ]` compares it as
    # decimal (the `test` builtin never applies bash's `$(( ))` octal-prefix
    # rule), and Python's `int("08")` likewise reads it as decimal 8 — so
    # `reap 08` sweeps exactly like `reap 8`, never like a rejected or
    # differently-valued input.
    [ "$override" -gt 0 ] || { echo "::error::reap: hours must be > 0"; return 2; }
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
  out="$(printf '%s' "$body" | python3 -c "
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
# There is deliberately no S3-backed sccache here any more. Measured on a live
# pod (ledger row 17): sccache gave ZERO cross-target-dir cache reuse for rustc
# units on this image (every populate-then-reuse pair against a FRESH
# CARGO_TARGET_DIR re-missed everything sccache had just written) while adding
# ~+33% wall clock to every build that ran it — row 1's earlier read of a "4x
# cache hit" was against a cache that was never actually warm for the
# `--release` profile under test (row 9's correction). `CARGO_BUILD_RUSTC_WRAPPER=`
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

# Wrapper-off (row 17; M3 of the pod-build-substrate contract). Every shell
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
