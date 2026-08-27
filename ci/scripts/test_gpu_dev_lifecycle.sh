#!/usr/bin/env bash
# Mocks-only, no-network regression suite for the 2026-08-25 pod-lifecycle
# incident (ledger rows 279-289), where the repo's own tooling terminated all
# four dev pods in one night, plus the mechanism defects three follow-up
# audit rounds found in this suite's own first pass. Covers the fixes that
# closed all of it:
#
#   (1) RP_POD_CREATED must be set the MOMENT a pod id comes back from the
#       deploy mutation — not at SSH-up, minutes later — or an EXIT trap
#       firing during the reachability wait (Ctrl-C, a cancelled CI run's
#       SIGTERM) leaks a freshly rented pod for its full deadline.
#   (2) a read-only subcommand's require_pod failure must never issue
#       podTerminate — the pod this invocation did not create is not this
#       invocation's to end (rp_cleanup's RP_POD_CREATED gate).
#   (3) `up` refuses (exit 2) a session alias that already has a recorded
#       pod, even an unreachable one, unless `--replace`; `--replace` must
#       not inherit the DEAD session's own RP_TTL_HOURS/RP_IMAGE from the
#       meta file `rp_session_load` dot-sources.
#   (4) `RP_DEV_TTL_HOURS` validates under its own name (positive-integer
#       AND >0), never misattributed to RP_TTL_HOURS.
#   (5) `down` verifies a recorded pod id against the account's own live pod
#       list by ID + this tooling's own "<prefix>-ttl<digits>" NAME SHAPE —
#       never an exact TTL number, which two earlier attempts both tried and
#       both had to be removed (see rp_pod_verify's own doc in
#       runpod_lib.sh: v1's exact-TTL match refused to release a
#       genuinely-owned pod when the session's meta predated TTL tracking;
#       v2's `RP_TTL_HOURS=<H>` override "remedy" was found INERT on every
#       input, since the session meta is always loaded before any override
#       is read). A refusal KEEPS the local record and names only the
#       RunPod console as the manual path.
#   (6) `down` CONFIRMS a terminate took, rather than trusting
#       `rp_terminate`'s own thrown-away result: it re-queries the account
#       and forgets the local record only once the id is confirmed absent;
#       a still-present id (a rejected mutation, most likely) keeps the
#       record and exits 1 asking for a retry, rather than leaking the pod
#       while also destroying the only record pointing at it.
#   (7) `reap 0` / `reap 00` must REFUSE (exit 2), never sweep — "0" is
#       all-digit (no non-digit character), so a digit-shape check alone
#       lets it through, and Python's `if override:` is true for the STRING
#       "0" exactly as for "8", giving `limit = 0` under which every
#       RUNNING pod reads as already past its deadline: `reap 0` would
#       mass-terminate the whole account's jammi-gpu* fleet.
#   (8) `down` finding the recorded id ABSENT from the account (as opposed
#       to present-but-mismatched) is NOT a refusal — it is the ordinary
#       shape of "this pod already ended on its own" (its own deadline, or
#       the sweep), the single most common way a session's pod goes away.
#       `down` forgets the record and exits 0, rather than getting stuck
#       refusing forever until an operator remembers `up --replace`.
#   (9) `rp_pod_verify` captures the account query's body BEFORE parsing it,
#       rather than piping `rp_gql | python3` directly — under `set -o
#       pipefail` (every caller of this file) a direct pipe's exit status is
#       the LAST command to fail non-zero, so a `curl` exit that happens to
#       equal 3 would alias with this function's own semantic code 3 ("id
#       ABSENT — down forgets the record") even when the body curl still
#       delivered showed the pod genuinely present and correctly shaped.
#  (10) `shell` force-clears RP_SESSION to "" before sourcing runpod_lib.sh,
#       rather than merely leaving it unset — an EXPORTED RP_SESSION
#       inherited from an earlier `up`/`attach` in the same terminal would
#       otherwise point `shell`'s throwaway RP_WORK at that LIVE session's
#       own directory, and `rp_init` (called before anything is rented)
#       writes into RP_WORK regardless of whether a pod is ever actually
#       deployed.
#
# There is deliberately no "pause the sweep for one pod" feature (an earlier
# version of this branch shipped `hold`/`unhold`, backed by a `podEditJob`
# rename mutation that does not exist in RunPod's public schema —
# `PodEditJobInput` has no `name` field and two other required fields this
# tooling never had reason to send). rp_sweep's own age-based judgement
# (finding: past-deadline terminates, within-deadline does not) is still
# covered below.
#
# Every network-facing call (`curl`, i.e. every RunPod GraphQL request) is
# mocked: the function-level groups source runpod_lib.sh directly and
# override rp_gql / rp_terminate as plain bash function reassignments (legal
# because we source the file ourselves before redefining them); the CLI-level
# groups drive gpu-dev.sh as a real subprocess (to exercise its own dispatch,
# not just the library primitives) with a stub `curl` prepended onto PATH.
# `ssh` is never mocked — every fixture that needs an "unreachable pod"
# points at 127.0.0.1 on a closed local port, which the kernel refuses
# instantly with no network traffic; no fixture in this file needs a
# REACHABLE pod, so no sshd mock is needed. `ssh-keygen` runs for real (local
# keypair generation only).
#
# Run: bash ci/scripts/test_gpu_dev_lifecycle.sh
# Hermetic: no network, no GPU, no real RunPod account.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

RESULTS="$SANDBOX/results.log"
: > "$RESULTS"
# Used by function-level (sourced) subshells, which cannot write bash
# variables back to this parent process — every group appends PASS:<name> or
# FAIL:<name> lines here instead, tallied once at the very end.
record() { echo "$1:$2" >> "$RESULTS"; }

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

# ── shared fixture plumbing ─────────────────────────────────────────────────

STUBBIN="$SANDBOX/bin"
mkdir -p "$STUBBIN"
CALL_LOG="$SANDBOX/calls.log"
: > "$CALL_LOG"

# A stub `curl` that never touches the network. `rp_gql` invokes real curl as
# `curl -s URL -H '...' --data-binary "$PAYLOAD"`, so the GraphQL payload
# arrives as a plain argument (never stdin) — this stub scans "$@" for it,
# classifies by the mutation/query name it carries, logs the FULL payload (so
# a test can assert both what WAS/WASN'T sent and what it actually said), and
# answers from a per-test fixture selected by the matching MOCK_* env var.
#
# A single gpu-dev.sh invocation can query the account MORE THAN ONCE (e.g.
# `down`'s pre-terminate rp_pod_verify, then its post-terminate rp_pod_gone
# confirmation), and a test may need the account's apparent state to differ
# between those two calls (a pod present, then absent — a successful
# terminate; or present both times — a rejected one). Each `myself{...}`
# call therefore consumes MOCK_ACCOUNT_RESPONSE_1 on the FIRST call and
# MOCK_ACCOUNT_RESPONSE_2 (falling back to _1 if unset) on every call after
# — tracked via a counter FILE, since each curl invocation is a fresh
# process with no memory of the last one.
cat > "$STUBBIN/curl" <<'STUB'
#!/usr/bin/env bash
payload=""
for a in "$@"; do
  case "$a" in
    *podTerminate*|*podFindAndDeployOnDemand*|*'myself{'*|*'pod(input:'*)
      payload="$a" ;;
  esac
done
[ -n "$payload" ] && printf '%s\n' "$payload" >> "${MOCK_CALL_LOG:-/dev/null}"
case "$payload" in
  *podTerminate*) echo '{"data":{"podTerminate":true}}' ;;
  *podFindAndDeployOnDemand*) cat "${MOCK_DEPLOY_RESPONSE:?MOCK_DEPLOY_RESPONSE unset}" ;;
  *'myself{'*)
    n=0
    [ -f "${MOCK_ACCOUNT_CALL_COUNTER:?MOCK_ACCOUNT_CALL_COUNTER unset}" ] && n="$(cat "$MOCK_ACCOUNT_CALL_COUNTER")"
    n=$((n + 1))
    echo "$n" > "$MOCK_ACCOUNT_CALL_COUNTER"
    if [ "$n" = "1" ]; then
      cat "${MOCK_ACCOUNT_RESPONSE_1:?MOCK_ACCOUNT_RESPONSE_1 unset}"
    else
      cat "${MOCK_ACCOUNT_RESPONSE_2:-${MOCK_ACCOUNT_RESPONSE_1:?MOCK_ACCOUNT_RESPONSE_1 unset}}"
    fi
    ;;
  # rp_deploy_live's SSH-reachability poll (the `pod(input:{podId:...})`
  # runtime/ports query). No public port-22 mapping by default — "still
  # booting, keep polling" — so a caller that wants to exercise the poll's
  # OWN wall-clock deadline (RP_SSH_WAIT_SECS) never needs to fake a
  # reachable host; it can just let the deadline run out against this
  # default. A caller that DOES want a specific reply (a reachable mapping,
  # a malformed body) sets MOCK_PORT_RESPONSE to a fixture file.
  *'pod(input:'*)
    if [ -n "${MOCK_PORT_RESPONSE:-}" ] && [ -f "$MOCK_PORT_RESPONSE" ]; then
      cat "$MOCK_PORT_RESPONSE"
    else
      echo '{"data":{"pod":{"runtime":{"ports":[]}}}}'
    fi
    ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$STUBBIN/curl"

export PATH="$STUBBIN:$PATH"
export MOCK_CALL_LOG="$CALL_LOG"
export MOCK_ACCOUNT_CALL_COUNTER="$SANDBOX/acct-call-count"
export RUNPOD_API_KEY="test-dummy-key"
export RP_SESSION_ROOT="$SANDBOX/sessions"
export RP_S3_CONF="$SANDBOX/no-such-s3-conf"
export RP_SSH_CONFIG="$SANDBOX/ssh_config"
mkdir -p "$RP_SESSION_ROOT"

reset_log() { : > "$CALL_LOG"; }
log_has() { grep -q -- "$1" "$CALL_LOG"; }   # $1=substring

# Resets the account-query sequence for a fresh `down` invocation: no prior
# call count, and no leftover MOCK_ACCOUNT_RESPONSE_1/_2 from an earlier
# test (an unset _2 falls back to _1, so a stale _2 left exported by a
# PREVIOUS test could silently leak into a test that never meant to set one).
reset_account_seq() {
  rm -f "$MOCK_ACCOUNT_CALL_COUNTER"
  unset MOCK_ACCOUNT_RESPONSE_1 MOCK_ACCOUNT_RESPONSE_2
}

# A recorded session pointed at 127.0.0.1 on a reserved, unlisted port: the
# kernel refuses the connection immediately (ECONNREFUSED), so
# rp_session_alive fails FAST with zero network traffic — exactly the "the
# pod may have been reaped" shape require_pod exists to handle.
write_meta() { # $1=session $2=podId $3=ttlHours
  mkdir -p "$RP_SESSION_ROOT/$1"
  cat > "$RP_SESSION_ROOT/$1/meta" <<EOF
RP_POD_ID=$2
RP_HOST=127.0.0.1
RP_PORT=1
RP_ARCH=a100
RP_IMAGE=ghcr.io/f-inverse/jammi-ai-ci-cuda:latest
RP_REF=main
RP_TTL_HOURS=$3
EOF
  chmod 600 "$RP_SESSION_ROOT/$1/meta"
}

# A legacy meta -- no RP_TTL_HOURS line at all, exactly what an `up` on base
# (before RP_TTL_HOURS was added to rp_session_save) would have written.
# `down` no longer looks at RP_TTL_HOURS AT ALL (see rp_pod_verify's own
# doc), so this fixture now proves that residual case is trivially fine
# rather than exercising any special-case handling. $1=session $2=podId
write_meta_legacy() {
  mkdir -p "$RP_SESSION_ROOT/$1"
  cat > "$RP_SESSION_ROOT/$1/meta" <<EOF
RP_POD_ID=$2
RP_HOST=127.0.0.1
RP_PORT=1
RP_ARCH=a100
RP_IMAGE=ghcr.io/f-inverse/jammi-ai-ci-cuda:latest
RP_REF=main
EOF
  chmod 600 "$RP_SESSION_ROOT/$1/meta"
}

iso_from_epoch() { # $1=epoch seconds -- portable across GNU and BSD `date`
  date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ
}

echo "=== gpu-dev lifecycle safety: mocks-only regression suite ==="

# ═════════════════════════════════════════════════════════════════════════
# Group 0 — rp_cleanup's RP_POD_CREATED gate (function-level; finding 2's
# actual mechanism). Sanity-checks that the harness can detect a termination
# at all (case A) before trusting the "no termination" assertions elsewhere.
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  G0_LOG="$SANDBOX/g0-terminate.log"; : > "$G0_LOG"
  rp_terminate() { echo "$1" >> "$G0_LOG"; }
  rp_gql() { echo '{}'; }

  : > "$G0_LOG"; RP_POD_ID="podA"; RP_POD_CREATED=1; RP_KEEP=0
  rp_cleanup
  if grep -q "^podA$" "$G0_LOG"; then
    record PASS "group0-created-pod-terminates (positive control)"
  else
    record FAIL "group0-created-pod-terminates (positive control)"
  fi

  : > "$G0_LOG"; RP_POD_ID="podB"; RP_POD_CREATED=0; RP_KEEP=0
  rp_cleanup
  if grep -q "^podB$" "$G0_LOG"; then
    record FAIL "group0-uncreated-pod-not-terminated (finding 2 mechanism)"
  else
    record PASS "group0-uncreated-pod-not-terminated (finding 2 mechanism)"
  fi

  : > "$G0_LOG"; RP_POD_ID="podC"; RP_POD_CREATED=1; RP_KEEP=1
  rp_cleanup
  if grep -q "^podC$" "$G0_LOG"; then
    record FAIL "group0-kept-pod-not-terminated"
  else
    record PASS "group0-kept-pod-not-terminated"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 1 — CLI-level: a read-only subcommand (`logs`) against an unreachable
# recorded session must exit non-zero and must never issue podTerminate
# (finding 2, driven through the real gpu-dev.sh dispatch).
# ═════════════════════════════════════════════════════════════════════════
SESSION_RO="ro-fail"
write_meta "$SESSION_RO" "pod-unrelated-1" "8"
reset_log
bash "$DIR/gpu-dev.sh" logs "$SESSION_RO" >"$SANDBOX/out-ro.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  bad "finding-2: 'logs' against an unreachable session should exit non-zero (got 0)"
else
  ok "finding-2: 'logs' against an unreachable session exits non-zero (rc=$rc)"
fi
if log_has "podTerminate"; then
  bad "finding-2: 'logs' against an unreachable session issued podTerminate (regression)"
else
  ok "finding-2: 'logs' against an unreachable session issued no podTerminate"
fi
if grep -q "did not create it" "$SANDBOX/out-ro.log"; then
  ok "finding-2: cleanup reports it did not create the pod"
else
  bad "finding-2: expected the 'did not create it' notice in cleanup output"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 1b — CLI-level: an EXIT trap firing DURING the SSH-readiness wait
# (Ctrl-C, a cancelled CI run's SIGTERM) must still terminate the pod THIS
# invocation just rented (finding 1). Every deploy candidate is mocked as an
# immediate success with no reachable ports, so `shell` enters the
# reachability loop's own `sleep 10` — the exact window the audit reproduced
# the leak in — without ever needing a real SSH attempt.
# ═════════════════════════════════════════════════════════════════════════
echo '{"data":{"podFindAndDeployOnDemand":{"id":"pod-leak-test"}}}' > "$SANDBOX/deploy-leak-test.json"
export MOCK_DEPLOY_RESPONSE="$SANDBOX/deploy-leak-test.json"
reset_log
bash "$DIR/gpu-dev.sh" shell a100 --ref abcdef1234567890 >"$SANDBOX/out-sigterm.log" 2>&1 &
LEAK_PID=$!
# Bounded poll for the deploy call to actually land, then a short extra beat
# to be inside the reachability loop's own `sleep 10` before signalling —
# never a blind guess at timing.
deadline=$((SECONDS + 20))
while ! log_has "podFindAndDeployOnDemand" && [ "$SECONDS" -lt "$deadline" ]; do sleep 0.1; done
sleep 0.5
kill -TERM "$LEAK_PID" 2>/dev/null
# Bounded wait, never an indefinite `wait`: if the EXIT trap somehow does not
# fire promptly, force-kill rather than hang the whole suite — the assertion
# below then fails honestly instead of the runner wedging.
wait_deadline=$((SECONDS + 15))
while kill -0 "$LEAK_PID" 2>/dev/null && [ "$SECONDS" -lt "$wait_deadline" ]; do sleep 0.2; done
kill -0 "$LEAK_PID" 2>/dev/null && kill -KILL "$LEAK_PID" 2>/dev/null
wait "$LEAK_PID" 2>/dev/null
# The stub logs the FULL raw GraphQL payload (see the curl stub above), not a
# bare id — a podTerminate payload embeds the target id inside
# `podId:\"<id>\"`, so match the mutation name AND the id together on the
# same logged line, not an exact-match on a bare id that is never logged.
term_count="$(grep -c "podTerminate.*pod-leak-test" "$CALL_LOG")"
if [ "$term_count" = "1" ]; then
  ok "finding-1: SIGTERM during the SSH-readiness wait terminates the freshly rented pod exactly once"
else
  bad "finding-1: expected exactly one podTerminate for pod-leak-test after SIGTERM during the SSH wait (got ${term_count}); output: $(cat "$SANDBOX/out-sigterm.log")"
fi
unset MOCK_DEPLOY_RESPONSE

# ═════════════════════════════════════════════════════════════════════════
# Group 1c — CLI-level: `shell` must force RP_SESSION="" before sourcing
# runpod_lib.sh, even when the caller's own shell already has RP_SESSION
# EXPORTED (e.g. `export RP_SESSION=a100` earlier in the SAME terminal — a
# one-off `RP_SESSION=a100 gpu-dev.sh attach a100` command-prefix assignment
# does NOT persist past that single invocation, so it is not what this
# fixture reproduces). An inherited RP_SESSION would point `shell`'s
# throwaway RP_WORK at the LIVE session's own directory instead of a fresh
# temp dir, and `rp_init` (called before anything is rented) writes into
# RP_WORK unconditionally.
#
# Strengthened (round-2 audit on #388): a100's session dir is PRE-SEEDED
# with its own real keypair below, exactly like a genuine `up`/`attach`
# session already would have. Without this, rp_init's own
# `[ -f "$RP_SSH_KEY" ] || ssh-keygen ...` would REUSE an already-missing-
# turned-present key path only on the SECOND run — on an EMPTY a100 dir (the
# original version of this fixture) `ssh-keygen` would run whether or not
# the bug reproduced, so "a new file appeared" was not actually pinned to
# the bug. Pre-seeding removes that false-negative risk: any key material
# written into a100's dir by THIS run can only be an overwrite.
#
# The direct, positive signal for "RP_WORK resolved to a temp dir" comes
# from TMPDIR: runpod_lib.sh's own `mktemp -d "${TMPDIR:-/tmp}/jammi-gpu-dev.XXXXXX"`
# (see its own comment) builds its throwaway path from $TMPDIR explicitly —
# portable across GNU and BSD `mktemp`, unlike a bare `mktemp -d` — so
# overriding TMPDIR to an otherwise-empty watch directory for this one
# invocation makes a NEWLY generated keypair appearing under it direct,
# unambiguous proof that `rp_init` wrote into a FRESH temp directory, never
# a100's own (which already has its own pre-seeded key and would need no
# `ssh-keygen` call at all).
#
# The deploy itself is mocked as a SUCCESS (`podFindAndDeployOnDemand`
# returns a real id, same shape as finding 1's own fixture above) rather
# than a no-capacity refusal, so this exercises the actual pod-rental path,
# not merely the pre-flight one — and since no fixture in this suite mocks
# a reachable SSH server (see this file's own header doc), the invocation is
# SIGTERM'd once the deploy call lands, exactly like finding 1's fixture.
# ═════════════════════════════════════════════════════════════════════════
SESSION_LIVE_A100="a100"
write_meta "$SESSION_LIVE_A100" "pod-live-a100" "72"
ssh-keygen -t ed25519 -N '' -f "$RP_SESSION_ROOT/$SESSION_LIVE_A100/id_ed25519" -q
before_listing="$(ls -A "$RP_SESSION_ROOT/$SESSION_LIVE_A100")"
before_meta_sum="$(cat "$RP_SESSION_ROOT/$SESSION_LIVE_A100/meta")"
before_key_sum="$(cat "$RP_SESSION_ROOT/$SESSION_LIVE_A100/id_ed25519")"

TMPWATCH="$SANDBOX/tmpwatch-shell-isolation"
mkdir -p "$TMPWATCH"

echo '{"data":{"podFindAndDeployOnDemand":{"id":"pod-shell-isolation"}}}' > "$SANDBOX/deploy-shell-isolation.json"
export MOCK_DEPLOY_RESPONSE="$SANDBOX/deploy-shell-isolation.json"
reset_log
TMPDIR="$TMPWATCH" RP_SESSION="$SESSION_LIVE_A100" bash "$DIR/gpu-dev.sh" shell a100 --ref abcdef1234567890 \
  >"$SANDBOX/out-shell-isolation.log" 2>&1 &
ISO_PID=$!
deadline=$((SECONDS + 20))
while ! log_has "podFindAndDeployOnDemand" && [ "$SECONDS" -lt "$deadline" ]; do sleep 0.1; done

# rp_init (which resolves RP_WORK and generates its keypair if one is not
# already there) always runs BEFORE rp_deploy_arch is ever called, so by the
# time the deploy call has landed in the log, any keypair rp_init generated
# already exists on disk. Checked HERE, before the SIGTERM below, and NOT
# after the process has exited: rp_cleanup's own EXIT trap `rm -rf`s a
# throwaway RP_WORK on the way out (correct behaviour for a real throwaway
# pod), so inspecting the watch dir post-exit would always find it already
# deleted regardless of whether the fix actually took effect — the earlier
# version of this assertion did exactly that and failed even against the
# FIXED code.
if find "$TMPWATCH" -name 'id_ed25519' 2>/dev/null | grep -q .; then
  ok "finding(shell-session-isolation): RP_WORK resolved to a fresh \$TMPDIR temp dir (a new keypair was generated there)"
else
  bad "finding(shell-session-isolation): expected a NEW keypair under \$TMPDIR (RP_WORK never resolved to a throwaway temp dir); watch dir contents: $(ls -A "$TMPWATCH" 2>/dev/null)"
fi

sleep 0.5
kill -TERM "$ISO_PID" 2>/dev/null
wait_deadline=$((SECONDS + 15))
while kill -0 "$ISO_PID" 2>/dev/null && [ "$SECONDS" -lt "$wait_deadline" ]; do sleep 0.2; done
kill -0 "$ISO_PID" 2>/dev/null && kill -KILL "$ISO_PID" 2>/dev/null
wait "$ISO_PID" 2>/dev/null

after_listing="$(ls -A "$RP_SESSION_ROOT/$SESSION_LIVE_A100")"
after_meta_sum="$(cat "$RP_SESSION_ROOT/$SESSION_LIVE_A100/meta")"
after_key_sum="$(cat "$RP_SESSION_ROOT/$SESSION_LIVE_A100/id_ed25519")"

if log_has "podTerminate.*pod-shell-isolation"; then
  ok "finding(shell-session-isolation): the freshly rented throwaway pod was terminated on SIGTERM"
else
  bad "finding(shell-session-isolation): expected a podTerminate for pod-shell-isolation after SIGTERM; output: $(cat "$SANDBOX/out-shell-isolation.log")"
fi
if [ "$before_listing" = "$after_listing" ]; then
  ok "finding(shell-session-isolation): the live a100 session dir's listing is unchanged (pre-seeded keypair, no new/overwritten file)"
else
  bad "finding(shell-session-isolation): the live a100 session dir's listing changed (before=[$before_listing] after=[$after_listing]) — an inherited exported RP_SESSION leaked into shell's throwaway RP_WORK"
fi
if [ "$before_meta_sum" = "$after_meta_sum" ]; then
  ok "finding(shell-session-isolation): the live a100 session's meta content is byte-identical after shell exits"
else
  bad "finding(shell-session-isolation): the live a100 session's meta content changed (regression!)"
fi
if [ "$before_key_sum" = "$after_key_sum" ]; then
  ok "finding(shell-session-isolation): the live a100 session's pre-seeded keypair is byte-identical after shell exits"
else
  bad "finding(shell-session-isolation): the live a100 session's pre-seeded keypair changed (regression!)"
fi
unset MOCK_DEPLOY_RESPONSE

# ═════════════════════════════════════════════════════════════════════════
# Group 1d — CLI-level: a failed `mktemp -d` (round-3 audit advisory) must
# abort the invocation outright, not fall through with RP_WORK unset/empty
# — every later use (the ssh key path, the meta path, the pod deploy
# itself) would then operate on a garbage or empty path instead of failing
# loudly at the one place the real cause is known. TMPDIR pointed at a
# nonexistent directory reproduces a real `mktemp` failure with zero mocked
# network state required: `shell` must exit 2 before `rp_init` even reaches
# ssh-keygen, so no deploy call is ever issued.
# ═════════════════════════════════════════════════════════════════════════
reset_log
TMPDIR="$SANDBOX/no-such-tmpdir-$$" bash "$DIR/gpu-dev.sh" shell a100 --ref abcdef1234567890 \
  >"$SANDBOX/out-mktemp-fail.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding(mktemp-failure): an unwritable/nonexistent \$TMPDIR exits 2" \
  || bad "finding(mktemp-failure): expected exit 2 for an unwritable \$TMPDIR (got $rc); $(cat "$SANDBOX/out-mktemp-fail.log")"
grep -q "could not create a temp work dir" "$SANDBOX/out-mktemp-fail.log" \
  && ok "finding(mktemp-failure): the refusal names the real cause (could not create a temp work dir)" \
  || bad "finding(mktemp-failure): expected a 'could not create a temp work dir' message ($(cat "$SANDBOX/out-mktemp-fail.log"))"
if log_has "podFindAndDeployOnDemand"; then
  bad "finding(mktemp-failure): a failed mktemp must issue NO deploy call (regression!)"
else
  ok "finding(mktemp-failure): a failed mktemp issues no deploy call"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 2 — rp_pod_verify semantics (function-level): id + this tooling's
# own "<prefix>-ttl<digits>" NAME SHAPE, never an exact TTL number.
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  rp_gql() { cat "$MOCK_RESPONSE_FILE"; }

  echo '{"data":{"myself":{"pods":[{"id":"pod-x","name":"jammi-gpu-ttl72"}]}}}' > "$SANDBOX/acct1.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct1.json"
  out="$(rp_pod_verify "pod-x" 2>/dev/null)"; rc=$?
  [ "$rc" -eq 0 ] && [ "$out" = "jammi-gpu-ttl72" ] \
    && record PASS "group2-verify-ttl-shaped-name-matches" || record FAIL "group2-verify-ttl-shaped-name-matches (rc=$rc out=$out)"

  # The id is real, but its account name is NOT this tooling's shape at all
  # (as opposed to merely a different NUMBER, which is fine — see the
  # ttl-differs CLI test below) -- must refuse.
  echo '{"data":{"myself":{"pods":[{"id":"pod-x","name":"some-unrelated-name"}]}}}' > "$SANDBOX/acct2.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct2.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 1 ] && record PASS "group2-verify-non-shaped-name-refused" || record FAIL "group2-verify-non-shaped-name-refused"

  # An ABSENT id is its own return code (3), never folded into the
  # present-but-mismatched code (1): `down` treats them completely
  # differently — absent is normal cleanup (round-4 audit advisory), not a
  # refusal — and needs rp_pod_verify to tell them apart.
  echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct3.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct3.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 3 ] && record PASS "group2-verify-absent-id-distinct-code" || record FAIL "group2-verify-absent-id-distinct-code"

  echo 'not json' > "$SANDBOX/acct4.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct4.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 2 ] && record PASS "group2-verify-query-failure-refused" || record FAIL "group2-verify-query-failure-refused"

  # THE pipeline-status fix's own reproduction: rp_gql (curl) itself exits 3
  # while STILL printing a COMPLETE body that lists the pod (a network
  # hiccup after the response was already fully received, or any transport
  # that reports a non-zero exit alongside a good body). This suite runs
  # under `set -uo pipefail` (inherited from the top of this file), so
  # BEFORE the fix — `rp_gql ... | python3 -c ...` piped directly — the
  # pipeline's own exit status would be curl's 3 regardless of what python
  # found, aliasing with rp_pod_verify's OWN semantic code 3 ("id ABSENT,
  # down forgets the record") even though the body actually shows the pod
  # PRESENT and correctly shaped. Must return 2 ("could not query"), never
  # 3 — the fixed function captures the body with its own `||` gate before
  # ever handing it to the parser, so a curl failure can never reach python
  # at all, regardless of what curl printed alongside it.
  rp_gql() { cat "$MOCK_RESPONSE_FILE"; return 3; }
  echo '{"data":{"myself":{"pods":[{"id":"pod-x","name":"jammi-gpu-ttl72"}]}}}' > "$SANDBOX/acct5.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct5.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 2 ] && record PASS "group2-verify-curl-exit3-with-complete-body-returns-2-never-3" \
    || record FAIL "group2-verify-curl-exit3-with-complete-body-returns-2-never-3"
)

# ═════════════════════════════════════════════════════════════════════════
# Group 2c — rp_pod_gone semantics (function-level; finding 6's actual
# confirmation primitive).
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  rp_gql() { cat "$MOCK_RESPONSE_FILE"; }

  echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/gone1.json"
  MOCK_RESPONSE_FILE="$SANDBOX/gone1.json"
  rp_pod_gone "pod-y" >/dev/null 2>&1
  [ $? -eq 0 ] && record PASS "group2c-gone-absent-confirmed" || record FAIL "group2c-gone-absent-confirmed"

  echo '{"data":{"myself":{"pods":[{"id":"pod-y","name":"jammi-gpu-ttl8"}]}}}' > "$SANDBOX/gone2.json"
  MOCK_RESPONSE_FILE="$SANDBOX/gone2.json"
  rp_pod_gone "pod-y" >/dev/null 2>&1
  [ $? -eq 1 ] && record PASS "group2c-gone-still-present-not-confirmed" || record FAIL "group2c-gone-still-present-not-confirmed"

  echo 'not json' > "$SANDBOX/gone3.json"
  MOCK_RESPONSE_FILE="$SANDBOX/gone3.json"
  rp_pod_gone "pod-y" >/dev/null 2>&1
  [ $? -eq 1 ] && record PASS "group2c-gone-query-failure-not-confirmed" || record FAIL "group2c-gone-query-failure-not-confirmed"
)

# ═════════════════════════════════════════════════════════════════════════
# Group 2b — CLI-level: an invalid RP_DEV_TTL_HOURS must be reported under
# its OWN name, not misattributed to RP_TTL_HOURS. Covers BOTH the
# digit-shape check (round-2 audit finding 4) and the >0 check (round-3
# audit finding 2: "0" is all-digit and slips past a digit-only pattern).
# ═════════════════════════════════════════════════════════════════════════
reset_log
RP_DEV_TTL_HOURS=not-a-number bash "$DIR/gpu-dev.sh" up a100 >"$SANDBOX/out-bad-dev-ttl.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding-4: an invalid (non-numeric) RP_DEV_TTL_HOURS exits 2" \
  || bad "finding-4: expected exit 2 for a non-numeric RP_DEV_TTL_HOURS (got $rc)"
grep -q "RP_DEV_TTL_HOURS must be a positive integer" "$SANDBOX/out-bad-dev-ttl.log" \
  && ok "finding-4: the non-numeric validation error names RP_DEV_TTL_HOURS, not RP_TTL_HOURS" \
  || bad "finding-4: expected the error to name RP_DEV_TTL_HOURS ($(cat "$SANDBOX/out-bad-dev-ttl.log"))"
if grep -q "RP_TTL_HOURS must be" "$SANDBOX/out-bad-dev-ttl.log"; then
  bad "finding-4: the non-numeric error must not misattribute to RP_TTL_HOURS (regression!)"
else
  ok "finding-4: the non-numeric error does not misattribute to RP_TTL_HOURS"
fi

reset_log
RP_DEV_TTL_HOURS=0 bash "$DIR/gpu-dev.sh" up a100 >"$SANDBOX/out-zero-dev-ttl.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding-2(round-3): RP_DEV_TTL_HOURS=0 exits 2" \
  || bad "finding-2(round-3): expected exit 2 for RP_DEV_TTL_HOURS=0 (got $rc)"
grep -q "RP_DEV_TTL_HOURS must be > 0" "$SANDBOX/out-zero-dev-ttl.log" \
  && ok "finding-2(round-3): the >0 validation error names RP_DEV_TTL_HOURS, not RP_TTL_HOURS" \
  || bad "finding-2(round-3): expected 'RP_DEV_TTL_HOURS must be > 0' ($(cat "$SANDBOX/out-zero-dev-ttl.log"))"
if grep -q "RP_TTL_HOURS must be" "$SANDBOX/out-zero-dev-ttl.log"; then
  bad "finding-2(round-3): the >0 error must not misattribute to RP_TTL_HOURS (regression!)"
else
  ok "finding-2(round-3): the >0 error does not misattribute to RP_TTL_HOURS"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 3 — CLI-level: `up` refuses a session alias that already has a
# recorded pod unless --replace, and --replace must not inherit the dead
# session's own RP_TTL_HOURS/RP_IMAGE (finding 3, driven through gpu-dev.sh).
# ═════════════════════════════════════════════════════════════════════════
SESSION_UP="up-stale"
write_meta "$SESSION_UP" "pod-stale-1" "8"
reset_log
RP_SESSION="$SESSION_UP" bash "$DIR/gpu-dev.sh" up a100 >"$SANDBOX/out-up-refuse.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding-3(up): refuses (exit 2) an alias with a recorded-but-unreachable pod" \
  || bad "finding-3(up): expected exit 2 refusing an existing alias (got $rc)"
# Pin the MESSAGE, not just the exit code — exit 2 is also gpu-dev.sh's
# generic bad-usage code, so a passing assertion on rc alone could mask an
# argument-parsing mistake in this very test producing the "right" exit code
# for the wrong reason.
grep -q "already has a recorded pod" "$SANDBOX/out-up-refuse.log" \
  && ok "finding-3(up): refusal names the recorded pod (not a usage error)" \
  || bad "finding-3(up): expected the 'already has a recorded pod' refusal text ($(cat "$SANDBOX/out-up-refuse.log"))"
if log_has "podFindAndDeployOnDemand"; then
  bad "finding-3(up): refusal path must issue no deploy call at all (regression)"
else
  ok "finding-3(up): refusal path issued no network call"
fi
# The refusal message must tell the operator to replay with the ARCH `up`
# actually takes positionally, not the session name — `up <session-name>`
# parses the session name AS an arch and fails differently.
grep -q -- "up a100 --replace" "$SANDBOX/out-up-refuse.log" \
  && ok "finding-3(up): refusal names the arch ('up a100 --replace'), not the session, in its --replace suggestion" \
  || bad "finding-3(up): expected 'up a100 --replace' (arch, not session) in the refusal text ($(cat "$SANDBOX/out-up-refuse.log"))"

# `--replace` bypasses the refusal and reaches the real deploy path. Every
# candidate is mocked as SUPPLY_CONSTRAINT so rp_deploy_live's SSH-readiness
# poll is never entered (no real wait, no real network) — this proves
# --replace REACHED deploy, not that a pod was rented. The mocked RESPONSE
# says "no capacity", but the OUTGOING request (what this stub logs) still
# carries the real payload gpu-dev.sh actually built, which is what the
# assertions below inspect.
echo '{"errors":[{"extensions":{"code":"SUPPLY_CONSTRAINT"},"message":"no capacity"}]}' > "$SANDBOX/deploy-no-capacity.json"
export MOCK_DEPLOY_RESPONSE="$SANDBOX/deploy-no-capacity.json"
reset_log
RP_SESSION="$SESSION_UP" bash "$DIR/gpu-dev.sh" up a100 --replace --ref abcdef1234567890 \
  >"$SANDBOX/out-up-replace.log" 2>&1
rc=$?
[ "$rc" -eq 75 ] && ok "finding-3(up --replace): bypasses the refusal and reaches deploy (rc=75, no capacity)" \
  || bad "finding-3(up --replace): expected rc=75 once it reaches the (mocked, no-capacity) deploy path (got $rc)"
log_has "podFindAndDeployOnDemand" \
  && ok "finding-3(up --replace): a deploy call was actually made" \
  || bad "finding-3(up --replace): expected a deploy call after --replace"
grep -q -- "--replace: overwriting session" "$SANDBOX/out-up-replace.log" \
  && ok "finding-3(up --replace): prints the overwrite warning" \
  || bad "finding-3(up --replace): expected the overwrite warning in output"
# The dead session's own meta recorded RP_TTL_HOURS=8. `up` (with no
# RP_TTL_HOURS set by this test's own environment) computes the 72h dev
# default BEFORE rp_session_load can dot-source the dead meta and clobber it
# — this is the exact "jammi-gpu-ttl8 / sleep 28800 on --replace over an 8h
# meta" defect the audit reproduced. Both the pod's NAME and its in-pod
# watchdog's sleep duration must reflect 72h, never the dead session's 8h.
if grep -q '"name": "jammi-gpu-ttl72"' "$CALL_LOG"; then
  ok "finding-3(up --replace): deploy payload name is jammi-gpu-ttl72 (the dev default), not the dead session's jammi-gpu-ttl8"
else
  bad "finding-3(up --replace): expected the deploy payload name to be jammi-gpu-ttl72; log: $(cat "$CALL_LOG")"
fi
if grep -q "sleep 259200" "$CALL_LOG"; then
  ok "finding-3(up --replace): deploy payload watchdog uses sleep 259200 (72h), not the dead session's 8h"
else
  bad "finding-3(up --replace): expected 'sleep 259200' (72h) in the deploy payload; log: $(cat "$CALL_LOG")"
fi
unset MOCK_DEPLOY_RESPONSE

# ═════════════════════════════════════════════════════════════════════════
# Group 3b — CLI-level: `down` verifies by id + name SHAPE only (finding 5)
# and confirms a terminate actually took before forgetting the local record
# (finding 6), driven through the real gpu-dev.sh dispatch.
# ═════════════════════════════════════════════════════════════════════════

# A legacy (no-TTL) meta for a genuinely-owned pod: verifies, terminates,
# and the SECOND account query (post-terminate) shows the id gone —
# confirmed, so the local record is forgotten. Proves TTL-in-meta is now
# irrelevant end to end, not just at the rp_pod_verify layer.
SESSION_DOWN_OK="down-ok"
write_meta_legacy "$SESSION_DOWN_OK" "pod-ok"
echo '{"data":{"myself":{"pods":[{"id":"pod-ok","name":"jammi-gpu-ttl8"}]}}}' > "$SANDBOX/acct-down-ok-1.json"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-down-ok-2.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-down-ok-1.json"
export MOCK_ACCOUNT_RESPONSE_2="$SANDBOX/acct-down-ok-2.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_OK" >"$SANDBOX/out-down-ok.log" 2>&1
rc=$?
term_count_ok="$(grep -c "podTerminate.*pod-ok" "$CALL_LOG")"
[ "$term_count_ok" = "1" ] && [ "$rc" -eq 0 ] \
  && ok "finding-5/6(down): a verified, confirmed terminate succeeds exactly once (legacy meta, no TTL line)" \
  || bad "finding-5/6(down): expected exactly one confirmed podTerminate for pod-ok (got count=${term_count_ok} rc=${rc}); $(cat "$SANDBOX/out-down-ok.log")"
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_OK/meta" ] \
  && bad "finding-6(down): the local record should be forgotten after a CONFIRMED terminate" \
  || ok "finding-6(down): the local record is forgotten after a confirmed terminate"

# The recorded id is real and the account name IS this tooling's shape, but
# the NUMBER differs from what the session's own meta recorded (8 vs 72) —
# the TTL never gated release even before this round; this pins that the id
# alone (once name-shaped) is enough, regardless of the specific number.
SESSION_DOWN_TTL_DIFFERS="down-ttl-differs"
write_meta "$SESSION_DOWN_TTL_DIFFERS" "pod-ttl-differs" "8"
echo '{"data":{"myself":{"pods":[{"id":"pod-ttl-differs","name":"jammi-gpu-ttl72"}]}}}' > "$SANDBOX/acct-ttl-differs-1.json"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-ttl-differs-2.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-ttl-differs-1.json"
export MOCK_ACCOUNT_RESPONSE_2="$SANDBOX/acct-ttl-differs-2.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_TTL_DIFFERS" >"$SANDBOX/out-down-ttl-differs.log" 2>&1
term_count_differs="$(grep -c "podTerminate.*pod-ttl-differs" "$CALL_LOG")"
[ "$term_count_differs" = "1" ] \
  && ok "finding-5(down): a differing TTL NUMBER (meta ttl8, account ttl72, same id) still terminates — the id is authoritative" \
  || bad "finding-5(down): expected exactly one podTerminate for pod-ttl-differs despite the differing TTL number (got ${term_count_differs}); $(cat "$SANDBOX/out-down-ttl-differs.log")"

# A recorded id absent from the account entirely is NOT a refusal — it is
# the ordinary shape of "this pod already ended on its own" (its own
# deadline, or the sweep), the single most common way a session's pod goes
# away. `down` must forget the record and exit 0, not get stuck refusing
# forever until someone remembers `up --replace` (round-4 audit advisory).
SESSION_DOWN_GONE="down-gone-already"
write_meta "$SESSION_DOWN_GONE" "pod-gone-already" "8"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-down-gone-1.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-down-gone-1.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_GONE" >"$SANDBOX/out-down-gone.log" 2>&1
rc=$?
if log_has "podTerminate"; then
  bad "finding(round-4, down): an already-absent id must never attempt a terminate (nothing to release; regression!)"
else
  ok "finding(round-4, down): an already-absent id issues no terminate call (nothing to release)"
fi
[ "$rc" -eq 0 ] && ok "finding(round-4, down): an already-absent id exits 0 (not a refusal)" \
  || bad "finding(round-4, down): expected exit 0 for an already-absent id (got $rc)"
grep -q "is gone from the account" "$SANDBOX/out-down-gone.log" \
  && ok "finding(round-4, down): an already-absent id prints the gone-from-account message" \
  || bad "finding(round-4, down): expected the 'is gone from the account' message ($(cat "$SANDBOX/out-down-gone.log"))"
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_GONE/meta" ] \
  && bad "finding(round-4, down): the local record should be forgotten once the id is confirmed absent (regression!)" \
  || ok "finding(round-4, down): the local record is forgotten once the id is confirmed absent"

# The recorded id IS present, but its account name is not this tooling's
# shape at all (some entirely unrelated pod happens to share the id space —
# never realistic for RunPod's globally-unique ids, but the check must still
# hold the line): refuses, keeps the record.
SESSION_DOWN_UNSHAPED="down-unshaped-name"
write_meta "$SESSION_DOWN_UNSHAPED" "pod-unshaped" "8"
echo '{"data":{"myself":{"pods":[{"id":"pod-unshaped","name":"some-unrelated-name"}]}}}' > "$SANDBOX/acct-down-unshaped-1.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-down-unshaped-1.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_UNSHAPED" >"$SANDBOX/out-down-unshaped.log" 2>&1
if log_has "podTerminate"; then
  bad "finding-5(down): a non-tooling-shaped name must refuse to terminate (regression!)"
else
  ok "finding-5(down): a non-tooling-shaped name refused to terminate"
fi
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_UNSHAPED/meta" ] \
  && ok "finding-5(down): the local record is KEPT after a non-shaped-name refusal" \
  || bad "finding-5(down): the local record must survive a non-shaped-name refusal (regression!)"

# The account query ITSELF fails (a GraphQL `errors` body — e.g. rate
# limiting), never reaching a pod list at all: rp_pod_verify's own code 2
# ("could not query the account"). This is the follow-up fixture for the
# curl/python pipeline-status split above rp_pod_verify's own definition in
# runpod_lib.sh — a query failure must refuse (never terminate) and keep the
# local record, exactly like the non-shaped-name refusal, never aliasing with
# the ABSENT-id forget arm (verify_rc 3).
SESSION_DOWN_QUERY_FAIL="down-query-fail"
write_meta "$SESSION_DOWN_QUERY_FAIL" "pod-query-fail" "8"
echo '{"errors":[{"message":"rate limited"}]}' > "$SANDBOX/acct-down-query-fail-1.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-down-query-fail-1.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_QUERY_FAIL" >"$SANDBOX/out-down-query-fail.log" 2>&1
rc=$?
[ "$rc" -eq 1 ] && ok "finding(query-failure, down): a failed account query exits 1 (refuses)" \
  || bad "finding(query-failure, down): expected exit 1 for a failed account query (got $rc)"
if log_has "podTerminate"; then
  bad "finding(query-failure, down): a failed account query must never issue podTerminate (regression!)"
else
  ok "finding(query-failure, down): a failed account query issues no podTerminate"
fi
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_QUERY_FAIL/meta" ] \
  && ok "finding(query-failure, down): the local record is KEPT after a query-failure refusal" \
  || bad "finding(query-failure, down): the local record must survive a query-failure refusal (regression!)"

# Verified (id + shape match), but the SECOND account query — the
# post-terminate confirmation — still shows the pod present: the terminate
# was attempted but is NOT confirmed (a rejected mutation, most likely).
# Must keep the local record and exit 1 asking for a retry, never both leak
# the pod AND destroy the record (finding 6).
SESSION_DOWN_REJECTED="down-terminate-rejected"
write_meta "$SESSION_DOWN_REJECTED" "pod-reject" "8"
echo '{"data":{"myself":{"pods":[{"id":"pod-reject","name":"jammi-gpu-ttl8"}]}}}' > "$SANDBOX/acct-reject-1.json"
echo '{"data":{"myself":{"pods":[{"id":"pod-reject","name":"jammi-gpu-ttl8"}]}}}' > "$SANDBOX/acct-reject-2.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-reject-1.json"
export MOCK_ACCOUNT_RESPONSE_2="$SANDBOX/acct-reject-2.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_REJECTED" >"$SANDBOX/out-down-rejected.log" 2>&1
rc=$?
term_count_rejected="$(grep -c "podTerminate.*pod-reject" "$CALL_LOG")"
[ "$term_count_rejected" = "1" ] \
  && ok "finding-6(down): a terminate IS attempted even when not later confirmed" \
  || bad "finding-6(down): expected the terminate to still be attempted once (got ${term_count_rejected})"
[ "$rc" -eq 1 ] && ok "finding-6(down): an unconfirmed terminate exits 1" \
  || bad "finding-6(down): expected exit 1 for an unconfirmed terminate (got $rc)"
grep -q "terminate not confirmed" "$SANDBOX/out-down-rejected.log" \
  && ok "finding-6(down): an unconfirmed terminate names itself, not a generic verification refusal" \
  || bad "finding-6(down): expected a 'terminate not confirmed' message ($(cat "$SANDBOX/out-down-rejected.log"))"
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_REJECTED/meta" ] \
  && ok "finding-6(down): the local record is KEPT when the terminate is not confirmed (no leak-and-destroy)" \
  || bad "finding-6(down): the local record must survive an unconfirmed terminate (regression!)"

# Neither real refusal path (non-shaped name, unconfirmed terminate)
# suggests `reap <hours>` (account-wide, never a per-pod remedy) or the
# removed, INERT RP_TTL_HOURS=<H> override. The already-gone case above is
# deliberately NOT checked here — it is not a refusal, so it never had
# either suggestion to begin with.
if grep -q -- "reap <hours>" "$SANDBOX/out-down-unshaped.log" "$SANDBOX/out-down-rejected.log"; then
  bad "finding: down's refusal must not suggest 'reap <hours>' as a per-pod remedy (it is account-wide; regression!)"
else
  ok "finding: down's refusal does not suggest reap as a per-pod remedy"
fi
if grep -q -- "RP_TTL_HOURS=<H>" "$SANDBOX/out-down-unshaped.log" "$SANDBOX/out-down-rejected.log"; then
  bad "finding: down's refusal must not promise the removed, inert RP_TTL_HOURS=<H> override (regression!)"
else
  ok "finding: down's refusal does not promise the removed RP_TTL_HOURS=<H> override"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 3c — CLI-level: an exported RP_SESSION must never silently OUTRANK
# an explicit positional session argument for down/attach/run/logs/push/pull
# (round-2 audit on #388, reproduced: `RP_SESSION=a100 gpu-dev.sh down l40s`
# terminated pod-a100 and forgot ITS record, never touching l40s — the
# positional the caller actually typed was discarded outright). Two live
# sessions are recorded so a wrong pick is directly observable (the WRONG
# one's pod would be terminated, not merely "a" pod).
# ═════════════════════════════════════════════════════════════════════════
SESSION_CONFLICT_A="a100"
SESSION_CONFLICT_L="l40s"
write_meta "$SESSION_CONFLICT_A" "pod-conflict-a100" "8"
write_meta "$SESSION_CONFLICT_L" "pod-conflict-l40s" "8"

# A differing exported RP_SESSION and positional: refuse (exit 2), issue NO
# account query at all (the ambiguity is resolved BEFORE anything is rented
# or acted on), terminate nothing, and keep BOTH local records intact.
reset_log; reset_account_seq
RP_SESSION="$SESSION_CONFLICT_A" bash "$DIR/gpu-dev.sh" down "$SESSION_CONFLICT_L" \
  >"$SANDBOX/out-conflict-refuse.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding(session-conflict): a differing positional vs exported RP_SESSION exits 2" \
  || bad "finding(session-conflict): expected exit 2 for a differing positional vs RP_SESSION (got $rc); $(cat "$SANDBOX/out-conflict-refuse.log")"
grep -q "conflicting session" "$SANDBOX/out-conflict-refuse.log" \
  && ok "finding(session-conflict): the refusal names itself, not a generic usage error" \
  || bad "finding(session-conflict): expected a 'conflicting session' message ($(cat "$SANDBOX/out-conflict-refuse.log"))"
grep -q -- "'${SESSION_CONFLICT_L}'" "$SANDBOX/out-conflict-refuse.log" \
  && grep -q -- "RP_SESSION='${SESSION_CONFLICT_A}'" "$SANDBOX/out-conflict-refuse.log" \
  && ok "finding(session-conflict): the refusal names BOTH the positional and the exported RP_SESSION" \
  || bad "finding(session-conflict): expected both '${SESSION_CONFLICT_L}' and RP_SESSION='${SESSION_CONFLICT_A}' named ($(cat "$SANDBOX/out-conflict-refuse.log"))"
if log_has "podTerminate"; then
  bad "finding(session-conflict): a conflicting session must issue NO podTerminate (regression!)"
else
  ok "finding(session-conflict): a conflicting session issues no podTerminate"
fi
if log_has "myself{"; then
  bad "finding(session-conflict): a conflicting session must issue NO account query at all — resolved before anything is rented (regression!)"
else
  ok "finding(session-conflict): a conflicting session issues no account query"
fi
[ -f "$RP_SESSION_ROOT/$SESSION_CONFLICT_A/meta" ] && [ -f "$RP_SESSION_ROOT/$SESSION_CONFLICT_L/meta" ] \
  && ok "finding(session-conflict): BOTH local records survive the refusal" \
  || bad "finding(session-conflict): both records must survive the refusal (regression!)"

# The SAME positional and exported RP_SESSION (no conflict): acts on it
# normally, exactly like the pre-existing down tests above.
echo '{"data":{"myself":{"pods":[{"id":"pod-conflict-a100","name":"jammi-gpu-ttl8"}]}}}' \
  > "$SANDBOX/acct-conflict-match-1.json"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-conflict-match-2.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-conflict-match-1.json"
export MOCK_ACCOUNT_RESPONSE_2="$SANDBOX/acct-conflict-match-2.json"
RP_SESSION="$SESSION_CONFLICT_A" bash "$DIR/gpu-dev.sh" down "$SESSION_CONFLICT_A" \
  >"$SANDBOX/out-conflict-match.log" 2>&1
rc=$?
term_count_match="$(grep -c "podTerminate.*pod-conflict-a100" "$CALL_LOG")"
[ "$term_count_match" = "1" ] && [ "$rc" -eq 0 ] \
  && ok "finding(session-conflict): a MATCHING positional and RP_SESSION acts normally (terminates pod-conflict-a100)" \
  || bad "finding(session-conflict): expected exactly one confirmed terminate for a matching positional/RP_SESSION (got count=${term_count_match} rc=${rc}); $(cat "$SANDBOX/out-conflict-match.log")"
if log_has "podTerminate.*pod-conflict-l40s"; then
  bad "finding(session-conflict): a matching down on a100 must never touch l40s's pod (regression!)"
else
  ok "finding(session-conflict): a matching down on a100 never touches l40s's pod"
fi
[ -f "$RP_SESSION_ROOT/$SESSION_CONFLICT_L/meta" ] \
  && ok "finding(session-conflict): l40s's record is untouched by a matching down on a100" \
  || bad "finding(session-conflict): l40s's record must survive a matching down on a100 (regression!)"

# ═════════════════════════════════════════════════════════════════════════
# Group 3d — CLI-level: RP_SESSION is validated as a CONTAINMENT blacklist
# (empty/'.'/'..' /a '/' anywhere / a leading '-'), not a character
# whitelist, at the point RP_WORK is derived from it — a traversing value
# must be refused BEFORE rp_cleanup's or rp_session_forget's unconditional
# `rm -rf "$RP_WORK"` could ever act on it (round-2 audit on #388). `down`
# (with NO positional, so RP_SESSION alone resolves SESSION) is the driving
# subcommand here — a NAMED-session verb, unlike `ls`/`reap`, which never
# apply this check at all (see Group 3e below).
#
# Round-3 audit on #388: the ORIGINAL fix used a `[A-Za-z0-9_-]+` character
# WHITELIST, which rejected every session name containing a `.` —
# including `bench.1`, a shape gpu-dev.sh's own dispatch is happy to
# create (`RP_SESSION=bench.1 gpu-dev.sh up`) — so a live pod under a
# dotted session name became unreachable by EVERY verb, stranded for its
# full deadline. The fix below is a blacklist of the shapes that are
# actually dangerous; a dot anywhere but as the whole name is accepted.
# ═════════════════════════════════════════════════════════════════════════
TRAVERSAL_TARGET="$SANDBOX/must-not-be-touched"
mkdir -p "$TRAVERSAL_TARGET"
echo "sentinel" > "$TRAVERSAL_TARGET/sentinel.txt"
RP_SESSION="../$(basename "$TRAVERSAL_TARGET")" bash "$DIR/gpu-dev.sh" down \
  >"$SANDBOX/out-traversal.log" 2>&1
rc=$?
[ "$rc" -eq 2 ] && ok "finding(RP_SESSION-blacklist): a '..'-containing RP_SESSION exits 2" \
  || bad "finding(RP_SESSION-blacklist): expected exit 2 for a traversing RP_SESSION (got $rc); $(cat "$SANDBOX/out-traversal.log")"
grep -q "RP_SESSION may" "$SANDBOX/out-traversal.log" \
  && ok "finding(RP_SESSION-blacklist): the refusal names the session-name rule" \
  || bad "finding(RP_SESSION-blacklist): expected an 'RP_SESSION may ...' refusal text ($(cat "$SANDBOX/out-traversal.log"))"
[ -f "$TRAVERSAL_TARGET/sentinel.txt" ] \
  && ok "finding(RP_SESSION-blacklist): the out-of-root target directory is untouched" \
  || bad "finding(RP_SESSION-blacklist): the out-of-root target must survive (regression — rm -rf escaped RP_SESSION_ROOT!)"
# A bare '..' (no leading slash) must refuse the same way.
RP_SESSION=".." bash "$DIR/gpu-dev.sh" down >"$SANDBOX/out-dotdot.log" 2>&1
[ $? -eq 2 ] && ok "finding(RP_SESSION-blacklist): a bare '..' exits 2" \
  || bad "finding(RP_SESSION-blacklist): expected exit 2 for a bare '..' ($(cat "$SANDBOX/out-dotdot.log"))"
# A slash anywhere (no traversal, but still a multi-segment path) refuses.
RP_SESSION="a100/evil" bash "$DIR/gpu-dev.sh" down >"$SANDBOX/out-slash.log" 2>&1
[ $? -eq 2 ] && ok "finding(RP_SESSION-blacklist): an embedded '/' (no '..') also exits 2" \
  || bad "finding(RP_SESSION-blacklist): expected exit 2 for an embedded '/' ($(cat "$SANDBOX/out-slash.log"))"
# A leading '-' refuses (reads as an option to any tool it is later passed
# to positionally).
RP_SESSION="-x" bash "$DIR/gpu-dev.sh" down >"$SANDBOX/out-dash.log" 2>&1
[ $? -eq 2 ] && ok "finding(RP_SESSION-blacklist): a leading '-' exits 2" \
  || bad "finding(RP_SESSION-blacklist): expected exit 2 for a leading '-' ($(cat "$SANDBOX/out-dash.log"))"
# THE round-3 regression itself: a DOTTED session name (not the whole name
# being '.'/'..') is accepted, not refused. No recorded pod for it, so
# `down` simply reports that and exits 0 -- the point is that it reaches
# THAT far at all, rather than refusing at the RP_SESSION gate.
RP_SESSION="a100.2" bash "$DIR/gpu-dev.sh" down >"$SANDBOX/out-dotted.log" 2>&1
rc=$?
[ "$rc" -eq 0 ] && ok "finding(RP_SESSION-blacklist): a dotted session name ('a100.2') is accepted, not refused" \
  || bad "finding(RP_SESSION-blacklist): a dotted session name must be accepted (got rc=$rc); $(cat "$SANDBOX/out-dotted.log")"
grep -q "RP_SESSION may" "$SANDBOX/out-dotted.log" \
  && bad "finding(RP_SESSION-blacklist): a dotted session name must NOT trigger the RP_SESSION refusal (regression to the whitelist!)" \
  || ok "finding(RP_SESSION-blacklist): a dotted session name triggers no RP_SESSION refusal"
# An ordinary session name (letters, digits, hyphen, underscore) is
# unaffected — pinned directly here alongside the blacklist shapes.
RP_SESSION="a100-2" bash "$DIR/gpu-dev.sh" down >"$SANDBOX/out-ordinary.log" 2>&1
[ $? -eq 0 ] && ok "finding(RP_SESSION-blacklist): an ordinary [A-Za-z0-9_-] session name is still accepted" \
  || bad "finding(RP_SESSION-blacklist): an ordinary session name must still work ($(cat "$SANDBOX/out-ordinary.log"))"

# ═════════════════════════════════════════════════════════════════════════
# Group 3e — CLI-level: a dotted session name works END TO END, not merely
# past the RP_SESSION gate — `down bench.1` actually releases ITS pod
# (round-3 audit's own reproduction: a live pod under a dotted session name
# was stranded for its full deadline because every verb refused against
# it). Same pattern as the pre-existing "down-ok" fixture above, just with
# a dotted session name.
# ═════════════════════════════════════════════════════════════════════════
SESSION_DOTTED="bench.1"
write_meta_legacy "$SESSION_DOTTED" "pod-bench-1"
echo '{"data":{"myself":{"pods":[{"id":"pod-bench-1","name":"jammi-gpu-ttl8"}]}}}' > "$SANDBOX/acct-dotted-1.json"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-dotted-2.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-dotted-1.json"
export MOCK_ACCOUNT_RESPONSE_2="$SANDBOX/acct-dotted-2.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOTTED" >"$SANDBOX/out-down-dotted.log" 2>&1
rc=$?
term_count_dotted="$(grep -c "podTerminate.*pod-bench-1" "$CALL_LOG")"
[ "$term_count_dotted" = "1" ] && [ "$rc" -eq 0 ] \
  && ok "finding(dotted-session-e2e): 'down bench.1' releases its pod end to end" \
  || bad "finding(dotted-session-e2e): expected exactly one confirmed terminate for pod-bench-1 (got count=${term_count_dotted} rc=${rc}); $(cat "$SANDBOX/out-down-dotted.log")"
[ -f "$RP_SESSION_ROOT/$SESSION_DOTTED/meta" ] \
  && bad "finding(dotted-session-e2e): bench.1's local record should be forgotten after a confirmed terminate" \
  || ok "finding(dotted-session-e2e): bench.1's local record is forgotten after a confirmed terminate"

# ═════════════════════════════════════════════════════════════════════════
# Group 3f — CLI-level: the RP_SESSION gate never applies to `ls`/`reap`
# (account-level verbs; they never read RP_SESSION at all). An exported
# RP_SESSION that would refuse under a NAMED-session verb — dotted, or even
# a traversal shape — must leave both `ls` and `reap` completely unaffected.
# ═════════════════════════════════════════════════════════════════════════
RP_SESSION="a100.2" bash "$DIR/gpu-dev.sh" ls >"$SANDBOX/out-ls-dotted.log" 2>&1
[ $? -eq 0 ] && ok "finding(ls-reap-exempt): 'ls' works under RP_SESSION=a100.2" \
  || bad "finding(ls-reap-exempt): 'ls' must work under RP_SESSION=a100.2 ($(cat "$SANDBOX/out-ls-dotted.log"))"
RP_SESSION="../evil" bash "$DIR/gpu-dev.sh" ls >"$SANDBOX/out-ls-traversal.log" 2>&1
[ $? -eq 0 ] && ok "finding(ls-reap-exempt): 'ls' is unaffected even by a traversal-shaped RP_SESSION" \
  || bad "finding(ls-reap-exempt): 'ls' must be unaffected by a traversal-shaped RP_SESSION ($(cat "$SANDBOX/out-ls-traversal.log"))"

echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-reap-dotted.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-reap-dotted.json"
RP_SESSION="a100.2" bash "$DIR/gpu-dev.sh" reap >"$SANDBOX/out-reap-dotted.log" 2>&1
rc=$?
[ "$rc" -eq 0 ] && ok "finding(ls-reap-exempt): 'reap' works under RP_SESSION=a100.2" \
  || bad "finding(ls-reap-exempt): 'reap' must work under RP_SESSION=a100.2 (got rc=$rc); $(cat "$SANDBOX/out-reap-dotted.log")"
if log_has "podTerminate"; then
  bad "finding(ls-reap-exempt): 'reap' against an empty account must terminate nothing (regression!)"
else
  ok "finding(ls-reap-exempt): 'reap' against an empty account terminates nothing"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 4 — rp_sweep's age-based judgement (function-level). Age math is
# REAL (computed from actual createdAt timestamps against the real clock);
# only rp_gql (the account response) and rp_terminate are mocked.
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  G4_TERM="$SANDBOX/g4-terminate.log"; : > "$G4_TERM"
  rp_terminate() { echo "$1" >> "$G4_TERM"; }

  now_epoch="$(date -u +%s)"
  stale_iso="$(iso_from_epoch $((now_epoch - 50 * 3600)))"    # 50h old, ttl8, past deadline
  fresh_iso="$(iso_from_epoch $((now_epoch - 1 * 3600)))"     # 1h old, ttl8, within deadline

  cat > "$SANDBOX/g4-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-stale","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${stale_iso}","runtime":{"uptimeInSeconds":180000}},
  {"id":"pod-fresh","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${fresh_iso}","runtime":{"uptimeInSeconds":3600}}
]}}}
JSON
  rp_gql() { cat "$SANDBOX/g4-account.json"; }

  rp_sweep >/dev/null 2>&1

  if grep -q "^pod-stale$" "$G4_TERM"; then
    record PASS "group4-stale-pod-terminated (past its own deadline)"
  else
    record FAIL "group4-stale-pod-terminated"
  fi
  if grep -q "^pod-fresh$" "$G4_TERM"; then
    record FAIL "group4-fresh-pod-not-terminated (well within its deadline)"
  else
    record PASS "group4-fresh-pod-not-terminated"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 4b — rp_sweep's override validation (function-level; round-4 audit
# BLOCK). "0" and "00" are all-digit (no non-digit character), so the
# digit-shape check alone lets them through; Python's `if override:` is then
# true for the STRING "0" exactly as it is for "8", giving `limit = 0` —
# every RUNNING pod's age is "past-deadline-0s", so `reap 0` mass-terminates
# the whole account's jammi-gpu* fleet instead of refusing.
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  G4B_TERM="$SANDBOX/g4b-terminate.log"; : > "$G4B_TERM"
  rp_terminate() { echo "$1" >> "$G4B_TERM"; }
  # The validation must reject BEFORE any account query — a query the fix
  # doesn't even need should never be reached on a rejected override.
  rp_gql() { echo "SHOULD_NOT_BE_CALLED" >> "$SANDBOX/g4b-query.log"; echo '{}'; }

  : > "$G4B_TERM"; : > "$SANDBOX/g4b-query.log"
  rp_sweep 0 >/dev/null 2>&1
  rc0=$?
  [ "$rc0" -eq 2 ] && record PASS "group4b-reap-zero-exits-2" || record FAIL "group4b-reap-zero-exits-2 (rc=$rc0)"
  [ -s "$G4B_TERM" ] && record FAIL "group4b-reap-zero-terminates-nothing (regression!)" \
    || record PASS "group4b-reap-zero-terminates-nothing"
  [ -s "$SANDBOX/g4b-query.log" ] && record FAIL "group4b-reap-zero-issues-no-query (regression!)" \
    || record PASS "group4b-reap-zero-issues-no-query"

  : > "$G4B_TERM"; : > "$SANDBOX/g4b-query.log"
  rp_sweep 00 >/dev/null 2>&1
  rc00=$?
  [ "$rc00" -eq 2 ] && record PASS "group4b-reap-double-zero-exits-2" || record FAIL "group4b-reap-double-zero-exits-2 (rc=$rc00)"
  [ -s "$G4B_TERM" ] && record FAIL "group4b-reap-double-zero-terminates-nothing (regression!)" \
    || record PASS "group4b-reap-double-zero-terminates-nothing"

  # Positive control: `reap 1` (a REAL positive override) must still work —
  # this fix must not have broken force-reap itself, only the zero case.
  # Two pods: one 4h old (past a 1h force-reap ceiling) and one 20m old
  # (well within it) — only the 4h one is swept.
  rp_gql() { cat "$SANDBOX/g4b-account.json"; }
  now_epoch="$(date -u +%s)"
  h4_iso="$(iso_from_epoch $((now_epoch - 4 * 3600)))"
  m20_iso="$(iso_from_epoch $((now_epoch - 20 * 60)))"
  cat > "$SANDBOX/g4b-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-4h","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${h4_iso}","runtime":{"uptimeInSeconds":14000}},
  {"id":"pod-20m","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${m20_iso}","runtime":{"uptimeInSeconds":1200}}
]}}}
JSON
  : > "$G4B_TERM"
  rp_sweep 1 >/dev/null 2>&1
  if grep -q "^pod-4h$" "$G4B_TERM"; then
    record PASS "group4b-reap-one-still-sweeps-the-4h-pod"
  else
    record FAIL "group4b-reap-one-still-sweeps-the-4h-pod"
  fi
  if grep -q "^pod-20m$" "$G4B_TERM"; then
    record FAIL "group4b-reap-one-does-not-sweep-the-20m-pod (well within a 1h force-reap ceiling)"
  else
    record PASS "group4b-reap-one-does-not-sweep-the-20m-pod"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 5 — RP_SSHO must pin every connection to the tooling's own key via
# IdentitiesOnly=yes (2026-08-26 incident, ledger row 328): without it, an
# ssh-agent holding many identities offers all of them before RP_SSH_KEY,
# and the reachability probe in rp_deploy_live can exhaust the pod's own
# MaxAuthTries before RP_SSH_KEY is ever tried — reading a perfectly
# reachable pod as unreachable and terminating it. `ssh -G` resolves the
# EFFECTIVE configuration for a candidate host with zero network traffic (it
# never actually connects), so this is a hermetic, direct assertion on the
# real option array rp_deploy_live and every gpu-dev.sh ssh/rsync call
# share — not a grep on the source text, which could pass with the option
# spelled right but never actually reaching ssh's own argv.
# ═════════════════════════════════════════════════════════════════════════
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  rp_init
  g5_out="$(ssh -G "${RP_SSHO[@]}" placeholder-host 2>&1)"
  if printf '%s\n' "$g5_out" | grep -qi '^identitiesonly yes$'; then
    record PASS "group5-RP_SSHO-identitiesonly-yes"
  else
    record FAIL "group5-RP_SSHO-identitiesonly-yes (ssh -G output: $g5_out)"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 6 — CLI-level: rp_deploy_live's SSH-reachability poll honours
# RP_SSH_WAIT_SECS as a real WALL-CLOCK deadline, not the old fixed
# 24-iteration/10s-sleep budget (a hard-coded ~4 minutes no caller could
# raise or shrink). The port query is mocked to NEVER return a public
# port-22 mapping (the curl stub's own default for `pod(input:...)` — see
# above), so the loop runs out its own deadline rather than taking the
# reachable-pod branch; a 2s deadline keeps this test fast while still
# proving the deadline is real wall-clock time — a fixed-iteration budget
# would run ~4 minutes regardless of what RP_SSH_WAIT_SECS says.
# ═════════════════════════════════════════════════════════════════════════
echo '{"data":{"podFindAndDeployOnDemand":{"id":"pod-wait-secs-test"}}}' > "$SANDBOX/deploy-wait-secs.json"
export MOCK_DEPLOY_RESPONSE="$SANDBOX/deploy-wait-secs.json"
reset_log
g6_start="$SECONDS"
RP_SSH_WAIT_SECS=2 bash "$DIR/gpu-dev.sh" shell a100 --ref abcdef1234567890 \
  >"$SANDBOX/out-wait-secs.log" 2>&1 &
G6_PID=$!
# Bounded wait: this run needs no signal — it naturally exits once the
# poll's own 2s deadline expires. Never an indefinite `wait`: a regression
# that reintroduced the old fixed-iteration budget fails this assertion
# instead of wedging the whole suite for minutes.
g6_deadline=$((SECONDS + 20))
while kill -0 "$G6_PID" 2>/dev/null && [ "$SECONDS" -lt "$g6_deadline" ]; do sleep 0.2; done
if kill -0 "$G6_PID" 2>/dev/null; then
  kill -KILL "$G6_PID" 2>/dev/null
  wait "$G6_PID" 2>/dev/null
  bad "finding(RP_SSH_WAIT_SECS): 'shell' with RP_SSH_WAIT_SECS=2 did not exit within 20s (regression to a fixed-iteration budget?); output: $(cat "$SANDBOX/out-wait-secs.log")"
else
  wait "$G6_PID" 2>/dev/null
  g6_rc=$?
  g6_elapsed=$((SECONDS - g6_start))
  # >=2 proves the deadline was actually honoured (not skipped or zeroed);
  # <15 proves it is bounded by RP_SSH_WAIT_SECS, not the old ~240s budget —
  # a wide ceiling since this also carries process start/CLI-parse overhead.
  if [ "$g6_elapsed" -ge 2 ] && [ "$g6_elapsed" -lt 15 ]; then
    ok "finding(RP_SSH_WAIT_SECS): the reachability poll's wall-clock deadline is honoured (${g6_elapsed}s for a 2s budget)"
  else
    bad "finding(RP_SSH_WAIT_SECS): expected roughly 2-15s elapsed for RP_SSH_WAIT_SECS=2 (got ${g6_elapsed}s, rc=${g6_rc}); output: $(cat "$SANDBOX/out-wait-secs.log")"
  fi
  [ "$g6_rc" -eq 75 ] && ok "finding(RP_SSH_WAIT_SECS): exhausting the deadline with no reachable candidate returns 75 (neutral skip)" \
    || bad "finding(RP_SSH_WAIT_SECS): expected exit 75 once the deadline is exhausted (got $g6_rc)"
  grep -q "never became reachable within 2s" "$SANDBOX/out-wait-secs.log" \
    && ok "finding(RP_SSH_WAIT_SECS): the refusal names the actual budget (2s), not a hard-coded '4m'" \
    || bad "finding(RP_SSH_WAIT_SECS): expected 'never became reachable within 2s' in output ($(cat "$SANDBOX/out-wait-secs.log"))"
fi
unset MOCK_DEPLOY_RESPONSE

# ═════════════════════════════════════════════════════════════════════════
# Group 7 — CLI-level: `wait-seed`/`wait-job` (the fail-open-watcher lesson).
# `ssh` is deliberately mocked ONLY inside this group, via its own bin dir
# prepended to PATH for these specific invocations — never added to the
# shared $STUBBIN every other group's PATH carries, so no earlier/later
# group's "ssh to 127.0.0.1:1 refuses instantly" assumption is disturbed.
#
# The stub answers CALL ORDER, not script content: response #1 always
# services `require_pod`'s own `rp_session_alive` liveness probe (`ssh ...
# true`), which rp_wait_poll's own polls run AFTER. Response files are
# "<rc>\n<output...>"; a call past the last scripted response repeats it
# (mirrors the curl stub's own _2-falls-back-to-_1 convention above).
# ═════════════════════════════════════════════════════════════════════════
WAITBIN="$SANDBOX/waitbin"
mkdir -p "$WAITBIN"
cat > "$WAITBIN/ssh" <<'STUB'
#!/usr/bin/env bash
cat > /dev/null   # discard the piped poll script -- the stub never reads it
n=0
[ -f "${MOCK_SSH_CALL_COUNTER:?MOCK_SSH_CALL_COUNTER unset}" ] && n="$(cat "$MOCK_SSH_CALL_COUNTER")"
n=$((n + 1))
echo "$n" > "$MOCK_SSH_CALL_COUNTER"
dir="${MOCK_SSH_RESPONSES_DIR:?MOCK_SSH_RESPONSES_DIR unset}"
resp="$dir/$n"
[ -f "$resp" ] || resp="$dir/$(ls "$dir" | sort -n | tail -1)"
rc="$(head -n1 "$resp")"
tail -n +2 "$resp"
exit "$rc"
STUB
chmod +x "$WAITBIN/ssh"

# $1=dir $2=call-number $3=rc $4...=output lines
write_ssh_resp() {
  local dir="$1" n="$2" rc="$3"; shift 3
  { echo "$rc"; printf '%s\n' "$@"; } > "$dir/$n"
}

# --- 7a: wait-seed SUCCESS -------------------------------------------------
G7A_SESSION="wsA"; write_meta "$G7A_SESSION" "pod-wsA" "8"
G7A_DIR="$SANDBOX/g7a-ssh"; mkdir -p "$G7A_DIR"
write_ssh_resp "$G7A_DIR" 1 0                                    # require_pod liveness
write_ssh_resp "$G7A_DIR" 2 0 "seed complete: {\"tuples\":[\"T1\"]}"  # first poll: COMPLETE
rm -f "$SANDBOX/g7a-counter"
MOCK_SSH_CALL_COUNTER="$SANDBOX/g7a-counter" MOCK_SSH_RESPONSES_DIR="$G7A_DIR" \
  RP_WAIT_INTERVAL_SECS=1 \
  PATH="$WAITBIN:$PATH" bash "$DIR/gpu-dev.sh" wait-seed "$G7A_SESSION" --timeout 10 \
  >"$SANDBOX/out-g7a.log" 2>&1
g7a_rc=$?
if [ "$g7a_rc" -eq 0 ] && grep -q "SUCCESS" "$SANDBOX/out-g7a.log"; then
  ok "wait-seed: success path exits 0 and reports SUCCESS"
else
  bad "wait-seed: success path expected rc=0 + SUCCESS (got rc=$g7a_rc): $(cat "$SANDBOX/out-g7a.log")"
fi

# --- 7b: wait-seed FAILURE (failure marker) --------------------------------
G7B_SESSION="wsB"; write_meta "$G7B_SESSION" "pod-wsB" "8"
G7B_DIR="$SANDBOX/g7b-ssh"; mkdir -p "$G7B_DIR"
write_ssh_resp "$G7B_DIR" 1 0                                     # require_pod liveness
write_ssh_resp "$G7B_DIR" 2 1 "seed FAILED -- nvcc fatal: out of memory"
rm -f "$SANDBOX/g7b-counter"
MOCK_SSH_CALL_COUNTER="$SANDBOX/g7b-counter" MOCK_SSH_RESPONSES_DIR="$G7B_DIR" \
  RP_WAIT_INTERVAL_SECS=1 \
  PATH="$WAITBIN:$PATH" bash "$DIR/gpu-dev.sh" wait-seed "$G7B_SESSION" --timeout 10 \
  >"$SANDBOX/out-g7b.log" 2>&1
g7b_rc=$?
if [ "$g7b_rc" -ne 0 ] && grep -q "FAILURE" "$SANDBOX/out-g7b.log" && grep -q "nvcc fatal" "$SANDBOX/out-g7b.log"; then
  ok "wait-seed: failure-marker path exits non-zero and names the failure"
else
  bad "wait-seed: failure-marker path expected rc!=0 + FAILURE naming nvcc (got rc=$g7b_rc): $(cat "$SANDBOX/out-g7b.log")"
fi

# --- 7c: wait-seed TRANSPORT FAILURE (the load-bearing case) --------------
# Three consecutive unreachable polls (rc 255, ssh's own "could not connect"
# convention) must exit loudly -- NEVER read as "still running", which is
# exactly the silent-idle failure mode this verb replaces.
G7C_SESSION="wsC"; write_meta "$G7C_SESSION" "pod-wsC" "8"
G7C_DIR="$SANDBOX/g7c-ssh"; mkdir -p "$G7C_DIR"
write_ssh_resp "$G7C_DIR" 1 0                                     # require_pod liveness
write_ssh_resp "$G7C_DIR" 2 255 "ssh: connect to host 127.0.0.1 port 1: Connection refused"
write_ssh_resp "$G7C_DIR" 3 255 "ssh: connect to host 127.0.0.1 port 1: Connection refused"
write_ssh_resp "$G7C_DIR" 4 255 "ssh: connect to host 127.0.0.1 port 1: Connection refused"
rm -f "$SANDBOX/g7c-counter"
MOCK_SSH_CALL_COUNTER="$SANDBOX/g7c-counter" MOCK_SSH_RESPONSES_DIR="$G7C_DIR" \
  RP_WAIT_INTERVAL_SECS=1 \
  PATH="$WAITBIN:$PATH" bash "$DIR/gpu-dev.sh" wait-seed "$G7C_SESSION" --timeout 30 \
  >"$SANDBOX/out-g7c.log" 2>&1
g7c_rc=$?
if [ "$g7c_rc" -eq 2 ] && grep -q "TRANSPORT FAILURE" "$SANDBOX/out-g7c.log" \
  && grep -q "NOT evidence" "$SANDBOX/out-g7c.log"; then
  ok "wait-seed: 3 consecutive unreachable polls is a LOUD transport failure (rc=2), never silent 'still running'"
else
  bad "wait-seed: expected rc=2 + TRANSPORT FAILURE naming the count (got rc=$g7c_rc): $(cat "$SANDBOX/out-g7c.log")"
fi
if [ "$(cat "$SANDBOX/g7c-counter" 2>/dev/null)" = "4" ]; then
  ok "wait-seed: transport failure gives up after exactly RP_WAIT_MAX_TRANSPORT_FAILS(=3) consecutive misses, not before or after"
else
  bad "wait-seed: expected exactly 4 ssh calls (1 liveness + 3 failed polls), got $(cat "$SANDBOX/g7c-counter" 2>/dev/null)"
fi

# --- 7d: wait-job SUCCESS (tmux session ended, .jammi.log present) --------
G7D_SESSION="wjD"; write_meta "$G7D_SESSION" "pod-wjD" "8"
G7D_DIR="$SANDBOX/g7d-ssh"; mkdir -p "$G7D_DIR"
write_ssh_resp "$G7D_DIR" 1 0                                     # require_pod liveness
write_ssh_resp "$G7D_DIR" 2 0 "job finished (tmux session jammi-jammi-ai ended) -- tail:" "all tests passed"
rm -f "$SANDBOX/g7d-counter"
MOCK_SSH_CALL_COUNTER="$SANDBOX/g7d-counter" MOCK_SSH_RESPONSES_DIR="$G7D_DIR" \
  RP_WAIT_INTERVAL_SECS=1 \
  PATH="$WAITBIN:$PATH" bash "$DIR/gpu-dev.sh" wait-job "$G7D_SESSION" --timeout 10 \
  >"$SANDBOX/out-g7d.log" 2>&1
g7d_rc=$?
if [ "$g7d_rc" -eq 0 ] && grep -q "SUCCESS" "$SANDBOX/out-g7d.log"; then
  ok "wait-job: success path (tmux session ended, log present) exits 0"
else
  bad "wait-job: success path expected rc=0 + SUCCESS (got rc=$g7d_rc): $(cat "$SANDBOX/out-g7d.log")"
fi

# --- 7e: wait-job FAILURE (no evidence -- never ran) -----------------------
G7E_SESSION="wjE"; write_meta "$G7E_SESSION" "pod-wjE" "8"
G7E_DIR="$SANDBOX/g7e-ssh"; mkdir -p "$G7E_DIR"
write_ssh_resp "$G7E_DIR" 1 0                                     # require_pod liveness
write_ssh_resp "$G7E_DIR" 2 1 "no job evidence for tree 'jammi-ai': no live tmux session and no .jammi.log"
rm -f "$SANDBOX/g7e-counter"
MOCK_SSH_CALL_COUNTER="$SANDBOX/g7e-counter" MOCK_SSH_RESPONSES_DIR="$G7E_DIR" \
  RP_WAIT_INTERVAL_SECS=1 \
  PATH="$WAITBIN:$PATH" bash "$DIR/gpu-dev.sh" wait-job "$G7E_SESSION" --timeout 10 \
  >"$SANDBOX/out-g7e.log" 2>&1
g7e_rc=$?
if [ "$g7e_rc" -ne 0 ] && grep -q "no job evidence" "$SANDBOX/out-g7e.log"; then
  ok "wait-job: no-evidence path (never ran) exits non-zero naming it"
else
  bad "wait-job: expected rc!=0 + 'no job evidence' (got rc=$g7e_rc): $(cat "$SANDBOX/out-g7e.log")"
fi

# --- 7f: wait-seed/wait-job honour the SAME RP_SESSION-vs-positional
# conflict refusal every other session verb does (never silently pick one).
G7F_OUT="$SANDBOX/out-g7f.log"
RP_SESSION="other-session" bash "$DIR/gpu-dev.sh" wait-seed "$G7A_SESSION" >"$G7F_OUT" 2>&1
g7f_rc=$?
if [ "$g7f_rc" -eq 2 ] && grep -q "conflicting session" "$G7F_OUT"; then
  ok "wait-seed: a positional session conflicting with exported RP_SESSION REFUSES (exit 2)"
else
  bad "wait-seed: expected exit 2 + 'conflicting session' (got rc=$g7f_rc): $(cat "$G7F_OUT")"
fi

# --- 7g: --timeout must reject a non-positive-integer value ---------------
G7G_OUT="$SANDBOX/out-g7g.log"
bash "$DIR/gpu-dev.sh" wait-seed "$G7A_SESSION" --timeout notanumber >"$G7G_OUT" 2>&1
g7g_rc=$?
if [ "$g7g_rc" -eq 2 ] && grep -q "must be a positive integer" "$G7G_OUT"; then
  ok "wait-seed: a non-integer --timeout is refused (exit 2), never silently defaulted"
else
  bad "wait-seed: expected exit 2 for --timeout notanumber (got rc=$g7g_rc): $(cat "$G7G_OUT")"
fi

# ── tally ────────────────────────────────────────────────────────────────
while IFS=: read -r status name; do
  [ -n "$status" ] || continue
  if [ "$status" = "PASS" ]; then
    ok "$name"
  else
    bad "$name"
  fi
done < "$RESULTS"

echo
echo "gpu-dev-lifecycle: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
