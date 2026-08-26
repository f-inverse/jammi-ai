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
    *podTerminate*|*podFindAndDeployOnDemand*|*'myself{'*)
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

  echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct3.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct3.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 1 ] && record PASS "group2-verify-ghost-id-refused" || record FAIL "group2-verify-ghost-id-refused"

  echo 'not json' > "$SANDBOX/acct4.json"
  MOCK_RESPONSE_FILE="$SANDBOX/acct4.json"
  rp_pod_verify "pod-x" >/dev/null 2>&1
  [ $? -eq 2 ] && record PASS "group2-verify-query-failure-refused" || record FAIL "group2-verify-query-failure-refused"
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

# A recorded id absent from the account entirely — the ghost-id shape —
# refuses without ever attempting a terminate, and keeps the local record.
SESSION_DOWN_GHOST="down-ghost"
write_meta "$SESSION_DOWN_GHOST" "pod-ghost" "8"
echo '{"data":{"myself":{"pods":[]}}}' > "$SANDBOX/acct-down-ghost-1.json"
reset_log; reset_account_seq
export MOCK_ACCOUNT_RESPONSE_1="$SANDBOX/acct-down-ghost-1.json"
bash "$DIR/gpu-dev.sh" down "$SESSION_DOWN_GHOST" >"$SANDBOX/out-down-ghost.log" 2>&1
if log_has "podTerminate"; then
  bad "finding-5(down): a ghost id absent from the account must refuse to terminate (regression!)"
else
  ok "finding-5(down): a ghost id refused to terminate"
fi
grep -q "refusing to terminate" "$SANDBOX/out-down-ghost.log" \
  && ok "finding-5(down): ghost-id refusal prints a refusal message" \
  || bad "finding-5(down): expected a refusal message on a ghost id"
[ -f "$RP_SESSION_ROOT/$SESSION_DOWN_GHOST/meta" ] \
  && ok "finding-5(down): the local record is KEPT after a ghost-id refusal" \
  || bad "finding-5(down): the local record must survive a ghost-id refusal (regression!)"

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

# Neither refusal path suggests `reap <hours>` (account-wide, never a
# per-pod remedy) or the removed, INERT RP_TTL_HOURS=<H> override.
if grep -q -- "reap <hours>" "$SANDBOX/out-down-ghost.log" "$SANDBOX/out-down-unshaped.log"; then
  bad "finding: down's refusal must not suggest 'reap <hours>' as a per-pod remedy (it is account-wide; regression!)"
else
  ok "finding: down's refusal does not suggest reap as a per-pod remedy"
fi
if grep -q -- "RP_TTL_HOURS=<H>" "$SANDBOX/out-down-ghost.log" "$SANDBOX/out-down-unshaped.log"; then
  bad "finding: down's refusal must not promise the removed, inert RP_TTL_HOURS=<H> override (regression!)"
else
  ok "finding: down's refusal does not promise the removed RP_TTL_HOURS=<H> override"
fi

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
