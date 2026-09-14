#!/usr/bin/env bash
# GPU-gang-lane fixture suite. Mocks-only: no network, no GPU, no RunPod
# account, no pod. Shaped on `test_gpu_prove_lane.sh` (which owns the same
# job for the prove lane) and driving the REAL objects rather than
# paraphrases of them: `runpod_gpu_gang.sh` is `source`d, not executed — see
# that file's own "sourced-execution guard" comment, which skips the
# sweep/deploy/heredoc flow when sourced — so `rp_gang_verdict`,
# `GANG_GROUPS` and the lane's own env pins are the committed ones.
#
# Cases:
#   G0  sourcing the driver invokes no curl/ssh/scp/runpodctl — the
#       "mocks-only" claim measured through a PATH shim, not asserted in a
#       comment.
#   G1  `rp_gang_verdict` over EVERY rc arm: a clean pass; ssh 0 with a
#       missing marker; ssh 0 with a non-zero group; a cut/hang (76, 124)
#       with no `PROVE_EXIT`; an in-suite exit returned verbatim (1, 97,
#       255); an abrupt cut whose final line has no newline; an unreadable
#       log (every marker lost -> FAIL, never a silent pass).
#   G2  the GANG_GROUPS closure: every `::group::` name in the driver minus
#       `device` equals GANG_GROUPS exactly.
#   G3  the never-vacuous arms of the `gang-proof` group, driven as the
#       EXPANDED REMOTE TEXT the driver really sends (the `<<REMOTE`
#       heredoc, expanded exactly as the driver expands it): "running 0
#       tests" is a FAIL, and a pass that wrote no artifact is a FAIL.
#       Plus the unescaped-backtick/`$(` static guard on that heredoc, the
#       same class the prove suite catches (an unescaped pair is evaluated
#       LOCALLY, on the runner, before a byte reaches ssh).
#   G5  the post-run artifact retrieval (U7b-A1-pull P1/P2), driven as the
#       EXPANDED LOCAL TEXT the driver really runs after the remote heredoc
#       exits (extracted the same way G3 extracts the remote text, but from
#       the driver's own local block, under a real `rsync` shim on PATH so
#       no network call is made): a failed pull JOINS the leg's own `rc`
#       (never a silently-warned second exit path), and a run of >=128
#       contiguous hex characters (the shape of a hex-encoded NCCL id,
#       256 at full length) found in the pulled
#       artifact dir OR the run's own log fails the leg too — a clean pull
#       with a clean artifact/log stays silent.
#   G6  the two `JAMMI_REQUIRE_*` exports (U7b-A1-pull P3) are present in
#       the `<<REMOTE` heredoc body, beside the existing env block; removing
#       either is a mutation this case catches.
#   G7  NO `schedule:` key exists anywhere in the committed `gpu-gang.yml`
#       (U7b-A1-pull P5) — re-adding one (any cron) is a mutation this case
#       catches. This row is itself deleted the moment U7b-A3 re-adds the
#       schedule block with its own never-vacuous writer; a permanent
#       "no schedule" assertion would then reject the correct end state.
#   G4  the cost bound is what the MECHANISM produces: the `$/run` figure
#       printed in `gpu-gang.yml`, in the driver's own header and in
#       `docs/maintainer/dev-gpu.md` is re-derived here from the `a100`
#       candidate list in `runpod_lib.sh`, this lane's own
#       RP_SSH_WAIT_SECS/RP_TTL_HOURS/RP_GPU_COUNT, the workflow's
#       MAX_ATTEMPTS and the measured $3.18/h rate — a drifted pin or a
#       silently raised term is a failure here, not a number nobody
#       re-checks. The GPU count is a term of the RATE, not of the hours:
#       $3.18/h prices a 2-GPU pod, so a lane that moved to 4 would keep
#       printing a bound it no longer bills at.
#
# Run: bash ci/scripts/test_gpu_gang_lane.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
GANG_SH="$DIR/runpod_gpu_gang.sh"
LIB_SH="$DIR/runpod_lib.sh"
GANG_YML="$REPO_ROOT/.github/workflows/gpu-gang.yml"
DEV_GPU_MD="$REPO_ROOT/docs/maintainer/dev-gpu.md"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

SANDBOX="$(mktemp -d)"
NETPROBE_LOG="$SANDBOX/netprobe.log"
# This EXIT trap does NOT survive the `source` below: runpod_lib.sh installs
# `trap rp_cleanup EXIT` of its own at source time, replacing whatever was
# registered here (measured, not assumed). So nothing in this suite may rely
# on an exit-time hook, and the external-call claim below is asserted inline.
trap 'rm -rf "$SANDBOX"' EXIT

# --------------------------------------------------------------------------
# Source the real driver. Sourcing makes NO network call (the guard skips
# rp_sweep/rp_init/rp_deploy_arch and the heredoc), but runpod_lib.sh does
# validate its own env at source time, hence the dummy key.
# --------------------------------------------------------------------------
export RUNPOD_API_KEY="test-dummy-key"

# "Sourcing rents nothing" is a MEASURED claim here, not a comment.
#
# A bash-function stub for `rp_deploy_arch`/`rp_deploy_live` would measure
# NOTHING: `runpod_gpu_gang.sh` sources `runpod_lib.sh`, whose own
# definitions REPLACE any same-named function defined beforehand (checked:
# after sourcing, `declare -f rp_deploy_arch` is the library's body). So the
# probe is placed one level down, on the primitives the library cannot
# redefine — `curl`, which every deploy/terminate/query goes through via
# `rp_gql`, and `ssh`, the only other external effect on that path. Each is
# a real executable on a PATH prefix that appends its argv to a log; the
# claim is then the log, not an assumption about the guard.
#
# The shims exit 1, so a driver that reaches one fails LOUDLY rather than
# proceeding on a fabricated response. That also means a guard regression
# aborts this suite inside the `source`, before the assertion below prints:
# the observable is then the suite's own non-zero exit (measured: with the
# guard neutralised to `if true`, the suite exits 1 and the driver's RunPod
# query/deploy errors are what stopped it — a real key would have rented).
# The shims stay on PATH for the whole run: nothing in a mocks-only suite
# has any business calling them later either.
NETPROBE_BIN="$SANDBOX/netprobe-bin"
mkdir -p "$NETPROBE_BIN"
for tool in curl ssh scp runpodctl; do
  cat >"$NETPROBE_BIN/$tool" <<PROBE
#!/usr/bin/env bash
printf '%s %s\n' "$tool" "\$*" >>"$NETPROBE_LOG"
exit 1
PROBE
  chmod +x "$NETPROBE_BIN/$tool"
done
: >"$NETPROBE_LOG"
PATH="$NETPROBE_BIN:$PATH"

# shellcheck source=ci/scripts/runpod_gpu_gang.sh
source "$GANG_SH"

netprobe_calls="$(grep -c . "$NETPROBE_LOG" 2>/dev/null || true)" # tripwire-ok: grep -c on an EMPTY log legitimately exits 1; zero IS the pass condition, asserted on the next line.
if [ "${netprobe_calls:-0}" -eq 0 ]; then
  ok "sourcing runpod_gpu_gang.sh invoked NO curl/ssh/scp/runpodctl (counted through a PATH shim, not assumed) — the sourced-execution guard really does cover the rent/deploy path"
else
  bad "sourcing runpod_gpu_gang.sh invoked ${netprobe_calls} external call(s): $(tr '\n' '; ' <"$NETPROBE_LOG") — the sourced-execution guard no longer covers the rent path, and this suite would spend money"
fi

if declare -f rp_gang_verdict >/dev/null && declare -f rp_run_remote_watched >/dev/null; then
  ok "sourcing runpod_gpu_gang.sh (no network) defines rp_gang_verdict and (via runpod_lib.sh) rp_run_remote_watched"
else
  bad "sourcing runpod_gpu_gang.sh did not define the expected functions"
fi

# ============================================================================
# G2: GANG_GROUPS closure — {::group:: names} - {device} == GANG_GROUPS.
# ============================================================================
mapfile -t script_groups < <(grep -oE '::group::[a-z0-9-]+' "$GANG_SH" | sed 's/::group:://' | sort -u)
mapfile -t declared_groups < <(printf '%s\n' "${GANG_GROUPS[@]}" | sort -u)
non_members=()
for g in "${script_groups[@]}"; do
  [ "$g" = "device" ] && continue
  non_members+=("$g")
done
mapfile -t non_members_sorted < <(printf '%s\n' "${non_members[@]}" | sort -u)
if [ "${#non_members_sorted[@]}" -eq "${#declared_groups[@]}" ] \
  && [ "$(printf '%s\n' "${non_members_sorted[@]}")" = "$(printf '%s\n' "${declared_groups[@]}")" ]; then
  ok "G2: GANG_GROUPS closure: {::group:: names} - {device} == GANG_GROUPS exactly (${#declared_groups[@]} members)"
else
  bad "G2: GANG_GROUPS closure mismatch: script non-member groups=[${non_members_sorted[*]}] vs GANG_GROUPS=[${declared_groups[*]}]"
fi

# ============================================================================
# G1: rp_gang_verdict over every rc arm.
# ============================================================================
all_pass_log() {
  local log="$1"
  {
    for g in "${GANG_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
    echo "PROVE_EXIT=0"
  } > "$log"
}

drive() { # $1=raw_rc $2=log -> prints "rc|stderr"
  local out rc
  out="$(rp_gang_verdict "$1" "$2" 2>&1 >/dev/null)"
  rp_gang_verdict "$1" "$2" >/dev/null 2>&1
  rc=$?
  printf '%s|%s' "$rc" "$out"
}

log="$SANDBOX/g1-clean.log"
all_pass_log "$log"
res="$(drive 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 0 ] && [ -z "$err" ]; then
  ok "G1: every gating group rc=0 with ssh 0 -> 0, no diagnostic"
else
  bad "G1: expected a silent 0; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-missing.log"
{
  echo "PROVE_GROUP_RC name=gang-build rc=0"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"gang-proof=<missing>"* ]]; then
  ok "G1: ssh 0 with a MISSING group marker -> 1, naming the group as <missing>"
else
  bad "G1: expected 1 + gang-proof=<missing>; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-nonzero.log"
{
  echo "PROVE_GROUP_RC name=gang-build rc=0"
  echo "PROVE_GROUP_RC name=gang-proof rc=3"
  echo "PROVE_EXIT=0"
} > "$log"
res="$(drive 0 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"gang-proof=3"* ]]; then
  ok "G1: ssh 0 with a NON-ZERO group marker -> 1 (a bare zero is never trusted against its own markers)"
else
  bad "G1: expected 1 + gang-proof=3; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-hang.log"
echo "PROVE_GROUP_RC name=gang-build rc=0" > "$log"
res="$(drive 76 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 76 ] && [[ "$err" == *"cut/hang (raw rc=76)"* ]] && [[ "$err" == *"gang-proof=<missing>"* ]]; then
  ok "G1: inactivity kill (76, no PROVE_EXIT) -> 76, naming the unresolved group"
else
  bad "G1: expected 76 + cut/hang naming gang-proof; got rc=$rc err=$err"
fi

log="$SANDBOX/g1-cut.log"
echo "PROVE_GROUP_RC name=gang-build rc=0" > "$log"
res="$(drive 124 "$log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 124 ] && [[ "$err" == *"cut/hang (raw rc=124)"* ]]; then
  ok "G1: budget cut (124, no PROVE_EXIT) -> 124, naming the unresolved group"
else
  bad "G1: expected 124 + cut/hang; got rc=$rc err=$err"
fi

for insuite in 1 97 255; do
  log="$SANDBOX/g1-insuite-$insuite.log"
  {
    echo "PROVE_GROUP_RC name=gang-build rc=0"
    echo "PROVE_GROUP_RC name=gang-proof rc=${insuite}"
    echo "PROVE_EXIT=${insuite}"
  } > "$log"
  res="$(drive "$insuite" "$log")"; rc="${res%%|*}"; err="${res#*|}"
  if [ "$rc" -eq "$insuite" ] && [[ "$err" != *"cut/hang"* ]]; then
    ok "G1: an in-suite exit ${insuite} (PROVE_EXIT present) is returned VERBATIM, no cut diagnostic"
  else
    bad "G1: expected ${insuite} verbatim; got rc=$rc err=$err"
  fi
done

# An abrupt ssh cut leaves the final line unterminated; a bare `while read`
# loop drops it. The marker on that line must still credit.
log="$SANDBOX/g1-partial.log"
printf 'PROVE_GROUP_RC name=gang-build rc=0\nPROVE_GROUP_RC name=gang-proof rc=0' > "$log"
res="$(drive 0 "$log")"; rc="${res%%|*}"
if [ "$rc" -eq 0 ]; then
  ok "G1: a marker on an UNTERMINATED final line is still parsed (abrupt-cut shape)"
else
  bad "G1: expected 0 from an unterminated final marker line; got rc=$rc"
fi

# An unreadable log loses every marker: with ssh 0 that must FAIL, never
# read as "nothing went wrong".
res="$(drive 0 "$SANDBOX/g1-does-not-exist.log")"; rc="${res%%|*}"; err="${res#*|}"
if [ "$rc" -eq 1 ] && [[ "$err" == *"<missing>"* ]]; then
  ok "G1: an unreadable log (every marker lost) with ssh 0 -> 1, fail-closed"
else
  bad "G1: expected 1 from an unreadable log; got rc=$rc err=$err"
fi

# ============================================================================
# G3: the remote text's never-vacuous arms, driven as the driver expands it.
# ============================================================================
hd_start="$(grep -n '<<REMOTE' "$GANG_SH" | head -1 | cut -d: -f1)"
hd_end="$(grep -n '^REMOTE$' "$GANG_SH" | head -1 | cut -d: -f1)"
hd_body="$(sed -n "$((hd_start + 1)),$((hd_end - 1))p" "$GANG_SH")"

# Static guard first: the heredoc is UNQUOTED by design (the local shell
# must expand ${GIT_REF} etc. into the remote text), so an unescaped
# backtick pair or `$(...)` is evaluated on the RUNNER before a byte reaches
# ssh — silently deleting text, or running an unintended command. Checked
# BEFORE the expansion below, which would otherwise be the thing that runs it.
unsafe="$(python3 - "$GANG_SH" "$hd_start" "$hd_end" <<'PY'
import re, sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(path, encoding="utf-8").read().splitlines()
for i, line in enumerate(lines[start:end - 1], start=start + 1):
    if re.search(r"(?<!\\)`", line) or re.search(r"(?<!\\)\$\(", line):
        print(f"{i}: {line}")
PY
)"
if [ -z "$unsafe" ]; then
  ok "G3: no unescaped backtick or \$(...) in the <<REMOTE heredoc body (nothing runs locally at construction time)"
else
  bad "G3: unescaped local expansion in the heredoc body: $unsafe"
fi

if [ -z "$unsafe" ]; then
  remote_text="$(eval "cat <<REMOTE
${hd_body}
REMOTE
")"
else
  remote_text=""
fi

# The `gang-proof` group's two never-vacuous arms, lifted VERBATIM out of
# that expanded text (from the zero-test grep through the empty-artifact
# check) and run against fixtures. `grc` is the group's own rc variable.
arms="$(printf '%s\n' "$remote_text" | sed -n '/^if grep -q "running 0 tests"/,/^\[ "\$grc" -ne 0 \] && rc=\$grc$/p' | sed '$d')"
if [ -n "$arms" ]; then
  ok "G3: extracted the gang-proof never-vacuous arms from the expanded remote text"
else
  bad "G3: could not extract the gang-proof never-vacuous arms from the expanded remote text"
fi

# The arms MUTATE `grc`, so the eval runs in THIS function's own shell (a
# `$(eval …)` capture would discard the assignment and read back the
# pre-arm value — which is exactly how a never-vacuous check gets tested
# into always-passing). Output goes to a file instead.
run_arms() { # $1=initial grc  $2=gang_log contents  $3=artifact dir state (empty|filled)
  local grc="$1"
  local gang_log="$SANDBOX/gang_proof.log"
  printf '%s\n' "$2" > "$gang_log"
  local JAMMI_GANG_ARTIFACT_DIR="$SANDBOX/artifact-$3"
  rm -rf "$JAMMI_GANG_ARTIFACT_DIR"; mkdir -p "$JAMMI_GANG_ARTIFACT_DIR"
  [ "$3" = "filled" ] && echo '{}' > "$JAMMI_GANG_ARTIFACT_DIR/gang.json"
  eval "$arms" > "$SANDBOX/arms.out" 2>&1
  printf '%s|%s' "$grc" "$(cat "$SANDBOX/arms.out")"
}

res="$(run_arms 0 "running 0 tests
test result: ok. 0 passed; 0 failed" filled)"
if [ "${res%%|*}" -eq 1 ] && [[ "${res#*|}" == *"matched ZERO tests"* ]]; then
  ok "G3: a zero-test filter match is a FAILURE with a named reason, never a silent pass"
else
  bad "G3: expected grc=1 + 'matched ZERO tests'; got $res"
fi

res="$(run_arms 0 "running 2 tests
test result: ok. 2 passed; 0 failed" empty)"
if [ "${res%%|*}" -eq 1 ] && [[ "${res#*|}" == *"wrote NO artifact"* ]]; then
  ok "G3: a passing run that wrote NO artifact is a FAILURE with a named reason"
else
  bad "G3: expected grc=1 + 'wrote NO artifact'; got $res"
fi

res="$(run_arms 0 "running 2 tests
test result: ok. 2 passed; 0 failed" filled)"
if [ "${res%%|*}" -eq 0 ] && [ -z "${res#*|}" ]; then
  ok "G3: a real pass with a written artifact stays 0 (the arms are not blanket-refusing)"
else
  bad "G3: expected a silent grc=0; got $res"
fi

# ============================================================================
# G5: the post-run artifact retrieval is FATAL on a failed pull, and the
# id-secrecy scan over the pulled artifact dir + this run's own log is a
# backstop (U7b-A1-pull P1/P2). Extracted from the driver's OWN local text
# the same way G3 extracts the remote heredoc's, and eval'd directly in this
# function's shell (never inside a `$(...)` capture, which would discard the
# `rc` mutation the arms make — the same reason G3's run_arms does not
# capture its own eval).
# ============================================================================
retrieval_start_ln="$(grep -n '^mkdir -p "\$GANG_ARTIFACT_DIR"$' "$GANG_SH" | head -1 | cut -d: -f1)"
retrieval_end_ln="$(grep -n '^# --- end artifact retrieval and id-secrecy check ---$' "$GANG_SH" | head -1 | cut -d: -f1)"
if [ -n "$retrieval_start_ln" ] && [ -n "$retrieval_end_ln" ]; then
  retrieval_body="$(sed -n "${retrieval_start_ln},${retrieval_end_ln}p" "$GANG_SH")"
  ok "G5: extracted the artifact-retrieval + id-secrecy block from the driver (lines ${retrieval_start_ln}-${retrieval_end_ln})"
else
  bad "G5: could not locate the artifact-retrieval + id-secrecy block in the driver (anchors moved or were deleted)"
  retrieval_body=""
fi

RSYNC_BIN="$SANDBOX/rsync-bin"
mkdir -p "$RSYNC_BIN"

run_retrieval() { # $1=initial rc  $2=rsync exit code  $3=artifact content(none|clean|leak)  $4=log content(clean|leak) -> "rc|stderr"
  cat >"$RSYNC_BIN/rsync" <<RS
#!/usr/bin/env bash
exit $2
RS
  chmod +x "$RSYNC_BIN/rsync"

  local rc="$1"
  local GANG_ARTIFACT_DIR="$SANDBOX/g5-pulled"
  rm -rf "$GANG_ARTIFACT_DIR"; mkdir -p "$GANG_ARTIFACT_DIR"
  case "$3" in
    clean) echo '{"world":2}' > "$GANG_ARTIFACT_DIR/gang.json" ;;
    leak) python3 -c "print('a' * 256)" > "$GANG_ARTIFACT_DIR/gang.json" ;;
    none) : ;;
  esac
  local LOG="$SANDBOX/g5-retrieval.log"
  case "$4" in
    clean) echo "ordinary run output, no secrets" > "$LOG" ;;
    leak) python3 -c "print('b' * 256)" > "$LOG" ;;
  esac

  # shellcheck disable=SC2034  # read only inside the eval'd retrieval_body below, which shellcheck cannot see into
  local RP_HOST="203.0.113.1" RP_PORT="22"
  # shellcheck disable=SC2034  # same: read only inside the eval'd retrieval_body
  local RP_SSHO=(-o StrictHostKeyChecking=no)
  # shellcheck disable=SC2034  # same: read only inside the eval'd retrieval_body
  local GANG_REMOTE_ARTIFACT_DIR="/root/jammi-ai/.gang-artifact"
  local old_path="$PATH"
  PATH="$RSYNC_BIN:$PATH"
  eval "$retrieval_body" > "$SANDBOX/g5.out" 2>&1
  PATH="$old_path"
  printf '%s|%s' "$rc" "$(cat "$SANDBOX/g5.out")"
}

res="$(run_retrieval 0 0 clean clean)"
if [ "${res%%|*}" -eq 0 ] && [[ "${res#*|}" == *"pulled the gang artifact"* ]] && [[ "${res#*|}" != *"::error::"* ]]; then
  ok "G5: a clean pull with a clean artifact/log stays rc=0, no error"
else
  bad "G5: expected a silent rc=0 pull; got $res"
fi

res="$(run_retrieval 0 17 none clean)"
if [ "${res%%|*}" -eq 17 ] && [[ "${res#*|}" == *"gang artifact pull failed"* ]]; then
  ok "G5: a FAILED pull (rsync rc=17) with rc=0 so far joins the leg's own rc -> 17, never a silent warning"
else
  bad "G5: expected rc=17 + 'gang artifact pull failed'; got $res"
fi

res="$(run_retrieval 5 17 none clean)"
if [ "${res%%|*}" -eq 5 ] && [[ "${res#*|}" == *"gang artifact pull failed"* ]]; then
  ok "G5: a failed pull on an ALREADY-failing leg (rc=5) reports the pull failure but never overwrites the leg's own rc"
else
  bad "G5: expected rc=5 (unchanged) + the pull-failed message; got $res"
fi

res="$(run_retrieval 0 0 leak clean)"
if [ "${res%%|*}" -eq 1 ] && [[ "${res#*|}" == *"hex-encoded NCCL id"* ]]; then
  ok "G5: a 256-hex-char run in the PULLED ARTIFACT dir fails the leg (rc=1), naming the id-secrecy reason"
else
  bad "G5: expected rc=1 + the id-secrecy message from a leaked artifact; got $res"
fi

res="$(run_retrieval 0 0 clean leak)"
if [ "${res%%|*}" -eq 1 ] && [[ "${res#*|}" == *"hex-encoded NCCL id"* ]]; then
  ok "G5: a 256-hex-char run in the RUN'S OWN LOG fails the leg (rc=1), naming the id-secrecy reason"
else
  bad "G5: expected rc=1 + the id-secrecy message from a leaked log; got $res"
fi

res="$(run_retrieval 0 0 clean clean)"
if [ "${res%%|*}" -eq 0 ] && [[ "${res#*|}" != *"hex-encoded"* ]]; then
  ok "G5: a clean artifact and log never trip the id-secrecy scan (not blanket-refusing)"
else
  bad "G5: expected the id-secrecy scan to stay silent on clean input; got $res"
fi

# ============================================================================
# G6: the two JAMMI_REQUIRE_* exports (U7b-A1-pull P3) are present in the
# expanded <<REMOTE heredoc body, beside the existing env block.
# ============================================================================
if [[ "$remote_text" == *"export JAMMI_REQUIRE_CUDA=1"* ]] && [[ "$remote_text" == *"export JAMMI_REQUIRE_CUDA_GANG=1"* ]]; then
  ok "G6: the expanded remote heredoc exports both JAMMI_REQUIRE_CUDA=1 and JAMMI_REQUIRE_CUDA_GANG=1"
else
  bad "G6: the expanded remote heredoc is missing one or both JAMMI_REQUIRE_* exports"
fi

require_env_mutation="$(printf '%s\n' "$remote_text" | grep -c '^export JAMMI_REQUIRE_CUDA')"
if [ "$require_env_mutation" -eq 2 ]; then
  ok "G6: exactly two JAMMI_REQUIRE_CUDA* export lines (removing either is the mutation this case catches)"
else
  bad "G6: expected exactly 2 JAMMI_REQUIRE_CUDA* export lines; found ${require_env_mutation}"
fi

# ============================================================================
# G7: NO schedule: key anywhere in the committed gpu-gang.yml (U7b-A1-pull
# P5). This row is itself deleted the moment U7b-A3 re-adds the schedule
# block with its own never-vacuous writer, in the same diff as that writer —
# a permanent "no schedule" assertion would then reject the correct end
# state this repo is meant to reach.
# ============================================================================
if grep -n '^\s*schedule:' "$GANG_YML" >/dev/null 2>&1; then
  bad "G7: gpu-gang.yml carries a schedule: key -- between U7b-A1-pull's merge and U7b-A3's re-add, NO cron of any kind may fire this paid two-GPU lane"
else
  ok "G7: gpu-gang.yml carries no schedule: key (hermetic, source-text check)"
fi

# ============================================================================
# G4: the cost bound is what the mechanism produces.
# ============================================================================
# Every term is read back OUT of the committed artifacts, never retyped:
#   candidates  <- runpod_lib.sh's own `a100)` candidate list
#   wait/ttl    <- the values the sourced driver actually exports
#   attempts    <- gpu-gang.yml's MAX_ATTEMPTS
#   rate        <- GANG_RATE_USD_PER_HOUR below (spike S4's measured SECURE
#                  2-GPU A100-SXM4-80GB rate; the one term with no producer
#                  in this tree, so it is named here and in all three prose
#                  sites, and the COMMUNITY rate is stated as unmeasured)
GANG_RATE_USD_PER_HOUR=3.18

a100_candidates="$(grep -oE '^\s*a100\)\s*cand=\([^)]*\)' "$LIB_SH" | grep -oE '"[^"]+"' | grep -c .)"
if [ "$a100_candidates" -ge 1 ]; then
  ok "G4: derived the a100 candidate count from runpod_lib.sh: ${a100_candidates}"
else
  bad "G4: could not derive the a100 candidate count from runpod_lib.sh"
fi

max_attempts="$(grep -oE 'MAX_ATTEMPTS:[[:space:]]*"[0-9]+"' "$GANG_YML" | grep -oE '[0-9]+' | head -1)"
if [ "$max_attempts" = "1" ]; then
  ok "G4: gpu-gang.yml pins MAX_ATTEMPTS to \"1\" (one deploy search per run — the bound is stated for one)"
else
  bad "G4: gpu-gang.yml's MAX_ATTEMPTS is '${max_attempts:-<absent>}', not \"1\" — bound (i) no longer holds"
fi

if grep -qE '^export RP_SSH_WAIT_SECS="\$\{RP_SSH_WAIT_SECS:-[0-9]+\}"' "$GANG_SH"; then
  ok "G4: the driver exports its own lane-local RP_SSH_WAIT_SECS default (RP_SSH_WAIT_SECS=${RP_SSH_WAIT_SECS})"
else
  bad "G4: the driver has no lane-local 'export RP_SSH_WAIT_SECS=\${RP_SSH_WAIT_SECS:-N}' line"
fi

if [ "$RP_TTL_HOURS" = "1" ]; then
  ok "G4: the driver pins RP_TTL_HOURS=1 (the pod's own entrypoint deadline)"
else
  bad "G4: RP_TTL_HOURS is '${RP_TTL_HOURS}', not 1 — the TTL term of both bounds moved"
fi

# The rate is a PER-POD rate measured at a GPU COUNT: every figure below
# multiplies hours by $3.18/h, and that price was read off a SECURE 2-GPU
# A100-SXM4-80GB pod (S4). A lane that quietly rented 4 GPUs would keep
# printing the same bound while billing something else, so the count is read
# back OUT of the sourced driver — the same variable the shared deploy
# payload reads — and pinned here. This suite ASSERTS the count; it does not
# set one (test_pod_substrate.sh's `(ab/gpuCount D5)` closed set records why
# this file is allowed to name the variable at all).
if [ "$RP_GPU_COUNT" = "2" ]; then
  ok "G4: the driver pins RP_GPU_COUNT=2 — the pod shape the \$${GANG_RATE_USD_PER_HOUR}/h rate was measured at"
else
  bad "G4: RP_GPU_COUNT is '${RP_GPU_COUNT}', not 2 — every bound here is priced at a rate measured for a 2-GPU pod, so the figures no longer price this lane"
fi

read -r bound_i bound_ii runner_minutes < <(python3 - \
  "$a100_candidates" "$RP_SSH_WAIT_SECS" "$RP_TTL_HOURS" "$max_attempts" "$GANG_RATE_USD_PER_HOUR" <<'PY'
import sys
cands, wait_s, ttl_h, attempts, rate = (
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
)
searches = cands * attempts
# (i) every rp_terminate takes: each abandoned candidate bills for its own
# reachability wait, then the winning pod bills to its TTL.
bound_i = searches * wait_s / 3600.0 * rate + ttl_h * rate
# (ii) every rp_terminate FAILS: nothing is torn down early, so each pod
# billed — the abandoned ones and the winner — runs to its baked-in TTL.
bound_ii = (searches + 1) * ttl_h * rate
# The runner's own budget, from the same terms: the library RP_TIMEOUT
# default (50m) + the deploy worst case + 10m of runner overhead.
runner = 50 + searches * wait_s // 60 + 10
print(f"{bound_i:.2f} {bound_ii:.2f} {runner}")
PY
)

if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 4.25 else 1)" "$bound_i"; then
  ok "G4: bound (i) = \$${bound_i} per run, within the approved \$4.25 ceiling"
else
  bad "G4: bound (i) = \$${bound_i} per run, ABOVE the approved \$4.25 ceiling — a term was raised"
fi

for site in "$GANG_YML" "$GANG_SH" "$DEV_GPU_MD"; do
  rel="${site#"$REPO_ROOT"/}"
  if grep -qF "\$${bound_i}" "$site"; then
    ok "G4: ${rel} states bound (i) as \$${bound_i} — the figure the mechanism produces"
  else
    bad "G4: ${rel} does not state bound (i) (\$${bound_i}) — its prose and the mechanism disagree"
  fi
  if grep -qF "\$${bound_ii}" "$site"; then
    ok "G4: ${rel} states bound (ii) as \$${bound_ii} (the sweep-only path)"
  else
    bad "G4: ${rel} does not state bound (ii) (\$${bound_ii})"
  fi
  if grep -qF "\$${GANG_RATE_USD_PER_HOUR}/h" "$site"; then
    ok "G4: ${rel} names the \$${GANG_RATE_USD_PER_HOUR}/h rate both bounds are computed at"
  else
    bad "G4: ${rel} does not name the \$${GANG_RATE_USD_PER_HOUR}/h rate"
  fi
  if grep -qiE 'COMMUNITY[^.]*(unmeasured|not been (priced|measured)|nothing has priced)' "$site"; then
    ok "G4: ${rel} states the COMMUNITY 2-GPU rate is unmeasured (no bound is claimed for it)"
  else
    bad "G4: ${rel} does not state that the COMMUNITY 2-GPU rate is unmeasured"
  fi
done

yml_timeout="$(grep -oE '^\s*timeout-minutes:\s*[0-9]+' "$GANG_YML" | grep -oE '[0-9]+' | head -1)"
if [ "$yml_timeout" = "$runner_minutes" ]; then
  ok "G4: gpu-gang.yml's timeout-minutes (${yml_timeout}) is what the same terms produce"
else
  bad "G4: gpu-gang.yml's timeout-minutes is ${yml_timeout:-<absent>}, but the terms produce ${runner_minutes}"
fi

# The shared candidate list must NOT be narrowed to buy the bound: a
# one-day PCIe capacity observation reorders the search (SXM4 first above
# gpuCount 1), it never drops a candidate.
if grep -qE '^\s*a100\)\s*cand=\(.*A100 80GB PCIe.*A100-SXM4-80GB' "$LIB_SH" \
  && [ "$a100_candidates" -eq 4 ]; then
  ok "G4: the shared a100 candidate list still carries all four PCIe+SXM4 entries (reordered above gpuCount 1, never narrowed)"
else
  bad "G4: the a100 candidate list was narrowed (${a100_candidates} entries) — a multi-GPU rental reorders the search, it does not drop candidates"
fi

echo
echo "gpu-gang-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
