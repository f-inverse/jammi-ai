#!/usr/bin/env bash
# GPU-prove-lane fixture suite (esc-080/esc-082/esc-083). Mocks-only, no
# network, no GPU, no RunPod account: drives the REAL `rp_run_remote_watched`
# (runpod_lib.sh) and the REAL `rp_prove_verdict`/`PROVE_GROUPS`
# (runpod_gpu_prove.sh, `source`d rather than executed — see that file's own
# "sourced-execution guard" comment, which skips the deploy-a-pod flow when
# sourced) against a STUBBED `ssh`.
#
# Cases (contract esc-080..083, D6):
#   F1  inactivity kill -> 76, named group, `[name:rc…]` list.
#   F2  budget cut (ssh 124, no PROVE_EXIT) -> 124, `[name:rc…]` list;
#       N=0 groups -> `[]`.
#   F3  in-suite exits pass through VERBATIM: 255; in-suite 124 WITH
#       `PROVE_EXIT=124` (no BUDGET line); in-suite 76; an early `exit 97`
#       with no markers at all.
#   F4  bench is non-gating: bench exit 1 with every proof group rc=0 -> 0 +
#       `BENCH_EXIT=1`; a served-proof failure followed by a HUNG bench ->
#       FAIL; a served-proof failure with a normal exit -> FAIL; a cut
#       INSIDE a proof group -> FAIL; a cut inside bench with every proof
#       group already rc=0 -> 0 + `::warning::`; a marker written after the
#       final poll tick still credits.
#   F7  `rp_cleanup` (the `trap ... EXIT` runpod_lib.sh installs at source
#       time) fires on the 76 path and on the exit-0 bench-cut path.
#   Plus: the `PROVE_GROUPS` closure fixture (every `::group::` name in the
#   script minus `device`/`bench` equals `PROVE_GROUPS`, exactly);
#   partial-line-at-poll-boundary (a marker split across two 5s polls is
#   still parsed); `PROVE_EXIT` disagreeing with its own markers -> FAIL.
#
# F5 (`check_flash_attn_closure.py --self-test`) and F6
# (`check_gpu_prove_timings.py --self-test`) are NOT duplicated here — they
# are already required, standalone steps in `ci.yml`'s guard matrix; this
# suite owns only the driver/watchdog mechanism itself.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
PROVE_SH="$DIR/runpod_gpu_prove.sh"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

# --------------------------------------------------------------------------
# Source the real driver (no network call merely from sourcing: the
# sourced-execution guard skips rp_sweep/rp_init/rp_deploy_arch/the heredoc).
# --------------------------------------------------------------------------
export RUNPOD_API_KEY="test-dummy-key"
export GPU_PROVE_ARCH="sm_80"
# shellcheck source=ci/scripts/runpod_gpu_prove.sh
source "$PROVE_SH"

if declare -f rp_prove_verdict >/dev/null && declare -f rp_run_remote_watched >/dev/null; then
  ok "sourcing runpod_gpu_prove.sh (no network) defines rp_prove_verdict and (via runpod_lib.sh) rp_run_remote_watched"
else
  bad "sourcing runpod_gpu_prove.sh did not define the expected functions"
fi

# --------------------------------------------------------------------------
# PROVE_GROUPS closure: the script's own `::group::` names, minus device and
# bench, must equal PROVE_GROUPS exactly.
# --------------------------------------------------------------------------
mapfile -t script_groups < <(grep -oE '::group::[a-z0-9-]+' "$PROVE_SH" | sed 's/::group:://' | sort -u)
mapfile -t declared_groups < <(printf '%s\n' "${PROVE_GROUPS[@]}" | sort -u)
non_members=()
for g in "${script_groups[@]}"; do
  [ "$g" = "device" ] && continue
  [ "$g" = "bench" ] && continue
  non_members+=("$g")
done
mapfile -t non_members_sorted < <(printf '%s\n' "${non_members[@]}" | sort -u)
if [ "${#non_members_sorted[@]}" -eq "${#declared_groups[@]}" ] && [ "$(printf '%s\n' "${non_members_sorted[@]}")" = "$(printf '%s\n' "${declared_groups[@]}")" ]; then
  ok "PROVE_GROUPS closure: {::group:: names} - {device, bench} == PROVE_GROUPS exactly (${#declared_groups[@]} members)"
else
  bad "PROVE_GROUPS closure mismatch: script non-member groups=[${non_members_sorted[*]}] vs PROVE_GROUPS=[${declared_groups[*]}]"
fi

# --------------------------------------------------------------------------
# helper: build a log fixture from a marker spec, run rp_prove_verdict, and
# report the resulting rc plus stderr.
# --------------------------------------------------------------------------
all_six_pass_log() {
  local log="$1"
  {
    for g in "${PROVE_GROUPS[@]}"; do
      echo "PROVE_GROUP_RC name=${g} rc=0"
    done
  } > "$log"
}

# ============================================================================
# F1: rp_run_remote_watched itself — inactivity kill -> 76, named group,
# [name:rc...] list (real function, stubbed ssh).
# ============================================================================
ssh() {
  cat >/dev/null
  echo "::group::served-client-server-proof"
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  sleep 30
  return 0
}
export -f ssh
RP_SSHO=(); RP_PORT=22; RP_HOST=localhost
# A fixture-local threshold passed as a plain function ARGUMENT (never a
# committed `RP_INACTIVITY=<n>` assignment, which check_gpu_prove_timings.
# py's R1 setter-predicate scan would -- correctly -- flag as a second
# source of truth for the real default) -- see rp_run_remote_watched's own
# `$1` doc.
out="$(rp_run_remote_watched 3 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] \
  && echo "$out" | grep -q 'NO PROGRESS' \
  && echo "$out" | grep -q 'group "served-client-server-proof"' \
  && echo "$out" | grep -q 'groups: \[capability-surface-build:0\]'; then
  ok "F1: inactivity kill -> 76, names the open group, lists [name:rc...]"
else
  bad "F1: expected 76 + named group + list; got rc=$rc out=$out"
fi

# ============================================================================
# F2: budget cut (ssh 124, no PROVE_EXIT) -> 124 with list; N=0 -> [].
# ============================================================================
ssh() {
  cat >/dev/null
  echo "::group::kernels-cuda"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'BUDGET' && echo "$out" | grep -q 'groups: \[kernels-default:0\]'; then
  ok "F2: budget cut -> 124, lists [name:rc...]"
else
  bad "F2: expected 124 + list; got rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "no groups at all"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'groups: \[\]'; then
  ok "F2: N=0 groups renders as []"
else
  bad "F2: expected groups: []; got rc=$rc out=$out"
fi

# ============================================================================
# F3: in-suite exits pass through VERBATIM.
# ============================================================================
ssh() { cat >/dev/null; return 255; }
export -f ssh
rp_run_remote_watched <<< "noop" >/dev/null 2>&1
[ $? -eq 255 ] && ok "F3: ssh 255 passes through verbatim" || bad "F3: expected 255"

ssh() {
  cat >/dev/null
  echo "::group::bench"
  echo "PROVE_EXIT=124"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && ! echo "$out" | grep -q 'BUDGET'; then
  ok "F3: in-suite 124 (PROVE_EXIT=124 present) -> 124 verbatim, no BUDGET line"
else
  bad "F3: expected 124 with no BUDGET line; got rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "PROVE_EXIT=76"
  return 76
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && ! echo "$out" | grep -q 'NO PROGRESS'; then
  ok "F3: in-suite 76 (PROVE_EXIT=76 present) passes through unchanged, no watchdog diagnostic"
else
  bad "F3: expected in-suite 76 unchanged; got rc=$rc out=$out"
fi

ssh() { cat >/dev/null; return 97; }
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 97 ] && ! echo "$out" | grep -qE 'NO PROGRESS|BUDGET'; then
  ok "F3: an early exit 97 with no markers passes through verbatim"
else
  bad "F3: expected 97 verbatim with no diagnostic; got rc=$rc out=$out"
fi

# ============================================================================
# F4: bench is non-gating (driven through rp_prove_verdict directly).
# ============================================================================
log="$SANDBOX/f4a.log"
{
  for g in "${PROVE_GROUPS[@]}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
  echo "BENCH_EXIT=1"
  echo "PROVE_GROUP_RC name=bench rc=1"
  echo "PROVE_EXIT=0"
} > "$log"
rp_prove_verdict 0 "$log"
rc=$?
[ "$rc" -eq 0 ] && ok "F4: bench exit 1 with every proof group rc=0 -> 0 (BENCH_EXIT=1 is non-gating)" \
  || bad "F4: expected 0, got $rc"

log="$SANDBOX/f4b.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  echo "PROVE_GROUP_RC name=capability-surface-proof rc=0"
  echo "PROVE_GROUP_RC name=served-client-server-proof rc=1"
  echo "PROVE_GROUP_RC name=engine-core-sweep rc=0"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  echo "PROVE_GROUP_RC name=kernels-cuda rc=0"
  # bench then hangs -- no BENCH_EXIT, no PROVE_EXIT line at all.
} > "$log"
rp_prove_verdict 76 "$log"   # the watchdog would have killed the hung bench
rc=$?
[ "$rc" -eq 76 ] && ok "F4: served-proof failure (rc=1) then a hung bench -> FAIL (not the bench exception, since a REAL proof group failed)" \
  || bad "F4: expected 76 (FAIL), got $rc"

log="$SANDBOX/f4c.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  echo "PROVE_GROUP_RC name=capability-surface-proof rc=0"
  echo "PROVE_GROUP_RC name=served-client-server-proof rc=1"
  echo "PROVE_GROUP_RC name=engine-core-sweep rc=0"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  echo "PROVE_GROUP_RC name=kernels-cuda rc=0"
  echo "BENCH_EXIT=0"
  echo "PROVE_GROUP_RC name=bench rc=0"
  echo "PROVE_EXIT=1"
} > "$log"
rp_prove_verdict 1 "$log"
rc=$?
[ "$rc" -eq 1 ] && ok "F4: served-proof failure (rc=1) with a NORMAL exit -> FAIL, verbatim" \
  || bad "F4: expected 1, got $rc"

log="$SANDBOX/f4d.log"
{
  echo "PROVE_GROUP_RC name=capability-surface-build rc=0"
  # cut happens INSIDE capability-surface-proof -- its own marker never lands.
} > "$log"
rp_prove_verdict 124 "$log"
rc=$?
[ "$rc" -eq 124 ] && ok "F4: a cut INSIDE a proof group (marker missing) -> FAIL" \
  || bad "F4: expected 124, got $rc"

log="$SANDBOX/f4e.log"
all_six_pass_log "$log"
# cut happens inside bench -- no BENCH_EXIT, no PROVE_EXIT.
rp_prove_verdict 76 "$log"
rc=$?
out="$(rp_prove_verdict 76 "$log" 2>&1 >/dev/null)"
if [ "$rc" -eq 0 ] && echo "$out" | grep -q '::warning::'; then
  ok "F4: a cut INSIDE bench with every proof group already rc=0 -> 0 + ::warning::"
else
  bad "F4: expected 0 + warning, got rc=$rc out=$out"
fi

# A marker written after the final poll tick still credits: rp_prove_verdict
# only ever reads the FINAL drained log file, so any marker present in it —
# regardless of when it landed — is honored. Simulate by writing the LAST
# group's marker only after constructing the rest of the log (order in the
# file, not wall-clock time, is what a static read sees; the wall-clock
# guarantee itself is exercised by rp_run_remote_watched's own final-drain
# step, proven directly below).
log="$SANDBOX/f4f.log"
: > "$log"
for g in "${PROVE_GROUPS[@]:0:5}"; do echo "PROVE_GROUP_RC name=${g} rc=0" >> "$log"; done
echo "PROVE_GROUP_RC name=${PROVE_GROUPS[5]} rc=0" >> "$log"   # the "late" marker
echo "PROVE_EXIT=0" >> "$log"
rp_prove_verdict 0 "$log"
[ $? -eq 0 ] && ok "F4: a marker landing last in the log still credits" || bad "F4: late marker did not credit"

# ============================================================================
# partial-line-at-poll-boundary: a marker split across two 5s polls is still
# reassembled by rp_run_remote_watched's own carry logic.
# ============================================================================
ssh() {
  cat >/dev/null
  printf 'PROVE_GROUP_RC name=engine'
  sleep 6
  printf -- '-core-sweep rc=0\n'
  sleep 30
  return 0
}
export -f ssh
out="$(rp_run_remote_watched 6 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && echo "$out" | grep -q 'groups: \[engine-core-sweep:0\]'; then
  ok "partial-line-at-poll-boundary: a marker split across two polls is reassembled correctly"
else
  bad "partial-line-at-poll-boundary: expected engine-core-sweep:0 in the group list; got rc=$rc out=$out"
fi

# ============================================================================
# BLOCK 2 audit fix: the FINAL drain's own carry handling (distinct from the
# poll-boundary case above, which never reaches EOF mid-marker) -- an
# unterminated last `::group::` line, and separately an unterminated last
# `PROVE_GROUP_RC` marker, both landing right at ssh's own (non-inactivity)
# exit. `_rrw_scan_new_text` prepends `parse_carry` to its OWN `$1`
# internally; the final-drain caller must pass ONLY the missing newline
# (`$'\n'`), never `"$parse_carry"$'\n'` again -- doubling it once produced
# `cut group "kernels-cuda::group::kernels-cuda"` live.
# ============================================================================
ssh() {
  cat >/dev/null
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  printf '::group::kernels-cuda'   # no trailing newline -- ssh exits right here
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] \
  && echo "$out" | grep -q 'group "kernels-cuda"' \
  && ! echo "$out" | grep -q 'kernels-cuda::group::kernels-cuda' \
  && echo "$out" | grep -q 'groups: \[kernels-default:0\]'; then
  ok "final-drain carry: an unterminated last ::group:: line names the group once, never doubled"
else
  bad "final-drain carry: unterminated ::group:: line mishandled; rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "::group::kernels-cuda"
  printf 'PROVE_GROUP_RC name=kernels-cuda rc=0'   # no trailing newline
  return 124
}
export -f ssh
out="$(rp_run_remote_watched <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'groups: \[kernels-cuda:0\]'; then
  ok "final-drain carry: an unterminated last PROVE_GROUP_RC marker is still parsed (not dropped, not doubled)"
else
  bad "final-drain carry: unterminated PROVE_GROUP_RC marker mishandled; rc=$rc out=$out"
fi

# ============================================================================
# BLOCK 3 audit fix -- cross-parser fixture: the SAME unterminated-final-
# marker log content, fed independently to (a) rp_run_remote_watched
# (already exercised above), (b) rp_prove_verdict (the log-file path), and
# (c) the Python producer's parse_log/PROVE_GROUP_RC_RE -- all three must
# extract the identical (name, rc) pair, never drop/double it.
# ============================================================================

# (b) rp_prove_verdict: all six groups pass, the LAST one's marker is
# unterminated (no trailing newline in the log FILE at all) and no
# PROVE_EXIT was reached (a genuine cut) -- if the parser dropped the
# unterminated marker, all_proof_pass would read 0 and the bench-cut
# exception would NOT apply, leaving rc=124 instead of 0.
xp_log="$SANDBOX/xparser.log"
{
  for g in "${PROVE_GROUPS[@]:0:5}"; do echo "PROVE_GROUP_RC name=${g} rc=0"; done
} > "$xp_log"
printf 'PROVE_GROUP_RC name=%s rc=0' "${PROVE_GROUPS[5]}" >> "$xp_log"   # no trailing newline
rp_prove_verdict 124 "$xp_log"
xp_bash_rc=$?
if [ "$xp_bash_rc" -eq 0 ]; then
  ok "cross-parser (bash, rp_prove_verdict): unterminated final marker credited (all six groups pass -> bench-cut exception -> 0)"
else
  bad "cross-parser (bash, rp_prove_verdict): unterminated final marker NOT credited (rc=$xp_bash_rc, expected 0)"
fi

# (a)+(c) SAME literal marker line text (byte-identical, no trailing
# newline), fed to the bash-side shared parser (rp_parse_prove_marker,
# used by both rp_run_remote_watched and rp_prove_verdict) and to the
# Python-side prove_surface.PROVE_GROUP_RC_RE (imported, never a second
# regex, by gpu_prove_timings.py) directly -- both must extract the
# IDENTICAL (name, rc) pair from the identical text.
xp_marker_line="PROVE_GROUP_RC name=${PROVE_GROUPS[5]} rc=0"
if rp_parse_prove_marker "$xp_marker_line"; then
  xp_bash_name="$RP_PARSED_MARKER_NAME"
  xp_bash_rcval="$RP_PARSED_MARKER_RC"
else
  xp_bash_name=""
  xp_bash_rcval=""
fi
xp_py_pair="$(python3 -c "
import sys
sys.path.insert(0, 'ci/scripts')
import prove_surface
m = prove_surface.PROVE_GROUP_RC_RE.search('$xp_marker_line')
print(m.group('name') + ' ' + m.group('rc') if m else 'NOMATCH')
")"
if [ "$xp_bash_name $xp_bash_rcval" = "$xp_py_pair" ] && [ "$xp_py_pair" != "NOMATCH" ]; then
  ok "cross-parser: bash rp_parse_prove_marker and Python prove_surface.PROVE_GROUP_RC_RE extract the IDENTICAL (name, rc) from the same marker text ($xp_py_pair)"
else
  bad "cross-parser: bash extracted '$xp_bash_name $xp_bash_rcval' but python extracted '$xp_py_pair' -- grammars disagree"
fi

# (c) the same marker text, embedded in a minimal GH-raw-log-shaped
# synthetic log with NO trailing newline on the file's last line, must
# resolve via the producer's OWN import of that same constant.
xp_py_log="$SANDBOX/xparser_gh.log"
{
  printf 'GPU prove on RunPod (sm_80)\tUNKNOWN STEP\t2026-01-01T00:00:00.0000000Z ##[group]%s\n' "${PROVE_GROUPS[5]}"
  printf 'GPU prove on RunPod (sm_80)\tUNKNOWN STEP\t2026-01-01T00:00:01.0000000Z %s' "$xp_marker_line"
} > "$xp_py_log"
xp_py_out="$(python3 -c "
import sys
sys.path.insert(0, 'ci/scripts/perf')
import gpu_prove_timings as g
text = open('$xp_py_log').read()
print('MATCH' if any(m.group('name') == '${PROVE_GROUPS[5]}' and m.group('rc') == '0' for m in g._GROUP_RC_RE.finditer(text)) else 'NOMATCH')
")"
if [ "$xp_py_out" = "MATCH" ]; then
  ok "cross-parser (python producer, via its OWN imported PROVE_GROUP_RC_RE): the unterminated marker text extracts name=${PROVE_GROUPS[5]} rc=0"
else
  bad "cross-parser (python producer): expected a MATCH on the unterminated marker; got $xp_py_out"
fi

# ============================================================================
# PROVE_EXIT disagreeing with its own markers -> FAIL (never trusted blindly).
# ============================================================================
log="$SANDBOX/disagree.log"
all_six_pass_log "$log"
# Corrupt ONE marker after the fact so the file disagrees with a PROVE_EXIT=0.
sed -i.bak 's/name=kernels-cuda rc=0/name=kernels-cuda rc=1/' "$log"
echo "PROVE_EXIT=0" >> "$log"
rp_prove_verdict 0 "$log"
rc=$?
[ "$rc" -ne 0 ] && ok "PROVE_EXIT disagreeing with its own markers -> FAIL (rc=$rc)" \
  || bad "expected a nonzero rc when PROVE_EXIT=0 disagrees with a failed marker"

# ============================================================================
# F7 (BLOCK audit fix): drives the REAL, EXECUTED runpod_gpu_prove.sh top-
# level exit path (rp_sweep -> rp_init -> rp_deploy_arch -> rp_run_remote_
# watched -> rp_prove_verdict -> exit) end to end, in a SEPARATE bash
# process, with `curl`/`ssh` stubbed (never a bare function override --
# sourcing runpod_lib.sh would clobber a re-defined bash function, but it
# never touches the real `curl`/`ssh` EXTERNAL BINARIES, so shadowing them
# on PATH survives the source). `rp_cleanup` itself is NEVER stubbed (it
# cannot be, for the same clobber reason) -- its call is recorded
# INDIRECTLY but unambiguously: `rp_terminate` (which `rp_cleanup` calls
# for every pod it created) makes a REAL `podTerminate` GraphQL call
# through `curl`, and the stub below writes a marker file the moment it
# sees one. No `trap rp_cleanup EXIT` firing means no `podTerminate` call
# means no marker -- this fails against a scratch copy with that trap
# commented out (verified by hand while writing this fixture).
# ============================================================================
F7_STUBBIN="$SANDBOX/f7-bin"
mkdir -p "$F7_STUBBIN"

cat > "$F7_STUBBIN/curl" <<'CURLSTUB'
#!/usr/bin/env bash
payload=""
for a in "$@"; do
  case "$a" in
    *podTerminate*|*podFindAndDeployOnDemand*|*'myself{'*|*'pod(input:'*) payload="$a" ;;
  esac
done
case "$payload" in
  *podTerminate*)
    echo "TERMINATED" >> "${F7_TERMINATE_MARKER:?F7_TERMINATE_MARKER unset}"
    echo '{"data":{"podTerminate":true}}'
    ;;
  *podFindAndDeployOnDemand*) echo '{"data":{"podFindAndDeployOnDemand":{"id":"f7-fake-pod"}}}' ;;
  *'myself{'*) echo '{"data":{"myself":{"pods":[]}}}' ;;
  *'pod(input:'*) echo '{"data":{"pod":{"runtime":{"ports":[{"ip":"127.0.0.1","publicPort":2222,"privatePort":22,"isIpPublic":true,"type":"tcp"}]}}}}' ;;
  *) echo '{}' ;;
esac
CURLSTUB
chmod +x "$F7_STUBBIN/curl"

# Distinguishes the THREE real ssh invocation shapes rp_deploy_live/
# rp_run_remote_watched make, by the LAST argument's own content: a bare
# `true` (the reachability liveness probe), an `nvidia-smi` driver query,
# or the heredoc's `timeout N bash -s` remote-script form (the ONLY one
# that reads stdin and needs a scenario-specific reply).
cat > "$F7_STUBBIN/ssh" <<'SSHSTUB'
#!/usr/bin/env bash
last="${@: -1}"
case "$last" in
  true) exit 0 ;;
  *nvidia-smi*) echo "570.195.03"; exit 0 ;;
  *"bash -s"*)
    cat >/dev/null
    if [ "${F7_SCENARIO:-}" = "watchdog" ]; then
      echo "::group::capability-surface-build"
      sleep 30
    else
      for g in capability-surface-build capability-surface-proof served-client-server-proof engine-core-sweep kernels-default kernels-cuda; do
        echo "PROVE_GROUP_RC name=${g} rc=0"
      done
      echo "::group::bench"
      sleep 30
    fi
    ;;
  *) exit 0 ;;
esac
SSHSTUB
chmod +x "$F7_STUBBIN/ssh"

f7_case() {
  # $1 = "watchdog" | "bench-cut"; $2 = expected exit code.
  local mode="$1" want_rc="$2"
  local marker="$SANDBOX/f7-terminate-marker-$mode"
  rm -f "$marker"
  # A single `env ...` line (never a `VAR=val \`-per-line assignment chain):
  # check_gpu_prove_timings.py's own R1 setter-predicate scan matches
  # `^\s*(export\s+)?RP_(TIMEOUT|INACTIVITY)=` at the START of a physical
  # line, which a bare `RP_INACTIVITY=3 \` continuation line would satisfy
  # -- this is a plain per-invocation env-var override, not a second
  # committed default, and must not be misread as one.
  env RUNPOD_API_KEY=test-dummy-key PATH="$F7_STUBBIN:$PATH" F7_TERMINATE_MARKER="$marker" F7_SCENARIO="$mode" GPU_PROVE_ARCH=sm_80 RP_INACTIVITY=3 RP_SSH_WAIT_SECS=10 bash "$PROVE_SH" > "$SANDBOX/f7-$mode.out" 2>&1
  local rc=$?
  if [ "$rc" -eq "$want_rc" ] && [ -f "$marker" ]; then
    ok "F7 ($mode): the real executed exit path returns $want_rc AND rp_cleanup's own podTerminate call fired (marker recorded)"
  else
    bad "F7 ($mode): expected rc=$want_rc with a recorded podTerminate call; got rc=$rc marker-present=$([ -f "$marker" ] && echo yes || echo no); out=$(cat "$SANDBOX/f7-$mode.out")"
  fi
}
# Every earlier F1-F4/cross-parser case above did `export -f ssh` with a
# succession of bash FUNCTION overrides -- an exported bash function takes
# PRECEDENCE over a same-named PATH executable in any child process, so
# without unsetting it here the f7_case subprocess below would inherit the
# LAST test's own `ssh` function instead of ever reaching $F7_STUBBIN/ssh
# on PATH.
unset -f ssh
f7_case "watchdog" 76
f7_case "bench-cut" 0

echo
echo "gpu-prove-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
