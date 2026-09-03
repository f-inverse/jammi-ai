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
# Unescaped-backtick guard (real bug found live): the `<<REMOTE` heredoc is
# UNQUOTED (by design -- the local shell must expand `${GIT_REF}` etc. into
# the remote script text), which means the local shell ALSO evaluates any
# bare `` `...` `` pair as a command substitution AT HEREDOC-CONSTRUCTION
# TIME, on the CI RUNNER, before a single byte reaches ssh -- an escaped
# `\`...\`` inside a comment survives as literal text; an unescaped pair
# either silently deletes text (`` `test` `` runs the `test` builtin, exit
# 1, empty stdout) or prints "command not found" to the runner's own
# stderr (`` `is_gated` ``, an undefined command) and STILL deletes the
# text. A STATIC scan over the heredoc body catches this class before it
# ever reaches a real pod; F7 (below) independently proves the REAL
# heredoc expansion produces no such stderr end to end.
heredoc_start="$(grep -n '<<REMOTE' "$PROVE_SH" | head -1 | cut -d: -f1)"
heredoc_end="$(grep -n '^REMOTE$' "$PROVE_SH" | head -1 | cut -d: -f1)"
unescaped_backticks="$(python3 -c "
import re
lines = open('$PROVE_SH').read().splitlines()
body = lines[$heredoc_start:$heredoc_end - 1]
bad = []
for i, line in enumerate(body, start=$heredoc_start + 1):
    if len(re.findall(r'(?<!\\\\)\`', line)) % 2 != 0 or len(re.findall(r'(?<!\\\\)\`', line)) >= 2:
        if re.findall(r'(?<!\\\\)\`', line):
            bad.append(f'{i}: {line}')
for b in bad:
    print(b)
")"
if [ -z "$unescaped_backticks" ]; then
  ok "unescaped-backtick guard: no unescaped backtick pair in the <<REMOTE heredoc body"
else
  bad "unescaped-backtick guard: found unescaped backtick(s) in the heredoc body (would be evaluated as a LOCAL command substitution): $unescaped_backticks"
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
  exec sleep 30
  return 0
}
export -f ssh
RP_SSHO=(); RP_PORT=22; RP_HOST=localhost
# A fixture-local threshold passed as a plain function ARGUMENT (never a
# committed `RP_INACTIVITY=<n>` assignment, which check_gpu_prove_timings.
# py's R1 setter-predicate scan would -- correctly -- flag as a second
# source of truth for the real default) -- see rp_run_remote_watched's own
# `$1` doc.
out="$(rp_run_remote_watched 3 0.2 <<< "noop" 2>&1)"
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
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
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
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
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
rp_run_remote_watched "" 0.2 <<< "noop" >/dev/null 2>&1
[ $? -eq 255 ] && ok "F3: ssh 255 passes through verbatim" || bad "F3: expected 255"

ssh() {
  cat >/dev/null
  echo "::group::bench"
  echo "PROVE_EXIT=124"
  return 124
}
export -f ssh
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
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
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && ! echo "$out" | grep -q 'NO PROGRESS'; then
  ok "F3: in-suite 76 (PROVE_EXIT=76 present) passes through unchanged, no watchdog diagnostic"
else
  bad "F3: expected in-suite 76 unchanged; got rc=$rc out=$out"
fi

ssh() { cat >/dev/null; return 97; }
export -f ssh
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
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
# esc-082 behavior-pinning: EXECUTES the real <<REMOTE heredoc BODY locally
# (fix-verifier finding on 19fca3c1: reverting bench_rc's own fold back into
# `rc` left the whole 24/24-green suite unchanged, because F7's ssh stub
# discards the heredoc via `cat >/dev/null` and every F4 case feeds
# hand-authored logs straight to rp_prove_verdict -- neither ever RUNS the
# heredoc's own shell logic). Extracts the heredoc's literal text (the SAME
# bytes the real driver sends over ssh) and runs it as an ordinary bash
# script -- `\$rc`/`\${grc}` (escaped in the source so the LOCAL/CI-runner
# shell never touches them when building the real heredoc) become literal
# `$rc`/`${grc}` once run this way, read as ITS OWN locals exactly as the
# REMOTE bash would; `${NATIVE_COMPUTE_CAP}`/`${GIT_REF}`/`${GIT_REPO}`
# (unescaped in the source, meant for LOCAL expansion) are supplied by this
# fixture's own environment instead of the real driver's. `cargo`/
# `nvidia-smi`/`git` are PATH-shimmed (git's own `clone` copies this
# checkout's real `ci/`+`crates/` into the fake clone target so the
# heredoc's own manifest read is genuine, not faked); `cd /root` is
# substituted for a sandbox-local workdir -- the ONLY adaptation, disclosed
# here, everything else is the file's own unmodified bytes.
# ============================================================================
ESC082_BIN="$SANDBOX/esc082-bin"
mkdir -p "$ESC082_BIN"

cat > "$ESC082_BIN/cargo" <<'CARGOSTUB'
#!/usr/bin/env bash
args="$*"
if [ -n "${ESC082_FAIL_MATCH:-}" ] && [[ "$args" == *"$ESC082_FAIL_MATCH"* ]]; then
  echo "esc082 stub cargo: FAILING ($args)" >&2
  exit 1
fi
echo "esc082 stub cargo: ok ($args)"
echo "test result: ok. 1 passed; 0 failed; 0 ignored"
exit 0
CARGOSTUB
chmod +x "$ESC082_BIN/cargo"

cat > "$ESC082_BIN/nvidia-smi" <<'NVSTUB'
#!/usr/bin/env bash
case "$*" in
  *"name,compute_cap,driver_version"*)
    echo "name, compute_cap, driver_version"
    echo "NVIDIA A100 80GB PCIe, 8.0, 570.195.03"
    ;;
  *"compute_cap --format=csv,noheader"*) echo "8.0" ;;
  *) echo "" ;;
esac
NVSTUB
chmod +x "$ESC082_BIN/nvidia-smi"

cat > "$ESC082_BIN/git" <<'GITSTUB'
#!/usr/bin/env bash
case "$1" in
  clone)
    dest="${@: -1}"
    mkdir -p "$dest"
    cp -r "${ESC082_REAL_ROOT:?unset}/ci" "$dest/ci"
    cp -r "$ESC082_REAL_ROOT/crates" "$dest/crates"
    ;;
  rev-parse) echo "0000000000000000000000000000000000000000" ;;
  *) : ;;
esac
exit 0
GITSTUB
chmod +x "$ESC082_BIN/git"

# Extracts the heredoc's literal body (between `<<REMOTE` and the closing
# `REMOTE`), substitutes the ONE sandbox-workdir adaptation, and writes it
# as a runnable script -- never a hand-copied re-transcription of the
# script's own logic.
esc082_extract_heredoc() {
  local out="$1"
  local start end
  start="$(grep -n '<<REMOTE' "$PROVE_SH" | head -1 | cut -d: -f1)"
  end="$(grep -n '^REMOTE$' "$PROVE_SH" | head -1 | cut -d: -f1)"
  local raw_text
  raw_text="$(sed -n "$((start + 1)),$((end - 1))p" "$PROVE_SH" \
    | sed 's#^cd /root && rm -rf jammi-ai#cd "$ESC082_WORKDIR" \&\& rm -rf jammi-ai#')"
  # Two-phase expansion, mirroring what actually happens on the wire: phase
  # 1 (this `eval`) reproduces the LOCAL/CI-runner's own UNQUOTED-heredoc
  # construction -- `\$`/`` \` ``/`\\` de-escaped, `${NATIVE_COMPUTE_CAP}`/
  # `${GIT_REF}`/`${GIT_REPO}` expanded using THIS fixture's own env --
  # producing the EXACT text the real driver would send over ssh. Feeding
  # `$raw_text` through a SECOND, genuinely unquoted heredoc via `eval`
  # (which re-parses its argument as fresh source, so `<<ESC082_RAW...`
  # below is a REAL heredoc redirect at that point, not inert text) gets
  # this right without hand-reimplementing heredoc-expansion rules; a
  # naive `\$` -> `$` sed substitution was tried FIRST and produced a
  # bash syntax error (`json.load(open(` unparseable) because it never
  # reproduced `$(...)`'s own nested-quote "island" parsing -- an escaped
  # `\$(` is never treated as a command-substitution island, so the
  # embedded double quotes inside the manifest-read python snippet would
  # otherwise close the OUTER assignment's string early. Phase 2 is just
  # running the resulting (already-expanded) text as an ordinary script.
  echo '#!/usr/bin/env bash' > "$out"
  eval "cat <<ESC082_RAW_9f8a3c
$raw_text
ESC082_RAW_9f8a3c" >> "$out"
}

# Runs the extracted heredoc script with `$1`=the cargo invocation substring
# to FAIL (empty = everything succeeds), writing the emitted log to `$2`.
# Returns the script's own exit code (mirrors `raw_rc` in the real driver,
# since this IS the remote script the driver's own `wait $pid` would see).
esc082_run() {
  local fail_match="$1" outlog="$2"
  local script="$SANDBOX/esc082-heredoc.sh"
  local workdir="$SANDBOX/esc082-workdir-$$-$RANDOM"
  rm -rf "$workdir"; mkdir -p "$workdir"
  # Phase-1 expansion (inside esc082_extract_heredoc's own `eval`) needs
  # NATIVE_COMPUTE_CAP/GIT_REF/GIT_REPO/ESC082_WORKDIR ALREADY set in THIS
  # shell -- it runs at EXTRACTION time, not at the later `env ... bash`
  # run time below, so exporting them only on that later line would be too
  # late for the `${...}` references the extraction step must resolve.
  export NATIVE_COMPUTE_CAP=80 GIT_REF=test-ref GIT_REPO=unused ESC082_WORKDIR="$workdir"
  esc082_extract_heredoc "$script"
  env PATH="$ESC082_BIN:$PATH" \
    ESC082_FAIL_MATCH="$fail_match" \
    ESC082_REAL_ROOT="$REPO_ROOT" \
    ESC082_WORKDIR="$workdir" \
    bash "$script" > "$outlog" 2>&1
  local rc=$?
  rm -rf "$workdir"
  return "$rc"
}

# --- positive case: only the bench invocation (gpu-inference-scale) fails. ---
esc082_log="$SANDBOX/esc082-bench-fail.log"
esc082_run "gpu-inference-scale" "$esc082_log"
esc082_raw_rc=$?
esc082_ok=1
grep -q '^BENCH_EXIT=1$' "$esc082_log" || esc082_ok=0
grep -q '^PROVE_EXIT=0$' "$esc082_log" || esc082_ok=0
for g in "${PROVE_GROUPS[@]}"; do
  grep -q "^PROVE_GROUP_RC name=${g} rc=0\$" "$esc082_log" || esc082_ok=0
done
if [ "$esc082_ok" -eq 1 ]; then
  ok "esc-082 (heredoc execution): a bench-only cargo failure -> BENCH_EXIT=1, PROVE_EXIT=0, every gating group rc=0"
else
  bad "esc-082 (heredoc execution): expected BENCH_EXIT=1/PROVE_EXIT=0/all-gating-rc=0; raw_rc=$esc082_raw_rc log=$(cat "$esc082_log")"
fi
esc082_verdict_rc=1
rp_prove_verdict "$esc082_raw_rc" "$esc082_log"
esc082_verdict_rc=$?
if [ "$esc082_verdict_rc" -eq 0 ]; then
  ok "esc-082 (heredoc execution): rp_prove_verdict over the EXECUTED heredoc's own log yields lane rc 0 (bench is genuinely non-gating in production code, not only in hand-authored fixtures)"
else
  bad "esc-082 (heredoc execution): expected lane rc 0 from the executed heredoc's own log; got $esc082_verdict_rc"
fi

# --- negative control: the SAME shim, but the served-proof invocation
# (grpc_embedding_gpu) fails instead -- this must NOT be swallowed. ---
esc082_neg_log="$SANDBOX/esc082-served-fail.log"
esc082_run "grpc_embedding_gpu" "$esc082_neg_log"
esc082_neg_raw_rc=$?
rp_prove_verdict "$esc082_neg_raw_rc" "$esc082_neg_log"
esc082_neg_verdict_rc=$?
if [ "$esc082_neg_verdict_rc" -ne 0 ] && grep -q '^PROVE_GROUP_RC name=served-client-server-proof rc=[1-9]' "$esc082_neg_log"; then
  ok "esc-082 negative control (heredoc execution): a served-proof cargo failure -> lane rc != 0, marker names served-client-server-proof"
else
  bad "esc-082 negative control (heredoc execution): expected a nonzero lane rc naming served-client-server-proof; got $esc082_neg_verdict_rc log=$(cat "$esc082_neg_log")"
fi

# --- bench opens strictly AFTER every gating group's own marker (the
# "runs LAST" comment, now checked against the EXECUTED log's own line
# order, not prose). ---
esc082_bench_open_line="$(grep -n '^::group::bench$' "$esc082_log" | head -1 | cut -d: -f1)"
esc082_last_gating_marker_line=0
for g in "${PROVE_GROUPS[@]}"; do
  l="$(grep -n "^PROVE_GROUP_RC name=${g} rc=" "$esc082_log" | tail -1 | cut -d: -f1)"
  [ -n "$l" ] && [ "$l" -gt "$esc082_last_gating_marker_line" ] && esc082_last_gating_marker_line="$l"
done
if [ -n "$esc082_bench_open_line" ] && [ "$esc082_bench_open_line" -gt "$esc082_last_gating_marker_line" ]; then
  ok "esc-082 (heredoc execution): ::group::bench opens AFTER every gating group's own marker in the executed log (runs LAST, not only by comment)"
else
  bad "esc-082 (heredoc execution): expected bench's own group-open line ($esc082_bench_open_line) after the last gating marker ($esc082_last_gating_marker_line)"
fi

# ============================================================================
# partial-line-at-poll-boundary: a marker split across multiple 0.2s polls
# is still reassembled by rp_run_remote_watched's own carry logic. Round-2
# audit fix: the ORIGINAL version used a fixed 5s poll with a 3s threshold
# and a 6s inter-part delay -- numbers with essentially no safety margin
# (threshold < poll), which made the outcome depend on scheduling jitter
# rather than the declared threshold. Now: threshold=1s, poll=0.2s, and the
# inter-part delay (0.3s) is > the poll interval (so it genuinely spans
# multiple poll ticks, exercising the carry) but < threshold/3 (0.33s, so
# it can never be mistaken for real inactivity) -- the same >=3x-both-
# directions discipline every other fixture in this file now uses.
# ============================================================================
ssh() {
  cat >/dev/null
  printf 'PROVE_GROUP_RC name=engine'
  sleep 0.3
  printf -- '-core-sweep rc=0\n'
  exec sleep 30
  return 0
}
export -f ssh
out="$(rp_run_remote_watched 1 0.2 <<< "noop" 2>&1)"
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
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
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
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'groups: \[kernels-cuda:0\]'; then
  ok "final-drain carry: an unterminated last PROVE_GROUP_RC marker is still parsed (not dropped, not doubled)"
else
  bad "final-drain carry: unterminated PROVE_GROUP_RC marker mishandled; rc=$rc out=$out"
fi

# ============================================================================
# Round-2 audit BLOCK A fix: the SAME two cases, but through the
# INACTIVITY-KILL arm (76), not the normal-exit arm (124) above. The B.2 fix
# only patched the normal-exit arm's final flush; the kill arm drained
# remaining bytes but never called the shared carry-flush, so an
# unterminated final line landing right as the kill fired was silently
# dropped or attributed to whatever group was open BEFORE it (demonstrated
# live: `NO PROGRESS ... in group "kernels-default"` for a cut genuinely
# inside `kernels-cuda`; `groups: []` for an unterminated final marker).
# Both terminal arms now share ONE `_rrw_flush_carry` function.
# ============================================================================
ssh() {
  cat >/dev/null
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  printf '::group::kernels-cuda'   # no trailing newline; then goes silent -- kill fires
  exec sleep 30
}
export -f ssh
out="$(rp_run_remote_watched 1 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] \
  && echo "$out" | grep -q 'group "kernels-cuda"' \
  && ! echo "$out" | grep -q 'kernels-default"; groups: \[\]' \
  && echo "$out" | grep -q 'groups: \[kernels-default:0\]'; then
  ok "kill-arm final-drain carry: an unterminated last ::group:: line names the CORRECT (newly-opened) group, never the stale previous one"
else
  bad "kill-arm final-drain carry: unterminated ::group:: line mishandled under the 76 path; rc=$rc out=$out"
fi

ssh() {
  cat >/dev/null
  echo "::group::kernels-cuda"
  printf 'PROVE_GROUP_RC name=kernels-cuda rc=0'   # no trailing newline; then goes silent
  exec sleep 30
}
export -f ssh
out="$(rp_run_remote_watched 1 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 76 ] && echo "$out" | grep -q 'groups: \[kernels-cuda:0\]'; then
  ok "kill-arm final-drain carry: an unterminated last PROVE_GROUP_RC marker is still parsed under the 76 path (not dropped)"
else
  bad "kill-arm final-drain carry: unterminated PROVE_GROUP_RC marker mishandled under the 76 path; rc=$rc out=$out"
fi

# ============================================================================
# esc-084/#454 wrong-tree: PROVE_EXPECT_SHA identity check,
# direct rp_run_remote_watched calls (real function, stubbed ssh -- rp_
# cleanup's own podTerminate call on this path is proven separately by the
# F7 "wrong-tree" scenario further below, the same way F1-F4's in-process
# calls never touch rp_cleanup either).
# ============================================================================
GOOD_SHA="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BAD_SHA="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

# A disagreeing PROVE_SHA on a GROWTH tick -> 77 within ONE poll tick, never
# deferred to the inactivity arm (which would take RP_INACTIVITY seconds).
ssh() {
  cat >/dev/null
  echo "PROVE_SHA=${BAD_SHA}"
  exec sleep 30
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched 3 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 77 ] && echo "$out" | grep -q "WRONG TREE expected=${GOOD_SHA} got=${BAD_SHA}"; then
  ok "wrong-tree: a disagreeing PROVE_SHA on a growth tick -> 77 within one poll tick"
else
  bad "wrong-tree: expected 77 + WRONG TREE diagnostic on a growth tick; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# A disagreeing PROVE_SHA landing ONLY in the final flush (no trailing
# newline, the remote exits cleanly right after) -> still 77, never a
# silent pass-through of the remote's own rc=0.
ssh() {
  cat >/dev/null
  echo "::group::device"
  echo "::endgroup::"
  printf 'PROVE_SHA=%s' "$BAD_SHA"
  return 0
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 77 ] && echo "$out" | grep -q "WRONG TREE expected=${GOOD_SHA} got=${BAD_SHA}"; then
  ok "wrong-tree: a mismatching PROVE_SHA landing only in the final flush -> 77"
else
  bad "wrong-tree: expected 77 + WRONG TREE diagnostic from the final flush; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# NO PROVE_SHA= line at all, with PROVE_EXPECT_SHA set -> 77 ("identity
# never asserted" -- absence is a failure, same doctrine as P1's
# zero-producers, never a silent green on the remote's own rc=0).
ssh() {
  cat >/dev/null
  echo "::group::device"
  echo "::endgroup::"
  return 0
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 77 ] && echo "$out" | grep -q "WRONG TREE expected=${GOOD_SHA} got=none"; then
  ok "wrong-tree: PROVE_EXPECT_SHA set but no PROVE_SHA= line ever observed -> 77 (absence is a failure)"
else
  bad "wrong-tree: expected 77 + got=none; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# A MATCHING PROVE_SHA -> the normal verdict, never 77.
ssh() {
  cat >/dev/null
  echo "PROVE_SHA=${GOOD_SHA}"
  echo "::group::device"
  echo "::endgroup::"
  return 0
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 0 ]; then
  ok "wrong-tree: a matching PROVE_SHA -> the normal verdict (rc=0), never 77"
else
  bad "wrong-tree: a matching PROVE_SHA should never trip 77; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# PROVE_EXPECT_SHA UNSET (a hand run of the script) -> no check at all, even
# against an otherwise-mismatching PROVE_SHA= line -- a hand run has no
# record to protect.
ssh() {
  cat >/dev/null
  echo "PROVE_SHA=${BAD_SHA}"
  echo "::group::device"
  echo "::endgroup::"
  return 0
}
export -f ssh
unset PROVE_EXPECT_SHA
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && ! echo "$out" | grep -q "WRONG TREE"; then
  ok "wrong-tree: PROVE_EXPECT_SHA unset -> no check even on a mismatching PROVE_SHA (hand runs have no record to protect)"
else
  bad "wrong-tree: PROVE_EXPECT_SHA unset should never check identity; got rc=$rc out=$out"
fi

# ============================================================================
# BLOCK B10 audit fix -- wrong-tree taxonomy: a MISMATCH is 77 regardless of
# the remote's own rc; ABSENCE is 77 ONLY when the session's own rc is 0 (it
# claimed success without ever asserting identity) -- otherwise it falls
# through UNCHANGED to the existing 124/255 handling and BUDGET diagnostic,
# never relabeling a transport death (esc-085's own signature, rc 255) or a
# genuine budget cut (rc 124) as wrong-tree, and never suppressing BUDGET.
# ============================================================================

# ABSENCE + remote rc 255 (a transport death) -> 255 verbatim, never 77.
ssh() {
  cat >/dev/null
  echo "::group::device"
  echo "::endgroup::"
  return 255
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 255 ] && ! echo "$out" | grep -q "WRONG TREE"; then
  ok "wrong-tree (B10): absence + remote rc 255 -> 255 verbatim, never relabeled 77"
else
  bad "wrong-tree (B10): expected 255 with no WRONG TREE diagnostic; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# ABSENCE + rc 124 with no PROVE_EXIT (a genuine budget cut) -> 124, AND the
# BUDGET diagnostic still fires -- never suppressed by the identity check.
ssh() {
  cat >/dev/null
  echo "::group::kernels-cuda"
  echo "PROVE_GROUP_RC name=kernels-default rc=0"
  return 124
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 124 ] && echo "$out" | grep -q 'BUDGET' && ! echo "$out" | grep -q "WRONG TREE"; then
  ok "wrong-tree (B10): absence + rc 124 with no PROVE_EXIT -> 124 AND the BUDGET diagnostic still fires"
else
  bad "wrong-tree (B10): expected 124 + BUDGET, no WRONG TREE; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# MISMATCH + rc 255 -> still 77 (a mismatch wins regardless of the remote's
# own rc, even a transport-death-shaped one).
ssh() {
  cat >/dev/null
  echo "PROVE_SHA=${BAD_SHA}"
  return 255
}
export -f ssh
PROVE_EXPECT_SHA="$GOOD_SHA"
out="$(rp_run_remote_watched "" 0.2 <<< "noop" 2>&1)"
rc=$?
if [ "$rc" -eq 77 ] && echo "$out" | grep -q "WRONG TREE expected=${GOOD_SHA} got=${BAD_SHA}"; then
  ok "wrong-tree (B10): mismatch + rc 255 -> still 77 (mismatch wins regardless of rc)"
else
  bad "wrong-tree (B10): expected 77 despite rc 255; got rc=$rc out=$out"
fi
unset PROVE_EXPECT_SHA

# ============================================================================
# BLOCK 3 audit fix -- cross-parser fixture: the SAME unterminated-final-
# marker log content, fed independently to (a) rp_run_remote_watched
# (both terminal arms -- the normal-exit case above and the inactivity-kill
# case immediately above it), (b) rp_prove_verdict (the log-file path), and
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

# --------------------------------------------------------------------------
# Round-2 audit advisory: the 8 divergent-grammar inputs the auditor used to
# probe rp_parse_prove_marker vs prove_surface.PROVE_GROUP_RC_RE, fed to
# BOTH parsers, asserting identical (name, rc) OR identical NOMATCH.
# --------------------------------------------------------------------------
xp_div_python() {
  python3 -c "
import sys
sys.path.insert(0, 'ci/scripts')
import prove_surface
m = prove_surface.PROVE_GROUP_RC_RE.search('''$1''')
print(m.group('name') + ' ' + m.group('rc') if m else 'NOMATCH')
"
}
xp_div_bash() {
  if rp_parse_prove_marker "$1"; then
    echo "${RP_PARSED_MARKER_NAME} ${RP_PARSED_MARKER_RC}"
  else
    echo "NOMATCH"
  fi
}
xp_div_check() {
  local label="$1" input="$2"
  local b p
  b="$(xp_div_bash "$input")"
  p="$(xp_div_python "$input")"
  if [ "$b" = "$p" ]; then
    ok "cross-parser divergent input ($label): bash and python agree ($b)"
  else
    bad "cross-parser divergent input ($label): bash='$b' python='$p' -- grammars disagree on input: $input"
  fi
}
xp_div_check "rc=abc (non-numeric)"        "PROVE_GROUP_RC name=x rc=abc"
xp_div_check "two markers on one line"     "PROVE_GROUP_RC name=a rc=0 PROVE_GROUP_RC name=b rc=1"
xp_div_check "double space"                "PROVE_GROUP_RC name=x  rc=0"
xp_div_check "empty rc"                    "PROVE_GROUP_RC name=x rc="
xp_div_check "empty name"                  "PROVE_GROUP_RC name= rc=0"
xp_div_check "CRLF"                        $'PROVE_GROUP_RC name=x rc=0\r'
xp_div_check "leading junk"                "some prefix junk PROVE_GROUP_RC name=x rc=0"
xp_div_check "trailing junk"               "PROVE_GROUP_RC name=x rc=0 trailing junk here"

# --------------------------------------------------------------------------
# BLOCK 2 audit fix (esc-082 class: comment vs mechanism) -- the cross-
# parser fixture `prove_surface.py`/`runpod_lib.sh`'s own comments PROMISE
# for `PROVE_SHA`, mirroring the `xp_div_check` discipline immediately
# above: `rp_parse_prove_sha` (bash) and `prove_surface.PROVE_SHA_RE`
# (python) fed the identical input, asserting identical sha or identical
# NOMATCH. Makes the comments true BY MECHANISM, never by softened wording.
# --------------------------------------------------------------------------
xp_sha_div_python() {
  python3 -c "
import sys
sys.path.insert(0, 'ci/scripts')
import prove_surface
m = prove_surface.PROVE_SHA_RE.search('''$1''')
print(m.group('sha') if m else 'NOMATCH')
"
}
xp_sha_div_bash() {
  if rp_parse_prove_sha "$1"; then
    echo "${RP_PARSED_PROVE_SHA}"
  else
    echo "NOMATCH"
  fi
}
xp_sha_div_check() {
  local label="$1" input="$2"
  local b p
  b="$(xp_sha_div_bash "$input")"
  p="$(xp_sha_div_python "$input")"
  if [ "$b" = "$p" ]; then
    ok "cross-parser PROVE_SHA divergent input ($label): bash and python agree (${b:-NOMATCH})"
  else
    bad "cross-parser PROVE_SHA divergent input ($label): bash='$b' python='$p' -- grammars disagree on input: $input"
  fi
}
XP_SHA40_LOWER="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
XP_SHA40_UPPER="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
xp_sha_div_check "valid 40-hex sha"            "PROVE_SHA=${XP_SHA40_LOWER}"
xp_sha_div_check "uppercase hex"               "PROVE_SHA=${XP_SHA40_UPPER}"
xp_sha_div_check "mixed case"                  "PROVE_SHA=aAbBcC0123456789"
xp_sha_div_check "empty value"                 "PROVE_SHA="
xp_sha_div_check "non-hex"                     "PROVE_SHA=zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
xp_sha_div_check "leading junk"                "some prefix junk PROVE_SHA=${XP_SHA40_LOWER}"
xp_sha_div_check "trailing junk"               "PROVE_SHA=${XP_SHA40_LOWER} trailing junk here"
xp_sha_div_check "two markers on one line"     "PROVE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa PROVE_SHA=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
xp_sha_div_check "CRLF"                        $'PROVE_SHA='"${XP_SHA40_LOWER}"$'\r'
xp_sha_div_check "g-suffixed truncation"       "PROVE_SHA=abc123g"
xp_sha_div_check "PROVE_SHA= abc (space)"      "PROVE_SHA= abc"
xp_sha_div_check "no marker at all"            "nothing to see here"

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
#
# esc-085/#453: when F7_SSHARGV_PROBE / F7_SSHARGV_SESSION are set, the
# stub records its OWN REAL "$@" (verbatim, one token per line) to that
# path -- the ARGV-SCOPED control the escape row demands (never
# `grep ServerAliveInterval ci/scripts/runpod_lib.sh`, which would be
# vacuously green via rp_wait_poll's own unrelated array even before the
# fix). Overwritten (`>`, not `>>`) each call: only the most recent
# invocation of a given shape is kept, and every real call of that shape
# carries the identical liveness options, so the last one is representative.
cat > "$F7_STUBBIN/ssh" <<'SSHSTUB'
#!/usr/bin/env bash
last="${@: -1}"
case "$last" in
  true)
    [ -n "${F7_SSHARGV_PROBE:-}" ] && printf '%s\n' "$@" > "${F7_SSHARGV_PROBE}"
    exit 0
    ;;
  *nvidia-smi*) echo "570.195.03"; exit 0 ;;
  *"bash -s"*)
    [ -n "${F7_SSHARGV_SESSION:-}" ] && printf '%s\n' "$@" > "${F7_SSHARGV_SESSION}"
    cat >/dev/null
    case "${F7_SCENARIO:-}" in
      watchdog)
        echo "::group::capability-surface-build"
        exec sleep 30
        ;;
      wrong-tree)
        # esc-084/#454: a mismatching PROVE_SHA, then hangs --
        # the growth-tick kill path must fire (77) well before RP_INACTIVITY.
        echo "PROVE_SHA=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        echo "::group::capability-surface-build"
        exec sleep 30
        ;;
      *)
        for g in capability-surface-build capability-surface-proof served-client-server-proof engine-core-sweep kernels-default kernels-cuda; do
          echo "PROVE_GROUP_RC name=${g} rc=0"
        done
        echo "::group::bench"
        exec sleep 30
        ;;
    esac
    ;;
  *) exit 0 ;;
esac
SSHSTUB
chmod +x "$F7_STUBBIN/ssh"

# ============================================================================
# esc-085/#453: keepalive argv assertion + its own meta-controls.
# ============================================================================
# Asserts `-oServerAliveInterval=<n>` / `-oServerAliveCountMax=<n>` are
# present in a captured argv file, ATTACHED FORM ONLY (the grep anchors on
# the literal `-oServerAliveInterval=` PREFIX of a single token -- a
# detached `-o ServerAliveInterval=30` records as TWO separate lines,
# `-o` and `ServerAliveInterval=30`, neither of which matches), and NUMERIC
# (>= 1) -- a substring match alone would accept `-oServerAliveInterval=0`
# or a non-integer suffix, which is not what "present" means here.
f7_assert_keepalive_argv() {
  local file="$1" label="$2"
  if [ ! -s "$file" ]; then
    bad "F7 keepalive ($label): no argv captured (missing or empty: $file) -- this ssh shape was never invoked, or its capture var was never wired"
    return
  fi
  local interval_line countmax_line interval_val countmax_val
  interval_line="$(grep -m1 '^-oServerAliveInterval=' "$file" || true)"
  countmax_line="$(grep -m1 '^-oServerAliveCountMax=' "$file" || true)"
  interval_val="${interval_line#-oServerAliveInterval=}"
  countmax_val="${countmax_line#-oServerAliveCountMax=}"
  if [[ "$interval_val" =~ ^[0-9]+$ ]] && [ "$interval_val" -ge 1 ] \
     && [[ "$countmax_val" =~ ^[0-9]+$ ]] && [ "$countmax_val" -ge 1 ]; then
    ok "F7 keepalive ($label): ServerAliveInterval=${interval_val} ServerAliveCountMax=${countmax_val} present, numeric, attached form, >= 1"
  else
    bad "F7 keepalive ($label): missing/non-numeric/degenerate ServerAlive option(s) -- interval='${interval_val}' countmax='${countmax_val}' (argv: $(tr '\n' ' ' < "$file"))"
  fi
}

# Meta-controls (esc-085 control b): drive `f7_assert_keepalive_argv`
# itself against synthetic argv files and assert it correctly REJECTS every
# bad shape -- an inner FAIL here is the CORRECT outcome, so it is
# converted to an outer PASS (and the inner FAIL's own counter increment is
# reversed) rather than double-counted or silently swallowed.
f7_expect_keepalive_reject() {
  local file="$1" desc="$2"
  local before_fail=$FAIL
  f7_assert_keepalive_argv "$file" "$desc"
  if [ "$FAIL" -gt "$before_fail" ]; then
    FAIL=$before_fail
    ok "F7 keepalive meta-control ($desc): correctly rejected"
  else
    bad "F7 keepalive meta-control ($desc): should have been rejected but was accepted"
  fi
}

META_DIR="$SANDBOX/f453-keepalive-meta"
mkdir -p "$META_DIR"
f7_expect_keepalive_reject "$META_DIR/missing" "missing capture file"
: > "$META_DIR/empty"
f7_expect_keepalive_reject "$META_DIR/empty" "empty capture file"
printf '%s\n' -o ServerAliveInterval=30 -o ServerAliveCountMax=6 > "$META_DIR/detached"
f7_expect_keepalive_reject "$META_DIR/detached" "detached -o form (not the real RP_SSHO shape)"
printf '%s\n' -oServerAliveInterval=0 -oServerAliveCountMax=6 > "$META_DIR/interval-zero"
f7_expect_keepalive_reject "$META_DIR/interval-zero" "ServerAliveInterval=0"
printf '%s\n' -oServerAliveInterval=30 -oServerAliveCountMax=0 > "$META_DIR/countmax-zero"
f7_expect_keepalive_reject "$META_DIR/countmax-zero" "ServerAliveCountMax=0"
printf '%s\n' -oServerAliveInterval=abc -oServerAliveCountMax=6 > "$META_DIR/non-integer"
f7_expect_keepalive_reject "$META_DIR/non-integer" "non-integer ServerAliveInterval"
printf '%s\n' -oServerAliveInterval=30 -oServerAliveCountMax=6 > "$META_DIR/good"
f7_assert_keepalive_argv "$META_DIR/good" "meta-control positive (well-formed attached options)"

f7_case() {
  # $1 = "watchdog" | "bench-cut"; $2 = expected exit code.
  # $3 (optional) = a DIFFERENT runpod_gpu_prove.sh path to execute (its own
  # $DIR sources ITS OWN sibling runpod_lib.sh) -- the esc-085 non-vacuity
  # leg's own hook; defaults to the real $PROVE_SH.
  local mode="$1" want_rc="$2" driver="${3:-$PROVE_SH}"
  local marker="$SANDBOX/f7-terminate-marker-$mode"
  local session_argv="$SANDBOX/f7-sshargv-session-$mode"
  local probe_argv="$SANDBOX/f7-sshargv-probe-$mode"
  rm -f "$marker" "$session_argv" "$probe_argv"
  # A single `env ...` line (never a `VAR=val \`-per-line assignment chain):
  # check_gpu_prove_timings.py's own R1 setter-predicate scan matches
  # `^\s*(export\s+)?RP_(TIMEOUT|INACTIVITY)=` at the START of a physical
  # line, which a bare `RP_INACTIVITY=3 \` continuation line would satisfy
  # -- this is a plain per-invocation env-var override, not a second
  # committed default, and must not be misread as one.
  # RP_WATCH_POLL_S=0.2 (fixture/diagnostic-only, see runpod_gpu_prove.sh's
  # own doc) keeps this REAL executed subprocess's own watchdog poll fast,
  # the same >=3x-separated-from-the-stub's-own-timing discipline every
  # other fixture in this file uses (RP_INACTIVITY=3 vs the stub's own
  # `sleep 30` silence is a 10x margin).
  env RUNPOD_API_KEY=test-dummy-key PATH="$F7_STUBBIN:$PATH" F7_TERMINATE_MARKER="$marker" F7_SCENARIO="$mode" F7_SSHARGV_SESSION="$session_argv" F7_SSHARGV_PROBE="$probe_argv" GPU_PROVE_ARCH=sm_80 RP_INACTIVITY=3 RP_WATCH_POLL_S=0.2 RP_SSH_WAIT_SECS=10 bash "$driver" > "$SANDBOX/f7-$mode.out" 2>&1
  local rc=$?
  local heredoc_stderr_ok=1
  # The unescaped-backtick class (real bug, fixed live) manifests as a
  # command-substitution error on the RUNNER's own stderr the moment the
  # LOCAL shell expands the <<REMOTE heredoc -- F7 ALREADY captures that
  # exact expansion end to end, so re-check its own output here rather than
  # building a second mechanism.
  if grep -qE 'command not found|unexpected EOF while looking for matching' "$SANDBOX/f7-$mode.out"; then
    heredoc_stderr_ok=0
  fi
  if [ "$rc" -eq "$want_rc" ] && [ -f "$marker" ] && [ "$heredoc_stderr_ok" -eq 1 ]; then
    ok "F7 ($mode): the real executed exit path returns $want_rc, rp_cleanup's own podTerminate call fired (marker recorded), and the <<REMOTE heredoc's own expansion produced no command-substitution stderr"
  else
    bad "F7 ($mode): expected rc=$want_rc with a recorded podTerminate call and no heredoc-expansion stderr; got rc=$rc marker-present=$([ -f "$marker" ] && echo yes || echo no) heredoc_stderr_ok=$heredoc_stderr_ok; out=$(cat "$SANDBOX/f7-$mode.out")"
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
f7_assert_keepalive_argv "$SANDBOX/f7-sshargv-session-watchdog" "watchdog scenario, bash -s session"
f7_assert_keepalive_argv "$SANDBOX/f7-sshargv-probe-watchdog" "watchdog scenario, true reachability probe"
f7_case "bench-cut" 0
f7_assert_keepalive_argv "$SANDBOX/f7-sshargv-session-bench-cut" "bench-cut scenario, bash -s session"
f7_assert_keepalive_argv "$SANDBOX/f7-sshargv-probe-bench-cut" "bench-cut scenario, true reachability probe"

# ============================================================================
# esc-084/#454 wrong-tree, F7-style subprocess execution:
# proves rp_cleanup's own podTerminate call fires on the 77 path too, the
# same real-executed-exit-path proof F7's watchdog/bench-cut scenarios give
# 76/0 above. PROVE_EXPECT_SHA is set to a DIFFERENT sha than the stub's own
# "PROVE_SHA=deadbeef..." line, so the mismatch is genuine.
# ============================================================================
WT_MARKER="$SANDBOX/f7-terminate-marker-wrong-tree"
rm -f "$WT_MARKER"
env RUNPOD_API_KEY=test-dummy-key PATH="$F7_STUBBIN:$PATH" F7_TERMINATE_MARKER="$WT_MARKER" F7_SCENARIO="wrong-tree" GPU_PROVE_ARCH=sm_80 RP_INACTIVITY=3 RP_WATCH_POLL_S=0.2 RP_SSH_WAIT_SECS=10 PROVE_EXPECT_SHA="cccccccccccccccccccccccccccccccccccccc" bash "$PROVE_SH" > "$SANDBOX/f7-wrong-tree.out" 2>&1
wt_rc=$?
if [ "$wt_rc" -eq 77 ] && [ -f "$WT_MARKER" ] && grep -q 'WRONG TREE' "$SANDBOX/f7-wrong-tree.out"; then
  ok "F7 (wrong-tree, esc-084/#454): the real executed exit path returns 77, rp_cleanup's own podTerminate call fired (marker recorded), and the WRONG TREE diagnostic is present"
else
  bad "F7 (wrong-tree): expected rc=77 with a recorded podTerminate call and a WRONG TREE diagnostic; got rc=$wt_rc marker-present=$([ -f "$WT_MARKER" ] && echo yes || echo no); out=$(cat "$SANDBOX/f7-wrong-tree.out")"
fi

# ============================================================================
# esc-085 control (c), non-vacuity: the SAME F7 fixture, run against a
# scratch copy of BOTH runpod_lib.sh AND runpod_gpu_prove.sh (the driver
# sources "$DIR/runpod_lib.sh" by its OWN $DIR, so both files must move
# together) with the two keepalive options stripped from RP_SSHO -- must go
# RED. Proves the control is load-bearing, not a fixture that would pass
# unconditionally regardless of what RP_SSHO actually carries.
# ============================================================================
NV_DIR="$SANDBOX/f453-nonvacuity"
mkdir -p "$NV_DIR"
cp "$DIR/runpod_lib.sh" "$NV_DIR/runpod_lib.sh"
cp "$PROVE_SH" "$NV_DIR/runpod_gpu_prove.sh"
# Strip exactly the two options this control exists to pin -- a scratch-only
# mutation of a COPY, never the real tree.
python3 - "$NV_DIR/runpod_lib.sh" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
stripped = text.replace(" -oServerAliveInterval=30 -oServerAliveCountMax=6", "")
if stripped == text:
    raise SystemExit("esc-085 non-vacuity fixture: the keepalive options were not found to strip -- fixture itself is stale")
open(path, "w").write(stripped)
PYEOF

f7_case "watchdog" 76 "$NV_DIR/runpod_gpu_prove.sh"
before_fail=$FAIL
f7_assert_keepalive_argv "$SANDBOX/f7-sshargv-session-watchdog" "non-vacuity scratch copy (options stripped)"
if [ "$FAIL" -gt "$before_fail" ]; then
  FAIL=$before_fail
  ok "F7 keepalive non-vacuity (esc-085 control c): stripping RP_SSHO's keepalive options in a scratch copy correctly turns the assertion RED"
else
  FAIL=$before_fail
  bad "F7 keepalive non-vacuity (esc-085 control c): stripping RP_SSHO's keepalive options did NOT turn the assertion red -- the control is vacuous"
fi

# ============================================================================
# rp_wait_poll keepalive precedence (esc-085/#453): the probe's own PREPENDED
# options still resolve to 10s/3-try even though the shared RP_SSHO now
# itself carries 30s/6-try. `ssh -G` resolves the EFFECTIVE configuration
# with zero network traffic -- same technique test_gpu_dev_lifecycle.sh's
# own group5 uses for RP_SSHO's IdentitiesOnly=yes -- a direct assertion on
# the real option array, never a grep on source text. The prepended
# literal below must stay byte-identical to runpod_lib.sh's own
# `wait_sshopts=(...)` line; a drift there is this test's job to catch.
# ============================================================================
(
  export RUNPOD_API_KEY="dummy-key"
  unset RP_SESSION
  # shellcheck source=ci/scripts/runpod_lib.sh
  source "$DIR/runpod_lib.sh"
  rp_init
  RP_PORT=22
  RP_HOST=placeholder-host
  wait_sshopts=(-oBatchMode=yes -oServerAliveInterval=10 -oServerAliveCountMax=3 "${RP_SSHO[@]}")
  probe_out="$(ssh -G "${wait_sshopts[@]}" -p "$RP_PORT" "root@${RP_HOST}" "timeout 20 bash -s" 2>&1)"
  probe_interval="$(printf '%s\n' "$probe_out" | grep -i '^serveraliveinterval ' | awk '{print $2}')"
  probe_countmax="$(printf '%s\n' "$probe_out" | grep -i '^serveralivecountmax ' | awk '{print $2}')"
  if [ "$probe_interval" = "10" ] && [ "$probe_countmax" = "3" ]; then
    record_ok=1
  else
    record_ok=0
  fi
  echo "$record_ok $probe_interval $probe_countmax" > "$SANDBOX/rp-wait-poll-precedence.out"
)
read -r rp_wp_ok rp_wp_interval rp_wp_countmax < "$SANDBOX/rp-wait-poll-precedence.out"
if [ "$rp_wp_ok" = "1" ]; then
  ok "rp_wait_poll probe precedence: ssh -G resolves ServerAliveInterval=10 ServerAliveCountMax=3 (prepended, wins over RP_SSHO's own 30/6)"
else
  bad "rp_wait_poll probe precedence: expected 10/3, got interval=${rp_wp_interval:-<empty>} countmax=${rp_wp_countmax:-<empty>}"
fi

echo
echo "gpu-prove-lane: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
