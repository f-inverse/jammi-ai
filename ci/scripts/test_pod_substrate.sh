#!/usr/bin/env bash
# Mocks-only, no-network regression suite for the pod-build-substrate unit
# (contract ci/pod-build-substrate; ledger rows 1-4, 8-9, 13, 17, 19, 21, 29,
# 34 — five pressure-test rounds). Covers acceptance A1 (a)-(h):
#
#   (a) rp_bootstrap's generated /root/.jammi_env carries
#       CARGO_BUILD_RUSTC_WRAPPER= and CARGO_INCREMENTAL=0, and NOWHERE in
#       runpod_lib.sh does SCCACHE_* or rp_s3_load survive (row 17: a
#       measured S3-backed sccache gave zero cross-target-dir reuse on this
#       image and cost ~+33% wall).
#   (b) pod_target_clone.sh refuses (exit 3) without a seed marker; clones
#       correctly with a marker present; --verify on a fixture `cargo build
#       -v` log correctly distinguishes a Fresh-jammi-* poisoned clone from a
#       clean one.
#   (c) rp_s3_load is entirely absent from runpod_lib.sh (the S3-backed
#       sccache mechanism is REMOVED, not merely disabled).
#   (d) pod_timing_lock.sh's flock exclusivity: two concurrent non-blocking
#       acquires -> exactly one 0 and one 75; the holder dying (kill -9)
#       frees the lock for the next acquirer; a tmux-detached job holds the
#       lock against an outsider for its lifetime. Linux-only (flock from
#       util-linux); under JAMMI_REQUIRE_LOCK_TEST=1 a skip here is RED —
#       CI (ubuntu) sets this, so the mechanism is never silently unproven.
#   (e) key-manifest RED tests (i)/(ii): every name-shaped string literal /
#       every `rerun-if-env-changed=` literal in jammi-kernels' and
#       jammi-wire's build.rs, and in the vendored bindgen_cuda 0.1.6
#       source, is accounted for by pod_seed_key_inputs.toml — against the
#       REAL sources in this checkout, never a fixture (a fixture is used
#       ONLY to prove the scanner itself catches an unlisted literal).
#   (f) push excludes: pod_push_stamp.sh's `excludes` output is pinned
#       (a regression tripwire) AND gpu-dev.sh's `push` case sources it from
#       that SAME function (grep), so the real rsync and the push-stamp's
#       manifest hash can never see two different exclude sets.
#   (g) no unanchored/unquoted `-t jammi` remains in gpu-dev.sh; every window
#       op uses `"=<session>:"`.
#   (h) exactly two "/root/jammi-ai" literal sites in runpod_lib.sh
#       (rp_tree_dir's own default + rp_bootstrap's clone destination,
#       counting only non-comment lines) and zero in dev-gpu-recipes.md.
#
# Every network-facing call is either avoided entirely (a/c/e/f/g/h are pure
# source-text/fixture checks) or intercepted at the FUNCTION boundary the
# same way test_gpu_dev_lifecycle.sh's own module doc describes: (a)
# overrides `rp_run_remote` itself (a plain bash function reassignment,
# legal because this suite sources runpod_lib.sh directly) to CAPTURE the
# heredoc rp_bootstrap would have sent, rather than faking ssh's network
# behaviour. `ssh` itself is never mocked; no fixture here needs a reachable
# pod. `flock`/`tmux` in (d) run for REAL against local files/processes —
# not mocked, genuinely exercising the kernel's own lock semantics.
#
# Run: bash ci/scripts/test_pod_substrate.sh
# Hermetic: no network, no GPU, no real RunPod account, no cargo build.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

PASS=0
FAIL=0
SKIP=0
ok()   { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }
skip() { SKIP=$((SKIP + 1)); echo "skip - $*"; }

# ═════════════════════════════════════════════════════════════════════════
# (a) + (c): .jammi_env writer + no S3/sccache anywhere in runpod_lib.sh
# ═════════════════════════════════════════════════════════════════════════
{
  CAPTURE="$SANDBOX/bootstrap_heredoc.txt"
  DRIVER="$SANDBOX/probe_bootstrap.sh"
  cat > "$DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export RUNPOD_API_KEY=test-dummy-key
export RP_SESSION_ROOT="$SANDBOX/sessions"
export RP_SSH_CONFIG="$SANDBOX/ssh_config"
mkdir -p "\$RP_SESSION_ROOT"
DIR="$REPO_ROOT/ci/scripts"
# shellcheck disable=SC1091
. "\$DIR/runpod_lib.sh"
# Function-level intercept (see this suite's own module doc): capture the
# heredoc rp_bootstrap builds instead of actually shelling out over ssh —
# no ssh mock, no network, no pod.
rp_run_remote() { cat > "$CAPTURE"; return 0; }
RP_SESSION=""
RP_HOST=127.0.0.1
RP_PORT=1
rp_bootstrap main >/dev/null 2>&1
DRV
  chmod +x "$DRIVER"
  bash "$DRIVER"

  # The captured text is rp_bootstrap's REMOTE SCRIPT SOURCE (what would run
  # on the pod), not its output — it writes /root/.jammi_env via `echo
  # 'export NAME=VALUE' >> /root/.jammi_env` lines, which this repo cannot
  # actually execute here (root-only paths, yum). Asserting the exact `echo`
  # statement that PRODUCES each line is the faithful, hermetic proxy for
  # "the writer emits this line" without running remote-only commands.
  if [ -f "$CAPTURE" ] && grep -Fq "echo 'export CARGO_BUILD_RUSTC_WRAPPER='" "$CAPTURE"; then
    ok "(a) rp_bootstrap's .jammi_env writer emits CARGO_BUILD_RUSTC_WRAPPER= (empty)"
  else
    bad "(a) CARGO_BUILD_RUSTC_WRAPPER= not found in the captured bootstrap heredoc"
  fi
  if [ -f "$CAPTURE" ] && grep -Fq "echo 'export CARGO_INCREMENTAL=0'" "$CAPTURE"; then
    ok "(a) rp_bootstrap's .jammi_env writer emits CARGO_INCREMENTAL=0"
  else
    bad "(a) CARGO_INCREMENTAL=0 not found in the captured bootstrap heredoc"
  fi
  # SCCACHE_* / rp_s3_load / --start-server are the MECHANISM (env vars,
  # function calls, live commands) — a bare English "sccache" in an
  # explanatory comment about why it is gone is not the thing this check
  # bans.
  if [ -f "$CAPTURE" ] && grep -qE 'SCCACHE_|rp_s3_load|sccache --start-server|sccache --stop-server|AWS_ACCESS_KEY_ID' "$CAPTURE"; then
    bad "(a) captured bootstrap heredoc still contains the sccache/S3 MECHANISM (env var/function/command), not just a comment about its removal"
  else
    ok "(a) captured bootstrap heredoc carries no sccache/S3 mechanism"
  fi
}

{
  if grep -q 'rp_s3_load' "$REPO_ROOT/ci/scripts/runpod_lib.sh"; then
    bad "(c) rp_s3_load still present in runpod_lib.sh"
  else
    ok "(c) rp_s3_load is entirely absent from runpod_lib.sh"
  fi
  if grep -qi 'SCCACHE_' "$REPO_ROOT/ci/scripts/runpod_lib.sh"; then
    bad "(c) SCCACHE_* still present in runpod_lib.sh"
  else
    ok "(c) SCCACHE_* is entirely absent from runpod_lib.sh"
  fi
  if grep -q 'RP_S3_CONF' "$REPO_ROOT/ci/scripts/runpod_lib.sh"; then
    bad "(c) RP_S3_CONF still present in runpod_lib.sh"
  else
    ok "(c) RP_S3_CONF is entirely absent from runpod_lib.sh"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (b) pod_target_clone.sh
# ═════════════════════════════════════════════════════════════════════════
{
  CLONE_SH="$REPO_ROOT/ci/scripts/pod_target_clone.sh"

  # No marker -> refuse, exit 3.
  NOMARKER_SEED="$SANDBOX/b_noseed"
  mkdir -p "$NOMARKER_SEED"
  bash "$CLONE_SH" "$NOMARKER_SEED" "$SANDBOX/b_dest_should_not_exist" >/dev/null 2>"$SANDBOX/b1.err"
  rc=$?
  if [ "$rc" -eq 3 ] && [ ! -e "$SANDBOX/b_dest_should_not_exist" ]; then
    ok "(b) pod_target_clone.sh refuses (exit 3) without .jammi-seed-complete, and creates nothing"
  else
    bad "(b) expected exit 3 + no destination without a seed marker (got rc=$rc); stderr: $(cat "$SANDBOX/b1.err")"
  fi

  # Marker present -> real copy, content matches, no deletion of the seed.
  SEED="$SANDBOX/b_seed"
  mkdir -p "$SEED/release"
  echo "seed-artifact" > "$SEED/release/thing"
  : > "${SEED}.jammi-seed-complete"
  DEST="$SANDBOX/b_dest"
  bash "$CLONE_SH" "$SEED" "$DEST" > "$SANDBOX/b2.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$DEST/release/thing" ] && [ "$(cat "$DEST/release/thing")" = "seed-artifact" ] && [ -f "$SEED/release/thing" ]; then
    ok "(b) pod_target_clone.sh clones content faithfully and leaves the seed intact (no deletion step)"
  else
    bad "(b) clone did not reproduce seed content or removed the seed (rc=$rc); output: $(cat "$SANDBOX/b2.out")"
  fi

  # Refuses to clone onto an existing destination.
  bash "$CLONE_SH" "$SEED" "$DEST" >/dev/null 2>"$SANDBOX/b3.err"
  rc=$?
  if [ "$rc" -eq 2 ]; then
    ok "(b) pod_target_clone.sh refuses to clone over an existing destination"
  else
    bad "(b) expected exit 2 cloning over an existing destination (got rc=$rc)"
  fi

  # --verify: a log WITHOUT a Fresh jammi-* line passes.
  CLEAN_LOG="$SANDBOX/b_clean.log"
  { echo "   Compiling jammi-kernels v0.47.0"; echo "   Compiling jammi-bench v0.47.0"; echo "    Finished release"; } > "$CLEAN_LOG"
  bash "$CLONE_SH" "" "" --verify < "$CLEAN_LOG" > "$SANDBOX/b4.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    ok "(b) --verify PASSES a log with no Fresh jammi-* unit"
  else
    bad "(b) --verify unexpectedly failed a clean log (rc=$rc): $(cat "$SANDBOX/b4.out")"
  fi

  # --verify: a log WITH a Fresh jammi-* line fails (the poisoned-clone case).
  DIRTY_LOG="$SANDBOX/b_dirty.log"
  { echo "       Fresh jammi-kernels v0.47.0"; echo "   Compiling jammi-bench v0.47.0"; } > "$DIRTY_LOG"
  bash "$CLONE_SH" "" "" --verify < "$DIRTY_LOG" > "$SANDBOX/b5.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && grep -q 'Fresh' "$SANDBOX/b5.out"; then
    ok "(b) --verify FAILS a log carrying a Fresh jammi-* unit (poisoned-clone detection)"
  else
    bad "(b) --verify should have failed on a Fresh jammi-* line (rc=$rc): $(cat "$SANDBOX/b5.out")"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (d) pod_timing_lock.sh — flock exclusivity. Linux-only (util-linux flock);
# a skip is RED under JAMMI_REQUIRE_LOCK_TEST=1 (CI sets this).
# ═════════════════════════════════════════════════════════════════════════
{
  LOCK_SH="$REPO_ROOT/ci/scripts/pod_timing_lock.sh"
  if ! command -v flock >/dev/null 2>&1; then
    if [ "${JAMMI_REQUIRE_LOCK_TEST:-0}" = "1" ]; then
      bad "(d) flock (util-linux) not found and JAMMI_REQUIRE_LOCK_TEST=1 — a skip here is a RED, not a pass"
    else
      skip "(d) flock (util-linux) not found on this host — lock tests skipped (set JAMMI_REQUIRE_LOCK_TEST=1 to make this fatal)"
    fi
  else
    LOCKFILE="$SANDBOX/d.lock"

    # Two concurrent -n acquires: exactly one 0, one 75.
    export JAMMI_TIMING_LOCK="$LOCKFILE"
    OUT1="$SANDBOX/d1.out"; OUT2="$SANDBOX/d2.out"
    ( JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'sleep 1; exit 0'; echo "rc=$?" > "$OUT1" ) &
    P1=$!
    sleep 0.2
    ( JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'exit 0'; echo "rc=$?" > "$OUT2" ) &
    P2=$!
    wait "$P1" "$P2" 2>/dev/null
    r1="$(cat "$OUT1" 2>/dev/null)"; r2="$(cat "$OUT2" 2>/dev/null)"
    if { [ "$r1" = "rc=0" ] && [ "$r2" = "rc=75" ]; } || { [ "$r1" = "rc=75" ] && [ "$r2" = "rc=0" ]; }; then
      ok "(d) two concurrent 'acquire -n' -> exactly one exit 0, one exit 75 (${r1}, ${r2})"
    else
      bad "(d) expected exactly one 0 and one 75 across two concurrent 'acquire -n' (got ${r1}, ${r2})"
    fi

    # Holder file written under the lock, tmp+rename: readable and complete
    # immediately after a successful acquire.
    rm -f "$LOCKFILE" "${LOCKFILE}.holder"
    JAMMI_TIMING_LABEL="probe-d" bash "$LOCK_SH" acquire -n -- bash -c 'exit 0' >/dev/null 2>&1
    if [ -f "${LOCKFILE}.holder" ] && grep -q '^holder=probe-d$' "${LOCKFILE}.holder"; then
      ok "(d) holder file is written (tmp+rename) under the lock with the caller's label"
    else
      bad "(d) holder file missing or malformed after a successful acquire"
    fi

    # Holder dies -> the lock frees (kernel-owned liveness: no stale marker,
    # nothing to steal). `flock file command` forks to run `command`, and a
    # POSIX flock() lock is bound to the OPEN FILE DESCRIPTION, which fork()
    # shares — so the CHILD (the real job) inherits the hold, and outlives a
    # kill aimed only at the flock PARENT. This is the same fd-inheritance
    # this suite's own module doc / dev-gpu.md warns about, and it is real
    # POSIX behaviour, not specific to this repo's flock. The realistic
    # "holder dies" shape this tooling actually relies on is tmux's own
    # `kill-session`, which SIGHUPs the WHOLE pane process group at once
    # (gpu-dev.sh's `run` does exactly this before starting the next job) —
    # so this leg kills the FULL descendant tree, not just the top pid.
    rm -f "$LOCKFILE"
    ( JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'sleep 30' ) &
    HOLDER_PID=$!
    sleep 0.3
    kill_tree() { # $1=pid
      local child
      for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
      kill -9 "$1" 2>/dev/null
    }
    kill_tree "$HOLDER_PID"
    wait "$HOLDER_PID" 2>/dev/null
    # Give the kernel a moment to release the fd on process death.
    deadline=$((SECONDS + 10))
    freed=0
    while [ "$SECONDS" -lt "$deadline" ]; do
      if JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'exit 0' >/dev/null 2>&1; then
        freed=1; break
      fi
      sleep 0.2
    done
    if [ "$freed" = "1" ]; then
      ok "(d) killing the FULL holder tree (matching tmux kill-session's own process-group signal) frees the lock for the next acquirer — kernel-owned liveness, no stale state"
    else
      bad "(d) lock did not free after killing the full holder tree"
    fi

    # A tmux-detached job holds the lock against an outsider for its
    # lifetime (the shape `run --timing`'s own launcher relies on).
    if command -v tmux >/dev/null 2>&1; then
      rm -f "$LOCKFILE"
      TSESS="jammi-pod-substrate-test-$$"
      tmux kill-session -t "=${TSESS}" 2>/dev/null
      tmux new-session -d -s "$TSESS" "JAMMI_TIMING_LOCK='$LOCKFILE' bash '$LOCK_SH' acquire -n -- bash -c 'sleep 5'"
      sleep 0.5
      if JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'exit 0' >/dev/null 2>&1; then
        bad "(d) an outsider acquired the lock while the tmux-detached job should still be holding it"
      else
        ok "(d) a tmux-detached job holds the lock against an outsider for its lifetime"
      fi
      tmux kill-session -t "=${TSESS}" 2>/dev/null
    else
      if [ "${JAMMI_REQUIRE_LOCK_TEST:-0}" = "1" ]; then
        bad "(d) tmux not found and JAMMI_REQUIRE_LOCK_TEST=1 — a skip here is a RED, not a pass"
      else
        skip "(d) tmux not found — the tmux-detached-job lock leg is skipped"
      fi
    fi
    unset JAMMI_TIMING_LOCK
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (e) key-manifest RED tests (i)/(ii) — against the REAL sources
# ═════════════════════════════════════════════════════════════════════════
{
  SEED_TARGET_SH="$REPO_ROOT/ci/scripts/pod_seed_target.sh"
  MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
  BINDGEN=""
  for f in "$HOME"/.cargo/registry/src/*/bindgen_cuda-*/src/lib.rs; do
    [ -f "$f" ] && BINDGEN="$f" && break
  done

  DRIVER="$SANDBOX/probe_scan.sh"
  cat > "$DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$SEED_TARGET_SH"
NAMES="\$(mktemp)"
pod_seed_manifest_names "$MANIFEST" > "\$NAMES"
mode="\$1"; src="\$2"
pod_seed_scan_source "\$src" "\$NAMES" "\$mode"
rm -f "\$NAMES"
DRV
  chmod +x "$DRIVER"

  KERNELS_BUILDRS="$REPO_ROOT/crates/jammi-kernels/build.rs"
  WIRE_BUILDRS="$REPO_ROOT/crates/jammi-wire/build.rs"

  unlisted="$(bash "$DRIVER" all "$KERNELS_BUILDRS")"
  [ -z "$unlisted" ] && ok "(e-i) every name-shaped literal in jammi-kernels/build.rs is manifest-accounted" \
    || bad "(e-i) unlisted literal(s) in jammi-kernels/build.rs: $unlisted"

  unlisted="$(bash "$DRIVER" all "$WIRE_BUILDRS")"
  [ -z "$unlisted" ] && ok "(e-i) every name-shaped literal in jammi-wire/build.rs is manifest-accounted" \
    || bad "(e-i) unlisted literal(s) in jammi-wire/build.rs: $unlisted"

  if [ -n "$BINDGEN" ]; then
    unlisted="$(bash "$DRIVER" all "$BINDGEN")"
    [ -z "$unlisted" ] && ok "(e-i) every name-shaped literal in vendored bindgen_cuda's src/lib.rs is manifest-accounted ($BINDGEN)" \
      || bad "(e-i) unlisted literal(s) in bindgen_cuda: $unlisted"

    unlisted="$(bash "$DRIVER" rerun_only "$BINDGEN")"
    [ -z "$unlisted" ] && ok "(e-ii) every rerun-if-env-changed literal in bindgen_cuda is manifest-accounted" \
      || bad "(e-ii) unlisted rerun-if-env-changed literal(s) in bindgen_cuda: $unlisted"
  else
    bad "(e) vendored bindgen_cuda source not found under \$HOME/.cargo/registry/src — this box's registry appears unfetched; the (e-i)/(e-ii) bindgen_cuda legs cannot run (fetch the workspace's dependencies first)"
  fi

  unlisted="$(bash "$DRIVER" rerun_only "$KERNELS_BUILDRS")"
  [ -z "$unlisted" ] && ok "(e-ii) every rerun-if-env-changed literal in jammi-kernels/build.rs is manifest-accounted" \
    || bad "(e-ii) unlisted rerun-if-env-changed literal(s) in jammi-kernels/build.rs: $unlisted"

  # Negative control: the scanner itself must actually catch a genuinely
  # unlisted literal — proves (e-i)/(e-ii) are RED tests, not vacuously
  # green because the scan itself is a no-op.
  FAKE="$SANDBOX/e_fake_buildrs.rs"
  cat > "$FAKE" <<'FAKESRC'
fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=JAMMI_TOTALLY_NEW_VAR");
    let _ = std::env::var("SOME_UNLISTED_THING");
}
FAKESRC
  unlisted="$(bash "$DRIVER" all "$FAKE")"
  [ "$unlisted" = "SOME_UNLISTED_THING" ] && ok "(e-i) negative control: the all-literal scanner catches a genuinely unlisted var" \
    || bad "(e-i) negative control FAILED — expected exactly 'SOME_UNLISTED_THING', got: $unlisted"
  unlisted="$(bash "$DRIVER" rerun_only "$FAKE")"
  [ "$unlisted" = "JAMMI_TOTALLY_NEW_VAR" ] && ok "(e-ii) negative control: the rerun-only scanner catches a genuinely unlisted announced var" \
    || bad "(e-ii) negative control FAILED — expected exactly 'JAMMI_TOTALLY_NEW_VAR', got: $unlisted"
}

# ═════════════════════════════════════════════════════════════════════════
# (f) push excludes — pinned + single-sourced
# ═════════════════════════════════════════════════════════════════════════
{
  EXPECTED='.claude
.sccache
.gpu-pull
scratchpad
target
.git
.venv*
crates/jammi-kernels/third_party/cutlass'
  ACTUAL="$(bash "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" excludes)"
  [ "$ACTUAL" = "$EXPECTED" ] && ok "(f) pod_push_stamp.sh excludes list is exactly the contracted set" \
    || bad "(f) excludes list drifted — expected:\n$EXPECTED\ngot:\n$ACTUAL"

  if grep -q 'pod_push_stamp\.sh" excludes' "$REPO_ROOT/ci/scripts/gpu-dev.sh"; then
    ok "(f) gpu-dev.sh's push case sources its excludes from pod_push_stamp.sh (single source of truth)"
  else
    bad "(f) gpu-dev.sh's push case does not call pod_push_stamp.sh excludes — the exclude set could drift from the stamp's own manifest hash"
  fi

  # manifest_sha256 reacts to a change under the exclude set's scope and
  # ignores one outside it (functional proof, not just a pinned string).
  SB="$SANDBOX/f_repo"
  mkdir -p "$SB/target"
  ( cd "$SB" && git init -q && git config user.email a@b.c && git config user.name t \
      && echo hi > tracked.txt && echo junk > target/junk && git add tracked.txt && git commit -q -m init )
  before="$(bash "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" compute "$SB" s1 | grep manifest_sha256)"
  echo more >> "$SB/target/junk"
  after_excluded="$(bash "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" compute "$SB" s1 | grep manifest_sha256)"
  echo more >> "$SB/tracked.txt"
  after_included="$(bash "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" compute "$SB" s1 | grep manifest_sha256)"
  if [ "$before" = "$after_excluded" ] && [ "$before" != "$after_included" ]; then
    ok "(f) manifest_sha256 ignores excluded-path content and reacts to included-path content"
  else
    bad "(f) manifest_sha256 did not behave as expected under the exclude set (before=$before after_excluded=$after_excluded after_included=$after_included)"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (g) no unanchored/unquoted `-t jammi`; window ops use "=name:"
# ═════════════════════════════════════════════════════════════════════════
{
  GPU_DEV="$REPO_ROOT/ci/scripts/gpu-dev.sh"
  # An unanchored/unquoted literal `-t jammi`/`-s jammi` (used AS the whole
  # session name "jammi", not as a prefix of "jammi-<tree>") would be the
  # shipped bug this closes (gpu-dev.sh:273/:439 pre-fix). Every tmux
  # `-t`/`-s` target in the fixed file is either `"=...` (quoted,
  # =-anchored) or a variable expansion — never the bare literal. A small
  # Python scan (not a hand-escaped shell regex) for the exact token
  # sequence `-t`/`-s`, whitespace, then the bareword `jammi` with nothing
  # session-name-shaped glued onto it.
  unanchored="$(python3 - "$GPU_DEV" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
# A bare `-t jammi` / `-s jammi` token: flag jammi/-s followed by literal
# "jammi" then a character that is NOT part of a longer session name
# (letters/digits/_/-/.) and NOT immediately preceded by '='. Comment-only
# lines (prose ABOUT the rule necessarily contains the literal string being
# described) are excluded, same as the (h) manifest-site count below.
bad = []
for i, line in enumerate(lines, start=1):
    if line.strip().startswith("#"):
        continue
    for m in re.finditer(r'-[ts]\s+jammi(?![A-Za-z0-9_.\-])', line):
        start = m.start()
        if start > 0 and line[start - 1] == '=':
            continue
        bad.append(str(i))
print(",".join(bad))
PY
)"
  if [ -z "$unanchored" ]; then
    ok "(g) no unanchored '-t/-s jammi' literal remains in gpu-dev.sh"
  else
    bad "(g) an unanchored '-t/-s jammi' (no '=' anchor) remains in gpu-dev.sh at line(s): $unanchored"
  fi
  # Every window-level tmux op (set-option -w, and any future window/pane
  # op) must target "=<session>:" — the ':' after the =-anchored session is
  # what makes it a WINDOW target rather than a session target.
  window_ops="$(grep -n 'tmux set-option -w' "$GPU_DEV")"
  if [ -n "$window_ops" ] && ! printf '%s\n' "$window_ops" | grep -qvE '"=[^"]+:"'; then
    ok "(g) every tmux window-level op targets \"=<session>:\""
  else
    bad "(g) a tmux window-level op does not target \"=<session>:\" — $window_ops"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (h) exactly two "/root/jammi-ai" literal sites in runpod_lib.sh (excluding
# comment lines); zero in dev-gpu-recipes.md
# ═════════════════════════════════════════════════════════════════════════
{
  LIB="$REPO_ROOT/ci/scripts/runpod_lib.sh"
  # Strip full-line comments (a line whose trimmed content starts with '#')
  # before counting — a doc comment DESCRIBING the two-site rule necessarily
  # contains the literal string itself in prose, and must not count as a
  # third "site".
  count="$(grep -v -E '^[[:space:]]*#' "$LIB" | grep -c -F '/root/jammi-ai')"
  if [ "$count" -eq 2 ]; then
    ok "(h) exactly two non-comment '/root/jammi-ai' literal sites in runpod_lib.sh"
  else
    bad "(h) expected exactly 2 non-comment '/root/jammi-ai' sites in runpod_lib.sh, got ${count}"
  fi

  RECIPES="$REPO_ROOT/docs/maintainer/dev-gpu-recipes.md"
  count_recipes="$(grep -c -F '/root/jammi-ai' "$RECIPES" 2>/dev/null || true)"
  count_recipes="${count_recipes:-0}"
  if [ "$count_recipes" -eq 0 ]; then
    ok "(h) zero '/root/jammi-ai' literal sites in dev-gpu-recipes.md"
  else
    bad "(h) expected zero '/root/jammi-ai' sites in dev-gpu-recipes.md, got ${count_recipes}"
  fi
}

echo
echo "test_pod_substrate: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
[ "$FAIL" -eq 0 ]
