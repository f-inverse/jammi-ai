#!/usr/bin/env bash
# needs: cargo-registry, jq, rsync, tmux, flock, shasum, en-us-locale
# Mocks-only, no-network regression suite for the pod build substrate:
# the seed/clone target dirs, the timing lock, the seed-cache key manifest
# and the pushed-tree stamp. Its first legs, (a)-(h):
#
#   (a) rp_bootstrap's generated /root/.jammi_env carries
#       CARGO_BUILD_RUSTC_WRAPPER= and CARGO_INCREMENTAL=0, and NOWHERE in
#       runpod_lib.sh does SCCACHE_* or rp_s3_load appear (a measured
#       S3-backed sccache gave zero cross-target-dir reuse on this image and
#       cost ~+33% wall).
#   (b) pod_target_clone.sh refuses (exit 3) without a seed marker; clones
#       correctly with a marker present; --verify on a fixture `cargo build
#       -v` log correctly distinguishes a Fresh-jammi-* poisoned clone from a
#       clean one.
#   (c) rp_s3_load is entirely absent from runpod_lib.sh (no S3-backed
#       sccache mechanism exists, disabled or otherwise).
#   (d) pod_timing_lock.sh's flock exclusivity: two concurrent non-blocking
#       acquires -> exactly one 0 and one 75; the holder dying (kill -9)
#       frees the lock for the next acquirer; a tmux-detached job holds the
#       lock against an outsider for its lifetime. Needs flock (util-linux)
#       and tmux, which the guard declares.
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
# Hermetic: no network, no GPU, no real RunPod account. Host tools the
# suite runs for real (flock, tmux, rsync, jq, shasum, the en_US.UTF-8
# locale, a warm cargo registry) are this suite's declared needs (its
# `# needs:` line, ci/needs.toml); a leg whose tool is absent fails naming
# it, never skips.
# The `(q/real-build)` leg runs a real, tiny, offline `cargo build`/`cargo
# build --release`/`cargo clean` cycle (a two-member scratch workspace
# under `mktemp -d`; Cargo resolves nothing beyond std), and
# `(w/cutlass tarball-golden)` runs `cargo package --list --offline`;
# everything else avoids cargo build/test/clippy entirely (stubbed or
# structural). The build cycle also
# unsets RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS/CARGO_BUILD_RUSTFLAGS before
# invoking cargo: those are process env vars, not directory-scoped like
# .cargo/config.toml, so the fixture would otherwise inherit whatever
# linker/warnings posture the CALLING job exports (e.g. a bare
# ubuntu-latest guard runner's own -fuse-ld=mold RUSTFLAGS, which fails to
# link when mold is not on that runner's PATH — mold only ships inside the
# jammi-ai-ci container image other jobs build in).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL - $*"; }

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
  # REPO_ROOT (this actual jammi-ai checkout) as the tree-dir — a real cargo
  # workspace for the member-freedom check to resolve
  # `cargo metadata` against; the fixture seed/dest carry no jammi-named
  # artifacts, so the check passes.
  bash "$CLONE_SH" "$SEED" "$DEST" "$REPO_ROOT" > "$SANDBOX/b2.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$DEST/release/thing" ] && [ "$(cat "$DEST/release/thing")" = "seed-artifact" ] && [ -f "$SEED/release/thing" ]; then
    ok "(b) pod_target_clone.sh clones content faithfully and leaves the seed intact (no deletion step)"
  else
    bad "(b) clone did not reproduce seed content or removed the seed (rc=$rc); output: $(cat "$SANDBOX/b2.out")"
  fi

  # The clone stamps a marker INSIDE the destination recording the seed
  # dir, the seed's own completion-marker content (mtime + sha256), and the
  # clone timestamp. gpu-dev.sh `run`'s preflight reads exactly this file's
  # PRESENCE; this test also cross-checks its CONTENT is honest (never just
  # "the file exists").
  CLONE_MARKER="$DEST/.jammi-clone-of-seed"
  if [ -f "$CLONE_MARKER" ]; then
    marker_check="$(python3 -c '
import hashlib, json, sys
seed_dir, complete_marker, dest_dir, marker_path = sys.argv[1:5]
with open(marker_path) as f:
    m = json.load(f)
with open(complete_marker, "rb") as f:
    expected_sha = hashlib.sha256(f.read()).hexdigest()
ok = (
    m.get("seed_dir") == seed_dir
    and m.get("seed_complete_marker") == complete_marker
    and m.get("seed_complete_marker_sha256") == expected_sha
    and m.get("dest_dir") == dest_dir
    and bool(m.get("seed_complete_marker_mtime"))
    and bool(m.get("clone_timestamp"))
)
print("OK" if ok else "MISMATCH: " + json.dumps(m))
' "$SEED" "${SEED}.jammi-seed-complete" "$DEST" "$CLONE_MARKER")"
    if [ "$marker_check" = "OK" ]; then
      ok "(b/clone-marker) pod_target_clone.sh stamps \$DEST/.jammi-clone-of-seed with honest seed_dir/seed-marker-sha256/dest_dir/timestamps"
    else
      bad "(b/clone-marker) clone marker content is wrong: $marker_check"
    fi
  else
    bad "(b/clone-marker) pod_target_clone.sh did not stamp $CLONE_MARKER on a successful clone"
  fi

  # Snapshot the marker's own bytes BEFORE the
  # refusal attempt below, so "survives" is a real before/after comparison,
  # never a file compared against itself.
  CLONE_MARKER_BEFORE="$(cat "$CLONE_MARKER" 2>/dev/null)"

  # Refuses to clone onto an existing destination.
  bash "$CLONE_SH" "$SEED" "$DEST" "$REPO_ROOT" >/dev/null 2>"$SANDBOX/b3.err"
  rc=$?
  if [ "$rc" -eq 2 ]; then
    ok "(b) pod_target_clone.sh refuses to clone over an existing destination"
  else
    bad "(b) expected exit 2 cloning over an existing destination (got rc=$rc)"
  fi

  # The refusal path exits before `cp`/the marker-stamp step even run, so
  # the marker from the first successful clone must survive byte-for-byte.
  if [ -f "$CLONE_MARKER" ] && [ "$(cat "$CLONE_MARKER")" = "$CLONE_MARKER_BEFORE" ]; then
    ok "(b/clone-marker) the existing-destination refusal path leaves the prior successful clone's marker untouched"
  else
    bad "(b/clone-marker) the existing-destination refusal path unexpectedly mutated or removed the clone marker"
  fi

  # A POISONED seed (a fake .fingerprint entry named after
  # a REAL workspace member, jammi-bench) must be refused — unconditionally,
  # never opt-in — at clone time, and the poisoned clone removed rather
  # than left behind for a caller to discover later.
  POISON_SEED="$SANDBOX/b_poison_seed"
  rm -rf "$POISON_SEED"
  mkdir -p "$POISON_SEED/debug/.fingerprint/jammi-bench-deadbeef"
  echo x > "$POISON_SEED/debug/.fingerprint/jammi-bench-deadbeef/lib-jammi-bench.json"
  : > "${POISON_SEED}.jammi-seed-complete"
  POISON_DEST="$SANDBOX/b_poison_dest"
  rm -rf "$POISON_DEST"
  bash "$CLONE_SH" "$POISON_SEED" "$POISON_DEST" "$REPO_ROOT" > "$SANDBOX/b6.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && [ ! -e "$POISON_DEST" ] && grep -q 'jammi-bench-deadbeef' "$SANDBOX/b6.out"; then
    ok "(b/member-freedom) a clone of a POISONED seed (fake .fingerprint/jammi-bench-deadbeef/) is refused unconditionally and the poisoned clone removed"
  else
    bad "(b/member-freedom) expected a poisoned seed's clone to be refused and removed (rc=$rc, dest exists=$([ -e "$POISON_DEST" ] && echo yes || echo no)): $(cat "$SANDBOX/b6.out")"
  fi

  # --adopt (the executable remedy for `run`'s UNMARKED_WARM refusal): a
  # target dir that is already WARM but carries no marker — one built
  # before the marker scheme existed, or by any path other than the
  # `target` verb — is stamped in place. The clone path cannot serve this
  # case at all: it refuses an existing destination (pinned right above),
  # which is exactly why the old refusal's remedy could not execute.
  #
  # Adoption is CHECKED, not a rubber stamp: it runs the SAME content
  # validation every clone runs (pod_seed_assert_member_free, off the real
  # workspace member list resolved from REPO_ROOT) and demands the OPPOSITE
  # answer — member artifacts MUST be present, because those artifacts are
  # the warmth being claimed.
  ADOPT_WARM="$SANDBOX/b_adopt_warm"
  rm -rf "$ADOPT_WARM"
  mkdir -p "$ADOPT_WARM/debug/.fingerprint/jammi-bench-deadbeef"
  echo x > "$ADOPT_WARM/debug/.fingerprint/jammi-bench-deadbeef/lib-jammi-bench.json"
  bash "$CLONE_SH" "$SEED" "$ADOPT_WARM" "$REPO_ROOT" --adopt > "$SANDBOX/b7.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$ADOPT_WARM/.jammi-clone-of-seed" ]; then
    ok "(b/adopt) --adopt stamps the marker on an EXISTING warm target dir the clone path refuses outright"
  else
    bad "(b/adopt) expected --adopt to accept and stamp a warm dir (rc=$rc): $(cat "$SANDBOX/b7.out")"
  fi
  # The marker is HONEST about what it is: an adoption, with no seed
  # provenance to claim (an adopted dir came from no seed).
  adopt_marker_check="$(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
print("OK" if (m.get("adopted") is True and m.get("seed_dir") is None
                and m.get("dest_dir") == sys.argv[2]
                and bool(m.get("adopted_timestamp"))) else "MISMATCH: " + json.dumps(m))
' "$ADOPT_WARM/.jammi-clone-of-seed" "$ADOPT_WARM" 2>&1)"
  [ "$adopt_marker_check" = "OK" ] \
    && ok "(b/adopt) the adoption marker records adopted=true with NO fabricated seed provenance" \
    || bad "(b/adopt) adoption marker content is wrong: $adopt_marker_check"
  # The warm dir's own content is untouched — adoption copies and deletes
  # nothing.
  [ -f "$ADOPT_WARM/debug/.fingerprint/jammi-bench-deadbeef/lib-jammi-bench.json" ] \
    && ok "(b/adopt) adoption leaves the warm dir's own artifacts in place (nothing copied, nothing deleted)" \
    || bad "(b/adopt) adoption must not disturb the warm dir's existing artifacts"
  # A genuinely COLD dir (a real cargo profile tree with no member
  # artifacts) is still refused — --adopt can never launder cold into
  # marked.
  ADOPT_COLD="$SANDBOX/b_adopt_cold"
  rm -rf "$ADOPT_COLD"
  mkdir -p "$ADOPT_COLD/debug/.fingerprint/serde-1234abcd"
  bash "$CLONE_SH" "$SEED" "$ADOPT_COLD" "$REPO_ROOT" --adopt > "$SANDBOX/b8.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && [ ! -f "$ADOPT_COLD/.jammi-clone-of-seed" ] && grep -q 'NO workspace-member artifacts' "$SANDBOX/b8.out"; then
    ok "(b/adopt) a COLD dir (no workspace-member artifacts) is refused and left unmarked"
  else
    bad "(b/adopt) expected a cold dir to be refused unmarked (rc=$rc): $(cat "$SANDBOX/b8.out")"
  fi
  # A dir that is not a built CARGO_TARGET_DIR at all (no debug/, no
  # release/) is a structural refusal, not a silent stamp.
  ADOPT_EMPTY="$SANDBOX/b_adopt_empty"
  rm -rf "$ADOPT_EMPTY"; mkdir -p "$ADOPT_EMPTY"
  bash "$CLONE_SH" "$SEED" "$ADOPT_EMPTY" "$REPO_ROOT" --adopt > "$SANDBOX/b9.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && [ ! -f "$ADOPT_EMPTY/.jammi-clone-of-seed" ]; then
    ok "(b/adopt) a dir with neither debug/ nor release/ is refused (structurally not a target dir), never stamped"
  else
    bad "(b/adopt) expected a structural refusal for a non-target dir (rc=$rc): $(cat "$SANDBOX/b9.out")"
  fi
  # A missing dir is refused too (there is nothing warm to adopt).
  bash "$CLONE_SH" "$SEED" "$SANDBOX/b_adopt_absent" "$REPO_ROOT" --adopt > "$SANDBOX/b10.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && [ ! -e "$SANDBOX/b_adopt_absent" ]; then
    ok "(b/adopt) --adopt against a MISSING dir is refused and creates nothing"
  else
    bad "(b/adopt) expected --adopt to refuse a missing dir and create nothing (rc=$rc): $(cat "$SANDBOX/b10.out")"
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
  # The Fresh line sits FIRST (cargo prints Fresh units before anything
  # else) with 1,320,035 B of filler AFTER it — what makes the SIGPIPE'd
  # `printf | grep -q` shape this guards against fail on EVERY run is a
  # stream past ONE PIPE BUFFER (64 KiB), not any specific "cargo log" byte
  # count, so this fixture clears it by >20x. A smaller (~70-byte) fixture
  # fails the buggy shape only when its writer is descheduled between
  # lines, so it would pass almost every run.
  DIRTY_LOG="$SANDBOX/b_dirty.log"
  {
    echo "       Fresh jammi-kernels v0.47.0"
    awk 'BEGIN{for(i=0;i<40000;i++) print "   Compiling jammi-bench v0.47.0"}'
  } > "$DIRTY_LOG"
  bash "$CLONE_SH" "" "" --verify < "$DIRTY_LOG" > "$SANDBOX/b5.out" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && grep -q 'Fresh' "$SANDBOX/b5.out"; then
    ok "(b) --verify FAILS a log carrying a Fresh jammi-* unit (poisoned-clone detection)"
  else
    bad "(b) --verify should have failed on a Fresh jammi-* line (rc=$rc): $(cat "$SANDBOX/b5.out")"
  fi

  # --verify against a genuinely UNCREATABLE/UNWRITABLE temp log, fed the
  # SAME poisoned DIRTY_LOG on stdin, must REFUSE (non-zero, naming the
  # real cause) — never silently fall through and report the poisoned
  # clone as clean. A PATH stub forcing `mktemp`/`cat` to fail (not an
  # invalid $TMPDIR: bare `mktemp` on some userlands silently falls back
  # to another directory instead of erroring, so that would not reliably
  # reproduce this) is the portable repro across GNU and BSD. Pre-fix,
  # `log_file="$(mktemp)"` and `cat > "$log_file"` were unchecked: a
  # failed mktemp left `log_file` empty, the detection `grep` then errored
  # against a nonexistent path, and the unchecked `if grep -Eq …; then`
  # read that error the SAME as "no match" — this exact DIRTY_LOG (a real
  # `Fresh jammi-*` line) reported "clone verify OK", exit 0.
  B_STUB_MKTEMP="$SANDBOX/b_stub_mktemp_fail"
  mkdir -p "$B_STUB_MKTEMP"
  cat > "$B_STUB_MKTEMP/mktemp" <<'STUB'
#!/usr/bin/env bash
exit 1
STUB
  chmod +x "$B_STUB_MKTEMP/mktemp"
  PATH="$B_STUB_MKTEMP:$PATH" bash "$CLONE_SH" "" "" --verify < "$DIRTY_LOG" > "$SANDBOX/b6.out" 2>&1
  rc=$?
  if [ "$rc" -eq 2 ] && grep -q 'could not create a temp file' "$SANDBOX/b6.out"; then
    ok "(b/mktemp-failure) --verify against an uncreatable temp log REFUSES (exit 2), never reads a poisoned log as clean"
  else
    bad "(b/mktemp-failure) expected exit 2 naming 'could not create a temp file' (rc=$rc): $(cat "$SANDBOX/b6.out")"
  fi

  B_STUB_CAT="$SANDBOX/b_stub_cat_fail"
  mkdir -p "$B_STUB_CAT"
  cat > "$B_STUB_CAT/cat" <<'STUB'
#!/usr/bin/env bash
exit 1
STUB
  chmod +x "$B_STUB_CAT/cat"
  PATH="$B_STUB_CAT:$PATH" bash "$CLONE_SH" "" "" --verify < "$DIRTY_LOG" > "$SANDBOX/b7.out" 2>&1
  rc=$?
  if [ "$rc" -eq 2 ] && grep -q 'could not write the --verify log' "$SANDBOX/b7.out"; then
    ok "(b/cat-write-failure) --verify against an unwritable temp log REFUSES (exit 2), never reads a poisoned log as clean"
  else
    bad "(b/cat-write-failure) expected exit 2 naming 'could not write the --verify log' (rc=$rc): $(cat "$SANDBOX/b7.out")"
  fi

  # --verify with the DETECTION grep itself failing (a genuine read
  # failure on the already-written, already-poisoned log_file — distinct
  # from the two fixtures above, which fail BEFORE log_file has any real
  # content). A PATH-stubbed `grep` that always exits 2 forces exactly the
  # `grep_rc` value neither "poisoned" (0) nor "clean" (1) covers; the
  # fixed script's third `if`/`elif`/`else` arm must refuse (exit 2,
  # naming the real cause) rather than falling through to "clean" the way
  # a naive `if grep_rc -eq 0 … else clean fi` (only two arms, no
  # standalone case for "other") would. Same poisoned DIRTY_LOG as input —
  # this fixture exists specifically to prove that arm is load-bearing.
  B_STUB_GREP="$SANDBOX/b_stub_grep_fail"
  mkdir -p "$B_STUB_GREP"
  cat > "$B_STUB_GREP/grep" <<'STUB'
#!/usr/bin/env bash
exit 2
STUB
  chmod +x "$B_STUB_GREP/grep"
  PATH="$B_STUB_GREP:$PATH" bash "$CLONE_SH" "" "" --verify < "$DIRTY_LOG" > "$SANDBOX/b8.out" 2>&1
  rc=$?
  if [ "$rc" -eq 2 ] && grep -q 'could not read the --verify log' "$SANDBOX/b8.out"; then
    ok "(b/grep-read-failure) --verify whose detection grep itself errors REFUSES (exit 2), never reads a poisoned log as clean"
  else
    bad "(b/grep-read-failure) expected exit 2 naming 'could not read the --verify log' (rc=$rc): $(cat "$SANDBOX/b8.out")"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (d) pod_timing_lock.sh — flock exclusivity. Linux-only (util-linux flock);
# the guard declares flock and tmux as needs.
# ═════════════════════════════════════════════════════════════════════════
{
  LOCK_SH="$REPO_ROOT/ci/scripts/pod_timing_lock.sh"
  if ! command -v flock >/dev/null 2>&1; then
    bad "(d) flock (util-linux) not found — the lock tests need it"
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
    # WHILE held (checked mid-hold, on a backgrounded slow command:
    # pod_timing_lock.sh removes the holder file on release, so a check
    # after the wrapped command returns would always see it gone).
    rm -f "$LOCKFILE" "${LOCKFILE}.holder"
    ( JAMMI_TIMING_LABEL="probe-d" JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'sleep 1' ) >/dev/null 2>&1 &
    D_HOLDER_BG=$!
    sleep 0.3
    if [ -f "${LOCKFILE}.holder" ] && grep -q '^holder=probe-d$' "${LOCKFILE}.holder"; then
      ok "(d) holder file is written (tmp+rename) under the lock with the caller's label, WHILE genuinely held"
    else
      bad "(d) holder file missing or malformed while the lock is genuinely held"
    fi
    wait "$D_HOLDER_BG" 2>/dev/null
    # The holder file must be REMOVED on release: a holder file left on
    # disk makes a downstream reader (pod_build_timings.sh's own LOCK_HELD
    # check) read "held" for every later run, whether or not the lock is
    # actually free.
    if [ ! -f "${LOCKFILE}.holder" ]; then
      ok "(d/holder-release) holder file is REMOVED on release — a subsequent reader cannot mistake a past hold for a current one"
    else
      bad "(d/holder-release) holder file survived release: $(cat "${LOCKFILE}.holder" 2>/dev/null)"
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

      # Negative control: a
      # LAUNCHER-held flock does NOT protect a tmux job. `tmux new-session
      # -d` hands the pane's command off to the tmux SERVER, which forks
      # ITS OWN child to run it — the flock-wrapped LAUNCHER (the client
      # invocation of `tmux new-session -d ...` itself) is a completely
      # separate process tree that exits almost immediately once the
      # server acknowledges the new session, releasing the lock long
      # before the detached job (still running) is done. This is exactly
      # why the acquisition has to happen INSIDE the pane's own
      # command line (the correct shape, proven above), never by wrapping
      # the launcher that merely REQUESTS the session.
      rm -f "$LOCKFILE"
      TSESS2="jammi-pod-substrate-negtest-$$"
      tmux kill-session -t "=${TSESS2}" 2>/dev/null
      # A tmux SERVER must already be running before this leg's own `tmux
      # new-session -d` client call: with NO server up, that client call
      # itself FORKS a fresh server as a side effect, and fork() duplicates
      # the launcher's already-open flock fd into that (long-lived) server
      # process too — an entirely separate, accidental way the lock could
      # end up held, unrelated to the mechanism this negative control means
      # to isolate. A throwaway keepalive session guarantees a warm server,
      # so the launcher client below is a "thin" client (message an
      # existing server, exit) with nothing left to inherit its fd.
      KEEPALIVE="jammi-pod-substrate-keepalive-$$"
      tmux kill-session -t "=${KEEPALIVE}" 2>/dev/null
      tmux new-session -d -s "$KEEPALIVE" "sleep 10"
      sleep 0.3
      JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- \
        tmux new-session -d -s "$TSESS2" "sleep 5" >/dev/null 2>&1
      sleep 0.5
      if JAMMI_TIMING_LOCK="$LOCKFILE" bash "$LOCK_SH" acquire -n -- bash -c 'exit 0' >/dev/null 2>&1; then
        ok "(d-neg) a LAUNCHER-held flock (wrapping 'tmux new-session -d' itself, not the pane's own command) does NOT protect the detached job — an outsider acquires the lock while the job is still running, confirming the launcher released it the instant the client returned"
      else
        bad "(d-neg) expected the launcher-held lock to have ALREADY released (the launcher client returns almost instantly) — an outsider should have acquired it"
      fi
      tmux kill-session -t "=${TSESS2}" 2>/dev/null
      tmux kill-session -t "=${KEEPALIVE}" 2>/dev/null

      # Assert the SHIPPED shape: gpu-dev.sh's `run --timing` dispatches the
      # pane's own command DIRECTLY to `.jammi-job.sh` (never a bare `bash
      # -c "flock ... bash job.sh"` split across two processes), and the
      # flock acquisition lives INSIDE that same generated script
      # (rp_job_wrapper_with_marker_lines, runpod_lib.sh) — i.e. INSIDE the
      # pane's own command, the correct shape the two legs above just
      # contrasted. The acquisition lives in the wrapper script, not the
      # outer LAUNCH string, so a lock refusal and the job's own real exit
      # code can never collide on the literal value 75 — see that
      # function's own doc. No hardcoded indent anchor, and `tmux
      # new-session` is confirmed to actually CONSUME the LAUNCH variable,
      # not merely to exist somewhere nearby.
      GPU_DEV_SH="$REPO_ROOT/ci/scripts/gpu-dev.sh"
      RUNPOD_LIB_SH="$REPO_ROOT/ci/scripts/runpod_lib.sh"
      launch_line="$(grep -F "LAUNCH=\"bash '\${TREE_DIR}'/.jammi-job.sh > '\${TREE_DIR}'/.jammi.log 2>&1\"" "$GPU_DEV_SH")"
      tmux_consumes_launch="$(grep -F 'tmux new-session -d -s "${TMUX_SESSION}" "${LAUNCH}"' "$GPU_DEV_SH")"
      flock_inside_wrapper="$(grep -F 'flock -n 9' "$RUNPOD_LIB_SH")"
      if [ -n "$launch_line" ] && [ -n "$tmux_consumes_launch" ] && [ -n "$flock_inside_wrapper" ]; then
        ok "(d-neg) gpu-dev.sh's shipped --timing LAUNCH dispatches directly to .jammi-job.sh, flock lives INSIDE that generated script (runpod_lib.sh), and tmux new-session actually consumes LAUNCH"
      else
        bad "(d-neg) gpu-dev.sh's --timing LAUNCH shape/consumption check failed — launch_line='${launch_line}' tmux_consumes='${tmux_consumes_launch}' flock_inside_wrapper='${flock_inside_wrapper}'"
      fi
    else
      bad "(d) tmux not found — the tmux-detached-job lock leg needs it"
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
  for f in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/bindgen_cuda-*/src/lib.rs; do
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
    bad "(e) vendored bindgen_cuda source not found under \${CARGO_HOME:-\$HOME/.cargo}/registry/src — this box's registry appears unfetched; the (e-i)/(e-ii) bindgen_cuda legs cannot run (fetch the workspace's dependencies first)"
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

  # (e-full) the RED test scope is not a hand-picked file subset — every
  # package with a build script in the RESOLVED dependency graph (`cargo
  # metadata --features jammi-kernels/cuda`), cc-dependents allowlisted at
  # the package level (derived from the SAME metadata graph, never a
  # hardcoded name list). A hand-picked subset misses CUDA-toolchain
  # build scripts such as cudarc's.
  FULL_DRIVER="$SANDBOX/probe_full_scan.sh"
  cat > "$FULL_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$SEED_TARGET_SH"
pod_seed_scan_all_vendored_buildrs "$MANIFEST" "\$1"
DRV
  chmod +x "$FULL_DRIVER"

  full_out="$(bash "$FULL_DRIVER" all 2>"$SANDBOX/e_full_all.stderr")"
  full_rc=$?
  full_summary="$(cat "$SANDBOX/e_full_all.stderr")"
  if [ "$full_rc" -eq 0 ]; then
    ok "(e-full-i) every name-shaped literal in EVERY build-script package from cargo metadata is manifest-accounted — ${full_summary#*: }"
  else
    bad "(e-full-i) unlisted literal(s) across the full vendored-package enumeration: $full_out ($full_summary)"
  fi

  full_out="$(bash "$FULL_DRIVER" rerun_only 2>"$SANDBOX/e_full_rerun.stderr")"
  full_rc=$?
  full_summary="$(cat "$SANDBOX/e_full_rerun.stderr")"
  if [ "$full_rc" -eq 0 ]; then
    ok "(e-full-ii) every rerun-if-env-changed literal across the full vendored-package enumeration is manifest-accounted — ${full_summary#*: }"
  else
    bad "(e-full-ii) unlisted rerun-if-env-changed literal(s) across the full vendored-package enumeration: $full_out ($full_summary)"
  fi

  # Negative control for the FULL scan too: prove it is not vacuously green
  # because cc-allowlisting (or the enumeration itself) swallows everything.
  FAKE_PKG_DIR="$SANDBOX/e_fake_pkg"
  mkdir -p "$FAKE_PKG_DIR"
  cat > "$FAKE_PKG_DIR/build.rs" <<'FAKESRC'
fn main() {
    let _ = std::env::var("SOME_OTHER_UNLISTED_VENDORED_THING");
}
FAKESRC
  FULL_NEG_DRIVER="$SANDBOX/probe_full_scan_neg.sh"
  cat > "$FULL_NEG_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$SEED_TARGET_SH"
NAMES="\$(mktemp)"
pod_seed_manifest_names "$MANIFEST" > "\$NAMES"
pod_seed_scan_source "$FAKE_PKG_DIR/build.rs" "\$NAMES" all
rm -f "\$NAMES"
DRV
  chmod +x "$FULL_NEG_DRIVER"
  neg_out="$(bash "$FULL_NEG_DRIVER")"
  [ "$neg_out" = "SOME_OTHER_UNLISTED_VENDORED_THING" ] \
    && ok "(e-full) negative control: a genuinely unlisted literal in a synthetic vendored-shaped package is still caught (not swallowed by cc-allowlisting/enumeration)" \
    || bad "(e-full) negative control FAILED — expected exactly 'SOME_OTHER_UNLISTED_VENDORED_THING', got: $neg_out"
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
  [ "$ACTUAL" = "$EXPECTED" ] && ok "(f) pod_push_stamp.sh excludes list is exactly the pinned set" \
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
  # session name "jammi", not as a prefix of "jammi-<tree>") would let
  # tmux prefix-match the wrong session. Every tmux
  # `-t`/`-s` target in the file is either `"=...` (quoted,
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
  if [ -n "$window_ops" ] && ! grep -qvE '"=[^"]+:"' <<<"$window_ops"; then
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

# ═════════════════════════════════════════════════════════════════════════
# (i) the build-substrate clone (a CARGO_TARGET_DIR)
# and a tree's own source checkout are DISJOINT directories, so `push --tree`
# can never delete a `target` clone; the job wrapper wires CARGO_TARGET_DIR
# to the clone.
# ═════════════════════════════════════════════════════════════════════════
{
  RUNPOD_LIB_SH="$REPO_ROOT/ci/scripts/runpod_lib.sh"
  RUNPOD_DRIVER="$SANDBOX/probe_runpod_lib.sh"
  cat > "$RUNPOD_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export RUNPOD_API_KEY=test-dummy-key
export RP_SESSION_ROOT="$SANDBOX/i_sessions"
export RP_SSH_CONFIG="$SANDBOX/i_ssh_config"
mkdir -p "\$RP_SESSION_ROOT"
# shellcheck disable=SC1091
. "$RUNPOD_LIB_SH"
"\$@"
DRV
  chmod +x "$RUNPOD_DRIVER"

  tree_dir="$(bash "$RUNPOD_DRIVER" rp_tree_dir mytree)"
  target_dir="$(bash "$RUNPOD_DRIVER" rp_target_dir mytree)"
  if [ "$tree_dir" != "$target_dir" ] && [ -n "$tree_dir" ] && [ -n "$target_dir" ]; then
    ok "(i) rp_tree_dir('mytree')=${tree_dir} and rp_target_dir('mytree')=${target_dir} are disjoint directories"
  else
    bad "(i) rp_tree_dir/rp_target_dir did not resolve to disjoint paths (tree=${tree_dir} target=${target_dir})"
  fi
  case "$target_dir" in
    "$tree_dir"/*|"$tree_dir")
      bad "(i) target_dir (${target_dir}) is nested under tree_dir (${tree_dir}) — a push --delete of the tree would still reach it" ;;
    *) ok "(i) target_dir is not nested under tree_dir (push --tree's rsync --delete cannot reach it)" ;;
  esac

  # rp_job_wrapper_lines carries the CARGO_TARGET_DIR export.
  wrapper_text="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_lines "/root/trees/mytree" "/root/target-mytree" "cargo test")"
  if grep -qF "export CARGO_TARGET_DIR='/root/target-mytree'" <<<"$wrapper_text"; then
    ok "(i) rp_job_wrapper_lines emits the CARGO_TARGET_DIR export pointing at the tree's own clone"
  else
    bad "(i) rp_job_wrapper_lines did not emit the expected CARGO_TARGET_DIR export — got: $wrapper_text"
  fi
  grep -qF "cd '/root/trees/mytree'" <<<"$wrapper_text" \
    && ok "(i) rp_job_wrapper_lines still cd's into the tree (source), not the target dir" \
    || bad "(i) rp_job_wrapper_lines did not cd into the tree dir — got: $wrapper_text"
  grep -qF "cargo test" <<<"$wrapper_text" \
    && ok "(i) rp_job_wrapper_lines carries the caller's job command verbatim" \
    || bad "(i) rp_job_wrapper_lines dropped the job command — got: $wrapper_text"

  # gpu-dev.sh's `run` case must build the job wrapper via
  # rp_job_wrapper_with_marker_lines (single source, never re-inlined) — a
  # structural check that the wiring above is actually USED, not merely
  # available. The marker-bearing variant is what lets wait-job tell a
  # job's own real completion apart from a stale log or a flock-refused
  # invocation.
  if grep -q 'rp_job_wrapper_with_marker_lines "\$TREE_DIR" "\$TARGET_DIR" "\$JOB" "\$RUN_TOKEN" "\$TIMING" "\$WAVE" "\$TREE"' "$REPO_ROOT/ci/scripts/gpu-dev.sh"; then
    ok "(i) gpu-dev.sh's run case builds its job wrapper via rp_job_wrapper_with_marker_lines"
  else
    bad "(i) gpu-dev.sh's run case does not call rp_job_wrapper_with_marker_lines with (TREE_DIR, TARGET_DIR, JOB, RUN_TOKEN, TIMING, WAVE, TREE)"
  fi

  # (i/deployment-gap) `target` must execute its OWN checkout's staged
  # pod-side scripts (`/root/.jammi-caller-scripts/...`), never the pod's
  # bootstrapped `/root/jammi-ai/ci/scripts/...` copies directly — a pod
  # booted from an older `main` would otherwise run a pod_target_clone.sh
  # that may stamp no clone marker, and the very next `run` would refuse a
  # perfectly legitimate clone. A structural check (rsync's
  # own wire protocol cannot be driven through this file's mocked-ssh
  # idiom at all — the SSH-mocked behavioral coverage for the actual rsync
  # argv lives in test_gpu_dev_lifecycle.sh's Group 11).
  GPU_DEV_SRC="$REPO_ROOT/ci/scripts/gpu-dev.sh"
  if grep -q 'bash \${STAGE_DIR}/pod_target_clone.sh' "$GPU_DEV_SRC" \
    && grep -q 'bash \${STAGE_DIR}/pod_provision_cutlass.sh' "$GPU_DEV_SRC" \
    && grep -q 'bash \${STAGE_DIR}/pod_target_clone.sh .* --verify' "$GPU_DEV_SRC" \
    && ! grep -q 'bash /root/jammi-ai/ci/scripts/pod_target_clone.sh' "$GPU_DEV_SRC" \
    && ! grep -q 'bash /root/jammi-ai/ci/scripts/pod_provision_cutlass.sh' "$GPU_DEV_SRC"; then
    ok "(i/deployment-gap) target's clone/--verify/--with-cutlass call sites all run from \${STAGE_DIR}, never gpu-dev.sh's old /root/jammi-ai/ci/scripts/... literal"
  else
    bad "(i/deployment-gap) expected every target call site to run from \${STAGE_DIR} and NONE to still hardcode /root/jammi-ai/ci/scripts/..."
  fi
  if grep -qF '"$DIR/pod_target_clone.sh" "$DIR/pod_seed_target.sh"' "$GPU_DEV_SRC" \
    && grep -qF '"$DIR/pod_provision_cutlass.sh" "$DIR/pod_push_stamp.sh"' "$GPU_DEV_SRC"; then
    ok "(i/deployment-gap) target's staging rsync ships all four THIS-checkout scripts pod_target_clone.sh needs (self-sources pod_seed_target.sh) and pod_provision_cutlass.sh needs (invokes pod_push_stamp.sh as a subprocess)"
  else
    bad "(i/deployment-gap) expected target's staging rsync to list pod_target_clone.sh, pod_seed_target.sh, pod_provision_cutlass.sh, and pod_push_stamp.sh as sources"
  fi

  # rp_job_wrapper_with_marker_lines itself: same
  # env/cd/job carriage as rp_job_wrapper_lines above, PLUS the completion
  # marker wait-job actually reads, PLUS (one-pod-per-wave, WAVE-scoped) the
  # active-wave claim write/clear.
  marker_wrapper_text="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_with_marker_lines "/root/trees/mytree" "/root/target-mytree" "cargo test" "tok123" "0" "mywave" "mytree")"
  if grep -qF "export CARGO_TARGET_DIR='/root/target-mytree'" <<<"$marker_wrapper_text"; then
    ok "(i/job-marker) rp_job_wrapper_with_marker_lines still emits the CARGO_TARGET_DIR export"
  else
    bad "(i/job-marker) rp_job_wrapper_with_marker_lines did not emit the expected CARGO_TARGET_DIR export — got: $marker_wrapper_text"
  fi
  grep -qF "rm -f '/root/trees/mytree/.jammi.exit'" <<<"$marker_wrapper_text" \
    && ok "(i/job-marker) rp_job_wrapper_with_marker_lines removes any stale .jammi.exit at the VERY START" \
    || bad "(i/job-marker) rp_job_wrapper_with_marker_lines did not remove .jammi.exit up front — got: $marker_wrapper_text"
  grep -qF '"token":"tok123"' <<<"$marker_wrapper_text" \
    && ok "(i/job-marker) rp_job_wrapper_with_marker_lines carries the caller's own token into the marker" \
    || bad "(i/job-marker) rp_job_wrapper_with_marker_lines dropped the caller's token — got: $marker_wrapper_text"
  grep -q 'flock -n 9' <<<"$marker_wrapper_text" \
    && bad "(i/job-marker) timing=0 must NOT emit a flock acquisition — got: $marker_wrapper_text" \
    || ok "(i/job-marker) timing=0 emits no flock acquisition"
  # (one-pod-per-wave) the active-wave claim is written with
  # the caller's wave/tree at the very START (alongside .jammi.exit removal)
  # and removed again at the NORMAL-completion exit path.
  grep -qF "printf '%s\\n' 'WAVE=mywave' 'TREE=mytree' 'TS=" <<<"$marker_wrapper_text" \
    && ok "(i/one-pod-per-wave) rp_job_wrapper_with_marker_lines writes the active-wave claim (wave+tree) up front" \
    || bad "(i/one-pod-per-wave) rp_job_wrapper_with_marker_lines did not write the expected active-wave claim — got: $marker_wrapper_text"
  # Every claim write and every claim removal is inside a flock critical
  # section (fd 8 on RP_CLAIM_LOCK) — a reader never sees a half-written
  # claim, and two holders launching at once never race on the store.
  unlocked="$(printf '%s\n' "$marker_wrapper_text" | grep -F '.jammi-active-wave.d/' | grep -vc 'flock 8')"
  [ "$unlocked" -eq 0 ] \
    && ok "(i/one-pod-per-wave) every claim-store statement runs inside the flock critical section" \
    || bad "(i/one-pod-per-wave) $unlocked claim-store statement(s) run OUTSIDE the flock — got: $marker_wrapper_text"
  # The claim write puts caller-supplied values in the remote printf's
  # ARGUMENT position, never its FORMAT position. With the value
  # interpolated into the format string, `--wave '%d'` makes the pod's own
  # printf consume a missing argument and write `WAVE=0` (a claim for a
  # wave nobody asked for), and a `"` closes the format string, turning the
  # remainder of the generated line into remote shell code. Driven with the
  # exotic values directly, BELOW the entrypoint allowlist, so the
  # generator is pinned independently of it.
  marker_wrapper_fmt="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_with_marker_lines "/root/trees/t" "/root/target-t" "cargo test" "tok123" "0" '%d' 'a"b')"
  claim_line="$(printf '%s\n' "$marker_wrapper_fmt" | grep -F ".claim'" | grep -vF 'rm -f')"
  case "$claim_line" in
    "( flock 8; printf '%s\n' 'WAVE=%d' 'TREE=a\"b' 'TS="*)
      ok "(i/one-pod-per-wave) a '%d' wave and a quote-bearing tree stay INERT DATA in the claim write (argument position, single-quoted)" ;;
    *)
      bad "(i/one-pod-per-wave) caller-supplied wave/tree must not reach the remote printf's format position — got: $claim_line" ;;
  esac
  # And the generated text, executed, writes those values VERBATIM — the
  # end-to-end form of the same property (no format expansion, no escape,
  # and the claim FILENAME the tree name lands in stays inert too).
  claim_sandbox="$SANDBOX/i_claimfmt"; rm -rf "$claim_sandbox"; mkdir -p "$claim_sandbox"
  printf '%s\n' "$marker_wrapper_fmt" | grep -F '.jammi-active-wave' | grep -vF 'rm -f' \
    | sed "s#/root/.jammi-active-wave#$claim_sandbox/.jammi-active-wave#g" > "$claim_sandbox/write.sh"
  bash "$claim_sandbox/write.sh"
  claim_head="$(head -2 "$claim_sandbox/.jammi-active-wave.d/a\"b.claim" 2>/dev/null)"
  if [ "$claim_head" = "$(printf 'WAVE=%%d\nTREE=a"b')" ]; then
    ok "(i/one-pod-per-wave) executing the claim write records the %d wave and the quote-bearing tree VERBATIM — no format expansion, no injection"
  else
    bad "(i/one-pod-per-wave) the executed claim write mangled its values — got: $claim_head"
  fi
  clear_count="$(printf '%s\n' "$marker_wrapper_text" | grep -cF "rm -f '/root/.jammi-active-wave.d/mytree.claim'")"
  [ "$clear_count" -ge 1 ] \
    && ok "(i/one-pod-per-wave) rp_job_wrapper_with_marker_lines clears the active-wave claim on normal completion" \
    || bad "(i/one-pod-per-wave) rp_job_wrapper_with_marker_lines never clears the active-wave claim — got: $marker_wrapper_text"
  marker_wrapper_timing="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_with_marker_lines "/root/trees/mytree" "/root/target-mytree" "cargo test" "tok123" "1" "mywave" "mytree")"
  grep -q 'flock -n 9' <<<"$marker_wrapper_timing" \
    && ok "(i/job-marker) timing=1 emits the fd-based flock acquisition INSIDE the wrapper" \
    || bad "(i/job-marker) timing=1 did not emit a flock acquisition — got: $marker_wrapper_timing"
  grep -qF '"rc":75,"lock_refused":true' <<<"$marker_wrapper_timing" \
    && ok "(i/job-marker) timing=1's refusal arm writes an rc=75/lock_refused=true marker before exiting" \
    || bad "(i/job-marker) timing=1 did not write the lock-refused marker — got: $marker_wrapper_timing"
  # (one-pod-per-wave) the lock-refused early-exit path must
  # ALSO clear the claim (the job's tmux session ends there too, even
  # though the actual job command never ran) — two `rm -f` occurrences
  # total in the timing=1 text: one on the lock-refused arm, one on normal
  # completion.
  clear_count_timing="$(printf '%s\n' "$marker_wrapper_timing" | grep -cF "rm -f '/root/.jammi-active-wave.d/mytree.claim'")"
  [ "$clear_count_timing" -ge 2 ] \
    && ok "(i/one-pod-per-wave) timing=1's lock-refused early exit ALSO clears this holder's claim, not only normal completion" \
    || bad "(i/one-pod-per-wave) expected the claim cleared on BOTH the lock-refused and normal-completion paths (got $clear_count_timing occurrence(s)) — got: $marker_wrapper_timing"
  # Ordering: under --timing the claim is written only AFTER the timing
  # lock is held. A run whose `flock -n 9` is REFUSED never ran a job, so
  # it must never have appeared in the store on behalf of that job.
  flock_ln="$(printf '%s\n' "$marker_wrapper_timing" | grep -n 'flock -n 9' | head -1 | cut -d: -f1)"
  claim_ln="$(printf '%s\n' "$marker_wrapper_timing" | grep -nF "> '/root/.jammi-active-wave.d/mytree.claim'" | head -1 | cut -d: -f1)"
  { [ -n "$flock_ln" ] && [ -n "$claim_ln" ] && [ "$claim_ln" -gt "$flock_ln" ]; } \
    && ok "(i/one-pod-per-wave) timing=1 writes the claim only AFTER acquiring the timing lock (claim line $claim_ln > flock line $flock_ln)" \
    || bad "(i/one-pod-per-wave) the claim write must follow the timing-lock acquisition (flock line '$flock_ln', claim line '$claim_ln') — got: $marker_wrapper_timing"

  # Co-tenancy, executed: TWO same-wave holders on different trees each
  # write their OWN claim file, both survive, and the first holder's own
  # completion path removes ONLY its own — the second holder stays visible
  # to the concurrency gate. With one shared claim file, holder B's launch
  # would overwrite holder A's claim and holder A's completion would delete
  # the claim B still relies on, leaving a busy pod reading as unclaimed.
  cot="$SANDBOX/i_cotenancy"; rm -rf "$cot"; mkdir -p "$cot"
  rebase_claim_store() { # $1=generated text -> the same text against $cot
    printf '%s\n' "${1//\/root\/.jammi-active-wave/$cot/.jammi-active-wave}"
  }
  hold_a="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_with_marker_lines "/root/trees/tree-a" "/root/target-a" ":" "tokA" "0" "wave-A" "tree-a")"
  hold_b="$(bash "$RUNPOD_DRIVER" rp_job_wrapper_with_marker_lines "/root/trees/tree-b" "/root/target-b" ":" "tokB" "0" "wave-A" "tree-b")"
  rebase_claim_store "$hold_a" | grep -F '.jammi-active-wave' | grep -vF 'rm -f' > "$cot/write-a.sh"
  rebase_claim_store "$hold_b" | grep -F '.jammi-active-wave' | grep -vF 'rm -f' > "$cot/write-b.sh"
  rebase_claim_store "$hold_a" | grep -F '.jammi-active-wave' | grep -F 'rm -f' > "$cot/clear-a.sh"
  bash "$cot/write-a.sh"; bash "$cot/write-b.sh"
  claims_after_both="$(find "$cot/.jammi-active-wave.d" -name '*.claim' 2>/dev/null | wc -l | tr -d ' ')"
  [ "$claims_after_both" = "2" ] \
    && ok "(i/one-pod-per-wave) two same-wave holders on different trees BOTH hold a claim (neither clobbers the other)" \
    || bad "(i/one-pod-per-wave) expected 2 surviving claims for two same-wave holders (got $claims_after_both)"
  bash "$cot/clear-a.sh"
  if [ ! -f "$cot/.jammi-active-wave.d/tree-a.claim" ] && [ -f "$cot/.jammi-active-wave.d/tree-b.claim" ]; then
    ok "(i/one-pod-per-wave) a holder's completion removes ONLY its own claim — the co-tenant's claim survives"
  else
    bad "(i/one-pod-per-wave) holder A's completion must remove exactly its own claim (a=$([ -f "$cot/.jammi-active-wave.d/tree-a.claim" ] && echo present || echo gone), b=$([ -f "$cot/.jammi-active-wave.d/tree-b.claim" ] && echo present || echo gone))"
  fi
  [ "$(cat "$cot/.jammi-active-wave.d/tree-b.claim" 2>/dev/null | head -2)" = "$(printf 'WAVE=wave-A\nTREE=tree-b')" ] \
    && ok "(i/one-pod-per-wave) the surviving co-tenant's claim still names ITS own wave and tree" \
    || bad "(i/one-pod-per-wave) the surviving claim's content is wrong — got: $(cat "$cot/.jammi-active-wave.d/tree-b.claim" 2>/dev/null)"

  # ── (i/build-sha) JAMMI_BUILD_SHA defaults from the push stamp, and ONLY
  # from a CLEAN one ─────────────────────────────────────────────────────
  # `push` rsyncs the working tree WITHOUT `.git`, so a bench binary built
  # on a pushed tree has no repository for build.rs's `git rev-parse`
  # fallback and bakes build_sha="unknown" — every producer's provenance
  # cross-check then refuses its output. `<tree>/.jammi-push-stamp.json`
  # already records the missing fact, so rp_job_env_lines emits a default
  # that reads it.
  #
  # The property under test is not "a sha appears" but WHICH trees get one:
  # `push` sends uncommitted work too, so `laptop_head` names the commit
  # the tree was BASED on. Exporting it for a DIRTY push would fabricate a
  # provenance — a false claim no downstream reader can detect, strictly
  # worse than "unknown". The generated text is EXECUTED here (not merely
  # grepped) against real fixture stamps, because a grep cannot tell the
  # clean branch from the dirty one.
  bsha="$SANDBOX/i_buildsha"; rm -rf "$bsha"; mkdir -p "$bsha"
  bsha_clean_digest="$(python3 -c 'import hashlib; print(hashlib.sha256(b"").hexdigest())')"
  bsha_head40="0123456789abcdef0123456789abcdef01234567"
  bsha_write_stamp() { # $1=tree_dir $2=laptop_head $3=porcelain $4=diff
    mkdir -p "$1"
    python3 - "$1/.jammi-push-stamp.json" "$2" "$3" "$4" <<'BSHAPY'
import json, sys
json.dump(
    {
        "laptop_head": sys.argv[2],
        "porcelain_sha256": sys.argv[3],
        "diff_head_sha256": sys.argv[4],
        "manifest_sha256": "fixture",
        "cutlass_gitlink": None,
        "ts": "fixture",
        "session": "fixture",
    },
    open(sys.argv[1], "w"),
)
BSHAPY
  }
  # Run the emitted preamble and report what JAMMI_BUILD_SHA ended up as.
  # $1=tree_dir; $2 (optional) = a pre-set JAMMI_BUILD_SHA the caller brings.
  bsha_resolve() {
    local text
    text="$(bash "$RUNPOD_DRIVER" rp_job_build_sha_lines "$1")"
    if [ -n "${2:-}" ]; then
      JAMMI_BUILD_SHA="$2" bash -c "${text}; printf '%s' \"\${JAMMI_BUILD_SHA:-}\"" 2>/dev/null # tripwire-ok: the emitted text writes an intentional ::warning:: to stderr on every refusal branch; this helper reports the RESOLVED VALUE only, and each caller below asserts that value explicitly — a discarded warning is never a silent pass
    else
      bash -c "${text}; printf '%s' \"\${JAMMI_BUILD_SHA:-}\"" 2>/dev/null # tripwire-ok: same as the branch above
    fi
  }

  bsha_write_stamp "$bsha/clean" "$bsha_head40" "$bsha_clean_digest" "$bsha_clean_digest"
  [ "$(bsha_resolve "$bsha/clean")" = "$bsha_head40" ] \
    && ok "(i/build-sha) a CLEAN push stamp's laptop_head becomes the job's JAMMI_BUILD_SHA (no hand-typed sha, no build_sha=unknown)" \
    || bad "(i/build-sha) a clean push stamp did not yield its laptop_head — got: '$(bsha_resolve "$bsha/clean")'"

  # Dirty on EITHER hash: the pushed bytes are not that commit.
  bsha_write_stamp "$bsha/dirty_status" "$bsha_head40" "0000000000000000000000000000000000000000000000000000000000000000" "$bsha_clean_digest"
  [ -z "$(bsha_resolve "$bsha/dirty_status")" ] \
    && ok "(i/build-sha) a push stamp with a DIRTY working tree (porcelain) exports nothing — a fabricated provenance is worse than \"unknown\"" \
    || bad "(i/build-sha) a dirty-porcelain stamp fabricated a sha — got: '$(bsha_resolve "$bsha/dirty_status")'"
  bsha_write_stamp "$bsha/dirty_diff" "$bsha_head40" "$bsha_clean_digest" "1111111111111111111111111111111111111111111111111111111111111111"
  [ -z "$(bsha_resolve "$bsha/dirty_diff")" ] \
    && ok "(i/build-sha) a push stamp with unstaged diff content exports nothing either (both hashes are load-bearing)" \
    || bad "(i/build-sha) a dirty-diff stamp fabricated a sha — got: '$(bsha_resolve "$bsha/dirty_diff")'"

  # `laptop_head` is the literal "unknown" when the pushed repo-root was
  # not a git checkout — the one value that is shaped like a field but is
  # not a sha.
  bsha_write_stamp "$bsha/unknown_head" "unknown" "$bsha_clean_digest" "$bsha_clean_digest"
  [ -z "$(bsha_resolve "$bsha/unknown_head")" ] \
    && ok "(i/build-sha) laptop_head=\"unknown\" is not shaped like a sha and exports nothing" \
    || bad "(i/build-sha) laptop_head=unknown leaked into JAMMI_BUILD_SHA — got: '$(bsha_resolve "$bsha/unknown_head")'"

  # No stamp at all, and a corrupt one: both refuse quietly-valued (empty),
  # loudly-messaged — never a traceback, never a partial value.
  mkdir -p "$bsha/no_stamp"
  [ -z "$(bsha_resolve "$bsha/no_stamp")" ] \
    && ok "(i/build-sha) a tree with no push stamp exports nothing" \
    || bad "(i/build-sha) a stampless tree produced a sha — got: '$(bsha_resolve "$bsha/no_stamp")'"
  # What the job log says about a stampless tree depends on whether build.rs
  # has a repository to read: a git checkout resolves its own sha (a note),
  # a bare tree bakes "unknown" (the warning).
  bsha_message() { bash -c "$(bash "$RUNPOD_DRIVER" rp_job_build_sha_lines "$1")" 2>&1 >/dev/null; }
  bsha_out="$(bsha_message "$bsha/no_stamp")" \
    && grep -qF "::warning::JAMMI_BUILD_SHA left UNSET" <<<"$bsha_out" \
    && ok "(i/build-sha) a stampless tree with no git checkout warns that the build bakes \"unknown\"" \
    || bad "(i/build-sha) a stampless non-checkout tree did not warn — got: '$(bsha_message "$bsha/no_stamp")'"
  mkdir -p "$bsha/checkout"
  git -C "$bsha/checkout" init -q && git -C "$bsha/checkout" -c user.name=t -c user.email=t@t commit -q --allow-empty -m fixture
  bsha_checkout_msg="$(bsha_message "$bsha/checkout")"
  [ -z "$(bsha_resolve "$bsha/checkout")" ] && ! grep -qF "::warning::" <<<"$bsha_checkout_msg" \
    && grep -qF "$(git -C "$bsha/checkout" rev-parse HEAD)" <<<"$bsha_checkout_msg" \
    && ok "(i/build-sha) a stampless git checkout exports nothing and names the HEAD build.rs will resolve — no false \"unknown\" warning" \
    || bad "(i/build-sha) a stampless git checkout was misreported — got: '$bsha_checkout_msg'"
  mkdir -p "$bsha/corrupt"; printf 'not json' > "$bsha/corrupt/.jammi-push-stamp.json"
  [ -z "$(bsha_resolve "$bsha/corrupt")" ] \
    && ok "(i/build-sha) a corrupt push stamp exports nothing (and does not abort the job with a traceback)" \
    || bad "(i/build-sha) a corrupt push stamp produced a sha — got: '$(bsha_resolve "$bsha/corrupt")'"

  # The manual escape hatch still wins: an operator who KNOWS the pushed
  # tip (the dirty-tree case) types it and is never overridden.
  [ "$(bsha_resolve "$bsha/dirty_status" "ffffffffffffffffffffffffffffffffffffffff")" = "ffffffffffffffffffffffffffffffffffffffff" ] \
    && ok "(i/build-sha) a caller-supplied JAMMI_BUILD_SHA is never overwritten by the stamp default" \
    || bad "(i/build-sha) the stamp default clobbered a caller-supplied JAMMI_BUILD_SHA"

  # Wired into the preamble EVERY job runs through, not merely defined —
  # the same single-source discipline the CARGO_TARGET_DIR assertions above
  # check for their own line.
  grep -qF "__jammi_push_stamp='/root/trees/mytree/.jammi-push-stamp.json'" <<<"$marker_wrapper_text" \
    && ok "(i/build-sha) the real job wrapper carries the stamp-derived JAMMI_BUILD_SHA default (rp_job_env_lines, one definition)" \
    || bad "(i/build-sha) rp_job_wrapper_with_marker_lines does not carry the build-sha default — got: $marker_wrapper_text"

  # revert-RED: neuter rp_job_build_sha_lines on a SCRATCH copy of
  # runpod_lib.sh and confirm the clean-stamp case stops resolving. Without
  # this the assertions above could pass against a tree where the default
  # came from somewhere else entirely (or from the ambient environment).
  bsha_scratch_lib="$bsha/runpod_lib_neutered.sh"
  python3 - "$RUNPOD_LIB_SH" "$bsha_scratch_lib" <<'BSHAREVERTPY'
import re
import sys

src = open(sys.argv[1]).read()
# Replace the function BODY with a no-op emitter — the pre-change shape,
# where the job preamble said nothing about JAMMI_BUILD_SHA at all.
start = src.index("rp_job_build_sha_lines() {")
end = src.index("\nBUILDSHAEOF\n}", start) + len("\nBUILDSHAEOF\n}")
neutered = "rp_job_build_sha_lines() {\n  : \"${1:?needs a tree dir}\"\n}"
open(sys.argv[2], "w").write(src[:start] + neutered + src[end:])
BSHAREVERTPY
  bsha_scratch_driver="$bsha/probe_neutered.sh"
  sed "s#^\. \"$RUNPOD_LIB_SH\"\$#. \"$bsha_scratch_lib\"#" "$RUNPOD_DRIVER" > "$bsha_scratch_driver"
  bsha_reverted_text="$(bash "$bsha_scratch_driver" rp_job_build_sha_lines "$bsha/clean")"
  bsha_reverted_value="$(bash -c "${bsha_reverted_text}; printf '%s' \"\${JAMMI_BUILD_SHA:-}\"" 2>/dev/null)" # tripwire-ok: the neutered emitter is EXPECTED to produce nothing at all; the empty resolved value is the assertion right below, never a silent pass
  [ -z "$bsha_reverted_value" ] \
    && ok "(i/build-sha revert-RED) neutering rp_job_build_sha_lines on a scratch copy leaves the default unset — the same CLEAN stamp now resolves nothing, so the default is genuinely load-bearing" \
    || bad "(i/build-sha revert-RED) the neutered copy still resolved a sha ('$bsha_reverted_value') — this leg is not testing what it claims"

  # target-then-push composition, end to end, on a LOCAL fixture pod
  # filesystem (real pod_target_clone.sh, real rsync, real exclude list —
  # no ssh needed since the mechanism under test is path separation, not
  # SSH plumbing; ssh is never mocked elsewhere in this suite either).
  #
  # Both paths are REBASED from the real rp_target_dir/rp_tree_dir outputs
  # (captured above as target_dir/tree_dir, absolute /root/... paths) onto
  # the sandbox POD_ROOT, never hand-built: a hand-built pair would pass
  # even if `target` cloned into the same dir `push` writes to, and a
  # change to the real naming scheme changes what this leg tests too.
  POD_ROOT="$SANDBOX/i_podroot"
  rm -rf "$POD_ROOT"
  mkdir -p "$POD_ROOT/seed/release"
  echo "seed-artifact" > "$POD_ROOT/seed/release/thing"
  : > "$POD_ROOT/seed.jammi-seed-complete"

  CLONE_DEST="${POD_ROOT}${target_dir#/root}"
  # pod_target_clone.sh's own member-freedom check needs
  # a REAL cargo workspace to resolve `cargo metadata` against — this repo
  # itself (REPO_ROOT) is the only one available hermetically; the fixture
  # seed above carries no actual jammi-* artifacts, so the check passes.
  bash "$REPO_ROOT/ci/scripts/pod_target_clone.sh" "$POD_ROOT/seed" "$CLONE_DEST" "$REPO_ROOT" > "$SANDBOX/i_clone.out" 2>&1
  clone_rc=$?

  SRC_REPO="$SANDBOX/i_src_repo"
  rm -rf "$SRC_REPO"; mkdir -p "$SRC_REPO/target"
  ( cd "$SRC_REPO" && git init -q && git config user.email a@b.c && git config user.name t \
      && echo hi > tracked.txt && echo junk > target/junk && git add tracked.txt && git commit -q -m init )

  TREE_DEST="${POD_ROOT}${tree_dir#/root}"
  mkdir -p "$(dirname "$TREE_DEST")"
  EXCLUDE_ARGS=()
  while IFS= read -r pat; do
    [ -n "$pat" ] || continue
    EXCLUDE_ARGS+=(--exclude "$pat")
  done < <(bash "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" excludes)
  # The rsync FLAGS themselves (not just the exclude
  # list) are read from gpu-dev.sh's own real `push` command, never
  # re-typed here — a drift in gpu-dev.sh's actual flags (e.g. dropping
  # `-c` or `--no-times`) would otherwise go unnoticed by this leg.
  rsync_flags_line="$(grep -oE 'rsync -[A-Za-z]+( --[A-Za-z-]+)*' "$REPO_ROOT/ci/scripts/gpu-dev.sh" | grep -- '--delete' | head -1)"
  [ -n "$rsync_flags_line" ] || { bad "(i) could not extract gpu-dev.sh's own push rsync flags — check the grep pattern still matches"; rsync_flags_line="rsync -azc --no-times --delete"; }
  # shellcheck disable=SC2086  # intentional word-split: a small, known set of flag tokens
  rsync ${rsync_flags_line#rsync } "${EXCLUDE_ARGS[@]}" "$SRC_REPO/" "$TREE_DEST/" > "$SANDBOX/i_push.out" 2>&1
  push_rc=$?

  if [ "$clone_rc" -eq 0 ] && [ "$push_rc" -eq 0 ] && [ -f "$CLONE_DEST/release/thing" ] \
     && [ "$(cat "$CLONE_DEST/release/thing")" = "seed-artifact" ] && [ -f "$TREE_DEST/tracked.txt" ]; then
    ok "(i) target-then-push composition: the clone at ${CLONE_DEST} (derived from rp_target_dir) SURVIVES a subsequent push --tree into the disjoint ${TREE_DEST} (derived from rp_tree_dir)"
  else
    bad "(i) target-then-push composition failed (clone_rc=$clone_rc push_rc=$push_rc); clone: $(cat "$SANDBOX/i_clone.out"); push: $(cat "$SANDBOX/i_push.out")"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (j) pod_seed_target.sh's failure arm — a REAL
# subshell (not a `{ }` group whose internal `exit` kills the whole
# script), the FAILED marker + log tail, and EITHER-marker gating.
# ═════════════════════════════════════════════════════════════════════════
{
  SEED_SANDBOX="$SANDBOX/j_seed"
  rm -rf "$SEED_SANDBOX"
  mkdir -p "$SEED_SANDBOX/tree"
  ( cd "$SEED_SANDBOX/tree" && git init -q -b main && git config user.email a@b.c && git config user.name t \
      && echo x > f.txt && git add f.txt && git commit -q -m init )

  FAIL_STUBBIN="$SANDBOX/j_bin_fail"
  mkdir -p "$FAIL_STUBBIN"
  CARGO_CALLS="$SANDBOX/j_cargo_calls"
  : > "$CARGO_CALLS"
  cat > "$FAIL_STUBBIN/cargo" <<CARGOSTUB
#!/usr/bin/env bash
echo "\$*" >> "$CARGO_CALLS"
echo "simulated T1 failure" >&2
exit 1
CARGOSTUB
  chmod +x "$FAIL_STUBBIN/cargo"

  JDRIVER="$SANDBOX/probe_seed_fail.sh"
  cat > "$JDRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export PATH="$FAIL_STUBBIN:\$PATH"
export JAMMI_SEED_DIR="$SEED_SANDBOX/seed"
export JAMMI_TREE_DIR="$SEED_SANDBOX/tree"
export JAMMI_SEED_MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_target_main --no-lock "\$@"
DRV
  chmod +x "$JDRIVER"

  out1="$(bash "$JDRIVER" 2>&1)"
  rc1=$?
  FAILED_MARKER_PATH="${SEED_SANDBOX}/seed.jammi-seed-failed"
  if [ "$rc1" -ne 0 ]; then
    ok "(j) a T1 failure returns non-zero (rc=$rc1) — the failure arm is REACHABLE, not dead code behind a killed script"
  else
    bad "(j) expected non-zero exit on a simulated T1 failure, got rc=$rc1"
  fi
  if [ -f "$FAILED_MARKER_PATH" ] && grep -q "seed build FAILED" "$FAILED_MARKER_PATH"; then
    ok "(j) .jammi-seed-failed marker is written with a log tail after the failure"
  else
    bad "(j) .jammi-seed-failed marker missing or malformed after the failure"
  fi
  grep -q "seed build FAILED" <<<"$out1" \
    && ok "(j) the failure's log tail is printed to stdout" \
    || bad "(j) the failure's log tail was not printed to stdout — got: $out1"

  calls_after_first="$(wc -l < "$CARGO_CALLS" | tr -d ' ')"
  out2="$(bash "$JDRIVER" 2>&1)"
  rc2=$?
  calls_after_second="$(wc -l < "$CARGO_CALLS" | tr -d ' ')"
  if [ "$rc2" -ne 0 ] && [ "$calls_after_second" = "$calls_after_first" ]; then
    ok "(j) a second invocation WITHOUT --reseed refuses (rc=$rc2) and does not re-invoke cargo (gated on EITHER marker, not just COMPLETE)"
  else
    bad "(j) second invocation should have refused without retrying cargo (rc=$rc2, cargo calls ${calls_after_first}->${calls_after_second})"
  fi
  grep -qi "previously FAILED" <<<"$out2" \
    && ok "(j) the refusal names the prior failure, not a generic error" \
    || bad "(j) the refusal did not name the prior failure — got: $out2"

  # `--reseed` must remove a STALE
  # COMPLETE_MARKER too, not just FAILED_MARKER — a marker left in place
  # for the whole rebuild reads as "done" to any consumer checking its mere
  # EXISTENCE (pod_target_clone.sh's own gate, gpu-dev.sh's `wait-seed`).
  # Simulated by hand-writing a COMPLETE marker from an OLDER, unrelated
  # successful build, then re-invoking with --reseed against the SAME
  # always-fails cargo stub: this rebuild attempt fails too (irrelevant to
  # what is being asserted), but the marker must be gone the instant the
  # rebuild started, regardless of how it ends.
  : > "${SEED_SANDBOX}/seed.jammi-seed-complete"
  bash "$JDRIVER" --reseed >/dev/null 2>&1
  if [ ! -f "${SEED_SANDBOX}/seed.jammi-seed-complete" ]; then
    ok "(j/reseed) --reseed removes a stale COMPLETE marker at rebuild start, even though this rebuild attempt itself then fails"
  else
    bad "(j/reseed) --reseed left a stale COMPLETE marker in place — a consumer reading marker EXISTENCE alone would read a lie about a PREVIOUS build"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (k) the --no-lock re-exec passes an ARRAY, never
# a quoted conditional-empty-string — the documented default invocation (no
# --reseed) must not die on "unknown argument ''". Requires real flock
# (same gate as (d)); JAMMI_SEED_DRY_RUN short-circuits before any git/cargo
# work so this is still hermetic and fast.
# ═════════════════════════════════════════════════════════════════════════
{
  if ! command -v flock >/dev/null 2>&1; then
    bad "(k) flock not found — the re-exec dry-run test needs it"
  else
    KDRIVER="$SANDBOX/probe_seed_dryrun.sh"
    cat > "$KDRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export JAMMI_TIMING_LOCK="$SANDBOX/k.lock"
export JAMMI_SEED_DIR="$SANDBOX/k_seed"
export JAMMI_TREE_DIR="$SANDBOX/k_tree"
export JAMMI_SEED_DRY_RUN=1
export JAMMI_SEED_LOCK_WAIT_SECS=5
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_target_main
DRV
    chmod +x "$KDRIVER"
    rm -f "$SANDBOX/k.lock"
    out="$(bash "$KDRIVER" 2>&1)"
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q "dry-run: args parsed OK" <<<"$out"; then
      ok "(k) the documented default invocation (no args) re-execs through pod_timing_lock.sh and reaches the inner script with no argument-parsing error"
    else
      bad "(k) the default (no-args) re-exec path failed (rc=$rc): $out"
    fi

    # Negative control: pod_seed_target.sh's own arg loop genuinely REJECTS
    # an empty-string argument (`*) echo "::error::unknown argument..."`),
    # so a quoted `"$(cond && echo --reseed)"` shape (which passes exactly
    # one empty-string arg when cond is false) would be fatal on the
    # documented default invocation — the array form above is load-bearing.
    NEG_DRIVER="$SANDBOX/probe_seed_argloop_neg.sh"
    cat > "$NEG_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_target_main --no-lock ""
DRV
    chmod +x "$NEG_DRIVER"
    neg_out="$(bash "$NEG_DRIVER" 2>&1)"
    neg_rc=$?
    if [ "$neg_rc" -eq 2 ] && grep -q "unknown argument ''" <<<"$neg_out"; then
      ok "(k) negative control: passing a literal empty-string argument (what a quoted conditional shape produces) is genuinely rejected by the arg loop — the array form avoids a real failure mode"
    else
      bad "(k) negative control did not reproduce the empty-string rejection (rc=$neg_rc): $neg_out"
    fi
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (l) pod_seed_check_stdout_subset's cross-check must not be vacuous: an
# unlisted var -> RED; an EMPTY capture dir -> RED (an empty glob must
# never fall through with bad=0); a conforming capture -> GREEN; a
# zero-byte capture file beside a real one -> GREEN.
# ═════════════════════════════════════════════════════════════════════════
{
  SEED_TARGET_SH="$REPO_ROOT/ci/scripts/pod_seed_target.sh"
  MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
  SUBSET_DRIVER="$SANDBOX/probe_subset.sh"
  cat > "$SUBSET_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$SEED_TARGET_SH"
pod_seed_check_stdout_subset "\$1" "$MANIFEST"
DRV
  chmod +x "$SUBSET_DRIVER"

  # Leg 1: an unlisted var -> RED.
  SUBSET_UNLISTED="$SANDBOX/subset_unlisted"
  mkdir -p "$SUBSET_UNLISTED"
  printf 'cargo:rerun-if-env-changed=NVCC\ncargo:rerun-if-env-changed=JAMMI_N4_UNLISTED_VAR\n' > "$SUBSET_UNLISTED/release__jammi-kernels-abc.output"
  out="$(bash "$SUBSET_DRIVER" "$SUBSET_UNLISTED" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && grep -q 'JAMMI_N4_UNLISTED_VAR' <<<"$out"; then
    ok "(l/stdout-subset) an unlisted announced var reddens the cross-check"
  else
    bad "(l/stdout-subset) expected RED naming JAMMI_N4_UNLISTED_VAR (rc=$rc): $out"
  fi

  # Leg 2: an EMPTY capture dir -> RED (bash leaves 'dir/*' unexpanded
  # when nothing matches, `[ -f ]` fails, the loop body never runs, and a
  # naive bad=0 would stamp a seed complete having checked nothing).
  SUBSET_EMPTY="$SANDBOX/subset_empty"
  rm -rf "$SUBSET_EMPTY"; mkdir -p "$SUBSET_EMPTY"
  out="$(bash "$SUBSET_DRIVER" "$SUBSET_EMPTY" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && grep -q 'capture_count=0' <<<"$out"; then
    ok "(l/stdout-subset) an EMPTY capture dir reddens the cross-check (capture_count=0), never a silent pass"
  else
    bad "(l/stdout-subset) expected RED naming capture_count=0 for an empty capture dir (rc=$rc): $out"
  fi

  # Leg 3: conforming (real manifest names only) -> GREEN.
  SUBSET_CONFORM="$SANDBOX/subset_conform"
  rm -rf "$SUBSET_CONFORM"; mkdir -p "$SUBSET_CONFORM"
  printf 'cargo:rerun-if-env-changed=NVCC\ncargo:rerun-if-env-changed=CUDA_HOME\n' > "$SUBSET_CONFORM/release__jammi-kernels-abc.output"
  out="$(bash "$SUBSET_DRIVER" "$SUBSET_CONFORM" 2>&1)"; rc=$?
  [ "$rc" -eq 0 ] && ok "(l/stdout-subset) a conforming capture (real manifest names, non-empty) passes cleanly" \
    || bad "(l/stdout-subset) a conforming capture unexpectedly failed (rc=$rc): $out"

  # Leg 4: a captured-but-EMPTY (zero-byte) `output` file is a LEGITIMATE
  # "the build script announced nothing" state (cargo writes this file for
  # every build script it runs, REGARDLESS of whether it prints anything),
  # never a defect on its own. A real seed build on this workspace's own
  # `--features jammi-kernels/cuda` graph captures several (chrono-tz,
  # rustls, snap, ...), so a per-file empty-is-an-error rule would abort
  # every real seed. The revert-RED below proves such a rule does fire on
  # this fixture, so the GREEN here is not vacuous.
  SUBSET_EMPTYFILE="$SANDBOX/subset_emptyfile"
  rm -rf "$SUBSET_EMPTYFILE"; mkdir -p "$SUBSET_EMPTYFILE"
  printf 'cargo:rerun-if-env-changed=NVCC\n' > "$SUBSET_EMPTYFILE/release__jammi-kernels-abc.output"
  : > "$SUBSET_EMPTYFILE/release__chrono-tz-def.output"
  out="$(bash "$SUBSET_DRIVER" "$SUBSET_EMPTYFILE" 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ]; then
    ok "(l/stdout-subset) a legitimately zero-byte captured output file (alongside a real one) passes cleanly — cargo writes this file for every build script regardless of whether it prints anything"
  else
    bad "(l/stdout-subset) a zero-byte captured output file should NOT redden the cross-check on its own (rc=$rc): $out"
  fi

  # revert-RED: a per-file "zero bytes -> error" rule, run against the
  # SAME fixture.
  SUBSET_OLD_CHECK_SCRIPT="$SANDBOX/subset_old_check.sh"
  cat > "$SUBSET_OLD_CHECK_SCRIPT" <<'DRV'
#!/usr/bin/env bash
set -uo pipefail
bad=0
for f in "$1"/*; do
  [ -f "$f" ] || continue
  if [ ! -s "$f" ]; then
    echo "OLD-RULE: captured build-script output file is EMPTY: $f"
    bad=1
  fi
done
exit "$bad"
DRV
  chmod +x "$SUBSET_OLD_CHECK_SCRIPT"
  old_out="$(bash "$SUBSET_OLD_CHECK_SCRIPT" "$SUBSET_EMPTYFILE" 2>&1)"; old_rc=$?
  if [ "$old_rc" -ne 0 ] && grep -q 'chrono-tz' <<<"$old_out"; then
    ok "(l/stdout-subset revert-RED) a per-file empty-is-an-error rule genuinely DOES fire on the SAME zero-byte fixture — the GREEN above is not a vacuous no-op"
  else
    bad "(l/stdout-subset revert-RED) the old-rule reproduction did not fire as expected (rc=$old_rc): $old_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (m) pod_push_cutlass_matches — the stamp's
# cutlass_gitlink must be recorded and a mismatch against the actual
# submodule commit must refuse. This is the SAME script (pod_push_stamp.sh)
# `target --with-cutlass` invokes remotely, never a second copy.
# ═════════════════════════════════════════════════════════════════════════
{
  PUSH_STAMP_SH="$REPO_ROOT/ci/scripts/pod_push_stamp.sh"

  # pod_push_compute against THIS repo actually records a cutlass_gitlink
  # (the submodule really is pinned at crates/jammi-kernels/third_party/cutlass).
  M_STAMP="$SANDBOX/m_stamp.json"
  bash "$PUSH_STAMP_SH" compute "$REPO_ROOT" m-session > "$M_STAMP" 2>"$SANDBOX/m_compute.err"
  m_gitlink="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cutlass_gitlink") or "")' "$M_STAMP" 2>/dev/null)"
  if [ -n "$m_gitlink" ]; then
    ok "(m/cutlass-stamp) pod_push_stamp.sh compute records a real cutlass_gitlink (${m_gitlink}) from THIS repo's own submodule pin"
  else
    bad "(m/cutlass-stamp) pod_push_stamp.sh compute did not record a cutlass_gitlink against this repo — stamp: $(cat "$M_STAMP" 2>/dev/null); stderr: $(cat "$SANDBOX/m_compute.err")"
  fi

  # cutlass-check: matching sha -> 0.
  bash "$PUSH_STAMP_SH" cutlass-check "$M_STAMP" "$m_gitlink" >/dev/null 2>&1
  [ $? -eq 0 ] && ok "(m/cutlass-stamp) cutlass-check PASSES when the actual sha matches the stamp" \
    || bad "(m/cutlass-stamp) cutlass-check unexpectedly failed on a matching sha"

  # cutlass-check: mismatched sha -> 1, refusal, both shas printed.
  m_out="$(bash "$PUSH_STAMP_SH" cutlass-check "$M_STAMP" "0000000000000000000000000000000000000000" 2>&1)"
  m_rc=$?
  if [ "$m_rc" -eq 1 ] && grep -q "$m_gitlink" <<<"$m_out" && grep -q '0000000000000000000000000000000000000000' <<<"$m_out"; then
    ok "(m/cutlass-stamp) cutlass-check REFUSES a mismatched sha and names BOTH shas"
  else
    bad "(m/cutlass-stamp) expected a refusal naming both shas (rc=$m_rc): $m_out"
  fi

  # cutlass-check: missing/stale stamp (no cutlass_gitlink field at all) -> 2.
  M_STALE_STAMP="$SANDBOX/m_stale_stamp.json"
  printf '{"laptop_head":"abc"}' > "$M_STALE_STAMP"
  bash "$PUSH_STAMP_SH" cutlass-check "$M_STALE_STAMP" "$m_gitlink" >/dev/null 2>&1
  [ $? -eq 2 ] && ok "(m/cutlass-stamp) cutlass-check returns 2 (not usable) for a stale stamp with no cutlass_gitlink field" \
    || bad "(m/cutlass-stamp) expected exit 2 for a stamp with no cutlass_gitlink field"

  # gpu-dev.sh's target --with-cutlass remote block must actually DELEGATE
  # to pod_provision_cutlass.sh (never a heredoc-embedded copy inside
  # gpu-dev.sh — see that file's own module doc for why), which
  # in turn must actually CALL cutlass-check (never a re-implemented
  # comparison) — a structural check that the wiring is used, not merely
  # available (ssh is never mocked, so the remote invocation itself is not
  # exercised end to end here; the REAL mechanism fixture is below).
  PROVISION_SH="$REPO_ROOT/ci/scripts/pod_provision_cutlass.sh"
  if grep -q 'pod_provision_cutlass.sh' "$REPO_ROOT/ci/scripts/gpu-dev.sh"; then
    ok "(m/cutlass-stamp) gpu-dev.sh's target --with-cutlass delegates to pod_provision_cutlass.sh"
  else
    bad "(m/cutlass-stamp) gpu-dev.sh's target --with-cutlass does not delegate to pod_provision_cutlass.sh"
  fi
  if grep -q 'pod_push_stamp.sh cutlass-check' "$PROVISION_SH"; then
    ok "(m/cutlass-stamp) pod_provision_cutlass.sh invokes pod_push_stamp.sh cutlass-check"
  else
    bad "(m/cutlass-stamp) pod_provision_cutlass.sh does not call pod_push_stamp.sh cutlass-check"
  fi

  # `cp -a` of the cutlass submodule copies its own
  # `.git` gitlink FILE into the destination tree, which is a real git
  # checkout of its own — an un-registered, foreign `.git` nested inside it
  # makes `git status`/`git add` at the tree's root fail. Structural (the
  # remote heredoc only runs on a live pod) for the SOURCE TEXT: the
  # gitlink must be stripped from the destination and its absence asserted,
  # never left in place; EXECUTABLE against a real fixture for the actual
  # mechanism (pod_provision_cutlass.sh's own real bytes, never a
  # reimplementation) to prove `cp -a` really does copy `.git` and the
  # strip really does remove it while preserving real content.
  if grep -q 'rm -rf "\${TREE_SOURCE_DIR:?}/\${CUTLASS_PATH:?}/.git"' "$PROVISION_SH" \
     && grep -q '\[ -e "\$TREE_SOURCE_DIR/\$CUTLASS_PATH/.git" \]' "$PROVISION_SH"; then
    ok "(m/strip-gitlink) pod_provision_cutlass.sh strips the copied .git gitlink and asserts its absence"
  else
    bad "(m/strip-gitlink) pod_provision_cutlass.sh does not strip+assert-absent the copied cutlass/.git"
  fi

  STRIP_SRC="$SANDBOX/strip_cutlass_src"
  STRIP_DEST="$SANDBOX/strip_tree_dest/crates/jammi-kernels/third_party/cutlass"
  rm -rf "$STRIP_SRC" "$SANDBOX/strip_tree_dest"
  mkdir -p "$STRIP_SRC"
  echo "gitdir: /root/jammi-ai/.git/modules/crates/jammi-kernels/third_party/cutlass" > "$STRIP_SRC/.git"
  echo "real header content" > "$STRIP_SRC/header.h"
  mkdir -p "$(dirname "$STRIP_DEST")"
  cp -a "$STRIP_SRC" "$STRIP_DEST"
  if [ -e "$STRIP_DEST/.git" ]; then
    ok "(m/strip-gitlink) negative control: cp -a genuinely DOES copy the submodule's .git gitlink (the strip is load-bearing)"
  else
    bad "(m/strip-gitlink) negative control failed: cp -a did not copy .git — the fixture itself is wrong"
  fi
  rm -rf "$STRIP_DEST/.git"
  if [ ! -e "$STRIP_DEST/.git" ] && [ -f "$STRIP_DEST/header.h" ] && [ "$(cat "$STRIP_DEST/header.h")" = "real header content" ]; then
    ok "(m/strip-gitlink) stripping .git removes the foreign gitlink while preserving real cutlass content"
  else
    bad "(m/strip-gitlink) strip either left .git behind or damaged real content"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (m/provision) pod_provision_cutlass.sh run for REAL — not a heredoc grep —
# against a genuine two-commit submodule fixture (a real upstream cutlass
# repo with two commits, a real superproject with a real submodule, a real
# destination tree with a push stamp). Legs: match -> copy; drift ->
# remediation fetch+checkout -> re-verify OK -> copy; no submodule
# registered ("deinit") -> refuse; unreachable stamped sha -> loud
# fetch-failure refuse; emptied submodule content -> refuse; an already
# populated destination -> proceed; unreachable submodule remote -> named
# error. Plus the revert-RED: a bare `cutlass-check` command under
# `set -e` stops at MISMATCH before the remediation arm is ever reached.
# ═════════════════════════════════════════════════════════════════════════
{
  PROVISION_SH="$REPO_ROOT/ci/scripts/pod_provision_cutlass.sh"
  PROV_ROOT="$SANDBOX/prov_cutlass"
  rm -rf "$PROV_ROOT"; mkdir -p "$PROV_ROOT"

  prov_git() { git -c protocol.file.allow=always -c init.defaultBranch=main "$@"; }

  # A real upstream cutlass repo with two real commits. Includes
  # include/cutlass/cutlass.h: pod_provision_cutlass.sh's content
  # validation requires this exact path plus an on-disk file count >= the
  # pinned commit's own `git ls-tree` count, so a fixture without it would
  # trip that check on every leg below, never reaching the mechanism these
  # legs exist to exercise.
  PROV_UPSTREAM="$PROV_ROOT/cutlass-upstream"
  prov_git init -q "$PROV_UPSTREAM"
  prov_git -C "$PROV_UPSTREAM" config user.email t@t; prov_git -C "$PROV_UPSTREAM" config user.name t
  mkdir -p "$PROV_UPSTREAM/include/cutlass"
  echo v1 > "$PROV_UPSTREAM/header.h"
  echo 'v1 cutlass.h' > "$PROV_UPSTREAM/include/cutlass/cutlass.h"
  prov_git -C "$PROV_UPSTREAM" add header.h include/cutlass/cutlass.h
  prov_git -C "$PROV_UPSTREAM" commit -q -m v1
  PROV_SHA1="$(prov_git -C "$PROV_UPSTREAM" rev-parse HEAD)"
  echo v2 > "$PROV_UPSTREAM/header.h"
  echo 'v2 cutlass.h' > "$PROV_UPSTREAM/include/cutlass/cutlass.h"
  prov_git -C "$PROV_UPSTREAM" commit -q -am v2
  PROV_SHA2="$(prov_git -C "$PROV_UPSTREAM" rev-parse HEAD)"

  # prov_make_super <dir> <checked-out-sha>: a superproject (/root/jammi-ai
  # stand-in) with a real submodule pinned at the given sha.
  prov_make_super() {
    local dir="$1" sha="$2"
    rm -rf "$dir"; prov_git init -q "$dir"
    prov_git -C "$dir" config user.email t@t; prov_git -C "$dir" config user.name t
    mkdir -p "$dir/crates/jammi-kernels/third_party"
    prov_git -C "$dir" submodule add -q "$PROV_UPSTREAM" crates/jammi-kernels/third_party/cutlass >/dev/null 2>&1
    prov_git -C "$dir/crates/jammi-kernels/third_party/cutlass" checkout -q "$sha"
    prov_git -C "$dir" add crates/jammi-kernels/third_party/cutlass .gitmodules
    prov_git -C "$dir" commit -q -m "pin cutlass"
  }

  # prov_make_tree <dir> <stamp-sha-or-empty>: a destination tree with a push
  # stamp naming the given cutlass_gitlink (JSON written directly — this
  # test exercises pod_provision_cutlass.sh, not pod_push_stamp.sh compute).
  prov_make_tree() {
    local dir="$1" stamp_sha="$2"
    rm -rf "$dir"; mkdir -p "$dir"
    printf '{"cutlass_gitlink": "%s"}\n' "$stamp_sha" > "$dir/.jammi-push-stamp.json"
  }

  # ---- leg 1: MATCH -> copy, no remediation -------------------------------
  PROV_SUPER_MATCH="$PROV_ROOT/super_match"
  prov_make_super "$PROV_SUPER_MATCH" "$PROV_SHA1"
  PROV_TREE_MATCH="$PROV_ROOT/tree_match"
  prov_make_tree "$PROV_TREE_MATCH" "$PROV_SHA1"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_MATCH" "$PROV_SUPER_MATCH" 2>&1)"; prov_rc=$?
  if [ "$prov_rc" -eq 0 ] && [ -f "$PROV_TREE_MATCH/crates/jammi-kernels/third_party/cutlass/header.h" ] \
     && [ ! -e "$PROV_TREE_MATCH/crates/jammi-kernels/third_party/cutlass/.git" ] \
     && ! grep -q 'attempting to fetch' <<<"$prov_out"; then
    ok "(m/provision match) matching stamp/submodule -> copies straight through, no remediation attempted"
  else
    bad "(m/provision match) expected a clean copy with no remediation (rc=$prov_rc): $prov_out"
  fi

  # ---- leg 2: DRIFT -> remediation fetch+checkout -> re-verify OK -> copy --
  PROV_SUPER_DRIFT="$PROV_ROOT/super_drift"
  prov_make_super "$PROV_SUPER_DRIFT" "$PROV_SHA1"
  PROV_TREE_DRIFT="$PROV_ROOT/tree_drift"
  prov_make_tree "$PROV_TREE_DRIFT" "$PROV_SHA2"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_DRIFT" "$PROV_SUPER_DRIFT" 2>&1)"; prov_rc=$?
  if [ "$prov_rc" -eq 0 ] && grep -q 'attempting to fetch' <<<"$prov_out" \
     && [ -f "$PROV_TREE_DRIFT/crates/jammi-kernels/third_party/cutlass/header.h" ] \
     && [ "$(cat "$PROV_TREE_DRIFT/crates/jammi-kernels/third_party/cutlass/header.h")" = v2 ]; then
    ok "(m/provision drift) a genuine mismatch reaches the remediation arm, fetches+checks out the STAMPED commit, re-verifies OK, and copies the CORRECT (v2) content"
  else
    bad "(m/provision drift) expected remediation to run and copy v2 content (rc=$prov_rc): $prov_out"
  fi

  # ---- leg 3: no submodule registered ("deinit"-shaped) -> refuse --------
  PROV_SUPER_DEINIT="$PROV_ROOT/super_deinit"
  rm -rf "$PROV_SUPER_DEINIT"; prov_git init -q "$PROV_SUPER_DEINIT"
  prov_git -C "$PROV_SUPER_DEINIT" config user.email t@t; prov_git -C "$PROV_SUPER_DEINIT" config user.name t
  echo x > "$PROV_SUPER_DEINIT/README"; prov_git -C "$PROV_SUPER_DEINIT" add README; prov_git -C "$PROV_SUPER_DEINIT" commit -q -m x
  PROV_TREE_DEINIT="$PROV_ROOT/tree_deinit"
  prov_make_tree "$PROV_TREE_DEINIT" "$PROV_SHA1"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_DEINIT" "$PROV_SUPER_DEINIT" 2>&1)"; prov_rc=$?
  # `set -euo pipefail`'s own FIRST command (`git submodule update --init`)
  # aborts immediately with git's own "pathspec ... did not match" when NO
  # submodule is registered at all — the loudest possible refusal, before
  # this script's own "no .git after submodule update" message (reachable
  # only when `submodule update --init` itself SUCCEEDS but leaves no
  # .git, not exercised by this leg) is ever printed. Either message is a
  # genuine, loud, non-silent refusal — the fixture accepts both.
  if [ "$prov_rc" -ne 0 ] && { grep -q 'no .git after submodule update' <<<"$prov_out" \
     || grep -q 'pathspec.*did not match' <<<"$prov_out"; } \
     && [ ! -e "$PROV_TREE_DEINIT/crates/jammi-kernels/third_party/cutlass" ]; then
    ok "(m/provision deinit) no submodule registered at all -> refuses loudly, nothing copied"
  else
    bad "(m/provision deinit) expected a loud refusal naming the missing .git (rc=$prov_rc): $prov_out"
  fi

  # ---- leg 4: stamped sha unreachable -> loud fetch-failure refuse -------
  PROV_SUPER_FETCHFAIL="$PROV_ROOT/super_fetchfail"
  prov_make_super "$PROV_SUPER_FETCHFAIL" "$PROV_SHA1"
  PROV_TREE_FETCHFAIL="$PROV_ROOT/tree_fetchfail"
  prov_make_tree "$PROV_TREE_FETCHFAIL" "0000000000000000000000000000000000000000"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_FETCHFAIL" "$PROV_SUPER_FETCHFAIL" 2>&1)"; prov_rc=$?
  if [ "$prov_rc" -ne 0 ] && grep -q 'could not fetch/checkout' <<<"$prov_out" \
     && [ ! -e "$PROV_TREE_FETCHFAIL/crates/jammi-kernels/third_party/cutlass" ]; then
    ok "(m/provision fetch-failure) an unreachable stamped sha -> loud refusal, nothing copied"
  else
    bad "(m/provision fetch-failure) expected a loud fetch-failure refusal (rc=$prov_rc): $prov_out"
  fi

  # ---- leg 5: SUPER_DIR's own submodule checkout is EMPTY (a .git + a
  # resolvable HEAD sha, but the checked-out content itself deleted out
  # from under it, e.g. by another tree's push) -> refuse loudly,
  # content-validated BEFORE ever reading a stamp or attempting a copy. ---
  PROV_SUPER_EMPTYCONTENT="$PROV_ROOT/super_emptycontent"
  prov_make_super "$PROV_SUPER_EMPTYCONTENT" "$PROV_SHA1"
  rm -f "$PROV_SUPER_EMPTYCONTENT/crates/jammi-kernels/third_party/cutlass/header.h" \
        "$PROV_SUPER_EMPTYCONTENT/crates/jammi-kernels/third_party/cutlass/include/cutlass/cutlass.h"
  PROV_TREE_EMPTYCONTENT="$PROV_ROOT/tree_emptycontent"
  prov_make_tree "$PROV_TREE_EMPTYCONTENT" "$PROV_SHA1"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_EMPTYCONTENT" "$PROV_SUPER_EMPTYCONTENT" 2>&1)"; prov_rc=$?
  if [ "$prov_rc" -ne 0 ] && grep -q 'missing include/cutlass/cutlass.h' <<<"$prov_out" \
     && [ ! -e "$PROV_TREE_EMPTYCONTENT/crates/jammi-kernels/third_party/cutlass" ]; then
    ok "(m/provision empty-content) SUPER_DIR's own submodule has a valid .git/HEAD but its checked-out content was deleted -> refused loudly, nothing copied"
  else
    bad "(m/provision empty-content) expected a loud content-validation refusal (rc=$prov_rc): $prov_out"
  fi

  # ---- leg 6: a tree whose cutlass path is ALREADY populated by an
  # EARLIER copy-provisioning call (.git-stripped, real content), the
  # precondition an in-tree `git submodule update --init` collides with.
  # pod_provision_cutlass.sh's `rm -rf` + `cp -a` never asks git to touch
  # that path, so it PROCEEDS cleanly; the revert-RED runs a bare `git
  # submodule update --init` DIRECTLY against the SAME already-populated
  # path and shows it FAILS. ----------------------------------------------
  PROV_SUPER_COLLIDE="$PROV_ROOT/super_collide"
  prov_make_super "$PROV_SUPER_COLLIDE" "$PROV_SHA1"
  # The "tree" is ITSELF git-backed with a REAL submodule reference (the
  # same shape as pod_build_timings.sh's own $JAMMI_TREE_DIR after its
  # FA2-tip checkout) — never a plain rsync-pushed stand-in — so the
  # revert-RED's `git submodule update --init` below is a faithful
  # reproduction, not a "not a git repository" false negative.
  PROV_TREE_COLLIDE="$PROV_ROOT/tree_collide"
  prov_make_super "$PROV_TREE_COLLIDE" "$PROV_SHA1"
  # First call: an EARLIER `target --with-cutlass`-shaped provisioning —
  # overwrites the tree's own (currently real submodule) cutlass path
  # with a plain, .git-stripped copy, simulating a prior copy-
  # provisioning call against this SAME git-backed tree.
  env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_COLLIDE" "$PROV_SUPER_COLLIDE" >/dev/null 2>&1
  [ -f "$PROV_TREE_COLLIDE/crates/jammi-kernels/third_party/cutlass/header.h" ] \
    && [ ! -e "$PROV_TREE_COLLIDE/crates/jammi-kernels/third_party/cutlass/.git" ] \
    || bad "(m/provision collide) setup: the first provisioning call did not leave the expected populated, .git-stripped precondition"
  # Second call (the SAME shape pod_build_timings.sh's own call takes):
  # must PROCEED despite the destination already being populated.
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_COLLIDE" "$PROV_SUPER_COLLIDE" 2>&1)"; prov_rc=$?
  if [ "$prov_rc" -eq 0 ] && [ -f "$PROV_TREE_COLLIDE/crates/jammi-kernels/third_party/cutlass/header.h" ]; then
    ok "(m/provision collide) a SECOND provisioning call against an ALREADY-populated (.git-stripped) destination PROCEEDS cleanly — the filesystem-level rm -rf + cp -a never asks git to touch that path"
  else
    bad "(m/provision collide) expected the second provisioning call to proceed despite the already-populated destination (rc=$prov_rc): $prov_out"
  fi
  # revert-RED: an in-tree `git submodule update --init`, run DIRECTLY
  # against the SAME already-populated path, fails. Strip the tree's OWN
  # `.git/modules/.../cutlass` backing repo first so this is a FIRST-EVER
  # `--init` hitting an occupied destination: a submodule already
  # initialised locally once tolerates a re-checkout over stray files
  # (git silently overwrites them and returns rc=0), while a genuinely
  # fresh `--init` clone into an occupied, non-empty, non-submodule
  # directory does not.
  rm -rf "$PROV_TREE_COLLIDE/.git/modules/crates/jammi-kernels/third_party/cutlass"
  prov_collide_revert_out="$(git -C "$PROV_TREE_COLLIDE" submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass 2>&1)"; prov_collide_revert_rc=$?
  if [ "$prov_collide_revert_rc" -ne 0 ]; then
    ok "(m/provision collide revert-RED) an in-tree 'git submodule update --init', run directly against the SAME already-populated path, FAILS — never running that command against the destination is load-bearing"
  else
    bad "(m/provision collide revert-RED) expected an in-tree git submodule update to fail against an already-populated, non-submodule path (rc=$prov_collide_revert_rc): $prov_collide_revert_out"
  fi

  # ---- leg 7: SUPER_DIR's own `submodule update
  # --init` needs network + a reachable submodule remote — an unreachable
  # remote must produce the NAMED error, never git's bare stderr with no
  # step name. -------------------------------------------------------------
  PROV_SUPER_UNREACHABLE="$PROV_ROOT/super_unreachable"
  prov_make_super "$PROV_SUPER_UNREACHABLE" "$PROV_SHA1"
  # Deinitialise the submodule so the NEXT `--init` is a genuinely fresh
  # clone attempt (not a re-checkout of an already-local backing repo —
  # same reasoning as the collide leg's own revert-RED above), then make
  # the ONLY remote it could clone from unreachable.
  prov_git -C "$PROV_SUPER_UNREACHABLE" submodule deinit -f crates/jammi-kernels/third_party/cutlass >/dev/null 2>&1
  rm -rf "$PROV_SUPER_UNREACHABLE/.git/modules/crates/jammi-kernels/third_party/cutlass"
  PROV_UPSTREAM_MOVED="$PROV_ROOT/cutlass-upstream-MOVED-away"
  mv "$PROV_UPSTREAM" "$PROV_UPSTREAM_MOVED"
  PROV_TREE_UNREACHABLE="$PROV_ROOT/tree_unreachable"
  prov_make_tree "$PROV_TREE_UNREACHABLE" "$PROV_SHA1"
  prov_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROVISION_SH" "$PROV_TREE_UNREACHABLE" "$PROV_SUPER_UNREACHABLE" 2>&1)"; prov_rc=$?
  mv "$PROV_UPSTREAM_MOVED" "$PROV_UPSTREAM"
  if [ "$prov_rc" -ne 0 ] && grep -q 'pod_provision_cutlass: submodule update failed (network/remote unreachable?)' <<<"$prov_out"; then
    ok "(m/provision unreachable-remote) SUPER_DIR's own submodule update, with its remote made unreachable, FAILS with the NAMED error (never git's bare stderr alone)"
  else
    bad "(m/provision unreachable-remote) expected the named submodule-update-failed error (rc=$prov_rc): $prov_out"
  fi

  # ---- revert-RED: the bare-command form ---------------------------------
  # A BARE `cutlass-check` command (no if/else) under `set -e`, on the SAME
  # drift fixture: the shell aborts at the MISMATCH error, `CHECK_RC` never
  # read, the remediation arm (and the copy) never reached.
  PROV_REVERT_DIR="$SANDBOX/prov_revert_dir"
  rm -rf "$PROV_REVERT_DIR"; mkdir -p "$PROV_REVERT_DIR"
  # pod_provision_cutlass.sh computes its own $DIR from its own location and
  # calls "$DIR/pod_push_stamp.sh" — a REAL copy must sit alongside the
  # reverted script for that resolution to find it (same technique the
  # (p2/member-freedom) fixture uses for pod_seed_target.sh/pod_push_stamp.sh).
  cp "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" "$PROV_REVERT_DIR/pod_push_stamp.sh"
  PROV_REVERTED="$PROV_REVERT_DIR/pod_provision_cutlass_reverted.sh"
  cp "$PROVISION_SH" "$PROV_REVERTED"
  python3 - "$PROV_REVERTED" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '''if bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA"; then
  CHECK_RC=0
else
  CHECK_RC=$?
fi
'''
new = '''bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA"
CHECK_RC=$?
'''
assert old in t, "revert fixture: could not locate the if/else block to neuter"
open(p, "w").write(t.replace(old, new))
PY
  if bash -n "$PROV_REVERTED"; then
    PROV_SUPER_REVERT="$PROV_ROOT/super_revert"
    prov_make_super "$PROV_SUPER_REVERT" "$PROV_SHA1"
    PROV_TREE_REVERT="$PROV_ROOT/tree_revert"
    prov_make_tree "$PROV_TREE_REVERT" "$PROV_SHA2"
    provr_out="$(env GIT_ALLOW_PROTOCOL=file bash "$PROV_REVERTED" "$PROV_TREE_REVERT" "$PROV_SUPER_REVERT" 2>&1)"; provr_rc=$?
    if [ "$provr_rc" -ne 0 ] && grep -q 'MISMATCH' <<<"$provr_out" \
       && ! grep -q 'attempting to fetch' <<<"$provr_out" \
       && [ ! -e "$PROV_TREE_REVERT/crates/jammi-kernels/third_party/cutlass/header.h" ]; then
      ok "(m/provision revert-RED) the bare-command form on the SAME drift fixture stops at MISMATCH under set -e — the remediation arm is dead code without the if/else, so the if/else is load-bearing"
    else
      bad "(m/provision revert-RED) expected the reverted (bare-command) form to stop at MISMATCH without reaching remediation (rc=$provr_rc): $provr_out"
    fi
  else
    bad "(m/provision revert-RED) revert fixture has a syntax error"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (n) pod_seed_pkg_has_feature — live detection, never a hardcoded feature
# path, with THREE distinct return codes (0 declared / 1 genuinely absent /
# 2 could not determine) — and the seed helpers around it.
# ═════════════════════════════════════════════════════════════════════════
{
  N_FIXTURE="$SANDBOX/n_fixture"
  rm -rf "$N_FIXTURE"; mkdir -p "$N_FIXTURE/src"
  cat > "$N_FIXTURE/Cargo.toml" <<'EOF'
[package]
name = "jammi-n-fixture"
version = "0.1.0"
edition = "2021"

[features]
default = []
has-me = []
EOF
  echo 'fn main() {}' > "$N_FIXTURE/src/main.rs"
  ( cd "$N_FIXTURE" && cargo generate-lockfile -q 2>/dev/null )

  N_DRIVER="$SANDBOX/probe_n.sh"
  cat > "$N_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
cd "$N_FIXTURE" || exit 9
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_pkg_has_feature "\$1" "\$2"
DRV
  chmod +x "$N_DRIVER"

  bash "$N_DRIVER" jammi-n-fixture has-me >/dev/null 2>&1
  [ $? -eq 0 ] && ok "(n) pod_seed_pkg_has_feature returns 0 for a feature that genuinely exists" \
    || bad "(n) expected 0 for jammi-n-fixture/has-me"

  bash "$N_DRIVER" jammi-n-fixture no-such-feature >/dev/null 2>&1
  [ $? -eq 1 ] && ok "(n) pod_seed_pkg_has_feature returns 1 (genuinely absent) for a feature that does not exist on a FOUND package" \
    || bad "(n) expected 1 for jammi-n-fixture/no-such-feature"

  bash "$N_DRIVER" no-such-package anything >/dev/null 2>&1
  [ $? -eq 2 ] && ok "(n) pod_seed_pkg_has_feature returns 2 (could not determine) for a package not in the graph — distinct from 1" \
    || bad "(n) expected 2 for an unknown package"

  # Metadata-query failure (no Cargo.toml at all) -> 2, not 1: a failed
  # query is never evidence that a feature is absent.
  N_NOPROJECT="$SANDBOX/n_noproject"
  rm -rf "$N_NOPROJECT"; mkdir -p "$N_NOPROJECT"
  N_DRIVER2="$SANDBOX/probe_n2.sh"
  cat > "$N_DRIVER2" <<DRV
#!/usr/bin/env bash
set -uo pipefail
cd "$N_NOPROJECT" || exit 9
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_pkg_has_feature anything anything
DRV
  chmod +x "$N_DRIVER2"
  bash "$N_DRIVER2" >/dev/null 2>&1
  [ $? -eq 2 ] && ok "(n) pod_seed_pkg_has_feature returns 2 when the metadata query itself fails (no Cargo.toml) — never 1 ('genuinely absent' would be a false claim)" \
    || bad "(n) expected 2 when cargo metadata itself has nothing to query"

  # T1b's own dispatch differentiates the two messages.
  if grep -q 'could not determine whether jammi-kernels declares flash-attn' "$REPO_ROOT/ci/scripts/pod_seed_target.sh"; then
    ok "(n) T1b's dispatch has a DISTINCT message for 'could not determine' vs 'genuinely absent'"
  else
    bad "(n) T1b's dispatch does not distinguish a metadata-query failure from a genuine feature absence"
  fi

  # rc=2 ("could not determine") ABORTS the seed's own T1b gate, never
  # silently skips it as a soft "treat as absent". Structural (reaching
  # this branch live needs a real `main`-branch checkout mid-seed-build):
  # the source must `exit 1` on rc=2; only rc=1 reads "T1b skipped".
  if grep -q 'refusing to guess .absent.; see pod_seed_cargo_metadata_locked' "$REPO_ROOT/ci/scripts/pod_seed_target.sh"; then
    ok "(n/seed-helpers) T1b's rc=2 branch names the real cause and refuses to guess, rather than silently skipping"
  else
    bad "(n/seed-helpers) T1b's rc=2 branch does not name the refusal-to-guess"
  fi
  SEEDH_SEEDSH="$REPO_ROOT/ci/scripts/pod_seed_target.sh"
  SEEDH_T1B_START="$(grep -n 'pod_seed_pkg_has_feature jammi-kernels flash-attn || feat_rc=\$?' "$SEEDH_SEEDSH" | head -1 | cut -d: -f1)"
  # The feat_rc dispatch (if/elif/else/fi) sits OUTSIDE the
  # `_seed_is_main`-only block, so its own else/fi are at 4-space indent;
  # the awk anchors below track that nesting.
  SEEDH_T1B_ELSE="$(awk -v s="$SEEDH_T1B_START" 'NR>s && /^    else$/{print NR; exit}' "$SEEDH_SEEDSH")"
  SEEDH_T1B_FI="$(awk -v s="$SEEDH_T1B_START" 'NR>s && /^    fi$/{print NR; exit}' "$SEEDH_SEEDSH")"
  if [ -n "$SEEDH_T1B_ELSE" ] && [ -n "$SEEDH_T1B_FI" ] \
     && grep -q 'exit 1' <<<"$(sed -n "${SEEDH_T1B_ELSE},${SEEDH_T1B_FI}p" "$SEEDH_SEEDSH")"; then
    ok "(n/seed-helpers) T1b's rc=2 else-branch genuinely contains 'exit 1' (aborts the seed subshell), not merely a warning"
  else
    bad "(n/seed-helpers) could not confirm T1b's rc=2 else-branch aborts (start=${SEEDH_T1B_START:-?} else=${SEEDH_T1B_ELSE:-?} fi=${SEEDH_T1B_FI:-?})"
  fi

  # pod_build_timings.sh's OWN, separate FA2 *measurement* leg ABORTS on
  # rc=2 too. It cannot rely on the seed's T1b abort: whenever a seed
  # already exists (the ordinary case — `up`/`shell` kick off the seed at
  # bootstrap), step (i)'s pod_seed_target.sh --no-lock call short-circuits
  # at "seed already complete" without ever reaching the T1b/rc=2 abort.
  if grep -q 'FA2 leg: could not determine whether jammi-kernels declares flash-attn' "$REPO_ROOT/ci/scripts/perf/pod_build_timings.sh"; then
    ok "(n/seed-helpers) pod_build_timings.sh's OWN FA2 measurement leg ABORTS on rc=2, never downgrading 'could not determine' to 'absent'"
  else
    bad "(n/seed-helpers) pod_build_timings.sh's FA2 leg dispatch text changed unexpectedly"
  fi

  # EXECUTABLE fixture: the REAL FA2-leg bytes (sed-extracted from
  # pod_build_timings.sh, never a reimplementation), run against a real git
  # repo + a real "origin" remote, gated on the RESOLVED sha (never an
  # abbrev-ref-vs-"main" comparison: a checkout-by-sha — the ordinary case,
  # since JAMMI_FA2_TIP_REF is normally a sha — ALWAYS leaves HEAD
  # detached, so such a gate could never match at all). Legs: (a) checked out ON the branch "main" ->
  # gate matches, reaches pod_seed_pkg_has_feature (stubbed rc=2) ->
  # ABORTS naming the cause; (b) checked out DETACHED at the EXACT SAME
  # sha as origin/main (the realistic FA2-tip-by-sha shape) -> the gate
  # STILL matches (sha-based, not branch-based) -> same abort; (c)
  # checked out DETACHED at a DIFFERENT sha (not on main) -> the leg is
  # correctly SKIPPED, with fa2_ran=false and a real, non-empty
  # fa2_reason recorded — never silently vanishing with no explanation.
  PBT_SH="$REPO_ROOT/ci/scripts/perf/pod_build_timings.sh"
  FA2_START="$(grep -n '^fa2_wall=""$' "$PBT_SH" | head -1 | cut -d: -f1)"
  FA2_END="$(awk -v s="$FA2_START" 'NR>=s && /^echo "::endgroup::"$/{print NR; exit}' "$PBT_SH")"
  if [ -n "$FA2_START" ] && [ -n "$FA2_END" ]; then
    fa2_git() { git -c protocol.file.allow=always -c init.defaultBranch=main "$@"; }
    FA2_UPSTREAM="$SANDBOX/fa2_upstream"
    rm -rf "$FA2_UPSTREAM"; fa2_git init -q "$FA2_UPSTREAM"
    fa2_git -C "$FA2_UPSTREAM" config user.email t@t; fa2_git -C "$FA2_UPSTREAM" config user.name t
    echo x > "$FA2_UPSTREAM/f"; fa2_git -C "$FA2_UPSTREAM" add f; fa2_git -C "$FA2_UPSTREAM" commit -q -m x
    FA2_MAIN_SHA="$(fa2_git -C "$FA2_UPSTREAM" rev-parse HEAD)"
    echo y > "$FA2_UPSTREAM/f2"; fa2_git -C "$FA2_UPSTREAM" add f2; fa2_git -C "$FA2_UPSTREAM" commit -q -m y
    FA2_OTHER_SHA="$(fa2_git -C "$FA2_UPSTREAM" rev-parse HEAD)"

    FA2_CLONE="$SANDBOX/fa2_clone"
    rm -rf "$FA2_CLONE"
    export GIT_ALLOW_PROTOCOL=file
    fa2_git clone -q "$FA2_UPSTREAM" "$FA2_CLONE" || echo "fa2 clone failed rc=$?" >&2
    unset GIT_ALLOW_PROTOCOL

    fa2_build_driver() { # $1=checkout-target -> writes+returns driver path
      local checkout_to="$1" driver="$SANDBOX/fa2_driver_$$_${RANDOM}.sh"
      {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        echo 'fail() { echo "::error::$*" >&2; exit 1; }'
        echo 'pod_seed_pkg_has_feature() { return 2; }'
        echo "cd '$FA2_CLONE' || exit 9"
        echo "env GIT_ALLOW_PROTOCOL=file git -c protocol.file.allow=always checkout -q '$checkout_to' || exit 8"
        sed -n "${FA2_START},${FA2_END}p" "$PBT_SH"
        echo 'printf "FA2_RAN=%s FA2_REASON=%s\n" "$fa2_ran" "$fa2_reason"'
      } > "$driver"
      chmod +x "$driver"
      printf '%s' "$driver"
    }

    # (a) on the branch "main"
    fa2_driver_a="$(fa2_build_driver main)"
    if bash -n "$fa2_driver_a"; then
      fa2_out_a="$(bash "$fa2_driver_a" 2>&1)"; fa2_rc_a=$?
      if [ "$fa2_rc_a" -ne 0 ] && grep -q 'could not determine whether jammi-kernels declares flash-attn' <<<"$fa2_out_a"; then
        ok "(n/fa2-leg a) on branch 'main': the real FA2-leg bytes reach pod_seed_pkg_has_feature (stubbed rc=2) and ABORT naming the real cause"
      else
        bad "(n/fa2-leg a) expected the real FA2-leg bytes to abort on rc=2 when on main (rc=$fa2_rc_a): $fa2_out_a"
      fi
    else
      bad "(n/fa2-leg a) driver fixture has a syntax error"
    fi

    # (b) DETACHED at the exact same sha as origin/main — the realistic
    # FA2-tip-by-sha shape, where an abbrev-ref gate never matches at all.
    # FA2_OTHER_SHA (the SECOND commit, made after FA2_MAIN_SHA was
    # captured but BEFORE cloning) is origin/main's ACTUAL tip once
    # cloned — FA2_MAIN_SHA is merely an ANCESTOR of
    # main, not main itself; the two legs below use whichever sha is
    # ACTUALLY origin/main's tip vs ACTUALLY not, never by variable name
    # alone.
    fa2_driver_b="$(fa2_build_driver "$FA2_OTHER_SHA")"
    fa2_out_b="$(bash "$fa2_driver_b" 2>&1)"; fa2_rc_b=$?
    if [ "$fa2_rc_b" -ne 0 ] && grep -q 'could not determine whether jammi-kernels declares flash-attn' <<<"$fa2_out_b"; then
      ok "(n/fa2-leg b) DETACHED HEAD at the SAME sha as origin/main: the gate STILL matches (sha-based, not abbrev-ref) — reaches pod_seed_pkg_has_feature and ABORTS naming the cause, exactly like leg (a)"
    else
      bad "(n/fa2-leg b) expected a detached HEAD at origin/main's own sha to still match the gate (rc=$fa2_rc_b): $fa2_out_b"
    fi

    # (c) DETACHED at a DIFFERENT sha (not main's tip — an ANCESTOR of
    # it) — correctly skipped, fa2_ran=false, with a real, non-empty
    # reason recorded — never a silent null.
    fa2_driver_c="$(fa2_build_driver "$FA2_MAIN_SHA")"
    fa2_out_c="$(bash "$fa2_driver_c" 2>&1)"; fa2_rc_c=$?
    if [ "$fa2_rc_c" -eq 0 ] && grep -q 'FA2_RAN=false' <<<"$fa2_out_c" \
       && grep -q 'FA2_REASON=resolved sha' <<<"$fa2_out_c"; then
      ok "(n/fa2-leg c) DETACHED HEAD at a sha that is NOT origin/main: fa2_ran=false with a real, non-empty reason recorded — never silently vanishing with no explanation"
    else
      bad "(n/fa2-leg c) expected fa2_ran=false with a real reason for a non-main detached sha (rc=$fa2_rc_c): $fa2_out_c"
    fi
  else
    bad "(n/fa2-leg) could not locate the FA2 leg's start/end lines in pod_build_timings.sh (start=${FA2_START:-?} end=${FA2_END:-?})"
  fi

  # A complete seed marker WITHOUT FA2 -> the timings JSON copies
  # seed_tuples/seed_t1b_flash_attn_* from the seed's own marker. Exercises
  # the REAL python snippets
  # pod_build_timings.sh's step (i) block uses to read the marker back
  # (sed-extracted, never reimplemented), against a real marker JSON
  # shaped exactly like a real "not on main" seed.
  FA2_MARKER_PY_START="$(grep -n "^seed_tuples_json=" "$PBT_SH" | head -1 | cut -d: -f1)"
  FA2_MARKER_PY_END="$(awk -v s="$FA2_MARKER_PY_START" 'NR>=s && /^echo "::endgroup::"$/{print NR; exit}' "$PBT_SH")"
  if [ -n "$FA2_MARKER_PY_START" ] && [ -n "$FA2_MARKER_PY_END" ]; then
    FA2_SEED_NOFA2="$SANDBOX/fa2_seed_no_fa2"
    python3 -c '
import json
json.dump({
  "ref": "ci/pod-build-substrate-r5", "sha": "deadbeef", "date": "2026-01-01T00:00:00Z",
  "tuples": ["T1", "T2", "T3"], "rustflags": "", "size_bytes": 0,
  "manifest_sha256": "abc", "seed_source": "built",
  "t1b_flash_attn_ran": False,
  "t1b_flash_attn_reason": "ref != main (ref=ci/pod-build-substrate-r5) - T1b is main-only by design",
}, open("'"$FA2_SEED_NOFA2"'.jammi-seed-complete", "w"))
'
    FA2_MARKER_DRIVER="$SANDBOX/fa2_marker_driver.sh"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      echo "JAMMI_SEED_DIR='$FA2_SEED_NOFA2'"
      sed -n "${FA2_MARKER_PY_START},${FA2_MARKER_PY_END}p" "$PBT_SH" | grep -v '^echo "::endgroup::"$'
      echo 'printf "TUPLES=%s RAN=%s REASON=%s\n" "$seed_tuples_json" "$seed_t1b_ran" "$seed_t1b_reason"'
    } > "$FA2_MARKER_DRIVER"
    chmod +x "$FA2_MARKER_DRIVER"
    if bash -n "$FA2_MARKER_DRIVER"; then
      fa2_marker_out="$(bash "$FA2_MARKER_DRIVER" 2>&1)"
      if grep -q 'TUPLES=\["T1", "T2", "T3"\] RAN=false REASON=ref != main' <<<"$fa2_marker_out" \
         || grep -qE 'TUPLES=\[.T1., .T2., .T3.\] RAN=false REASON=ref != main' <<<"$fa2_marker_out"; then
        ok "(n/fa2-leg) the real marker-reading python snippets correctly copy seed_tuples (no T1b) / seed_t1b_flash_attn_ran=false / the real reason out of a real seed-complete marker"
      else
        bad "(n/fa2-leg) expected the marker fields to be copied verbatim: $fa2_marker_out"
      fi
    else
      bad "(n/fa2-leg) marker-reader driver fixture has a syntax error"
    fi
  else
    bad "(n/fa2-leg) could not locate the marker-reading python snippets in pod_build_timings.sh (start=${FA2_MARKER_PY_START:-?} end=${FA2_MARKER_PY_END:-?})"
  fi

  # Every `--frozen` metadata call site goes through the
  # shared stderr-capturing helper — never a raw `cargo metadata --frozen
  # ... 2>/dev/null` (a tripwire: an edit introducing a raw discarding
  # call site fails this).
  SEEDH_RAW_SITES="$(grep -c 'cargo metadata --frozen --format-version 1 2>/dev/null\|cargo metadata --frozen --format-version 1 --features jammi-kernels/cuda 2>/dev/null' "$SEEDH_SEEDSH" 2>/dev/null || true)"
  if [ "${SEEDH_RAW_SITES:-0}" -eq 0 ]; then
    ok "(n/seed-helpers) no raw, stderr-discarding 'cargo metadata --frozen ... 2>/dev/null' call site remains in pod_seed_target.sh"
  else
    bad "(n/seed-helpers) found ${SEEDH_RAW_SITES} raw stderr-discarding cargo metadata call site(s) still in pod_seed_target.sh"
  fi

  # pod_seed_cargo_metadata_locked actually surfaces real
  # stderr (never silently returns empty) — exercised against a REAL
  # broken --frozen query (no Cargo.lock at all) rather than asserted.
  SEEDH_METASH="$SANDBOX/seedh_metash"
  mkdir -p "$SEEDH_METASH"
  # shellcheck disable=SC1090
  SEEDH_META_OUT="$( (cd "$SEEDH_METASH" && . "$SEEDH_SEEDSH" && pod_seed_cargo_metadata_locked) 2>&1 1>/dev/null )"
  SEEDH_META_RC=0
  # shellcheck disable=SC1090
  (cd "$SEEDH_METASH" && . "$SEEDH_SEEDSH" && pod_seed_cargo_metadata_locked >/dev/null 2>/dev/null) || SEEDH_META_RC=$?
  if [ "$SEEDH_META_RC" -eq 2 ] && [ -n "$SEEDH_META_OUT" ]; then
    ok "(n/seed-helpers) pod_seed_cargo_metadata_locked returns 2 and prints REAL stderr on a genuinely broken query (no silent empty string)"
  else
    bad "(n/seed-helpers) expected rc=2 with non-empty stderr on a broken metadata query (rc=$SEEDH_META_RC, stderr empty=$([ -z "$SEEDH_META_OUT" ] && echo yes || echo no))"
  fi

  # The seed's own one-time, network-allowed priming call
  # (`cargo metadata --locked`, never `--frozen`) runs BEFORE T1 — a
  # structural, line-position check (a real priming run needs network, out
  # of scope for a hermetic suite).
  SEEDH_PRIME_LINE="$(grep -n 'cargo metadata --locked --format-version 1 --features jammi-kernels/cuda' "$SEEDH_SEEDSH" | head -1 | cut -d: -f1)"
  SEEDH_T1_LINE="$(grep -n 'echo "=== T1: release -p jammi-bench --features cuda ==="' "$SEEDH_SEEDSH" | head -1 | cut -d: -f1)"
  if [ -n "$SEEDH_PRIME_LINE" ] && [ -n "$SEEDH_T1_LINE" ] && [ "$SEEDH_PRIME_LINE" -lt "$SEEDH_T1_LINE" ]; then
    ok "(n/seed-helpers) the seed's one-time network-allowed metadata priming call (line ${SEEDH_PRIME_LINE}) runs BEFORE T1 (line ${SEEDH_T1_LINE})"
  else
    bad "(n/seed-helpers) expected the priming call to precede T1 (prime=${SEEDH_PRIME_LINE:-?} T1=${SEEDH_T1_LINE:-?})"
  fi

  # pod_seed_write_failure_marker captures a diagnostic buried far above a
  # plain tail, against a REAL fixture log, alongside a plain tail-only
  # form's miss on the same log so the fixture is known to discriminate.
  SEEDH_FIXTURE_LOG="$SANDBOX/seedh_fixture.log"
  { echo "   Compiling jammi-kernels v0.47.0"
    echo "error[E0433]: failed to resolve: use of undeclared crate or module foo"
    for i in $(seq 1 200); do echo "   Checking some-other-crate-$i v0.1.$i"; done
    echo "    Finished release [optimized] target(s) in 40.12s"
  } > "$SEEDH_FIXTURE_LOG"
  SEEDH_NEW_MARKER="$SANDBOX/seedh_new_marker.txt"
  # shellcheck disable=SC1090
  (cd "$SEEDH_METASH" && . "$SEEDH_SEEDSH" && pod_seed_write_failure_marker "$SEEDH_FIXTURE_LOG" "$SEEDH_NEW_MARKER" 1)
  if grep -q 'E0433' "$SEEDH_NEW_MARKER"; then
    ok "(n/seed-helpers) pod_seed_write_failure_marker catches an error buried 200 lines above the tail"
  else
    bad "(n/seed-helpers) pod_seed_write_failure_marker missed a buried E0433 error"
  fi
  SEEDH_OLD_MARKER="$SANDBOX/seedh_old_marker.txt"
  { echo "seed build FAILED (exit 1) — log tail:"; tail -100 "$SEEDH_FIXTURE_LOG"; } > "$SEEDH_OLD_MARKER"
  if ! grep -q 'E0433' "$SEEDH_OLD_MARKER"; then
    ok "(n/seed-helpers revert-RED) a plain tail-100 form genuinely MISSES the same buried error on the SAME fixture — the marker's wider capture is load-bearing"
  else
    bad "(n/seed-helpers revert-RED) a plain tail-100 form unexpectedly caught the error too — the fixture does not discriminate"
  fi

  # pod_seed_assert_required_tools fails loudly, naming every missing tool,
  # and passes when everything required is present. POSIX prefix-assignment
  # resolves the COMMAND NAME using the assignment's OWN new PATH, not the
  # caller's — `PATH="$narrow_dir" bash -c ...` fails to find "bash" itself
  # once PATH no longer contains it — so bash is captured once, as an
  # ABSOLUTE path, and invoked as "$SEEDH_REAL_BASH" below.
  SEEDH_REAL_BASH="$(command -v bash)"

  SEEDH_TOOLBIN_MISSING="$SANDBOX/seedh_toolbin_missing"
  mkdir -p "$SEEDH_TOOLBIN_MISSING"
  # dirname is needed by pod_seed_target.sh's own TOP-LEVEL code (computing
  # $DIR at source time, before any function is even called) — omitting it
  # from a narrow-PATH sandbox breaks sourcing itself with unrelated noise
  # ("dirname: command not found"), not the specific behavior under test.
  for t in dirname git python3 sha256sum shasum; do
    REAL="$(command -v "$t" 2>/dev/null || true)"
    [ -n "$REAL" ] && ln -sf "$REAL" "$SEEDH_TOOLBIN_MISSING/$t"
  done
  # cargo deliberately left OFF PATH here.
  SEEDH_TOOLS_OUT="$(PATH="$SEEDH_TOOLBIN_MISSING" "$SEEDH_REAL_BASH" -c '. "'"$SEEDH_SEEDSH"'"; pod_seed_assert_required_tools' 2>&1)"
  SEEDH_TOOLS_RC=$?
  if [ "$SEEDH_TOOLS_RC" -ne 0 ] && grep -q 'cargo' <<<"$SEEDH_TOOLS_OUT"; then
    ok "(n/seed-helpers) pod_seed_assert_required_tools fails and NAMES 'cargo' when it is missing from PATH"
  else
    bad "(n/seed-helpers) expected a failure naming 'cargo' (rc=$SEEDH_TOOLS_RC): $SEEDH_TOOLS_OUT"
  fi
  if (PATH="$HOME/.cargo/bin:$PATH" "$SEEDH_REAL_BASH" -c '. "'"$SEEDH_SEEDSH"'"; pod_seed_assert_required_tools') >/dev/null 2>&1; then
    ok "(n/seed-helpers) pod_seed_assert_required_tools passes cleanly when every required tool is present"
  else
    bad "(n/seed-helpers) pod_seed_assert_required_tools unexpectedly failed with a normal PATH"
  fi

  # pod_sha256_of_file prefers sha256sum, falls back to
  # shasum, and refuses loudly (never a silent empty string) if neither
  # exists — exercised against real fake-tool PATH sandboxes.
  SEEDH_HASH_FILE="$SANDBOX/seedh_hash_target.txt"
  echo "hash me" > "$SEEDH_HASH_FILE"
  SEEDH_REAL_HASH="$(shasum -a 256 "$SEEDH_HASH_FILE" 2>/dev/null | awk '{print $1}')"

  SEEDH_NOTOOLS="$SANDBOX/seedh_notools_bin"
  mkdir -p "$SEEDH_NOTOOLS"
  SEEDH_REAL_DIRNAME="$(command -v dirname 2>/dev/null || true)"
  [ -n "$SEEDH_REAL_DIRNAME" ] && ln -sf "$SEEDH_REAL_DIRNAME" "$SEEDH_NOTOOLS/dirname"
  SEEDH_NOHASH_OUT="$(PATH="$SEEDH_NOTOOLS" "$SEEDH_REAL_BASH" -c '. "'"$SEEDH_SEEDSH"'"; pod_sha256_of_file "'"$SEEDH_HASH_FILE"'"' 2>&1)"
  SEEDH_NOHASH_RC=$?
  if [ "$SEEDH_NOHASH_RC" -eq 2 ] && [ -z "$(printf '%s' "$SEEDH_NOHASH_OUT" | grep -v '::error::')" ]; then
    ok "(n/seed-helpers) pod_sha256_of_file refuses loudly (rc=2, no hash printed) when neither sha256sum nor shasum exists"
  else
    bad "(n/seed-helpers) pod_sha256_of_file should have refused loudly with no hashing tool available (rc=$SEEDH_NOHASH_RC): $SEEDH_NOHASH_OUT"
  fi

  SEEDH_SHASUM_ONLY="$SANDBOX/seedh_shasum_only_bin"
  mkdir -p "$SEEDH_SHASUM_ONLY"
  [ -n "$SEEDH_REAL_DIRNAME" ] && ln -sf "$SEEDH_REAL_DIRNAME" "$SEEDH_SHASUM_ONLY/dirname"
  # pod_sha256_of_file's own shasum-fallback branch pipes through awk.
  SEEDH_REAL_AWK="$(command -v awk 2>/dev/null || true)"
  [ -n "$SEEDH_REAL_AWK" ] && ln -sf "$SEEDH_REAL_AWK" "$SEEDH_SHASUM_ONLY/awk"
  # shasum is a declared need of this suite (ci/needs.toml
  # [need.shasum]); the fallback leg fails, naming it, on a host without it.
  REAL_SHASUM="$(command -v shasum 2>/dev/null || true)"
  if [ -n "$REAL_SHASUM" ]; then
    ln -sf "$REAL_SHASUM" "$SEEDH_SHASUM_ONLY/shasum"
    SEEDH_SHASUM_HASH="$(PATH="$SEEDH_SHASUM_ONLY" "$SEEDH_REAL_BASH" -c '. "'"$SEEDH_SEEDSH"'"; pod_sha256_of_file "'"$SEEDH_HASH_FILE"'"' 2>/dev/null)"
    if [ "$SEEDH_SHASUM_HASH" = "$SEEDH_REAL_HASH" ] && [ -n "$SEEDH_SHASUM_HASH" ]; then
      ok "(n/sha256) pod_sha256_of_file falls back to shasum -a 256 when sha256sum is absent, and computes the CORRECT hash"
    else
      bad "(n/sha256) shasum fallback hash mismatch or empty (got '$SEEDH_SHASUM_HASH', want '$SEEDH_REAL_HASH')"
    fi
  else
    bad "(n/sha256) the shasum fallback leg needs shasum on PATH (ci/needs.toml need 'shasum': perl-Digest-SHA); it is absent on this host"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (o) pod_build_timings.sh: the timings-under-lock live witness, the FA2
# leg's ordering against the T1 snapshot, the features its byte-equality
# compares, and the cold leg's empty start. None is exercisable end to end
# without a real CUDA toolchain; each is checked as precisely as a hermetic
# suite can: the LIVE-WITNESS mechanism directly (function-level), the
# rest as static assertions on the real script (a real cargo/nvcc run
# happens only on a live pod).
# ═════════════════════════════════════════════════════════════════════════
{
  PBT_SH="$REPO_ROOT/ci/scripts/perf/pod_build_timings.sh"

  # LOCK_HELD reads the ACTUAL holder file, not a hardcoded constant.
  # Fixture: a holder file that does
  # NOT name "pod_build_timings" must yield LOCK_HELD=false.
  O_LOCK="$SANDBOX/o.lock"
  printf 'holder=someone-else\n' > "${O_LOCK}.holder"
  o_out="$(JAMMI_TIMING_LOCK="$O_LOCK" bash -c '
    _LOCK_FILE="${JAMMI_TIMING_LOCK:-/root/.jammi-timing.lock}"
    if [ -f "${_LOCK_FILE}.holder" ] && grep -q "^holder=pod_build_timings\$" "${_LOCK_FILE}.holder" 2>/dev/null; then
      echo true
    else
      echo false
    fi
  ')"
  [ "$o_out" = "false" ] && ok "(o) LOCK_HELD reads false when the holder file does not name pod_build_timings (a live witness, not a hardcoded true)" \
    || bad "(o) LOCK_HELD should have read false for a mismatched holder — got: $o_out"

  printf 'holder=pod_build_timings\n' > "${O_LOCK}.holder"
  o_out2="$(JAMMI_TIMING_LOCK="$O_LOCK" bash -c '
    _LOCK_FILE="${JAMMI_TIMING_LOCK:-/root/.jammi-timing.lock}"
    if [ -f "${_LOCK_FILE}.holder" ] && grep -q "^holder=pod_build_timings\$" "${_LOCK_FILE}.holder" 2>/dev/null; then
      echo true
    else
      echo false
    fi
  ')"
  [ "$o_out2" = "true" ] && ok "(o) LOCK_HELD reads true when the holder file genuinely names pod_build_timings" \
    || bad "(o) LOCK_HELD should have read true for a matching holder — got: $o_out2"

  # pod_build_timings.sh itself uses this exact live-witness shape, not a
  # hardcoded LOCK_HELD=true.
  if grep -q 'grep -q .\^holder=pod_build_timings\$.' "$PBT_SH"; then
    ok "(o) pod_build_timings.sh reads LOCK_HELD from the holder file (live witness), not a hardcoded constant"
  else
    bad "(o) pod_build_timings.sh does not appear to read LOCK_HELD from the holder file"
  fi
  if grep -qE '^LOCK_HELD=true$' "$PBT_SH"; then
    bad "(o) pod_build_timings.sh has an unconditional 'LOCK_HELD=true' — LOCK_HELD must be a live witness"
  else
    ok "(o) pod_build_timings.sh has no unconditional 'LOCK_HELD=true'"
  fi

  # FA2 ordering: clone_hashes/recompiled must be captured BEFORE the FA2
  # leg's own code (line-position check on the real script; the two
  # markers below are unique substrings of it).
  snap_line="$(grep -n 'clone_hashes="\$(snapshot_hashes "\$CLONE_DIR")"' "$PBT_SH" | head -1 | cut -d: -f1)"
  fa2_line="$(grep -n 'CLONE_FA2_DIR="/root/.jammi-clone-fa2-a2"' "$PBT_SH" | head -1 | cut -d: -f1)"
  if [ -n "$snap_line" ] && [ -n "$fa2_line" ] && [ "$snap_line" -lt "$fa2_line" ]; then
    ok "(o/fa2-order) clone_hashes is snapshotted (line ${snap_line}) BEFORE the FA2 leg's own clone dir is even created (line ${fa2_line})"
  else
    bad "(o/fa2-order) expected the clone_hashes snapshot to precede the FA2 leg's own clone dir in source order (snap=${snap_line:-?} fa2=${fa2_line:-?})"
  fi
  # The FA2 leg builds into its OWN clone dir, never CLONE_DIR.
  if grep -q 'CARGO_TARGET_DIR="\$CLONE_FA2_DIR"' "$PBT_SH"; then
    ok "(o/fa2-order) the FA2 leg builds into its own CLONE_FA2_DIR, never the T1 CLONE_DIR"
  else
    bad "(o/fa2-order) the FA2 leg does not appear to use a separate CARGO_TARGET_DIR"
  fi
  # byte_equal compares like with like only when the clone (T1) build and
  # the cold build pass the same `--features`, and the recorded
  # clone_features/cold_features are literals that must name what each
  # build actually passed. Each value is read from its one site in the real
  # script; a missing or duplicated site reads as a mismatch.
  o_one() { sed -nE "$1" "$PBT_SH" | awk 'NR == 1 { v = $0 } END { if (NR == 1) print v }'; }
  o_clone_built="$(o_one 's/^cargo build --release -p jammi-bench --features ([^ ]+) .*clone_t1\.log.*/\1/p')"
  o_cold_built="$(o_one 's/^CARGO_TARGET_DIR="\$COLD_DIR" cargo build --release -p jammi-bench --features ([^ ]+).*/\1/p')"
  o_clone_recorded="$(o_one 's/^clone_features="([^"]*)"$/\1/p')"
  o_cold_recorded="$(o_one 's/^cold_features="([^"]*)"$/\1/p')"
  if [ -n "$o_clone_built" ] && [ "$o_clone_built" = "$o_cold_built" ] \
     && [ "$o_clone_recorded" = "$o_clone_built" ] && [ "$o_cold_recorded" = "$o_cold_built" ]; then
    ok "(o/features) the clone and cold builds pass the same --features (${o_clone_built}), and clone_features/cold_features record exactly that"
  else
    bad "(o/features) expected one clone build, one cold build and both recorded features to agree (clone build=${o_clone_built:-?} cold build=${o_cold_built:-?} clone_features=${o_clone_recorded:-?} cold_features=${o_cold_recorded:-?})"
  fi

  # The cold leg builds from a genuinely empty directory. Structural, since
  # a real cold build needs a CUDA toolchain: the cold-leg block must
  # `rm -rf`+`mkdir -p` COLD_DIR and must NOT `cp -a` the clone into it
  # anywhere in the same script.
  if grep -qE 'rm -rf "\$COLD_DIR"' "$PBT_SH" && grep -qE 'mkdir -p "\$COLD_DIR"' "$PBT_SH"; then
    ok "(o/cold-empty) the cold leg does rm -rf + mkdir -p on COLD_DIR (starts from nothing)"
  else
    bad "(o/cold-empty) the cold leg does not appear to rm -rf + mkdir -p COLD_DIR"
  fi
  if grep -qE 'cp -a "\$CLONE_DIR" "\$COLD_DIR"' "$PBT_SH"; then
    bad "(o/cold-empty) found 'cp -a \$CLONE_DIR \$COLD_DIR' — the cold leg is copying the clone again, defeating the whole point of a cold comparison"
  else
    ok "(o/cold-empty) no 'cp -a \$CLONE_DIR \$COLD_DIR' anywhere — the cold leg never reuses the clone's own artifacts"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (p) the OTHER two unconditional member-freedom call sites, beside
# pod_target_clone.sh's (b/member-freedom above), each with a revert-RED.
# pod_seed_target.sh's own completion-stamp gate (T4,
# right before .jammi-seed-complete is written) and pod_build_timings.sh's
# second, independent witness (right after step (i)'s seed-marker check)
# are exercised here against the REAL scripts, the REAL
# pod_seed_assert_member_free function, and a REAL `cargo metadata --frozen`
# query against THIS checkout's own workspace (never a fixture workspace,
# same technique (b) already uses) — only cargo build/test/clippy are
# stubbed to a silent exit 0 (metadata/fetch pass straight through to the
# real cargo binary), so the T1-T4 pipeline finishes in well under a second
# without ever compiling anything, and whatever is planted in JAMMI_SEED_DIR
# before the call survives untouched all the way to the gate under test.
# `git submodule` is stubbed the same way: the seed's CUTLASS provisioning
# step would otherwise clone the submodule over the network INTO this
# checkout whenever it has no cutlass.h ((w/cutlass) covers that step on
# local fixtures); every other git call reaches the real binary.
# ═════════════════════════════════════════════════════════════════════════
{
  P_STUBBIN="$SANDBOX/p_cargo_stubbin"
  mkdir -p "$P_STUBBIN"
  P_REAL_CARGO="$(command -v cargo)"
  cat > "$P_STUBBIN/cargo" <<STUB
#!/usr/bin/env bash
case "\$1" in
  metadata|fetch) exec "$P_REAL_CARGO" "\$@" ;;
  *) exit 0 ;;
esac
STUB
  chmod +x "$P_STUBBIN/cargo"
  P_REAL_GIT="$(command -v git)"
  cat > "$P_STUBBIN/git" <<STUB
#!/usr/bin/env bash
case "\$1" in
  submodule) exit 0 ;;
  *) exec "$P_REAL_GIT" "\$@" ;;
esac
STUB
  chmod +x "$P_STUBBIN/git"

  # ---- p1: pod_seed_target.sh's own completion-stamp gate (~line 557) ----
  P1_SEED="$SANDBOX/p1_seed"
  rm -rf "$P1_SEED"; mkdir -p "$P1_SEED/release/deps"
  : > "$P1_SEED/release/deps/libjammi_kernels-cafef00d.rlib"

  P1_DRIVER="$SANDBOX/probe_p1_seedmain.sh"
  cat > "$P1_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export PATH="$P_STUBBIN:\$PATH"
export JAMMI_SEED_DIR="$P1_SEED"
export JAMMI_TREE_DIR="$REPO_ROOT"
export JAMMI_SEED_MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_target_main --no-lock
DRV
  chmod +x "$P1_DRIVER"

  p1_out="$(bash "$P1_DRIVER" 2>&1)"; p1_rc=$?
  if [ "$p1_rc" -ne 0 ] && grep -q 'libjammi_kernels-cafef00d.rlib' <<<"$p1_out" \
     && [ ! -f "${P1_SEED}.jammi-seed-complete" ]; then
    ok "(p1/member-freedom) pod_seed_target_main's own completion-stamp gate refuses a lib-prefixed poisoned target dir (real T1-T4 pipeline, real cargo metadata) and writes no completion marker"
  else
    bad "(p1/member-freedom) expected the real seed pipeline to refuse the poisoned target dir before stamping complete (rc=$p1_rc): $p1_out"
  fi

  # Negative control: an UNPOISONED target dir must not trip THIS check —
  # asserted as the absence of a "NOT member-free" line, even though the run
  # still fails later for an UNRELATED reason (the env-surface cross-check,
  # since the stubbed cargo never produces real build-script capture
  # output) — isolating this gate's own pass/fail from the rest of the
  # pipeline's.
  P1_CLEAN="$SANDBOX/p1_clean"
  rm -rf "$P1_CLEAN"; mkdir -p "$P1_CLEAN"
  P1_CLEAN_DRIVER="$SANDBOX/probe_p1_clean.sh"
  cat > "$P1_CLEAN_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export PATH="$P_STUBBIN:\$PATH"
export JAMMI_SEED_DIR="$P1_CLEAN"
export JAMMI_TREE_DIR="$REPO_ROOT"
export JAMMI_SEED_MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_target_main --no-lock
DRV
  chmod +x "$P1_CLEAN_DRIVER"
  p1c_out="$(bash "$P1_CLEAN_DRIVER" 2>&1)"
  if ! grep -q 'NOT member-free' <<<"$p1c_out"; then
    ok "(p1/member-freedom) an unpoisoned target dir passes the member-free gate cleanly (no 'NOT member-free' from THIS check)"
  else
    bad "(p1/member-freedom) the member-free gate false-positived on a clean target dir: $p1c_out"
  fi

  # revert-RED proof: neutering pod_seed_target.sh's OWN completion-stamp
  # call, on a FRESH copy of the same poisoned fixture (a fresh dir, not
  # p1_seed itself — p1_seed's own .jammi-seed-failed marker would short-
  # circuit a second invocation via the "previously FAILED, not retrying"
  # arm before ever reaching T1-T4 again), makes the "NOT member-free"
  # catch DISAPPEAR — proving this exact call site, not some other check,
  # is what caught p1_seed above.
  P1_SEED_REVERT="$SANDBOX/p1_seed_revert"
  rm -rf "$P1_SEED_REVERT"; mkdir -p "$P1_SEED_REVERT/release/deps"
  : > "$P1_SEED_REVERT/release/deps/libjammi_kernels-cafef00d.rlib"

  P1_REVERTED="$SANDBOX/p1_seed_target_reverted.sh"
  cp "$REPO_ROOT/ci/scripts/pod_seed_target.sh" "$P1_REVERTED"
  REVERT_LINE="$(grep -n 'pod_seed_assert_member_free "\$JAMMI_SEED_DIR" "\$JAMMI_TREE_DIR" || exit 1' "$P1_REVERTED" | head -1 | cut -d: -f1)"
  if [ -n "$REVERT_LINE" ]; then
    sed -i.bak "${REVERT_LINE}s/^\([[:space:]]*\)pod_seed_assert_member_free/\1: pod_seed_assert_member_free_DISABLED_FOR_REVERT_PROOF #/" "$P1_REVERTED"
    if bash -n "$P1_REVERTED"; then
      P1_REVERT_DRIVER="$SANDBOX/probe_p1_revert.sh"
      cat > "$P1_REVERT_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
export PATH="$P_STUBBIN:\$PATH"
export JAMMI_SEED_DIR="$P1_SEED_REVERT"
export JAMMI_TREE_DIR="$REPO_ROOT"
export JAMMI_SEED_MANIFEST="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
# shellcheck disable=SC1091
. "$P1_REVERTED"
pod_seed_target_main --no-lock
DRV
      chmod +x "$P1_REVERT_DRIVER"
      p1r_out="$(bash "$P1_REVERT_DRIVER" 2>&1)"
      if ! grep -q 'NOT member-free' <<<"$p1r_out"; then
        ok "(p1/member-freedom revert-RED) neutering pod_seed_target.sh's own completion-stamp gate call (line ${REVERT_LINE}) on the SAME-shaped poisoned dir makes the 'NOT member-free' catch disappear — the call site is genuinely load-bearing, not a proxy"
      else
        bad "(p1/member-freedom revert-RED) expected the poison catch to disappear after neutering line ${REVERT_LINE}, but it is still present: $p1r_out"
      fi
    else
      bad "(p1/member-freedom) revert-RED fixture has a syntax error after neutering line ${REVERT_LINE}"
    fi
  else
    bad "(p1/member-freedom) could not locate pod_seed_target.sh's completion-stamp member-free call site to build the revert-RED fixture"
  fi

  # ---- p2: pod_build_timings.sh's second, independent witness (~line 128) ----
  # Everything past the (i) seed block needs a live cargo/nvcc toolchain
  # (as in (o) above), so this drives the
  # REAL bytes of pod_build_timings.sh from its shebang through the (i)
  # block (a verbatim `sed` extraction, never a hand-reimplementation) with
  # a synthetic tail appended, placed at the same relative depth
  # (<root>/perf/driver.sh) so the script's own `CI_SCRIPTS="$(cd
  # "$DIR/.." && pwd)"` resolution finds a REAL copy of pod_seed_target.sh
  # (for its `.` sourcing at (i)) and a REAL pod_push_stamp.sh alongside it.
  PBT_SH="$REPO_ROOT/ci/scripts/perf/pod_build_timings.sh"
  # Anchored to the actual CODE line (`echo "::endgroup::"`), never a bare
  # substring match — a comment mentioning the literal marker text would
  # otherwise be picked up first, truncating the driver before it ever
  # reaches the real (i) block.
  P2_ENDGROUP_LINE="$(grep -n '^echo "::endgroup::"$' "$PBT_SH" | head -1 | cut -d: -f1)"
  if [ -z "$P2_ENDGROUP_LINE" ]; then
    bad "(p2/member-freedom) could not locate the first '::endgroup::' in pod_build_timings.sh to build the truncated driver"
  else
    P2_ROOT="$SANDBOX/p2_pbt_root"
    rm -rf "$P2_ROOT"; mkdir -p "$P2_ROOT/perf"
    cp "$REPO_ROOT/ci/scripts/pod_seed_target.sh" "$P2_ROOT/pod_seed_target.sh"
    cp "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" "$P2_ROOT/pod_push_stamp.sh"
    sed -n "1,${P2_ENDGROUP_LINE}p" "$PBT_SH" > "$P2_ROOT/perf/driver.sh"
    { printf '\necho P2_REACHED_PAST_MEMBER_CHECK\nexit 0\n'; } >> "$P2_ROOT/perf/driver.sh"
    chmod +x "$P2_ROOT/perf/driver.sh"

    # Poisoned fixture: seed marker pre-created (so the (i) block's own
    # external call to the REAL pod_seed_target.sh --no-lock takes the fast
    # "seed already complete" path — verified idempotent, never touches
    # cargo) with a lib-prefixed poisoned artifact already in place.
    P2_SEED_POISON="$SANDBOX/p2_seed_poison"
    rm -rf "$P2_SEED_POISON"; mkdir -p "$P2_SEED_POISON/release/deps"
    : > "$P2_SEED_POISON/release/deps/libjammi_kernels-deadbeef.rlib"
    : > "${P2_SEED_POISON}.jammi-seed-complete"

    p2_poison_out="$(JAMMI_TREE_DIR="$REPO_ROOT" JAMMI_SEED_DIR="$P2_SEED_POISON" \
      JAMMI_FA2_TIP_REF=irrelevant JAMMI_BOX_LABEL=probe JAMMI_BUILD_TIMINGS_OUT=/dev/null \
      bash "$P2_ROOT/perf/driver.sh" --no-lock 2>&1)"
    p2_poison_rc=$?
    if [ "$p2_poison_rc" -ne 0 ] && grep -q 'NOT member-free' <<<"$p2_poison_out" \
       && ! grep -q 'P2_REACHED_PAST_MEMBER_CHECK' <<<"$p2_poison_out"; then
      ok "(p2/member-freedom) pod_build_timings.sh's own second, independent member-free witness refuses a poisoned seed and never reaches past it (real script bytes through the (i) block)"
    else
      bad "(p2/member-freedom) expected pod_build_timings.sh's own witness to refuse the poisoned seed and not reach past it (rc=$p2_poison_rc): $p2_poison_out"
    fi

    # Negative control: a clean seed reaches past the (i) block.
    P2_SEED_CLEAN="$SANDBOX/p2_seed_clean"
    rm -rf "$P2_SEED_CLEAN"; mkdir -p "$P2_SEED_CLEAN/release"
    echo unrelated > "$P2_SEED_CLEAN/release/thing.txt"
    : > "${P2_SEED_CLEAN}.jammi-seed-complete"
    p2_clean_out="$(JAMMI_TREE_DIR="$REPO_ROOT" JAMMI_SEED_DIR="$P2_SEED_CLEAN" \
      JAMMI_FA2_TIP_REF=irrelevant JAMMI_BOX_LABEL=probe JAMMI_BUILD_TIMINGS_OUT=/dev/null \
      bash "$P2_ROOT/perf/driver.sh" --no-lock 2>&1)"
    p2_clean_rc=$?
    if [ "$p2_clean_rc" -eq 0 ] && grep -q 'P2_REACHED_PAST_MEMBER_CHECK' <<<"$p2_clean_out"; then
      ok "(p2/member-freedom) a clean seed reaches past pod_build_timings.sh's own witness (negative control — the gate does not block legitimate runs)"
    else
      bad "(p2/member-freedom) expected a clean seed to reach past the witness (rc=$p2_clean_rc): $p2_clean_out"
    fi

    # revert-RED proof: neutering the REAL script's own call line, on the
    # SAME poisoned fixture, makes it reach past the check.
    CALL_LINE="$(grep -n 'pod_seed_assert_member_free "\$JAMMI_SEED_DIR" "\$JAMMI_TREE_DIR" || fail "seed at' "$P2_ROOT/perf/driver.sh" | head -1 | cut -d: -f1)"
    if [ -n "$CALL_LINE" ]; then
      cp "$P2_ROOT/perf/driver.sh" "$P2_ROOT/perf/driver_reverted.sh"
      sed -i.bak "${CALL_LINE}s/^pod_seed_assert_member_free/: pod_seed_assert_member_free_DISABLED_FOR_REVERT_PROOF #/" "$P2_ROOT/perf/driver_reverted.sh"
      if bash -n "$P2_ROOT/perf/driver_reverted.sh"; then
        p2_revert_out="$(JAMMI_TREE_DIR="$REPO_ROOT" JAMMI_SEED_DIR="$P2_SEED_POISON" \
          JAMMI_FA2_TIP_REF=irrelevant JAMMI_BOX_LABEL=probe JAMMI_BUILD_TIMINGS_OUT=/dev/null \
          bash "$P2_ROOT/perf/driver_reverted.sh" --no-lock 2>&1)"
        p2_revert_rc=$?
        if [ "$p2_revert_rc" -eq 0 ] && grep -q 'P2_REACHED_PAST_MEMBER_CHECK' <<<"$p2_revert_out"; then
          ok "(p2/member-freedom revert-RED) neutering pod_build_timings.sh's own witness call (line ${CALL_LINE}) on the SAME poisoned seed lets it reach past the check — the call site is genuinely load-bearing, not a proxy"
        else
          bad "(p2/member-freedom revert-RED) expected the poisoned seed to reach past the check after neutering line ${CALL_LINE} (rc=$p2_revert_rc): $p2_revert_out"
        fi
      else
        bad "(p2/member-freedom) revert-RED fixture has a syntax error after neutering line ${CALL_LINE}"
      fi
    else
      bad "(p2/member-freedom) could not locate pod_build_timings.sh's own member-free call site in the truncated driver to build the revert-RED fixture"
    fi
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (q/real-build) the real library-crate fixture pod_seed_target.sh's
# member-freedom doc points to: a REAL two-member cargo workspace (lib
# jammi-zzlib + bin jammi-zzbin), a REAL `cargo build` + `cargo build
# --release`, the artifact list taken from a REAL `find` (never hand-typed
# filenames) — every one of those real lib*.rlib/.rmeta/.d/.fingerprint/
# build entries must trip pod_seed_assert_member_free; after `cargo clean
# --workspace` (both profiles) + the incremental/ rm this SAME script's
# member-free-clean step performs, none may. The scratch workspace lives
# under `mktemp -d` and is removed at the end of this leg regardless of
# outcome (disk hygiene — a real cargo build/clean cycle leaves real bytes
# behind).
# ═════════════════════════════════════════════════════════════════════════
{
  Q_ROOT="$(mktemp -d)"
  q_cleanup() { rm -rf "$Q_ROOT"; }

  mkdir -p "$Q_ROOT/ws/jammi-zzlib/src" "$Q_ROOT/ws/jammi-zzbin/src"
  cat > "$Q_ROOT/ws/Cargo.toml" <<'EOF'
[workspace]
members = ["jammi-zzlib", "jammi-zzbin"]
resolver = "2"
EOF
  cat > "$Q_ROOT/ws/jammi-zzlib/Cargo.toml" <<'EOF'
[package]
name = "jammi-zzlib"
version = "0.1.0"
edition = "2021"
EOF
  echo 'pub fn f() -> i32 { 1 }' > "$Q_ROOT/ws/jammi-zzlib/src/lib.rs"
  cat > "$Q_ROOT/ws/jammi-zzbin/Cargo.toml" <<'EOF'
[package]
name = "jammi-zzbin"
version = "0.1.0"
edition = "2021"

[dependencies]
jammi-zzlib = { path = "../jammi-zzlib" }
EOF
  echo 'fn main() { jammi_zzlib::f(); }' > "$Q_ROOT/ws/jammi-zzbin/src/main.rs"

  Q_TGT="$Q_ROOT/tgt"
  # This fixture build must be hermetic to the CALLER's linker/warnings
  # posture: `RUSTFLAGS` (unlike `.cargo/config.toml`) is a process env var,
  # not directory-scoped, so a `cargo build` under this throwaway
  # `mktemp -d` workspace still inherits whatever the invoking shell
  # exports — including this guard leg's own job-scoped `RUSTFLAGS` (see
  # .github/actions/setup-rust-ci's "-D warnings -C link-arg=-fuse-ld=mold"
  # default, exported via $GITHUB_ENV for the whole "pod build substrate"
  # job). That default is correct INSIDE the `jammi-ai-ci` container image
  # (which bakes in mold — see this repo's own Dockerfile builder-stage
  # comment) but this guard leg runs on a bare `ubuntu-latest` runner with
  # no `container:` (the ONE guard leg that needs a real toolchain at all —
  # see the "Rust toolchain (pod build substrate only)" step), so `mold` is
  # not guaranteed to be on PATH there and `cc -fuse-ld=mold` fails
  # ("collect2: fatal error: cannot find 'ld'") even though the exact same
  # fixture links fine locally/on a pod with mold installed. This leg tests
  # pod_seed_assert_member_free's artifact-detection logic, not the
  # project's own reproducible-build posture, so it must not depend on
  # optional host tooling — unset the inherited flags and let `cc` pick
  # its own default linker.
  (cd "$Q_ROOT/ws" \
     && unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS \
     && CARGO_TARGET_DIR="$Q_TGT" cargo build -q \
     && CARGO_TARGET_DIR="$Q_TGT" cargo build --release -q) >/dev/null 2>"$Q_ROOT/build.err"
  q_build_rc=$?
  if [ "$q_build_rc" -ne 0 ]; then
    bad "(q/real-build) real cargo build of the a2fix fixture workspace failed (rc=$q_build_rc): $(cat "$Q_ROOT/build.err")"
    q_cleanup
  else
    # Real artifact list from `find`, scoped to the SAME four subdirs
    # pod_seed_assert_member_free itself scans ({.fingerprint,deps,build}
    # under debug/release; incremental checked separately below) — never a
    # hand-typed filename.
    find "$Q_TGT/debug/.fingerprint" "$Q_TGT/debug/deps" "$Q_TGT/debug/build" \
         "$Q_TGT/release/.fingerprint" "$Q_TGT/release/deps" "$Q_TGT/release/build" \
         -mindepth 1 -maxdepth 1 -iname '*zzlib*' 2>/dev/null | sort > "$Q_ROOT/expected.txt"
    q_expected_count="$(wc -l < "$Q_ROOT/expected.txt" | tr -d ' ')"

    Q_DRIVER="$SANDBOX/q_driver.sh"
    cat > "$Q_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_assert_member_free "\$1" "\$2"
DRV
    chmod +x "$Q_DRIVER"

    bash "$Q_DRIVER" "$Q_TGT" "$Q_ROOT/ws" > "$Q_ROOT/before.out" 2>&1
    q_before_rc=$?
    q_missing=0
    while IFS= read -r expected_path; do
      [ -n "$expected_path" ] || continue
      grep -qF "$expected_path" "$Q_ROOT/before.out" || q_missing=$((q_missing + 1))
    done < "$Q_ROOT/expected.txt"
    if [ "$q_before_rc" -eq 1 ] && [ "${q_expected_count:-0}" -ge 4 ] && [ "$q_missing" -eq 0 ]; then
      ok "(q/real-build) a REAL cargo build (lib+bin workspace) — every one of ${q_expected_count} real lib*.rlib/.rmeta/.d/.fingerprint/build entries from a real 'find' trips pod_seed_assert_member_free (rc=1)"
    else
      bad "(q/real-build) expected rc=1 with all ${q_expected_count:-0} real artifacts flagged (rc=$q_before_rc, missing=$q_missing): $(cat "$Q_ROOT/before.out")"
    fi

    # `cargo clean` never invokes the linker, but stays symmetric with the
    # build above (unset, not just left alone) so a future addition to this
    # block can't silently reintroduce the inherited-RUSTFLAGS class here.
    (cd "$Q_ROOT/ws" \
       && unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS \
       && CARGO_TARGET_DIR="$Q_TGT" cargo clean --workspace --frozen -q \
       && CARGO_TARGET_DIR="$Q_TGT" cargo clean --workspace --release --frozen -q) >/dev/null 2>"$Q_ROOT/clean.err"
    q_clean_rc=$?
    rm -rf "$Q_TGT"/*/incremental

    bash "$Q_DRIVER" "$Q_TGT" "$Q_ROOT/ws" > "$Q_ROOT/after.out" 2>&1
    q_after_rc=$?
    if [ "$q_clean_rc" -eq 0 ] && [ "$q_after_rc" -eq 0 ]; then
      ok "(q/real-build) after 'cargo clean --workspace' (both profiles) + the incremental/ rm, the SAME real target dir trips NOTHING (rc=0)"
    else
      bad "(q/real-build) expected a clean pass after cargo clean (clean_rc=$q_clean_rc, after_rc=$q_after_rc): $(cat "$Q_ROOT/after.out")"
    fi
  fi
  q_cleanup
}

# ═════════════════════════════════════════════════════════════════════════
# (r/byte-equal) byte_equal's tri-state (invalid/true/false, plus
# set_mismatch) in pod_build_timings.sh, exercised against
# the REAL bytes of the tri-state decision + JSON-assembly block (sed-
# extracted, never reimplemented) via a minimal harness supplying only the
# inputs that block reads (clone_hashes/cold_hashes/clone_features/
# cold_features and the handful of other positional values the real JSON
# writer takes). Three legs: empty snapshot on either side -> INVALID
# (rc!=0, byte_equal_state: invalid in the artifact); equal non-empty
# hashes -> true; differing non-empty hashes -> false. Revert-RED on the
# `>= 1` floor: a bare `[ "$clone_hashes" = "$cold_hashes" ]` (no
# empty-set guard) reads two empty strings as equal — a silent, wrong
# "true" on the SAME empty-snapshot fixture that correctly reads INVALID.
# ═════════════════════════════════════════════════════════════════════════
{
  PBT_SH="$REPO_ROOT/ci/scripts/perf/pod_build_timings.sh"
  # The tri-state block starts at clone_paths= (the SET comparison behind
  # byte_equal="set_mismatch") — the extraction must include it, or
  # clone_paths/cold_paths are unbound (set -u) inside the harness below.
  R_TRISTATE_START="$(grep -n '^clone_paths="\$(printf' "$PBT_SH" | head -1 | cut -d: -f1)"
  R_TRISTATE_END="$(awk -v s="$R_TRISTATE_START" 'NR>=s && /^fi$/{print NR; exit}' "$PBT_SH")"

  r_run_tristate() { # $1=clone_hashes $2=cold_hashes -> prints "byte_equal=<val> diff_nonempty=<yes|no>" on stdout
    local clone_hashes="$1" cold_hashes="$2"
    local driver="$SANDBOX/r_tristate_driver.sh"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      # The extracted range calls pod_sha256_of_stdin (for
      # clone_path_set_sha256/cold_path_set_sha256) — sourced from the
      # real pod_seed_target.sh, never reimplemented.
      echo "# shellcheck disable=SC1091"
      echo ". '$REPO_ROOT/ci/scripts/pod_seed_target.sh'"
      printf 'clone_hashes=%q\n' "$clone_hashes"
      printf 'cold_hashes=%q\n' "$cold_hashes"
      sed -n "${R_TRISTATE_START},${R_TRISTATE_END}p" "$PBT_SH"
      echo 'printf "byte_equal=%s diff_nonempty=%s\n" "$byte_equal" "$([ -n "$byte_equal_diff" ] && echo yes || echo no)"'
    } > "$driver"
    bash "$driver"
  }

  if [ -n "$R_TRISTATE_START" ] && [ -n "$R_TRISTATE_END" ]; then
    r_out="$(r_run_tristate "" "")"
    if grep -q 'byte_equal=invalid diff_nonempty=yes' <<<"$r_out"; then
      ok "(r/byte-equal) an empty snapshot on BOTH sides -> byte_equal=invalid, with a non-empty explanatory diff"
    else
      bad "(r/byte-equal) expected byte_equal=invalid for a doubly-empty snapshot: $r_out"
    fi

    r_out="$(r_run_tristate "" $'path\tsha')"
    if grep -q 'byte_equal=invalid' <<<"$r_out"; then
      ok "(r/byte-equal) an empty snapshot on ONE side (clone) -> byte_equal=invalid"
    else
      bad "(r/byte-equal) expected byte_equal=invalid for a one-sided-empty snapshot: $r_out"
    fi

    r_out="$(r_run_tristate $'a\tsha1' $'a\tsha1')"
    if grep -q 'byte_equal=true diff_nonempty=no' <<<"$r_out"; then
      ok "(r/byte-equal) equal, non-empty hash sets -> byte_equal=true"
    else
      bad "(r/byte-equal) expected byte_equal=true for identical non-empty snapshots: $r_out"
    fi

    r_out="$(r_run_tristate $'a\tsha1' $'a\tsha2')"
    if grep -q 'byte_equal=false diff_nonempty=yes' <<<"$r_out"; then
      ok "(r/byte-equal) differing, non-empty hash sets -> byte_equal=false, with a real diff"
    else
      bad "(r/byte-equal) expected byte_equal=false for differing non-empty snapshots: $r_out"
    fi

    # A clone snapshot with an EXTRA path the cold side doesn't have (the
    # shape a debug/-vs-release/ scoping mistake produces at the SET level;
    # this proves the tri-state ITSELF, independent of snapshot_hashes'
    # own release/ scoping) must be named "set_mismatch", never collapsed
    # into a bare "false" that reads exactly like a genuine
    # byte-reproducibility regression.
    r_out="$(r_run_tristate "$(printf 'a\tsha1\nb\tsha2')" $'a\tsha1')"
    if grep -q 'byte_equal=set_mismatch diff_nonempty=yes' <<<"$r_out"; then
      ok "(r/byte-equal) a clone snapshot with an EXTRA path the cold side lacks -> byte_equal=set_mismatch (never a bare false), with the symmetric difference"
    else
      bad "(r/byte-equal) expected byte_equal=set_mismatch for a path-set mismatch: $r_out"
    fi

    # revert-RED: the `>= 1` / non-vacuity floor, removed.
    r_reverted_out="$(
      driver="$SANDBOX/r_revert_driver.sh"
      {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        echo 'clone_hashes=""'
        echo 'cold_hashes=""'
        echo 'if [ "$clone_hashes" = "$cold_hashes" ]; then byte_equal=true; byte_equal_diff=""; else byte_equal=false; byte_equal_diff="differ"; fi'
        echo 'printf "byte_equal=%s\n" "$byte_equal"'
      } > "$driver"
      bash "$driver"
    )"
    if grep -q 'byte_equal=true' <<<"$r_reverted_out"; then
      ok "(r/byte-equal revert-RED) a bare-comparison form (no empty-set guard), on the SAME doubly-empty fixture, reads a silent WRONG 'true' — the invalid-state guard is load-bearing, not vacuous"
    else
      bad "(r/byte-equal revert-RED) expected the reverted form to read a false 'true' on the doubly-empty fixture: $r_reverted_out"
    fi
  else
    bad "(r/byte-equal) could not locate the byte_equal tri-state block in pod_build_timings.sh (start=${R_TRISTATE_START:-?} end=${R_TRISTATE_END:-?})"
  fi

  # JSON-written-after-the-validity-decision: byte_equal (the tri-state
  # decision) must be computed BEFORE the JSON assembly step that embeds
  # it — a structural, line-position check on the real script (the JSON is
  # a >100-line python heredoc; running it standalone needs a full set of
  # 17 positional args, out of proportion to what this specific ordering
  # claim needs).
  R_DECISION_LINE="$(grep -n '^if \[ -z "\$clone_hashes" \] || \[ -z "\$cold_hashes" \]; then$' "$PBT_SH" | head -1 | cut -d: -f1)"
  R_JSON_LINE="$(grep -n '^  python3 -' "$PBT_SH" | head -1 | cut -d: -f1)"
  if [ -n "$R_DECISION_LINE" ] && [ -n "$R_JSON_LINE" ] && [ "$R_DECISION_LINE" -lt "$R_JSON_LINE" ]; then
    ok "(r/byte-equal) the byte_equal validity decision (line ${R_DECISION_LINE}) runs BEFORE the JSON is assembled (line ${R_JSON_LINE}) — an INVALID run's own byte_equal_state is what lands in the artifact, never computed after the fact"
  else
    bad "(r/byte-equal) expected the validity decision to precede JSON assembly (decision=${R_DECISION_LINE:-?} json=${R_JSON_LINE:-?})"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (s/manifest) a synthetic-but-REAL-NAMED capture set — the literal
# `cargo:rerun-if-env-changed=<NAME>` announcements this repo's real
# `--features jammi-kernels/cuda` graph produces on a real seed build (see
# pod_seed_key_inputs.toml's own citation), a representative subset across
# all 11 announcing packages, never a smaller invented set —
# replayed through pod_seed_check_stdout_subset against the REAL, CURRENT
# manifest: must PASS. Revert-RED: the SAME capture set against a manifest
# copy with the `CC_*` wildcard removed must FAIL, naming the real unlisted
# names.
# ═════════════════════════════════════════════════════════════════════════
{
  MANIFEST_REAL="$REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
  S_CAPTURE="$SANDBOX/s_manifest_capture"
  rm -rf "$S_CAPTURE"; mkdir -p "$S_CAPTURE"
  # Names observed on a real seed build, one file per package, matching
  # cargo's own `<profile>__<pkg-dirname>.output` capture naming.
  cat > "$S_CAPTURE/release__zstd-sys-3db14a63a57ab829.output" <<'EOF'
cargo:rerun-if-env-changed=CC_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=CC_x86_64
cargo:rerun-if-env-changed=HOST_CC
cargo:rerun-if-env-changed=CFLAGS_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=CFLAGS_x86_64
cargo:rerun-if-env-changed=HOST_CFLAGS
cargo:rerun-if-env-changed=AR_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=AR_x86_64
cargo:rerun-if-env-changed=HOST_AR
cargo:rerun-if-env-changed=ARFLAGS_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=ARFLAGS_x86_64
cargo:rerun-if-env-changed=HOST_ARFLAGS
cargo:rerun-if-env-changed=CC_ENABLE_DEBUG_OUTPUT
cargo:rerun-if-env-changed=CC_FORCE_DISABLE
cargo:rerun-if-env-changed=ZSTD_SYS_USE_PKG_CONFIG
EOF
  cat > "$S_CAPTURE/release__cxx-6a8d7f42d5cebcb7.output" <<'EOF'
cargo:rerun-if-env-changed=CXX_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=CXX_x86_64
cargo:rerun-if-env-changed=HOST_CXX
cargo:rerun-if-env-changed=CXXFLAGS_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=CXXFLAGS_x86_64
cargo:rerun-if-env-changed=HOST_CXXFLAGS
cargo:rerun-if-env-changed=CARGO_MANIFEST_LINKS
EOF
  cat > "$S_CAPTURE/release__link-cplusplus-1ea710f5f6a71693.output" <<'EOF'
cargo:rerun-if-env-changed=CXXSTDLIB
cargo:rerun-if-env-changed=CXXSTDLIB_x86_64
cargo:rerun-if-env-changed=CXXSTDLIB_x86_64_unknown_linux_gnu
cargo:rerun-if-env-changed=HOST_CXXSTDLIB
EOF
  cat > "$S_CAPTURE/release__liblzma-sys-5090553e29feb974.output" <<'EOF'
cargo:rerun-if-env-changed=LZMA_API_STATIC
cargo:rerun-if-env-changed=LZMA_SYS_ENABLE_THREADS
EOF
  cat > "$S_CAPTURE/release__libsqlite3-sys-87badfb09dabf147.output" <<'EOF'
cargo:rerun-if-env-changed=LIBSQLITE3_FLAGS
cargo:rerun-if-env-changed=LIBSQLITE3_SYS_USE_PKG_CONFIG
cargo:rerun-if-env-changed=SQLITE_MAX_COLUMN
cargo:rerun-if-env-changed=SQLITE_MAX_EXPR_DEPTH
cargo:rerun-if-env-changed=SQLITE_MAX_VARIABLE_NUMBER
EOF
  cat > "$S_CAPTURE/release__onig_sys-c7636c527bb74bf4.output" <<'EOF'
cargo:rerun-if-env-changed=RUSTONIG_SYSTEM_LIBONIG
cargo:rerun-if-env-changed=RUSTONIG_STATIC_LIBONIG
cargo:rerun-if-env-changed=RUSTONIG_DYNAMIC_LIBONIG
EOF
  cat > "$S_CAPTURE/release__ring-c67535eaff4d1542.output" <<'EOF'
cargo:rerun-if-env-changed=RING_PREGENERATE_ASM
cargo:rerun-if-env-changed=CARGO_MANIFEST_LINKS
EOF
  cat > "$S_CAPTURE/release__numkong-145b6e225b03466d.output" <<'EOF'
cargo:rerun-if-env-changed=NK_MARCH_NATIVE
cargo:rerun-if-env-changed=NK_TARGET_ALDER
cargo:rerun-if-env-changed=NK_TARGET_SAPPHIRE
EOF
  cat > "$S_CAPTURE/debug__blake3-7de09f09feda0ae5.output" <<'EOF'
cargo:rerun-if-env-changed=CC_ENABLE_DEBUG_OUTPUT
EOF
  cat > "$S_CAPTURE/debug__psm-ce23a5f52daf72e5.output" <<'EOF'
cargo:rerun-if-env-changed=AR_x86_64
cargo:rerun-if-env-changed=CFLAGS_x86_64
EOF
  cat > "$S_CAPTURE/debug__usearch-0d714967eb0a6f36.output" <<'EOF'
cargo:rerun-if-env-changed=CXX_x86_64
cargo:rerun-if-env-changed=NK_DYNAMIC_DISPATCH
cargo:rerun-if-env-changed=NK_NATIVE_BF16
EOF

  S_DRIVER="$SANDBOX/s_manifest_driver.sh"
  cat > "$S_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$REPO_ROOT/ci/scripts/pod_seed_target.sh"
pod_seed_check_stdout_subset "\$1" "\$2"
DRV
  chmod +x "$S_DRIVER"

  s_out="$(bash "$S_DRIVER" "$S_CAPTURE" "$MANIFEST_REAL" 2>&1)"; s_rc=$?
  if [ "$s_rc" -eq 0 ]; then
    ok "(s/manifest) the REAL observed cc-family + package-specific announcements (11 packages, verbatim names), replayed against the CURRENT manifest, pass cleanly"
  else
    bad "(s/manifest) expected the current manifest to cover every real observed name (rc=$s_rc): $s_out"
  fi

  # revert-RED: the SAME capture set against a manifest missing the CC_*
  # wildcard family.
  S_MANIFEST_REVERTED="$SANDBOX/s_manifest_reverted.toml"
  python3 - "$MANIFEST_REAL" "$S_MANIFEST_REVERTED" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
needle = '"CC_*", "CXX_*", "AR_*", "RANLIB_*", "CFLAGS_*", "CXXFLAGS_*",'
replacement = '"CXX_*", "AR_*", "RANLIB_*", "CFLAGS_*", "CXXFLAGS_*",'
assert needle in t, "revert fixture: could not locate the CC_* wildcard entry to remove"
open(dst, "w").write(t.replace(needle, replacement, 1))
PY
  s_revert_out="$(bash "$S_DRIVER" "$S_CAPTURE" "$S_MANIFEST_REVERTED" 2>&1)"; s_revert_rc=$?
  if [ "$s_revert_rc" -ne 0 ] && grep -q 'CC_x86_64' <<<"$s_revert_out"; then
    ok "(s/manifest revert-RED) removing the CC_* wildcard from the SAME manifest makes the SAME real capture set fail, naming the real unlisted CC_x86_64* names — the wildcard is genuinely load-bearing"
  else
    bad "(s/manifest revert-RED) expected removing CC_* to reintroduce a real RED (rc=$s_revert_rc): $s_revert_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (t) class-shaped tripwire: every `2>/dev/null`, `|| true`, `|| :` site
# in the pod-substrate scripts must carry an in-script
# `# tripwire-ok: <reason>` allowlist annotation (same line, a contiguous
# comment block immediately above the site, or above the START of a
# backslash-continued multi-line statement the site is part of) — never an
# instance-shaped grep for one known-bad string, which misses every OTHER
# producing command that discards its stderr.
# ═════════════════════════════════════════════════════════════════════════
{
  T_SCANNER="$SANDBOX/t_tripwire_scanner.py"
  cat > "$T_SCANNER" <<'PYEOF'
import re, sys

# Also catch >/dev/null 2>&1, &>/dev/null, and
# 2>&- — the same "producing command's diagnostic silenced" class as
# 2>/dev/null, just spelled differently.
pat = re.compile(r'2>/dev/null|>/dev/null 2>&1|&>/dev/null|2>&-|\|\|\s*true\b|\|\|\s*:(\s|$)')

def is_annotated(lines, idx):
    if "tripwire-ok" in lines[idx]:
        return True
    j = idx
    while j > 0 and lines[j - 1].rstrip("\n").rstrip().endswith("\\"):
        j -= 1
        if "tripwire-ok" in lines[j]:
            return True
    k = j - 1
    while k >= 0 and lines[k].strip().startswith("#"):
        if "tripwire-ok" in lines[k]:
            return True
        k -= 1
    return False

bad = []
for path in sys.argv[1:]:
    lines = open(path, encoding="utf-8").readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        if pat.search(line) and not is_annotated(lines, i):
            bad.append("%s:%d: %s" % (path, i + 1, line.rstrip()))

if bad:
    print("UNANNOTATED:")
    for b in bad:
        print(b)
    sys.exit(1)
print("all sites annotated")
PYEOF

  T_FILES="$REPO_ROOT/ci/scripts/gpu-dev.sh $REPO_ROOT/ci/scripts/pod_seed_target.sh $REPO_ROOT/ci/scripts/pod_push_stamp.sh $REPO_ROOT/ci/scripts/perf/pod_build_timings.sh $REPO_ROOT/ci/scripts/pod_timing_lock.sh $REPO_ROOT/ci/scripts/pod_target_clone.sh $REPO_ROOT/ci/scripts/pod_provision_cutlass.sh"
  # shellcheck disable=SC2086
  t_out="$(python3 "$T_SCANNER" $T_FILES 2>&1)"; t_rc=$?
  if [ "$t_rc" -eq 0 ]; then
    ok "(t) every 2>/dev/null / || true / || : site in the pod-substrate scripts carries a tripwire-ok allowlist annotation"
  else
    bad "(t) unannotated tripwire site(s) found: $t_out"
  fi

  # revert-RED: strip ONE real annotation the scan above just relied on,
  # from a scratch COPY, and prove the SAME scanner catches it.
  T_MUTANT_DIR="$SANDBOX/t_mutant"
  rm -rf "$T_MUTANT_DIR"; mkdir -p "$T_MUTANT_DIR"
  T_MUTANT="$T_MUTANT_DIR/pod_push_stamp.sh"
  cp "$REPO_ROOT/ci/scripts/pod_push_stamp.sh" "$T_MUTANT"
  python3 - "$T_MUTANT" <<'PY'
import sys
p = sys.argv[1]
lines = open(p).readlines()
mutated = False
for i, l in enumerate(lines):
    if "tripwire-ok" in l and "2>/dev/null" in l:
        lines[i] = l.split("# tripwire-ok")[0].rstrip() + "\n"
        mutated = True
        break
if not mutated:
    raise SystemExit("no annotated 2>/dev/null line found to mutate")
open(p, "w").writelines(lines)
PY
  t_mutant_out="$(python3 "$T_SCANNER" "$T_MUTANT" 2>&1)"; t_mutant_rc=$?
  if [ "$t_mutant_rc" -ne 0 ] && grep -q 'UNANNOTATED' <<<"$t_mutant_out"; then
    ok "(t revert-RED) stripping ONE real tripwire-ok annotation from a scratch copy makes the scanner catch it — the scanner is genuinely load-bearing, not vacuous"
  else
    bad "(t revert-RED) expected the scanner to catch the stripped annotation (rc=$t_mutant_rc): $t_mutant_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (u) claim-tripwire: every verification claim the pod-substrate scripts
# make by citation (a backtick-quoted, parenthesised test label —
# `` `(label)` `` — e.g. `` `(q/real-build)` ``, `` `(m/provision drift)` ``)
# must name a label that genuinely exists as an `ok "(label)...`
# assertion in test_pod_substrate.sh — never a citation to a leg that is
# renamed, unwritten, or removed out from under the comment claiming it.
# ═════════════════════════════════════════════════════════════════════════
{
  U_SCANNER="$SANDBOX/u_claim_scanner.py"
  cat > "$U_SCANNER" <<'PYEOF'
import re, sys

citation_re = re.compile(r'`\(([^)]+)\)`')

def find_labels(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in citation_re.finditer(text):
        label = m.group(1).strip()
        if not label or not re.match(r'^[A-Za-z]', label):
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((path, line_no, label))
    return out

test_suite = sys.argv[1]
sources = sys.argv[2:]
suite_text = open(test_suite, encoding="utf-8").read()

bad = []
for src in sources:
    for path, line_no, label in find_labels(src):
        needle = 'ok "(%s)' % label
        if needle not in suite_text:
            bad.append("%s:%d: cites `(%s)` -- no such assertion label in %s" % (path, line_no, label, test_suite))

if bad:
    print("UNRESOLVED CLAIMS:")
    for b in bad:
        print(b)
    sys.exit(1)
print("every claim resolves to a real test label")
PYEOF

  U_TEST_SUITE="$REPO_ROOT/ci/scripts/test_pod_substrate.sh"
  U_SOURCES="$REPO_ROOT/ci/scripts/gpu-dev.sh $REPO_ROOT/ci/scripts/pod_seed_target.sh $REPO_ROOT/ci/scripts/pod_push_stamp.sh $REPO_ROOT/ci/scripts/perf/pod_build_timings.sh $REPO_ROOT/ci/scripts/pod_timing_lock.sh $REPO_ROOT/ci/scripts/pod_target_clone.sh $REPO_ROOT/ci/scripts/pod_provision_cutlass.sh $REPO_ROOT/ci/scripts/pod_seed_key_inputs.toml"
  # shellcheck disable=SC2086
  u_out="$(python3 "$U_SCANNER" "$U_TEST_SUITE" $U_SOURCES 2>&1)"; u_rc=$?
  if [ "$u_rc" -eq 0 ]; then
    ok "(u) every backtick-quoted test-label citation in the pod-substrate sources resolves to a real assertion in test_pod_substrate.sh"
  else
    bad "(u) unresolved verification claim(s): $u_out"
  fi

  # revert-RED: a scratch copy citing a label that does not exist.
  U_MUTANT_DIR="$SANDBOX/u_mutant"
  rm -rf "$U_MUTANT_DIR"; mkdir -p "$U_MUTANT_DIR"
  U_MUTANT="$U_MUTANT_DIR/fake_source.sh"
  printf '#!/usr/bin/env bash\n# verified against a real fixture -- see test_pod_substrate.sh'"'"'s `(z/DOES_NOT_EXIST)` leg\n' > "$U_MUTANT"
  u_mutant_out="$(python3 "$U_SCANNER" "$U_TEST_SUITE" "$U_MUTANT" 2>&1)"; u_mutant_rc=$?
  if [ "$u_mutant_rc" -ne 0 ] && grep -q 'z/DOES_NOT_EXIST' <<<"$u_mutant_out"; then
    ok "(u revert-RED) a citation to a label that genuinely does not exist is caught by the SAME scanner — not vacuous"
  else
    bad "(u revert-RED) expected the scanner to catch a nonexistent label citation (rc=$u_mutant_rc): $u_mutant_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (v/push) pod_push_stamp.sh against real fixtures (a notools bin/, an
# emptyrepo/, real repos): (1) PATH without sha256sum/shasum -> compute FAILS
# rc!=0 naming the tool, empty stdout (no stamp, poisoned or otherwise, is
# ever written); (2) a zero-file directory -> FAILS naming "empty
# manifest", never sha256(''); (3) determinism across two INDEPENDENT
# clones + an LC_ALL toggle -> byte-equal manifest_sha256, a one-byte
# content change flips it, and a revert-RED removing LC_ALL=C from the
# sort reproduces the cross-locale divergence the pin prevents (LC_ALL=C
# and LC_ALL=en_US.UTF-8 sort Banana.txt/apple.txt/Cherry.txt/banana.txt
# differently).
# ═════════════════════════════════════════════════════════════════════════
{
  PUSH_SH_REAL="$REPO_ROOT/ci/scripts/pod_push_stamp.sh"
  V_REAL_BASH="$(command -v bash)"

  # ---- leg 1: PATH without sha256sum/shasum ------------------------------
  V_NOTOOLS_BIN="$SANDBOX/v_notools_bin"
  rm -rf "$V_NOTOOLS_BIN"; mkdir -p "$V_NOTOOLS_BIN"
  for t in awk basename cat date dirname git mktemp python3 rm rsync sort stat; do
    V_REAL_TOOL="$(command -v "$t" 2>/dev/null || true)"
    [ -n "$V_REAL_TOOL" ] && ln -sf "$V_REAL_TOOL" "$V_NOTOOLS_BIN/$t"
  done

  V_REPO1="$SANDBOX/v_repo1"
  rm -rf "$V_REPO1"; mkdir -p "$V_REPO1"
  git init -q "$V_REPO1"
  git -C "$V_REPO1" config user.email t@t; git -C "$V_REPO1" config user.name t
  echo hello > "$V_REPO1/a.txt"
  git -C "$V_REPO1" add a.txt
  git -C "$V_REPO1" commit -q -m x

  V_NOTOOLS_STDOUT="$SANDBOX/v_notools_stdout.txt"
  V_NOTOOLS_STDERR="$SANDBOX/v_notools_stderr.txt"
  PATH="$V_NOTOOLS_BIN" "$V_REAL_BASH" "$PUSH_SH_REAL" compute "$V_REPO1" testsession > "$V_NOTOOLS_STDOUT" 2>"$V_NOTOOLS_STDERR"
  v_notools_rc=$?
  if [ "$v_notools_rc" -ne 0 ] && [ ! -s "$V_NOTOOLS_STDOUT" ] && grep -q 'sha256sum-or-shasum' "$V_NOTOOLS_STDERR"; then
    ok "(v/push notools) compute FAILS (rc=$v_notools_rc) naming the missing sha256sum-or-shasum tool, with EMPTY stdout — no stamp (poisoned or otherwise) is ever written"
  else
    bad "(v/push notools) expected a loud, empty-stdout refusal naming the missing tool (rc=$v_notools_rc, stdout_bytes=$(wc -c < "$V_NOTOOLS_STDOUT" | tr -d ' ')): $(cat "$V_NOTOOLS_STDERR")"
  fi

  # ---- leg 2: a zero-file directory ---------------------------------------
  V_EMPTYREPO="$SANDBOX/v_emptyrepo"
  rm -rf "$V_EMPTYREPO"; mkdir -p "$V_EMPTYREPO"
  V_EMPTY_STDOUT="$SANDBOX/v_empty_stdout.txt"
  V_EMPTY_STDERR="$SANDBOX/v_empty_stderr.txt"
  bash "$PUSH_SH_REAL" compute "$V_EMPTYREPO" testsession > "$V_EMPTY_STDOUT" 2>"$V_EMPTY_STDERR"
  v_empty_rc=$?
  if [ "$v_empty_rc" -ne 0 ] && grep -q 'empty manifest' "$V_EMPTY_STDERR" && [ ! -s "$V_EMPTY_STDOUT" ]; then
    ok "(v/push empty-dir) a zero-file repo-root FAILS naming 'empty manifest' — never sha256('') read as a computed digest"
  else
    bad "(v/push empty-dir) expected a loud 'empty manifest' refusal (rc=$v_empty_rc): $(cat "$V_EMPTY_STDERR")"
  fi

  # ---- leg 3: determinism across two independent clones + LC_ALL toggle --
  V_SRC_REPO="$SANDBOX/v_src_repo"
  rm -rf "$V_SRC_REPO"; mkdir -p "$V_SRC_REPO"
  git init -q "$V_SRC_REPO"
  git -C "$V_SRC_REPO" config user.email t@t; git -C "$V_SRC_REPO" config user.name t
  # Filenames deliberately chosen to sort DIFFERENTLY under LC_ALL=C vs a
  # real locale's collation (case-mixing) — a fixture that sorts the same
  # either way would prove nothing about the actual fix.
  printf 'a\n' > "$V_SRC_REPO/Banana.txt"
  printf 'b\n' > "$V_SRC_REPO/apple.txt"
  printf 'c\n' > "$V_SRC_REPO/Cherry.txt"
  printf 'd\n' > "$V_SRC_REPO/banana.txt"
  git -C "$V_SRC_REPO" add -A
  git -C "$V_SRC_REPO" commit -q -m seed

  V_CLONE1="$SANDBOX/v_clone1"; V_CLONE2="$SANDBOX/v_clone2"
  rm -rf "$V_CLONE1" "$V_CLONE2"
  git clone -q "$V_SRC_REPO" "$V_CLONE1"
  git clone -q "$V_SRC_REPO" "$V_CLONE2"

  V_MANIFEST_DRIVER="$SANDBOX/v_manifest_driver.sh"
  cat > "$V_MANIFEST_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$PUSH_SH_REAL"
pod_push_manifest_sha256 "\$1"
DRV
  chmod +x "$V_MANIFEST_DRIVER"

  # en_US.UTF-8 is a declared need of this suite (ci/needs.toml
  # [need.en-us-locale]). glibc lists it as `en_US.utf8`, macOS as
  # `en_US.UTF-8`; both answer to LC_ALL=en_US.UTF-8. V_LOCALES is captured
  # and fed to grep as a here-string, never piped into `grep -q`, whose
  # early exit can SIGPIPE the producer and read, under pipefail, as a
  # false "not found".
  V_LOCALES="$(locale -a 2>/dev/null)"
  if ! grep -qixE 'en_US\.utf-?8' <<<"$V_LOCALES"; then
    bad "(v/push locale) the cross-locale legs need the en_US.UTF-8 locale (ci/needs.toml need 'en-us-locale': glibc-langpack-en); this host's locale -a does not list it"
  fi

  V_SHA_C="$(LC_ALL=C bash "$V_MANIFEST_DRIVER" "$V_CLONE1" 2>/dev/null)"
  V_SHA_OTHER="$(LC_ALL=en_US.UTF-8 bash "$V_MANIFEST_DRIVER" "$V_CLONE2" 2>/dev/null)"
  if [ -n "$V_SHA_C" ] && [ "$V_SHA_C" = "$V_SHA_OTHER" ]; then
    ok "(v/push determinism) two INDEPENDENT clones (different paths) of the same content, hashed under DIFFERENT caller LC_ALL settings, produce the byte-equal manifest_sha256 (${V_SHA_C})"
  else
    bad "(v/push determinism) expected byte-equal manifest_sha256 across independent clones/locales (C=${V_SHA_C} other=${V_SHA_OTHER})"
  fi

  # A one-byte content change must flip the hash.
  printf 'CHANGED\n' > "$V_CLONE1/apple.txt"
  V_SHA_CHANGED="$(LC_ALL=C bash "$V_MANIFEST_DRIVER" "$V_CLONE1" 2>/dev/null)"
  if [ -n "$V_SHA_CHANGED" ] && [ "$V_SHA_CHANGED" != "$V_SHA_C" ]; then
    ok "(v/push determinism) a one-byte content change in ONE clone flips manifest_sha256"
  else
    bad "(v/push determinism) expected a content change to flip the manifest sha (before=${V_SHA_C} after=${V_SHA_CHANGED})"
  fi

  # revert-RED: LC_ALL=C removed from the sort, on a scratch copy.
  V_PUSH_REVERTED_DIR="$SANDBOX/v_push_reverted"
  rm -rf "$V_PUSH_REVERTED_DIR"; mkdir -p "$V_PUSH_REVERTED_DIR"
  V_PUSH_REVERTED="$V_PUSH_REVERTED_DIR/pod_push_stamp_reverted.sh"
  python3 - "$PUSH_SH_REAL" "$V_PUSH_REVERTED" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
needle = '| LC_ALL=C sort > "$manifest"'
replacement = '| sort > "$manifest"'
assert needle in t, "revert fixture: could not locate the LC_ALL=C sort call to neuter"
open(dst, "w").write(t.replace(needle, replacement, 1))
PY
  if bash -n "$V_PUSH_REVERTED"; then
    V_REVERT_DRIVER="$SANDBOX/v_manifest_revert_driver.sh"
    cat > "$V_REVERT_DRIVER" <<DRV
#!/usr/bin/env bash
set -uo pipefail
# shellcheck disable=SC1091
. "$V_PUSH_REVERTED"
pod_push_manifest_sha256 "\$1"
DRV
    chmod +x "$V_REVERT_DRIVER"
    # glibc's en_US collation folds case (apple, Banana, banana, Cherry);
    # C sorts by byte (Banana, Cherry, apple, banana). A locale that sorts
    # the fixture identically makes this leg prove nothing, so it fails.
    V_REVERT_SHA_C="$(LC_ALL=C bash "$V_REVERT_DRIVER" "$V_CLONE2" 2>/dev/null)"
    V_REVERT_SHA_OTHER="$(LC_ALL=en_US.UTF-8 bash "$V_REVERT_DRIVER" "$V_CLONE2" 2>/dev/null)"
    if [ -z "$V_REVERT_SHA_C" ] || [ -z "$V_REVERT_SHA_OTHER" ]; then
      bad "(v/push revert-RED) the pin-removed copy computed no manifest (C=${V_REVERT_SHA_C:-<none>} en_US=${V_REVERT_SHA_OTHER:-<none>})"
    elif [ "$V_REVERT_SHA_C" != "$V_REVERT_SHA_OTHER" ]; then
      ok "(v/push revert-RED) removing LC_ALL=C from the sort makes the SAME clone hash DIFFERENTLY under LC_ALL=C vs en_US.UTF-8 (${V_REVERT_SHA_C} != ${V_REVERT_SHA_OTHER}) — the LC_ALL=C pin is load-bearing"
    else
      bad "(v/push revert-RED) expected LC_ALL=C and en_US.UTF-8 to sort the case-mixed fixture differently once the pin is removed (C=${V_REVERT_SHA_C} en_US=${V_REVERT_SHA_OTHER}); this host's en_US.UTF-8 collation does not fold case"
    fi
  else
    bad "(v/push revert-RED) reverted pod_push_stamp.sh has a syntax error"
  fi

  rm -rf "$V_NOTOOLS_BIN" "$V_REPO1" "$V_EMPTYREPO" "$V_SRC_REPO" "$V_CLONE1" "$V_CLONE2" "$V_PUSH_REVERTED_DIR"
}

# ═════════════════════════════════════════════════════════════════════════
# (w/cutlass) the seed's CUTLASS provisioning on a fresh clone: a REAL git
# clone whose HEAD == its origin/main (the provenance that runs T1b —
# rp_bootstrap's `git clone`, which never inits submodules), carrying the
# crates/jammi-kernels/third_party/cutlass GITLINK with the submodule NOT
# initialized (include/ absent). Hermetic: local repos only, no network;
# the one cargo call is the offline tarball listing at the end. The leg
# drives ONLY the provisioning hunk's own REAL bytes (pod_seed_target.sh's
# `cutlass_inc=` block, sed-extracted by anchor — never a
# re-implementation in the test body). Fixture non-vacuity is asserted
# BEFORE the run (git-repo AND HEAD==origin/main AND include/ absent —
# else the leg aborts INVALID and drives nothing). GREEN: the arm runs
# `git submodule update --init --force --checkout --depth 1` and include/
# exists at the pinned commit afterwards. revert-RED: the SAME hunk with
# the update command neutered in a SCRATCH COPY leaves include/ absent on
# the same provenance, after which T1b dies in build.rs's 'CUTLASS
# submodule is not checked out' panic (the cargo half of that chain needs
# a live pod). Half-deleted state: submodule initialized, then worktree
# contents removed while the metadata still claims the pinned commit —
# plain `git submodule update --init --depth 1` exits 0 leaving include/
# absent (so --force --checkout is load-bearing), while the hunk restores
# it. Also: the predicate-mismatch state (include/ present, cutlass.h
# absent — the file-level guard re-provisions, a dir-exists guard skips),
# the guard lattice's two remaining arms (idempotence on a provisioned
# tree; non-git skip), provisioning on a non-main branch, and the tarball
# golden (cargo package --list: no cutlass, no artifacts/, license trio
# present).
{
  SEED_TARGET_SH="$REPO_ROOT/ci/scripts/pod_seed_target.sh"
  W_ROOT="$SANDBOX/w_cutlass"
  rm -rf "$W_ROOT"; mkdir -p "$W_ROOT"
  w_git() { git -c protocol.file.allow=always -c init.defaultBranch=main "$@"; }

  # A local "cutlass upstream" with an include/ dir in its tree.
  W_UPSTREAM="$W_ROOT/cutlass-upstream"
  w_git init -q "$W_UPSTREAM"
  w_git -C "$W_UPSTREAM" config user.email t@t; w_git -C "$W_UPSTREAM" config user.name t
  mkdir -p "$W_UPSTREAM/include/cutlass"
  echo 'cutlass.h' > "$W_UPSTREAM/include/cutlass/cutlass.h"
  w_git -C "$W_UPSTREAM" add include
  w_git -C "$W_UPSTREAM" commit -q -m v1
  W_PIN_SHA="$(w_git -C "$W_UPSTREAM" rev-parse HEAD)"

  # The superproject pinning the gitlink; then a BARE "origin" and fresh
  # clones FROM it — so origin/main genuinely RESOLVES in each clone and
  # HEAD == origin/main by construction (rp_bootstrap's own provenance),
  # never a repo where origin/main is empty and T1b's gate silently skips.
  W_SUPER="$W_ROOT/super"
  w_git init -q "$W_SUPER"
  w_git -C "$W_SUPER" config user.email t@t; w_git -C "$W_SUPER" config user.name t
  mkdir -p "$W_SUPER/crates/jammi-kernels/third_party"
  # tripwire-ok: submodule-add progress chatter only; a genuine add failure
  # surfaces as a missing gitlink in the non-vacuity assertion below, which
  # aborts the leg INVALID — never a silent pass.
  w_git -C "$W_SUPER" submodule add -q "$W_UPSTREAM" crates/jammi-kernels/third_party/cutlass >/dev/null 2>&1
  w_git -C "$W_SUPER" add .gitmodules crates/jammi-kernels/third_party/cutlass
  w_git -C "$W_SUPER" commit -q -m "pin cutlass"
  W_ORIGIN="$W_ROOT/origin.git"
  w_git clone -q --bare "$W_SUPER" "$W_ORIGIN"

  w_fresh_clone() { # $1=dest — a fresh main clone of the bare origin
    rm -rf "$1"
    w_git clone -q "$W_ORIGIN" "$1"
  }

  W_SUB_REL="crates/jammi-kernels/third_party/cutlass"
  W_INC_REL="$W_SUB_REL/include"

  # w_build_driver <script> <clone-dir>: sed-extract the REAL provisioning
  # hunk (anchor: the `cutlass_inc=` assignment, through its closing
  # 6-space `fi`) into a runnable driver — the same verbatim-extraction
  # technique the (p2/member-freedom) and (n/fa2-leg) legs use. The hunk
  # sits OUTSIDE the `_seed_is_main`-only block, so its closing `fi` is at
  # 6-space indentation; the anchor tracks that nesting.
  w_build_driver() {
    local script="$1" clone="$2" driver="$SANDBOX/w_driver_$$_${RANDOM}.sh"
    local start end
    start="$(grep -n 'cutlass_inc="crates/jammi-kernels/third_party/cutlass/include"' "$script" | head -1 | cut -d: -f1)"
    end="$(awk -v s="$start" 'NR>s && /^      fi$/{print NR; exit}' "$script")"
    [ -n "$start" ] && [ -n "$end" ] || return 1
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      echo '# the fixture submodule is pulled over the local file transport,'
      echo '# which git >= 2.38 blocks for submodules unless allowed:'
      echo 'export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always'
      echo "cd '$clone' || exit 9"
      sed -n "${start},${end}p" "$script"
      echo 'echo W_HUNK_COMPLETED'
    } > "$driver"
    chmod +x "$driver"
    printf '%s' "$driver"
  }

  W_CLONE1="$W_ROOT/clone_green"
  w_fresh_clone "$W_CLONE1"
  W_DRIVER1="$(w_build_driver "$SEED_TARGET_SH" "$W_CLONE1")" || W_DRIVER1=""

  # Fixture non-vacuity, asserted BEFORE the run:
  # git-repo AND HEAD==origin/main AND gitlink pinned AND include/ absent.
  W_FIXTURE_VALID=0
  w_head="$(w_git -C "$W_CLONE1" rev-parse HEAD)"
  w_omain="$(w_git -C "$W_CLONE1" rev-parse --verify --quiet origin/main)"
  w_gitlink_entry="$(w_git -C "$W_CLONE1" ls-tree HEAD -- "$W_SUB_REL")"
  # tripwire-ok: the probe's EXIT CODE is the assertion ("is the clone a git
  # repo at all"); a failure lands in the else-arm's INVALID abort below.
  if git -C "$W_CLONE1" rev-parse --git-dir >/dev/null 2>&1 \
     && [ -n "$w_head" ] && [ "$w_head" = "$w_omain" ] \
     && grep -q "^160000 commit $W_PIN_SHA" <<<"$w_gitlink_entry" \
     && [ ! -d "$W_CLONE1/$W_INC_REL" ] \
     && [ -n "$W_DRIVER1" ] && bash -n "$W_DRIVER1"; then
    W_FIXTURE_VALID=1
    ok "(w/cutlass fixture) non-vacuity: real git repo, HEAD==origin/main (${w_head}), gitlink pinned at ${W_PIN_SHA}, include/ absent — the exact triggering provenance, and the provisioning-hunk extraction is well-formed"
  else
    bad "(w/cutlass fixture) INVALID — head=${w_head:-?} origin/main=${w_omain:-?} gitlink='${w_gitlink_entry}' include_absent=$([ ! -d "$W_CLONE1/$W_INC_REL" ] && echo yes || echo no) driver='${W_DRIVER1:-EXTRACTION FAILED}'; the driven legs are aborted, not run against a vacuous fixture"
  fi

  if [ "$W_FIXTURE_VALID" = 1 ]; then
    # ---- GREEN: the real hunk provisions the submodule --------------------
    w_out="$(bash "$W_DRIVER1" 2>&1)"; w_rc=$?
    # tripwire-ok: a failed provisioning leaves no submodule repo to query —
    # the empty result already fails the sha assertion loudly just below.
    w_sub_head="$(w_git -C "$W_CLONE1/$W_SUB_REL" rev-parse HEAD 2>/dev/null)"
    if [ "$w_rc" -eq 0 ] \
       && grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$w_out" \
       && grep -q 'W_HUNK_COMPLETED' <<<"$w_out" \
       && [ -f "$W_CLONE1/$W_INC_REL/cutlass/cutlass.h" ] \
       && [ "$w_sub_head" = "$W_PIN_SHA" ]; then
      ok "(w/cutlass green) the REAL provisioning hunk, driven against the triggering provenance, runs 'git submodule update --init --force --checkout --depth 1' and include/ exists at the pinned commit afterwards"
    else
      bad "(w/cutlass green) expected the hunk to provision include/ at ${W_PIN_SHA} (rc=$w_rc, sub_head=${w_sub_head:-none}): $w_out"
    fi

    # ---- revert-RED: the hunk with the update command neutered ------------
    # ANCHOR COUPLING: removing the whole hunk also removes the
    # `cutlass_inc=` extraction anchor, so the fixture collapses to INVALID
    # (the (w/cutlass fixture) assertion catches the failed extraction and
    # aborts) rather than red-ing the green leg. The surgical single-line
    # neuter below is what shows the DISCRIMINATING failure — either way
    # the suite goes red, but only the surgical form shows the
    # include/-left-absent behavior itself.
    W_REVERT_DIR="$SANDBOX/w_revert"
    rm -rf "$W_REVERT_DIR"; mkdir -p "$W_REVERT_DIR"
    W_REVERTED="$W_REVERT_DIR/pod_seed_target_reverted.sh"
    cp "$SEED_TARGET_SH" "$W_REVERTED"
    python3 - "$W_REVERTED" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = "        git submodule update --init --force --checkout --depth 1 crates/jammi-kernels/third_party/cutlass || {\n"
new = "        true || {\n"
assert old in t, "revert fixture: could not locate the provisioning command to neuter"
open(p, "w").write(t.replace(old, new, 1))
PY
    W_CLONE2="$W_ROOT/clone_revert"
    w_fresh_clone "$W_CLONE2"
    W_DRIVER2="$(w_build_driver "$W_REVERTED" "$W_CLONE2")" || W_DRIVER2=""
    if [ -n "$W_DRIVER2" ] && bash -n "$W_DRIVER2" && [ ! -d "$W_CLONE2/$W_INC_REL" ]; then
      w2_out="$(bash "$W_DRIVER2" 2>&1)"; w2_rc=$?
      if [ "$w2_rc" -eq 0 ] && [ ! -d "$W_CLONE2/$W_INC_REL" ]; then
        ok "(w/cutlass revert-RED) the SAME hunk with the update command neutered in a scratch copy sails through (rc=0) leaving include/ ABSENT on the same provenance — the silent state that then panics in build.rs, so the provisioning command is load-bearing"
      else
        bad "(w/cutlass revert-RED) expected the neutered hunk to leave include/ absent with rc=0 (rc=$w2_rc, include_absent=$([ ! -d "$W_CLONE2/$W_INC_REL" ] && echo yes || echo no)): $w2_out"
      fi
    else
      bad "(w/cutlass revert-RED) scratch-copy fixture is broken (driver='${W_DRIVER2:-EXTRACTION FAILED}', include_absent=$([ ! -d "$W_CLONE2/$W_INC_REL" ] && echo yes || echo no))"
    fi

    # ---- half-deleted state: --force --checkout load-bearing -------------
    W_CLONE3="$W_ROOT/clone_halfdel"
    w_fresh_clone "$W_CLONE3"
    W_DRIVER3="$(w_build_driver "$SEED_TARGET_SH" "$W_CLONE3")" || W_DRIVER3=""
    w3_setup_out="$(bash "$W_DRIVER3" 2>&1)"; w3_setup_rc=$?
    W_SUB3="$W_CLONE3/$W_SUB_REL"
    find "$W_SUB3" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    # tripwire-ok: if the setup provisioning failed there is no submodule
    # repo to query — the empty sha already fails the fixture check loudly.
    w3_meta="$(w_git -C "$W_SUB3" rev-parse HEAD 2>/dev/null)"
    if [ "$w3_setup_rc" -eq 0 ] && [ ! -d "$W_CLONE3/$W_INC_REL" ] \
       && [ -e "$W_SUB3/.git" ] && [ "$w3_meta" = "$W_PIN_SHA" ]; then
      ok "(w/cutlass half-deleted fixture) submodule initialized, worktree contents removed, .git kept — metadata still claims the pinned commit while include/ is absent"
    else
      bad "(w/cutlass half-deleted fixture) could not construct the half-deleted state (setup_rc=$w3_setup_rc, meta=${w3_meta:-none}): $w3_setup_out"
    fi
    w3_old_out="$(w_git -C "$W_CLONE3" submodule update --init --depth 1 "$W_SUB_REL" 2>&1)"; w3_old_rc=$?
    if [ "$w3_old_rc" -eq 0 ] && [ ! -d "$W_CLONE3/$W_INC_REL" ]; then
      ok "(w/cutlass half-deleted RED) plain 'git submodule update --init --depth 1' exits 0 AND leaves include/ absent — a silent no-op on the half-deleted state, proving --force --checkout is load-bearing"
    else
      bad "(w/cutlass half-deleted RED) expected the plain update to no-op silently (rc=$w3_old_rc, include_absent=$([ ! -d "$W_CLONE3/$W_INC_REL" ] && echo yes || echo no)): $w3_old_out"
    fi
    w3_out="$(bash "$W_DRIVER3" 2>&1)"; w3_rc=$?
    if [ "$w3_rc" -eq 0 ] && [ -f "$W_CLONE3/$W_INC_REL/cutlass/cutlass.h" ]; then
      ok "(w/cutlass half-deleted green) the hunk (--force --checkout) restores include/ on the SAME half-deleted state the plain update just no-opped on"
    else
      bad "(w/cutlass half-deleted green) expected the hunk to restore include/ (rc=$w3_rc): $w3_out"
    fi

    # ---- predicate mismatch: the guard is BUILD.RS'S OWN predicate
    # (include/cutlass/cutlass.h, a FILE), never the coarser dir-exists
    # check — the state the two differ on: include/ EXISTS but
    # cutlass/cutlass.h is gone (interrupted checkout, partial copy). The
    # hunk must RE-provision; a `[ ! -d ]` guard skips. --------------------
    W_CLONE4="$W_ROOT/clone_predicate"
    w_fresh_clone "$W_CLONE4"
    W_DRIVER4="$(w_build_driver "$SEED_TARGET_SH" "$W_CLONE4")" || W_DRIVER4=""
    w4_setup_out="$(bash "$W_DRIVER4" 2>&1)"; w4_setup_rc=$?
    rm -rf "$W_CLONE4/$W_INC_REL/cutlass"
    if [ "$w4_setup_rc" -eq 0 ] && [ -d "$W_CLONE4/$W_INC_REL" ] \
       && [ ! -f "$W_CLONE4/$W_INC_REL/cutlass/cutlass.h" ]; then
      ok "(w/cutlass predicate fixture) include/ EXISTS while include/cutlass/cutlass.h is absent — the state a dir-exists guard cannot distinguish from provisioned"
    else
      bad "(w/cutlass predicate fixture) could not construct the include-present/header-absent state (setup_rc=$w4_setup_rc): $w4_setup_out"
    fi
    w4_out="$(bash "$W_DRIVER4" 2>&1)"; w4_rc=$?
    if [ "$w4_rc" -eq 0 ] && [ -f "$W_CLONE4/$W_INC_REL/cutlass/cutlass.h" ] \
       && grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$w4_out"; then
      ok "(w/cutlass predicate green) the file-level guard fires on the include-present/header-absent state and RE-provisions — cutlass.h restored"
    else
      bad "(w/cutlass predicate green) expected the hunk to re-provision cutlass.h (rc=$w4_rc): $w4_out"
    fi
    # revert-RED: the guard coarsened to a dir-exists form in a scratch
    # copy skips the SAME state — the file-level predicate is load-bearing,
    # not cosmetic.
    W_GUARD_REVERTED="$W_REVERT_DIR/pod_seed_target_dirguard.sh"
    cp "$SEED_TARGET_SH" "$W_GUARD_REVERTED"
    python3 - "$W_GUARD_REVERTED" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '[ ! -f "$cutlass_inc/cutlass/cutlass.h" ]'
new = '[ ! -d "$cutlass_inc" ]'
assert old in t, "guard-revert fixture: could not locate the file-level guard to coarsen"
open(p, "w").write(t.replace(old, new, 1))
PY
    rm -rf "$W_CLONE4/$W_INC_REL/cutlass"
    W_DRIVER4R="$(w_build_driver "$W_GUARD_REVERTED" "$W_CLONE4")" || W_DRIVER4R=""
    if [ -n "$W_DRIVER4R" ] && bash -n "$W_DRIVER4R"; then
      w4r_out="$(bash "$W_DRIVER4R" 2>&1)"; w4r_rc=$?
      if [ "$w4r_rc" -eq 0 ] && [ ! -f "$W_CLONE4/$W_INC_REL/cutlass/cutlass.h" ] \
         && ! grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$w4r_out"; then
        ok "(w/cutlass predicate revert-RED) a dir-exists guard, in a scratch copy, SKIPS the same include-present/header-absent state (no provisioning banner, cutlass.h stays absent) — the file-level predicate is load-bearing"
      else
        bad "(w/cutlass predicate revert-RED) expected the coarsened guard to skip (rc=$w4r_rc, header_absent=$([ ! -f "$W_CLONE4/$W_INC_REL/cutlass/cutlass.h" ] && echo yes || echo no)): $w4r_out"
      fi
    else
      bad "(w/cutlass predicate revert-RED) guard-reverted scratch fixture is broken (driver='${W_DRIVER4R:-EXTRACTION FAILED}')"
    fi

    # ---- lattice completion: the two remaining guard arms -----------------
    # (a) idempotence: clone1 is fully provisioned by the green leg above —
    # a second run must SKIP (no banner), rc=0, content and pinned sha
    # intact.
    w5_out="$(bash "$W_DRIVER1" 2>&1)"; w5_rc=$?
    # tripwire-ok: same post-state query as the green leg's — an empty sha
    # already fails the assertion loudly below.
    w5_sub_head="$(w_git -C "$W_CLONE1/$W_SUB_REL" rev-parse HEAD 2>/dev/null)"
    if [ "$w5_rc" -eq 0 ] && [ -f "$W_CLONE1/$W_INC_REL/cutlass/cutlass.h" ] \
       && [ "$w5_sub_head" = "$W_PIN_SHA" ] \
       && ! grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$w5_out"; then
      ok "(w/cutlass idempotence) a second run on the fully-provisioned clone SKIPS (no provisioning banner), rc=0, cutlass.h present, submodule still at the pinned sha"
    else
      bad "(w/cutlass idempotence) expected a clean skip on an already-provisioned clone (rc=$w5_rc, sub_head=${w5_sub_head:-none}): $w5_out"
    fi
    # (b) non-git skip: a plain directory (no .git anywhere up-tree —
    # $SANDBOX lives under mktemp) with the cutlass path but no include/ —
    # the hunk must SKIP with rc=0 and leave include/ absent (the loud
    # failure on this arm is build.rs's own panic, out of a no-cargo
    # suite's scope; only the skip is asserted).
    W_PLAIN="$W_ROOT/plain_nongit"
    rm -rf "$W_PLAIN"; mkdir -p "$W_PLAIN/$W_SUB_REL"
    W_DRIVER6="$(w_build_driver "$SEED_TARGET_SH" "$W_PLAIN")" || W_DRIVER6=""
    if [ -n "$W_DRIVER6" ] && bash -n "$W_DRIVER6"; then
      w6_out="$(bash "$W_DRIVER6" 2>&1)"; w6_rc=$?
      if [ "$w6_rc" -eq 0 ] && [ ! -d "$W_PLAIN/$W_INC_REL" ] \
         && ! grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$w6_out"; then
        ok "(w/cutlass non-git skip) a plain non-git tree with include/ absent is SKIPPED (rc=0, no banner, include/ still absent) — build.rs, not the hunk, owns the loud failure on that arm"
      else
        bad "(w/cutlass non-git skip) expected the hunk to skip a non-git tree (rc=$w6_rc, include_absent=$([ ! -d "$W_PLAIN/$W_INC_REL" ] && echo yes || echo no)): $w6_out"
      fi
    else
      bad "(w/cutlass non-git skip) non-git driver fixture is broken (driver='${W_DRIVER6:-EXTRACTION FAILED}')"
    fi
  fi

  # ---- (w/cutlass any-branch) CUTLASS provisioning must run on EVERY
  # branch, not only inside the `_seed_is_main`-only block beside T1b's own
  # build: measurement pods boot on a FEATURE branch, and a later
  # flash-attn build on such a pod (not this script's) panics in build.rs
  # when the header is absent. This leg drives the SAME
  # full dispatch (feat_rc gate, the provisioning guard, and the
  # main-only T1b-build gate together — the anchor range (n/seed-helpers)'s
  # SEEDH_T1B_ELSE/FI legs already track) against a NON-main fixture (HEAD !=
  # origin/main), asserting provisioning fires REGARDLESS, while T1b's own
  # build still does not run. cargo is out of scope for a no-cargo suite —
  # the one line that calls the real (cargo-backed) pod_seed_pkg_has_
  # feature is stubbed to a preset $feat_rc in the driver harness; every
  # OTHER byte, including the provisioning guard and the main-only gate
  # around T1b's build, is the script's own real bytes.
  w_build_full_driver() { # $1=script $2=clone $3=feat_rc $4=is_main
    local script="$1" clone="$2" feat_rc="$3" is_main="$4"
    local driver="$SANDBOX/w_full_driver_$$_${RANDOM}.sh"
    local start end
    start="$(grep -n '^    t1b_ran="false"$' "$script" | head -1 | cut -d: -f1)"
    end="$(awk -v s="$start" 'NR>s && /^    fi$/{print NR; exit}' "$script")"
    [ -n "$start" ] && [ -n "$end" ] || return 1
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      echo 'export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always'
      echo "cd '$clone' || exit 9"
      echo "feat_rc=$feat_rc"
      echo "_seed_is_main=\"$is_main\""
      echo '_seed_main_reason="test fixture: resolved sha != origin/main"'
      # the ONLY line neutered: the real (cargo-backed) metadata probe,
      # replaced by a no-op that leaves the harness-preset $feat_rc alone —
      # every other byte in the range is sed-extracted verbatim, never
      # reimplemented.
      sed -n "${start},${end}p" "$script" \
        | sed 's/pod_seed_pkg_has_feature jammi-kernels flash-attn || feat_rc=\$?/: # test-stub: feat_rc preset by the harness, no cargo in this suite/'
      echo 'echo W_HUNK_COMPLETED'
      echo 'printf "T1B_RAN=%s T1B_REASON=%s\n" "$t1b_ran" "$t1b_reason"'
    } > "$driver"
    chmod +x "$driver"
    printf '%s' "$driver"
  }

  w_branch_clone() { # $1=dest — a fresh clone checked out onto a feature
    # branch with a commit of its own, so HEAD diverges from origin/main
    # (a measurement pod's own provenance) rather than a same-sha detached
    # checkout.
    w_fresh_clone "$1"
    w_git -C "$1" config user.email t@t; w_git -C "$1" config user.name t
    w_git -C "$1" checkout -q -b feature/any-branch
    echo "feature work" > "$1/feature.txt"
    w_git -C "$1" add feature.txt
    w_git -C "$1" commit -q -m "feature commit (diverges HEAD from origin/main)"
  }

  W_CLONE_BRANCH="$W_ROOT/clone_branch"
  w_branch_clone "$W_CLONE_BRANCH"

  W_AB_HEAD="$(w_git -C "$W_CLONE_BRANCH" rev-parse HEAD)"
  W_AB_OMAIN="$(w_git -C "$W_CLONE_BRANCH" rev-parse --verify --quiet origin/main)"
  W_AB_GITLINK="$(w_git -C "$W_CLONE_BRANCH" ls-tree HEAD -- "$W_SUB_REL")"
  W_AB_FIXTURE_VALID=0
  if git -C "$W_CLONE_BRANCH" rev-parse --git-dir >/dev/null 2>&1 \
     && [ -n "$W_AB_HEAD" ] && [ -n "$W_AB_OMAIN" ] && [ "$W_AB_HEAD" != "$W_AB_OMAIN" ] \
     && grep -q "^160000 commit $W_PIN_SHA" <<<"$W_AB_GITLINK" \
     && [ ! -d "$W_CLONE_BRANCH/$W_INC_REL" ]; then
    W_AB_FIXTURE_VALID=1
    ok "(w/cutlass any-branch fixture) non-vacuity: real git repo, HEAD (${W_AB_HEAD}) != origin/main (${W_AB_OMAIN}), gitlink pinned at ${W_PIN_SHA}, include/ absent — a measurement pod's own provenance"
  else
    bad "(w/cutlass any-branch fixture) INVALID — head=${W_AB_HEAD:-?} origin/main=${W_AB_OMAIN:-?} same=$([ "$W_AB_HEAD" = "$W_AB_OMAIN" ] && echo yes || echo no) gitlink='${W_AB_GITLINK}' include_absent=$([ ! -d "$W_CLONE_BRANCH/$W_INC_REL" ] && echo yes || echo no); the driven legs are aborted, not run against a vacuous fixture"
  fi

  if [ "$W_AB_FIXTURE_VALID" = 1 ]; then
    # ---- GREEN: the hunk provisions REGARDLESS of branch ----------------
    W_AB_DRIVER="$(w_build_full_driver "$SEED_TARGET_SH" "$W_CLONE_BRANCH" 0 false)" || W_AB_DRIVER=""
    if [ -n "$W_AB_DRIVER" ] && bash -n "$W_AB_DRIVER"; then
      wab_out="$(bash "$W_AB_DRIVER" 2>&1)"; wab_rc=$?
      wab_sub_head="$(w_git -C "$W_CLONE_BRANCH/$W_SUB_REL" rev-parse HEAD 2>/dev/null)" # tripwire-ok: a failed provisioning leaves no submodule repo to query — the empty result already fails the sha assertion loudly just below
      if [ "$wab_rc" -eq 0 ] \
         && grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$wab_out" \
         && [ -f "$W_CLONE_BRANCH/$W_INC_REL/cutlass/cutlass.h" ] \
         && [ "$wab_sub_head" = "$W_PIN_SHA" ] \
         && ! grep -q 'T1b (main only)' <<<"$wab_out" \
         && grep -q 'T1B_RAN=false' <<<"$wab_out" \
         && grep -q 'main-only by design' <<<"$wab_out"; then
        ok "(w/cutlass any-branch green) on a feature-branch fixture (HEAD != origin/main) the header exists and the submodule is at the pinned sha AFTER the seed's provisioning step, while T1b's own build did NOT run and is still reported skipped as main-only"
      else
        bad "(w/cutlass any-branch green) expected provisioning to fire and T1b to stay skipped (rc=$wab_rc, sub_head=${wab_sub_head:-none}): $wab_out"
      fi
    else
      bad "(w/cutlass any-branch green) driver fixture is broken (driver='${W_AB_DRIVER:-EXTRACTION FAILED}')"
    fi

    # ---- RED control: a main-only placement leaves the header absent on
    # the SAME non-main fixture. Constructed by wrapping the real
    # provisioning guard inside `if [ "$_seed_is_main" = "true" ]; then
    # ... fi` in a scratch copy, rather than a hand-written
    # reimplementation.
    W_AB_OLDPLACEMENT="$SANDBOX/w_ab_oldplacement.sh"
    cp "$SEED_TARGET_SH" "$W_AB_OLDPLACEMENT"
    python3 - "$W_AB_OLDPLACEMENT" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '      cutlass_inc="crates/jammi-kernels/third_party/cutlass/include"'
new = '      if [ "$_seed_is_main" = "true" ]; then\n' + old
assert old in t, "old-placement fixture: could not locate the cutlass_inc anchor to wrap"
t = t.replace(old, new, 1)
anchor = '      fi\n      if [ "$_seed_is_main" = "true" ]; then\n        echo "=== T1b (main only)'
assert anchor in t, "old-placement fixture: could not locate the guard's closing fi to wrap"
closer = '      fi\n' + anchor
t = t.replace(anchor, closer, 1)
open(p, "w").write(t)
PY
    # A SEPARATE fresh feature-branch clone — W_CLONE_BRANCH was already
    # provisioned by the green leg above; the RED control needs its OWN
    # unprovisioned fixture on the SAME (non-main) provenance.
    W_CLONE_BRANCH2="$W_ROOT/clone_branch_red"
    w_branch_clone "$W_CLONE_BRANCH2"
    W_AB_OLD_DRIVER="$(w_build_full_driver "$W_AB_OLDPLACEMENT" "$W_CLONE_BRANCH2" 0 false)" || W_AB_OLD_DRIVER=""
    if [ -n "$W_AB_OLD_DRIVER" ] && bash -n "$W_AB_OLD_DRIVER" && [ ! -d "$W_CLONE_BRANCH2/$W_INC_REL" ]; then
      wabold_out="$(bash "$W_AB_OLD_DRIVER" 2>&1)"; wabold_rc=$?
      if [ "$wabold_rc" -eq 0 ] && [ ! -d "$W_CLONE_BRANCH2/$W_INC_REL" ] \
         && ! grep -q 'T1b prerequisite: provisioning the CUTLASS submodule' <<<"$wabold_out"; then
        ok "(w/cutlass any-branch revert-RED) a main-only placement, in a scratch copy, SKIPS provisioning on the SAME non-main fixture (include/ stays absent) — provisioning outside the main-only block is load-bearing"
      else
        bad "(w/cutlass any-branch revert-RED) expected a main-only placement to skip provisioning (rc=$wabold_rc, include_absent=$([ ! -d "$W_CLONE_BRANCH2/$W_INC_REL" ] && echo yes || echo no)): $wabold_out"
      fi
    else
      bad "(w/cutlass any-branch revert-RED) old-placement scratch fixture is broken (driver='${W_AB_OLD_DRIVER:-EXTRACTION FAILED}', include_absent=$([ ! -d "$W_CLONE_BRANCH2/$W_INC_REL" ] && echo yes || echo no))"
    fi
  fi

  # ---- tarball golden: the crates.io jammi-kernels tarball ships WITHOUT
  # CUTLASS (a publish checkout has no submodules, so a registry consumer
  # enabling flash-attn meets build.rs's refusal) and WITHOUT the committed
  # cuda-runs evidence artifacts, while the vendored flash-attention
  # subtree's license/provenance trio IS present. `--offline` keeps the
  # listing hermetic: it resolves from the local registry the guard's
  # `cargo-registry` need provides, and a cold registry fails naming the
  # crate it could not find. Independent of the sandbox git fixture above.
  W_PKG_LIST="$SANDBOX/w_pkg_list.txt"
  W_PKG_ERR="$SANDBOX/w_pkg_err.txt"
  ( cd "$REPO_ROOT" && cargo package --list -p jammi-kernels --allow-dirty --offline ) > "$W_PKG_LIST" 2>"$W_PKG_ERR"
  w_pkg_rc=$?
  if [ "$w_pkg_rc" -ne 0 ]; then
    bad "(w/cutlass tarball-golden) cargo package --list --offline failed (rc=$w_pkg_rc); it needs the warm registry ci/needs.toml's 'cargo-registry' need provides (cargo fetch --locked): $(cat "$W_PKG_ERR")"
  elif [ ! -s "$W_PKG_LIST" ]; then
    bad "(w/cutlass tarball-golden) cargo package --list produced EMPTY output — an empty tarball listing is a FAIL, never a vacuous pass"
  else
    w_pkg_paths="$(wc -l < "$W_PKG_LIST" | tr -d ' ')"
    w_cutlass_count="$(grep -c '^third_party/cutlass/' "$W_PKG_LIST")"
    w_artifacts_count="$(grep -c '^artifacts/' "$W_PKG_LIST")"
    if [ "$w_cutlass_count" -eq 0 ] && [ "$w_artifacts_count" -eq 0 ] \
       && grep -qx 'third_party/flash-attention/LICENSE' "$W_PKG_LIST" \
       && grep -qx 'third_party/flash-attention/AUTHORS' "$W_PKG_LIST" \
       && grep -qx 'third_party/flash-attention/VENDORED.md' "$W_PKG_LIST"; then
      ok "(w/cutlass tarball-golden) cargo package --list (${w_pkg_paths} paths): zero third_party/cutlass/ paths, zero artifacts/ paths, and the flash-attention LICENSE/AUTHORS/VENDORED.md trio present"
    else
      bad "(w/cutlass tarball-golden) tarball contents drifted (paths=${w_pkg_paths}, cutlass=${w_cutlass_count}, artifacts=${w_artifacts_count}, trio=$(grep -cxE 'third_party/flash-attention/(LICENSE|AUTHORS|VENDORED\.md)' "$W_PKG_LIST")/3)"
    fi
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (x) the seed's T3 clippy tuple and the flash-attn-only items it lints —
# the hermetic controls a no-CUDA host can run (the live oracle, the
# dead_code diagnostic naming all three symbols under `--features cuda`,
# needs a real CUDA toolchain):
#   (i)  a STRUCTURAL assertion that the four items in
#        crates/jammi-kernels/tests/cuda_parity.rs (FlashUpstreamMeasurement
#        struct AND its impl, max_abs_diff_finite_first, FlashFixtureKind)
#        each carry an ADJACENT `#[cfg(feature = "flash-attn")]` — walked
#        up through the contiguous attribute/doc block above each item
#        definition, never a mere file-wide grep count (which four gates
#        ANYWHERE in a 8000-line file would satisfy vacuously). An
#        empty/ambiguous definition match set is itself a FAIL. revert-RED:
#        stripping ONE gate in a scratch copy is caught by the same scanner.
#   (ii) the tuple lockstep's SEED-SIDE half: the exact T3 command line
#        parsed out of pod_seed_target.sh must be character-for-character
#        `cargo clippy -p jammi-kernels --all-targets --features cuda --
#        -D warnings` — RED on zero extracted tuples (an empty match set
#        must fail), RED on any drift; exact-tuple string equality, never
#        substring, so `cuda,flash-attn` can never satisfy `cuda`. NOT
#        covered here: merge-path REACHABILITY of a blocking twin, a
#        workflow property outside this suite's scope. `ci.yml`'s hermetic
#        `Clippy jammi-kernels --features flash-attn --all-targets` step is
#        the merge-path twin (`check_lint_surface_closure.py`'s own module
#        doc), never a second live-GPU copy.
{
  SEED_TARGET_SH="$REPO_ROOT/ci/scripts/pod_seed_target.sh"
  CUDA_PARITY_RS="$REPO_ROOT/crates/jammi-kernels/tests/cuda_parity.rs"

  # ---- (i) structural cfg-gate scanner ------------------------------------
  X_SCANNER="$SANDBOX/x_cfg_gate_scanner.py"
  cat > "$X_SCANNER" <<'PYEOF'
import re, sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()

ITEMS = [
    ("struct FlashUpstreamMeasurement", re.compile(r'^struct FlashUpstreamMeasurement\b')),
    ("impl FlashUpstreamMeasurement", re.compile(r'^impl FlashUpstreamMeasurement\b')),
    ("fn max_abs_diff_finite_first", re.compile(r'^fn max_abs_diff_finite_first\b')),
    ("enum FlashFixtureKind", re.compile(r'^enum FlashFixtureKind\b')),
]
CFG = re.compile(r'^#\[cfg\(feature\s*=\s*"flash-attn"\)\]$')

def gated(idx):
    # Walk UP through the CONTIGUOUS attribute/doc-comment block immediately
    # above the item definition: the gate must be ADJACENT to the item.
    # A blank line or any code line ends the walk — a gate elsewhere in the
    # file never counts.
    j = idx - 1
    while j >= 0:
        s = lines[j].strip()
        if CFG.match(s):
            return True
        if s.startswith("#[") or s.startswith("//"):
            j -= 1
            continue
        return False
    return False

bad = []
for name, pat in ITEMS:
    hits = [i for i, l in enumerate(lines) if pat.match(l)]
    if len(hits) != 1:
        bad.append("%s: expected exactly 1 definition, found %d -- an empty or ambiguous match set is a FAIL, never a pass" % (name, len(hits)))
        continue
    if not gated(hits[0]):
        bad.append('%s (line %d): no ADJACENT #[cfg(feature = "flash-attn")] gate' % (name, hits[0] + 1))

if bad:
    print("UNGATED:")
    for b in bad:
        print(b)
    sys.exit(1)
print("all four items carry an adjacent flash-attn cfg gate")
PYEOF

  x_out="$(python3 "$X_SCANNER" "$CUDA_PARITY_RS" 2>&1)"; x_rc=$?
  if [ "$x_rc" -eq 0 ]; then
    ok "(x/cfg-gates) all four cuda_parity.rs items (FlashUpstreamMeasurement struct+impl, max_abs_diff_finite_first, FlashFixtureKind) carry an ADJACENT #[cfg(feature = \"flash-attn\")] — structurally matched, not a file-wide grep count"
  else
    bad "(x/cfg-gates) $x_out"
  fi

  X_MUTANT_DIR="$SANDBOX/x_mutant"
  rm -rf "$X_MUTANT_DIR"; mkdir -p "$X_MUTANT_DIR"
  X_MUTANT="$X_MUTANT_DIR/cuda_parity_stripped.rs"
  cp "$CUDA_PARITY_RS" "$X_MUTANT"
  python3 - "$X_MUTANT" <<'PY'
import re, sys
p = sys.argv[1]
lines = open(p).readlines()
def_idx = next(i for i, l in enumerate(lines) if l.startswith("fn max_abs_diff_finite_first"))
cfg = re.compile(r'^#\[cfg\(feature\s*=\s*"flash-attn"\)\]\s*$')
target = None
j = def_idx - 1
while j >= 0:
    s = lines[j].strip()
    if cfg.match(s):
        target = j
        break
    if s.startswith("#[") or s.startswith("//"):
        j -= 1
        continue
    break
assert target is not None, "mutation fixture: no adjacent cfg gate found to strip"
del lines[target]
open(p, "w").writelines(lines)
PY
  x_mut_out="$(python3 "$X_SCANNER" "$X_MUTANT" 2>&1)"; x_mut_rc=$?
  if [ "$x_mut_rc" -ne 0 ] && grep -q 'max_abs_diff_finite_first' <<<"$x_mut_out"; then
    ok "(x/cfg-gates revert-RED) stripping ONE gate (max_abs_diff_finite_first's) in a scratch copy is caught by the SAME scanner, naming the item — the scanner is load-bearing, not vacuous"
  else
    bad "(x/cfg-gates revert-RED) expected the scanner to catch the stripped gate (rc=$x_mut_rc): $x_mut_out"
  fi

  # ---- (ii) T3 tuple lockstep, seed-side half -----------------------------
  X_T3_EXPECTED='cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings'
  x_check_t3_lockstep() { # $1=script -> 0 iff >=1 extracted non-comment clippy tuple AND every one is character-for-character the expected tuple
    local extracted
    extracted="$(grep -vE '^[[:space:]]*#' "$1" | grep -F 'cargo clippy' | sed -e 's/^[[:space:]]*//' -e 's/ || exit 1$//')"
    if [ -z "$extracted" ]; then
      echo "zero extracted clippy tuples (empty match set)"
      return 1
    fi
    local bad_tuple=0 line
    while IFS= read -r line; do
      if [ "$line" != "$X_T3_EXPECTED" ]; then
        echo "tuple drift: extracted '$line' != expected '$X_T3_EXPECTED'"
        bad_tuple=1
      fi
    done <<< "$extracted"
    return "$bad_tuple"
  }

  x_t3_out="$(x_check_t3_lockstep "$SEED_TARGET_SH")"; x_t3_rc=$?
  if [ "$x_t3_rc" -eq 0 ]; then
    ok "(x/t3-lockstep) pod_seed_target.sh's T3 command line is character-for-character '${X_T3_EXPECTED}' — the seed-side half of the tuple-lockstep spec"
  else
    bad "(x/t3-lockstep) $x_t3_out"
  fi

  X_NOCLIPPY="$SANDBOX/x_noclippy_fixture.sh"
  printf '#!/usr/bin/env bash\necho "no clippy invocation anywhere in this file"\n' > "$X_NOCLIPPY"
  x_empty_out="$(x_check_t3_lockstep "$X_NOCLIPPY")"; x_empty_rc=$?
  if [ "$x_empty_rc" -ne 0 ] && grep -q 'empty match set' <<<"$x_empty_out"; then
    ok "(x/t3-lockstep empty-RED) a source with ZERO extractable clippy tuples fails the check (an empty match set must fail) — never a vacuous pass"
  else
    bad "(x/t3-lockstep empty-RED) expected the empty-match-set to fail (rc=$x_empty_rc): $x_empty_out"
  fi

  X_T3_MUTANT="$X_MUTANT_DIR/pod_seed_target_t3_drift.sh"
  cp "$SEED_TARGET_SH" "$X_T3_MUTANT"
  python3 - "$X_T3_MUTANT" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = "    cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings || exit 1\n"
new = "    cargo clippy -p jammi-kernels --all-targets --features cuda,flash-attn -- -D warnings || exit 1\n"
assert old in t, "drift fixture: could not locate the T3 line to mutate"
open(p, "w").write(t.replace(old, new, 1))
PY
  x_drift_out="$(x_check_t3_lockstep "$X_T3_MUTANT")"; x_drift_rc=$?
  if [ "$x_drift_rc" -ne 0 ] && grep -q 'cuda,flash-attn' <<<"$x_drift_out"; then
    ok "(x/t3-lockstep drift-RED) mutating the T3 line to '--features cuda,flash-attn' in a scratch copy is caught — exact-tuple match, never substring: cuda,flash-attn does NOT satisfy cuda"
  else
    bad "(x/t3-lockstep drift-RED) expected the mutated tuple to fail the exact-match check (rc=$x_drift_rc): $x_drift_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (y/push-parent) the trees root on a fresh pod: `push --tree <name>`
# rsyncs to /root/trees/<name>/, but rsync creates only the LAST path
# component of its own destination, and nothing in the pod bootstrap or
# the build-substrate seed provisions /root/trees itself — so without
# help the very FIRST push on a fresh pod fails with `rsync: mkdir
# "/root/trees/<name>" failed: No such file or directory (2)`.
# `rp_push_ensure_parent` (runpod_lib.sh) runs a bounded, idempotent
# remote `mkdir -p` on the tree's parent BEFORE gpu-dev.sh's push case
# ever calls rsync.
# ═════════════════════════════════════════════════════════════════════════
{
  Y_GPU_DEV="$REPO_ROOT/ci/scripts/gpu-dev.sh"
  Y_RUNPOD_LIB="$REPO_ROOT/ci/scripts/runpod_lib.sh"

  # ---- structural: single-sourced, and BEFORE the rsync it protects ------
  y_ensure_line="$(grep -n 'rp_push_ensure_parent "\$TREE_DIR"' "$Y_GPU_DEV" | head -1 | cut -d: -f1)"
  y_rsync_line="$(grep -n 'rsync -azc --no-times --no-owner --no-group --delete' "$Y_GPU_DEV" | head -1 | cut -d: -f1)"
  if [ -n "$y_ensure_line" ] && [ -n "$y_rsync_line" ] && [ "$y_ensure_line" -lt "$y_rsync_line" ]; then
    ok "(y/push-parent) gpu-dev.sh's push case calls rp_push_ensure_parent \"\$TREE_DIR\" (line ${y_ensure_line}) BEFORE its own rsync (line ${y_rsync_line})"
  else
    bad "(y/push-parent) expected rp_push_ensure_parent \"\$TREE_DIR\" to precede push's own rsync in gpu-dev.sh (ensure_line=${y_ensure_line:-<none>} rsync_line=${y_rsync_line:-<none>})"
  fi

  # ---- reproduce the underlying defect on a REAL local rsync -------------
  # No ssh/network needed to reproduce this: a purely LOCAL rsync (no -e,
  # no host: syntax) exhibits the identical "creates only the last path
  # component" behaviour — verified live against this box's own rsync
  # binary, not asserted from the man page.
  Y_PODROOT="$SANDBOX/y_podroot"
  Y_SRC="$SANDBOX/y_src"
  rm -rf "$Y_PODROOT" "$Y_SRC"
  mkdir -p "$Y_PODROOT" "$Y_SRC"
  echo hi > "$Y_SRC/file.txt"
  Y_DEST="$Y_PODROOT/trees/freshtree"   # neither "trees" nor "freshtree" exists yet
  y_before_out="$(rsync -a "$Y_SRC/" "$Y_DEST/" 2>&1)"; y_before_rc=$?
  if [ "$y_before_rc" -ne 0 ] && grep -qi 'no such file or directory' <<<"$y_before_out"; then
    ok "(y/push-parent repro) a bare rsync into a destination whose PARENT ('trees') does not exist fails (rc=$y_before_rc): $(printf '%s' "$y_before_out" | head -1)"
  else
    bad "(y/push-parent repro) expected the bare rsync to fail naming a missing parent directory (rc=$y_before_rc): $y_before_out"
  fi
  [ -e "$Y_PODROOT/trees" ] && bad "(y/push-parent repro) the failed rsync must not have created 'trees' either" \
    || ok "(y/push-parent repro) the failed rsync left NO trace of 'trees' — confirms rsync created zero components here, not one"

  # ---- rp_push_ensure_parent, exercised for REAL (function-boundary ------
  # intercept on rp_run_remote — this suite's own established technique,
  # see leg (a)'s module doc: never mock ssh, capture/execute the heredoc
  # rp_run_remote would have sent). The heredoc payload here is a plain,
  # portable `mkdir -p '<path>'` — genuinely provable against a real local
  # filesystem without any transport involved, so the override both
  # CAPTURES the text (for the structural assertion below) AND actually
  # RUNS it locally, proving real behaviour, not merely matching text.
  # A real driver SCRIPT (not `bash -c` positionals): rp_run_remote is
  # itself called with ZERO arguments by rp_push_ensure_parent, so an
  # override that reads ITS OWN $1/$2 (scoped to how rp_run_remote was
  # invoked, not to this outer script's own args) would read empty values —
  # the capture path is threaded through an exported env var instead, the
  # same shape leg (a)'s own CAPTURE variable uses.
  Y_ENSURE_DRIVER="$SANDBOX/y_ensure_driver.sh"
  cat > "$Y_ENSURE_DRIVER" <<'DRV'
#!/usr/bin/env bash
set -uo pipefail
export RUNPOD_API_KEY=test-dummy-key
export RP_SESSION_ROOT="${Y_DRV_SESSIONS}"
export RP_SSH_CONFIG="${Y_DRV_SSH_CONFIG}"
mkdir -p "$RP_SESSION_ROOT"
# shellcheck disable=SC1090
. "${Y_DRV_LIB}"
rp_run_remote() { cat > "${Y_DRV_CAPTURE}"; bash "${Y_DRV_CAPTURE}"; }
rp_push_ensure_parent "${Y_DRV_TREE_DIR}"
DRV
  chmod +x "$Y_ENSURE_DRIVER"

  Y_CAPTURE="$SANDBOX/y_capture.txt"
  Y_DRV_LIB="$Y_RUNPOD_LIB" Y_DRV_TREE_DIR="$Y_DEST" Y_DRV_CAPTURE="$Y_CAPTURE" \
    Y_DRV_SESSIONS="$SANDBOX/y_sessions" Y_DRV_SSH_CONFIG="$SANDBOX/y_ssh_config" \
    bash "$Y_ENSURE_DRIVER" > "$SANDBOX/y_ensure.out" 2>&1
  y_ensure_rc=$?
  if [ "$y_ensure_rc" -eq 0 ] && [ -d "$Y_PODROOT/trees" ] && grep -qF "mkdir -p '${Y_PODROOT}/trees'" "$Y_CAPTURE"; then
    ok "(y/push-parent fix) rp_push_ensure_parent's captured remote command is mkdir -p on the tree's PARENT, and running it for real creates '${Y_PODROOT}/trees'"
  else
    bad "(y/push-parent fix) rp_push_ensure_parent did not provision the parent as expected (rc=$y_ensure_rc, captured: $(cat "$Y_CAPTURE" 2>/dev/null); out: $(cat "$SANDBOX/y_ensure.out"))"
  fi

  # ---- idempotent (a second call against an already-existing parent) ----
  Y_DRV_LIB="$Y_RUNPOD_LIB" Y_DRV_TREE_DIR="$Y_DEST" Y_DRV_CAPTURE="$SANDBOX/y_capture2.txt" \
    Y_DRV_SESSIONS="$SANDBOX/y_sessions" Y_DRV_SSH_CONFIG="$SANDBOX/y_ssh_config" \
    bash "$Y_ENSURE_DRIVER" > "$SANDBOX/y_ensure2.out" 2>&1
  [ $? -eq 0 ] && ok "(y/push-parent fix) a second rp_push_ensure_parent against the now-existing parent is a silent no-op (mkdir -p is idempotent)" \
    || bad "(y/push-parent fix) a repeat call against an existing parent must still succeed: $(cat "$SANDBOX/y_ensure2.out")"

  # ---- the SAME local rsync succeeds once the parent is provisioned -----
  y_after_out="$(rsync -a "$Y_SRC/" "$Y_DEST/" 2>&1)"; y_after_rc=$?
  if [ "$y_after_rc" -eq 0 ] && [ -f "$Y_DEST/file.txt" ]; then
    ok "(y/push-parent fix) the identical rsync that failed above succeeds once rp_push_ensure_parent has provisioned the parent"
  else
    bad "(y/push-parent fix) expected the rsync to succeed once the parent exists (rc=$y_after_rc): $y_after_out"
  fi

  # ---- revert-RED: neutering rp_push_ensure_parent's mkdir on a scratch
  # copy of runpod_lib.sh brings the missing-parent failure back, proving
  # this leg is genuinely mutant-killing, not vacuous.
  Y_MUTANT_LIB="$SANDBOX/y_runpod_lib_mutant.sh"
  cp "$Y_RUNPOD_LIB" "$Y_MUTANT_LIB"
  python3 - "$Y_MUTANT_LIB" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = "  rp_run_remote <<EOF\nset -uo pipefail\nmkdir -p '${parent_dir}'\nEOF\n"
new = "  :\n"
assert old in t, "revert fixture: could not locate rp_push_ensure_parent's mkdir heredoc"
open(p, "w").write(t.replace(old, new, 1))
PY
  Y_MUTANT_PODROOT="$SANDBOX/y_mutant_podroot"
  rm -rf "$Y_MUTANT_PODROOT"; mkdir -p "$Y_MUTANT_PODROOT"
  Y_MUTANT_DEST="$Y_MUTANT_PODROOT/trees/freshtree"
  Y_DRV_LIB="$Y_MUTANT_LIB" Y_DRV_TREE_DIR="$Y_MUTANT_DEST" Y_DRV_CAPTURE="$SANDBOX/y_capture_mutant.txt" \
    Y_DRV_SESSIONS="$SANDBOX/y_sessions_mutant" Y_DRV_SSH_CONFIG="$SANDBOX/y_ssh_config_mutant" \
    bash "$Y_ENSURE_DRIVER" > "$SANDBOX/y_mutant_ensure.out" 2>&1
  y_mutant_rsync_out="$(rsync -a "$Y_SRC/" "$Y_MUTANT_DEST/" 2>&1)"; y_mutant_rsync_rc=$?
  if [ "$y_mutant_rsync_rc" -ne 0 ] && grep -qi 'no such file or directory' <<<"$y_mutant_rsync_out"; then
    ok "(y/push-parent revert-RED) neutering rp_push_ensure_parent's mkdir on a scratch copy brings the missing-parent failure back (rc=$y_mutant_rsync_rc) — the mkdir is genuinely load-bearing"
  else
    bad "(y/push-parent revert-RED) expected the neutered copy to fail on the missing parent (rc=$y_mutant_rsync_rc): $y_mutant_rsync_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (z/cwd-guard) gpu-dev.sh's REPO_ROOT is derived from THIS SCRIPT's own
# on-disk location, never from $PWD. On a multi-worktree laptop, invoking
# one tree's script copy from inside a DIFFERENT tree would silently
# push/run/target the WRONG tree (the push-stamp's own laptop_head field
# being the only tell), so push/run/target REFUSE (exit 2, naming both
# paths) on a mismatch, overridable via RP_ALLOW_ROOT_MISMATCH=1.
# ═════════════════════════════════════════════════════════════════════════
{
  Z_GPU_DEV="$REPO_ROOT/ci/scripts/gpu-dev.sh"

  # ---- refuses from a plain (non-git) mismatched cwd ----------------------
  Z_PLAIN_CWD="$SANDBOX/z_plain_cwd"
  rm -rf "$Z_PLAIN_CWD"; mkdir -p "$Z_PLAIN_CWD"
  z_plain_out="$(cd "$Z_PLAIN_CWD" && RUNPOD_API_KEY=test-dummy-key bash "$Z_GPU_DEV" push somesession --tree x 2>&1)"; z_plain_rc=$?
  if [ "$z_plain_rc" -eq 2 ] && grep -qF "REPO_ROOT=${REPO_ROOT}" <<<"$z_plain_out" \
     && grep -qF "$Z_PLAIN_CWD" <<<"$z_plain_out" \
     && grep -q 'RP_ALLOW_ROOT_MISMATCH' <<<"$z_plain_out"; then
    ok "(z/cwd-guard) 'push' from a plain (non-git) mismatched cwd REFUSES (exit 2), naming REPO_ROOT, the cwd, and the override"
  else
    bad "(z/cwd-guard) expected a named refusal (rc=$z_plain_rc): $z_plain_out"
  fi

  # ---- refuses from a DIFFERENT real git checkout (exercises the git ------
  # rev-parse --show-toplevel branch, not just the plain-cwd fallback) -----
  Z_OTHER_REPO="$SANDBOX/z_other_repo"
  rm -rf "$Z_OTHER_REPO"; mkdir -p "$Z_OTHER_REPO"
  ( cd "$Z_OTHER_REPO" && git init -q && git config user.email a@b.c && git config user.name t \
      && echo hi > f.txt && git add f.txt && git commit -q -m init )
  Z_OTHER_TOPLEVEL="$(cd "$Z_OTHER_REPO" && git rev-parse --show-toplevel)"
  z_other_out="$(cd "$Z_OTHER_REPO" && RUNPOD_API_KEY=test-dummy-key bash "$Z_GPU_DEV" run somesession echo hi 2>&1)"; z_other_rc=$?
  if [ "$z_other_rc" -eq 2 ] && grep -qF "REPO_ROOT=${REPO_ROOT}" <<<"$z_other_out" \
     && grep -qF "$Z_OTHER_TOPLEVEL" <<<"$z_other_out"; then
    ok "(z/cwd-guard) 'run' from a DIFFERENT real git checkout REFUSES, naming the OTHER checkout's own git toplevel (not just \$PWD verbatim)"
  else
    bad "(z/cwd-guard) expected a named refusal citing the other repo's git toplevel (rc=$z_other_rc): $z_other_out"
  fi

  # ---- 'target' is gated too -----------------------------------------------
  z_target_out="$(cd "$Z_PLAIN_CWD" && RUNPOD_API_KEY=test-dummy-key bash "$Z_GPU_DEV" target somesession sometree 2>&1)"; z_target_rc=$?
  [ "$z_target_rc" -eq 2 ] && grep -qF "REPO_ROOT=${REPO_ROOT}" <<<"$z_target_out" \
    && ok "(z/cwd-guard) 'target' is gated by the same cwd-mismatch check as push/run" \
    || bad "(z/cwd-guard) expected 'target' to refuse too (rc=$z_target_rc): $z_target_out"

  # ---- the override opts in deliberately -----------------------------------
  z_override_out="$(cd "$Z_PLAIN_CWD" && RUNPOD_API_KEY=test-dummy-key RP_ALLOW_ROOT_MISMATCH=1 bash "$Z_GPU_DEV" push somesession --tree x 2>&1)"; z_override_rc=$?
  if ! grep -q 'RP_ALLOW_ROOT_MISMATCH=1 to override' <<<"$z_override_out" && [ "$z_override_rc" -ne 2 ]; then
    ok "(z/cwd-guard) RP_ALLOW_ROOT_MISMATCH=1 opts past the refusal (proceeds to the next real check — no live session — rather than the mismatch error; rc=$z_override_rc)"
  else
    bad "(z/cwd-guard) RP_ALLOW_ROOT_MISMATCH=1 did not bypass the mismatch refusal (rc=$z_override_rc): $z_override_out"
  fi

  # ---- the ordinary (matching) invocation is never false-positived --------
  z_ok_out="$(cd "$REPO_ROOT" && RUNPOD_API_KEY=test-dummy-key bash "$Z_GPU_DEV" push somesession --tree x 2>&1)"; z_ok_rc=$?
  if ! grep -q "this script's own location resolves to REPO_ROOT" <<<"$z_ok_out"; then
    ok "(z/cwd-guard) invoking from REPO_ROOT itself never trips the mismatch refusal (rc=$z_ok_rc, a real 'no live session' failure downstream is expected and fine)"
  else
    bad "(z/cwd-guard) a matching cwd must never trip the mismatch refusal: $z_ok_out"
  fi

  # ---- verbs OUTSIDE {push,run,target} are never gated ---------------------
  z_attach_out="$(cd "$Z_PLAIN_CWD" && RUNPOD_API_KEY=test-dummy-key bash "$Z_GPU_DEV" attach somesession 2>&1)"; z_attach_rc=$?
  if ! grep -q "this script's own location resolves to REPO_ROOT" <<<"$z_attach_out"; then
    ok "(z/cwd-guard) 'attach' (outside the gated verb set) is unaffected by a mismatched cwd (rc=$z_attach_rc, a real 'no live session' failure downstream is expected and fine)"
  else
    bad "(z/cwd-guard) 'attach' must never be gated by the cwd-mismatch check: $z_attach_out"
  fi

  # ---- revert-RED: a scratch copy with the guard neutered no longer -------
  # refuses — proving this leg is genuinely mutant-killing. gpu-dev.sh
  # resolves its own sibling runpod_lib.sh via $DIR (its own on-disk
  # location), so a real copy of runpod_lib.sh must sit alongside the
  # mutant for that resolution to find it (same technique used throughout
  # this suite's other revert-RED fixtures, e.g. (m/provision revert-RED)).
  Z_MUTANT_ROOT="$SANDBOX/z_mutant_repo/ci/scripts"
  rm -rf "$SANDBOX/z_mutant_repo"; mkdir -p "$Z_MUTANT_ROOT"
  cp "$REPO_ROOT/ci/scripts/runpod_lib.sh" "$Z_MUTANT_ROOT/runpod_lib.sh"
  Z_MUTANT_GPU_DEV="$Z_MUTANT_ROOT/gpu-dev.sh"
  cp "$Z_GPU_DEV" "$Z_MUTANT_GPU_DEV"
  python3 - "$Z_MUTANT_GPU_DEV" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
m = re.search(r'\ncase "\$CMD" in\n  push\|run\|target\)\n.*?\n    ;;\nesac\n', t, re.S)
assert m, "revert fixture: could not locate the push|run|target cwd-mismatch guard"
t2 = t[:m.start()] + "\n" + t[m.end():]
assert t2 != t
open(p, "w").write(t2)
PY
  bash -n "$Z_MUTANT_GPU_DEV" || bad "(z/cwd-guard revert-RED) the neutered gpu-dev.sh copy has a syntax error"
  z_mut_out="$(cd "$Z_PLAIN_CWD" && RUNPOD_API_KEY=test-dummy-key bash "$Z_MUTANT_GPU_DEV" push somesession --tree x 2>&1)"; z_mut_rc=$?
  if [ "$z_mut_rc" -ne 2 ] && ! grep -q "this script's own location resolves to REPO_ROOT" <<<"$z_mut_out"; then
    ok "(z/cwd-guard revert-RED) neutering the cwd-mismatch guard on a scratch copy silently proceeds past a mismatched cwd (rc=$z_mut_rc) — the guard is genuinely load-bearing"
  else
    bad "(z/cwd-guard revert-RED) expected the neutered copy to NOT refuse (rc=$z_mut_rc): $z_mut_out"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (aa/write-ahead) `gpu-dev.sh up` records its session the MOMENT the pod
# id comes back from the deploy mutation: runpod_lib.sh's rp_deploy_live
# writes the session — pod id, arch, a host-unknown placeholder — before
# the SSH-reachability wait (and the driver-floor check), and the
# post-wait call UPDATES that same record with the real host/port rather
# than creating it. A session written only after the wait leaves a
# window, minutes long, in which a failure (an external kill that
# bypasses the EXIT trap, or a trap-time `rp_terminate` that itself
# silently fails) leaves a RUNNING, BILLING pod that `ls`/`down` cannot
# see.
# ═════════════════════════════════════════════════════════════════════════
{
  AA_RUNPOD_LIB="$REPO_ROOT/ci/scripts/runpod_lib.sh"

  # ---- structural: the write-ahead save precedes the reachability wait, ---
  # which precedes the post-wait UPDATE save (see the leg's own doc above).
  # `^    rp_session_save$` matches only the bare write-ahead call (4-space
  # indent, no trailing `; return 0`) so it cannot accidentally match the
  # post-wait update call (10-space indent) or rp_session_save's own
  # definition/other callers.
  aa_write_ahead_line="$(grep -n '^    rp_session_save$' "$AA_RUNPOD_LIB" | head -1 | cut -d: -f1)"
  aa_wait_line="$(grep -n 'while \[ "\$SECONDS" -lt "\$_rp_deploy_deadline" \]; do' "$AA_RUNPOD_LIB" | head -1 | cut -d: -f1)"
  aa_update_line="$(grep -n 'rp_session_save; return 0' "$AA_RUNPOD_LIB" | head -1 | cut -d: -f1)"
  if [ -n "$aa_write_ahead_line" ] && [ -n "$aa_wait_line" ] && [ -n "$aa_update_line" ] \
     && [ "$aa_write_ahead_line" -lt "$aa_wait_line" ] && [ "$aa_wait_line" -lt "$aa_update_line" ]; then
    ok "(aa/write-ahead) rp_deploy_live's write-ahead rp_session_save (line ${aa_write_ahead_line}) precedes the reachability wait (line ${aa_wait_line}), which precedes the post-wait update save (line ${aa_update_line})"
  else
    bad "(aa/write-ahead) expected write-ahead < wait < update ordering (write_ahead=${aa_write_ahead_line:-<none>} wait=${aa_wait_line:-<none>} update=${aa_update_line:-<none>})"
  fi

  # ---- behavioural: a candidate that never becomes reachable still leaves --
  # a durable, on-disk session record naming the pod that was created (an
  # unreachable/killed post-create window must not leave the pod recorded
  # NOWHERE). Function-level intercept on rp_gql (this
  # suite's own established technique — see leg (a)'s module doc): no
  # network, no real RunPod account. The port-lookup stub returns an empty
  # `ports` list, so `[ -n "${RP_HOST:-}" ]` short-circuits false and `ssh`
  # itself is never invoked — the same "never mock ssh" discipline this
  # suite applies throughout; the driver forces a fast, deterministic
  # timeout via RP_SSH_WAIT_SECS=1 rather than actually waiting.
  AA_DRIVER="$SANDBOX/aa_driver.sh"
  cat > "$AA_DRIVER" <<'DRV'
#!/usr/bin/env bash
set -uo pipefail
export RUNPOD_API_KEY=test-dummy-key
export RP_SESSION_ROOT="${AA_DRV_SESSIONS}"
export RP_SSH_CONFIG="${AA_DRV_SSH_CONFIG}"
export RP_SESSION="${AA_DRV_SESSION}"
export RP_SSH_WAIT_SECS=1
mkdir -p "$RP_SESSION_ROOT"
FAKE_POD_ID="${AA_DRV_POD_ID}"
# shellcheck disable=SC1090
. "${AA_DRV_LIB}"
RP_ARCH=fakearch
rp_gql() {
  case "$1" in
    *podFindAndDeployOnDemand*) printf '{"data":{"podFindAndDeployOnDemand":{"id":"%s"}}}' "$FAKE_POD_ID" ;;
    *'pod(input:'*) printf '{"data":{"pod":{"runtime":{"ports":[]}}}}' ;;
    *podTerminate*) printf '{"data":{"podTerminate":true}}' ;;
    *) printf '{}' ;;
  esac
}
rp_deploy_live "FAKE|FAKE-GPU"
echo "RC=$?"
DRV
  chmod +x "$AA_DRIVER"

  AA_POD_ID="aa-writeahead-fake-pod-1234"
  AA_SESSIONS="$SANDBOX/aa_sessions"
  AA_SESSION_NAME="aa-writeahead"
  AA_DRV_LIB="$AA_RUNPOD_LIB" AA_DRV_POD_ID="$AA_POD_ID" AA_DRV_SESSION="$AA_SESSION_NAME" \
    AA_DRV_SESSIONS="$AA_SESSIONS" AA_DRV_SSH_CONFIG="$SANDBOX/aa_ssh_config" \
    bash "$AA_DRIVER" > "$SANDBOX/aa_deploy.out" 2>&1
  AA_META="$AA_SESSIONS/$AA_SESSION_NAME/meta"
  if grep -qF 'RC=75' "$SANDBOX/aa_deploy.out" ; then
    ok "(aa/write-ahead) the unreachable candidate correctly fails the OVERALL deploy (rc=75, no capacity/reachability) after being terminated in-loop"
  else
    bad "(aa/write-ahead) expected rp_deploy_live to return 75 on an unreachable candidate: $(cat "$SANDBOX/aa_deploy.out")"
  fi
  if [ -f "$AA_META" ] && grep -qF "RP_POD_ID=$AA_POD_ID" "$AA_META"; then
    ok "(aa/write-ahead fix) the session file was written with the pod id BEFORE the (ultimately failed) reachability wait completed — a killed/unreachable post-create window is still visible to ls/down, not orphaned"
  else
    bad "(aa/write-ahead fix) expected the session file to record RP_POD_ID=${AA_POD_ID} despite the deploy ultimately failing (meta: $(cat "$AA_META" 2>/dev/null || echo '<missing>'))"
  fi

  # ---- revert-RED: remove the write-ahead record (leaving only the ------
  # post-wait/on-success save) and this leg's behavioural assertion above
  # must go RED: an unreachable candidate leaves NO session file at all.
  AA_MUTANT_LIB="$SANDBOX/aa_runpod_lib_mutant.sh"
  cp "$AA_RUNPOD_LIB" "$AA_MUTANT_LIB"
  python3 - "$AA_MUTANT_LIB" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = "\n    rp_session_save\n    # A wall-clock deadline"
new = "\n    # A wall-clock deadline"
assert old in t, "revert fixture: could not locate rp_deploy_live's write-ahead rp_session_save call"
assert t.count(old) == 1, "revert fixture: expected exactly one write-ahead call site"
open(p, "w").write(t.replace(old, new, 1))
PY
  bash -n "$AA_MUTANT_LIB" || bad "(aa/write-ahead revert-RED) the neutered runpod_lib.sh copy has a syntax error"

  AA_MUTANT_SESSIONS="$SANDBOX/aa_mutant_sessions"
  AA_DRV_LIB="$AA_MUTANT_LIB" AA_DRV_POD_ID="$AA_POD_ID" AA_DRV_SESSION="$AA_SESSION_NAME" \
    AA_DRV_SESSIONS="$AA_MUTANT_SESSIONS" AA_DRV_SSH_CONFIG="$SANDBOX/aa_ssh_config_mutant" \
    bash "$AA_DRIVER" > "$SANDBOX/aa_mutant_deploy.out" 2>&1
  AA_MUTANT_META="$AA_MUTANT_SESSIONS/$AA_SESSION_NAME/meta"
  if grep -qF 'RC=75' "$SANDBOX/aa_mutant_deploy.out" && { [ ! -f "$AA_MUTANT_META" ] || ! grep -qF "RP_POD_ID=$AA_POD_ID" "$AA_MUTANT_META"; }; then
    ok "(aa/write-ahead revert-RED) neutering the write-ahead save on a scratch copy makes the same unreachable candidate leave NO recorded pod id — the write-ahead save is genuinely load-bearing"
  else
    bad "(aa/write-ahead revert-RED) expected the neutered copy to leave no RP_POD_ID recorded (meta: $(cat "$AA_MUTANT_META" 2>/dev/null || echo '<missing>')): $(cat "$SANDBOX/aa_mutant_deploy.out")"
  fi
}

# ═════════════════════════════════════════════════════════════════════════
# (ab/gpuCount) the shared deploy payload's `gpuCount` is a PARAMETER
# (RP_GPU_COUNT, default 1), and for a count > 1 the a100 candidate order
# puts SXM4 before PCIe (a `gpuCount: 2` SXM4 pod provisions where the
# PCIe pool returns zero 2-GPU capacity). Four behavioural
# determinants, each with its OWN revert-RED mutant on a scratch copy of
# runpod_lib.sh — the default value, the parameter being honoured, the
# count-1 candidate order (unchanged), the count>1 order (SXM4 first) —
# plus a fifth, SET-shaped leg: no lane other than the reviewed two below
# sets RP_GPU_COUNT anywhere, so every EXISTING lane (gpu-prove,
# gpu-dev, howwell) still deploys `gpuCount: 1`. That leg's
# set is DERIVED (`git ls-files` over ci/scripts + .github/workflows, then
# grep), never a hand list, so a future lane that starts setting the
# variable reds here instead of silently changing three other lanes'
# payloads.
#
# No network and no RunPod account: `_rp_deploy_payload` is a pure
# JSON-emitting function, and the candidate legs intercept `rp_deploy_live`
# itself (this suite's own established function-boundary technique — see
# leg (a)'s module doc) so nothing is ever deployed.
# ═════════════════════════════════════════════════════════════════════════
{
  AB_LIB="$REPO_ROOT/ci/scripts/runpod_lib.sh"
  AB_DRIVER="$SANDBOX/ab_payload_driver.sh"
  cat > "$AB_DRIVER" <<'DRV'
#!/usr/bin/env bash
set -uo pipefail
export RUNPOD_API_KEY=test-dummy-key
export RP_SESSION_ROOT="${AB_DRV_SESSIONS}"
export RP_SSH_CONFIG="${AB_DRV_SSH_CONFIG}"
mkdir -p "$RP_SESSION_ROOT"
# shellcheck disable=SC1090
. "${AB_DRV_LIB}"
# rp_init (which normally mints this) is never called here: it writes an ssh
# key and installs an EXIT trap, neither of which a payload/candidate-order
# reading needs.
RP_PUBKEY="ssh-ed25519 AAAAfixturekey test@fixture"
case "${AB_DRV_MODE}" in
  payload)
    _rp_deploy_payload SECURE "NVIDIA A100-SXM4-80GB"
    ;;
  candidates)
    # The candidate list is the ARGUMENT rp_deploy_arch hands to
    # rp_deploy_live; intercepting that one function boundary reads the
    # order without deploying anything.
    rp_deploy_live() { printf '%s\n' "$@"; return 0; }
    rp_deploy_arch a100
    ;;
  *)
    echo "unknown AB_DRV_MODE ${AB_DRV_MODE}" >&2; exit 2
    ;;
esac
DRV
  chmod +x "$AB_DRIVER"

  # $1=lib path  $2=mode  $3=RP_GPU_COUNT value ("" = leave it unset)
  ab_run() {
    if [ -n "$3" ]; then
      AB_DRV_LIB="$1" AB_DRV_MODE="$2" RP_GPU_COUNT="$3" \
        AB_DRV_SESSIONS="$SANDBOX/ab_sessions" AB_DRV_SSH_CONFIG="$SANDBOX/ab_ssh_config" \
        bash "$AB_DRIVER" 2>&1
    else
      AB_DRV_LIB="$1" AB_DRV_MODE="$2" \
        AB_DRV_SESSIONS="$SANDBOX/ab_sessions" AB_DRV_SSH_CONFIG="$SANDBOX/ab_ssh_config" \
        bash "$AB_DRIVER" 2>&1
    fi
  }

  # Reads the deploy mutation's own `gpuCount`, and REFUSES a JSON string
  # (the RunPod API takes an unquoted number; `"2"` would be a silent
  # payload defect this leg exists to catch).
  ab_gpu_count() {
    python3 -c '
import json, sys
d = json.load(sys.stdin)
v = d["variables"]["i"]["gpuCount"]
print(v if isinstance(v, int) and not isinstance(v, bool) else "NOT-AN-INT:%r" % (v,))
' 2>&1
  }

  AB_PCIE_FIRST="SECURE|NVIDIA A100 80GB PCIe
COMMUNITY|NVIDIA A100 80GB PCIe
SECURE|NVIDIA A100-SXM4-80GB
COMMUNITY|NVIDIA A100-SXM4-80GB"
  AB_SXM_FIRST="SECURE|NVIDIA A100-SXM4-80GB
COMMUNITY|NVIDIA A100-SXM4-80GB
SECURE|NVIDIA A100 80GB PCIe
COMMUNITY|NVIDIA A100 80GB PCIe"

  # ---- D1: default (no RP_GPU_COUNT in the environment) is gpuCount 1 -----
  ab_d1="$(ab_run "$AB_LIB" payload "" | ab_gpu_count)"
  if [ "$ab_d1" = "1" ]; then
    ok "(ab/gpuCount D1) the deploy payload's default gpuCount is the JSON number 1 — every lane that never sets RP_GPU_COUNT deploys a single-GPU pod"
  else
    bad "(ab/gpuCount D1) expected default gpuCount=1, got '${ab_d1}'"
  fi

  # ---- D2: RP_GPU_COUNT=2 reaches the payload as the number 2 ------------
  ab_d2="$(ab_run "$AB_LIB" payload 2 | ab_gpu_count)"
  if [ "$ab_d2" = "2" ]; then
    ok "(ab/gpuCount D2) RP_GPU_COUNT=2 emits gpuCount as the JSON number 2 — the gang leg's 1 pod x 2 GPU shape"
  else
    bad "(ab/gpuCount D2) expected gpuCount=2 under RP_GPU_COUNT=2, got '${ab_d2}'"
  fi

  # ---- D3: the count-1 a100 candidate ORDER is unchanged (PCIe first) ----
  ab_d3="$(ab_run "$AB_LIB" candidates "")"
  if [ "$ab_d3" = "$AB_PCIE_FIRST" ]; then
    ok "(ab/gpuCount D3) at the default count the a100 candidate order is unchanged (PCIe SECURE/COMMUNITY, then SXM4) — no existing lane's provisioning order moved"
  else
    bad "(ab/gpuCount D3) count-1 candidate order changed; got: ${ab_d3}"
  fi

  # ---- D4: count > 1 puts SXM4 first, dropping no candidate --------------
  ab_d4="$(ab_run "$AB_LIB" candidates 2)"
  if [ "$ab_d4" = "$AB_SXM_FIRST" ]; then
    ok "(ab/gpuCount D4) at RP_GPU_COUNT=2 the a100 candidates are SXM4-first with the PCIe pair kept as a fallback and each pair's SECURE-before-COMMUNITY order preserved (the PCIe pool can return zero 2-GPU capacity)"
  else
    bad "(ab/gpuCount D4) expected the SXM4-first order at RP_GPU_COUNT=2; got: ${ab_d4}"
  fi

  # ---- D5 (set-shaped): which TRACKED files mention RP_GPU_COUNT at all --
  # The reviewed set is exactly runpod_lib.sh (the parameter's own home),
  # runpod_gpu_gang.sh (the ONE lane that asks for more than one GPU), this
  # suite, and test_gpu_gang_lane.sh — which READS the gang driver's count
  # back and asserts 2, because the gang lane's dollar bound is priced at a
  # rate measured for a 2-GPU pod. An ASSERTER of the value is not a lane
  # setting one; anything else appearing here means some OTHER lane started
  # moving its own deploy payload off `gpuCount: 1`.
  ab_mentions="$(cd "$REPO_ROOT" && git ls-files ci/scripts .github/workflows | while IFS= read -r f; do
    if grep -q 'RP_GPU_COUNT' "$f"; then printf '%s\n' "$f"; fi
  done | sort)"
  ab_unexpected="$(printf '%s\n' "$ab_mentions" | grep -v '^$' \
    | grep -vx 'ci/scripts/runpod_lib.sh' \
    | grep -vx 'ci/scripts/runpod_gpu_gang.sh' \
    | grep -vx 'ci/scripts/test_gpu_gang_lane.sh' \
    | grep -vx 'ci/scripts/test_pod_substrate.sh' || true)" # tripwire-ok: grep -v with no surviving line legitimately exits 1; the emptiness IS the pass condition, asserted on the next line.
  if grep -qx 'ci/scripts/runpod_lib.sh' <<<"$ab_mentions" && [ -z "$ab_unexpected" ]; then
    ok "(ab/gpuCount D5) RP_GPU_COUNT is mentioned only by runpod_lib.sh, the gang driver, the gang-lane suite that asserts its value and this suite across every tracked ci/scripts + .github/workflows file — gpu-prove, gpu-perf-ab, gpu-dev and howwell all inherit the default"
  else
    bad "(ab/gpuCount D5) unexpected RP_GPU_COUNT site(s) — a lane other than the gang driver is changing its own gpuCount: ${ab_unexpected:-<runpod_lib.sh itself never mentions it>}"
  fi

  # ---- revert-RED mutants: one per behavioural determinant above ---------
  AB_MUTANT_DIR="$SANDBOX/ab_mutants"
  rm -rf "$AB_MUTANT_DIR"; mkdir -p "$AB_MUTANT_DIR"

  # M1 (kills D2): the payload hard-codes the count again.
  AB_M1="$AB_MUTANT_DIR/m1_literal_count.sh"
  cp "$AB_LIB" "$AB_M1"
  python3 - "$AB_M1" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '"gpuCount": int(gpu_count),'
new = '"gpuCount": 1,'
assert t.count(old) == 1, "M1 fixture: expected exactly one parameterised gpuCount site"
open(p, "w").write(t.replace(old, new, 1))
PY
  bash -n "$AB_M1" || bad "(ab/gpuCount M1) the mutated runpod_lib.sh copy has a syntax error"
  ab_m1="$(ab_run "$AB_M1" payload 2 | ab_gpu_count)"
  if [ "$ab_m1" != "2" ]; then
    ok "(ab/gpuCount M1 revert-RED) hard-coding the payload's gpuCount back to a literal makes D2 fail (got '${ab_m1}') — D2 genuinely binds the parameter, not the literal"
  else
    bad "(ab/gpuCount M1 revert-RED) the literal-count mutant still emitted gpuCount=2 — D2 does not bite"
  fi

  # M2 (kills D1): the default becomes 2.
  AB_M2="$AB_MUTANT_DIR/m2_default_two.sh"
  cp "$AB_LIB" "$AB_M2"
  python3 - "$AB_M2" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = 'RP_GPU_COUNT="${RP_GPU_COUNT:-1}"'
new = 'RP_GPU_COUNT="${RP_GPU_COUNT:-2}"'
assert t.count(old) == 1, "M2 fixture: expected exactly one RP_GPU_COUNT default site"
open(p, "w").write(t.replace(old, new, 1))
PY
  bash -n "$AB_M2" || bad "(ab/gpuCount M2) the mutated runpod_lib.sh copy has a syntax error"
  ab_m2="$(ab_run "$AB_M2" payload "" | ab_gpu_count)"
  if [ "$ab_m2" != "1" ]; then
    ok "(ab/gpuCount M2 revert-RED) moving the default off 1 makes D1 fail (got '${ab_m2}') — D1 genuinely pins the default every existing lane inherits"
  else
    bad "(ab/gpuCount M2 revert-RED) the default-2 mutant still emitted gpuCount=1 — D1 does not bite"
  fi

  # M3 (kills D4): the multi-GPU reordering is removed.
  AB_M3="$AB_MUTANT_DIR/m3_no_reorder.sh"
  cp "$AB_LIB" "$AB_M3"
  python3 - "$AB_M3" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '  _rp_order_candidates_for_gpu_count\n'
assert t.count(old) == 1, "M3 fixture: expected exactly one candidate-ordering call site"
open(p, "w").write(t.replace(old, "", 1))
PY
  bash -n "$AB_M3" || bad "(ab/gpuCount M3) the mutated runpod_lib.sh copy has a syntax error"
  ab_m3="$(ab_run "$AB_M3" candidates 2)"
  if [ "$ab_m3" != "$AB_SXM_FIRST" ]; then
    ok "(ab/gpuCount M3 revert-RED) removing the multi-GPU reordering makes D4 fail — D4 genuinely binds the SXM4-first order, not the arch table's own literal order"
  else
    bad "(ab/gpuCount M3 revert-RED) the no-reorder mutant still produced the SXM4-first order — D4 does not bite"
  fi

  # M4 (kills D3): the reordering becomes unconditional (count-1 lanes move too).
  AB_M4="$AB_MUTANT_DIR/m4_unconditional_reorder.sh"
  cp "$AB_LIB" "$AB_M4"
  python3 - "$AB_M4" <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
old = '  [ "$RP_GPU_COUNT" -gt 1 ] || return 0\n'
assert t.count(old) == 1, "M4 fixture: expected exactly one count>1 guard"
open(p, "w").write(t.replace(old, "", 1))
PY
  bash -n "$AB_M4" || bad "(ab/gpuCount M4) the mutated runpod_lib.sh copy has a syntax error"
  ab_m4="$(ab_run "$AB_M4" candidates "")"
  if [ "$ab_m4" != "$AB_PCIE_FIRST" ]; then
    ok "(ab/gpuCount M4 revert-RED) dropping the count>1 guard reorders the DEFAULT lanes too and makes D3 fail — D3 genuinely pins the existing lanes' provisioning order"
  else
    bad "(ab/gpuCount M4 revert-RED) the unconditional-reorder mutant left the count-1 order intact — D3 does not bite"
  fi
}


echo
echo "test_pod_substrate: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ]
