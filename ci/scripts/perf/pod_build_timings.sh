#!/usr/bin/env bash
# A2 producer (pod-build-substrate acceptance, contract v6) — runs ON A LIVE
# POD, never in CI. Not run by this PR; the lead runs it on a100d after this
# commit lands and commits the JSON it prints under
# ci/artifacts/pod-build-timings/. Until that JSON exists, no doc in this
# repo may cite a number from it (see docs/maintainer/dev-gpu.md, which cites
# ledger rows 1/17 only and marks the RP_DISK_GB formula's S values
# "pending").
#
# Measures, in order (contract v6 acceptance A2):
#   (i)   seed + marker + manifest cross-check (iii) — builds the seed via
#         pod_seed_target.sh and asserts .jammi-seed-complete plus the
#         announced-env-surface subset check.
#   (ii)  a CLONE build (pod_target_clone.sh) of the FA2 PR tip under tuple
#         T1 (release -p jammi-bench --features cuda) recompiles ONLY member
#         units plus the transitive lock/feature diff between the seed's ref
#         and the clone's ref — enumerated from `cargo build -v` output, not
#         asserted as a bare pass/fail.
#   (iii) sccache request count, labelled "unchanged by construction": the
#         wrapper is off pod-wide (CARGO_BUILD_RUSTC_WRAPPER= in
#         /root/.jammi_env, M3), so this leg records `sccache --show-stats`
#         BEFORE and AFTER the clone build and asserts the delta is zero
#         requests — a live check that the wrapper really is off, not an
#         assumption.
#   (iv)  byte-equality of the jammi-bench binary and every emitted .ptx
#         between the clone build and a COLD build at the SAME
#         CARGO_TARGET_DIR path — the cold leg builds from a genuinely EMPTY
#         directory (`rm -rf && mkdir`), never a copy of the clone with only
#         `cargo clean --workspace` run over it (that still reuses every
#         third-party dependency rlib the clone already built, so a
#         poisoned SEED dependency artifact could never register — round-2
#         audit finding 8); both legs' walls are recorded. Deny-listing
#         files expected to differ run-to-run for reasons unrelated to code
#         (jammi_flash_build_times.txt — wall-clock timing text;
#         .rustc_info.json — a cache of the toolchain's OWN self-report, not
#         build output; CACHEDIR.TAG — a static marker; .cargo-*lock — cargo's
#         own transient lockfiles).
#   (v)   S_src (git source tree size), S_seed (seed CARGO_TARGET_DIR size),
#         S_clone (clone CARGO_TARGET_DIR size), copy wall clock, and whether
#         the filesystem actually reflinked (pod_target_clone.sh's own
#         du/du---apparent-size printout, parsed back out here).
#   (vi)  clone wall clock vs ledger row 1's cold baseline (284 s), with the
#         flash-attn term (T1b, main-only) reported SEPARATELY since row 1's
#         284 s baseline did not include it.
#
# Usage (on the pod, inside the checkout the seed was built from):
#   JAMMI_FA2_TIP_REF=<ref> ci/scripts/perf/pod_build_timings.sh
# Prints the result JSON to stdout; the caller redirects it into
# ci/artifacts/pod-build-timings/<ts>-<sha>.json and commits it.
#
# Runs under pod_timing_lock.sh itself (round-2 audit finding 7: every wall
# number this producer measures is meaningless if a concurrent `run
# --timing`/another producer is competing for the same nvcc/CPU/disk). This
# script self-wraps exactly like pod_seed_target.sh does (--no-lock re-exec
# through pod_timing_lock.sh acquire -w): the FIRST invocation (no
# --no-lock) re-execs itself under the lock; the re-exec'd invocation
# (--no-lock) does the real work AND passes --no-lock straight through to
# its OWN call into pod_seed_target.sh (a "pass-through", never a second
# acquire of the SAME flock from a child process — that would deadlock
# against the parent that already holds it). `lock_held: true` is recorded
# in the result JSON as a live witness, not merely asserted in a comment.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_SCRIPTS="$(cd "$DIR/.." && pwd)"

JAMMI_TREE_DIR="${JAMMI_TREE_DIR:-/root/jammi-ai}"
JAMMI_SEED_DIR="${JAMMI_SEED_DIR:-/root/.jammi-seed}"
JAMMI_FA2_TIP_REF="${JAMMI_FA2_TIP_REF:?set JAMMI_FA2_TIP_REF to the FA2 PR tip ref/sha to measure}"
BOX="${JAMMI_BOX_LABEL:?set JAMMI_BOX_LABEL, e.g. 'a100d (A100 PCIe, driver 570)'}"
JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS="${JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS:-3600}"

fail() { echo "::error::$*" >&2; exit 1; }

NO_LOCK=0
for _a in "$@"; do [ "$_a" = "--no-lock" ] && NO_LOCK=1; done
if [ "$NO_LOCK" != "1" ]; then
  JAMMI_TIMING_LABEL="pod_build_timings" JAMMI_TIMING_JOB="pod_build_timings" \
    exec "$CI_SCRIPTS/pod_timing_lock.sh" acquire -w "$JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS" -- \
      env JAMMI_TREE_DIR="$JAMMI_TREE_DIR" JAMMI_SEED_DIR="$JAMMI_SEED_DIR" \
          JAMMI_FA2_TIP_REF="$JAMMI_FA2_TIP_REF" JAMMI_BOX_LABEL="$BOX" \
      "$0" --no-lock
fi
LOCK_HELD=true

# shellcheck disable=SC1091
. "$CI_SCRIPTS/pod_seed_target.sh"
# shellcheck disable=SC1091
. "$CI_SCRIPTS/pod_push_stamp.sh" 2>/dev/null || true

cd "$JAMMI_TREE_DIR" || fail "no tree at $JAMMI_TREE_DIR"

# ---- (i) seed + marker + manifest cross-check --------------------------
echo "::group::(i) seed"
"$CI_SCRIPTS/pod_seed_target.sh" --no-lock
[ -f "${JAMMI_SEED_DIR}.jammi-seed-complete" ] || fail "seed marker missing after pod_seed_target.sh"
S_seed_bytes="$(du -sk "$JAMMI_SEED_DIR" 2>/dev/null | awk '{print $1*1024}')"
echo "::endgroup::"

# ---- source-tree size (for RP_DISK_GB's S_src) --------------------------
S_src_bytes="$(du -sk --exclude=.git "$JAMMI_TREE_DIR" 2>/dev/null | awk '{print $1*1024}')"

# ---- (ii) clone build at the FA2 tip -------------------------------------
echo "::group::(ii) clone build @ ${JAMMI_FA2_TIP_REF}"
git fetch --all --tags --prune --quiet || fail "git fetch failed"
git checkout --quiet "$JAMMI_FA2_TIP_REF" || fail "checkout of ${JAMMI_FA2_TIP_REF} failed"
CLONE_DIR="/root/.jammi-clone-a2"
rm -rf "$CLONE_DIR"
copy_t0=$(date +%s)
"$CI_SCRIPTS/pod_target_clone.sh" "$JAMMI_SEED_DIR" "$CLONE_DIR" | tee /tmp/pod_build_timings.clone.log
copy_t1=$(date +%s)
copy_wall=$((copy_t1 - copy_t0))
reflink_took="no"
grep -q 'reflink=auto' /tmp/pod_build_timings.clone.log && reflink_took="attempted (auto; may have fallen back — see du vs du --apparent-size above)"

sccache_before="$(sccache --show-stats 2>/dev/null || echo 'sccache not running')"
export CARGO_TARGET_DIR="$CLONE_DIR"
export CARGO_INCREMENTAL=0
export CARGO_BUILD_RUSTC_WRAPPER=
clone_t0=$(date +%s)
cargo build --release -p jammi-bench --features cuda -v > /tmp/pod_build_timings.clone_build.log 2>&1
clone_rc=$?
clone_t1=$(date +%s)
clone_wall=$((clone_t1 - clone_t0))
[ "$clone_rc" -eq 0 ] || fail "clone build failed (exit $clone_rc) — see /tmp/pod_build_timings.clone_build.log"
sccache_after="$(sccache --show-stats 2>/dev/null || echo 'sccache not running')"

fa2_wall=""
# Detected via pod_seed_pkg_has_feature (sourced from pod_seed_target.sh
# above), never hand-asserted: `jammi-encoders/flash-attn` does not exist
# (round-2 audit finding 3) — flash-attn lives on jammi-kernels, forwarded
# through jammi-bench's own direct dependency on it.
if [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] && pod_seed_pkg_has_feature jammi-kernels flash-attn; then
  fa2_t0=$(date +%s)
  cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn >> /tmp/pod_build_timings.clone_build.log 2>&1
  fa2_rc=$?
  fa2_t1=$(date +%s)
  [ "$fa2_rc" -eq 0 ] && fa2_wall=$((fa2_t1 - fa2_t0))
fi

# Recompiled-unit enumeration: every "Compiling <pkg>" line in the clone
# build's own `-v` output IS the recompile set (a member-free seed means
# every member is expected here; anything ELSE compiling is either a
# transitive dependency the FA2 tip's Cargo.lock/feature diff actually
# changed, or a real leak from the seed's own member-freedom).
recompiled="$(grep -oE '^ *Compiling [^ ]+' /tmp/pod_build_timings.clone_build.log | awk '{print $2}' | sort -u)"
echo "::endgroup::"

S_clone_bytes="$(du -sk "$CLONE_DIR" 2>/dev/null | awk '{print $1*1024}')"

# ---- (iii) sccache requests unchanged by construction --------------------
sccache_delta_note="wrapper is off (CARGO_BUILD_RUSTC_WRAPPER=); sccache --show-stats before/after recorded verbatim below — expect identical (0 additional requests) since rustc never invoked it"

# ---- (iv) byte-equality vs a cold build at the SAME target dir -----------
echo "::group::(iv) cold build @ same CARGO_TARGET_DIR path (clean first)"
DENYLIST_RE='(jammi_flash_build_times\.txt|\.rustc_info\.json|CACHEDIR\.TAG|\.cargo-.*lock)$'
snapshot_hashes() { # $1=dir -> "path<TAB>sha256" lines, denylist excluded, sorted
  find "$1" -type f \( -name 'jammi-bench' -o -name '*.ptx' \) \
    | grep -Ev "$DENYLIST_RE" \
    | while read -r f; do printf '%s\t%s\n' "${f#"$1"/}" "$(shasum -a 256 "$f" | awk '{print $1}')"; done | sort
}
clone_hashes="$(snapshot_hashes "$CLONE_DIR")"
COLD_DIR="/root/.jammi-cold-a2"
# A genuinely EMPTY directory — never `cp -a $CLONE_DIR $COLD_DIR` +
# `cargo clean --workspace` (round-2 audit finding 8): `cargo clean
# --workspace` only removes WORKSPACE-MEMBER (jammi-*) artifacts, so a
# copy-then-clean "cold" build still reuses every THIRD-PARTY dependency
# rlib the clone already built — a poisoned SEED dependency artifact would
# be present, byte-identical, in BOTH legs, and this comparison could never
# register it. Building from nothing is the only way this leg proves what
# it claims: that the seed/clone mechanism reproduces a truly cold build's
# bytes, not merely "whatever the clone already had, minus jammi's own
# code".
rm -rf "$COLD_DIR"
mkdir -p "$COLD_DIR"
cold_t0=$(date +%s)
CARGO_TARGET_DIR="$COLD_DIR" cargo build --release -p jammi-bench --features cuda \
  > /tmp/pod_build_timings.cold_build.log 2>&1 || fail "cold build (same target dir, from empty) failed"
cold_t1=$(date +%s)
cold_wall=$((cold_t1 - cold_t0))
cold_hashes="$(snapshot_hashes "$COLD_DIR")"
if [ "$clone_hashes" = "$cold_hashes" ]; then
  byte_equal="true"; byte_equal_diff=""
else
  byte_equal="false"
  byte_equal_diff="$(diff <(echo "$clone_hashes") <(echo "$cold_hashes") | head -50)"
fi
echo "::endgroup::"

# ---- assemble result JSON (single pass; every value passed explicitly,
# nothing read back from a scratch file) -----------------------------------
RECOMPILED="$recompiled" SCCACHE_DELTA_NOTE="$sccache_delta_note" \
  SCCACHE_BEFORE="$sccache_before" SCCACHE_AFTER="$sccache_after" \
  python3 - "$BOX" "$(git rev-parse HEAD)" "$JAMMI_FA2_TIP_REF" "$clone_wall" "${fa2_wall:-}" \
  "$copy_wall" "$reflink_took" "$S_src_bytes" "$S_seed_bytes" "$S_clone_bytes" \
  "$byte_equal" "$(date -u +%FT%TZ)" "$cold_wall" "$LOCK_HELD" <<'PY'
import json, sys, os
(box, sha, tip_ref, clone_wall, fa2_wall, copy_wall, reflink_took,
 s_src, s_seed, s_clone, byte_equal, ts, cold_wall, lock_held) = sys.argv[1:15]
result = {
  "schema_version": 1,
  "box": box,
  "git_sha": sha,
  "fa2_tip_ref": tip_ref,
  "ts": ts,
  "lock_held": lock_held == "true",
  "measurements": {
    "clone_build_wall_s": int(clone_wall),
    "cold_build_wall_s": int(cold_wall),
    "flash_attn_leg_wall_s": int(fa2_wall) if fa2_wall else None,
    "copy_wall_s": int(copy_wall),
    "reflink": reflink_took,
    "S_src_bytes": int(s_src) if s_src else None,
    "S_seed_bytes": int(s_seed) if s_seed else None,
    "S_clone_bytes": int(s_clone) if s_clone else None,
    "clone_vs_row1_284s_cold_baseline_delta_s": int(clone_wall) - 284,
  },
  "recompiled_units": os.environ.get("RECOMPILED", "").splitlines(),
  "byte_equal_clone_vs_cold_same_target_dir": byte_equal == "true",
  "sccache_note": os.environ.get("SCCACHE_DELTA_NOTE", ""),
  "sccache_before": os.environ.get("SCCACHE_BEFORE", ""),
  "sccache_after": os.environ.get("SCCACHE_AFTER", ""),
}
print(json.dumps(result, indent=2))
PY

if [ "$byte_equal" != "true" ]; then
  echo "::warning::byte-equality FAILED (iv) — see diff below" >&2
  echo "$byte_equal_diff" >&2
fi
echo "pipeline complete: pipe this script's stdout JSON into ci/artifacts/pod-build-timings/<ts>-<sha7>.json and commit it (see this file's header)." >&2
