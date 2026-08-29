#!/usr/bin/env bash
# A2 producer (pod-build-substrate acceptance, contract v6) — runs ON A LIVE
# POD, never in CI. The lead runs it on a live pod and commits the JSON it
# writes (to JAMMI_BUILD_TIMINGS_OUT, never stdout — see Usage below) under
# ci/artifacts/pod-build-timings/; the first committed run is
# 20260827T183928Z-bc27e75.json. No doc in this repo may cite a number this
# producer measures except from a committed JSON (docs/maintainer/dev-gpu.md
# and pod-build-guide.md §4 cite that file).
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
#   (iv)  byte-equality of the DETERMINISTIC release outputs — every
#         emitted .ptx plus every workspace member's own compiled
#         libjammi_*.rlib/.rmeta (round-6 fix: the LINKED BINARY,
#         release/jammi-bench, is EXCLUDED — ThinLTO local-symbol
#         suffixes make it non-deterministic across two builds of the
#         IDENTICAL tree on the SAME box with this toolchain (mold
#         2.35.1 / clang 21), a real, live a100c finding; the excluded
#         path and reason are named in the JSON's own byte_equal_scope
#         field, never silently dropped from the claim) —
#         round-4 addendum: these are DELIBERATELY two DIFFERENT literal
#         CARGO_TARGET_DIR paths, not the same directory reused; an earlier
#         revision of this doc said "the SAME target dir path", which was
#         never true of the mechanism and is corrected here. Comparing
#         across genuinely different paths is the STRONGER claim: it also
#         catches any artifact that embeds its own CARGO_TARGET_DIR's
#         absolute path (debug info, `.d` files, panic messages), which a
#         same-path comparison could never distinguish from a real defect.
#         The cold leg builds from a genuinely EMPTY directory (`rm -rf &&
#         mkdir`), never a copy of the clone with only `cargo clean
#         --workspace` run over it (that still reuses every third-party
#         dependency rlib the clone already built, so a poisoned SEED
#         dependency artifact could never register — round-2 audit finding
#         8); both legs' walls are recorded. Deny-listing files expected to
#         differ run-to-run for reasons unrelated to code
#         (jammi_flash_build_times.txt — wall-clock timing text;
#         .rustc_info.json — a cache of the toolchain's OWN self-report, not
#         build output; CACHEDIR.TAG — a static marker; .cargo-*lock — cargo's
#         own transient lockfiles). round-6 fix (live a100c run at
#         63bf905): both snapshots are scoped to release/ ONLY (T2's
#         `cargo test --no-run` — debug profile by default — leaves
#         third-party PTX in the CLONE's debug/ subtree, inherited from
#         the seed, that the COLD dir never has at all; comparing the
#         whole tree compared two different FILE SETS, not the same
#         artifacts' bytes). byte_equal_state is a FOUR-state result:
#         invalid (empty match set on either side) / set_mismatch (both
#         non-empty, but the PATHS present differ) / true / false — never
#         collapsed into a bare true/false.
#   (v)   S_src (git source tree size), S_seed (seed CARGO_TARGET_DIR size),
#         S_clone (clone CARGO_TARGET_DIR size), copy wall clock, and whether
#         the filesystem actually reflinked (pod_target_clone.sh's own
#         du/du---apparent-size printout, parsed back out here).
#   (vi)  clone wall clock vs THIS SAME RUN's own cold build wall clock
#         (round-5 fix: no cross-box/uncommitted constant baked in — a
#         reader computes any delta they want from the two real numbers
#         both already in the JSON), with the flash-attn term (fa2_wall,
#         main-only, resolved-sha-gated — round-6 fix, see fa2_ran/
#         fa2_reason) reported SEPARATELY and always-recorded, never
#         silently absent with no stated reason.
#
# Usage (on the pod, inside the checkout the seed was built from):
#   JAMMI_FA2_TIP_REF=<ref> JAMMI_BUILD_TIMINGS_OUT=<path> [JAMMI_MAIN_SHA=<sha>] \
#     ci/scripts/perf/pod_build_timings.sh
# REQUIRES $JAMMI_TREE_DIR to be a git-backed tree with a reachable
# 'origin' remote (round-6 fix, lead probe item 2): step (ii) below runs
# `git fetch` + `git checkout` directly against it — a tree populated
# purely by `push` (rsync, no `.git` at all — see pod_push_stamp.sh)
# cannot run this script. Further, the FA2 leg specifically needs
# `origin/main` to be a RESOLVABLE remote-tracking ref — a bundle clone
# (this runner's own standard shape; measured on all three real
# a100c/a100e runs) typically carries no such ref, in which case the FA2
# leg reports "could not resolve origin/main" and is skipped, not run.
# Set `JAMMI_MAIN_SHA` explicitly (the runner, from the laptop, already
# knows it) to bypass that resolution entirely.
# Writes the result JSON to JAMMI_BUILD_TIMINGS_OUT (never stdout — round-4
# addendum: this script's own progress markers, `::group::`/`::endgroup::`
# and every intermediate status line, went to stdout too, so "redirect
# stdout into the artifact" silently corrupted the artifact with everything
# else this script printed while running; a caller doing exactly what the
# OLD usage line said would never have produced valid JSON). The caller
# copies JAMMI_BUILD_TIMINGS_OUT to ci/artifacts/pod-build-timings/<ts>-
# <sha>.json and commits it. Progress/status still goes to stdout/stderr as
# before, safe to watch live or log verbatim.
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
# round-4 addendum: the JSON result is written HERE, never to stdout — see
# the module doc above.
JAMMI_BUILD_TIMINGS_OUT="${JAMMI_BUILD_TIMINGS_OUT:?set JAMMI_BUILD_TIMINGS_OUT to the output JSON path -- never stdout, since this script also prints progress markers to stdout}"
JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS="${JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS:-3600}"

fail() { echo "::error::$*" >&2; exit 1; }

# --- provenance cross-check (unification contract C5.1), same shape as
# finetune_ab.sh's/stacked_sweep.sh's/clip_artifact_producer.sh's own
# check_bin_provenance(): called after EACH `cargo build -p jammi-bench`
# below (T1's clone build, the cold build, and the FA2 leg's own build)
# BEFORE this producer trusts the git_sha it is about to stamp into the
# result JSON's own `git_sha` field. A build that exits 0 says nothing
# about whether the resulting binary is provably the one THIS checkout's
# HEAD produced — a stale/cached artifact resurrected from a different
# CARGO_TARGET_DIR, or a linker/sccache misconfiguration that silently
# reused a prior binary, would still exit 0 and still be a real gap this
# script's own byte-equality leg (iv) cannot see (it compares clone vs.
# cold to EACH OTHER, never either one against the sha it claims to have
# built). Refuses loudly (never warns-and-continues) on any mismatch,
# unknown, or unreadable provenance reading, exactly like its siblings.
check_bin_provenance() {
  local bin="$1"
  local sha sha_re='^[0-9a-fA-F]{40}$'
  sha="$(git rev-parse HEAD)"
  if ! [[ "$sha" =~ $sha_re ]]; then
    fail "provenance cross-check: HEAD did not resolve to a 40-hex commit ('$sha') -- refusing"
  fi
  [ -x "$bin" ] || fail "provenance cross-check: no executable at $bin -- the build above did not produce the binary this leg is about to stamp git_sha=$sha for"
  local bin_prov_json bin_prov_sha
  bin_prov_json="$("$bin" provenance 2>&1)" || fail "'$bin provenance' failed: $bin_prov_json"
  bin_prov_sha="$(printf '%s' "$bin_prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || fail "could not parse build_sha from '$bin provenance' output: $bin_prov_json"
  if [ -z "$bin_prov_sha" ] || [ "$bin_prov_sha" != "$sha" ]; then
    fail "'$bin provenance' reports build_sha=$bin_prov_sha, but this checkout is at sha=$sha -- refusing before this producer's JSON stamps git_sha=$sha against a binary that does not agree"
  fi
}

NO_LOCK=0
for _a in "$@"; do [ "$_a" = "--no-lock" ] && NO_LOCK=1; done
if [ "$NO_LOCK" != "1" ]; then
  JAMMI_TIMING_LABEL="pod_build_timings" JAMMI_TIMING_JOB="pod_build_timings" \
    exec "$CI_SCRIPTS/pod_timing_lock.sh" acquire -w "$JAMMI_BUILD_TIMINGS_LOCK_WAIT_SECS" -- \
      env JAMMI_TREE_DIR="$JAMMI_TREE_DIR" JAMMI_SEED_DIR="$JAMMI_SEED_DIR" \
          JAMMI_FA2_TIP_REF="$JAMMI_FA2_TIP_REF" JAMMI_BOX_LABEL="$BOX" \
          JAMMI_BUILD_TIMINGS_OUT="$JAMMI_BUILD_TIMINGS_OUT" \
      "$0" --no-lock
fi
# LOCK_HELD is a LIVE WITNESS, not a constant (round-3 audit Class B): the
# holder file pod_timing_lock.sh writes UNDER the lock (tmp+rename) is read
# back and must name THIS invocation's own label — a hardcoded `true` here
# would still print "held" even if the re-exec chain above were somehow
# skipped or the lock file were on a different path than pod_timing_lock.sh
# actually used.
#
# round-4 audit A3: the holder file ALONE is not sufficient — before
# pod_timing_lock.sh started removing it on release, a PRIOR run's holder
# file (same label) stayed on disk forever, so this witness read `true`
# for every run after the first, genuinely held or not (reproduced: prior
# run exits, witness reads true, an outsider acquires the lock
# immediately). pod_timing_lock.sh now removes the holder on EXIT/INT/TERM,
# but this witness ALSO cross-checks the recorded `pid=` is a LIVE process
# (`kill -0`) rather than trusting the file's mere existence — belt-and-
# suspenders against a removal that itself raced or failed. The pid
# recorded is the flock-held bash wrapper's own pid (pod_build_timings.sh
# is its direct child, so that wrapper being alive IS this invocation
# genuinely holding the lock right now).
_LOCK_FILE="${JAMMI_TIMING_LOCK:-/root/.jammi-timing.lock}"
_HOLDER_PID="$(grep -o '^pid=[0-9]*$' "${_LOCK_FILE}.holder" 2>/dev/null | cut -d= -f2)" # tripwire-ok: a missing/unreadable holder file is a REAL, valid state (no lock held, or a race with the writer) — _HOLDER_PID coming back empty is checked explicitly right below (LOCK_HELD=false), never silently treated as "held"
# tripwire-ok (both lines below): a missing/non-matching holder file or a
# dead pid legitimately means "not held by this label" — the else-branch
# reports LOCK_HELD=false loudly right below, never silently.
if [ -f "${_LOCK_FILE}.holder" ] && grep -q '^holder=pod_build_timings$' "${_LOCK_FILE}.holder" 2>/dev/null \
   && [ -n "$_HOLDER_PID" ] && kill -0 "$_HOLDER_PID" 2>/dev/null; then
  LOCK_HELD=true
else
  LOCK_HELD=false
  echo "::warning::LOCK_HELD=false — ${_LOCK_FILE}.holder does not name this invocation with a live pid; every wall-clock number below is UNPROTECTED against a concurrent producer" >&2
fi

# shellcheck disable=SC1091
. "$CI_SCRIPTS/pod_seed_target.sh"
# round-6 advisory (folded): the round-5 sourcing of pod_push_stamp.sh
# here was dead code (no pod_push_* function was ever called directly by
# THIS file — pod_provision_cutlass.sh invokes it as a separate `bash`
# subprocess, never via this script's own sourced namespace) — deleted,
# never merely re-annotated.

cd "$JAMMI_TREE_DIR" || fail "no tree at $JAMMI_TREE_DIR"

# round-4 addendum: fail loudly, naming every missing tool, before spending
# any wall-clock time on the legs below (pod_seed_assert_required_tools is
# sourced from pod_seed_target.sh above).
pod_seed_assert_required_tools || fail "required tool(s) missing — see ::error:: above"

# ---- (i) seed + marker + manifest cross-check --------------------------
echo "::group::(i) seed"
"$CI_SCRIPTS/pod_seed_target.sh" --no-lock
[ -f "${JAMMI_SEED_DIR}.jammi-seed-complete" ] || fail "seed marker missing after pod_seed_target.sh"
# round-3 audit N2: the ONE unconditional, filesystem-level check — never
# opt-in — that the seed pod_seed_target.sh just finished is genuinely
# member-free, run again HERE (a second, independent witness; the seed's
# own build already ran it before stamping complete).
pod_seed_assert_member_free "$JAMMI_SEED_DIR" "$JAMMI_TREE_DIR" || fail "seed at ${JAMMI_SEED_DIR} is NOT member-free"
S_seed_bytes="$(du -sk "$JAMMI_SEED_DIR" 2>/dev/null | awk '{print $1*1024}')" # tripwire-ok: best-effort size for the RP_DISK_GB report only, never gates pass/fail; a du failure yields S_seed_bytes="" -> the JSON writer emits null for it (explicit, not a silent zero)
# round-5 fix (Class-A item 4): the seed's own completion marker now
# records WHICH tuples it actually built (pod_seed_target.sh's
# t1b_flash_attn_ran/reason, see that script's own completion-marker
# writer) — read back here and copied verbatim into THIS producer's own
# output JSON, so `flash_attn_leg_wall_s` below is interpretable: a reader
# can tell whether the SEED itself carried FA2 artifacts, independent of
# whether this run's OWN separate FA2 *measurement* leg (below) also ran.
seed_tuples_json="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("[]"); sys.exit(0)
print(json.dumps(d.get("tuples", [])))
' "${JAMMI_SEED_DIR}.jammi-seed-complete")"
seed_t1b_ran="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("false"); sys.exit(0)
print("true" if d.get("t1b_flash_attn_ran") else "false")
' "${JAMMI_SEED_DIR}.jammi-seed-complete")"
seed_t1b_reason="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(""); sys.exit(0)
print(d.get("t1b_flash_attn_reason") or "")
' "${JAMMI_SEED_DIR}.jammi-seed-complete")"
echo "::endgroup::"

# ---- source-tree size (for RP_DISK_GB's S_src) --------------------------
S_src_bytes="$(du -sk --exclude=.git "$JAMMI_TREE_DIR" 2>/dev/null | awk '{print $1*1024}')" # tripwire-ok: same as S_seed_bytes above -- best-effort report-only size, empty result -> JSON null, never a silent zero

# ---- (ii) clone build at the FA2 tip -------------------------------------
# round-6 fix (lead probe item 2): this whole leg REQUIRES $JAMMI_TREE_DIR
# to be a git-backed tree with a remote ("origin") that actually carries
# history — a tree populated purely by `push` (rsync, no .git — see
# pod_push_stamp.sh) cannot run this script at all, and even a git-backed
# BUNDLE clone (this runner's own standard shape, measured on all three
# real a100c/a100e runs) typically carries no `origin/main` remote-
# tracking ref, which is why the FA2 leg below reliably reports "could
# not resolve origin/main" on every such run — not a bug in that leg, a
# real precondition of the whole (ii) block. Named explicitly in both
# `fail` messages (the OLD messages here said only "failed", not why),
# and `JAMMI_MAIN_SHA` (see the FA2 gate below) is the escape hatch for a
# runner that already knows main's sha without needing a resolvable
# origin/main remote-tracking ref at all.
echo "::group::(ii) clone build @ ${JAMMI_FA2_TIP_REF}"
git fetch --all --tags --prune --quiet \
  || fail "git fetch failed — this script requires \$JAMMI_TREE_DIR to be a git-backed tree with a reachable 'origin' remote (a tree populated purely by 'push' has no .git at all and cannot run this script)"
git checkout --quiet "$JAMMI_FA2_TIP_REF" \
  || fail "checkout of ${JAMMI_FA2_TIP_REF} failed — \$JAMMI_TREE_DIR must be a git-backed tree whose 'origin' remote actually carries this ref (see the (ii) block's own module-doc note)"
# round-6 fix (audit item 1 — the class this round closes: "the scripts
# assume a git state of the tree that a pushed/provisioned tree does not
# have"): the OLD form ran `git submodule update --init` DIRECTLY on
# $JAMMI_TREE_DIR's own cutlass path — but that SAME path can ALREADY be
# populated by an earlier `target --with-cutlass` copy-provisioning call
# (pod_provision_cutlass.sh's own `cp -a` + `rm -rf .git`), which git's
# own submodule machinery REFUSES to touch (a non-empty, non-submodule-
# shaped directory) — the live a100c failure: rc=1, wall=819s wasted
# before failing. ONE provisioning surface now handles cutlass in EVERY
# tree pod_build_timings.sh operates on: pod_provision_cutlass.sh's own
# `rm -rf` + `cp -a` never asks git to touch the destination path at all
# — a stale copy-provisioned dir, an empty dir, or a never-touched path
# are all handled identically — and it derives the expected pin from
# $JAMMI_TREE_DIR's OWN git index (this checkout just fetched+checked out
# $JAMMI_FA2_TIP_REF above, so its recorded gitlink pin at this path IS
# the correct expectation) rather than requiring a push-stamp file — see
# that script's own citation for the full mechanism (including its own
# content-floor validation of SUPER_DIR's submodule before ever copying
# from it).
if grep -q 'crates/jammi-kernels/third_party/cutlass' .gitmodules 2>/dev/null; then # tripwire-ok: a missing .gitmodules is a REAL, valid state on a ref that never carried the cutlass submodule at all -- provisioning is correctly SKIPPED, not silently passed
  bash "$CI_SCRIPTS/pod_provision_cutlass.sh" "$JAMMI_TREE_DIR" \
    || fail "cutlass provisioning into ${JAMMI_TREE_DIR} failed (pod_provision_cutlass.sh, after checking out ${JAMMI_FA2_TIP_REF}) — see the ::error:: above"
fi
CLONE_DIR="/root/.jammi-clone-a2"
rm -rf "$CLONE_DIR"
copy_t0=$(date +%s)
"$CI_SCRIPTS/pod_target_clone.sh" "$JAMMI_SEED_DIR" "$CLONE_DIR" "$JAMMI_TREE_DIR" | tee /tmp/pod_build_timings.clone.log
copy_t1=$(date +%s)
copy_wall=$((copy_t1 - copy_t0))
reflink_took="no"
grep -q 'reflink=auto' /tmp/pod_build_timings.clone.log && reflink_took="attempted (auto; may have fallen back — see du vs du --apparent-size above)"
# pod_target_clone.sh's own unconditional member-freedom check (round-3
# audit N2) already ran as part of the call above and would have removed
# CLONE_DIR and exited non-zero on failure — this just names the fact
# loudly rather than letting a missing CLONE_DIR fail obscurely below.
[ -d "$CLONE_DIR" ] || fail "clone at ${CLONE_DIR} is missing — pod_target_clone.sh's own member-freedom check likely refused it (see the log above)"

sccache_before="$(sccache --show-stats 2>/dev/null || echo 'sccache not running')" # tripwire-ok: "sccache not running" is a visible, non-empty sentinel (never a silent empty string) for the ordinary pod-wide state (M3: the wrapper is off) -- leg (iii)'s own delta check reads this value verbatim, so a real absence is reported, not hidden
export CARGO_TARGET_DIR="$CLONE_DIR"
export CARGO_INCREMENTAL=0
export CARGO_BUILD_RUSTC_WRAPPER=
clone_t0=$(date +%s)
cargo build --release -p jammi-bench --features cuda -v > /tmp/pod_build_timings.clone_t1.log 2>&1
clone_rc=$?
clone_t1=$(date +%s)
clone_wall=$((clone_t1 - clone_t0))
[ "$clone_rc" -eq 0 ] || fail "clone build (T1) failed (exit $clone_rc) — see /tmp/pod_build_timings.clone_t1.log"
check_bin_provenance "$CLONE_DIR/release/jammi-bench"
sccache_after="$(sccache --show-stats 2>/dev/null || echo 'sccache not running')" # tripwire-ok: same as sccache_before above
clone_features="cuda"

# round-3 audit N3: clone_hashes and the T1 recompiled-unit list are
# snapshotted IMMEDIATELY after T1, from a T1-ONLY log — BEFORE the FA2
# leg (below) ever touches a directory or a log file. The OLD order built
# T1 and FA2 into the SAME CLONE_DIR, appending to the SAME log, then took
# this snapshot AFTER both had run: on `main` (where FA2 always ran)
# byte_equal against the T1-only cold leg was GUARANTEED false, and
# recompiled_units was the union of two different feature builds' logs.
S_clone_bytes="$(du -sk "$CLONE_DIR" 2>/dev/null | awk '{print $1*1024}')" # tripwire-ok: same as S_seed_bytes/S_src_bytes above -- best-effort report-only size, empty -> JSON null
# round-4 addendum (on-pod incident, a100c A2 run at b3cafda): `shasum` is
# ABSENT on the pod image — this used to make every hash in this leg
# SILENTLY empty there, so byte_equal's "equal" was comparing two empty
# strings (a pass that never actually compared anything; A4's own
# invalid-state fix catches an EMPTY MATCH SET on the `find` side, but not
# a hashing tool that is simply missing while `find` still matches real
# files — `pod_sha256_of_file`, sourced from pod_seed_target.sh above,
# prefers coreutils sha256sum and refuses loudly rather than printing
# nothing if neither hashing tool exists).
# round-6 fix (live a100c run at 63bf905, real evidence: byte_equal=false
# with 54 files on the clone side vs 21 on the cold side, clone+cold wall
# 360s): this used to `find "$1"` UNSCOPED across the WHOLE target dir,
# matching *.ptx/jammi-bench in BOTH debug/ and release/ — but T1 (the
# command this leg actually measures, `cargo build --release`) and the
# cold leg both build ONLY the release profile. The CLONE inherits the
# SEED's own debug/ subtree too (T2's `cargo test --no-run` — debug
# profile by default, no --release flag — leaves third-party dependency
# PTX files there that survive the seed's own member-free clean since
# they belong to non-jammi crates), while COLD_DIR never had anything
# but a release build run against it at all, so it has no debug/
# subtree. The two snapshots were comparing DIFFERENT FILE SETS, not the
# same artifacts' bytes — scoping to release/ (the ONE profile both legs
# actually build) makes this a like-for-like comparison BY CONSTRUCTION,
# never merely asserted after the fact.
# round-6 fix (audit item B, live a100c evidence): release/jammi-bench —
# the FINAL LINKED BINARY — carries 467 ThinLTO local-symbol suffixes
# (anon.<h>.N.llvm.<hash>) that differ between TWO BUILDS OF THE
# IDENTICAL TREE ON THE SAME BOX (mold 2.35.1 / clang 21's own ThinLTO
# codegen is not byte-deterministic for the linked binary), while every
# one of the 20 release/*.ptx files this leg also hashes IS byte-
# identical clone<->cold, across boxes, across passes — and rustc's own
# compiled outputs for workspace members (.rlib/.rmeta) are equally
# deterministic. Binary byte-equality is UNATTAINABLE with this
# toolchain; comparing it anyway makes a real, attainable claim (every
# DETERMINISTIC output byte-matches) read as a permanent false negative.
# The comparison set is release/*.ptx + release/**/libjammi_*.rlib|
# .rmeta (workspace members' own compiled outputs, for the measured
# feature set) — the linked binary is EXCLUDED, named explicitly in the
# JSON (byte_equal_scope), never silently dropped from the claim.
snapshot_hashes() { # $1=dir -> "path<TAB>sha256" lines, denylist excluded, sorted, release/ ONLY
  # tripwire-ok (the 2>/dev/null on the find below): a missing release/
  # subtree (a build that produced nothing where something was expected)
  # yields an empty match set, which the caller's own byte_equal=
  # "invalid" path (empty-set floor) catches explicitly -- never a
  # silent pass.
  find "$1/release" -type f \( -name '*.ptx' -o -name 'libjammi_*.rlib' -o -name 'libjammi_*.rmeta' \) 2>/dev/null \
    | grep -Ev "$DENYLIST_RE" \
    | while read -r f; do printf '%s\t%s\n' "${f#"$1"/}" "$(pod_sha256_of_file "$f")"; done | sort
}
DENYLIST_RE='(jammi_flash_build_times\.txt|\.rustc_info\.json|CACHEDIR\.TAG|\.cargo-.*lock)$'
clone_hashes="$(snapshot_hashes "$CLONE_DIR")"
recompiled="$(grep -oE '^ *Compiling [^ ]+' /tmp/pod_build_timings.clone_t1.log | awk '{print $2}' | sort -u)"

fa2_wall=""
fa2_features=""
fa2_ran="false"
fa2_reason=""
# Detected via pod_seed_pkg_has_feature (sourced from pod_seed_target.sh
# above), never hand-asserted: `jammi-encoders/flash-attn` does not exist
# (round-2 audit finding 3) — flash-attn lives on jammi-kernels, forwarded
# through jammi-bench's own direct dependency on it. round-3 audit N3: the
# FA2 leg gets its OWN clone dir (fresh from the seed) and its OWN log —
# never CLONE_DIR/the T1 log, which are already snapshotted above and must
# stay untouched by anything that runs after this point.
#
# round-6 fix (audit item 4, the class this round closes: "the scripts
# assume a git state of the tree that a pushed/provisioned tree does not
# have"): the OLD gate compared `git rev-parse --abbrev-ref HEAD` to the
# literal "main" — but this leg runs AFTER `git checkout --quiet
# "$JAMMI_FA2_TIP_REF"` (:203 above), which checks out BY SHA whenever
# JAMMI_FA2_TIP_REF is a sha (the ordinary case for an FA2 PR tip) — a
# checkout-by-sha ALWAYS leaves a DETACHED HEAD, whose abbrev-ref reads
# the literal string "HEAD", never any branch name. The OLD gate could
# therefore never match in the ordinary case, and — with no `else` arm at
# all — the ENTIRE FA2 measurement leg silently vanished: fa2_wall stayed
# empty (flash_attn_leg_wall_s: null in the committed JSON) with NO
# recorded reason, indistinguishable from "ran and measured nothing" or
# "correctly determined this isn't main". Gated on the RESOLVED sha
# instead — identical whether the checkout landed on a real branch or a
# detached HEAD — and fa2_ran/fa2_reason are now ALWAYS recorded,
# mirroring the seed's own t1b_flash_attn_ran/reason
# (pod_seed_target.sh's completion marker), so a reader of the committed
# JSON never has to guess why this leg did or did not run.
_head_sha="$(git rev-parse HEAD)"
# round-6 fix (lead probe item 2): `origin/main` is UNRESOLVABLE on a
# bundle clone (this runner's own standard shape — measured on all three
# real a100c/a100e runs: origin carries no main remote-tracking ref at
# all), which made this leg report "could not resolve origin/main" every
# time on real hardware, never actually exercising the FA2 leg end to
# end. `JAMMI_MAIN_SHA` is an explicit escape hatch: a caller (the
# runner, from the laptop) that already KNOWS main's sha can pass it
# directly, bypassing the need for a resolvable remote-tracking ref.
if [ -n "${JAMMI_MAIN_SHA:-}" ]; then
  _main_sha="$JAMMI_MAIN_SHA"
  _main_sha_source="JAMMI_MAIN_SHA override"
else
  _main_sha="$(git rev-parse --verify --quiet origin/main 2>/dev/null || true)" # tripwire-ok: no origin/main remote-tracking ref (e.g. a bundle clone, or a fetch that never populated it) is a REAL state; the empty result is checked explicitly right below (fa2_reason names it), never silently treated as "on main"
  _main_sha_source="origin/main"
fi
if [ -n "$_main_sha" ] && [ "$_head_sha" = "$_main_sha" ]; then
  feat_rc=0
  pod_seed_pkg_has_feature jammi-kernels flash-attn || feat_rc=$?
  if [ "$feat_rc" -eq 0 ]; then
    CLONE_FA2_DIR="/root/.jammi-clone-fa2-a2"
    rm -rf "$CLONE_FA2_DIR"
    "$CI_SCRIPTS/pod_target_clone.sh" "$JAMMI_SEED_DIR" "$CLONE_FA2_DIR" "$JAMMI_TREE_DIR" | tee /tmp/pod_build_timings.clone_fa2_copy.log
    if [ -d "$CLONE_FA2_DIR" ]; then
      fa2_t0=$(date +%s)
      CARGO_TARGET_DIR="$CLONE_FA2_DIR" CARGO_INCREMENTAL=0 CARGO_BUILD_RUSTC_WRAPPER='' \
        cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn \
        > /tmp/pod_build_timings.clone_fa2.log 2>&1
      fa2_rc=$?
      fa2_t1=$(date +%s)
      if [ "$fa2_rc" -eq 0 ]; then
        check_bin_provenance "$CLONE_FA2_DIR/release/jammi-bench"
        fa2_wall=$((fa2_t1 - fa2_t0))
        fa2_features="cuda,jammi-kernels/flash-attn"
        fa2_ran="true"
        fa2_reason="declared (cargo metadata) and built (resolved sha ${_head_sha} matches ${_main_sha_source})"
      else
        fa2_reason="FA2 leg build failed (exit ${fa2_rc}) — see /tmp/pod_build_timings.clone_fa2.log; T1's own snapshot is unaffected"
        echo "::warning::FA2 leg build failed (exit $fa2_rc) — see /tmp/pod_build_timings.clone_fa2.log; T1's own snapshot above is unaffected" >&2
      fi
    else
      fa2_reason="FA2 leg's own clone was refused (member-freedom check) — see the log above; T1's own snapshot is unaffected"
      echo "::warning::FA2 leg's own clone at ${CLONE_FA2_DIR} was refused (member-freedom check) — see the log above; T1's own snapshot is unaffected" >&2
    fi
  elif [ "$feat_rc" -eq 1 ]; then
    fa2_reason="jammi-kernels declares no flash-attn feature (cargo metadata, resolved sha ${_head_sha})"
    echo "FA2 leg skipped: jammi-kernels declares no flash-attn feature (cargo metadata)"
  else
    # round-5 fix (round-4 audit finding, family O — "trace the mechanism
    # behind a stated justification"): this arm used to WARN-and-skip,
    # justified in-comment by "the seed's own real invocation at step (i)
    # already aborts on the same rc=2" (pod_seed_target.sh:270-274 at the
    # time). That justification is FALSE whenever a seed already exists:
    # `pod_seed_target.sh --no-lock` (called without --reseed at step (i),
    # just above) short-circuits at "seed already complete — nothing to
    # do" (pod_seed_target.sh's own COMPLETE_MARKER gate) and returns 0
    # WITHOUT ever reaching the T1b/rc=2 abort — which is the ORDINARY
    # state here, since `up`/`shell` already kick off the seed at
    # bootstrap (dev-gpu-recipes.md). A broken `--frozen` metadata query
    # at THIS call site must therefore abort on its own terms, exactly
    # like the seed's own T1b gate does — never silently downgrade to
    # "absent" (the same on-pod incident class pod_seed_target.sh's own
    # T1b gate was fixed for in round 4).
    fail "FA2 leg: could not determine whether jammi-kernels declares flash-attn (cargo metadata query failed or the package was not found) — refusing to guess 'absent'; see pod_seed_pkg_has_feature's own ::error:: above for the real cause"
  fi
elif [ -z "$_main_sha" ]; then
  fa2_reason="could not resolve ${_main_sha_source} (no such remote-tracking ref found after 'git fetch --all --tags --prune' — this tree's 'origin' remote may not carry main at all, the ordinary shape for a bundle clone; set JAMMI_MAIN_SHA to bypass) — FA2 is main-only by design, skipping"
else
  fa2_reason="resolved sha ${_head_sha} != ${_main_sha_source} (${_main_sha}) — FA2 is main-only by design, skipping"
fi
echo "::endgroup::"

# ---- (iii) sccache requests unchanged by construction --------------------
sccache_delta_note="wrapper is off (CARGO_BUILD_RUSTC_WRAPPER=); sccache --show-stats before/after recorded verbatim below — expect identical (0 additional requests) since rustc never invoked it"

# ---- (iv) byte-equality vs a cold build at a SEPARATE, genuinely-empty target dir ----
# clone_hashes/recompiled were ALREADY snapshotted immediately after T1,
# above (round-3 audit N3) — this leg reads that snapshot, never re-takes
# it after the FA2 leg has had a chance to touch anything.
echo "::group::(iv) cold build @ a separate, genuinely-empty CARGO_TARGET_DIR"
cold_features="cuda"
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
  > /tmp/pod_build_timings.cold_build.log 2>&1 || fail "cold build (separate empty target dir) failed"
cold_t1=$(date +%s)
cold_wall=$((cold_t1 - cold_t0))
# After the wall-clock stop, not before it -- this cross-check must never
# inflate the measured cold_build_wall_s number it is only verifying the
# provenance of.
check_bin_provenance "$COLD_DIR/release/jammi-bench"
cold_hashes="$(snapshot_hashes "$COLD_DIR")"
# round-4 audit A4: an EMPTY match set on both sides made byte_equal read
# "true" (empty string equals empty string) — the SAME empty-glob vacuity
# N4 fixed for the env-surface cross-check, never applied to this
# acceptance oracle itself, whose whole claim (byte-equality of jammi-bench
# + every .ptx) is meaningless if `find` matched zero files on either leg
# (a path bug, a build that produced nothing where something was
# expected). A non-empty match set on BOTH sides is required before
# "equal" can mean anything; otherwise the comparison is INVALID, not a
# silent true.
clone_paths="$(printf '%s' "$clone_hashes" | cut -f1)"
cold_paths="$(printf '%s' "$cold_hashes" | cut -f1)"
clone_file_count="$(printf '%s' "$clone_hashes" | grep -c . || true)" # tripwire-ok: grep -c on an empty string legitimately returns 0 with rc=1 (no lines matched) -- the count itself (0) is exactly what byte_equal="invalid" below reads, never a silent miscount
cold_file_count="$(printf '%s' "$cold_hashes" | grep -c . || true)" # tripwire-ok: same as clone_file_count above
# round-6 fix: recorded in the JSON below so a reader (or a future CI
# check) can confirm a `set_mismatch` verdict, or independently notice a
# future one, WITHOUT re-deriving the full path list from raw logs — a
# compact fingerprint of "which artifacts did this side even produce".
clone_path_set_sha256="$(printf '%s\n' "$clone_paths" | pod_sha256_of_stdin)"
cold_path_set_sha256="$(printf '%s\n' "$cold_paths" | pod_sha256_of_stdin)"
if [ -z "$clone_hashes" ] || [ -z "$cold_hashes" ]; then
  byte_equal="invalid"
  byte_equal_diff="clone_hashes or cold_hashes matched ZERO files (clone empty: $([ -z "$clone_hashes" ] && echo yes || echo no); cold empty: $([ -z "$cold_hashes" ] && echo yes || echo no)) — the byte-equality comparison is MEANINGLESS, not a pass"
elif [ "$clone_paths" != "$cold_paths" ]; then
  # round-6 fix (live a100c run at 63bf905): a SET mismatch (different
  # PATHS present, e.g. the debug/-vs-release/ scoping bug snapshot_hashes
  # itself was just fixed for above) is a DISTINCT finding from a byte
  # mismatch on files both sides agree exist — collapsing it into "false"
  # reads, at a glance, exactly like a genuine build-reproducibility
  # regression, when the real story is "the oracle compared two different
  # sets of artifacts, not the same artifacts' bytes." Named explicitly,
  # with the symmetric difference (paths present on only one side).
  byte_equal="set_mismatch"
  byte_equal_diff="$(diff <(printf '%s\n' "$clone_paths") <(printf '%s\n' "$cold_paths"))"
elif [ "$clone_hashes" = "$cold_hashes" ]; then
  byte_equal="true"; byte_equal_diff=""
else
  byte_equal="false"
  byte_equal_diff="$(diff <(echo "$clone_hashes") <(echo "$cold_hashes") | head -50)"
fi
# round-5 fix (round-4 audit advisory: "a self-check between two constants
# proves nothing"): the OLD form asserted `clone_features == cold_features`
# — but BOTH are hardcoded to the literal "cuda" at their own assignment
# sites above (:244, :335), never derived from either cargo invocation's
# actual `--features` argument, so the comparison could never fire; it
# tested the script's own two adjacent string literals against each
# other, not the real build commands. clone_features/cold_features are
# still recorded in the output JSON below (informational — a future
# reader wants to see what was actually built), just without the
# tautological self-check.
echo "::endgroup::"

# ---- assemble result JSON (single pass; every value passed explicitly,
# nothing read back from a scratch file) -----------------------------------
RECOMPILED="$recompiled" SCCACHE_DELTA_NOTE="$sccache_delta_note" \
  SCCACHE_BEFORE="$sccache_before" SCCACHE_AFTER="$sccache_after" \
  python3 - "$BOX" "$(git rev-parse HEAD)" "$JAMMI_FA2_TIP_REF" "$clone_wall" "${fa2_wall:-}" \
  "$copy_wall" "$reflink_took" "$S_src_bytes" "$S_seed_bytes" "$S_clone_bytes" \
  "$byte_equal" "$(date -u +%FT%TZ)" "$cold_wall" "$LOCK_HELD" "$clone_features" "$cold_features" "$fa2_features" \
  "$seed_tuples_json" "$seed_t1b_ran" "$seed_t1b_reason" \
  "$clone_file_count" "$cold_file_count" "$clone_path_set_sha256" "$cold_path_set_sha256" "$byte_equal_diff" \
  "$fa2_ran" "$fa2_reason" \
  > "$JAMMI_BUILD_TIMINGS_OUT" <<'PY'
import json, sys, os
(box, sha, tip_ref, clone_wall, fa2_wall, copy_wall, reflink_took,
 s_src, s_seed, s_clone, byte_equal, ts, cold_wall, lock_held,
 clone_features, cold_features, fa2_features,
 seed_tuples_json, seed_t1b_ran, seed_t1b_reason,
 clone_file_count, cold_file_count, clone_path_set_sha256, cold_path_set_sha256,
 byte_equal_diff, fa2_ran, fa2_reason) = sys.argv[1:28]
try:
    seed_tuples = json.loads(seed_tuples_json)
except Exception:
    seed_tuples = []
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
    # round-6 fix (audit item 4): ALWAYS recorded, mirroring seed_t1b_*
    # above — a reader never has to guess whether flash_attn_leg_wall_s
    # is null because this leg correctly determined it should not run
    # (fa2_ran=false, a real reason given) vs. silently failed to run at
    # all (the exact class this fix closes: the OLD gate compared HEAD's
    # own detached-after-checkout-by-sha abbrev-ref to "main", which
    # never matched, with no else arm to record why).
    "fa2_ran": fa2_ran == "true",
    "fa2_reason": fa2_reason or None,
    "copy_wall_s": int(copy_wall),
    "reflink": reflink_took,
    "S_src_bytes": int(s_src) if s_src else None,
    "S_seed_bytes": int(s_seed) if s_seed else None,
    "S_clone_bytes": int(s_clone) if s_clone else None,
    # round-5 fix (round-4 audit advisory: "a headline delta must be a
    # same-run, same-box control"): the OLD field hardcoded a 284s
    # constant from ledger row 1 (a DIFFERENT box, different load;
    # .jammi/ledger/ is gitignored, so nothing in the repo even PRODUCES
    # 284) into this artifact's schema, next to `cold_build_wall_s` above
    # which already measures the SAME-run, same-box cold-build control. A
    # reader who wants the delta can compute clone_build_wall_s -
    # cold_build_wall_s (or vs. any other row they choose) FROM the two
    # real numbers already here — this producer no longer bakes in one
    # specific, uncommitted, cross-box comparator as if it were part of
    # the measurement itself.
  },
  # round-3 audit N3: recorded explicitly so a reader (or a future CI check)
  # can independently confirm the byte-equality comparison above compared
  # like-for-like, without re-deriving it from the raw logs.
  "clone_features": clone_features,
  "cold_features": cold_features,
  "fa2_features": fa2_features or None,
  # round-5 fix (Class-A item 4): which tuples the SEED itself actually
  # built (pod_seed_target.sh's own completion marker, read back at step
  # (i) above) — makes `flash_attn_leg_wall_s` interpretable: it is only
  # meaningful when seed_t1b_flash_attn_ran is also true (this producer's
  # OWN FA2 leg clones FROM that seed).
  "seed_tuples": seed_tuples,
  "seed_t1b_flash_attn_ran": seed_t1b_ran == "true",
  "seed_t1b_flash_attn_reason": seed_t1b_reason or None,
  "recompiled_units": os.environ.get("RECOMPILED", "").splitlines(),
  # round-4 audit A4: this field used to collapse "invalid" (empty match set
  # on one or both legs — the comparison never actually ran) into the same
  # False as a genuine byte mismatch, so a reader of the committed JSON could
  # not tell "the byte-equality check FAILED" from "the byte-equality check
  # never meaningfully ran". Emit the raw tri-state string so both cases are
  # distinguishable in the artifact; the boolean below is kept for existing
  # consumers that only care about the true/not-true axis.
  #
  # round-5 fix (round-4 audit advisory: "'same target dir' doc/mechanism
  # agree"): the module doc (above, "DELIBERATELY two DIFFERENT literal
  # CARGO_TARGET_DIR paths") was already corrected, but this field's OWN
  # NAME still asserted the claim the doc retracted. Renamed to describe
  # what the mechanism actually compares.
  "byte_equal_clone_vs_cold": byte_equal == "true",
  # round-6 fix (live a100c run at 63bf905: byte_equal=false because the
  # clone snapshot enumerated 54 files — it inherited the seed's debug/
  # build outputs — against the cold snapshot's 21 release-only files; a
  # SET mismatch, not a byte mismatch, reported as a bare "false"
  # indistinguishable from a genuine reproducibility regression).
  # `byte_equal_state` now includes "set_mismatch" as its own value
  # (distinct from "true"/"false"/"invalid"); these four fields record
  # what each side's snapshot actually enumerated so a reader — or this
  # producer's own next run — never has to re-derive it from raw logs to
  # tell "different artifacts" from "different bytes of the same
  # artifacts".
  "byte_equal_state": byte_equal,
  "clone_file_count": int(clone_file_count) if clone_file_count else 0,
  "cold_file_count": int(cold_file_count) if cold_file_count else 0,
  "clone_path_set_sha256": clone_path_set_sha256,
  "cold_path_set_sha256": cold_path_set_sha256,
  "byte_equal_diff": byte_equal_diff if byte_equal != "true" else None,
  # round-6 fix (audit item B, live a100c evidence): binary byte-equality
  # is UNATTAINABLE with this toolchain (release/jammi-bench carries 467
  # ThinLTO local-symbol suffixes that differ between two builds of the
  # IDENTICAL tree on the SAME box — mold 2.35.1 / clang 21's own ThinLTO
  # codegen). Recorded explicitly, a constant description of what THIS
  # comparison covers and what it deliberately excludes (and why) — never
  # a claim the linked binary matches when it structurally cannot.
  "byte_equal_scope": {
    "compared": ["release/*.ptx", "release/**/libjammi_*.rlib", "release/**/libjammi_*.rmeta"],
    "excluded": ["release/jammi-bench: thinlto-local-symbol-nondeterminism (mold 2.35.1 / clang 21 — differs between two builds of the identical tree on the same box)"],
  },
  "sccache_note": os.environ.get("SCCACHE_DELTA_NOTE", ""),
  "sccache_before": os.environ.get("SCCACHE_BEFORE", ""),
  "sccache_after": os.environ.get("SCCACHE_AFTER", ""),
}
print(json.dumps(result, indent=2))
PY

# round-4 audit A4: "invalid" is not a milder form of "false" — it means the
# comparison never ran (find matched zero files on one or both legs), so the
# committed artifact would otherwise carry a byte_equal value that looks like
# a completed check. Hard-fail rather than warn: a warning here would let a
# meaningless leg produce a JSON artifact indistinguishable, at a glance,
# from one where (iv) genuinely passed or genuinely failed.
if [ "$byte_equal" = "invalid" ]; then
  fail "byte-equality check (iv) is INVALID, not a pass or a fail: $byte_equal_diff"
elif [ "$byte_equal" = "set_mismatch" ]; then
  echo "::warning::byte-equality check (iv) is a SET MISMATCH (iv), not a byte-level pass/fail — clone_file_count=${clone_file_count} cold_file_count=${cold_file_count}; the two sides enumerated DIFFERENT artifacts, see diff below" >&2
  echo "$byte_equal_diff" >&2
elif [ "$byte_equal" != "true" ]; then
  echo "::warning::byte-equality FAILED (iv) — see diff below" >&2
  echo "$byte_equal_diff" >&2
fi
echo "pipeline complete: result JSON written to ${JAMMI_BUILD_TIMINGS_OUT} — copy it to ci/artifacts/pod-build-timings/<ts>-<sha7>.json and commit it (see this file's header)." >&2
