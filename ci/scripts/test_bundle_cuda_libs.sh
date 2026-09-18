#!/usr/bin/env bash
# Hermetic suite for `ci/scripts/bundle_cuda_libs.sh` — the derivation and
# loader-verification arm that decide which shared libraries the CUDA
# (`cu12`) release tarball carries, and whether they actually load from the
# stage directory.
#
# Why a suite at all: the only caller of that script is
# `.github/workflows/release-binaries.yml`'s `server-cu12-build` job, whose
# promote leg runs on a `v*` TAG. Nothing on the merge path executes the
# derivation, so without this file its first real exercise would be a release
# — the shape that let the retired hand-written soname list ship a binary with
# an unsatisfiable `DT_NEEDED libnccl.so.2` in the first place (#535), and the
# shape that let a retired loader-verification arm pass a stage that could not
# actually run standalone (#534).
#
# Hermetic in the strict sense: no `readelf`, no real `ldd` run, no ELF file,
# no network, no CUDA install — so it runs identically on the Linux `Guard`
# runner and on a maintainer's macOS box. Two seams make that possible and
# keep the shipped code in the loop rather than re-implementing it: the suite
# `source`s the actual script and replaces `bundle_needed_sonames` (its ONE
# ELF-reading function) with a fixture map, and it feeds
# `bundle_verify_loader_resolution` literal captured-report TEXT rather than
# ever invoking the real loader. Everything else — resolution, ordering, the
# host-provided partition, the transitive closure, the refusal, the staging
# copies, the report parse, the resolved-path check — is the real code path.
#
# The fixture link set is MEASURED, not invented: it is the `DT_NEEDED` list
# `packaging/server-cu12/verify_link_set.py`'s own suite pins for the shipped
# `jammi-server` binary (read off the `server-cu12-binary` artifact of run
# 34717957779), plus `libnccl.so.2`, which is what `candle-core/nccl` in
# `jammi-ai`'s `cuda` feature adds.
#
# One edge a PRIOR revision of this fixture inferred, rather than measured,
# was WRONG: `libnvrtc.so.12 -> libnvrtc-builtins`. A MEASUREMENT — `docker run
# --rm --platform linux/amd64 nvidia/cuda:12.6.3-devel-ubi8 readelf -d
# /usr/local/cuda-12.6/lib64/libnvrtc.so.12` — shows the real `libnvrtc.so.12`'s
# `Dynamic section` names exactly these `NEEDED` entries (an S1-style fact,
# recorded here because nothing else in this repo pins it):
#
#   libpthread.so.0
#   librt.so.1
#   libdl.so.2
#   libm.so.6
#   libc.so.6
#   ld-linux-x86-64.so.2
#
# Every one of those is host-provided (platform). There is NO `NEEDED` entry
# for `libnvrtc-builtins` anywhere in that list — NVRTC `dlopen`s its builtins
# library at runtime rather than linking it, so a `DT_NEEDED`-only closure
# walk, however faithfully implemented, structurally cannot discover it. This
# is exactly why `bundle_stage_floor` exists as a mechanism independent of the
# `DT_NEEDED` closure: the fixture below matches the measured fact (`libnvrtc.
# so.12`'s own needed-sonames case names only a host-provided dependency), and
# the suite's own assertions below prove `libnvrtc-builtins` still ends up
# staged — through the floor, never through the closure.
#
# `ci/scripts/fixtures/cu12_loader_report_real.txt` (arm 1a, DETECTION) is a
# REAL `ldd` report captured from the actual cu12 binary in the release
# lane's own CUDA container, and check 14 below verifies it against the
# measured `DT_NEEDED` set with no `BUNDLE_FIXTURE_PROVISIONAL` needed.
# `ci/scripts/fixtures/cu12_jail_report_real.txt` (arm 1b, THE JAIL — #534's
# chroot half) is still the CLEARLY-LABELLED `captured: pending` placeholder
# (see check 20 below): `BUNDLE_FIXTURE_PROVISIONAL=1` is required to run
# this suite at all until the lead dispatches `release-binaries.yml`,
# downloads the `cu12-jail-report` workflow artifact, and commits its
# content in place of the placeholder — the intended effect: the merge path
# stays red on this file until that real report lands as its own commit.
#
# Run today: `BUNDLE_FIXTURE_PROVISIONAL=1 bash ci/scripts/test_bundle_cuda_libs.sh`
# Run once the jail fixture is real: `bash ci/scripts/test_bundle_cuda_libs.sh`
#
# shellcheck disable=SC2329,SC2034,SC2012,SC2086,SC2016
# File-level, not per-site, because every instance across this file is the
# SAME five deliberate shapes: SC2086 ("double quote to prevent word
# splitting") fires on every `bundle_verify_loader_resolution ... $NEEDED`
# call below — deliberately unquoted for the identical reason `bundle_cuda_
# libs.sh`'s own `bundle_main` disables it inline: a space-separated soname
# list is meant to become SEPARATE positional arguments (sonames contain no
# whitespace). SC2329 ("function never invoked") fires on
# every `bundle_*`/`cp` fixture-replacement function below — shellcheck
# cannot see that `bundle_main`/`bundle_copy_sources`/etc., defined in the
# SOURCED `bundle_cuda_libs.sh`, call them indirectly, which is the entire
# point of the fixture seam. SC2034 ("appears unused") fires on several
# `*_out` captures kept only for a human re-running a failing case by hand
# (their exit code, captured the line after via `$?`, is what every
# assertion actually drives on) — this file's OWN suite asserts other
# `*_out` captures directly by name, so this is a deliberate per-check
# choice, not an oversight worth flagging repo-wide. SC2012 (`ls` over
# `find`) fires on two `ls | wc -l`/`ls | sort` pipelines over a `mktemp -d`
# fixture tree this same file creates with only alphanumeric names -- the
# non-alphanumeric-filename hazard `find` would close does not exist here.
# SC2016 ("expressions don't expand in single quotes") fires on every
# single-quoted `$ORIGIN`/`${ORIGIN}` literal in the
# RPATH/RUNPATH fixtures below -- deliberately literal, the same reason
# `bundle_cuda_libs.sh`'s own `bundle_is_origin_only_rpath` disables it
# inline: `$ORIGIN` is glibc's dynamic string token, never a shell variable.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/bundle_cuda_libs.sh"
REAL_REPORT_FIXTURE="${HERE}/fixtures/cu12_loader_report_real.txt"
REAL_JAIL_REPORT_FIXTURE="${HERE}/fixtures/cu12_jail_report_real.txt"
WORKFLOW="${HERE}/../../.github/workflows/release-binaries.yml"

failures=0
checks=0

fail() {
  failures=$((failures + 1))
  echo "FAIL[$1]: $2" >&2
}

ok() {
  echo "ok[$1]"
}

assert_contains() {
  local name="$1" haystack="$2" needle="$3"
  checks=$((checks + 1))
  case "$haystack" in
    *"$needle"*) ok "$name" ;;
    *) fail "$name" "expected to find '${needle}' in:
${haystack}" ;;
  esac
}

assert_not_contains() {
  local name="$1" haystack="$2" needle="$3"
  checks=$((checks + 1))
  case "$haystack" in
    *"$needle"*) fail "$name" "expected NOT to find '${needle}' in:
${haystack}" ;;
    *) ok "$name" ;;
  esac
}

assert_eq() {
  local name="$1" actual="$2" expected="$3"
  checks=$((checks + 1))
  if [ "$actual" = "$expected" ]; then
    ok "$name"
  else
    fail "$name" "expected '${expected}', got '${actual}'"
  fi
}

# ---------------------------------------------------------------------------
# Step 0: the script parses. A syntax error in a file whose only other caller
# is a tag-triggered release job would otherwise surface during that release.
# ---------------------------------------------------------------------------
checks=$((checks + 1))
if bash -n "$SCRIPT"; then
  ok "bundle_cuda_libs.sh parses"
else
  fail "bundle_cuda_libs.sh parses" "bash -n reported a syntax error"
fi

# shellcheck source=ci/scripts/bundle_cuda_libs.sh
. "$SCRIPT"
# The sourced script sets `-e` for its own execution as a program; this suite
# asserts on the exit codes of the refusals, so it takes `-e` back off here.
# `-u`/`pipefail` stay.
set +e

# ---------------------------------------------------------------------------
# The fake library tree. Empty files: nothing here is ever read as an ELF, only
# resolved by name, which is precisely the part of the derivation under test.
# ---------------------------------------------------------------------------
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bundle-cuda-libs-test.XXXXXX")"
trap 'rm -rf "$ROOT"' EXIT

TOOLKIT="${ROOT}/usr/local/cuda-12.6/lib64"
SYSLIB="${ROOT}/usr/lib64"
mkdir -p "$TOOLKIT" "$SYSLIB"

# The toolkit dir, as the CUDA 12.6 install lays it out: each library present
# both as its SONAME and as the fully-versioned object the SONAME links to.
for f in \
  libcudart.so.12 libcudart.so.12.6.77 \
  libcublas.so.12 libcublas.so.12.6.4.1 \
  libcublasLt.so.12 libcublasLt.so.12.6.4.1 \
  libcurand.so.10 libcurand.so.10.3.7.77 \
  libnvrtc.so.12 libnvrtc.so.12.6.85 \
  libnvrtc-builtins.so.12.6 libnvrtc-builtins.so.12.6.85 \
  libstdc++.so.6; do
  : >"${TOOLKIT}/${f}"
done

# `/usr/lib64`: where the image's `libnccl` RPM installs, alongside the
# platform's own libraries. `libcudart.so.12` is present here TOO, as a decoy:
# the search path's order is a determinant of the derivation, and a test that
# never presents a choice cannot see it.
for f in \
  libnccl.so.2 libnccl.so.2.23.4 \
  libcudart.so.12 \
  libstdc++.so.6 libc.so.6; do
  : >"${SYSLIB}/${f}"
done

SEARCH="${TOOLKIT}:${SYSLIB}"
BINARY="${ROOT}/fake-jammi-server"
: >"$BINARY"

# The measured `DT_NEEDED` list of the shipped cu12 binary (see the module
# doc), in link order, plus the `libnccl.so.2` that `candle-core/nccl` adds.
BINARY_NEEDED="libcudart.so.12
libstdc++.so.6
libcuda.so.1
libnvrtc.so.12
libcurand.so.10
libcublas.so.12
libcublasLt.so.12
libnccl.so.2
libdl.so.2
libgcc_s.so.1
librt.so.1
libpthread.so.0
libm.so.6
libmvec.so.1
libc.so.6
ld-linux-x86-64.so.2"

# The fixture that replaces the script's one ELF-reading function, keyed by
# BASENAME so a resolved absolute path answers the same as a bare soname.
#
# `libnvrtc.so.12`'s and `libnccl.so.2`'s own `NEEDED` sets are MEASURED (see
# the module doc for `libnvrtc.so.12`'s recorded `readelf -d` output): both
# name only host-provided (platform) libraries, so neither contributes a
# further non-host soname to the closure. In particular this fixture no
# longer claims `libnvrtc.so.12 -> libnvrtc-builtins` — that edge does not
# exist on the real object — which is exactly why `bundle_stage_floor` is
# exercised below as the ONLY mechanism this suite has for staging
# `libnvrtc-builtins` at all.
install_fixture_needed() {
  bundle_needed_sonames() {
    case "$(basename "$1")" in
      fake-jammi-server) printf '%s\n' "$BINARY_NEEDED" ;;
      libnvrtc.so.12) printf '%s\n' "libc.so.6" ;;
      libnccl.so.2) printf '%s\n' "libc.so.6" ;;
      *) : ;;
    esac
  }
}
install_fixture_needed

# shellcheck disable=SC2046  # one soname per line, and no soname has whitespace
sources="$(bundle_copy_sources "$SEARCH" $(bundle_needed_sonames "$BINARY"))"
sources_rc=$?
assert_eq "derivation succeeds on the measured link set" "$sources_rc" "0"

# ---------------------------------------------------------------------------
# 0b. The DEFAULT search path is itself a determinant, and the suite cannot see
#     it through any of the cases below: they all pass an explicit fake tree,
#     as a hermetic suite must. The release workflow passes NO search path at
#     all — the constant below is the single site that decides where a soname
#     may come from — so it is asserted directly. A mutation dropping
#     `/usr/lib64` from it (the whole NCCL half of this unit) otherwise leaves
#     every other check in this file green.
# ---------------------------------------------------------------------------
assert_contains "the default search path carries the CUDA 12.6 toolkit" \
  "$BUNDLE_DEFAULT_SEARCH_PATH" "/usr/local/cuda-12.6/lib64"
assert_contains "the default search path carries /usr/lib64 (where libnccl lives)" \
  "$BUNDLE_DEFAULT_SEARCH_PATH" "/usr/lib64"
assert_eq "the toolkit precedes /usr/lib64 in the default search path" \
  "$BUNDLE_DEFAULT_SEARCH_PATH" "/usr/local/cuda-12.6/lib64:/usr/lib64"

# ---------------------------------------------------------------------------
# 1. This unit's own fact: NCCL is carried, and it comes from /usr/lib64 — the
#    toolkit dir holds no NCCL at all, which is why a derivation keyed only to
#    the toolkit dir would silently drop it.
# ---------------------------------------------------------------------------
assert_contains "nccl soname staged from /usr/lib64" "$sources" "${SYSLIB}/libnccl.so.2
"
# No trailing newline in this needle (unlike the sibling check above): with
# the fictional `libnvrtc -> libnvrtc-builtins` edge gone (measured false;
# see the module doc), `libnccl.so.2.23.4` is now genuinely the LAST line
# `bundle_copy_sources` emits, and `$(...)` command substitution strips a
# trailing newline — a needle anchored on one would never match the true
# last line regardless of correctness.
assert_contains "nccl versioned object staged from /usr/lib64" "$sources" "${SYSLIB}/libnccl.so.2.23.4"

# ---------------------------------------------------------------------------
# 2. Search-path ORDER: `libcudart.so.12` exists in both directories; the
#    toolkit's copy wins because the toolkit is first.
# ---------------------------------------------------------------------------
assert_contains "cudart resolves from the toolkit" "$sources" "${TOOLKIT}/libcudart.so.12
"
assert_not_contains "cudart does not resolve from /usr/lib64" "$sources" "${SYSLIB}/libcudart.so.12
"
assert_eq "resolver reports the first matching directory" \
  "$(bundle_resolve_soname libcudart.so.12 "$SEARCH")" "$TOOLKIT"
assert_eq "resolver reports /usr/lib64 for a soname only it holds" \
  "$(bundle_resolve_soname libnccl.so.2 "$SEARCH")" "$SYSLIB"

# ---------------------------------------------------------------------------
# 3. Every name the retired hand list carried, that a `DT_NEEDED` closure walk
#    CAN reach, is still carried by the DERIVATION alone (`$sources`) —
#    `libnvrtc-builtins` is deliberately excluded from this list: measurement
#    shows it is not reachable by any `DT_NEEDED` edge (see the module doc),
#    so it is asserted separately, below, as a FLOOR fact rather than a
#    derivation fact.
# ---------------------------------------------------------------------------
for stem in libcudart libcublas libcublasLt libcurand libnvrtc; do
  assert_contains "hand-list member ${stem} still staged by the closure" "$sources" "${TOOLKIT}/${stem}.so."
done
assert_not_contains "libnvrtc-builtins is NOT reachable by the DT_NEEDED closure alone (measured fact)" \
  "$sources" "libnvrtc-builtins"

# ---------------------------------------------------------------------------
# 4. The host-provided partition: neither the platform's own libraries nor the
#    driver's are staged, even though the fake tree holds them.
# ---------------------------------------------------------------------------
assert_not_contains "libstdc++ is never staged" "$sources" "libstdc++"
assert_not_contains "libc is never staged" "$sources" "libc.so"
assert_not_contains "the driver's libcuda is never staged" "$sources" "libcuda.so"
for soname in libc.so.6 libm.so.6 libmvec.so.1 libdl.so.2 librt.so.1 libpthread.so.0 \
  libgcc_s.so.1 libstdc++.so.6 ld-linux-x86-64.so.2 ld-linux-aarch64.so.1 \
  libcuda.so.1 libnvidia-ptxjitcompiler.so.1; do
  checks=$((checks + 1))
  if bundle_is_host_provided "$soname"; then
    ok "host-provided: ${soname}"
  else
    fail "host-provided: ${soname}" "classified as the tarball's to carry"
  fi
done
for soname in libnccl.so.2 libcudart.so.12 libcublasLt.so.12 libnvrtc-builtins.so.12.6; do
  checks=$((checks + 1))
  if bundle_is_host_provided "$soname"; then
    fail "the tarball's to carry: ${soname}" "classified as host-provided"
  else
    ok "the tarball's to carry: ${soname}"
  fi
done

# ---------------------------------------------------------------------------
# 5. The refusal. A soname no search directory holds fails the derivation and
#    is NAMED — the case the retired hand list could not have had, because a
#    name it did not list was never looked for at all.
# ---------------------------------------------------------------------------
missing_out="$(bundle_copy_sources "$SEARCH" libcudart.so.12 libcusparse.so.12 2>&1)"
missing_rc=$?
assert_eq "an unresolvable soname fails the derivation" "$missing_rc" "1"
assert_contains "the unresolvable soname is named" "$missing_out" "libcusparse.so.12"

# The same refusal reached through a TRANSITIVE edge rather than a direct one.
bundle_needed_sonames() {
  case "$(basename "$1")" in
    libnvrtc.so.12) printf '%s\n' "libgone.so.1" ;;
    *) : ;;
  esac
}
transitive_out="$(bundle_copy_sources "$SEARCH" libnvrtc.so.12 2>&1)"
transitive_rc=$?
assert_eq "an unresolvable TRANSITIVE soname fails the derivation" "$transitive_rc" "1"
assert_contains "the unresolvable transitive soname is named" "$transitive_out" "libgone.so.1"
install_fixture_needed

# ---------------------------------------------------------------------------
# 6. The unversioned-soname hole (executed attack A1): a `DT_NEEDED` entry
#    that IS its own final object (`libfoo.so`, no trailing version) present
#    in the search dir. RED before this fix: the copy loop's stem glob
#    (`"$dir/$stem.so".*`) demands a LITERAL `.` immediately after `.so`,
#    which a bare `libfakeunversioned.so` — nothing after it — can never
#    satisfy, so the soname resolved (`-e "$dir/$soname"` passed) yet was
#    staged nowhere and reported nowhere: a silent omission on a real
#    `DT_NEEDED` entry the tarball genuinely cannot `exec` without.
# ---------------------------------------------------------------------------
: >"${TOOLKIT}/libfakeunversioned.so"
unver_sources="$(bundle_copy_sources "$SEARCH" libfakeunversioned.so)"
unver_rc=$?
assert_eq "an unversioned soname's derivation still succeeds" "$unver_rc" "0"
assert_contains "an unversioned soname is copied by its exact resolved name" \
  "$unver_sources" "${TOOLKIT}/libfakeunversioned.so"

# The floor's own resolver gets the same fallback (advisory carried from
# #535: `bundle_resolve_stem_dir`/the floor's copy step used to lack it
# entirely). A directory that holds ONLY the bare unversioned object for a
# stem (no versioned sibling at all) must still resolve and stage it.
: >"${ROOT}/only-unversioned-stem.so"
mkdir -p "${ROOT}/onlyunver"
: >"${ROOT}/onlyunver/libonlyunver.so"
assert_eq "bundle_stem_objects finds a bare unversioned object with no versioned sibling" \
  "$(bundle_stem_objects "${ROOT}/onlyunver" libonlyunver)" "${ROOT}/onlyunver/libonlyunver.so"
assert_eq "bundle_resolve_stem_dir resolves a directory holding only the unversioned object" \
  "$(bundle_resolve_stem_dir libonlyunver "${ROOT}/onlyunver")" "${ROOT}/onlyunver"

# ---------------------------------------------------------------------------
# 7. `bundle_main` end to end — the entry point the workflow calls. Nothing is
#    stubbed: the derivation and its copy step are pure filesystem operations
#    over the fake tree, which is what makes them hermetic and exercisable
#    through the real entry point rather than only in isolation.
# ---------------------------------------------------------------------------
STAGE="${ROOT}/stage/lib"
main_out="$(bundle_main "$BINARY" "$STAGE" "$SEARCH" 2>&1)"
main_rc=$?
assert_eq "bundle_main succeeds over the fixture tree" "$main_rc" "0"
assert_contains "bundle_main names what it stages" "$main_out" "staging ${SYSLIB}/libnccl.so.2"
staged="$(LC_ALL=C ls "$STAGE" | LC_ALL=C sort | tr '\n' ' ')"
assert_eq "the staged tree is exactly the derived closure UNION the floor" "$staged" \
  "libcublas.so.12 libcublas.so.12.6.4.1 libcublasLt.so.12 libcublasLt.so.12.6.4.1 libcudart.so.12 libcudart.so.12.6.77 libcurand.so.10 libcurand.so.10.3.7.77 libnccl.so.2 libnccl.so.2.23.4 libnvrtc-builtins.so.12.6 libnvrtc-builtins.so.12.6.85 libnvrtc.so.12 libnvrtc.so.12.6.85 "
assert_eq "libnvrtc-builtins ends up staged (via the floor, not the closure)" \
  "$([ -f "${STAGE}/libnvrtc-builtins.so.12.6" ] && echo yes || echo no)" "yes"

# A binary that needs nothing bundle-able is a build defect, not an empty-but-
# correct staging run: a CUDA build always links at least the CUDA runtime.
bundle_needed_sonames() {
  printf '%s\n' "libc.so.6" "libstdc++.so.6"
}
empty_out="$(bundle_main "$BINARY" "${ROOT}/stage-empty" "$SEARCH" 2>&1)"
empty_rc=$?
assert_eq "a link set with nothing to bundle is refused" "$empty_rc" "1"
assert_contains "the empty-set refusal explains itself" "$empty_out" "names no bundle-able DT_NEEDED library"
install_fixture_needed

# ---------------------------------------------------------------------------
# 8. The "lead's own anticipation" run: `bundle_main` over a SYNTHETIC tree
#    whose derived closure includes BOTH an unversioned soname AND the
#    ordinary versioned objects, together, through the real entry point —
#    not `bundle_copy_sources` in isolation. Reuses the same fixture tree
#    plus the one extra unversioned file from step 6, injected as a genuine
#    `DT_NEEDED` entry of the binary rather than only resolved directly.
# ---------------------------------------------------------------------------
UNVER_BINARY="${ROOT}/fake-jammi-server-unver"
: >"$UNVER_BINARY"
UNVER_STAGE="${ROOT}/stage-unver/lib"
bundle_needed_sonames() {
  case "$(basename "$1")" in
    fake-jammi-server-unver) printf '%s\n' "$BINARY_NEEDED" "libfakeunversioned.so" ;;
    libnvrtc.so.12) printf '%s\n' "libc.so.6" ;;
    libnccl.so.2) printf '%s\n' "libc.so.6" ;;
    *) : ;;
  esac
}
main_unver_out="$(bundle_main "$UNVER_BINARY" "$UNVER_STAGE" "$SEARCH" 2>&1)"
main_unver_rc=$?
assert_eq "bundle_main succeeds over a tree with an unversioned soname AND ordinary versioned objects" \
  "$main_unver_rc" "0"
assert_eq "the unversioned soname is staged by its exact name" \
  "$([ -f "${UNVER_STAGE}/libfakeunversioned.so" ] && echo yes || echo no)" "yes"
assert_eq "an ordinary versioned-only object (cudart) is staged alongside it" \
  "$([ -f "${UNVER_STAGE}/libcudart.so.12" ] && echo yes || echo no)" "yes"
assert_eq "an ordinary versioned-only object (cudart's real object) is staged alongside it" \
  "$([ -f "${UNVER_STAGE}/libcudart.so.12.6.77" ] && echo yes || echo no)" "yes"
assert_eq "the floor-only member (libnvrtc-builtins) is staged in the same run" \
  "$([ -f "${UNVER_STAGE}/libnvrtc-builtins.so.12.6" ] && echo yes || echo no)" "yes"
install_fixture_needed

# ---------------------------------------------------------------------------
# 9. The floor (`bundle_stage_floor`), driven directly — resolves and stages
#    the seven fixed stems from the search path, independent of any binary's
#    `DT_NEEDED` graph.
# ---------------------------------------------------------------------------
FLOOR_STAGE="${ROOT}/stage-floor/lib"
mkdir -p "$FLOOR_STAGE"
floor_out="$(bundle_stage_floor "$SEARCH" "$FLOOR_STAGE" 2>&1)"
floor_rc=$?
assert_eq "the floor succeeds over a tree that holds all seven stems" "$floor_rc" "0"
for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10 \
  libnvrtc.so.12 libnvrtc-builtins.so.12.6 libnccl.so.2; do
  checks=$((checks + 1))
  if [ -f "${FLOOR_STAGE}/${f}" ]; then
    ok "the floor stages ${f}"
  else
    fail "the floor stages ${f}" "expected a regular file at ${FLOOR_STAGE}/${f}"
  fi
done

# Mutation-shaped check: a search path missing the ONE directory that carries
# NCCL (`/usr/lib64`) still holds every other floor stem in the toolkit dir —
# the floor's refusal must name exactly the stem it could not resolve, not
# fail vacuously or fail the whole set silently.
FLOOR_STAGE_PARTIAL="${ROOT}/stage-floor-partial/lib"
mkdir -p "$FLOOR_STAGE_PARTIAL"
floor_partial_out="$(bundle_stage_floor "$TOOLKIT" "$FLOOR_STAGE_PARTIAL" 2>&1)"
floor_partial_rc=$?
assert_eq "the floor fails when a stem is not in any search directory" "$floor_partial_rc" "1"
assert_contains "the floor names the exact missing stem" "$floor_partial_out" "libnccl"
assert_not_contains "the floor does not spuriously name a stem it DID resolve" \
  "$floor_partial_out" "libcudart missing"

# ---------------------------------------------------------------------------
# 9b. The floor's collision refusal (T3/F5, #535's carried advisory): the
#     floor refuses ONLY when it would write a DESTINATION BASENAME the
#     derivation already staged from a DIFFERENT SOURCE OBJECT (compared by
#     realpath) — never keyed on which directory either side started in. Two
#     fixtures: a genuine conflict (two DIFFERENT objects both named
#     `libcublasLt.so.12`), and the tolerated carve-out (the SAME real object
#     reached through two different paths — a symlink bridging directories —
#     which is not a conflict at all and is simply skipped, not copied
#     twice).
# ---------------------------------------------------------------------------
COLL_A="${ROOT}/collision/dirA"
COLL_B="${ROOT}/collision/dirB"
mkdir -p "$COLL_A" "$COLL_B"
# dirA: everything the floor would independently resolve for all seven
# stems — a complete, self-consistent set. Nothing here is "wrong" by
# itself; the conflict comes from the closure_sources fixture below claiming
# a DIFFERENT real object at the exact same destination filename.
for f in libcudart.so.12 libcudart.so.12.6.77 libcublas.so.12 libcublas.so.12.6.4.1 \
  libcublasLt.so.12 libcublasLt.so.12.6.4.1 libcurand.so.10 libcurand.so.10.3.7.77 \
  libnvrtc.so.12 libnvrtc.so.12.6.85 libnvrtc-builtins.so.12.6 libnvrtc-builtins.so.12.6.85 \
  libnccl.so.2 libnccl.so.2.23.4; do
  : >"${COLL_A}/${f}"
done
# dirB: a DIFFERENT, non-empty object under the EXACT SAME destination
# basename the floor is about to write (`libcublasLt.so.12`) — standing in
# for "the derivation already staged a different build of this same-named
# library from elsewhere". Non-empty so its content (and therefore its
# identity) is genuinely distinct from dirA's empty fixture file.
printf 'a different object entirely' >"${COLL_B}/libcublasLt.so.12"

coll_sources="${COLL_B}/libcublasLt.so.12
"
COLL_STAGE="${ROOT}/collision/stage/lib"
mkdir -p "$COLL_STAGE"
coll_out="$(bundle_stage_floor "$COLL_A" "$COLL_STAGE" "$coll_sources" 2>&1)"
coll_rc=$?
assert_eq "the floor refuses when it would write a destination the derivation already staged from a DIFFERENT source object" \
  "$coll_rc" "1"
assert_contains "the collision refusal names the destination basename" "$coll_out" "libcublasLt.so.12"
assert_contains "the collision refusal names the floor's own source path" "$coll_out" "${COLL_A}/libcublasLt.so.12"
assert_contains "the collision refusal names the derivation's real source path" "$coll_out" \
  "$(bundle_realpath "${COLL_B}/libcublasLt.so.12")"
coll_staged_count="$(ls "$COLL_STAGE" 2>/dev/null | wc -l | tr -d ' ')"
assert_eq "the collision refusal stages nothing at all — refuses before any copy runs" "$coll_staged_count" "0"

# The tolerated carve-out: the closure's claimed source is a SYMLINK to the
# EXACT SAME real file the floor would independently resolve — one object,
# reached two ways. Not a conflict; the floor simply skips re-copying it and
# stages every other stem normally.
COLL_SAME_STAGE="${ROOT}/collision-same/stage/lib"
mkdir -p "$COLL_SAME_STAGE"
COLL_LINKDIR="${ROOT}/collision-same/linkdir"
mkdir -p "$COLL_LINKDIR"
ln -s "${COLL_A}/libcublasLt.so.12" "${COLL_LINKDIR}/libcublasLt.so.12"
coll_same_sources="${COLL_LINKDIR}/libcublasLt.so.12
"
coll_same_out="$(bundle_stage_floor "$COLL_A" "$COLL_SAME_STAGE" "$coll_same_sources" 2>&1)"
coll_same_rc=$?
assert_eq "the floor tolerates the SAME real object reached through two different paths (no conflict)" \
  "$coll_same_rc" "0"
assert_eq "the floor still stages every other stem when the same-object carve-out applies" \
  "$([ -f "${COLL_SAME_STAGE}/libcudart.so.12" ] && echo yes || echo no)" "yes"

# No closure_sources argument at all (the floor's stand-alone caller,
# `bundle_stage_floor "$SEARCH" "$FLOOR_STAGE"` in check 9 above) never
# collides, because there is nothing to compare against — asserted already
# by check 9's own success.

# ---------------------------------------------------------------------------
# 10. The stage-set assertion (`bundle_assert_staged`), driven directly: a
#     regular file of exactly the required SONAME must exist under `lib_dir`.
#     This is deliberately NOT the same claim `bundle_copy_sources` makes by
#     exiting 0 (see the module doc) — mutation-shaped: remove a file after a
#     successful derivation and the assertion, not the derivation, is what
#     catches it.
# ---------------------------------------------------------------------------
ASSERT_STAGE="${ROOT}/stage-assert/lib"
mkdir -p "$ASSERT_STAGE"
: >"${ASSERT_STAGE}/libcudart.so.12"
: >"${ASSERT_STAGE}/libnccl.so.2"
assert_out="$(bundle_assert_staged "$ASSERT_STAGE" libcudart.so.12 libnccl.so.2 2>&1)"
assert_rc=$?
assert_eq "the stage-set assertion passes when every required file is present" "$assert_rc" "0"

rm -f "${ASSERT_STAGE}/libnccl.so.2"
assert_out2="$(bundle_assert_staged "$ASSERT_STAGE" libcudart.so.12 libnccl.so.2 2>&1)"
assert_rc2=$?
assert_eq "the stage-set assertion fails when a required file is gone" "$assert_rc2" "1"
assert_contains "the stage-set assertion names the missing file" "$assert_out2" "libnccl.so.2"
assert_not_contains "the stage-set assertion does not name a file that IS present" \
  "$assert_out2" "libcudart.so.12 => "

# A symlink whose target does not exist is not a "regular file" either — `[ -f
# ]` follows the link and reports false for a broken one, which is the
# correct outcome for a soname that LOOKS staged but resolves to nothing.
ln -s "${ASSERT_STAGE}/does-not-exist.so.1" "${ASSERT_STAGE}/libbroken.so.1"
assert_broken_out="$(bundle_assert_staged "$ASSERT_STAGE" libbroken.so.1 2>&1)"
assert_broken_rc=$?
assert_eq "the stage-set assertion fails on a broken symlink" "$assert_broken_rc" "1"
assert_contains "the stage-set assertion names the broken symlink's soname" "$assert_broken_out" "libbroken.so.1"

# `bundle_main` itself now runs the stage-set assertion (and the floor) after
# the copy step — driven end to end, over the same fixture tree used above,
# rather than only in isolation.
assert_contains "bundle_main's own stage-set assertion reports success" "$main_out" \
  "every non-host-provided soname of the DT_NEEDED closure, plus the floor, is staged"

# ---------------------------------------------------------------------------
# 11. `bundle_main`-level (T2): each phase's failure is `bundle_main`'s own
#     failure, not swallowed by the unconditional success echo #535
#     reproduced at `326785ef` (`bundle_main` discarded the return status of
#     `bundle_stage_floor`/`bundle_resolve_closure`/`bundle_assert_staged`
#     and printed the sentence regardless).
# ---------------------------------------------------------------------------

# 11a. A floor-only stem (libnvrtc-builtins — absent from the DT_NEEDED
#      closure by measured fact, see the module doc) missing from EVERY
#      search directory must fail `bundle_main` end to end, even though the
#      derivation itself (the closure) succeeds cleanly.
MAIN2_TOOLKIT="${ROOT}/main2/toolkit"
MAIN2_SYS="${ROOT}/main2/sys"
mkdir -p "$MAIN2_TOOLKIT" "$MAIN2_SYS"
for f in libcudart.so.12 libcudart.so.12.6.77 libcublas.so.12 libcublas.so.12.6.4.1 \
  libcublasLt.so.12 libcublasLt.so.12.6.4.1 libcurand.so.10 libcurand.so.10.3.7.77 \
  libnvrtc.so.12 libnvrtc.so.12.6.85 libstdc++.so.6; do
  : >"${MAIN2_TOOLKIT}/${f}"
done
for f in libnccl.so.2 libnccl.so.2.23.4 libc.so.6; do
  : >"${MAIN2_SYS}/${f}"
done
MAIN2_SEARCH="${MAIN2_TOOLKIT}:${MAIN2_SYS}"
MAIN2_BINARY="${ROOT}/fake-jammi-server-nofloor"
: >"$MAIN2_BINARY"
bundle_needed_sonames() {
  case "$(basename "$1")" in
    fake-jammi-server-nofloor) printf '%s\n' "$BINARY_NEEDED" ;;
    libnvrtc.so.12) printf '%s\n' "libc.so.6" ;;
    libnccl.so.2) printf '%s\n' "libc.so.6" ;;
    *) : ;;
  esac
}
MAIN2_STAGE="${ROOT}/stage-nofloor/lib"
main2_out="$(bundle_main "$MAIN2_BINARY" "$MAIN2_STAGE" "$MAIN2_SEARCH" 2>&1)"
main2_rc=$?
install_fixture_needed
assert_eq "bundle_main fails end-to-end when a floor-only stem (libnvrtc-builtins) is absent from every search directory" \
  "$main2_rc" "1"
assert_contains "bundle_main's floor-missing failure names the stem" "$main2_out" "libnvrtc-builtins"
assert_not_contains "bundle_main's floor-missing failure does not print the success sentence" \
  "$main2_out" "is staged under"

# 11b. A required file missing from `$lib_dir` AFTER the copy step (a
#      resolution that succeeded but a copy that silently landed nothing)
#      must fail `bundle_main` via the stage-set assertion — the exact defect
#      the retired revision's discarded return values hid.
MAIN3_STAGE="${ROOT}/stage-missingfile/lib"
mkdir -p "$MAIN3_STAGE"
cp() {
  case "$2" in
    *libnccl*) return 0 ;; # simulate a copy that silently staged nothing
    *) command cp "$@" ;;
  esac
}
main3_out="$(bundle_main "$BINARY" "$MAIN3_STAGE" "$SEARCH" 2>&1)"
main3_rc=$?
unset -f cp
assert_eq "bundle_main fails end-to-end when a required file is missing after the copy step" \
  "$main3_rc" "1"
assert_contains "bundle_main's post-copy failure names the missing file" "$main3_out" "libnccl"

# ---------------------------------------------------------------------------
# 12. `bundle_parse_loader_report` (T4), driven directly: the three line
#     shapes a real `ldd`/`ld.so --list` report can carry.
# ---------------------------------------------------------------------------
parsed_ordinary="$(printf '%s\n' 'libcudart.so.12 => /opt/lib/libcudart.so.12 (0x00007f0000000000)' | bundle_parse_loader_report)"
assert_eq "parse: an ordinary resolved line" "$parsed_ordinary" "libcudart.so.12 RESOLVED /opt/lib/libcudart.so.12"

parsed_notfound="$(printf '%s\n' 'libnccl.so.2 => not found' | bundle_parse_loader_report)"
assert_eq "parse: a not-found line" "$parsed_notfound" "libnccl.so.2 NOTFOUND"

parsed_selfnamed="$(printf '%s\n' '	/lib64/ld-linux-x86-64.so.2 (0x00007ffff7fc0000)' | bundle_parse_loader_report)"
assert_eq "parse: the loader's own no-'=>' line resolves to its basename" \
  "$parsed_selfnamed" "ld-linux-x86-64.so.2 RESOLVED /lib64/ld-linux-x86-64.so.2"

parsed_blank="$(printf '\n  \n' | bundle_parse_loader_report)"
assert_eq "parse: blank/whitespace-only lines produce nothing" "$parsed_blank" ""

# ---------------------------------------------------------------------------
# 13. `bundle_verify_loader_resolution` (T4), driven directly against literal
#     captured-report TEXT — no `ldd`, no real loader, anywhere in this suite.
# ---------------------------------------------------------------------------
LOADER_LIB="${ROOT}/loader-lib"
mkdir -p "$LOADER_LIB"

# A minimal but REAL-SHAPED "full DT_NEEDED" set for these checks: two
# bundled (non-host) sonames, one platform member, one driver member.
LOADER_NEEDED="libcudart.so.12 libnccl.so.2 libc.so.6 libcuda.so.1"

# 13a. Defect (1) closed: a bundled soname the tarball never staged resolves
#      from a HOST path outside lib_dir — the loader "sees" it only because
#      the build host itself carries a copy, exactly the shape
#      `LD_LIBRARY_PATH` prepending (never restricting) makes possible. Must
#      FAIL, naming the wrong path.
good_report="libcudart.so.12 => ${LOADER_LIB}/libcudart.so.12 (0x1)
libnccl.so.2 => /usr/lib64/libnccl.so.2 (0x2)
libc.so.6 => /lib64/libc.so.6 (0x3)
libcuda.so.1 => /lib64/libcuda.so.1 (0x4)"
hostcopy_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$good_report" $LOADER_NEEDED 2>&1)"
hostcopy_rc=$?
assert_eq "loader verify: a bundled library resolved from a host path (not lib_dir) fails" "$hostcopy_rc" "1"
assert_contains "loader verify: the host-copy failure names the wrong path" "$hostcopy_out" "/usr/lib64/libnccl.so.2"

# 13b. Defect (2), three fixtures: an empty report, a "not a dynamic
#      executable" report, and a linux-vdso.so.1-only report must ALL fail —
#      none contains a resolved entry for any of `LOADER_NEEDED`, and this
#      suite demands that absence be a hard failure, not a vacuous pass.
empty_report=""
empty_report_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$empty_report" $LOADER_NEEDED 2>&1)"
empty_report_rc=$?
assert_eq "loader verify: an empty report fails (vacuous pass closed)" "$empty_report_rc" "1"

crashed_report="not a dynamic executable"
crashed_report_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$crashed_report" $LOADER_NEEDED 2>&1)"
crashed_report_rc=$?
assert_eq "loader verify: a crashed-loader report fails (vacuous pass closed)" "$crashed_report_rc" "1"

vdso_report="	linux-vdso.so.1 (0x00007ffff7fc0000)"
vdso_report_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$vdso_report" $LOADER_NEEDED 2>&1)"
vdso_report_rc=$?
assert_eq "loader verify: a vdso-only report fails (vacuous pass closed)" "$vdso_report_rc" "1"
assert_contains "loader verify: the vdso-only failure names an absent required soname" "$vdso_report_out" "libcudart.so.12"

# 13c. Defect (3) closed: the loader's own self-named (no `=>`) line is
#      parsed as a RESOLVED platform entry, so an otherwise CORRECT stage —
#      every bundled member from lib_dir, every platform/driver member from
#      the host, the loader naming itself with no `=>` — passes cleanly.
correct_report="libcudart.so.12 => ${LOADER_LIB}/libcudart.so.12 (0x1)
libnccl.so.2 => ${LOADER_LIB}/libnccl.so.2 (0x2)
libc.so.6 => /lib64/libc.so.6 (0x3)
libcuda.so.1 => /lib64/libcuda.so.1 (0x4)
	/lib64/ld-linux-x86-64.so.2 (0x00007ffff7fc0000)"
correct_needed="libcudart.so.12 libnccl.so.2 libc.so.6 libcuda.so.1 ld-linux-x86-64.so.2"
correct_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$correct_report" $correct_needed 2>&1)"
correct_rc=$?
assert_eq "loader verify: a correct stage with the loader's self-named line passes" "$correct_rc" "0"

# 13d. Defect (4) closed: lib_dir='/' is refused explicitly rather than
#      silently treating every absolute path as "under lib_dir" (which would
#      make 13a's own check vacuous at the root).
root_out="$(bundle_verify_loader_resolution "/" "$good_report" $LOADER_NEEDED 2>&1)"
root_rc=$?
assert_eq "loader verify: lib_dir='/' is refused explicitly" "$root_rc" "1"
assert_contains "loader verify: the lib_dir='/' refusal explains itself" "$root_out" "lib_dir='/'"

# 13e. Driver members are tolerated `not found` by the STATED list, never a
#      pattern — a driver-less CI host legitimately cannot resolve
#      `libcuda.so.1`, and that alone must not fail the arm.
driverless_report="libcudart.so.12 => ${LOADER_LIB}/libcudart.so.12 (0x1)
libnccl.so.2 => ${LOADER_LIB}/libnccl.so.2 (0x2)
libc.so.6 => /lib64/libc.so.6 (0x3)
libcuda.so.1 => not found"
driverless_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$driverless_report" $LOADER_NEEDED 2>&1)"
driverless_rc=$?
assert_eq "loader verify: a driver library 'not found' on a driver-less host is tolerated" "$driverless_rc" "0"

# A soname NOT on the driver-members list that is simply absent from the
# report must still fail — tolerance is by NAME, never inferred from shape.
notdriver_report="libcudart.so.12 => ${LOADER_LIB}/libcudart.so.12 (0x1)
libc.so.6 => /lib64/libc.so.6 (0x3)
libcuda.so.1 => not found"
notdriver_out="$(bundle_verify_loader_resolution "$LOADER_LIB" "$notdriver_report" $LOADER_NEEDED 2>&1)"
notdriver_rc=$?
assert_eq "loader verify: an absent NON-driver required soname still fails" "$notdriver_rc" "1"
assert_contains "loader verify: the non-driver absence is named" "$notdriver_out" "libnccl.so.2"

# ---------------------------------------------------------------------------
# 14. The real captured report (T4(3)'s own fixture; see this file's module
#     doc). Refuses to run at all unless BUNDLE_FIXTURE_PROVISIONAL=1 while
#     the fixture is still the CLEARLY-LABELLED `captured: pending`
#     placeholder — the intended effect: this suite, and therefore the merge
#     path, stays red on this file until the lead commits the real report
#     (`ci/scripts/capture_loader_report.sh`, run on a driver-only-style
#     host) as its own commit.
# ---------------------------------------------------------------------------
checks=$((checks + 1))
if [ ! -f "$REAL_REPORT_FIXTURE" ]; then
  fail "the real loader-report fixture exists" "expected a file at ${REAL_REPORT_FIXTURE}"
else
  ok "the real loader-report fixture exists"
  real_report_head="$(head -n1 "$REAL_REPORT_FIXTURE")"
  checks=$((checks + 1))
  case "$real_report_head" in
    "# captured: pending"*)
      if [ "${BUNDLE_FIXTURE_PROVISIONAL:-0}" != "1" ]; then
        fail "the real loader-report fixture is not provisional" \
          "fixture is still 'captured: pending' -- set BUNDLE_FIXTURE_PROVISIONAL=1 to run this suite anyway (the merge path itself must NOT set it), or land the real captured report"
      else
        ok "the real loader-report fixture is provisional, and BUNDLE_FIXTURE_PROVISIONAL=1 is set"
      fi
      ;;
    *)
      ok "the real loader-report fixture is a real capture (no 'captured: pending' header)"
      # Once real, it must actually verify clean against the measured
      # BINARY_NEEDED set (defect 3's true test: a REAL correct stage
      # passes). `libnvrtc-builtins` is deliberately NOT in this required
      # set — it is dlopen-only (see the module doc), so no `ldd` report can
      # ever carry a resolved entry for it; requiring it here would fail
      # every real, correct capture by construction.
      #
      # `lib_dir` is not hardcoded: it is read off the real report's OWN
      # `libcudart.so.12` entry (the directory `capture_loader_report.sh`'s
      # `LD_LIBRARY_PATH` actually pointed at on the capturing host), since
      # this suite has no other way to know that host's stage path.
      real_report_body="$(tail -n +2 "$REAL_REPORT_FIXTURE")"
      real_parsed="$(printf '%s\n' "$real_report_body" | bundle_parse_loader_report)"
      real_lib_dir="$(printf '%s\n' "$real_parsed" | while IFS=' ' read -r n s p; do
        if [ "$n" = "libcudart.so.12" ] && [ "$s" = "RESOLVED" ]; then
          dirname "$p"
          break
        fi
      done)"
      checks=$((checks + 1))
      if [ -z "$real_lib_dir" ]; then
        fail "the real report names a resolved libcudart.so.12 entry" \
          "no RESOLVED libcudart.so.12 line found in ${REAL_REPORT_FIXTURE} -- cannot infer the capturing host's stage directory"
      else
        ok "the real report names a resolved libcudart.so.12 entry"
        real_out="$(bundle_verify_loader_resolution "$real_lib_dir" "$real_report_body" $BINARY_NEEDED 2>&1)"
        real_rc=$?
        assert_eq "the real captured report verifies against the measured DT_NEEDED set" "$real_rc" "0"
      fi
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# 15a. F1, driven directly and by name: the closure EXCLUDES the
#      host-provided set even when a platform soname is directly resolvable
#      in the search path — a contract-literal "closure UNION floor" with no
#      exclusion would stage `libstdc++`/`libc`/`ld-linux-*` from
#      `/usr/lib64` and the launcher's `LD_LIBRARY_PATH` would then load a
#      glibc-2.28-toolkit-image libc under the host's own (likely newer)
#      loader — an ABI hazard the tarball must never ship, not merely an
#      oversight. Check 4 above already covers this implicitly for the
#      whole staged set; this is F1's OWN explicit, by-name oracle.
# ---------------------------------------------------------------------------
platform_excl_sources="$(bundle_copy_sources "$SEARCH" libstdc++.so.6 libcudart.so.12)"
assert_not_contains "F1: a resolvable platform soname (libstdc++) is excluded from the staged set" \
  "$platform_excl_sources" "libstdc++"
assert_contains "F1: a bundle-able soname in the SAME call is still staged" \
  "$platform_excl_sources" "libcudart.so.12"

# ---------------------------------------------------------------------------
# 15b. F4: `bundle_assert_no_runpath` — the binary carries no
#      `DT_RPATH`/`DT_RUNPATH` dynamic-section entry, checked before
#      anything else in `bundle_main`. `bundle_dynamic_section`, the SECOND
#      ELF-reading function, is replaced by the suite the same way
#      `bundle_needed_sonames` is.
# ---------------------------------------------------------------------------
bundle_dynamic_section() {
  case "$(basename "$1")" in
    fake-jammi-server-clean)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000000c (INIT)                0x1000"
      ;;
    fake-jammi-server-rpath)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000000f (RPATH)               Library rpath: [/opt/build-host/cuda/lib64]"
      ;;
    fake-jammi-server-runpath)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [/opt/build-host/cuda/lib64]"
      ;;
    # The REAL `readelf -d` line, verbatim, measured
    # against the CUDA 12.6 toolkit's own libcublas.so/libcublasLt.so/
    # libcurand.so (`readelf -d` inside `nvidia/cuda:12.6.3-devel-ubi8`) —
    # committed as a fixture so this exact shape is pinned, not re-typed.
    libcublasLt.so.12-origin-runpath)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [\$ORIGIN]"
      ;;
    fake-jammi-server-origin-rpath)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000000f (RPATH)               Library rpath: [\$ORIGIN]"
      ;;
    fake-jammi-server-origin-braced)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [\${ORIGIN}]"
      ;;
    fake-jammi-server-origin-dotdot)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [\$ORIGIN/..]"
      ;;
    fake-jammi-server-origin-dotdot-lib)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [\$ORIGIN/../lib]"
      ;;
    fake-jammi-server-usrlib64)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [/usr/lib64]"
      ;;
    fake-jammi-server-mixed-origin)
      printf '%s\n' \
        " 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]" \
        " 0x000000000000001d (RUNPATH)             Library runpath: [\$ORIGIN:/usr/lib]"
      ;;
    *) printf '%s\n' "" ;;
  esac
}
CLEAN_BINARY="${ROOT}/fake-jammi-server-clean"
RPATH_BINARY="${ROOT}/fake-jammi-server-rpath"
RUNPATH_BINARY="${ROOT}/fake-jammi-server-runpath"
ORIGIN_RUNPATH_BINARY="${ROOT}/libcublasLt.so.12-origin-runpath"
ORIGIN_RPATH_BINARY="${ROOT}/fake-jammi-server-origin-rpath"
ORIGIN_BRACED_BINARY="${ROOT}/fake-jammi-server-origin-braced"
ORIGIN_DOTDOT_BINARY="${ROOT}/fake-jammi-server-origin-dotdot"
ORIGIN_DOTDOT_LIB_BINARY="${ROOT}/fake-jammi-server-origin-dotdot-lib"
USRLIB64_BINARY="${ROOT}/fake-jammi-server-usrlib64"
MIXED_ORIGIN_BINARY="${ROOT}/fake-jammi-server-mixed-origin"
: >"$CLEAN_BINARY"
: >"$RPATH_BINARY"
: >"$RUNPATH_BINARY"
: >"$ORIGIN_RUNPATH_BINARY"
: >"$ORIGIN_RPATH_BINARY"
: >"$ORIGIN_BRACED_BINARY"
: >"$ORIGIN_DOTDOT_BINARY"
: >"$ORIGIN_DOTDOT_LIB_BINARY"
: >"$USRLIB64_BINARY"
: >"$MIXED_ORIGIN_BINARY"

clean_runpath_out="$(bundle_assert_no_runpath "$CLEAN_BINARY" 2>&1)"
clean_runpath_rc=$?
assert_eq "F4: a binary with no RPATH/RUNPATH passes" "$clean_runpath_rc" "0"

rpath_out="$(bundle_assert_no_runpath "$RPATH_BINARY" 2>&1)"
rpath_rc=$?
assert_eq "F4: a binary with DT_RPATH fails" "$rpath_rc" "1"
assert_contains "F4: the RPATH failure names the RPATH line" "$rpath_out" "RPATH"

runpath_out="$(bundle_assert_no_runpath "$RUNPATH_BINARY" 2>&1)"
runpath_rc=$?
assert_eq "F4: a binary with DT_RUNPATH fails" "$runpath_rc" "1"
assert_contains "F4: the RUNPATH failure names the RUNPATH line" "$runpath_out" "RUNPATH"

# ---------------------------------------------------------------------------
# 15b2. `$ORIGIN`-only RPATH/RUNPATH is ACCEPTED, by
#       component — the real shape NVIDIA ships (measured `readelf -d`
#       against the CUDA 12.6 toolkit's libcublasLt.so.12/libcublas.so.12/
#       libcurand.so.10), never refused by presence alone.
# ---------------------------------------------------------------------------
origin_only_rc=0
bundle_is_origin_only_rpath '$ORIGIN' || origin_only_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('\$ORIGIN') succeeds" "$origin_only_rc" "0"
origin_braced_rc=0
bundle_is_origin_only_rpath '${ORIGIN}' || origin_braced_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('\${ORIGIN}') succeeds" "$origin_braced_rc" "0"
origin_dotdot_rc=0
bundle_is_origin_only_rpath '$ORIGIN/..' || origin_dotdot_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('\$ORIGIN/..') fails (escapes the object's own directory)" "$origin_dotdot_rc" "1"
origin_dotdot_lib_rc=0
bundle_is_origin_only_rpath '$ORIGIN/../lib' || origin_dotdot_lib_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('\$ORIGIN/../lib') fails" "$origin_dotdot_lib_rc" "1"
usrlib64_rc=0
bundle_is_origin_only_rpath '/usr/lib64' || usrlib64_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('/usr/lib64') fails" "$usrlib64_rc" "1"
mixed_origin_rc=0
bundle_is_origin_only_rpath '$ORIGIN:/usr/lib' || mixed_origin_rc=$?
assert_eq "origin: bundle_is_origin_only_rpath('\$ORIGIN:/usr/lib') fails (mixed component)" "$mixed_origin_rc" "1"

# The same rule, end to end through `bundle_assert_no_runpath`, over a
# committed real-shape fixture and each mutation named above.
origin_runpath_out="$(bundle_assert_no_runpath "$ORIGIN_RUNPATH_BINARY" 2>&1)"
origin_runpath_rc=$?
assert_eq "origin: a real-shape \$ORIGIN RUNPATH (libcublasLt.so.12) passes" "$origin_runpath_rc" "0"

origin_rpath_out="$(bundle_assert_no_runpath "$ORIGIN_RPATH_BINARY" 2>&1)"
origin_rpath_rc=$?
assert_eq "origin: a \$ORIGIN RPATH (0x0f, same rule as RUNPATH) passes" "$origin_rpath_rc" "0"

origin_braced_out="$(bundle_assert_no_runpath "$ORIGIN_BRACED_BINARY" 2>&1)"
origin_braced_rc=$?
assert_eq "origin: a \${ORIGIN} (braced) RUNPATH passes" "$origin_braced_rc" "0"

origin_dotdot_out="$(bundle_assert_no_runpath "$ORIGIN_DOTDOT_BINARY" 2>&1)"
origin_dotdot_bin_rc=$?
assert_eq "origin: \$ORIGIN/.. fails (escapes the object's own directory)" "$origin_dotdot_bin_rc" "1"
assert_contains "origin: the \$ORIGIN/.. failure names the value" "$origin_dotdot_out" '$ORIGIN/..'

origin_dotdot_lib_out="$(bundle_assert_no_runpath "$ORIGIN_DOTDOT_LIB_BINARY" 2>&1)"
origin_dotdot_lib_bin_rc=$?
assert_eq "origin: \$ORIGIN/../lib fails" "$origin_dotdot_lib_bin_rc" "1"

usrlib64_out="$(bundle_assert_no_runpath "$USRLIB64_BINARY" 2>&1)"
usrlib64_bin_rc=$?
assert_eq "origin: /usr/lib64 (a bare absolute path, no \$ORIGIN at all) fails" "$usrlib64_bin_rc" "1"

mixed_origin_out="$(bundle_assert_no_runpath "$MIXED_ORIGIN_BINARY" 2>&1)"
mixed_origin_bin_rc=$?
assert_eq "origin: \$ORIGIN:/usr/lib (mixed components) fails" "$mixed_origin_bin_rc" "1"
assert_contains "origin: the mixed-component failure names the value" "$mixed_origin_out" '$ORIGIN:/usr/lib'

# `bundle_main` itself refuses a RUNPATH'd binary before staging anything —
# no stage directory is even created.
bundle_needed_sonames() {
  case "$(basename "$1")" in
    fake-jammi-server-runpath) printf '%s\n' "libcudart.so.12" ;;
    *) : ;;
  esac
}
RUNPATH_STAGE="${ROOT}/stage-runpath/lib"
main_runpath_out="$(bundle_main "$RUNPATH_BINARY" "$RUNPATH_STAGE" "$SEARCH" 2>&1)"
main_runpath_rc=$?
install_fixture_needed
assert_eq "bundle_main refuses a RUNPATH'd binary before staging anything" "$main_runpath_rc" "1"
assert_eq "bundle_main creates no stage directory for a RUNPATH'd binary (refused before mkdir)" \
  "$([ -d "$RUNPATH_STAGE" ] && echo exists || echo absent)" "absent"

# 15c. `bundle_assert_no_runpath_dir` extends
#      the SAME check over every regular file in a directory — reuses the
#      `bundle_dynamic_section` fixture above (still installed).
RUNPATH_DIR_CLEAN="${ROOT}/runpath-dir-clean"
mkdir -p "$RUNPATH_DIR_CLEAN"
cp "$CLEAN_BINARY" "${RUNPATH_DIR_CLEAN}/fake-jammi-server-clean"
runpath_dir_clean_rc=0
bundle_assert_no_runpath_dir "$RUNPATH_DIR_CLEAN" >/dev/null 2>&1 || runpath_dir_clean_rc=$?
assert_eq "runpath-dir: bundle_assert_no_runpath_dir passes over a directory with no RPATH/RUNPATH file" "$runpath_dir_clean_rc" "0"

RUNPATH_DIR_DIRTY="${ROOT}/runpath-dir-dirty"
mkdir -p "$RUNPATH_DIR_DIRTY"
cp "$CLEAN_BINARY" "${RUNPATH_DIR_DIRTY}/fake-jammi-server-clean"
cp "$RUNPATH_BINARY" "${RUNPATH_DIR_DIRTY}/fake-jammi-server-runpath"
runpath_dir_dirty_out="$(bundle_assert_no_runpath_dir "$RUNPATH_DIR_DIRTY" 2>&1)"
runpath_dir_dirty_rc=$?
assert_eq "runpath-dir: bundle_assert_no_runpath_dir fails when ANY file in the directory carries RUNPATH" \
  "$runpath_dir_dirty_rc" "1"
assert_contains "runpath-dir: the directory-level failure names the offending file's own error" "$runpath_dir_dirty_out" "fake-jammi-server-runpath"

# Restore the real ELF reader for anything below (nothing does, today, but
# leaving a fixture installed past its own section is exactly the kind of
# state leak this suite's `install_fixture_needed` idiom exists to avoid).
bundle_dynamic_section() {
  readelf -d "$1" 2>&1
}

# ---------------------------------------------------------------------------
# 16. `bundle_jail_platform_basenames` (see #534), driven
#     directly: pure — the platform SUBSET of a `DT_NEEDED` list, nothing
#     bundle-able and nothing driver.
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086
jail_platform_out="$(bundle_jail_platform_basenames $BINARY_NEEDED)"
assert_contains "jail platform basenames includes a glibc member (libc)" "$jail_platform_out" "libc.so.6"
assert_contains "jail platform basenames includes the loader itself" "$jail_platform_out" "ld-linux-x86-64.so.2"
assert_not_contains "jail platform basenames excludes a bundle-able member (cudart)" "$jail_platform_out" "libcudart.so.12"
assert_not_contains "jail platform basenames excludes the driver (libcuda)" "$jail_platform_out" "libcuda.so.1"

# ---------------------------------------------------------------------------
# 17a. `bundle_jail_platform_closure` (the platform closure of the
#      binary AND every staged object), driven directly over the REAL
#      fixture tree section 7 above already built ($BINARY, the already-
#      populated $STAGE — `bundle_needed_sonames` is still the fixture
#      `install_fixture_needed` installed). Proof this reads EVERY staged
#      object's own needed-sonames, not only the binary's: libnvrtc.so.12
#      and libnccl.so.2 both (fixture-)report `libc.so.6` as their own
#      dependency, on top of the binary naming it directly — the closure
#      must still emit it exactly ONCE.
# ---------------------------------------------------------------------------
jail_closure_out="$(bundle_jail_platform_closure "$BINARY" "$STAGE")"
assert_contains "jail platform closure includes a glibc member (libc)" "$jail_closure_out" "libc.so.6"
assert_contains "jail platform closure includes the loader itself" "$jail_closure_out" "ld-linux-x86-64.so.2"
assert_not_contains "jail platform closure excludes a bundle-able member (cudart)" "$jail_closure_out" "libcudart.so.12"
assert_not_contains "jail platform closure excludes the driver (libcuda)" "$jail_closure_out" "libcuda.so.1"
jail_closure_libc_count="$(printf '%s\n' "$jail_closure_out" | grep -c '^libc\.so\.6$')"
assert_eq "jail platform closure deduplicates a member named by multiple objects" "$jail_closure_libc_count" "1"

# ---------------------------------------------------------------------------
# 17b. `bundle_jail_expected_relpaths`, driven directly over fixture LISTINGS
#     (text, no filesystem read) — the jail builder's file-set rule as a
#     PURE function. `interp_relpath` is a measured-shape example: the
#     loader's OWN `PT_INTERP` path, without its leading `/`.
# ---------------------------------------------------------------------------
JAIL_LIB_LISTING="libcudart.so.12
libcudart.so.12.6.77
libnccl.so.2"
JAIL_PLATFORM_BASENAMES="libc.so.6
ld-linux-x86-64.so.2"
jail_expected_out="$(bundle_jail_expected_relpaths "$JAIL_LIB_LISTING" "$JAIL_PLATFORM_BASENAMES" "lib64/ld-linux-x86-64.so.2")"
assert_contains "jail expected paths: the binary at the jail root" "$jail_expected_out" "jammi-server"
assert_contains "jail expected paths: the loader at its own PT_INTERP path" "$jail_expected_out" "lib64/ld-linux-x86-64.so.2"
assert_contains "jail expected paths: a staged lib_dir entry under lib/" "$jail_expected_out" "lib/libcudart.so.12"
assert_contains "jail expected paths: a staged lib_dir entry's versioned sibling under lib/" "$jail_expected_out" "lib/libcudart.so.12.6.77"
# Platform copies land under the SEPARATE platform/ directory, never lib/
# (the isolation property — no shared namespace with the hardlinked stage).
assert_contains "jail expected paths: a platform member under platform/" "$jail_expected_out" "platform/libc.so.6"
assert_not_contains "jail expected paths: a platform member is NEVER also placed under lib/" "$jail_expected_out" "lib/libc.so.6"
assert_not_contains "jail expected paths exclude the driver" "$jail_expected_out" "libcuda.so.1"
assert_not_contains "jail expected paths never carry the old fixed 'ld.so' name" "$jail_expected_out" "ld.so"

# ---------------------------------------------------------------------------
# 18. `bundle_assert_jail_file_set` (see #534, the builder's own
#     file-set rule checked over a REAL fixture tree — no ELF, pure
#     filesystem, same idiom as `bundle_assert_staged`). Mutation: a file
#     at a path the builder never wrote
#     (`usr/lib/libnccl.so.2`, standing in for a host copy leaking into the
#     jail at a path outside `lib/`) must FAIL, named.
# ---------------------------------------------------------------------------
JAIL_FIXTURE="${ROOT}/jail-fixture"
mkdir -p "${JAIL_FIXTURE}/lib" "${JAIL_FIXTURE}/lib64"
: >"${JAIL_FIXTURE}/jammi-server"
: >"${JAIL_FIXTURE}/lib64/ld-linux-x86-64.so.2"
: >"${JAIL_FIXTURE}/lib/libcudart.so.12"
: >"${JAIL_FIXTURE}/lib/libc.so.6"
JAIL_EXPECTED_LIST="jammi-server
lib64/ld-linux-x86-64.so.2
lib/libcudart.so.12
lib/libc.so.6"
# shellcheck disable=SC2086
jail_fileset_out="$(bundle_assert_jail_file_set "$JAIL_FIXTURE" $JAIL_EXPECTED_LIST 2>&1)"
jail_fileset_rc=$?
assert_eq "the jail file-set check passes on an exactly-matching tree" "$jail_fileset_rc" "0"

mkdir -p "${JAIL_FIXTURE}/usr/lib"
: >"${JAIL_FIXTURE}/usr/lib/libnccl.so.2"
# shellcheck disable=SC2086
jail_fileset_mut_out="$(bundle_assert_jail_file_set "$JAIL_FIXTURE" $JAIL_EXPECTED_LIST 2>&1)"
jail_fileset_mut_rc=$?
assert_eq "the jail file-set check fails when an unexpected host-shaped file leaks in (/usr/lib/libnccl.so.2)" \
  "$jail_fileset_mut_rc" "1"
assert_contains "the jail file-set failure names the leaked file" "$jail_fileset_mut_out" "usr/lib/libnccl.so.2"
rm -rf "${JAIL_FIXTURE:?}/usr"

rm -f "${JAIL_FIXTURE}/lib/libc.so.6"
# shellcheck disable=SC2086
jail_fileset_missing_out="$(bundle_assert_jail_file_set "$JAIL_FIXTURE" $JAIL_EXPECTED_LIST 2>&1)"
jail_fileset_missing_rc=$?
assert_eq "the jail file-set check fails when an expected file is missing" "$jail_fileset_missing_rc" "1"
assert_contains "the jail file-set failure names the missing file" "$jail_fileset_missing_out" "lib/libc.so.6"

# ---------------------------------------------------------------------------
# 18a. `bundle_build_jail` hardlinks the shipped stage into the jail's own
#      BUNDLE_JAIL_LIB_DIR with `cp -al`, sharing INODES. First, directly:
#      the sharing is proven REAL (a write through either path is a write
#      to the SAME file), which is exactly why `bundle_build_jail`'s own
#      code must never write there again after the copy. Second, end to
#      end, over `bundle_build_jail`'s own real code (`bundle_binary_
#      interp` stubbed the same way the other two ELF-reading functions
#      already are in this suite): the shipped stage's file is BYTE-
#      IDENTICAL, same inode, after the call as before — proof the
#      platform-copy step that follows writes only into
#      BUNDLE_JAIL_PLATFORM_DIR, never back into BUNDLE_JAIL_LIB_DIR.
# ---------------------------------------------------------------------------
bundle_inode() {
  python3 -c 'import os, sys; print(os.stat(sys.argv[1]).st_ino)' "$1"
}

CPAL_STAGE="${ROOT}/cpal-isolation/stage"
mkdir -p "$CPAL_STAGE"
printf 'original content\n' > "${CPAL_STAGE}/libfoo.so.1"
CPAL_JAIL="${ROOT}/cpal-isolation/jail"
mkdir -p "$CPAL_JAIL"
cp -al "$CPAL_STAGE" "${CPAL_JAIL}${BUNDLE_JAIL_LIB_DIR}"

cpal_stage_inode_before="$(bundle_inode "${CPAL_STAGE}/libfoo.so.1")"
cpal_jail_inode="$(bundle_inode "${CPAL_JAIL}${BUNDLE_JAIL_LIB_DIR}/libfoo.so.1")"
assert_eq "cp -al: the jail's lib/ file shares the SAME inode as the shipped stage's file" \
  "$cpal_jail_inode" "$cpal_stage_inode_before"

printf 'a write through the jail side\n' > "${CPAL_JAIL}${BUNDLE_JAIL_LIB_DIR}/libfoo.so.1"
cpal_stage_content_after="$(cat "${CPAL_STAGE}/libfoo.so.1")"
assert_eq "cp -al: a write through the jail's lib/ file DOES alter the shipped stage's own file — the sharing is real, not incidental" \
  "$cpal_stage_content_after" "a write through the jail side"

# `bundle_build_jail`'s own real code, run end to end.
REAL_JAIL_HOST_LOADER="${ROOT}/real-jail-isolation/host-lib64/fake-ld.so"
mkdir -p "$(dirname "$REAL_JAIL_HOST_LOADER")"
: >"$REAL_JAIL_HOST_LOADER"
bundle_binary_interp() {
  printf '%s\n' "$REAL_JAIL_HOST_LOADER"
}
REAL_JAIL_STAGE="${ROOT}/real-jail-isolation/stage/lib"
mkdir -p "$REAL_JAIL_STAGE"
printf 'bundle-able content\n' > "${REAL_JAIL_STAGE}/libbundleable.so.1"
REAL_JAIL_BINARY="${ROOT}/real-jail-isolation/fake-jammi-server"
: >"$REAL_JAIL_BINARY"
HOST_PLATFORM_DIR="${ROOT}/real-jail-isolation/host-platform"
mkdir -p "$HOST_PLATFORM_DIR"
: >"${HOST_PLATFORM_DIR}/libc.so.6"
real_jail_report="libc.so.6 => ${HOST_PLATFORM_DIR}/libc.so.6 (0x1)"
bundle_needed_sonames() {
  case "$(basename "$1")" in
    fake-jammi-server) printf '%s\n' "libbundleable.so.1" "libc.so.6" ;;
    *) : ;;
  esac
}
REAL_JAIL_DIR="${ROOT}/real-jail-isolation/jail"
real_jail_stage_inode_before="$(bundle_inode "${REAL_JAIL_STAGE}/libbundleable.so.1")"
bundle_build_jail "$REAL_JAIL_BINARY" "$REAL_JAIL_STAGE" "$REAL_JAIL_DIR" "$real_jail_report" >/dev/null
real_jail_build_rc=$?
install_fixture_needed
bundle_binary_interp() {
  readelf -l "$1" | sed -n 's/.*Requesting program interpreter: \(.*\)\]$/\1/p'
}
assert_eq "bundle_build_jail succeeds over the isolation fixture" "$real_jail_build_rc" "0"
real_jail_stage_inode_after="$(bundle_inode "${REAL_JAIL_STAGE}/libbundleable.so.1")"
real_jail_stage_content_after="$(cat "${REAL_JAIL_STAGE}/libbundleable.so.1")"
assert_eq "bundle_build_jail's own code never rewrites the shipped stage's file (same inode after the call)" \
  "$real_jail_stage_inode_after" "$real_jail_stage_inode_before"
assert_eq "bundle_build_jail's own code never rewrites the shipped stage's file (same content after the call)" \
  "$real_jail_stage_content_after" "bundle-able content"
# The platform member DID land, via the copy step, purely under
# BUNDLE_JAIL_PLATFORM_DIR, never touching BUNDLE_JAIL_LIB_DIR again.
assert_eq "bundle_build_jail stages the platform member under BUNDLE_JAIL_PLATFORM_DIR" \
  "$([ -f "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libc.so.6" ] && echo yes || echo no)" "yes"

# `bundle_assert_jail_class_provenance`, driven directly, over the SAME
# already-built (correct) jail: passes on its own real output.
provenance_ok_out="$(bundle_assert_jail_class_provenance "$REAL_JAIL_STAGE" "$REAL_JAIL_DIR" 2>&1)"
provenance_ok_rc=$?
assert_eq "bundle_assert_jail_class_provenance passes over a correctly-built jail" "$provenance_ok_rc" "0"

# Mutation: a PLATFORM member found under BUNDLE_JAIL_LIB_DIR (the
# bundle-able directory) is a named failure, by soname alone, regardless
# of its actual inode.
: >"${REAL_JAIL_DIR}${BUNDLE_JAIL_LIB_DIR}/libc.so.6"
provenance_platform_under_lib_out="$(bundle_assert_jail_class_provenance "$REAL_JAIL_STAGE" "$REAL_JAIL_DIR" 2>&1)"
provenance_platform_under_lib_rc=$?
assert_eq "bundle_assert_jail_class_provenance fails when a platform member is found under BUNDLE_JAIL_LIB_DIR" \
  "$provenance_platform_under_lib_rc" "1"
assert_contains "bundle_assert_jail_class_provenance names the platform-under-lib failure" \
  "$provenance_platform_under_lib_out" "a PLATFORM member found under"
rm -f "${REAL_JAIL_DIR}${BUNDLE_JAIL_LIB_DIR}/libc.so.6"

# Mutation: a BUNDLE-ABLE member found under BUNDLE_JAIL_PLATFORM_DIR (the
# platform directory) is a named failure, by soname alone.
: >"${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libbundleable.so.1"
provenance_bundleable_under_platform_out="$(bundle_assert_jail_class_provenance "$REAL_JAIL_STAGE" "$REAL_JAIL_DIR" 2>&1)"
provenance_bundleable_under_platform_rc=$?
assert_eq "bundle_assert_jail_class_provenance fails when a bundle-able member is found under BUNDLE_JAIL_PLATFORM_DIR" \
  "$provenance_bundleable_under_platform_rc" "1"
assert_contains "bundle_assert_jail_class_provenance names the bundle-able-under-platform failure" \
  "$provenance_bundleable_under_platform_out" "a BUNDLE-ABLE (non-platform) member found under"
rm -f "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libbundleable.so.1"

# Mutation: a platform-named file under BUNDLE_JAIL_PLATFORM_DIR whose
# inode happens to MATCH the staged tarball (accidentally hardlinked in
# bulk instead of copied individually from the host) is refused by
# PROVENANCE even though its name and directory otherwise agree.
rm -f "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libc.so.6"
ln "${REAL_JAIL_STAGE}/libbundleable.so.1" "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libc.so.6"
provenance_wrong_inode_out="$(bundle_assert_jail_class_provenance "$REAL_JAIL_STAGE" "$REAL_JAIL_DIR" 2>&1)"
provenance_wrong_inode_rc=$?
assert_eq "bundle_assert_jail_class_provenance fails when a 'platform' file under BUNDLE_JAIL_PLATFORM_DIR is actually hardlinked from the stage" \
  "$provenance_wrong_inode_rc" "1"
assert_contains "bundle_assert_jail_class_provenance names the wrong-provenance failure" \
  "$provenance_wrong_inode_out" "accidentally hardlinked from the stage"
rm -f "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libc.so.6"
cp -L "${HOST_PLATFORM_DIR}/libc.so.6" "${REAL_JAIL_DIR}${BUNDLE_JAIL_PLATFORM_DIR}/libc.so.6"

# ---------------------------------------------------------------------------
# 18b. `bundle_normalize_path`, driven
#      directly: LEXICAL `.`/`..` collapse, filesystem-free.
# ---------------------------------------------------------------------------
assert_eq "normalize: a '..'-composed path collapses to where it really points" \
  "$(bundle_normalize_path "/lib/../usr/lib/x.so")" "/usr/lib/x.so"
assert_eq "normalize: an already-clean path is unchanged" \
  "$(bundle_normalize_path "/lib/libc.so.6")" "/lib/libc.so.6"
assert_eq "normalize: a trailing slash is stripped" \
  "$(bundle_normalize_path "/lib/")" "/lib"
assert_eq "normalize: a relative (non-absolute) path stays relative" \
  "$(bundle_normalize_path "linux-vdso.so.1")" "linux-vdso.so.1"

# ---------------------------------------------------------------------------
# 19. `bundle_verify_jail_report` (see #534), driven directly
#     against literal captured-report TEXT — no `chroot`, no real loader,
#     anywhere in this suite. `JAIL_NEEDED` carries one bundle-able member,
#     one platform member, the driver, and the loader itself, so every
#     branch of the rule has a fixture. `JAIL_LOADER_PATH` is the
#     loader's own measured-shape `PT_INTERP` path, passed as this
#     function's first argument.
# ---------------------------------------------------------------------------
JAIL_LOADER_PATH="/lib64/ld-linux-x86-64.so.2"
JAIL_NEEDED="libcudart.so.12 libnccl.so.2 libc.so.6 libcuda.so.1 ld-linux-x86-64.so.2"

# 19a. A correct jail report: bundle-able members resolve under /lib,
#      PLATFORM members resolve under /platform (a
#      class-keyed expected directory, not a single literal), the driver is
#      not found, and the loader's own self-named (no `=>`) line resolves
#      to EXACTLY its PT_INTERP path (measured — at that path the self
#      line is the no-`=>` fixture shape and this rule passes).
jail_correct_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /lib/libnccl.so.2 (0x2)
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_correct_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_correct_report" $JAIL_NEEDED 2>&1)"
jail_correct_rc=$?
assert_eq "jail verify: a correct report (driver absent, loader at its own PT_INTERP path) passes" "$jail_correct_rc" "0"

# 19a2. The loader resolved from the
#       WRONG path (e.g. copied to an arbitrary name/location instead of
#       its own PT_INTERP path) is a NAMED failure, never tolerated.
#       Measured shape: when the loader is INVOKED somewhere other
#       than its own PT_INTERP path, its self line carries a `=>` (its true
#       soname mapped to wherever it was actually invoked from) — it is
#       ONLY at its own PT_INTERP path that the self line degenerates to
#       the bare, no-`=>` shape (19a above).
jail_wrong_loader_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /lib/libnccl.so.2 (0x2)
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => not found
ld-linux-x86-64.so.2 => /ld.so (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_wrong_loader_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_wrong_loader_report" $JAIL_NEEDED 2>&1)"
jail_wrong_loader_rc=$?
assert_eq "jail verify: the loader resolved from the WRONG path fails" "$jail_wrong_loader_rc" "1"
assert_contains "jail verify: the wrong-loader-path failure names the actual resolved path" "$jail_wrong_loader_out" "/ld.so"

# 19a3. The loader's own entry ABSENT from the report entirely is now a
#       named failure too — no exemption: a real trace always
#       carries the self line, so its absence is itself a defect signal.
jail_no_loader_line_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /lib/libnccl.so.2 (0x2)
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => not found"
# shellcheck disable=SC2086
jail_no_loader_line_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_no_loader_line_report" $JAIL_NEEDED 2>&1)"
jail_no_loader_line_rc=$?
assert_eq "jail verify: fails when the loader's own soname has no line at all" "$jail_no_loader_line_rc" "1"
assert_contains "jail verify: names the loader's own soname as absent" "$jail_no_loader_line_out" "ld-linux-x86-64.so.2"

# 19b. Bundle-able member `not found` inside the jail.
jail_bundleable_missing_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => not found
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_bundleable_missing_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_bundleable_missing_report" $JAIL_NEEDED 2>&1)"
jail_bundleable_missing_rc=$?
assert_eq "jail verify: a bundle-able member 'not found' fails" "$jail_bundleable_missing_rc" "1"
assert_contains "jail verify: the bundle-able-not-found failure names it" "$jail_bundleable_missing_out" "libnccl.so.2"

# 19c. Driver member RESOLVED inside the jail — proof the jail failed to
#      exclude it.
jail_driver_resolved_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /lib/libnccl.so.2 (0x2)
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => /lib/libcuda.so.1 (0x4)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_driver_resolved_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_driver_resolved_report" $JAIL_NEEDED 2>&1)"
jail_driver_resolved_rc=$?
assert_eq "jail verify: a driver member RESOLVED fails" "$jail_driver_resolved_rc" "1"
assert_contains "jail verify: the driver-resolved failure names it" "$jail_driver_resolved_out" "libcuda.so.1"

# 19d. A member resolved from OUTSIDE /lib (e.g. /usr/lib) — impossible in a
#      real bare jail, but the rule still refuses it rather than assuming
#      the impossibility.
jail_outside_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /usr/lib/libnccl.so.2 (0x2)
libc.so.6 => /platform/libc.so.6 (0x3)
libcuda.so.1 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_outside_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_outside_report" $JAIL_NEEDED 2>&1)"
jail_outside_rc=$?
assert_eq "jail verify: a member resolved from outside /lib fails" "$jail_outside_rc" "1"
assert_contains "jail verify: the outside-/lib failure names the wrong path" "$jail_outside_out" "/usr/lib/libnccl.so.2"

# 19d2. A PLATFORM member resolved from `/lib` — where `bundle_build_jail`
#       never stages a platform member, only `/platform` — must FAIL: the
#       rule is class-keyed in BOTH directions (a platform member accepted
#       under `/platform`, rejected under `/lib`), never merely widened to
#       accept a platform member from wherever it happens to resolve.
jail_platform_old_layout_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libnccl.so.2 => /lib/libnccl.so.2 (0x2)
libc.so.6 => /lib/libc.so.6 (0x3)
libcuda.so.1 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_platform_old_layout_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_platform_old_layout_report" $JAIL_NEEDED 2>&1)"
jail_platform_old_layout_rc=$?
assert_eq "jail verify: a platform member resolved from /lib (the bundle-able directory) fails" \
  "$jail_platform_old_layout_rc" "1"
assert_contains "jail verify: the wrong-directory failure names the platform member" "$jail_platform_old_layout_out" "libc.so.6"
assert_contains "jail verify: the wrong-directory failure names the expected /platform directory" "$jail_platform_old_layout_out" "not /platform"

# 19e. Vacuous pass closed — an empty report contains no resolved entry for
#      any required name and must fail, never pass silently.
jail_empty_report=""
# shellcheck disable=SC2086
jail_empty_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_empty_report" $JAIL_NEEDED 2>&1)"
jail_empty_rc=$?
assert_eq "jail verify: an empty report fails (vacuous pass closed)" "$jail_empty_rc" "1"

# ---------------------------------------------------------------------------
# 19f-h. The quantifier is EVERY LINE the report
#        carries, never only the names in `needed` — each fixture below
#        names a soname NOT present in `JAIL_NEEDED` at all (a transitive
#        member of a STAGED object the binary itself never names directly).
# ---------------------------------------------------------------------------
JAIL_NEEDED_NARROW="libcudart.so.12 libc.so.6 ld-linux-x86-64.so.2"

# 19f. A transitive platform member `not found` — a jail
#      missing libm.so.6 traces `libm.so.6 => not found`, and the binary
#      genuinely cannot run even though libm.so.6 is
#      never in the narrow `needed` list passed here.
jail_transitive_notfound_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
libm.so.6 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_transitive_notfound_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_transitive_notfound_report" $JAIL_NEEDED_NARROW 2>&1)"
jail_transitive_notfound_rc=$?
assert_eq "jail verify: a transitive member 'not found' fails even though it is absent from 'needed'" \
  "$jail_transitive_notfound_rc" "1"
assert_contains "jail verify: the transitive-not-found failure names it" "$jail_transitive_notfound_out" "libm.so.6"

# 19g. A transitive PLATFORM member resolved from OUTSIDE /platform.
#      libm.so.6 is itself a
#      platform member (glibc's math library), so its correct location is
#      `/platform`, never `/lib` — /usr/lib is wrong under EITHER layout.
jail_transitive_outside_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
libm.so.6 => /usr/lib/libm.so.6 (0x2)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_transitive_outside_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_transitive_outside_report" $JAIL_NEEDED_NARROW 2>&1)"
jail_transitive_outside_rc=$?
assert_eq "jail verify: a transitive platform member resolved outside /platform fails even though it is absent from 'needed'" \
  "$jail_transitive_outside_rc" "1"
assert_contains "jail verify: the transitive-outside-/platform failure names the wrong path" "$jail_transitive_outside_out" "/usr/lib/libm.so.6"

# 19h. A DRIVER member resolved from OUTSIDE the jail, transitively, never a
#      direct DT_NEEDED entry.
jail_transitive_driver_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
libnvidia-ptxjitcompiler.so.1 => /usr/lib64/libnvidia-ptxjitcompiler.so.1 (0x5)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_transitive_driver_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_transitive_driver_report" $JAIL_NEEDED_NARROW 2>&1)"
jail_transitive_driver_rc=$?
assert_eq "jail verify: a transitive DRIVER member resolved fails even though it is absent from 'needed'" \
  "$jail_transitive_driver_rc" "1"
assert_contains "jail verify: the transitive-driver-resolved failure names it" "$jail_transitive_driver_out" "libnvidia-ptxjitcompiler.so.1"

# 19i. A `..`-composed path that textually
#      starts with "/lib/" but NORMALIZES to somewhere else must still fail
#      — a bare prefix match (`case "$p" in "/lib"/*)`) would have passed
#      this, since the string literally begins with "/lib/". Mutates a
#      BUNDLE-ABLE member (libcudart.so.12, expected under /lib) so this
#      fixture stays about path normalization alone, independent of the
#      class-keyed directory logic.
jail_dotdot_report="libcudart.so.12 => /lib/../usr/lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_dotdot_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_dotdot_report" $JAIL_NEEDED_NARROW 2>&1)"
jail_dotdot_rc=$?
assert_eq "jail verify: a '..'-composed path that normalizes outside /lib fails" "$jail_dotdot_rc" "1"
assert_contains "jail verify: the normalized-path failure names the raw resolved path" "$jail_dotdot_out" "/lib/../usr/lib/libcudart.so.12"

# 19j. The `linux-vdso.so.1` carve-out: a no-`=>` line with a
#      RELATIVE, synthetic "path" must be tolerated, never judged as
#      "resolved outside /lib" (which a naive normalize-then-prefix-check
#      would otherwise flag, since a relative path never starts with "/").
jail_vdso_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
linux-vdso.so.1 (0x00007ffff7fc0000)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_vdso_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_vdso_report" $JAIL_NEEDED_NARROW 2>&1)"
jail_vdso_rc=$?
assert_eq "jail verify: the linux-vdso.so.1 carve-out tolerates its synthetic relative path" "$jail_vdso_rc" "0"

# 19k-m. The vDSO carve-out must be
#        CONDITIONED, never matched on the soname alone — three
#        spoof shapes, each otherwise a report that would pass cleanly.
JAIL_NEEDED_VDSO="libcudart.so.12 libc.so.6 ld-linux-x86-64.so.2"

# 19k. Spoofed vdso RESOLVED from a real host-shaped path.
jail_vdso_spoof_evil_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
linux-vdso.so.1 => /usr/lib/evil.so (0x4)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_vdso_spoof_evil_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_vdso_spoof_evil_report" $JAIL_NEEDED_VDSO 2>&1)"
jail_vdso_spoof_evil_rc=$?
assert_eq "jail verify: a spoofed 'linux-vdso.so.1 => /usr/lib/evil.so' fails" "$jail_vdso_spoof_evil_rc" "1"
assert_contains "jail verify: the spoofed-vdso-evil failure names the wrong path" "$jail_vdso_spoof_evil_out" "/usr/lib/evil.so"

# 19l. Spoofed vdso reported as `not found`.
jail_vdso_spoof_notfound_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
linux-vdso.so.1 => not found
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_vdso_spoof_notfound_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_vdso_spoof_notfound_report" $JAIL_NEEDED_VDSO 2>&1)"
jail_vdso_spoof_notfound_rc=$?
assert_eq "jail verify: a spoofed 'linux-vdso.so.1 => not found' fails" "$jail_vdso_spoof_notfound_rc" "1"
assert_contains "jail verify: the spoofed-vdso-not-found failure names it" "$jail_vdso_spoof_notfound_out" "linux-vdso.so.1"

# 19m. Spoofed vdso: a `=>`-shaped line whose path TEXT equals the bare
#      soname -- parses to the SAME (RESOLVED, "linux-vdso.so.1") pair the
#      genuine no-`=>` self-line produces; a real trace never emits this
#      shape (a real `=>` resolution always carries an absolute path).
jail_vdso_spoof_selfarrow_report="libcudart.so.12 => /lib/libcudart.so.12 (0x1)
libc.so.6 => /platform/libc.so.6 (0x3)
linux-vdso.so.1 => linux-vdso.so.1 (0x1)
	${JAIL_LOADER_PATH} (0x00007ffff7fc0000)"
# shellcheck disable=SC2086
jail_vdso_spoof_selfarrow_out="$(bundle_verify_jail_report "$JAIL_LOADER_PATH" "$jail_vdso_spoof_selfarrow_report" $JAIL_NEEDED_VDSO 2>&1)"
jail_vdso_spoof_selfarrow_rc=$?
assert_eq "jail verify: a spoofed 'linux-vdso.so.1 => linux-vdso.so.1' (self-text via a REAL arrow) fails" \
  "$jail_vdso_spoof_selfarrow_rc" "1"
assert_contains "jail verify: the spoofed-vdso-self-arrow failure names it" "$jail_vdso_spoof_selfarrow_out" "linux-vdso.so.1"

# ---------------------------------------------------------------------------
# 20. The real captured JAIL report (see this
#     file's module doc and `cu12_jail_report_real.txt`'s own header):
#     the fixture carries BOTH the report AND the SAME lane run's measured
#     `DT_NEEDED` set, delimited by a `# --- BINARY_NEEDED ---` line, so the
#     two halves of this oracle can never desync. Refuses to run at all
#     unless `BUNDLE_FIXTURE_PROVISIONAL=1` while the fixture is still the
#     CLEARLY-LABELLED `captured: pending` placeholder — the intended
#     effect: this suite, and therefore the merge path, stays red on this
#     file until the lead commits the real report (captured by
#     `release-binaries.yml`'s `server-cu12-build` job, uploaded as the
#     `cu12-jail-report` workflow artifact) as its own commit.
# ---------------------------------------------------------------------------
checks=$((checks + 1))
if [ ! -f "$REAL_JAIL_REPORT_FIXTURE" ]; then
  fail "the real jail-report fixture exists" "expected a file at ${REAL_JAIL_REPORT_FIXTURE}"
else
  ok "the real jail-report fixture exists"
  real_jail_head="$(head -n1 "$REAL_JAIL_REPORT_FIXTURE")"
  checks=$((checks + 1))
  case "$real_jail_head" in
    "# captured: pending"*)
      if [ "${BUNDLE_FIXTURE_PROVISIONAL:-0}" != "1" ]; then
        fail "the real jail-report fixture is not provisional" \
          "fixture is still 'captured: pending' -- set BUNDLE_FIXTURE_PROVISIONAL=1 to run this suite anyway (the merge path itself must NOT set it), or land the real captured report"
      else
        ok "the real jail-report fixture is provisional, and BUNDLE_FIXTURE_PROVISIONAL=1 is set"
      fi
      ;;
    *)
      ok "the real jail-report fixture is a real capture (no 'captured: pending' header)"
      real_jail_report_part="$(sed -n '1,/^# --- BINARY_NEEDED ---$/p' "$REAL_JAIL_REPORT_FIXTURE" | sed '$d')"
      real_jail_needed_part="$(sed -n '/^# --- BINARY_NEEDED ---$/,$p' "$REAL_JAIL_REPORT_FIXTURE" | tail -n +2)"
      checks=$((checks + 1))
      if [ -z "$real_jail_needed_part" ]; then
        fail "the real jail-report fixture carries its own committed DT_NEEDED section" \
          "expected a '# --- BINARY_NEEDED ---'-delimited section (the report and its DT_NEEDED set are captured together, same lane run)"
      else
        ok "the real jail-report fixture carries its own committed DT_NEEDED section"
        real_jail_parsed="$(printf '%s\n' "$real_jail_report_part" | bundle_parse_loader_report)"
        real_jail_loader_path=""
        while IFS=' ' read -r rj_soname rj_state rj_path; do
          [ -n "$rj_soname" ] || continue
          if [ "$rj_state" = "RESOLVED" ] && bundle_is_loader_soname "$rj_soname"; then
            real_jail_loader_path="$rj_path"
            break
          fi
        done <<EOF
$real_jail_parsed
EOF
        checks=$((checks + 1))
        if [ -z "$real_jail_loader_path" ]; then
          fail "the real jail report names a resolved loader (ld-linux-*) entry" \
            "no RESOLVED ld-linux-* line found in ${REAL_JAIL_REPORT_FIXTURE} -- cannot infer the loader's own PT_INTERP path"
        else
          ok "the real jail report names a resolved loader (ld-linux-*) entry"
          # shellcheck disable=SC2086
          real_jail_out="$(bundle_verify_jail_report "$real_jail_loader_path" "$real_jail_report_part" $real_jail_needed_part 2>&1)"
          real_jail_rc=$?
          assert_eq "the real captured jail report verifies against its own committed DT_NEEDED set" "$real_jail_rc" "0"
        fi
      fi
      ;;
  esac
fi


# ---------------------------------------------------------------------------
# 20b. `strip_inline_comment`'s own
#      correctness, self-tested directly in Python before it is ever
#      trusted against the real workflow file below — including a code
#      line that "unwires" a call by moving its name into a TRAILING
#      comment (`: skip   # bundle_assert_jail_file_set`), which must NOT
#      satisfy a presence check for that name.
# ---------------------------------------------------------------------------
checks=$((checks + 1))
b3_selftest_out="$(python3 - <<'PYEOF' 2>&1
def strip_inline_comment(line):
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def strip_comment_lines(text):
    return "\n".join(strip_inline_comment(line) for line in text.splitlines())


cases = [
    ("a whole-line comment disappears", "# bundle_assert_jail_file_set", ""),
    ("a bare code line is untouched", "bundle_assert_jail_file_set foo bar", "bundle_assert_jail_file_set foo bar"),
    (
        "an inline trailing comment is stripped, code kept",
        "bundle_build_jail a b  # calls bundle_assert_jail_file_set too",
        "bundle_build_jail a b  ",
    ),
    (
        "a name moved into a trailing comment "
        "on an unrelated code line reads as ABSENT",
        ": skip   # bundle_assert_jail_file_set",
        ": skip   ",
    ),
    (
        "a '#' inside single quotes is not a comment starter",
        "echo 'a # b' # real comment",
        "echo 'a # b' ",
    ),
    (
        "a '#' inside double quotes is not a comment starter",
        'echo "a # b" # real comment',
        'echo "a # b" ',
    ),
]
failures = []
for name, given, expected in cases:
    got = strip_inline_comment(given)
    if got != expected:
        failures.append(f"{name}: strip_inline_comment({given!r}) = {got!r}, expected {expected!r}")

# The mutation itself, end to end: a synthetic run_text where
# bundle_build_jail is called for real,
# bundle_assert_jail_file_set's call is REPLACED by a no-op with its name
# surviving only in a trailing comment, then jail_trace.py is called for real.
mutated_run_text = strip_comment_lines(
    "bundle_build_jail x y z w\n"
    ": skip   # bundle_assert_jail_file_set x y\n"
    "python3 ci/scripts/jail_trace.py a b c d\n"
)
if "bundle_assert_jail_file_set" in mutated_run_text:
    failures.append(
        "a name moved into a trailing comment still satisfies "
        "'bundle_assert_jail_file_set' in run_text"
    )

if failures:
    print("FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("ok")
PYEOF
)"
b3_selftest_rc=$?
if [ "$b3_selftest_rc" -eq 0 ]; then
  ok "strip_inline_comment strips shell-aware, keeps quoted '#', and a trailing-comment mutation reads as absent"
else
  fail "strip_inline_comment strips shell-aware, keeps quoted '#', and a trailing-comment mutation reads as absent" \
    "$b3_selftest_out"
fi

# ---------------------------------------------------------------------------
# 21. The release lane wires this script in, and the retired hand list
#     is gone from it. Parsed YAML (PyYAML), never a regex over the file
#     text: the property under test is "the Package step's shell script
#     calls bundle_cuda_libs.sh and never re-declares the seven-name list",
#     read off the parsed `run:` scalar of the exact step, not a grep that
#     would pass just as well on a comment mentioning the same words.
# ---------------------------------------------------------------------------
checks=$((checks + 1))
workflow_check_out="$(python3 - "$WORKFLOW" <<'PYEOF' 2>&1
import sys
import yaml

path = sys.argv[1]
with open(path) as f:
    doc = yaml.safe_load(f)

jobs = doc.get("jobs", {})
job = jobs.get("server-cu12-build")
if job is None:
    print("no 'server-cu12-build' job in release-binaries.yml")
    sys.exit(1)

steps = job.get("steps", [])


def strip_inline_comment(line):
    # A WHOLE-LINE-ONLY comment strip
    # (dropping a line only when its stripped content starts with `#`)
    # leaves an INLINE trailing comment on a real code line untouched — a
    # code line reading `: skip   # bundle_assert_jail_file_set` still
    # contains the literal substring `bundle_assert_jail_file_set`, so
    # every presence/ordering check below would read it as "the call is
    # here" even though the real call was DELETED and only its name
    # survives in a comment. Dropping from
    # the first UNQUOTED `#` to end of line, character by character,
    # tracking single/double-quote state — a `#` inside `'...'` or `"..."`
    # (e.g. a `#`-containing string literal some future step might pass to
    # a script) is never treated as a comment starter, matching what the
    # shell itself does. Deliberately NOT a full shell parser (no
    # backslash-escape handling, no `$'...'` ANSI-C quoting, no here-doc
    # awareness) — this is a presence/ordering CHECK over generated
    # workflow YAML, not a shell interpreter, and every `run:` block in
    # this repo's workflows sticks to plain `'...'`/`"..."` quoting.
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def strip_comment_lines(text):
    # The ORDERING assertions below (`bundle_
    # build_jail` before `bundle_assert_jail_file_set` before `jail_trace.
    # py`, `jail_idx < detect_idx`) are meaningless if a `#`-prefixed
    # COMMENT mentioning one of these names earlier in the script text
    # shifts its `str.find()` position ahead of the real call site — exactly
    # what this file's own module doc above already warns a presence-only
    # check ("not a grep that would pass just as well on a comment
    # mentioning the same words") must avoid; dropping every comment
    # (whole-line OR inline) before ANY check below makes both the
    # presence AND the ordering checks read the real code, never prose
    # describing it.
    return "\n".join(strip_inline_comment(line) for line in text.splitlines())


package_runs = [strip_comment_lines(s.get("run", "")) for s in steps if s.get("id") == "package"]
if not package_runs:
    print("no step with id: package under server-cu12-build")
    sys.exit(1)
run_text = "\n".join(package_runs)

if "bundle_cuda_libs.sh" not in run_text:
    print("the package step never calls bundle_cuda_libs.sh")
    sys.exit(1)

hand_list = "libcudart libcublas libcublasLt libcurand libnvrtc libnvrtc-builtins libnccl"
if hand_list in run_text:
    print("the retired seven-name hand list is still literally present in the package step")
    sys.exit(1)

# See #534: both arms present, arm 1a (detection) ordered
# strictly before arm 1b (the jail) — a future edit that drops the jail call
# or reorders it behind the tarball assembly (a silent fallback to
# detection-only) fails this check rather than only the hermetic function
# tests above, which never read the workflow at all.
detect_idx = run_text.find("bundle_verify_loader_resolution")
if detect_idx == -1:
    print("the package step never calls bundle_verify_loader_resolution (arm 1a, detection)")
    sys.exit(1)

jail_idx = run_text.find("jail_trace.py")
if jail_idx == -1:
    print("the package step never invokes ci/scripts/jail_trace.py (arm 1b, the jail)")
    sys.exit(1)
build_idx = run_text.find("bundle_build_jail")
if build_idx == -1:
    print("the package step never calls bundle_build_jail (arm 1b, the jail)")
    sys.exit(1)
if "bundle_verify_jail_report" not in run_text:
    print("the package step never calls bundle_verify_jail_report (arm 1b, the jail)")
    sys.exit(1)
if jail_idx < detect_idx:
    print("the jail arm (jail_trace.py) appears before arm 1a's detection call — arm 1a must run FIRST")
    sys.exit(1)

# The jail builder's own file-set assert must be
# WIRED into the lane, between the build call and the trace call — an
# assertion that exists in bundle_cuda_libs.sh but is never called from the
# workflow enforces nothing on a real release.
assert_idx = run_text.find("bundle_assert_jail_file_set")
if assert_idx == -1:
    print("the package step never calls bundle_assert_jail_file_set (the jail builder's own file-set rule)")
    sys.exit(1)
if not (build_idx < assert_idx < jail_idx):
    print(
        "bundle_assert_jail_file_set must run strictly between bundle_build_jail and the "
        "jail_trace.py call, not before the jail exists or after it has already been traced"
    )
    sys.exit(1)

print("ok")
sys.exit(0)
PYEOF
)"
workflow_check_rc=$?
if [ "$workflow_check_rc" -eq 0 ]; then
  ok "release-binaries.yml's server-cu12-build package step calls bundle_cuda_libs.sh, drops the hand list, and runs arm 1a (detection) before arm 1b (the jail)"
else
  fail "release-binaries.yml's server-cu12-build package step calls bundle_cuda_libs.sh, drops the hand list, and runs arm 1a (detection) before arm 1b (the jail)" \
    "$workflow_check_out"
fi

# ---------------------------------------------------------------------------
if [ "$failures" -ne 0 ]; then
  echo "test_bundle_cuda_libs.sh: ${failures} of ${checks} check(s) FAILED" >&2
  exit 1
fi
echo "test_bundle_cuda_libs.sh: all ${checks} check(s) passed."
