#!/usr/bin/env bash
# Hermetic suite for `ci/scripts/bundle_cuda_libs.sh` — the derivation that
# decides which shared libraries the CUDA (`cu12`) release tarball carries.
#
# Why a suite at all: the only caller of that script is
# `.github/workflows/release-binaries.yml`'s `server-cu12-build` job, whose
# promote leg runs on a `v*` TAG. Nothing on the merge path executes the
# derivation, so without this file its first real exercise would be a release
# — the shape that let the retired hand-written soname list ship a binary with
# an unsatisfiable `DT_NEEDED libnccl.so.2` in the first place.
#
# Hermetic in the strict sense: no `readelf`, no `ldd`, no ELF file, no
# network, no CUDA install — so it runs identically on the Linux `Guard`
# runner and on a maintainer's macOS box. Two seams make that possible, and
# both keep the shipped code in the loop rather than re-implementing it: the
# suite `source`s the actual script and replaces `bundle_needed_sonames` (its
# ONE ELF-reading function) with a fixture map, and it drives
# `bundle_unresolved_from_loader_output` (the loader-report rule, split out of
# `bundle_verify_stage` for exactly this) over fixture `ldd` text. Everything
# between those two seams — resolution, ordering, the host-provided partition,
# the transitive closure, the refusal, the staging copies — is the real code
# path, entered through `bundle_main`, the same entry point the workflow calls.
#
# The fixture link set is MEASURED, not invented: it is the `DT_NEEDED` list
# `packaging/server-cu12/verify_link_set.py`'s own suite pins for the shipped
# `jammi-server` binary (read off the `server-cu12-binary` artifact of run
# 34717957779), plus `libnccl.so.2`, which is what `candle-core/nccl` in
# `jammi-ai`'s `cuda` feature adds. One fixture edge is DERIVED rather than
# measured and is marked as such where it is declared: `libnvrtc.so.12` ->
# `libnvrtc-builtins`.
#
# Run: `bash ci/scripts/test_bundle_cuda_libs.sh`
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/bundle_cuda_libs.sh"

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
# `libnvrtc.so.12 -> libnvrtc-builtins.so.12.6` is the one DERIVED edge here,
# and it is stated as derived rather than measured: the binary's own measured
# `DT_NEEDED` list above does NOT name `libnvrtc-builtins`, yet the hand list
# this unit retires bundled it, which leaves a transitive edge through
# `libnvrtc` as the explanation. This suite pins the CONSEQUENCE — a
# transitively needed library is staged — rather than the edge's exact owner;
# on real bytes, the release lane's independent loader check is the arm that
# would object if the edge sat elsewhere.
install_fixture_needed() {
  bundle_needed_sonames() {
    case "$(basename "$1")" in
      fake-jammi-server) printf '%s\n' "$BINARY_NEEDED" ;;
      libnvrtc.so.12) printf '%s\n' "libnvrtc-builtins.so.12.6" "libc.so.6" ;;
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
assert_contains "nccl versioned object staged from /usr/lib64" "$sources" "${SYSLIB}/libnccl.so.2.23.4
"

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
# 3. Every name the retired hand list carried is still carried — the
#    derivation replaces that list, it does not shrink it. `libnvrtc-builtins`
#    included, which the binary itself never names: the closure is transitive,
#    and a direct-only walk would drop it.
# ---------------------------------------------------------------------------
for stem in libcudart libcublas libcublasLt libcurand libnvrtc libnvrtc-builtins; do
  assert_contains "hand-list member ${stem} still staged" "$sources" "${TOOLKIT}/${stem}.so."
done

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
# 6. `bundle_main` end to end — the entry point the workflow calls. Its loader
#    arm is stubbed here and only here: verifying a fake ELF with the real
#    `ldd` is not a thing a hermetic suite can do, and that arm's own rule is
#    driven directly in step 7 below.
# ---------------------------------------------------------------------------
STAGE="${ROOT}/stage/lib"
bundle_verify_stage() {
  echo "stub bundle_verify_stage($1, $2)"
}
main_out="$(bundle_main "$BINARY" "$STAGE" "$SEARCH" 2>&1)"
main_rc=$?
assert_eq "bundle_main succeeds over the fixture tree" "$main_rc" "0"
assert_contains "bundle_main names what it stages" "$main_out" "staging ${SYSLIB}/libnccl.so.2"
staged="$(LC_ALL=C ls "$STAGE" | LC_ALL=C sort | tr '\n' ' ')"
assert_eq "the staged tree is exactly the derived set" "$staged" \
  "libcublas.so.12 libcublas.so.12.6.4.1 libcublasLt.so.12 libcublasLt.so.12.6.4.1 libcudart.so.12 libcudart.so.12.6.77 libcurand.so.10 libcurand.so.10.3.7.77 libnccl.so.2 libnccl.so.2.23.4 libnvrtc-builtins.so.12.6 libnvrtc-builtins.so.12.6.85 libnvrtc.so.12 libnvrtc.so.12.6.85 "

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
# 7. The second, independent arm: the loader-report rule. It checks not only
#    THAT a bundled soname resolved, but WHERE: `LD_LIBRARY_PATH` PREPENDS to
#    the loader's search, it does not RESTRICT it, so a soname the tarball
#    never staged can still come back resolved if the build host happens to
#    carry it too — a system `libnccl` at `/usr/lib64`, say — and a rule that
#    only greps for `not found` calls that report clean. `$STAGE` already
#    holds the real files `bundle_main` staged in step 6, and `$SYSLIB`
#    already exists from the setup fixture, so these reports point at real
#    directories rather than invented strings, which is what makes the
#    symlink case below meaningful.
# ---------------------------------------------------------------------------
clean_report="	linux-vdso.so.1 (0x00007ffd8c9f2000)
	libcudart.so.12 => ${STAGE}/libcudart.so.12 (0x00007f1c00000000)
	libnccl.so.2 => ${STAGE}/libnccl.so.2 (0x00007f1bf0000000)
	libcuda.so.1 => not found
	libnvidia-ptxjitcompiler.so.1 => not found
	libc.so.6 => ${SYSLIB}/libc.so.6 (0x00007f1be0000000)"
assert_eq "a driver 'not found' and a platform soname resolved outside the stage dir both stay clean" \
  "$(bundle_unresolved_from_loader_output "$clean_report" "$STAGE")" ""

# This unit's own regression: `libnccl.so.2` resolved from `/usr/lib64` — the
# builder's own system copy — rather than the stage dir. The retired rule
# called this clean, because it names no 'not found' line at all; RED at
# f74943b5 (see the report's `red_observed`): the same fixture text, run
# against the pre-fix `bundle_unresolved_from_loader_output`, returns empty.
nccl_outside_report="	linux-vdso.so.1 (0x00007ffd8c9f2000)
	libcudart.so.12 => ${STAGE}/libcudart.so.12 (0x00007f1c00000000)
	libnccl.so.2 => ${SYSLIB}/libnccl.so.2 (0x00007f1bf0000000)
	libcuda.so.1 => not found
	libc.so.6 => ${SYSLIB}/libc.so.6 (0x00007f1be0000000)"
assert_contains "a bundled soname resolved from outside the stage dir is a defect" \
  "$(bundle_unresolved_from_loader_output "$nccl_outside_report" "$STAGE")" \
  "libnccl.so.2 => ${SYSLIB}/libnccl.so.2 (resolved outside ${STAGE})"

# Every bundled soname resolved UNDER the stage dir: clean.
all_under_report="	libcudart.so.12 => ${STAGE}/libcudart.so.12 (0x1)
	libnccl.so.2 => ${STAGE}/libnccl.so.2 (0x2)
	libnvrtc-builtins.so.12.6 => ${STAGE}/libnvrtc-builtins.so.12.6 (0x3)
	libcuda.so.1 => not found
	libc.so.6 => ${SYSLIB}/libc.so.6 (0x4)"
assert_eq "every bundled soname resolved under the stage dir passes" \
  "$(bundle_unresolved_from_loader_output "$all_under_report" "$STAGE")" ""

dirty_report="${all_under_report}
	libcusparse.so.12 => not found"
assert_contains "an unresolved bundled library is a defect" \
  "$(bundle_unresolved_from_loader_output "$dirty_report" "$STAGE")" "libcusparse.so.12 => not found"

# A SYMLINKED stage dir: the workflow's own runner can put its temp root
# behind a symlink (macOS's `/tmp` -> `/private/tmp` is exactly this shape;
# a Linux CI runner's `$RUNNER_TEMP` is not guaranteed not to be one either),
# so the loader's report and the `lib_dir` argument this script was called
# with can name the SAME directory by two different literal strings.
# `realpath`-normalising the comparison (`bundle_realpath_dir`), not the
# literal strings, is what makes this equal; a literal-string comparison
# would wrongly flag every entry as "resolved outside".
STAGE_LINK="${ROOT}/stage/lib-link"
ln -s "$STAGE" "$STAGE_LINK"
symlink_report="	libcudart.so.12 => ${STAGE_LINK}/libcudart.so.12 (0x5)"
assert_eq "a symlinked stage dir still normalises to a pass" \
  "$(bundle_unresolved_from_loader_output "$symlink_report" "$STAGE")" ""

# A soname staged AS a symlink whose OWN target escapes `$lib_dir` — distinct
# from the case above, where the STAGE DIR itself is reached through a
# symlink but the FILE it names is a real object underneath. Here the
# directory component normalises inside `$lib_dir` just fine (`$lib_dir`
# itself is not a symlink); only resolving the FILE's own chain
# (`bundle_realpath_file`) exposes that the object it ultimately names lives
# elsewhere. RED at c9d20550: `bundle_unresolved_from_loader_output`
# `realpath`-normalised only `dirname "$resolved_path"`, so this case read as
# clean.
OUTSIDE="${ROOT}/outside"
mkdir -p "$OUTSIDE"
: >"${OUTSIDE}/libnccl.so.2"
ESCAPE_STAGE="${ROOT}/stage/lib-escape"
mkdir -p "$ESCAPE_STAGE"
: >"${ESCAPE_STAGE}/libcudart.so.12"
ln -s "${OUTSIDE}/libnccl.so.2" "${ESCAPE_STAGE}/libnccl.so.2"
escape_report="	libcudart.so.12 => ${ESCAPE_STAGE}/libcudart.so.12 (0x6)
	libnccl.so.2 => ${ESCAPE_STAGE}/libnccl.so.2 (0x7)"
assert_contains "a soname symlinked to a path outside lib_dir is a defect" \
  "$(bundle_unresolved_from_loader_output "$escape_report" "$ESCAPE_STAGE")" \
  "libnccl.so.2 => ${ESCAPE_STAGE}/libnccl.so.2 (resolved outside ${ESCAPE_STAGE})"

# Contrast: a symlink INSIDE the stage dir pointing to another object also
# INSIDE the stage dir stays clean — the property is about where the file
# ultimately resolves, not whether it is a symlink at all.
: >"${ESCAPE_STAGE}/libnccl.so.2.23.4"
ln -s "${ESCAPE_STAGE}/libnccl.so.2.23.4" "${ESCAPE_STAGE}/libnccl.so.2.inside"
inside_report="	libcudart.so.12 => ${ESCAPE_STAGE}/libcudart.so.12 (0x8)
	libnccl.so.2 => ${ESCAPE_STAGE}/libnccl.so.2.inside (0x9)"
assert_eq "a symlink inside the stage dir pointing inside it stays clean" \
  "$(bundle_unresolved_from_loader_output "$inside_report" "$ESCAPE_STAGE")" ""

# ---------------------------------------------------------------------------
# 8. The set cross-check: the report must name every soname
#    `bundle_needed_sonames` derived for the binary, not merely resolve the
#    ones it happens to mention. Without the third (derived-set) argument this
#    function only ever inspects lines the report actually contains, which is
#    exactly the shape that calls a VACUOUS report clean — none of these
#    three carry a single bad line, because none of them says anything about
#    a staged dependency at all. RED at c9d20550 (2-arg calls, no derived set
#    to compare against — this IS the vacuous-pass bug the fix closes):
# ---------------------------------------------------------------------------
for empty_shape in "" "not a dynamic executable" "	linux-vdso.so.1 (0x00007ffd8c9f2000)"; do
  checks=$((checks + 1))
  got="$(bundle_unresolved_from_loader_output "$empty_shape" "$STAGE" "libcudart.so.12")"
  case "$got" in
    *"libcudart.so.12"*) ok "a vacuous report ('${empty_shape}') is caught once a derived set is given" ;;
    *) fail "a vacuous report ('${empty_shape}') is caught once a derived set is given" "expected 'libcudart.so.12' to be named missing, got: ${got}" ;;
  esac
done

# A report naming SOME but not all of the derived set: the one it drops is
# named, the one it has is not re-flagged.
derived_two="libcudart.so.12
libnvrtc.so.12"
partial_report="	libcudart.so.12 => ${STAGE}/libcudart.so.12 (0x1)"
assert_contains "a derived soname the report never mentions is named missing" \
  "$(bundle_unresolved_from_loader_output "$partial_report" "$STAGE" "$derived_two")" \
  "libnvrtc.so.12 => missing from loader report"
assert_not_contains "a derived soname the report DOES mention is not flagged missing" \
  "$(bundle_unresolved_from_loader_output "$partial_report" "$STAGE" "$derived_two")" \
  "libcudart.so.12 => missing"

# A report naming every derived soname (plus entries the derived set does not
# mention, e.g. the driver) still passes.
assert_eq "a report naming every derived soname passes the cross-check" \
  "$(bundle_unresolved_from_loader_output "$all_under_report" "$STAGE" "libcudart.so.12
libnccl.so.2
libnvrtc-builtins.so.12.6")" ""

# ---------------------------------------------------------------------------
# 9. `bundle_verify_stage` itself, driven directly (not stubbed) via a fake
#    `ldd` shell function — a function shadows the real command in PATH
#    lookup, which is what makes this hermetic on a host with no loader
#    worth asking and no real ELF to ask it about. Step 6 above replaced
#    `bundle_verify_stage` with its own stub (to drive `bundle_main` without
#    a real loader); re-source the script to get the REAL function back
#    before testing it directly.
# ---------------------------------------------------------------------------
. "$SCRIPT"
set +e
VERIFY_BINARY="${ROOT}/fake-verify-binary"
: >"$VERIFY_BINARY"
VERIFY_STAGE="${ROOT}/stage/lib-verify"
mkdir -p "$VERIFY_STAGE"
: >"${VERIFY_STAGE}/libcudart.so.12"

bundle_needed_sonames() {
  case "$(basename "$1")" in
    fake-verify-binary) printf '%s\n' "libcudart.so.12" ;;
    *) : ;;
  esac
}

# 9a. `ldd` itself exits non-zero. RED at c9d20550: the `|| true` on the
#     command substitution discarded this exit status entirely, so the
#     (empty) output was read as a clean report.
ldd() {
  echo "ldd: simulated non-zero exit" >&2
  return 1
}
verify_out="$(bundle_verify_stage "$VERIFY_BINARY" "$VERIFY_STAGE" 2>&1)"
verify_rc=$?
assert_eq "a non-zero ldd exit fails bundle_verify_stage" "$verify_rc" "1"
assert_contains "a non-zero ldd exit names the tool" "$verify_out" "ldd exited"
unset -f ldd

# 9b. `ldd` exits 0 but the report names none of the binary's own derived
#     sonames (a vdso-only report) — the vacuous-pass shape a bare `not
#     found` grep cannot see, because there is no `not found` line to find.
#     RED at c9d20550: no derived set was ever passed to
#     `bundle_unresolved_from_loader_output`.
ldd() {
  printf '\tlinux-vdso.so.1 (0x00007ffd8c9f2000)\n'
}
verify_out="$(bundle_verify_stage "$VERIFY_BINARY" "$VERIFY_STAGE" 2>&1)"
verify_rc=$?
assert_eq "a vacuous-but-zero-exit ldd report fails bundle_verify_stage" "$verify_rc" "1"
assert_contains "the missing derived soname is named" "$verify_out" "libcudart.so.12 => missing from loader report"
unset -f ldd

# 9c. `ldd` exits 0 and names the derived soname resolved under the stage
#     dir: clean, end to end.
ldd() {
  printf '\tlibcudart.so.12 => %s/libcudart.so.12 (0x1)\n' "$VERIFY_STAGE"
}
verify_out="$(bundle_verify_stage "$VERIFY_BINARY" "$VERIFY_STAGE" 2>&1)"
verify_rc=$?
assert_eq "a complete, correctly-resolved ldd report passes bundle_verify_stage" "$verify_rc" "0"
unset -f ldd
install_fixture_needed

# ---------------------------------------------------------------------------
if [ "$failures" -ne 0 ]; then
  echo "test_bundle_cuda_libs.sh: ${failures} of ${checks} check(s) FAILED" >&2
  exit 1
fi
echo "test_bundle_cuda_libs.sh: all ${checks} check(s) passed."
