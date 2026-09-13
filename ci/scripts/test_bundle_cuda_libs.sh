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
# runner and on a maintainer's macOS box. One seam makes that possible and
# keeps the shipped code in the loop rather than re-implementing it: the
# suite `source`s the actual script and replaces `bundle_needed_sonames` (its
# ONE ELF-reading function) with a fixture map. Everything else — resolution,
# ordering, the host-provided partition, the transitive closure, the refusal,
# the staging copies — is the real code path, entered through `bundle_main`,
# the same entry point the workflow calls.
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
    libnvrtc.so.12) printf '%s\n' "libnvrtc-builtins.so.12.6" "libc.so.6" ;;
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
install_fixture_needed

# ---------------------------------------------------------------------------
if [ "$failures" -ne 0 ]; then
  echo "test_bundle_cuda_libs.sh: ${failures} of ${checks} check(s) FAILED" >&2
  exit 1
fi
echo "test_bundle_cuda_libs.sh: all ${checks} check(s) passed."
