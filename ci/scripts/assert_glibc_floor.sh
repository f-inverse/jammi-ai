#!/usr/bin/env bash
# Asserts a Linux ELF binary's highest linked GLIBC symbol version does not
# exceed a floor (the manylinux_2_28 baseline every Linux release artifact
# of this workspace -- the CLI tarball, the server tarball, the server
# wheel, and the native engine wheel -- is built to run on). Building INSIDE
# the manylinux_2_28 CI container is necessary but not sufficient: a
# bare-runner fallback, a toolchain bump inside the container, or a future
# base-image change could silently raise the linked floor without anything
# else noticing. This is the SUFFICIENT check -- it reads the actual
# DT_VERNEED / dynamic-symbol versions the binary carries.
#
# The `GLIBC_x.y` versions a binary references are read off `objdump -T`
# (falling back to `readelf --dyn-syms`, whichever is on PATH first -- both
# ship in the manylinux_2_28 CI image's binutils) and compared with
# `sort -V` -- a LEXICAL sort of version strings returns "2.9" as "greater"
# than "2.34" (wrong: 2.9 < 2.34), which is exactly the kind of miss this
# assert exists to avoid.
#
# Usage:
#   assert_glibc_floor.sh <binary> [max=2.28]
#     (exits 0 if every referenced GLIBC_x.y version is <= max, 1 otherwise)
#   assert_glibc_floor.sh --self-test
#     (drives the comparison logic against synthetic version lists, no ELF
#     tooling, no network; exits 0 iff every fixture's exit code matches
#     what it should be)
set -euo pipefail

# Extracts the GLIBC_x.y tokens a binary references, one per line,
# de-duplicated. Reads from stdout of `objdump -T`/`readelf --dyn-syms`
# (whichever the caller piped in) via stdin -- kept separate from the ELF
# tool invocation itself so --self-test can drive it with a synthetic
# fixture, no ELF tooling required.
_extract_versions() {
  grep -o 'GLIBC_[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?' | sed 's/^GLIBC_//' | sort -uV
}

# The real comparison: given a newline-separated list of GLIBC_x.y version
# strings (already extracted, no "GLIBC_" prefix) on stdin and a floor as
# $1, returns (not exits, so --self-test can call this without killing its
# own shell under `set -e`) 0 if every version is <= floor, 1 otherwise.
_assert_versions() {
  local floor="$1"
  local versions
  versions="$(cat)"

  if [ -z "$versions" ]; then
    echo "assert-glibc-floor: no GLIBC_x.y symbol versions found -- either a static binary (nothing to assert) or the ELF tool produced no output" >&2
    return 0
  fi

  local highest
  highest="$(printf '%s\n' "$versions" | sort -V | tail -1)"

  # `sort -V` between {highest, floor}: if their sorted-highest is NOT the
  # floor itself, `highest` sorts ABOVE `floor` -- i.e. highest > floor.
  local winner
  winner="$(printf '%s\n%s\n' "$highest" "$floor" | sort -V | tail -1)"
  if [ "$winner" != "$floor" ]; then
    echo "assert-glibc-floor: highest linked GLIBC version is $highest, exceeds the floor $floor" >&2
    return 1
  fi

  echo "assert-glibc-floor: OK -- highest linked GLIBC version $highest <= floor $floor"
  return 0
}

_self_test() {
  local failures=0
  local rc

  rc=0
  printf '2.17\n2.25\n2.28\n' | _assert_versions "2.28" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[at-floor]: OK (exit 0, expected 0)"
  else
    echo "self-test[at-floor]: FAIL (exit $rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  # The exact lexical-sort miss this script exists to avoid: "2.9" as a
  # STRING sorts after "2.34" lexically, but 2.9 < 2.34 numerically -- a
  # naive `sort` (not `sort -V`) would call this a floor VIOLATION when it
  # is not. `sort -V` must call it fine.
  rc=0
  printf '2.9\n2.17\n' | _assert_versions "2.28" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[lexical-sort-trap-under-floor]: OK (exit 0, expected 0)"
  else
    echo "self-test[lexical-sort-trap-under-floor]: FAIL (exit $rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  rc=0
  printf '2.17\n2.34\n' | _assert_versions "2.28" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[above-floor]: OK (exit 1, expected 1)"
  else
    echo "self-test[above-floor]: FAIL (exit $rc, expected 1)" >&2
    failures=$((failures + 1))
  fi

  rc=0
  printf '' | _assert_versions "2.28" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[no-versions-found]: OK (exit 0, expected 0)"
  else
    echo "self-test[no-versions-found]: FAIL (exit $rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  # `_extract_versions` itself: a synthetic `objdump -T`-shaped line carrying
  # a GLIBC_x.y.z (three-component, seen on some symbols) must parse.
  rc=0
  extracted="$(printf '0000000000000000      DF *UND*  0000000000000000  GLIBC_2.2.5 memcpy\n' | _extract_versions)"
  if [ "$extracted" = "2.2.5" ]; then
    echo "self-test[extract-three-component]: OK"
  else
    echo "self-test[extract-three-component]: FAIL (got '$extracted', expected '2.2.5')" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "assert-glibc-floor --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "assert-glibc-floor --self-test: all 5 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

binary="${1:?usage: assert_glibc_floor.sh <binary> [max=2.28]}"
floor="${2:-2.28}"

if [ ! -f "$binary" ]; then
  echo "::error::assert_glibc_floor.sh: no such file: $binary" >&2
  exit 1
fi

if command -v objdump >/dev/null 2>&1; then
  versions="$(objdump -T "$binary" 2>/dev/null | _extract_versions)"
elif command -v readelf >/dev/null 2>&1; then
  versions="$(readelf --dyn-syms --wide "$binary" 2>/dev/null | _extract_versions)"
else
  echo "::error::assert_glibc_floor.sh: neither objdump nor readelf is on PATH -- cannot read the binary's linked symbol versions" >&2
  exit 1
fi

printf '%s\n' "$versions" | _assert_versions "$floor"
