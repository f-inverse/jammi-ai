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
# The three-way lattice (explicit, not collapsed): (1) genuinely static --
# `readelf -d` reports no dynamic section -- passes with a printed notice,
# checked FIRST and independently of the dump tool below; (2) dynamically
# linked but the dump tool itself exits non-zero -- a real failure, never
# treated as "static" or as "nothing to assert"; (3) dynamically linked,
# dump tool succeeded but extracted zero GLIBC_x.y tokens -- also a real
# failure (unexplained, not a pass). Only a dynamically linked binary with
# at least one extracted token reaches the actual floor compare.
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
# fixture, no ELF tooling required. `grep -o` exits 1 on no match, which
# under `pipefail` would otherwise kill this pipeline (and, at the top
# level under `set -e`, the whole script) BEFORE the caller ever gets a
# chance to decide whether zero matches means "static binary" or "real
# failure" -- so the trailing `|| true` here is deliberate: this function
# itself never fails, only reports what it found (possibly nothing).
_extract_versions() {
  grep -o 'GLIBC_[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?' | sed 's/^GLIBC_//' | sort -uV || true
}

# Returns (not exits) 0 if stdin (a `readelf -d` dump) reports the binary
# as carrying no dynamic section -- the genuine "statically linked, nothing
# to assert" case -- 1 otherwise. Kept separate from the caller so
# --self-test can drive it with synthetic `readelf -d` output, no ELF
# tooling required. The exact GNU binutils wording is "There is no dynamic
# section in this file." -- matched case-insensitively and loosely (`no
# dynamic section`) so a wording tweak across binutils versions still
# matches.
_is_static_elf_dump() {
  grep -qi 'no dynamic section'
}

# The real comparison: given a newline-separated list of GLIBC_x.y version
# strings (already extracted, no "GLIBC_" prefix) on stdin, a floor as $1,
# and (optional) the tool that produced those versions as $2, returns (not
# exits, so --self-test can call this without killing its own shell under
# `set -e`) 0 if every version is <= floor, 1 otherwise. `$2` defaults to a
# generic label when omitted (every --self-test fixture below calls this
# with a floor only, driving the comparison logic with no real tool in the
# loop) -- the real invocation at the bottom of this file always passes its
# own `$tool_desc`, so an operator reading a pass/fail line always sees
# which of objdump/readelf actually produced the versions being judged,
# never a message that is silent about which tool ran.
#
# Empty input is a FAILURE here, not a pass: the caller only reaches this
# function after `_is_static_elf_dump` has already ruled out "genuinely
# static" (that case returns 0 before this is ever called) -- so empty
# input at this point means the binary is dynamically linked but the ELF
# tool found no GLIBC_x.y tokens at all, which this cannot assert a floor
# against and must not silently wave through.
_assert_versions() {
  local floor="$1"
  local tool_desc="${2:-the symbol-version tool}"
  local versions
  versions="$(cat)"

  if [ -z "$versions" ]; then
    echo "assert-glibc-floor: ${tool_desc} extracted no GLIBC_x.y symbol versions from a dynamically-linked binary -- cannot assert the floor (the static-binary case is ruled out separately, by 'readelf -d', before this is called)" >&2
    return 1
  fi

  local highest
  highest="$(printf '%s\n' "$versions" | sort -V | tail -1)"

  # `sort -V` between {highest, floor}: if their sorted-highest is NOT the
  # floor itself, `highest` sorts ABOVE `floor` -- i.e. highest > floor.
  local winner
  winner="$(printf '%s\n%s\n' "$highest" "$floor" | sort -V | tail -1)"
  if [ "$winner" != "$floor" ]; then
    echo "assert-glibc-floor: highest linked GLIBC version is $highest (per ${tool_desc}), exceeds the floor $floor" >&2
    return 1
  fi

  echo "assert-glibc-floor: OK -- highest linked GLIBC version $highest (per ${tool_desc}) <= floor $floor"
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

  # Changed from the old (unreachable) "empty -> pass" arm: by the time
  # `_assert_versions` is reached in the real flow, the static case has
  # already been ruled out by `_is_static_elf_dump` -- so empty input here
  # means "dynamically linked but nothing extracted", a real failure.
  rc=0
  printf '' | _assert_versions "2.28" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[empty-versions-fails]: OK (exit 1, expected 1)"
  else
    echo "self-test[empty-versions-fails]: FAIL (exit $rc, expected 1)" >&2
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

  # `_extract_versions` must never itself fail (no `pipefail` kill) when a
  # dump genuinely carries zero GLIBC_x.y tokens -- the exact bug this
  # round fixes: under `set -euo pipefail`, a bare `grep -o` with no match
  # used to abort the whole script before this point was ever reached.
  rc=0
  printf 'no glibc tokens in this line\n' | _extract_versions > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[extract-never-fails-on-no-match]: OK"
  else
    echo "self-test[extract-never-fails-on-no-match]: FAIL (rc=$rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  # `_is_static_elf_dump`: the genuine static case (GNU binutils' exact
  # wording) must be detected...
  rc=0
  printf 'There is no dynamic section in this file.\n' | _is_static_elf_dump || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[static-dump-detected]: OK"
  else
    echo "self-test[static-dump-detected]: FAIL (rc=$rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  # ...and a real dynamic-section dump must NOT be mistaken for it.
  rc=0
  printf ' 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]\n' | _is_static_elf_dump || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[dynamic-dump-not-static]: OK"
  else
    echo "self-test[dynamic-dump-not-static]: FAIL (rc=$rc, expected 1)" >&2
    failures=$((failures + 1))
  fi

  # The tool-exit-code arm of the lattice: the `if out=$(cmd); then rc=0;
  # else rc=$?; fi` idiom the real flow uses to capture a symbol-dump
  # tool's exit code WITHOUT tripping `set -e` -- driven here with `false`/
  # `true` instead of objdump/readelf so no ELF tooling is required.
  rc=0
  if dump_output="$(false)"; then tool_rc=0; else tool_rc=$?; fi
  if [ "$tool_rc" -ne 0 ]; then
    echo "self-test[tool-exit-code-captured-on-failure]: OK (rc=$tool_rc)"
  else
    echo "self-test[tool-exit-code-captured-on-failure]: FAIL (rc=$tool_rc, expected non-zero)" >&2
    failures=$((failures + 1))
  fi

  if dump_output="$(printf 'GLIBC_2.17\n')"; then tool_rc=0; else tool_rc=$?; fi
  if [ "$tool_rc" -eq 0 ] && [ "$dump_output" = "GLIBC_2.17" ]; then
    echo "self-test[tool-exit-code-captured-on-success]: OK"
  else
    echo "self-test[tool-exit-code-captured-on-success]: FAIL (rc=$tool_rc, output='$dump_output')" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "assert-glibc-floor --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "assert-glibc-floor --self-test: all 9 fixture(s) passed."
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

# Lattice arm 1: genuinely static, per `readelf -d` -- checked FIRST and
# independently of which tool does the symbol dump below, so a static
# binary short-circuits to a pass with a printed notice before the
# dump-tool's exit code or output is even considered.
if command -v readelf >/dev/null 2>&1; then
  if readelf -d "$binary" 2>/dev/null | _is_static_elf_dump; then
    echo "assert-glibc-floor: ${binary} is statically linked (readelf -d reports no dynamic section) -- nothing to assert, floor check N/A"
    exit 0
  fi
fi

# Lattice arm 2 vs 3: run the symbol-dump tool, capturing its OWN exit code
# via the `if var=$(cmd); then rc=0; else rc=$?; fi` idiom (never trips
# `set -e`, unlike a bare assignment) -- a non-zero exit here is arm 2 (a
# real failure: the binary is dynamically linked but this tool could not
# read it), never silently treated the same as "static" or as "zero tokens
# found but the tool was fine".
if command -v objdump >/dev/null 2>&1; then
  tool_desc="objdump -T"
  if dump_output="$(objdump -T "$binary" 2>/dev/null)"; then
    tool_rc=0
  else
    tool_rc=$?
  fi
elif command -v readelf >/dev/null 2>&1; then
  tool_desc="readelf --dyn-syms"
  if dump_output="$(readelf --dyn-syms --wide "$binary" 2>/dev/null)"; then
    tool_rc=0
  else
    tool_rc=$?
  fi
else
  echo "::error::assert_glibc_floor.sh: neither objdump nor readelf is on PATH -- cannot read the binary's linked symbol versions" >&2
  exit 1
fi

if [ "$tool_rc" -ne 0 ]; then
  echo "::error::assert_glibc_floor.sh: ${tool_desc} exited ${tool_rc} against ${binary} (and it is not statically linked, per 'readelf -d') -- cannot assert the GLIBC floor" >&2
  exit 1
fi

# Lattice arm 3: dynamically linked, tool succeeded -- the real floor
# compare. `_assert_versions` itself now fails closed on zero tokens
# extracted (the binary is dynamic and the tool worked, so an empty result
# here is unexplained, not a pass).
versions="$(printf '%s\n' "$dump_output" | _extract_versions)"
printf '%s\n' "$versions" | _assert_versions "$floor" "$tool_desc"
