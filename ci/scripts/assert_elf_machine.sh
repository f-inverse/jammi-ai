#!/usr/bin/env bash
# Asserts an ELF (or Mach-O) binary's own machine field matches an arch a
# caller is about to STAMP onto an artifact -- a release tarball's triple, a
# PyPI wheel's `--platform-tag`, or a native wheel's own platform-derived
# tag -- rather than trusting the runner label / container image the binary
# happened to be built on. Factored out of `package_release_bin.sh`'s S4
# assert (#482) so every leg that stamps an architecture runs the identical
# check instead of a per-caller reimplementation: `package_release_bin.sh`
# itself, `_pypi-server.yml`'s wheel-tagging step (BLOCK 1, #482 -- nothing
# else there asserted the binary's actual arch before
# `python -m wheel tags --platform-tag` relabeled it), and `pypi.yml`'s
# native-Linux legs, run against the `.so` unzipped out of the maturin
# wheel.
#
# Usage:
#   assert_elf_machine.sh <binary> <expected-arch>
#     <expected-arch> is "x86_64" or "aarch64" -- canonical and OS-
#     independent. The script itself decides which tool to trust (`readelf
#     -h` for ELF, `file -b` for Mach-O -- detected off the binary, not the
#     caller's OS) and which substring that tool prints for the expected
#     arch (`readelf` says "X86-64"/"AArch64"; `file` says "x86_64"/"arm64"
#     for Apple Silicon -- the two tools disagree on the AArch64 name, so
#     the substring map is picked per tool, never a shared one). A Mach-O
#     "universal" (fat) binary is refused outright, never substring-matched
#     -- its description carries every bundled arch's substring at once, so
#     a plain substring test would satisfy whichever arch was expected.
#     Exits 0 on a match, 1 otherwise, naming both the binary and the
#     expected arch.
#   assert_elf_machine.sh --self-test
#     Drives the substring-match and arch-mapping logic against synthetic
#     tool output, no ELF tooling, no network; exits 0 iff every fixture
#     matches.
set -euo pipefail

# Maps a canonical <expected-arch> token to the substring `readelf -h`'s
# "Machine:" line carries for it.
_elf_machine_substr() {
  case "$1" in
    x86_64) echo "X86-64" ;;
    aarch64) echo "AArch64" ;;
    *)
      echo "::error::assert_elf_machine.sh: no known ELF machine mapping for arch '$1' (expected x86_64 or aarch64)" >&2
      return 1
      ;;
  esac
}

# Maps a canonical <expected-arch> token to the substring `file -b` carries
# for a Mach-O binary of that arch (Apple Silicon reports "arm64", never
# "aarch64").
_macho_machine_substr() {
  case "$1" in
    x86_64) echo "x86_64" ;;
    aarch64) echo "arm64" ;;
    *)
      echo "::error::assert_elf_machine.sh: no known Mach-O machine mapping for arch '$1' (expected x86_64 or aarch64)" >&2
      return 1
      ;;
  esac
}

# The real comparison, factored so --self-test can drive it with synthetic
# tool output -- no ELF tooling required. Returns (not exits) 0 if `$2`
# (the machine string already read off the tool) contains `$1` (the
# expected substring), 1 otherwise.
_match_machine() {
  local expected_substr="$1"
  local got="$2"
  case "$got" in
    *"$expected_substr"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Refuses a Mach-O "universal" (fat) binary outright. `file -b`'s single-line
# description of a fat binary contains EVERY bundled arch's substring in one
# string (e.g. both "x86_64" and "arm64"), so `_match_machine` above -- a
# plain substring test -- would report a match for WHICHEVER arch the
# caller expected, regardless of which one this leg is actually stamping. A
# release leg promises a single-arch binary under its triple's tag; a fat
# binary satisfies that promise for every arch at once, which is exactly the
# "any arch via substring" gap this closes. Factored so --self-test can
# drive it with synthetic `file -b` output, no Mach-O tooling required.
_is_macho_universal() {
  case "$1" in
    *'universal binary'*) return 0 ;;
    *) return 1 ;;
  esac
}

_self_test() {
  local failures=0
  local rc

  rc=0
  _match_machine "X86-64" "  Machine:                           Advanced Micro Devices X86-64" || rc=$?
  if [ "$rc" -eq 0 ]; then echo "self-test[elf-x86_64-match]: OK"; else echo "self-test[elf-x86_64-match]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  rc=0
  _match_machine "AArch64" "  Machine:                           AArch64" || rc=$?
  if [ "$rc" -eq 0 ]; then echo "self-test[elf-aarch64-match]: OK"; else echo "self-test[elf-aarch64-match]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  # The exact mismatch this script exists to catch: an aarch64-expecting
  # caller fed an x86_64 binary's readelf output must be rejected.
  rc=0
  _match_machine "AArch64" "  Machine:                           Advanced Micro Devices X86-64" || rc=$?
  if [ "$rc" -eq 1 ]; then echo "self-test[elf-mismatch-rejected]: OK"; else echo "self-test[elf-mismatch-rejected]: FAIL (rc=$rc, expected 1)" >&2; failures=$((failures + 1)); fi

  rc=0
  _match_machine "arm64" "Mach-O 64-bit executable arm64" || rc=$?
  if [ "$rc" -eq 0 ]; then echo "self-test[macho-arm64-match]: OK"; else echo "self-test[macho-arm64-match]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  rc=0
  _match_machine "x86_64" "Mach-O 64-bit executable x86_64" || rc=$?
  if [ "$rc" -eq 0 ]; then echo "self-test[macho-x86_64-match]: OK"; else echo "self-test[macho-x86_64-match]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  rc=0
  _match_machine "arm64" "Mach-O 64-bit executable x86_64" || rc=$?
  if [ "$rc" -eq 1 ]; then echo "self-test[macho-mismatch-rejected]: OK"; else echo "self-test[macho-mismatch-rejected]: FAIL (rc=$rc, expected 1)" >&2; failures=$((failures + 1)); fi

  rc=0
  _elf_machine_substr "x86_64" > /dev/null || rc=$?
  if [ "$rc" -eq 0 ]; then echo "self-test[elf-map-known]: OK"; else echo "self-test[elf-map-known]: FAIL" >&2; failures=$((failures + 1)); fi

  rc=0
  _elf_machine_substr "riscv64" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then echo "self-test[elf-map-unknown-arch-fails]: OK"; else echo "self-test[elf-map-unknown-arch-fails]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  rc=0
  _macho_machine_substr "riscv64" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then echo "self-test[macho-map-unknown-arch-fails]: OK"; else echo "self-test[macho-map-unknown-arch-fails]: FAIL (rc=$rc)" >&2; failures=$((failures + 1)); fi

  # The exact substring-satisfies-any-arch shape this round closes: a fat
  # binary's `file -b` description carries every bundled arch's substring at
  # once, so it must be flagged BEFORE any substring match is attempted --
  # never treated as a match for whichever arch happened to be expected.
  rc=0
  if _is_macho_universal "Mach-O universal binary with 2 architectures: [x86_64:Mach-O 64-bit executable x86_64] [arm64:Mach-O 64-bit executable arm64]"; then rc=0; else rc=1; fi
  if [ "$rc" -eq 0 ]; then echo "self-test[macho-universal-detected]: OK"; else echo "self-test[macho-universal-detected]: FAIL" >&2; failures=$((failures + 1)); fi

  rc=0
  if _is_macho_universal "Mach-O 64-bit executable arm64"; then rc=0; else rc=1; fi
  if [ "$rc" -eq 1 ]; then echo "self-test[macho-single-arch-not-flagged-universal]: OK"; else echo "self-test[macho-single-arch-not-flagged-universal]: FAIL" >&2; failures=$((failures + 1)); fi

  if [ "$failures" -gt 0 ]; then
    echo "assert-elf-machine --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "assert-elf-machine --self-test: all 11 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

binary="${1:?usage: assert_elf_machine.sh <binary> <expected-arch>}"
expected_arch="${2:?usage: assert_elf_machine.sh <binary> <expected-arch>}"

if [ ! -f "$binary" ]; then
  echo "::error::assert_elf_machine.sh: no such file: $binary" >&2
  exit 1
fi

# The binary's own container format decides which tool to trust -- `file
# -b` names it without requiring the caller to already know (this script
# runs on both native Linux release legs and the macOS/Darwin release
# legs `package_release_bin.sh` packages).
file_desc="$(file -b "$binary" 2> /dev/null || true)"

case "$file_desc" in
  *Mach-O*)
    if _is_macho_universal "$file_desc"; then
      echo "::error::assert_elf_machine.sh: ${binary} is a Mach-O universal (fat) binary ('${file_desc}') -- refusing: a release leg must stamp a single-arch binary under arch ${expected_arch}, and a fat binary's description substring-matches every arch it bundles, not just the one being asserted" >&2
      exit 1
    fi
    expected_substr="$(_macho_machine_substr "$expected_arch")"
    got="$file_desc"
    tool_desc="file -b"
    ;;
  *)
    if ! command -v readelf > /dev/null 2>&1; then
      echo "::error::assert_elf_machine.sh: readelf is not on PATH -- cannot read ${binary}'s ELF machine field" >&2
      exit 1
    fi
    expected_substr="$(_elf_machine_substr "$expected_arch")"
    got="$(readelf -h "$binary" 2> /dev/null | grep -i '^ *Machine:' || true)"
    tool_desc="readelf -h"
    ;;
esac

if [ -z "$got" ]; then
  echo "::error::assert_elf_machine.sh: ${tool_desc} produced no machine field for ${binary} -- cannot assert arch ${expected_arch}" >&2
  exit 1
fi

if _match_machine "$expected_substr" "$got"; then
  echo "assert-elf-machine: OK -- ${binary} (${tool_desc}: '${got}') matches expected arch ${expected_arch}"
  exit 0
fi

echo "::error::assert_elf_machine.sh: ${binary} (${tool_desc}: '${got}') does not report '${expected_substr}' -- expected arch ${expected_arch}" >&2
exit 1
