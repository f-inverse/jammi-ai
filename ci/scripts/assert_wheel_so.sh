#!/usr/bin/env bash
# The `.so` inside a native wheel is the machine its platform tag claims and,
# on Linux, links no symbol above the GLIBC floor the tag promises. Built in
# a manylinux container does not by itself prove either: both are read off
# the bytes inside the wheel.
#
#   bash ci/scripts/assert_wheel_so.sh WHEEL ARCH [GLIBC]
#
# ARCH is x86_64 or aarch64; GLIBC (e.g. 2.28) is asserted when given and
# skipped for a macOS wheel, which has no GLIBC.
set -euo pipefail

wheel="${1:?WHEEL}"
arch="${2:?ARCH}"
glibc="${3:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
python3 -m zipfile -e "$wheel" "$workdir"
so="$(find "$workdir" -name '*.so' -print -quit)"
test -n "$so" || { echo "::error::no .so inside $wheel" >&2; exit 1; }
if [ -n "$glibc" ]; then
  bash "$here/assert_glibc_floor.sh" "$so" "$glibc"
fi
bash "$here/assert_elf_machine.sh" "$so" "$arch"
