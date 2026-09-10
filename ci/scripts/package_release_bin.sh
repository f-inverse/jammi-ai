#!/usr/bin/env bash
# Packages one release binary into the versioned tarball asset the release
# lanes upload (release-binaries.yml): reads the workspace version from
# Cargo.toml, asserts the binary was actually built for the triple it claims
# (S4) and, for a `*-linux-gnu` triple, that it links no GLIBC symbol above
# the manylinux_2_28 floor (W3/V2 -- beside the machine assert, so a
# bare-runner build can never be promoted), strips the binary, and tars it as
# `<bin>-<version>-<triple>.tar.gz`. Emits `asset=<name>` to $GITHUB_OUTPUT
# so the upload steps can reference it. One definition instead of a repeated
# per-leg version-parse + strip + tar block. (The CUDA server tarball is NOT
# packaged here — its lib-bundling + launcher assembly is bespoke and stays
# inline in its own job.)
#
# Usage: package_release_bin.sh <binary-name> <target-triple>
set -euo pipefail

bin="${1:?usage: package_release_bin.sh <binary-name> <target-triple>}"
triple="${2:?usage: package_release_bin.sh <binary-name> <target-triple>}"

# `sed` echoes the line unchanged on no-substitution, so a shape assert is
# the difference between failing here and uploading a garbage-named asset.
version=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
case "$version" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *)
    echo "::error::could not parse a semver workspace version out of Cargo.toml (got: '${version}')" >&2
    exit 1
    ;;
esac

bin_path="target/release/${bin}"

# S4: assert the binary's own ELF/Mach-O machine field matches the triple's
# arch -- a mismatched leg (wrong runner, wrong cross-target) fails here,
# before packaging, rather than shipping a binary that cannot exec on the
# host its filename promises. Delegates to the shared
# `assert_elf_machine.sh` (#482) -- the same script `_pypi-server.yml`'s
# wheel-tagging step and `pypi.yml`'s native Linux legs call, so every leg
# that stamps an architecture onto an artifact runs one identical check
# instead of a per-caller reimplementation.
case "$triple" in
  x86_64-*-linux-gnu | x86_64-apple-darwin)
    want_arch="x86_64"
    ;;
  aarch64-*-linux-gnu | aarch64-apple-darwin)
    want_arch="aarch64"
    ;;
  *)
    echo "::error::package_release_bin.sh: no known machine-assert mapping for triple '${triple}'" >&2
    exit 1
    ;;
esac

bash "$(dirname "${BASH_SOURCE[0]}")/assert_elf_machine.sh" "$bin_path" "$want_arch"

# W3/V2: the sufficient ABI floor assert for every *-linux-gnu triple --
# building inside the manylinux_2_28 CI container is necessary but not
# sufficient; this reads the binary's own linked GLIBC symbol versions.
case "$triple" in
  *-linux-gnu)
    bash "$(dirname "${BASH_SOURCE[0]}")/assert_glibc_floor.sh" "$bin_path" 2.28
    ;;
esac

asset="${bin}-${version}-${triple}.tar.gz"
strip "$bin_path"
tar -C target/release -czf "$asset" "$bin"
echo "asset=$asset" >> "$GITHUB_OUTPUT"
