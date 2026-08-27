#!/usr/bin/env bash
# Packages one release binary into the versioned tarball asset the release
# lanes upload (release-binaries.yml): reads the workspace version from
# Cargo.toml, strips the binary, and tars it as
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

version=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
asset="${bin}-${version}-${triple}.tar.gz"
strip "target/release/${bin}"
tar -C target/release -czf "$asset" "$bin"
echo "asset=$asset" >> "$GITHUB_OUTPUT"
