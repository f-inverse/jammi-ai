#!/usr/bin/env bash
# The `jammi` CLI out of the one release tarball an artifact directory holds,
# executable, into DEST. Artifacts do not preserve the exec bit, so every
# consumer unpacks a CLI through this.
#
#   bash ci/scripts/unpack_cli.sh ARTIFACT_DIR DEST
set -euo pipefail

dir="${1:?ARTIFACT_DIR}"
dest="${2:?DEST}"

tarballs=("$dir"/jammi-*.tar.gz)
[ "${#tarballs[@]}" -eq 1 ] && [ -e "${tarballs[0]}" ] \
  || { echo "::error::expected one jammi-*.tar.gz in $dir, found: ${tarballs[*]}" >&2; exit 1; }
mkdir -p "$dest"
tar -xzf "${tarballs[0]}" -C "$dest" jammi
chmod +x "$dest/jammi"
