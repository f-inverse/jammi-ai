#!/usr/bin/env bash
# Installs the pinned actionlint release into /usr/local/bin, verified against
# its published checksum. The guard runner calls this to provide the
# `actionlint` need.
set -euo pipefail
version=1.7.12
case "$(uname -m)" in
  x86_64) arch=amd64; sum=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8 ;;
  aarch64 | arm64) arch=arm64; sum=325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6 ;;
  *) echo "no pinned actionlint for $(uname -m)" >&2; exit 1 ;;
esac
tarball="actionlint_${version}_linux_${arch}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "https://github.com/rhysd/actionlint/releases/download/v${version}/${tarball}" -o "$tmp/$tarball"
echo "$sum  $tmp/$tarball" | sha256sum -c --quiet -
tar -xzf "$tmp/$tarball" -C "$tmp" actionlint
install -m 0755 "$tmp/actionlint" /usr/local/bin/actionlint
