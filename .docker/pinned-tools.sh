#!/usr/bin/env bash
# The third-party tools the CI lanes run, each pinned once, here. The CI image
# installs them at build time (this file is an input of its content tag, so a
# moved pin is a new image); a lane on a bare runner, or a host outside the
# image, provides them through this same script, so a version and its checksum
# never live in two places.
#
#   .docker/pinned-tools.sh TOOL...
#
# A TOOL already installed at its pin is left alone, so in the CI image every
# call is a no-op that touches no network. A release binary is verified
# against its published SHA-256 before it is installed into /usr/local/bin; a
# cargo tool is `cargo install --locked` at its exact version.
set -euo pipefail

case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "no pinned tools for $(uname -m)" >&2; exit 1 ;;
esac

# at_pin VERSION CMD... — CMD runs and reports VERSION. A here-string, not a
# pipe: under pipefail, `grep -q` exiting at its first match SIGPIPEs CMD and
# reads as absent.
at_pin() {
  local version="$1"; shift
  command -v "$1" >/dev/null && grep -qwF "$version" <<< "$("$@" 2>/dev/null)"
}

# release URL SHA256 MEMBER — fetch the tarball, verify it, install MEMBER.
release() {
  local url="$1" sum="$2" member="$3" tmp
  tmp="$(mktemp -d)"
  curl -fsSL "$url" -o "$tmp/release.tar.gz"
  echo "$sum  $tmp/release.tar.gz" | sha256sum -c --quiet -
  tar -xzf "$tmp/release.tar.gz" -C "$tmp" "$member"
  install -m 0755 "$tmp/$member" "/usr/local/bin/$member"
  rm -rf "$tmp"
}

provide() {
  local v
  case "$1" in
    actionlint)
      v=1.7.12
      at_pin "$v" actionlint --version && return
      declare -A sum=(
        [amd64]=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
        [arm64]=325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6
      )
      release "https://github.com/rhysd/actionlint/releases/download/v${v}/actionlint_${v}_linux_${arch}.tar.gz" \
        "${sum[$arch]}" actionlint
      ;;
    kustomize)
      v=5.8.1
      at_pin "v$v" kustomize version && return
      declare -A sum=(
        [amd64]=029a7f0f4e1932c52a0476cf02a0fd855c0bb85694b82c338fc648dcb53a819d
        [arm64]=0953ea3e476f66d6ddfcd911d750f5167b9365aa9491b2326398e289fef2c142
      )
      release "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv${v}/kustomize_v${v}_linux_${arch}.tar.gz" \
        "${sum[$arch]}" kustomize
      ;;
    kubeconform)
      v=0.8.0
      at_pin "v$v" kubeconform -v && return
      declare -A sum=(
        [amd64]=9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883
        [arm64]=1f53fc8e81258197a35e8603054162a5af1de8c5af13746c71ab680d9534ed87
      )
      release "https://github.com/yannh/kubeconform/releases/download/v${v}/kubeconform-linux-${arch}.tar.gz" \
        "${sum[$arch]}" kubeconform
      ;;
    # The test runner every hermetic lane uses (`.config/nextest.toml`).
    cargo-nextest)
      v=0.9.146
      at_pin "$v" cargo-nextest nextest --version && return
      declare -A triple=([amd64]=x86_64-unknown-linux-gnu [arm64]=aarch64-unknown-linux-gnu)
      declare -A sum=(
        [amd64]=682c21b777c333e96fd532e114d3a5a894e0729ab88d94c0a9f20f8419695428
        [arm64]=b2e33d7c72de7ade0ff7b3a948ac37516b24f8a836b7a8870c1f634a94be9de9
      )
      release "https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-${v}/cargo-nextest-${v}-${triple[$arch]}.tar.gz" \
        "${sum[$arch]}" cargo-nextest
      ;;
    # deny.toml's shape asserts this major version's schema behavior; the
    # RustSec advisory database the tool reads stays deliberately live.
    cargo-deny)
      v=0.20.2
      at_pin "$v" cargo-deny --version && return
      cargo install cargo-deny --version "$v" --locked
      ;;
    # The dep-DAG generator's source of truth; the committed block matches
    # what this version renders.
    build-graph)
      v=0.1.0
      at_pin "$v" cargo-build-graph build-graph --version && return
      cargo install build-graph --version "$v" --locked
      ;;
    # The linker `.cargo/config.toml` names for every Linux target; its
    # `bin/` and `lib/` unpack under /usr/local as the release lays them out.
    mold)
      v=2.35.1
      at_pin "$v" mold --version && return
      declare -A sum=(
        [amd64]=e58ff420ba4b034222a3519227a142a1aef4915f6feea1ecc42006d1e0b85111
        [arm64]=4d06acf1d7f92a495c0bea0cbdb3a827ee65b004771cab3cf8a7acc280f80c13
      )
      declare -A machine=([amd64]=x86_64 [arm64]=aarch64)
      local tmp
      tmp="$(mktemp -d)"
      curl -fsSL "https://github.com/rui314/mold/releases/download/v${v}/mold-${v}-${machine[$arch]}-linux.tar.gz" -o "$tmp/mold.tar.gz"
      echo "${sum[$arch]}  $tmp/mold.tar.gz" | sha256sum -c --quiet -
      tar -xzf "$tmp/mold.tar.gz" --strip-components=1 -C /usr/local
      rm -rf "$tmp"
      ;;
    *) echo "no pin for tool '$1'" >&2; exit 1 ;;
  esac
}

[ "$#" -gt 0 ] || { echo "usage: $0 TOOL..." >&2; exit 2; }
for tool in "$@"; do provide "$tool"; done
