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
# cargo tool is `cargo install --locked` at its exact version; maturin on a
# bare host is the pip release at the version the image's cargo install pins.
# Linux tools come as amd64 and arm64; a tool the macOS legs need comes as
# darwin arm64 too (the x86_64 macOS wheel is cross-compiled on the arm64
# runner).
set -euo pipefail

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  *) echo "no pinned tools for $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "no pinned tools for $(uname -m)" >&2; exit 1 ;;
esac
platform="$os-$arch"

# at_pin VERSION CMD... — CMD runs and reports VERSION. A here-string, not a
# pipe: under pipefail, `grep -q` exiting at its first match SIGPIPEs CMD and
# reads as absent.
at_pin() {
  local version="$1"; shift
  command -v "$1" >/dev/null && grep -qwF "$version" <<< "$("$@" 2>/dev/null)"
}

sha256() { # FILE SUM — the file's SHA-256 is SUM, by whichever tool the OS has.
  local got
  if command -v sha256sum >/dev/null; then got="$(sha256sum "$1" | cut -d' ' -f1)"; else got="$(shasum -a 256 "$1" | cut -d' ' -f1)"; fi
  [ "$got" = "$2" ] || { echo "$1: sha256 $got, expected $2" >&2; return 1; }
}

# release URL SHA256 MEMBER — fetch the tarball, verify it, install MEMBER.
release() {
  local url="$1" sum="$2" member="$3" tmp
  tmp="$(mktemp -d)"
  curl -fsSL "$url" -o "$tmp/release.tar"
  sha256 "$tmp/release.tar" "$sum"
  tar -xf "$tmp/release.tar" -C "$tmp" "$member"
  install -m 0755 "$tmp/$member" "/usr/local/bin/$(basename "$member")"
  rm -rf "$tmp"
}

# A pin the current platform lacks is a usage error, never a silent no-op.
# Plain variables named `<table>_<os>_<arch>`, read by indirection: the macOS
# runner's /bin/bash is 3.2, which has neither associative arrays nor namerefs.
pinned() { # TABLE — the variable `<TABLE>_<os>_<arch>` must be set.
  local name="${1}_${os}_${arch}"
  [ -n "${!name:-}" ] || { echo "no $1 pin for $platform" >&2; exit 1; }
  echo "${!name}"
}

provide() {
  local v sum
  case "$1" in
    actionlint)
      v=1.7.12
      at_pin "$v" actionlint --version && return
      sums_linux_amd64=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
      sums_linux_arm64=325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6
      release "https://github.com/rhysd/actionlint/releases/download/v${v}/actionlint_${v}_${os}_${arch}.tar.gz" \
        "$(pinned sums)" actionlint
      ;;
    # actionlint lints every `run:` block through it when it is on PATH.
    shellcheck)
      v=0.11.0
      at_pin "$v" shellcheck --version && return
      sums_linux_amd64=8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198
      sums_linux_arm64=12b331c1d2db6b9eb13cfca64306b1b157a86eb69db83023e261eaa7e7c14588
      machine_linux_amd64=x86_64
      machine_linux_arm64=aarch64
      release "https://github.com/koalaman/shellcheck/releases/download/v${v}/shellcheck-v${v}.${os}.$(pinned machine).tar.xz" \
        "$(pinned sums)" "shellcheck-v${v}/shellcheck"
      ;;
    kustomize)
      v=5.8.1
      at_pin "v$v" kustomize version && return
      sums_linux_amd64=029a7f0f4e1932c52a0476cf02a0fd855c0bb85694b82c338fc648dcb53a819d
      sums_linux_arm64=0953ea3e476f66d6ddfcd911d750f5167b9365aa9491b2326398e289fef2c142
      release "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv${v}/kustomize_v${v}_${os}_${arch}.tar.gz" \
        "$(pinned sums)" kustomize
      ;;
    kubeconform)
      v=0.8.0
      at_pin "v$v" kubeconform -v && return
      sums_linux_amd64=9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883
      sums_linux_arm64=1f53fc8e81258197a35e8603054162a5af1de8c5af13746c71ab680d9534ed87
      release "https://github.com/yannh/kubeconform/releases/download/v${v}/kubeconform-${os}-${arch}.tar.gz" \
        "$(pinned sums)" kubeconform
      ;;
    # The test runner every hermetic lane uses (`.config/nextest.toml`).
    cargo-nextest)
      v=0.9.146
      at_pin "$v" cargo-nextest nextest --version && return
      triple_linux_amd64=x86_64-unknown-linux-gnu
      triple_linux_arm64=aarch64-unknown-linux-gnu
      sums_linux_amd64=682c21b777c333e96fd532e114d3a5a894e0729ab88d94c0a9f20f8419695428
      sums_linux_arm64=b2e33d7c72de7ade0ff7b3a948ac37516b24f8a836b7a8870c1f634a94be9de9
      release "https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-${v}/cargo-nextest-${v}-$(pinned triple).tar.gz" \
        "$(pinned sums)" cargo-nextest
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
    # PyO3 wheel builds: the image installs it from crates.io at this
    # version; a bare host takes the same version's pip release.
    maturin)
      v=1.14.1
      at_pin "$v" maturin --version && return
      python3 -m pip install --quiet "maturin==$v"
      ;;
    # prost-build (via substrait) invokes protoc at build time; the image
    # installs the same version from the same release.
    protoc)
      v=28.3
      at_pin "$v" protoc --version && return
      sums_darwin_arm64=92ceefda6a7293ec014e6ecac82d64719357145cb6fc2865badadeb5e62c0431
      asset_darwin_arm64=osx-aarch_64
      sum="$(pinned sums)"
      local tmp
      tmp="$(mktemp -d)"
      curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${v}/protoc-${v}-$(pinned asset).zip" -o "$tmp/protoc.zip"
      sha256 "$tmp/protoc.zip" "$sum"
      unzip -q "$tmp/protoc.zip" -d /usr/local bin/protoc 'include/*'
      chmod 0755 /usr/local/bin/protoc
      rm -rf "$tmp"
      ;;
    # The rustc wrapper `.cargo/config.toml` names; the image installs the
    # same version's musl build.
    sccache)
      v=0.14.0
      at_pin "$v" sccache --version && return
      sums_darwin_arm64=a781e8018260ab128e7690d8497736fa231b6ca895d57131d5b5b966ca987594
      triple_darwin_arm64=aarch64-apple-darwin
      release "https://github.com/mozilla/sccache/releases/download/v${v}/sccache-v${v}-$(pinned triple).tar.gz" \
        "$(pinned sums)" "sccache-v${v}-$(pinned triple)/sccache"
      ;;
    # The linker `.cargo/config.toml` names for every Linux target; its
    # `bin/` and `lib/` unpack under /usr/local as the release lays them out.
    mold)
      v=2.35.1
      at_pin "$v" mold --version && return
      sums_linux_amd64=e58ff420ba4b034222a3519227a142a1aef4915f6feea1ecc42006d1e0b85111
      sums_linux_arm64=4d06acf1d7f92a495c0bea0cbdb3a827ee65b004771cab3cf8a7acc280f80c13
      machine_linux_amd64=x86_64
      machine_linux_arm64=aarch64
      sum="$(pinned sums)"
      local tmp
      tmp="$(mktemp -d)"
      curl -fsSL "https://github.com/rui314/mold/releases/download/v${v}/mold-${v}-$(pinned machine)-linux.tar.gz" -o "$tmp/mold.tar.gz"
      sha256 "$tmp/mold.tar.gz" "$sum"
      tar -xzf "$tmp/mold.tar.gz" --strip-components=1 -C /usr/local
      rm -rf "$tmp"
      ;;
    *) echo "no pin for tool '$1'" >&2; exit 1 ;;
  esac
}

[ "$#" -gt 0 ] || { echo "usage: $0 TOOL..." >&2; exit 2; }
for tool in "$@"; do provide "$tool"; done
