#!/usr/bin/env bash
# `.cargo/config.toml`'s `[target.aarch64-unknown-linux-gnu]` stanza reaches a
# native aarch64 build: `+fp16` is a compile floor (`gemm`'s f16 kernels are
# selected at run time behind `is_aarch64_feature_detected!`, and the opt-level
# 0 build of that code compiles only with the feature on), and the mold link
# argument beside it is what every Linux target links with. Two readers of the
# same stanza, each asserted: cargo's cfg set with this job's own
# `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS` joined in (what a CI job
# gets), and with that variable unset (what a bare `cargo build` gets). A bare
# `RUSTFLAGS` would replace the stanza instead of joining it, so its absence is
# asserted first. Every assertion runs; none can quietly no-op.
#
#   bash ci/scripts/check_aarch64_floor.sh    # on an aarch64 Linux host with the pinned toolchain and mold
set -euo pipefail

if [ -n "${RUSTFLAGS+set}" ]; then
  echo "::error::RUSTFLAGS is set ('$RUSTFLAGS'): a bare RUSTFLAGS replaces .cargo/config.toml's per-target rustflags, dropping mold and the fp16 floor" >&2
  exit 1
fi
[ "$(uname -m)" = aarch64 ] || { echo "::error::this oracle reads the aarch64 stanza; the host is $(uname -m)" >&2; exit 1; }

# No stderr redirect: a compile failure must be visible, never read as a
# missing cfg flag.
cfg_joined="$(cargo rustc -p jammi-numerics --lib -- --print cfg)"
grep -q 'target_feature="fp16"' <<<"$cfg_joined" \
  || { echo "::error::config.toml joined with CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS does not carry +fp16" >&2; exit 1; }
cfg_alone="$(env -u CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS cargo rustc -p jammi-numerics --lib -- --print cfg)"
grep -q 'target_feature="fp16"' <<<"$cfg_alone" \
  || { echo "::error::config.toml alone does not carry +fp16" >&2; exit 1; }
echo "the aarch64 floor holds: +fp16 in cargo's cfg set, joined with the CI variable and alone"
