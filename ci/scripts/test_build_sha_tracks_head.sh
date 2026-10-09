#!/usr/bin/env bash
# needs: cargo
# The build-sha oracle's one leg that needs a real commit in a real checkout
# (`check_build_sha_tracks_head.sh` makes an empty commit): run over a clone
# of this repository that is discarded with the test, never over the checkout
# the test was started from. The other legs are `provenance_baked.rs`'s.
set -euo pipefail

root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
clone="$(mktemp -d)"
trap 'rm -rf "$clone"' EXIT
git clone --quiet --shared --no-checkout "$root" "$clone/repo"
git -C "$clone/repo" checkout --quiet "$(git -C "$root" rev-parse HEAD)"
cd "$clone/repo"
# The clone's build lands in the caller's target directory, so a warm cache
# makes the two builds the oracle compares seconds each.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}"
bash ci/scripts/check_build_sha_tracks_head.sh
