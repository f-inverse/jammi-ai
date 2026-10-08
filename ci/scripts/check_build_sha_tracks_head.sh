#!/usr/bin/env bash
# The build-sha hermeticity oracle: a new commit, on a rebuild, moves the sha
# `jammi-bench` bakes (`crates/jammi-bench/build.rs`) to match it. This is the
# one leg that needs a REAL commit in a real checkout, so it is a CI step over
# an ephemeral clone, never a `#[test]`; `provenance_baked.rs` holds the other
# two legs (an uncommitted edit dirties a rebuild; a built binary ignores its
# run-time environment) as fast tests over a scratch repository.
#
#   bash ci/scripts/check_build_sha_tracks_head.sh
#
# Makes an empty commit in the current checkout. Run it only where that
# commit is discarded with the clone.
#
# The `sleep 1` between reading the first sha and committing guards a timing
# false reading: cargo's `rerun-if-changed` compares the watched file's mtime
# with its fingerprint, and on a one-second-granularity filesystem a commit in
# the same second as the prior build is indistinguishable from "unchanged".
set -euo pipefail

field() { python3 -c 'import json,sys; print(json.load(sys.stdin)["'"$1"'"])'; }

before="$(cargo run -p jammi-bench --quiet -- provenance | field build_sha)"
sleep 1
git -c user.email=ci@jammi.invalid -c user.name=jammi-ci \
  commit --quiet --allow-empty -m "build-sha hermeticity probe (never pushed)"
new_head="$(git rev-parse HEAD)"
after="$(cargo run -p jammi-bench --quiet -- provenance | field build_sha)"
echo "before=$before after=$after new_head=$new_head"
if [ "$before" = "$after" ]; then
  echo "::error::the baked build_sha did not change after a real commit and rebuild (before=$before after=$after)"
  exit 1
fi
case "$after" in
  "$new_head" | "$new_head"-dirty) ;;
  *)
    echo "::error::the rebuilt build_sha ($after) does not track the new commit ($new_head)"
    exit 1
    ;;
esac
