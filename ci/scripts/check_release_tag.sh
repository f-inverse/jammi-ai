#!/usr/bin/env bash
# A release tag names the version the checkout ships: `vX.Y.Z` or `py-vX.Y.Z`
# equals `[workspace.package] version` in Cargo.toml. Every publisher runs this
# on its tag ref before promoting anything (`_proof-required.yml`), and the
# crates.io publish keys its idempotence probes on the same version.
#
#   bash ci/scripts/check_release_tag.sh TAG
#
# Prints the version on success.
set -euo pipefail

tag="${1:?TAG}"
case "$tag" in
  py-v*) version="${tag#py-v}" ;;
  v*) version="${tag#v}" ;;
  *) echo "::error::$tag is not a release tag (vX.Y.Z or py-vX.Y.Z)" >&2; exit 1 ;;
esac
workspace="$(python3 -c 'import tomllib; print(tomllib.load(open("Cargo.toml","rb"))["workspace"]["package"]["version"])')"
if [ "$version" != "$workspace" ]; then
  echo "::error::tag $tag names version $version but the workspace is at $workspace -- refusing to release a tree under another version" >&2
  exit 1
fi
echo "$version"
