#!/usr/bin/env bash
# The `py-v*` release tag a published-cookbook run uses: the one requested,
# or the newest the repository holds; either must exist. Prints `tag=<tag>`
# for `$GITHUB_OUTPUT`.
#
#   bash ci/scripts/resolve_release_tag.sh [REQUESTED]
set -euo pipefail

requested="${1:-}"
repo="https://github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY}.git"

if [ -n "$requested" ]; then
  tag="$requested"
else
  tag="$(git ls-remote --tags --refs "$repo" 'py-v*' | sed 's#.*refs/tags/##' | sort -V | tail -n1)"
fi
case "$tag" in
  py-v[0-9]*) ;;
  *) echo "::error::'$tag' is not a py-vX.Y.Z release tag" >&2; exit 1 ;;
esac
git ls-remote --exit-code --tags "$repo" "refs/tags/$tag" >/dev/null \
  || { echo "::error::tag $tag does not exist" >&2; exit 1; }
echo "tag=$tag"
