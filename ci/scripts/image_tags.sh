#!/usr/bin/env bash
# The one tag policy of every published server image, as the `tags:` block
# `docker/metadata-action` reads. A release tag `vX.Y.Z` publishes `X.Y.Z`,
# `X.Y`, `sha-<sha>` and `latest` (the newest release); the self-contained
# variant of the CPU image carries the same four under its own name:
# `selfcontained-X.Y.Z`, `selfcontained-X.Y`, `selfcontained-sha-<sha>` and
# `selfcontained`. Nothing but a release tag ever moves one of these.
#
#   bash ci/scripts/image_tags.sh VARIANT [--immutable]
#
# VARIANT is `generic`, `selfcontained` or `cuda` (the CUDA image's own
# tags, the generic scheme on its own name). `--immutable` prints only the
# commit tag, the one a per-arch leg pushes before the merge moves the rest.
set -euo pipefail

variant="${1:?VARIANT}"; shift
immutable_only=false
[ "${1:-}" != "--immutable" ] || immutable_only=true
case "$variant" in
  generic | cuda) prefix="" moving="latest" ;;
  selfcontained) prefix="selfcontained-" moving="selfcontained" ;;
  *) echo "::error::unknown image variant '$variant' (generic, selfcontained, cuda)" >&2; exit 1 ;;
esac
echo "type=sha,format=long,prefix=${prefix}sha-"
$immutable_only && exit 0
echo "type=semver,pattern={{version}},prefix=${prefix}"
echo "type=semver,pattern={{major}}.{{minor}},prefix=${prefix}"
echo "type=raw,value=${moving}"
