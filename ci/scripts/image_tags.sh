#!/usr/bin/env bash
# The one tag policy of every published server image, as the `tags:` block
# `docker/metadata-action` reads. A release tag `vX.Y.Z` publishes `X.Y.Z`,
# `X.Y`, `sha-<sha>` and `latest` (the newest release); the self-contained
# variant of the CPU image carries the same four under its own name:
# `selfcontained-X.Y.Z`, `selfcontained-X.Y`, `selfcontained-sha-<sha>` and
# `selfcontained`. Nothing but a release tag ever moves one of these.
#
#   bash ci/scripts/image_tags.sh VARIANT [--immutable | --commit-tag SHA]
#
# VARIANT is `generic`, `selfcontained` or `cuda` (the CUDA image's own
# tags, the generic scheme on its own name). `--immutable` prints only the
# commit tag's entry, the one a per-arch leg pushes before the merge moves
# the rest; `--commit-tag SHA` prints that tag's name for SHA, the one the
# merge re-reads (`docker/metadata-action` orders its output by type, never
# by the spec, so the name is derived here, not read back).
set -euo pipefail

variant="${1:?VARIANT}"; shift
mode="${1:-}"
case "$variant" in
  generic | cuda) prefix="" moving="latest" ;;
  selfcontained) prefix="selfcontained-" moving="selfcontained" ;;
  *) echo "::error::unknown image variant '$variant' (generic, selfcontained, cuda)" >&2; exit 1 ;;
esac
case "$mode" in
  --commit-tag)
    sha="${2:?SHA}"
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "::error::'$sha' is not a full commit sha" >&2; exit 1; }
    echo "${prefix}sha-${sha}"
    exit 0
    ;;
  "" | --immutable) ;;
  *) echo "usage: $0 VARIANT [--immutable | --commit-tag SHA]" >&2; exit 2 ;;
esac
echo "type=sha,format=long,prefix=${prefix}sha-"
[ "$mode" != --immutable ] || exit 0
echo "type=semver,pattern={{version}},prefix=${prefix}"
echo "type=semver,pattern={{major}}.{{minor}},prefix=${prefix}"
echo "type=raw,value=${moving}"
