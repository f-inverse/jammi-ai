#!/usr/bin/env bash
# Merges per-arch image sources into one multi-arch index under real tags,
# verify-then-promote: the merge is dry-run and its index's platform set
# asserted before anything is pushed, every tag's previous digest is read
# first, and after the push the immutable first tag is re-read and asserted
# again. The one definition of an index promotion: the tree's CI image
# (`_ci-base-image.yml`) and every published server image (`server-image.yml`)
# merge through it.
#
#   bash ci/scripts/merge_image_index.sh --platforms linux/amd64,linux/arm64 \
#     --tag IMAGE:TAG [--tag IMAGE:TAG ...] -- SOURCE...
#
# The first `--tag` is the immutable one (a content or commit tag): it is the
# tag re-read after the push, and the digest it resolves to is the one line
# on stdout, `digest=<sha256>`, for `$GITHUB_OUTPUT`; every other line is
# progress, on stderr.
#
# Failure arms, each named:
#   - the dry-run index's platform set differs from --platforms: nothing pushed;
#   - a tag's previous digest cannot be read for any reason but "not found":
#     nothing pushed (a tag this cannot even read it must not move);
#   - the post-push re-read fails, or returns no digest: whether the merge
#     landed is unknown, so no restore is suggested;
#   - the post-push index is wrong: the merge landed wrong, and the restore
#     command for EVERY tag, each to its OWN previous digest, is printed.
set -euo pipefail

platforms="" tags=()
while [ $# -gt 0 ]; do
  case "$1" in
    --platforms) platforms="$2"; shift 2 ;;
    --tag) tags+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "usage: $0 --platforms P --tag T [--tag T...] -- SOURCE..." >&2; exit 2 ;;
  esac
done
sources=("$@")
[ -n "$platforms" ] && [ "${#tags[@]}" -gt 0 ] && [ "${#sources[@]}" -gt 0 ] \
  || { echo "usage: $0 --platforms P --tag T [--tag T...] -- SOURCE..." >&2; exit 2; }
check="$(dirname "${BASH_SOURCE[0]}")/check_merged_index_platforms.sh"

tag_args=()
for t in "${tags[@]}"; do tag_args+=(-t "$t"); done

docker buildx imagetools create --dry-run "${tag_args[@]}" "${sources[@]}" | bash "$check" "$platforms" >&2

# A digest, or empty for a tag that does not exist yet; any other failure
# refuses the promotion.
previous_digest() {
  local tag="$1" out rc stderr
  stderr="$(mktemp)"
  set +e
  out="$(docker buildx imagetools inspect "$tag" --format '{{.Manifest.Digest}}' 2>"$stderr")"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    if grep -q "not found" "$stderr"; then
      rm -f "$stderr"; echo ""; return 0
    fi
    echo "::error::cannot read $tag before promoting (not a not-found): $(cat "$stderr")" >&2
    rm -f "$stderr"; exit 1
  fi
  rm -f "$stderr"
  [[ "$out" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || { echo "::error::$tag resolved to '$out', not a digest -- refusing to promote" >&2; exit 1; }
  echo "$out"
}

declare -A previous
for t in "${tags[@]}"; do
  previous["$t"]="$(previous_digest "$t")"
  if [ -z "${previous[$t]}" ]; then echo "$t: first publish" >&2; else echo "$t: currently ${previous[$t]}" >&2; fi
done

docker buildx imagetools create "${tag_args[@]}" "${sources[@]}" >&2

immutable="${tags[0]}"
set +e
digest="$(docker buildx imagetools inspect "$immutable" --format '{{.Manifest.Digest}}' 2>/tmp/merge-confirm-stderr)"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "::error::re-reading $immutable after the push failed ($(cat /tmp/merge-confirm-stderr)) -- whether the merge landed is unknown; verify by hand before any restore" >&2
  exit 1
fi
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || { echo "::error::$immutable re-read as '$digest', not a digest -- whether the merge landed is unknown; verify by hand" >&2; exit 1; }
if ! docker buildx imagetools inspect "$immutable" --format '{{json .Manifest}}' | bash "$check" "$platforms" >&2; then
  echo "::error::the pushed index under $immutable is wrong; restore each tag to its own previous digest:" >&2
  for t in "${tags[@]}"; do
    if [ -n "${previous[$t]}" ]; then
      echo "::error::  docker buildx imagetools create -t $t ${t%%:*}@${previous[$t]}" >&2
    else
      echo "::error::  $t was a first publish: nothing to restore" >&2
    fi
  done
  exit 1
fi
echo "digest=$digest"
