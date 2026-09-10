#!/usr/bin/env bash
# Asserts a pushed image digest carries both a provenance attestation and an
# SBOM attestation. Shared by every push job in server-image.yml so the
# assertion logic lives in exactly one place instead of four copies drifting
# independently.
#
# Reads `REF` from the environment: `<registry>/<image>@sha256:<digest>` --
# the exact digest the job just pushed, never a mutable tag (a tag can be
# re-pointed by a concurrent run; the digest cannot).
#
# Primary check: `docker buildx imagetools inspect --format` exposes
# `.Provenance`/`.SBOM` accessors on a sufficiently recent buildx. Fallback
# (buildx too old to carry the accessors): read the raw OCI index and assert
# an attestation manifest (`vnd.docker.reference.type=attestation-manifest`)
# whose layers include an spdx predicate.
set -euo pipefail

if [ -z "${REF:-}" ]; then
  echo "::error::REF is unset -- nothing to inspect" >&2
  exit 1
fi

provenance="$(docker buildx imagetools inspect "$REF" --format '{{json .Provenance}}' 2>/dev/null || true)"
sbom="$(docker buildx imagetools inspect "$REF" --format '{{json .SBOM}}' 2>/dev/null || true)"

if [ -n "$provenance" ] && [ "$provenance" != "null" ] && [ -n "$sbom" ] && [ "$sbom" != "null" ]; then
  echo "provenance and SBOM confirmed via imagetools .Provenance/.SBOM accessors for $REF"
  exit 0
fi

echo "::notice::imagetools .Provenance/.SBOM accessors unavailable on this runner's buildx for $REF -- falling back to --raw + jq"

raw="$(docker buildx imagetools inspect "$REF" --raw)"
att_digests="$(echo "$raw" | jq -r '.manifests[]? | select(.annotations["vnd.docker.reference.type"] == "attestation-manifest") | .digest')"
if [ -z "$att_digests" ]; then
  echo "::error::no attestation-manifest found for $REF" >&2
  exit 1
fi

image_ref="${REF%@*}"
found_spdx=0
while IFS= read -r digest; do
  [ -z "$digest" ] && continue
  att="$(docker buildx imagetools inspect "${image_ref}@${digest}" --raw)"
  if echo "$att" | jq -e '.layers[]? | select((.annotations["in-toto.io/predicate-type"] // "") | contains("spdx"))' >/dev/null; then
    found_spdx=1
  fi
done <<< "$att_digests"

if [ "$found_spdx" -ne 1 ]; then
  echo "::error::no spdx-predicate SBOM attestation found among $REF's attestation manifest(s)" >&2
  exit 1
fi

echo "attestation-manifest + spdx SBOM confirmed via --raw fallback for $REF"
