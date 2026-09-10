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

# `--format {{json .Provenance}}` / `{{json .SBOM}}` print the JSON literal
# `{}` -- not `null`, not the empty string -- when the accessor's attestation
# is absent, so presence must be a shape test, not a non-emptiness test on
# the printed string. A multi-platform attestation is keyed by "os/arch"
# (each value itself a non-empty object); a single-platform attestation is a
# flat non-empty object. Either way, an absent attestation prints `{}`.
non_empty_attestation() {
  jq -e '
    def is_platform_map: (type == "object") and (length > 0) and (to_entries | all(.value | type == "object"));
    if is_platform_map then
      (to_entries | all(.value | (type == "object") and (length > 0)))
    else
      (type == "object") and (length > 0)
    end
  ' >/dev/null 2>&1
}

provenance_ok=0
sbom_ok=0
if [ -n "$provenance" ] && printf '%s' "$provenance" | non_empty_attestation; then
  provenance_ok=1
fi
if [ -n "$sbom" ] && printf '%s' "$sbom" | non_empty_attestation; then
  sbom_ok=1
fi

if [ "$provenance_ok" -eq 1 ] && [ "$sbom_ok" -eq 1 ]; then
  echo "provenance and SBOM confirmed via imagetools .Provenance/.SBOM accessors for $REF"
  exit 0
fi

if [ "$provenance_ok" -eq 1 ] && [ "$sbom_ok" -eq 0 ]; then
  echo "::notice::provenance confirmed but SBOM missing (or accessor unavailable) via imagetools for $REF -- falling back to --raw + jq"
elif [ "$provenance_ok" -eq 0 ] && [ "$sbom_ok" -eq 1 ]; then
  echo "::notice::SBOM confirmed but provenance missing (or accessor unavailable) via imagetools for $REF -- falling back to --raw + jq"
else
  echo "::notice::imagetools .Provenance/.SBOM accessors unavailable on this runner's buildx for $REF -- falling back to --raw + jq"
fi

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
