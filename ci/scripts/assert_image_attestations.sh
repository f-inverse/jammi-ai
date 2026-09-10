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
# `.Provenance`/`.SBOM` accessors on a sufficiently recent buildx; on such a
# buildx the command exits 0 and prints the JSON literal `{}` -- not `null`,
# not the empty string -- when the attestation itself is absent. A `{}` is
# therefore a REAL finding (the attestation is missing), so it fails closed
# immediately here and never falls through to the fallback below. The
# fallback is reserved for the one case the accessor genuinely can't speak
# to: a buildx too old to recognize `.Provenance`/`.SBOM`, which errors
# (nonzero exit) rather than returning `{}`. There the raw OCI index is read
# directly, and BOTH an spdx SBOM predicate and a SLSA provenance predicate
# must be found among the attestation manifests -- checking only the SBOM
# predicate previously let a provenance-less image pass the fallback arm.
set -euo pipefail

if [ -z "${REF:-}" ]; then
  echo "::error::REF is unset -- nothing to inspect" >&2
  exit 1
fi

# `--format {{json .Provenance}}` / `{{json .SBOM}}` print the JSON literal
# `{}` when the attestation is absent, so presence must be a shape test, not
# a non-emptiness test on the printed string. A multi-platform attestation is
# keyed by "os/arch" (each value itself a non-empty object); a
# single-platform attestation is a flat non-empty object. `is_platform_map`
# is the single source of truth for "looks like a per-platform map": it
# fails on ANY non-object or empty-object value, so a malformed or partially
# missing per-platform map cannot slip past as if it were a flat attestation.
non_empty_attestation() {
  jq -e '
    def is_platform_map: (type == "object") and (length > 0) and
      (to_entries | all(.value | (type == "object") and (length > 0)));
    if is_platform_map then
      true
    else
      (type == "object") and (length > 0)
    end
  ' >/dev/null 2>&1
}

provenance=""
provenance_rc=0
provenance="$(docker buildx imagetools inspect "$REF" --format '{{json .Provenance}}' 2>/dev/null)" || provenance_rc=$?

sbom=""
sbom_rc=0
sbom="$(docker buildx imagetools inspect "$REF" --format '{{json .SBOM}}' 2>/dev/null)" || sbom_rc=$?

if [ "$provenance_rc" -eq 0 ] && [ "$sbom_rc" -eq 0 ]; then
  provenance_ok=0
  sbom_ok=0
  printf '%s' "$provenance" | non_empty_attestation && provenance_ok=1
  printf '%s' "$sbom" | non_empty_attestation && sbom_ok=1

  if [ "$provenance_ok" -eq 1 ] && [ "$sbom_ok" -eq 1 ]; then
    echo "provenance and SBOM confirmed via imagetools .Provenance/.SBOM accessors for $REF"
    exit 0
  fi

  # Both accessors ran and answered -- a positively empty `{}` is a real
  # finding, not an "accessor unavailable" signal, so this fails closed
  # immediately instead of falling through to the --raw fallback below.
  [ "$provenance_ok" -eq 1 ] || echo "::error::provenance attestation missing via imagetools .Provenance accessor for $REF" >&2
  [ "$sbom_ok" -eq 1 ] || echo "::error::SBOM attestation missing via imagetools .SBOM accessor for $REF" >&2
  exit 1
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
found_provenance=0
while IFS= read -r digest; do
  [ -z "$digest" ] && continue
  att="$(docker buildx imagetools inspect "${image_ref}@${digest}" --raw)"
  if echo "$att" | jq -e '.layers[]? | select((.annotations["in-toto.io/predicate-type"] // "") | contains("spdx"))' >/dev/null; then
    found_spdx=1
  fi
  if echo "$att" | jq -e '.layers[]? | select((.annotations["in-toto.io/predicate-type"] // "") | contains("slsa.dev/provenance"))' >/dev/null; then
    found_provenance=1
  fi
done <<< "$att_digests"

if [ "$found_spdx" -ne 1 ] || [ "$found_provenance" -ne 1 ]; then
  [ "$found_spdx" -eq 1 ] || echo "::error::no spdx-predicate SBOM attestation found among $REF's attestation manifest(s)" >&2
  [ "$found_provenance" -eq 1 ] || echo "::error::no slsa.dev/provenance-predicate attestation found among $REF's attestation manifest(s)" >&2
  exit 1
fi

echo "attestation-manifest + spdx SBOM + SLSA provenance confirmed via --raw fallback for $REF"
