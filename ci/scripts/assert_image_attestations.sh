#!/usr/bin/env bash
# Asserts a pushed image digest carries both a provenance attestation and an
# SBOM attestation, and -- when the index is multi-platform -- that BOTH
# attestations actually cover every platform the index carries, not just
# some of them. Shared by every push job in server-image.yml so the
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
#
# Expected shape (primary-accessor path only): the two possible shapes --
# a multi-platform per-platform map keyed "os/arch[/variant]" (one entry
# per platform BuildKit attested) versus the FLAT single-platform predicate
# object real buildx emits, e.g. `{"SLSA":{...}}` / `{"SPDX":{...}}` -- are
# structurally indistinguishable by key inspection alone: both are "a
# non-empty object whose values are themselves non-empty objects"
# (`is_platform_map`'s test). A single-platform push's flat `{"SLSA":{...}}`
# passes that structural test exactly as a genuine per-platform map does, so
# `is_platform_map` on its own cannot decide which shape an attestation OWES.
# The decision instead comes from the index's OWN platform count, read once
# via `imagetools inspect --raw`'s `.manifests[]` (excluding
# attestation-manifest entries): >=2 platforms means the index is genuinely
# multi-platform and the per-platform map is required, checked for EXACT
# key-set equality against the index's platform set; 0 or 1 platforms means
# the flat predicate object is required, and a would-be per-platform map
# whose key set happens to equal that single platform (the shape mistakenly
# handed to a single-platform index) is rejected. A merge that lands an
# attested arm64 leg alongside an unattested amd64 leg previously passed
# here: the arm64 leg's non-empty provenance/SBOM satisfied the old "is
# there SOMETHING" test with no check that every index platform had its OWN
# entry. `--self-test` drives both the shape decision and the key-set
# comparison against synthetic JSON, no docker, no registry, no network
# required.
#
# Usage:
#   REF=<registry>/<image>@sha256:<digest> assert_image_attestations.sh
#   assert_image_attestations.sh --self-test
set -euo pipefail

# Structural test ONLY: a non-empty object all of whose values are
# themselves non-empty objects. This shape is worn by BOTH a genuine
# per-platform map (keyed "os/arch[/variant]") AND the real flat
# single-platform predicate object (e.g. `{"SLSA":{...}}`) -- the two are
# indistinguishable by structure alone, which is exactly why
# `assert_attestation_shape` below never uses this function to decide
# WHICH shape is owed; it uses it only to confirm a value already known
# (from the index's own platform count) to owe the per-platform map
# actually has that structure. Fails on ANY non-object or empty-object
# value, so a malformed or partially missing per-platform map cannot slip
# past. `non_empty_attestation` below composes with this for a pure
# presence check (rejects only the literal `{}` buildx prints for an
# absent attestation), not a shape-correctness check.
is_platform_map() {
  jq -e '
    (type == "object") and (length > 0) and
      (to_entries | all(.value | (type == "object") and (length > 0)))
  ' > /dev/null 2>&1
}

# `--format {{json .Provenance}}` / `{{json .SBOM}}` print the JSON literal
# `{}` when the attestation is absent, so presence must be a shape test, not
# a non-emptiness test on the printed string. A platform-keyed map passes via
# `is_platform_map`; otherwise a flat single-platform attestation must be a
# non-empty object.
non_empty_attestation() {
  local input
  input="$(cat)"
  if printf '%s' "$input" | is_platform_map; then
    return 0
  fi
  printf '%s' "$input" | jq -e '(type == "object") and (length > 0)' > /dev/null 2>&1
}

# Parses a raw OCI index (`imagetools inspect --raw`'s stdout, on stdin)
# into the index's OWN platform set: one "os/arch[/variant]" string per
# line, sorted and de-duplicated, EXCLUDING attestation-manifest entries
# (those carry a synthetic "unknown/unknown" platform, never a real leg).
# Pure jq, no docker -- separated from `index_platform_set` below so
# --self-test can drive it with a synthetic raw index, no registry
# required.
parse_index_platforms() {
  jq -r '
    .manifests[]?
    | select((.annotations["vnd.docker.reference.type"] // "") != "attestation-manifest")
    | .platform
    | select(. != null)
    | if ((.variant // "") != "") then "\(.os)/\(.architecture)/\(.variant)" else "\(.os)/\(.architecture)" end
  ' | sort -u
}

# The real (network-touching) read: `$1` is the same `<repo>@sha256:<digest>`
# ref the caller inspects for provenance/SBOM.
index_platform_set() {
  docker buildx imagetools inspect "$1" --raw | parse_index_platforms
}

# Given a platform-map attestation JSON (`$1`, already confirmed non-empty
# via `is_platform_map`) and the index's own platform set (`$2`, one
# "os/arch[/variant]" per line) and a label (`$3`, for the error message),
# fails if the map's key set is not EXACTLY the index's platform set --
# printing every platform missing from the map and every platform the map
# names that the index does not have. Pure jq/comm, no docker -- so
# --self-test can drive it with synthetic fixtures, including the exact
# partial-map shape (an attested arm64 leg, an unattested amd64 leg) that
# previously passed silently.
platform_keys_match() {
  local map_json="$1" index_platforms="$2" label="$3"
  local map_keys missing extra
  map_keys="$(printf '%s' "$map_json" | jq -r 'keys[]' | sort -u)"
  missing="$(comm -23 <(printf '%s\n' "$index_platforms") <(printf '%s\n' "$map_keys"))"
  extra="$(comm -13 <(printf '%s\n' "$index_platforms") <(printf '%s\n' "$map_keys"))"
  if [ -n "$missing" ] || [ -n "$extra" ]; then
    if [ -n "$missing" ]; then
      echo "::error::${label} attestation is missing platform(s) the index carries: $(printf '%s' "$missing" | tr '\n' ' ')" >&2
    fi
    if [ -n "$extra" ]; then
      echo "::error::${label} attestation names platform(s) the index does not carry: $(printf '%s' "$extra" | tr '\n' ' ')" >&2
    fi
    return 1
  fi
  return 0
}

# Decides WHICH shape a single attestation ($1) owes and enforces it,
# using the index's OWN platform set ($2, one "os/arch[/variant]" per
# line -- the same ground truth `platform_keys_match` compares against) as
# the determinant, never the attestation's own key spelling:
#
#   >=2 index platforms (genuinely multi-platform): the per-platform map
#   is required. `is_platform_map` confirms the structure, then
#   `platform_keys_match` requires the key set to be EXACTLY the index's
#   platform set.
#
#   0 or 1 index platforms (a single-platform push, or a ref that isn't an
#   index at all): the FLAT predicate object real buildx emits is
#   required -- a non-empty object, full stop. The one thing it must NOT
#   be is the per-platform map mistakenly applied to a single-platform
#   index: if the attestation's own key set is exactly the index's (one
#   entry) platform set, that IS the per-platform shape, wrongly handed to
#   a flat single-platform image, and is rejected by name.
assert_attestation_shape() {
  local attestation="$1" index_platforms="$2" label="$3"
  local platform_count
  platform_count="$(printf '%s\n' "$index_platforms" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [ "$platform_count" -ge 2 ]; then
    if ! printf '%s' "$attestation" | is_platform_map; then
      echo "::error::${label} attestation is missing platform(s) the index carries: $(printf '%s' "$index_platforms" | tr '\n' ' ')" >&2
      return 1
    fi
    platform_keys_match "$attestation" "$index_platforms" "$label"
    return $?
  fi

  if ! printf '%s' "$attestation" | jq -e '(type == "object") and (length > 0)' > /dev/null 2>&1; then
    echo "::error::${label} attestation is not a non-empty flat predicate object for a single-platform image" >&2
    return 1
  fi

  if [ -n "$index_platforms" ]; then
    local att_keys
    att_keys="$(printf '%s' "$attestation" | jq -r 'keys[]' | sort -u)"
    if [ "$att_keys" = "$(printf '%s\n' "$index_platforms" | sort -u)" ]; then
      echo "::error::${label} attestation for a single-platform image is shaped as a per-platform map (keyed \"$(printf '%s' "$att_keys" | tr '\n' ' ')\") instead of a flat predicate object" >&2
      return 1
    fi
  fi
  return 0
}

_self_test() {
  local failures=0

  if printf '%s' '{"linux/amd64":{"a":1},"linux/arm64":{"b":2}}' | is_platform_map; then
    echo "self-test[is-platform-map-true]: OK"
  else
    echo "self-test[is-platform-map-true]: FAIL" >&2
    failures=$((failures + 1))
  fi

  if printf '%s' '{"predicateType":"x"}' | is_platform_map; then
    echo "self-test[is-platform-map-flat-rejected]: FAIL (flat object accepted as a platform map)" >&2
    failures=$((failures + 1))
  else
    echo "self-test[is-platform-map-flat-rejected]: OK"
  fi

  if printf '%s' '{}' | is_platform_map; then
    echo "self-test[is-platform-map-empty-rejected]: FAIL" >&2
    failures=$((failures + 1))
  else
    echo "self-test[is-platform-map-empty-rejected]: OK"
  fi

  if printf '%s' '{"linux/amd64":{},"linux/arm64":{"b":1}}' | is_platform_map; then
    echo "self-test[is-platform-map-empty-value-rejected]: FAIL (a platform key with an empty value must not count as attested)" >&2
    failures=$((failures + 1))
  else
    echo "self-test[is-platform-map-empty-value-rejected]: OK"
  fi

  # The REAL flat shape buildx emits for a single-platform attestation --
  # `{"SLSA":{...}}` -- not a fixture with scalar values the producer never
  # emits. This is structurally identical to a per-platform map
  # (`is_platform_map` returns true for it too), which is exactly why
  # presence and shape-correctness are two separate checks: this fixture
  # only proves buildx's real output is recognized as PRESENT.
  local real_flat_provenance
  real_flat_provenance='{"SLSA":{"predicateType":"https://slsa.dev/provenance/v1","predicate":{"buildDefinition":{"buildType":"https://mobyproject.org/buildkit@v1"}}}}'
  if printf '%s' "$real_flat_provenance" | non_empty_attestation; then
    echo "self-test[non-empty-attestation-flat]: OK"
  else
    echo "self-test[non-empty-attestation-flat]: FAIL" >&2
    failures=$((failures + 1))
  fi

  if printf '%s' '{}' | non_empty_attestation; then
    echo "self-test[non-empty-attestation-empty-rejected]: FAIL" >&2
    failures=$((failures + 1))
  else
    echo "self-test[non-empty-attestation-empty-rejected]: OK"
  fi

  # A raw index carrying two real legs and one attestation-manifest entry
  # (synthetic "unknown/unknown" platform): the attestation-manifest entry
  # must be excluded from the parsed platform set.
  local raw_two_leg parsed
  raw_two_leg='{"manifests":[
    {"platform":{"os":"linux","architecture":"amd64"}},
    {"platform":{"os":"linux","architecture":"arm64"}},
    {"annotations":{"vnd.docker.reference.type":"attestation-manifest"},"platform":{"os":"unknown","architecture":"unknown"}}
  ]}'
  parsed="$(printf '%s' "$raw_two_leg" | parse_index_platforms)"
  if [ "$parsed" = "$(printf 'linux/amd64\nlinux/arm64')" ]; then
    echo "self-test[parse-index-platforms-excludes-attestation-manifest]: OK"
  else
    echo "self-test[parse-index-platforms-excludes-attestation-manifest]: FAIL (got: $parsed)" >&2
    failures=$((failures + 1))
  fi

  # A variant-bearing platform (e.g. linux/arm/v7) must render with its
  # variant suffix, not collapse to "linux/arm".
  local raw_variant parsed_variant
  raw_variant='{"manifests":[{"platform":{"os":"linux","architecture":"arm","variant":"v7"}}]}'
  parsed_variant="$(printf '%s' "$raw_variant" | parse_index_platforms)"
  if [ "$parsed_variant" = "linux/arm/v7" ]; then
    echo "self-test[parse-index-platforms-variant]: OK"
  else
    echo "self-test[parse-index-platforms-variant]: FAIL (got: $parsed_variant)" >&2
    failures=$((failures + 1))
  fi

  local index_two full_map partial_map out rc

  index_two="$(printf 'linux/amd64\nlinux/arm64')"
  full_map='{"linux/amd64":{"a":1},"linux/arm64":{"b":2}}'
  if platform_keys_match "$full_map" "$index_two" "provenance" > /dev/null 2>&1; then
    echo "self-test[platform-keys-match-full-coverage]: OK"
  else
    echo "self-test[platform-keys-match-full-coverage]: FAIL" >&2
    failures=$((failures + 1))
  fi

  # The exact bug this closes: an attestation map covering only ONE of the
  # index's two legs (the attested-arm64/unattested-amd64 shape a live
  # merge produced) must fail, naming the missing platform.
  partial_map='{"linux/arm64":{"b":2}}'
  rc=0
  out="$(platform_keys_match "$partial_map" "$index_two" "provenance" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'linux/amd64'; then
    echo "self-test[platform-keys-match-partial-names-missing]: OK"
  else
    echo "self-test[platform-keys-match-partial-names-missing]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  # An attestation map naming a platform the index does NOT carry must also
  # fail (strict set equality, not "index is a subset of the map").
  local extra_map
  extra_map='{"linux/amd64":{"a":1},"linux/arm64":{"b":2},"linux/riscv64":{"c":3}}'
  rc=0
  out="$(platform_keys_match "$extra_map" "$index_two" "SBOM" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'linux/riscv64'; then
    echo "self-test[platform-keys-match-extra-names-unexpected]: OK"
  else
    echo "self-test[platform-keys-match-extra-names-unexpected]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  # assert_attestation_shape: the index's OWN platform count -- not the
  # attestation's key spelling -- decides which shape is owed. All six
  # combinations below drive the actual function the script calls, not
  # just its building blocks.

  local index_one
  index_one="linux/amd64"

  # single-platform flat -> pass: the real `{"SLSA":{...}}` shape is
  # exactly what a single-platform image owes.
  if assert_attestation_shape "$real_flat_provenance" "$index_one" "provenance" > /dev/null 2>&1; then
    echo "self-test[assert-shape-single-platform-flat-pass]: OK"
  else
    echo "self-test[assert-shape-single-platform-flat-pass]: FAIL" >&2
    failures=$((failures + 1))
  fi

  # single-platform given a map -> fail: an attestation keyed EXACTLY by
  # the index's one real platform is the per-platform shape, wrongly
  # applied where a flat predicate object belongs.
  local single_given_map
  single_given_map='{"linux/amd64":{"a":1}}'
  rc=0
  out="$(assert_attestation_shape "$single_given_map" "$index_one" "provenance" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[assert-shape-single-platform-given-map-fail]: OK"
  else
    echo "self-test[assert-shape-single-platform-given-map-fail]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  # multi-platform flat -> fail: the exact bug this closes -- the real
  # flat `{"SLSA":{...}}` shape handed to a genuinely multi-platform index
  # must fail, naming every platform it doesn't cover.
  rc=0
  out="$(assert_attestation_shape "$real_flat_provenance" "$index_two" "provenance" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'linux/amd64' && printf '%s' "$out" | grep -q 'linux/arm64'; then
    echo "self-test[assert-shape-multi-platform-flat-fail]: OK"
  else
    echo "self-test[assert-shape-multi-platform-flat-fail]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  # multi-platform map exact -> pass.
  if assert_attestation_shape "$full_map" "$index_two" "provenance" > /dev/null 2>&1; then
    echo "self-test[assert-shape-multi-platform-map-exact-pass]: OK"
  else
    echo "self-test[assert-shape-multi-platform-map-exact-pass]: FAIL" >&2
    failures=$((failures + 1))
  fi

  # multi-platform map missing one -> fail, naming it.
  rc=0
  out="$(assert_attestation_shape "$partial_map" "$index_two" "provenance" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'linux/amd64'; then
    echo "self-test[assert-shape-multi-platform-map-missing-fail]: OK"
  else
    echo "self-test[assert-shape-multi-platform-map-missing-fail]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  # multi-platform map with an extra platform -> fail, naming it.
  rc=0
  out="$(assert_attestation_shape "$extra_map" "$index_two" "SBOM" 2>&1)" || rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'linux/riscv64'; then
    echo "self-test[assert-shape-multi-platform-map-extra-fail]: OK"
  else
    echo "self-test[assert-shape-multi-platform-map-extra-fail]: FAIL (rc=$rc, out=$out)" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "assert-image-attestations --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "assert-image-attestations --self-test: all 17 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

if [ -z "${REF:-}" ]; then
  echo "::error::REF is unset -- nothing to inspect" >&2
  exit 1
fi

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

  if [ "$provenance_ok" -ne 1 ] || [ "$sbom_ok" -ne 1 ]; then
    # Both accessors ran and answered -- a positively empty `{}` is a real
    # finding, not an "accessor unavailable" signal, so this fails closed
    # immediately instead of falling through to the --raw fallback below.
    [ "$provenance_ok" -eq 1 ] || echo "::error::provenance attestation missing via imagetools .Provenance accessor for $REF" >&2
    [ "$sbom_ok" -eq 1 ] || echo "::error::SBOM attestation missing via imagetools .SBOM accessor for $REF" >&2
    exit 1
  fi

  # Both attestations are non-empty (a literal `{}` was already rejected
  # above). Which SHAPE each one owes -- the per-platform map or the flat
  # single-platform predicate object -- is decided by the index's OWN
  # platform count, read once here and never by either attestation's key
  # spelling: an attested arm64 leg beside an unattested amd64 leg is
  # non-empty (passes the check above) but covers only half a genuinely
  # multi-platform index; a real single-platform push's flat
  # `{"SLSA":{...}}` must not be forced through the per-platform key-set
  # check its structure superficially resembles.
  index_platforms="$(index_platform_set "$REF")"
  if [ -z "$index_platforms" ]; then
    echo "::error::could not read $REF's own platform set from the raw index -- refusing to confirm attestation coverage" >&2
    exit 1
  fi

  platform_ok=1
  assert_attestation_shape "$provenance" "$index_platforms" "provenance" || platform_ok=0
  assert_attestation_shape "$sbom" "$index_platforms" "SBOM" || platform_ok=0

  if [ "$platform_ok" -eq 1 ]; then
    echo "provenance and SBOM confirmed via imagetools .Provenance/.SBOM accessors for $REF"
    exit 0
  fi
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
