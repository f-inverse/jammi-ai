#!/usr/bin/env bash
# Asserts that a `docker buildx imagetools` manifest-list JSON carries
# EXACTLY a requested platform set, plus at least one `unknown/unknown`
# attestation-referrer entry per platform. Gates the promotion of a
# consumer-facing tag: `_ci-base-image.yml`'s `merge-manifest` job runs this
# TWICE per invocation -- once against a `docker buildx imagetools create
# --dry-run`'s output BEFORE anything is pushed (a failing assertion here
# means `:latest`/`sha-<sha>` never moves), and once more against
# `docker buildx imagetools inspect <tag> --format '{{json .Manifest}}'`
# right after the real, pushing `imagetools create` runs, to confirm the
# push matched what the dry run predicted. Both commands return the same
# top-level OCI index JSON shape (`{"manifests": [...]}`), verified
# empirically against a live registry.
#
# The `unknown/unknown` entries are `docker/build-push-action`'s default
# provenance attestation on a public repo, but ONLY when the build ran under
# the buildx `docker-container` driver (`docker/setup-buildx-action`); the
# default `docker` driver silently omits them, which is why every leg that
# feeds this assertion's sources must run `setup-buildx-action` first.
#
# Usage:
#   check_merged_index_platforms.sh <comma-separated-platforms>
#     (manifest-list JSON on stdin; exits 0 on match, 1 on any mismatch)
#   check_merged_index_platforms.sh --self-test
#     (drives the assertion above against synthetic fixtures, no docker, no
#     network; exits 0 iff every fixture's exit code matches what it
#     should be)
set -euo pipefail

# The real assertion. Reads the manifest JSON from stdin, takes the wanted
# platform set as $1. Returns (not exits, so --self-test can call this
# without killing its own shell under `set -e`) 0 on match, 1 otherwise.
_assert() {
  local platforms="$1"
  local manifest
  manifest="$(cat)"

  local want want_count got unknown_count
  want="$(printf '%s\n' "${platforms//,/$'\n'}" | sort)"
  want_count="$(printf '%s\n' "${platforms//,/$'\n'}" | grep -c .)"

  # The real (os != unknown) platform manifests must equal the requested set
  # exactly.
  got="$(jq -r '.manifests[] | select(.platform.os != "unknown") | "\(.platform.os)/\(.platform.architecture)"' <<< "$manifest" | sort)"
  if [ "$got" != "$want" ]; then
    echo "check-merged-index-platforms: platform set mismatch (want: [$want]; got: [$got])" >&2
    return 1
  fi

  # A merge that silently dropped attestations would still pass the
  # platform-set check above, so assert this too.
  unknown_count="$(jq '[.manifests[] | select(.platform.os == "unknown")] | length' <<< "$manifest")"
  if [ "$unknown_count" -lt "$want_count" ]; then
    echo "check-merged-index-platforms: expected at least $want_count unknown/unknown provenance entries (one per platform), found $unknown_count" >&2
    return 1
  fi

  echo "check-merged-index-platforms: OK -- platform set [$want] present, $unknown_count/$want_count provenance entries"
  return 0
}

# Four fixtures, hermetic (no docker, no network): the two PASS shapes and
# the two FAIL shapes production actually hits, plus the single-platform
# shape `image-cuda.yml`'s one-arch caller exercises. Each asserts the
# EXACT exit code `_assert` returns, never just "did it print something".
_self_test() {
  local failures=0
  local rc

  local good_2plat='{"manifests":[
    {"platform":{"architecture":"amd64","os":"linux"}},
    {"platform":{"architecture":"arm64","os":"linux"}},
    {"platform":{"architecture":"unknown","os":"unknown"}},
    {"platform":{"architecture":"unknown","os":"unknown"}}
  ]}'
  rc=0
  _assert "linux/amd64,linux/arm64" <<< "$good_2plat" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[good-2-platform]: OK (exit 0, expected 0)"
  else
    echo "self-test[good-2-platform]: FAIL (exit $rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  local missing_attest='{"manifests":[
    {"platform":{"architecture":"amd64","os":"linux"}},
    {"platform":{"architecture":"arm64","os":"linux"}}
  ]}'
  rc=0
  _assert "linux/amd64,linux/arm64" <<< "$missing_attest" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[missing-attestation]: OK (exit 1, expected 1)"
  else
    echo "self-test[missing-attestation]: FAIL (exit $rc, expected 1)" >&2
    failures=$((failures + 1))
  fi

  # Requested amd64 only; the index actually carries both -- a mismatched
  # platform SET, not merely a missing attestation.
  rc=0
  _assert "linux/amd64" <<< "$good_2plat" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "self-test[mismatched-platform-set]: OK (exit 1, expected 1)"
  else
    echo "self-test[mismatched-platform-set]: FAIL (exit $rc, expected 1)" >&2
    failures=$((failures + 1))
  fi

  local good_1plat='{"manifests":[
    {"platform":{"architecture":"amd64","os":"linux"}},
    {"platform":{"architecture":"unknown","os":"unknown"}}
  ]}'
  rc=0
  _assert "linux/amd64" <<< "$good_1plat" > /dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "self-test[single-platform-good]: OK (exit 0, expected 0)"
  else
    echo "self-test[single-platform-good]: FAIL (exit $rc, expected 0)" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "check-merged-index-platforms --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "check-merged-index-platforms --self-test: all 4 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

platforms="${1:?usage: check_merged_index_platforms.sh <comma-separated-platforms> (manifest JSON on stdin) | check_merged_index_platforms.sh --self-test}"
_assert "$platforms"
