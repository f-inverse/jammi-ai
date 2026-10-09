#!/usr/bin/env bash
# Every backend the distributed harness needs is provisioned. The harness
# early-returns to a silent pass when any of these is unset
# (`Backends::from_env_or_skip`); on a lane that provisions them all, a
# missing variable is a misconfigured lane, never a legitimate skip, so it
# fails here rather than reporting a green run that exercised nothing.
#
#   bash ci/scripts/require_distributed_backends.sh
set -euo pipefail

missing=0
for var in JAMMI_TEST_PG_URL JAMMI_TEST_S3_ENDPOINT JAMMI_TEST_S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "::error::$var is empty: the distributed lane would skip silently (hollow green)" >&2
    missing=1
  fi
done
exit "$missing"
