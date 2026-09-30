# Sourced by the perf producers whose legs reach a fleet rung (`shape-d`):
# the catalog and store those rungs run over.
#
#   fleet_backends_up DIR
#       Start the pinned Postgres catalog (`pg_test_catalog.sh`) and S3-class
#       store (`s3_test_store.sh`) with their data under DIR, stop both when
#       the sourcing script exits, and export the variables the fleet reads
#       (`JAMMI_TEST_PG_URL`, `JAMMI_TEST_S3_*`, `AWS_*`).

FLEET_BACKENDS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fleet_backends_up() {
  FLEET_BACKENDS_DATA="$1"
  mkdir -p "$FLEET_BACKENDS_DATA/catalog" "$FLEET_BACKENDS_DATA/store"
  bash "$FLEET_BACKENDS_SCRIPTS/pg_test_catalog.sh" start --data "$FLEET_BACKENDS_DATA/catalog" >/dev/null
  bash "$FLEET_BACKENDS_SCRIPTS/s3_test_store.sh" start --data "$FLEET_BACKENDS_DATA/store" >/dev/null
  trap 'fleet_backends_down' EXIT
  set -a
  eval "$(bash "$FLEET_BACKENDS_SCRIPTS/pg_test_catalog.sh" env)"
  eval "$(bash "$FLEET_BACKENDS_SCRIPTS/s3_test_store.sh" env)"
  set +a
}

fleet_backends_down() {
  bash "$FLEET_BACKENDS_SCRIPTS/pg_test_catalog.sh" stop --data "$FLEET_BACKENDS_DATA/catalog"
  bash "$FLEET_BACKENDS_SCRIPTS/s3_test_store.sh" stop --data "$FLEET_BACKENDS_DATA/store"
}
