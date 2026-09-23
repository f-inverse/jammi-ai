# Sourced by the perf producers whose legs reach the compute plane's fleet
# rungs (`placed`, `shape-d`): the catalog and store those rungs run over.
#
#   plane_backends_up DIR
#       Start the pinned Postgres catalog (`pg_test_catalog.sh`) and S3-class
#       store (`s3_test_store.sh`) with their data under DIR, stop both when
#       the sourcing script exits, and export the variables the fleet reads
#       (`JAMMI_TEST_PG_URL`, `JAMMI_TEST_S3_*`, `AWS_*`).

PLANE_BACKENDS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

plane_backends_up() {
  PLANE_BACKENDS_DATA="$1"
  mkdir -p "$PLANE_BACKENDS_DATA/catalog" "$PLANE_BACKENDS_DATA/store"
  bash "$PLANE_BACKENDS_SCRIPTS/pg_test_catalog.sh" start --data "$PLANE_BACKENDS_DATA/catalog" >/dev/null
  bash "$PLANE_BACKENDS_SCRIPTS/s3_test_store.sh" start --data "$PLANE_BACKENDS_DATA/store" >/dev/null
  trap 'plane_backends_down' EXIT
  set -a
  eval "$(bash "$PLANE_BACKENDS_SCRIPTS/pg_test_catalog.sh" env)"
  eval "$(bash "$PLANE_BACKENDS_SCRIPTS/s3_test_store.sh" env)"
  set +a
}

plane_backends_down() {
  bash "$PLANE_BACKENDS_SCRIPTS/pg_test_catalog.sh" stop --data "$PLANE_BACKENDS_DATA/catalog"
  bash "$PLANE_BACKENDS_SCRIPTS/s3_test_store.sh" stop --data "$PLANE_BACKENDS_DATA/store"
}
