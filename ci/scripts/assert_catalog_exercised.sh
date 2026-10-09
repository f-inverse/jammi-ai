#!/usr/bin/env bash
# A deploy shape's Postgres catalog was exercised at runtime: after the smoke
# registered a source, the `sources` table holds a row. An empty table means
# the shape's catalog URL is not backing the catalog, however green the smoke
# itself read.
#
#   bash ci/scripts/assert_catalog_exercised.sh EXEC...
#
# EXEC... runs a command inside the shape's Postgres (e.g. `docker compose
# -f ... exec -T postgres`, or `kubectl -n ns exec statefulset/postgres --`).
set -euo pipefail

[ "$#" -gt 0 ] || { echo "usage: $0 EXEC..." >&2; exit 2; }
count="$("$@" psql -U jammi -d jammi -tAc 'select count(*) from sources')"
echo "postgres catalog sources rows: $count"
[ "$count" -ge 1 ] \
  || { echo "::error::no row in the Postgres 'sources' table after add_source: the shape's JAMMI_CATALOG__POSTGRES__URL is not backing the catalog" >&2; exit 1; }
