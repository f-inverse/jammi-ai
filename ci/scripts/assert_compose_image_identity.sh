#!/usr/bin/env bash
# The container a compose service runs is the image this job built, by image
# id: a smoke that drove a stale image would prove nothing about the bytes
# it packaged.
#
#   COMPOSE_FILE=a.yml:b.yml bash ci/scripts/assert_compose_image_identity.sh IMAGE SERVICE
set -euo pipefail

image="${1:?IMAGE}"
service="${2:?SERVICE}"

built_id="$(docker image inspect --format '{{.Id}}' "$image")"
container_id="$(docker compose ps -q "$service")"
test -n "$container_id" || { echo "::error::compose service '$service' has no container" >&2; exit 1; }
running_id="$(docker inspect --format '{{.Image}}' "$container_id")"
[ "$built_id" = "$running_id" ] \
  || { echo "::error::the running container's image ($running_id) is not the image this job built ($built_id)" >&2; exit 1; }
