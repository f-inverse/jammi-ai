#!/usr/bin/env bash
# The service images a deploy shape runs beside, at the digests
# `ci/service-images.env` pins, under the names the deploy files use: each
# pinned reference is pulled and tagged with its own name and tag, so a
# compose file or a manifest that names `postgres:16` runs the pinned image
# where this ran (compose uses a present image; `kind load docker-image`
# loads it). `ci_image.py check` holds the deploy files to those names.
#
#   bash ci/scripts/pull_service_images.sh
#
# Prints each `name:tag` it provided, one per line.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
while IFS='=' read -r key ref; do
  case "$key" in ""|\#*) continue ;; esac
  name="${ref%%@*}"
  docker pull --quiet "$ref" >/dev/null
  docker tag "$ref" "$name"
  echo "$name"
done < ci/service-images.env
