#!/usr/bin/env bash
# Every pod of a kind-deployed workload runs the image this host built: each
# pod's kubelet-reported imageID equals the local image's id, or one of the
# repoDigests containerd holds for the loaded image. A pulled or stale image
# fails by name; an empty pod list fails rather than passing vacuously.
#
#   bash ci/scripts/assert_kind_image_identity.sh IMAGE NAMESPACE SELECTOR [KIND_NODE]
#
# `kind load docker-image` registers only a tag reference, so containerd's
# repoDigests is usually empty and kubelet reports the bare config digest;
# the equality arm is the expected one.
set -euo pipefail

image="${1:?IMAGE}" namespace="${2:?NAMESPACE}" selector="${3:?SELECTOR}" node="${4:-kind-control-plane}"

built="sha256:$(docker image inspect --format '{{.Id}}' "$image" | sed 's/^sha256://')"
digests="$(docker exec "$node" crictl inspecti -o json "docker.io/library/$image" | jq -r '.status.repoDigests[]')"
ids="$(kubectl -n "$namespace" get pod -l "$selector" -o jsonpath='{.items[*].status.containerStatuses[*].imageID}')"
if [ -z "$ids" ]; then
  echo "::error::no pod matching $selector reported an imageID -- an empty match would pass vacuously" >&2
  exit 1
fi
for id in $ids; do
  if [ "$id" = "$built" ] || grep -qx "$id" <<<"$digests"; then
    echo "pod imageID $id is the built image ($built) or one of its repoDigests"
  else
    echo "::error::pod imageID $id is neither the built image ($built) nor one of its repoDigests ($digests)" >&2
    exit 1
  fi
done
