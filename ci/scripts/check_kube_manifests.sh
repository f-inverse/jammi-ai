#!/usr/bin/env bash
# Every Kubernetes manifest a kustomization renders is valid against the
# pinned Kubernetes schema: `kustomize build | kubeconform --strict`, so an
# unknown key or a wrong-typed field fails here, not at `kubectl apply`.
#
#   bash ci/scripts/check_kube_manifests.sh KUSTOMIZATION_DIR...
#
# The schema version and the schema repository commit are pinned here, once:
# the manifest-validity guard in `ci.yml` and the kind smoke, which validates
# its overlay before applying it, both run this. Fetched schemas are cached
# under KUBECONFORM_CACHE (default `$HOME/.cache/kubeconform`); kubeconform
# refuses a missing cache directory, so it is created first.
set -euo pipefail

KUBERNETES_VERSION=1.34.11
SCHEMA_COMMIT=b582a12a09aa9b5d1edad577a964c504130214fc
cache="${KUBECONFORM_CACHE:-$HOME/.cache/kubeconform}"

[ "$#" -gt 0 ] || { echo "usage: $0 KUSTOMIZATION_DIR..." >&2; exit 2; }
mkdir -p "$cache"
for dir in "$@"; do
  echo "::group::$dir"
  kustomize build "$dir" | kubeconform --strict --summary \
    --kubernetes-version "$KUBERNETES_VERSION" \
    -schema-location "https://raw.githubusercontent.com/yannh/kubernetes-json-schema/${SCHEMA_COMMIT}/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json" \
    -cache "$cache" \
    -
  echo "::endgroup::"
done
