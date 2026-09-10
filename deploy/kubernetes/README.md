# `deploy/kubernetes` — kustomize reference manifests

This is a reference shape for running `jammi-server` on Kubernetes, not a
production-ready chart: no Ingress, no TLS, no HorizontalPodAutoscaler, no
NetworkPolicy, no operator, no Helm, and no per-cloud overlay. Orchestration
— which scheduler, how replicas are placed, ingress, TLS termination,
autoscaling, network policy, each cloud's managed-service annotations — is
the deployer's runtime, not the engine's (see
`docs/guide/src/philosophy.md`'s "How it deploys"). `base/` is the whole
shape the engine cares about; anything beyond it is a kustomize patch on
this Deployment/Service's metadata, never a change to the engine's own
knobs.

## Layout

- **`base/`** — Shape C, the multi-tenant query tier: a `Deployment`
  (`replicas: 3`), a `Service`, and a `ConfigMap` carrying `jammi.toml`'s
  non-secret knobs. Every replica runs `[worker] enabled = false` — it
  accepts every job submission but claims none.
- **`overlays/shape-d/`** — the compute tier: a GPU-scheduled `Deployment`
  (`jammi-server-compute`) running the `cu12` image, `[worker] enabled =
  true` claiming the training job kinds. **PROVISIONAL**: a plain
  `Deployment` today. #500 (multi-GPU / multi-node gangs as engine
  mechanism) decides the gang primitive; once ranks need stable per-rank
  identity and ordered startup this becomes a `StatefulSet` or an indexed
  `Job`. The Shape C base is unaffected.
  https://github.com/f-inverse/jammi-ai/issues/500. This overlay is
  kubeconform-validated only — no GPU node is available in CI, so it never
  runs a real pod there.
- **`overlays/ci/`** — the kind smoke's stack: upstream `postgres:16` and
  `nats:2.10-alpine` `StatefulSet`s alongside the base query tier, pinned to
  the image the workflow already built and `kind load`ed. Never a production
  shape.

## Secrets

No `Secret` manifest ships in git. Create it out-of-band, once per cluster
namespace, with the three env keys the Deployment's `envFrom` names
(the same keys `deploy/docker-compose.yml` sets):

```
kubectl -n <namespace> create secret generic jammi-server-secrets \
  --from-literal=JAMMI_AUDIT_MASTER_KEY=<...> \
  --from-literal=JAMMI_CATALOG__POSTGRES__URL=<...> \
  --from-literal=JAMMI_BROKER__JET_STREAM__URL=<...>
```

`base/jammi.toml`'s `result_root = "s3://jammi-results/prod"` is an example, not a
literal to deploy unchanged. The S3 driver reads its credentials from the
process environment, not from this config file (`AmazonS3Builder::from_env`,
`crates/jammi-db/src/config/mod.rs`'s `CloudSection::S3` doc) — add
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` to the same Secret above when
using an S3 (or R2) result root.

## Config precedence

Env wins over the `ConfigMap`: the `JAMMI_<PATH>` env layer resolves after
the baked `/etc/jammi/jammi.toml`, so an operator can override any non-secret
knob without editing the `ConfigMap`. Secrets never belong in the
`ConfigMap` — only the `Secret` above.

## Validating locally

Every kustomization in this tree is proven the same way — build it, then
validate every resource against the pinned Kubernetes schema:

```
kustomize build deploy/kubernetes/base | kubeconform --strict --summary --kubernetes-version 1.34.11 -
kustomize build deploy/kubernetes/overlays/shape-d | kubeconform --strict --summary --kubernetes-version 1.34.11 -
kustomize build deploy/kubernetes/overlays/ci | kubeconform --strict --summary --kubernetes-version 1.34.11 -
```

Fetch the pinned binaries (never a distro package, never a third-party
action): kustomize `kustomize/v5.8.1`, kubeconform `v0.8.0`. `kubeconform`
never inspects `ConfigMap.data` — the `jammi.toml` bodies are proven by the
guide's own fence test against the real config loader
(`docs/guide/src/reference-topologies.md`), not by this validation.

## Image pin advice

`:latest` is re-pointed by every `v*` release tag. Pin an exact `:vX.Y.Z`
tag for reproducible deploys — this applies to both the CPU image
(`base/deployment.yaml`) and the GPU image
(`overlays/shape-d/deployment-compute.yaml`).

## Notes

- Every kustomization in this tree declares `generatorOptions:
  {disableNameSuffixHash: true}` — the Deployments reference their
  ConfigMap by a static name, so a generated hash suffix would break the
  reference.
