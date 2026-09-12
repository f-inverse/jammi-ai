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

## Shutdown: DRAIN and RELEASE

`jammi-server` has two shutdown modes (PostgreSQL's mapping): **SIGTERM =
DRAIN** — the in-flight job finishes, the lease keeps renewing, every epoch
bundle lands, then the process exits 0 — and **SIGINT = RELEASE** — every job
lease is handed back to the catalog at once (the row stays `running` with a
NULL lease and `releases + 1`; a compute job's building-table lease with it),
the loop stops and the process exits **0 when its own evidence confirms
every lease was handed back, or exit code 3 when it does not** (see
`docs/guide/src/deploy-server.md`'s RELEASE section), so a successor claims
a CONFIRMED release (exit 0) within one `[worker] idle_poll_secs` at no
attempt cost (`releases` offsets it in the `attempts - releases` cap). A
DEGRADED release (exit 3) does not universally cost an attempt, and a
degraded determinant is defined by MISSING evidence — it gets a definite
consequence only where the evidence establishes one, never a single
universal outcome: when the sweep statement for that lease's own table
itself failed, the lease was never written, falls to the expiry path, and
costs one attempt (`attempts + 1`, `releases` untouched) for `jobs`, or a
one-time back-off for the linked `result_tables` row (no `attempts`/
`releases` columns there to increment); but when the sweep confirms and only the
hold-observation or stop-witness evidence is missing, every row the sweep
itself matched was already handed back (`releases + 1`, lease NULLed) and a
successor claims it within one idle poll at no attempt cost — nothing is
established about a row still under an active hold, or about a claim that
commits after the sweep runs, and either of those keeps a live lease that
falls to the expiry path instead. Any signal while
draining is a RELEASE. There is no engine-side timeout: the pod's
`terminationGracePeriodSeconds` bounds a DRAIN, then SIGKILL.

**Operative rule:** the grace must cover one epoch's wall time, or the drain
never lands its final bundle before SIGKILL and the job takes the expiry
path — one `[lease] duration_secs` window before a successor requeues it,
and one attempt consumed. The compute overlay ships **600 s**: the
cluster-autoscaler's `--max-graceful-termination-sec` default, the upper
anchor a scale-down honours; a larger value is honoured only by rollouts and
`kubectl delete`, so raise both together. The Shape C base keeps the
Kubernetes default (30 s) — its query replicas hold no training job.

**Rollout arithmetic** at 3 compute replicas: the Deployment defaults
(`maxSurge` 25% rounds up to 1, `maxUnavailable` 25% rounds down to 0) make
the rollout serial, so the worst case is 3 × 600 s = 30 min of drains;
`maxSurge: 100%` / `maxUnavailable: 0` drains all three at once — 10 min at
double GPU demand. Caps a DRAIN cannot cross: kubelet graceful node shutdown
is off by default (0 s); AWS Spot gives a 2-minute interruption notice; GCP
Spot ≤ 30 s. On spot capacity use RELEASE instead — the job is claimable at
once and no attempt burns:

```yaml
lifecycle:
  preStop:
    exec:
      # The uniform actuator: sends SIGINT to pid 1. Works on every image.
      command: ["/usr/local/bin/jammi-server", "release"]
```

```yaml
lifecycle:
  preStop:
    exec:
      # cu12 only — that image is ubi8 and has a shell; the CPU images are
      # distroless and do not.
      command: ["/bin/sh", "-c", "kill -INT 1"]
```

A `preStop` hook runs before SIGTERM is sent and inside the grace countdown.
Alternatives: a derived image with `STOPSIGNAL SIGINT`; `lifecycle.stopSignal`
once the `ContainerStopSignals` feature gate leaves alpha; and, outside
Kubernetes, `docker kill --signal=INT` / `docker compose kill -s SIGINT`.

**Autoscaling** input: `jammi_jobs_queued{kind}` on `/metrics` (a worker
process samples it from the catalog every `[worker] metrics_sample_secs`) is
the HPA/KEDA signal for the compute tier; `jammi_worker_jobs_in_flight` says
whether a replica is busy.

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
