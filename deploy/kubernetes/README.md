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
- **`overlays/shape-d/`** — the query tier with the Ballista client role
  (`jammi-query.toml` replaces the base ConfigMap: `[ballista.client]`
  pointed at `jammi-server-scheduler:50050`, so a batch statement the tier
  receives runs on the compute tier) plus the compute tier: a GPU-scheduled `StatefulSet`
  (`jammi-server-compute`, `replicas: 2`) behind a headless `Service`
  (`clusterIP: None`), running the `cu12` image with `[worker] enabled =
  true` claiming the training job kinds and `nvidia.com/gpu: 2` per pod
  (one device per `[worker] local_ranks`), PLUS a single-replica CPU
  scheduler `Deployment` (`jammi-server-scheduler`) behind a plain
  `Service`. Each `jammi-server-compute` pod's `peer_advertise` is its own
  stable DNS name under the headless Service
  (`<pod>.jammi-server-compute.<namespace>.svc.cluster.local`), so a rank's
  peer identity survives a pod restart — the property a plain `Deployment`
  cannot hold; the same name is its Ballista `advertise_host`, since each
  pod also registers as a Ballista executor with `jammi-server-scheduler`.
  This overlay admits single-pod gangs (`W ≤ 2`, `Local` on one pod's two
  devices); a cross-pod `Peer` gang of world `W` needs `W > local_ranks`,
  `max_world_size ≥ W` on the submit edge (`base/jammi.toml`; the key is
  read only where jobs are enqueued), and at least `W` compute pods able
  to hold a rank (the coordinator's included). This overlay is
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

**Rollout arithmetic** for the `jammi-server-compute` `StatefulSet` (2
replicas): a `StatefulSet`'s `RollingUpdate` has no `maxSurge`;
`maxUnavailable` exists only behind the alpha `MaxUnavailableStatefulSet`
feature gate — assume strictly serial, one ordinal at a time in descending
order (pod `-1` first, then pod `-0`), each wait for the
PREVIOUS ordinal's own DRAIN (≤ 600 s) to finish before it is touched, so
the worst case is strictly serial: 2 × 600 s = 20 min of drains for this
overlay's 2 replicas. `rollingUpdate.partition` (unset here, default `0`)
is the only lever that changes this — a nonzero partition pins every
ordinal AT OR ABOVE it to the old spec, useful for a canary rollout of the
highest ordinal alone. `podManagementPolicy: Parallel` governs SCALE
up/down only (replicas created or deleted without waiting on a sibling); it
does not change a `RollingUpdate`'s own strictly-ordered, one-at-a-time
replacement. The single-replica `jammi-server-scheduler` `Deployment`
claims no job and runs no task, so it has nothing to drain: `maxSurge: 25%`
rounds up to 1 (the Deployment default), so a fresh scheduler pod starts
before the old one stops, and the two briefly serve the SAME
`[ballista.scheduler]`-hosted role behind the same Service (executor
registrations and slot counts are read from the shared catalog, never a
scheduler-local cache that a rollout could split). A plan in flight on the
old scheduler when it stops fails at its submitter, whose own path re-runs
it — a materialization in the replica that received it, a training attempt
through the lease reclaim. Caps a DRAIN cannot cross: kubelet graceful
node shutdown is off by default (0 s); AWS Spot gives a 2-minute
interruption notice; GCP Spot ≤ 30 s. On spot capacity use RELEASE instead
— the job is claimable at once and no attempt burns:

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
Kubernetes, `docker kill --signal=INT` / `docker compose kill -s SIGINT` — note
that the daemon records a kill as a manual stop, so a `restart` policy does not
bring that container back; signal the process from the host (`kill -INT` on
`docker inspect`'s `.State.Pid`) when the policy is meant to restart it.

**Autoscaling** input: `jammi_jobs_queued{kind}` on `/metrics` (a worker
process samples it from the catalog every `[worker] metrics_sample_secs`) is
the HPA/KEDA signal for the compute tier; `jammi_worker_jobs_in_flight` says
whether a replica is busy.

## Compute plane

Three roles under `[ballista]` (`docs/guide/src/configuration.md`). The
query tier (`jammi-server`, `[worker] enabled = false`) is a CLIENT of the
scheduler: a `CREATE TABLE … AS` over Flight SQL or a materialization a
verb builds runs its plan on the compute tier when a live executor holds
every device kind it requires, in the replica otherwise; a `SELECT` never
leaves the replica.

A claimed training attempt — `fine_tune`, `graph_fine_tune` or
`context_predictor`, of any world size — is submitted the same way when its
claimant holds the client role: as one Ballista task, bound to an executor
other than the claimant that lists the claimant's OWN device kind, and
trained in the claimant's process when no live executor does
(byte-identical either way, which is why the kind must match). That rule
decides who claims here: the query tier and the scheduler are CPU pods and
the executors list `cuda`, so an attempt claimed on either would train
there. The compute pods claim the training kinds, and nothing else does:

- **`jammi-server-scheduler`** (a single-replica CPU `Deployment`):
  `[ballista.scheduler]` set (advertised as its Service name, so an
  executor's task-status report dials the Service, never the pod's `0.0.0.0`
  bind), `[worker] enabled = false`, no `[ballista.executor]`, no
  `[ballista.client]`. It binds submitted tasks to executors; it claims no
  job, runs no task and submits nothing of its own.
- **`jammi-server-compute`** (the GPU `StatefulSet`): `[worker] enabled =
  true` claiming `["fine_tune", "graph_fine_tune", "context_predictor"]`,
  and `[ballista.executor]` pointed at `jammi-server-scheduler`'s Service.
  Each pod trains the jobs it claims in-process under its own `[worker]
  local_ranks` topology, and runs the tasks the scheduler binds to it. A
  placed training attempt and the pod's own claim take the same job slot,
  and run the same body.

DRAIN on a `jammi-server-compute` pod stops Ballista task admission at once
— the executor reports `Terminating` to the scheduler the instant DRAIN
begins, before the pod's in-flight worker job is joined, a terminating
executor is never bound, and a training attempt the pod is still dialled
with inside its grace is refused before any claim transfer — but WAITS for any in-flight
placed task to finish before the pod itself stops — the same "finish what's
running, refuse what's new" shape DRAIN already gives an in-process claim.
RELEASE tears the executor down immediately regardless of an in-flight
placed task: the scheduler's own `ExecutorLost` handling and jammi's
`transfer_claim` guard (never Ballista's own task retry, which is pinned at
zero — `task_max_failures = 0`) are what put the job back in front of a
successor.

## Image pin advice

`:latest` is re-pointed by every `v*` release tag. Pin an exact `:X.Y.Z`
tag for reproducible deploys — this applies to both the CPU image
(`base/deployment.yaml`) and the GPU image
(`overlays/shape-d/statefulset-compute.yaml`).

## Notes

- Every kustomization in this tree declares `generatorOptions:
  {disableNameSuffixHash: true}` — the Deployments reference their
  ConfigMap by a static name, so a generated hash suffix would break the
  reference.
