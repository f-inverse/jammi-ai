# Reference Topologies

The same engine binary — and the same Rust crate / Python wheel for the
embedded case — serves every deployment shape below. Nothing here is a
different code path: each shape is a point on the [backend-driver
configuration surface](./philosophy.md#how-it-deploys-one-binary-pluggable-backends)
(catalog, trigger broker, object storage) plus a process count. This page
pins each shape to the concrete artifact that realises it: a config, a
tested `deploy/` Compose file, or an orchestration sketch.

## Shape A — single-process embedded

**Artifact:** the `jammi-db` / `jammi-ai` Rust crates, or the `jammi-ai`
Python wheel. No server process.

The workspace defaults ARE Shape A: SQLite catalog under `artifact_dir`,
result tables on local disk, the in-memory trigger broker, an in-process
model cache. No config file is required.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
use jammi_db::config::JammiConfig;
use jammi_ai::session::InferenceSession;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let config = JammiConfig::load(None)?; // SQLite + local fs + in-memory broker
let session = InferenceSession::new(config).await?;
# let _ = session;
# Ok(())
# }
```

```python
import jammi

# `file://` resolves to the compiled in-process engine (the `jammi-ai[embedded]`
# extra), constructed through the same `JammiConfig` resolution chain as
# `jammi-server` — via the engine's own `jammi_native.open_local`.
db = jammi.connect("file:///var/lib/jammi")
```

Notebooks, the `jammi` CLI against a local directory, single-machine batch
jobs, and laptop development all run this shape unmodified — see
[Quickstart: Rust](./quickstart-rust.md) and [Quickstart:
Python](./quickstart-python.md).

## Shape B — single-tenant server

**Artifact:** [`deploy/docker-compose.yml`](https://github.com/f-inverse/jammi-ai/blob/main/deploy/docker-compose.yml)
— one `jammi-server` process, a Postgres catalog, and a JetStream broker,
tested end to end by the `compose-smoke` workflow on every push to `main`
and nightly.

```yaml
{{#include ../../../deploy/docker-compose.yml}}
```

The `jammi-server` service's configuration is entirely environment-driven,
through the same `JAMMI_<PATH>` layer every deployment shape uses:

- `JAMMI_CATALOG__POSTGRES__URL` — the Postgres catalog connection.
- `JAMMI_BROKER__JET_STREAM__URL` — the JetStream broker connection.
- `JAMMI_SERVER__SERVICES` — `all` here (every compiled-in service tier); see
  [Service tiers](./deploy-server.md#service-tiers) for narrower selections.
- `JAMMI_AUDIT_MASTER_KEY` — read from `deploy/.env` (copy
  `deploy/.env.example`); an absent key makes Compose itself refuse to come
  up (`${JAMMI_AUDIT_MASTER_KEY:?…}`) before the container ever starts.
  Signing is not a "disabled" mode: it stays configured and, unset, only
  fails at the first audit write; a present-but-malformed key instead makes
  `jammi-server` refuse to start.

The healthcheck is exec-form `jammi-server probe` (see [`jammi-server
probe`](#the-jammi-server-probe-subcommand) below) — the runtime image is
distroless and ships no shell, so a `curl`/`wget`-based `HEALTHCHECK` is not
an option; `probe` is the same generic-CLI shape as Postgres's own
`pg_isready`.

**What the smoke proves.** `tests/compose/shape_b_remote.py` asserts
`get_server_info().broker == "jet_stream"` — the RUNTIME driver kind, which
fails if `JAMMI_BROKER__JET_STREAM__URL` were ever dropped from the compose
file, unlike the compile-time `features` list alone — registers the bundled
`patents.parquet` fixture, generates embeddings with the bundled `tiny_bert`
fixture, searches for the stored vector's own nearest neighbor (an exact
self-hit), then restarts the `jammi-server` container and repeats the same
search. It asserts durability across the restart: identical result ids and
scores within `1e-6`, and an unchanged `list_index_segments` — the segment
bundle on the Postgres-backed catalog and the `jammi-data` volume survives a
container restart, not merely that the server answers again afterward. The
workflow itself queries the `postgres` container directly afterward for the
`sources` table's row count — the matching runtime oracle on the catalog
side, failing if `JAMMI_CATALOG__POSTGRES__URL` were ever dropped.

## Shape C — multi-tenant server

**Artifact:** N `jammi-server` replicas behind a load balancer, a shared
Postgres catalog, a shared object store, and a shared JetStream (or
Postgres-as-)broker. Every replica runs the identical config; only the
process count differs from Shape B.

```toml
artifact_dir = "/var/lib/jammi"

[catalog.postgres]
url = "${POSTGRES_URL}?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
pool_size = 16
max_lifetime_secs = 1800

[broker.jet_stream]
url = "nats://${NATS_HOST}:4222"
retention_seconds = 604800
credentials = { file = "/var/run/secrets/nats.creds" }

[lease]
duration_secs = 30
heartbeat_secs = 10

[server]
services = "all"
```

The env-only equivalent (see [Deploy as a Server: a production shape,
entirely from the
environment](./deploy-server.md#a-production-shape-entirely-from-the-environment)
for the fuller walkthrough, including the object-store result root and a
file-backed audit signing key):

```bash
export JAMMI_CATALOG__POSTGRES__URL="postgres://jammi:${POSTGRES_PASSWORD}@postgres.internal:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
export JAMMI_CATALOG__POSTGRES__POOL_SIZE=16
export JAMMI_BROKER__JET_STREAM__URL="nats://nats.internal:4222"
export JAMMI_BROKER__JET_STREAM__CREDENTIALS__FILE=/run/secrets/nats.creds
export JAMMI_LEASE__DURATION_SECS=30
export JAMMI_LEASE__HEARTBEAT_SECS=10
export JAMMI_SERVER__SERVICES=all
jammi-server
```

**The guarantee.** With every replica pointed at the same Postgres catalog,
the one lease primitive documented in [Catalog Backend and Trigger
Broker: Multi-writer
safety](./catalog-and-broker.md#multi-writer-safety) holds across the whole
fleet: concurrent writers never corrupt or reap each other's building
tables; a crashed replica's rows are reclaimed at the next session boot or
by `jammi reconcile`; migrations are serialised by an advisory lock.
`[lease]` (`duration_secs` / `heartbeat_secs`, `JAMMI_LEASE__*`) is the one
timing knob every leased catalog row shares across the fleet — see
[Configuration](./configuration.md) for its full field reference.

### Kubernetes (sketch, not a shipped manifest)

Orchestration — which scheduler, how replicas are placed, ingress, TLS
termination, autoscaling — is the deployer's runtime, not the engine's (see
["How it deploys"](./philosophy.md#how-it-deploys-one-binary-pluggable-backends)).
The engine ships no Helm chart and no manifest tree; the sketch below shows
only the shape — a Deployment running the published image against the
config above, wired to the two knobs the engine itself cares about
(`readinessProbe` against `/readyz`, `runAsNonRoot`) — for a reader assembling
their own cluster's manifests.

```yaml
# sketch, not a shipped manifest -- orchestration is your runtime.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jammi-server
spec:
  replicas: 3
  selector:
    matchLabels: { app: jammi-server }
  template:
    metadata:
      labels: { app: jammi-server }
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
      containers:
        - name: jammi-server
          image: ghcr.io/f-inverse/jammi-ai-server:latest
          ports:
            - { containerPort: 8081, name: flight }
            - { containerPort: 8080, name: health }
          readinessProbe:
            httpGet: { path: /readyz, port: 8080 }
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
            periodSeconds: 10
          envFrom:
            - secretRef: { name: jammi-server-secrets } # JAMMI_CATALOG__POSTGRES__URL, JAMMI_BROKER__JET_STREAM__URL, JAMMI_AUDIT_MASTER_KEY, ...
          volumeMounts:
            - { name: config, mountPath: /etc/jammi, readOnly: true }
            - { name: scratch, mountPath: /var/lib/jammi }
      volumes:
        - name: config
          configMap: { name: jammi-server-config } # the non-secret knobs only
        - name: scratch
          emptyDir: {} # local scratch only -- the catalog/broker/result-root carry the durable state
---
apiVersion: v1
kind: Service
metadata:
  name: jammi-server
spec:
  selector: { app: jammi-server }
  ports:
    - { name: flight, port: 8081, targetPort: 8081 }
    - { name: health, port: 8080, targetPort: 8080 }
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: jammi-server-config
data:
  jammi.toml: |
    [storage]
    result_root = "s3://jammi-results/prod"

    [storage.cloud.s3]
    region = "us-east-1"

    [server]
    services = "all"
```

`jammi-server-secrets` is a `Secret` carrying the same env keys the Compose
and bare-env forms above use — `JAMMI_CATALOG__POSTGRES__URL`,
`JAMMI_BROKER__JET_STREAM__CREDENTIALS__FILE` (or `__URL`),
`JAMMI_AUDIT_MASTER_KEY` — mounted as env vars, never baked into the
ConfigMap.

## Shape D — disaggregated

**Artifact:** the query tier above (Shape C's Deployment) unchanged, plus a
second Deployment on GPU nodes running the SAME image family, scheduled
separately.

Today's branch has no dedicated worker-process driver: isolating the
training claim loop onto its own replica uses the existing service-tier
mechanism (see [Service tiers](./deploy-server.md#service-tiers)) — narrow
every query-tier replica's `[server] services` to exclude `train`, and give
the GPU-node Deployment `services = ["train"]` (`JAMMI_SERVER__SERVICES=train`)
so only it runs `TrainingService`'s claim loop against the shared catalog.

```yaml
# sketch: a second Deployment, GPU variant, GPU-node-scheduled -- the compute
# tier. Not a shipped manifest; see the Kubernetes sketch above for the
# query tier and the shared Secret/ConfigMap this Deployment reuses.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jammi-server-train
spec:
  replicas: 1
  selector:
    matchLabels: { app: jammi-server-train }
  template:
    metadata:
      labels: { app: jammi-server-train }
    spec:
      nodeSelector:
        gpu-node-pool: "true" # your cluster's own GPU node label
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
      containers:
        - name: jammi-server
          image: ghcr.io/f-inverse/jammi-ai-server-cu12:latest
          resources:
            limits: { nvidia.com/gpu: 1 }
          readinessProbe:
            httpGet: { path: /readyz, port: 8080 }
            periodSeconds: 5
          envFrom:
            - secretRef: { name: jammi-server-secrets } # same catalog/broker as the query tier
          env:
            - { name: JAMMI_SERVER__SERVICES, value: "train" }
          volumeMounts:
            - { name: config, mountPath: /etc/jammi, readOnly: true }
      volumes:
        - name: config
          configMap: { name: jammi-server-config }
```

Both `:latest` tags are re-pointed by every `v*` release tag (never by a
prerelease); the CPU `:latest` can additionally be re-pointed to the current
`main` by a manual `build-and-push-main` dispatch. Pin an exact `:vX.Y.Z`
tag for reproducible GPU-node deploys.

Very high scale, specialized GPU pools, and a split compliance posture
(query tier vs. training tier on separate node pools / network policies)
are the shapes this topology serves.

## The `jammi-server probe` subcommand

Every shape above that runs `jammi-server` — B, C, D — uses the same
readiness mechanism: `jammi-server probe [--url URL] [--config PATH]
[--timeout-secs N]` GETs `/readyz` once and exits `0` on HTTP `200`, `1`
otherwise (never following a redirect), printing the failure status and body
to stderr. Without `--url`, the target derives from the resolved `[server]
health_listen` through the identical config-resolution chain `serve` uses.
It is a generic CLI a supervisor drives from a subprocess exit code — Compose's
`HEALTHCHECK`, systemd's `ExecStartPost=`, Nomad's `check { type = "script" }`
— the same shape as Postgres's `pg_isready`. Kubernetes needs none of this:
its `readinessProbe.httpGet` already speaks HTTP directly against `/readyz`.

## What the published images can and cannot do

The published CPU image (`ghcr.io/f-inverse/jammi-ai-server`) is built with
`cargo build --features jammi-server/jetstream-broker,jammi-server/storage-cloud`
— nothing else. `GetServerInfo.features` on that image therefore reports
`["jetstream-broker"]`, and `storage_backends` reports the schemes
`storage-cloud` pulls in (`s3`, `r2`, `gs`, `azure`, alongside the always-on
`file`/`memory`).

A **Postgres catalog** (`[catalog.postgres]`, Shapes B/C/D above) works on
this image unconditionally: `sqlx`'s Postgres driver is not gated behind any
cargo feature, so it is not a `features` entry at all — it is simply always
present.

The separate `postgres` **cargo feature** (`datafusion-table-providers/postgres-federation`)
is a different capability — querying a remote Postgres database as a
federated SQL *source*, not the catalog backend — and is **not** compiled
into the published image. It would appear in `features` as `"postgres"`
only on a custom build that opts into it explicitly; do not read this
image's Postgres-catalog support as evidence that source federation is
available.

**Node architecture.** The CPU image's generic tags (`ghcr.io/f-inverse/jammi-ai-server:latest`/`:vX.Y.Z`/`:vX.Y`
and their `sha-<sha>` equivalents) are a multi-arch index — `linux/amd64` and
`linux/arm64` — so Shapes B/C's query tier and Shape D's disaggregated query
tier can schedule onto either an amd64 or an arm64 node pool without a
per-arch tag; `docker pull`/Kubernetes resolve the right member
automatically. Shape D's GPU compute tier is unaffected by this: the CUDA
image (`-cu12`) is `linux/amd64` only, so the GPU node pool stays amd64. The
same CPU image name's self-contained tags (`:selfcontained`,
`:selfcontained-sha-<sha>`) are also `linux/amd64` only — never schedule
those onto an arm64 node pool.
