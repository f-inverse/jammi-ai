# Reference Topologies

The same engine binary — and the same Rust crate / Python wheel for the
embedded case — serves every deployment shape below. Nothing here is a
different code path: each shape is a point on the [backend-driver
configuration surface](./philosophy.md#how-it-deploys-one-binary-pluggable-backends)
(catalog, trigger broker, object storage) plus a process count. This page
pins each shape to the concrete artifact that realises it: a config, a
tested `deploy/` Compose file, or a kustomize manifest tree.

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

The published ports (`8081`, `8080`) are bound to `127.0.0.1`, not
`0.0.0.0`: the compose file publishes them for a TLS-terminating proxy
running on the same host to reach, never for direct exposure to an
untrusted network (see [Security Posture](./security.md#transport-encryption-is-the-deployers-runtime-not-the-engines)).

The healthcheck is exec-form `jammi-server probe` (see [`jammi-server
probe`](#the-jammi-server-probe-subcommand) below) — the runtime image is
distroless and ships no shell, so a `curl`/`wget`-based `HEALTHCHECK` is not
an option; `probe` is the same generic-CLI shape as Postgres's own
`pg_isready`.

**What the smoke proves.** `tests/compose/remote_smoke.py` is the shared
smoke oracle; `shape_b_remote.py` (this Compose shape) and
`shape_c_kube_remote.py` (the Kubernetes shape below) are its two drivers,
differing only in how they restart the server and in what they assert
afterwards — durability on the Compose volume here, the shared catalog on
the emptyDir pods there. The oracle asserts
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
services = ["core", "event", "eval"]

[worker]
enabled = false
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

### Kubernetes (deploy/kubernetes)

Orchestration — which scheduler, how replicas are placed, ingress, TLS
termination, autoscaling — is the deployer's runtime, not the engine's (see
["How it deploys"](./philosophy.md#how-it-deploys-one-binary-pluggable-backends)).
`deploy/kubernetes/base` is the whole shape the engine cares about — a
Deployment with `readinessProbe` against `/readyz` and `runAsNonRoot`, a
Service, a ConfigMap of the non-secret knobs, an `emptyDir` for scratch;
ingress, TLS, autoscaling, network policy and each cloud's managed-service
annotations are the deployer's overlay, and the seam is a kustomize patch on
the Deployment/Service metadata, never a change to the engine's knobs. Every
PR validates `kustomize build` + `kubeconform --strict --kubernetes-version
1.34.11` over every kustomization in the tree, in `ci.yml`'s `Guard
(kubernetes manifests)`; the `kube-smoke` workflow additionally stands the
`ci` overlay up on a real `kind` cluster on push to `main`, nightly, on
manual dispatch, and on any pull request that touches
`deploy/kubernetes/**` or `tests/compose/**`.

```yaml
{{#include ../../../deploy/kubernetes/base/deployment.yaml}}
```

```yaml
{{#include ../../../deploy/kubernetes/base/service.yaml}}
```

```toml
{{#include ../../../deploy/kubernetes/base/jammi.toml}}
```

`jammi-server-secrets` is a `Secret` carrying the same env keys the Compose
and bare-env forms above use — `JAMMI_CATALOG__POSTGRES__URL`,
`JAMMI_BROKER__JET_STREAM__CREDENTIALS__FILE` (or `__URL`),
`JAMMI_AUDIT_MASTER_KEY` — mounted as env vars, never baked into the
ConfigMap. No `Secret` manifest ships in git; create it out-of-band, once per
cluster namespace:

```bash
kubectl -n <namespace> create secret generic jammi-server-secrets \
  --from-literal=JAMMI_AUDIT_MASTER_KEY=<...> \
  --from-literal=JAMMI_CATALOG__POSTGRES__URL=<...> \
  --from-literal=JAMMI_BROKER__JET_STREAM__URL=<...>
```

never in git.

**What `kube-smoke` proves.** Against the `ci` overlay on a real `kind`
cluster: readiness (`kubectl rollout status`), `get_server_info().broker ==
"jet_stream"`, the Postgres `sources` table's row count via `psql` against
the `postgres` StatefulSet, one-hop image identity — every pod's
`containerStatuses[].imageID` traces back to the image `kind load`ed, never
a registry pull — and a `rollout restart` after which the new pod still
sees the source the pod it replaced registered. What it does NOT prove:
durability across that restart — the scratch volume is an `emptyDir`, so the
Postgres catalog and the JetStream broker, not the pod's local disk, carry
the state a fresh pod recovers.

## Shape D — disaggregated

**Artifact:** the query tier above (Shape C's Deployment) unchanged, plus a
second Deployment on GPU nodes running the SAME image family, scheduled
separately.

Running jobs is not a service tier (see [Service
tiers](./deploy-server.md#service-tiers)): whether a process *claims and
executes* the jobs it accepted is `[worker] enabled`. Every query-tier
replica runs `[worker] enabled = false` (`JAMMI_WORKER__ENABLED=false`) —
it still mounts `core`/`event`/`eval` and accepts every submission — and the
GPU-node Deployment runs `[worker] enabled = true` (`JAMMI_WORKER__ENABLED=true`,
optionally `JAMMI_WORKER__KINDS='["fine_tune", "graph_fine_tune", "context_predictor"]'`
to claim only the training kinds) so only it runs the job worker's claim
loop against the shared catalog. Its `[server] services` is whatever the
compute node should also serve — `services = []` for a pure compute node.

The compute tier's Deployment carries `terminationGracePeriodSeconds: 600`
— SIGTERM drains (the in-flight training job finishes, every epoch bundle
lands) and SIGKILL follows the grace; SIGINT, or `jammi-server release` from
a `preStop` hook, RELEASES — on a CONFIRMED release (exit 0), the job's
lease is handed back at once and any other replica claims it within one
idle poll at no attempt cost; a DEGRADED release (exit 3) does not cost an
attempt universally, and its per-lease outcome depends on which determinant
degraded (see the RELEASE breakdown in `deploy/kubernetes/README.md` and
`deploy-server.md` below — never assume the CONFIRMED cost here for a
degraded exit). The grace must cover one epoch's wall time; on spot
capacity use RELEASE. The operative rule, the rollout arithmetic and both
`preStop` recipes are in `deploy/kubernetes/README.md` ("Shutdown: DRAIN
and RELEASE"); the modes themselves are in
[Shutdown](./deploy-server.md#shutdown-drain-and-release).

The compute tier is a plain Deployment today and is **provisional**:
[#500](https://github.com/f-inverse/jammi-ai/issues/500) decides the gang
primitive for multi-GPU and multi-node training; once ranks need stable
per-rank identity and ordered startup, this overlay becomes a `StatefulSet`
or an indexed `Job`. The Shape C base is unaffected. This overlay is
validated by `kubeconform` only — CI has no GPU node.

```yaml
{{#include ../../../deploy/kubernetes/overlays/shape-d/deployment-compute.yaml}}
```

```toml
{{#include ../../../deploy/kubernetes/overlays/shape-d/jammi-compute.toml}}
```

Both `:latest` tags are re-pointed by every `v*` release tag (never by a
prerelease); the CPU `:latest` can additionally be re-pointed to the current
`main` by a manual `build-and-push-main` dispatch. Pin an exact `:vX.Y.Z`
tag for reproducible GPU-node deploys.

Very high scale, specialized GPU pools, and a split compliance posture
(query tier vs. training tier on separate node pools / network policies)
are the shapes this topology serves.

### Beyond-one-node retrieval

A query-tier replica answers `Search` over a table whose ANN index segments
it does not all hold by fanning the query out to the replicas that own
them: each owner searches its segments and returns `(row_id, distance)`
hits — never vectors — and the coordinator merges them under the same total
order a single node merges its own segments with (the distributed data
plane; see [Security Posture](./security.md#the-peer-listener-i-peer) for
the I-PEER invariant and [Operability](./operability.md#failure-mode-matrix)
for the failure ladder). Three facts fix the shape:

- **The default is `AllLocal`.** With `[server] peer_bind` unset — every
  shape above — every segment is this replica's, there is no third listener,
  and the search is exactly the single-node search (the same kernels, the
  same bytes, the same exact-read count at every segment count). Nothing on
  this page changes until a deployment opts in.
- **`peer_bind` makes a replica an owner.** Setting `[server] peer_bind`
  (`JAMMI_SERVER__PEER_BIND`) opens the internal `PeerService` listener on
  that replica. Bind it on a private interface behind network policy / mTLS
  from the runtime — its clients are other jammi coordinators and it
  authenticates nothing itself.
- **Precondition: a shared, replica-readable `result_root`.** A segment an
  owner serves must be a bundle every replica can reach: `[storage]
  result_root` on an object store every replica reads (a local
  `artifact_dir` is one node's). Placement — which replica owns which
  segment — is derived at query time from the live replica ring
  (`[server] peer_advertise` + the catalog's `instances` rows, rendezvous-
  hashed over `(table, segment)`), never declared; the membership half of
  that ring is the compute-tier substrate's (`peer_advertise`,
  `instances.peer_addr`), and until it lands a library process supplies an
  explicit `StaticPlacement` through
  `InferenceSession::open_with_placement`. Batch builders (the neighbor
  graph, eval) never fan out: they load the whole table's segment set on the
  building replica.

**A REFRESHED table's `Mixed` arm is not version-aware.** The single-node
(`AllLocal`, every segment this replica's own) search path always resolves a
versioned table's CURRENT version before searching it. The multi-node
`Mixed` arm — reached only when `peer_bind` is set and this table's segments
span more than one replica — does not: it plans off `list_index_segments`'
flat, unversioned segment set, the same limitation the single-node path
carried before its own version-aware resolution was added. If a Shape D
deployment places a table `refresh_embeddings` has since published a new
version of across more than one owning replica, a `Search` served through
peers can surface rows from a version older than the table's current one;
keep a refreshed table's segments on a single owning replica (or force-local
it) until this closes.

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

The separate `postgres` **cargo feature** (`datafusion-table-providers/postgres`)
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
