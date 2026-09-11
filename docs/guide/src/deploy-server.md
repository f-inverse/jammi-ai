# Deploy as a Server

Jammi can run as an Arrow Flight SQL server, making all registered sources and embedding tables queryable from any Arrow-compatible client. Use this when multiple services, BI tools, or non-Rust/Python consumers need to query Jammi's data.

## The workflow

The server is a **read path**. Deploy the server, then set up your data through
it — with the `jammi` CLI (a strict gRPC client) or the library — so other
systems can query it:

```bash
# 1. Start the server
jammi-server

# 2. Register sources against the running server with the CLI
jammi --target grpc://127.0.0.1:8081 \
  sources add patents --url /data/patents.parquet --format parquet

# 3. Generate embeddings (library or Python — not available over Flight SQL)
python3 -c '
import jammi
db = jammi.connect("file:///var/lib/jammi")
db.generate_embeddings(source="patents", model="sentence-transformers/all-MiniLM-L6-v2", columns=["abstract"], key="id", modality="text")
'
```

## Connecting with Arrow Flight SQL

### Python (pyarrow)

```python
from pyarrow.flight import FlightClient, FlightDescriptor

client = FlightClient("grpc://localhost:8081")

# Run a SQL query
info = client.get_flight_info(
    FlightDescriptor.for_command(b"SELECT id, title, year FROM patents.public.patents WHERE year > 2020")
)
reader = client.do_get(info.endpoints[0].ticket)
table = reader.read_all()
print(table.to_pandas())
```

### Query embedding tables

Embedding tables are registered in DataFusion and queryable via SQL:

```python
# List all embedding tables
info = client.get_flight_info(
    FlightDescriptor.for_command(b"SELECT table_name FROM information_schema.tables WHERE table_schema = 'jammi'")
)

# Query vectors directly
info = client.get_flight_info(
    FlightDescriptor.for_command(b"SELECT _row_id, _model_id FROM \"jammi.patents__embedding__all-MiniLM-L6-v2__20260325\" LIMIT 10")
)
```

### JDBC

Flight SQL is compatible with JDBC drivers that support the Arrow Flight SQL protocol, enabling access from Java applications, BI tools (Superset, DBeaver, Tableau), and SQL editors.

## Server configuration

```toml
[server]
flight_listen = "0.0.0.0:8081"
preload_models = ["sentence-transformers/all-MiniLM-L6-v2"]

[logging]
level = "info"
format = "json"    # structured logging for production
```

### Preloading models

Models listed in `preload_models` are loaded into the cache at startup,
BEFORE `/readyz` reports ready — it answers `503 {"status":"not_ready",
"detail":"preloading i/n"}` meanwhile, `/healthz` stays `200` (no
`startupProbe` needed), and this process's claim loop waits at its gate
with its `workers.state` row reading `warming`, so no job is claimed by a
cold process. An entry is a bare id, whose task comes from the catalog's
`models` row, or `{ id, task }` naming the task explicitly — required for a
`local:` path, which has no row. A listed model that cannot load, a bare id
with no `models` row, or an unknown task token is a **startup error**: the
server exits non-zero instead of serving. A shutdown signal during the
preload exits 0 without ever serving.

```toml
[server]
preload_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    { id = "local:/models/bge-small", task = "text_embedding" },
]
```

## Service tiers

One server binary scales to many deployment shapes by mounting only the gRPC
service tiers a deployment needs — no per-shape rebuild. The **core** tier is
always mounted: `CatalogService` (the control plane — tenant binding, the
`GetServerInfo` handshake, and source / model / channel / mutable-table /
topic administration), `EmbeddingService`, `InferenceService`,
`PipelineService`, `AuditService`, `JobService` (durable job submission — every
deployment accepts a job and reports its status), and the Flight SQL surface.
Two optional tiers are runtime-selectable via `[server] services`:

| Tier | Service | Role |
|---|---|---|
| `event` | `TriggerService` | topic / publish / subscribe streams |
| `eval`  | `EvalService` | per-query evaluation arrays |

```toml
[server]
services = "all"             # all-in-one: every tier (the default)
# services = ["event"]       # serve + event box
# services = []              # serve-only: core tier only
```

Running jobs is not a tier. Whether a process *claims and executes* the jobs
it accepted is `[worker] enabled` (see [Configuration](configuration.md)):
a request node runs `[worker] enabled = false` and still accepts every
submission; a compute node runs `services = []` with `[worker] enabled =
true, kinds = [...]` and works what the request nodes queued.

A deployment advertises exactly the tiers it mounted over the wire, so a client
can negotiate capability before calling a verb:

```python
info = db.get_server_info()
# {"version": "...", "features": [...], "storage_backends": [...],
#  "services": ["core", "eval", "event"]}
if "eval" in info["services"]:
    db.eval_per_query(...)
```

Reaching a verb whose tier was **not** mounted returns a truthful `Unimplemented`
("not enabled on this deployment") rather than a misleading success — the
service-mount analog of the client's build-by-capability `connect(target)`.

**Runtime config, no compile features.** Every tier compiles into every
build — no cargo feature gates a tier, so there is no compile ceiling for the
selection to hit; a token naming no tier is a startup error, not a silent
drop. Override the selection with `JAMMI_SERVER__SERVICES` (`all`, or a
comma-separated token list — empty for serve-only).

## GPU configuration

For GPU-accelerated inference in production:

```toml
[gpu]
device = 0            # CUDA device index
memory_limit = "auto"
memory_fraction = 0.9
require_gpu = false   # fail fast if the GPU is unavailable instead of CPU fallback

[inference]
batch_size = 64
max_loaded_models = 3
```

Set `gpu.device = -1` for CPU-only deployment. On a GPU build, an unavailable device degrades to CPU with a warning by default; set `gpu.require_gpu = true` to fail fast instead.

## Environment variable overrides

Every config field can be overridden with environment variables, useful for
containerized deployments: `JAMMI_<PATH>`, path segments joined by `__`, works
for every field in the tree — not a hand-picked subset — and a bare
`JAMMI_<FIELD>` (no `__`) works for a top-level one. See
[Configuration](./configuration.md#environment-variable-overrides) for the
full rule.

```bash
JAMMI_SERVER__FLIGHT_LISTEN=0.0.0.0:9081 \
JAMMI_GPU__DEVICE=-1 \
JAMMI_LOGGING__FORMAT=json \
jammi-server
```

### A production shape, entirely from the environment

A Postgres catalog, a JetStream broker, an S3 result root, and a file-backed
audit signing key — Shape C's stack — need no TOML file at all: every field
resolves from `JAMMI_*` variables through the same layered loader.

```bash
export JAMMI_CATALOG__POSTGRES__URL="postgres://jammi:${POSTGRES_PASSWORD}@postgres.internal:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
export JAMMI_CATALOG__POSTGRES__POOL_SIZE=16
export JAMMI_BROKER__JET_STREAM__URL="nats://nats.internal:4222"
export JAMMI_BROKER__JET_STREAM__CREDENTIALS__FILE=/run/secrets/nats.creds
export JAMMI_STORAGE__RESULT_ROOT="s3://jammi-results/prod"
export JAMMI_STORAGE__CLOUD__S3__REGION=us-east-1
export JAMMI_SIGNING_KEY__FILE__PATH=/run/secrets/jammi-audit-master-key
export JAMMI_MODELS__HUB_TOKEN__FILE=/run/secrets/hf-token
export JAMMI_SERVER__SERVICES=all
jammi-server
```

The equivalent TOML file — the two are interchangeable, and either one
overrides the other's fields when both are present:

```toml
[catalog.postgres]
url = "postgres://jammi:${POSTGRES_PASSWORD}@postgres.internal:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
pool_size = 16

[broker.jet_stream]
url = "nats://nats.internal:4222"
credentials = { file = "/run/secrets/nats.creds" }

[storage]
result_root = "s3://jammi-results/prod"

[storage.cloud.s3]
region = "us-east-1"

[signing_key.file]
path = "/run/secrets/jammi-audit-master-key"

[models]
hub_token = { file = "/run/secrets/hf-token" }

[server]
services = "all"
```

## Health, readiness, and metrics

The server exposes three HTTP side-channel endpoints on port `8080`:

```bash
curl http://localhost:8080/healthz
# {"status":"ok","version":"0.8.0"}

curl http://localhost:8080/readyz
# {"status":"ready"}

curl http://localhost:8080/metrics
# jammi_grpc_requests_total 0
# jammi_flight_queries_total 0
# jammi_eval_invocations_total 0
# jammi_search_latency_seconds_bucket{...} 0
```

`/healthz` is a liveness probe — `200` while the process can keep its
leases and its claim loop alive; `503 {"status":"unhealthy","lease_keeper":
false|true,"claim_loop":"…"}` when the lease keeper thread is dead (every
lease this process holds is lost) or the claim loop task panicked
(`claim_loop: "failed"`). A stopped or aborted loop, a process with no loop,
and a DRAIN in progress are all `200` — liveness decides restarts, never
routing, and there is no slow-step detection (the runtime owns "how long is
too long"). `/readyz` is a readiness probe — `200` means the catalog backend
responded; `503` means it didn't, or the server is draining (`"detail":
"draining"`), and traffic should be drained from this instance. Point your
load balancer at `/readyz`.

`/metrics` exposes a small, substrate-level set of Prometheus counters
(gRPC requests, Flight SQL queries, eval invocations, refusals at the
`[server.limits]` edge) plus a search-latency histogram, and five gauges:

| Gauge | Present on | Source |
|---|---|---|
| `jammi_jobs_queued{kind}` | worker-enabled processes | the catalog, sampled every `[worker] metrics_sample_secs` (default 5) by a dedicated task — one `GROUP BY kind, status` per tick, never on a scrape; a held (`claimable = false`) row counts as queued |
| `jammi_jobs_running{kind}` | worker-enabled processes | the same sample |
| `jammi_worker_jobs_in_flight` | worker-enabled processes | loop-claimed jobs running under a live lease hold (0 or 1); an inline `run_now` is never counted |
| `jammi_worker_claim_loop_up` | worker-enabled processes | 1 while the claim loop task runs, 0 once it stopped, aborted or failed |
| `jammi_lease_heartbeat_age_seconds` | every process | seconds since the lease keeper last completed a renewal pass |

A process without a claim loop omits the worker families (absent, never 0).
No gauge carries a tenant label. `jammi_jobs_queued{kind}` is the
autoscaling input for a compute tier; a scrape issues no catalog statement.

## What the server can and cannot do

| Operation | Available over Flight SQL? | Available over typed gRPC? |
|-----------|--------------------------|----------------------------|
| SQL queries on source tables | Yes | — (use Flight SQL) |
| SQL queries on embedding tables | Yes | — (use Flight SQL) |
| Joins, aggregations, filters | Yes | — (use Flight SQL) |
| Generate embeddings | No — use library or Python package | Yes — `EmbeddingService.GenerateEmbeddings` |
| Semantic vector search | No — use library or Python package | Yes — `EmbeddingService.Search` |
| Inference | No — use library or Python package | Yes — `InferenceService.Infer` |
| Fine-tuning (and graph / context-predictor training) | No — use library or Python package | Yes — `JobService.SubmitJob` (core; runs where `[worker] enabled`) |
| Context-predictor prediction | No — use library or Python package | Yes — `InferenceService.Predict` |
| Evaluation | No — use library or Python package | Yes — `EvalService` (eval tier) |

The Flight SQL surface is a **query** interface (read path); the ML operations are not SQL, so they ride the **typed gRPC** surface instead. Set up your data and run training/inference through the Rust library, the `jammi-ai` / `jammi-client` Python package, or — for a remote engine — those same verbs over gRPC, then query the results over Flight SQL. The CLI is a strict gRPC client that registers sources and drives the admin surfaces against a running server; it carries no ML verbs and does not run the engine in-process.

The typed gRPC surface is what an edge runtime speaks (it has no HTTP/2 client for Flight SQL's bidirectional streaming). `EmbeddingService` serves `AddSource`, `GenerateEmbeddings`, `EncodeQuery`, and `Search` over plain gRPC — and, since tonic-web is mounted, over **gRPC-web** — so an edge function running the engine as a sidecar can ingest, encode, **and** search without the library. `Search` accepts a precomputed vector or an existing `row_key` (query-by-example, with the vector resolved inside the engine); see [Semantic Search](./semantic-search.md#search-over-grpc-edge-runtimes). `JobService` (core, always mounted) serves all three training kinds over gRPC and `InferenceService.Predict` serves a trained context predictor — so a client can offload training and prediction to a GPU server with the same verb surface the embedded engine exposes.

## Shutdown: DRAIN and RELEASE

The server has two shutdown modes, PostgreSQL's mapping: **SIGTERM = DRAIN**
(smart) and **SIGINT = RELEASE** (fast); any signal received while draining
is a RELEASE. There is no engine-side drain timeout — the runtime's grace
period (`terminationGracePeriodSeconds`, `stop_grace_period`) bounds a
DRAIN, then SIGKILL.

**DRAIN** (`kill -TERM`, `docker stop`, a Kubernetes pod deletion):
`/readyz` flips to `503 {"status":"not_ready","detail":"draining"}`; the
listener closes and in-flight requests finish; every idle `WaitJob` /
`Subscribe` stream is ended with `UNAVAILABLE` "server draining" (counted
under `jammi_grpc_refused_total{reason="draining"}`); the embedded worker
finishes the job it is running — its lease keeps renewing, every epoch
bundle lands, the job reaches `completed` under the same attempt — and
claims no more. Then the catalog is released and the process exits 0. An
in-flight **unary** (an inline `run_now` such as `GenerateEmbeddings`) is
bounded only by the grace period. A DRAIN that outlives the grace is
SIGKILLed: the running job's lease then expires after one `[lease]
duration_secs`, a successor requeues it (resuming from its last epoch
bundle) and it consumes one attempt.

**RELEASE** (`kill -INT`, Ctrl+C, `jammi-server release`, or a second
SIGTERM): connections are severed at once; every job lease this process
holds is handed back — the row stays `running` under this instance with
`lease_expires_at = NULL` and `releases + 1`, and a compute job's linked
building-table lease is NULLed with it — the loop is stopped (cooperatively
while no job is under a hold, by abort while one is), the `workers` row is
deleted, the catalog released, and the process exits 0 at once. A released
row is claimable by any other worker within one `[worker] idle_poll_secs`,
never one lease window; the reclaim cap counts `attempts - releases`, so a
rollout storm of releases never burns the three attempts a genuine crash
does. The abandoned training thread never finalizes: its lease is gone and
its next epoch boundary bails without a bundle, so the `_resume` manifest
epoch never advances past the last landed one. Two named exceptions (§3.5 of
the design): a claim caught between its COMMIT and its hold registration
past one heartbeat keeps its live lease and is recovered by the expiry path
(one lease window, `attempts + 1`, never `failed`); and a compute job whose
linked building sweep errored while the jobs sweep succeeded makes the
successor back off once (one lease window) before it re-materializes.

`jammi-server release [--pid N]` sends SIGINT to `N` (default 1, the
container entrypoint) and exits 0 when the signal was sent — the uniform
RELEASE actuator for a `preStop` hook, since the distroless images carry no
shell for `kill`. It knows nothing about jobs. The library reaches the same
mechanism through `EmbeddedWorker::release_and_stop` and Python's
`close(release=True)`; there the process survives, so a thread that reaches
finalize before a successor claims may still land `completed` — the one
documented divergence from the server, which exits.

`ListWorkers` / `jammi workers` show each claim loop's `state`: `warming`
(the process is preloading; nothing claimed yet), `claiming`, or `draining`.

## The identity seam

The server performs **no authentication** on its own — treat it as
**trusted-network** and put access control in front of it, or supply your own
`TenantResolver` at the seam the engine ships for exactly this. The contract
of record for that seam is on the security page; the sketch below is only
the minimal shape a caller wires in. See:

- [Security Posture](./security.md) for the full threat model — what the
  engine defends, what it explicitly does not, and the trusted-network
  assumption every deployment inherits; and
- [Bring your own auth](./multi-tenant.md#bring-your-own-auth) for the
  `TenantResolver` seam itself: one resolver, plugged into
  `assemble_grpc_chain` once, authenticates both the gRPC control plane and
  the Flight `db.sql` lane.

**Sketch: an authenticating proxy in front.** The engine does not invent
tenants (the one rule everything else follows from —
[Design Philosophy](./philosophy.md#the-one-rule-everything-else-follows-from));
a proxy that already verified the caller injects the fact, and the resolver
only reads it:

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_server;
# extern crate tonic;
# extern crate uuid;
use jammi_db::TenantId;
use jammi_server::grpc::session::{TenantResolver, TenantScope};
use tonic::{metadata::MetadataMap, Status};
use uuid::Uuid;

struct ProxyHeaderResolver;

#[tonic::async_trait]
impl TenantResolver for ProxyHeaderResolver {
    // The proxy verified the caller upstream and sets this header itself —
    // never a client-controlled one. Read ONLY the proxy-set value and
    // reject when it is absent: no header, no fallback tenant.
    async fn resolve(&self, metadata: &MetadataMap) -> Result<TenantScope, Status> {
        let raw = metadata
            .get("x-jammi-verified-tenant")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("no verified tenant"))?;
        let uuid = Uuid::parse_str(raw).map_err(|_| Status::unauthenticated("bad tenant"))?;
        let tenant = TenantId::from_uuid(uuid).map_err(|_| Status::unauthenticated("bad tenant"))?;
        Ok(TenantScope::Tenant(tenant))
    }
}
```

That is the whole pattern: the proxy authenticates and sets one header the
client cannot forge (metadata stripped from the inbound request and
re-added by the proxy itself); the resolver trusts only its own header and
fails closed when it is missing.

Transport encryption is a separate decision from the identity seam above —
see [Security
Posture](./security.md#transport-encryption-is-the-deployers-runtime-not-the-engines)
for why the engine ships no TLS code path and how a deployer's runtime
terminates it in front.

Run the server where only trusted clients can reach it (a private network /
VPC with the gRPC + health ports, `8081` / `8080`, closed to the public
internet; network policy or a firewall; or an authenticating reverse proxy) —
or wire an authenticating `TenantResolver` in front — before exposing it
beyond a trusted caller.

**The peer listener (I-PEER).** `[server] peer_bind` — unset by default —
opens a THIRD listener that serves `jammi.v1.peer.PeerService` to other
replicas of the same deployment (segment search for the segments this replica
owns; see [Beyond one node](./reference-topologies.md#beyond-one-node-retrieval)).
It is deliberately outside the identity seam: the peer routes are built
outside `assemble_grpc_chain`, are never wrapped by the `TenantResolverLayer`,
never advertised by `GetServerInfo`, and the public listener answers
`UNIMPLEMENTED` for their paths. The owner binds no tenant — the request
carries none — because tenant scope was already enforced by the coordinator
(the replica that received the `Search`), which resolved the table through its
own tenant-scoped catalog read before fanning out; the owner enforces only
that every requested segment belongs to the named table. The invariant every
deployment inherits: **every client of `peer_bind` is a jammi coordinator.**
Bind it on a private interface behind network policy and, where the runtime
provides it, mTLS; on a routable interface without them it exposes
cross-tenant segment reads to anyone who can reach the port.

## Deploying as a container

The OSS server ships as two public Docker images on GHCR:

- `ghcr.io/f-inverse/jammi-ai-server` — **CPU**, built from a distroless base.
- `ghcr.io/f-inverse/jammi-ai-server-cu12` — **CUDA**, for GPU-accelerated inference (see [GPU serving](#gpu-serving)).

The generic CPU tags (`:latest`, `:vX.Y.Z`, `:vX.Y`, and their `sha-<sha>`
equivalents) are multi-arch image indexes: `linux/amd64` and `linux/arm64`,
so `docker pull`/`docker run` resolves the right member for the host's
architecture automatically. The self-contained CPU tags
(`:selfcontained`, `:selfcontained-sha-<sha>`) and the CUDA (`-cu12`) tags
are `linux/amd64` only, pushed under the same CPU image name in the
self-contained case.

Both run as the nonroot user (uid `65532`), expose the same `8080` / `8081` ports the local binary listens on, and share the same tag scheme (`:latest`, `:vX.Y.Z`, `:vX.Y`). Both `:latest` tags are re-pointed by every `v*` release tag (never by a prerelease); the CPU `:latest` can additionally be re-pointed to the current `main` by a manual `build-and-push-main` dispatch. The image entrypoint is `jammi-server`, so `docker run <image>` brings up the server with **zero config** — a local SQLite catalog, the in-memory broker, and every service tier, no TOML required. The `jammi` admin CLI also ships in the image for running verbs against the server. The examples below use the CPU image, and bind both published ports to `127.0.0.1`: the server itself performs no authentication (see [The identity seam](#the-identity-seam)), so publishing to every interface would expose an unauthenticated admin surface to the host's whole network — a terminator or reverse proxy that itself binds a public interface is what a deployment fronts these loopback-bound ports with.

```bash
# Turnkey: zero config, no TOML.
docker run --rm \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server:latest
```

To supply your own config, pass `--config` to the `jammi-server` entrypoint:

```bash
docker run --rm \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  -v jammi_data:/var/lib/jammi \
  -v $(pwd)/jammi.toml:/etc/jammi/jammi.toml:ro \
  ghcr.io/f-inverse/jammi-ai-server:latest --config /etc/jammi/jammi.toml
```

A tested Compose stack (server + Postgres catalog + JetStream broker) lives
at `deploy/docker-compose.yml`, exercised end to end by the `compose-smoke`
workflow — see [Reference Topologies: Shape
B](./reference-topologies.md#shape-b--single-tenant-server) for the full
file, its environment variables, and what the smoke proves.

### Persistence

`/var/lib/jammi` holds the catalog DB, model weights, and indices. Zero-config `jammi-server` writes its SQLite catalog there (the image sets `JAMMI_ARTIFACT_DIR=/var/lib/jammi`) and its Hugging Face Hub cache at `/var/lib/jammi/hf` (the image sets `HF_HOME=/var/lib/jammi/hf`, the fallback [`[models] hub_cache_dir`](./configuration.md#catalog-broker-signing-key-storage-and-model-source) reads when unset). On the `jammi-ai-server-cu12` image the same volume also holds the CUDA PTX-JIT compute cache at `/var/lib/jammi/.nv-cache` (see [GPU serving](#gpu-serving)) — mounting the volume is what makes both caches durable across container restarts. The Dockerfile declares `/var/lib/jammi` as a `VOLUME` owned by uid `65532` — a named Docker volume or no mount at all just works; a bind mount must have the host directory writable by uid `65532`:

```bash
# Bind mount on the host.
sudo chown -R 65532:65532 /opt/jammi/data
docker run -v /opt/jammi/data:/var/lib/jammi ...
```

A named Docker volume (the compose default) sidesteps that step because Docker provisions ownership for the container's user automatically.

### Configuration

The image needs no config — it boots zero-config, running `jammi-server
serve` (the implicit default subcommand) via the entrypoint. To override
defaults under `docker run`, pass `--config` as shown above, or set
`JAMMI_*` env vars — the `[gpu]`, `[server]`, and `services` knobs documented
above all apply. Under Compose, the same two options carry over unchanged
(`command:` appends to the entrypoint the same way `docker run <image>
--config ...` does; `environment:` sets the same `JAMMI_*` vars) — see
[Reference Topologies: Shape B](./reference-topologies.md#shape-b--single-tenant-server)
for a complete, tested Compose file doing exactly this against a Postgres
catalog and JetStream broker.

### GPU serving

The `jammi-ai-server-cu12` image builds with candle's CUDA backend on an NVIDIA CUDA 12.6 runtime base, so `libcudart` and the rest of the CUDA runtime libraries are present in the image. It carries the same turnkey `jammi` CLI as the CPU image. Run it on a host with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and pass `--gpus all`. Both `:latest` tags are re-pointed by every `v*` release tag (never by a prerelease); the CPU `:latest` can additionally be re-pointed to the current `main` by a manual `build-and-push-main` dispatch — pin an exact `:vX.Y.Z` tag for a reproducible GPU-node deploy:

```bash
# Turnkey: zero config, GPU inference.
docker run --rm --gpus all \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest
```

With no TOML the server selects GPU device `0` by default. To override the device or any other knob, pass a config with `--config` (or bind-mount it straight at `/etc/jammi/jammi.toml`, one of the resolution order's own default locations — see [Configuration](./configuration.md) — and drop the flag entirely):

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  -v jammi_data:/var/lib/jammi \
  -v $(pwd)/jammi.toml:/etc/jammi/jammi.toml:ro \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest --config /etc/jammi/jammi.toml
```

Set `gpu.device = 0` in `jammi.toml` (or `JAMMI_GPU__DEVICE=0`) to select the CUDA device; see [GPU configuration](#gpu-configuration). The image is compiled for compute capability `8.0` (Ampere) and runs on `8.0` and every newer datacenter GPU — A10/A6000 (`8.6`), L40S (`8.9`), H100 (`9.0`) — via PTX forward-compatibility. Turing GPUs (e.g. Tesla T4, `7.5`) are not supported.

**Minimum NVIDIA driver:** the image is built against the CUDA 12.6 toolkit and ships single-architecture PTX, so on any GPU newer than `8.0` the driver **JIT-compiles** that PTX at first model load. This requires a driver new enough for the CUDA 12.6 runtime — **Linux: `r560` or later** (`≥ 560.28.03`). An older driver (for example `550.x`, which tops out at the CUDA 12.4 PTX ISA) can reject the image's newer PTX at load with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` / `CUDA_ERROR_INVALID_PTX`, even on a supported architecture. `nvidia-smi` reports the installed driver and its max CUDA version.

**JIT cache persistence:** the image sets `CUDA_CACHE_PATH=/var/lib/jammi/.nv-cache`, so the driver's compiled PTX→SASS cache lands on the `/var/lib/jammi` volume rather than the container's ephemeral filesystem. With the volume mounted, the JIT cost above is paid once — a subsequent cold start on the same host reuses the cached SASS instead of re-JIT-ing every model load. Without the volume mounted, each container restart starts with an empty cache and re-pays the JIT. `CUDA_CACHE_MAXSIZE` (bytes) caps the cache size if the default cap is too small for the set of models you serve.

The CPU image ignores GPU config and runs inference on the CPU.

### Building from source

The Dockerfile lives at the workspace root and uses BuildKit cache mounts for the cargo registry and target directory:

```bash
# CPU image (default).
DOCKER_BUILDKIT=1 docker build -t jammi-ai-server:dev -f Dockerfile .

# CUDA image — selected by the RUNTIME_VARIANT build-arg.
DOCKER_BUILDKIT=1 docker build -t jammi-ai-server-cu12:dev \
  --build-arg RUNTIME_VARIANT=runtime-cuda -f Dockerfile .
```

Cold builds take ~30 minutes (the workspace is large); warm builds with cache hits land at ~3 minutes. The CUDA build additionally compiles candle's CUDA kernels, so its cold build is longer.

### Supply chain: SBOM, provenance, attestations

Every image `server-image.yml` pushes to GHCR — the CPU `:latest` / `:vX.Y.Z`
tags, the CUDA `-cu12` tags, and the dispatch-only `:selfcontained` build —
carries a `docker/build-push-action` SPDX SBOM and `mode=max` build
provenance attached to the image manifest, plus a Sigstore-signed
[`actions/attest-build-provenance`](https://github.com/actions/attest-build-provenance)
attestation published to the repository's attestation store for the exact
digest that job pushed — never a mutable tag, which a concurrent run could
re-point. Verify an image you pulled against that digest:

```bash
gh attestation verify oci://ghcr.io/f-inverse/jammi-ai-server@sha256:<digest> \
  --repo f-inverse/jammi-ai
```

CI checks both of these, in two separate steps, on every push job:
`ci/scripts/assert_image_attestations.sh` asserts (via `docker buildx
imagetools inspect`) that the pushed digest's own OCI index carries BOTH a
non-empty SBOM and a non-empty provenance attestation manifest; a positively
empty accessor fails the job immediately rather than falling back to a
looser check. Which SHAPE each attestation must then match is decided by
the index's own platform count, read off the raw OCI index, never by the
attestation's own key spelling: a multi-platform index (the merged CPU
manifest list, `linux/amd64` + `linux/arm64`) requires the per-platform map,
its key set checked for exact equality against the index's platform set, so
an attestation covering only one of the merged legs — the other landed
unattested — fails the job by name, rather than passing because *some*
platform was attested. A single-platform push (the CUDA `-cu12` tags, the
dispatch-only `:selfcontained` build, and each per-arch `sha-<sha>-<arch>`
leg before it is merged) instead requires the FLAT predicate object buildx
actually emits for a single platform — `{"SLSA":{...}}` / `{"SPDX":{...}}`
— since that flat object and the per-platform map are structurally
indistinguishable by key inspection alone (both are "a non-empty object of
non-empty objects"); the index's platform count is what breaks the tie. A
separate `gh attestation verify oci://... --bundle-from-oci` step then
verifies the Sigstore-signed bundle `attest-build-provenance` published as
an OCI referrer — the check above never touches that bundle.
The `compose-smoke` workflow's own build (`load: true`, loaded into the
runner's daemon, never pushed) carries neither: `sbom` and `provenance` are
explicitly `false` there, since the stock Docker exporter a `load` build
uses cannot carry attestations.
