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

Models listed in `preload_models` are downloaded and loaded into memory at startup. This ensures the session is warm before the server accepts connections.

```toml
[server]
preload_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]
```

## Service tiers

One server binary scales to many deployment shapes by mounting only the gRPC
service tiers a deployment needs — no per-shape rebuild. The **core** tier is
always mounted: `CatalogService` (the control plane — tenant binding, the
`GetServerInfo` handshake, and source / model / channel / mutable-table /
topic administration), `EmbeddingService`, `InferenceService`,
`AuditService`, and the Flight SQL surface. Three optional tiers are
runtime-selectable via `[server] services`:

| Tier | Service | Role |
|---|---|---|
| `train` | `TrainingService` | model training (fine-tune, graph fine-tune, context predictor) |
| `event` | `TriggerService` | topic / publish / subscribe streams |
| `eval`  | `EvalService` | per-query evaluation arrays |

```toml
[server]
services = "all"             # all-in-one: every tier compiled in (the default)
# services = ["event"]       # serve + event box
# services = ["train"]       # serve + training box
# services = []              # serve-only: core tier only
```

A deployment advertises exactly the tiers it mounted over the wire, so a client
can negotiate capability before calling a verb:

```python
info = db.get_server_info()
# {"version": "...", "features": [...], "storage_backends": [...],
#  "services": ["core", "eval", "event", "train"]}
if "train" in info["services"]:
    db.fine_tune(...)
```

Reaching a verb whose tier was **not** mounted returns a truthful `Unimplemented`
("not enabled on this deployment") rather than a misleading success — the
service-mount analog of the client's build-by-capability `connect(target)`.

**Runtime config vs. compile features.** The `train` tier additionally requires
the `train` compile feature (on by default). A `--no-default-features`
serve-only build carries no training surface at all; requesting `train` in
config on such a build is a startup error, not a silent drop. The `event` and
`eval` tiers always compile and are gated at runtime only. Override the
selection with `JAMMI_SERVER__SERVICES` (`all`, or a comma-separated token
list — empty for serve-only).

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

`/healthz` is a liveness probe — a `200` means the process is running.
`/readyz` is a readiness probe — `200` means the catalog backend
responded; `503` means it didn't and traffic should be drained from
this instance. Point your load balancer at `/readyz`.

`/metrics` exposes a small, substrate-level set of Prometheus counters
(gRPC requests, Flight SQL queries, eval invocations) plus a search-
latency histogram.

## What the server can and cannot do

| Operation | Available over Flight SQL? | Available over typed gRPC? |
|-----------|--------------------------|----------------------------|
| SQL queries on source tables | Yes | — (use Flight SQL) |
| SQL queries on embedding tables | Yes | — (use Flight SQL) |
| Joins, aggregations, filters | Yes | — (use Flight SQL) |
| Generate embeddings | No — use library or Python package | Yes — `EmbeddingService.GenerateEmbeddings` |
| Semantic vector search | No — use library or Python package | Yes — `EmbeddingService.Search` |
| Inference | No — use library or Python package | Yes — `InferenceService.Infer` |
| Fine-tuning (and graph / context-predictor training) | No — use library or Python package | Yes — `TrainingService.StartTraining` (train tier) |
| Context-predictor prediction | No — use library or Python package | Yes — `InferenceService.Predict` |
| Evaluation | No — use library or Python package | Yes — `EvalService` (eval tier) |

The Flight SQL surface is a **query** interface (read path); the ML operations are not SQL, so they ride the **typed gRPC** surface instead. Set up your data and run training/inference through the Rust library, the `jammi-ai` / `jammi-client` Python package, or — for a remote engine — those same verbs over gRPC, then query the results over Flight SQL. The CLI is a strict gRPC client that registers sources and drives the admin surfaces against a running server; it carries no ML verbs and does not run the engine in-process.

The typed gRPC surface is what an edge runtime speaks (it has no HTTP/2 client for Flight SQL's bidirectional streaming). `EmbeddingService` serves `AddSource`, `GenerateEmbeddings`, `EncodeQuery`, and `Search` over plain gRPC — and, since tonic-web is mounted, over **gRPC-web** — so an edge function running the engine as a sidecar can ingest, encode, **and** search without the library. `Search` accepts a precomputed vector or an existing `row_key` (query-by-example, with the vector resolved inside the engine); see [Semantic Search](./semantic-search.md#search-over-grpc-edge-runtimes). With the **train** tier mounted, `TrainingService` serves all three training kinds over gRPC and `InferenceService.Predict` serves a trained context predictor — so a client can offload training and prediction to a GPU server with the same verb surface the embedded engine exposes.

## Graceful shutdown

The server drains active connections on SIGTERM / Ctrl+C before exiting. In-flight queries complete; long-running operations started via the library are unaffected.

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

## Deploying as a container

The OSS server ships as two public Docker images on GHCR:

- `ghcr.io/f-inverse/jammi-ai-server` — **CPU**, built from a distroless base.
- `ghcr.io/f-inverse/jammi-ai-server-cu12` — **CUDA**, for GPU-accelerated inference (see [GPU serving](#gpu-serving)).

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
looser check. A separate `gh attestation verify oci://... --bundle-from-oci`
step then verifies the Sigstore-signed bundle `attest-build-provenance`
published as an OCI referrer — the check above never touches that bundle.
The `compose-smoke` workflow's own build (`load: true`, loaded into the
runner's daemon, never pushed) carries neither: `sbom` and `provenance` are
explicitly `false` there, since the stock Docker exporter a `load` build
uses cannot carry attestations.
