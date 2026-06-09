# Deploy as a Server

Jammi can run as an Arrow Flight SQL server, making all registered sources and embedding tables queryable from any Arrow-compatible client. Use this when multiple services, BI tools, or non-Rust/Python consumers need to query Jammi's data.

## The workflow

The server is a **read path**. Set up your data with the library or CLI, then deploy the server so other systems can query it:

```bash
# 1. Register sources and generate embeddings (CLI or library)
jammi sources add patents --path /data/patents.parquet --format parquet

# 2. Generate embeddings (library or Python — not available over Flight SQL)
python3 -c '
import jammi_ai
db = jammi_ai.connect("file:///var/lib/jammi")
db.generate_embeddings(source="patents", model="sentence-transformers/all-MiniLM-L6-v2", columns=["abstract"], key="id", modality="text")
'

# 3. Start the server
jammi serve
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
always mounted: `SessionService` (tenant binding + the `GetServerInfo`
handshake), `EmbeddingService`, `InferenceService`, `MutableTableService`,
`ChannelService`, `AuditService`, and the Flight SQL surface. Three optional
tiers are runtime-selectable via `[server] services`:

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

Every config field can be overridden with environment variables, useful for containerized deployments:

```bash
JAMMI_SERVER__FLIGHT_LISTEN=0.0.0.0:9081 \
JAMMI_GPU__DEVICE=-1 \
JAMMI_LOGGING__FORMAT=json \
jammi serve
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

The Flight SQL surface is a **query** interface (read path); the ML operations are not SQL, so they ride the **typed gRPC** surface instead. Set up your data and run training/inference through the Rust library, the `jammi-ai` / `jammi-client` Python package, or — for a remote engine — those same verbs over gRPC, then query the results over Flight SQL. The CLI registers sources and starts the server; it carries no ML verbs.

The typed gRPC surface is what an edge runtime speaks (it has no HTTP/2 client for Flight SQL's bidirectional streaming). `EmbeddingService` serves `AddSource`, `GenerateEmbeddings`, `EncodeQuery`, and `Search` over plain gRPC — and, since tonic-web is mounted, over **gRPC-web** — so an edge function running the engine as a sidecar can ingest, encode, **and** search without the library. `Search` accepts a precomputed vector or an existing `row_key` (query-by-example, with the vector resolved inside the engine); see [Semantic Search](./semantic-search.md#search-over-grpc-edge-runtimes). With the **train** tier mounted, `TrainingService` serves all three training kinds over gRPC and `InferenceService.Predict` serves a trained context predictor — so a client can offload training and prediction to a GPU server with the same verb surface the embedded engine exposes.

## Graceful shutdown

The server drains active connections on SIGTERM / Ctrl+C before exiting. In-flight queries complete; long-running operations started via the library are unaffected.

## Deploying as a container

The OSS server ships as two public Docker images on GHCR:

- `ghcr.io/f-inverse/jammi-ai-server` — **CPU**, built from a distroless base.
- `ghcr.io/f-inverse/jammi-ai-server-cu12` — **CUDA**, for GPU-accelerated inference (see [GPU serving](#gpu-serving)).

Both run as the nonroot user (uid `65532`), expose the same `8080` / `8081` ports the local binary listens on, and share the same tag scheme (`:latest`, `:vX.Y.Z`, `:vX.Y`). Both carry the `jammi` CLI and are **turnkey**: the image entrypoint is `jammi serve`, so `docker run <image>` brings up the server with **zero config** — a local SQLite catalog, the in-memory broker, and every service tier, no TOML required. The examples below use the CPU image.

```bash
# Turnkey: zero config, no TOML.
docker run --rm \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server:latest
```

To supply your own config, pass it to the `serve` subcommand (the image
entrypoint is the CLI, so any `jammi` verb works):

```bash
docker run --rm \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  -v $(pwd)/jammi.toml:/etc/jammi/jammi.toml:ro \
  ghcr.io/f-inverse/jammi-ai-server:latest serve --config /etc/jammi/jammi.toml
```

A minimal compose file lives in the workspace at `examples/docker-compose/oss-server.yml`:

```bash
cd examples/docker-compose
docker compose -f oss-server.yml up
```

### Persistence

`/var/lib/jammi` holds the catalog DB, model weights, and indices. Zero-config `jammi serve` writes its SQLite catalog there (the image sets `JAMMI_ARTIFACT_DIR=/var/lib/jammi`). The Dockerfile declares it as a `VOLUME` owned by uid `65532` — a named Docker volume or no mount at all just works; a bind mount must have the host directory writable by uid `65532`:

```bash
# Bind mount on the host.
sudo chown -R 65532:65532 /opt/jammi/data
docker run -v /opt/jammi/data:/var/lib/jammi ...
```

A named Docker volume (the compose default) sidesteps that step because Docker provisions ownership for the container's user automatically.

### Configuration

The image needs no config — it boots zero-config. To override defaults, bind-mount a TOML and point `serve` at it with a `command:`. The `[gpu]`, `[server]`, and `services` knobs documented above (and `JAMMI_*` env overrides) all apply:

```yaml
# oss-server.yml
services:
  jammi-server:
    image: ghcr.io/f-inverse/jammi-ai-server:latest
    command: ["serve", "--config", "/etc/jammi/jammi.toml"]
    volumes:
      - ./jammi.toml:/etc/jammi/jammi.toml:ro
      - jammi_data:/var/lib/jammi
    ports:
      - "8080:8080"
      - "8081:8081"
```

Or skip the TOML entirely and tune via environment variables:

```yaml
services:
  jammi-server:
    image: ghcr.io/f-inverse/jammi-ai-server:latest
    environment:
      JAMMI_SERVER__SERVICES: "event"      # serve + event tier only
      JAMMI_LOGGING__FORMAT: "json"
    volumes:
      - jammi_data:/var/lib/jammi
    ports:
      - "8080:8080"
      - "8081:8081"
```

### GPU serving

The `jammi-ai-server-cu12` image builds with candle's CUDA backend on an NVIDIA CUDA 12.6 runtime base, so `libcudart` and the rest of the CUDA runtime libraries are present in the image. It carries the same turnkey `jammi` CLI as the CPU image. Run it on a host with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and pass `--gpus all`:

```bash
# Turnkey: zero config, GPU inference.
docker run --rm --gpus all \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest
```

With no TOML the server selects GPU device `0` by default. To override the device or any other knob, pass a config to `serve`:

```bash
docker run --rm --gpus all \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  -v $(pwd)/jammi.toml:/etc/jammi/jammi.toml:ro \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest serve --config /etc/jammi/jammi.toml
```

Set `gpu.device = 0` in `jammi.toml` (or `JAMMI_GPU__DEVICE=0`) to select the CUDA device; see [GPU configuration](#gpu-configuration). The image is compiled for compute capability `8.6`. The CPU image ignores GPU config and runs inference on the CPU.

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
