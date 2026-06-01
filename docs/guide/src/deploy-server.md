# Deploy as a Server

Jammi can run as an Arrow Flight SQL server, making all registered sources and embedding tables queryable from any Arrow-compatible client. Use this when multiple services, BI tools, or non-Rust/Python consumers need to query Jammi's data.

## The workflow

The server is a **read path**. Set up your data with the library or CLI, then deploy the server so other systems can query it:

```bash
# 1. Register sources and generate embeddings (CLI or library)
jammi sources add patents --path /data/patents.parquet --format parquet

# 2. Generate embeddings (library or Python — not available over Flight SQL)
python3 -c "
import jammi_ai
db = jammi_ai.connect()
db.generate_embeddings(source='patents', model='sentence-transformers/all-MiniLM-L6-v2', columns=['abstract'], key='id')
"

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

## GPU configuration

For GPU-accelerated inference in production:

```toml
[gpu]
device = 0            # CUDA device index
memory_limit = "auto"
memory_fraction = 0.9

[inference]
batch_size = 64
max_loaded_models = 3
```

Set `gpu.device = -1` for CPU-only deployment.

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

| Operation | Available over Flight SQL? |
|-----------|--------------------------|
| SQL queries on source tables | Yes |
| SQL queries on embedding tables | Yes |
| Joins, aggregations, filters | Yes |
| Generate embeddings | No — use library or CLI |
| Semantic vector search | No — use library or CLI |
| Fine-tuning | No — use library or CLI |
| Evaluation | No — use library or CLI |

The server is a query interface. ML operations (embeddings, search, fine-tuning) are done through the Rust library, Python package, or CLI — then the results are queryable over Flight SQL.

The typed gRPC surface is the exception for the embedding path. `EmbeddingService` serves `AddSource`, `GenerateAudioEmbeddings`, `EncodeAudioQuery`, and `Search` over plain gRPC — and, since tonic-web is mounted, over **gRPC-web**. That is the transport an edge runtime can speak (it has no HTTP/2 client for Flight SQL's bidirectional streaming), so an edge function running the engine as a sidecar can ingest, encode, **and** search without the library or CLI. `Search` accepts a precomputed vector or an existing `row_key` (query-by-example, with the vector resolved inside the engine); see [Semantic Search](./semantic-search.md#search-over-grpc-edge-runtimes).

## Graceful shutdown

The server drains active connections on SIGTERM / Ctrl+C before exiting. In-flight queries complete; long-running operations started via the library are unaffected.

## Deploying as a container

The OSS server ships as a public Docker image at `ghcr.io/f-inverse/jammi-ai-server`. The image is built from a distroless base, runs as the nonroot user (uid `65532`), and exposes the same `8080` / `8081` ports the local binary listens on.

```bash
docker run --rm \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  -v $(pwd)/jammi.toml:/etc/jammi/jammi.toml:ro \
  ghcr.io/f-inverse/jammi-ai-server:latest
```

A minimal compose file lives in the workspace at `examples/docker-compose/oss-server.yml`:

```bash
cd examples/docker-compose
docker compose -f oss-server.yml up
```

### Persistence

`/var/lib/jammi` holds the catalog DB, model weights, and indices. The Dockerfile declares it as a `VOLUME` — bind mounts work, but the host directory must be writable by uid `65532`:

```bash
# Bind mount on the host.
sudo chown -R 65532:65532 /opt/jammi/data
docker run -v /opt/jammi/data:/var/lib/jammi ...
```

A named Docker volume (the compose default) sidesteps that step because Docker provisions ownership for the container's user automatically.

### Configuration

The container's entrypoint expects `/etc/jammi/jammi.toml`. Bind-mount your config there:

```yaml
# oss-server.yml
services:
  jammi-server:
    image: ghcr.io/f-inverse/jammi-ai-server:latest
    volumes:
      - ./jammi.toml:/etc/jammi/jammi.toml:ro
      - jammi_data:/var/lib/jammi
    ports:
      - "8080:8080"
      - "8081:8081"
```

### Building from source

The Dockerfile lives at the workspace root and uses BuildKit cache mounts for the cargo registry and target directory:

```bash
DOCKER_BUILDKIT=1 docker build -t jammi-ai-server:dev -f Dockerfile .
```

Cold builds take ~30 minutes (the workspace is large); warm builds with cache hits land at ~3 minutes.
