# Deploy as a Server

Jammi can run as an Arrow Flight SQL server, making all registered sources and embedding tables queryable from any Arrow-compatible client. Use this when multiple services, BI tools, or non-Rust/Python consumers need to query Jammi's data.

## The workflow

The server is a **read path**. Set up your data with the library or CLI, then deploy the server so other systems can query it:

```bash
# 1. Register sources and generate embeddings (CLI or library)
jammi sources add patents --path /data/patents.parquet --format parquet

# 2. Generate embeddings (library or Python — not available over Flight SQL)
python3 -c "
import jammi
db = jammi.connect()
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

## Health checks

The server also exposes an HTTP endpoint for container liveness probes:

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

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

## Graceful shutdown

The server drains active connections on SIGTERM / Ctrl+C before exiting. In-flight queries complete; long-running operations started via the library are unaffected.
