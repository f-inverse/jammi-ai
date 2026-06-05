# Jammi AI

Jammi is an embeddable AI engine that brings model inference into your data pipeline. Register data sources, run SQL queries, generate embeddings, search with vector similarity, fine-tune models on your domain, and evaluate results — all without leaving your application.

## Install

```bash
pip install jammi-ai
```

The embed wheel runs the engine in-process and bundles
[`jammi-client`](./clients/python/) for remote targets. For a lean,
engine-free deploy footprint that talks to a remote server, install the client
on its own:

```bash
pip install jammi-client
```

(GPU/CUDA lives on the server image — the CUDA variant
[`jammi-ai-server-cu12`](https://github.com/f-inverse/jammi-ai/pkgs/container/jammi-ai-server-cu12) —
not the embed wheel.)

## Quickstart

The 5-minute walkthrough — install, connect, register a source, generate
embeddings, search — lives in [`cookbook/quickstart/`](./cookbook/quickstart/)
with a runnable [`quickstart.py`](./cookbook/quickstart/quickstart.py)
gated by CI. The condensed version:

```python
import jammi_ai

# One front door. `file://` runs the in-process engine; flip to a
# `https://` / `grpc://` target — no code change — to talk to a remote server.
db = jammi_ai.connect("file://.jammi")
db.add_source("corpus", url="cookbook/fixtures/tiny_corpus.parquet", format="parquet")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
db.generate_embeddings(source="corpus", model=MODEL, columns=["content"], key="id", modality="text")

query_vec = db.encode_query(model=MODEL, query="quantum computing applications")
results = db.search("corpus", query=query_vec, k=5)  # pyarrow.Table
print(results.to_pandas())
```

For runnable end-to-end recipes — mutable tables, trigger streams, eval,
fine-tuning, Flight SQL — see [`cookbook/`](./cookbook/).

## Features

- **SQL over local files** — query Parquet, CSV, and JSON via DataFusion
- **Federated queries** — join local files with PostgreSQL or MySQL
- **Text embeddings** — load any BERT-family model from Hugging Face Hub (or local safetensors / ONNX) and persist results to Parquet with ANN indexes
- **Image embeddings** — CLIP-style vision encoders
- **Vector search** — ANN similarity search with automatic brute-force fallback; `search` returns a table directly, same shape embedded or remote
- **Similarity graphs** — `build_neighbor_graph` materializes the whole k-nearest-neighbour edge set of an embedding table as a queryable relation, for dedup, clustering, and graph-aware training-data prep
- **Compound query** — `join` / `filter` / `select` and model inference (the `annotate` SQL table function) over your data, in-process or over the Flight SQL lane in one round-trip
- **Evidence provenance** — `retrieved_by` and `annotated_by` tracking on the fluent Rust query builder's results
- **Fine-tuning** — LoRA / deep LoRA adapters with contrastive loss to improve embeddings for your domain
- **Evaluation** — recall@k, precision@k, MRR, nDCG, accuracy, F1, and A/B model comparison
- **Model caching** — LRU eviction, ref-counted guards, single-flight loading
- **GPU scheduling** — memory-budget admission control with RAII permits
- **Crash recovery** — recovers embedding tables stuck in "building" state on restart

## Search and compound query

`search` is the bounded primitive — nearest-neighbor top-k with optional `filter` / `select`, returning a `pyarrow.Table` directly (the same call, embedded or remote):

```python
results = db.search(
    "patents", query=query_vec, k=20,
    filter="year >= 2020", select=["id", "title", "similarity"],
)   # pyarrow.Table
```

For open, compound retrieval + inference — `join`, `filter`, and running a model over a relation — use SQL. The `annotate(...)` table function runs a model over a relation's columns; it works identically in-process (embed wheel) and over the Flight SQL lane (remote engine via `jammi-client`):

```python
results = db.sql("""
    SELECT p.title, a.vector
    FROM annotate('all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS a
    JOIN patents.public.patents AS p ON a._row_id = arrow_cast(p.id, 'Utf8')
""")
```

All results are returned as `pyarrow.Table` — zero-copy from the Rust engine.

## Fine-tuning

```python
job = db.fine_tune(
    source="patents",
    model="sentence-transformers/all-MiniLM-L6-v2",
    triplets="triplets_train.parquet",
)
job.wait()
```

## Requirements

- Python 3.9+
- Linux (x86_64) or macOS (Apple Silicon or Intel)

> **Windows** is not yet supported due to a dependency on POSIX memory-mapping APIs.

## Running the OSS server

For deployments that need a long-running Flight SQL + gRPC service rather than an embedded library, the workspace ships a Docker image:

```bash
docker run --rm \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server:latest

curl http://localhost:8080/healthz
# {"status":"ok","version":"0.8.0"}
```

For GPU-accelerated inference, pull the CUDA variant `ghcr.io/f-inverse/jammi-ai-server-cu12:latest` and run it with `--gpus all` on a host with the NVIDIA Container Toolkit.

The OSS server is single-tenant — the deployer's network is the auth boundary. See [Deploy as a Server](https://f-inverse.github.io/jammi-ai/deploy-server.html) for the full guide.

## Documentation

Full documentation, including guides for SQL queries, embeddings, search, fine-tuning, and evaluation:

**[https://f-inverse.github.io/jammi-ai/](https://f-inverse.github.io/jammi-ai/)**

For the engine's design philosophy — what belongs in Jammi versus a consumer's own repo, how
embeddings are consumed, and how it deploys — see
[Design Philosophy](./docs/guide/src/philosophy.md).

## License

Apache-2.0
