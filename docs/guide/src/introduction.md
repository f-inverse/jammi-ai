# Jammi AI

Jammi is an embeddable AI engine that brings model inference into your data pipeline. Register data sources, run SQL queries, generate embeddings, search with vector similarity, fine-tune models on your domain, and evaluate results — all without leaving your application.

## What Jammi does

- **Query local data with SQL** — register Parquet, CSV, and JSON files, run full SQL via DataFusion
- **Federate external databases** — query PostgreSQL and MySQL alongside local files
- **Generate embeddings** — load any BERT-family model from HuggingFace Hub (or local safetensors / ONNX), persist results to Parquet with sidecar ANN indexes
- **Vector search** — ANN similarity search over embedding tables with automatic fallback to brute-force
- **Search builder** — fluent API for `.join()`, `.annotate()`, `.filter()`, `.sort()`, `.limit()`, `.select()`, `.run()`
- **Evidence provenance** — `retrieved_by` and `annotated_by` tracking on every search result
- **Fine-tuning** — LoRA adapters with contrastive loss to improve embeddings for your domain
- **Evaluation** — retrieval metrics (recall@k, precision@k, MRR, nDCG), classification (accuracy, F1), and A/B model comparison
- **Per-row error handling** — null or invalid text produces error status per row, not a batch failure
- **Model caching** — LRU eviction, ref-counted guards, single-flight loading
- **GPU scheduling** — memory-budget admission control with RAII permits
- **Crash recovery** — on restart, recovers result tables stuck in "building" state
- **Inference observability** — attach observers to hook into every output batch

## Three ways to use Jammi

| Interface | Best for | Install |
|-----------|----------|---------|
| **Rust library** | Embedding Jammi into Rust applications | `cargo add jammi-ai` |
| **Python package** | Data science, notebooks, scripts | `pip install jammi` |
| **CLI** | Shell workflows, quick queries, ops | `cargo install jammi-cli` |

All three interfaces share the same engine, configuration, and storage format. Embeddings generated from Python are queryable from the CLI, and vice versa.

For multi-language access or BI tool integration, `jammi serve` starts an Arrow Flight SQL server — any Arrow client can connect and query via standard SQL.

## Crates

| Crate | Purpose |
|-------|---------|
| `jammi-engine` | Query engine, configuration, catalog, source management, Parquet storage, ANN indexes |
| `jammi-ai` | Model loading, inference execution, embedding pipeline, vector search, evidence model, fine-tuning, evaluation |
| `jammi-server` | Arrow Flight SQL server and HTTP health endpoint |
| `jammi-cli` | Command-line interface |
| `jammi-python` | Python bindings via PyO3 |

`jammi-engine` has no dependency on `jammi-ai`. You can use it standalone for SQL queries over local data without pulling in the AI layer.
