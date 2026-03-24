# Jammi AI

Jammi is an embeddable AI engine that brings model inference into your data pipeline. Register data sources, run SQL queries, generate embeddings, search with vector similarity, and build ML-powered applications — all from Rust.

## What Jammi does today (Checkpoint 3)

- **Query local data with SQL** — register Parquet and CSV files, run full SQL via DataFusion
- **Generate embeddings** — load any BERT-family model from HuggingFace Hub (or local safetensors), persist results to Parquet with sidecar ANN indexes
- **Vector search** — ANN similarity search over embedding tables with automatic fallback to brute-force
- **Search builder** — fluent API for `.join()`, `.annotate()`, `.filter()`, `.sort()`, `.limit()`, `.select()`, `.run()`
- **Evidence provenance** — `retrieved_by` and `annotated_by` tracking on every search result
- **Per-row error handling** — null or invalid text produces error status per row, not a batch failure
- **Model caching** — LRU eviction, ref-counted guards, single-flight loading
- **Crash recovery** — on restart, recovers result tables stuck in "building" state
- **Inference observability** — attach observers to hook into every output batch

## What's coming next

- **Fine-tuning** — LoRA fine-tuning with contrastive loss
- **Evaluation** — metrics, golden sets, A/B comparison
- **GPU scheduling** — memory-budget admission control
- **Python API** — PyO3 bindings
- **Server** — REST + Arrow Flight SQL

## Crates

| Crate | Purpose |
|-------|---------|
| `jammi-engine` | Query engine, configuration, catalog, source management, Parquet storage, ANN indexes |
| `jammi-ai` | Model loading, inference execution, embedding pipeline, vector search, evidence model |
| `jammi-server` | HTTP and Arrow Flight SQL server (future) |
| `jammi-cli` | Command-line interface (future) |
| `jammi-python` | Python bindings via PyO3 (future) |
