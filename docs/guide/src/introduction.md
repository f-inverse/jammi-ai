# Jammi AI

Jammi is an embeddable AI engine that brings model inference into your data pipeline. Register data sources, run SQL queries, generate embeddings, and build ML-powered applications — all from Rust.

## What Jammi does today (Checkpoint 2)

- **Query local data with SQL** — register Parquet and CSV files, run full SQL via DataFusion
- **Generate embeddings** — load any BERT-family model from HuggingFace Hub (or local safetensors), tokenize text columns, run inference, get back Arrow RecordBatches with normalized embedding vectors
- **Per-row error handling** — null or invalid text produces error status per row, not a batch failure
- **Model caching** — LRU eviction, ref-counted guards, single-flight loading
- **Inference observability** — attach observers to hook into every output batch

## What's coming next

- **Vector search** — store embeddings in LanceDB, run similarity queries
- **Evidence model** — multi-channel provenance for search results
- **Fine-tuning** — LoRA fine-tuning with contrastive loss
- **Evaluation** — metrics, golden sets, A/B comparison
- **GPU scheduling** — memory-budget admission control
- **Python API** — PyO3 bindings
- **Server** — REST + Arrow Flight SQL

## Crates

| Crate | Purpose |
|-------|---------|
| `jammi-engine` | Query engine, configuration, catalog, source management |
| `jammi-ai` | Model loading, inference execution, output adapters |
| `jammi-server` | HTTP and Arrow Flight SQL server (Phase 11) |
| `jammi-cli` | Command-line interface (Phase 11) |
| `jammi-python` | Python bindings via PyO3 (Phase 10) |
