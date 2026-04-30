# Jammi AI

Jammi is an embeddable AI engine that brings model inference into your data pipeline. Register data sources, run SQL queries, generate embeddings, search with vector similarity, fine-tune models on your domain, and evaluate results — all without leaving your application.

## Install

```bash
pip install jammi-ai
```

For CUDA/GPU support:

```bash
pip install jammi-ai-cu12
```

## Quickstart

```python
import jammi

# Connect (pass gpu_device=-1 to force CPU)
db = jammi.connect()

# Register a local data source
db.add_source("patents", path="patents.parquet", format="parquet")

# Query with SQL — returns a pyarrow.Table
table = db.sql("SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5")
print(table.to_pandas())

# Generate and persist embeddings (with an ANN index)
db.generate_text_embeddings(
    source="patents",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["title"],
    key="id",
)

# Semantic search
query_vec = db.encode_text_query("sentence-transformers/all-MiniLM-L6-v2", "quantum computing applications")

search = db.search("patents", query=query_vec, k=5)
search.sort("similarity", descending=True)
results = search.run()   # pyarrow.Table

print(results.to_pandas())
```

## Features

- **SQL over local files** — query Parquet, CSV, and JSON via DataFusion
- **Federated queries** — join local files with PostgreSQL or MySQL
- **Text embeddings** — load any BERT-family model from Hugging Face Hub (or local safetensors / ONNX) and persist results to Parquet with ANN indexes
- **Image embeddings** — CLIP-style vision encoders
- **Vector search** — ANN similarity search with automatic brute-force fallback
- **SearchBuilder** — fluent API for `.filter()`, `.sort()`, `.join()`, `.annotate()`, `.limit()`, `.select()`, `.run()`
- **Evidence provenance** — `retrieved_by` and `annotated_by` tracking on every search result
- **Fine-tuning** — LoRA / deep LoRA adapters with contrastive loss to improve embeddings for your domain
- **Evaluation** — recall@k, precision@k, MRR, nDCG, accuracy, F1, and A/B model comparison
- **Model caching** — LRU eviction, ref-counted guards, single-flight loading
- **GPU scheduling** — memory-budget admission control with RAII permits
- **Crash recovery** — recovers embedding tables stuck in "building" state on restart

## SearchBuilder

```python
search = db.search("patents", query=query_vec, k=20)
search.filter("year >= 2020")
search.sort("similarity", descending=True)
search.limit(5)
search.select(["id", "title", "similarity"])
results = search.run()   # pyarrow.Table
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

## Documentation

Full documentation, including guides for SQL queries, embeddings, search, fine-tuning, and evaluation:

**[https://f-inverse.github.io/jammi-ai/](https://f-inverse.github.io/jammi-ai/)**

## License

Apache-2.0
