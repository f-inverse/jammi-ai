# Generate Embeddings

Generate vector embeddings by running a model over text columns from a registered source. Results are persisted to Parquet with sidecar ANN indexes for fast similarity search.

## Basic usage

### Rust

```rust
let record = session.generate_text_embeddings(
    "patents",
    "sentence-transformers/all-MiniLM-L6-v2",
    &["abstract".to_string()],
    "id",
).await?;

println!("Embedded {} rows, {} dimensions", record.row_count, record.dimensions.unwrap());
```

### Python

```python
db.generate_text_embeddings(
    source="patents",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["abstract"],
    key="id",
)
```

## What gets created

Each call creates a timestamped Parquet file plus a sidecar ANN index bundle:

```
{artifact_dir}/jammi_db/
├── patents__embedding__all-MiniLM-L6-v2__20260325T120000.parquet
├── patents__embedding__all-MiniLM-L6-v2__20260325T120000.usearch
├── patents__embedding__all-MiniLM-L6-v2__20260325T120000.rowmap
└── patents__embedding__all-MiniLM-L6-v2__20260325T120000.manifest.json
```

- **Parquet file** — source of truth. Contains `_row_id`, `_source_id`, `_model_id`, `vector`. Readable by external tools (DuckDB, Polars, pandas).
- **`.usearch`** — USearch HNSW graph for ANN search.
- **`.rowmap`** — maps internal USearch keys to `_row_id` strings.
- **`.manifest.json`** — metadata (dimensions, count, metric, backend).

The sidecar files are disposable — deleting them falls back to brute-force exact search. The Parquet file is the only thing that matters.

## Embedding table schema

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key column value cast to string |
| `_source_id` | Utf8 | Source identifier |
| `_model_id` | Utf8 | Model identifier |
| `vector` | FixedSizeList(Float32, N) | L2-normalized embedding vector |

Failed rows (null or empty text) are excluded — only successfully embedded rows appear in the output.

## Multiple text columns

Pass multiple column names to concatenate them (space-separated) before embedding:

### Rust

```rust
session.generate_text_embeddings(
    "papers",
    "sentence-transformers/all-MiniLM-L6-v2",
    &["title".to_string(), "abstract".to_string()],
    "doi",
).await?;
```

### Python

```python
db.generate_text_embeddings(
    source="papers",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["title", "abstract"],
    key="doi",
)
```

## Multiple embedding tables

Each call creates a new table. Multiple tables can coexist for the same source (different models, different columns):

```rust
session.generate_text_embeddings("patents", "all-MiniLM-L6-v2", &["abstract".into()], "id").await?;
session.generate_text_embeddings("patents", "bge-small-en-v1.5", &["title".into()], "id").await?;
```

When searching, the latest ready embedding table is used by default.

## Supported models

Any encoder model on HuggingFace Hub with safetensors weights. Supported architectures:

**BERT family** — BERT, RoBERTa, DistilBERT, CamemBERT, XLM-RoBERTa:

- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, higher quality)
- `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`

**ModernBERT** — modernized encoder with rotary embeddings, 8192-token context, GeGLU:

- `answerdotai/ModernBERT-base` (768-dim)
- `answerdotai/ModernBERT-large` (1024-dim)

Or any local directory with `config.json` + `model.safetensors` + `tokenizer.json`. The architecture is detected automatically from `model_type` in config.json.

Use a local model:

```rust
let model = ModelSource::local("/path/to/my-model");
```

## Raw inference (no persistence)

To get embeddings as `RecordBatch` without writing to disk:

### Rust

```rust
use jammi_ai::model::{ModelSource, ModelTask};

let model = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
let results = session.infer("patents", &model, ModelTask::Embedding, &["abstract".into()], "id").await?;
```

### Python

```python
results = db.infer(
    source="patents",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["abstract"],
    task="embedding",
    key="id",
)
```

Each `RecordBatch` has prefix columns (`_row_id`, `_source`, `_model`, `_status`, `_error`, `_latency_ms`) plus task-specific columns (e.g., `vector` for embeddings).

## Error handling

Inference never panics on bad input. Errors are tracked per-row:

| Condition | `_status` | `_error` | `vector` |
|-----------|-----------|----------|----------|
| Valid text | `"ok"` | null | 384-dim float vector |
| Null text | `"error"` | `"Empty or null text input"` | null |
| Empty text | `"error"` | `"Empty or null text input"` | null |

The batch continues processing even when individual rows fail.

## Dynamic batch sizing

The runner starts with the configured `inference.batch_size` (default: 32). If an out-of-memory error occurs:

1. Halve the batch size
2. Retry (up to 3 times)
3. If OOM persists at batch size 1, mark the row as error and continue

The reduced batch size is sticky for the remainder of the stream.

## Crash recovery

If the process dies mid-generation, the table is left in "building" status. On the next session start, recovery runs automatically:

- **Parquet missing** — mark as failed
- **Parquet corrupt** — delete file, mark as failed
- **Parquet valid but stuck in "building"** — promote to "ready", rebuild ANN index

No data is lost if the Parquet file was fully written.

## DataFusion integration

Result tables are automatically registered in DataFusion and queryable via SQL:

```rust
let results = session.sql(&format!(
    "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 10",
    record.table_name
)).await?;
```
