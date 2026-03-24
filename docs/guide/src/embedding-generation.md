# Embedding Generation

`generate_embeddings()` runs a model over a source and persists the results to Jammi DB — a Parquet-native store with sidecar ANN indexes for fast similarity search.

## Basic usage

```rust
let record = session.generate_embeddings(
    "patents",                                     // source id
    "sentence-transformers/all-MiniLM-L6-v2",     // model (HF Hub or local:path)
    &["abstract".to_string()],                     // text columns to embed
    "id",                                          // key column
).await?;

assert_eq!(record.status, "ready");
println!("Embedded {} rows, {} dimensions", record.row_count, record.dimensions.unwrap());
```

## What gets created

Each call creates a new timestamped Parquet file plus a sidecar ANN index bundle:

```
{artifact_dir}/jammi_db/
├── patents__embedding__all-MiniLM-L6-v2__20260324T120000123.parquet
├── patents__embedding__all-MiniLM-L6-v2__20260324T120000123.usearch
├── patents__embedding__all-MiniLM-L6-v2__20260324T120000123.rowmap
└── patents__embedding__all-MiniLM-L6-v2__20260324T120000123.manifest.json
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

## Multiple embedding tables

Each `generate_embeddings()` call creates a new table. Multiple tables can coexist for the same source (different models, different columns):

```rust
session.generate_embeddings("patents", "all-MiniLM-L6-v2", &["abstract".into()], "id").await?;
session.generate_embeddings("patents", "bge-small-en-v1.5", &["title".into()], "id").await?;

let tables = session.catalog().find_result_tables("patents", Some("embedding"), None)?;
assert!(tables.len() >= 2);
```

When searching, the latest ready embedding table is used by default. You can also specify one explicitly.

## Crash recovery

If the process dies mid-generation, the table is left in "building" status. On the next `InferenceSession::new()`, recovery runs automatically:

- **Parquet missing** → mark as failed
- **Parquet corrupt** (no valid footer) → delete file, mark as failed
- **Parquet valid** but status stuck in "building" → promote to "ready", rebuild ANN index

No data is lost if the Parquet file was fully written.

## DataFusion integration

Result tables are automatically registered in DataFusion and queryable via SQL:

```rust
let results = session.sql(&format!(
    "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 10",
    record.table_name
)).await?;
```

Tables are re-registered on session restart via `load_existing_tables()`.
