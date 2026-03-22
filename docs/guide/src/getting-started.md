# Getting Started

## Prerequisites

- Rust 1.88+ (`rustup update`)
- A HuggingFace model will be downloaded on first use (~90MB for MiniLM)

## Add Jammi to your project

```toml
# Cargo.toml
[dependencies]
jammi-engine = { git = "https://github.com/f-inverse/jammi-ai" }
jammi-ai = { git = "https://github.com/f-inverse/jammi-ai" }
tokio = { version = "1", features = ["full"] }
```

## Query local data

```rust
use jammi_engine::config::JammiConfig;
use jammi_engine::session::JammiSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JammiConfig::load(None)?;
    let session = JammiSession::new(config).await?;

    // Register a local Parquet file
    session.add_source("patents", SourceType::Local, SourceConnection {
        url: Some("file:///path/to/patents.parquet".into()),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }).await?;

    // Query with SQL
    let results = session.sql(
        "SELECT title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year"
    ).await?;

    for batch in &results {
        println!("{batch:?}");
    }
    Ok(())
}
```

CSV works the same way — use `FileFormat::Csv` instead.

## Generate embeddings

```rust
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JammiConfig::load(None)?;
    let session = InferenceSession::new(config).await?;

    session.add_source("patents", SourceType::Local, SourceConnection {
        url: Some("file:///path/to/patents.parquet".into()),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }).await?;

    // Run embedding inference on the "abstract" text column
    let results = session.infer(
        "patents",                                    // source id
        "sentence-transformers/all-MiniLM-L6-v2",    // HF Hub model
        ModelTask::Embedding,                         // task
        &["abstract".to_string()],                    // text column(s)
        "id",                                         // key column
    ).await?;

    // Each RecordBatch has:
    //   _row_id, _source, _model, _status, _error, _latency_ms, vector
    for batch in &results {
        println!("Rows: {}, Columns: {:?}",
            batch.num_rows(),
            batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        );
    }
    Ok(())
}
```

The first run downloads the model from HuggingFace Hub (~90MB). Subsequent runs load from cache.

## Output schema

Every inference output includes these prefix columns:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key column value (cast to string) |
| `_source` | Utf8 | Source ID |
| `_model` | Utf8 | Model ID |
| `_status` | Utf8 | `"ok"` or `"error"` |
| `_error` | Utf8 (nullable) | Error message if status is error |
| `_latency_ms` | Float32 | Inference latency for this batch |

For embedding tasks, the output also includes:

| Column | Type | Description |
|--------|------|-------------|
| `vector` | FixedSizeList(Float32, N) | L2-normalized embedding vector |

Rows with null or empty text get `_status = "error"` with a message in `_error`. The `vector` column is null for error rows.

## Supported models

Any BERT-family model on HuggingFace Hub with safetensors weights:

- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, higher quality)
- `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`
- Any local directory with `config.json` + `model.safetensors` + `tokenizer.json`

Use a local model by passing its absolute path as the model ID:

```rust
session.infer("source", "/path/to/my-model", ModelTask::Embedding, &["text".into()], "id").await?;
```

## Observe inference batches

```rust
use jammi_ai::inference::observer::InferenceObserver;
use std::sync::Arc;

struct MyObserver;
impl InferenceObserver for MyObserver {
    fn on_batch(&self, batch: &arrow::record_batch::RecordBatch, model_id: &str, latency: std::time::Duration) {
        println!("Batch: {} rows from {model_id} in {latency:?}", batch.num_rows());
    }
}

let session = InferenceSession::with_observer(
    config,
    Some(Arc::new(MyObserver) as Arc<dyn InferenceObserver>),
).await?;
```
