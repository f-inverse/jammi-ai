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

## Generate embeddings and search

```rust
use std::sync::Arc;
use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JammiConfig::load(None)?;
    let session = Arc::new(InferenceSession::new(config).await?);

    session.add_source("patents", SourceType::Local, SourceConnection {
        url: Some("file:///path/to/patents.parquet".into()),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }).await?;

    // Generate embeddings — persists to Parquet with sidecar ANN index
    let record = session.generate_embeddings(
        "patents",
        "sentence-transformers/all-MiniLM-L6-v2",
        &["abstract".to_string()],
        "id",
    ).await?;
    println!("Embedded {} rows, dimensions={:?}", record.row_count, record.dimensions);

    // Encode a query and search
    let query_vec = session.encode_query(
        "sentence-transformers/all-MiniLM-L6-v2",
        "quantum computing applications",
    ).await?;

    let results = session.search("patents", query_vec, 5).await?
        .sort("similarity", true)?
        .run().await?;

    for batch in &results {
        println!("{batch:?}");
    }
    Ok(())
}
```

The first run downloads the model from HuggingFace Hub (~90MB). Subsequent runs load from cache.

## Run raw inference

For inference without persistence (returns `Vec<RecordBatch>` directly):

```rust
use jammi_ai::model::{ModelSource, ModelTask};

let model = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
let results = session.infer(
    "patents",
    &model,
    ModelTask::Embedding,
    &["abstract".to_string()],
    "id",
).await?;
```

Each `RecordBatch` has prefix columns (`_row_id`, `_source`, `_model`, `_status`, `_error`, `_latency_ms`) plus task-specific columns (e.g., `vector` for embeddings).

## Supported models

Any BERT-family model on HuggingFace Hub with safetensors weights:

- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, higher quality)
- `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`
- Any local directory with `config.json` + `model.safetensors` + `tokenizer.json`

Use a local model:

```rust
let model = ModelSource::local("/path/to/my-model");
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
