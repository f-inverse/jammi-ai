# Quickstart: Rust

This walkthrough registers a local data file, runs a SQL query, generates embeddings, and performs a semantic search — all in one program.

## Full example

```rust
use std::sync::Arc;
use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JammiConfig::load(None)?;
    let session = Arc::new(InferenceSession::new(config).await?);

    // 1. Register a data source
    session.add_source("patents", SourceType::Local, SourceConnection {
        url: Some("file:///path/to/patents.parquet".into()),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }).await?;

    // 2. Query with SQL
    let rows = session.sql(
        "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5"
    ).await?;
    for batch in &rows {
        println!("{batch:?}");
    }

    // 3. Generate embeddings
    let record = session.generate_text_embeddings(
        "patents",
        "sentence-transformers/all-MiniLM-L6-v2",
        &["title".to_string()],
        "id",
    ).await?;
    println!("Embedded {} rows", record.row_count);

    // 4. Semantic search
    let query = session.encode_text_query(
        "sentence-transformers/all-MiniLM-L6-v2",
        "quantum computing applications",
    ).await?;

    let results = session.search("patents", query, 5).await?
        .sort("similarity", true)?
        .run().await?;

    for batch in &results {
        println!("{batch:?}");
    }

    Ok(())
}
```

The first run downloads the model from HuggingFace Hub (~90MB). Subsequent runs load from cache.

## What's happening

1. **`JammiConfig::load(None)`** loads config from `jammi.toml`, `$JAMMI_CONFIG`, or defaults
2. **`InferenceSession`** wraps the query engine with model loading, caching, and GPU scheduling
3. **`add_source`** registers a file in the catalog — it survives session restarts
4. **`sql`** runs any SQL query via DataFusion, returns `Vec<RecordBatch>`
5. **`generate_text_embeddings`** runs the model over every row, persists vectors to Parquet with a sidecar ANN index
6. **`encode_text_query`** encodes a text string into the same vector space
7. **`search`** finds the nearest neighbors, hydrates all source columns, and returns results with similarity scores

## Next steps

- [Query Your Data with SQL](./query-data.md) — SQL features, joins, aggregations
- [Generate Embeddings](./generate-embeddings.md) — persistence, multiple models, crash recovery
- [Semantic Search](./semantic-search.md) — SearchBuilder API, filtering, evidence provenance
