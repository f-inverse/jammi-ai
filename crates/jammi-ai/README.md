# jammi-ai

Embeddable AI engine for inference, embeddings, vector search, and fine-tuning. Part of [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-ai` builds on `jammi-engine` to add model loading (HuggingFace Hub, local safetensors, ONNX), embedding generation with persistent Parquet + ANN indexes, semantic vector search with a fluent SearchBuilder API, LoRA fine-tuning, and evaluation (retrieval, classification, summarization).

## Usage

```rust
use std::sync::Arc;
use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

let config = JammiConfig::load(None)?;
let session = Arc::new(InferenceSession::new(config).await?);

// Register data and generate embeddings
session.add_source("patents", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/patents.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;

session.generate_embeddings(
    "patents", "sentence-transformers/all-MiniLM-L6-v2", &["abstract".into()], "id",
).await?;

// Semantic search
let query = session.encode_query("sentence-transformers/all-MiniLM-L6-v2", "quantum computing").await?;
let results = session.search("patents", query, 10).await?
    .sort("similarity", true)?
    .limit(5)
    .run().await?;
```

## Documentation

See the [Jammi AI Cookbook](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
