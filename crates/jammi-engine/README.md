# jammi-engine

Query engine, catalog, and storage layer for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-engine` provides the foundation for Jammi: SQL queries via DataFusion, source registration (Parquet, CSV, JSON, PostgreSQL, MySQL), Parquet storage with sidecar ANN indexes, crash recovery, and configuration management.

## Usage

```rust
use jammi_engine::config::JammiConfig;
use jammi_engine::session::JammiSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

let config = JammiConfig::load(None)?;
let session = JammiSession::new(config).await?;

session.add_source("data", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/data.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;

let results = session.sql("SELECT * FROM data.public.data LIMIT 10").await?;
```

## Documentation

See the [Jammi AI Cookbook](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
