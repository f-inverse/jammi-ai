# jammi-db

Vector database, SQL federation, mutable companion tables, and trigger broker for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-db` provides the foundation for Jammi: SQL queries via DataFusion, source registration (Parquet, CSV, JSON, PostgreSQL, MySQL), Parquet storage with sidecar ANN indexes, mutable companion tables with crash-safe WAL, trigger broker for provenance channels, and configuration management.

## Usage

```rust
use jammi_db::config::JammiConfig;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

let config = JammiConfig::load(None)?;
let session = JammiSession::new(config).await?;

session.add_source("data", SourceType::File, SourceConnection {
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
