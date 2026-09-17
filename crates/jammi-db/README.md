# jammi-db

Vector database, SQL federation, mutable companion tables, and trigger broker for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-db` provides the foundation for Jammi: SQL queries via DataFusion, source registration (Parquet, CSV, JSON, JSONL, PostgreSQL, MySQL), Parquet storage with sidecar ANN indexes, mutable companion tables with crash-safe WAL, trigger broker for provenance channels, and configuration management.

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

## Build requirements

The `postgres` and `mysql` features pull `datafusion-table-providers`'s
federation drivers, which link a native TLS stack (`native-tls` ->
OpenSSL) rather than `rustls`, unconditionally at every published
version. Building either feature requires OpenSSL's development headers
on the build host (e.g. `libssl-dev`/`openssl-devel`, or the `openssl`
Homebrew formula plus `OPENSSL_DIR`/`PKG_CONFIG_PATH`). Neither feature
is enabled by default, and no release lane in this repo enables them
today (`ci/release-feature-manifest.json` carries no row that reaches
either — enforced by `ci/scripts/check_release_manifest_pg_mysql_closure.py`).

## Documentation

See the [Jammi AI Guide](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
