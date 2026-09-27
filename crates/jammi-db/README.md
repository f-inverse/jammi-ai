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
federation drivers, which use a native TLS stack (`native-tls`) rather
than `rustls`. Both features turn on `native-tls`'s `vendored` feature, so
on Linux OpenSSL is built from source and linked statically, and `mysql`
links zlib statically: a build host needs a C toolchain and Perl (with the
`IPC::Cmd` and `Time::Piece` modules OpenSSL's `Configure` uses), not
OpenSSL's development headers, and the result carries no `libssl`,
`libcrypto` or `libz` runtime dependency. macOS and Windows use the
platform TLS stack. Neither feature is on by default; every published
server binary, image and Python wheel enables both
(`ci/release-feature-manifest.json`), and
`ci/scripts/check_database_drivers_static.py` keeps the static linkage a
checked property of these declarations.

## Documentation

See the [Jammi AI Guide](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
