# Data Sources

Jammi registers external data as named sources, then queries them via SQL. Sources are persisted in the catalog — they survive session restarts.

## Registering sources

```rust
session.add_source("my_data", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/data/".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;
```

### Supported formats

| Format | Extension | Notes |
|--------|-----------|-------|
| `FileFormat::Parquet` | `.parquet` | Columnar, compressed, recommended for large datasets |
| `FileFormat::Csv` | `.csv` | Auto-detected schema, custom delimiters via `file_extension` |

### Source types

| Type | Description | Feature flag |
|------|-------------|--------------|
| `SourceType::Local` | Local filesystem (file:// URLs) | (always available) |
| `SourceType::Postgres` | PostgreSQL database | `postgres` |
| `SourceType::Mysql` | MySQL / MariaDB database | `mysql` |

## Querying sources

Sources are accessible via three-part SQL names: `<source_id>.public.<table_name>`.

The table name is derived from the file name (e.g., `patents.parquet` becomes `patents`).

```rust
// Simple query
let results = session.sql("SELECT * FROM my_data.public.patents LIMIT 10").await?;

// Filter and aggregate
let results = session.sql("
    SELECT category, COUNT(*) as count
    FROM my_data.public.patents
    WHERE year > 2020
    GROUP BY category
    ORDER BY count DESC
").await?;

// Join across sources
session.add_source("companies", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/companies.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.sql("
    SELECT p.title, c.company_name
    FROM my_data.public.patents p
    JOIN companies.public.companies c ON p.assignee_id = c.id
").await?;
```

## Source lifecycle

```rust
// List registered sources
let sources = session.catalog().list_sources()?;

// Remove a source
session.catalog().remove_source("my_data")?;
```

Sources persist in the SQLite catalog at `<artifact_dir>/catalog.db`. Registering the same source ID twice returns an error — remove it first.
