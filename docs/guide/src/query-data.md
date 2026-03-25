# Query Your Data with SQL

Register data files as named sources, then query them with full SQL. Sources are persisted in the catalog and survive session restarts.

## Register a source

### Rust

```rust
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

session.add_source("patents", SourceType::Local, SourceConnection {
    url: Some("file:///data/patents.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;
```

### Python

```python
db.add_source("patents", path="/data/patents.parquet", format="parquet")
```

### CLI

```bash
jammi sources add patents --path /data/patents.parquet --format parquet
```

## Supported formats

| Format | Rust | Python/CLI | Notes |
|--------|------|------------|-------|
| Parquet | `FileFormat::Parquet` | `"parquet"` | Columnar, compressed, recommended for large datasets |
| CSV | `FileFormat::Csv` | `"csv"` | Auto-detected schema |
| JSON | `FileFormat::Json` | `"json"` | Line-delimited JSON |

## Run a SQL query

Sources are accessible via three-part SQL names: `<source_id>.public.<table_name>`. The table name is derived from the file name (e.g., `patents.parquet` becomes `patents`).

### Rust

```rust
let results = session.sql(
    "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year"
).await?;

for batch in &results {
    println!("{batch:?}");
}
```

### Python

```python
table = db.sql("SELECT id, title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year")
print(table.to_pandas())
```

### CLI

```bash
jammi query "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year"
```

## Aggregations

```sql
SELECT category, COUNT(*) as count, AVG(citation_count) as avg_citations
FROM patents.public.patents
WHERE year > 2020
GROUP BY category
ORDER BY count DESC
```

## Joins across sources

Register multiple sources and join them in a single query:

### Rust

```rust
session.add_source("companies", SourceType::Local, SourceConnection {
    url: Some("file:///data/companies.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.sql("
    SELECT p.title, c.company_name
    FROM patents.public.patents p
    JOIN companies.public.companies c ON p.assignee_id = c.id
").await?;
```

### Python

```python
db.add_source("companies", path="/data/companies.csv", format="csv")

table = db.sql("""
    SELECT p.title, c.company_name
    FROM patents.public.patents p
    JOIN companies.public.companies c ON p.assignee_id = c.id
""")
```

## Source lifecycle

### Rust

```rust
// List registered sources
let sources = session.catalog().list_sources()?;

// Remove a source
session.catalog().remove_source("patents")?;
```

### CLI

```bash
jammi sources list
```

Sources persist in the SQLite catalog at `<artifact_dir>/catalog.db`. Registering the same source ID twice returns an error — remove it first.

## Execution plans

Use `EXPLAIN` (or the CLI `explain` command) to see how DataFusion will execute your query:

```bash
jammi explain "SELECT * FROM patents.public.patents WHERE year > 2020"
```
