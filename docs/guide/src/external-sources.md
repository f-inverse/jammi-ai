# Connect to PostgreSQL / MySQL

Jammi federates external databases alongside local files. Register a database as a source and query it with the same SQL interface — joins across local files and databases work seamlessly.

## PostgreSQL

### Rust

```rust
use jammi_engine::source::{SourceConnection, SourceType};

session.add_source("pg_data", SourceType::Postgres, SourceConnection {
    url: Some("postgresql://user:pass@localhost:5432/mydb".into()),
    ..Default::default()
}).await?;

let results = session.sql(
    "SELECT id, title FROM pg_data.public.articles WHERE published = true LIMIT 10"
).await?;
```

## MySQL

### Rust

```rust
session.add_source("mysql_data", SourceType::Mysql, SourceConnection {
    url: Some("mysql://user:pass@localhost:3306/mydb".into()),
    ..Default::default()
}).await?;
```

## Cross-source joins

Once registered, external databases are queryable with the same three-part naming convention and can be joined with local files:

```sql
SELECT p.title, a.author_name
FROM local_data.public.papers p
JOIN pg_data.public.authors a ON p.author_id = a.id
WHERE a.institution = 'MIT'
```

## Generate embeddings from external sources

External databases work as sources for embedding generation:

```python
db.add_source("pg_articles", path="postgresql://user:pass@localhost/mydb", format="parquet")
# Note: for external databases, use the Rust API to register with the appropriate source_type

db.generate_embeddings(
    source="pg_articles",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["title", "abstract"],
    key="id",
)
```

## Feature flags

External source support requires feature flags when building from source:

| Source | Feature flag |
|--------|-------------|
| PostgreSQL | `postgres` |
| MySQL | `mysql` |

These are enabled by default in published crates and pre-built binaries.

## Supported source types

| Type | Description | Status |
|------|-------------|--------|
| Local files | Parquet, CSV, JSON | Always available |
| PostgreSQL | Any PostgreSQL-compatible database | Available |
| MySQL | MySQL / MariaDB | Available |
| SQLite | SQLite databases | Not supported (rusqlite version conflict) |
