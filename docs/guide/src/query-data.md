# Query Your Data with SQL

Register data files as named sources, then query them with full SQL. Sources are persisted in the catalog and survive session restarts.

## Register a source

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

session.add_source("patents", SourceType::File, SourceConnection {
    url: Some("file:///data/patents.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;
# Ok(()) }
```

### Python

```python
db.add_source("patents", url="/data/patents.parquet", format="parquet")
```

### CLI

```bash
jammi sources add patents --url /data/patents.parquet --format parquet
```

## Supported formats

| Format | Rust | Python/CLI | Notes |
|--------|------|------------|-------|
| Parquet | `FileFormat::Parquet` | `"parquet"` | Columnar, compressed, recommended for large datasets |
| CSV | `FileFormat::Csv` | `"csv"` | Auto-detected schema |
| JSON | `FileFormat::Json` | `"json"` | Line-delimited JSON |
| JSON Lines | `FileFormat::JsonLines` | `"jsonl"` or `"ndjson"` | Same line-delimited reader as JSON; `.jsonl` preferred, `.ndjson` used when no `.jsonl` files match. Resolved once at registration and pinned — a later directory change never flips it. |

## Run a SQL query

Sources are accessible via three-part SQL names: `<source_id>.public.<table_name>`. The table name is derived from the file name (e.g., `patents.parquet` becomes `patents`).

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let results = session.sql(
    "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year"
).await?;

for batch in &results {
    println!("{batch:?}");
}
# Ok(()) }
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

## Materialize a query as a table

`CREATE TABLE <name> AS <query>` materializes the query's rows as a result table — the same
kind of table an embedding or an as-of join produces: bytes on the object store under the
store's root, a `result_tables` row of the `statement` kind carrying the table's attestation,
visible to every replica through the catalog. Read it as `"jammi.<name>"` (every result table's
relation), on any replica, from the moment the statement returns:

```sql
CREATE TABLE recent AS SELECT id, title FROM patents.public.patents WHERE year >= 2022;
SELECT id, title FROM "jammi.recent" ORDER BY id;
DROP TABLE recent;
```

The query runs where the compute plane says (see `[ballista.client]` in
[Configuration](./configuration.md)): on a process holding the client role, the whole
materialization — the query and the write — runs on an executor, and the submitting process
finishes the catalog row. `IF NOT EXISTS` leaves an existing table as it is; `OR REPLACE` drops
it first; a name already taken is refused otherwise.

`DROP TABLE <name>` is the store's drop of the result table under your tenant: the catalog row
and its segment and version rows go, then every object the row referenced, then the binding on
the replica that ran it (every other replica drops its own at its next resolution, which finds
no row). A table another writer is still building is refused; `IF EXISTS` makes an absent
table a no-op.

The table records its query as SQL, so it is a producer the engine replays: `recompute(name)`
re-runs the query over the sources' current rows, under the same name, and records fresh
anchors on the relations it scans (see [Materialization contract](./materialization-contract.md)).

`CREATE TABLE <name> (<columns>)` — a column list and no query — is refused: a result table is
what a query produced, and there are no empty ones. So is a qualified name (`a.b`): a result
table is named by one identifier. So is a query the engine cannot render back to SQL — a
`WITH RECURSIVE` query, a `VALUES` list — since the table could never replay; the refusal
names the node.

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

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_db::source::{FileFormat, SourceConnection, SourceType};
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
session.add_source("companies", SourceType::File, SourceConnection {
    url: Some("file:///data/companies.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.sql("
    SELECT p.title, c.company_name
    FROM patents.public.patents p
    JOIN companies.public.companies c ON p.assignee_id = c.id
").await?;
# Ok(()) }
```

### Python

```python
db.add_source("companies", url="/data/companies.csv", format="csv")

table = db.sql("""
    SELECT p.title, c.company_name
    FROM patents.public.patents p
    JOIN companies.public.companies c ON p.assignee_id = c.id
""")
```

## Source lifecycle

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
// List registered sources
let sources = session.catalog().list_sources().await?;

// Remove a source
session.remove_source("patents").await?;
# Ok(()) }
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
