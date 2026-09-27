# Connect to PostgreSQL / MySQL

> **Measured companion:** [Databases as sources — Postgres, MySQL and a file in one query](https://f-inverse.github.io/jammi-ai/cookbook/chapters/30-federation/federation.html)
> federates a Postgres table and a MariaDB table, joins them with a CSV file, and embeds
> from the Postgres table, on the embedded engine and on a server.

Jammi federates external databases alongside local files. Register a database by its
URL and every table in it becomes a table of the SQL surface, queried in place: the
largest part of a query that reads only one database's tables — filters, projections,
joins, aggregates, limits — runs in that database, and the engine joins the result with
any other source. A function only the engine has (one it installs, such as the content
hash an embedding job computes) is never sent to a database: the part of the query that
calls it runs in the engine, over the rows the database returns.

## Python

`add_source` takes the URL that names the data. A `postgres://` or `postgresql://` URL is
a PostgreSQL database, a `mysql://` URL a MySQL or MariaDB database; neither takes a
`format`. Any other URL or path is a file, which needs one. The embedded engine and a
remote session take the same call.

```python
db.add_source("pg_data", url="postgresql://reader:secret@db.internal:5432/sales")
db.add_source("mysql_data", url="mysql://reader:secret@mysql.internal:3306/registry")
db.add_source("papers", url="/data/papers.parquet", format="parquet")

db.sql("SELECT id, title FROM pg_data.public.articles WHERE published LIMIT 10")
```

A Postgres source serves the tables of the database's `public` schema; a MySQL source
serves the tables of the database its URL names. Both appear under the source's name
with the `public` schema: `pg_data.public.articles`, `mysql_data.public.assignees`.

## Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
use jammi_db::source::{SourceConnection, SourceType};

session.add_source("pg_data", SourceType::Postgres, SourceConnection {
    url: Some("postgresql://user:pass@localhost:5432/mydb".into()),
    ..Default::default()
}).await?;

session.add_source("mysql_data", SourceType::Mysql, SourceConnection {
    url: Some("mysql://user:pass@localhost:3306/mydb".into()),
    ..Default::default()
}).await?;

let results = session.sql(
    "SELECT id, title FROM pg_data.public.articles WHERE published = true LIMIT 10"
).await?;
# Ok(()) }
```

`SourceDefinition::from_url` chooses the source type from the URL the way Python does.

## Connection URLs

A PostgreSQL URL means what it means to libpq. The host, port, user, password and
database come from the URL, and the query parameters `host`, `port`, `user`,
`password`, `dbname`, `sslmode`, `sslrootcert`, `application_name` and `options`
override them; any other parameter is refused, and so is a list of hosts. Every part
is percent-decoded (a `+` stays a `+`), and an empty value is no value. A `host` that
is a path names a Unix-socket directory, so a local server's socket URL works as handed
out: `postgresql://postgres:@/postgres?host=/tmp/pgdata`. `sslmode` is `disable`,
`prefer` (the default, as in libpq), `require`, `verify-ca` or `verify-full`.

A MySQL URL is `mysql://user:password@host:port/database`, and must name its database.
Its `sslmode` parameter is `required` (the default), `preferred` or `disabled`; a server
with no TLS — a local development instance — is reached with `sslmode=disabled`.

## Cross-source joins

Once registered, external databases are queryable with the same three-part naming
convention and can be joined with local files:

```sql
SELECT p.title, a.author_name
FROM local_data.public.papers p
JOIN pg_data.public.authors a ON p.author_id = a.id
WHERE a.institution = 'MIT'
```

## Generate embeddings from external sources

A database source feeds every verb a file does. A verb that names a source rather than a
table — `generate_embeddings` among them — reads the source's first table in name order,
so a database registered for one holds the table it reads first:

```python
db.generate_embeddings(
    source="pg_articles",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["title", "abstract"],
    key="id",
)
db.search("pg_articles", query=vector, k=10)
```

## What ships

Every published engine carries both drivers: the `jammi-ai-native` and
`jammi-ai-native-cu12` wheels, and every `jammi-server` wheel, tarball and container
image. The drivers speak TLS through OpenSSL, built from source and linked statically
into the artifact, so none needs a system OpenSSL or a database client library at
runtime.

Building from source, the cargo features are:

| Source | Feature flag |
|--------|-------------|
| PostgreSQL | `postgres` |
| MySQL | `mysql` |

Neither is on by default for a bare `cargo build`. Enabling either compiles OpenSSL
from source on Linux, which needs a C toolchain and Perl with the `IPC::Cmd` and
`Time::Piece` modules — see [Build dependencies (Linux)](./installation.md#build-dependencies-linux).
macOS and Windows use the platform's TLS stack.

## Supported source types

| Type | Description | Status |
|------|-------------|--------|
| File (`file://`) | Parquet, CSV, JSON, JSONL on local disk | Always available |
| File (`s3://` / `gs://` / `azure://`) | Same formats over cloud object stores | Feature-gated — see [Cloud Storage](./cloud-storage.md) |
| PostgreSQL | Any PostgreSQL-compatible database | Available |
| MySQL | MySQL / MariaDB | Available |
| SQLite | SQLite databases | Not supported (rusqlite version conflict) |
