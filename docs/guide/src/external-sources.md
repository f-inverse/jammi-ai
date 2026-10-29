# Connect to PostgreSQL / MySQL

> **Measured companion:** [Databases as sources — Postgres, MySQL and a file in one query](https://f-inverse.github.io/jammi-ai/cookbook/chapters/federation/federation.html)
> federates a Postgres table and a MariaDB table, joins them with a CSV file, and embeds
> from the Postgres table, on the embedded engine and on a server.

Jammi federates external databases alongside local files. Register a database by its
URL and every table in it becomes a table of the SQL surface, queried in place: the
largest part of a query that reads only one database's tables, and that the database
reads exactly as the engine does, runs in that database, and the engine joins the result
with any other source. Anything the database would read otherwise stays in the engine,
over the rows the database returns, so an answer never depends on where it was computed
([What runs in the database](#what-runs-in-the-database)).

## Python

`add_source` takes the URL that names the data. A `postgres://` or `postgresql://` URL is
a PostgreSQL database, a `mysql://` URL a MySQL or MariaDB database; neither takes a
`format`. Any other URL or path is a file, which needs one. The embedded engine and a
remote session take the same call.

```python
db.add_source("pg_data", url="postgresql://reader:${JAMMI_SOURCE_PG_PASSWORD}@db.internal:5432/sales")
db.add_source("mysql_data", url="mysql://reader:${JAMMI_SOURCE_MYSQL_PASSWORD}@mysql.internal:3306/registry")
db.add_source("papers", url="/data/papers.parquet", format="parquet")

db.sql("SELECT id, title FROM pg_data.public.articles WHERE published LIMIT 10")
```

A Postgres source serves the tables, views and materialized views of every schema the
connecting role may use, each under its own schema: `pg_data.public.articles`,
`pg_data.sales.orders`. Which schemas and relations a source serves is the role's
grants; the system schemas are never served. A MySQL source serves the tables of the
database its URL names under `public`: `mysql_data.public.assignees`.

`describe_source` lists what a source serves: each table's schema, its name and its
columns as the engine reads them, a `pyarrow.Schema` in Python. A remote session gets the
same answer as the embedded engine, so a client finds a database's tables without holding
its credentials. `list_sources` reads the registry alone; `describe_source` reaches the
database, so a database it cannot reach fails it.

```python
for table in db.describe_source("pg_data")["tables"]:
    print(f"{table['schema']}.{table['table']}", table["columns"].names)
```

A query over a Postgres source whose database has stopped answering fails at once — an
`Unavailable` error (gRPC `UNAVAILABLE`, the retryable class) naming the database's address
and why it was not reached — and answers again, with no re-registration, once the database
does.

## Postgres column types

Every column of every type reads as the value it holds, and no column's type keeps its
table, or any other table of the source, from registering. A column is read as stored
when it is a number, a bounded or unbounded `numeric`, `boolean`, a string type, `uuid`,
`bytea`, `date`, `time`, `timestamptz`, `interval`, or an array of numbers, booleans or
strings. Postgres converts the rest before the engine reads them, and a filter, sort or
aggregate the database runs compares the converted value:

| Type | Read as |
|---|---|
| a domain | its base type |
| `money` | `numeric(19, s)`, `s` the database's money scale (read when the source registers) |
| `timestamp` | the instant it names at UTC; every session runs with `TimeZone` set to UTC |
| every other type (`jsonb`, `json`, `citext`, `timetz`, an enum, `inet`, `tsvector`, `xml`, `bit`, …) | its text |

## Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
use jammi_db::source::{SourceConnection, SourceType};

session.add_source("pg_data", SourceType::Postgres, SourceConnection {
    url: Some("postgresql://user:${JAMMI_SOURCE_PG_PASSWORD}@localhost:5432/mydb".into()),
    ..Default::default()
}).await?;

session.add_source("mysql_data", SourceType::Mysql, SourceConnection {
    url: Some("mysql://user:${JAMMI_SOURCE_MYSQL_PASSWORD}@localhost:3306/mydb".into()),
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

A MySQL URL is `mysql://user:${JAMMI_SOURCE_PASSWORD}@host:port/database`, and must name its database.
Its `sslmode` parameter is `required` (the default), `preferred` or `disabled`. `required`
verifies the server's certificate and the host it names; `preferred` encrypts without
verifying either; a server with no TLS — a local development instance — is reached with
`sslmode=disabled`.

### Passwords

A source's URL names its password with a placeholder, never the password itself. The
engine keeps a source's URL in its catalog, to connect again after a restart and from
every server sharing the catalog, so a password written into the URL would be written
into the catalog, its WAL and every backup of it. `${NAME}` in a URL is read from the
environment of the process that connects, each time it connects, and names a variable
whose name starts `JAMMI_SOURCE_` (and holds no `__`, which the config loader reads as a
config section's path):

```python
import os

os.environ["JAMMI_SOURCE_PG_PASSWORD"] = "…"  # or set it where the engine runs
db.add_source("pg_data", url="postgresql://reader:${JAMMI_SOURCE_PG_PASSWORD}@db.internal:5432/sales")
```

- **Embedded**, the variable is read from your own process's environment. **Remote**, it
  is read from the **server's**: whoever runs the server sets it there, and a client
  names it.
- Only a `JAMMI_SOURCE_` variable can be named. The environment is the server's and
  the URL is the client's, so a URL naming any variable could have the server send its
  own secrets (its catalog's password, its cloud keys) as the password to a host the
  client chose. A placeholder naming any other variable is refused with an
  `InvalidArgument` on `url`. Any client that may add a source may still name any
  `JAMMI_SOURCE_` variable, so set there only the passwords those clients may use.
- A URL that writes its password, in its user info or as a Postgres `password`
  parameter, is refused with an `InvalidArgument` on `url` that shows the placeholder
  form; the refusal never repeats the URL.
- A variable that is not set fails the source's build, naming the variable.
- The variable's value goes into the URL as written, so a password holding a character
  a URL reserves (`@`, `:`, `/`, `?`, `#`, `%`) is percent-encoded in it. `$$` in a URL
  is one `$`, as in the config file.
- A source registered before placeholders, whose URL holds its password, still
  connects, and logs a warning to register it again with a placeholder.

### A server certificate the engine's host does not trust

A database whose certificate a private CA signed — a managed database's own CA, or a
self-hosted server's — is verified against that CA when the source holds it: pass the
CA's PEM text, not a path, so a remote client hands it over the same way the embedded
engine takes it.

```python
ca = open("private-ca.pem").read()
db.add_source("orders", url="mysql://reader:${JAMMI_SOURCE_ORDERS_PASSWORD}@orders.internal:3306/shop", ca_certificate=ca)
```

A source holding a CA connects with a mode that verifies the server: `required` for
MySQL, `verify-full` for Postgres when the URL names no mode. A URL naming a mode that
does not verify (`preferred`, `disabled`, `prefer`, `require`, `disable`) is refused with
the CA, as is a Postgres URL that also names its own `sslrootcert`, a path on the engine's
host. A file source takes no CA.

## What runs in the database

A database is sent only constructs it is known to mean as the engine means them, each
backed by a test that runs it over the database and over the same rows in a file. Every
database runs scans, projections, filters, limits, inner, left and right joins, boolean
logic, `IS [NOT] NULL`, comparisons and `BETWEEN` on numbers, booleans, dates and
zone-less timestamps, `+ - *` on numbers, integer casts, `COUNT`, `SUM` over numbers, and
`MIN`/`MAX` over ordered types.

| Construct | PostgreSQL | MySQL |
|---|---|---|
| `=`, `<>`, `IN`, grouping and join keys on strings | in the database (byte equality) | in the engine (the default collation folds case) |
| `\|\|` | in the database | in the engine (a logical OR in MySQL) |
| `/` and `%` on integers and floats | in the database | in the engine (`/` on integers is decimal division) |
| `ORDER BY` on non-string keys | in the database | in the engine (NULLs sort first ascending) |
| `AVG` over integers and floats | in the database | in the engine (a four-place decimal) |
| `COUNT(DISTINCT …)`, `FULL JOIN` | in the database | in the engine |

Always in the engine: ordering comparisons and sorts on strings (a database orders them
by its collation), `AVG` over decimals, date arithmetic, string casts, every scalar
function — DataFusion's (`approx_distinct`, `date_trunc`, `upper`) and every function a
session installs — and `unnest`. A filter or a sort the engine runs above a database's
rows stays above them; only a limit reaches the database.

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
table — `generate_embeddings` among them — reads the source's first table: its `public`
schema's first relation in name order, or, with no `public` relation, the first schema's
by name. A database registered for one holds the table it reads first:

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
