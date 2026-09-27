# Scope a Federated Source by Tenant

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Tenancy](https://f-inverse.github.io/jammi-ai/cookbook/chapters/11-tenancy/tenancy.html).

The session-scoped tenant binding ([`multi-tenant.md`](./multi-tenant.md))
relies on every table the engine reads carrying a `tenant_id` column. That
works for mutable companion tables and Parquet result tables Jammi
produced itself — both emit the column per the tenant-identifier discipline
([Design Philosophy](./philosophy.md#the-one-rule-everything-else-follows-from)).
But a *federated* source
— a remote Postgres warehouse, a S3 Parquet lake, a CSV from someone
else's pipeline — usually doesn't. It may carry a `customer_id`, an
`organization`, a `workspace` column, or no tenant discriminator at all.

This recipe shows how to tell Jammi which column on a federated source
plays the role of the tenant discriminator, so the predicate-injection
analyzer rule scopes scans against that column instead of looking for the
engine's built-in `tenant_id` name.

## Goal

After this recipe you can:

1. Register a federated source whose tenant discriminator is named
   differently from `tenant_id`, declaring that column as you register it.
2. Verify two tenants get disjoint rows from the same physical source.
3. Recognise what the declaration does *not* do.

## Setup

The recipe assumes you have a Parquet file (or any other federated
source) whose schema includes a column that already carries the tenant
identifier — for example a `customer_id` column populated with the UUID
of the customer who owns each row. The column's value must be the same
canonical hyphenated lowercase form `TenantId::Display` emits; the
analyzer rule does a string comparison after coercing the column to
`Utf8`.

## Register the source with its tenant column

The tenant column is part of the source's definition: it is declared when
the source is registered, persisted with it in the catalog, and replayed by
every session and every replica that reads the source. Register the source
once, as a global source (unscoped), so every tenant can name it.

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate tokio;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

# async fn ex(session: &JammiSession) -> jammi_db::error::Result<()> {
session
    .add_source(
        "notes",
        SourceType::File,
        SourceConnection {
            url: Some("file:///data/notes.parquet".into()),
            format: Some(FileFormat::Parquet),
            tenant_column: Some("customer_id".into()),
            ..Default::default()
        },
    )
    .await?;
# Ok(())
# }
```

### Python

The same declaration, on either transport — an embedded engine or a
server:

```python
db.add_source("notes", url="/data/notes.parquet", format="parquet",
              tenant_column="customer_id")
```

From then on, a scan of `notes.public.notes` under a tenant-bound session
is rewritten with
`WHERE CAST(customer_id AS Utf8) = $current_tenant OR CAST(customer_id AS Utf8) IS NULL`
(or `IS NULL` alone when the session is unscoped). Every verb that reads
the source reads through that scan — `sql`, `generate_embeddings` (which
therefore embeds only the tenant's rows), and `search` (whose embedding
table and hydrated columns are the tenant's own).

A source whose tables carry a `tenant_id` column needs no declaration:
that column scopes it on its own.

## Verify the predicate

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate tokio;
# use std::str::FromStr;
# use jammi_db::TenantId;
# use jammi_db::session::JammiSession;
# use jammi_db::config::JammiConfig;
# async fn ex() -> jammi_db::error::Result<()> {
let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a")?;
let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9b")?;

let session_a = JammiSession::new(JammiConfig::default()).await?.with_tenant(alice);
let session_b = JammiSession::new(JammiConfig::default()).await?.with_tenant(bob);

let count_a = session_a.sql("SELECT COUNT(*) FROM notes.public.notes").await?;
let count_b = session_b.sql("SELECT COUNT(*) FROM notes.public.notes").await?;

// Each session sees only its own rows — `count_a` and `count_b` are
// disjoint subsets of the on-disk file.
# Ok(())
# }
```

For a file with 10 rows split 6 (`customer_id = alice`) + 4
(`customer_id = bob`), the two sessions get `6` and `4` respectively.

## What you cannot do

- **You cannot declare a column the source does not have.** Registration
  refuses it with a typed `Schema` error naming the column (Python:
  `InvalidArgument`), and nothing is persisted.
- **You cannot declare a second discriminator.** A table that carries its
  own `tenant_id` column is scoped by it; declaring another column on such a
  source is refused at registration the same way.
- **You cannot change the declaration in place.** It is part of the
  source's definition: remove the source and register it again with the
  new column (or none — a source with no discriminator is readable by
  every tenant).

If the federated source you are wrapping carries no tenant
discriminator, two options are open: (1) re-shape upstream so each
tenant lands in its own table, registered as a separate source, or
(2) accept that the source is globally visible to every session and
gate access at a higher layer (Flight SQL session interceptor, gRPC
auth middleware). The engine itself does not authenticate;
[Design Philosophy](./philosophy.md#the-one-rule-everything-else-follows-from) §
*the engine does not invent tenants* applies.

## See also

- [Scope a Session to a Tenant](./multi-tenant.md) — the broader
  session-binding recipe this one extends.
- [Register a Mutable Companion Table](./register-mutable-table.md) —
  for sources Jammi owns, the `tenant_id` column is emitted by default.
