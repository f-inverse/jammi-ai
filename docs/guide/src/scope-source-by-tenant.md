# Scope a Federated Source by Tenant

The session-scoped tenant binding ([`multi-tenant.md`](./multi-tenant.md))
relies on every table the engine reads carrying a `tenant_id` column. That
works for mutable companion tables and Parquet result tables Jammi
produced itself — both emit the column by ADR-00. But a *federated* source
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
   differently from `tenant_id`.
2. Tell the analyzer rule which column to use.
3. Verify two tenants get disjoint rows from the same physical source.
4. Recognise what `set_source_tenant_column` does *not* do.

## Setup

The recipe assumes you have a Parquet file (or any other federated
source) whose schema includes a column that already carries the tenant
identifier — for example a `customer_id` column populated with the UUID
of the customer who owns each row. The column's value must be the same
canonical hyphenated lowercase form `TenantId::Display` emits; the
analyzer rule does a string comparison after coercing the column to
`Utf8`.

## Register a federated source

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
            ..Default::default()
        },
    )
    .await?;
# Ok(())
# }
```

### Python

```python
db.add_source("notes", path="/data/notes.parquet", format="parquet")
```

## Declare its tenant column

### Rust

```rust,no_run
# extern crate jammi_db;
# use jammi_db::session::JammiSession;
# fn ex(session: &JammiSession) {
session.set_source_tenant_column("notes", Some("customer_id".into()));
# }
```

`set_source_tenant_column` registers the override on the session's
`SourceTenantColumns` map. The next time the analyzer rule sees a scan
against `notes.public.notes`, it discovers the override and injects
`WHERE CAST(customer_id AS Utf8) = $current_tenant OR CAST(customer_id AS Utf8) IS NULL`
(or `IS NULL` only when the session is unscoped).

> The Python and CLI surfaces do not expose this method today — it lives
> on Rust `JammiSession` only. If you embed Jammi as a library this is
> the right hook; if you reach Jammi over Flight SQL / gRPC, the source
> registration and tenant-column declaration happen on the server side
> before the server starts accepting client connections.

Schema column `tenant_id` always wins. Only call
`set_source_tenant_column` when your federated source carries the
discriminator under a different name, or when it carries the
discriminator at all — sources without any tenant column remain
globally visible.

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
session_a.set_source_tenant_column("notes", Some("customer_id".into()));

let session_b = JammiSession::new(JammiConfig::default()).await?.with_tenant(bob);
session_b.set_source_tenant_column("notes", Some("customer_id".into()));

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

- **You cannot point `set_source_tenant_column` at a column that doesn't
  exist on the source.** The analyzer rule emits a column reference that
  DataFusion later fails to resolve at execution time, surfacing as a
  `DataFusionError::SchemaError`. The override is a trust contract — the
  engine does not validate the column's presence at registration time.
- **You cannot mix `tenant_id` and a non-`tenant_id` column on the same
  source.** When the source's schema already declares `tenant_id`, the
  built-in column wins and the override is ignored.
- **You cannot remove the discriminator at runtime once tenants are
  actively querying.** Call `set_source_tenant_column("notes", None)` to
  drop the override; subsequent queries on `notes.public.notes` will not
  be tenant-scoped at all.

If the federated source you are wrapping carries no tenant
discriminator, two options are open: (1) re-shape upstream so each
tenant lands in its own table, registered as a separate source, or
(2) accept that the source is globally visible to every session and
gate access at a higher layer (Flight SQL session interceptor, gRPC
auth middleware). The engine itself does not authenticate; ADR-00 §
*Engine does not invent tenants* applies.

## See also

- [Scope a Session to a Tenant](./multi-tenant.md) — the broader
  session-binding recipe this one extends.
- [Register a Mutable Companion Table](./register-mutable-table.md) —
  for sources Jammi owns, the `tenant_id` column comes from
  [ADR-00](https://github.com/f-inverse/jammi-ai/blob/main/docs/plans/cp9-substrate-primitives/ADR-00-tenant-identifier.md)
  by default.
