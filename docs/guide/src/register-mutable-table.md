# Register a Mutable Companion Table

A *mutable companion table* lives in the same backend database as the
Jammi catalog (SQLite by default, Postgres in shared deployments), supports
transactional `INSERT` / `UPDATE` / `DELETE` through DataFusion DML, and
federates with Parquet result tables and external sources in one SQL
surface. Reach for it when a tenant needs a relation it can edit row by row
— a feature-store slowly-changing dimension table, a per-user state table,
a config-driven lookup — that still has to participate in the same JOINs as
your immutable result tables.

The primitive carries only what every consumer needs: a schema, a primary
key, optional tenant scope, optional secondary indexes, optional ordering
column. No history semantics, no lifecycle vocabulary, no audit columns.

## Goal

This recipe walks through registering one mutable companion table for a
neutral third-party use case (a feature-store team called *Polaris Features*
maintaining slowly-changing dimensions for their recommender) and shows the
equivalent Rust / Python / CLI surface.

## Setup

Assumes a working `JammiSession`. The session opens the catalog at the
configured artifact directory; nothing else is needed.

## Define the schema

Polaris keeps one row per `(item_id, valid_from, valid_to)` interval:

```rust,no_run
# extern crate arrow_schema;
# fn make() {
use std::sync::Arc;
use arrow_schema::{DataType, Field, Schema};

let schema = Arc::new(Schema::new(vec![
    Field::new("item_id",      DataType::Utf8,    false),
    Field::new("price_tier",   DataType::Utf8,    false),
    Field::new("availability", DataType::Utf8,    false),
    Field::new("valid_from",   DataType::Int64,   false),  // epoch milliseconds
    Field::new("valid_to",     DataType::Int64,   true),   // epoch milliseconds; NULL = open
]));
# }
```

The catalog encoder accepts the closed primitive subset enforced by every
`MutableBackend` impl — `Boolean`, the integer family, `Float32` / `Float64`,
`Utf8`, `Binary`. Wider types (e.g. `Timestamp`, `Decimal`) round-trip via
their natural numeric encoding (`Int64` epoch milliseconds, scaled `Int64`)
so the schema stays narrow and the rule stays one-line at the boundary.

The engine reserves `tenant_id` and any column whose name starts with `_`
— the schema builder rejects them at build time per ADR-00. (The
`tenant_id` column is always present on the storage table; the engine
appends it implicitly.)

## Build the definition

`MutableTableDefinitionBuilder` chains the field validations:

```rust,no_run
# extern crate jammi_engine;
# extern crate arrow_schema;
# use std::sync::Arc;
# use arrow_schema::Schema;
use jammi_engine::store::mutable::definition::{
    MutableIndexDef, MutableTableDefinitionBuilder, MutableTableId,
};

# fn make(schema: Arc<Schema>) -> jammi_engine::store::mutable::definition::MutableTableDefinition {
let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("item_dimensions").unwrap(),
        schema,
    )
    .primary_key(vec!["item_id".into(), "valid_from".into()])
    .index(MutableIndexDef {
        name: "idx_item_dim_active".into(),
        columns: vec!["item_id".into(), "valid_to".into()],
        unique: false,
    })
    .build()
    .unwrap();
# def
# }
```

The primary key must be a non-empty subset of the schema; secondary indexes
are optional but persisted on the storage table so the backend can use
them for `WHERE` clauses.

## Register

The registration is atomic: catalog row + storage `CREATE TABLE` + every
secondary `CREATE INDEX` commit together. If any step fails, nothing lands.

### Rust

```rust,no_run
# extern crate jammi_engine;
# extern crate tokio;
# use jammi_engine::store::mutable::definition::MutableTableDefinition;
# use jammi_engine::session::JammiSession;
# async fn ex(session: &JammiSession, def: MutableTableDefinition) -> jammi_engine::error::Result<()> {
let id = session.create_mutable_table(def).await?;
// The table is now queryable as `mutable.public.item_dimensions` in the
// same SQL surface that federates result tables and external sources.
# Ok(())
# }
```

### Python

```python
import pyarrow as pa
import jammi

db = jammi.connect(artifact_dir="/var/lib/jammi")
# The Python wrapper exposes mutable-table registration through the
# `create_mutable_table` accessor (see `jammi.mutable`). The recipe below
# is illustrative; consult the API reference for the binding shape your
# version ships.
```

### CLI

The `jammi` CLI exposes mutable-table registration through the lower-level
`sources` surface for now; programmatic clients should use the Rust or
Python APIs.

## Verify

```rust,no_run
# extern crate jammi_engine;
# extern crate tokio;
# async fn ex(session: &jammi_engine::session::JammiSession) -> jammi_engine::error::Result<()> {
let zero_rows = session
    .sql("SELECT * FROM mutable.public.item_dimensions LIMIT 0")
    .await?;
assert_eq!(zero_rows[0].schema().fields().len(), 5);
# Ok(())
# }
```

The query returns a zero-row batch with the declared schema — confirmation
that the table is registered and DataFusion can route `mutable.public.<id>`
correctly.

## Federation tease

The mutable table now JOINs with your existing result tables and sources:

```sql
SELECT  d.item_id, d.price_tier, e.embedding
FROM    mutable.public.item_dimensions d
JOIN    itemembs.public.item_embeddings e ON e.item_id = d.item_id
WHERE   d.valid_to IS NULL
  AND   d.price_tier = 'premium'
LIMIT 10;
```

See the [Run Transactional Updates on a Mutable Table](./update-mutable-table.md)
recipe for INSERT / UPDATE / DELETE round-trips and the SCD Type 2
close-and-open pattern.
