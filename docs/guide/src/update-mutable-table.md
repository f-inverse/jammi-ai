# Run Transactional Updates on a Mutable Table

Once a mutable companion table is registered (see
[Register a Mutable Companion Table](./register-mutable-table.md)), you
change its rows with the same SQL surface that runs your read queries.
Each `INSERT`, `REPLACE INTO`, `UPDATE`, and `DELETE` statement is one
backend transaction — every row it touches commits or none does — and
federates with your immutable result tables in subsequent SELECTs.

## Goal

Walk through the DML verbs against the `item_dimensions` table from the
previous recipe (primary key `(item_id, valid_from)`), then record a
price-tier change as a Slowly-Changing Dimension Type 2 close-and-open.

## Insert

```sql
INSERT INTO mutable.public.item_dimensions
    (item_id, price_tier, availability, valid_from)
VALUES
    ('sku-1842', 'standard', 'in_stock', '2026-04-01T00:00:00Z'),
    ('sku-2901', 'premium',  'in_stock', '2026-04-01T00:00:00Z'),
    ('sku-3457', 'standard', 'out_of_stock', '2026-04-01T00:00:00Z');
```

Every DML statement answers with a single-row `UInt64` column called
`count` — here, 3. An `INSERT` whose key already exists fails, and fails
whole: none of its rows land.

## Upsert — `REPLACE INTO`

```sql
REPLACE INTO mutable.public.item_dimensions
    (item_id, price_tier, availability, valid_from)
VALUES
    ('sku-2901', 'premium', 'low_stock', '2026-04-01T00:00:00Z'),
    ('sku-5120', 'standard', 'in_stock', '2026-04-01T00:00:00Z');
```

`REPLACE INTO` upserts by primary key: a row whose key exists replaces the
existing row, and a new key is inserted. `count` is the number of rows
written — here, 2.

## Update

```sql
UPDATE mutable.public.item_dimensions
   SET availability = 'low_stock'
 WHERE item_id = 'sku-3457';
```

The predicate and each assignment may be any expression over the row —
`SET score = score * 10 WHERE score > 1` evaluates against each matched
row's own values. A row whose predicate is `NULL` is not matched (SQL
three-valued logic). `count` is the number of rows matched; a predicate
that matches nothing succeeds with `count = 0`.

## Delete

```sql
DELETE FROM mutable.public.item_dimensions
 WHERE item_id = 'sku-5120';
```

A `DELETE` with no `WHERE` clause empties the table. Row-level cascades
are the backend's job (the foreign-key declarations on the storage
table); the engine does not model cascades above the backend.

## Tenancy

A session bound to a tenant reads its own rows and the global
(`tenant_id IS NULL`) rows, but an `UPDATE`, `DELETE`, or `REPLACE INTO`
rewrites only its own. An unfiltered `DELETE` from a tenant-bound session
leaves the global rows, and every other tenant's rows, in place — no
statement moves a row across tenants.

## SCD Type 2 — close-and-open

A price-tier change closes the active row's `valid_to` and inserts a new
row with the new tier and a new `valid_from`:

```sql
UPDATE mutable.public.item_dimensions
   SET valid_to = '2026-05-15T12:00:00Z'
 WHERE item_id = 'sku-1842' AND valid_to IS NULL;
```

```sql
INSERT INTO mutable.public.item_dimensions
    (item_id, price_tier, availability, valid_from)
VALUES
    ('sku-1842', 'premium', 'in_stock', '2026-05-15T12:00:00Z');
```

`session.sql` plans one statement per call, so these are two
transactions. A reader between them sees the item with no open row. When
the close and the open must land together, write both through one
caller-owned transaction on the direct-access path below.

## Federation join

The mutable table now joins with the embedding table to surface
recommender candidates filtered by current tier:

```sql
SELECT  d.item_id, d.price_tier, e.embedding
  FROM  mutable.public.item_dimensions d
  JOIN  itemembs.public.item_embeddings e ON e.item_id = d.item_id
 WHERE  d.valid_to IS NULL
   AND  d.price_tier = 'premium'
 LIMIT 10;
```

The federation is the engine's existing `FederationOptimizerRule` work
— no special integration needed; mutable tables resolve in the same
session as your Parquet result tables and external sources.

## Crash recovery

If the process dies mid-write, no partial commit is visible on restart.
SQLite's WAL mode ([documentation](https://www.sqlite.org/wal.html))
and Postgres's MVCC each guarantee that an open transaction either
commits as a whole or is rolled back on connection loss. The engine
inherits that guarantee through the `CatalogBackend::transaction`
closure shape: when the closure returns `Err(_)`, the backend rolls
back; when the process is killed mid-execution, the backend rolls back
the in-flight transaction.

## Direct-access append + replay

Two lower-level methods bypass DataFusion's planner for high-throughput
event paths:

```rust,no_run
# extern crate jammi_db;
# extern crate arrow;
# extern crate tokio;
# async fn ex(
#     session: &jammi_db::session::JammiSession,
#     batch: arrow::array::RecordBatch,
# ) -> jammi_db::error::Result<()> {
use jammi_db::store::mutable::definition::MutableTableId;
use jammi_db::catalog::backend::TxOptions;

let id = MutableTableId::new("events").unwrap();
let registry = session.mutable_tables_arc();
let backend = session.catalog().backend_arc();

// Direct INSERT via insert_batch — caller owns the transaction.
backend
    .transaction(TxOptions::default(), move |tx| {
        let registry = registry.clone();
        let id = id.clone();
        let batch = batch.clone();
        Box::pin(async move {
            registry
                .insert_batch(tx, &id, &batch)
                .await
                .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))?;
            Ok::<(), jammi_db::BackendError>(())
        })
    })
    .await?;
# Ok(())
# }
```

```rust,no_run
# extern crate jammi_db;
# extern crate futures;
# extern crate tokio;
# use futures::StreamExt;
# async fn ex(
#     session: &jammi_db::session::JammiSession,
# ) -> jammi_db::error::Result<()> {
use jammi_db::store::mutable::definition::MutableTableId;

let id = MutableTableId::new("events").unwrap();
// Stream rows where the registered `order_column` value > 100.
let mut stream = session
    .mutable_tables()
    .scan_after(&id, 100)
    .await
    .map_err(|e| jammi_db::error::JammiError::Catalog(e.to_string()))?;
while let Some(batch) = stream.next().await {
    let _batch = batch
        .map_err(|e| jammi_db::error::JammiError::Catalog(e.to_string()))?;
    // …
}
# Ok(())
# }
```

These are the surface the trigger broker uses to publish events into a
backing table and replay subscribers. The SQL surface is the general one;
reach for `insert_batch` when several writes must share one transaction.
