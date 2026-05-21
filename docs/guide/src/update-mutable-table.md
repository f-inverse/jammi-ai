# Run Transactional Updates on a Mutable Table

Once a mutable companion table is registered (see
[Register a Mutable Companion Table](./register-mutable-table.md)), you
update its rows with the same SQL surface that runs your read queries.
Every `INSERT` / `UPDATE` / `DELETE` lands in one backend transaction —
either every row commits or none does — and federates with your immutable
result tables in subsequent SELECTs.

## Goal

Walk through the three DML verbs against the `item_dimensions` table from
the previous recipe, then demonstrate the Slowly-Changing Dimension Type
2 close-and-open pattern Polaris uses to record a price-tier change.

## Insert

```sql
INSERT INTO mutable.public.item_dimensions
    (item_id, price_tier, availability, valid_from)
VALUES
    ('sku-1842', 'standard', 'in_stock', '2026-04-01T00:00:00Z'),
    ('sku-2901', 'premium',  'in_stock', '2026-04-01T00:00:00Z'),
    ('sku-3457', 'standard', 'out_of_stock', '2026-04-01T00:00:00Z');
```

The `RecordBatch` returned by `session.sql(...)` carries a single-row
`UInt64` column called `count` per DataFusion's `TableProvider::insert_into`
contract. Three rows landed; the JOIN-against-result-tables query from
the previous recipe now returns three rows.

## Update

```sql
UPDATE mutable.public.item_dimensions
   SET availability = 'low_stock'
 WHERE item_id = 'sku-2901';
```

Predicate columns that participate in an index pushdown become a backend
`WHERE` clause; the rest filter above the scan node. The update commits
in one transaction; if the predicate matches zero rows, the call succeeds
with `rows_affected = 0`.

## Delete

```sql
DELETE FROM mutable.public.item_dimensions
 WHERE item_id = 'sku-3457';
```

`DELETE` follows the same shape. Row-level cascades are SQLite's job (the
foreign-key declarations on the storage table); the engine does not model
cascades above the backend.

## SCD Type 2 — close-and-open

Polaris records a price-tier change by closing the active row's
`valid_to` and inserting a new row with the new tier. Both statements
must land atomically; today the supported pattern is to issue them as a
single multi-statement SQL string through `session.sql`, which DataFusion
plans as one DML batch under one transaction:

```sql
-- Single sql() call so both statements land in one transaction.
UPDATE mutable.public.item_dimensions
   SET valid_to = '2026-05-15T12:00:00Z'
 WHERE item_id = 'sku-1842' AND valid_to IS NULL;

INSERT INTO mutable.public.item_dimensions
    (item_id, price_tier, availability, valid_from)
VALUES
    ('sku-1842', 'premium', 'in_stock', '2026-05-15T12:00:00Z');
```

A future `JammiSession::transaction(|tx| async { … })` API will make
multi-statement DML atomicity explicit; today the multi-statement SQL
string is the supported surface.

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
— no special integration needed; mutable tables register under the same
`SessionContext` as your Parquet result tables and external sources.

## Crash recovery

If the process dies mid-write, no partial commit is visible on restart.
SQLite's WAL mode ([documentation](https://www.sqlite.org/wal.html))
and Postgres's MVCC each guarantee that an open transaction either
commits as a whole or is rolled back on connection loss. The engine
inherits that guarantee through the `CatalogBackend::transaction`
closure shape: when the closure returns `Err(_)`, the backend rolls
back; when the process is killed mid-execution, the backend rolls back
the in-flight transaction.

## Direct-access append + replay (Phase 4 trigger streams)

Two lower-level methods bypass DataFusion's planner for high-throughput
event paths:

```rust,no_run
# extern crate jammi_engine;
# extern crate arrow;
# extern crate tokio;
# async fn ex(
#     session: &jammi_engine::session::JammiSession,
#     batch: arrow::array::RecordBatch,
# ) -> jammi_engine::error::Result<()> {
use jammi_engine::store::mutable::definition::MutableTableId;
use jammi_engine::catalog::backend::TxOptions;

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
                .map_err(|e| jammi_engine::BackendError::Execution(e.to_string()))?;
            Ok::<(), jammi_engine::BackendError>(())
        })
    })
    .await?;
# Ok(())
# }
```

```rust,no_run
# extern crate jammi_engine;
# extern crate futures;
# extern crate tokio;
# use futures::StreamExt;
# async fn ex(
#     session: &jammi_engine::session::JammiSession,
# ) -> jammi_engine::error::Result<()> {
use jammi_engine::store::mutable::definition::MutableTableId;

let id = MutableTableId::new("events").unwrap();
// Stream rows where the registered `order_column` value > 100.
let mut stream = session
    .mutable_tables()
    .scan_after(&id, 100)
    .await
    .map_err(|e| jammi_engine::error::JammiError::Catalog(e.to_string()))?;
while let Some(batch) = stream.next().await {
    let _batch = batch
        .map_err(|e| jammi_engine::error::JammiError::Catalog(e.to_string()))?;
    // …
}
# Ok(())
# }
```

These are the surface Phase 4's trigger broker uses to publish events
into a backing table and replay subscribers; general consumers should
prefer the SQL surface.
