//! Live-Postgres parity tests for mutable companion tables.
//!
//! Mirrors the substrate-level coverage of `mutable_tables.rs` against a
//! real Postgres backend, exercising the same `Catalog` + `MutableTableRegistry`
//! lifecycle the SQLite harness does. Gated behind `live-postgres-tests` so
//! the default `cargo test` stays hermetic; the dedicated `test-pg` CI job
//! enables the feature with `JAMMI_TEST_PG_URL` pointing at a `postgres:15`
//! service container.
//!
//! Tests use per-call UUID-suffixed mutable-table identifiers so the same
//! database can host every test sequentially without clean-up between them;
//! the CI job runs `--test-threads=1` to serialise `migrate()` ledger writes
//! against the shared catalog tables.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use jammi_engine::catalog::backend::{BackendImpl, TxOptions};
use jammi_engine::catalog::backend_postgres::PostgresBackend;
use jammi_engine::catalog::Catalog;
use jammi_engine::store::mutable::definition::{
    MutableTableDefinitionBuilder, MutableTableError, MutableTableId,
};
use jammi_engine::store::mutable::postgres::PostgresMutableBackend;
use jammi_engine::store::mutable::MutableBackend;
use jammi_engine::tenant::TenantContext;

const ENV_VAR: &str = "JAMMI_TEST_PG_URL";

/// Connect to the live Postgres instance described by `JAMMI_TEST_PG_URL`,
/// apply pending migrations, and return a Catalog + matching mutable backend
/// suitable for direct-access registry construction.
async fn open_pg_catalog() -> (Arc<Catalog>, Arc<dyn MutableBackend>) {
    let url = std::env::var(ENV_VAR).unwrap_or_else(|_| {
        panic!(
            "{ENV_VAR} is required for live-postgres tests (e.g. \
             postgres://postgres:postgres@localhost:5432/jammi_test)"
        )
    });
    let pg = PostgresBackend::open(&url)
        .await
        .expect("open postgres backend");
    let backend_impl = BackendImpl::Postgres(pg);
    backend_impl
        .migrate()
        .await
        .expect("migrations apply against postgres");

    let catalog = Arc::new(Catalog::from_backend(backend_impl));
    let mutable: Arc<dyn MutableBackend> =
        Arc::new(PostgresMutableBackend::new(catalog.backend_arc()));
    (catalog, mutable)
}

fn widget_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
    ]))
}

fn events_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("seq", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, true),
    ]))
}

/// Mutable-table id suffixed with a monotonically increasing counter +
/// process-start epoch so concurrent CI runs (or repeated invocations) don't
/// collide on the shared catalog.
fn unique_id(prefix: &str) -> MutableTableId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    MutableTableId::new(format!("{prefix}_{epoch_ns:x}_{n:x}")).unwrap()
}

fn new_registry(
    catalog: Arc<Catalog>,
    backend: Arc<dyn MutableBackend>,
) -> Arc<jammi_engine::source::mutable::MutableTableRegistry> {
    let tenant_binding = Arc::new(RwLock::new(TenantContext::Unscoped));
    Arc::new(jammi_engine::source::mutable::MutableTableRegistry::new(
        catalog,
        backend,
        tenant_binding,
    ))
}

#[tokio::test]
async fn migrations_apply_against_live_postgres() {
    // Bare smoke test: opening the backend + running migrate() is itself the
    // assertion. If any DEFAULT expression or DDL statement is SQLite-only,
    // this test fails at migration time, before the rest of the harness even
    // touches the catalog.
    let (_catalog, _backend) = open_pg_catalog().await;
}

#[tokio::test]
async fn register_persists_catalog_row_and_storage_table() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    let loaded = registry
        .get(&id)
        .await
        .unwrap()
        .expect("table must be visible after register");
    assert_eq!(loaded.id.as_str(), id.as_str());
    assert_eq!(loaded.schema.fields().len(), 3);
    assert_eq!(loaded.primary_key, vec!["id".to_string()]);

    registry.drop_table(&id).await.unwrap();
    assert!(registry.get(&id).await.unwrap().is_none());
}

#[tokio::test]
async fn order_column_round_trips_through_catalog() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    let reloaded = registry
        .get(&id)
        .await
        .unwrap()
        .expect("table must be present");
    assert_eq!(reloaded.order_column.as_deref(), Some("seq"));

    registry.drop_table(&id).await.unwrap();
}

#[tokio::test]
async fn insert_batch_writes_via_caller_owned_transaction() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    let batch = RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int64Array::from(vec![100_i64, 101, 102])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let pg_backend = catalog.backend_arc();
    let registry_for_closure = Arc::clone(&registry);
    let id_for_closure = id.clone();
    let written = pg_backend
        .transaction(TxOptions::default(), move |tx| {
            let id = id_for_closure.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry_for_closure);
            Box::pin(async move {
                let n = registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| jammi_engine::BackendError::Execution(e.to_string()))?;
                Ok::<u64, jammi_engine::BackendError>(n)
            })
        })
        .await
        .unwrap();
    assert_eq!(written, 3);

    registry.drop_table(&id).await.unwrap();
}

#[tokio::test]
async fn scan_after_streams_rows_in_order() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    // Seed three rows via insert_batch.
    let batch = RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int64Array::from(vec![100_i64, 200, 300])),
            Arc::new(StringArray::from(vec!["old", "mid", "new"])),
        ],
    )
    .unwrap();
    let pg_backend = catalog.backend_arc();
    let registry_for_closure = Arc::clone(&registry);
    let id_for_closure = id.clone();
    pg_backend
        .transaction(TxOptions::default(), move |tx| {
            let id = id_for_closure.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry_for_closure);
            Box::pin(async move {
                registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| jammi_engine::BackendError::Execution(e.to_string()))?;
                Ok::<(), jammi_engine::BackendError>(())
            })
        })
        .await
        .unwrap();

    let mut stream = registry.scan_after(&id, 150).await.unwrap();
    let mut seqs: Vec<i64> = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        let col = batch
            .column_by_name("seq")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..col.len() {
            seqs.push(col.value(i));
        }
    }
    assert_eq!(seqs, vec![200, 300]);

    registry.drop_table(&id).await.unwrap();
}

#[tokio::test]
async fn scan_after_errors_when_order_column_missing() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("noorder");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    match registry.scan_after(&id, 0).await {
        Ok(_) => panic!("scan_after must reject a table with no order_column"),
        Err(MutableTableError::NoOrderColumn) => {}
        Err(other) => panic!("expected NoOrderColumn, got {other:?}"),
    }

    registry.drop_table(&id).await.unwrap();
}

#[tokio::test]
async fn insert_batch_rejects_schema_mismatch() {
    let (catalog, backend) = open_pg_catalog().await;
    let registry = new_registry(Arc::clone(&catalog), Arc::clone(&backend));

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    registry.register(def).await.unwrap();

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();

    let pg_backend = catalog.backend_arc();
    let registry_for_closure = Arc::clone(&registry);
    let id_for_closure = id.clone();
    let err = pg_backend
        .transaction(TxOptions::default(), move |tx| {
            let id = id_for_closure.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry_for_closure);
            Box::pin(async move {
                match registry.insert_batch(tx, &id, &batch).await {
                    Ok(_) => Ok::<(), jammi_engine::BackendError>(()),
                    Err(MutableTableError::Schema(msg)) => Err(
                        jammi_engine::BackendError::Execution(format!("SCHEMA_MISMATCH:{msg}")),
                    ),
                    Err(other) => Err(jammi_engine::BackendError::Execution(other.to_string())),
                }
            })
        })
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SCHEMA_MISMATCH"),
        "expected schema mismatch error, got: {msg}"
    );

    registry.drop_table(&id).await.unwrap();
}
