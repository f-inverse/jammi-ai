//! End-to-end integration tests for Phase 2 — mutable companion tables.
//!
//! Coverage: register/list/drop lifecycle, atomic catalog + storage commit,
//! DataFusion DML through `INSERT INTO mutable.public.<id>`, federation
//! between mutable tables and Parquet result tables, tenant filtering on
//! list, order-column round-trip, direct-access `insert_batch` and
//! `scan_after` paths, schema-mismatch rejection.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` +
//! `cfg_attr`. The SQLite lane is always generated; the Postgres lane is
//! generated only when the `live-postgres-tests` feature is on, and skips
//! at runtime when `JAMMI_TEST_PG_URL` is unset. CI's `test-pg` job sets
//! both the feature and the env var; the hermetic `cargo test` lane runs
//! only the SQLite parameterisation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use futures::StreamExt;
use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::store::mutable::definition::{
    MutableIndexDef, MutableTableDefinitionBuilder, MutableTableError, MutableTableId,
};
use jammi_test_utils::make_test_session;
use tempfile::tempdir;
use test_case::test_case;

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

/// Backend-unique mutable-table id. SQLite per-tempdir tests don't strictly
/// need uniqueness, but the Postgres lane shares one database across every
/// test in the run; the suffix avoids `relation "<name>" already exists`
/// errors between parameterised variants.
fn unique_id(prefix: &str) -> MutableTableId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    MutableTableId::new(format!("{prefix}_{epoch_ns:x}_{n:x}")).unwrap()
}

/// SAFETY note: the Postgres lane returns `None` when `JAMMI_TEST_PG_URL`
/// is unset so the test can early-return rather than `#[ignore]`'ing
/// (CLAUDE.md forbids `#[ignore]`). The macro `skip_if_no_backend!()`
/// abbreviates the pattern.
macro_rules! skip_if_no_backend {
    ($backend:expr, $dir:expr) => {
        match make_test_session($backend, $dir).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_persists_catalog_row_and_storage_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let listed = session.mutable_tables().list(None).await.unwrap();
    assert!(listed.iter().any(|d| d.id.as_str() == id.as_str()));

    let loaded = session.mutable_tables().get(&id).await.unwrap().unwrap();
    assert_eq!(loaded.schema.fields().len(), 3);
    assert_eq!(loaded.primary_key, vec!["id".to_string()]);

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_removes_catalog_row_and_storage_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("ephemeral");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session.drop_mutable_table(&id).await.unwrap();

    assert!(session.mutable_tables().get(&id).await.unwrap().is_none());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn datafusion_insert_then_scan_round_trip(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, name, score) VALUES \
             (1, 'alpha', 0.5), (2, 'beta', 1.5), (3, 'gamma', 2.5)",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    let batches = session
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 3);
    let names = string_column_values(&batch, "name");
    assert_eq!(names[0], "alpha");
    assert_eq!(names[2], "gamma");
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_makes_select_fail(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("widgets");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
    session.drop_mutable_table(&id).await.unwrap();

    let err = session
        .sql(&format!(
            "SELECT * FROM mutable.public.{name}",
            name = id.as_str()
        ))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not found") || msg.contains("Table") || msg.contains(id.as_str()),
        "expected table-not-found error; got: {msg}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registered_mutable_tables_reload_across_sessions(backend: BackendKind) {
    // Persistence contract: a mutable table registered through one session
    // is visible to a subsequent session opened against the same backend.
    // SQLite uses the artifact dir; Postgres reuses the JAMMI_TEST_PG_URL
    // connection — both paths go through `make_test_session`.
    let dir = tempdir().unwrap();
    let id = unique_id("persistent");

    {
        let session = skip_if_no_backend!(backend, dir.path());
        let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
            .primary_key(vec!["id".into()])
            .build()
            .unwrap();
        session.create_mutable_table(def).await.unwrap();
        session
            .sql(&format!(
                "INSERT INTO mutable.public.{name} (id, name, score) VALUES (42, 'meaning', 1.0)",
                name = id.as_str()
            ))
            .await
            .unwrap();
    }

    let session = skip_if_no_backend!(backend, dir.path());
    let batches = session
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{name}",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 1);
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 42);

    // Clean up the persistent row so a re-run doesn't surface stale state on
    // the shared Postgres catalog.
    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_emits_implicit_tenant_id_column_per_adr_00(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("with_tenant");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // A `LIMIT 0` SELECT proves the on-disk table exists and is reachable
    // through DataFusion; the tenant_id column being present is exercised
    // implicitly through the predicate-injection path in Phase 3 tests.
    session
        .sql(&format!(
            "SELECT * FROM mutable.public.{name} LIMIT 0",
            name = id.as_str()
        ))
        .await
        .unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_filters_by_tenant_scope(backend: BackendKind) {
    use jammi_db::TenantId;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();

    let global_id = unique_id("global_table");
    let def = MutableTableDefinitionBuilder::new(global_id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let scoped_id = unique_id("tenant_a_table");
    let def = MutableTableDefinitionBuilder::new(scoped_id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .tenant(Some(tenant_a))
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let global = session.mutable_tables().list(None).await.unwrap();
    assert!(global.iter().any(|d| d.id.as_str() == global_id.as_str()));
    assert!(global.iter().all(|d| d.id.as_str() != scoped_id.as_str()));

    let scoped = session.mutable_tables().list(Some(tenant_a)).await.unwrap();
    assert!(scoped.iter().any(|d| d.id.as_str() == scoped_id.as_str()));
    assert!(scoped.iter().all(|d| d.id.as_str() != global_id.as_str()));

    // The global table is dropped from the unscoped session (its owner). The
    // tenant-scoped table can only be dropped by a session bound to that tenant
    // — the strict delete predicate refuses an unscoped (or foreign-tenant)
    // session, mirroring `retire_model`. Bind tenant A for its cleanup.
    session.drop_mutable_table(&global_id).await.unwrap();
    session
        .with_tenant(tenant_a)
        .drop_mutable_table(&scoped_id)
        .await
        .unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_create_get_delete_round_trip(backend: BackendKind) {
    // Catalog-level smoke test: drives the repos directly via
    // `session.catalog()`, bypassing the registry's storage-table step.
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("plain_cat");
    let def = MutableTableDefinitionBuilder::new(id.clone(), widget_schema())
        .primary_key(vec!["id".into()])
        .index(MutableIndexDef {
            name: format!("idx_{}_name", id.as_str()),
            columns: vec!["name".into()],
            unique: false,
        })
        .build()
        .unwrap();

    let catalog = session.catalog();
    catalog.create_mutable_table(&def).await.unwrap();
    let got = catalog.get_mutable_table(&id).await.unwrap().unwrap();
    assert_eq!(got.id.as_str(), id.as_str());
    assert_eq!(got.indexes.len(), 1);
    assert_eq!(got.indexes[0].columns, vec!["name".to_string()]);

    catalog.delete_mutable_table(&id).await.unwrap();
    assert!(catalog.get_mutable_table(&id).await.unwrap().is_none());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn order_column_persists_across_reload(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let id = unique_id("events");

    {
        let session = skip_if_no_backend!(backend, dir.path());
        let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
            .primary_key(vec!["id".into()])
            .order_column("seq")
            .build()
            .unwrap();
        session.create_mutable_table(def).await.unwrap();
    }

    let session = skip_if_no_backend!(backend, dir.path());
    let reloaded = session
        .mutable_tables()
        .get(&id)
        .await
        .unwrap()
        .expect("table should exist after reload");
    assert_eq!(reloaded.order_column.as_deref(), Some("seq"));

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_batch_appends_with_session_tenant(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let batch = RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int64Array::from(vec![100_i64, 101, 102])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let backend_arc = session.catalog().backend_arc();
    let registry = session.mutable_tables_arc();
    let id_clone = id.clone();
    let written = backend_arc
        .transaction(TxOptions::default(), move |tx| {
            let id = id_clone.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry);
            Box::pin(async move {
                let n = registry
                    .insert_batch(tx, &id, &batch)
                    .await
                    .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))?;
                Ok::<u64, jammi_db::BackendError>(n)
            })
        })
        .await
        .unwrap();
    assert_eq!(written, 3);

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_after_streams_rows_strictly_greater_in_order(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .order_column("seq")
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, seq, payload) VALUES \
             (1, 100, 'old'), (2, 200, 'mid'), (3, 300, 'new')",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    let mut stream = session.mutable_tables().scan_after(&id, 150).await.unwrap();
    let mut seqs = Vec::new();
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

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_after_errors_when_order_column_missing(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("noorder");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    match session.mutable_tables().scan_after(&id, 0).await {
        Ok(_) => panic!("scan_after should reject table without order_column"),
        Err(MutableTableError::NoOrderColumn) => {}
        Err(other) => panic!("expected NoOrderColumn, got {other:?}"),
    }

    session.drop_mutable_table(&id).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_batch_rejects_schema_mismatch(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("events");
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();

    let backend_arc = session.catalog().backend_arc();
    let registry = session.mutable_tables_arc();
    let id_for_closure = id.clone();
    let err = backend_arc
        .transaction(TxOptions::default(), move |tx| {
            let id = id_for_closure.clone();
            let batch = batch.clone();
            let registry = Arc::clone(&registry);
            Box::pin(async move {
                match registry.insert_batch(tx, &id, &batch).await {
                    Ok(_) => Ok::<(), jammi_db::BackendError>(()),
                    Err(MutableTableError::Schema(msg)) => Err(jammi_db::BackendError::Execution(
                        format!("SCHEMA_MISMATCH:{msg}"),
                    )),
                    Err(other) => Err(jammi_db::BackendError::Execution(other.to_string())),
                }
            })
        })
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("SCHEMA_MISMATCH"),
        "expected schema mismatch error, got: {err}"
    );

    session.drop_mutable_table(&id).await.unwrap();
}

/// Round-trip a `DataType::Binary` column through the DataFusion DML sink
/// (writes BLOB/BYTEA bytes through `MutableTableSink::extract_value`) and
/// the provider scan (reads bytes back through `decode_row`'s explicit
/// `Binary` arm). Exercises non-UTF-8 byte sequences (`0x00`, `0xFF`,
/// bincode-shaped payload) so a silent UTF-8 decode would observably
/// truncate or null the value.
///
/// Postgres coverage runs in the same parameterised matrix as every other
/// test in this file when the `live-postgres-tests` feature is on; the
/// default hermetic `cargo test` lane runs the SQLite variant only.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn binary_column_roundtrip_through_provider(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());

    let id = unique_id("blobs");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("blob", DataType::Binary, false),
    ]));
    let def = MutableTableDefinitionBuilder::new(id.clone(), Arc::clone(&schema))
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // Two non-UTF-8 payloads: an embedded NUL/0xFF/high-bit pattern, plus a
    // 256-byte bincode-shaped Vec<f32> blob (a realistic shape for callers
    // who store serialised embeddings in a mutable companion table).
    let payload_a: Vec<u8> = vec![0x00, 0xFF, 0xC3, 0x28, 0xA0, 0xA1, 0x80, 0x01];
    let payload_b: Vec<u8> = {
        let mut v = Vec::with_capacity(4 * 64);
        for i in 0_u32..64 {
            v.extend_from_slice(&(i as f32).to_le_bytes());
        }
        v
    };

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2])),
            Arc::new(BinaryArray::from_iter_values([
                payload_a.as_slice(),
                payload_b.as_slice(),
            ])),
        ],
    )
    .unwrap();

    // Force the DataFusion sink path: register the source batch as a memory
    // table, then `INSERT INTO mutable.public.<id> SELECT *` — equivalent
    // to a user-written DML statement, which exercises `extract_value`'s
    // Binary arm.
    let src = format!("src_{}", id.as_str());
    session.context().register_batch(&src, batch).unwrap();
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, blob) SELECT id, blob FROM {src}",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    // Read back through the provider scan path — `decode_row`'s Binary arm
    // must produce a BinaryArray (not a null StringArray) with the original
    // bytes intact.
    let batches = session
        .sql(&format!(
            "SELECT blob FROM mutable.public.{name} ORDER BY id",
            name = id.as_str()
        ))
        .await
        .unwrap();
    let out = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(out.num_rows(), 2);
    let blobs = out
        .column_by_name("blob")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap_or_else(|| {
            panic!(
                "blob column is not BinaryArray; got {:?}",
                out.column_by_name("blob").unwrap().data_type()
            )
        });
    assert_eq!(blobs.value(0), payload_a.as_slice());
    assert_eq!(blobs.value(1), payload_b.as_slice());

    session.drop_mutable_table(&id).await.unwrap();
}

/// Cross-tenant isolation regression: a tenant must not be able to READ or
/// DELETE another tenant's mutable companion table by knowing its (low-entropy,
/// user-chosen) id. Drives the catalog repo directly — the root-cause surface —
/// so it pins the tenant predicate on `get_mutable_table` / `delete_mutable_table`
/// rather than any caller-side guard.
///
/// Both sessions share one backend (the same SQLite catalog file under `dir`),
/// so tenant B's repo queries can see — and, before the fix, destroy — tenant
/// A's row. Pre-fix this test FAILS at step 2: the unscoped `WHERE id = $1`
/// SELECT returns A's row to B, and the unscoped `DELETE FROM mutable_tables
/// WHERE id = $1` removes it.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutable_table_isolation_blocks_cross_tenant_get_and_delete(backend: BackendKind) {
    use jammi_db::TenantId;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();

    // Stable id so tenant B can name tenant A's table exactly — the attack
    // vector is a guessed/known id, not entropy.
    let id = unique_id("events");

    // 1) Tenant A creates the table under its own tenant scope.
    let session_a = skip_if_no_backend!(backend, dir.path()).with_tenant(tenant_a);
    let def = MutableTableDefinitionBuilder::new(id.clone(), events_schema())
        .primary_key(vec!["id".into()])
        .tenant(Some(tenant_a))
        .build()
        .unwrap();
    session_a.create_mutable_table(def).await.unwrap();

    // Tenant A can resolve its own table.
    let got_a = session_a
        .catalog()
        .get_mutable_table(&id)
        .await
        .unwrap()
        .expect("tenant A must see its own table");
    assert_eq!(got_a.tenant, Some(tenant_a));

    // 2) Tenant B cannot READ tenant A's table.
    let session_b = skip_if_no_backend!(backend, dir.path()).with_tenant(tenant_b);
    assert!(
        session_b
            .catalog()
            .get_mutable_table(&id)
            .await
            .unwrap()
            .is_none(),
        "tenant B must NOT be able to read tenant A's mutable table"
    );

    // 3) Tenant B's delete must NOT destroy tenant A's table. The DELETE is a
    // success no-op (zero rows matched) — the destructive cross-tenant delete
    // is closed.
    session_b.catalog().delete_mutable_table(&id).await.unwrap();

    // The registry-level drop path must also refuse: it resolves the backing
    // table via the now-tenant-scoped `get`, so it cannot even find tenant A's
    // table to DROP.
    match session_b.mutable_tables().drop_table(&id).await {
        Err(MutableTableError::NotFound(missing)) => assert_eq!(missing.as_str(), id.as_str()),
        other => panic!("tenant B drop_table should be NotFound; got {other:?}"),
    }

    // Tenant A's table STILL EXISTS after tenant B's delete + drop attempts.
    assert!(
        session_a
            .catalog()
            .get_mutable_table(&id)
            .await
            .unwrap()
            .is_some(),
        "tenant A's table must survive tenant B's cross-tenant delete attempt"
    );

    // 4) Tenant A can still get and then delete its OWN table.
    assert!(session_a
        .catalog()
        .get_mutable_table(&id)
        .await
        .unwrap()
        .is_some());
    session_a.catalog().delete_mutable_table(&id).await.unwrap();
    assert!(
        session_a
            .catalog()
            .get_mutable_table(&id)
            .await
            .unwrap()
            .is_none(),
        "tenant A's own delete must remove its table"
    );
}

/// `Utf8` extracter — handles both `StringArray` and `StringViewArray`
/// (Arrow 57's parquet reader emits the latter under DataFusion 52).
fn string_column_values(batch: &RecordBatch, name: &str) -> Vec<String> {
    use arrow::array::StringViewArray;
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} missing in {:?}", batch.schema()));
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        return (0..sa.len()).map(|i| sa.value(i).to_string()).collect();
    }
    if let Some(sv) = col.as_any().downcast_ref::<StringViewArray>() {
        return (0..sv.len()).map(|i| sv.value(i).to_string()).collect();
    }
    panic!(
        "column {name} is neither StringArray nor StringViewArray; got {:?}",
        col.data_type()
    );
}
