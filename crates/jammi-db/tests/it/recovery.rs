//! Crash-consistency of the catalog↔result-table-storage boundary.
//!
//! Object storage cannot join the catalog transaction, so a result table is
//! published in two steps — write the Parquet (and sidecar) bytes, then flip a
//! single catalog row `building → ready`. The guarantee is **crash-consistent
//! eventual reconciliation** via that status gate plus a startup sweep
//! ([`ResultStore::recover`]), not a distributed transaction.
//!
//! These tests prove the guarantee by *constructing* each post-crash torn
//! state directly — the recovery logic does not care how a torn state arose, so
//! we write the catalog row plus the bytes in the exact shape a crash would
//! leave them, then run the real `recover()` + `load_existing_tables` and assert
//! the invariants. A test that passed *without* first constructing the torn
//! state would be vacuous; non-vacuity is the bar.
//!
//! Invariants asserted after recovery:
//! - **I1** a non-`Ready`/torn table is not queryable (never registered);
//! - **I2** reconciliation is terminal — no row left `building` (`Ready` XOR
//!   `Failed`);
//! - **I3** no `Ready` row points at missing bytes;
//! - **I4** no live orphan — every Parquet object under the result root is
//!   pointed at by a `Ready` row, or was reaped;
//! - **I5** a promoted `Ready` row's `row_count` equals the true Parquet footer
//!   row count;
//! - **I6** an embedding table's sidecar self-heals — `resolve_search_mode`
//!   returns a working index post-recovery, rebuilt from the Parquet if the
//!   sidecar was absent or torn.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case`: the SQLite
//! lane runs on the hermetic `cargo test` lane, the Postgres lane is generated
//! under the `live-postgres-tests` feature and skips at runtime when
//! `JAMMI_TEST_PG_URL` is unset (CI's "Test (Postgres)" job runs it). The
//! Postgres lane shares one catalog DB across the run, so each test first clears
//! `result_tables` (CI runs that lane `--test-threads=1`, so the
//! reset-then-populate cannot race a sibling — important because `recover()` is
//! a cross-tenant admin scan that would otherwise see a sibling's rows).

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use bytes::Bytes;
use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::{BackendImpl, BackendKind, TxOptions};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{ResultStore, ResultTableInfo};
use jammi_db::TenantId;
use jammi_test_utils::pg_url_for_tests;
use tempfile::tempdir;
use test_case::test_case;

const DIMS: usize = 4;

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
}

/// Build a catalog backend of `kind`, or `None` for the Postgres lane when
/// `JAMMI_TEST_PG_URL` is unset (early-return rather than `#[ignore]`, which
/// CLAUDE.md forbids). The SQLite catalog lives at `<dir>/catalog.db`; both
/// backends keep result-table Parquet under `<dir>/jammi_db`.
async fn open_backend(kind: BackendKind, dir: &Path) -> Option<BackendImpl> {
    match kind {
        BackendKind::Sqlite => {
            let backend = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
            Some(BackendImpl::Sqlite(backend))
        }
        BackendKind::Postgres => {
            let url = pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .expect("open postgres backend");
            Some(BackendImpl::Postgres(pg))
        }
    }
}

/// Clear `result_tables` so a cross-tenant `recover()` scan sees only the rows
/// this test creates. The SQLite lane has a fresh tempdir per test, but running
/// the reset on both lanes keeps one code path; on the shared Postgres DB it is
/// load-bearing.
async fn reset_result_tables(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move { tx.execute("DELETE FROM result_tables", &[]).await })
        })
        .await
        .unwrap();
}

/// Migrate `backend`, build an unscoped (GLOBAL) [`Catalog`] over it — the
/// shape a startup recovery session has — and clear `result_tables`. Catalog
/// handles derived via [`Catalog::pinned_to_tenant`] share the same backend.
async fn fresh_catalog(backend: BackendImpl) -> Arc<Catalog> {
    backend.migrate().await.unwrap();
    let catalog = Arc::new(Catalog::from_backend(backend));
    reset_result_tables(&catalog).await;
    catalog
}

/// A `ResultStore` rooted at `dir` over `catalog`. Every store built on the same
/// `dir` shares the `<dir>/jammi_db` Parquet root, so an unscoped recovery store
/// reaches every tenant's bytes.
fn result_store(dir: &Path, catalog: Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, catalog, AnnIndexConfig::default()).unwrap()
}

/// Register a `building` embedding result table and return its generated paths.
async fn create_building_embedding(store: &ResultStore) -> ResultTableInfo {
    store
        .create_table(
            "src1",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            None,
        )
        .await
        .unwrap()
}

/// A valid, closed embedding Parquet with `n` rows written to the table's URL —
/// the bytes a writer leaves on disk *before* the catalog row is flipped to
/// `ready`. The catalog row stays `building`: this is the
/// "valid-but-never-finalized" torn state.
async fn write_closed_embedding_parquet(store: &ResultStore, info: &ResultTableInfo, n: usize) {
    let schema = embedding_table_schema(DIMS);
    let row_ids: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    let row_id_arr = StringArray::from_iter_values(row_ids.iter().map(|s| s.as_str()));
    let source_arr = StringArray::from_iter_values((0..n).map(|_| "src1"));
    let model_arr = StringArray::from_iter_values((0..n).map(|_| "test-model"));
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..DIMS).map(move |d| (i * DIMS + d) as f32))
        .collect();
    let item = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_id_arr),
            Arc::new(source_arr),
            Arc::new(model_arr),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    let mut writer = store.open_writer(&info.parquet_url, schema).await.unwrap();
    if n > 0 {
        writer.write_batch(&batch).await.unwrap();
    }
    let written = writer.close().await.unwrap();
    assert_eq!(written, n, "writer wrote the rows we asked for");
}

/// Overwrite the table's Parquet object with `bytes` that are *not* a valid
/// closed Parquet (a torn write — header without footer, or garbage). Recovery
/// must classify this as corrupt, reap the bytes, and mark the row `failed`.
async fn write_torn_parquet(store: &ResultStore, info: &ResultTableInfo, bytes: Bytes) {
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    let path = handle.data_path().unwrap();
    handle.put_bytes(&path, bytes).await.unwrap();
}

/// Fetch the catalog record for `name`, asserting it exists.
async fn record(catalog: &Catalog, name: &str) -> ResultTableRecord {
    catalog
        .get_result_table(name)
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("table {name} should still exist"))
}

/// True if `name` is registered (queryable) in `ctx`.
fn is_registered(ctx: &SessionContext, name: &str) -> bool {
    let table_ref = datafusion::sql::TableReference::bare(format!("jammi.{name}"));
    ctx.table_exist(table_ref).unwrap()
}

/// `.parquet` objects physically present under the result root.
fn parquet_files_on_disk(dir: &Path) -> Vec<std::path::PathBuf> {
    let root = dir.join("jammi_db");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("parquet") {
                out.push(p);
            }
        }
    }
    out
}

// =====================================================================
//  (a) building row, NO bytes → recover marks Failed (I1, I2, I3, I4)
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn building_with_missing_bytes_fails(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    // Torn state: a building row, but the writer crashed before any bytes
    // landed. We deliberately write NO Parquet.
    let info = create_building_embedding(&store).await;
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    assert!(
        !handle.exists(&handle.data_path().unwrap()).await.unwrap(),
        "precondition: no bytes were written"
    );

    store.recover().await.unwrap();

    // I2: terminal — no building rows remain.
    assert!(catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap()
        .is_empty());
    // The missing-bytes arm fails the row.
    let rec = record(&catalog, &info.table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Failed.to_string());

    // I1: a failed table is never registered/queryable.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();
    assert!(
        !is_registered(&ctx, &info.table_name),
        "I1: failed table not queryable"
    );
    // I4: no orphan bytes (none were ever written).
    assert!(
        parquet_files_on_disk(dir.path()).is_empty(),
        "I4: no orphan"
    );
}

// =====================================================================
//  (b) building row + TORN parquet → Failed + bytes reaped (I2, I4)
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn building_with_torn_parquet_fails_and_reaps(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    // Construct a torn Parquet: start from a valid closed file, then truncate
    // it so the footer is gone — exactly what a crash mid-flush leaves. The
    // file exists and has a plausible header but is NOT a valid closed Parquet.
    write_closed_embedding_parquet(&store, &info, 5).await;
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    let path = handle.data_path().unwrap();
    let full = handle.get_bytes(&path).await.unwrap();
    let truncated = full.slice(0..full.len() / 2);
    write_torn_parquet(&store, &info, truncated).await;

    store.recover().await.unwrap();

    // I2: terminal, and the torn arm fails it.
    let rec = record(&catalog, &info.table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Failed.to_string());
    // I4: the corrupt bytes were reaped.
    assert!(
        !handle.exists(&path).await.unwrap(),
        "I4: torn Parquet bytes deleted"
    );
    assert!(
        parquet_files_on_disk(dir.path()).is_empty(),
        "I4: no orphan Parquet left on disk"
    );

    // I1: not queryable.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();
    assert!(!is_registered(&ctx, &info.table_name));
}

// =====================================================================
//  (c)/(d) building row + valid closed parquet, never finalized →
//  Ready with TRUE row_count, sidecar rebuilt (I2, I5, I6). This is also
//  the finalize-ordering window (d): bytes durable, row still `building`
//  because the crash landed between the write and the status flip.
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn building_with_valid_parquet_promotes_with_true_count(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    // Torn state: a fully-valid closed Parquet with 7 rows and its
    // materialization manifest, but the catalog row is still `building` (crash
    // between the manifest write and the status flip). Because the manifest
    // landed, recovery promotes it with the footer's true count.
    write_closed_embedding_parquet(&store, &info, 7).await;
    jammi_test_utils::write_manifest_sidecar_for(&store, &info.parquet_url, "src1", DIMS).await;
    if let Some(ref idx_url) = info.index_url {
        let idx_handle = store.open_index(idx_url).unwrap();
        // Prove the sidecar is genuinely absent before recovery.
        let sidecar = idx_handle.sibling_path("usearch").unwrap();
        assert!(!idx_handle.exists(&sidecar).await.unwrap());
    }

    store.recover().await.unwrap();

    // I2 + promotion: terminal Ready.
    let rec = record(&catalog, &info.table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    // I5: promoted row_count is the TRUE footer count, not the writer's intent.
    assert_eq!(rec.row_count, 7, "I5: row_count == actual Parquet rows");

    // I6: the sidecar self-heals — resolve_search_mode returns a working index
    // rebuilt from the Parquet even though no sidecar was on disk pre-recovery.
    let index = store
        .resolve_search_mode(&rec)
        .await
        .unwrap()
        .expect("I6: sidecar rebuilt from Parquet");
    let hits = index.search(&[0.0, 1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(hits.len(), 1, "I6: rebuilt index is queryable");

    // I1/I3: a Ready table whose bytes exist IS registered.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();
    assert!(
        is_registered(&ctx, &info.table_name),
        "Ready table is queryable"
    );
    // I4: the one Parquet on disk is the one the Ready row points at.
    let on_disk = parquet_files_on_disk(dir.path());
    assert_eq!(on_disk.len(), 1, "I4: exactly the Ready table's Parquet");
}

// =====================================================================
//  (b-promote) building row + valid closed parquet holding fewer rows
//  than a full run → promoted with the FOOTER truth (the valid-but-
//  partial arm of (b): a closed Parquet, just shorter).
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn partial_but_valid_parquet_promotes_with_footer_count(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    // A closed Parquet that holds only 2 rows (a flush landed mid-run, then the
    // file was closed cleanly before the crash), plus its manifest. Recovery
    // must trust the footer (and the manifest's presence) to promote.
    write_closed_embedding_parquet(&store, &info, 2).await;
    jammi_test_utils::write_manifest_sidecar_for(&store, &info.parquet_url, "src1", DIMS).await;

    store.recover().await.unwrap();

    let rec = record(&catalog, &info.table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec.row_count, 2, "I5: footer is the source of truth");
}

// =====================================================================
//  (e) Ready row whose bytes are missing → NOT loaded (I1, I3)
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn ready_with_missing_bytes_not_loaded(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    // Torn state: the catalog committed `ready` (e.g. a power loss reordered
    // the bytes-fsync after the row commit), but the bytes are absent. The row
    // is Ready, no Parquet exists.
    let info = create_building_embedding(&store).await;
    catalog
        .update_result_table_status(&info.table_name, ResultTableStatus::Ready, 9)
        .await
        .unwrap();
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    assert!(!handle.exists(&handle.data_path().unwrap()).await.unwrap());

    // load_existing_tables must skip a Ready row whose bytes are gone.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();

    // I1/I3: never registered, so never queryable.
    assert!(
        !is_registered(&ctx, &info.table_name),
        "I3/I1: Ready-but-missing-bytes not registered"
    );
}

// =====================================================================
//  (idempotence) recover() re-run after reconciliation is a no-op.
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn recover_is_idempotent(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 3).await;
    // A promotable torn state: the manifest sidecar landed before the crash, so
    // recovery promotes (a manifest-less valid Parquet would be reaped instead).
    jammi_test_utils::write_manifest_sidecar_for(&store, &info.parquet_url, "src1", DIMS).await;

    store.recover().await.unwrap();
    let after_first = record(&catalog, &info.table_name).await;
    assert_eq!(after_first.status, ResultTableStatus::Ready.to_string());
    assert_eq!(after_first.row_count, 3);

    // Re-running over an already-reconciled catalog touches nothing.
    store.recover().await.unwrap();
    let after_second = record(&catalog, &info.table_name).await;
    assert_eq!(after_second.status, ResultTableStatus::Ready.to_string());
    assert_eq!(after_second.row_count, 3);
}

// =====================================================================
//  Tenant-scoped recovery: A and B each leave a torn building table.
//  recover() (a cross-tenant admin scan) must reconcile BOTH, keep each
//  row's tenant_id, and leak nothing across tenants.
// =====================================================================

#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn recover_reconciles_every_tenant(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    // Unscoped (GLOBAL) catalog — the shape a startup recovery session has.
    let global = fresh_catalog(backend).await;

    // Tenant-pinned catalog handles share the same backend + Parquet root.
    let cat_a = Arc::new(global.pinned_to_tenant(Some(tenant_a())));
    let cat_b = Arc::new(global.pinned_to_tenant(Some(tenant_b())));
    let store_a = result_store(dir.path(), Arc::clone(&cat_a));
    let store_b = result_store(dir.path(), Arc::clone(&cat_b));

    // Each tenant leaves a torn building table with a valid closed Parquet that
    // was never finalized — both should promote to Ready under recovery.
    let info_a = create_building_embedding(&store_a).await;
    write_closed_embedding_parquet(&store_a, &info_a, 4).await;
    jammi_test_utils::write_manifest_sidecar_for(&store_a, &info_a.parquet_url, "src1", DIMS).await;
    let info_b = create_building_embedding(&store_b).await;
    write_closed_embedding_parquet(&store_b, &info_b, 6).await;
    jammi_test_utils::write_manifest_sidecar_for(&store_b, &info_b.parquet_url, "src1", DIMS).await;

    // Each tenant sees ONLY its own building table before recovery (proves the
    // rows are genuinely tenant-bound, not GLOBAL).
    assert_eq!(
        cat_a
            .list_result_tables_by_status(ResultTableStatus::Building)
            .await
            .unwrap()
            .len(),
        1,
        "tenant A sees only its own building row"
    );
    assert_eq!(
        cat_b
            .list_result_tables_by_status(ResultTableStatus::Building)
            .await
            .unwrap()
            .len(),
        1
    );
    // The unscoped/GLOBAL session sees NEITHER tenant's row outside admin scope
    // — this is exactly why a naive tenant-scoped recover() would skip them.
    assert!(
        global
            .list_result_tables_by_status(ResultTableStatus::Building)
            .await
            .unwrap()
            .is_empty(),
        "unscoped session does not see tenant-owned building rows outside admin scope"
    );

    // Recovery runs from the unscoped store; internally it enters an admin
    // scope and reconciles BOTH tenants' orphans.
    let recovery_store = result_store(dir.path(), Arc::clone(&global));
    recovery_store.recover().await.unwrap();

    // Both tenants' tables are now terminal Ready with their true counts, and
    // each kept its own tenant_id.
    let rec_a = record(&cat_a, &info_a.table_name).await;
    assert_eq!(rec_a.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec_a.row_count, 4);
    assert_eq!(
        rec_a.tenant_id.as_deref(),
        Some(tenant_a().to_string().as_str())
    );

    let rec_b = record(&cat_b, &info_b.table_name).await;
    assert_eq!(rec_b.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec_b.row_count, 6);
    assert_eq!(
        rec_b.tenant_id.as_deref(),
        Some(tenant_b().to_string().as_str())
    );

    // I2 across tenants: no building row remains for either tenant.
    assert!(cat_a
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap()
        .is_empty());
    assert!(cat_b
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap()
        .is_empty());

    // Cross-tenant visibility: A cannot see B's promoted table and vice versa.
    assert!(
        cat_a
            .get_result_table(&info_b.table_name)
            .await
            .unwrap()
            .is_none(),
        "tenant A must not see tenant B's table"
    );
    assert!(
        cat_b
            .get_result_table(&info_a.table_name)
            .await
            .unwrap()
            .is_none(),
        "tenant B must not see tenant A's table"
    );
}
