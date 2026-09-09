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
//! state would be vacuous; non-vacuity is the bar. Every torn state is a DEAD
//! writer's: the fixture detaches the writer's [`BuildingTable`] handle and
//! forces its lease into the past ([`jammi_test_utils::abandon_building`],
//! which first asserts the row was `building` under a live lease), so each
//! test observes the transition recovery performs, never just an end state a
//! live-lease skip could leave vacuously.
//!
//! The lease-ownership half (esc-094, issue #479): a `building` row is owned by
//! its writer under a heartbeated lease, recovery touches only rows whose lease
//! is absent or expired, and every transition on a building row is a
//! compare-and-set naming the owner. The two-writer oracles
//! (`live_writer_survives_peer_recover_*`, feature `test-hooks`) park a live
//! writer at a named point, run a peer session's `recover()` beside it, and
//! prove the writer's row, bytes, and segments survive and the table completes
//! with the true row count.
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
use jammi_db::catalog::backend::{BackendImpl, BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{
    Owner, ResultTableCas, ResultTableKind, ResultTableRecord, TenantArm,
};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::{AnnIndexConfig, LeaseConfig};
use jammi_db::error::JammiError;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{
    ComputeDevice, ContextAggregator, ContextCandidateSource, InputAnchor, Materialization,
    MaterializationEnv, ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{BuildingTable, ResultStore};
#[cfg(feature = "test-hooks")]
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use jammi_test_utils::{abandon_building, pg_url_for_tests};
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

/// Require-gate (KO-7) for the `JAMMI_TEST_PG_URL`-unset skip every
/// `open_backend(BackendKind::Postgres, ..)` call site in this file falls
/// through to: by default (unset) the Postgres arm still silently skips,
/// exactly as before — a lane that wants to REQUIRE the real Postgres arm
/// run (never silently skip it) sets `JAMMI_REQUIRE_PG`, and this call
/// panics instead.
fn require_live_pg(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_PG").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_PG is set but JAMMI_TEST_PG_URL is unset -- this lane \
             must run the real Postgres arm, not skip it"
        );
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

/// Register a `building` embedding result table and return the writer's
/// lease-owned handle.
async fn create_building_embedding(store: &ResultStore) -> BuildingTable {
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

/// A store whose building tables are held under a lease long enough that no
/// heartbeat fires during a test — for the tests that forge lease state by
/// hand while the writer's handle is still alive.
fn long_lease_store(dir: &Path, catalog: Arc<Catalog>) -> ResultStore {
    result_store(dir, catalog).with_lease_intervals(
        LeaseConfig {
            duration_secs: 600,
            heartbeat_secs: 200,
        }
        .intervals()
        .unwrap(),
    )
}

/// The catalog record for `name`, read across every tenant.
#[cfg(feature = "test-hooks")]
async fn record_admin(catalog: &Catalog, name: &str) -> ResultTableRecord {
    TenantBinding::admin_scope(catalog.get_result_table(name))
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("table {name} should still exist"))
}

/// Force a row's lease into the past by hand (the writer's handle stays alive
/// — its heartbeat, if it fired, would renew; the callers use a
/// [`long_lease_store`] so it never does within the test).
async fn expire_lease_by_hand(catalog: &Catalog, name: &str) {
    let name = name.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE result_tables SET lease_expires_at = '1970-01-01T00:00:00.000000Z' \
                     WHERE table_name = $1",
                    &[SqlValue::TextOwned(name)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Delete a row by hand — the `RowGone` fixture.
async fn delete_row_by_hand(catalog: &Catalog, name: &str) {
    let name = name.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM result_tables WHERE table_name = $1",
                    &[SqlValue::TextOwned(name)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// The producing descriptor + environment a writer attests its table with.
fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::ContextSet {
        encoder_id: "synthetic-embed".into(),
        source_id: "src1".into(),
        embedding_table: None,
        candidate_source: ContextCandidateSource::Ann { k: 5 },
        value_columns: Vec::new(),
        aggregator: ContextAggregator::Mean,
        exclude_self: true,
        split: None,
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(ComputeDevice::Cpu, Vec::new())
}

fn inputs() -> Vec<InputAnchor> {
    vec![InputAnchor::unpinned_at_instant(
        "src1",
        "1970-01-01T00:00:00Z",
    )]
}

/// A built one-segment index over the same rows [`write_closed_embedding_parquet`]
/// writes, at the store's default precision.
fn built_index(n: usize) -> SidecarIndex {
    let mut idx = SidecarIndex::new(
        DIMS,
        &AnnIndexConfig::default(),
        AnnIndexConfig::default().storage_precision,
    )
    .unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..DIMS).map(|d| (i * DIMS + d) as f32).collect();
        idx.add(&format!("row-{i}"), &v).unwrap();
    }
    idx.build().unwrap();
    idx
}

/// `SELECT count(*)` over the registered table in `ctx`.
#[cfg(feature = "test-hooks")]
async fn select_count(ctx: &SessionContext, name: &str) -> usize {
    let batches = ctx
        .sql(&format!("SELECT count(*) AS n FROM \"jammi.{name}\""))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let col = batches[0]
        .column_by_name("n")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    col.value(0) as usize
}

/// Whether the Parquet object behind `url` exists — an `Err` from the probe
/// FAILS the test rather than reading as "absent".
async fn parquet_exists(store: &ResultStore, url: &StorageUrl) -> bool {
    let handle = store.open_parquet(url).unwrap();
    handle
        .exists(&handle.data_path().unwrap())
        .await
        .expect("exists() probe must not error")
}

/// A valid, closed embedding Parquet with `n` rows written to the table's URL —
/// the bytes a writer leaves on disk *before* the catalog row is flipped to
/// `ready`. The catalog row stays `building`: this is the
/// "valid-but-never-finalized" torn state.
async fn write_closed_embedding_parquet(store: &ResultStore, info: &BuildingTable, n: usize) {
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

    let mut writer = store.open_writer(info.parquet_url(), schema).await.unwrap();
    if n > 0 {
        writer.write_batch(&batch).await.unwrap();
    }
    let written = writer.close().await.unwrap();
    assert_eq!(written, n, "writer wrote the rows we asked for");
}

/// Overwrite the table's Parquet object with `bytes` that are *not* a valid
/// closed Parquet (a torn write — header without footer, or garbage). Recovery
/// must classify this as corrupt, reap the bytes, and mark the row `failed`.
async fn write_torn_parquet(store: &ResultStore, url: &StorageUrl, bytes: Bytes) {
    let handle = store.open_parquet(url).unwrap();
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
    // Tables now land one directory deeper, under their tenant segment
    // (`{root}/_global/{table}.parquet` or `{root}/{tenant}/{table}.parquet`)
    // — walk two levels so this helper still finds every table regardless
    // of which segment it landed under.
    let root = dir.join("jammi_db");
    let mut out = Vec::new();
    walk_parquet(&root, &mut out);
    out
}

fn walk_parquet(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_parquet(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("parquet") {
            out.push(p);
        }
    }
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
        require_live_pg("building_with_missing_bytes_fails");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    // Torn state: a building row, but the writer crashed before any bytes
    // landed. We deliberately write NO Parquet.
    let info = create_building_embedding(&store).await;
    let url = info.parquet_url().clone();
    assert!(
        !parquet_exists(&store, &url).await,
        "precondition: no bytes were written"
    );
    let table_name = abandon_building(&catalog, info).await;

    store.recover().await.unwrap();

    // I2: terminal — no building rows remain.
    assert!(catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap()
        .is_empty());
    // The missing-bytes arm fails the row.
    let rec = record(&catalog, &table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Failed.to_string());
    assert!(
        rec.lease_expires_at.is_none(),
        "a terminal row carries no lease"
    );

    // I1: a failed table is never registered/queryable.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();
    assert!(
        !is_registered(&ctx, &table_name),
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
        require_live_pg("building_with_torn_parquet_fails_and_reaps");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    // Construct a torn Parquet: start from a valid closed file, then truncate
    // it so the footer is gone — exactly what a crash mid-flush leaves. The
    // file exists and has a plausible header but is NOT a valid closed Parquet.
    write_closed_embedding_parquet(&store, &info, 5).await;
    let url = info.parquet_url().clone();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    let full = handle.get_bytes(&path).await.unwrap();
    let truncated = full.slice(0..full.len() / 2);
    write_torn_parquet(&store, &url, truncated).await;
    let table_name = abandon_building(&catalog, info).await;

    store.recover().await.unwrap();

    // I2: terminal, and the torn arm fails it.
    let rec = record(&catalog, &table_name).await;
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
    assert!(!is_registered(&ctx, &table_name));
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
        require_live_pg("building_with_valid_parquet_promotes_with_true_count");
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
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    // Prove the ANN index is genuinely absent before recovery: the building
    // table has no segments yet (none were ever appended).
    assert!(
        store
            .catalog()
            .list_index_segments(info.table_name())
            .await
            .unwrap()
            .is_empty(),
        "no index segments exist before recovery"
    );
    let dead_writer = info.writer_id().to_string();
    let table_name = abandon_building(&catalog, info).await;

    // A restart is a different writer: recover from a peer store.
    let peer = result_store(dir.path(), global_sibling(&catalog));
    peer.recover().await.unwrap();

    // I2 + promotion: terminal Ready.
    let rec = record(&catalog, &table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    // I5: promoted row_count is the TRUE footer count, not the writer's intent.
    assert_eq!(rec.row_count, 7, "I5: row_count == actual Parquet rows");
    // Recovery claimed the row before rebuilding and promoting: the recoverer
    // is the writer of record, the dead writer's id is history.
    assert_eq!(rec.writer_id.as_deref(), Some(peer.writer_id()));
    assert_ne!(rec.writer_id.as_deref(), Some(dead_writer.as_str()));
    assert!(rec.lease_expires_at.is_none(), "promote clears the lease");

    // I6: the sidecar self-heals — resolve_search_mode returns a working index
    // rebuilt from the Parquet even though no sidecar was on disk pre-recovery.
    // Recovery rebuilds the whole table as a single fresh segment (segment 0).
    let segs = store
        .catalog()
        .list_index_segments(&table_name)
        .await
        .unwrap();
    assert_eq!(
        segs.len(),
        1,
        "I6: recovery rebuilds the index as one segment"
    );
    assert_eq!(segs[0].segment_id, 0);
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
    assert!(is_registered(&ctx, &table_name), "Ready table is queryable");
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
        require_live_pg("partial_but_valid_parquet_promotes_with_footer_count");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    // A closed Parquet that holds only 2 rows (a flush landed mid-run, then the
    // file was closed cleanly before the crash), plus its manifest. Recovery
    // must trust the footer (and the manifest's presence) to promote.
    write_closed_embedding_parquet(&store, &info, 2).await;
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    let table_name = abandon_building(&catalog, info).await;

    store.recover().await.unwrap();

    let rec = record(&catalog, &table_name).await;
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
        require_live_pg("ready_with_missing_bytes_not_loaded");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    // Torn state: the catalog committed `ready` (e.g. a power loss reordered
    // the bytes-fsync after the row commit), but the bytes are absent. The row
    // is Ready, no Parquet exists.
    let info = create_building_embedding(&store).await;
    catalog
        .update_result_table_status(info.table_name(), ResultTableStatus::Ready, 9)
        .await
        .unwrap();
    assert!(!parquet_exists(&store, info.parquet_url()).await);

    // load_existing_tables must skip a Ready row whose bytes are gone.
    let ctx = SessionContext::new();
    store.load_existing_tables(&ctx).await.unwrap();

    // I1/I3: never registered, so never queryable.
    assert!(
        !is_registered(&ctx, info.table_name()),
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
        require_live_pg("recover_is_idempotent");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 3).await;
    // A promotable torn state: the manifest sidecar landed before the crash, so
    // recovery promotes (a manifest-less valid Parquet would be reaped instead).
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    let table_name = abandon_building(&catalog, info).await;

    store.recover().await.unwrap();
    let after_first = record(&catalog, &table_name).await;
    assert_eq!(after_first.status, ResultTableStatus::Ready.to_string());
    assert_eq!(after_first.row_count, 3);

    // Re-running over an already-reconciled catalog touches nothing.
    store.recover().await.unwrap();
    let after_second = record(&catalog, &table_name).await;
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
        require_live_pg("recover_reconciles_every_tenant");
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
    jammi_test_utils::write_manifest_sidecar_for(&store_a, info_a.parquet_url(), "src1", DIMS)
        .await;
    let info_b = create_building_embedding(&store_b).await;
    write_closed_embedding_parquet(&store_b, &info_b, 6).await;
    jammi_test_utils::write_manifest_sidecar_for(&store_b, info_b.parquet_url(), "src1", DIMS)
        .await;

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

    // Both writers are dead and their leases have run out.
    let name_a = abandon_building(&cat_a, info_a).await;
    let name_b = abandon_building(&cat_b, info_b).await;

    // Recovery runs from the unscoped store; internally it enters an admin
    // scope and reconciles BOTH tenants' orphans — the one named
    // implicit-admin pass reaps/promotes a tenant-owned expired-lease row.
    let recovery_store = result_store(dir.path(), Arc::clone(&global));
    recovery_store.recover().await.unwrap();

    // Both tenants' tables are now terminal Ready with their true counts, and
    // each kept its own tenant_id.
    let rec_a = record(&cat_a, &name_a).await;
    assert_eq!(rec_a.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec_a.row_count, 4);
    assert_eq!(
        rec_a.tenant_id.as_deref(),
        Some(tenant_a().to_string().as_str())
    );

    let rec_b = record(&cat_b, &name_b).await;
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
        cat_a.get_result_table(&name_b).await.unwrap().is_none(),
        "tenant A must not see tenant B's table"
    );
    assert!(
        cat_b.get_result_table(&name_a).await.unwrap().is_none(),
        "tenant B must not see tenant A's table"
    );
}

// =====================================================================
//  esc-094 — lease ownership. A `building` row belongs to its writer under
//  a heartbeated lease; recovery touches only rows whose lease is absent or
//  expired; every building-row transition is a CAS naming the owner.
// =====================================================================

/// Control: with the writer dead and its lease expired, recovery reaps a
/// manifest-less valid Parquet exactly as it always did — `failed`, bytes and
/// segments gone — and it does so only AFTER its own one-row CAS (the
/// transition is observed: `building` + live lease before, `failed` after).
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn expired_lease_building_row_is_reaped(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("expired_lease_building_row_is_reaped");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 4).await;
    info.append_segment(&built_index(4)).await.unwrap();
    let url = info.parquet_url().clone();
    let table_name = abandon_building(&catalog, info).await;
    assert!(
        parquet_exists(&store, &url).await,
        "precondition: bytes present"
    );
    assert_eq!(
        catalog
            .list_index_segments(&table_name)
            .await
            .unwrap()
            .len(),
        1,
        "precondition: one segment registered"
    );

    // A second session over the same catalog runs the sweep.
    let peer = result_store(dir.path(), global_sibling(&catalog));
    peer.recover().await.unwrap();

    let rec = record(&catalog, &table_name).await;
    assert_eq!(rec.status, ResultTableStatus::Failed.to_string());
    assert!(rec.lease_expires_at.is_none());
    assert!(
        !parquet_exists(&store, &url).await,
        "bytes reaped after the fail CAS"
    );
    assert!(
        catalog
            .list_index_segments(&table_name)
            .await
            .unwrap()
            .is_empty(),
        "segments purged after the fail CAS"
    );
}

/// A sibling unscoped catalog handle over the same backend — "a second
/// session over the SAME catalog URL".
fn global_sibling(catalog: &Catalog) -> Arc<Catalog> {
    Arc::new(catalog.pinned_to_tenant(None))
}

/// W2 (esc-094): a live writer parked between its lease renew and the manifest
/// write — Parquet valid, no manifest, row `building` — survives a peer
/// session's `recover()`: status, bytes, and segments untouched; released, it
/// completes with the true row count. ONE arm is cross-tenant: the writer is
/// bound to tenant A, the peer to tenant B, and the peer's sweep (the one
/// implicit-admin pass) still leaves A's live row alone.
#[cfg(feature = "test-hooks")]
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_writer_survives_peer_recover_w2(kind: BackendKind) {
    use jammi_db::store::mutable::test_hook::{arm, MaterializationPoint};

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("live_writer_survives_peer_recover_w2");
        return;
    };
    let global = fresh_catalog(backend).await;
    let cat_a = Arc::new(global.pinned_to_tenant(Some(tenant_a())));
    let cat_b = Arc::new(global.pinned_to_tenant(Some(tenant_b())));
    let store_a = result_store(dir.path(), Arc::clone(&cat_a));
    let store_b = result_store(dir.path(), Arc::clone(&cat_b));
    const N: usize = 6;

    let armed = arm(MaterializationPoint::Materialization, store_a.writer_id());
    let ctx_a = SessionContext::new();
    let writer = {
        let store_a = store_a.clone();
        let ctx_a = ctx_a.clone();
        tokio::spawn(async move {
            let building = create_building_embedding(&store_a).await;
            write_closed_embedding_parquet(&store_a, &building, N).await;
            building.append_segment(&built_index(N)).await.unwrap();
            let (descriptor, env) = (descriptor(), env());
            building
                .finish(&ctx_a, N, Materialization::new(&descriptor, &env, inputs()))
                .await
        })
    };
    armed
        .wait_parked()
        .await
        .expect("writer A must reach the materialization point");

    // Positive preconditions: A is exactly in the W2 window.
    let building_rows =
        TenantBinding::admin_scope(cat_a.list_result_tables_by_status(ResultTableStatus::Building))
            .await
            .unwrap();
    assert_eq!(
        building_rows.len(),
        1,
        "precondition: A's row is `building`"
    );
    let row = &building_rows[0];
    let table_name = row.table_name.clone();
    let url = StorageUrl::parse(&row.parquet_path).unwrap();
    let handle = store_a.open_parquet(&url).unwrap();
    assert!(
        parquet_exists(&store_a, &url).await,
        "precondition: Parquet present"
    );
    assert!(
        jammi_db::storage::reader::is_valid_parquet(&handle)
            .await
            .unwrap(),
        "precondition: Parquet valid"
    );
    assert!(
        store_a
            .read_materialization_manifest(&url)
            .await
            .unwrap()
            .is_none(),
        "precondition: manifest absent"
    );
    // Liveness is asserted through the catalog's own backend-correct
    // predicate, not a Rust-side string compare against `row.lease_expires_at`
    // — that stored value is a Postgres-clock expression's text rendering on
    // Postgres (`(now() + make_interval(...))::text`), not the
    // `lease_now()`-shaped string a naive `>` compare here would assume.
    assert!(
        cat_a
            .list_live_building_tables()
            .await
            .unwrap()
            .iter()
            .any(|t| t.table_name == table_name),
        "precondition: A's lease is live"
    );
    assert_eq!(
        cat_a.list_index_segments(&table_name).await.unwrap().len(),
        1
    );

    // B (a different tenant) boots and sweeps.
    store_b.recover().await.unwrap();

    let after = record_admin(&global, &table_name).await;
    assert_eq!(
        after.status,
        ResultTableStatus::Building.to_string(),
        "a live writer's row survives a peer's recover()"
    );
    assert!(parquet_exists(&store_a, &url).await, "bytes survive");
    assert_eq!(
        cat_a.list_index_segments(&table_name).await.unwrap().len(),
        1,
        "segments survive"
    );

    // Release A and let it complete.
    armed.release();
    let rec = writer.await.unwrap().expect("A's finish succeeds");
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec.row_count, N);
    assert_eq!(
        jammi_db::storage::reader::count_parquet_rows(&handle)
            .await
            .unwrap(),
        N
    );
    assert!(rec.definition_hash.is_some(), "the attestation landed");
    assert!(rec.lease_expires_at.is_none());
    assert_eq!(select_count(&ctx_a, &table_name).await, N);
}

/// U2 (block #1, phase-4 fix): the SAME two-writer shape as W2, but the peer
/// runs `reconcile(apply=true)` instead of the startup `recover()` sweep — a
/// live-lease `building` row's bytes must survive a reconcile pass exactly as
/// they survive recovery. `own_seg` scoping means the peer must reconcile
/// under A's OWN tenant (a cross-tenant `reconcile()` would not even list A's
/// prefix at all, which would pass vacuously); the cross-tenant half of this
/// contract is `reconcile_all`, covered by `tenant_isolation_oracle.rs`.
#[cfg(feature = "test-hooks")]
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_writer_survives_peer_reconcile_apply_u2(kind: BackendKind) {
    use jammi_db::store::mutable::test_hook::{arm, MaterializationPoint};
    use jammi_db::store::ReconcileOptions;

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("live_writer_survives_peer_reconcile_apply_u2");
        return;
    };
    let global = fresh_catalog(backend).await;
    let cat_a = Arc::new(global.pinned_to_tenant(Some(tenant_a())));
    let store_a = long_lease_store(dir.path(), Arc::clone(&cat_a));
    // A peer SESSION over the SAME tenant, the shape a second replica process
    // reconciling A's own prefix takes.
    let peer = long_lease_store(dir.path(), Arc::clone(&cat_a));
    const N: usize = 4;

    let armed = arm(MaterializationPoint::Materialization, store_a.writer_id());
    let ctx_a = SessionContext::new();
    let writer = {
        let store_a = store_a.clone();
        let ctx_a = ctx_a.clone();
        tokio::spawn(async move {
            let building = create_building_embedding(&store_a).await;
            write_closed_embedding_parquet(&store_a, &building, N).await;
            building.append_segment(&built_index(N)).await.unwrap();
            let (descriptor, env) = (descriptor(), env());
            building
                .finish(&ctx_a, N, Materialization::new(&descriptor, &env, inputs()))
                .await
        })
    };
    armed
        .wait_parked()
        .await
        .expect("writer A must reach the materialization point");

    let building_rows = cat_a
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(
        building_rows.len(),
        1,
        "precondition: A's row is `building`"
    );
    let table_name = building_rows[0].table_name.clone();
    let url = StorageUrl::parse(&building_rows[0].parquet_path).unwrap();

    // The peer reconciles A's own prefix with `apply=true` — the lease
    // duration `long_lease_store` uses (600s) sets the grace floor.
    let report = peer
        .reconcile(ReconcileOptions {
            apply: true,
            grace: std::time::Duration::from_secs(600),
        })
        .await
        .unwrap();
    assert!(
        report.orphans.is_empty(),
        "a live-lease building row's objects must never be reaped by reconcile: {report:?}"
    );
    assert!(
        report
            .orphans
            .iter()
            .chain(report.pending.iter())
            .all(|k| !k.contains(&table_name)),
        "no object of the live row may even be flagged: {report:?}"
    );
    assert!(parquet_exists(&store_a, &url).await, "bytes survive");
    assert_eq!(
        record_admin(&global, &table_name).await.status,
        ResultTableStatus::Building.to_string(),
        "a live writer's row survives a peer's reconcile(apply=true)"
    );

    armed.release();
    let rec = writer.await.unwrap().expect("A's finish succeeds");
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec.row_count, N);
}

/// U2b (block #1, phase-4 fix, RED first): an EXPIRED-lease `building` row is
/// reconciled through the RECOVERY arm — claimed (its `writer_id` changes)
/// BEFORE its bytes go — never through the orphan arm with no claim/CAS at
/// all. Before this fix, `reconcile`'s orphan loop reaped such a row's
/// Parquet/sidecar objects directly (no claim, no CAS), so a stalled original
/// writer's later `renew`/`promote` would have raced a promote over deleted
/// bytes; this pins that the row is claimed first (fencing the stale writer)
/// and the stale writer's own subsequent CAS is refused.
#[cfg(feature = "test-hooks")]
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn expired_lease_building_row_is_claimed_before_reconcile_reaps_it_u2b(kind: BackendKind) {
    use jammi_db::store::ReconcileOptions;

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("expired_lease_building_row_is_claimed_before_reconcile_reaps_it_u2b");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = long_lease_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 3).await;
    info.append_segment(&built_index(3)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    let writer_id = info.writer_id().to_string();
    let url = info.parquet_url().clone();
    let name = abandon_building(&catalog, info).await;

    // A peer reconciles under the SAME tenant scope, `apply=true`, past the
    // configured (long) lease's grace floor.
    let peer = result_store(dir.path(), global_sibling(&catalog));
    let report = peer
        .reconcile(ReconcileOptions {
            apply: true,
            grace: std::time::Duration::from_secs(600),
        })
        .await
        .unwrap();
    assert!(
        report.orphans.iter().all(|k| !k.contains(&name)),
        "the expired row's bytes must be reconciled through the recovery arm, \
         never orphan-reaped directly: {report:?}"
    );

    // The row was CLAIMED (a new writer_id) then promoted — never left
    // `building`, and never simply deleted.
    let after = record_admin(&catalog, &name).await;
    assert_ne!(
        after.writer_id.as_deref(),
        Some(writer_id.as_str()),
        "the row must be re-stamped with a NEW writer_id (fencing the stale one) \
         before any byte touches it"
    );
    assert_eq!(
        after.status,
        ResultTableStatus::Ready.to_string(),
        "a valid Parquet with a landed manifest self-heals to ready via the claim arm"
    );
    assert!(
        parquet_exists(&store, &url).await,
        "bytes survive the claim"
    );

    // The stale original writer's own CAS is now refused, never re-promotes.
    let late = ResultTableCas::writer(&name, &writer_id, None);
    let err = catalog
        .promote_result_table_with_manifest(&late, 3, "deadbeef", "[]")
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::CasFailed { table, status } if *table == name && status == "ready")
            || matches!(&err, JammiError::LeaseLost { table } if *table == name),
        "the fenced writer's late CAS must be refused, never re-promote: {err:?}"
    );
}

/// W1 (esc-094): a live writer parked right after `create_table` — row
/// `building`, no bytes yet — survives a peer's `recover()` (which would
/// otherwise take the missing-bytes arm and fail it); released, it completes.
#[cfg(feature = "test-hooks")]
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_writer_survives_peer_recover_w1(kind: BackendKind) {
    use jammi_db::store::mutable::test_hook::{arm, MaterializationPoint};

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("live_writer_survives_peer_recover_w1");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store_a = result_store(dir.path(), Arc::clone(&catalog));
    let store_b = result_store(dir.path(), global_sibling(&catalog));
    const N: usize = 3;

    let armed = arm(MaterializationPoint::TableCreated, store_a.writer_id());
    let ctx = SessionContext::new();
    let writer = {
        let store_a = store_a.clone();
        let ctx = ctx.clone();
        tokio::spawn(async move {
            let building = create_building_embedding(&store_a).await;
            write_closed_embedding_parquet(&store_a, &building, N).await;
            let (descriptor, env) = (descriptor(), env());
            building
                .finish(&ctx, N, Materialization::new(&descriptor, &env, inputs()))
                .await
        })
    };
    armed
        .wait_parked()
        .await
        .expect("writer A must reach the table_created point");

    let rows = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "precondition: A's row is `building`");
    let table_name = rows[0].table_name.clone();
    let url = StorageUrl::parse(&rows[0].parquet_path).unwrap();
    assert!(
        !parquet_exists(&store_a, &url).await,
        "precondition: no bytes yet"
    );
    assert!(rows[0].lease_expires_at.is_some(), "precondition: leased");

    store_b.recover().await.unwrap();

    assert_eq!(
        record(&catalog, &table_name).await.status,
        ResultTableStatus::Building.to_string(),
        "a live writer's byte-less row survives a peer's recover()"
    );

    armed.release();
    let rec = writer.await.unwrap().expect("A's finish succeeds");
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec.row_count, N);
    assert_eq!(select_count(&ctx, &table_name).await, N);
}

// ---- the zero-row outcomes: exactly one typed error each, none deletes ----

/// `RowGone`: the row was deleted underneath the writer. `abort` reports it
/// and deletes nothing.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn zero_rows_row_gone_deletes_nothing(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("zero_rows_row_gone_deletes_nothing");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = long_lease_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 2).await;
    let url = info.parquet_url().clone();
    let name = info.table_name().to_string();
    delete_row_by_hand(&catalog, &name).await;

    let err = info.abort().await.unwrap_err();
    assert!(
        matches!(&err, JammiError::RowGone { table } if *table == name),
        "got {err:?}"
    );
    assert!(parquet_exists(&store, &url).await, "RowGone never deletes");
}

/// `TenantMismatch` / STRICT promote: a GLOBAL row is not promoted by a
/// tenant-scoped writer's CAS (the leaky `OR tenant_id IS NULL` arm is gone);
/// the row stays `building` and nothing is deleted. The row's own (unscoped)
/// writer then promotes it.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn strict_tenant_predicate_on_promote(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("strict_tenant_predicate_on_promote");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = long_lease_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    let rows = 3;
    write_closed_embedding_parquet(&store, &info, rows).await;
    let (descriptor, env) = (descriptor(), env());
    let (manifest, anchors) = store
        .write_attestation(
            info.parquet_url(),
            Materialization::new(&descriptor, &env, inputs()),
        )
        .await
        .unwrap();
    let name = info.table_name().to_string();

    // The same writer id, but a tenant-B-scoped binding: STRICT refuses.
    let scoped = ResultTableCas {
        table: name.clone(),
        tenant_arm: TenantArm::Strict(Some(tenant_b())),
        owner: Owner::Writer(info.writer_id().to_string()),
    };
    let err = catalog
        .promote_result_table_with_manifest(
            &scoped,
            rows,
            manifest.definition_hash.as_str(),
            &anchors,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::TenantMismatch { table } if *table == name),
        "got {err:?}"
    );
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Building.to_string()
    );
    assert!(parquet_exists(&store, info.parquet_url()).await);

    // The GLOBAL writer's own CAS promotes.
    let owner = catalog
        .promote_result_table_with_manifest(
            &info.cas(),
            rows,
            manifest.definition_hash.as_str(),
            &anchors,
        )
        .await
        .unwrap();
    assert_eq!(owner, None, "a GLOBAL row's owner is None");
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Ready.to_string()
    );
}

/// `CasFailed{failed}`: recovery reaped the (dead-writer) row; the writer's
/// late promote is refused with the row's status and deletes nothing more.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn zero_rows_reaped_row_is_cas_failed(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("zero_rows_reaped_row_is_cas_failed");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 2).await;
    let writer_id = info.writer_id().to_string();
    let name = abandon_building(&catalog, info).await;
    result_store(dir.path(), global_sibling(&catalog))
        .recover()
        .await
        .unwrap();
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Failed.to_string()
    );

    let late = ResultTableCas::writer(&name, &writer_id, None);
    let err = catalog
        .promote_result_table_with_manifest(&late, 2, "deadbeef", "[]")
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::CasFailed { table, status } if *table == name && status == "failed"),
        "got {err:?}"
    );
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Failed.to_string()
    );
}

/// `CasFailed{ready}`: recovery promoted the expired-lease row from its own
/// sidecar; the writer's late promote is refused, never re-promotes, and the
/// bytes it would have "cleaned" are the live table's — untouched.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn zero_rows_recovery_promoted_row_is_cas_failed_ready(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("zero_rows_recovery_promoted_row_is_cas_failed_ready");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 5).await;
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    let url = info.parquet_url().clone();
    let writer_id = info.writer_id().to_string();
    let name = abandon_building(&catalog, info).await;
    result_store(dir.path(), global_sibling(&catalog))
        .recover()
        .await
        .unwrap();
    let promoted = record(&catalog, &name).await;
    assert_eq!(promoted.status, ResultTableStatus::Ready.to_string());
    assert_eq!(promoted.row_count, 5);

    let late = ResultTableCas::writer(&name, &writer_id, None);
    let err = catalog
        .promote_result_table_with_manifest(&late, 5, "deadbeef", "[]")
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::CasFailed { table, status } if *table == name && status == "ready"),
        "got {err:?}"
    );
    let after = record(&catalog, &name).await;
    assert_eq!(
        after.definition_hash, promoted.definition_hash,
        "never re-promoted"
    );
    assert!(
        parquet_exists(&store, &url).await,
        "the live table's bytes are untouched"
    );
}

/// `LeaseLost`: recovery claimed the expired-lease row (the recoverer is now
/// the writer of record); the original writer's renew and abort both report
/// it and delete nothing — the claimant owns the bytes.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn zero_rows_claimed_row_is_lease_lost_and_deletes_nothing(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("zero_rows_claimed_row_is_lease_lost_and_deletes_nothing");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = long_lease_store(dir.path(), Arc::clone(&catalog));
    let recoverer = result_store(dir.path(), global_sibling(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 2).await;
    info.append_segment(&built_index(2)).await.unwrap();
    let url = info.parquet_url().clone();
    let name = info.table_name().to_string();
    // The writer stalled past its lease (forged by hand; its heartbeat is
    // 200 s away) and a recoverer claimed the row.
    expire_lease_by_hand(&catalog, &name).await;
    let claimed = catalog
        .claim_expired_building_table(
            &ResultTableCas::expired(&name, None),
            recoverer.writer_id(),
            recoverer.lease_intervals().lease(),
        )
        .await
        .unwrap();
    assert!(claimed, "the expired row is claimable");
    assert_eq!(
        record(&catalog, &name).await.writer_id.as_deref(),
        Some(recoverer.writer_id())
    );

    let err = catalog
        .renew_lease(&info.cas(), std::time::Duration::from_secs(30))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::LeaseLost { table } if *table == name),
        "got {err:?}"
    );
    let err = catalog
        .insert_index_segment(&info.cas(), 7, "file:///never", 1)
        .await
        .unwrap_err();
    assert!(matches!(&err, JammiError::LeaseLost { .. }), "got {err:?}");

    let err = info.abort().await.unwrap_err();
    assert!(
        matches!(&err, JammiError::LeaseLost { table } if *table == name),
        "got {err:?}"
    );
    assert!(
        parquet_exists(&store, &url).await,
        "LeaseLost never deletes the claimant's bytes"
    );
    assert_eq!(
        catalog.list_index_segments(&name).await.unwrap().len(),
        1,
        "the claimant's segments are untouched"
    );
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Building.to_string()
    );
}

/// `abort()` after the writer's own one-row CAS is the writer's only deletion
/// arm: `failed`, bytes + sidecar + segments gone.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn abort_deletes_only_after_its_own_cas(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("abort_deletes_only_after_its_own_cas");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 2).await;
    info.append_segment(&built_index(2)).await.unwrap();
    let url = info.parquet_url().clone();
    let name = info.table_name().to_string();
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Building.to_string()
    );

    info.abort().await.unwrap();

    let rec = record(&catalog, &name).await;
    assert_eq!(rec.status, ResultTableStatus::Failed.to_string());
    assert!(rec.lease_expires_at.is_none());
    assert!(!parquet_exists(&store, &url).await);
    assert!(catalog.list_index_segments(&name).await.unwrap().is_empty());
}

/// Dropping the handle without `finish`/`abort` (an error path unwinding
/// through `?`) marks the row `failed` by the writer's own CAS and deletes
/// NOTHING — the objects are reconcile's to reap later.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn drop_without_finish_marks_failed_or_expires(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("drop_without_finish_marks_failed_or_expires");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 2).await;
    let url = info.parquet_url().clone();
    let name = info.table_name().to_string();
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Building.to_string()
    );

    drop(info);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let rec = record(&catalog, &name).await;
        if rec.status == ResultTableStatus::Failed.to_string() {
            assert!(rec.lease_expires_at.is_none());
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "a dropped handle must mark its row failed under a runtime; got {}",
            rec.status
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(parquet_exists(&store, &url).await, "Drop deletes nothing");
}

/// `delete_result_tables_for_source` refuses while a live writer holds a
/// `building` row over the source (`SourceBusy`), and deletes — expired
/// lease rows included — once the writer is dead.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn remove_source_refuses_live_building_row(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("remove_source_refuses_live_building_row");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    let name = info.table_name().to_string();

    let err = catalog
        .delete_result_tables_for_source("src1")
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::SourceBusy { source_id, table } if source_id == "src1" && *table == name),
        "got {err:?}"
    );
    assert_eq!(
        record(&catalog, &name).await.status,
        ResultTableStatus::Building.to_string()
    );

    // The writer dies and its lease runs out: the row is a dead writer's and
    // is deleted with the rest of the source's tables.
    abandon_building(&catalog, info).await;
    let deleted = catalog
        .delete_result_tables_for_source("src1")
        .await
        .unwrap();
    assert_eq!(deleted.len(), 1);
    assert_eq!(deleted[0].table_name, name);
    assert!(catalog.get_result_table(&name).await.unwrap().is_none());
}

/// Block #6: `delete_result_tables_for_source` is ONE atomic statement, not a
/// SELECT-then-DELETE — pinned by an interleaving this source has TWO rows
/// for: a terminal (`ready`) row the DELETE's own `WHERE` would otherwise
/// happily remove, and a live-lease `building` row it must refuse. A
/// SELECT-then-DELETE bug would delete the `ready` row regardless of the
/// `building` row's fate (they are independent rows, no shared predicate ties
/// their deletion together in a naive implementation); the atomic rewrite
/// makes `SourceBusy` roll back the WHOLE transaction, so the terminal row
/// must survive exactly as the busy row does — the caller's returned
/// cleanup set and the catalog's actual state can never diverge.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn source_busy_rolls_back_the_whole_delete_not_only_the_busy_row(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("source_busy_rolls_back_the_whole_delete_not_only_the_busy_row");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    // A terminal row for the SAME source — the naive SELECT-then-DELETE's
    // own unconditional `WHERE source_id = $1` would remove this one
    // regardless of the busy row's outcome.
    let ready = create_building_embedding(&store).await;
    let ready_name = ready.table_name().to_string();
    write_closed_embedding_parquet(&store, &ready, 2).await;
    let ctx = SessionContext::new();
    let (descriptor, env) = (descriptor(), env());
    ready
        .finish(&ctx, 2, Materialization::new(&descriptor, &env, inputs()))
        .await
        .unwrap();
    assert_eq!(
        record(&catalog, &ready_name).await.status,
        ResultTableStatus::Ready.to_string(),
        "precondition: the sibling row is terminal"
    );

    // A live-lease `building` row, same source.
    let building = create_building_embedding(&store).await;
    let building_name = building.table_name().to_string();

    let err = catalog
        .delete_result_tables_for_source("src1")
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::SourceBusy { source_id, table }
            if source_id == "src1" && *table == building_name),
        "got {err:?}"
    );

    // BOTH rows survive — the terminal one is not a partial casualty of a
    // refusal that only concerned the busy row.
    assert!(
        catalog
            .get_result_table(&ready_name)
            .await
            .unwrap()
            .is_some(),
        "SourceBusy must roll back the WHOLE transaction: the terminal sibling \
         row must survive exactly as the busy row does"
    );
    assert!(catalog
        .get_result_table(&building_name)
        .await
        .unwrap()
        .is_some());
    let _ = building; // keep the live lease held for the duration of the call above
}

/// Two recoverers racing on ONE expired-lease row (valid Parquet + sidecar):
/// exactly one claims and promotes; the other's claim matches zero rows and it
/// writes nothing — one segment set, one owner, the true row count.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_recoverers_race_on_one_expired_row(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("two_recoverers_race_on_one_expired_row");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = result_store(dir.path(), Arc::clone(&catalog));

    let info = create_building_embedding(&store).await;
    write_closed_embedding_parquet(&store, &info, 8).await;
    // A stale two-segment set the dead writer left behind; the winner's
    // rebuild replaces it with exactly one.
    info.append_segment(&built_index(4)).await.unwrap();
    info.append_segment(&built_index(8)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "src1", DIMS).await;
    let url = info.parquet_url().clone();
    let name = abandon_building(&catalog, info).await;

    let r1 = result_store(dir.path(), global_sibling(&catalog));
    let r2 = result_store(dir.path(), global_sibling(&catalog));
    let (a, b) = tokio::join!(r1.recover(), r2.recover());
    a.unwrap();
    b.unwrap();

    let rec = record(&catalog, &name).await;
    assert_eq!(rec.status, ResultTableStatus::Ready.to_string());
    assert_eq!(rec.row_count, 8);
    assert!(rec.definition_hash.is_some());
    let owner = rec
        .writer_id
        .as_deref()
        .expect("the winner is the writer of record");
    assert!(
        owner == r1.writer_id() || owner == r2.writer_id(),
        "owner {owner} must be one of the two recoverers"
    );
    let segs = catalog.list_index_segments(&name).await.unwrap();
    assert_eq!(segs.len(), 1, "exactly one rebuilt segment: {segs:?}");
    assert!(parquet_exists(&store, &url).await);
    assert!(store.resolve_search_mode(&rec).await.unwrap().is_some());
}
