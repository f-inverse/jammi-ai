//! `ResultStore::reconcile` / `reconcile_all` — the object-store cross-check
//! against the catalog. Every oracle here is engine-level (`file://` and
//! `memory://`, SQLite catalog); the wire/CLI surface lands in a later
//! commit.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ComputeDevice, ComputePrecision, MaterializationEnv, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{
    BuildingTable, EmbeddingTableSpec, Materialization, ReconcileOptions, ResultStore,
};
use jammi_db::TenantId;
use tempfile::tempdir;
use uuid::Uuid;

const DIMS: usize = 4;

fn fresh_tenant() -> TenantId {
    TenantId::from_uuid(Uuid::new_v4()).unwrap()
}

fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::Embedding {
        model_id: "test-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: "docs".into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "test-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("reconcile-fixture-digest".into()),
            quantization: None,
        }],
    )
}

fn sample_rows(n: usize) -> Vec<(String, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let vec = (0..DIMS).map(|d| (i * DIMS + d) as f32 + 1.0).collect();
            (format!("row-{i}"), vec)
        })
        .collect()
}

async fn materialize_healthy_table(
    store: &ResultStore,
    ctx: &SessionContext,
    source_id: &str,
) -> jammi_db::catalog::result_repo::ResultTableRecord {
    let rows = sample_rows(5);
    store
        .materialize_embedding_table(
            ctx,
            EmbeddingTableSpec {
                source_id,
                model_id: "test-model",
                derived_from: None,
                dimensions: DIMS,
                key_column: Some("_row_id"),
                text_columns: Some("body"),
            },
            &rows,
            Materialization::new(&descriptor(), &env(), vec![]),
        )
        .await
        .unwrap()
}

/// Backdate the mtime of every regular file directly under `dir` by
/// `by` — lets a test manufacture an "already past grace" orphan candidate
/// deterministically, with no real-time sleep.
fn backdate_dir(dir: &std::path::Path, by: Duration) {
    let target = std::time::SystemTime::now() - by;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::File::options()
                .write(true)
                .open(entry.path())
                .unwrap()
                .set_modified(target)
                .unwrap();
        }
    }
}

/// The lease-duration floor `apply=true` must respect. This test never
/// actually waits for expiry — it only checks the synchronous config
/// guard — so a short-but-valid whole-second pair
/// (`LeaseConfig::intervals`'s only public constructor) is enough.
fn short_lease() -> jammi_db::catalog::lease::LeaseIntervals {
    jammi_db::config::LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap()
}

/// Register a `building` embedding row and write a valid, closed Parquet
/// under it directly — bypassing the catalog's `building -> ready` flip, so
/// the row stays `building` with real bytes on disk and NO
/// `.materialization.json` sidecar: the "torn write before manifest" shape
/// [`ResultStore::recover`]'s expired-building pre-pass reaps via
/// `reap_after_fail_cas` (fails the row, then deletes the Parquet).
async fn create_building_embedding_with_parquet(
    store: &ResultStore,
    source_id: &str,
    n: usize,
) -> BuildingTable {
    let info = store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            None,
        )
        .await
        .unwrap();

    let schema = embedding_table_schema(DIMS);
    let row_ids: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    let row_id_arr = StringArray::from_iter_values(row_ids.iter().map(|s| s.as_str()));
    let source_arr = StringArray::from_iter_values((0..n).map(|_| source_id));
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
    writer.close().await.unwrap();
    info
}

// ─── report-count correctness: a key the expired-building pre-pass already
//     claimed/deleted must never ALSO be counted by this pass's own
//     orphan/bytes_reclaimed accounting off a listing snapshot taken before
//     the pre-pass ran ───────────────────────────────────────────────────────

#[tokio::test]
async fn pre_pass_deleted_table_is_not_double_counted() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // A `building` row: valid Parquet, no manifest, expired lease — exactly
    // the state the pre-pass reaps via `reap_after_fail_cas` (fails the row,
    // deletes the Parquet) under `apply=true`.
    let info = create_building_embedding_with_parquet(&store, "docs-torn", 5).await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    // `abandon_building` asserts the row is `building` under a live lease,
    // detaches the writer's handle (so no background heartbeat can renew it
    // out from under the next line), THEN forces the lease into the past.
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;
    // Backdate the Parquet so it would ALSO qualify as a past-grace orphan
    // candidate if a stale (pre-pre-pass) listing snapshot were checked
    // against it — the exact condition that would double-count it.
    let table_dir = std::path::Path::new(&parquet_local).parent().unwrap();
    backdate_dir(table_dir, Duration::from_secs(3600));

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    // The pre-pass reaped the row: it is `failed`, not still `building`.
    let row = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "{report:?}"
    );
    // The pre-pass's own delete actually removed the bytes.
    assert!(
        !std::path::Path::new(&parquet_local).exists(),
        "the pre-pass must have deleted the torn row's Parquet"
    );

    // The key the pre-pass already deleted must NEVER show up in this SAME
    // pass's own orphan accounting — it was never listed in the first place
    // (listing runs AFTER the pre-pass), so it cannot be double-counted.
    let parquet_key = std::path::Path::new(&parquet_local)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        !report.orphans.iter().any(|o| o.ends_with(&parquet_key)),
        "a pre-pass-deleted key must not double-count as this pass's own orphan: {report:?}"
    );
    assert_eq!(
        report.bytes_reclaimed, 0,
        "the pre-pass's own delete must not be double-counted in bytes_reclaimed: {report:?}"
    );
}

// ─── S1: a healthy F32 indexed `ready` table survives `apply=true` ─────────

#[tokio::test]
async fn healthy_table_survives_reconcile_apply() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();

    assert!(report.rows_failed.is_empty(), "{report:?}");
    assert!(report.orphans.is_empty(), "{report:?}");
    assert!(report.unattributed.is_empty(), "{report:?}");
    assert_eq!(report.bytes_reclaimed, 0);

    // The table is still there, byte-for-byte reachable.
    let still_there = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap();
    assert!(
        still_there.is_some(),
        "reconcile must not touch a live table"
    );
}

// ─── apply=false mutates nothing, even with a missing object ───────────────

#[tokio::test]
async fn apply_false_never_mutates() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    // Delete the Parquet out from under the row directly (never through the
    // engine) to manufacture a missing-required-object condition.
    std::fs::remove_file(record.parquet_path.trim_start_matches("file://")).ok();

    let report = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(0),
        })
        .await
        .unwrap();
    assert!(
        report.rows_failed.contains(&record.table_name),
        "a missing required object must be REPORTED even under apply=false: {report:?}"
    );

    // But nothing was mutated: the row is still `ready`.
    let still_ready = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        still_ready.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "apply=false must not flip the row"
    );
}

// ─── missing required object -> `ready -> failed`, then its objects are
//     orphans (reaped only past grace, under `apply`) ────────────────────

#[tokio::test]
async fn missing_object_fails_the_row_and_is_reaped_past_grace() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    // Remove ONE required sidecar sibling (not the Parquet itself) so the
    // Parquet object survives to become the orphan candidate reconcile then
    // reaps.
    let segs = store
        .catalog()
        .list_index_segments(&record.table_name)
        .await
        .unwrap();
    assert_eq!(segs.len(), 1, "5 rows fit in one segment");
    let idx_path = segs[0].index_path.trim_start_matches("file://");
    let (stem, _ext) = idx_path.rsplit_once('.').unwrap();
    let usearch_path = format!("{stem}.usearch");
    std::fs::remove_file(&usearch_path).unwrap();
    // Backdate every OTHER surviving object of this table so it is already
    // past a short grace window when reconcile runs — no real-time sleep
    // needed to prove "past grace" reclaim.
    let table_dir = std::path::Path::new(idx_path).parent().unwrap();
    backdate_dir(table_dir, Duration::from_secs(3600));

    let report = store
        .clone()
        .with_lease_intervals(short_lease())
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report.rows_failed.contains(&record.table_name),
        "missing sidecar must fail the row: {report:?}"
    );
    // At grace=0 every orphan candidate (now including the Parquet + rowmap +
    // manifest siblings, since the row is no longer `ready`) is reclaimed.
    assert!(
        !report.orphans.is_empty(),
        "the failed row's now-unreferenced objects must be orphan candidates: {report:?}"
    );
    assert!(report.bytes_reclaimed > 0);

    let row = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string()
    );
}

// ─── a young orphan is `pending`, never deleted, at any `apply` ───────────

#[tokio::test]
async fn young_orphan_is_pending_not_deleted() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();

    // A stray object under the `_global` segment with no catalog row at all
    // (a losing writer's abandoned bytes) — attributable, but unreferenced.
    let stray = dir.path().join("jammi_db").join("_global");
    std::fs::create_dir_all(&stray).unwrap();
    std::fs::write(stray.join("orphan_table.parquet"), b"stray-bytes").unwrap();

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        report
            .pending
            .iter()
            .any(|p| p.ends_with("orphan_table.parquet")),
        "a fresh orphan must be `pending` under a long grace: {report:?}"
    );
    assert!(report.orphans.is_empty());
    assert!(stray.join("orphan_table.parquet").exists());
}

// ─── an unattributed key survives even at the SHORTEST valid grace under
//     `apply=true` (unattributed is skipped before the age gate at all,
//     so this holds independent of the object's age) — reported ONLY by
//     the admin cross-tenant pass (a scoped pass, even an
//     unscoped/GLOBAL one, reports NOTHING it cannot attribute to its own
//     prefix; an unattributed key is store-wide by definition) ───────────

#[tokio::test]
async fn unattributed_key_never_deleted_regardless_of_grace() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // A pre-layout / garbage key directly at the root — its own first path
    // segment does not parse through `TenantSegment::parse` at all.
    let root = dir.path().join("jammi_db");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pre_layout_table.parquet"), b"ancient-bytes").unwrap();
    backdate_dir(&root, Duration::from_secs(3600));

    let report = store
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report
            .unattributed
            .contains(&"pre_layout_table.parquet".to_string()),
        "{report:?}"
    );
    assert_eq!(report.unattributed_count, 1);
    assert!(report.orphans.is_empty());
    assert!(root.join("pre_layout_table.parquet").exists());
}

/// RED first: the SAME stray key as above, but seen through the
/// SCOPED arm (`ResultStore::reconcile`, even on an unscoped/GLOBAL store —
/// scoped is scoped regardless of which tenant) — must report NOTHING for
/// it. Before the fix this failed: a scoped pass listed every unattributed
/// key store-wide.
#[tokio::test]
async fn scoped_reconcile_never_reports_unattributed_keys() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let root = dir.path().join("jammi_db");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pre_layout_table.parquet"), b"ancient-bytes").unwrap();
    backdate_dir(&root, Duration::from_secs(3600));

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report.unattributed.is_empty(),
        "a scoped pass must report NO unattributed entries: {report:?}"
    );
    assert_eq!(report.unattributed_count, 0, "{report:?}");
    // Never deleted either way — out of scope, not merely "not old enough".
    assert!(root.join("pre_layout_table.parquet").exists());
}

// ─── apply=true requires grace >= lease.duration ──────────────────────────

#[tokio::test]
async fn apply_requires_grace_at_least_the_lease_duration() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), catalog, AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let err = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_millis(1),
        })
        .await
        .unwrap_err();
    assert!(
        matches!(err, jammi_db::error::JammiError::Config(_)),
        "a too-short grace under apply=true must be a typed Config refusal, got: {err:?}"
    );

    // apply=false tolerates ANY grace.
    store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_millis(1),
        })
        .await
        .unwrap();
}

// ─── two-tenant scope: a tenant's own reconcile never touches / reports
//     another tenant's rows or stray objects; reconcile_all covers both ───

#[tokio::test]
async fn tenant_scoped_reconcile_never_touches_another_tenants_prefix() {
    let dir = tempdir().unwrap();
    let base_catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();
    let catalog_a = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_a)));
    let catalog_b = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_b)));

    let store_a = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_a),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let store_b = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_b),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let ctx = SessionContext::new();
    let record_a = materialize_healthy_table(&store_a, &ctx, "docs-a").await;
    let record_b = materialize_healthy_table(&store_b, &ctx, "docs-b").await;

    // A stray object under tenant B's own segment — B's problem, not A's.
    let stray_dir = dir.path().join("jammi_db").join(tenant_b.to_string());
    std::fs::create_dir_all(&stray_dir).unwrap();
    std::fs::write(stray_dir.join("stray.parquet"), b"stray").unwrap();
    backdate_dir(&stray_dir, Duration::from_secs(3600));

    // Tenant A's own reconcile: sees nothing of B's, reports nothing of B's,
    // and never deletes B's stray (it is out of A's scope, not A's orphan).
    let report_a = store_a
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(report_a.rows_failed.is_empty(), "{report_a:?}");
    assert!(report_a.orphans.is_empty(), "{report_a:?}");
    assert!(stray_dir.join("stray.parquet").exists());

    // Both tables (A's and B's) are still intact after A's own pass.
    assert!(store_a
        .catalog()
        .get_result_table(&record_a.table_name)
        .await
        .unwrap()
        .is_some());
    assert!(store_b
        .catalog()
        .get_result_table(&record_b.table_name)
        .await
        .unwrap()
        .is_some());

    // reconcile_all (admin) reaps ONLY the stray, leaving both tenants'
    // live tables byte-identical.
    let report_all = store_a
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(report_all.rows_failed.is_empty(), "{report_all:?}");
    assert!(
        report_all
            .orphans
            .iter()
            .any(|o| o.ends_with("stray.parquet")),
        "{report_all:?}"
    );
    assert!(!stray_dir.join("stray.parquet").exists());
    assert!(store_a
        .catalog()
        .get_result_table(&record_a.table_name)
        .await
        .unwrap()
        .is_some());
    assert!(store_b
        .catalog()
        .get_result_table(&record_b.table_name)
        .await
        .unwrap()
        .is_some());
}

// ─── a scoped pass reports NOTHING it cannot act on: a GLOBAL ready row
//     with a missing required object is invisible to a tenant-scoped
//     dry-run AND apply alike — only an admin (`all=true`) pass ever
//     touches it. RED before the fix: `list_result_tables_by_status` (an
//     ordinary READ — GLOBAL visible to every tenant, like any other read on
//     this table) fed straight into the row->object CAS below it, so a
//     tenant-scoped dry-run REPORTED the GLOBAL row in `rows_failed` while
//     the matching `apply` pass's `fail_ready_result_table` CAS (`Strict` on
//     the caller's own tenant) missed it entirely — dry-run and apply
//     disagreed on the identical state. ─────────────────────────────────────

#[tokio::test]
async fn scoped_pass_reports_nothing_for_a_global_row_it_cannot_act_on() {
    let dir = tempdir().unwrap();
    let base_catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let tenant_b = fresh_tenant();
    let catalog_b = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_b)));

    let store_global = ResultStore::new(
        dir.path(),
        Arc::clone(&base_catalog),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let store_b = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_b),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store_global, &ctx, "docs-global").await;

    // Remove the GLOBAL row's `.materialization.json` sidecar so it fails
    // `required_row_objects_present` — a missing required object, exactly
    // the condition that would (incorrectly) drive a scoped dry-run to
    // report it.
    let parquet_local = record.parquet_path.trim_start_matches("file://");
    let (stem, _ext) = parquet_local.rsplit_once('.').unwrap();
    let mat_path = format!("{stem}.materialization.json");
    assert!(
        std::path::Path::new(&mat_path).exists(),
        "fixture must actually produce a materialization sidecar to remove"
    );
    std::fs::remove_file(&mat_path).unwrap();

    // Tenant B's dry-run: reports NOTHING for the GLOBAL row.
    let dry = store_b
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(dry.rows_failed.is_empty(), "{dry:?}");
    assert_eq!(dry.rows_failed_count, 0, "{dry:?}");

    // Tenant B's apply: agrees with its own dry-run on the same state.
    let apply = store_b
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(apply.rows_failed.is_empty(), "{apply:?}");
    assert_eq!(apply.rows_failed_count, 0, "{apply:?}");

    // The GLOBAL row stayed `ready` through both of tenant B's passes.
    let row = store_global
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "a scoped pass must never flip a row it cannot even report on"
    );

    // An admin (`all=true`) pass sees and reports it in BOTH modes.
    let admin_dry = store_b
        .reconcile_all(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        admin_dry.rows_failed.contains(&record.table_name),
        "{admin_dry:?}"
    );

    let admin_apply = store_b
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        admin_apply.rows_failed.contains(&record.table_name),
        "{admin_apply:?}"
    );
    let row_after = store_global
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row_after.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "the admin apply pass must actually act on the row it reported"
    );
}

// ─── artifact arm: a running job's checkpoints survive; a canonical-UUID
//     job id never lands in `unattributed` ─────────────────────────────────

#[tokio::test]
async fn running_jobs_artifact_prefix_survives_and_is_never_unattributed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();

    catalog
        .register_model(RegisterModelParams {
            model_id: "base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    let job_id = Uuid::new_v4().to_string();
    catalog
        .create_training_job(CreateTrainingJobParams {
            job_id: &job_id,
            base_model_id: "base::1",
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    catalog
        .claim_next_training_job("worker-1", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("the freshly queued job is claimable");

    let bundle = vec![(
        "adapter.safetensors".to_string(),
        bytes::Bytes::from_static(b"weights"),
    )];
    store
        .artifact_store()
        .put_artifact(None, &[&job_id, "worker-1", "0"], &bundle)
        .await
        .unwrap();

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(30),
        })
        .await
        .unwrap();
    assert!(
        report.unattributed.is_empty(),
        "a canonical-UUID job id under a running job must never be unattributed: {report:?}"
    );
    assert!(
        report.orphans.is_empty(),
        "a running job's published bytes must survive reconcile: {report:?}"
    );

    // The bytes are still there.
    assert!(dir
        .path()
        .join("jammi_db")
        .join("models")
        .join("_global")
        .join(&job_id)
        .join("worker-1")
        .join("0")
        .join("adapter.safetensors")
        .exists());
}

// ─── RED first: a `models` row names a prefix whose `manifest.json`
//     is absent — the row is still live, so the prefix is NEVER reclaimed
//     through the orphan arm, at any grace or `apply`; it is reported as
//     `damaged` instead (row present, manifest absent) ────────────────────

#[tokio::test]
async fn models_row_prefix_with_no_manifest_is_damaged_never_reclaimed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let job_id = Uuid::new_v4().to_string();
    let bundle = vec![(
        "adapter.safetensors".to_string(),
        bytes::Bytes::from_static(b"weights"),
    )];
    let prefix_url = store
        .artifact_store()
        .put_artifact(None, &[&job_id], &bundle)
        .await
        .unwrap();

    // A `models` row names exactly this prefix as its winning attempt.
    catalog
        .register_model(RegisterModelParams {
            model_id: "damaged-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some(prefix_url.as_str()),
            config_json: None,
        })
        .await
        .unwrap();

    // Torn write / corrupted publish: the manifest is gone, the weights
    // survive. Backdate everything so it is well past a short grace.
    let prefix_dir = dir
        .path()
        .join("jammi_db")
        .join("models")
        .join("_global")
        .join(&job_id);
    std::fs::remove_file(prefix_dir.join("manifest.json")).unwrap();
    backdate_dir(&prefix_dir, Duration::from_secs(3600));

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    assert!(
        report
            .damaged
            .iter()
            .any(|d| d.ends_with("adapter.safetensors")),
        "a models-row-named prefix with no manifest must be reported damaged: {report:?}"
    );
    assert_eq!(report.damaged_count, 1, "{report:?}");
    assert!(
        report
            .orphans
            .iter()
            .all(|o| !o.ends_with("adapter.safetensors")),
        "damaged bytes must never fall through to the orphan arm: {report:?}"
    );
    assert!(
        prefix_dir.join("adapter.safetensors").exists(),
        "damaged bytes are never reclaimed, even past grace under apply=true"
    );
}
