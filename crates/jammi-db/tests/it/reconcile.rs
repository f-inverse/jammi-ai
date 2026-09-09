//! `ResultStore::reconcile` / `reconcile_all` — the object-store cross-check
//! against the catalog. Every oracle here is engine-level (`file://` and
//! `memory://`, SQLite catalog); the wire/CLI surface lands in a later
//! commit.

use std::sync::Arc;
use std::time::Duration;

use datafusion::prelude::SessionContext;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ComputeDevice, ComputePrecision, MaterializationEnv, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::{EmbeddingTableSpec, Materialization, ReconcileOptions, ResultStore};
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
//     so this holds independent of the object's age) ─────────────────────

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
        .reconcile(ReconcileOptions {
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
    assert!(report.orphans.is_empty());
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
