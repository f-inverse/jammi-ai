//! OPS (#482) — the building-table lease class under RELEASE: the linked
//! sweep `Catalog::release_building_tables_of_claimant` (scoped through the
//! `jobs` linkage, never through `writer_id` alone), the `lease_present`
//! arm every `renew_lease` carries so a released building lease is never
//! re-armed, and the keeper's behaviour on a released `ResultTable` hold.
//!
//! Every test is parameterised over [`BackendKind`] the way `jobs_queue.rs`
//! is (SQLite always; Postgres under `live-postgres-tests`, skipped at
//! runtime when `JAMMI_TEST_PG_URL` is unset).

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::lease_keeper::LeaseTarget;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{
    CreateResultTableParams, JobAttempt, ResultTableCas, ResultTableKind,
};
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::config::{LeaseConfig, StoragePrecision};
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;

use crate::common::keeper_for_backend;

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

/// A backend-parameterised catalog with the FK base model registered and
/// the queue tables cleared (the Postgres lane shares one database).
macro_rules! catalog_for {
    ($backend:expr, $dir:expr) => {{
        let session = skip_if_no_backend!($backend, $dir);
        let catalog = Arc::clone(session.catalog());
        catalog
            .backend_arc()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute("DELETE FROM jobs", &[]).await?;
                    tx.execute("DELETE FROM workers", &[]).await?;
                    tx.execute("DELETE FROM instances", &[]).await?;
                    Ok(())
                })
            })
            .await
            .unwrap();
        catalog
            .register_model(RegisterModelParams {
                model_id: "rt-base",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .ok();
        (session, catalog)
    }};
}

fn compute_job(job_id: &str, execution: JobExecution) -> SubmitJobParams<'_> {
    SubmitJobParams {
        job_id,
        kind: "embedding",
        execution,
        spec: "{}",
        model_ref: None,
        output_model_id: None,
        model_source: None,
        priority: 0,
    }
}

fn building_row<'a>(
    table: &'a str,
    writer_id: &'a str,
    job_attempt: Option<JobAttempt<'a>>,
) -> CreateResultTableParams<'a> {
    CreateResultTableParams {
        table_name: table,
        source_id: "src",
        model_id: "rt-base",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/rt.parquet",
        dimensions: Some(4),
        key_column: None,
        text_columns: None,
        storage_precision: StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::backend::now_sortable(),
        writer_id: Some(writer_id),
        lease: Some(Duration::from_secs(3600)),
        job_attempt,
    }
}

async fn lease_of(catalog: &Catalog, table: &str) -> Option<String> {
    catalog
        .get_result_table(table)
        .await
        .unwrap()
        .expect("row present")
        .lease_expires_at
}

/// Short, valid lease timing: `heartbeat * 2 < lease`.
fn fast_intervals() -> jammi_db::catalog::lease::LeaseIntervals {
    LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap()
}

/// The sweep oracle (D11's two-class form): after `release_jobs_claimed_by`
/// + `release_building_tables_of_claimant`, exactly the building rows of
/// THIS instance's loop-claimed compute jobs have a NULL lease and are
/// claimable by a successor at once; a second sweep matches 0 rows; an
/// inline `run_now` row's building row, a library materialization with no
/// jobs linkage, and a `ready` row — all under the SAME `writer_id` — are
/// untouched. A writer-scoped sweep would NULL the inline and library rows;
/// base has no sweep at all (a live building lease is claimable only after
/// expiry).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn released_building_table_is_claimable_by_the_successor_at_once(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = catalog_for!(backend, dir.path());
    let suffix = unique_suffix();
    let writer = format!("writer-{suffix}");
    let queued_table = format!("sw_queued_{suffix}");
    let inline_table = format!("sw_inline_{suffix}");
    let library_table = format!("sw_library_{suffix}");
    let ready_table = format!("sw_ready_{suffix}");
    let lease = Duration::from_secs(3600);

    // The loop-claimed compute job and its linked building row.
    catalog
        .submit_job(compute_job("sw-q", JobExecution::Queued))
        .await
        .unwrap();
    let q = catalog
        .claim_next("me", &["embedding"], lease)
        .await
        .unwrap()
        .expect("queued job claimed");
    catalog
        .create_result_table(building_row(
            &queued_table,
            &writer,
            Some(JobAttempt {
                job_id: "sw-q",
                instance_id: "me",
                attempts: q.attempts,
            }),
        ))
        .await
        .unwrap();
    // The inline `run_now` row under the same instance and writer.
    catalog
        .submit_job(compute_job("sw-i", JobExecution::Inline))
        .await
        .unwrap();
    let i = catalog
        .claim_by_id("sw-i", "me", lease)
        .await
        .unwrap()
        .expect("inline job claimed");
    catalog
        .create_result_table(building_row(
            &inline_table,
            &writer,
            Some(JobAttempt {
                job_id: "sw-i",
                instance_id: "me",
                attempts: i.attempts,
            }),
        ))
        .await
        .unwrap();
    // A library materialization: same writer, no jobs row at all.
    catalog
        .create_result_table(building_row(&library_table, &writer, None))
        .await
        .unwrap();
    // A `ready` row of the same writer.
    catalog
        .create_result_table(building_row(&ready_table, &writer, None))
        .await
        .unwrap();
    catalog
        .promote_result_table_with_manifest(
            &ResultTableCas::writer(&ready_table, &writer, None),
            1,
            "hash",
            "[]",
        )
        .await
        .unwrap();

    // Base fact: a live building lease is NOT claimable by a successor.
    assert!(
        !catalog
            .claim_expired_building_table(
                &ResultTableCas::expired(&queued_table, None),
                "successor",
                lease
            )
            .await
            .unwrap(),
        "a live building lease is not claimable"
    );

    // Sweep #1: jobs first, then the linked building rows.
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 1);
    assert_eq!(
        catalog
            .release_building_tables_of_claimant("me", &writer)
            .await
            .unwrap(),
        1,
        "exactly the loop-claimed job's building row is released"
    );
    assert!(lease_of(&catalog, &queued_table).await.is_none());
    assert!(
        lease_of(&catalog, &inline_table).await.is_some(),
        "the inline row's building lease is untouched"
    );
    assert!(
        lease_of(&catalog, &library_table).await.is_some(),
        "a library materialization's lease is untouched"
    );
    let ready = catalog
        .get_result_table(&ready_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ready.status, "ready");
    assert!(ready.lease_expires_at.is_none());
    let job = catalog.get_job("sw-q").await.unwrap();
    assert_eq!(job.status, JobStatus::Running.to_string());
    assert!(job.lease_expires_at.is_none());
    assert_eq!(job.releases, 1);
    assert!(
        catalog
            .get_job("sw-i")
            .await
            .unwrap()
            .lease_expires_at
            .is_some(),
        "the inline jobs row keeps its lease"
    );

    // Sweep #2 is idempotent on both classes.
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 0);
    assert_eq!(
        catalog
            .release_building_tables_of_claimant("me", &writer)
            .await
            .unwrap(),
        0
    );

    // The successor claims the released building row at once; the inline
    // row's live lease still refuses.
    assert!(catalog
        .claim_expired_building_table(
            &ResultTableCas::expired(&queued_table, None),
            "successor",
            lease
        )
        .await
        .unwrap());
    assert!(!catalog
        .claim_expired_building_table(
            &ResultTableCas::expired(&inline_table, None),
            "successor",
            lease
        )
        .await
        .unwrap());
    assert_eq!(
        catalog
            .get_result_table(&queued_table)
            .await
            .unwrap()
            .unwrap()
            .writer_id
            .as_deref(),
        Some("successor")
    );
}

/// The re-arm guard on the building class: a released (NULL) building lease
/// renews 0 rows under the writer's own CAS — `renew_lease` always carries
/// the `lease_present` arm, so the miss surfaces as `CasFailed { status:
/// "building" }` (the fifth `classify_cas_miss` cause) — and a keeper hold
/// registered on the released row flips `lost` within one heartbeat while
/// the lease stays NULL. Base: the renewal re-arms the lease.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_released_building_lease_is_never_re_armed_by_the_keeper(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = catalog_for!(backend, dir.path());
    let suffix = unique_suffix();
    let writer = format!("writer-{suffix}");
    let table = format!("rearm_{suffix}");

    catalog
        .submit_job(compute_job("rearm-q", JobExecution::Queued))
        .await
        .unwrap();
    let q = catalog
        .claim_next("me", &["embedding"], Duration::from_secs(3))
        .await
        .unwrap()
        .expect("queued job claimed");
    catalog
        .create_result_table(building_row(
            &table,
            &writer,
            Some(JobAttempt {
                job_id: "rearm-q",
                instance_id: "me",
                attempts: q.attempts,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 1);
    assert_eq!(
        catalog
            .release_building_tables_of_claimant("me", &writer)
            .await
            .unwrap(),
        1
    );
    assert!(lease_of(&catalog, &table).await.is_none());

    // The writer's own renewal (the shape `BuildingTable::finish` and the
    // keeper both use) misses with the status-named error.
    let renewed = catalog
        .renew_lease(
            &ResultTableCas::writer(&table, &writer, None),
            Duration::from_secs(3),
        )
        .await;
    match renewed {
        Err(JammiError::CasFailed { status, .. }) => assert_eq!(status, "building"),
        other => panic!("a released building lease must not renew; got {other:?}"),
    }
    assert!(
        lease_of(&catalog, &table).await.is_none(),
        "the renewal must not re-arm the released lease"
    );

    // A keeper hold on the released row: lost within one heartbeat, lease
    // still NULL after two.
    let keeper = keeper_for_backend(backend, dir.path().to_path_buf(), fast_intervals()).await;
    let hold = keeper.hold(LeaseTarget::ResultTable {
        table: table.clone(),
        writer_id: writer.clone(),
    });
    tokio::time::sleep(Duration::from_millis(2_200)).await;
    assert!(
        hold.lost(),
        "the keeper's renewal matched 0 rows, so the hold reads lost"
    );
    assert!(
        lease_of(&catalog, &table).await.is_none(),
        "two heartbeats later the released lease is still NULL"
    );
    drop(hold);
    keeper
        .shutdown_and_join(Duration::from_secs(10))
        .await
        .unwrap();
}

/// The `lease_present` arm is a plain `pub` field: `false` in every builder
/// (the expired-lease owner arm must never carry it — its own predicate is
/// `IS NULL OR < now`), set by `with_lease_present`, and rendered only when
/// set.
#[test]
fn lease_present_is_false_in_every_builder_and_set_by_with_lease_present() {
    assert!(!ResultTableCas::writer("t", "w", None).lease_present);
    assert!(!ResultTableCas::writer_any_tenant("t", "w").lease_present);
    assert!(!ResultTableCas::expired("t", None).lease_present);
    assert!(
        ResultTableCas::writer("t", "w", None)
            .with_lease_present()
            .lease_present
    );
}

/// A released lease is a value the row carries, not an in-memory flag: a
/// raw `SELECT` after the sweep reads `lease_expires_at IS NULL` on the
/// released building row (the successor's `claim_expired_building_table`
/// predicate is `IS NULL OR < now`).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_sweep_writes_a_null_lease_the_backend_reads_back(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = catalog_for!(backend, dir.path());
    let suffix = unique_suffix();
    let writer = format!("writer-{suffix}");
    let table = format!("raw_{suffix}");
    catalog
        .submit_job(compute_job("raw-q", JobExecution::Queued))
        .await
        .unwrap();
    let q = catalog
        .claim_next("me", &["embedding"], Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();
    catalog
        .create_result_table(building_row(
            &table,
            &writer,
            Some(JobAttempt {
                job_id: "raw-q",
                instance_id: "me",
                attempts: q.attempts,
            }),
        ))
        .await
        .unwrap();
    catalog.release_jobs_claimed_by("me").await.unwrap();
    catalog
        .release_building_tables_of_claimant("me", &writer)
        .await
        .unwrap();
    let table_for_query = table.clone();
    let null_count: i64 = catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, i64>(
                        "SELECT COUNT(*) AS n FROM result_tables \
                         WHERE table_name = $1 AND lease_expires_at IS NULL",
                        &[SqlValue::TextOwned(table_for_query)],
                        |row| row.get("n"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(null_count, 1);
}
