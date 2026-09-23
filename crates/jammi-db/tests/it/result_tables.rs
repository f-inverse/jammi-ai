//! The building-table lease class under RELEASE: the linked
//! sweep `Catalog::release_building_tables_of_claimant` (scoped through the
//! `jobs` linkage, never through `writer_id` alone), the `lease_present`
//! arm every `renew_lease` carries so a released building lease is never
//! re-armed, and the keeper's behaviour on a released `ResultTable` hold.
//!
//! Every test is parameterised over [`BackendKind`] the way `jobs_queue.rs`
//! is (SQLite always; Postgres under `live-postgres-tests`).

use std::sync::Arc;
use std::time::Duration;

use jammi_datafusion::ModelTask;
use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::lease_keeper::LeaseTarget;
use jammi_db::catalog::result_repo::{
    CreateResultTableParams, JobAttempt, ResultTableCas, ResultTableKind,
};
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::config::{LeaseConfig, StoragePrecision};
use jammi_db::error::JammiError;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{keeper_for_backend, queue_session, unique_suffix, BASE_MODEL_ID};

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
        model_id: BASE_MODEL_ID,
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/rt.parquet",
        dimensions: Some(4),
        key_column: None,
        text_columns: None,
        storage_precision: StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::lease::canonical_stamp_now(),
        writer_id: Some(writer_id),
        lease: Some(Duration::from_secs(3600)),
        job_attempt,
        replaces: None,
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

/// The sweep oracle (two-class form): after `release_jobs_claimed_by`
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
    let (_session, catalog) = queue_session(backend, dir.path()).await;
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
    let (_session, catalog) = queue_session(backend, dir.path()).await;
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
    let (_session, catalog) = queue_session(backend, dir.path()).await;
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

// ---------------------------------------------------------------------------
// `Catalog::get_result_table_for_tenant` — the STRICT tenant-pinned resolver
// the gang admission handler resolves a `world_size > 1` job's
// `training_set_location` through.
// ---------------------------------------------------------------------------

fn strict_tenant(n: u8) -> jammi_db::TenantId {
    use std::str::FromStr;
    jammi_db::TenantId::from_str(&format!("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e{n:02x}")).unwrap()
}

/// The strict-predicate property, on BOTH backends: a NULL-tenant
/// `result_tables` row (created outside any tenant scope) is visible to a
/// real tenant through the RELAXED `get_result_table` read — the hazard —
/// and NEVER through the strict resolver: `Some(tenant)` resolves nothing,
/// `None` (a GLOBAL caller) resolves it, and ambient admin scope does not
/// widen the strict predicate (the same `None` under `with_admin_scope`).
/// Mutation proof: replacing the strict predicate with the relaxed one
/// (`tenant_id = $2 OR tenant_id IS NULL`) flips the `Some(tenant)` arm
/// to `Some(row)`; adding an admin-scope arm that drops the tenant
/// predicate flips the admin-scope arm the same way.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant(
    kind: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (session, catalog) = queue_session(kind, dir.path()).await;
    let table = format!("strict_null_tenant_{}", unique_suffix());
    catalog
        .create_result_table(building_row(&table, "writer-strict", None))
        .await
        .unwrap();
    let tenant_a = strict_tenant(0x0a);

    let relaxed = session
        .with_tenant_scoped(tenant_a, |scope| {
            let table = table.clone();
            async move { scope.catalog().get_result_table(&table).await.unwrap() }
        })
        .await;
    assert!(
        relaxed.is_some(),
        "the RELAXED get_result_table read must still hand tenant A the NULL-tenant row — \
         the hazard the strict resolver exists to close"
    );

    let strict_for_a = catalog
        .get_result_table_for_tenant(&table, Some(tenant_a))
        .await
        .unwrap();
    assert!(
        strict_for_a.is_none(),
        "the strict resolver must never match a NULL-tenant row for a real tenant, got {strict_for_a:?}"
    );
    let strict_for_global = catalog
        .get_result_table_for_tenant(&table, None)
        .await
        .unwrap();
    assert_eq!(
        strict_for_global.as_ref().map(|r| r.table_name.as_str()),
        Some(table.as_str()),
        "a GLOBAL caller (tenant None) resolves the NULL-tenant row — NULL equals NULL under the strict predicate"
    );
    // Ambient admin scope is task-local: the SAME catalog handle, called
    // inside `with_admin_scope`, is what every admin-widened verb reads —
    // the strict resolver must not.
    let under_admin = session
        .with_admin_scope(|_admin| {
            let table = table.clone();
            let catalog = Arc::clone(&catalog);
            async move {
                catalog
                    .get_result_table_for_tenant(&table, Some(tenant_a))
                    .await
                    .unwrap()
            }
        })
        .await;
    assert!(
        under_admin.is_none(),
        "ambient admin scope must not widen the strict resolver: tenant A still resolves \
         nothing for a NULL-tenant row, got {under_admin:?}"
    );
}

/// A row created UNDER tenant B resolves for B alone: not for tenant A, not
/// for a GLOBAL (`None`) caller, and not for A under ambient admin scope —
/// the other half of the strict predicate (`tenant_id = $t`), on both
/// backends. Together with the NULL-row test above this quantifies the
/// property over every `(row tenant, caller tenant)` class: equal real
/// tenants, unequal real tenants, NULL row vs real caller, real row vs NULL
/// caller, NULL vs NULL.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_result_table_for_tenant_resolves_only_the_owning_tenant(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let (session, catalog) = queue_session(kind, dir.path()).await;
    let table = format!("strict_owned_{}", unique_suffix());
    let tenant_a = strict_tenant(0x1a);
    let tenant_b = strict_tenant(0x1b);
    session
        .with_tenant_scoped(tenant_b, |scope| {
            let table = table.clone();
            async move {
                scope
                    .catalog()
                    .create_result_table(building_row(&table, "writer-strict-b", None))
                    .await
                    .unwrap()
            }
        })
        .await;

    let for_b = catalog
        .get_result_table_for_tenant(&table, Some(tenant_b))
        .await
        .unwrap();
    assert_eq!(
        for_b.as_ref().and_then(|r| r.tenant_id.clone()),
        Some(tenant_b.to_string()),
        "the owning tenant resolves its own row"
    );
    for (label, caller) in [("tenant A", Some(tenant_a)), ("a GLOBAL caller", None)] {
        let got = catalog
            .get_result_table_for_tenant(&table, caller)
            .await
            .unwrap();
        assert!(
            got.is_none(),
            "{label} must not resolve tenant B's row through the strict resolver, got {got:?}"
        );
    }
    // Ambient admin scope is task-local: the SAME catalog handle, called
    // inside `with_admin_scope`, is what every admin-widened verb reads —
    // the strict resolver must not.
    let under_admin = session
        .with_admin_scope(|_admin| {
            let table = table.clone();
            let catalog = Arc::clone(&catalog);
            async move {
                catalog
                    .get_result_table_for_tenant(&table, Some(tenant_a))
                    .await
                    .unwrap()
            }
        })
        .await;
    assert!(
        under_admin.is_none(),
        "ambient admin scope must not let tenant A resolve tenant B's row, got {under_admin:?}"
    );
}
