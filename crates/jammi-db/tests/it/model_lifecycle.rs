//! Model DELETE lifecycle on the `models` catalog table.
//!
//! DELETE removes the row, so it is refused while any reference still points at
//! the model. The load-bearing rule is the referential scan over four edges,
//! each keyed by what it actually stores — the model NAME for the two no-FK
//! edges (`result_tables.model_id`, `jobs.output_model_id`) and the catalog PK
//! for the two FK-backed ones (`jobs.model_ref`, `eval_runs.model_id`).
//! A pk-keyed scan would silently miss the two name-keyed edges, so each is
//! exercised directly. DELETE is strictly tenant-scoped — a tenant touches only a
//! row it owns. The two `jobs` edges are additionally age-gated (N9): a
//! `[jobs] retention_days`-aged terminal row stops blocking, while a
//! non-terminal row blocks indefinitely and a young terminal row still blocks.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` + `cfg_attr`.
//! The SQLite lane is always generated; the Postgres lane is generated only when
//! the `live-postgres-tests` feature is on, and skips at runtime when
//! `JAMMI_TEST_PG_URL` is unset (an early return, never `#[ignore]`). The
//! Postgres lane is where the contract bites hardest: the four-edge scan runs
//! under PG `Serializable` and still surfaces the typed `ModelReferenced` rather
//! than a raw FK error. On the Postgres lane that one catalog DB is shared across
//! the whole run, so each test first clears the referential tables via
//! [`reset_catalog`]; CI's `test-pg` job runs the lane with `--test-threads=1`,
//! so the reset-then-populate sequence cannot race a sibling test.

use std::str::FromStr;

use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;
use test_case::test_case;

/// The Postgres lane returns `None` when `JAMMI_TEST_PG_URL` is unset so the
/// test can early-return rather than `#[ignore]`'ing (CLAUDE.md forbids
/// `#[ignore]`). Yields the base (unscoped) catalog, with the shared referential
/// tables cleared so the four-edge scan and the partial-index checks see only
/// this test's rows.
macro_rules! lifecycle_catalog {
    ($backend:expr, $dir:expr) => {{
        let session = match make_test_session($backend, $dir).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        };
        let catalog = std::sync::Arc::clone(session.catalog());
        reset_catalog(&catalog).await;
        (session, catalog)
    }};
}

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
}

/// Clear every row from the referential tables (children before parents, so the
/// FK-backed deletes do not block) so the global referential scan and the
/// partial-index checks see only the rows this test creates. The SQLite lane has
/// a fresh tempdir per test, but running the reset there too keeps both lanes on
/// one path. Run under `--test-threads=1` on the Postgres lane, so it cannot race
/// a sibling test.
async fn reset_catalog(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                for table in ["eval_runs", "jobs", "result_tables", "models", "sources"] {
                    tx.execute(&format!("DELETE FROM {table}"), &[]).await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();
}

fn register_params(model_id: &str) -> RegisterModelParams<'_> {
    RegisterModelParams {
        model_id,
        version: 1,
        model_type: "embedding",
        backend: "candle",
        task: ModelTask::TextEmbedding,
        base_model_id: None,
        artifact_path: None,
        config_json: None,
    }
}

/// Register a file source so a `result_tables` row can satisfy the
/// `source_id REFERENCES sources(source_id)` FK on the Postgres lane (where FKs
/// are enforced).
async fn register_source(cat: &Catalog, source_id: &str) {
    cat.register_source(
        source_id,
        SourceType::File,
        &SourceConnection {
            url: Some("file:///tmp/src.parquet".into()),
            format: Some(FileFormat::Parquet),
            ..Default::default()
        },
    )
    .await
    .unwrap();
}

/// Resolve the catalog PK an FK-backed reference binds to.
async fn pk_of(cat: &Catalog, name: &str) -> String {
    cat.get_model(name).await.unwrap().unwrap().catalog_pk
}

/// Submit a job naming `model_ref` (PK-keyed, `jobs.model_ref`) and/or
/// `output_model_id` (NAME-keyed, `jobs.output_model_id`) — the two edges
/// `scan_model_references` walks over `jobs`. No claim/finish/fail: the
/// referential scan reads these columns directly off the row `submit_job`
/// writes, so no lease dance is needed to exercise it.
async fn submit_referencing_job(
    cat: &Catalog,
    job_id: &str,
    model_ref: Option<&str>,
    output_model_id: Option<&str>,
) {
    cat.submit_job(SubmitJobParams {
        job_id,
        kind: "fine_tune",
        execution: JobExecution::Queued,
        spec: "{}",
        model_ref,
        output_model_id,
        model_source: None,
        priority: 0,
    })
    .await
    .unwrap();
}

/// Force `jobs.status` and `jobs.updated_at` directly (bypassing every
/// lease guard) so the retention age-predicate (N9) can be exercised without
/// a claim/finish/fail dance for every fixture row. `days_ago` ages
/// `updated_at`; `status` is written byte-for-byte (`"queued"`, `"running"`,
/// `"completed"`, `"failed"`, or any other string a caller wants to probe).
async fn force_job_age(cat: &Catalog, job_id: &str, status: &str, days_ago: i64) {
    let cutoff = (chrono::Utc::now() - chrono::Duration::days(days_ago))
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string();
    cat.backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let status = status.to_string();
            let job_id = job_id.to_string();
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET status = $1, updated_at = $2 WHERE job_id = $3",
                    &[
                        jammi_db::catalog::backend::SqlValue::TextOwned(status),
                        jammi_db::catalog::backend::SqlValue::TextOwned(cutoff),
                        jammi_db::catalog::backend::SqlValue::TextOwned(job_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// HEADLINE: an unreferenced model deletes cleanly and is then absent.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_unreferenced_model_succeeds(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    cat.delete_model("acme/embed-mini", None, false, 30)
        .await
        .expect("an unreferenced model deletes");

    assert!(
        cat.get_model("acme/embed-mini").await.unwrap().is_none(),
        "a deleted model is gone from the catalog entirely (not merely retired)"
    );
}

/// A reference through `result_tables.model_id` (the NAME-keyed, no-FK edge)
/// blocks the delete. A pk-keyed scan would miss this, so it is load-bearing.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_result_table_name_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    register_source(&cat, "src").await;
    // result_tables.model_id stores the model NAME.
    cat.create_result_table(CreateResultTableParams {
        writer_id: None,
        lease: None,
        table_name: "acme_embeddings",
        source_id: "src",
        model_id: "acme/embed-mini",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "/tmp/p.parquet",
        dimensions: Some(384),
        key_column: Some("id"),
        text_columns: None,
        storage_precision: jammi_db::config::StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::backend::now_sortable(),
        job_attempt: None,
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/embed-mini", None, false, 30)
        .await
        .expect_err("a result-table reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"result_tables".to_string()),
            "the blocking edge is reported as result_tables, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced, got {other:?}"),
    }
    assert!(
        cat.get_model("acme/embed-mini").await.unwrap().is_some(),
        "a blocked delete leaves the model in place"
    );
}

/// A reference through `jobs.output_model_id` (the other NAME-keyed, no-FK
/// edge) blocks the delete — matched by the model NAME.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_job_output_name_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    // A base model the job's FK points at, and the output model whose NAME
    // the job records under `output_model_id`.
    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    cat.register_model(register_params("acme/tuned"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;

    submit_referencing_job(&cat, "job-1", Some(&base_pk), Some("acme/tuned")).await;

    let err = cat
        .delete_model("acme/tuned", None, false, 30)
        .await
        .expect_err("an output-model reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"jobs.output_model_id".to_string()),
            "the blocking edge is reported as jobs.output_model_id, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced, got {other:?}"),
    }
}

/// A reference through `jobs.model_ref` (the FK-backed, PK-keyed edge) blocks
/// the delete. The scan — not the database FK — raises the typed error, so it
/// never leaks as a raw constraint violation.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_job_model_ref_pk_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;

    submit_referencing_job(&cat, "job-1", Some(&base_pk), None).await;

    let err = cat
        .delete_model("acme/base", None, false, 30)
        .await
        .expect_err("a model_ref reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"jobs.model_ref".to_string()),
            "the blocking edge is reported as jobs.model_ref, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced (not a raw FK violation), got {other:?}"),
    }
}

/// N9: a TERMINAL `jobs` row past `retention_days` no longer blocks — the
/// referential predicate is age-gated, not sweep-dependent (`prune_jobs`
/// never has to run first for the delete to succeed).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_unblocked_by_a_terminal_job_past_the_retention_window(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;
    submit_referencing_job(&cat, "job-old-terminal", Some(&base_pk), None).await;
    force_job_age(&cat, "job-old-terminal", "completed", 31).await;

    cat.delete_model("acme/base", None, false, 30)
        .await
        .expect("a terminal job past the retention window must not block delete");
}

/// N9: a NON-TERMINAL `jobs` row blocks indefinitely, regardless of age — the
/// age gate only ever lifts a TERMINAL row's block.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_still_blocked_by_an_old_non_terminal_job(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;
    submit_referencing_job(&cat, "job-old-running", Some(&base_pk), None).await;
    force_job_age(&cat, "job-old-running", "running", 3650).await;

    let err = cat
        .delete_model("acme/base", None, false, 30)
        .await
        .expect_err("an old but non-terminal job must still block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"jobs.model_ref".to_string()),
            "got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced, got {other:?}"),
    }
}

/// N9: a YOUNG terminal `jobs` row (inside the retention window) still
/// blocks — the gate is on age, not merely on `status`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_still_blocked_by_a_young_terminal_job(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;
    submit_referencing_job(&cat, "job-young-terminal", Some(&base_pk), None).await;
    force_job_age(&cat, "job-young-terminal", "failed", 1).await;

    let err = cat
        .delete_model("acme/base", None, false, 30)
        .await
        .expect_err("a young terminal job (inside the retention window) must still block");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"jobs.model_ref".to_string()),
            "got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced, got {other:?}"),
    }
}

/// A reference through `eval_runs.model_id` (the FK-backed, PK-keyed edge)
/// blocks the delete — again via the typed scan, not the database FK.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_eval_run_pk_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    let pk = pk_of(&cat, "acme/embed-mini").await;
    cat.record_eval_run(&EvalRunRecord {
        eval_run_id: "run-1".into(),
        eval_type: "embedding".into(),
        model_id: Some(pk),
        source_id: "src".into(),
        golden_source: "golden".into(),
        k: Some(10),
        metrics_json: "{}".into(),
        status: "completed".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/embed-mini", None, false, 30)
        .await
        .expect_err("an eval-run reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"eval_runs".to_string()),
            "the blocking edge is reported as eval_runs, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced (not a raw FK violation), got {other:?}"),
    }
}

/// Volume ("scale tier") dimension, kept honest: with ~1000 unrelated models,
/// result tables, and training jobs seeded through the typed verbs, the
/// four-edge referential scan still surfaces the correct typed `ModelReferenced`
/// for the one referenced model. The point is correctness at volume on Postgres
/// `Serializable` — NO wall-clock/latency assertion (that proves no contract and
/// is flaky); the scan's result, not its speed, is the invariant.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_under_volume(backend: BackendKind) {
    const N: usize = 1000;
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    register_source(&cat, "src").await;
    // Seed N base models and one training job each — none referencing the target
    // — so the scan must filter a populated table down to the single real edge.
    for i in 0..N {
        let base_name = format!("acme/base-{i}");
        cat.register_model(register_params(&base_name))
            .await
            .unwrap();
        let base_pk = pk_of(&cat, &base_name).await;
        submit_referencing_job(&cat, &format!("job-{i}"), Some(&base_pk), None).await;
        // An unrelated result table per model, exercising the no-FK name edge at
        // volume without referencing the target.
        cat.create_result_table(CreateResultTableParams {
            writer_id: None,
            lease: None,
            table_name: &format!("acme_rt_{i}"),
            source_id: "src",
            model_id: &base_name,
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "/tmp/p.parquet",
            dimensions: Some(384),
            key_column: Some("id"),
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
            job_attempt: None,
        })
        .await
        .unwrap();
    }

    // The one model under test, referenced through the name-keyed result-table
    // edge — the scan must find it among the N unrelated rows.
    cat.register_model(register_params("acme/target"))
        .await
        .unwrap();
    cat.create_result_table(CreateResultTableParams {
        writer_id: None,
        lease: None,
        table_name: "acme_target_rt",
        source_id: "src",
        model_id: "acme/target",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "/tmp/p.parquet",
        dimensions: Some(384),
        key_column: Some("id"),
        text_columns: None,
        storage_precision: jammi_db::config::StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::backend::now_sortable(),
        job_attempt: None,
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/target", None, false, 30)
        .await
        .expect_err("the referenced model is blocked even among N unrelated rows");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"result_tables".to_string()),
            "the blocking edge is reported as result_tables at volume, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced at volume, got {other:?}"),
    }

    // An unrelated, unreferenced model from the seeded set still deletes cleanly
    // at volume — the scan correctly finds NO edge for it.
    let free_base_pk = pk_of(&cat, "acme/base-0").await;
    submit_referencing_job(&cat, "job-free", Some(&free_base_pk), None).await;
    cat.register_model(register_params("acme/unreferenced"))
        .await
        .unwrap();
    cat.delete_model("acme/unreferenced", None, false, 30)
        .await
        .expect("an unreferenced model deletes cleanly even at volume");
}

/// Tenant B cannot delete tenant A's model: the strict tenant predicate matches
/// no row B owns, so the read path resolves nothing for B and the delete is a
/// NotFound — A's row is untouched.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn cross_tenant_delete_is_not_found(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    cat_a
        .register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();

    let err = cat_b
        .delete_model("acme/embed-mini", None, false, 30)
        .await
        .expect_err("tenant B must not delete tenant A's model");
    assert!(
        matches!(err, JammiError::ModelNotFound { .. }),
        "cross-tenant delete is a model NotFound, got {err:?}"
    );
    assert!(
        cat_a.get_model("acme/embed-mini").await.unwrap().is_some(),
        "tenant A's model is untouched by tenant B's failed delete"
    );
}

/// `if_exists = true` makes deleting an absent model a success no-op.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_absent_with_if_exists_is_noop(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.delete_model("acme/never-registered", None, true, 30)
        .await
        .expect("if_exists makes an absent delete a no-op");
}

/// `if_exists = false` on an absent model is a NotFound.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_absent_without_if_exists_is_not_found(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    let err = cat
        .delete_model("acme/never-registered", None, false, 30)
        .await
        .expect_err("a strict delete of an absent model is NotFound");
    assert!(
        matches!(err, JammiError::ModelNotFound { .. }),
        "absent delete without if_exists is a model NotFound, got {err:?}"
    );
}
