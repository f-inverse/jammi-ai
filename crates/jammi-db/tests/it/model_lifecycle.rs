//! Model DELETE and PROMOTE lifecycle on the `models` catalog table.
//!
//! DELETE is the hard counterpart of RETIRE: it removes the row, so it is
//! refused while any reference still points at the model. The load-bearing rule
//! is the four-edge referential scan, each edge keyed by what it actually stores
//! — the model NAME for the two no-FK edges (`result_tables.model_id`,
//! `training_jobs.output_model_id`) and the catalog PK for the two FK-backed
//! ones (`training_jobs.base_model_id`, `eval_runs.model_id`). A pk-keyed scan
//! would silently miss the two name-keyed edges, so each is exercised directly.
//!
//! PROMOTE is a single nullable flag: promoting a version demotes the prior one,
//! so at most one row per `(tenant, name)` is promoted; the partial unique
//! indexes back that even against a direct double-promote. Both verbs are
//! strictly tenant-scoped — a tenant touches only a row it owns.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` + `cfg_attr`.
//! The SQLite lane is always generated; the Postgres lane is generated only when
//! the `live-postgres-tests` feature is on, and skips at runtime when
//! `JAMMI_TEST_PG_URL` is unset (an early return, never `#[ignore]`). The
//! Postgres lane is where the contract bites hardest: the partial unique indexes
//! are real `CREATE UNIQUE INDEX … WHERE …` constraints a direct double-promote
//! violates with a typed `BackendError::Constraint`, the four-edge scan runs
//! under PG `Serializable` and still surfaces the typed `ModelReferenced` rather
//! than a raw FK error, and migration 021's two partial indexes are asserted to
//! exist via `pg_indexes` (the PG analogue of the SQLite `sqlite_master` check in
//! `migrations.rs`). On the Postgres lane that one catalog DB is shared across
//! the whole run, so each test first clears the referential tables via
//! [`reset_catalog`]; CI's `test-pg` job runs the lane with `--test-threads=1`,
//! so the reset-then-populate sequence cannot race a sibling test.

use std::str::FromStr;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendError, BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
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
                for table in [
                    "eval_runs",
                    "training_jobs",
                    "result_tables",
                    "models",
                    "sources",
                ] {
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
    cat.delete_model("acme/embed-mini", None, false)
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
        table_name: "acme_embeddings",
        source_id: "src",
        model_id: "acme/embed-mini",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "/tmp/p.parquet",
        index_path: None,
        dimensions: Some(384),
        key_column: Some("id"),
        text_columns: None,
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/embed-mini", None, false)
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

/// A reference through `training_jobs.output_model_id` (the other NAME-keyed,
/// no-FK edge) blocks the delete — set on `finalize_training_job`, matched by
/// the model NAME.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_training_output_name_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    // A base model the job's FK points at, and the output model whose NAME the
    // finalize records under `output_model_id`.
    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    cat.register_model(register_params("acme/tuned"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;

    cat.create_training_job(CreateTrainingJobParams {
        job_id: "job-1",
        base_model_id: &base_pk,
        training_source: "src.csv",
        loss_type: "contrastive",
        hyperparams: "{}",
        kind: "fine_tune",
        training_spec: "{}",
    })
    .await
    .unwrap();
    let lease = Duration::from_secs(30);
    cat.claim_next_training_job("worker-a", lease)
        .await
        .unwrap()
        .expect("the queued job is claimable");
    // output_model_id is set to the output model NAME on finalize.
    let finalized = cat
        .finalize_training_job("job-1", "worker-a", "acme/tuned", "/tmp/out", None)
        .await
        .unwrap();
    assert!(finalized, "the lease holder finalizes the job");

    let err = cat
        .delete_model("acme/tuned", None, false)
        .await
        .expect_err("an output-model reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"training_jobs.output_model_id".to_string()),
            "the blocking edge is reported as training_jobs.output_model_id, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced, got {other:?}"),
    }
}

/// A reference through `training_jobs.base_model_id` (the FK-backed, PK-keyed
/// edge) blocks the delete. The scan — not the database FK — raises the typed
/// error, so it never leaks as a raw constraint violation.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn delete_blocked_by_training_base_pk_edge(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/base"))
        .await
        .unwrap();
    let base_pk = pk_of(&cat, "acme/base").await;

    cat.create_training_job(CreateTrainingJobParams {
        job_id: "job-1",
        base_model_id: &base_pk,
        training_source: "src.csv",
        loss_type: "contrastive",
        hyperparams: "{}",
        kind: "fine_tune",
        training_spec: "{}",
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/base", None, false)
        .await
        .expect_err("a base-model reference must block the delete");
    match err {
        JammiError::ModelReferenced { referenced_by, .. } => assert!(
            referenced_by.contains(&"training_jobs.base_model_id".to_string()),
            "the blocking edge is reported as training_jobs.base_model_id, got {referenced_by:?}"
        ),
        other => panic!("expected ModelReferenced (not a raw FK violation), got {other:?}"),
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
        .delete_model("acme/embed-mini", None, false)
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
        cat.create_training_job(CreateTrainingJobParams {
            job_id: &format!("job-{i}"),
            base_model_id: &base_pk,
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
        // An unrelated result table per model, exercising the no-FK name edge at
        // volume without referencing the target.
        cat.create_result_table(CreateResultTableParams {
            table_name: &format!("acme_rt_{i}"),
            source_id: "src",
            model_id: &base_name,
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "/tmp/p.parquet",
            index_path: None,
            dimensions: Some(384),
            key_column: Some("id"),
            text_columns: None,
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
        table_name: "acme_target_rt",
        source_id: "src",
        model_id: "acme/target",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "/tmp/p.parquet",
        index_path: None,
        dimensions: Some(384),
        key_column: Some("id"),
        text_columns: None,
    })
    .await
    .unwrap();

    let err = cat
        .delete_model("acme/target", None, false)
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
    cat.create_training_job(CreateTrainingJobParams {
        job_id: "job-free",
        base_model_id: &pk_of(&cat, "acme/base-0").await,
        training_source: "src.csv",
        loss_type: "contrastive",
        hyperparams: "{}",
        kind: "fine_tune",
        training_spec: "{}",
    })
    .await
    .unwrap();
    cat.register_model(register_params("acme/unreferenced"))
        .await
        .unwrap();
    cat.delete_model("acme/unreferenced", None, false)
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
        .delete_model("acme/embed-mini", None, false)
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

    cat.delete_model("acme/never-registered", None, true)
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
        .delete_model("acme/never-registered", None, false)
        .await
        .expect_err("a strict delete of an absent model is NotFound");
    assert!(
        matches!(err, JammiError::ModelNotFound { .. }),
        "absent delete without if_exists is a model NotFound, got {err:?}"
    );
}

/// HEADLINE: promote sets the flag; the projected `promoted` bool reads true.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn promote_sets_the_flag(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    assert!(
        cat.get_model("acme/embed-mini")
            .await
            .unwrap()
            .unwrap()
            .promoted_at
            .is_none(),
        "a freshly registered model is not promoted"
    );

    cat.promote_model("acme/embed-mini", None).await.unwrap();
    assert!(
        cat.get_model("acme/embed-mini")
            .await
            .unwrap()
            .unwrap()
            .promoted_at
            .is_some(),
        "promote sets the promoted flag"
    );
}

/// Promoting v2 demotes v1: at most one version per `(tenant, name)` is promoted.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn promoting_a_version_demotes_the_prior(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    let v1 = RegisterModelParams {
        version: 1,
        ..register_params("acme/embed-mini")
    };
    let v2 = RegisterModelParams {
        version: 2,
        ..register_params("acme/embed-mini")
    };
    cat.register_model(v1).await.unwrap();
    cat.register_model(v2).await.unwrap();

    cat.promote_model("acme/embed-mini", Some(1)).await.unwrap();
    assert!(
        cat.get_model_version("acme/embed-mini", 1)
            .await
            .unwrap()
            .unwrap()
            .promoted_at
            .is_some(),
        "v1 is promoted"
    );

    cat.promote_model("acme/embed-mini", Some(2)).await.unwrap();
    assert!(
        cat.get_model_version("acme/embed-mini", 2)
            .await
            .unwrap()
            .unwrap()
            .promoted_at
            .is_some(),
        "v2 is now promoted"
    );
    assert!(
        cat.get_model_version("acme/embed-mini", 1)
            .await
            .unwrap()
            .unwrap()
            .promoted_at
            .is_none(),
        "promoting v2 demoted v1 — exactly one version is promoted"
    );
}

/// The partial unique index rejects a direct double-promote (one that skips the
/// in-verb demote) in the TENANT-scoped namespace — a second tenant-owned row
/// promoted under `idx_models_promoted` collides. On the Postgres lane the
/// collision surfaces as the typed `BackendError::Constraint` naming the `models`
/// table; the SQLite lane raises the equivalent unique-violation, so both lanes
/// assert the same typed constraint error rather than a bare `is_err()`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn double_promote_rejected_by_partial_index_tenant_scope(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    let v1 = RegisterModelParams {
        version: 1,
        ..register_params("acme/embed-mini")
    };
    let v2 = RegisterModelParams {
        version: 2,
        ..register_params("acme/embed-mini")
    };
    cat.register_model(v1).await.unwrap();
    cat.register_model(v2).await.unwrap();

    cat.promote_model("acme/embed-mini", Some(1)).await.unwrap();
    // A raw UPDATE that sets v2 promoted WITHOUT demoting v1 must hit the
    // partial unique index `idx_models_promoted(tenant_id, name)`.
    let pk_v2 = cat
        .get_model_version("acme/embed-mini", 2)
        .await
        .unwrap()
        .unwrap()
        .catalog_pk;
    assert_constraint_violation(
        direct_promote(&cat, &pk_v2).await,
        "a direct second promote in the same (tenant, name) must violate the partial unique index",
    );
}

/// The GLOBAL-namespace partial index (`idx_models_promoted_global`, on `name`
/// WHERE `tenant_id IS NULL`) rejects a double-promote among GLOBAL rows — the
/// gap the composite `(tenant_id, name)` index leaves, since SQL treats the NULL
/// tenants as distinct.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn double_promote_rejected_by_partial_index_global_scope(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let global = base.pinned_to_tenant(None);

    let v1 = RegisterModelParams {
        version: 1,
        ..register_params("shared/global-embed")
    };
    let v2 = RegisterModelParams {
        version: 2,
        ..register_params("shared/global-embed")
    };
    global.register_model(v1).await.unwrap();
    global.register_model(v2).await.unwrap();

    global
        .promote_model("shared/global-embed", Some(1))
        .await
        .unwrap();
    let pk_v2 = global
        .get_model_version("shared/global-embed", 2)
        .await
        .unwrap()
        .unwrap()
        .catalog_pk;
    assert_constraint_violation(
        direct_promote(&global, &pk_v2).await,
        "a direct second GLOBAL promote must violate the global partial unique index",
    );
}

/// Tenant B cannot promote tenant A's model: the strict tenant predicate matches
/// no row B owns, so it is a NotFound.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn cross_tenant_promote_is_not_found(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    cat_a
        .register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();

    let err = cat_b
        .promote_model("acme/embed-mini", None)
        .await
        .expect_err("tenant B must not promote tenant A's model");
    assert!(
        matches!(err, JammiError::ModelNotFound { .. }),
        "cross-tenant promote is a model NotFound, got {err:?}"
    );
}

/// Migration-chain assertion: after the full chain (through 021) the two partial
/// unique indexes that back the at-most-one-promoted invariant exist on `models`.
/// SQLite reads them from `sqlite_master`; Postgres from `pg_indexes` — the PG
/// analogue of the SQLite index checks in `migrations.rs`. This is the
/// schema-shape complement to the behavioural double-promote tests above: the
/// indexes are present, not merely inferred from a rejected write.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn migration_021_partial_indexes_exist(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, base) = lifecycle_catalog!(backend, dir.path());

    for idx in ["idx_models_promoted", "idx_models_promoted_global"] {
        let exists = index_exists_on_models(&base, backend, idx).await;
        assert!(
            exists,
            "partial unique index '{idx}' must exist on `models` after migration 021 on {backend:?}"
        );
    }
}

/// Whether a named index exists on `models`, read from the backend's catalog
/// metadata — `sqlite_master` on SQLite, `pg_indexes` on Postgres.
async fn index_exists_on_models(cat: &Catalog, backend: BackendKind, index: &str) -> bool {
    // COUNT(*) so the result column is a bigint on Postgres (read as i64);
    // a bare `SELECT 1` is int4 there and fails the i64 read (SQLite's flexible
    // typing tolerated it, Postgres does not).
    let sql = match backend {
        BackendKind::Sqlite => {
            "SELECT COUNT(*) AS n FROM sqlite_master \
             WHERE type = 'index' AND name = $1 AND tbl_name = 'models'"
        }
        BackendKind::Postgres => {
            "SELECT COUNT(*) AS n FROM pg_indexes \
             WHERE indexname = $1 AND tablename = 'models'"
        }
    };
    let index = index.to_string();
    cat.backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let counts: Vec<i64> = tx
                        .query(sql, &[SqlValue::TextOwned(index)], |row| {
                            row.get::<i64>("n")
                        })
                        .await?;
                    Ok(counts.first().copied().unwrap_or(0) > 0)
                })
            },
        )
        .await
        .unwrap()
}

/// Assert a `direct_promote` outcome is the typed unique-constraint violation
/// the partial index raises (classified from the backend's unique-violation and
/// wrapped as `JammiError::BackendDriver`), not some other failure. Postgres
/// populates the offending table on the `DatabaseError`, so the `models` table
/// is asserted there; SQLite's driver does not surface a table name on a unique
/// violation (it reports `<unknown>`), so the table is asserted only when the
/// backend reports it — the variant is the load-bearing invariant on both lanes.
fn assert_constraint_violation(result: Result<(), JammiError>, context: &str) {
    match result {
        Err(JammiError::BackendDriver(BackendError::Constraint { table, .. })) => {
            if table != "<unknown>" {
                assert_eq!(
                    table, "models",
                    "{context}: a reported constraint table must name `models`"
                );
            }
        }
        Err(other) => panic!("{context}: expected a constraint violation, got {other:?}"),
        Ok(()) => panic!("{context}: the direct double-promote was accepted"),
    }
}

/// Set `promoted_at` on a row directly (no demote of a sibling) to exercise the
/// partial unique index in isolation — the backstop a `promote_model` skips. The
/// index is keyed by the row's stored `tenant_id`, not the session.
async fn direct_promote(cat: &Catalog, pk: &str) -> Result<(), JammiError> {
    let pk = pk.to_string();
    cat.backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE models SET promoted_at = CAST(CURRENT_TIMESTAMP AS TEXT) \
                     WHERE model_id = $1",
                    &[SqlValue::TextOwned(pk)],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .map_err(JammiError::from)
}
