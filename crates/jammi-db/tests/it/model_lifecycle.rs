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

use std::str::FromStr;
use std::time::Duration;

use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::{BackendImpl, TenantId};
use tempfile::tempdir;

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
}

/// Open one shared SQLite backend and return an unscoped base catalog over it.
async fn shared_catalog(dir: &std::path::Path) -> Catalog {
    let backend = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
    let catalog = Catalog::from_backend(BackendImpl::Sqlite(backend));
    catalog.backend_arc().migrate().await.unwrap();
    catalog
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

/// Resolve the catalog PK an FK-backed reference binds to.
async fn pk_of(cat: &Catalog, name: &str) -> String {
    cat.get_model(name).await.unwrap().unwrap().catalog_pk
}

/// HEADLINE: an unreferenced model deletes cleanly and is then absent.
#[tokio::test]
async fn delete_unreferenced_model_succeeds() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
#[tokio::test]
async fn delete_blocked_by_result_table_name_edge() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
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
#[tokio::test]
async fn delete_blocked_by_training_output_name_edge() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
#[tokio::test]
async fn delete_blocked_by_training_base_pk_edge() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
#[tokio::test]
async fn delete_blocked_by_eval_run_pk_edge() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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

/// Tenant B cannot delete tenant A's model: the strict tenant predicate matches
/// no row B owns, so the read path resolves nothing for B and the delete is a
/// NotFound — A's row is untouched.
#[tokio::test]
async fn cross_tenant_delete_is_not_found() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
        matches!(err, JammiError::Model { .. }),
        "cross-tenant delete is a model NotFound, got {err:?}"
    );
    assert!(
        cat_a.get_model("acme/embed-mini").await.unwrap().is_some(),
        "tenant A's model is untouched by tenant B's failed delete"
    );
}

/// `if_exists = true` makes deleting an absent model a success no-op.
#[tokio::test]
async fn delete_absent_with_if_exists_is_noop() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.delete_model("acme/never-registered", None, true)
        .await
        .expect("if_exists makes an absent delete a no-op");
}

/// `if_exists = false` on an absent model is a NotFound.
#[tokio::test]
async fn delete_absent_without_if_exists_is_not_found() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    let err = cat
        .delete_model("acme/never-registered", None, false)
        .await
        .expect_err("a strict delete of an absent model is NotFound");
    assert!(
        matches!(err, JammiError::Model { .. }),
        "absent delete without if_exists is a model NotFound, got {err:?}"
    );
}

/// HEADLINE: promote sets the flag; the projected `promoted` bool reads true.
#[tokio::test]
async fn promote_sets_the_flag() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
#[tokio::test]
async fn promoting_a_version_demotes_the_prior() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
/// promoted under `idx_models_promoted` collides.
#[tokio::test]
async fn double_promote_rejected_by_partial_index_tenant_scope() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
    let direct = direct_promote(&cat, Some(tenant_a()), &pk_v2).await;
    assert!(
        direct.is_err(),
        "a direct second promote in the same (tenant, name) must violate the partial unique index"
    );
}

/// The GLOBAL-namespace partial index (`idx_models_promoted_global`, on `name`
/// WHERE `tenant_id IS NULL`) rejects a double-promote among GLOBAL rows — the
/// gap the composite `(tenant_id, name)` index leaves, since SQL treats the NULL
/// tenants as distinct.
#[tokio::test]
async fn double_promote_rejected_by_partial_index_global_scope() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
    let direct = direct_promote(&global, None, &pk_v2).await;
    assert!(
        direct.is_err(),
        "a direct second GLOBAL promote must violate the global partial unique index"
    );
}

/// Tenant B cannot promote tenant A's model: the strict tenant predicate matches
/// no row B owns, so it is a NotFound.
#[tokio::test]
async fn cross_tenant_promote_is_not_found() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
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
        matches!(err, JammiError::Model { .. }),
        "cross-tenant promote is a model NotFound, got {err:?}"
    );
}

/// Set `promoted_at` on a row directly (no demote of a sibling) to exercise the
/// partial unique index in isolation — the backstop a `promote_model` skips.
/// `tenant` is unused beyond documenting which scope's row the PK belongs to;
/// the index is keyed by the row's stored `tenant_id`, not the session.
async fn direct_promote(
    cat: &Catalog,
    _tenant: Option<TenantId>,
    pk: &str,
) -> Result<(), JammiError> {
    use jammi_db::catalog::backend::{SqlValue, TxOptions};
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
