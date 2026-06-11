//! Model RETIRE lifecycle: the soft state transition and its visibility split.
//!
//! The load-bearing rule under test is the three-way visibility split:
//! `get_model` still returns a retired model (it is the reference-resolution /
//! provenance path), `list_models` hides it (active listing), and the strict
//! tenant scope means a tenant retires only its own row — never a peer's, never
//! a GLOBAL model from a tenant session.

use std::str::FromStr;

use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::ModelStatus;
use jammi_db::catalog::Catalog;
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

fn retired() -> String {
    ModelStatus::Retired.to_string()
}

/// HEADLINE: retire hides the model from `list_models` but `get_model` still
/// returns it — the reference-resolution path is preserved.
#[tokio::test]
async fn retire_hides_from_list_but_get_model_still_resolves() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();

    // Before retire: present in the active listing.
    assert!(
        cat.list_models()
            .await
            .unwrap()
            .iter()
            .any(|m| m.model_id == "acme/embed-mini"),
        "registered model must be in the active listing before retire"
    );

    cat.retire_model("acme/embed-mini", None).await.unwrap();

    // After retire: GONE from list_models.
    assert!(
        !cat.list_models()
            .await
            .unwrap()
            .iter()
            .any(|m| m.model_id == "acme/embed-mini"),
        "retired model must be hidden from the active listing"
    );

    // But get_model STILL returns it — the FK / provenance resolution path.
    let resolved = cat
        .get_model("acme/embed-mini")
        .await
        .unwrap()
        .expect("get_model must still resolve a retired model");
    assert_eq!(
        resolved.status,
        retired(),
        "the resolved record carries the retired status"
    );
}

/// Idempotent: retiring an already-retired model succeeds as a no-op.
#[tokio::test]
async fn retire_is_idempotent() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    cat.retire_model("acme/embed-mini", None).await.unwrap();
    cat.retire_model("acme/embed-mini", None)
        .await
        .expect("retiring an already-retired model is a success no-op");

    assert_eq!(
        cat.get_model("acme/embed-mini")
            .await
            .unwrap()
            .unwrap()
            .status,
        retired()
    );
}

/// A second tenant CANNOT retire tenant A's model: the retire is filtered on a
/// STRICT `tenant_id = $current`, so tenant B asking to retire A's row gets a
/// NotFound and A's row is untouched.
#[tokio::test]
async fn second_tenant_cannot_retire_first_tenants_model() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    cat_a
        .register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();

    // Tenant B cannot even see A's tenant-scoped model, so retire is NotFound.
    let err = cat_b
        .retire_model("acme/embed-mini", None)
        .await
        .expect_err("tenant B must not retire tenant A's model");
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "cross-tenant retire is reported as a model NotFound, got: {err:?}"
    );

    // A's row is untouched — still active for A.
    assert_eq!(
        cat_a
            .get_model("acme/embed-mini")
            .await
            .unwrap()
            .unwrap()
            .status,
        ModelStatus::Registered.to_string(),
        "tenant A's model must be unchanged by tenant B's failed retire"
    );
}

/// Retiring a GLOBAL (`tenant_id IS NULL`) model from a tenant session is
/// rejected: the read path can surface the global row, but the strict retire
/// predicate matches no tenant-owned row, so it is NotFound — a tenant may not
/// retire a shared model.
#[tokio::test]
async fn tenant_cannot_retire_global_model() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;

    // Register a GLOBAL model on the unscoped (tenant = None) catalog.
    let global = base.pinned_to_tenant(None);
    global
        .register_model(register_params("shared/global-embed"))
        .await
        .unwrap();

    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    // The tenant can SEE the global model through the read path…
    assert!(
        cat_a
            .get_model("shared/global-embed")
            .await
            .unwrap()
            .is_some(),
        "a tenant can resolve a GLOBAL model through get_model"
    );
    // …but it cannot RETIRE it.
    let err = cat_a
        .retire_model("shared/global-embed", None)
        .await
        .expect_err("a tenant must not retire a GLOBAL model");
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "retiring a GLOBAL model from a tenant session is NotFound, got: {err:?}"
    );

    // The global row is still active.
    assert_eq!(
        global
            .get_model("shared/global-embed")
            .await
            .unwrap()
            .unwrap()
            .status,
        ModelStatus::Registered.to_string()
    );
}

/// An eval run referencing the retired model still resolves the model's catalog
/// PK through `get_model` — the FK-resolution path the retire must not break.
#[tokio::test]
async fn eval_run_referencing_retired_model_still_resolves_via_get_model() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    cat.register_model(register_params("acme/embed-mini"))
        .await
        .unwrap();
    // Resolve the catalog PK the eval run binds to (an eval run records the
    // model's PK, not its bare name).
    let model_pk = cat
        .get_model("acme/embed-mini")
        .await
        .unwrap()
        .unwrap()
        .catalog_pk;

    cat.record_eval_run(&EvalRunRecord {
        eval_run_id: "run-1".into(),
        eval_type: "embedding".into(),
        model_id: Some(model_pk.clone()),
        source_id: "src".into(),
        golden_source: "golden".into(),
        k: Some(10),
        metrics_json: "{}".into(),
        status: "completed".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    })
    .await
    .expect("record_eval_run with a valid model FK");

    // Now retire the model.
    cat.retire_model("acme/embed-mini", None).await.unwrap();

    // The eval run's referenced model still resolves through get_model — the
    // retired row is intact and the FK still points at a real row.
    let resolved = cat
        .get_model("acme/embed-mini")
        .await
        .unwrap()
        .expect("a retired model an eval run references must still resolve");
    assert_eq!(resolved.catalog_pk, model_pk);
    assert_eq!(resolved.status, retired());
}
