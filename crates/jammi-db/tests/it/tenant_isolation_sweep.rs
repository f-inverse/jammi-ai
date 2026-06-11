//! Adversarial tenant-isolation sweep (slice D) over catalog control-surface
//! repos. Each test binds tenant A, creates a resource, binds tenant B, and
//! asserts B cannot read, overwrite, or otherwise reach A's resource through
//! the catalog repo APIs.
//!
//! The shared backend is opened ONCE and wrapped in per-tenant `Catalog`
//! handles via `Catalog::pinned_to_tenant`, so all handles see the same
//! database rows — the realistic single-process multi-tenant topology.

use std::str::FromStr;

use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::model_repo::RegisterModelParams;
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

fn register_params<'a>(model_id: &'a str, backend: &'a str) -> RegisterModelParams<'a> {
    RegisterModelParams {
        model_id,
        version: 1,
        model_type: "embedding",
        backend,
        task: ModelTask::TextEmbedding,
        base_model_id: None,
        artifact_path: None,
        config_json: None,
    }
}

/// HEADLINE: tenant B must NOT be able to overwrite tenant A's model row.
///
/// `models.model_id` is a GLOBAL primary key (`name::version`), and
/// `register_model` uses `INSERT ... ON CONFLICT(model_id) DO UPDATE` with no
/// tenant predicate in the conflict clause. The write-side guard
/// `assert_tenant_matches` only checks that the *new* row's tenant equals the
/// session binding — it never compares against the *existing* row's owner.
///
/// So tenant B registering a model whose `model_id::version` collides with
/// tenant A's existing row silently UPDATEs A's row (its `backend`, `task`,
/// `model_type`, `metadata`, `artifact_path`), corrupting another tenant's
/// catalog entry. This is a cross-tenant write leak.
#[tokio::test]
async fn tenant_b_cannot_overwrite_tenant_a_model() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;

    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    // Tenant A registers "shared-name::1" with backend "candle".
    cat_a
        .register_model(register_params("shared-name", "candle"))
        .await
        .unwrap();

    // Sanity: A sees its row with backend "candle".
    let a_before = cat_a.get_model("shared-name").await.unwrap().unwrap();
    assert_eq!(a_before.backend, "candle");

    // Tenant B registers the SAME model id+version with a different backend.
    // Under correct isolation this must EITHER (a) create a separate B-owned
    // row, or (b) be rejected — it must NOT touch A's row.
    cat_b
        .register_model(register_params("shared-name", "vllm"))
        .await
        .unwrap();

    // INVARIANT: tenant A's row is unchanged. A's backend is still "candle".
    let a_after = cat_a.get_model("shared-name").await.unwrap().unwrap();
    assert_eq!(
        a_after.backend, "candle",
        "CROSS-TENANT WRITE LEAK: tenant B's register_model overwrote tenant A's \
         model row (backend changed {} -> {})",
        a_before.backend, a_after.backend,
    );
}

/// Companion read-path check: after both tenants "register" the same global
/// model id, tenant B must see ITS OWN model definition, and tenant A must see
/// ITS OWN — not a single shared row whose contents depend on write order.
#[tokio::test]
async fn same_model_id_yields_per_tenant_rows() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;

    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    cat_a
        .register_model(register_params("collide", "candle"))
        .await
        .unwrap();
    cat_b
        .register_model(register_params("collide", "vllm"))
        .await
        .unwrap();

    let a = cat_a.get_model("collide").await.unwrap().unwrap();
    let b = cat_b.get_model("collide").await.unwrap().unwrap();

    // Each tenant should observe the backend IT registered.
    assert_eq!(
        a.backend, "candle",
        "tenant A must see its own model backend"
    );
    assert_eq!(b.backend, "vllm", "tenant B must see its own model backend");
}
