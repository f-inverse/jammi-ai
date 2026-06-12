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
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::{BackendImpl, ChannelId, TenantId};
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

// --- evidence channels (D1) -----------------------------------------------

/// One Utf8 column named `marker`, carrying `value` so the two tenants'
/// otherwise-identically-named channels are distinguishable.
fn channel_spec(name: &str, priority: i32, column: &str) -> ChannelSpec {
    ChannelSpec {
        id: ChannelId::new(name).unwrap(),
        priority,
        columns: vec![ChannelColumn {
            name: column.to_string(),
            data_type: ChannelColumnType::Utf8,
        }],
    }
}

/// HEADLINE: the channel catalog is tenant-scoped. Tenant B must not see, nor
/// collide with, tenant A's channel.
///
/// Before the D1 fix, `evidence_channels.channel_name` was a GLOBAL primary key
/// and `register`/`list` carried no tenant predicate, so tenant A's channel "X"
/// was visible to — and blocked registration by — every other tenant: a
/// cross-tenant read leak plus a false "already exists" collision.
#[tokio::test]
async fn channel_namespace_is_per_tenant() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;

    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    // Tenant A registers channel "X".
    cat_a
        .channels()
        .register(&channel_spec("chan_x", 10, "a_marker"))
        .await
        .unwrap();

    // INVARIANT 1: tenant B's list() must NOT contain A's "chan_x". B sees only the
    // global seed channels (vector/inference/bm25), never another tenant's row.
    let b_names: Vec<String> = cat_b
        .channels()
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.id.as_str().to_string())
        .collect();
    assert!(
        !b_names.contains(&"chan_x".to_string()),
        "CROSS-TENANT READ LEAK: tenant B sees tenant A's channel 'chan_x' in list(): {b_names:?}"
    );

    // INVARIANT 2: tenant B registering "chan_x" must SUCCEED — separate namespace,
    // no false "already exists" collision against A's row.
    cat_b
        .channels()
        .register(&channel_spec("chan_x", 20, "b_marker"))
        .await
        .expect("tenant B must be able to register 'chan_x' despite tenant A owning one");

    // INVARIANT 3: each tenant resolves ITS OWN "chan_x" (own column + priority).
    let a_x = cat_a
        .channels()
        .get(&ChannelId::new("chan_x").unwrap())
        .await
        .unwrap()
        .unwrap();
    let b_x = cat_b
        .channels()
        .get(&ChannelId::new("chan_x").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(a_x.priority, 10);
    assert_eq!(a_x.columns[0].name, "a_marker");
    assert_eq!(b_x.priority, 20);
    assert_eq!(b_x.columns[0].name, "b_marker");

    // INVARIANT 4: re-registering "chan_x" under A still rejects as a duplicate
    // (per-tenant uniqueness is enforced by the UNIQUE constraint even though
    // both backends treat NULL tenants as distinct in that constraint).
    let err = cat_a
        .channels()
        .register(&channel_spec("chan_x", 99, "dup"))
        .await
        .unwrap_err();
    match err {
        JammiError::EvidenceChannel(m) => assert!(
            m.contains("already exists"),
            "expected per-tenant duplicate reject, got: {m}"
        ),
        other => panic!("expected EvidenceChannel(already exists), got {other:?}"),
    }
}

/// `tenant = None` (the global namespace) sees ONLY the global, `tenant_id IS
/// NULL` rows — never any tenant's channel — and a tenant sees the global seed
/// channels (own-shadows-global read precedence, #140).
#[tokio::test]
async fn global_namespace_sees_only_null_tenant_rows() {
    let dir = tempdir().unwrap();
    let base = shared_catalog(dir.path()).await;

    let cat_global = base.pinned_to_tenant(None);
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));

    // Tenant A registers a private channel.
    cat_a
        .channels()
        .register(&channel_spec("private_a", 7, "marker"))
        .await
        .unwrap();

    let global_names: Vec<String> = cat_global
        .channels()
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.id.as_str().to_string())
        .collect();

    // The global view shows the NULL-tenant seed channels and NOT A's private
    // channel.
    assert!(
        global_names.contains(&"vector".to_string()),
        "global list must include the seed 'vector' channel: {global_names:?}"
    );
    assert!(
        !global_names.contains(&"private_a".to_string()),
        "CROSS-TENANT READ LEAK: the global (tenant=None) view sees tenant A's \
         private channel: {global_names:?}"
    );

    // Tenant A still resolves the global seed channels (own-shadows-global): a
    // tenant-bound query merging the dense 'vector' channel must succeed.
    let a_names: Vec<String> = cat_a
        .channels()
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.id.as_str().to_string())
        .collect();
    assert!(
        a_names.contains(&"vector".to_string()),
        "tenant A must resolve the global seed channels: {a_names:?}"
    );
    assert!(
        a_names.contains(&"private_a".to_string()),
        "tenant A must see its own channel: {a_names:?}"
    );
    assert!(
        cat_a
            .channels()
            .get(&ChannelId::new("vector").unwrap())
            .await
            .unwrap()
            .is_some(),
        "tenant A must resolve the global 'vector' seed via get()"
    );
}
