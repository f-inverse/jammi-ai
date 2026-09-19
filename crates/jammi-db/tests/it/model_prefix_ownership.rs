//! I5 (#547), restated by the wave-5 pressure round (F4): `delete_model`
//! deletes a catalog ROW only — it carries no owner-vs-reuser refusal, and
//! needs none, because byte safety over a prefix TWO `models` rows name is
//! already GLOBAL: `ResultStore::prefix_is_referenced` /
//! `Catalog::count_models_naming_prefix_all_tenants` counts across every
//! tenant (and untenanted rows) regardless of which tenant's session is
//! asking. This file is the executed oracle #547 asks for in place of a
//! migration and an owner-refusal: two `models` rows naming ONE artifact
//! prefix, deleted in BOTH orders, same-tenant and cross-tenant, proving
//! U3's acceptance (a) — "delete one row, the other loads; delete both, the
//! sweep reaps" — holds regardless of which row is called the "owner" and
//! regardless of deletion order, with NO tenant-bound call ever learning the
//! other row exists (delete_model's own referential scan never queries
//! `models` at all — see `model_repo.rs::delete_model`'s doc — so there is
//! no channel for it to disclose anything about a peer row through).

use std::str::FromStr;

use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_db::TenantId;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;
use test_case::test_case;

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8f1a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8f1b").unwrap()
}

/// One shared artifact prefix two `models` rows will name — computed
/// through [`jammi_db::store::ArtifactStore::prefix_url`] (never a
/// hand-built string) so it is genuinely under `store`'s own models root,
/// exactly as `delete_artifact_prefix`'s I1 `debug_assert!` requires. Never
/// resolved to real bytes; this file exercises the CATALOG-level reference
/// count only, the same predicate `ResultStore::delete_unreferenced_prefix`
/// consults in production.
fn shared_prefix(store: &ResultStore) -> StorageUrl {
    store
        .artifact_store()
        .prefix_url(None, &["job-owner", "worker-1", "0"])
        .unwrap()
}

fn register_params<'a>(model_id: &'a str, artifact_path: &'a str) -> RegisterModelParams<'a> {
    RegisterModelParams {
        model_id,
        version: 1,
        model_type: "lora",
        backend: "candle",
        task: ModelTask::TextEmbedding,
        base_model_id: None,
        artifact_path: Some(artifact_path),
        config_json: None,
    }
}

/// Register two rows naming the SAME `artifact_path`, delete them in the
/// order `(first, second)`, and assert the shape F4 pins at every step:
/// after `first` is gone, `second` still `get_model`s and the prefix is
/// still referenced exactly once (never reapable); after `second` is also
/// gone, the prefix is unreferenced (the sweep would now reap it).
/// `first_cat`/`second_cat` are the tenant-pinned catalogs each row's OWN
/// delete runs under — same value for the same-tenant variant, different
/// values (tenant A, tenant B) for the cross-tenant variant.
#[allow(clippy::too_many_arguments)]
async fn both_orders_hold(
    store: &ResultStore,
    base: &Catalog,
    first_cat: &Catalog,
    first_id: &str,
    second_cat: &Catalog,
    second_id: &str,
) {
    let prefix = shared_prefix(store);
    first_cat
        .register_model(register_params(first_id, prefix.as_str()))
        .await
        .unwrap();
    second_cat
        .register_model(register_params(second_id, prefix.as_str()))
        .await
        .unwrap();

    // Both rows are live: the prefix is referenced twice.
    assert_eq!(
        store.prefix_is_referenced(&prefix).await.unwrap(),
        2,
        "two rows naming the same prefix must both count"
    );
    assert!(store.delete_unreferenced_prefix(&prefix).await.is_err_and(
        |e| matches!(e, jammi_db::error::JammiError::Storage(
                jammi_db::storage::StorageError::Referenced { count, .. }
            ) if count == 2)
    ));

    // Delete `first` — never refused: `delete_model` scans no `models` edge
    // at all (model_repo.rs's own doc: `result_tables`/`jobs`/`eval_runs`
    // only), so a peer row naming the SAME prefix is never even consulted.
    first_cat
        .delete_model(first_id, None, false, 30)
        .await
        .unwrap();

    // `second` still resolves — untouched by `first`'s delete.
    let still_loads = base
        .pinned_to_tenant(second_cat.current_tenant())
        .get_model(second_id)
        .await
        .unwrap();
    assert!(
        still_loads.is_some(),
        "the surviving row must still load after the OTHER row naming the same prefix is gone"
    );
    assert_eq!(
        still_loads.unwrap().artifact_path.as_deref(),
        Some(prefix.as_str())
    );

    // The prefix is still referenced exactly once — the sweep must not reap
    // it while `second` is alive.
    assert_eq!(
        store.prefix_is_referenced(&prefix).await.unwrap(),
        1,
        "one surviving row must keep the prefix reachable"
    );
    assert!(matches!(
        store.delete_unreferenced_prefix(&prefix).await,
        Err(jammi_db::error::JammiError::Storage(
            jammi_db::storage::StorageError::Referenced { count: 1, .. }
        ))
    ));

    // Delete `second` too — now both rows are gone.
    second_cat
        .delete_model(second_id, None, false, 30)
        .await
        .unwrap();
    assert_eq!(
        store.prefix_is_referenced(&prefix).await.unwrap(),
        0,
        "once BOTH rows are gone the prefix must be unreferenced"
    );
    // The sweep would now reap it — `delete_unreferenced_prefix` itself
    // best-effort-deletes a `file://` prefix with no manifest published
    // (never having called `put_artifact` here), which is a documented
    // no-op success, not a failure — proving the REFERENCE gate, not the
    // byte-delete, is what changed.
    store.delete_unreferenced_prefix(&prefix).await.unwrap();
}

/// HEADLINE (same tenant): owner-first order. Two rows, same tenant, same
/// prefix; delete the FIRST-REGISTERED row first.
#[test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn same_tenant_two_rows_one_prefix_owner_first(
    backend: jammi_db::catalog::backend::BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let base = std::sync::Arc::clone(session.catalog());
    reset_models(&base).await;
    let store = ResultStore::new(
        dir.path(),
        std::sync::Arc::clone(&base),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    both_orders_hold(&store, &base, &cat, "i5-same-owner", &cat, "i5-same-reuser").await;
}

/// Same fixture, REVERSED delete order: the reuser first, the owner second
/// — U3's acceptance (a) must hold in both orders, not just one.
#[test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn same_tenant_two_rows_one_prefix_reuser_first(
    backend: jammi_db::catalog::backend::BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let base = std::sync::Arc::clone(session.catalog());
    reset_models(&base).await;
    let store = ResultStore::new(
        dir.path(),
        std::sync::Arc::clone(&base),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let cat = base.pinned_to_tenant(Some(tenant_a()));

    both_orders_hold(&store, &base, &cat, "i5-rev-reuser", &cat, "i5-rev-owner").await;
}

/// HEADLINE (cross-tenant): tenant A's row and tenant B's row name the SAME
/// prefix (the cache-hit fan-out shape `find_models_by_definition`'s own doc
/// describes: a `NULL`-tenant global row's prefix, reused by a second,
/// tenant-scoped row) — global-reachability byte safety, never a
/// tenant-scoped view of it. Both delete orders.
#[test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn cross_tenant_two_rows_one_prefix_both_orders(
    backend: jammi_db::catalog::backend::BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let base = std::sync::Arc::clone(session.catalog());
    reset_models(&base).await;
    let store = ResultStore::new(
        dir.path(),
        std::sync::Arc::clone(&base),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    // Order 1: tenant A's row first.
    both_orders_hold(&store, &base, &cat_a, "i5-cross-a1", &cat_b, "i5-cross-b1").await;
    // Order 2 (fresh rows, reversed caller order): tenant B's row first.
    both_orders_hold(&store, &base, &cat_b, "i5-cross-b2", &cat_a, "i5-cross-a2").await;
}

/// The disclosure half of F4's property, stated as its own oracle rather
/// than folded into `both_orders_hold`'s assertions: a tenant-bound
/// `delete_model` NEVER queries the `models` table for a peer row at all
/// (`model_repo.rs::delete_model`'s own doc: the referential scan is over
/// `result_tables`/`jobs`/`eval_runs` only) — so tenant B's delete of its
/// own row cannot even in principle learn tenant A's row exists. Proven by
/// mutation: if `delete_model` were changed to consult
/// `count_models_naming_prefix_all_tenants` and refuse while a peer exists
/// (the owner-refusal #547's design round considered and F4 rejects), THIS
/// test's `tenant_b`-scoped delete would start failing with `Referenced`
/// instead of succeeding — reddening the assertion below. Run the mutation
/// by hand: temporarily add such a refusal to `delete_model` and re-run;
/// the assertion changes from `Ok` to `Err`, confirming today's absence of
/// an owner-refusal is exactly what this test pins.
#[test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_tenant_bound_delete_never_refuses_on_a_peer_tenants_reuse(
    backend: jammi_db::catalog::backend::BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let base = std::sync::Arc::clone(session.catalog());
    reset_models(&base).await;
    let store = ResultStore::new(
        dir.path(),
        std::sync::Arc::clone(&base),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let prefix = shared_prefix(&store);
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));

    cat_a
        .register_model(register_params("i5-disclosure-owner", prefix.as_str()))
        .await
        .unwrap();
    cat_b
        .register_model(register_params("i5-disclosure-reuser", prefix.as_str()))
        .await
        .unwrap();

    // Tenant A deletes its own row while tenant B's row is very much alive
    // and naming the exact same prefix — never refused, never told B's row
    // exists.
    cat_a
        .delete_model("i5-disclosure-owner", None, false, 30)
        .await
        .expect(
            "a tenant-bound delete must never refuse merely because a PEER tenant's row reuses \
             the same prefix",
        );
}

async fn reset_models(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(jammi_db::catalog::backend::TxOptions::default(), |tx| {
            Box::pin(async move {
                for table in ["eval_runs", "jobs", "result_tables", "models"] {
                    tx.execute(&format!("DELETE FROM {table}"), &[]).await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();
}
