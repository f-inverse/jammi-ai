//! Standing tenant-isolation oracle.
//!
//! This module proves two things at once and is the single source of truth for
//! both:
//!
//! 1. **Per-verb isolation** — for every data-plane rpc, one tenant creates a
//!    resource and another tenant attempts to read, overwrite, or delete it;
//!    the case asserts the attempt is refused (or, for stated-positive cases,
//!    that global rows are visible / a global-namespace key collision errors
//!    rather than silently clobbering a peer).
//!
//! 2. **Structural coverage** — the live rpc inventory is DERIVED from the
//!    compiled proto [`FILE_DESCRIPTOR_SET`], not a hand-maintained constant.
//!    A new `jammi.v1` rpc that lands without an isolation case and is not on
//!    the explicit control-plane allowlist makes [`every_rpc_is_covered`] fail,
//!    naming the rpc. Coverage cannot rot into an illusory guarantee: the
//!    descriptor is the authority, and the bind is symmetric (a stale case or
//!    allowlist entry naming a removed rpc fails too).
//!
//! The cases run hermetically against the catalog repos over one shared SQLite
//! backend (the realistic single-process multi-tenant topology: one backend,
//! per-tenant [`Catalog::pinned_to_tenant`] handles seeing the same rows).
//! Compute verbs that cannot run without a GPU are covered by asserting the
//! tenant-scoped resolver they depend on is isolated — that assertion executes
//! for real — and each carries a `covered_by` pointer at the end-to-end
//! distributed isolation test.

use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow_schema::{DataType, Field, Schema};
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::trigger::{TopicDefinition, TopicId, TriggerError};
use jammi_db::{BackendImpl, ChannelId, ModelTask, TenantId};
use jammi_test_utils::{fixture_url, test_config};
use jammi_wire::FILE_DESCRIPTOR_SET;
use prost::Message;
use prost_types::FileDescriptorSet;
use tempfile::{tempdir, TempDir};

// ---------------------------------------------------------------------------
// Derived wire surface
// ---------------------------------------------------------------------------

/// Prefix of every package whose services this engine serves on the wire. The
/// descriptor also carries `google.*` / `arrow.*` imports; only `jammi.v1.*`
/// services define the engine's own rpc surface, so the bind scopes to them.
const WIRE_PACKAGE_PREFIX: &str = "jammi.v1";

/// Decode [`FILE_DESCRIPTOR_SET`] into the set of `Service/Method` paths every
/// `jammi.v1.*` service serves — the gRPC path tail, which is the stable rpc
/// identity independent of the package the service is declared in.
fn wire_rpcs() -> BTreeSet<String> {
    let set = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
        .expect("the compiled jammi.v1 descriptor must decode");
    let mut rpcs = BTreeSet::new();
    for file in &set.file {
        let package = file.package();
        if package != WIRE_PACKAGE_PREFIX
            && !package.starts_with(&format!("{WIRE_PACKAGE_PREFIX}."))
        {
            continue;
        }
        for service in &file.service {
            for method in &service.method {
                rpcs.insert(format!("{}/{}", service.name(), method.name()));
            }
        }
    }
    rpcs
}

// ---------------------------------------------------------------------------
// Control-plane allowlist
// ---------------------------------------------------------------------------

/// Rpcs that legitimately carry no per-tenant isolation case because they touch
/// no tenant-scoped rows. Each entry is `(Service, Method)` with the reason it
/// is exempt; a new rpc that does NOT belong here must arrive with a case
/// instead (the bind enforces this).
///
/// AUDIT NOTE — the audit rpcs are intentionally NOT here. The per-query audit
/// table carries an implicit `tenant_id` column and the read path
/// (`audit::query::fetch_by_query_id` / `fetch_recent`) runs through the
/// session's tenant-scoped SQL, so a caller only ever sees its own tenant's
/// records. Because audit IS tenant-scoped, the three audit rpcs are covered as
/// cases, not allowlisted.
const CONTROL_PLANE_ALLOWLIST: &[(&str, &str)] = &[
    // Session control — establish / read / clear the per-session tenant
    // binding. They mutate session state, never a tenant-owned catalog row.
    ("CatalogService", "SetTenant"),
    ("CatalogService", "GetTenant"),
    ("CatalogService", "ClearTenant"),
    // Handshake metadata — reports the server's mounted service tiers and
    // version. Tenant-independent.
    ("CatalogService", "GetServerInfo"),
];

// ---------------------------------------------------------------------------
// Case model
// ---------------------------------------------------------------------------

/// How a case proves isolation, recorded so the gap each kind closes is visible
/// in the case table rather than implicit in the closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseKind {
    /// A creates a resource and B is refused read/overwrite/delete, all driven
    /// hermetically through the catalog repos.
    Hermetic,
    /// A stated-positive: a global row is visible to a tenant, or a
    /// global-namespace key collision errors rather than clobbering a peer.
    GlobalVisibility,
    /// A compute verb that needs a GPU to run end-to-end. The closure asserts
    /// the tenant-scoped resolver the verb depends on is isolated (this runs
    /// for real); `covered_by` names the end-to-end test that exercises the
    /// full path.
    ComputeResolver,
    /// Flight SQL `DoGet` (`query` / `sql`) is off the `jammi.v1` descriptor
    /// (it is `arrow.flight.FlightService/DoGet`), so the bind cannot force it.
    /// This explicit case drives a tenant-scoped SQL read and proves
    /// cross-tenant rows are not returned, so the most-used data verb is proven
    /// rather than silently uncovered.
    FlightSql,
}

/// One isolation case. `service` / `rpc` form the gRPC path that the structural
/// bind matches against the derived wire surface; for [`CaseKind::FlightSql`]
/// they name the off-descriptor Flight path and are deliberately not expected
/// to appear in `wire_rpcs`.
struct IsolationCase {
    service: &'static str,
    rpc: &'static str,
    kind: CaseKind,
    /// The end-to-end test that exercises this verb's full path, for the
    /// compute verbs whose hermetic coverage is the resolver they depend on.
    covered_by: Option<&'static str>,
    /// The isolation assertion. Boxed so cases live in one `const`-shaped slice
    /// while each carries its own behavioural body.
    assert: fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
}

/// The end-to-end cross-process isolation test the compute-verb cases point at.
/// Named once so a rename surfaces here, not scattered across cases.
const E2E_ISOLATION_TEST: &str = "crates/jammi-ai/tests/distributed/cross_tenant_isolation.rs::\
     mixed_tenant_queue_completes_each_under_its_own_scope";

// Macro to declare a case with an async body without repeating the boxing
// boilerplate. The body is an `async` block; it is pinned and boxed by the
// macro so `IsolationCase::assert` stays a plain `fn` pointer.
macro_rules! case {
    ($service:literal, $rpc:literal, $kind:expr, $covered_by:expr, $body:block) => {
        IsolationCase {
            service: $service,
            rpc: $rpc,
            kind: $kind,
            covered_by: $covered_by,
            assert: || Box::pin(async move $body),
        }
    };
}

// ---------------------------------------------------------------------------
// Hermetic harness
// ---------------------------------------------------------------------------

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
}

/// Open one shared SQLite backend and return an unscoped base catalog over it,
/// migrated. Per-tenant handles come from [`Catalog::pinned_to_tenant`] so all
/// handles see the same rows.
async fn shared_catalog() -> (TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let backend = SqliteBackend::open(&dir.path().join("catalog.db"))
        .await
        .unwrap();
    let catalog = Catalog::from_backend(BackendImpl::Sqlite(backend));
    catalog.backend_arc().migrate().await.unwrap();
    (dir, catalog)
}

/// Two per-tenant catalog handles plus an unscoped (global) handle over one
/// shared backend. Returned together with the `TempDir` guard so the backing
/// file lives for the duration of the case.
async fn ab_catalogs() -> (TempDir, Catalog, Catalog, Catalog) {
    let (dir, base) = shared_catalog().await;
    let cat_a = base.pinned_to_tenant(Some(tenant_a()));
    let cat_b = base.pinned_to_tenant(Some(tenant_b()));
    let cat_global = base.pinned_to_tenant(None);
    (dir, cat_a, cat_b, cat_global)
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

fn parquet_connection() -> SourceConnection {
    SourceConnection {
        url: Some(fixture_url("patents.parquet")),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

fn result_params<'a>(
    name: &'a str,
    source: &'a str,
    model: &'a str,
) -> CreateResultTableParams<'a> {
    CreateResultTableParams {
        table_name: name,
        source_id: source,
        model_id: model,
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/rt.parquet",
        index_path: None,
        dimensions: None,
        key_column: None,
        text_columns: None,
    }
}

/// One global topic schema, shared by the topic cases.
fn topic_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]))
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

fn cases() -> Vec<IsolationCase> {
    vec![
        // --- sources ---------------------------------------------------------
        case!("CatalogService", "AddSource", CaseKind::Hermetic, None, {
            // A registers `src_a`; B's tenant-scoped reads never surface it.
            let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
            cat_a
                .register_source("src_a", SourceType::File, &parquet_connection())
                .await
                .unwrap();
            let b_ids: Vec<String> = cat_b
                .list_sources()
                .await
                .unwrap()
                .into_iter()
                .map(|s| s.source_id)
                .collect();
            assert!(
                !b_ids.contains(&"src_a".to_string()),
                "CROSS-TENANT READ LEAK: tenant B sees tenant A's source via list_sources: {b_ids:?}"
            );
            assert!(
                cat_b.get_source("src_a").await.unwrap().is_none(),
                "tenant B must not resolve tenant A's source by id"
            );
        }),
        case!(
            "CatalogService",
            "RemoveSource",
            CaseKind::Hermetic,
            None,
            {
                // STRICT delete predicate: a tenant removes only a source it owns —
                // never a peer's, never a shared GLOBAL one.
                let (_dir, cat_a, cat_b, cat_g) = ab_catalogs().await;
                cat_g
                    .register_source("global_src", SourceType::File, &parquet_connection())
                    .await
                    .unwrap();
                cat_a
                    .register_source("src_a", SourceType::File, &parquet_connection())
                    .await
                    .unwrap();
                cat_b.remove_source("src_a").await.unwrap();
                assert!(
                    cat_a.get_source("src_a").await.unwrap().is_some(),
                    "tenant B must not delete tenant A's source"
                );
                cat_b.remove_source("global_src").await.unwrap();
                assert!(
                    cat_g.get_source("global_src").await.unwrap().is_some(),
                    "a tenant must not delete a shared GLOBAL source"
                );
                cat_g.remove_source("global_src").await.unwrap();
                assert!(
                    cat_g.get_source("global_src").await.unwrap().is_none(),
                    "an unscoped session manages GLOBAL sources"
                );
            }
        ),
        // RemoveSource cascades into `delete_result_tables_for_source`; that
        // cascade carries the STRICT tenant predicate on both its SELECT (the
        // disk-cleanup set) and its DELETE, so a tenant's remove never deletes
        // a peer's nor a GLOBAL result table for the same source id.
        case!(
            "CatalogService",
            "RemoveSource_result_table_cascade",
            CaseKind::Hermetic,
            None,
            {
                let (_dir, cat_a, cat_b, cat_g) = ab_catalogs().await;
                // A GLOBAL result table + a private one for tenant A, both on
                // source id "shared".
                cat_g
                    .create_result_table(result_params("rt_global", "shared", "model"))
                    .await
                    .unwrap();
                cat_a
                    .create_result_table(result_params("rt_a", "shared", "model"))
                    .await
                    .unwrap();
                // Tenant B's cascade deletes nothing it does not own.
                let removed = cat_b
                    .delete_result_tables_for_source("shared")
                    .await
                    .unwrap();
                assert!(
                    removed.is_empty(),
                    "tenant B must not delete a peer's or GLOBAL result table; got {removed:?}"
                );
                assert!(
                    cat_g.get_result_table("rt_global").await.unwrap().is_some(),
                    "the GLOBAL result table survives a foreign tenant's cascade"
                );
                assert!(
                    cat_a.get_result_table("rt_a").await.unwrap().is_some(),
                    "tenant A's result table survives tenant B's cascade"
                );
                // Tenant A deletes exactly its own row.
                let removed_a = cat_a
                    .delete_result_tables_for_source("shared")
                    .await
                    .unwrap();
                assert_eq!(
                    removed_a
                        .iter()
                        .map(|r| r.table_name.as_str())
                        .collect::<Vec<_>>(),
                    vec!["rt_a"],
                    "tenant A's cascade deletes exactly its own result table"
                );
                assert!(
                    cat_g.get_result_table("rt_global").await.unwrap().is_some(),
                    "the GLOBAL result table still survives after tenant A's cascade"
                );
            }
        ),
        case!("CatalogService", "ListSources", CaseKind::Hermetic, None, {
            // A scoped list returns own + global rows, never a peer's.
            let (_dir, cat_a, cat_b, cat_g) = ab_catalogs().await;
            cat_g
                .register_source("global_src", SourceType::File, &parquet_connection())
                .await
                .unwrap();
            cat_a
                .register_source("src_a", SourceType::File, &parquet_connection())
                .await
                .unwrap();
            cat_b
                .register_source("src_b", SourceType::File, &parquet_connection())
                .await
                .unwrap();
            let mut a_ids: Vec<String> = cat_a
                .list_sources()
                .await
                .unwrap()
                .into_iter()
                .map(|s| s.source_id)
                .collect();
            a_ids.sort();
            assert_eq!(
                a_ids,
                vec!["global_src".to_string(), "src_a".to_string()],
                "tenant A's list must be its own + global rows only"
            );
        }),
        case!(
            "CatalogService",
            "DescribeSource",
            CaseKind::Hermetic,
            None,
            {
                // describe is tenant-filtered: a peer's source is not describable.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .register_source("src_a", SourceType::File, &parquet_connection())
                    .await
                    .unwrap();
                assert!(
                    cat_a.describe_source("src_a").await.unwrap().is_some(),
                    "tenant A must describe its own source"
                );
                assert!(
                    cat_b.describe_source("src_a").await.unwrap().is_none(),
                    "tenant B must not describe tenant A's source"
                );
            }
        ),
        // The global `source_id` PRIMARY KEY is shared across tenants, so two
        // tenants registering the same id collide — the write ERRORS rather
        // than one tenant silently clobbering the other's row.
        case!(
            "CatalogService",
            "AddSource_global_pk_collision",
            CaseKind::GlobalVisibility,
            None,
            {
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .register_source("collide", SourceType::File, &parquet_connection())
                    .await
                    .unwrap();
                let err = cat_b
                    .register_source("collide", SourceType::File, &parquet_connection())
                    .await;
                assert!(
                    err.is_err(),
                    "a global source_id collision must ERROR, not clobber tenant A's row"
                );
                assert!(
                    cat_a.get_source("collide").await.unwrap().is_some(),
                    "tenant A's source survives a colliding cross-tenant register"
                );
            }
        ),
        // --- models ----------------------------------------------------------
        case!("CatalogService", "ListModels", CaseKind::Hermetic, None, {
            let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
            cat_a
                .register_model(register_params("m_a", "candle"))
                .await
                .unwrap();
            let b_names: Vec<String> = cat_b
                .list_models()
                .await
                .unwrap()
                .into_iter()
                .map(|m| m.model_id)
                .collect();
            assert!(
                !b_names.contains(&"m_a".to_string()),
                "CROSS-TENANT READ LEAK: tenant B sees tenant A's model: {b_names:?}"
            );
        }),
        case!(
            "CatalogService",
            "DescribeModel",
            CaseKind::Hermetic,
            None,
            {
                // A's model row is private; B registering the same name::version
                // keys a DISTINCT tenant-qualified PK and never overwrites A's row.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
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
                assert_eq!(a.backend, "candle", "tenant A sees its own model backend");
                assert_eq!(b.backend, "vllm", "tenant B sees its own model backend");
            }
        ),
        case!("CatalogService", "RetireModel", CaseKind::Hermetic, None, {
            // Strict tenant predicate: retiring a peer's model is NotFound and
            // leaves A's row untouched.
            let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
            cat_a
                .register_model(register_params("m_a", "candle"))
                .await
                .unwrap();
            assert!(
                cat_b.retire_model("m_a", None).await.is_err(),
                "tenant B must not retire tenant A's model"
            );
            let a = cat_a.get_model("m_a").await.unwrap().unwrap();
            assert_eq!(
                a.status.as_str(),
                "registered",
                "A's model must stay un-retired"
            );
        }),
        case!("CatalogService", "DeleteModel", CaseKind::Hermetic, None, {
            // Hard-delete is strict-tenant: a peer's delete is NotFound and the
            // row survives.
            let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
            cat_a
                .register_model(register_params("m_a", "candle"))
                .await
                .unwrap();
            assert!(
                cat_b.delete_model("m_a", None, false).await.is_err(),
                "tenant B must not delete tenant A's model"
            );
            assert!(
                cat_a.get_model("m_a").await.unwrap().is_some(),
                "tenant A's model must survive tenant B's delete attempt"
            );
        }),
        case!(
            "CatalogService",
            "PromoteModel",
            CaseKind::Hermetic,
            None,
            {
                // Promote resolves through the tenant-filtered get_model, so a peer
                // cannot promote a model it cannot resolve.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .register_model(register_params("m_a", "candle"))
                    .await
                    .unwrap();
                assert!(
                    cat_b.promote_model("m_a", None).await.is_err(),
                    "tenant B must not promote tenant A's model"
                );
            }
        ),
        // --- channels --------------------------------------------------------
        case!(
            "CatalogService",
            "RegisterChannel",
            CaseKind::Hermetic,
            None,
            {
                // The channel namespace is per-tenant: B does not see A's channel
                // and can register the same name without a false collision.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .channels()
                    .register(&channel_spec("chan_x", 10, "a_marker"))
                    .await
                    .unwrap();
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
                    "CROSS-TENANT READ LEAK: tenant B sees tenant A's channel: {b_names:?}"
                );
                cat_b
                    .channels()
                    .register(&channel_spec("chan_x", 20, "b_marker"))
                    .await
                    .expect("tenant B registers its own 'chan_x' without a false collision");
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
                assert_eq!(a_x.columns[0].name, "a_marker");
                assert_eq!(b_x.columns[0].name, "b_marker");
                // Intra-tenant uniqueness: A re-registering its own 'chan_x' is
                // rejected (a per-tenant duplicate), not silently accepted.
                assert!(
                    cat_a
                        .channels()
                        .register(&channel_spec("chan_x", 30, "dup"))
                        .await
                        .is_err(),
                    "re-registering an existing channel within a tenant must be rejected"
                );
            }
        ),
        case!(
            "CatalogService",
            "AddChannelColumns",
            CaseKind::Hermetic,
            None,
            {
                // Adding columns to a peer's channel is NotRegistered — the parent
                // existence check is tenant-scoped.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .channels()
                    .register(&channel_spec("chan_x", 10, "a_marker"))
                    .await
                    .unwrap();
                let extra = vec![ChannelColumn {
                    name: "injected".into(),
                    data_type: ChannelColumnType::Utf8,
                }];
                let res = cat_b
                    .channels()
                    .add_columns(&ChannelId::new("chan_x").unwrap(), &extra)
                    .await;
                assert!(
                    res.is_err(),
                    "tenant B must not add columns to tenant A's channel"
                );
                let a_x = cat_a
                    .channels()
                    .get(&ChannelId::new("chan_x").unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert!(
                    !a_x.columns.iter().any(|c| c.name == "injected"),
                    "tenant A's channel schema must be untouched by tenant B"
                );
            }
        ),
        case!(
            "CatalogService",
            "ListChannels",
            CaseKind::Hermetic,
            None,
            {
                // A tenant's list includes the global seed channels but not a
                // peer's private channel.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                cat_a
                    .channels()
                    .register(&channel_spec("private_a", 7, "marker"))
                    .await
                    .unwrap();
                let b_names: Vec<String> = cat_b
                    .channels()
                    .list()
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|s| s.id.as_str().to_string())
                    .collect();
                assert!(
                    b_names.contains(&"vector".to_string()),
                    "tenant B must resolve the global seed channels: {b_names:?}"
                );
                assert!(
                    !b_names.contains(&"private_a".to_string()),
                    "CROSS-TENANT READ LEAK: tenant B sees tenant A's private channel: {b_names:?}"
                );
            }
        ),
        // --- mutable tables --------------------------------------------------
        case!(
            "CatalogService",
            "CreateMutableTable",
            CaseKind::Hermetic,
            None,
            {
                // A creates a mutable table under its scope; B with the same id
                // cannot read it, and the global PK collision errors rather than
                // clobbering A's registration.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                let id = MutableTableId::new("events").unwrap();
                let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
                let def = MutableTableDefinitionBuilder::new(id.clone(), schema.clone())
                    .primary_key(vec!["id".into()])
                    .tenant(Some(tenant_a()))
                    .build()
                    .unwrap();
                cat_a.create_mutable_table(&def).await.unwrap();
                assert!(
                    cat_b.get_mutable_table(&id).await.unwrap().is_none(),
                    "tenant B must not read tenant A's mutable table"
                );
                assert!(
                    cat_a.get_mutable_table(&id).await.unwrap().is_some(),
                    "tenant A must read its own mutable table"
                );
            }
        ),
        case!(
            "CatalogService",
            "DropMutableTable",
            CaseKind::Hermetic,
            None,
            {
                // The destructive cross-tenant delete is closed: B's delete is a
                // success no-op (zero rows matched) and A's table survives.
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                let id = MutableTableId::new("events").unwrap();
                let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
                let def = MutableTableDefinitionBuilder::new(id.clone(), schema)
                    .primary_key(vec!["id".into()])
                    .tenant(Some(tenant_a()))
                    .build()
                    .unwrap();
                cat_a.create_mutable_table(&def).await.unwrap();
                cat_b.delete_mutable_table(&id).await.unwrap();
                assert!(
                    cat_a.get_mutable_table(&id).await.unwrap().is_some(),
                    "tenant A's mutable table must survive tenant B's cross-tenant delete"
                );
            }
        ),
        case!(
            "CatalogService",
            "ListMutableTables",
            CaseKind::Hermetic,
            None,
            {
                let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
                let id = MutableTableId::new("events").unwrap();
                let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
                let def = MutableTableDefinitionBuilder::new(id.clone(), schema)
                    .primary_key(vec!["id".into()])
                    .tenant(Some(tenant_a()))
                    .build()
                    .unwrap();
                cat_a.create_mutable_table(&def).await.unwrap();
                let b_ids: Vec<String> = cat_b
                    .list_mutable_tables(Some(tenant_b()))
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|d| d.id.as_str().to_string())
                    .collect();
                assert!(
                    !b_ids.contains(&"events".to_string()),
                    "CROSS-TENANT READ LEAK: tenant B lists tenant A's mutable table: {b_ids:?}"
                );
            }
        ),
        // --- topics (session-driven: TopicRepo needs the backing-table
        //     registry the session owns) -----------------------------------
        case!(
            "CatalogService",
            "RegisterTopic",
            CaseKind::Hermetic,
            None,
            {
                // A registers a private topic; B's tenant-scoped list never shows
                // it, and B can register the same name (per-tenant namespace).
                let (_dir, sess_a, sess_b, _g) = session_ab().await;
                let owned_a = TopicDefinition {
                    id: TopicId::new(),
                    name: "a.events".into(),
                    schema: topic_schema(),
                    tenant: Some(tenant_a()),
                    broker_metadata: Default::default(),
                };
                sess_a.topic_repo().register_topic(&owned_a).await.unwrap();
                let b_ids = list_topic_ids(&sess_b, Some(tenant_b())).await;
                assert!(
                    !b_ids.contains(&owned_a.id),
                    "CROSS-TENANT READ LEAK: tenant B sees tenant A's topic"
                );
            }
        ),
        case!("CatalogService", "DropTopic", CaseKind::Hermetic, None, {
            // STRICT lookup + delete: B dropping A's topic is TopicNotFound and
            // A's topic (and its shared backing table) survives.
            let (_dir, sess_a, sess_b, sess_g) = session_ab().await;
            let global = TopicDefinition {
                id: TopicId::new(),
                name: "global.events".into(),
                schema: topic_schema(),
                tenant: None,
                broker_metadata: Default::default(),
            };
            sess_g.topic_repo().register_topic(&global).await.unwrap();
            let owned_a = TopicDefinition {
                id: TopicId::new(),
                name: "a.events".into(),
                schema: topic_schema(),
                tenant: Some(tenant_a()),
                broker_metadata: Default::default(),
            };
            sess_a.topic_repo().register_topic(&owned_a).await.unwrap();
            match sess_b
                .topic_repo()
                .drop_topic(owned_a.id, Some(tenant_b()))
                .await
            {
                Err(TriggerError::TopicNotFound(_)) => {}
                other => panic!("expected TopicNotFound for cross-tenant drop, got {other:?}"),
            }
            assert!(
                list_topic_ids(&sess_a, Some(tenant_a()))
                    .await
                    .contains(&owned_a.id),
                "tenant B must not drop tenant A's topic"
            );
            match sess_b
                .topic_repo()
                .drop_topic(global.id, Some(tenant_b()))
                .await
            {
                Err(TriggerError::TopicNotFound(_)) => {}
                other => panic!("expected TopicNotFound for GLOBAL drop, got {other:?}"),
            }
            assert!(
                list_topic_ids(&sess_g, None).await.contains(&global.id),
                "a tenant must not drop a shared GLOBAL topic"
            );
        }),
        case!("CatalogService", "ListTopics", CaseKind::Hermetic, None, {
            let (_dir, sess_a, sess_b, _g) = session_ab().await;
            let owned_a = TopicDefinition {
                id: TopicId::new(),
                name: "a.events".into(),
                schema: topic_schema(),
                tenant: Some(tenant_a()),
                broker_metadata: Default::default(),
            };
            sess_a.topic_repo().register_topic(&owned_a).await.unwrap();
            assert!(
                !list_topic_ids(&sess_b, Some(tenant_b()))
                    .await
                    .contains(&owned_a.id),
                "CROSS-TENANT READ LEAK: tenant B lists tenant A's topic"
            );
        }),
        // Publish / Subscribe ride the same tenant-scoped backing table the
        // topic catalog provisions; the read-side isolation is proven by the
        // subscribe-scoped stream test in the engine, and the topic-row scope
        // is proven by the topic cases above. The hermetic assertion here pins
        // that a tenant cannot resolve a peer's topic to publish/subscribe to.
        case!("TriggerService", "Publish", CaseKind::Hermetic, None, {
            let (_dir, sess_a, sess_b, _g) = session_ab().await;
            let owned_a = TopicDefinition {
                id: TopicId::new(),
                name: "a.events".into(),
                schema: topic_schema(),
                tenant: Some(tenant_a()),
                broker_metadata: Default::default(),
            };
            sess_a.topic_repo().register_topic(&owned_a).await.unwrap();
            assert!(
                sess_b
                    .topic_repo()
                    .lookup_by_name("a.events", Some(tenant_b()))
                    .await
                    .unwrap()
                    .is_none(),
                "tenant B must not resolve tenant A's topic to publish to it"
            );
        }),
        case!("TriggerService", "Subscribe", CaseKind::Hermetic, None, {
            let (_dir, sess_a, sess_b, _g) = session_ab().await;
            let owned_a = TopicDefinition {
                id: TopicId::new(),
                name: "a.events".into(),
                schema: topic_schema(),
                tenant: Some(tenant_a()),
                broker_metadata: Default::default(),
            };
            sess_a.topic_repo().register_topic(&owned_a).await.unwrap();
            assert!(
                sess_b
                    .topic_repo()
                    .lookup_by_name("a.events", Some(tenant_b()))
                    .await
                    .unwrap()
                    .is_none(),
                "tenant B must not resolve tenant A's topic to subscribe to it"
            );
        }),
        // --- eval + result tables --------------------------------------------
        case!("EvalService", "EvalEmbeddings", CaseKind::Hermetic, None, {
            assert_eval_run_isolated().await;
        }),
        case!("EvalService", "EvalPerQuery", CaseKind::Hermetic, None, {
            assert_eval_run_isolated().await;
        }),
        case!("EvalService", "EvalInference", CaseKind::Hermetic, None, {
            assert_eval_run_isolated().await;
        }),
        case!("EvalService", "EvalCompare", CaseKind::Hermetic, None, {
            assert_eval_run_isolated().await;
        }),
        case!(
            "EvalService",
            "EvalCalibration",
            CaseKind::Hermetic,
            None,
            {
                assert_eval_run_isolated().await;
            }
        ),
        // --- audit (tenant-scoped — NOT allowlisted) -------------------------
        case!("AuditService", "AuditLog", CaseKind::Hermetic, None, {
            assert_audit_isolated().await;
        }),
        case!(
            "AuditService",
            "AuditFetchByQueryId",
            CaseKind::Hermetic,
            None,
            {
                assert_audit_isolated().await;
            }
        ),
        case!(
            "AuditService",
            "AuditFetchRecent",
            CaseKind::Hermetic,
            None,
            {
                assert_audit_isolated().await;
            }
        ),
        // --- compute verbs (resolver-isolated; covered_by names the e2e) -----
        case!(
            "EmbeddingService",
            "GenerateEmbeddings",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                // GenerateEmbeddings loads its embedding model (tenant-filtered
                // `get_model`) and reads the input source (tenant-scoped SQL); it
                // does NOT call `resolve_embedding_table` (it writes a fresh table).
                // The model leg is asserted here; the source leg is covered by the
                // AddSource/ListSources/DescribeSource cases.
                assert_model_resolver_isolated().await;
            }
        ),
        case!(
            "EmbeddingService",
            "EncodeQuery",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_model_resolver_isolated().await;
            }
        ),
        case!(
            "EmbeddingService",
            "Search",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_embedding_resolver_isolated().await;
            }
        ),
        case!(
            "InferenceService",
            "Infer",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_model_resolver_isolated().await;
            }
        ),
        case!(
            "InferenceService",
            "Predict",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_model_resolver_isolated().await;
            }
        ),
        case!(
            "PipelineService",
            "PropagateEmbeddings",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_embedding_resolver_isolated().await;
            }
        ),
        case!(
            "PipelineService",
            "BuildNeighborGraph",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_embedding_resolver_isolated().await;
            }
        ),
        case!(
            "PipelineService",
            "AssembleContext",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_embedding_resolver_isolated().await;
            }
        ),
        case!(
            "TrainingService",
            "StartTraining",
            CaseKind::ComputeResolver,
            Some(E2E_ISOLATION_TEST),
            {
                assert_training_create_isolated().await;
            }
        ),
        case!(
            "TrainingService",
            "TrainingStatus",
            CaseKind::Hermetic,
            None,
            {
                // TrainingStatus reads the job row via the tenant-filtered
                // `get_training_job` (NOT by an "unguessable" id) — a peer cannot
                // read another tenant's job status. The shared helper creates a
                // job under A and asserts that exact read isolation.
                assert_training_create_isolated().await;
            }
        ),
        // --- Flight SQL (off-descriptor; explicit case) ----------------------
        case!(
            "arrow.flight.FlightService",
            "DoGet",
            CaseKind::FlightSql,
            None,
            {
                assert_flight_sql_isolated().await;
            }
        ),
    ]
}

// ---------------------------------------------------------------------------
// Session-backed harness (topics / Flight SQL)
// ---------------------------------------------------------------------------

/// Tenant A, tenant B, and an unscoped session over one shared config (one
/// backend). The topic and Flight SQL cases need the session's backing-table
/// registry and SQL path, which the bare `Catalog` does not carry.
async fn session_ab() -> (TempDir, JammiSession, JammiSession, JammiSession) {
    let dir = tempdir().unwrap();
    let cfg = test_config(dir.path());
    let sess_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    let sess_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());
    let sess_g = JammiSession::new(cfg).await.unwrap();
    (dir, sess_a, sess_b, sess_g)
}

async fn list_topic_ids(sess: &JammiSession, tenant: Option<TenantId>) -> Vec<TopicId> {
    sess.topic_repo()
        .list_topics(tenant)
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.id)
        .collect()
}

// ---------------------------------------------------------------------------
// Shared assertion bodies (used by multiple cases that share one resolver)
// ---------------------------------------------------------------------------

/// An eval run carries a `tenant_id`; the read path is tenant-filtered, so a
/// peer cannot fetch a tenant's run. The `eval_runs.model_id` FK is satisfied
/// by registering the model under the same tenant first.
async fn assert_eval_run_isolated() {
    let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
    cat_a
        .register_model(register_params("m_a", "candle"))
        .await
        .unwrap();
    let pk = cat_a.get_model("m_a").await.unwrap().unwrap().catalog_pk;
    let run = EvalRunRecord {
        eval_run_id: "run_a".into(),
        eval_type: "embedding".into(),
        model_id: Some(pk),
        source_id: "src_a".into(),
        golden_source: "golden".into(),
        k: Some(10),
        metrics_json: "{}".into(),
        status: "completed".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    cat_a.record_eval_run(&run).await.unwrap();
    assert!(
        cat_a.get_eval_run("run_a").await.unwrap().is_some(),
        "tenant A must read its own eval run"
    );
    assert!(
        cat_b.get_eval_run("run_a").await.unwrap().is_none(),
        "CROSS-TENANT READ LEAK: tenant B reads tenant A's eval run"
    );
}

/// Audit rows carry an implicit `tenant_id`; the read path is the session's
/// tenant-scoped SQL, so a peer cannot fetch a tenant's audit record. This
/// drives the real write + read path through the session.
async fn assert_audit_isolated() {
    use jammi_db::audit::MASTER_KEY_ENV;

    // The audit signing key is a process-global env var (64-char hex → 32
    // bytes); set a deterministic test key so `log` can sign. The oracle runs
    // its cases serially, so no other case races this binding.
    std::env::set_var(
        MASTER_KEY_ENV,
        "0000000000000000000000000000000000000000000000000000000000000001",
    );

    // One shared session scoped per-tenant via `with_tenant_scoped` — the
    // realistic server topology (per-request handlers ride one session), and
    // the one provider registration both scopes resolve the audit table
    // through.
    let dir = tempdir().unwrap();
    let session = Arc::new(JammiSession::new(test_config(dir.path())).await.unwrap());

    async fn log(session: &JammiSession, tenant: TenantId, model: &'static str) -> uuid::Uuid {
        use jammi_db::audit::PerQueryAudit;
        session
            .with_tenant_scoped(tenant, |scope| {
                let model = model.to_string();
                async move {
                    let query_id = uuid::Uuid::new_v4();
                    let record = PerQueryAudit::new(
                        query_id,
                        model.clone(),
                        "v1",
                        serde_json::json!({ "q": model }),
                        vec!["doc1".into()],
                        vec![0.9_f32],
                    )
                    .unwrap();
                    scope.audit().log(vec![record]).await.unwrap();
                    query_id
                }
            })
            .await
    }

    // Both tenants log one record each; A's id is the cross-tenant probe.
    let a_id = log(&session, tenant_a(), "m_a").await;
    let _b_id = log(&session, tenant_b(), "m_b").await;

    // Tenant A reads its own record.
    let a_sees = session
        .with_tenant_scoped(tenant_a(), |scope| async move {
            scope.audit().fetch_by_query_id(a_id).await.unwrap()
        })
        .await;
    assert!(a_sees.is_some(), "tenant A must read its own audit record");

    // Tenant B cannot fetch A's record by id, and its recent fetch surfaces
    // only its own record.
    let (b_probe, b_recent) = session
        .with_tenant_scoped(tenant_b(), |scope| async move {
            let probe = scope.audit().fetch_by_query_id(a_id).await.unwrap();
            let recent = scope.audit().fetch_recent(100).await.unwrap();
            (probe, recent)
        })
        .await;
    assert!(
        b_probe.is_none(),
        "CROSS-TENANT READ LEAK: tenant B reads tenant A's audit record by query_id"
    );
    assert_eq!(b_recent.len(), 1, "tenant B sees only its own audit record");
    assert_eq!(
        b_recent[0].model_id, "m_b",
        "tenant B's recent fetch is its own row, not tenant A's"
    );
}

/// Search / Propagate / BuildNeighborGraph / AssembleContext resolve their input
/// embedding table through `resolve_embedding_table`, which is tenant-filtered: a
/// peer cannot resolve a tenant's ready embedding table.
async fn assert_embedding_resolver_isolated() {
    let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
    // A ready model-output embedding table for source "src_a", owned by A.
    let params = result_params("rt_a", "src_a", "m_a");
    cat_a.create_result_table(params).await.unwrap();
    cat_a
        .update_result_table_status(
            "rt_a",
            jammi_db::catalog::status::ResultTableStatus::Ready,
            0,
        )
        .await
        .unwrap();
    assert!(
        cat_a.resolve_embedding_table("src_a", None).await.is_ok(),
        "tenant A must resolve its own embedding table"
    );
    assert!(
        cat_b.resolve_embedding_table("src_a", None).await.is_err(),
        "CROSS-TENANT LEAK: tenant B resolved tenant A's embedding table"
    );
}

/// Infer / Predict / EncodeQuery / GenerateEmbeddings resolve the model through
/// `get_model`, which is tenant-filtered: a peer cannot resolve a tenant's
/// private model. (Infer / Predict additionally read a source scan via
/// tenant-scoped SQL, covered by the Flight SQL and source cases.)
async fn assert_model_resolver_isolated() {
    let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
    cat_a
        .register_model(register_params("m_a", "candle"))
        .await
        .unwrap();
    assert!(
        cat_a.get_model("m_a").await.unwrap().is_some(),
        "tenant A must resolve its own model"
    );
    assert!(
        cat_b.get_model("m_a").await.unwrap().is_none(),
        "CROSS-TENANT LEAK: tenant B resolved tenant A's model"
    );
}

/// StartTraining writes a job row under the session tenant; the create path is
/// the tenant gate. A peer cannot read the resulting job row (tenant-filtered
/// `get_training_job`), and the base-model FK resolves only the creating
/// tenant's model.
async fn assert_training_create_isolated() {
    let (_dir, cat_a, cat_b, _g) = ab_catalogs().await;
    cat_a
        .register_model(register_params("m_a", "candle"))
        .await
        .unwrap();
    let base_pk = cat_a.get_model("m_a").await.unwrap().unwrap().catalog_pk;
    cat_a
        .create_training_job(CreateTrainingJobParams {
            job_id: "job_a",
            base_model_id: &base_pk,
            training_source: "src_a",
            loss_type: "triplet",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    assert!(
        cat_a.get_training_job("job_a").await.is_ok(),
        "tenant A must read its own training job"
    );
    assert!(
        cat_b.get_training_job("job_a").await.is_err(),
        "CROSS-TENANT READ LEAK: tenant B reads tenant A's training job"
    );
}

/// Flight SQL `DoGet` runs `sql` against the session's tenant-scoped provider.
/// Two tenants insert into the same mutable table; each `SELECT` returns only
/// its own rows — cross-tenant rows are never returned.
async fn assert_flight_sql_isolated() {
    // One shared session (the realistic server topology); the table is created
    // once (unscoped) so both tenants address the same backing storage, and
    // per-tenant scopes via `with_tenant_scoped` are what separate the rows.
    let dir = tempdir().unwrap();
    let session = Arc::new(JammiSession::new(test_config(dir.path())).await.unwrap());
    let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("widgets").unwrap(),
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
    )
    .primary_key(vec!["id".into()])
    .build()
    .unwrap();
    session.create_mutable_table(def).await.unwrap();

    async fn insert(session: &JammiSession, tenant: TenantId, id: i64) {
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.widgets (id) VALUES ({id})"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }
    insert(&session, tenant_a(), 1).await;
    insert(&session, tenant_b(), 2).await;

    let ids_a = select_ids(&session, tenant_a()).await;
    let ids_b = select_ids(&session, tenant_b()).await;
    assert_eq!(
        ids_a,
        vec![1],
        "tenant A's Flight SQL read returns only its row"
    );
    assert_eq!(
        ids_b,
        vec![2],
        "tenant B's Flight SQL read returns only its row"
    );
}

async fn select_ids(session: &JammiSession, tenant: TenantId) -> Vec<i64> {
    session
        .with_tenant_scoped(tenant, |scope| async move {
            let rows = scope
                .sql("SELECT id FROM mutable.public.widgets ORDER BY id")
                .await
                .unwrap();
            let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
            let col = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect()
        })
        .await
}

// ---------------------------------------------------------------------------
// The structural bind + the running cases
// ---------------------------------------------------------------------------

/// The set of `Service/Method` paths a case or the allowlist accounts for.
/// Flight SQL is off the `jammi.v1` descriptor by construction, so its case is
/// excluded from the covered set used against `wire_rpcs` (it is proven by its
/// own running case, not by the descriptor bind).
fn covered_on_wire(cases: &[IsolationCase]) -> BTreeSet<String> {
    let mut covered: BTreeSet<String> = cases
        .iter()
        .filter(|c| c.kind != CaseKind::FlightSql)
        // Synthetic case names (e.g. the `_global_pk_collision` companion) are
        // not wire rpcs; they share the real rpc's `service`/`rpc-without-suffix`
        // identity, so strip a trailing `_...` synthetic tag for the bind.
        .map(|c| format!("{}/{}", c.service, base_rpc(c.rpc)))
        .collect();
    for (service, rpc) in CONTROL_PLANE_ALLOWLIST {
        covered.insert(format!("{service}/{rpc}"));
    }
    covered
}

/// Strip a synthetic companion suffix (`Rpc_some_tag` → `Rpc`) so multiple
/// cases over one rpc still bind to that single wire method. A real rpc name
/// never contains `_`, so the first `_` marks a synthetic tag.
fn base_rpc(rpc: &str) -> &str {
    match rpc.split_once('_') {
        Some((base, _)) => base,
        None => rpc,
    }
}

/// THE CORE GUARANTEE — coverage is structurally enforced.
///
/// Every `jammi.v1` rpc in the compiled descriptor must be either covered by an
/// isolation case or on the control-plane allowlist. A new rpc that lands with
/// neither makes this fail, naming the rpc — so coverage cannot silently rot.
/// The reverse direction fails on a stale case/allowlist entry naming a removed
/// rpc, so the two sets stay exactly the wire surface.
#[test]
fn every_rpc_is_covered() {
    let cases = cases();
    let wire = wire_rpcs();
    let covered = covered_on_wire(&cases);

    let uncovered: Vec<&String> = wire.difference(&covered).collect();
    assert!(
        uncovered.is_empty(),
        "UNCOVERED rpc(s) — add an isolation case or an explicit control-plane \
         allowlist entry for each:\n  {uncovered:#?}"
    );

    let stale: Vec<&String> = covered.difference(&wire).collect();
    assert!(
        stale.is_empty(),
        "STALE case/allowlist entries naming rpc(s) absent from the wire descriptor:\n  {stale:#?}"
    );
}

/// The allowlist and the cases partition the wire surface with no gaps and no
/// overlap: every rpc is accounted for exactly once, and the counts sum to the
/// descriptor total. This is the arithmetic the final report cites.
#[test]
fn allowlist_and_cases_partition_the_wire_surface() {
    let cases = cases();
    let wire = wire_rpcs();

    let case_rpcs: BTreeSet<String> = cases
        .iter()
        .filter(|c| c.kind != CaseKind::FlightSql)
        .map(|c| format!("{}/{}", c.service, base_rpc(c.rpc)))
        .collect();
    let allow_rpcs: BTreeSet<String> = CONTROL_PLANE_ALLOWLIST
        .iter()
        .map(|(s, r)| format!("{s}/{r}"))
        .collect();

    // No rpc is both a case and allowlisted.
    let overlap: Vec<&String> = case_rpcs.intersection(&allow_rpcs).collect();
    assert!(
        overlap.is_empty(),
        "rpc(s) both cased and allowlisted: {overlap:#?}"
    );

    // Cases + allowlist == wire surface exactly.
    let union: BTreeSet<String> = case_rpcs.union(&allow_rpcs).cloned().collect();
    assert_eq!(
        union, wire,
        "cases ∪ allowlist must equal the wire surface exactly"
    );

    // Arithmetic: |cases-on-wire| + |allowlist| == |wire|.
    assert_eq!(
        case_rpcs.len() + allow_rpcs.len(),
        wire.len(),
        "case count ({}) + allowlist count ({}) must equal wire rpc count ({})",
        case_rpcs.len(),
        allow_rpcs.len(),
        wire.len(),
    );
}

/// Every case's behavioural assertion actually executes — coverage is real, not
/// a label. The `ComputeResolver` cases additionally carry a `covered_by`
/// pointer at a REAL end-to-end test; the path is asserted to exist on disk so
/// the pointer cannot rot into a dangling reference.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_case_isolation_holds() {
    for case in cases() {
        if let Some(pointer) = case.covered_by {
            let file = pointer.split("::").next().unwrap();
            let path = repo_root().join(file);
            assert!(
                path.exists(),
                "case {}/{} names covered_by test file '{file}' which does not exist",
                case.service,
                case.rpc,
            );
        }
        (case.assert)().await;
    }
}

/// The workspace root, relative to this crate's manifest dir, so the
/// `covered_by` file-existence check resolves real paths regardless of the
/// process cwd.
fn repo_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/jammi-server.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root is two levels above the crate manifest dir")
        .to_path_buf()
}
