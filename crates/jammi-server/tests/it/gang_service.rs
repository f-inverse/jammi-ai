//! `GangService`'s U5a-1 slice.
//!
//! The first two tests below prove the §W2 Resolution hazards CONTRACT-U5a.md
//! round-11 folds 4 and 5 name (both b1' rows) directly against
//! `crate::grpc::gang::resolve_training_set_identity` and
//! `jammi_db::Catalog::get_result_table_for_tenant` — the way §W2 Resolution
//! states the property (a repo-level predicate plus an explicit guard at the
//! resolution call site), matching `get_job_for_rank`'s own
//! enumerating-caller-oracle treatment elsewhere in this contract. Neither
//! wires into `RunRank`'s own control flow yet: that needs `get_job_for_rank`,
//! this unit's still-pending db step, so there is no rpc-level "call
//! `RunRank`" surface to drive for either row today.
//!
//! The remaining tests drive the real `RunRank` rpc over the production
//! `peer_bind` listener (`start_engine_server_with_peer_bind`) — the wire-level
//! K2 edges (§W1: `world == 0`, `rank >= world`, refused before I-GANG runs)
//! and the interim state between this unit's merge and the day
//! `get_job_for_rank` lands: with no I-GANG determinant wired in yet, every
//! wire-valid `Assign` currently ends `UNIMPLEMENTED`.

use std::str::FromStr;
use std::sync::Arc;

use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::model_task::ModelTask;
use jammi_db::store::ResultStore;
use jammi_db::TenantId;
use jammi_server::grpc::gang::resolve_training_set_identity;
use tempfile::tempdir;
use tonic::Code;

fn null_tenant_row<'a>(table: &'a str) -> CreateResultTableParams<'a> {
    CreateResultTableParams {
        table_name: table,
        source_id: "src",
        model_id: "rt-base",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/does-not-exist.parquet",
        dimensions: Some(4),
        key_column: None,
        text_columns: None,
        storage_precision: StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::backend::now_sortable(),
        writer_id: None,
        lease: None,
        job_attempt: None,
    }
}

/// CONTRACT-U5a.md §W2 Resolution (round-11 fold, ruling 5), b1' row: "a rank
/// on tenant A resolves a NULL-tenant table of the requested name through the
/// relaxed `get_result_table` read." A NULL-tenant `result_tables` row exists
/// (as if created by a coordinator outside any tenant scope); from tenant A,
/// the RELAXED `get_result_table` (today's only seam) returns it — the
/// pre-existing hazard, unconditionally true at base — while the NEW STRICT
/// `get_result_table_for_tenant` refuses to match it, and
/// `resolve_training_set_identity` built on the strict verb refuses
/// `FAILED_PRECONDITION` rather than resolving the wrong tenant's table.
#[tokio::test]
async fn strict_resolver_never_matches_a_null_tenant_row_for_a_real_tenant() {
    let dir = tempdir().unwrap();
    let session = jammi_test_utils::make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session");
    let catalog = Arc::clone(session.catalog());

    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
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

    let table = format!("gang_null_tenant_{}", jammi_test_utils::unique_suffix());
    // Created with NO tenant scope active — the row's `tenant_id` lands NULL,
    // as `create_result_table`'s ambient `self.current_tenant()` resolves to
    // `None` on this unbound session.
    catalog
        .create_result_table(null_tenant_row(&table))
        .await
        .unwrap();

    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();

    // The pre-existing hazard: tenant A, reading through the RELAXED
    // resolver `session.with_tenant_scoped` puts every other tenant-aware
    // surface behind, sees the NULL-tenant row.
    let leaked = session
        .with_tenant_scoped(tenant_a, |scope| {
            let table = table.clone();
            async move { scope.catalog().get_result_table(&table).await.unwrap() }
        })
        .await;
    assert!(
        leaked.is_some(),
        "the relaxed `get_result_table` read must still see the NULL-tenant row \
         (today's only seam) — the hazard this row demonstrates"
    );

    // The NEW strict verb, called with tenant A pinned EXPLICITLY (never
    // ambient), never matches the NULL-tenant row.
    let strict = catalog
        .get_result_table_for_tenant(&table, Some(tenant_a))
        .await
        .unwrap();
    assert!(
        strict.is_none(),
        "the strict resolver must never match a NULL-tenant row for a real tenant, got {strict:?}"
    );

    // The resolution wrapper built on the strict verb refuses
    // FAILED_PRECONDITION — never resolving the NULL-tenant table.
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .expect("result store");
    let err = resolve_training_set_identity(&store, Some(tenant_a), "any-digest", &table)
        .await
        .expect_err("resolution must refuse a NULL-tenant table for a real tenant");
    assert_eq!(err.code(), Code::FailedPrecondition);
}

/// CONTRACT-U5a.md §W2 Resolution (round-11 fold, ruling 4), b1' row: "a
/// resolution wrapped in `with_admin_scope` resolves any tenant's table."
/// `TenantBinding::is_admin_scope()` is ambient — the strict verb's OWN
/// admin-scope behaviour is UNCHANGED (it mirrors every other repo verb's
/// convention and drops the tenant predicate under admin scope, matching a
/// row by primary key alone regardless of the tenant argument passed) —
/// so a guard-less call to the raw verb inside `with_admin_scope` resolves a
/// table belonging to a DIFFERENT tenant than the one it was asked to match.
/// `resolve_training_set_identity`'s own EXPLICIT guard, added at the
/// resolution site rather than relying on the verb, refuses
/// `FAILED_PRECONDITION` before ever calling the verb, regardless of which
/// table exists or which tenant is named.
#[tokio::test]
async fn resolution_site_refuses_under_admin_scope_even_when_the_raw_verb_would_resolve() {
    use datafusion::prelude::SessionContext;
    use jammi_ai::session::InferenceSession;
    use jammi_db::store::manifest::{
        ComputeDevice, ComputePrecision, InputAnchor, Materialization, MaterializationEnv,
        ModelContentDigest, ModelIdentity, ProducingDescriptor,
    };
    use jammi_db::store::EmbeddingTableSpec;

    let dir = tempdir().unwrap();
    let engine = Arc::new(
        InferenceSession::new(jammi_test_utils::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();
    let source_id = format!("gang_admin_src_{}", jammi_test_utils::unique_suffix());

    // A REAL ready table, sidecar and all — materialized for tenant B, so a
    // resolution naming tenant A has no legitimate way to reach it. Zero rows
    // keeps the fixture minimal: `finish` still writes the Parquet + the
    // manifest sidecar and promotes the row to `ready` regardless of row
    // count.
    let descriptor = ProducingDescriptor::Embedding {
        model_id: "rt-base".into(),
        task: ModelTask::TextEmbedding,
        source_id: source_id.clone(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: 4,
    };
    let env = MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "rt-base".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("gang-fixture-digest".into()),
            quantization: None,
        }],
    );
    let ctx = SessionContext::new();
    let source_id_for_scope = source_id.clone();
    let engine_for_scope = Arc::clone(&engine);
    let record = engine
        .with_tenant_scoped(tenant_b, move |_scope| async move {
            let store = engine_for_scope.result_store();
            store
                .materialize_embedding_table(
                    &ctx,
                    EmbeddingTableSpec {
                        source_id: &source_id_for_scope,
                        model_id: "rt-base",
                        derived_from: None,
                        dimensions: 4,
                        key_column: None,
                        text_columns: None,
                    },
                    &[],
                    Materialization::new(
                        &descriptor,
                        &env,
                        vec![InputAnchor::mutable_version(&source_id_for_scope, 1)],
                    ),
                    None,
                )
                .await
                .unwrap()
        })
        .await;
    let table = record.table_name.clone();

    // The pinned digest a genuine caller would carry on the `jobs` row —
    // read back the same way `resolve_training_set_identity` itself would
    // verify it, so this fixture's own honesty is legible: it is testing
    // against a digest that is ACTUALLY the row's own sidecar artifact, not
    // an arbitrary string.
    let store = engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("finish() must have written the sidecar");
    let digest = manifest.artifact.0.clone();

    // Guard-less call to the raw strict verb, inside `with_admin_scope`,
    // passing tenant A: the verb's OWN admin-scope branch drops the tenant
    // predicate entirely and matches by primary key alone — tenant B's row
    // comes back despite the caller naming tenant A. This is the hazard
    // ruling 4 names: the verb's own behaviour under admin scope is
    // unchanged by this contract.
    let table_for_admin = table.clone();
    let guardless = engine
        .with_admin_scope(|admin| {
            let table = table_for_admin.clone();
            async move {
                admin
                    .session()
                    .catalog()
                    .get_result_table_for_tenant(&table, Some(tenant_a))
                    .await
                    .unwrap()
            }
        })
        .await;
    assert!(
        guardless.is_some(),
        "the raw strict verb must still resolve cross-tenant under admin scope \
         (unchanged verb behaviour) — the hazard ruling 4 names"
    );

    // The resolution-site wrapper, called the SAME way (inside
    // `with_admin_scope`, naming tenant A, against a table that genuinely
    // resolves and verifies), refuses before ever calling the verb —
    // regardless of which table exists. The mutation this row proves: delete
    // the admin-scope guard and this SAME call succeeds `Ok(())`, since
    // every other conjunct (status `ready`, matching sidecar digest) is
    // genuinely satisfied — proving the guard, not a coincidental downstream
    // check, is what refuses here.
    let table_for_guard = table.clone();
    let digest_for_guard = digest.clone();
    let guarded =
        engine
            .with_admin_scope(|_admin| {
                let store = &store;
                let table = table_for_guard.clone();
                let digest = digest_for_guard.clone();
                async move {
                    resolve_training_set_identity(store, Some(tenant_a), &digest, &table).await
                }
            })
            .await;
    let err = guarded.expect_err("resolution must refuse under admin scope, unconditionally");
    assert_eq!(err.code(), Code::FailedPrecondition);
}

// ---------------------------------------------------------------------------
// §W1 K2 (wire-level edges, decided before I-GANG runs) + the interim state
// between U5a-1's own merge and the day `get_job_for_rank` lands.
// ---------------------------------------------------------------------------

fn assign_frame(world: u32, rank: u32) -> jammi_wire::proto::gang::RankControl {
    use jammi_wire::proto::gang::{rank_control, Assign, RankControl};
    RankControl {
        control: Some(rank_control::Control::Assign(Assign {
            job_id: "job-1".into(),
            attempt: 0,
            rank,
            world,
            coordinator_instance_id: "coord-1".into(),
        })),
    }
}

/// §W1 K2: `world == 0` is refused `INVALID_ARGUMENT`, before I-GANG (which
/// needs no row read at all here — no `jobs` row could ever satisfy this
/// wire-level edge).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_zero() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(0, 0));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("world == 0 must be refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    // `world == 0` also trips `rank >= world` for every unsigned `rank`
    // (`rank >= 0` is trivially true) — the message, not just the code,
    // distinguishes which K2 edge actually refused, so a mutation that
    // deletes the `world == 0` check specifically (leaving `rank >= world`
    // to catch this exact input by coincidence) still fails this assertion.
    assert!(
        err.message().contains("greater than zero"),
        "expected the world == 0 message specifically (not the rank >= world \
         message, which this exact input also trips since rank >= 0 is \
         trivially true for every unsigned rank), got: {}",
        err.message()
    );
}

/// §W1 K2: `rank >= world` is refused `INVALID_ARGUMENT` — the boundary case
/// (`rank == world`), not just a wildly out-of-range one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_rank_at_world_boundary() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(2, 2));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("rank >= world must be refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// A `RunRank` stream that opens with `Cancel` rather than `Assign` is a
/// protocol violation refused `INVALID_ARGUMENT` — never silently accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_stream_opening_with_cancel() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
    use jammi_wire::proto::gang::{rank_control, Cancel, RankControl};

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(RankControl {
        control: Some(rank_control::Control::Cancel(Cancel {})),
    });
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a stream must open with Assign");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// An empty stream (closed before any frame at all) is refused
/// `INVALID_ARGUMENT` — never left to hang for the admission bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_stream_closed_before_assign() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
    use jammi_wire::proto::gang::RankControl;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::empty::<RankControl>();
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an empty stream must be refused, never accepted silently");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// f1' (U5a-1's OWN interim state, before `get_job_for_rank` lands): a
/// wire-valid `Assign` — every §W1 K2 edge satisfied — reaches the handler
/// and, with no I-GANG determinant wired in yet and no `HostAdmission`
/// session to hand the call to, ends `UNIMPLEMENTED`. Under U5a-2 (§I5) the
/// SAME call instead receives `Admitted`; this row's own claim is narrower
/// than full f1' (I-GANG is not yet DECIDED here, only bypassed) and is
/// superseded once `get_job_for_rank` is wired into this handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_wire_valid_assign_is_unimplemented_before_i_gang_is_wired() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(1, 0));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("no HostAdmission session exists yet to admit into");
    assert_eq!(err.code(), tonic::Code::Unimplemented);
}
