//! `GangService`'s U5a-1 slice.
//!
//! The first two tests below prove the §W2 Resolution hazards CONTRACT-U5a.md
//! round-11 folds 4 and 5 name (both b1' rows) directly against
//! `crate::grpc::gang::resolve_training_set_identity` and
//! `jammi_db::Catalog::get_result_table_for_tenant` — the way §W2 Resolution
//! states the property (a repo-level predicate plus an explicit guard at the
//! resolution call site), matching `get_job_for_rank`'s own
//! enumerating-caller-oracle treatment elsewhere in this contract.
//!
//! The remaining tests drive the real `RunRank` rpc over the production
//! `peer_bind` listener (`start_engine_server_with_peer_bind` /
//! `start_no_worker_server`) — the wire-level K2 edges (§W1: `world == 0`,
//! `rank >= world`, refused before I-GANG runs), every I-GANG determinant
//! `get_job_for_rank` decides (§I1(a): job not found, not `running`, wrong
//! claimant, wrong attempt, lease not live), and f1': a call satisfying
//! EVERY determinant still ends `UNIMPLEMENTED` — this unit has no
//! `HostAdmission` session to hand it to (U5a-2 builds that).

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
    assign_frame_full("job-1", 0, rank, world, "coord-1")
}

fn assign_frame_full(
    job_id: &str,
    attempt: i64,
    rank: u32,
    world: u32,
    coordinator_instance_id: &str,
) -> jammi_wire::proto::gang::RankControl {
    use jammi_wire::proto::gang::{rank_control, Assign, RankControl};
    RankControl {
        control: Some(rank_control::Control::Assign(Assign {
            job_id: job_id.into(),
            attempt,
            rank,
            world,
            coordinator_instance_id: coordinator_instance_id.into(),
        })),
    }
}

/// A peer-bound server with `[worker] enabled = false` — every fixture below
/// drives `Catalog::submit_job`/`claim_next` directly (matching
/// `jobs_queue.rs`'s own fixture style) and needs the row's `status`/
/// `claimed_by`/`attempts` to stay exactly what the fixture set, never raced
/// by this same process's own production claim loop (`[worker] enabled`
/// defaults to `true`, `config/mod.rs`).
async fn start_no_worker_server() -> crate::common::grpc::PeerEngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = crate::common::grpc::peer_bind_config(dir.path());
    cfg.worker.enabled = false;
    crate::common::grpc::start_engine_server_from_config(cfg, Some(dir)).await
}

/// A minimal `world_size == 1` job, submitted and claimed on `server`'s own
/// engine catalog directly (bypassing the wire `JobService`, matching
/// `jobs_queue.rs`'s own fixture style — the gang admission surface reads
/// the row, not the submission RPC). `model_ref: None` avoids needing a
/// registered model FK target (the column is nullable). Returns the
/// `attempts` value the claim landed at (always `1`, the first claim), for
/// the caller to build a matching `Assign` frame with.
async fn submit_and_claim(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
    coordinator_instance_id: &str,
    lease: std::time::Duration,
) -> i64 {
    use jammi_db::catalog::jobs_repo::SubmitJobParams;
    use jammi_db::catalog::status::JobExecution;

    let catalog = server.engine.catalog();
    catalog
        .submit_job(SubmitJobParams {
            job_id,
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next(coordinator_instance_id, &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("must claim the only queued job");
    i64::from(claimed.attempts)
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

/// Observability: `jammi_gang_requests_total{rpc="RunRank"}` counts a
/// `RunRank` call reaching this member — the same shape `PeerService`'s own
/// `jammi_peer_requests_total{rpc}` counter proves (`peer_service.rs`), and
/// counted at the whole-server [`jammi_server::metrics_layer`] regardless of
/// how the call is ultimately decided (this one is refused at the wire-level
/// K2 edge, `world == 0`, before I-GANG ever runs).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_increments_gang_requests_metric() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    assert_eq!(
        server
            .metrics
            .gang_requests
            .with_label_values(&["RunRank"])
            .get(),
        0,
        "no RunRank call has reached this member yet"
    );
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(0, 0));
    let _ = client.run_rank(outbound).await;
    assert_eq!(
        server
            .metrics
            .gang_requests
            .with_label_values(&["RunRank"])
            .get(),
        1,
        "the call above must be counted regardless of its own refusal"
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

// ---------------------------------------------------------------------------
// §I1 (I-GANG, the full row predicate) + f1' (this unit's own terminal state)
// ---------------------------------------------------------------------------

/// f1' (this unit's own success path): a call satisfying EVERY I-GANG
/// determinant — the row is `running`, claimed by the caller's own
/// `coordinator_instance_id`, at the matching `attempt`, under a live
/// lease, and (`world_size == 1` here, so the training-set pair is not
/// gated) the coordinator's own `instances` row is fresh — still reaches
/// `UNIMPLEMENTED`: this unit has no `HostAdmission` session to hand the
/// call to (U5a-2 builds that). Proves every determinant was actually
/// DECIDED (not skipped) — a call that satisfies all of them does not stop
/// short at some earlier, easier-to-satisfy refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_every_i_gang_determinant_satisfied_is_unimplemented() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance("coord-full", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-full",
        "coord-full",
        std::time::Duration::from_secs(30),
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full("job-full", attempt, 0, 1, "coord-full"));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("no HostAdmission session exists yet to admit into");
    assert_eq!(err.code(), tonic::Code::Unimplemented);
}

/// b1': a job id no row exists for is refused `FAILED_PRECONDITION` — the
/// SAME status and message every other I-GANG determinant refuses with
/// (§I1 Non-disclosure), never a distinguishing "not found" text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_job_not_found() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full("no-such-job", 0, 0, 1, "coord-1"));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an absent job must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// b1': a `queued` (never claimed) row is refused `FAILED_PRECONDITION` —
/// the "not `running`" determinant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_job_not_running() {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied — this determinant is isolated from "coordinator
    // not fresh" (a separate conjunct, §I1), never a confound this test
    // would accidentally also exercise.
    server
        .engine
        .catalog()
        .upsert_instance("coord-1", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-completed",
        "coord-1",
        std::time::Duration::from_secs(30),
    )
    .await;
    // Force status away from `running` WITHOUT touching `claimed_by` /
    // `attempts` / the lease, so this row fails ONLY the "not running"
    // conjunct — every other conjunct (`claimant_matches`, `attempt_matches`,
    // `lease_live`) still holds. A row this engine's own claim path can
    // reach this way (`finish_job`), manufactured directly via raw SQL so
    // the fixture does not depend on that path's own guards.
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET status = 'completed' WHERE job_id = 'job-completed'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full("job-completed", attempt, 0, 1, "coord-1"));
    let err = client.run_rank(outbound).await.expect_err(
        "a non-running job must be refused, even with a matching claimant/attempt/lease",
    );
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// b1': a lease claimed for 1ms, then allowed to expire, is refused
/// `FAILED_PRECONDITION` — the "lease not live" determinant (never a
/// `NULL`-vs-expired distinction the caller can observe).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_lease_expired() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied — isolates "lease not live" from "coordinator not
    // fresh".
    server
        .engine
        .catalog()
        .upsert_instance("coord-expired", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-expired",
        "coord-expired",
        std::time::Duration::from_millis(1),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-expired",
        attempt,
        0,
        1,
        "coord-expired",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an expired lease must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// b1': every OTHER I-GANG determinant satisfied, but the coordinator's own
/// `instances` row was never upserted (absent) — refused
/// `FAILED_PRECONDITION`, the "coordinator not fresh" determinant §I1 names
/// beside the row predicate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_coordinator_not_fresh() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Deliberately no `upsert_instance` call for "coord-stale" — every other
    // conjunct is satisfied by a genuine claim.
    let attempt = submit_and_claim(
        &server,
        "job-stale-coord",
        "coord-stale",
        std::time::Duration::from_secs(30),
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-stale-coord",
        attempt,
        0,
        1,
        "coord-stale",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a coordinator with no instances row must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// b1': a caller naming the RIGHT coordinator but the WRONG attempt (a
/// zombie of a prior attempt this job already moved past) is refused
/// `FAILED_PRECONDITION` — the "wrong attempt" determinant, isolated from
/// "not claimed" by using the SAME coordinator the row was actually claimed
/// by.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_attempt_does_not_match() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance("coord-1", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-wrong-attempt",
        "coord-1",
        std::time::Duration::from_secs(30),
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-wrong-attempt",
        attempt + 1,
        0,
        1,
        "coord-1",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a caller naming a stale attempt must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// b1': a job claimed by a DIFFERENT coordinator than the caller names is
/// refused `FAILED_PRECONDITION` — the "not claimed [by this caller]"
/// determinant. Caller-supplied identity is never trusted over the row's
/// own `claimed_by` (I-GANG).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_claimed_by_a_different_coordinator() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied for the NAMED (impostor) coordinator — isolates
    // "not claimed by this caller" from "coordinator not fresh": `run_rank`
    // checks freshness against `assign.coordinator_instance_id`
    // ("coord-impostor"), never the row's own `claimed_by`.
    server
        .engine
        .catalog()
        .upsert_instance("coord-impostor", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-wrong-claimant",
        "coord-real",
        std::time::Duration::from_secs(30),
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-wrong-claimant",
        attempt,
        0,
        1,
        "coord-impostor",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a caller naming a different coordinator than claimed_by must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}
