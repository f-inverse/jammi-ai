//! `GangService`'s admission-wire tests.
//!
//! The first two tests below prove the sidecar-verify hazards directly
//! against `crate::grpc::gang::resolve_training_set_identity` and
//! `jammi_db::Catalog::get_result_table_for_tenant` — the way the sidecar
//! verify states the property (a repo-level predicate plus an explicit
//! guard at the resolution call site), matching `get_job_for_rank`'s own
//! enumerating-caller-oracle treatment elsewhere in this contract (see
//! `docs/rigor/contracts/feat_500-C-U5a-1.md` § A2).
//!
//! The remaining tests drive the real `RunRank` rpc over the production
//! `peer_bind` listener (`start_engine_server_with_peer_bind` /
//! `start_no_worker_server`) — the wire-level K2 edges (`world == 0`,
//! `rank >= world`, refused before I-GANG runs), every I-GANG determinant
//! `get_job_for_rank` decides (job not found, not `running`, wrong
//! claimant, wrong attempt, lease not live); a call satisfying
//! EVERY determinant still ends `UNIMPLEMENTED` — this unit has no
//! `HostAdmission` session to hand it to
//! (docs/plans/67-distributed-training/UNITS.md § U5a-2 builds it).
//!
//! The final group drives `RunRank` itself with `world_size > 1` — the ONLY
//! path through this handler that ever reads a tenant-scoped catalog row
//! (threading `row.tenant_id` into `resolve_training_set_identity`).
//! Every `world_size == 1` test above (and the two direct unit tests at the
//! top of this file, which call `resolve_training_set_identity` directly,
//! never through the RPC) leaves this block cold. These tests drive it
//! through the wire: a job's own `training_set_location` naming a
//! `result_tables` row registered under ANOTHER tenant, or under NO tenant
//! while the job is tenant-bound, must refuse `FAILED_PRECONDITION` without
//! disclosing the other tenant's row; a pair that genuinely resolves for the
//! job's OWN tenant must still reach the handler's own `UNIMPLEMENTED`
//! terminal (proving the sidecar path was actually decided, not merely
//! "always denies"); and the pair missing entirely at `world_size > 1` must
//! refuse the same way.
//!
//! Every `world_size > 1` test below is keyed on the
//! ROW's own `world_size` (a spec naming `{"common":{"world_size":2}}`,
//! `WORLD2_SPEC`), never on the caller's own `assign.world` — a
//! `world_size > 1` job admitted at a
//! caller-supplied `world = 1` would otherwise skip the pair conjunct and
//! the sidecar verify entirely. `run_rank_refuses_when_assign_world_mismatches_row_world_size`
//! is the determinant that closes this: `assign.world != row.world_size`
//! is itself a refusal. Also covered: a training-set row found for the
//! job's OWN tenant but not `ready`, and one that IS `ready` but whose
//! sidecar digest does not match — both refuse the same way as every other
//! determinant.
//!
//! `run_rank_refusal_is_non_disclosing_across_every_determinant`
//! is the ONE table-driven non-disclosure oracle — every I-GANG determinant
//! refuses with the pairwise-identical `(code, message)`. The
//! `test-hooks`-gated `run_rank_last_refusal_reason_distinguishes_every_determinant`
//! drives the exact SAME scenarios and asserts `GangServer::last_refusal_reason`
//! (via `PeerEngineServer::gang_last_refusal_reason`) distinguishes every one
//! of them same-process — the plain lane and the `test-hooks` lane therefore
//! run a DIFFERENT number of gang-prefixed test cases (stated at each test).

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

/// The strict-predicate property: a NULL-tenant `result_tables` row exists
/// (as if created by a coordinator outside any tenant scope); from tenant A,
/// the RELAXED `get_result_table` (the other read seam) returns it — the
/// hazard this predicate guards against — while the STRICT
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
         (the only relaxed seam) — the hazard this row demonstrates"
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

/// The admin-scope-guard property: a resolution wrapped in
/// `with_admin_scope` would resolve any tenant's table through the raw
/// verb, so the resolution site guards it explicitly instead.
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
    // manifest sidecar and marks the row `ready` regardless of row
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
    // comes back despite the caller naming tenant A. This is the hazard the
    // resolution-site guard exists for: the verb's own behaviour under
    // admin scope is unchanged.
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
         (unchanged verb behaviour) — the hazard the resolution-site guard exists for"
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
// Wire-level K2 edges (`world == 0`, `rank >= world`), decided before
// I-GANG runs.
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

/// A job whose `spec` names no `world_size` at all — `RankAdmissionRow`
/// decodes it to `1`, `jammi_db`'s own `WORLD_SIZE_IF_ABSENT` default (see
/// `jobs_repo.rs`), the shape a non-training job kind (or a spec predating
/// `world_size`) persists.
const WORLD1_SPEC: &str = "{}";

/// A job whose `spec` names `world_size: 2` under the `common` key — the
/// SAME shape `jammi-ai`'s `TrainingCommon` actually persists, and the SAME
/// literal `crates/jammi-db/tests/it/gang_rank_admission.rs` uses for its own
/// `get_job_for_rank_reflects_the_row_world_size` fixture. Every
/// `world_size > 1` test in this file submits with THIS spec — never a
/// `world_size == 1` spec paired with an `Assign.world == 2`, which the
/// world-mismatch conjunct refuses before the pair conjunct or the sidecar
/// verify ever runs.
const WORLD2_SPEC: &str = r#"{"common":{"world_size":2}}"#;

/// A job submitted and claimed on `server`'s own engine catalog directly
/// (bypassing the wire `JobService`, matching `jobs_queue.rs`'s own fixture
/// style — the gang admission surface reads the row, not the submission
/// RPC), with `spec` controlling the row's own `world_size` — never the
/// caller's `Assign.world`. `model_ref: None` avoids needing a registered
/// model FK target (the column is nullable). Returns the `attempts` value
/// the claim landed at (always `1`, the first claim), for the caller to
/// build a matching `Assign` frame with.
async fn submit_and_claim(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
    coordinator_instance_id: &str,
    lease: std::time::Duration,
    spec: &str,
) -> i64 {
    use jammi_db::catalog::jobs_repo::SubmitJobParams;
    use jammi_db::catalog::status::JobExecution;

    let catalog = server.engine.catalog();
    catalog
        .submit_job(SubmitJobParams {
            job_id,
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec,
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
// I-GANG (the full row predicate) + the not-yet-implemented terminal state
// ---------------------------------------------------------------------------

/// A call satisfying EVERY I-GANG
/// determinant — the row is `running`, claimed by the caller's own
/// `coordinator_instance_id`, at the matching `attempt`, under a live
/// lease, and (`world_size == 1` here, so the training-set pair is not
/// gated) the coordinator's own `instances` row is fresh — still reaches
/// `UNIMPLEMENTED`: this unit has no `HostAdmission` session to hand the
/// call to (docs/plans/67-distributed-training/UNITS.md § U5a-2 builds it).
/// Proves every determinant was actually
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
        WORLD1_SPEC,
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

/// A job id no row exists for is refused `FAILED_PRECONDITION` — the
/// SAME status and message every other I-GANG determinant refuses with
/// (non-disclosure), never a distinguishing "not found" text.
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

/// A `queued` (never claimed) row is refused `FAILED_PRECONDITION` —
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
        WORLD1_SPEC,
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

/// A lease claimed for 1ms, then allowed to expire, is refused
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
        WORLD1_SPEC,
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

/// Every OTHER I-GANG determinant satisfied, but the coordinator's own
/// `instances` row was never upserted (absent) — refused
/// `FAILED_PRECONDITION`, the "coordinator not fresh" determinant I-GANG
/// names beside the row predicate.
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
        WORLD1_SPEC,
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

/// A caller naming the RIGHT coordinator but the WRONG attempt (a
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
        WORLD1_SPEC,
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

/// A job claimed by a DIFFERENT coordinator than the caller names is
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
        WORLD1_SPEC,
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

// ---------------------------------------------------------------------------
// §I1(b) (world_size > 1), through the RPC: the ONLY tenant-scoped catalog
// read `run_rank` ever performs. No test above ever sets `world > 1` on a
// row that reaches this block (the lone `world == 2` frame, above, is
// refused at the §W1 K2 edge before any row is read) — these tests are the
// first to drive it.
// ---------------------------------------------------------------------------

/// A job submitted under an explicit tenant scope — mirroring
/// [`submit_and_claim`], but pinning the row's own `tenant_id` the way a
/// genuine tenant-bound submission would, via `with_tenant_scoped`
/// (`submit_job`'s own ambient `current_tenant()` picks it up) — then
/// claimed exactly like [`submit_and_claim`]. `claim_next` itself carries no
/// tenant predicate at all (the production worker claims globally, then
/// reads tenant off the claimed row), so the claim step runs unscoped.
/// Returns the claimed `attempts` value.
async fn submit_and_claim_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    job_id: &str,
    coordinator_instance_id: &str,
    lease: std::time::Duration,
    spec: &str,
) -> i64 {
    use jammi_db::catalog::jobs_repo::SubmitJobParams;
    use jammi_db::catalog::status::JobExecution;

    let job_id_owned = job_id.to_string();
    let spec_owned = spec.to_string();
    server
        .engine
        .with_tenant_scoped(tenant, move |scope| {
            let job_id = job_id_owned.clone();
            let spec = spec_owned.clone();
            async move {
                scope
                    .catalog()
                    .submit_job(SubmitJobParams {
                        job_id: &job_id,
                        kind: "fine_tune",
                        execution: JobExecution::Queued,
                        spec: &spec,
                        model_ref: None,
                        output_model_id: None,
                        model_source: None,
                        priority: 0,
                    })
                    .await
                    .unwrap()
            }
        })
        .await;
    let claimed = server
        .engine
        .catalog()
        .claim_next(coordinator_instance_id, &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("must claim the only queued job");
    i64::from(claimed.attempts)
}

/// Materializes a REAL `ready` `result_tables` row (Parquet + sidecar
/// manifest, the identical fixture shape
/// `resolution_site_refuses_under_admin_scope_even_when_the_raw_verb_would_resolve`
/// uses above) under `tenant`, and returns the `(training_set_location,
/// training_set_ref)` pair a genuine coordinator's
/// `fill_training_set_identity` call would carry: the table name, and the
/// digest read back from the SAME sidecar `resolve_training_set_identity`
/// itself verifies against.
async fn materialize_ready_table_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    source_id: &str,
) -> (String, String) {
    use datafusion::prelude::SessionContext;
    use jammi_db::store::manifest::{
        ComputeDevice, ComputePrecision, InputAnchor, Materialization, MaterializationEnv,
        ModelContentDigest, ModelIdentity, ProducingDescriptor,
    };
    use jammi_db::store::EmbeddingTableSpec;

    let descriptor = ProducingDescriptor::Embedding {
        model_id: "rt-base".into(),
        task: ModelTask::TextEmbedding,
        source_id: source_id.to_string(),
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
    let source_id_owned = source_id.to_string();
    let engine_for_scope = Arc::clone(&server.engine);
    let record = server
        .engine
        .with_tenant_scoped(tenant, move |_scope| async move {
            let store = engine_for_scope.result_store();
            store
                .materialize_embedding_table(
                    &ctx,
                    EmbeddingTableSpec {
                        source_id: &source_id_owned,
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
                        vec![InputAnchor::mutable_version(&source_id_owned, 1)],
                    ),
                    None,
                )
                .await
                .unwrap()
        })
        .await;
    let store = server.engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("finish() must have written the sidecar");
    (record.table_name.clone(), manifest.artifact.0.clone())
}

/// §I1(b) tenant isolation: `training_set_location` names a `result_tables`
/// row that genuinely resolves and verifies — but for ANOTHER tenant than
/// the one `get_job_for_rank` resolves the calling job under. Refused
/// `FAILED_PRECONDITION`, the SAME status and (§I1 Non-disclosure) the SAME
/// fixed message every other I-GANG determinant refuses with — the response
/// discloses neither the other tenant's id nor its table name. Mutation
/// proof: deleting `run_rank`'s entire `if assign.world > 1 { .. }` block
/// (the only tenant-scoped catalog read this handler performs) leaves the
/// full `jammi-server` `it` suite green — this test, driving the RPC at
/// `world == 2` on a filled pair, is what catches that deletion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_training_set_another_tenant_owns() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let tenant_owner = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let tenant_caller = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();
    let source_id = format!(
        "gang_cross_tenant_src_{}",
        jammi_test_utils::unique_suffix()
    );
    let (table, digest) =
        materialize_ready_table_for_tenant(&server, tenant_owner, &source_id).await;

    server
        .engine
        .catalog()
        .upsert_instance("coord-cross-tenant", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim_for_tenant(
        &server,
        tenant_caller,
        "job-cross-tenant",
        "coord-cross-tenant",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    let outcome = server
        .engine
        .catalog()
        .fill_training_set_identity(
            "job-cross-tenant",
            "coord-cross-tenant",
            attempt as u32,
            &digest,
            &table,
        )
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            jammi_db::catalog::jobs_repo::TrainingSetFillOutcome::Filled
        ),
        "the fixture's own CAS must land cleanly on a freshly claimed row"
    );

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-cross-tenant",
        attempt,
        0,
        2,
        "coord-cross-tenant",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a training set owned by another tenant must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert_eq!(
        err.message(),
        "gang admission refused",
        "the fixed §I1 Non-disclosure message, never a distinguishing reason"
    );
    assert!(
        !err.message().contains(&table) && !err.message().contains(&tenant_owner.to_string()),
        "the refusal must disclose neither the other tenant's table name nor its id, got: {}",
        err.message()
    );
}

/// §I1(b) tenant isolation, the NULL-tenant variant: `training_set_location`
/// names a `result_tables` row created with NO tenant scope active (as if a
/// coordinator materialized it outside any tenant binding) while the
/// calling job IS tenant-bound. The strict resolver
/// (`get_result_table_for_tenant`) never matches a NULL-tenant row for a
/// real tenant (the same property the direct unit test at the top of this
/// file proves against the resolver alone) — refused `FAILED_PRECONDITION`
/// through the RPC too, never resolving the orphaned row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let tenant_caller = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e03").unwrap();
    let table = format!("gang_null_tenant_rpc_{}", jammi_test_utils::unique_suffix());
    // Created with NO tenant scope active, exactly like this file's
    // `strict_resolver_never_matches_a_null_tenant_row_for_a_real_tenant` —
    // the row's `tenant_id` lands NULL.
    server
        .engine
        .catalog()
        .create_result_table(null_tenant_row(&table))
        .await
        .unwrap();

    server
        .engine
        .catalog()
        .upsert_instance("coord-null-tenant", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim_for_tenant(
        &server,
        tenant_caller,
        "job-null-tenant-table",
        "coord-null-tenant",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    server
        .engine
        .catalog()
        .fill_training_set_identity(
            "job-null-tenant-table",
            "coord-null-tenant",
            attempt as u32,
            "any-digest",
            &table,
        )
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-null-tenant-table",
        attempt,
        0,
        2,
        "coord-null-tenant",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a NULL-tenant training set must be refused for a tenant-bound job");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert_eq!(err.message(), "gang admission refused");
}

/// At `world_size > 1`: a training-set pair that genuinely resolves and
/// verifies for the job's OWN tenant (materialized and filled under the SAME
/// tenant `get_job_for_rank` resolves the job under) still reaches this
/// unit's own terminal — `UNIMPLEMENTED`, never `FAILED_PRECONDITION`.
/// Without this test, the three refusal tests above could all pass for the
/// wrong reason (a handler that unconditionally refuses every `world_size >
/// 1` call, tenant match or not) — this is the admitting case that proves
/// the tenant-scoped path was actually DECIDED, not skipped to an easy
/// refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_world_two_own_tenant_training_set_reaches_unimplemented() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let tenant = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e04").unwrap();
    let source_id = format!("gang_own_tenant_src_{}", jammi_test_utils::unique_suffix());
    let (table, digest) = materialize_ready_table_for_tenant(&server, tenant, &source_id).await;

    server
        .engine
        .catalog()
        .upsert_instance("coord-own-tenant", Some("label"), Some("host"))
        .await
        .unwrap();
    let attempt = submit_and_claim_for_tenant(
        &server,
        tenant,
        "job-own-tenant-table",
        "coord-own-tenant",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    server
        .engine
        .catalog()
        .fill_training_set_identity(
            "job-own-tenant-table",
            "coord-own-tenant",
            attempt as u32,
            &digest,
            &table,
        )
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-own-tenant-table",
        attempt,
        0,
        2,
        "coord-own-tenant",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("no HostAdmission session exists yet to admit into");
    assert_eq!(err.code(), tonic::Code::Unimplemented);
}

/// §I1(b): at `world_size > 1`, the training-set pair itself missing
/// (never filled) is refused `FAILED_PRECONDITION` — isolated from every
/// other I-GANG determinant, which all hold (a genuine claim, live lease,
/// fresh coordinator).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_gt_one_when_training_set_pair_missing() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance("coord-no-pair", Some("label"), Some("host"))
        .await
        .unwrap();
    // `submit_and_claim` never fills the training-set pair — it stays NULL.
    let attempt = submit_and_claim(
        &server,
        "job-no-pair",
        "coord-no-pair",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-no-pair",
        attempt,
        0,
        2,
        "coord-no-pair",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("world_size > 1 with an unset training-set pair must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert_eq!(err.message(), "gang admission refused");
}

// ---------------------------------------------------------------------------
// The world-mismatch determinant, the non-disclosure oracle, and the
// `test-hooks` seam.
// ---------------------------------------------------------------------------

/// A GENUINELY resolvable `result_tables` row for `tenant` — real Parquet,
/// real sidecar manifest, the identical fixture shape
/// [`materialize_ready_table_for_tenant`] builds — forced back to `status =
/// 'building'` via raw SQL AFTER materialization finishes (mirroring
/// `run_rank_refuses_when_job_not_running`'s own technique above: manufacture
/// the ONE conjunct under test directly, never through a path whose own
/// guards could refuse the setup itself). This is the property that matters:
/// a `null_tenant_row`-style row whose `parquet_path` never resolves would
/// ALSO fail the sidecar verify once found — making a "drop the Ready
/// conjunct" mutation invisible, since the outcome (`FailedPrecondition`)
/// would stay the same for the WRONG reason. Returns the `(table,
/// training_set_ref)` pair, exactly like [`materialize_ready_table_for_tenant`]
/// — this row verifies fine; only its `status` isolates the Ready conjunct.
async fn create_not_ready_but_verifiable_table_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    source_id: &str,
) -> (String, String) {
    use jammi_db::catalog::backend::TxOptions;

    let (table, digest) = materialize_ready_table_for_tenant(server, tenant, source_id).await;
    // Table names here are test-controlled (`jammi_test_utils::unique_suffix`
    // suffixed, never external input) — a fixed literal SQL string built the
    // same way `run_rank_refuses_when_job_not_running`'s own mutation above
    // does, never a parameterized statement this fixture needs `SqlValue`
    // plumbing for.
    let sql = format!("UPDATE result_tables SET status = 'building' WHERE table_name = '{table}'");
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let sql = sql.clone();
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await
        .unwrap();
    (table, digest)
}

/// Drives ONE `RunRank` call end-to-end for a NAMED
/// [`jammi_server::grpc::gang::GangRefusalReason`] — every
/// scenario satisfies every OTHER I-GANG determinant, isolating the named
/// one, the same discipline the individual determinant tests earlier in
/// this file already follow, collected here ONCE so both the plain lane's
/// pairwise non-disclosure oracle and the `test-hooks` lane's
/// reason-distinguishing oracle drive the identical fixtures — never two
/// copies of this setup that could quietly drift apart. Returns the server
/// the call was driven against (a `test-hooks` caller reads
/// `PeerEngineServer::gang_last_refusal_reason` off this SAME instance
/// afterward) paired with the `Status` the RPC returned.
async fn refusal_scenario(
    reason: jammi_server::grpc::gang::GangRefusalReason,
) -> (crate::common::grpc::PeerEngineServer, tonic::Status) {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let outbound_frame = match reason {
        GangRefusalReason::NotRunning => {
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-not-running", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-not-running",
                "nd-coord-not-running",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            server
                .engine
                .catalog()
                .backend_arc()
                .transaction(TxOptions::default(), |tx| {
                    Box::pin(async move {
                        tx.execute(
                            "UPDATE jobs SET status = 'completed' WHERE job_id = 'nd-job-not-running'",
                            &[],
                        )
                        .await
                    })
                })
                .await
                .unwrap();
            assign_frame_full("nd-job-not-running", attempt, 0, 1, "nd-coord-not-running")
        }
        GangRefusalReason::WrongClaimant => {
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-impostor", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-wrong-claimant",
                "nd-coord-real",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full("nd-job-wrong-claimant", attempt, 0, 1, "nd-coord-impostor")
        }
        GangRefusalReason::WrongAttempt => {
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-wrong-attempt", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-wrong-attempt",
                "nd-coord-wrong-attempt",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-wrong-attempt",
                attempt + 1,
                0,
                1,
                "nd-coord-wrong-attempt",
            )
        }
        GangRefusalReason::LeaseDead => {
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-lease-dead", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-lease-dead",
                "nd-coord-lease-dead",
                std::time::Duration::from_millis(1),
                WORLD1_SPEC,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assign_frame_full("nd-job-lease-dead", attempt, 0, 1, "nd-coord-lease-dead")
        }
        GangRefusalReason::NotFound => {
            assign_frame_full("nd-job-not-found", 0, 0, 1, "nd-coord-not-found")
        }
        GangRefusalReason::CoordinatorNotFresh => {
            let attempt = submit_and_claim(
                &server,
                "nd-job-coord-not-fresh",
                "nd-coord-stale",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full("nd-job-coord-not-fresh", attempt, 0, 1, "nd-coord-stale")
        }
        GangRefusalReason::WorldMismatch => {
            // Strengthened (lead mutation `if false && assign.world !=
            // row.world_size` survived against the original fixture here):
            // the row must be OTHERWISE ADMISSIBLE at its own `world_size`
            // (2) — own tenant, `ready`, digest-verified pair, fresh
            // coordinator, exactly `run_rank_world_two_own_tenant_
            // training_set_reaches_unimplemented`'s own fixture shape —
            // so the mismatch conjunct is the ONLY arm that can refuse
            // `assign.world = 1` against it. The original fixture (no
            // tenant, no pair) let the deleted-gate mutation survive: with
            // the mismatch conjunct gone, the SAME request fell into the
            // `row.world_size > 1` block and was refused by the UNFILLED
            // PAIR conjunct instead — the SAME fixed message, so the
            // pairwise/RPC-level oracles could not tell which arm refused.
            let tenant = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e15").unwrap();
            let source_id = format!(
                "gang_nd_world_mismatch_{}",
                jammi_test_utils::unique_suffix()
            );
            let (table, digest) =
                materialize_ready_table_for_tenant(&server, tenant, &source_id).await;
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-world-mismatch", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim_for_tenant(
                &server,
                tenant,
                "nd-job-world-mismatch",
                "nd-coord-world-mismatch",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            server
                .engine
                .catalog()
                .fill_training_set_identity(
                    "nd-job-world-mismatch",
                    "nd-coord-world-mismatch",
                    attempt as u32,
                    &digest,
                    &table,
                )
                .await
                .unwrap();
            // The row's own `world_size` is 2 (`WORLD2_SPEC`) and is
            // otherwise fully admissible at world 2 (see the control call
            // `run_rank_refuses_when_assign_world_mismatches_row_world_size`
            // drives against this SAME row, below); the caller names
            // `world = 1` — case (f). This is the ONLY conjunct that can
            // refuse this row.
            assign_frame_full(
                "nd-job-world-mismatch",
                attempt,
                0,
                1,
                "nd-coord-world-mismatch",
            )
        }
        GangRefusalReason::TrainingSetPairMissing => {
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-pair-missing", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-pair-missing",
                "nd-coord-pair-missing",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-pair-missing",
                attempt,
                0,
                2,
                "nd-coord-pair-missing",
            )
        }
        GangRefusalReason::TrainingSetOtherTenant => {
            let tenant_owner = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e11").unwrap();
            let tenant_caller = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e12").unwrap();
            let source_id = format!("gang_nd_other_tenant_{}", jammi_test_utils::unique_suffix());
            let (table, digest) =
                materialize_ready_table_for_tenant(&server, tenant_owner, &source_id).await;
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-other-tenant", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim_for_tenant(
                &server,
                tenant_caller,
                "nd-job-other-tenant",
                "nd-coord-other-tenant",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            server
                .engine
                .catalog()
                .fill_training_set_identity(
                    "nd-job-other-tenant",
                    "nd-coord-other-tenant",
                    attempt as u32,
                    &digest,
                    &table,
                )
                .await
                .unwrap();
            assign_frame_full(
                "nd-job-other-tenant",
                attempt,
                0,
                2,
                "nd-coord-other-tenant",
            )
        }
        GangRefusalReason::TrainingSetNotReady => {
            let tenant = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e13").unwrap();
            let source_id = format!("gang_nd_not_ready_{}", jammi_test_utils::unique_suffix());
            let (table, digest) =
                create_not_ready_but_verifiable_table_for_tenant(&server, tenant, &source_id).await;
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-not-ready", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim_for_tenant(
                &server,
                tenant,
                "nd-job-not-ready",
                "nd-coord-not-ready",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            server
                .engine
                .catalog()
                .fill_training_set_identity(
                    "nd-job-not-ready",
                    "nd-coord-not-ready",
                    attempt as u32,
                    &digest,
                    &table,
                )
                .await
                .unwrap();
            assign_frame_full("nd-job-not-ready", attempt, 0, 2, "nd-coord-not-ready")
        }
        GangRefusalReason::TrainingSetDigestMismatch => {
            let tenant = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e14").unwrap();
            let source_id = format!(
                "gang_nd_digest_mismatch_{}",
                jammi_test_utils::unique_suffix()
            );
            let (table, digest) =
                materialize_ready_table_for_tenant(&server, tenant, &source_id).await;
            server
                .engine
                .catalog()
                .upsert_instance("nd-coord-digest-mismatch", Some("l"), Some("h"))
                .await
                .unwrap();
            let attempt = submit_and_claim_for_tenant(
                &server,
                tenant,
                "nd-job-digest-mismatch",
                "nd-coord-digest-mismatch",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            let wrong_digest = format!("{digest}-wrong");
            server
                .engine
                .catalog()
                .fill_training_set_identity(
                    "nd-job-digest-mismatch",
                    "nd-coord-digest-mismatch",
                    attempt as u32,
                    &wrong_digest,
                    &table,
                )
                .await
                .unwrap();
            assign_frame_full(
                "nd-job-digest-mismatch",
                attempt,
                0,
                2,
                "nd-coord-digest-mismatch",
            )
        }
    };

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(outbound_frame);
    let status = client
        .run_rank(outbound)
        .await
        .expect_err("every fixture this function builds must refuse");
    (server, status)
}

/// §I1(b): at `world_size > 1`, a training-set row found for the job's OWN
/// tenant but NOT `ready` is refused `FAILED_PRECONDITION` — case (e), the
/// determinant no test before this round covered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_gt_one_when_training_set_not_ready() {
    use jammi_server::grpc::gang::GangRefusalReason;

    let (_server, status) = refusal_scenario(GangRefusalReason::TrainingSetNotReady).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
}

/// §I1(b): at `world_size > 1`, a `ready` training-set row whose sidecar
/// digest does NOT match `training_set_ref` is refused `FAILED_PRECONDITION`
/// — case (d), the determinant no test before this round covered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_gt_one_when_training_set_digest_mismatches() {
    use jammi_server::grpc::gang::GangRefusalReason;

    let (_server, status) = refusal_scenario(GangRefusalReason::TrainingSetDigestMismatch).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
}

/// The world-mismatch determinant — case (f): `assign.world` disagreeing with the
/// ROW's own `world_size` (`WORLD2_SPEC`'s `2` here, named `world = 1`) is
/// itself a refusal, the SAME fixed message every other I-GANG determinant
/// refuses with.
///
/// The row `refusal_scenario`'s `WorldMismatch` arm builds is OTHERWISE
/// ADMISSIBLE at its own `world_size` (own tenant, `ready`, digest-verified
/// pair, fresh coordinator) — so the mismatch conjunct is the ONLY arm that
/// can refuse `assign.world = 1` against it (a lead mutation,
/// `if false && assign.world != row.world_size`, survived against an
/// earlier fixture with no training-set pair filled: with the mismatch
/// conjunct gone, that request fell into the `row.world_size > 1` block and
/// was refused by the unfilled-pair conjunct instead, the SAME fixed
/// message, so the oracle could not tell which arm actually refused). The
/// control below drives the SAME row a second time at `assign.world = 2`
/// (matching its own `world_size`) and asserts it reaches `UNIMPLEMENTED` —
/// proving world 1 above refused for the mismatch alone, never because this
/// row was unresolvable for some other reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_assign_world_mismatches_row_world_size() {
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let (server, status) = refusal_scenario(GangRefusalReason::WorldMismatch).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    {
        assert_eq!(
            server.gang_last_refusal_reason(),
            Some(GangRefusalReason::WorldMismatch),
            "the served GangServer must record WorldMismatch, not some other determinant, \
             for this exact request"
        );
    }

    // Control: the SAME row (same job, same claim, same genuinely-verified
    // pair), driven a SECOND time on this SAME server, at `assign.world = 2`
    // (matching the row's own `world_size`) — admits all the way to
    // `UNIMPLEMENTED`, `run_rank_world_two_own_tenant_training_set_reaches_
    // unimplemented`'s own outcome (case c). `run_rank` never mutates the
    // `jobs` row (a read-only classification), so re-driving the identical
    // job here is sound: nothing about the row's own state changed between
    // the two calls. `attempt = 1`: `refusal_scenario`'s `WorldMismatch` arm
    // calls `submit_and_claim_for_tenant` exactly once on a freshly
    // submitted job with no prior claim — `Catalog::claim_next`'s first
    // (and only) claim always lands at `attempts = 1` (documented on
    // `submit_and_claim`/`submit_and_claim_for_tenant` above: "always `1`,
    // the first claim"). Reading the row back via `get_job_for_rank` here
    // instead would ADD a second call site outside this crate's own
    // `RunRank` handler — `only_the_gang_run_rank_handler_calls_get_job_
    // for_rank` (`gang_rank_admission_oracle.rs`) enumerates every caller of
    // that primary-key-only, non-tenant-scoped verb and would (correctly)
    // fail on exactly that.
    let attempt = 1;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "nd-job-world-mismatch",
        attempt,
        0,
        2,
        "nd-coord-world-mismatch",
    ));
    let control_err = client
        .run_rank(outbound)
        .await
        .expect_err("no HostAdmission session exists yet to admit into");
    assert_eq!(
        control_err.code(),
        tonic::Code::Unimplemented,
        "the SAME row at its own world_size must admit, proving world 1 above refused for \
         the mismatch alone"
    );
}

/// The full set of I-GANG determinants [`refusal_scenario`] can drive,
/// shared by the plain lane's non-disclosure oracle and the `test-hooks`
/// lane's reason-distinguishing oracle below — one definition, so adding a
/// determinant to one automatically covers it in the other.
fn every_gang_refusal_reason() -> [jammi_server::grpc::gang::GangRefusalReason; 11] {
    use jammi_server::grpc::gang::GangRefusalReason;
    [
        GangRefusalReason::NotRunning,
        GangRefusalReason::WrongClaimant,
        GangRefusalReason::WrongAttempt,
        GangRefusalReason::LeaseDead,
        GangRefusalReason::NotFound,
        GangRefusalReason::CoordinatorNotFresh,
        GangRefusalReason::WorldMismatch,
        GangRefusalReason::TrainingSetPairMissing,
        GangRefusalReason::TrainingSetOtherTenant,
        GangRefusalReason::TrainingSetNotReady,
        GangRefusalReason::TrainingSetDigestMismatch,
    ]
}

/// The ONE non-disclosure oracle. Every I-GANG determinant
/// (not running / wrong claimant / wrong attempt / lease dead / not found /
/// coordinator not fresh / world mismatch / pair NULL / other tenant /
/// digest mismatch / not Ready — eleven total) refuses with the
/// PAIRWISE-IDENTICAL `(code, message)` — compared pairwise so a single
/// differing pair fails naming exactly that pair, never merely "some
/// determinant's message differs somewhere". Mutation proof: make any ONE
/// determinant's message leak (e.g. append the reason) and this fails,
/// naming the leaking pair.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refusal_is_non_disclosing_across_every_determinant() {
    let reasons = every_gang_refusal_reason();
    let mut statuses = Vec::with_capacity(reasons.len());
    for reason in reasons {
        let (_server, status) = refusal_scenario(reason).await;
        statuses.push((reason, status));
    }
    for i in 0..statuses.len() {
        for j in (i + 1)..statuses.len() {
            let (reason_a, status_a) = &statuses[i];
            let (reason_b, status_b) = &statuses[j];
            assert_eq!(
                (status_a.code(), status_a.message()),
                (status_b.code(), status_b.message()),
                "{reason_a:?} and {reason_b:?} must refuse with the pairwise-identical \
                 (code, message) — non-disclosure requires this for every pair, not just \
                 some of them"
            );
        }
    }
}

/// `test-hooks` only: drives the SAME eleven scenarios
/// [`run_rank_refusal_is_non_disclosing_across_every_determinant`] does, but
/// asserts `PeerEngineServer::gang_last_refusal_reason` names the EXACT
/// determinant each one refused for — the seam that lets this lane
/// distinguish what the plain lane's own non-disclosure oracle just proved
/// is (correctly) indistinguishable on the wire. This is what makes the
/// `test-hooks` lane's `cargo test -p jammi-server --test it -- gang`
/// execute ONE MORE gang-prefixed test-fn than the plain lane (this
/// function itself is compiled only under `test-hooks`; the plain lane's
/// case above still runs the same eleven RPC calls, just without this
/// additional reason assertion).
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_last_refusal_reason_distinguishes_every_determinant() {
    for reason in every_gang_refusal_reason() {
        let (server, status) = refusal_scenario(reason).await;
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
        assert_eq!(
            server.gang_last_refusal_reason(),
            Some(reason),
            "the served GangServer must record {reason:?} for this scenario, distinguishing \
             it from every other determinant same-process, without that distinction ever \
             reaching the wire"
        );
    }
}
