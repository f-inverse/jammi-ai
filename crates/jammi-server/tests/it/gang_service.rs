//! `GangService`'s admission-wire tests.
//!
//! The tests below drive the real `RunRank` rpc over the production
//! `peer_bind` listener (`start_engine_server_with_peer_bind` /
//! `start_no_worker_server`) — the wire-level K2 edges (`world == 0`,
//! `rank >= world`, refused before I-GANG runs), ambient admin scope
//! (refused before any row is even read), every I-GANG determinant
//! `get_job_for_rank` decides (job not found, not `running`, wrong
//! claimant, wrong attempt, lease not live, an undecodable `world_size`);
//! a call satisfying EVERY determinant still ends `UNIMPLEMENTED` — this
//! unit has no `HostAdmission` session to hand it to
//! (docs/plans/67-distributed-training/UNITS.md § U5a-2 builds it).
//!
//! **This unit ships the `world_size == 1` lattice only.** A row whose own
//! `world_size` (decoded from `spec`, never the caller's `assign.world`) is
//! not exactly `1` refuses the SAME fixed way every other determinant does
//! — whether or not the caller's `assign.world` happens to agree with it —
//! since the training-set pair conjunct and its sidecar verify that would
//! admit a genuine multi-host row are `HostAdmission`'s to build
//! (docs/plans/67-distributed-training/UNITS.md § U5a-2); this handler never
//! attempts them.
//! `run_rank_refuses_when_assign_world_mismatches_row_world_size` drives
//! BOTH directions of `assign.world != row.world_size` (below the row's own
//! value, and above it); `run_rank_refuses_world_gt_one_when_matching_caller_world`
//! drives the case where `assign.world` genuinely AGREES with a row's own
//! `world_size > 1` — refused all the same, distinguishably
//! (`test-hooks`) from a genuine mismatch.
//!
//! `run_rank_refusal_is_non_disclosing_across_every_determinant`
//! is the ONE table-driven non-disclosure oracle — every I-GANG determinant
//! refuses with the pairwise-identical `(code, message)`. The
//! `test-hooks`-gated `run_rank_last_refusal_reason_distinguishes_every_determinant`
//! drives the exact SAME scenarios and asserts `GangServer::last_refusal_reason`
//! (via `PeerEngineServer::gang_last_refusal_reason`, or — for the
//! `AdminScope` determinant, which no network client can ever trigger — a
//! standalone `GangServer` invoked in-process, see `refusal_scenario`'s own
//! doc) distinguishes every one of them same-process — the plain lane and
//! the `test-hooks` lane therefore run a DIFFERENT number of gang-prefixed
//! test cases (stated at each test).

use std::sync::Arc;

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

/// Wire-level K2 (`docs/rigor/contracts/feat_500-C-U5a-1.md` §1.2):
/// `world == 0` is refused `INVALID_ARGUMENT`, before I-GANG (which
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

/// Wire-level K2 (`docs/rigor/contracts/feat_500-C-U5a-1.md` §1.2):
/// `rank >= world` is refused `INVALID_ARGUMENT` — the boundary case
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-full",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
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
    // not fresh" (a separate conjunct, `docs/rigor/contracts/feat_500-C-U5a-1.md`
    // §1.5), never a confound this test would accidentally also exercise.
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-1",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-expired",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-1",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-impostor",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
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

/// A genuine catalog fault reading `Catalog::fresh_instance` (the
/// `instances` table itself gone, mirroring `jammi-db`'s own
/// `get_job_for_rank_driver_fault_still_surfaces_as_err` DROP-TABLE
/// technique) maps through `admission_catalog_fault` to `Unavailable` — the
/// SAME admission-time classification every other catalog read on this path
/// uses, never `map_engine_error`'s generic mapping (which has no case for a
/// raw backend fault and would fall through to `Internal`). Every OTHER
/// I-GANG determinant is satisfied by construction, isolating this fault.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_fresh_instance_fault_is_unavailable() {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-fresh-fault",
            Some("l"),
            Some("h"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-fresh-fault",
        "coord-fresh-fault",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move { tx.execute("DROP TABLE instances", &[]).await })
        })
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-fresh-fault",
        attempt,
        0,
        1,
        "coord-fresh-fault",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a dropped instances table must surface as a genuine catalog fault");
    assert_eq!(
        err.code(),
        tonic::Code::Unavailable,
        "fresh_instance erroring must map through admission_catalog_fault, never \
         map_engine_error's generic mapping"
    );
}

// ---------------------------------------------------------------------------
// The world-mismatch determinant, the non-disclosure oracle, and the
// `test-hooks` seam.
// ---------------------------------------------------------------------------

/// Builds a `tonic::Streaming<RankControl>` by hand from `frames`,
/// length-prefixed exactly the way the real gRPC wire frames a message
/// (`tonic-prost`'s own codec tests use the identical shape: a `0`
/// compression byte, a big-endian `u32` length, then the encoded bytes) —
/// never a real network connection. The ONLY caller needing this is
/// [`refusal_scenario`]'s `AdminScope` arm: ambient admin scope is
/// per-process task-local state that never crosses a network hop (a client
/// cannot set it on the server's own request-handling task, nor does it
/// survive a `tokio::spawn` boundary), so exercising that determinant
/// requires calling `GangServer::run_rank` directly, in the SAME task tree a
/// test wraps in `with_admin_scope` — which needs a real `tonic::Streaming`
/// value to satisfy `run_rank`'s own signature, not a bare `Stream`.
fn streaming_of(
    frames: Vec<jammi_wire::proto::gang::RankControl>,
) -> tonic::Streaming<jammi_wire::proto::gang::RankControl> {
    use bytes::{BufMut, Bytes, BytesMut};
    use jammi_wire::proto::gang::RankControl;
    use prost::Message;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tonic::codec::BufferSettings;
    use tonic_prost::ProstCodec;

    /// A one-shot `http_body::Body` handing back every framed byte in a
    /// single poll — sufficient since `tonic::Streaming`'s own decode loop
    /// reads incrementally from whatever bytes a `poll_frame` call yields,
    /// never requiring one HTTP frame per gRPC message.
    struct OnceBody(Option<Bytes>);
    impl http_body::Body for OnceBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;
        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(self.0.take().map(|b| Ok(http_body::Frame::data(b))))
        }
    }

    let mut buf = BytesMut::new();
    for frame in frames {
        let mut msg = BytesMut::new();
        frame.encode(&mut msg).expect("encode RankControl");
        buf.put_u8(0); // uncompressed
        buf.put_u32(msg.len() as u32);
        buf.put_slice(&msg);
    }

    let decoder = ProstCodec::<RankControl, RankControl>::raw_decoder(BufferSettings::default());
    tonic::Streaming::new_request(decoder, OnceBody(Some(buf.freeze())), None, None)
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
///
/// `GangRefusalReason::AdminScope` is the ONE exception to "drives it over
/// the real listener": no network client can ever make ambient admin scope
/// visible on the SERVER's own request-handling task (see [`streaming_of`]'s
/// own doc), so this arm builds a STANDALONE `GangServer` from the SAME
/// `server.engine` (never the instance mounted on `server.peer_addr`) and
/// calls `run_rank` on it DIRECTLY, inside `server.engine.with_admin_scope`
/// — swapping the standalone server's (`test-hooks` only) refusal handle
/// onto the returned `server` so `PeerEngineServer::gang_last_refusal_reason`
/// still observes the call this function actually made.
async fn refusal_scenario(
    reason: jammi_server::grpc::gang::GangRefusalReason,
) -> (crate::common::grpc::PeerEngineServer, tonic::Status) {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    #[allow(unused_mut)]
    let mut server = start_no_worker_server().await;

    if reason == GangRefusalReason::AdminScope {
        use jammi_server::grpc::gang::GangServer;
        use jammi_server::grpc::proto::gang::gang_service_server::GangService;

        let coord = "nd-coord-admin-scope";
        server
            .engine
            .catalog()
            .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                coord,
                Some("l"),
                Some("h"),
                None,
                None,
            ))
            .await
            .unwrap();
        let attempt = submit_and_claim(
            &server,
            "nd-job-admin-scope",
            coord,
            std::time::Duration::from_secs(30),
            WORLD1_SPEC,
        )
        .await;
        // A standalone `GangServer`, never the one mounted on
        // `server.peer_addr` — see this function's own doc for why.
        let standalone = GangServer::new(
            Arc::clone(&server.engine),
            std::time::Duration::from_secs(30),
        );
        #[cfg(feature = "test-hooks")]
        {
            server.gang_refusal_handle = Some(standalone.refusal_reason_handle());
        }
        let request = tonic::Request::new(streaming_of(vec![assign_frame_full(
            "nd-job-admin-scope",
            attempt,
            0,
            1,
            coord,
        )]));
        let status = server
            .engine
            .with_admin_scope(|_admin| async {
                // `Response<Self::RunRankStream>` (the `Ok` type) is not
                // `Debug` (a boxed `dyn Stream` has no such impl), so
                // `.expect_err` (which requires it) cannot be used here —
                // a plain `match` avoids the bound entirely.
                match standalone.run_rank(request).await {
                    Ok(_) => panic!(
                        "admin scope must be refused, even though every OTHER I-GANG \
                         determinant this job satisfies would otherwise admit it"
                    ),
                    Err(status) => status,
                }
            })
            .await;
        return (server, status);
    }

    let outbound_frame = match reason {
        GangRefusalReason::NotRunning => {
            server
                .engine
                .catalog()
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-not-running",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
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
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-impostor",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
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
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-wrong-attempt",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
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
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-lease-dead",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
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
        GangRefusalReason::SpecUndecodable => {
            server
                .engine
                .catalog()
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-undecodable",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
                .await
                .unwrap();
            // A `world_size` key present but not a valid non-negative rank
            // count — `world_size_from_spec_json` (jammi-db) classifies this
            // `Undecodable`, never a fault of the read that found it.
            let attempt = submit_and_claim(
                &server,
                "nd-job-undecodable",
                "nd-coord-undecodable",
                std::time::Duration::from_secs(30),
                r#"{"common":{"world_size":"not-a-number"}}"#,
            )
            .await;
            // `assign.world` is irrelevant here (any value >= 1 satisfies
            // the wire-level K2 edges) — the row's own spec never decodes,
            // so this refuses before `assign.world` is ever compared to
            // anything.
            assign_frame_full("nd-job-undecodable", attempt, 0, 1, "nd-coord-undecodable")
        }
        GangRefusalReason::WorldMismatch => {
            // Direction (a): `assign.world` (1) BELOW the row's own
            // `world_size` (2, `WORLD2_SPEC`) — no training-set fixture
            // needed at all, since this unit ships the `world_size == 1`
            // lattice only: a row whose own `world_size` disagrees with `1`
            // refuses regardless of any pair/sidecar state (`MultiHostUnsupported`,
            // its own determinant below), so isolating the MISMATCH conjunct
            // specifically needs the `test-hooks` reason, not the wire
            // status/message alone — see `run_rank_refuses_when_assign_world_
            // mismatches_row_world_size` for the executed both-directions
            // proof (including the direction whose plain-lane status alone
            // already distinguishes the mismatch conjunct from every other
            // determinant).
            server
                .engine
                .catalog()
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-world-mismatch",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-world-mismatch",
                "nd-coord-world-mismatch",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-world-mismatch",
                attempt,
                0,
                1,
                "nd-coord-world-mismatch",
            )
        }
        GangRefusalReason::MultiHostUnsupported => {
            // `assign.world` (2) genuinely AGREES with the row's own
            // `world_size` (2, `WORLD2_SPEC`) — no mismatch, and (unlike
            // every prior round of this fixture) no training-set pair is
            // filled either, since this unit never reaches that conjunct at
            // all: a row whose own `world_size` is not `1` refuses outright,
            // pair state irrelevant.
            server
                .engine
                .catalog()
                .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                    "nd-coord-multi-host",
                    Some("l"),
                    Some("h"),
                    None,
                    None,
                ))
                .await
                .unwrap();
            let attempt = submit_and_claim(
                &server,
                "nd-job-multi-host",
                "nd-coord-multi-host",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            assign_frame_full("nd-job-multi-host", attempt, 0, 2, "nd-coord-multi-host")
        }
        GangRefusalReason::AdminScope => {
            unreachable!("handled above, before this match: no network client can trigger it")
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

/// The byte-identity oracle an undecodable `world_size` must satisfy: a row
/// whose `spec` does not decode a `world_size` — either shape (`world_size`
/// present but non-numeric, or `spec` text that
/// is not valid JSON at all — the identical two shapes
/// `jammi-db`'s own `gang_rank_admission.rs` decodes) — refuses
/// BYTE-IDENTICALLY (code, message) to a job id no row exists for at all:
/// an undecodable spec is a ROW FACT, never `admission_catalog_fault`'s
/// `Unavailable` (reserved for the READ itself faulting). The `test-hooks`
/// seam names `SpecUndecodable` specifically for both shapes, distinguishing
/// them from `NotFound` same-process even though the wire cannot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_an_undecodable_world_size_byte_identically_to_not_found() {
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let (_absent, not_found_status) = refusal_scenario(GangRefusalReason::NotFound).await;

    for (label, coord, job_id, spec) in [
        (
            "non-numeric world_size",
            "nd-coord-poison-numeric",
            "nd-job-poison-numeric",
            r#"{"common":{"world_size":"two"}}"#,
        ),
        (
            "spec not valid JSON",
            "nd-coord-poison-json",
            "nd-job-poison-json",
            "not-even-json",
        ),
    ] {
        let server = start_no_worker_server().await;
        server
            .engine
            .catalog()
            .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                coord,
                Some("l"),
                Some("h"),
                None,
                None,
            ))
            .await
            .unwrap();
        let attempt = submit_and_claim(
            &server,
            job_id,
            coord,
            std::time::Duration::from_secs(30),
            spec,
        )
        .await;
        let channel = crate::common::grpc::channel(server.peer_addr).await;
        let mut client = GangServiceClient::new(channel);
        let poisoned = client
            .run_rank(tokio_stream::once(assign_frame_full(
                job_id, attempt, 0, 1, coord,
            )))
            .await
            .expect_err("an undecodable world_size must be refused");
        assert_eq!(
            (poisoned.code(), poisoned.message()),
            (not_found_status.code(), not_found_status.message()),
            "{label}: an undecodable world_size must refuse byte-identically to an absent job"
        );
        #[cfg(feature = "test-hooks")]
        assert_eq!(
            server.gang_last_refusal_reason(),
            Some(GangRefusalReason::SpecUndecodable),
            "{label}: the test-hooks seam must name SpecUndecodable specifically"
        );
    }
}

/// `assign.world` genuinely AGREES with a row's own `world_size`, but that
/// shared value is not `1` — refused all the same (this unit ships the
/// `world_size == 1` lattice only, the training-set pair conjunct and its
/// sidecar verify are `HostAdmission`'s to build), distinguishably
/// (`test-hooks`) from a genuine `assign.world != row.world_size` mismatch —
/// see `run_rank_refuses_when_assign_world_mismatches_row_world_size`'s own
/// direction-(a) control, below, for the executed side-by-side proof.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_gt_one_when_caller_world_matches_the_row() {
    use jammi_server::grpc::gang::GangRefusalReason;

    #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
    let (server, status) = refusal_scenario(GangRefusalReason::MultiHostUnsupported).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server.gang_last_refusal_reason(),
        Some(GangRefusalReason::MultiHostUnsupported)
    );
}

/// The world-mismatch determinant, `assign.world != row.world_size`,
/// exercised in BOTH directions — the caller's `assign.world` below the
/// row's own value, and above it — each with a CONTROL call against the
/// SAME row proving the mismatch conjunct specifically is what refused,
/// never a coincidence with some other determinant.
///
/// **Direction (a)** (`assign.world` BELOW `row.world_size`): the row's own
/// `world_size` (`WORLD2_SPEC`'s `2`) is ALSO refused outright by
/// `GangRefusalReason::MultiHostUnsupported` regardless of the caller's
/// `world` — this unit ships the `world_size == 1` lattice only — so this
/// direction's control is distinguishable ONLY via the `test-hooks` reason,
/// never the wire status/message (both refuse `FAILED_PRECONDITION` with the
/// identical fixed message, by design — non-disclosure,
/// `docs/rigor/contracts/feat_500-C-U5a-1.md` §2 (P2)).
///
/// **Direction (b)** (`assign.world` ABOVE `row.world_size`): the row's own
/// `world_size` is `1` (`WORLD1_SPEC`), so its control (`assign.world`
/// matching, `= 1`) admits all the way to `UNIMPLEMENTED` — a
/// WIRE-VISIBLE distinction from the mismatched call's
/// `FAILED_PRECONDITION`, catching the mutation `if false && assign.world !=
/// row.world_size` on the PLAIN lane alone (direction (a)'s control cannot,
/// since `MultiHostUnsupported` refuses the SAME fixed way).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_assign_world_mismatches_row_world_size() {
    #[cfg(feature = "test-hooks")]
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    // Direction (a): assign.world (1) BELOW row.world_size (2).
    let server_a = start_no_worker_server().await;
    server_a
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "nd-coord-mismatch-below",
            Some("l"),
            Some("h"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt_a = submit_and_claim(
        &server_a,
        "nd-job-mismatch-below",
        "nd-coord-mismatch-below",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    let channel_a1 = crate::common::grpc::channel(server_a.peer_addr).await;
    let mut client_a1 = GangServiceClient::new(channel_a1);
    let mismatched = client_a1
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-below",
            attempt_a,
            0,
            1,
            "nd-coord-mismatch-below",
        )))
        .await
        .expect_err("assign.world (1) below row.world_size (2) must be refused");
    assert_eq!(mismatched.code(), tonic::Code::FailedPrecondition);
    assert_eq!(mismatched.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_a.gang_last_refusal_reason(),
        Some(GangRefusalReason::WorldMismatch),
        "must record WorldMismatch specifically, not MultiHostUnsupported"
    );
    // Control: the SAME row, assign.world (2) matching row.world_size (2).
    let channel_a2 = crate::common::grpc::channel(server_a.peer_addr).await;
    let mut client_a2 = GangServiceClient::new(channel_a2);
    let matching = client_a2
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-below",
            attempt_a,
            0,
            2,
            "nd-coord-mismatch-below",
        )))
        .await
        .expect_err("the row's own world_size (2) is unsupported in this unit regardless");
    assert_eq!(matching.code(), tonic::Code::FailedPrecondition);
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_a.gang_last_refusal_reason(),
        Some(GangRefusalReason::MultiHostUnsupported),
        "the SAME row at its own world_size must refuse for MultiHostUnsupported, \
         distinguishing it from the mismatch above"
    );

    // Direction (b): assign.world (2) ABOVE row.world_size (1).
    let server_b = start_no_worker_server().await;
    server_b
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "nd-coord-mismatch-above",
            Some("l"),
            Some("h"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt_b = submit_and_claim(
        &server_b,
        "nd-job-mismatch-above",
        "nd-coord-mismatch-above",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    let channel_b1 = crate::common::grpc::channel(server_b.peer_addr).await;
    let mut client_b1 = GangServiceClient::new(channel_b1);
    let mismatched_b = client_b1
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-above",
            attempt_b,
            0,
            2,
            "nd-coord-mismatch-above",
        )))
        .await
        .expect_err("assign.world (2) above row.world_size (1) must be refused");
    assert_eq!(mismatched_b.code(), tonic::Code::FailedPrecondition);
    assert_eq!(mismatched_b.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_b.gang_last_refusal_reason(),
        Some(GangRefusalReason::WorldMismatch)
    );
    // Control: the SAME row, assign.world (1) matching row.world_size (1) —
    // admits all the way to UNIMPLEMENTED, a WIRE-VISIBLE distinction from
    // the mismatched call above, proving the mismatch conjunct (never some
    // other reason) is what refused it.
    let channel_b2 = crate::common::grpc::channel(server_b.peer_addr).await;
    let mut client_b2 = GangServiceClient::new(channel_b2);
    let matching_b = client_b2
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-above",
            attempt_b,
            0,
            1,
            "nd-coord-mismatch-above",
        )))
        .await
        .expect_err("no HostAdmission session exists yet to admit into");
    assert_eq!(
        matching_b.code(),
        tonic::Code::Unimplemented,
        "the SAME row at its own world_size (1) must admit, proving direction (b) above \
         refused for the mismatch alone"
    );
}

/// The full set of I-GANG determinants [`refusal_scenario`] can drive,
/// shared by the plain lane's non-disclosure oracle and the `test-hooks`
/// lane's reason-distinguishing oracle below — one definition, so adding a
/// determinant to one automatically covers it in the other.
///
/// Derived from an EXHAUSTIVE match over a witness of each variant, never a
/// hardcoded `[..; N]` array: `assert_every_variant_is_a_witness`'s own
/// match has NO wildcard arm, so a new `GangRefusalReason` variant fails
/// THIS FILE to compile — naming the missing arm — until a matching arm is
/// added there; the exhaustive match forces every variant into a scenario
/// arm. `WITNESSES` below is a separate list that match does NOT force to
/// grow — a fixed arm-list gap the compiler catches is not the same as a
/// witness-list gap it does not; adding the new arm without also adding the
/// variant to `WITNESSES` still compiles.
fn every_gang_refusal_reason() -> Vec<jammi_server::grpc::gang::GangRefusalReason> {
    use jammi_server::grpc::gang::GangRefusalReason as R;

    const WITNESSES: &[R] = &[
        R::AdminScope,
        R::NotFound,
        R::NotRunning,
        R::WrongClaimant,
        R::WrongAttempt,
        R::LeaseDead,
        R::SpecUndecodable,
        R::WorldMismatch,
        R::MultiHostUnsupported,
        R::CoordinatorNotFresh,
    ];

    fn assert_every_variant_is_a_witness(variant: R) {
        match variant {
            R::AdminScope
            | R::NotFound
            | R::NotRunning
            | R::WrongClaimant
            | R::WrongAttempt
            | R::LeaseDead
            | R::SpecUndecodable
            | R::WorldMismatch
            | R::MultiHostUnsupported
            | R::CoordinatorNotFresh => {}
        }
    }
    for w in WITNESSES {
        assert_every_variant_is_a_witness(*w);
    }
    WITNESSES.to_vec()
}

/// The ONE non-disclosure oracle. Every I-GANG determinant
/// (ambient admin scope / not found / not running / wrong claimant / wrong
/// attempt / lease dead / undecodable world_size / world mismatch /
/// multi-host unsupported / coordinator not fresh — ten total) refuses with
/// the PAIRWISE-IDENTICAL `(code, message)` — compared pairwise so a single
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

/// `test-hooks` only: drives the SAME ten scenarios
/// [`run_rank_refusal_is_non_disclosing_across_every_determinant`] does, but
/// asserts `PeerEngineServer::gang_last_refusal_reason` names the EXACT
/// determinant each one refused for — the seam that lets this lane
/// distinguish what the plain lane's own non-disclosure oracle just proved
/// is (correctly) indistinguishable on the wire. This is what makes the
/// `test-hooks` lane's `cargo test -p jammi-server --test it -- gang`
/// execute ONE MORE gang-prefixed test-fn than the plain lane (this
/// function itself is compiled only under `test-hooks`; the plain lane's
/// case above still runs the same ten RPC calls, just without this
/// additional reason assertion). One of the ten (`AdminScope`) is driven
/// in-process by `refusal_scenario` itself (see its own doc) rather than
/// over the real listener — this test's own assertion still holds, since it
/// reads `PeerEngineServer::gang_last_refusal_reason`, which `refusal_scenario`
/// re-points at the standalone `GangServer` it actually called for that one
/// scenario.
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
