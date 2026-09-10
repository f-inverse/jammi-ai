//! `[server.limits]` end-to-end over the wire (PLAN-C §5): the ONE oracle
//! this suite owns that `crates/jammi-server/src/limits.rs`'s own unit tests
//! cannot -- everything that genuinely needs a live socket (an inbound
//! message tonic's own codec actually decodes and rejects, a
//! `JobService.WaitJob` stream that stays open across real network I/O and
//! whose permit release depends on the CLIENT actually disconnecting).
//!
//! The concurrency-refusal proofs (`(max_in_flight+1)`-th caller refused,
//! per-connection variant) are deliberately NOT re-proven here: forcing that
//! race deterministically over a real socket would need an artificial
//! per-RPC delay hook this codebase's RPC handlers do not have (every one
//! completes near-instantly), so `crate::limits`'s own unit tests exercise
//! that exact code path (the identical `Service::call`) with a controllable
//! held future instead -- a deterministic proof, not a network race. See
//! that module's `tests` submodule.

use std::time::Duration;

use jammi_db::config::LimitsConfig;
use jammi_server::grpc::proto::job::job_service_client::JobServiceClient;
use jammi_server::grpc::proto::job::submit_job_request::Spec;
use jammi_server::grpc::proto::job::{CancelJobRequest, JobHandle, SubmitJobRequest};
use jammi_server::grpc::proto::training::{FineTuneMethod, FineTuneSpec};
use tonic::Code;

use super::common::grpc::{channel, start_engine_server_with_limits};

/// A `SubmitJob` request that is ALWAYS accepted (submission never validates
/// the source) and, with no worker claiming it, stays `queued` forever --
/// the fixture every stream-budget test in this file needs to hold a
/// `WaitJob` stream open deterministically.
fn never_runs_job_request() -> SubmitJobRequest {
    SubmitJobRequest {
        spec: Some(Spec::FineTune(FineTuneSpec {
            source: "does-not-exist".into(),
            columns: vec!["a".into(), "b".into(), "c".into()],
            method: FineTuneMethod::Lora as i32,
            task: jammi_server::grpc::proto::inference::ModelTask::TextEmbedding as i32,
        })),
        base_model: "local:does-not-exist".into(),
        config: None,
        idempotency_key: String::new(),
    }
}

/// An oversize inbound message is refused -- never silently truncated -- and
/// counted under the `message_size` reason (N5: tonic's own per-service
/// `max_decoding_message_size` codec rejection, with no `RefusedBound`
/// extension, is what `RefusalStatusLayer` defaults to that label for).
/// Verified against the vendored tonic 0.14.5 source
/// (`codec/decode.rs:185-195`): this specific rejection is `OUT_OF_RANGE`,
/// not `RESOURCE_EXHAUSTED` -- see `crate::limits`'s N5 module doc.
#[tokio::test]
async fn oversize_inbound_message_is_refused_and_counted_as_message_size() {
    let server = start_engine_server_with_limits(LimitsConfig {
        max_message_bytes: 128,
        ..LimitsConfig::default()
    })
    .await;
    let mut client = JobServiceClient::new(channel(server.addr).await);

    // A `CancelJobRequest` is a single string field -- padding it well past
    // 128 bytes exceeds the configured cap with no other RPC machinery
    // involved.
    let oversize_job_id = "x".repeat(10_000);
    let err = client
        .cancel_job(CancelJobRequest {
            job_id: oversize_job_id,
        })
        .await
        .expect_err("an oversize inbound message must be refused, never silently truncated");
    assert_eq!(err.code(), Code::OutOfRange);

    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["message_size"])
            .get(),
        1,
        "the refusal must be counted under the message_size reason"
    );
}

/// A normal-sized request is unaffected by a generous `max_message_bytes` --
/// the refusal above is about SIZE, not about `CancelJob` itself. `CancelJob`
/// is idempotent over an absent id (`cancelled = false`, not an error -- see
/// `CancelJobResponse`'s doc), so the oracle here is simply "no refusal was
/// counted", not a particular RPC-level error code.
#[tokio::test]
async fn a_normal_sized_message_is_not_refused() {
    let server = start_engine_server_with_limits(LimitsConfig::default()).await;
    let mut client = JobServiceClient::new(channel(server.addr).await);
    let resp = client
        .cancel_job(CancelJobRequest {
            job_id: "a-job-id-that-does-not-exist".into(),
        })
        .await
        .expect("a normal-sized CancelJob call must not be refused")
        .into_inner();
    assert!(
        !resp.cancelled,
        "cancelling an absent id has no effect, but is not an error"
    );
    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["message_size"])
            .get(),
        0
    );
}

/// `max_job_waits + 1` concurrent `WaitJob` streams: the budget-exceeding
/// caller is refused with `RESOURCE_EXHAUSTED` while the first stream stays
/// open, and dropping the first stream (a client disconnect) releases the
/// permit -- a THIRD call then succeeds. Proves the stream-budget's
/// release-on-drop contract over a REAL socket (`crate::limits::PermitBody`),
/// not just the unit-level future/body drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_job_waits_plus_one_is_refused_and_releases_when_the_first_disconnects() {
    let server = start_engine_server_with_limits(LimitsConfig {
        max_job_waits: 1,
        ..LimitsConfig::default()
    })
    .await;
    let mut client = JobServiceClient::new(channel(server.addr).await);

    let job = client
        .submit_job(never_runs_job_request())
        .await
        .expect("submit_job")
        .into_inner();

    // Stream #1: opens and stays open (no worker claims the job, so it
    // never goes terminal) -- read one frame to prove it is genuinely live,
    // not merely accepted.
    let mut stream1 = client
        .wait_job(JobHandle {
            job_id: job.job_id.clone(),
        })
        .await
        .expect("the first WaitJob stream must open")
        .into_inner();
    tokio::time::timeout(Duration::from_secs(5), stream1.message())
        .await
        .expect("the first WaitJob stream must emit a live progress frame")
        .expect("wait_job frame");

    // Stream #2: refused at the edge -- the budget is exhausted while #1 is
    // still open.
    let err = client
        .wait_job(JobHandle {
            job_id: job.job_id.clone(),
        })
        .await
        .expect_err("the (max_job_waits+1)-th concurrent WaitJob must be refused");
    assert_eq!(err.code(), Code::ResourceExhausted);
    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["job_waits"])
            .get(),
        1
    );

    // Disconnect stream #1: its permit releases, and a THIRD call now
    // succeeds. Server-side release lags the client-side drop by one
    // network round trip (RST_STREAM propagation), so poll rather than
    // asserting on the first attempt.
    drop(stream1);
    let mut opened = false;
    for _ in 0..50 {
        match client
            .wait_job(JobHandle {
                job_id: job.job_id.clone(),
            })
            .await
        {
            Ok(mut stream3) => {
                tokio::time::timeout(Duration::from_secs(5), stream3.get_mut().message())
                    .await
                    .expect("the third WaitJob stream must emit a live progress frame")
                    .expect("wait_job frame");
                opened = true;
                break;
            }
            Err(status) if status.code() == Code::ResourceExhausted => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(status) => panic!("unexpected error re-opening WaitJob: {status:?}"),
        }
    }
    assert!(
        opened,
        "dropping the first WaitJob stream must eventually release its \
         job_waits permit, letting a new stream open"
    );
}

/// `WaitJob` carrying a client-requested `grpc-timeout` above
/// `wait_timeout_secs` is refused AT THE EDGE -- before the stream ever
/// opens -- with `DEADLINE_EXCEEDED`, rather than being allowed to hold a
/// connection open past the operator's budget.
#[tokio::test]
async fn wait_job_with_a_timeout_above_the_configured_budget_is_refused_at_the_edge() {
    let server = start_engine_server_with_limits(LimitsConfig {
        wait_timeout_secs: Some(1),
        ..LimitsConfig::default()
    })
    .await;
    let mut client = JobServiceClient::new(channel(server.addr).await);

    let job = client
        .submit_job(never_runs_job_request())
        .await
        .expect("submit_job")
        .into_inner();

    let mut request = tonic::Request::new(JobHandle {
        job_id: job.job_id.clone(),
    });
    // Far above the 1-second budget -- refused immediately, not after
    // waiting anywhere near this long.
    request.set_timeout(Duration::from_secs(600));
    let started = std::time::Instant::now();
    let err = client
        .wait_job(request)
        .await
        .expect_err("a WaitJob deadline above the configured budget must be refused");
    assert_eq!(err.code(), Code::DeadlineExceeded);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the refusal must happen at the edge, not after waiting out the client's own deadline"
    );
    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["timeout"])
            .get(),
        1
    );
}

/// A `WaitJob` deadline WITHIN the configured budget is unaffected --
/// the stream opens normally.
#[tokio::test]
async fn wait_job_with_a_timeout_within_the_configured_budget_opens_normally() {
    let server = start_engine_server_with_limits(LimitsConfig {
        wait_timeout_secs: Some(60),
        ..LimitsConfig::default()
    })
    .await;
    let mut client = JobServiceClient::new(channel(server.addr).await);

    let job = client
        .submit_job(never_runs_job_request())
        .await
        .expect("submit_job")
        .into_inner();

    let mut request = tonic::Request::new(JobHandle {
        job_id: job.job_id.clone(),
    });
    request.set_timeout(Duration::from_secs(5));
    let mut stream = client
        .wait_job(request)
        .await
        .expect("a within-budget WaitJob deadline must open normally")
        .into_inner();
    tokio::time::timeout(Duration::from_secs(5), stream.message())
        .await
        .expect("the stream must emit a live progress frame")
        .expect("wait_job frame");
}

/// RED before the fix: a `WaitJob` call carrying NO `grpc-timeout` header at
/// all (HTTP/2's own no-deadline default) used to open normally whenever
/// `wait_timeout_secs` was set, even though
/// [`jammi_db::config::LimitsConfig::wait_timeout_secs`]'s own doc already
/// promised an unbounded request is refused exactly like an over-budget one
/// -- an unbounded caller is precisely the resource risk the cap exists to
/// close (it would hold a `max_job_waits` permit open forever). GREEN after:
/// refused at the edge with `DEADLINE_EXCEEDED`, identically to the
/// over-budget case.
#[tokio::test]
async fn wait_job_with_no_timeout_header_is_refused_when_a_budget_is_configured() {
    let server = start_engine_server_with_limits(LimitsConfig {
        wait_timeout_secs: Some(1),
        ..LimitsConfig::default()
    })
    .await;
    let mut client = JobServiceClient::new(channel(server.addr).await);

    let job = client
        .submit_job(never_runs_job_request())
        .await
        .expect("submit_job")
        .into_inner();

    // No `request.set_timeout(...)` at all -- an unbounded request.
    let started = std::time::Instant::now();
    let err = client
        .wait_job(JobHandle {
            job_id: job.job_id.clone(),
        })
        .await
        .expect_err("an unbounded WaitJob request must be refused when a budget is configured");
    assert_eq!(err.code(), Code::DeadlineExceeded);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the refusal must happen at the edge, not after waiting out any deadline"
    );
    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["timeout"])
            .get(),
        1
    );
}

/// K4: the SAME `max_message_bytes` bound refuses an oversize inbound
/// message identically on Flight SQL, not only on the `jammi.v1.*` gRPC
/// plane -- both transports are constructed with the identical
/// `.max_decoding_message_size(limits.max_message_bytes)` in
/// `assemble_grpc_chain` (`crate::runtime`), off the SAME `[server.limits]`
/// value. `FlightSqlServiceClient::execute` folds a `tonic::Status` into
/// `ArrowError::IpcError(format!("{status:?}"))` (arrow-flight's own
/// `status_to_arrow_error`), losing the typed `Code` -- so this asserts on
/// the Debug-rendered code name rather than a `tonic::Code` equality, the
/// most precise oracle this client surface exposes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversize_flight_sql_query_is_refused_identically_to_grpc() {
    use arrow_flight::sql::client::FlightSqlServiceClient;

    let server = start_engine_server_with_limits(LimitsConfig {
        max_message_bytes: 128,
        ..LimitsConfig::default()
    })
    .await;
    let mut client = FlightSqlServiceClient::new(channel(server.addr).await);

    // A `CommandStatementQuery` is dominated by its `query` string -- padding
    // it past 128 bytes exceeds the same cap the gRPC plane enforces.
    let oversize_sql = format!("SELECT 1 /* {} */", "x".repeat(10_000));
    let err = client
        .execute(oversize_sql, None)
        .await
        .expect_err("an oversize Flight SQL query must be refused, never silently truncated");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("OutOfRange"),
        "expected the same OUT_OF_RANGE refusal the gRPC plane gets, got: {rendered}"
    );

    assert_eq!(
        server
            .metrics
            .grpc_refused
            .with_label_values(&["message_size"])
            .get(),
        1,
        "the Flight SQL refusal must be counted under message_size too"
    );
}
