//! `ExecutorRole::begin_drain`, in its
//! OWN test binary/process for the same reason `roles_drain.rs` is:
//! `begin_drain`'s first statement stores `ballista_executor::
//! executor_server::TERMINATING` (a crate-wide `static AtomicBool` nothing
//! resets), after which every executor hosted in this process reports
//! `Terminating` and is never bound again — so this test may never share a
//! process with an executor-hosting test that needs a task bound.

use std::sync::Arc;

use ballista_core::serde::protobuf::scheduler_grpc_client::SchedulerGrpcClient;
use ballista_core::serde::protobuf::{executor_status, ExecutorStatus, HeartBeatParams};
use ballista_core::utils::create_grpc_client_endpoint;
use jammi_db::catalog::status::ComputeExecutorStatus;
use jammi_db::config::BallistaSchedulerConfig;

use jammi_ai::session::InferenceSession;
use jammi_ballista::cluster::executor_is_live;
use jammi_ballista::roles::{host_executor, host_scheduler};
use jammi_db::config::BallistaExecutorConfig;

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

/// The DRAIN instant: `ExecutorRole::begin_drain` reports
/// `Terminating` to the scheduler and, over the catalog-backed cluster
/// state, the executor's OWN row reads `Terminating` and not live — the
/// binder stops binding here before the process's worker has finished
/// draining. Then the row STAYS draining: an `Active` heartbeat for this
/// executor, delivered through the scheduler's real gRPC after the drain
/// report (the executor's own heartbeater's report, built before the drain
/// flag flipped and landing after), refreshes the row's timestamp and never
/// its state. Mutation: make `begin_drain` flip only the local
/// `TERMINATING` flag (no heartbeat) and the row stays `Active`/live; write
/// the heartbeat's status unconditionally and the late `Active` report
/// revives it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn begin_drain_reports_terminating_to_the_catalog_before_the_executor_stops() {
    let session = session().await;
    let catalog = Arc::clone(session.catalog_arc());
    let scheduler = host_scheduler(
        &session,
        &BallistaSchedulerConfig {
            bind: "127.0.0.1:0".into(),
            advertise_host: None,
        },
    )
    .await
    .expect("scheduler role hosts");
    let executor_cfg = BallistaExecutorConfig {
        scheduler_address: format!("127.0.0.1:{}", scheduler.addr.port()),
        bind: "127.0.0.1:0".to_string(),
        grpc_bind: "127.0.0.1:0".to_string(),
        advertise_host: Some("127.0.0.1".to_string()),
        work_dir: None,
        task_slots: 1,
    };
    let executor = host_executor(&session, &executor_cfg)
        .await
        .expect("executor role hosts and registers");
    let id = executor.executor_id().to_string();
    let before = catalog
        .get_compute_executor(&id)
        .await
        .unwrap()
        .expect("registered row");
    assert_eq!(before.status, ComputeExecutorStatus::Active);
    assert!(executor_is_live(&before, chrono::Utc::now()));

    executor.begin_drain().await;

    let after = catalog
        .get_compute_executor(&id)
        .await
        .unwrap()
        .expect("the row is still there while the executor drains");
    assert_eq!(
        after.status,
        ComputeExecutorStatus::Terminating,
        "the DRAIN instant's own report"
    );
    assert!(
        !executor_is_live(&after, chrono::Utc::now()),
        "a terminating executor is not live to the binder or the submit edge"
    );

    // The late `Active` report, through the same scheduler gRPC the
    // executor's heartbeater uses.
    let channel =
        create_grpc_client_endpoint(format!("http://127.0.0.1:{}", scheduler.addr.port()), None)
            .expect("scheduler endpoint")
            .connect()
            .await
            .expect("the scheduler accepts a connection");
    SchedulerGrpcClient::new(channel)
        .heart_beat_from_executor(HeartBeatParams {
            executor_id: id.clone(),
            metrics: vec![],
            status: Some(ExecutorStatus {
                status: Some(executor_status::Status::Active(String::default())),
            }),
            metadata: None,
        })
        .await
        .expect("the scheduler records the heartbeat");
    let late = catalog
        .get_compute_executor(&id)
        .await
        .unwrap()
        .expect("the row is still there while the executor drains");
    assert_eq!(
        late.status,
        ComputeExecutorStatus::Terminating,
        "an Active heartbeat after the drain report never revives the row"
    );
    assert!(
        late.heartbeat_at >= after.heartbeat_at,
        "the late heartbeat still refreshes the timestamp"
    );
    assert!(
        !executor_is_live(&late, chrono::Utc::now()),
        "a draining executor stays not live"
    );

    executor.stop().await;
    scheduler.stop().await;
    catalog.remove_compute_executor(&id).await.ok();
}
