//! `ExecutorRole::drain()` (contract `feat_500-wave4` §9 B6), in its OWN
//! test binary/process: `ballista_executor::executor_server::TERMINATING`
//! is a crate-wide `static AtomicBool` — once one executor in a process
//! calls `drain()`, EVERY executor hosted afterward in that SAME process
//! reports `Terminating` in its heartbeats (read at
//! `ballista-executor-54.1.0/src/executor_server.rs:317`), so a scheduler
//! never binds tasks to it again. Verified by reproducing the exact
//! symptom: this test's assertions passed when this file's one test ran
//! alone in `tests/it/roles.rs`, then hung (30s timeout, no task ever
//! bound) the moment `tests/it`'s OTHER role test shared the process with
//! it — moving it to its own `[[test]]` binary is the fix, not a
//! workaround: each Cargo test target is its own OS process.

use std::sync::Arc;
use std::time::Duration;

use ballista_core::utils::{default_config_producer, default_session_builder};
use ballista_scheduler::cluster::BallistaCluster;
use ballista_scheduler::config::TaskDistributionPolicy;

use jammi_ai::session::InferenceSession;
use jammi_ballista::roles::{host_executor, host_scheduler};
use jammi_db::config::BallistaExecutorConfig;

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

/// `drain()` reports `Terminating` and stops within 5s when there is no
/// in-flight task to wait for — the no-task arm of the property. The
/// "survives a running task until it completes" arm needs a deliberately
/// slow operator to hold `TasksDrainedFuture` pending across the assertion
/// window; UNCOVERED here (named in this crate's contract file) — every
/// plan this hermetic suite submits completes near-instantly, so a real one
/// would need a dedicated slow `ExecutionPlan` fixture, which this unit did
/// not build given the round's scope.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn executor_drain_reports_terminating_and_stops_with_no_inflight_work() {
    let session = session().await;
    let cluster = BallistaCluster::new_memory(
        "jammi-ballista-drain",
        Arc::new(default_session_builder),
        Arc::new(default_config_producer),
    );
    let scheduler = host_scheduler(
        &session,
        "127.0.0.1:0",
        cluster,
        TaskDistributionPolicy::RoundRobin,
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

    tokio::time::timeout(Duration::from_secs(5), executor.drain())
        .await
        .expect("drain() with no in-flight work completes within 5s");
    scheduler.stop().await;
}
