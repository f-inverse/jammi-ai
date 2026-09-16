//! `ExecutorRole::begin_drain` (contract `feat_500-wave4` §9b F2), in its
//! OWN test binary/process for the same reason `roles_drain.rs` is:
//! `begin_drain`'s first statement stores `ballista_executor::
//! executor_server::TERMINATING` (a crate-wide `static AtomicBool` nothing
//! resets), after which every executor hosted in this process reports
//! `Terminating` and is never bound again — so this test may never share a
//! process with an executor-hosting test that needs a task bound
//! (closing audit #5's block: the oracle first landed in `tests/it`).

use std::sync::Arc;

use ballista_core::utils::{default_config_producer, default_session_builder};
use ballista_scheduler::cluster::BallistaCluster;
use ballista_scheduler::config::TaskDistributionPolicy;

use jammi_ai::session::InferenceSession;
use jammi_ballista::cluster::{executor_is_live, CatalogClusterState, CatalogJobState};
use jammi_ballista::roles::{host_executor, host_scheduler};
use jammi_db::config::BallistaExecutorConfig;

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

/// The DRAIN instant (contract §9 B6): `ExecutorRole::begin_drain` reports
/// `Terminating` to the scheduler and, over the catalog-backed cluster
/// state, the executor's OWN row reads `Terminating` and not live — the
/// binder stops binding here before the process's worker has finished
/// draining. Mutation: make `begin_drain` flip only the local `TERMINATING`
/// flag (no heartbeat) and the row stays `Active`/live.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn begin_drain_reports_terminating_to_the_catalog_before_the_executor_stops() {
    let session = session().await;
    let catalog = Arc::clone(session.catalog_arc());
    let cluster = BallistaCluster::new(
        Arc::new(CatalogClusterState::new(Arc::clone(&catalog))),
        Arc::new(CatalogJobState::new(
            Arc::clone(&catalog),
            "jammi-ballista-it-drain",
            Arc::new(default_session_builder),
            Arc::new(default_config_producer),
        )),
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
    let id = executor.executor_id().to_string();
    let before = catalog
        .get_compute_executor(&id)
        .await
        .unwrap()
        .expect("registered row");
    assert_eq!(before.status, "Active");
    assert!(executor_is_live(&before, chrono::Utc::now()));

    executor.begin_drain().await;

    let after = catalog
        .get_compute_executor(&id)
        .await
        .unwrap()
        .expect("the row is still there while the executor drains");
    assert_eq!(
        after.status, "Terminating",
        "the DRAIN instant's own report"
    );
    assert!(
        !executor_is_live(&after, chrono::Utc::now()),
        "a terminating executor is not live to the binder or the submit edge"
    );

    executor.stop().await;
    scheduler.stop().await;
    catalog.remove_compute_executor(&id).await.ok();
}
