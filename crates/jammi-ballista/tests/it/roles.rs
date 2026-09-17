//! Hermetic role-hosting oracle (contract `feat_500-wave4` §7 acceptance,
//! one process, in-memory cluster): `host_scheduler` + `host_executor` come
//! up, `submit_physical_plan` of a shuffle-boundary plan returns rows equal
//! to in-process execution, and `stop()` closes both ports within 5s.
//!
//! Shape mirrors `ballista-54.1.0/tests/physical_plan_submission.rs`'s
//! `should_execute_submitted_physical_plan_across_shuffle_stages` (the
//! reference client usage the brief names), substituting jammi's own
//! `host_scheduler`/`host_executor`/`submit_physical_plan` for that test's
//! `setup_test_cluster`/raw `execute_physical_plan` call.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{self, ExecutionPlan, Partitioning};
use datafusion::prelude::SessionContext;

use ballista_core::utils::{default_config_producer, default_session_builder};
use ballista_scheduler::cluster::BallistaCluster;
use ballista_scheduler::config::TaskDistributionPolicy;

use jammi_ai::session::InferenceSession;
use jammi_ballista::client::submit_physical_plan;
use jammi_ballista::cluster::{CatalogClusterState, CatalogJobState};
use jammi_ballista::roles::{host_executor, host_scheduler};
use jammi_db::catalog::compute_repo::ComputeExecutorRecord;
use jammi_db::config::BallistaExecutorConfig;

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

/// A two-partition `MemTable`-backed scan wrapped in a hash `RepartitionExec`
/// — a physical plan with a shuffle boundary, so the scheduler splits it
/// into two stages (the same shape `ballista-54.1.0`'s own reference test
/// uses).
async fn build_shuffle_plan() -> (Arc<dyn ExecutionPlan>, SessionContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![40, 50])),
        ],
    )
    .unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch1], vec![batch2]]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(table)).unwrap();
    let scan = ctx
        .table("t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let repartitioned: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(
            scan,
            Partitioning::Hash(vec![Arc::new(Column::new("id", 0))], 4),
        )
        .unwrap(),
    );
    (repartitioned, ctx)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scheduler_and_executor_host_in_one_process_and_submit_round_trips() {
    let session = session().await;

    let cluster = BallistaCluster::new_memory(
        "jammi-ballista-it",
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
        task_slots: 2,
    };
    let executor = host_executor(&session, &executor_cfg)
        .await
        .expect("executor role hosts and registers");
    assert_eq!(executor.executor_id(), session.instance_id());

    let (plan, ctx) = build_shuffle_plan().await;
    let expected = physical_plan::collect(plan.clone(), ctx.task_ctx())
        .await
        .expect("in-process collect");

    let scheduler_url = format!("http://127.0.0.1:{}", scheduler.addr.port());
    let stream = tokio::time::timeout(
        Duration::from_secs(30),
        submit_physical_plan(&session, &scheduler_url, plan),
    )
    .await
    .expect("submit did not time out")
    .expect("submit_physical_plan succeeds");
    let actual = tokio::time::timeout(
        Duration::from_secs(30),
        datafusion::physical_plan::common::collect(stream),
    )
    .await
    .expect("collect did not time out")
    .expect("collect succeeds");

    let expected_rows: usize = expected.iter().map(|b| b.num_rows()).sum();
    let actual_rows: usize = actual.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        actual_rows, expected_rows,
        "the placed plan must return the same row count as in-process execution \
         (a registered, task-slotted executor ran the shuffle stages)"
    );

    tokio::time::timeout(Duration::from_secs(5), executor.stop())
        .await
        .expect("executor stop() within 5s");
    tokio::time::timeout(Duration::from_secs(5), scheduler.stop())
        .await
        .expect("scheduler stop() within 5s");
}

#[tokio::test]
async fn unset_ballista_config_hosts_no_roles() {
    // The config-level property (contract §2.1): unset `[ballista]` = no
    // roles = today's process. Exercised at the config layer (jammi-server's
    // `tests/it/ballista_roles.rs` hosts the full negative case against a
    // real `OssServer`); this crate's own oracle is that `hosts_scheduler`/
    // `hosts_executor` are false on the default config, so `OssServer::bind`
    // never calls `host_scheduler`/`host_executor` in the first place.
    let cfg = jammi_db::config::BallistaConfig::default();
    assert!(!cfg.hosts_scheduler());
    assert!(!cfg.hosts_executor());
}

/// An executor role started BEFORE its scheduler is bound waits for it (a
/// bounded window, `SCHEDULER_CONNECT_WINDOW`) and registers once the
/// scheduler comes up, instead of refusing on the first refused connect —
/// the shape of a fleet whose processes start concurrently.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn executor_waits_for_a_scheduler_that_binds_later() {
    let session = session().await;
    let port = {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
        probe.local_addr().expect("probe addr").port()
    };
    let executor_cfg = BallistaExecutorConfig {
        scheduler_address: format!("127.0.0.1:{port}"),
        bind: "127.0.0.1:0".to_string(),
        grpc_bind: "127.0.0.1:0".to_string(),
        advertise_host: Some("127.0.0.1".to_string()),
        work_dir: None,
        task_slots: 1,
    };
    let executor_session = Arc::clone(&session);
    let executor_task =
        tokio::spawn(async move { host_executor(&executor_session, &executor_cfg).await });
    // The scheduler binds only after the executor has already been refused
    // at least a few times (250 ms between attempts).
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let cluster = BallistaCluster::new_memory(
        "jammi-ballista-it-late",
        Arc::new(default_session_builder),
        Arc::new(default_config_producer),
    );
    let scheduler = host_scheduler(
        &session,
        &format!("127.0.0.1:{port}"),
        cluster,
        TaskDistributionPolicy::RoundRobin,
    )
    .await
    .expect("scheduler role hosts on the pre-chosen port");
    let executor = tokio::time::timeout(Duration::from_secs(30), executor_task)
        .await
        .expect("the executor registered within the window")
        .expect("executor task joins")
        .expect("executor role hosts and registers after the scheduler came up");
    assert_eq!(executor.executor_id(), session.instance_id());
    executor.stop().await;
    scheduler.stop().await;
}

/// `placement_available` (the scheduler role's `PlacedGangSubmitter`) answers
/// from LIVE executors only — the binder's and the submit edge's own
/// predicate: a row a dead executor left behind (stale `heartbeat_at`) is
/// not a peer; a fresh row is. Mutation: drop `executor_is_live` from
/// `placement_available` and the stale row reads as a peer (the first
/// assertion reds). Hosts a scheduler and NO executor (nothing here touches
/// the executor's process-wide `TERMINATING` flag).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placement_available_counts_live_peers_only() {
    let session = session().await;
    let catalog = Arc::clone(session.catalog_arc());
    let cluster = BallistaCluster::new(
        Arc::new(CatalogClusterState::new(Arc::clone(&catalog))),
        Arc::new(CatalogJobState::new(
            Arc::clone(&catalog),
            "jammi-ballista-it-placement",
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
    let submitter = session
        .host_admission()
        .placed_gang_submitter()
        .expect("host_scheduler installs the placed-gang submitter");
    assert!(
        !submitter.placement_available(),
        "no executor registered at all: nothing to place on"
    );

    let record = |id: &str, heartbeat_at: String| ComputeExecutorRecord {
        executor_id: id.to_string(),
        instance_id: id.to_string(),
        host: "127.0.0.1".to_string(),
        port: 0,
        grpc_port: 0,
        task_slots: 1,
        available_slots: 1,
        status: "Active".to_string(),
        heartbeat_at,
        metadata: String::new(),
        devices: vec![],
    };
    let stale_id = format!("stale-peer-{}", jammi_test_utils::unique_suffix());
    catalog
        .upsert_compute_executor(&record(
            &stale_id,
            "2026-01-01T00:00:00.000000000Z".to_string(),
        ))
        .await
        .unwrap();
    assert!(
        !submitter.placement_available(),
        "a row a dead executor left behind is not a peer"
    );

    let live_id = format!("live-peer-{}", jammi_test_utils::unique_suffix());
    catalog
        .upsert_compute_executor(&record(
            &live_id,
            jammi_db::catalog::lease::canonical_stamp_now(),
        ))
        .await
        .unwrap();
    assert!(
        submitter.placement_available(),
        "a live registered executor other than this instance is a peer"
    );

    catalog.remove_compute_executor(&stale_id).await.ok();
    catalog.remove_compute_executor(&live_id).await.ok();
    scheduler.stop().await;
}
