//! `CatalogClusterState` / `CatalogJobState` / `DevicePlacement` — hermetic
//! oracle over BOTH catalog backends (contract `feat_500-wave4.md` §3;
//! UNITS §U8b; acceptance b2/b3/b6's substrate, the `unbind_tasks`
//! atomicity property, and A10's CAS-before-stamp ordering).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use ballista_core::serde::scheduler::{
    ExecutorData, ExecutorMetadata, ExecutorOperatingSystemSpecification, ExecutorSpecification,
};
use ballista_core::JobId;
use ballista_scheduler::cluster::{ClusterState, DistributionPolicy, ExecutorSlot};
use ballista_scheduler::planner::DefaultDistributedPlanner;
use ballista_scheduler::state::execution_graph::{ExecutionGraphBox, StaticExecutionGraph};
use ballista_scheduler::state::task_manager::JobInfoCache;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::gang_exec::{GangDescriptor, GangExec};
use jammi_ai::operator::inference_exec::InferenceExecBuilder;
use jammi_ballista::cluster::CatalogClusterState;
use jammi_ballista::placement::DevicePlacement;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::compute_repo::ComputeExecutorRecord;
use jammi_db::catalog::instance::DeviceFact;
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;

async fn catalog(kind: BackendKind) -> Option<Arc<Catalog>> {
    let dir = tempfile::tempdir().unwrap();
    let session = jammi_test_utils::make_test_session(kind, dir.path()).await?;
    std::mem::forget(dir);
    Some(Arc::clone(session.catalog()))
}

/// A throwaway `InferenceSession` (always SQLite — unrelated to the
/// per-backend catalog under test) whose ONLY purpose is to hand
/// [`InferenceExecBuilder`] a real `model_cache()`; the plan built against
/// it is never executed, only inspected/planned, so which catalog backs it
/// is immaterial.
async fn inference_session() -> Arc<jammi_ai::session::InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = jammi_ai::session::InferenceSession::new(cfg)
        .await
        .expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

fn scan() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec!["a"]))]).unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

fn executor_metadata(id: &str, task_slots: u32) -> (ExecutorMetadata, ExecutorData) {
    (
        ExecutorMetadata {
            id: id.to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            grpc_port: 0,
            specification: ExecutorSpecification { task_slots },
            os_info: ExecutorOperatingSystemSpecification::default(),
        },
        ExecutorData {
            executor_id: id.to_string(),
            total_task_slots: task_slots,
            available_task_slots: task_slots,
        },
    )
}

/// Build a real, one-stage `JobInfoCache` around `plan` — no shuffle
/// boundary, so `DefaultDistributedPlanner` produces exactly one
/// `ShuffleWriter`-rooted stage that is immediately `Running` (contract §2.3:
/// this is the SAME shape a placed gang's own single task takes).
fn job_info_cache(job_id: &JobId, plan: Arc<dyn ExecutionPlan>) -> JobInfoCache {
    let mut planner = DefaultDistributedPlanner::new();
    let graph = StaticExecutionGraph::new(
        "test-scheduler",
        job_id,
        "job",
        "session-1",
        plan,
        0,
        Arc::new(SessionConfig::default()),
        &mut planner,
        None,
    )
    .expect("plan_query_stages builds a graph for a single-stage plan");
    let boxed: ExecutionGraphBox = Box::new(graph);
    JobInfoCache::new(boxed)
}

fn active_jobs(job_id: JobId, cache: JobInfoCache) -> Arc<HashMap<JobId, JobInfoCache>> {
    let mut map = HashMap::new();
    map.insert(job_id, cache);
    Arc::new(map)
}

async fn run_both_backends<F, Fut>(f: F)
where
    F: Fn(BackendKind) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    f(BackendKind::Sqlite).await;
    if jammi_test_utils::pg_url_for_tests().is_some() {
        f(BackendKind::Postgres).await;
    } else {
        eprintln!("cluster: JAMMI_TEST_PG_URL unset -- skipping the Postgres arm");
    }
}

/// Register / heartbeat / remove round-trips through the `ClusterState`
/// trait — every verb `CatalogClusterState` maps onto `compute_repo`.
/// Mutation: drop `cache_heartbeat`'s write inside `register_executor` and
/// `executor_heartbeats()` reds (empty where a fresh registration should
/// already be visible).
#[tokio::test]
async fn register_heartbeat_remove_round_trip() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("exec-{}", jammi_test_utils::unique_suffix());
        let (meta, spec) = executor_metadata(&id, 4);

        state
            .register_executor(meta.clone(), spec)
            .await
            .expect("register_executor");
        assert_eq!(state.get_executor_heartbeat(&id).map(|_| ()), Some(()));
        assert!(state.executor_heartbeats().contains_key(&id));
        let listed = state.registered_executor_metadata().await;
        assert!(listed.iter().any(|m| m.id == id));

        state
            .save_executor_heartbeat(ballista_core::serde::protobuf::ExecutorHeartbeat {
                executor_id: id.clone(),
                timestamp: 123,
                metrics: vec![],
                status: None,
                peak_proc_physical_memory: 0,
                peak_proc_virtual_memory: 0,
            })
            .await
            .expect("save_executor_heartbeat");
        assert_eq!(
            state.get_executor_heartbeat(&id).map(|h| h.timestamp),
            Some(123)
        );

        state.remove_executor(&id).await.expect("remove_executor");
        assert!(state.get_executor_heartbeat(&id).is_none());
        let listed = state.registered_executor_metadata().await;
        assert!(!listed.iter().any(|m| m.id == id));
    })
    .await;
}

/// `unbind_tasks` is atomic: a batch that would push ANY executor's
/// `available_slots` past `task_slots` leaves EVERY row in the batch
/// untouched, including one whose own delta was in-bounds. Mutation: change
/// `CatalogClusterState::unbind_tasks` to issue one `adjust_compute_slots`
/// call per executor (instead of one batched call) and this reds (the
/// in-bounds executor's row WOULD move while the other's rejected write
/// leaves the property "no partial batch" false).
#[tokio::test]
async fn unbind_tasks_is_all_or_none() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let ok_id = format!("ok-{}", jammi_test_utils::unique_suffix());
        let bad_id = format!("bad-{}", jammi_test_utils::unique_suffix());
        let (ok_meta, ok_spec) = executor_metadata(&ok_id, 4);
        let (bad_meta, bad_spec) = executor_metadata(&bad_id, 2);
        state.register_executor(ok_meta, ok_spec).await.unwrap();
        state.register_executor(bad_meta, bad_spec).await.unwrap();
        // `bad_id` starts at 2/2 available; unbinding +1 would take it to 3,
        // above its `task_slots = 2` — the whole batch (including `ok_id`'s
        // in-bounds +1) must be refused.
        let batch: Vec<ExecutorSlot> = vec![(ok_id.clone(), 1), (bad_id.clone(), 1)];
        let err = state.unbind_tasks(batch).await;
        assert!(
            err.is_err(),
            "an out-of-bounds executor must fail the whole batch"
        );

        let ok_row = catalog.get_compute_executor(&ok_id).await.unwrap().unwrap();
        assert_eq!(
            ok_row.available_slots, 4,
            "the in-bounds executor's row must be untouched by a batch that failed elsewhere"
        );
    })
    .await;
}

/// (b2's substrate) Two `CatalogClusterState`s over the SAME catalog see
/// each other's registrations — there is no in-process cache a second
/// scheduler instance would miss. Mutation: construct the second state over
/// a FRESH, unrelated catalog and this reds (the registration becomes
/// invisible).
#[tokio::test]
async fn two_cluster_states_over_one_catalog_see_each_others_registrations() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let a = CatalogClusterState::new(Arc::clone(&catalog));
        let b = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("shared-{}", jammi_test_utils::unique_suffix());
        let (meta, spec) = executor_metadata(&id, 2);
        a.register_executor(meta, spec).await.unwrap();
        let seen_by_b = b.registered_executor_metadata().await;
        assert!(seen_by_b.iter().any(|m| m.id == id));
    })
    .await;
}

/// (b3) A GPU-bound stage (an `InferenceExec` naming `Cuda`) never binds to
/// a device-less executor, even when it is the ONLY registered executor and
/// even when a device-bearing one is also available — it binds to the
/// device-bearing one. Mutation: drop the device predicate in
/// `DevicePlacement::bind_tasks` (`eligible = slots[idx].slots > 0 && ...`
/// with the `gpu_bound` conjunct removed) and this reds (the task lands on
/// the device-less executor first, by round-robin order).
#[tokio::test]
async fn gpu_bound_stage_never_binds_to_a_device_less_executor() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let cpu_id = format!("cpu-{}", jammi_test_utils::unique_suffix());
        let gpu_id = format!("gpu-{}", jammi_test_utils::unique_suffix());
        // `cpu_id` registers FIRST so plain round-robin would try it first.
        let (cpu_meta, cpu_spec) = executor_metadata(&cpu_id, 1);
        catalog
            .upsert_compute_executor(&ComputeExecutorRecord {
                executor_id: cpu_id.clone(),
                instance_id: cpu_id.clone(),
                host: cpu_meta.host.clone(),
                port: cpu_meta.port,
                grpc_port: cpu_meta.grpc_port,
                task_slots: cpu_spec.total_task_slots,
                available_slots: cpu_spec.available_task_slots,
                status: "Active".to_string(),
                heartbeat_at: jammi_db::catalog::backend::now_sortable(),
                metadata: String::new(),
                devices: vec![],
            })
            .await
            .unwrap();
        catalog
            .upsert_compute_executor(&ComputeExecutorRecord {
                executor_id: gpu_id.clone(),
                instance_id: gpu_id.clone(),
                host: "127.0.0.1".to_string(),
                port: 0,
                grpc_port: 0,
                task_slots: 1,
                available_slots: 1,
                status: "Active".to_string(),
                heartbeat_at: jammi_db::catalog::backend::now_sortable(),
                metadata: String::new(),
                devices: vec![DeviceFact {
                    kind: "cuda".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let session = inference_session().await;
        let node = InferenceExecBuilder::new(
            scan(),
            ModelSource::hf("m"),
            ModelTask::TextEmbedding,
            vec!["text".to_string()],
            "text".to_string(),
            "src-1".to_string(),
            Arc::clone(session.model_cache()),
            jammi_db::store::manifest::ComputeDeviceKind::Cuda,
        )
        .embedding_dim(Some(2))
        .build()
        .unwrap();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(node);

        let job_id: JobId = "job-gpu".to_string().into();
        let cache = job_info_cache(&job_id, plan);
        let jobs = active_jobs(job_id, cache);

        let policy = DevicePlacement::new(Arc::clone(&catalog));
        let mut cpu_slot = ballista_core::serde::protobuf::AvailableTaskSlots {
            executor_id: cpu_id.clone(),
            slots: 1,
        };
        let mut gpu_slot = ballista_core::serde::protobuf::AvailableTaskSlots {
            executor_id: gpu_id.clone(),
            slots: 1,
        };
        let bound = policy
            .bind_tasks(vec![&mut cpu_slot, &mut gpu_slot], jobs)
            .await
            .expect("bind_tasks");
        assert_eq!(bound.len(), 1, "exactly one task to bind");
        assert_eq!(
            bound[0].0, gpu_id,
            "a GPU-bound task must bind to the device-bearing executor, never the device-less one"
        );
    })
    .await;
}

/// (b6, first half) A `GangExec` stage whose job row is already
/// `claimed_by` an instance OTHER than the stage's own submitter is never
/// bound to ANY slot — the re-launch guard. Mutation: drop the `claim_of`
/// skip in `DevicePlacement::bind_tasks` and this reds (the task binds to
/// whichever executor round-robin's turn lands on).
#[tokio::test]
async fn already_transferred_gang_is_never_bound() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let exec_id = format!("exec-{}", jammi_test_utils::unique_suffix());
        let (meta, spec) = executor_metadata(&exec_id, 2);
        catalog
            .upsert_compute_executor(&ComputeExecutorRecord {
                executor_id: exec_id.clone(),
                instance_id: exec_id.clone(),
                host: meta.host.clone(),
                port: meta.port,
                grpc_port: meta.grpc_port,
                task_slots: spec.total_task_slots,
                available_slots: spec.available_task_slots,
                status: "Active".to_string(),
                heartbeat_at: jammi_db::catalog::backend::now_sortable(),
                metadata: String::new(),
                // A `GangExec` is always GPU-bound (`stage_is_gpu_bound`) —
                // this executor needs a device so the ONLY reason binding
                // can fail is the claim guard under test, not refinement 2.
                devices: vec![DeviceFact {
                    kind: "cuda".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let job_id_s = format!("gang-job-{}", jammi_test_utils::unique_suffix());
        catalog
            .submit_job(SubmitJobParams {
                job_id: &job_id_s,
                kind: "gang_test",
                execution: JobExecution::Inline,
                spec: "{}",
                model_ref: None,
                output_model_id: None,
                model_source: None,
                priority: 0,
            })
            .await
            .expect("submit_job");
        let transferee = format!("transferee-{}", jammi_test_utils::unique_suffix());
        catalog
            .claim_by_id(&job_id_s, &transferee, Duration::from_secs(300))
            .await
            .expect("claim_by_id")
            .expect("row claimed");

        let submitter = format!("submitter-{}", jammi_test_utils::unique_suffix());
        let descriptor = GangDescriptor {
            job_id: job_id_s.clone(),
            attempt: 0,
            world: 2,
            submitter: submitter.clone(),
        };
        let plan: Arc<dyn ExecutionPlan> = Arc::new(GangExec::new(descriptor));
        let job_id: JobId = job_id_s.clone().into();
        let cache = job_info_cache(&job_id, plan);
        let jobs = active_jobs(job_id, cache);

        let policy = DevicePlacement::new(Arc::clone(&catalog));
        let mut slot = ballista_core::serde::protobuf::AvailableTaskSlots {
            executor_id: exec_id.clone(),
            slots: 2,
        };
        let bound = policy
            .bind_tasks(vec![&mut slot], jobs)
            .await
            .expect("bind_tasks");
        assert!(
            bound.is_empty(),
            "a gang whose claim already transferred to {transferee} (!= submitter {submitter}) \
             must never be bound: {bound:?}"
        );
    })
    .await;
}

/// A11 (this unit's own): the slot CAS happens BEFORE the graph's task info
/// is stamped — an executor with ZERO available slots never has a task
/// bound to it even though it is the only registered executor (contract §9
/// A10). Mutation: skip the `bind_compute_slots` CAS and always stamp, and
/// this reds (a task is "bound" to an executor with no real capacity).
#[tokio::test]
async fn a_slot_less_executor_never_gets_a_task_stamped() {
    run_both_backends(|kind| async move {
        let catalog = catalog(kind).await.expect("catalog opens");
        let exec_id = format!("full-{}", jammi_test_utils::unique_suffix());
        catalog
            .upsert_compute_executor(&ComputeExecutorRecord {
                executor_id: exec_id.clone(),
                instance_id: exec_id.clone(),
                host: "127.0.0.1".to_string(),
                port: 0,
                grpc_port: 0,
                task_slots: 1,
                available_slots: 0, // fully booked in the catalog
                status: "Active".to_string(),
                heartbeat_at: jammi_db::catalog::backend::now_sortable(),
                metadata: String::new(),
                devices: vec![],
            })
            .await
            .unwrap();

        let plan = scan();
        let job_id: JobId = "job-full".to_string().into();
        let cache = job_info_cache(&job_id, plan);
        let jobs = active_jobs(job_id, cache);

        let policy = DevicePlacement::new(Arc::clone(&catalog));
        // The in-memory snapshot says 1 free slot (a stale read, or a
        // concurrent scheduler's own optimistic guess) — the CAS against the
        // catalog's REAL count of 0 must still refuse it.
        let mut slot = ballista_core::serde::protobuf::AvailableTaskSlots {
            executor_id: exec_id.clone(),
            slots: 1,
        };
        let bound = policy.bind_tasks(vec![&mut slot], jobs).await.unwrap();
        assert!(
            bound.is_empty(),
            "the catalog's committed available_slots (0) must win over the in-memory \
             snapshot (1): {bound:?}"
        );
    })
    .await;
}
