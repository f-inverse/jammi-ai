//! `CatalogClusterState` / `CatalogJobState` / `DevicePlacement` — oracle over
//! BOTH catalog backends: shared-catalog scheduler state, the `unbind_tasks`
//! atomicity property, and the CAS-before-stamp ordering. Each property is one
//! body taking the backend, run by a `_sqlite` test and, under
//! `live-postgres-tests`, a `_postgres` test.

use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;
use futures::FutureExt;

use ballista_core::serde::scheduler::{
    ExecutorData, ExecutorMetadata, ExecutorOperatingSystemSpecification, ExecutorSpecification,
};
use ballista_core::JobId;
use ballista_scheduler::cluster::{ClusterState, DistributionPolicy, ExecutorSlot};
use ballista_scheduler::config::TaskDistributionPolicy;
use ballista_scheduler::planner::DefaultDistributedPlanner;
use ballista_scheduler::state::execution_graph::{ExecutionGraphBox, StaticExecutionGraph};
use ballista_scheduler::state::task_manager::JobInfoCache;

use jammi_ai::operator::placed_attempt_exec::{PlacedAttempt, PlacedAttemptExec};
use jammi_ballista::client::submit_physical_plan;
use jammi_ballista::cluster::{
    executor_is_live, executor_liveness_window, removal_is_a_loss, CatalogClusterState,
};
use jammi_ballista::placement::DevicePlacement;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::compute_repo::ComputeExecutorRecord;
use jammi_db::catalog::instance::DeviceFact;
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::status::{ComputeExecutorStatus, JobExecution};
use jammi_db::catalog::Catalog;

/// The catalog of a test session on `kind`, over an artifact dir that is
/// never deleted (the SQLite catalog file must outlive every handle the test
/// gives away).
async fn catalog(kind: BackendKind) -> Arc<Catalog> {
    let dir = tempfile::tempdir().unwrap().keep();
    Arc::clone(
        jammi_test_utils::make_test_session(kind, &dir)
            .await
            .catalog(),
    )
}

/// A throwaway `InferenceSession` (always SQLite — unrelated to the
/// per-backend catalog under test) whose ONLY purpose is to hand an
/// inference plan a real model cache; the plan built against it is never
/// executed, only inspected/planned, so which catalog backs it is
/// immaterial.
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
/// `ShuffleWriter`-rooted stage that is immediately `Running` (the SAME
/// shape a placed attempt's own single task takes).
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

/// A heartbeat as `ballista-executor`'s heartbeater builds one: the
/// executor's id, its clock, and one status claim.
fn heartbeat(
    executor_id: &str,
    timestamp: u64,
    status: ballista_core::serde::protobuf::executor_status::Status,
) -> ballista_core::serde::protobuf::ExecutorHeartbeat {
    ballista_core::serde::protobuf::ExecutorHeartbeat {
        executor_id: executor_id.to_string(),
        timestamp,
        metrics: vec![],
        status: Some(ballista_core::serde::protobuf::ExecutorStatus {
            status: Some(status),
        }),
        peak_proc_physical_memory: 0,
        peak_proc_virtual_memory: 0,
    }
}

/// Tests own their rows: the Postgres arm shares one database with every
/// other lane on this host, so every executor row a test registers is
/// removed on the way out — on the GREEN arm and on a PANIC alike (the
/// body runs under `catch_unwind`, the rows are removed, then the panic
/// resumes). A `cuda` row a red test left behind would make a later lane's
/// device-kind refusal admit a plan no live executor can run.
async fn with_owned_rows<F: std::future::Future<Output = ()>>(
    catalog: &Arc<Catalog>,
    owned: &RefCell<Vec<String>>,
    body: F,
) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    let ids = owned.borrow().clone();
    for id in ids {
        catalog.remove_compute_executor(&id).await.ok();
    }
    if let Err(payload) = outcome {
        std::panic::resume_unwind(payload);
    }
}

/// Register / heartbeat / remove round-trips through the `ClusterState`
/// trait — every verb `CatalogClusterState` maps onto `compute_repo`.
/// Mutation: drop `cache_heartbeat`'s write inside `register_executor` and
/// `executor_heartbeats()` reds (empty where a fresh registration should
/// already be visible).
async fn register_heartbeat_remove_round_trip(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("exec-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let (meta, spec) = executor_metadata(&id, 4);

        state
            .register_executor(meta.clone(), spec)
            .await
            .expect("register_executor");
        let cached = state
            .get_executor_heartbeat(&id)
            .expect("a fresh registration is already a cached heartbeat");
        assert!(state.executor_heartbeats().contains_key(&id));
        let listed = state.registered_executor_metadata().await;
        assert!(listed.iter().any(|m| m.id == id));
        // The row and the cache record ONE instant, the whole second the
        // scheduler's expiry sweep compares.
        assert_eq!(
            row_heartbeat_at(&catalog, &id).await,
            stamp_of_seconds(cached.timestamp),
            "the registration row carries the cached heartbeat's own instant"
        );

        state
            .save_executor_heartbeat(heartbeat(
                &id,
                123,
                ballista_core::serde::protobuf::executor_status::Status::Active(String::new()),
            ))
            .await
            .expect("save_executor_heartbeat");
        assert_eq!(
            state.get_executor_heartbeat(&id).map(|h| h.timestamp),
            Some(123)
        );
        assert_eq!(
            row_heartbeat_at(&catalog, &id).await,
            "1970-01-01T00:02:03.000000Z",
            "the row carries the heartbeat's own instant, never a second reading of the clock"
        );
        // A heartbeat that claims no status is a foreign sender, refused
        // typed and recorded nowhere: the cache still reads the last one.
        let refused = state
            .save_executor_heartbeat(ballista_core::serde::protobuf::ExecutorHeartbeat {
                executor_id: id.clone(),
                timestamp: 124,
                metrics: vec![],
                status: None,
                peak_proc_physical_memory: 0,
                peak_proc_virtual_memory: 0,
            })
            .await
            .expect_err("a status-less heartbeat is refused");
        assert!(
            refused.to_string().contains("carries no status"),
            "{refused}"
        );
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

#[tokio::test]
async fn register_heartbeat_remove_round_trip_sqlite() {
    register_heartbeat_remove_round_trip(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn register_heartbeat_remove_round_trip_postgres() {
    register_heartbeat_remove_round_trip(BackendKind::Postgres).await;
}

/// Ballista removes an executor whose task launch failed and THEN unbinds
/// the slots it had reserved there: `unbind_tasks` over a removed
/// executor's id is a no-op for that id, never a refusal, and the live
/// executors in the same batch still get their slots back. Mutation: drop
/// the `registered` filter in `CatalogClusterState::unbind_tasks` and this
/// reds (`adjust_compute_slots` refuses the whole batch on the missing row,
/// so the live executor's slot is never returned).
async fn unbind_tasks_drops_a_removed_executors_slots_and_returns_the_rest(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let live_id = format!("live-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(live_id.clone());
        let gone_id = format!("gone-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(gone_id.clone());
        let (live_meta, live_spec) = executor_metadata(&live_id, 4);
        let (gone_meta, gone_spec) = executor_metadata(&gone_id, 4);
        state.register_executor(live_meta, live_spec).await.unwrap();
        state.register_executor(gone_meta, gone_spec).await.unwrap();
        // Reserve one slot on each (the bind path's own CAS), then lose
        // `gone_id` the way the scheduler does on a failed launch.
        assert!(catalog
            .bind_compute_slots(&live_id, 1)
            .await
            .expect("bind live"));
        assert!(catalog
            .bind_compute_slots(&gone_id, 1)
            .await
            .expect("bind gone"));
        state
            .remove_executor(&gone_id)
            .await
            .expect("remove_executor");

        let batch: Vec<ExecutorSlot> = vec![(live_id.clone(), 1), (gone_id.clone(), 1)];
        state
            .unbind_tasks(batch)
            .await
            .expect("a removed executor's slots are dropped, not refused");

        let live_row = catalog
            .get_compute_executor(&live_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            live_row.available_slots, 4,
            "the live executor's slot came back (3 -> 4)"
        );
        assert!(
            catalog
                .get_compute_executor(&gone_id)
                .await
                .unwrap()
                .is_none(),
            "the removed executor stays removed (no resurrection by unbind)"
        );
    })
    .await;
}

#[tokio::test]
async fn unbind_tasks_drops_a_removed_executors_slots_and_returns_the_rest_sqlite() {
    unbind_tasks_drops_a_removed_executors_slots_and_returns_the_rest(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn unbind_tasks_drops_a_removed_executors_slots_and_returns_the_rest_postgres() {
    unbind_tasks_drops_a_removed_executors_slots_and_returns_the_rest(BackendKind::Postgres).await;
}

/// `unbind_tasks` is atomic: a batch that would push ANY executor's
/// `available_slots` past `task_slots` leaves EVERY row in the batch
/// untouched, including one whose own delta was in-bounds. Mutation: change
/// `CatalogClusterState::unbind_tasks` to issue one `adjust_compute_slots`
/// call per executor (instead of one batched call) and this reds (the
/// in-bounds executor's row WOULD move while the other's rejected write
/// leaves the property "no partial batch" false).
async fn unbind_tasks_is_all_or_none(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let ok_id = format!("ok-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(ok_id.clone());
        let bad_id = format!("bad-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(bad_id.clone());
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

#[tokio::test]
async fn unbind_tasks_is_all_or_none_sqlite() {
    unbind_tasks_is_all_or_none(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn unbind_tasks_is_all_or_none_postgres() {
    unbind_tasks_is_all_or_none(BackendKind::Postgres).await;
}

/// (b2's substrate) Two `CatalogClusterState`s over the SAME catalog see
/// each other's registrations — there is no in-process cache a second
/// scheduler instance would miss. Mutation: construct the second state over
/// a FRESH, unrelated catalog and this reds (the registration becomes
/// invisible).
async fn two_cluster_states_over_one_catalog_see_each_others_registrations(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let a = CatalogClusterState::new(Arc::clone(&catalog));
        let b = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("shared-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let (meta, spec) = executor_metadata(&id, 2);
        a.register_executor(meta, spec).await.unwrap();
        let seen_by_b = b.registered_executor_metadata().await;
        assert!(seen_by_b.iter().any(|m| m.id == id));
    })
    .await;
}

#[tokio::test]
async fn two_cluster_states_over_one_catalog_see_each_others_registrations_sqlite() {
    two_cluster_states_over_one_catalog_see_each_others_registrations(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn two_cluster_states_over_one_catalog_see_each_others_registrations_postgres() {
    two_cluster_states_over_one_catalog_see_each_others_registrations(BackendKind::Postgres).await;
}

/// (b3) KIND MATCH: a `Cuda`-stamped `InferenceExec` never binds to a
/// device-LESS executor (empty `devices`), even when it is the ONLY
/// registered executor and even when a `cuda`-bearing one is also
/// available — it binds to the `cuda`-bearing one. Mutation: drop the
/// device predicate in `DevicePlacement::bind_tasks` (`eligible =
/// slots[idx].slots > 0 && ...` with the kind-match conjunct removed) and
/// this reds (the task lands on the device-less executor first, by
/// round-robin order).
async fn cuda_stamped_stage_never_binds_to_a_device_less_executor(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let cpu_id = format!("cpu-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(cpu_id.clone());
        let gpu_id = format!("gpu-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(gpu_id.clone());
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
                metadata: String::new(),
                devices: vec![DeviceFact {
                    kind: "cuda".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let session = inference_session().await;
        let plan = crate::inference_plan(
            &session,
            scan(),
            jammi_db::store::manifest::ComputeDeviceKind::Cuda,
            1,
        );

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
            "a Cuda-stamped task must bind to the cuda-bearing executor, never the device-less one"
        );
    })
    .await;
}

#[tokio::test]
async fn cuda_stamped_stage_never_binds_to_a_device_less_executor_sqlite() {
    cuda_stamped_stage_never_binds_to_a_device_less_executor(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn cuda_stamped_stage_never_binds_to_a_device_less_executor_postgres() {
    cuda_stamped_stage_never_binds_to_a_device_less_executor(BackendKind::Postgres).await;
}

/// (b3) KIND MATCH, the properly-registered case: a `Cuda`-stamped
/// `InferenceExec` never binds to a CPU-ONLY executor (`devices =
/// [{cpu,0}]`, a legitimate registration, not an empty one) — it binds to
/// the `cuda`-bearing one. The companion positive case
/// (`cpu_stamped_stage_binds_to_a_cpu_only_executor`) proves the SAME
/// uniform rule does not accidentally exclude CPU workloads when CPU
/// devices are properly registered.
async fn cuda_stamped_stage_never_binds_to_a_cpu_only_executor(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let cpu_id = format!("cpu-only-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(cpu_id.clone());
        let gpu_id = format!("gpu-only-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(gpu_id.clone());
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
                metadata: String::new(),
                devices: vec![DeviceFact {
                    kind: "cpu".to_string(),
                    ordinal: 0,
                }],
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
                metadata: String::new(),
                devices: vec![DeviceFact {
                    kind: "cuda".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let session = inference_session().await;
        let plan = crate::inference_plan(
            &session,
            scan(),
            jammi_db::store::manifest::ComputeDeviceKind::Cuda,
            1,
        );

        let job_id: JobId = "job-cuda".to_string().into();
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
            "a Cuda-stamped stage must bind to the cuda-bearing executor, never a properly \
         registered cpu-only one"
        );
    })
    .await;
}

#[tokio::test]
async fn cuda_stamped_stage_never_binds_to_a_cpu_only_executor_sqlite() {
    cuda_stamped_stage_never_binds_to_a_cpu_only_executor(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn cuda_stamped_stage_never_binds_to_a_cpu_only_executor_postgres() {
    cuda_stamped_stage_never_binds_to_a_cpu_only_executor(BackendKind::Postgres).await;
}

/// (b3) The positive KIND MATCH case: a `Cpu`-stamped `InferenceExec` DOES
/// bind to a properly-registered CPU-only executor (`devices =
/// [{cpu,0}]`), never refused by the uniform kind-match rule the previous
/// test exercises negatively.
async fn cpu_stamped_stage_binds_to_a_cpu_only_executor(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let cpu_id = format!("cpu-only-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(cpu_id.clone());
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
                metadata: String::new(),
                devices: vec![DeviceFact {
                    kind: "cpu".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let session = inference_session().await;
        let plan = crate::inference_plan(
            &session,
            scan(),
            jammi_db::store::manifest::ComputeDeviceKind::Cpu,
            1,
        );

        let job_id: JobId = "job-cpu".to_string().into();
        let cache = job_info_cache(&job_id, plan);
        let jobs = active_jobs(job_id, cache);

        let policy = DevicePlacement::new(Arc::clone(&catalog));
        let mut cpu_slot = ballista_core::serde::protobuf::AvailableTaskSlots {
            executor_id: cpu_id.clone(),
            slots: 1,
        };
        let bound = policy
            .bind_tasks(vec![&mut cpu_slot], jobs)
            .await
            .expect("bind_tasks");
        assert_eq!(
            bound.len(),
            1,
            "a Cpu-stamped stage must bind to the properly-registered cpu-only executor"
        );
        assert_eq!(bound[0].0, cpu_id);
    })
    .await;
}

#[tokio::test]
async fn cpu_stamped_stage_binds_to_a_cpu_only_executor_sqlite() {
    cpu_stamped_stage_binds_to_a_cpu_only_executor(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn cpu_stamped_stage_binds_to_a_cpu_only_executor_postgres() {
    cpu_stamped_stage_binds_to_a_cpu_only_executor(BackendKind::Postgres).await;
}

/// (b6, first half) A `PlacedAttemptExec` stage whose job row is already
/// `claimed_by` an instance OTHER than the stage's own submitter is never
/// bound to ANY slot — the re-launch guard. Mutation: drop the `claim_of`
/// skip in `DevicePlacement::bind_tasks` and this reds (the task binds to
/// whichever executor round-robin's turn lands on).
async fn already_transferred_attempt_is_never_bound(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let exec_id = format!("exec-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(exec_id.clone());
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
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
                metadata: String::new(),
                // The descriptor below is stamped `Cuda` — this executor
                // must list a matching device so the ONLY reason binding
                // can fail is the claim guard under test (refinement 3),
                // never refinement 2's kind match.
                devices: vec![DeviceFact {
                    kind: "cuda".to_string(),
                    ordinal: 0,
                }],
            })
            .await
            .unwrap();

        let job_id_s = format!("placed-job-{}", jammi_test_utils::unique_suffix());
        catalog
            .submit_job(SubmitJobParams {
                job_id: &job_id_s,
                kind: "placed_test",
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
        let descriptor = PlacedAttempt {
            job_id: job_id_s.clone(),
            attempt: 0,
            submitter: submitter.clone(),
            device_kind: jammi_db::store::manifest::ComputeDeviceKind::Cuda,
            claimed_at: chrono::Utc::now(),
        };
        let plan: Arc<dyn ExecutionPlan> = Arc::new(PlacedAttemptExec::new(descriptor));
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
            "an attempt whose claim already transferred to {transferee} (!= submitter {submitter}) \
         must never be bound: {bound:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn already_transferred_attempt_is_never_bound_sqlite() {
    already_transferred_attempt_is_never_bound(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn already_transferred_attempt_is_never_bound_postgres() {
    already_transferred_attempt_is_never_bound(BackendKind::Postgres).await;
}

/// The slot CAS happens BEFORE the graph's task info is stamped — an
/// executor with ZERO available slots never has a task bound to it even
/// though it is the only registered executor. Mutation: skip the
/// `bind_compute_slots` CAS and always stamp, and
/// this reds (a task is "bound" to an executor with no real capacity).
async fn a_slot_less_executor_never_gets_a_task_stamped(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let exec_id = format!("full-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(exec_id.clone());
        catalog
            .upsert_compute_executor(&ComputeExecutorRecord {
                executor_id: exec_id.clone(),
                instance_id: exec_id.clone(),
                host: "127.0.0.1".to_string(),
                port: 0,
                grpc_port: 0,
                task_slots: 1,
                available_slots: 0, // fully booked in the catalog
                status: ComputeExecutorStatus::Active,
                heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
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

#[tokio::test]
async fn a_slot_less_executor_never_gets_a_task_stamped_sqlite() {
    a_slot_less_executor_never_gets_a_task_stamped(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn a_slot_less_executor_never_gets_a_task_stamped_postgres() {
    a_slot_less_executor_never_gets_a_task_stamped(BackendKind::Postgres).await;
}

/// A canonical stamp of a whole-second Unix instant — the shape every
/// `heartbeat_at` the cluster state writes has.
fn stamp_of_seconds(secs: u64) -> String {
    jammi_db::catalog::lease::canonical_stamp(
        chrono::DateTime::from_timestamp(i64::try_from(secs).unwrap(), 0).unwrap(),
    )
}

async fn row_heartbeat_at(catalog: &Catalog, id: &str) -> String {
    catalog
        .get_compute_executor(id)
        .await
        .unwrap()
        .expect("the executor's row")
        .heartbeat_at
}

fn record(id: &str, status: ComputeExecutorStatus, heartbeat_at: String) -> ComputeExecutorRecord {
    ComputeExecutorRecord {
        executor_id: id.to_string(),
        instance_id: id.to_string(),
        host: "127.0.0.1".to_string(),
        port: 0,
        grpc_port: 0,
        task_slots: 1,
        available_slots: 1,
        status,
        heartbeat_at,
        metadata: String::new(),
        devices: vec![],
    }
}

/// Ballista's own expiry test (`ExecutorManager::get_expired_executors`):
/// expired once `heartbeat <= now - executor_timeout_seconds`, every term
/// in whole Unix seconds and the threshold clamped at the epoch. The
/// tables below judge `executor_is_live` against THIS, never against a
/// second window arithmetic of the test's own.
fn ballista_expired(heartbeat_secs: u64, now: chrono::DateTime<chrono::Utc>) -> bool {
    let window = u64::try_from(executor_liveness_window().num_seconds()).unwrap();
    let threshold = u64::try_from(now.timestamp())
        .unwrap()
        .saturating_sub(window);
    heartbeat_secs <= threshold
}

/// `executor_is_live` — the ONE predicate the binder and the submit-edge
/// refusal share — is `Active` AND a heartbeat Ballista's expiry sweep
/// would not have expired; every other row (a `Terminating` one, a `Dead`
/// one, a stale timestamp, an unparseable one) is not live. Mutation: drop
/// the status arm and the `Terminating` row reads live; drop the window
/// and the stale row reads live; compare the stamp at sub-second
/// precision, or with `<` in place of Ballista's `<=`, and the edge rows
/// below read live while the sweep has already expired them.
#[test]
fn executor_is_live_table() {
    let now = chrono::Utc::now();
    let stamp = |t: chrono::DateTime<chrono::Utc>| t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();
    let fresh = stamp(now);

    // The window's edge, as a sweep meets it: the executor's last
    // heartbeat was cached at whole second `S`, and the sweep runs a
    // fraction of a second past `S + window`. Ballista expires it — the
    // row's verdict must be the same, whether the row's stamp is `S` itself
    // or a finer reading of the same clock taken a fraction later (`S +
    // 0.62 s`: within the window by sub-second arithmetic, expired by
    // Ballista's).
    let s = chrono::DateTime::from_timestamp(now.timestamp(), 0).unwrap();
    let sweep = s + executor_liveness_window() + chrono::Duration::milliseconds(590);
    for edge in [s, s + chrono::Duration::milliseconds(620)] {
        assert!(ballista_expired(
            u64::try_from(s.timestamp()).unwrap(),
            sweep
        ));
        assert!(
            !executor_is_live(
                &record("e", ComputeExecutorStatus::Active, stamp(edge)),
                sweep
            ),
            "a heartbeat the sweep expired is not live: stamp {edge}, sweep {sweep}"
        );
    }
    // One second later the sweep keeps it, and so does the row.
    let kept = s + chrono::Duration::seconds(1);
    assert!(!ballista_expired(
        u64::try_from(kept.timestamp()).unwrap(),
        sweep
    ));
    assert!(executor_is_live(
        &record("e", ComputeExecutorStatus::Active, stamp(kept)),
        sweep
    ));
    assert!(executor_is_live(
        &record("e", ComputeExecutorStatus::Active, fresh.clone()),
        now
    ));
    assert!(!executor_is_live(
        &record("e", ComputeExecutorStatus::Terminating, fresh.clone()),
        now
    ));
    assert!(!executor_is_live(
        &record("e", ComputeExecutorStatus::Dead, fresh),
        now
    ));
    let inside = stamp(now - executor_liveness_window() + chrono::Duration::seconds(1));
    assert!(executor_is_live(
        &record("e", ComputeExecutorStatus::Active, inside),
        now
    ));
    let outside = stamp(now - executor_liveness_window() - chrono::Duration::seconds(1));
    assert!(!executor_is_live(
        &record("e", ComputeExecutorStatus::Active, outside),
        now
    ));
    assert!(!executor_is_live(
        &record(
            "e",
            ComputeExecutorStatus::Active,
            "2026-01-01T00:00:00.000000Z".into()
        ),
        now
    ));
    assert!(!executor_is_live(
        &record("e", ComputeExecutorStatus::Active, "not-a-timestamp".into()),
        now
    ));
}

/// `removal_is_a_loss` — whether an executor's removal fails the jobs
/// bound to it — reads the catalog row through the ONE liveness predicate:
/// a live `Active` row (a launch that could not reach a healthy executor,
/// which registers again) is not a loss; a `Terminating` row, a `Dead`
/// one, an `Active` row whose heartbeat is past the window (the expiry
/// sweep's), and no row at all are. Mutation: decide from the status alone
/// and the stale `Active` row reads live; drop the row read and every
/// removal is a loss.
#[test]
fn removal_is_a_loss_table() {
    let now = chrono::Utc::now();
    let stamp = |t: chrono::DateTime<chrono::Utc>| t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();
    let fresh = stamp(now);
    let expired = stamp(now - executor_liveness_window() - chrono::Duration::seconds(1));
    assert!(!removal_is_a_loss(
        Some(&record("e", ComputeExecutorStatus::Active, fresh.clone())),
        now
    ));
    assert!(removal_is_a_loss(
        Some(&record(
            "e",
            ComputeExecutorStatus::Terminating,
            fresh.clone()
        )),
        now
    ));
    assert!(removal_is_a_loss(
        Some(&record("e", ComputeExecutorStatus::Dead, fresh)),
        now
    ));
    assert!(removal_is_a_loss(
        Some(&record("e", ComputeExecutorStatus::Active, expired)),
        now
    ));
    // The sweep's own case: it expired the executor at the window's edge,
    // and its removal is a loss — never the launch-failure reading, which
    // would leave the placed job waiting on an executor that is gone.
    let s = chrono::DateTime::from_timestamp(now.timestamp(), 0).unwrap();
    let sweep = s + executor_liveness_window() + chrono::Duration::milliseconds(590);
    assert!(ballista_expired(
        u64::try_from(s.timestamp()).unwrap(),
        sweep
    ));
    assert!(removal_is_a_loss(
        Some(&record("e", ComputeExecutorStatus::Active, stamp(s))),
        sweep
    ));
    assert!(removal_is_a_loss(None, now));
}

/// A scheduler that starts over rows other processes wrote seeds its
/// heartbeat cache from each row's own stamp — the instant the executor
/// was last heard from — so its first expiry sweep judges a dead
/// executor's row by that instant, not by this process's start. Mutation:
/// seed the cache with "now" and the second state reads the registration's
/// second, never `123`.
async fn init_seeds_the_heartbeat_cache_from_the_rows(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let first = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("exec-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let (meta, spec) = executor_metadata(&id, 1);
        first.register_executor(meta, spec).await.unwrap();
        first
            .save_executor_heartbeat(heartbeat(
                &id,
                123,
                ballista_core::serde::protobuf::executor_status::Status::Active(String::new()),
            ))
            .await
            .unwrap();

        let second = CatalogClusterState::new(Arc::clone(&catalog));
        second.init().await.expect("init reads the rows");
        assert_eq!(
            second.get_executor_heartbeat(&id).map(|h| h.timestamp),
            Some(123),
            "the restarted scheduler's cache carries the row's own instant"
        );
    })
    .await;
}

#[tokio::test]
async fn init_seeds_the_heartbeat_cache_from_the_rows_sqlite() {
    init_seeds_the_heartbeat_cache_from_the_rows(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn init_seeds_the_heartbeat_cache_from_the_rows_postgres() {
    init_seeds_the_heartbeat_cache_from_the_rows(BackendKind::Postgres).await;
}

/// The binder never binds to an executor that is not live: a `Terminating`
/// heartbeat (`ExecutorRole::begin_drain`'s report at the DRAIN instant)
/// and a stale `heartbeat_at` (a process that never ran its graceful
/// `remove_executor`) both drop the row from the candidate slots, so the
/// task lands on the one live executor even though all three have a free
/// slot. Mutation: drop `executor_is_live` from `bind_schedulable_tasks`'s
/// filter and the terminating or stale executor (registered FIRST, so
/// round-robin would try it first) takes the task.
async fn terminating_and_stale_executors_are_never_bound(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let term_id = format!("a-term-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(term_id.clone());
        let stale_id = format!("b-stale-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(stale_id.clone());
        let live_id = format!("c-live-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(live_id.clone());

        let (meta, spec) = executor_metadata(&term_id, 1);
        state.register_executor(meta, spec).await.unwrap();
        state
            .save_executor_heartbeat(heartbeat(
                &term_id,
                1,
                ballista_core::serde::protobuf::executor_status::Status::Terminating(String::new()),
            ))
            .await
            .unwrap();
        catalog
            .upsert_compute_executor(&record(
                &stale_id,
                ComputeExecutorStatus::Active,
                "2026-01-01T00:00:00.000000Z".to_string(),
            ))
            .await
            .unwrap();
        let (meta, spec) = executor_metadata(&live_id, 1);
        state.register_executor(meta, spec).await.unwrap();

        let plan = scan();
        let job_id: JobId = "job-liveness".to_string().into();
        let jobs = active_jobs(job_id.clone(), job_info_cache(&job_id, plan));
        // Scoped to this test's three executors (the trait's own
        // `executors` filter) so a concurrent test's live executor on
        // the shared Postgres never takes the task instead.
        let scope: std::collections::HashSet<String> =
            [term_id.clone(), stale_id.clone(), live_id.clone()]
                .into_iter()
                .collect();
        let bound = state
            .bind_schedulable_tasks(
                TaskDistributionPolicy::Custom(Arc::new(DevicePlacement::new(Arc::clone(
                    &catalog,
                )))),
                jobs,
                Some(scope),
            )
            .await
            .unwrap();
        let ids: Vec<&str> = bound.iter().map(|(id, _)| id.as_str()).collect();
        assert!(!ids.is_empty(), "the live executor takes the task");
        assert!(
            ids.iter().all(|id| *id == live_id),
            "only the live executor is ever bound; got {ids:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn terminating_and_stale_executors_are_never_bound_sqlite() {
    terminating_and_stale_executors_are_never_bound(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn terminating_and_stale_executors_are_never_bound_postgres() {
    terminating_and_stale_executors_are_never_bound(BackendKind::Postgres).await;
}

/// A draining executor never reads as `Active` again: after its
/// `Terminating` report (`ExecutorRole::begin_drain`), an `Active`
/// heartbeat — the executor's periodic heartbeater's report, built before
/// the drain flag flipped and delivered after — refreshes the row's
/// `heartbeat_at` and leaves it `Terminating`, so the row is still not live
/// and the binder still never binds it. Deterministic: the two heartbeats
/// are delivered in that order through the cluster state itself, no timing.
/// Mutation: write the heartbeat's status unconditionally and the second
/// report revives the row.
async fn an_active_heartbeat_after_terminating_never_revives_the_row(kind: BackendKind) {
    let catalog = catalog(kind).await;
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let state = CatalogClusterState::new(Arc::clone(&catalog));
        let id = format!("draining-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let (meta, spec) = executor_metadata(&id, 1);
        state.register_executor(meta, spec).await.unwrap();
        let registered = catalog.get_compute_executor(&id).await.unwrap().unwrap();
        assert_eq!(registered.status, ComputeExecutorStatus::Active);
        assert!(executor_is_live(&registered, chrono::Utc::now()));

        state
            .save_executor_heartbeat(heartbeat(
                &id,
                1,
                ballista_core::serde::protobuf::executor_status::Status::Terminating(String::new()),
            ))
            .await
            .unwrap();
        let draining = catalog.get_compute_executor(&id).await.unwrap().unwrap();
        assert_eq!(draining.status, ComputeExecutorStatus::Terminating);

        state
            .save_executor_heartbeat(heartbeat(
                &id,
                2,
                ballista_core::serde::protobuf::executor_status::Status::Active(String::new()),
            ))
            .await
            .unwrap();
        let after = catalog.get_compute_executor(&id).await.unwrap().unwrap();
        assert_eq!(
            after.status,
            ComputeExecutorStatus::Terminating,
            "an Active heartbeat after the drain report never revives the row"
        );
        assert!(
            after.heartbeat_at >= draining.heartbeat_at,
            "the late heartbeat still refreshes the timestamp"
        );
        assert!(
            !executor_is_live(&after, chrono::Utc::now()),
            "a draining executor is not live to the binder or the submit edge"
        );

        let plan = scan();
        let job_id: JobId = "job-draining".to_string().into();
        let jobs = active_jobs(job_id.clone(), job_info_cache(&job_id, plan));
        let scope: std::collections::HashSet<String> = [id.clone()].into_iter().collect();
        let bound = state
            .bind_schedulable_tasks(
                TaskDistributionPolicy::Custom(Arc::new(DevicePlacement::new(Arc::clone(
                    &catalog,
                )))),
                jobs,
                Some(scope),
            )
            .await
            .unwrap();
        let ids: Vec<&str> = bound.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.is_empty(),
            "the draining executor is never bound: {ids:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn an_active_heartbeat_after_terminating_never_revives_the_row_sqlite() {
    an_active_heartbeat_after_terminating_never_revives_the_row(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn an_active_heartbeat_after_terminating_never_revives_the_row_postgres() {
    an_active_heartbeat_after_terminating_never_revives_the_row(BackendKind::Postgres).await;
}

/// The submit edge's device-kind refusal reads LIVE executors only: a
/// `cuda` row a killed GPU executor left behind (stale `heartbeat_at`) never
/// admits a Cuda-stamped plan — it is refused before submitting, the
/// outcome the prose promises ("never parked unschedulable") — while a
/// fresh `cuda` row does admit it (the call then proceeds to the scheduler
/// and fails on the unreachable address, a DIFFERENT error). Mutation: drop
/// `executor_is_live` from `submit_physical_plan`'s read and the stale row
/// admits the plan (the first call does not return the refusal).
#[tokio::test]
async fn a_stale_cuda_row_never_admits_a_cuda_plan_at_the_submit_edge() {
    let session = inference_session().await;
    let catalog = Arc::clone(session.catalog_arc());
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let stale_id = format!("stale-cuda-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(stale_id.clone());
        let cuda = vec![DeviceFact {
            kind: "cuda".to_string(),
            ordinal: 0,
        }];
        let mut stale = record(
            &stale_id,
            ComputeExecutorStatus::Active,
            "2026-01-01T00:00:00.000000Z".to_string(),
        );
        stale.devices = cuda.clone();
        catalog.upsert_compute_executor(&stale).await.unwrap();

        let cuda_plan = || -> Arc<dyn ExecutionPlan> {
            crate::inference_plan(
                &session,
                scan(),
                jammi_db::store::manifest::ComputeDeviceKind::Cuda,
                1,
            )
        };
        let unreachable = "http://127.0.0.1:9";
        let err = submit_physical_plan(&session, unreachable, cuda_plan())
            .await
            .err()
            .expect("a stale cuda row admits nothing");
        // Refused typed at the submit edge, as the plane's one `Unheld`
        // class: the plan's `Cuda`, and no live executor listing it among
        // the kinds held.
        match jammi_db::error::JammiError::from(err) {
            jammi_db::error::JammiError::Unheld(
                jammi_db::compute_plane::Unheld::NoExecutorOfKind { required, held },
            ) => {
                assert_eq!(required, jammi_db::store::manifest::ComputeDeviceKind::Cuda);
                assert!(
                    !held.contains(&jammi_db::store::manifest::ComputeDeviceKind::Cuda),
                    "a stale cuda row must not count as held: {held:?}"
                );
            }
            other => panic!("expected Unheld(NoExecutorOfKind) at the submit edge, got {other:?}"),
        }

        let fresh_id = format!("fresh-cuda-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(fresh_id.clone());
        let mut fresh = record(
            &fresh_id,
            ComputeExecutorStatus::Active,
            jammi_db::catalog::lease::canonical_stamp_now(),
        );
        fresh.devices = cuda;
        catalog.upsert_compute_executor(&fresh).await.unwrap();
        let err = tokio::time::timeout(
            Duration::from_secs(30),
            submit_physical_plan(&session, unreachable, cuda_plan()),
        )
        .await
        .expect("the unreachable scheduler fails fast")
        .err()
        .expect("no scheduler listens on the address");
        assert!(
            !err.to_string()
                .contains("no live registered compute executor"),
            "a fresh cuda row admits the plan past the refusal: {err}"
        );
    })
    .await;
}
