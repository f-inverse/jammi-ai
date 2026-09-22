//! The three(+)-process Ballista lane: placed embedding and gang jobs, byte
//! parity with in-process execution, executor and scheduler death, and
//! standby schedulers. `required-features = ["live-distributed-tests"]`
//! (`Cargo.toml`); needs, on top of that:
//!
//! 1. `cargo build -p jammi-server --bin jammi-server --features
//!    storage-s3` into the SAME `CARGO_TARGET_DIR` this test binary is
//!    built into (`harness::jammi_server_binary`).
//! 2. `JAMMI_TEST_PG_URL` (a live Postgres).
//! 3. `JAMMI_TEST_S3_ENDPOINT` / `_S3_BUCKET` / `AWS_ACCESS_KEY_ID` /
//!    `AWS_SECRET_ACCESS_KEY` (an S3-compatible object store, the lane's S3 store in
//!    dev/CI).
//!
//! A test fails naming the first of these variables that is unset
//! ([`jammi_test_utils::DistributedBackends::from_env`]).
//!
//! **Not read through Ballista's API**: the embedding test's "both
//! executors executed at least one task of the job" is read from the
//! SCHEDULER PROCESS's OWN captured log (the `tracing::info!` line at the
//! one call site `DevicePlacement::bind_tasks` actually binds a task,
//! `crates/jammi-ballista/src/placement.rs`) rather than through
//! Ballista's `SchedulerGrpcClient::get_job_status`: the public
//! `ballista_core::execution_plans::execute_physical_plan` a client-side
//! `submit_physical_plan` caller uses never returns or exposes the
//! internally-minted `job_id` (confirmed by reading
//! `ballista-core-54.1.0/src/execution_plans/distributed_query.rs:332-405` —
//! the job id lives only in a private `Arc<Mutex<Option<JobId>>>` the
//! function never returns), so a caller outside the scheduler process has no
//! `job_id` to poll `get_job_status` with. The log line carries the SAME fact (which executor a stage/partition
//! bound to), read from the one process that actually knows it.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::operator::placed_attempt_exec::{PlacedAttempt, PlacedAttemptExec};
use jammi_ai::pipeline::embedding::build_embedding_plan;
use jammi_ai::session::InferenceSession;
use jammi_ballista::client::submit_physical_plan;
use jammi_ballista::placement::BOUND_TASK_LOG;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::SINK_WRITE_LOG;

use harness::{BallistaRole, Fleet, JobSize, ProcSpec, WorkerRole};
use jammi_test_utils::{flight_statement, DistributedBackends};

/// The deepest (leaf) plan node's own partition count — the scan stage's,
/// whatever wraps it (`jammi_ai::operator::inference_exec::plan_inference`'s
/// shape: the coalesce, the numbered input, the exchange, `InferenceExec`,
/// the merge).
fn leaf_partition_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let children = plan.children();
    match children.first() {
        Some(first) => leaf_partition_count(first),
        None => plan.output_partitioning().partition_count(),
    }
}

/// Arrow IPC (stream format) bytes of one batch — the unit the parity
/// assertions compare.
fn ipc_bytes(batch: &RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut writer =
            arrow::ipc::writer::StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
    buf
}

/// `batches` concatenated IN THE ORDER THEY ARRIVED, without `_latency_ms`
/// (wall-clock timing, legitimately different between two runs). Never
/// re-sorted: the plan's merge on `_ordinal` makes the row order part of what
/// a placed run must reproduce.
fn concat_in_arrival_order(batches: &[RecordBatch]) -> RecordBatch {
    let non_empty: Vec<RecordBatch> = batches
        .iter()
        .filter(|b| b.num_rows() > 0)
        .cloned()
        .collect();
    assert!(!non_empty.is_empty(), "no non-empty batches to concat");
    let combined = arrow::compute::concat_batches(&non_empty[0].schema(), &non_empty).unwrap();
    drop_column(&combined, "_latency_ms")
}

/// `batch` with `name` projected out, if present (a no-op otherwise) —
/// `RecordBatch` has no `drop_column`, only `project` by index.
fn drop_column(batch: &RecordBatch, name: &str) -> RecordBatch {
    let schema = batch.schema();
    let Some((idx, _)) = schema.column_with_name(name) else {
        return batch.clone();
    };
    let keep: Vec<usize> = (0..schema.fields().len()).filter(|&i| i != idx).collect();
    batch.project(&keep).expect("project drops one column")
}

/// Build the standard 3-process ballista fleet: `lane-1` hosts the
/// scheduler + an executor and is the ONLY process whose `[worker] kinds`
/// includes `fine_tune` (`jammi_db::catalog::jobs_repo::Catalog::list_gang_members`
/// admits
/// a candidate only when its `workers.kinds` token set contains the job's
/// own kind); `lane-2`/`lane-3` host executors only and list
/// `context_predictor`, so they are fleet MEMBERS (`[worker] enabled =
/// true`) but never claimants of a `fine_tune` job themselves — the
/// placed-executor-turned-coordinator's OWN `list_gang_members(kind =
/// "fine_tune")` call therefore always finds exactly `lane-1` as its sole
/// rank-1 candidate, deterministically, regardless of whether `lane-2` or
/// `lane-3` was the one placed onto.
fn standard_fleet_specs() -> (Vec<ProcSpec>, u16) {
    let scheduler_port = jammi_test_utils::free_port();
    let specs = vec![
        ProcSpec::fresh(
            BallistaRole::SchedulerAndExecutor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["fine_tune"]),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        ),
    ];
    (specs, scheduler_port)
}

/// This fleet member's minted instance id, resolved from its seeded label
/// via the shared catalog's `workers` rows (the reverse of
/// `harness::label_of`). The shared Postgres catalog is PERSISTENT across
/// every test run on this host (never per-test-isolated the way SQLite
/// fixtures are), so a lookup by a FIXED label would silently resolve a
/// prior run's stale row — `Fleet::spawn` mints a run-unique label per
/// process precisely so this lookup is unambiguous.
async fn instance_id_of_label(session: &InferenceSession, label: &str) -> String {
    let workers = session.catalog().list_workers().await.unwrap();
    workers
        .iter()
        .find(|w| w.label.as_deref() == Some(label))
        .map(|w| w.instance_id.clone())
        .unwrap_or_else(|| panic!("no worker row for label {label:?}; workers = {workers:?}"))
}

/// Wait until every executor-hosting member of `fleet`
/// (`Fleet::executor_labels` — the roles that render `[ballista.executor]`;
/// a client-role process registers no executor and is never waited on)
/// has (1) a `workers` row and (2) a `compute_executors` registration,
/// THEN return their instance ids in the same order as
/// `fleet.executor_labels()`. Never a bare row COUNT
/// (`list_compute_executors().len() >= n`): the shared Postgres catalog
/// accumulates rows from every OTHER test run on this host (SIGKILL never
/// runs a graceful `remove_executor`), so a count-based wait can
/// spuriously observe stale rows and return before THIS fleet's own
/// processes are actually up.
async fn await_fleet_registered(session: &Arc<InferenceSession>, fleet: &Fleet) -> Vec<String> {
    let labels = fleet.executor_labels();
    let ok = harness::await_condition(Duration::from_secs(60), || {
        futures::executor::block_on(async {
            let workers = session.catalog().list_workers().await.unwrap_or_default();
            labels
                .iter()
                .all(|l| workers.iter().any(|w| w.label.as_deref() == Some(*l)))
        })
    })
    .await;
    if !ok {
        fleet.dump_diagnostics("workers rows never appeared");
    }
    assert!(ok, "timed out waiting for `workers` rows for {labels:?}");

    let mut ids = Vec::with_capacity(labels.len());
    for label in labels {
        ids.push(instance_id_of_label(session, label).await);
    }

    let ok = harness::await_condition(Duration::from_secs(60), || {
        futures::executor::block_on(async {
            let execs = session
                .catalog()
                .list_compute_executor_devices()
                .await
                .unwrap_or_default();
            let ex_ids: std::collections::HashSet<&str> =
                execs.iter().map(|(id, _)| id.as_str()).collect();
            ids.iter().all(|id| ex_ids.contains(id.as_str()))
        })
    })
    .await;
    assert!(
        ok,
        "timed out waiting for compute_executors registrations for {ids:?}"
    );
    ids
}

// ─── an embedding plan across two executors ──────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embedding_job_across_two_executors_matches_in_process() {
    embedding_job_matches_in_process("embedding_job_across_two_executors_matches_in_process", 1)
        .await;
}

/// The same parity at a fan-out of four: the inference stage runs as four
/// tasks across the executors, each forwarding whole chunks, and the merge
/// stage restores the one row sequence — so the placed bytes equal the
/// in-process bytes in order, with nothing re-sorted on either side.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embedding_job_fanned_out_four_ways_matches_in_process() {
    embedding_job_matches_in_process("embedding_job_fanned_out_four_ways_matches_in_process", 4)
        .await;
}

/// The plan's `InferenceExec` partition count.
fn inference_partition_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    match plan.downcast_ref::<InferenceExec>() {
        Some(exec) => exec.properties().output_partitioning().partition_count(),
        None => plan
            .children()
            .into_iter()
            .map(inference_partition_count)
            .max()
            .unwrap_or(0),
    }
}

/// Build the embedding plan at a fan-out of `partitions`, collect it
/// in-process and through the fleet, and hold the two to the same bytes in
/// the same order.
async fn embedding_job_matches_in_process(test: &str, partitions: usize) {
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(test);
    // One row per forward chunk, so the source's eight rows are eight chunks
    // for the exchange to spread over the fan-out.
    let inference = jammi_db::config::InferenceConfig {
        batch_size: 1,
        partitions,
        ..Default::default()
    };
    let (session, dir) = harness::harness_session_with(&backends, &result_root, inference).await;

    // Force per-file partition splitting: DataFusion's default file-group
    // builder coalesces files below `repartition_file_min_size` (10 MiB)
    // into ONE partition regardless of `target_partitions` (confirmed by
    // executing this exact plan without this override:
    // `leaf_partition_count` read 1, not 2, even at 300 rows/file) —
    // dropping the threshold makes the two-file source scan with one
    // partition per file, the shape that makes this test non-vacuous.
    session
        .sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .expect("lower repartition_file_min_size so the two-file source splits per file");

    let source_name = harness::unique_source_name("two_files");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = harness::tiny_bert_model();
    let plan = build_embedding_plan(
        &session,
        &source_name,
        ModelSource::parse(&model),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        32,
    )
    .await
    .expect("build_embedding_plan");

    let scan_partitions = leaf_partition_count(&plan);
    assert!(
        scan_partitions >= 2,
        "the two-file source's scan stage must have >= 2 partitions for this test to be \
         non-vacuous, got {scan_partitions}"
    );
    assert_eq!(
        inference_partition_count(&plan),
        partitions,
        "the inference stage must run as {partitions} tasks"
    );

    // In-process, on the harness session — the parity baseline.
    let task_ctx = session.context().task_ctx();
    let in_process = datafusion::physical_plan::collect(plan.clone(), task_ctx)
        .await
        .expect("in-process collect");
    let in_process = concat_in_arrival_order(&in_process);

    // Across the fleet, through the scheduler.
    let (specs, scheduler_port) = standard_fleet_specs();
    let fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let ids = await_fleet_registered(&session, &fleet).await;
    let (lane2_id, lane3_id) = (ids[1].clone(), ids[2].clone());
    let scheduler_url = format!("http://127.0.0.1:{scheduler_port}");

    let stream = tokio::time::timeout(
        Duration::from_secs(60),
        submit_physical_plan(&session, &scheduler_url, plan.clone()),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics("submit_physical_plan timed out");
        panic!("submit_physical_plan timed out");
    })
    .unwrap_or_else(|e| {
        fleet.dump_diagnostics(&format!("submit_physical_plan failed: {e}"));
        panic!("submit_physical_plan failed: {e}");
    });
    let placed = tokio::time::timeout(
        Duration::from_secs(60),
        datafusion::physical_plan::common::collect(stream),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics("collecting the placed stream timed out");
        panic!("collecting the placed stream timed out");
    })
    .unwrap_or_else(|e| {
        fleet.dump_diagnostics(&format!("collecting the placed stream failed: {e}"));
        panic!("collecting the placed stream failed: {e}");
    });
    let placed = concat_in_arrival_order(&placed);

    assert_eq!(
        ipc_bytes(&in_process),
        ipc_bytes(&placed),
        "the plan collected through Ballista must be Arrow-IPC-byte-identical, row order \
         included, to the same plan collected in-process"
    );

    // Both non-submitter executors actually ran a task of this job — read
    // from the SCHEDULER process's own log (the module doc explains why not
    // through get_job_status).
    let scheduler_log = fleet.log_contents(fleet.label(0));
    assert!(
        scheduler_log.contains(&lane2_id),
        "the scheduler's own log must show a task bound to lane-2 ({lane2_id}); log:\n{scheduler_log}"
    );
    assert!(
        scheduler_log.contains(&lane3_id),
        "the scheduler's own log must show a task bound to lane-3 ({lane3_id}); log:\n{scheduler_log}"
    );

    drop(fleet);
}

// ─── a placed gang, and its parity with an unplaced gang ───────────────────

/// A placed task's completion, reported to the scheduler at the host it
/// ADVERTISES, frees the executor's task slot: with ONE non-submitter
/// executor offering ONE slot, the catalog's `available_slots` for it
/// returns to its capacity after each placed job, and a second job is
/// placed on it. The scheduler binds every interface (`0.0.0.0`, the
/// deployment shape) and advertises loopback, so the report travels to
/// the advertised name; the name itself is `roles.rs`'s own test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_placed_task_reports_completion_to_the_advertised_scheduler_and_frees_its_slot() {
    const TEST: &str =
        "a_placed_task_reports_completion_to_the_advertised_scheduler_and_frees_its_slot";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::add_training_source(&session, &source).await;

    let scheduler_port = jammi_test_utils::free_port();
    let specs = vec![
        ProcSpec::fresh(
            BallistaRole::SchedulerAndExecutor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["fine_tune"]),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        ),
    ];
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let executor_id = instance_id_of_label(&session, fleet.label(1)).await;

    for round in 1..=2 {
        let (job_id, expected_model) =
            harness::submit_fine_tune(&session, &source, JobSize::Quick, 1).await;
        let record = harness::await_job(
            &mut fleet,
            &session,
            &job_id,
            "the job is placed on the one non-submitter executor and completes",
            |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
        )
        .await;
        assert_eq!(
            record.claimed_by.as_deref(),
            Some(executor_id.as_str()),
            "round {round}: the job must run as a placed task on the executor whose only slot \
             the previous round's task held"
        );
        assert_eq!(
            record.output_model_id.as_deref(),
            Some(expected_model.as_str()),
            "round {round}"
        );
        let freed = harness::await_condition(Duration::from_secs(30), || {
            futures::executor::block_on(async {
                session
                    .catalog()
                    .list_compute_executors()
                    .await
                    .unwrap()
                    .iter()
                    .any(|e| e.instance_id == executor_id && e.available_slots == e.task_slots)
            })
        })
        .await;
        assert!(
            freed,
            "round {round}: the executor's one slot is free again once the task's completion \
             reached the scheduler"
        );
    }
    drop(fleet);
}

/// Standard fleet + a submitted `world_size = 2` gang fine-tune, polled to
/// `running`. Returns `(fleet, job_id, expected_model, claimant_instance_id)`.
async fn submit_and_await_placed_claim(
    backends: &DistributedBackends,
    result_root: &str,
    session: &Arc<InferenceSession>,
    source: &str,
    size: JobSize,
) -> (Fleet, String, String, String) {
    let (specs, _) = standard_fleet_specs();
    let mut fleet = harness::spawn_fleet(backends, result_root, specs);
    await_fleet_registered(session, &fleet).await;

    let (job_id, expected_model) = harness::submit_fine_tune(session, source, size, 2).await;
    // The PLACED claim, never the first one: the scheduler-role process
    // (lane-1) claims the row itself and holds it `running` under its own
    // id until the executor's `run_placed_attempt` transfers it. A wait that
    // returned on any claimant would catch that pre-transfer state on a
    // slower runner (claimant == lane-1), so the predicate is the transfer
    // itself; a placement that never transfers
    // ends here as a timeout with the fleet's diagnostics.
    let submitter_id = instance_id_of_label(session, fleet.label(0)).await;
    let record = harness::await_job(
        &mut fleet,
        session,
        &job_id,
        "the job is claimed, running, and transferred to a placed executor",
        |r| {
            r.status == jammi_db::catalog::status::JobStatus::Running.to_string()
                && r.claimed_by.as_deref().is_some_and(|c| c != submitter_id)
        },
    )
    .await;
    let claimant = record.claimed_by.expect("running job has a claimant");
    (fleet, job_id, expected_model, claimant)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placed_gang_completes_on_a_registered_executor_other_than_the_submitter() {
    const TEST: &str = "placed_gang_completes_on_a_registered_executor_other_than_the_submitter";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::add_training_source(&session, &source).await;

    let (mut fleet, job_id, expected_model, claimant) =
        submit_and_await_placed_claim(&backends, &result_root, &session, &source, JobSize::Quick)
            .await;

    let lane1_id = instance_id_of_label(&session, fleet.label(0)).await;
    let lane2_id = instance_id_of_label(&session, fleet.label(1)).await;
    let lane3_id = instance_id_of_label(&session, fleet.label(2)).await;
    assert_ne!(
        claimant, lane1_id,
        "the placed gang must never be bound back to the submitter's own executor"
    );
    assert!(
        claimant == lane2_id || claimant == lane3_id,
        "claimant {claimant} must be one of the two non-submitter executors"
    );

    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        "the placed gang completes",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    assert_eq!(record.claimed_by.as_deref(), Some(claimant.as_str()));
    assert_eq!(
        record.attempts, 1,
        "exactly one transfer, zero net attempts"
    );
    assert_eq!(record.releases, 0);
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(expected_model.as_str())
    );

    // HandedOff evidence on the submitter's (lane-1's) own log. The line
    // lands after the gang's result stream drains, which is after the
    // coordinator's `completed` row the wait above returned on -- so the
    // read polls (harness::await_log_contains) instead of asserting one
    // snapshot.
    let lane1_label = fleet.label(0).to_string();
    harness::await_log_contains(
        &mut fleet,
        &lane1_label,
        "run_placed_attempt: submitter HandedOff",
        "the HandedOff arm's `tracing::info!` line",
    )
    .await;

    // The placed run's artifact bytes equal a SECOND fleet's unplaced run
    // (no `[ballista]` at all — process 1 coordinates locally), same base
    // model / config / device kind (CPU).
    let plain_result_root = backends.unique_result_root(&format!("{TEST}-plain"));
    let plain_source = harness::unique_source_name(&format!("{TEST}-plain"));
    harness::add_training_source(&session, &plain_source).await;
    let plain_specs = vec![
        ProcSpec::fresh(
            BallistaRole::None,
            WorkerRole {
                enabled: true,
                kinds: Some(&["fine_tune"]),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::None,
            WorkerRole {
                enabled: true,
                kinds: Some(&["fine_tune"]),
                idle_poll_secs: 1,
            },
        ),
    ];
    let mut plain_fleet = harness::spawn_fleet(&backends, &plain_result_root, plain_specs);
    let (plain_job_id, plain_model) =
        harness::submit_fine_tune(&session, &plain_source, JobSize::Quick, 2).await;
    let plain_record = harness::await_job(
        &mut plain_fleet,
        &session,
        &plain_job_id,
        "the plain (non-Ballista) gang completes",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    assert_eq!(
        plain_record.output_model_id.as_deref(),
        Some(plain_model.as_str())
    );

    let models = session.catalog().list_models().await.unwrap();
    let placed_path = models
        .iter()
        .find(|m| m.model_id == expected_model)
        .and_then(|m| m.location.as_ref())
        .expect("placed model references its artifact")
        .bundle_url()
        .unwrap();
    let plain_path = models
        .iter()
        .find(|m| m.model_id == plain_model)
        .and_then(|m| m.location.as_ref())
        .expect("plain model references its artifact")
        .bundle_url()
        .unwrap();

    let placed_local = session
        .artifact_store()
        .fetch_artifact(&placed_path)
        .await
        .unwrap();
    let plain_local = session
        .artifact_store()
        .fetch_artifact(&plain_path)
        .await
        .unwrap();
    let placed_digest =
        jammi_ai::fine_tune::worker::published_artifact_digest(&placed_local).unwrap();
    let plain_digest =
        jammi_ai::fine_tune::worker::published_artifact_digest(&plain_local).unwrap();
    assert_eq!(
        placed_digest, plain_digest,
        "the placed run's adapter artifact must be byte-identical to the unplaced run's \
         (per device kind — both CPU)"
    );

    drop(fleet);
    drop(plain_fleet);
}

// ─── a placed context predictor ────────────────────────────────────────────

/// Placement is a property of the training attempt, not of the kinds that
/// have a rank count: a context-predictor job in a fleet whose executors
/// claim nothing is claimed by the scheduler process — which hosts no
/// executor, so it never trains while a live executor can hold the attempt —
/// placed, trained on an executor that reads the episodes' source and
/// embedding table through the shared catalog and result root, and published
/// byte-identical to the same job trained in its claimant's own process.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_context_predictor_job_completes_on_an_executor_that_claims_nothing() {
    const TEST: &str = "a_context_predictor_job_completes_on_an_executor_that_claims_nothing";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::add_episodes_source(&session, dir.path(), &source).await;

    // The reference, before any fleet exists to claim it: the same job
    // trained by its claimant — this session holds no plane.
    let in_process_model = format!("predictor-in-process-{}", jammi_test_utils::unique_suffix());
    let in_process_job =
        harness::submit_context_predictor(&session, &source, &in_process_model).await;
    let worker = jammi_ai::fine_tune::worker::JobWorker::new(&session).unwrap();
    let claimed = session
        .catalog()
        .claim_next(
            worker.worker_id(),
            &["context_predictor"],
            Duration::from_secs(60),
        )
        .await
        .unwrap()
        .expect("the queued reference job is claimable");
    assert_eq!(claimed.job_id, in_process_job);
    worker.run_claimed_job(&session, claimed).await;
    let reference = session.catalog().get_job(&in_process_job).await.unwrap();
    assert_eq!(
        reference.status,
        jammi_db::catalog::status::JobStatus::Completed.to_string(),
        "{reference:?}"
    );

    let scheduler_port = jammi_test_utils::free_port();
    let claims_nothing = WorkerRole {
        enabled: true,
        kinds: Some(&[]),
        idle_poll_secs: 1,
    };
    let specs = vec![
        ProcSpec::fresh(
            BallistaRole::SchedulerAndClient { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(BallistaRole::Executor { scheduler_port }, claims_nothing),
        ProcSpec::fresh(BallistaRole::Executor { scheduler_port }, claims_nothing),
    ];
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let executors = await_fleet_registered(&session, &fleet).await;

    let placed_model = format!("predictor-placed-{}", jammi_test_utils::unique_suffix());
    let job_id = harness::submit_context_predictor(&session, &source, &placed_model).await;
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        "the placed context predictor completes",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    let trained_by = record
        .claimed_by
        .clone()
        .expect("a completed job names its holder");
    assert!(
        executors.contains(&trained_by),
        "the job must train on an executor ({executors:?}), not on its claimant: {record:?}"
    );
    assert_eq!(
        record.attempts, 1,
        "exactly one transfer, zero net attempts"
    );
    assert_eq!(record.releases, 0);
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(placed_model.as_str())
    );

    let claimant = fleet.label(0).to_string();
    harness::await_log_contains(
        &mut fleet,
        &claimant,
        "run_placed_attempt: submitter HandedOff",
        "the claimant handed the attempt off",
    )
    .await;

    let mut digests = Vec::new();
    for model_id in [&placed_model, &in_process_model] {
        let model = session
            .catalog()
            .get_model(model_id)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("model {model_id} is registered"));
        let bundle = model
            .location
            .as_ref()
            .expect("a trained predictor references its artifact")
            .bundle_url()
            .unwrap();
        let local = session
            .artifact_store()
            .fetch_artifact(&bundle)
            .await
            .unwrap();
        digests.push(jammi_ai::fine_tune::worker::published_artifact_digest(&local).unwrap());
    }
    assert_eq!(
        digests[0], digests[1],
        "the placed predictor's artifact must be byte-identical to the in-process run's"
    );

    drop(fleet);
}

// ─── a killed placed executor, then reclaim ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_executor_mid_gang_leaves_the_row_for_reclaim_then_a_successor_completes() {
    const TEST: &str =
        "killed_executor_mid_gang_leaves_the_row_for_reclaim_then_a_successor_completes";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::add_training_source(&session, &source).await;

    // The SAME 3-process fleet the placed-gang test uses (lane-0 is the ONLY
    // `fine_tune`-kind claimant/submitter, deterministically placed onto
    // lane-1 or lane-2). A SECOND, independent `fine_tune`-kind bidder is
    // spawned LATE — only AFTER the first claim+placement has already
    // resolved — so it plays no part in that race and stays free the whole
    // time. This two-step spawn is load-bearing: a bidder present from t=0
    // races lane-0 for the INITIAL claim and, when it wins, runs the whole
    // gang in-process (no client role of its own installs a compute
    // plane — `roles::host_client`'s doc), never
    // exercising the placed path this test needs; a SECOND SCHEDULER able to place
    // independently (a scheduler hosting no executor) fails a different way —
    // Ballista's OWN task binder gates on ITS OWN executor-heartbeat CACHE
    // (`ballista-scheduler-54.1.0/src/state/executor_manager.rs:117-121`'s
    // `get_alive_executors`), which a second scheduler never populates for
    // executors that registered with a DIFFERENT scheduler (`CatalogCluster
    // State`'s own documented heartbeat-cache-staleness caveat) — observed
    // as "There are no alive executors to bind tasks" forever. The reclaimer
    // that actually works has NO ballista role of its own: on reclaiming it
    // runs the recovered gang in-process (the unplaced `Peer` path, dialing
    // lane-0 as rank 1 — `Holder::Awaiting` admits a rank exactly like
    // `Free`, so lane-0's OWN still-blocked placed attempt never conflicts).
    let (mut fleet, job_id, expected_model, claimant) = submit_and_await_placed_claim(
        &backends,
        &result_root,
        &session,
        &source,
        JobSize::Crashable,
    )
    .await;
    let lane0_id = instance_id_of_label(&session, fleet.label(0)).await;
    assert_ne!(
        claimant, lane0_id,
        "the placed gang runs on a non-submitter executor"
    );

    let reclaimer_idx = fleet.spawn_more(
        &backends,
        &result_root,
        ProcSpec::fresh(
            BallistaRole::None,
            WorkerRole {
                enabled: true,
                kinds: Some(&["fine_tune"]),
                idle_poll_secs: 1,
            },
        ),
    );
    let ok = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            let workers = session.catalog().list_workers().await.unwrap_or_default();
            workers
                .iter()
                .any(|w| w.label.as_deref() == Some(fleet.label(reclaimer_idx)))
        })
    })
    .await;
    assert!(
        ok,
        "timed out waiting for the late-joining reclaimer's workers row"
    );

    let killed_label = harness::label_of(&session, &claimant).await;

    // Record the distinct `claimed_by` transitions we observe, polling every
    // 300ms — a subset of the true transitions: a faster flip between two
    // polls is never observed.
    let mut observed: Vec<String> = vec![lane0_id.clone(), claimant.clone()];
    let mut killed = false;
    // The successor re-runs the whole job from scratch, so this wait is
    // the job's own length plus the reclaim: twice the single-job bound.
    let reclaim_timeout = harness::TERMINAL_TIMEOUT * 2;
    let deadline = std::time::Instant::now() + reclaim_timeout;
    let final_record = loop {
        if !killed {
            // Give the gang a moment to actually be mid-run before crashing it.
            tokio::time::sleep(Duration::from_secs(1)).await;
            assert!(
                fleet.kill9(&killed_label),
                "the claimant's label {killed_label:?} must be one of the spawned workers"
            );
            killed = true;
        }
        let record = session
            .catalog()
            .pinned_to_tenant(None)
            .get_job(&job_id)
            .await
            .unwrap();
        if let Some(cb) = &record.claimed_by {
            if observed.last() != Some(cb) {
                observed.push(cb.clone());
            }
        }
        if record.status == jammi_db::catalog::status::JobStatus::Completed.to_string() {
            break record;
        }
        // A terminal failure is the answer, not something to poll past: the
        // successor's own error names what it hit.
        if record.status == jammi_db::catalog::status::JobStatus::Failed.to_string() {
            fleet.dump_diagnostics("the job failed instead of completing on a successor");
            panic!(
                "the job ended failed on attempt {} ({:?}) instead of completing on a \
                 successor; observed claimed_by sequence: {observed:?}",
                record.attempts, record.error
            );
        }
        if std::time::Instant::now() >= deadline {
            fleet.dump_diagnostics("timed out awaiting reclaim + successor completion");
            panic!(
                "timed out after {:?} awaiting reclaim + successor completion; observed \
                 claimed_by sequence: {observed:?}",
                reclaim_timeout
            );
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    };

    assert!(
        observed.iter().filter(|c| **c == claimant).count() == 1,
        "the killed executor {claimant} must never reappear as claimant after being killed; \
         observed sequence: {observed:?}"
    );
    let successor = final_record
        .claimed_by
        .clone()
        .expect("completed job has a claimant");
    assert_ne!(
        successor, claimant,
        "a survivor, never the killed executor, completed the job; observed: {observed:?}"
    );
    // The successor is either the late-joining reclaimer (running the
    // recovered gang in-process, the unplaced `Peer` path — it installs no
    // compute plane of its own) or lane-0 itself once its own
    // placed attempt's stream finally errors (Ballista's own heartbeat
    // timeout) and its NEXT poll re-claims the still-expired row — either
    // way, never the executor this test just killed.
    let reclaimer_id = instance_id_of_label(&session, fleet.label(reclaimer_idx)).await;
    assert!(
        successor == reclaimer_id || successor == lane0_id,
        "the successor {successor} must be the late-joining reclaimer ({reclaimer_id}) or \
         lane-0 ({lane0_id}), never a re-appearance of the killed claimant; \
         observed: {observed:?}"
    );
    assert!(
        final_record.attempts >= 2,
        "the crashed attempt is spent and a successor's claim bumps attempts: {}",
        final_record.attempts
    );
    assert_eq!(
        final_record.output_model_id.as_deref(),
        Some(expected_model.as_str())
    );
    let models = session.catalog().list_models().await.unwrap();
    assert_eq!(
        models
            .iter()
            .filter(|m| m.model_id == expected_model)
            .count(),
        1,
        "exactly one model row after the crash and the successor's completion"
    );

    eprintln!(
        "killed_executor_mid_gang: observed claimed_by sequence (poll interval 300ms): \
         {observed:?}"
    );

    drop(fleet);
}

// ─── a scheduler restart ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scheduler_restart_keeps_executors_and_serves_a_new_job() {
    const TEST: &str = "scheduler_restart_keeps_executors_and_serves_a_new_job";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, scheduler_port) = standard_fleet_specs();
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let ids = await_fleet_registered(&session, &fleet).await;
    let (lane2_id, lane3_id) = (ids[1].clone(), ids[2].clone());
    let lane1_label = fleet.label(0).to_string();

    // SIGKILL and respawn the scheduler process (lane-1) at the SAME
    // `scheduler.bind` port. `instance_id` (`InferenceSession::instance_id`)
    // is minted at session construction, never externally supplied, so the
    // replacement is a fresh instance — the assertions below need only the
    // OTHER executors' registrations and a NEW job's completion, which the
    // shared catalog carries regardless of the replacement's id.
    fleet.kill9(&lane1_label);
    fleet.respawn(&backends, &result_root, &lane1_label);

    // Read back the two untouched executors' registrations, through the
    // harness session's OWN catalog — the same shared store the
    // replacement scheduler itself reads, rather than a
    // `SchedulerGrpcClient` call.
    let ok = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            session
                .catalog()
                .list_compute_executors()
                .await
                .map(|v| {
                    let ids: Vec<&str> = v.iter().map(|r| r.executor_id.as_str()).collect();
                    ids.contains(&lane2_id.as_str()) && ids.contains(&lane3_id.as_str())
                })
                .unwrap_or(false)
        })
    })
    .await;
    assert!(
        ok,
        "the two untouched executors must still be registered after the scheduler restart"
    );
    // The replacement's OWN gRPC listener needs a moment to bind — `respawn`
    // only waits out a port-reuse race, never the new process's own startup
    // (Postgres connect + migrate check + tonic bind). A raw TCP connect
    // probe is the readiness check (a `submit_physical_plan` before the
    // listener binds fails with `ConnectionRefused`).
    let ready = harness::await_condition(Duration::from_secs(30), || {
        std::net::TcpStream::connect(format!("127.0.0.1:{scheduler_port}")).is_ok()
    })
    .await;
    assert!(
        ready,
        "the replacement scheduler's gRPC listener never came up"
    );

    // A NEW job places and completes through the replacement scheduler.
    let source_name = harness::unique_source_name("two_files_restart");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let plan = build_embedding_plan(
        &session,
        &source_name,
        ModelSource::parse(&harness::tiny_bert_model()),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        32,
    )
    .await
    .unwrap();
    let scheduler_url = format!("http://127.0.0.1:{scheduler_port}");
    // A raw TCP connect succeeding is not sufficient readiness — the
    // replacement's own tonic server can accept the TCP handshake into its
    // backlog moments before its gRPC service is actually registered and
    // serving (a bare readiness probe can still be followed by one
    // `ConnectionRefused` from `submit_physical_plan` itself). Retry the
    // real submission a few times with a short backoff, a bounded
    // "still starting up" tolerance — never silently swallowing a REAL
    // failure past this bounded window.
    let submit_deadline = std::time::Instant::now() + Duration::from_secs(60);
    let stream = loop {
        let attempt = tokio::time::timeout(
            Duration::from_secs(10),
            submit_physical_plan(&session, &scheduler_url, plan.clone()),
        )
        .await;
        match attempt {
            Ok(Ok(stream)) => break stream,
            Ok(Err(_)) if std::time::Instant::now() < submit_deadline => {
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            Ok(Err(e)) => {
                fleet.dump_diagnostics("submit_physical_plan (post-restart) kept failing");
                panic!("submit_physical_plan through the replacement scheduler failed: {e}");
            }
            // The inner 10s timeout elapsing is ALSO a retryable "still
            // reconnecting" symptom, never a fast typed error: Ballista's
            // executors hold a long-lived gRPC connection to the OLD
            // scheduler process and must independently notice it dropped
            // and reconnect to the replacement before a task can bind (the
            // same executor-heartbeat-cache mechanism the two-scheduler
            // test's comment names, `ballista-scheduler-54.1.0/src/state/executor_manager.
            // rs:117-121`) — bounded by the SAME outer `submit_deadline`,
            // never an unbounded retry.
            Err(_) if std::time::Instant::now() < submit_deadline => continue,
            Err(_) => {
                fleet.dump_diagnostics("submit_physical_plan (post-restart) timed out");
                panic!(
                    "submit_physical_plan through the replacement scheduler never succeeded \
                     within the 60s retry window"
                );
            }
        }
    };
    let batches = tokio::time::timeout(
        Duration::from_secs(60),
        datafusion::physical_plan::common::collect(stream),
    )
    .await
    .expect("collect did not time out")
    .expect("collect succeeds through the replacement scheduler");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        rows > 0,
        "the new job must actually produce rows through the replacement scheduler"
    );

    drop(fleet);
}

// ─── two schedulers over one catalog ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_schedulers_over_one_catalog_serve_jobs_sequentially() {
    const TEST: &str = "two_schedulers_over_one_catalog_serve_jobs_sequentially";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    // The training source is registered on the SHARED catalog before any
    // fleet process starts. A claiming process resolves a fine-tune job's
    // named source from the catalog row at the moment it needs it, so the
    // order is a convenience, not a requirement (a plan submitted whole via
    // `submit_physical_plan` carries concrete file paths in its scan and
    // never looks a source name up on the executor at all).
    let source = harness::unique_source_name(TEST);
    harness::add_training_source(&session, &source).await;

    // Scheduler 4 hosts its OWN local executor too (`SchedulerAndExecutor`).
    // A scheduler 4 hosting no executor of its own fails every submission
    // with Ballista's OWN "There are no
    // alive executors to bind tasks" (`ballista-scheduler-54.1.0/src/state/
    // executor_manager.rs:117-121`'s `get_alive_executors`, gated on THIS
    // scheduler's own executor-HEARTBEAT cache — never the raw
    // `compute_executors` row set `list_compute_executors` reads) —
    // executors 2/3 heartbeat ONLY to the scheduler they registered with
    // (scheduler 1), so scheduler 4 never learns they are alive, no
    // matter how long a plan submitted to it waits (this is the concrete
    // shape of `CatalogClusterState`'s own documented heartbeat-cache-
    // staleness caveat: a standby scheduler's liveness view of an executor
    // it does not itself serve is only as fresh as its last init).
    // Scheduler 4 therefore serves its OWN job on its OWN local executor —
    // two schedulers over one shared catalog, each independently able to
    // serve a job, never a claim that Ballista binds a task ACROSS two live
    // schedulers.
    let (mut specs, scheduler1_port) = standard_fleet_specs();
    let scheduler4_port = jammi_test_utils::free_port();
    specs.push(ProcSpec::fresh(
        BallistaRole::SchedulerAndExecutor {
            scheduler_port: scheduler4_port,
        },
        WorkerRole {
            enabled: true,
            kinds: Some(&["context_predictor"]),
            idle_poll_secs: 30,
        },
    ));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    // Scheduler 4's own executor registers with the rest: every member of
    // this fleet hosts one.
    await_fleet_registered(&session, &fleet).await;

    // A job through scheduler 1 completes first.
    let (job_id, model_id) = harness::submit_fine_tune(&session, &source, JobSize::Quick, 2).await;
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        "job via scheduler 1 completes",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    assert_eq!(record.output_model_id.as_deref(), Some(model_id.as_str()));

    // A plan submitted directly through scheduler 4 runs to completion on
    // scheduler 4's OWN local executor.
    let source_name = harness::unique_source_name("two_files_sched4");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let plan = build_embedding_plan(
        &session,
        &source_name,
        ModelSource::parse(&harness::tiny_bert_model()),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        32,
    )
    .await
    .unwrap();
    let scheduler4_url = format!("http://127.0.0.1:{scheduler4_port}");
    let stream = tokio::time::timeout(
        Duration::from_secs(60),
        submit_physical_plan(&session, &scheduler4_url, plan),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics("submit_physical_plan via scheduler 4 timed out");
        panic!("submit_physical_plan via scheduler 4 timed out");
    })
    .expect("submit_physical_plan via the second scheduler succeeds");
    let batches = tokio::time::timeout(
        Duration::from_secs(60),
        datafusion::physical_plan::common::collect(stream),
    )
    .await
    .expect("collect did not time out")
    .expect("collect via the second scheduler succeeds");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        rows > 0,
        "the plan submitted through scheduler 4 must produce rows"
    );
    assert_ne!(
        scheduler4_port, scheduler1_port,
        "the two schedulers listen on distinct ports"
    );

    drop(fleet);
}

// ─── device-kind refusal ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn device_less_cluster_refuses_gpu_bound_plan_and_accepts_cpu_plan() {
    const TEST: &str = "device_less_cluster_refuses_gpu_bound_plan_and_accepts_cpu_plan";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, scheduler_port) = standard_fleet_specs();
    let fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let scheduler_url = format!("http://127.0.0.1:{scheduler_port}");

    // KIND MATCH: a PlacedAttemptExec's required
    // kind is its OWN descriptor's stamp, never "is this node type
    // GPU-shaped" — a dummy descriptor (never actually run) stamped `Cuda`
    // is refused on this all-CPU cluster (no registered executor lists a
    // `cuda` device); a `Cpu`-stamped one would be accepted (exercised
    // below by the real embedding plan, whose `InferenceExec` is stamped
    // from the harness session's own CPU device).
    let attempt_plan: Arc<dyn ExecutionPlan> = Arc::new(PlacedAttemptExec::new(PlacedAttempt {
        job_id: "dummy-job".to_string(),
        attempt: 0,
        submitter: "dummy-submitter".to_string(),
        device_kind: jammi_db::store::manifest::ComputeDeviceKind::Cuda,
        claimed_at: chrono::Utc::now(),
    }));
    // The device check is a fast, purely client-side catalog read before
    // any RPC: a 20s timeout is generous headroom, never load-bearing for a
    // passing run, but turns an unexpected fall-through to a real submission
    // (e.g. a plan the codec's `try_encode_udf`/`try_decode_udf` cannot
    // carry) into a fast, diagnosable failure instead of the test hanging.
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        submit_physical_plan(&session, &scheduler_url, attempt_plan),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics(
            "the device-less refusal did not return within 20s — it should never reach the \
             network at all",
        );
        panic!(
            "a PlacedAttemptExec plan's device-less refusal must return fast (client-side, before any \
             RPC); it did not return within 20s"
        );
    });
    let refusal = match result {
        Ok(_) => {
            panic!("a PlacedAttemptExec plan must be refused on a device-less cluster, but it was accepted")
        }
        Err(e) => JammiError::from(e),
    };
    // The typed refusal: the plan's own kind, and every kind the live
    // executors list — this fleet's registrations all list `cpu` (the CPU
    // plan below binds to one of them), and nothing lists `cuda`.
    match refusal {
        JammiError::Unheld(jammi_db::compute_plane::Unheld::NoExecutorOfKind {
            required,
            held,
        }) => {
            assert_eq!(required, jammi_db::store::manifest::ComputeDeviceKind::Cuda);
            assert_eq!(
                held,
                vec![jammi_db::store::manifest::ComputeDeviceKind::Cpu]
            );
        }
        other => panic!("expected Unheld(NoExecutorOfKind), got {other:?}"),
    }

    // A CPU InferenceExec plan (the harness session's own device kind) is
    // accepted and actually runs.
    let source_name = harness::unique_source_name("two_files_device_kind");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let cpu_plan = build_embedding_plan(
        &session,
        &source_name,
        ModelSource::parse(&harness::tiny_bert_model()),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        32,
    )
    .await
    .unwrap();
    let stream = submit_physical_plan(&session, &scheduler_url, cpu_plan)
        .await
        .expect("a CPU-device-kind plan must be accepted on a CPU-only cluster");
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .expect("the accepted CPU plan collects");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(rows > 0, "the accepted CPU plan must actually produce rows");

    drop(fleet);
}

// ─── a typed failure survives a placed task ──────────────────────────────

/// The typed error a placed plan's failure classifies to: it surfaces from
/// the submission itself (Ballista awaits the job's terminal status before
/// handing back the stream) or from the stream.
async fn classified_placed_failure(
    submitted: jammi_ballista::error::Result<datafusion::execution::SendableRecordBatchStream>,
) -> JammiError {
    match submitted {
        Err(e) => JammiError::from(e),
        Ok(stream) => match datafusion::physical_plan::common::collect(stream).await {
            Ok(batches) => panic!(
                "the placed plan over a null key must refuse, but yielded {} batch(es)",
                batches.len()
            ),
            Err(e) => JammiError::from(e),
        },
    }
}

/// A placed inference plan that refuses a NULL key — `KeyCheckExec`'s
/// `InvalidKey`, raised inside a task on an executor — reaches the
/// submitter classified IDENTICALLY to the same plan collected
/// in-process: the same variant, the same column and null count, at a
/// fan-out of two.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placed_inference_refusing_a_null_key_classifies_as_the_in_process_one() {
    const TEST: &str = "placed_inference_refusing_a_null_key_classifies_as_the_in_process_one";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let inference = jammi_db::config::InferenceConfig {
        batch_size: 1,
        partitions: 2,
        ..Default::default()
    };
    let (session, dir) = harness::harness_session_with(&backends, &result_root, inference).await;

    let source_name = harness::unique_source_name("null_key");
    let url = jammi_test_utils::write_null_key_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let plan = build_embedding_plan(
        &session,
        &source_name,
        ModelSource::parse(&harness::tiny_bert_model()),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        32,
    )
    .await
    .expect("build_embedding_plan");
    assert_eq!(inference_partition_count(&plan), 2);

    // In-process, on the harness session — the parity baseline.
    let in_process = match datafusion::physical_plan::collect(
        plan.clone(),
        session.context().task_ctx(),
    )
    .await
    {
        Ok(batches) => panic!(
            "the in-process plan over a null key must refuse, but yielded {} batch(es)",
            batches.len()
        ),
        Err(e) => JammiError::from(e),
    };
    assert!(
        matches!(
            &in_process,
            JammiError::InvalidKey { column, null_count: 1 } if column == "id"
        ),
        "in-process: expected InvalidKey {{ id, 1 }}, got {in_process:?}"
    );

    // Across the fleet, through the scheduler.
    let (specs, scheduler_port) = standard_fleet_specs();
    let fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let scheduler_url = format!("http://127.0.0.1:{scheduler_port}");
    let submitted = tokio::time::timeout(
        Duration::from_secs(60),
        submit_physical_plan(&session, &scheduler_url, plan),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics("submit_physical_plan timed out");
        panic!("submit_physical_plan timed out");
    });
    let placed = tokio::time::timeout(
        Duration::from_secs(60),
        classified_placed_failure(submitted),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics("collecting the placed stream timed out");
        panic!("collecting the placed stream timed out");
    });
    assert_eq!(
        format!("{placed:?}"),
        format!("{in_process:?}"),
        "the placed refusal must classify as the in-process one, variant and fields"
    );

    drop(fleet);
}

/// A placed gang whose training source is removed between the job's
/// submission and its placement fails its attempt with `SourceNotFound`
/// naming the source — on the job row (the executor's own terminal
/// write, the same message the in-process path records) and to the
/// submitter (the task's typed error, named on the submitter's log).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placed_gang_over_a_removed_source_fails_typed_on_the_row_and_to_the_submitter() {
    const TEST: &str =
        "placed_gang_over_a_removed_source_fails_typed_on_the_row_and_to_the_submitter";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::add_training_source(&session, &source).await;

    // Submitted, then the source removed, BEFORE any fleet member exists
    // to claim it: the placement that follows resolves the source on the
    // executor and finds no row.
    let (job_id, _) = harness::submit_fine_tune(&session, &source, JobSize::Quick, 2).await;
    session
        .remove_source(&source)
        .await
        .expect("a queued job's source can be removed");

    let (specs, _) = standard_fleet_specs();
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let submitter_id = instance_id_of_label(&session, fleet.label(0)).await;

    let expected = JammiError::SourceNotFound {
        source_id: source.clone(),
    }
    .to_string();
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        "the placed attempt fails naming the missing source",
        |r| r.status == jammi_db::catalog::status::JobStatus::Failed.to_string(),
    )
    .await;
    assert_eq!(
        record.error.as_deref(),
        Some(expected.as_str()),
        "the job row carries the typed error's own message"
    );
    let claimant = record
        .claimed_by
        .expect("a failed attempt names its claimant");
    assert_ne!(
        claimant, submitter_id,
        "the attempt failed on the placed executor, never on the submitter"
    );

    // The submitter saw the same typed error as the task's own failure.
    let lane1_label = fleet.label(0).to_string();
    harness::await_log_contains(
        &mut fleet,
        &lane1_label,
        &expected,
        "the submitter's HandedOff arm naming the placed attempt's typed error",
    )
    .await;

    drop(fleet);
}

// ─── device inventory ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_workers_and_compute_executor_devices_report_registered_devices() {
    const TEST: &str = "list_workers_and_compute_executor_devices_report_registered_devices";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, _scheduler_port) = standard_fleet_specs();
    let fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let ids = await_fleet_registered(&session, &fleet).await;

    let cpu = jammi_db::catalog::instance::DeviceFact {
        kind: "cpu".to_string(),
        ordinal: 0,
    };

    let workers = session.catalog().list_workers().await.unwrap();
    for label in fleet.worker_labels() {
        let row = workers
            .iter()
            .find(|w| w.label.as_deref() == Some(label))
            .unwrap_or_else(|| panic!("no `workers` row for {label:?}"));
        assert_eq!(
            row.devices,
            vec![cpu.clone()],
            "list_workers must mirror {label:?}'s device inventory as [{{cpu, 0}}]"
        );
    }

    // Filtered to THIS fleet's own executor ids — the shared Postgres
    // catalog carries other test runs' rows too (see
    // `await_fleet_registered`'s doc), so a bare row count would be wrong.
    let executor_devices = session
        .catalog()
        .list_compute_executor_devices()
        .await
        .unwrap();
    for id in &ids {
        let devices = executor_devices
            .iter()
            .find(|(eid, _)| eid == id)
            .map(|(_, d)| d.clone())
            .unwrap_or_else(|| panic!("no compute_executors devices row for {id:?}"));
        assert_eq!(devices, vec![cpu.clone()]);
    }

    drop(fleet);
}

/// `jammi_test_utils::free_port`'s two stated properties, asserted: every port lies
/// below every platform's ephemeral floor (so no outgoing `connect()` of
/// this process can take it before the spawned server binds it), and no
/// port is handed out twice by one process. Needs no backend. Mutation:
/// pick from `bind(:0)` again and the range assertion reds.
#[test]
fn free_port_stays_below_the_ephemeral_floor_and_never_repeats() {
    let mut seen = std::collections::HashSet::new();
    for _ in 0..64 {
        let p = jammi_test_utils::free_port();
        assert!(
            (20_000..32_000).contains(&p),
            "port {p} outside the reserved range"
        );
        assert!(seen.insert(p), "port {p} handed out twice");
    }
}

// ─── a batch statement on a client-role query tier ─────────────────────────

/// The query-tier process of a statement fleet: the client role alone, no
/// worker (it claims nothing and hosts no executor), `services = []` (the
/// Flight SQL surface is mounted regardless).
fn client_spec(scheduler_port: u16) -> ProcSpec {
    ProcSpec::fresh(
        BallistaRole::Client { scheduler_port },
        WorkerRole {
            enabled: false,
            kinds: None,
            idle_poll_secs: 1,
        },
    )
}

/// A client-role process that claims `embedding` jobs and submits their
/// sink to the scheduler — the shape-d query tier with a worker.
fn embedding_client_spec(scheduler_port: u16) -> ProcSpec {
    ProcSpec::fresh(
        BallistaRole::Client { scheduler_port },
        WorkerRole {
            enabled: true,
            kinds: Some(&["embedding"]),
            idle_poll_secs: 1,
        },
    )
}

/// Wait until the query tier labelled `label` answers a statement.
async fn await_flight_up(fleet: &mut Fleet, label: &str) {
    let addr = fleet.flight_addr(label);
    let deadline = std::time::Instant::now() + harness::TERMINAL_TIMEOUT;
    loop {
        if flight_statement(addr, "SELECT 1").await.is_ok() {
            return;
        }
        if std::time::Instant::now() >= deadline {
            fleet.dump_diagnostics("the query tier never answered a statement");
            panic!("query tier {label} never answered a statement");
        }
        tokio::time::sleep(harness::POLL_INTERVAL).await;
    }
}

/// `batches` concatenated in arrival order; an empty result is an empty
/// batch of `schema`.
fn concat_or_empty(batches: &[RecordBatch], schema: arrow::datatypes::SchemaRef) -> RecordBatch {
    match batches.iter().find(|b| b.num_rows() > 0) {
        Some(_) => concat_in_arrival_order(batches),
        None => RecordBatch::new_empty(schema),
    }
}

/// The `ready` result table `name`'s row and the bytes of its Parquet on
/// the shared object store, read through the harness session's own store
/// — exactly what every fleet member wrote or reads.
async fn table_bytes(
    session: &InferenceSession,
    name: &str,
) -> (jammi_db::catalog::result_repo::ResultTableRecord, Vec<u8>) {
    let record = session
        .catalog()
        .get_result_table(name)
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("result table {name} has no row"));
    assert_eq!(
        record.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "{name} is ready"
    );
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let handle = session.result_store().open_parquet(&url).unwrap();
    let bytes = handle
        .get_bytes(&handle.data_path().unwrap())
        .await
        .unwrap()
        .to_vec();
    (record, bytes)
}

/// The fleet member whose captured log carries the sink's write line for
/// `table`, and the store writer id that line names — the process writing
/// the table and the identity its row is leased under — once one does.
/// Polled tightly, never at [`harness::POLL_INTERVAL`]: the line is the
/// earliest moment the row is the executor's, the killed-executor test
/// kills on it, and a small table's write is over in tens of milliseconds.
async fn await_sink_writer(fleet: &mut Fleet, labels: &[String], table: &str) -> (String, String) {
    let deadline = std::time::Instant::now() + harness::TERMINAL_TIMEOUT;
    loop {
        for label in labels {
            let log = fleet.log_contents(label);
            if let Some(writer_id) = log
                .lines()
                .filter(|line| line.contains(SINK_WRITE_LOG) && line.contains(table))
                .find_map(|line| {
                    line.split_whitespace()
                        .find_map(|token| token.strip_prefix("writer="))
                })
            {
                return (label.clone(), writer_id.to_string());
            }
        }
        if std::time::Instant::now() >= deadline {
            fleet.dump_diagnostics("no fleet member logged the sink's write");
            panic!("no member of {labels:?} logged the sink writing {table}");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A `CREATE TABLE … AS` over a result table, issued over Flight SQL to a
/// client-role query tier beside a scheduler and two other executors, is a
/// result table written on the compute plane: its `result_tables` row
/// exists, its Parquet on the shared object store is byte-identical to the
/// table the same statement produces in-process, an EXECUTOR's log names
/// the write and the query tier's does not, the scheduler bound a stage to
/// one of this fleet's executors, and a `SELECT` on a DIFFERENT query-tier
/// process reads it through the catalog.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_table_as_over_flight_sql_runs_on_the_compute_plane_and_matches_in_process() {
    const TEST: &str =
        "create_table_as_over_flight_sql_runs_on_the_compute_plane_and_matches_in_process";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let source_name = harness::unique_source_name("two_files");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    // The result table the statement reads: written to the shared result
    // root before the fleet comes up, so every member binds it at open.
    let (record, _) = session
        .generate_text_embeddings(
            &source_name,
            &harness::tiny_bert_model(),
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .expect("the embedding result table materializes in-process");
    let table = record.table_name.clone();

    let (mut specs, scheduler_port) = standard_fleet_specs();
    specs.push(client_spec(scheduler_port));
    specs.push(client_spec(scheduler_port));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let ids = await_fleet_registered(&session, &fleet).await;
    let executors: Vec<String> = (0..3).map(|i| fleet.label(i).to_string()).collect();
    let query_tier = fleet.label(3).to_string();
    let other_query_tier = fleet.label(4).to_string();
    await_flight_up(&mut fleet, &query_tier).await;
    await_flight_up(&mut fleet, &other_query_tier).await;
    let addr = fleet.flight_addr(&query_tier);

    // Two names on the shared catalog: the routed table and the in-process
    // one — `result_tables` is keyed by name across every run on this host.
    let routed_name = format!("recent_{}", jammi_test_utils::unique_suffix());
    let in_process_name = format!("recent_{}", jammi_test_utils::unique_suffix());
    let ctas = |name: &str| {
        format!(
            "CREATE TABLE {name} AS SELECT _row_id, _source_id, _model_id, vector \
             FROM \"jammi.{table}\" ORDER BY _row_id"
        )
    };
    let read_back = |name: &str| {
        format!(
            "SELECT _row_id, _source_id, _model_id, vector FROM \"jammi.{name}\" ORDER BY _row_id"
        )
    };

    let jobs_before = session.catalog().list_compute_jobs().await.unwrap().len();
    let created = flight_statement(addr, &ctas(&routed_name))
        .await
        .unwrap_or_else(|e| {
            fleet.dump_diagnostics(&format!("the routed CREATE TABLE AS failed: {e}"));
            panic!("the routed CREATE TABLE AS failed: {e}");
        });
    assert!(
        created.iter().all(|b| b.num_rows() == 0),
        "a CREATE TABLE AS returns no rows"
    );

    session
        .sql(&ctas(&in_process_name))
        .await
        .expect("the in-process CREATE TABLE AS");

    // The table IS catalogued state: a `ready` row of the statement kind,
    // and the same bytes the in-process statement wrote.
    let (routed, routed_bytes) = table_bytes(&session, &routed_name).await;
    let (_, in_process_bytes) = table_bytes(&session, &in_process_name).await;
    assert_eq!(
        routed.kind,
        jammi_db::catalog::result_repo::ResultTableKind::Statement
    );
    assert_eq!(
        routed_bytes, in_process_bytes,
        "the table's Parquet on the object store must be byte-identical to the one the same \
         statement writes in-process"
    );

    // The write ran on an executor, never on the query tier.
    let (writer, _) = await_sink_writer(&mut fleet, &executors, &routed_name).await;
    assert!(executors.contains(&writer));
    assert!(
        !fleet.log_contents(&query_tier).contains(SINK_WRITE_LOG),
        "the query tier submits the sink; it never writes"
    );
    let submitted = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            session.catalog().list_compute_jobs().await.unwrap().len() == jobs_before + 1
        })
    })
    .await;
    assert!(submitted, "the statement became exactly one compute job");
    let scheduler = fleet.label(0).to_string();
    let scheduler_log = harness::await_log_contains(
        &mut fleet,
        &scheduler,
        BOUND_TASK_LOG,
        "the statement's stages bound to an executor",
    )
    .await;
    // A stage of the statement bound to one of THIS fleet's executors (the
    // query tier is none). Never "every bound line": the shared catalog
    // still carries live-looking rows of executors earlier runs killed,
    // and a stage bound to one of those is re-bound once its launch fails.
    let bound_here = scheduler_log
        .lines()
        .filter(|line| line.contains(BOUND_TASK_LOG))
        .any(|line| ids.iter().any(|id| line.contains(id)));
    assert!(
        bound_here,
        "a bound stage names one of the fleet's executors {ids:?}; log:\n{scheduler_log}"
    );

    // A different query-tier process, up before the table existed, reads
    // it through the catalog: the rows are the in-process table's.
    let other_addr = fleet.flight_addr(&other_query_tier);
    let read_elsewhere = flight_statement(other_addr, &read_back(&routed_name))
        .await
        .unwrap_or_else(|e| {
            fleet.dump_diagnostics(&format!("reading the created table elsewhere failed: {e}"));
            panic!("reading the created table on another query tier failed: {e}");
        });
    let in_process = session
        .sql(&read_back(&in_process_name))
        .await
        .expect("the in-process read back");
    let schema = in_process[0].schema();
    assert_eq!(
        ipc_bytes(&concat_or_empty(&read_elsewhere, schema.clone())),
        ipc_bytes(&concat_or_empty(&in_process, schema)),
        "the rows read on another query tier must be Arrow-IPC-byte-identical to the in-process \
         table's"
    );

    drop(fleet);
}

/// A `SELECT` on the same client-role query tier never leaves it: its rows
/// come back, no `compute_jobs` row appears and the scheduler binds
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn select_over_flight_sql_on_a_client_never_submits_a_compute_job() {
    const TEST: &str = "select_over_flight_sql_on_a_client_never_submits_a_compute_job";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let source_name = harness::unique_source_name("two_files");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let scheduler_port = jammi_test_utils::free_port();
    let specs = vec![
        ProcSpec::fresh(
            BallistaRole::SchedulerAndExecutor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        ),
        client_spec(scheduler_port),
    ];
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    let ids = await_fleet_registered(&session, &fleet).await;
    let query_tier = fleet.label(1).to_string();
    await_flight_up(&mut fleet, &query_tier).await;
    let addr = fleet.flight_addr(&query_tier);

    let jobs_before = session.catalog().list_compute_jobs().await.unwrap().len();
    let select = format!("SELECT id, text FROM \"{source_name}\".public.two_files ORDER BY id");
    let rows = flight_statement(addr, &select).await.unwrap_or_else(|e| {
        fleet.dump_diagnostics(&format!("the SELECT failed: {e}"));
        panic!("the SELECT failed: {e}");
    });
    let expected = session.sql(&select).await.expect("the in-process SELECT");
    assert_eq!(
        ipc_bytes(&concat_in_arrival_order(&rows)),
        ipc_bytes(&concat_in_arrival_order(&expected))
    );

    let jobs_after = session.catalog().list_compute_jobs().await.unwrap().len();
    assert_eq!(jobs_after, jobs_before, "a SELECT submits no compute job");
    let scheduler_log = fleet.log_contents(fleet.label(0));
    assert!(
        !scheduler_log.contains(BOUND_TASK_LOG),
        "the scheduler bound nothing for a SELECT (its only executor is {}); log:\n\
         {scheduler_log}",
        ids[0]
    );
    assert!(
        !fleet.log_contents(&query_tier).contains(SINK_WRITE_LOG),
        "a SELECT writes no result table"
    );

    drop(fleet);
}

/// A routed statement that refuses raises the same typed error in-process
/// and routed: a `CREATE TABLE … AS` over an inference whose keyed input
/// carries a null key is `InvalidKey { column: "id", null_count: 1 }` on
/// the query tier's Flight SQL status exactly as on the in-process session
/// — the sink on the executor fails its row under its own writer, so no
/// table of either name is left `ready` or `building`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routed_create_table_as_refusing_a_null_key_raises_the_in_process_error() {
    const TEST: &str = "routed_create_table_as_refusing_a_null_key_raises_the_in_process_error";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let source_name = harness::unique_source_name("null_key");
    let url = jammi_test_utils::write_null_key_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let (mut specs, scheduler_port) = standard_fleet_specs();
    specs.push(client_spec(scheduler_port));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let query_tier = fleet.label(3).to_string();
    await_flight_up(&mut fleet, &query_tier).await;
    let addr = fleet.flight_addr(&query_tier);

    let routed_name = format!("keyed_{}", jammi_test_utils::unique_suffix());
    let in_process_name = format!("keyed_{}", jammi_test_utils::unique_suffix());
    let ctas = |name: &str| {
        format!(
            "CREATE TABLE {name} AS SELECT _row_id, _status FROM annotate('{model}', \
             'text_embedding', '{source_name}.public.null_key', 'id', 'text')",
            model = harness::tiny_bert_model()
        )
    };
    let jobs_before = session.catalog().list_compute_jobs().await.unwrap().len();

    let in_process = session
        .sql(&ctas(&in_process_name))
        .await
        .expect_err("a null key refuses in-process");
    let routed = flight_statement(addr, &ctas(&routed_name))
        .await
        .err()
        .unwrap_or_else(|| {
            fleet.dump_diagnostics("the routed statement did not refuse");
            panic!("the routed statement did not refuse");
        });
    let routed = jammi_wire::error_from_status(&routed);
    for (side, err) in [("in-process", in_process), ("routed", routed)] {
        match err {
            JammiError::InvalidKey { column, null_count } => {
                assert_eq!(column, "id", "{side}");
                assert_eq!(null_count, 1, "{side}");
            }
            other => panic!("{side}: expected InvalidKey, got {other:?}"),
        }
    }
    let submitted = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            session.catalog().list_compute_jobs().await.unwrap().len() == jobs_before + 1
        })
    })
    .await;
    assert!(
        submitted,
        "the refusal came from the compute plane: one compute job was submitted"
    );
    for name in [&routed_name, &in_process_name] {
        let row = session.catalog().get_result_table(name).await.unwrap();
        assert!(
            row.as_ref()
                .is_none_or(|r| r.status
                    == jammi_db::catalog::status::ResultTableStatus::Failed.to_string()),
            "a refused statement leaves no live table {name}: {row:?}"
        );
    }

    drop(fleet);
}

/// An `embedding` job claimed by a client-role process routes its sink to
/// an executor: the executor's log names the write, the claimant's does
/// not, and the table's Parquet on the shared object store is
/// byte-identical to the one the same job writes in-process.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embedding_job_on_a_client_routes_its_sink_to_an_executor_and_matches_in_process() {
    const TEST: &str =
        "embedding_job_on_a_client_routes_its_sink_to_an_executor_and_matches_in_process";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let source_name = harness::unique_source_name("two_files");
    let url = harness::write_two_file_source(dir.path());
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let model = harness::tiny_bert_model();

    let (mut specs, scheduler_port) = standard_fleet_specs();
    specs.push(embedding_client_spec(scheduler_port));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let executors: Vec<String> = (0..3).map(|i| fleet.label(i).to_string()).collect();
    let claimant = fleet.label(3).to_string();

    let job = session
        .enqueue(
            jammi_ai::jobs::JobSpec::Embedding {
                source_id: source_name.clone(),
                model_id: model.clone(),
                columns: vec!["text".to_string()],
                key_column: "id".to_string(),
                modality: jammi_wire::request::Modality::Text,
                cache: jammi_db::store::CachePolicy::Bypass,
            },
            0,
        )
        .await
        .expect("the embedding job enqueues");
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job.job_id,
        "the embedding job claimed on the client-role process reaches a terminal status",
        |r| {
            r.status == jammi_db::catalog::status::JobStatus::Completed.to_string()
                || r.status == jammi_db::catalog::status::JobStatus::Failed.to_string()
        },
    )
    .await;
    assert_eq!(
        record.status,
        jammi_db::catalog::status::JobStatus::Completed.to_string(),
        "the placed embedding job completes: {:?}",
        record.error
    );
    let claimant_id = instance_id_of_label(&session, &claimant).await;
    assert_eq!(record.claimed_by.as_deref(), Some(claimant_id.as_str()));

    let routed = session
        .catalog()
        .find_result_tables(&source_name, Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap()
        .into_iter()
        .find(|t| t.status == jammi_db::catalog::status::ResultTableStatus::Ready.to_string())
        .expect("the job's ready embedding table");
    let (writer, _) = await_sink_writer(&mut fleet, &executors, &routed.table_name).await;
    assert!(executors.contains(&writer));
    assert!(
        !fleet.log_contents(&claimant).contains(SINK_WRITE_LOG),
        "the claimant submits the sink; it never writes"
    );

    let (in_process, _) = session
        .generate_text_embeddings(
            &source_name,
            &model,
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .expect("the in-process embedding");
    let (_, routed_bytes) = table_bytes(&session, &routed.table_name).await;
    let (_, in_process_bytes) = table_bytes(&session, &in_process.table_name).await;
    assert_eq!(
        routed_bytes, in_process_bytes,
        "the embedding table an executor wrote must be byte-identical to the in-process one"
    );

    drop(fleet);
}

/// A client-role worker claiming exactly `kinds`.
fn client_worker_spec(scheduler_port: u16, kinds: &'static [&'static str]) -> ProcSpec {
    ProcSpec::fresh(
        BallistaRole::Client { scheduler_port },
        WorkerRole {
            enabled: true,
            kinds: Some(kinds),
            idle_poll_secs: 1,
        },
    )
}

/// An id-only account graph: a ring of `nodes` with two chords each.
fn write_ring_edge_source(dir: &std::path::Path, nodes: usize) -> String {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let name = |i: usize| format!("acct-{:04}", i % nodes);
    let (src, dst): (Vec<String>, Vec<String>) = (0..nodes)
        .flat_map(|i| {
            [
                (name(i), name(i + 1)),
                (name(i), name(i + 7)),
                (name(i), name(i + 31)),
            ]
        })
        .unzip();
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(src)),
            Arc::new(StringArray::from(dst)),
        ],
    )
    .unwrap();
    let path = dir.join("ring_edges.parquet");
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    format!("file://{}", path.display())
}

/// A `graph_structure` job and a `propagate` job, each claimed by a
/// client-role process, run on the compute plane: the output table's sink is
/// written by an executor and never by the claimant, and each table's Parquet
/// on the shared object store is byte-identical to the one the same request
/// writes in-process. Every hop of the placed plan reads the run's adjacency
/// snapshot from that same store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn graph_jobs_on_a_client_run_on_an_executor_and_match_in_process() {
    use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
    use jammi_ai::pipeline::graph_propagation::PropagateRequest;
    use jammi_ai::pipeline::graph_structure::StructureRequest;
    use jammi_db::catalog::status::{JobStatus, ResultTableStatus};
    use jammi_db::store::CachePolicy;

    const TEST: &str = "graph_jobs_on_a_client_run_on_an_executor_and_match_in_process";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let source_name = harness::unique_source_name("ring_edges");
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(write_ring_edge_source(dir.path(), 400)),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let edges = EdgeSourceRef::Registered {
        source_id: source_name.clone(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        type_column: None,
        weight_column: None,
        as_of_column: None,
    };
    let structure = StructureRequest::new(source_name.clone(), edges.clone())
        .with_dimensions(64)
        .with_weights([0.0, 1.0, 1.0]);

    // In-process first: the reference bytes, and the embedding table the
    // propagation job propagates.
    let (structure_in_process, _) = session
        .generate_structure_embeddings(&structure, CachePolicy::Bypass)
        .await
        .expect("the in-process structure encoding");
    let propagate = PropagateRequest::new(source_name.clone(), edges)
        .with_embedding_table(structure_in_process.table_name.clone())
        .with_direction(EdgeDirection::Undirected);
    let (propagate_in_process, _) = session
        .propagate_embeddings(&propagate, CachePolicy::Bypass)
        .await
        .expect("the in-process propagation");

    let (mut specs, scheduler_port) = standard_fleet_specs();
    specs.push(client_worker_spec(scheduler_port, &["graph_structure"]));
    specs.push(client_worker_spec(scheduler_port, &["propagate"]));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let executors: Vec<String> = (0..3).map(|i| fleet.label(i).to_string()).collect();

    let jobs = [
        (
            "graph_structure",
            fleet.label(3).to_string(),
            jammi_ai::jobs::JobSpec::GraphStructure {
                request: structure,
                cache: CachePolicy::Bypass,
            },
            structure_in_process,
        ),
        (
            "graph_propagate",
            fleet.label(4).to_string(),
            jammi_ai::jobs::JobSpec::Propagate {
                request: propagate,
                cache: CachePolicy::Bypass,
            },
            propagate_in_process,
        ),
    ];
    for (model, claimant, spec, in_process) in jobs {
        let job = session.enqueue(spec, 0).await.expect("the job enqueues");
        let record = harness::await_job(
            &mut fleet,
            &session,
            &job.job_id,
            "the graph job claimed on the client-role process reaches a terminal status",
            |r| {
                r.status == JobStatus::Completed.to_string()
                    || r.status == JobStatus::Failed.to_string()
            },
        )
        .await;
        assert_eq!(
            record.status,
            JobStatus::Completed.to_string(),
            "the placed {model} job completes: {:?}",
            record.error
        );
        let claimant_id = instance_id_of_label(&session, &claimant).await;
        assert_eq!(record.claimed_by.as_deref(), Some(claimant_id.as_str()));

        let routed = session
            .catalog()
            .find_result_tables(&source_name, Some(ModelTask::TextEmbedding), Some(model))
            .await
            .unwrap()
            .into_iter()
            .find(|t| {
                t.status == ResultTableStatus::Ready.to_string()
                    && t.table_name != in_process.table_name
            })
            .expect("the job's ready table");
        let (writer, _) = await_sink_writer(&mut fleet, &executors, &routed.table_name).await;
        assert!(
            executors.contains(&writer),
            "{model}: an executor wrote the table"
        );
        assert!(
            !fleet
                .log_contents(&claimant)
                .lines()
                .any(|line| line.contains(SINK_WRITE_LOG) && line.contains(&routed.table_name)),
            "{model}: the claimant submits the sink; it never writes the table"
        );

        let (_, routed_bytes) = table_bytes(&session, &routed.table_name).await;
        let (_, in_process_bytes) = table_bytes(&session, &in_process.table_name).await;
        assert_eq!(
            routed_bytes, in_process_bytes,
            "{model}: the table an executor wrote must be byte-identical to the in-process one"
        );
    }

    drop(fleet);
}

/// An executor killed while it holds a sink's row. The row is the
/// executor's — `building` under its store's writer id, never handed back
/// — until its lease expires and a successor's own dispatch reclaims it.
/// The plane gives the executor up on its own heartbeat timeout
/// (Ballista's `executor_timeout_seconds`, swept every
/// `expire_dead_executor_interval_seconds` — the bound this test measures
/// against), and AT that loss the scheduler fails every placed job bound
/// to the executor typed (`cluster::PlacedJobs`): the claimant's placed
/// stream ends with `JammiError::ExecutorLost` naming the executor and
/// the plane's job, no stage of the job is ever relaunched, and the
/// attempt is spent and left for reclaim — never terminal. A successor
/// claim (either client's) finds the predecessor's row `building` under
/// an expired lease, fails it, and writes the table anew on a surviving
/// executor, byte-identical to the in-process one. One job, two
/// attempts, no second submission.
///
/// The scheduler hosts no executor here: a placed task never lands on the
/// process the plane lives in, so the kill takes an executor and only an
/// executor (a scheduler's death is the scheduler-restart test's).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_executor_mid_sink_write_is_reclaimed_and_a_rerun_writes_the_identical_table() {
    const TEST: &str =
        "killed_executor_mid_sink_write_is_reclaimed_and_a_rerun_writes_the_identical_table";
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    // Enough rows that the sink's write — the index build over every
    // vector, after the rows have streamed — takes seconds on the executor
    // holding the row, so a kill sent on its write line lands inside the
    // write; each row's inference stays a few tokens.
    const ROWS: usize = 60_000;
    let source_name = harness::unique_source_name("many_rows");
    let url = harness::write_many_row_source(dir.path(), ROWS);
    session
        .add_source(
            &source_name,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let model = harness::tiny_bert_model();
    let building_status = jammi_db::catalog::status::ResultTableStatus::Building.to_string();
    let ready_status = jammi_db::catalog::status::ResultTableStatus::Ready.to_string();

    // A scheduler running no task, three executors — so the successor's
    // sink has somewhere to land after one dies — and two clients claiming
    // `embedding` jobs: one holds the first attempt until the plane fails
    // it, and either claims the successor attempt. Each executor is a
    // fleet member like the standard fleet's: a worker of a kind this test
    // never enqueues, so it claims nothing and is known to the catalog by
    // its label.
    let scheduler_port = jammi_test_utils::free_port();
    let mut specs = vec![ProcSpec::fresh(
        BallistaRole::Scheduler { scheduler_port },
        WorkerRole {
            enabled: false,
            kinds: None,
            idle_poll_secs: 1,
        },
    )];
    specs.extend((0..3).map(|_| {
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kinds: Some(&["context_predictor"]),
                idle_poll_secs: 1,
            },
        )
    }));
    specs.push(embedding_client_spec(scheduler_port));
    specs.push(embedding_client_spec(scheduler_port));
    let mut fleet = harness::spawn_fleet(&backends, &result_root, specs);
    await_fleet_registered(&session, &fleet).await;
    let scheduler = fleet.label(0).to_string();
    let executors: Vec<String> = (1..4).map(|i| fleet.label(i).to_string()).collect();
    let clients: Vec<String> = (4..6).map(|i| fleet.label(i).to_string()).collect();

    let job = session
        .enqueue(
            jammi_ai::jobs::JobSpec::Embedding {
                source_id: source_name.clone(),
                model_id: model.clone(),
                columns: vec!["text".to_string()],
                key_column: "id".to_string(),
                modality: jammi_wire::request::Modality::Text,
                cache: jammi_db::store::CachePolicy::Bypass,
            },
            0,
        )
        .await
        .expect("the embedding job enqueues");

    // The attempt's building row, and the executor writing it — killed on
    // the sink's own write line, the earliest moment the row is its.
    let building = loop {
        let rows = session
            .catalog()
            .find_result_tables(&source_name, Some(ModelTask::TextEmbedding), None)
            .await
            .unwrap();
        if let Some(row) = rows.into_iter().find(|t| t.status == building_status) {
            break row;
        }
        tokio::time::sleep(harness::POLL_INTERVAL).await;
    };
    let (writer, writer_id) = await_sink_writer(&mut fleet, &executors, &building.table_name).await;
    let first_attempt = session
        .catalog()
        .pinned_to_tenant(None)
        .get_job(&job.job_id)
        .await
        .unwrap();
    let claimant = first_attempt
        .claimed_by
        .clone()
        .expect("the placed attempt's row names its claimant");
    assert_eq!(first_attempt.attempts, 1);
    let claimant_label = harness::label_of(&session, &claimant).await;
    assert!(clients.contains(&claimant_label), "a client claims the job");
    assert!(
        fleet.kill9(&writer),
        "the writing executor {writer:?} is one of the spawned processes"
    );
    let killed_at = std::time::Instant::now();

    // The row is the killed executor's: `building` under its store's
    // writer id — the id its own write line named — never the claimant's.
    let row = session
        .catalog()
        .get_result_table(&building.table_name)
        .await
        .unwrap()
        .expect("the row outlives its writer");
    assert_eq!(
        (row.status.as_str(), row.writer_id.as_deref()),
        (building_status.as_str(), Some(writer_id.as_str())),
        "the row was handed to the executor and never handed back"
    );

    // The plane gives the executor up on its own heartbeat timeout
    // (Ballista's, longer than a single job's bound), and at that loss the
    // scheduler fails the placed job typed: the claimant's attempt ends
    // with `ExecutorLost` naming the killed executor, spent and left for
    // reclaim — the line its log carries.
    let writer_instance = instance_id_of_label(&session, &writer).await;
    let lost_line = format!("compute plane: executor `{writer_instance}` holding placed job `");
    let lost = harness::await_condition(harness::TERMINAL_TIMEOUT * 2, || {
        let log = fleet.log_contents(&claimant_label);
        log.contains(jammi_ai::jobs::EXECUTOR_LOST_ATTEMPT_LOG) && log.contains(&lost_line)
    })
    .await;
    if !lost {
        fleet.dump_diagnostics("the claimant never learned of the executor's loss");
    }
    assert!(
        lost,
        "the claimant's attempt ends with the typed loss naming the killed executor"
    );
    let loss_after = killed_at.elapsed();
    let scheduler_log = fleet.log_contents(&scheduler);
    assert!(
        scheduler_log.contains(jammi_ballista::cluster::EXECUTOR_LOST_LOG)
            && scheduler_log.contains(&writer_instance),
        "the scheduler failed the placed job at the loss, naming the executor"
    );
    assert!(
        session
            .catalog()
            .list_compute_executor_devices()
            .await
            .unwrap()
            .iter()
            .all(|(id, _)| id != &writer_instance),
        "the killed executor's registration is gone before its jobs are failed"
    );

    // The attempt is left for reclaim, never terminal: a successor claim
    // runs the job to completion — attempts spent, none released, no
    // second submission.
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job.job_id,
        "the successor attempt completes on a surviving executor",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    let completed_after = killed_at.elapsed();
    assert!(
        record.attempts >= 2 && record.releases == 0,
        "the lost attempt is spent and a successor's claim runs the job: attempts={} releases={}",
        record.attempts,
        record.releases
    );
    assert_eq!(record.error, None, "a completed job carries no error");
    let successor_label =
        harness::label_of(&session, record.claimed_by.as_deref().expect("a claimant")).await;
    assert!(
        clients.contains(&successor_label),
        "a client, never an executor, claims the successor attempt"
    );
    let jammi_ai::jobs::JobResult::Table { table, .. } =
        serde_json::from_str::<jammi_ai::jobs::JobResult>(
            record.result.as_deref().expect("a completed job's result"),
        )
        .expect("the result decodes")
    else {
        panic!("an embedding job's result is a table");
    };

    // The successor's dispatch reclaimed the predecessor's row — it is no
    // longer `building` — and wrote the table anew on a survivor,
    // byte-identical to the in-process one.
    let predecessor = session
        .catalog()
        .get_result_table(&building.table_name)
        .await
        .unwrap();
    assert!(
        predecessor.is_none_or(|r| r.status != building_status),
        "the successor reclaims the killed executor's building row"
    );
    assert_ne!(table, building.table_name, "the successor writes anew");
    let (rerun, rerun_bytes) = table_bytes(&session, &table).await;
    assert_eq!(rerun.status, ready_status);
    let (rerun_writer, _) = await_sink_writer(&mut fleet, &executors, &table).await;
    assert_ne!(
        rerun_writer, writer,
        "a survivor wrote the successor's table"
    );
    for client in &clients {
        assert!(
            !fleet.log_contents(client).contains(SINK_WRITE_LOG),
            "a client submits the sink; it never writes"
        );
    }
    let (in_process, _) = session
        .generate_text_embeddings(
            &source_name,
            &model,
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .expect("the in-process embedding");
    let (_, in_process_bytes) = table_bytes(&session, &in_process.table_name).await;
    assert_eq!(
        rerun_bytes, in_process_bytes,
        "the successor's table must be byte-identical to the in-process one"
    );

    eprintln!(
        "killed_executor_mid_sink_write: the loss reached the claimant {loss_after:?} after the \
         kill; the successor completed {completed_after:?} after it"
    );

    drop(fleet);
}
