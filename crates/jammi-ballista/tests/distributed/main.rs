//! The three(+)-process Ballista lane (contract `feat_500-wave4` §2.5, §7,
//! §9; acceptance (a3)-(a5), (b1)-(b6)). `required-features =
//! ["live-distributed-tests"]` (`Cargo.toml`); needs, on top of that:
//!
//! 1. `cargo build -p jammi-server --bin jammi-server --features
//!    storage-s3` into the SAME `CARGO_TARGET_DIR` this test binary is
//!    built into (`harness::jammi_server_binary`).
//! 2. `JAMMI_TEST_PG_URL` (a live Postgres).
//! 3. `JAMMI_TEST_S3_ENDPOINT` / `_S3_BUCKET` / `AWS_ACCESS_KEY_ID` /
//!    `AWS_SECRET_ACCESS_KEY` (an S3-compatible object store, MinIO in
//!    dev/CI).
//! 4. `JAMMI_REQUIRE_DISTRIBUTED=1` to turn an unconfigured-lane skip into a
//!    hard failure (CI's own posture).
//!
//! **Uncovered** (named here and in this crate's contract file, never a
//! silent skip in the exit code CI reads): acceptance (a3)'s "both
//! executors executed at least one task of the job" is read from the
//! SCHEDULER PROCESS's OWN captured log (a `tracing::info!` line this unit
//! added at the one call site `DevicePlacement::bind_tasks` actually binds a
//! task, `crates/jammi-ballista/src/placement.rs`) rather than through
//! Ballista's `SchedulerGrpcClient::get_job_status`: the public
//! `ballista_core::execution_plans::execute_physical_plan` a client-side
//! `submit_physical_plan` caller uses never returns or exposes the
//! internally-minted `job_id` (confirmed by reading
//! `ballista-core-54.1.0/src/execution_plans/distributed_query.rs:332-405` —
//! the job id lives only in a private `Arc<Mutex<Option<JobId>>>` the
//! function never returns), so a caller outside the scheduler process has no
//! `job_id` to poll `get_job_status` with. The log-line determinant is the
//! honest substitute: it is the SAME fact (which executor a stage/partition
//! bound to), read from the one process that actually knows it.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::gang_exec::{GangDescriptor, GangExec};
use jammi_ai::pipeline::embedding::build_embedding_plan;
use jammi_ai::session::InferenceSession;
use jammi_ballista::client::submit_physical_plan;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

use harness::{Backends, BallistaRole, Fleet, JobSize, ProcSpec, WorkerRole};

/// The deepest (leaf) plan node's own partition count — the scan stage's,
/// regardless of how many `CoalescePartitionsExec`/operator nodes wrap it
/// (contract §2.5: `InferenceExec(SortExec(KeyCheckExec(
/// CoalescePartitionsExec(scan))))`).
fn leaf_partition_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let children = plan.children();
    match children.first() {
        Some(first) => leaf_partition_count(first),
        None => plan.output_partitioning().partition_count(),
    }
}

/// Arrow IPC (stream format) bytes of one batch — the byte-comparison unit
/// contract §2.5 names ("compare the Arrow IPC bytes of the batches").
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

/// Sort `batches` by `key_column` (ascending), concatenate into one batch —
/// the sink's own deterministic order once collected out of order across
/// two executors' partitions — and drop `_latency_ms` (wall-clock timing,
/// legitimately different between the in-process and the through-Ballista
/// run — the same exclusion `crates/jammi-ai/tests/it/content_hash.rs`'s
/// own thread-count-invariance oracle names: "excluding the wall-clock
/// `_latency_ms`").
fn sort_and_concat(batches: &[RecordBatch], key_column: &str) -> RecordBatch {
    let non_empty: Vec<RecordBatch> = batches
        .iter()
        .filter(|b| b.num_rows() > 0)
        .cloned()
        .collect();
    assert!(!non_empty.is_empty(), "no non-empty batches to sort/concat");
    let schema = non_empty[0].schema();
    let combined = arrow::compute::concat_batches(&schema, &non_empty).unwrap();
    let sort_indices =
        arrow::compute::sort_to_indices(combined.column_by_name(key_column).unwrap(), None, None)
            .unwrap();
    let sorted = arrow::compute::take_record_batch(&combined, &sort_indices).unwrap();
    drop_column(&sorted, "_latency_ms")
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
/// includes `fine_tune` (LANE brief item 1's determinant, confirmed against
/// `jammi_db::catalog::jobs_repo::Catalog::list_gang_members`, which admits
/// a candidate only when its `workers.kinds` token set contains the job's
/// own kind); `lane-2`/`lane-3` host executors only and list
/// `context_predictor`, so they are fleet MEMBERS (`[worker] enabled =
/// true`) but never claimants of a `fine_tune` job themselves — the
/// placed-executor-turned-coordinator's OWN `list_gang_members(kind =
/// "fine_tune")` call therefore always finds exactly `lane-1` as its sole
/// rank-1 candidate, deterministically, regardless of whether `lane-2` or
/// `lane-3` was the one placed onto.
fn standard_fleet_specs() -> (Vec<ProcSpec>, u16) {
    let scheduler_port = harness::free_port();
    let specs = vec![
        ProcSpec::fresh(
            BallistaRole::SchedulerAndExecutor { scheduler_port },
            WorkerRole {
                enabled: true,
                kind: Some("fine_tune"),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kind: Some("context_predictor"),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::Executor { scheduler_port },
            WorkerRole {
                enabled: true,
                kind: Some("context_predictor"),
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

/// Wait until every `[worker]`-enabled label in `labels` has (1) a
/// `workers` row and (2) a `compute_executors` registration, THEN return
/// their instance ids in the SAME order as `labels`. Never a bare row
/// COUNT (`list_compute_executors().len() >= n`): the shared Postgres
/// catalog accumulates rows from every OTHER test run on this host
/// (SIGKILL never runs a graceful `remove_executor`), so a count-based
/// wait can spuriously observe stale rows and return before THIS fleet's
/// own processes are actually up.
async fn await_fleet_registered(session: &Arc<InferenceSession>, labels: &[&str]) -> Vec<String> {
    let ok = harness::await_condition(Duration::from_secs(60), || {
        futures::executor::block_on(async {
            let workers = session.catalog().list_workers().await.unwrap_or_default();
            labels
                .iter()
                .all(|l| workers.iter().any(|w| w.label.as_deref() == Some(*l)))
        })
    })
    .await;
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

// ─── (a3) ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embedding_job_across_two_executors_matches_in_process() {
    const TEST: &str = "embedding_job_across_two_executors_matches_in_process";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    // Force per-file partition splitting: DataFusion's default file-group
    // builder coalesces files below `repartition_file_min_size` (10 MiB)
    // into ONE partition regardless of `target_partitions` (confirmed by
    // executing this exact plan without this override:
    // `leaf_partition_count` read 1, not 2, even at 300 rows/file) —
    // dropping the threshold makes the two-file source scan with one
    // partition per file, the non-vacuous shape oracle (a3) needs.
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
        "the two-file source's scan stage must have >= 2 partitions for oracle (a3) to be \
         non-vacuous, got {scan_partitions}"
    );

    // In-process, on the harness session — the K4 baseline.
    let task_ctx = session.context().task_ctx();
    let in_process = datafusion::physical_plan::collect(plan.clone(), task_ctx)
        .await
        .expect("in-process collect");
    let in_process = sort_and_concat(&in_process, "_row_id");

    // Across the fleet, through the scheduler.
    let (specs, scheduler_port) = standard_fleet_specs();
    let fleet = Fleet::spawn(&backends, &result_root, specs);
    let ids =
        await_fleet_registered(&session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;
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
    let placed = sort_and_concat(&placed, "_row_id");

    assert_eq!(
        ipc_bytes(&in_process),
        ipc_bytes(&placed),
        "the plan collected through Ballista must be Arrow-IPC-byte-identical to the same \
         plan collected in-process (K4)"
    );

    // (ii): both non-submitter executors actually ran a task of this job —
    // read from the SCHEDULER process's own log (this module's Uncovered
    // note explains why not through get_job_status).
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

// ─── (a4) / (b5) ───────────────────────────────────────────────────────────

/// Standard fleet + a submitted `world_size = 2` gang fine-tune, polled to
/// `running`. Returns `(fleet, job_id, expected_model, claimant_instance_id)`.
async fn submit_and_await_placed_claim(
    backends: &Backends,
    result_root: &str,
    session: &Arc<InferenceSession>,
    source: &str,
    size: JobSize,
) -> (Fleet, String, String, String) {
    let (specs, scheduler_port) = standard_fleet_specs();
    let _ = scheduler_port;
    let mut fleet = Fleet::spawn(backends, result_root, specs);
    await_fleet_registered(session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;

    let (job_id, expected_model) = harness::submit_gang_fine_tune(session, source, size, 2).await;
    let record = harness::await_job(
        &mut fleet,
        session,
        &job_id,
        "the job is claimed and running",
        |r| r.status == "running" && r.claimed_by.is_some(),
    )
    .await;
    let claimant = record.claimed_by.expect("running job has a claimant");
    (fleet, job_id, expected_model, claimant)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placed_gang_completes_on_a_registered_executor_other_than_the_submitter() {
    const TEST: &str = "placed_gang_completes_on_a_registered_executor_other_than_the_submitter";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::register_training_source(&session, &source).await;

    let (mut fleet, job_id, expected_model, claimant) =
        submit_and_await_placed_claim(&backends, &result_root, &session, &source, JobSize::Quick)
            .await;

    let lane1_id = instance_id_of_label(&session, fleet.label(0)).await;
    let lane2_id = instance_id_of_label(&session, fleet.label(1)).await;
    let lane3_id = instance_id_of_label(&session, fleet.label(2)).await;
    assert_ne!(
        claimant, lane1_id,
        "the placed gang must never be bound back to the submitter's own executor (contract \
         §9 B1)"
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
        |r| r.status == "completed",
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

    // HandedOff evidence on the submitter's (lane-1's) own log.
    let lane1_log = fleet.log_contents(fleet.label(0));
    assert!(
        lane1_log.contains("run_placed_gang: submitter HandedOff"),
        "lane-1's log must show the HandedOff arm this unit's `tracing::info!` line names; \
         log:\n{lane1_log}"
    );

    // K4 (b5): the placed run's artifact bytes equal a SECOND fleet's
    // wave-3-path run (no `[ballista]` at all — process 1 coordinates
    // locally), same base model / config / device kind (CPU).
    let plain_result_root = backends.unique_result_root(&format!("{TEST}-plain"));
    let plain_source = harness::unique_source_name(&format!("{TEST}-plain"));
    harness::register_training_source(&session, &plain_source).await;
    let plain_specs = vec![
        ProcSpec::fresh(
            BallistaRole::None,
            WorkerRole {
                enabled: true,
                kind: Some("fine_tune"),
                idle_poll_secs: 1,
            },
        ),
        ProcSpec::fresh(
            BallistaRole::None,
            WorkerRole {
                enabled: true,
                kind: Some("fine_tune"),
                idle_poll_secs: 1,
            },
        ),
    ];
    let mut plain_fleet = Fleet::spawn(&backends, &plain_result_root, plain_specs);
    let (plain_job_id, plain_model) =
        harness::submit_gang_fine_tune(&session, &plain_source, JobSize::Quick, 2).await;
    let plain_record = harness::await_job(
        &mut plain_fleet,
        &session,
        &plain_job_id,
        "the plain (non-Ballista) gang completes",
        |r| r.status == "completed",
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
        .and_then(|m| m.artifact_path.clone())
        .expect("placed model has an artifact_path");
    let plain_path = models
        .iter()
        .find(|m| m.model_id == plain_model)
        .and_then(|m| m.artifact_path.clone())
        .expect("plain model has an artifact_path");

    let placed_local = session
        .artifact_store()
        .fetch_artifact(&jammi_db::storage::StorageUrl::parse(&placed_path).unwrap())
        .await
        .unwrap();
    let plain_local = session
        .artifact_store()
        .fetch_artifact(&jammi_db::storage::StorageUrl::parse(&plain_path).unwrap())
        .await
        .unwrap();
    let placed_digest =
        jammi_ai::fine_tune::worker::adapter_files_digest(placed_local.dir()).unwrap();
    let plain_digest =
        jammi_ai::fine_tune::worker::adapter_files_digest(plain_local.dir()).unwrap();
    assert_eq!(
        placed_digest, plain_digest,
        "the placed run's adapter artifact must be byte-identical to the wave-3-path run's \
         (K4, per device kind — both CPU)"
    );

    drop(fleet);
    drop(plain_fleet);
}

// ─── (a5) / half of (b6) ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_executor_mid_gang_leaves_the_row_for_reclaim_then_a_successor_completes() {
    const TEST: &str =
        "killed_executor_mid_gang_leaves_the_row_for_reclaim_then_a_successor_completes";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST);
    harness::register_training_source(&session, &source).await;

    // The SAME 3-process fleet a4 uses (proven: lane-0 is the ONLY
    // `fine_tune`-kind claimant/submitter, deterministically placed onto
    // lane-1 or lane-2). A SECOND, independent `fine_tune`-kind bidder is
    // spawned LATE — only AFTER the first claim+placement has already
    // resolved — so it plays no part in that race and stays free the whole
    // time. This two-step spawn is load-bearing, executed: a bidder present
    // from t=0 raced lane-0 for the INITIAL claim and, when it won, ran the
    // whole gang in-process (no scheduler role of its own installs a
    // `PlacedGangSubmitter` — `roles::host_scheduler`'s doc), never
    // exercising the placed path a5 needs; a SECOND SCHEDULER able to place
    // independently (`BallistaRole::SchedulerOnly`) reds a different way —
    // Ballista's OWN task binder gates on ITS OWN executor-heartbeat CACHE
    // (`ballista-scheduler-54.1.0/src/state/executor_manager.rs:117-121`'s
    // `get_alive_executors`), which a second scheduler never populates for
    // executors that registered with a DIFFERENT scheduler (`CatalogCluster
    // State`'s own documented heartbeat-cache-staleness caveat) — observed
    // as "There are no alive executors to bind tasks" forever. The reclaimer
    // that actually works has NO ballista role of its own: on reclaiming it
    // runs the recovered gang in-process (the wave-3 `Peer` path, dialing
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
                kind: Some("fine_tune"),
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
    // 300ms — the honest subset at this poll rate (contract §7 (a5)).
    let mut observed: Vec<String> = vec![lane0_id.clone(), claimant.clone()];
    let mut killed = false;
    let deadline = std::time::Instant::now() + harness::TERMINAL_TIMEOUT;
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
        if record.status == "completed" {
            break record;
        }
        if std::time::Instant::now() >= deadline {
            fleet.dump_diagnostics("timed out awaiting reclaim + successor completion");
            panic!(
                "timed out after {:?} awaiting reclaim + successor completion; observed \
                 claimed_by sequence: {observed:?}",
                harness::TERMINAL_TIMEOUT
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
    // recovered gang in-process, the wave-3 `Peer` path — it installs no
    // `PlacedGangSubmitter` of its own) or lane-0 itself once its own
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

// ─── (b1) ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scheduler_restart_keeps_executors_and_serves_a_new_job() {
    const TEST: &str = "scheduler_restart_keeps_executors_and_serves_a_new_job";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, scheduler_port) = standard_fleet_specs();
    let mut fleet = Fleet::spawn(&backends, &result_root, specs);
    let ids =
        await_fleet_registered(&session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;
    let (lane2_id, lane3_id) = (ids[1].clone(), ids[2].clone());
    let lane1_label = fleet.label(0).to_string();

    // SIGKILL and respawn the scheduler process (lane-1) at the SAME
    // `scheduler_bind` port. `instance_id` (`instance_id`,
    // crates/jammi-ai/src/session.rs:468) is minted at session
    // construction, never externally supplied, so the replacement is a fresh
    // instance — this oracle's own assertions below need only the OTHER
    // executors' registrations and a NEW job's completion, which the
    // shared catalog carries regardless (this unit's contract file states
    // this deviation from a literal "same instance id" reading).
    fleet.kill9(&lane1_label);
    fleet.respawn(&backends, &result_root, &lane1_label);

    // Read back the two untouched executors' registrations, through the
    // harness session's OWN catalog — the same shared store the
    // replacement scheduler itself reads (contract §7 (b1) offers this as
    // an alternative to a `SchedulerGrpcClient` call).
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
    // probe is the honest readiness check (a `submit_physical_plan` before
    // this reds with `ConnectionRefused`, executed).
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
    // serving (executed: a bare readiness probe still saw one
    // `ConnectionRefused` from `submit_physical_plan` itself). Retry the
    // real submission a few times with a short backoff, the honest
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
            Ok(Err(e)) if std::time::Instant::now() < submit_deadline => {
                let _ = e;
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
            // same executor-heartbeat-cache mechanism `(b2)`'s own doc
            // names, `ballista-scheduler-54.1.0/src/state/executor_manager.
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

// ─── (b2) ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_schedulers_over_one_catalog_serve_jobs_sequentially() {
    const TEST: &str = "two_schedulers_over_one_catalog_serve_jobs_sequentially";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    // The training source must be registered on the SHARED catalog BEFORE
    // any fleet process starts: a claiming process resolves a fine-tune
    // job's named source through ITS OWN local reload-at-startup, never a
    // dynamic re-read of another process's later write (unlike a plan
    // submitted whole via `submit_physical_plan`, whose scan already
    // carries concrete file paths — no source-name lookup on the
    // executor at all). Executed: registering AFTER `Fleet::spawn` reds
    // with "Source '…' not found" on the claiming executor.
    let source = harness::unique_source_name(TEST);
    harness::register_training_source(&session, &source).await;

    // Scheduler 4 hosts its OWN local executor too (`SchedulerAndExecutor`,
    // never `SchedulerOnly`). Executed refutation: a `SchedulerOnly`
    // scheduler 4 reds every submission with Ballista's OWN "There are no
    // alive executors to bind tasks" (`ballista-scheduler-54.1.0/src/state/
    // executor_manager.rs:117-121`'s `get_alive_executors`, gated on THIS
    // scheduler's own executor-HEARTBEAT cache — never the raw
    // `compute_executors` row set `list_compute_executors` reads) —
    // executors 2/3 heartbeat ONLY to the scheduler they registered with
    // (scheduler 1), so scheduler 4 never learns they are alive, no
    // matter how long a plan submitted to it waits (this is the concrete
    // shape of `CatalogClusterState`'s own documented heartbeat-cache-
    // staleness caveat, contract §3: "a standby scheduler's liveness view
    // of an executor it does not itself serve is only as fresh as its
    // last init"). Scheduler 4 therefore serves its OWN job on its OWN
    // local executor — still "two schedulers over one shared catalog,
    // each independently able to serve a job" (contract §7 (b2)), never
    // a claim that Ballista binds a task ACROSS two live schedulers.
    let (mut specs, scheduler1_port) = standard_fleet_specs();
    let scheduler4_port = harness::free_port();
    let scheduler4_idx = specs.len();
    specs.push(ProcSpec::fresh(
        BallistaRole::SchedulerAndExecutor {
            scheduler_port: scheduler4_port,
        },
        WorkerRole {
            enabled: true,
            kind: Some("context_predictor"),
            idle_poll_secs: 30,
        },
    ));
    let mut fleet = Fleet::spawn(&backends, &result_root, specs);
    await_fleet_registered(&session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;
    let ok = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            let workers = session.catalog().list_workers().await.unwrap_or_default();
            workers
                .iter()
                .any(|w| w.label.as_deref() == Some(fleet.label(scheduler4_idx)))
        })
    })
    .await;
    assert!(ok, "timed out waiting for scheduler 4's own workers row");
    let scheduler4_executor_id = instance_id_of_label(&session, fleet.label(scheduler4_idx)).await;
    let ok = harness::await_condition(Duration::from_secs(30), || {
        futures::executor::block_on(async {
            session
                .catalog()
                .list_compute_executor_devices()
                .await
                .map(|v| v.iter().any(|(id, _)| id == &scheduler4_executor_id))
                .unwrap_or(false)
        })
    })
    .await;
    assert!(
        ok,
        "timed out waiting for scheduler 4's own executor to register"
    );

    // A job through scheduler 1 completes first.
    let (job_id, model_id) =
        harness::submit_gang_fine_tune(&session, &source, JobSize::Quick, 2).await;
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        "job via scheduler 1 completes",
        |r| r.status == "completed",
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
    assert_eq!(scheduler1_port, scheduler1_port); // scheduler 1's port is fixed/used above only implicitly via the fleet.

    drop(fleet);
}

// ─── (b3) ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn device_less_cluster_refuses_gpu_bound_plan_and_accepts_cpu_plan() {
    const TEST: &str = "device_less_cluster_refuses_gpu_bound_plan_and_accepts_cpu_plan";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, scheduler_port) = standard_fleet_specs();
    let fleet = Fleet::spawn(&backends, &result_root, specs);
    await_fleet_registered(&session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;
    let scheduler_url = format!("http://127.0.0.1:{scheduler_port}");

    // KIND MATCH (LANE pressure-round correction): a GangExec's required
    // kind is its OWN descriptor's stamp, never "is this node type
    // GPU-shaped" — a dummy descriptor (never actually run) stamped `Cuda`
    // is refused on this all-CPU cluster (no registered executor lists a
    // `cuda` device); a `Cpu`-stamped one would be accepted (exercised
    // below by the real embedding plan, whose `InferenceExec` is stamped
    // from the harness session's own CPU device).
    let gang_plan: Arc<dyn ExecutionPlan> = Arc::new(GangExec::new(GangDescriptor {
        job_id: "dummy-job".to_string(),
        attempt: 0,
        world: 2,
        submitter: "dummy-submitter".to_string(),
        device_kind: jammi_db::store::manifest::ComputeDeviceKind::Cuda,
    }));
    // The device check is a fast, purely client-side catalog read before
    // any RPC (contract §3/§9): a 20s timeout is generous headroom, never
    // load-bearing for a passing run, but turns an unexpected fall-through
    // to a real submission (this crate's own regression class: see the
    // codec's `try_encode_udf`/`try_decode_udf` fix this unit's pass added)
    // into a fast, diagnosable failure instead of the test hanging.
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        submit_physical_plan(&session, &scheduler_url, gang_plan),
    )
    .await
    .unwrap_or_else(|_| {
        fleet.dump_diagnostics(
            "the device-less refusal did not return within 20s — it should never reach the \
             network at all",
        );
        panic!(
            "a GangExec plan's device-less refusal must return fast (client-side, before any \
             RPC); it did not return within 20s"
        );
    });
    let msg = match result {
        Ok(_) => {
            panic!("a GangExec plan must be refused on a device-less cluster, but it was accepted")
        }
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("Cuda") && msg.contains("cuda"),
        "the refusal must name the required kind (Debug) and the missing wire kind (cuda): {msg}"
    );

    // A CPU InferenceExec plan (the harness session's own device kind) is
    // accepted and actually runs.
    let source_name = harness::unique_source_name("two_files_b3");
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

// ─── (b4) ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_workers_and_compute_executor_devices_report_registered_devices() {
    const TEST: &str = "list_workers_and_compute_executor_devices_report_registered_devices";
    let Some(backends) = harness::required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    let (specs, _scheduler_port) = standard_fleet_specs();
    let fleet = Fleet::spawn(&backends, &result_root, specs);
    let ids =
        await_fleet_registered(&session, &[fleet.label(0), fleet.label(1), fleet.label(2)]).await;

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
