//! The cancel checkpoints, `run_now`'s finish compare-and-set, and
//! `enqueue`'s model-link derivation — each driven through the real
//! executor path with a compute run parked at a documented checkpoint
//! (`jammi_ai::jobs::compute_test_hooks`), never by re-implementing the
//! algorithm in the test.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::training_job::fine_tuned_model_id;
use jammi_ai::fine_tune::worker::{training_test_hooks, EmbeddedWorker, JobWorker};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::compute_test_hooks::{arm, ParkPoint};
use jammi_ai::jobs::{ComputeSpec, JobSpec};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::catalog::status::JobStatus;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// A source name no other test in this binary registers — the rendezvous
/// key the park hook is armed on, so parallel tests of the same kind never
/// park each other's runs.
fn unique_source(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

async fn session_with_source(source: &str) -> (Arc<InferenceSession>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            source,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    (session, dir)
}

/// An `infer` spec over `source` whose model can never load — the run is
/// meant to be stopped at a checkpoint BEFORE any producer touches it, so
/// reaching the producer at all would itself be the failure.
fn never_dispatched_infer(source: &str) -> ComputeSpec {
    ComputeSpec::Infer {
        source_id: source.to_string(),
        model_id: "local:/nonexistent/model/for-the-cancel-checkpoint".to_string(),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        cache: CachePolicy::Bypass,
    }
}

/// The `running` job row whose persisted spec names `source`.
async fn running_job_over(session: &InferenceSession, source: &str) -> JobRecord {
    session
        .catalog()
        .list_jobs()
        .await
        .unwrap()
        .into_iter()
        .find(|j| j.status == JobStatus::Running.to_string() && j.spec.contains(source))
        .unwrap_or_else(|| panic!("a running job over {source} must exist while parked"))
}

/// `run_now` observes a cancel that lands while the run is claimed and
/// about to dispatch: the caller gets `JobCancelled`, the row lands
/// `failed` with that message, and no producer ever ran (the spec's model
/// does not exist, so a dispatch would have failed differently).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_now_honours_a_cancel_requested_before_dispatch() {
    let source = unique_source("cancel-run-now");
    let (session, _dir) = session_with_source(&source).await;
    let park = arm(&source, ParkPoint::BeforeDispatch);

    let runner = Arc::clone(&session);
    let spec = never_dispatched_infer(&source);
    let run = tokio::spawn(async move { runner.run_now(spec).await });

    park.wait_parked().await;
    let job = running_job_over(&session, &source).await;
    assert_eq!(job.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(
        session.catalog().cancel_request(&job.job_id).await.unwrap(),
        "a cancel on the running inline job is recorded"
    );
    park.release();

    let outcome = run.await.unwrap();
    assert!(
        matches!(&outcome, Err(JammiError::JobCancelled { job_id }) if *job_id == job.job_id),
        "run_now must surface the honoured cancel as JobCancelled, got {outcome:?}"
    );
    let row = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Failed.to_string());
    assert!(
        row.error.as_deref().unwrap_or("").contains("cancelled"),
        "the row records the cancel as its terminal error, got {:?}",
        row.error
    );
    assert!(
        row.cancel_requested,
        "the request stays recorded on the row"
    );
}

/// The worker path: a job submitted through `enqueue`, claimed by a real
/// `JobWorker`, cancelled through its `JobHandle` while parked before
/// dispatch — `cancel()` reports the request recorded, the worker's
/// checkpoint fails the job with the cancel message, and `wait()` surfaces
/// it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_claimed_jobs_cancel_is_honoured_at_the_workers_checkpoint() {
    let source = unique_source("cancel-worker");
    let (session, _dir) = session_with_source(&source).await;
    let park = arm(&source, ParkPoint::BeforeDispatch);

    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    let worker = EmbeddedWorker::spawn_worker(
        &session,
        JobWorker::with_intervals(&session, session.worker_intervals().unwrap()),
    )
    .unwrap();

    park.wait_parked().await;
    assert_eq!(
        handle.status().await.unwrap(),
        JobStatus::Running.to_string()
    );
    assert!(
        handle.cancel().await.unwrap(),
        "JobHandle::cancel records the request on a running job"
    );
    park.release();

    let err = handle.wait().await.unwrap_err();
    assert!(
        err.to_string().contains("cancelled"),
        "wait() surfaces the cancel as the job's failure, got {err}"
    );
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Failed.to_string());
    assert!(row.error.as_deref().unwrap_or("").contains("cancelled"));
    worker.stop_and_join().await.unwrap();
}

/// A cancel requested while the job is still `queued` is honoured at the
/// worker's post-claim checkpoint — before the prior-attempt dispatch and
/// before any producer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_requested_while_queued_is_honoured_right_after_the_claim() {
    let source = unique_source("cancel-queued");
    let (session, _dir) = session_with_source(&source).await;

    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    assert!(
        handle.cancel().await.unwrap(),
        "cancel on a queued job is recorded"
    );

    let worker = EmbeddedWorker::spawn_worker(
        &session,
        JobWorker::with_intervals(&session, session.worker_intervals().unwrap()),
    )
    .unwrap();
    let err = tokio::time::timeout(Duration::from_secs(30), handle.wait())
        .await
        .expect("the worker claims and fails the cancelled job promptly")
        .unwrap_err();
    assert!(
        err.to_string().contains("cancelled"),
        "the post-claim checkpoint fails the job with the cancel message, got {err}"
    );
    worker.stop_and_join().await.unwrap();
}

/// `run_now`'s `Ok` is exactly "the row is `completed` with this result":
/// when the row went terminal underneath the run (here an operator's write
/// while the run is parked between producing and finishing), the finish
/// compare-and-set misses and the caller gets `JobAttemptSuperseded` —
/// never an `Ok` the row contradicts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_now_reports_superseded_when_its_row_went_terminal_before_the_finish() {
    let source = unique_source("cas-run-now");
    let (session, _dir) = session_with_source(&source).await;
    let park = arm(&source, ParkPoint::BeforeFinish);

    let spec = ComputeSpec::Embedding {
        source_id: source.clone(),
        model_id: tiny_bert_model(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality: jammi_wire::request::Modality::Text,
        cache: CachePolicy::Bypass,
    };
    let runner = Arc::clone(&session);
    let run = tokio::spawn(async move { runner.run_now(spec).await });

    park.wait_parked().await;
    let job = running_job_over(&session, &source).await;
    let failed = JobStatus::Failed.to_string();
    let job_id_for_sql = job.job_id.clone();
    session
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET status = $1, error = $2, lease_expires_at = NULL \
                     WHERE job_id = $3",
                    &[
                        SqlValue::TextOwned(failed),
                        SqlValue::Text("operator wrote the row"),
                        SqlValue::TextOwned(job_id_for_sql),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
    park.release();

    let outcome = run.await.unwrap();
    assert!(
        matches!(&outcome, Err(JammiError::JobAttemptSuperseded { job_id }) if *job_id == job.job_id),
        "run_now must report the missed finish CAS as JobAttemptSuperseded, got {outcome:?}"
    );
    let row = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Failed.to_string());
    assert_eq!(
        row.error.as_deref(),
        Some("operator wrote the row"),
        "the missed CAS wrote nothing over the row"
    );
    assert!(
        row.result.is_none(),
        "no result lands on a row the run no longer owns"
    );
}

/// `enqueue` links a training row exactly as the dedicated entry point
/// does (`model_ref` = the base model's catalog PK, registered first when
/// absent; `output_model_id` = the id the finish CAS will mint), and a
/// compute row with a model input by `model_source` — so a caller that
/// reaches the queue through `enqueue` gets a row the referential scan and
/// the status read both see as fully linked.
#[tokio::test]
async fn enqueue_derives_the_model_links_like_the_dedicated_entry_points() {
    let source = unique_source("enqueue-links");
    let (session, _dir) = session_with_source(&source).await;
    let base = tiny_bert_model();
    let canonical = ModelSource::parse(&base).to_string();

    let spec: JobSpec = TrainingSpec::FineTune {
        source: source.clone(),
        columns: vec!["abstract".to_string()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: base.clone(),
            config: FineTuneConfig::default(),
        },
    }
    .into();
    let handle = session.enqueue(spec, 0).await.unwrap();
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    let base_pk = session
        .catalog()
        .get_model(&canonical)
        .await
        .unwrap()
        .expect("enqueue registers the base model row when absent")
        .catalog_pk;
    assert_eq!(
        row.model_ref.as_deref(),
        Some(base_pk.as_str()),
        "model_ref is the base model's catalog PK"
    );
    assert_eq!(
        row.output_model_id.as_deref(),
        Some(fine_tuned_model_id(&handle.job_id).as_str()),
        "output_model_id is the id the finish CAS mints"
    );
    assert_eq!(
        row.model_source, None,
        "a training kind has no model_source"
    );

    // The dedicated entry point produces the same links for the same base.
    let dedicated = session
        .fine_tune(
            &source,
            &base,
            &["abstract".to_string()],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();
    let dedicated_row = session.catalog().get_job(&dedicated.job_id).await.unwrap();
    assert_eq!(dedicated_row.model_ref, row.model_ref);
    assert_eq!(
        dedicated_row.output_model_id.as_deref(),
        Some(fine_tuned_model_id(&dedicated.job_id).as_str())
    );

    // A compute kind with a model input carries its canonical model source.
    let compute: JobSpec = ComputeSpec::Infer {
        source_id: source.clone(),
        model_id: base.clone(),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        cache: CachePolicy::Bypass,
    }
    .into();
    let handle = session.enqueue(compute, 0).await.unwrap();
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.model_source.as_deref(), Some(canonical.as_str()));
    assert_eq!(row.model_ref, None);
    assert_eq!(row.output_model_id, None);

    // ...and one without a model input carries none.
    let asof: JobSpec = ComputeSpec::AsofJoin {
        spine: source.clone(),
        facts: source.clone(),
        spec: jammi_ai::pipeline::asof::AsofJoinSpecBuilder::new(
            jammi_ai::pipeline::asof::AsofKey {
                by: vec!["id".into()],
                time: "id".into(),
            },
            jammi_ai::pipeline::asof::AsofKey {
                by: vec!["id".into()],
                time: "id".into(),
            },
        )
        .build(),
    }
    .into();
    let handle = session.enqueue(asof, 0).await.unwrap();
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.model_source, None);
}

/// unit #485 (round-2 adversarial BLOCK F1): before this unit, a training
/// kind's branch of `JobWorker::run_claimed_job` threaded only the
/// lease-lost flag into `run_spec` — `jobs.cancel_requested` was never read
/// anywhere on that branch, so `CancelJob`/`JobHandle::cancel` on a real
/// training job was recorded on the row and then silently ignored: the run
/// trained to completion regardless. This drives a REAL, tiny LoRA fine-tune
/// (few epochs is not enough to guarantee the run is still in flight when
/// the cancel lands — `epochs` is deliberately large, mirroring
/// `fine_tune.rs`'s `cancelled_run_reclaims_epoch_checkpoints_that_actually_
/// existed`'s own reasoning for the same problem) through the real claimed-job
/// path, requests a cancel once the claim has landed, and asserts the row
/// reaches `failed` with `JammiError::JobCancelled`'s message within the
/// worker's own heartbeat cadence — never `completed`, and never silently
/// still `running`.
#[tokio::test(flavor = "multi_thread")]
async fn a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    // A short heartbeat (1s, with a real >2x lease margin) so the watcher
    // this unit adds polls `cancel_requested` promptly — the SAME cadence
    // the lease keeper itself renews at, per `spawn_cancel_request_watcher`'s
    // doc.
    config.lease = jammi_db::config::LeaseConfig {
        duration_secs: 30,
        heartbeat_secs: 1,
    };
    config.worker = jammi_db::config::WorkerConfig {
        idle_poll_secs: 1,
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let handle = session
        .enqueue(
            TrainingSpec::FineTune {
                source: "training".to_string(),
                columns: vec![
                    "text_a".to_string(),
                    "text_b".to_string(),
                    "score".to_string(),
                ],
                method: FineTuneMethod::Lora,
                task: ModelTask::TextEmbedding,
                common: TrainingCommon {
                    base_model: tiny_bert_model(),
                    config: FineTuneConfig {
                        // Deliberately large, matching `fine_tune.rs`'s own
                        // lease-loss test: a tiny real epoch is fast enough
                        // (single-digit milliseconds) that the count must be
                        // big enough the run is CERTAINLY still training
                        // when the cancel request lands below.
                        epochs: 20_000,
                        batch_size: 8,
                        lora_rank: 4,
                        warmup_steps: 0,
                        ..Default::default()
                    },
                },
            }
            .into(),
            0,
        )
        .await
        .unwrap();

    let worker = JobWorker::new(&session).expect("this config clears the worker margin");
    let claimed = session
        .catalog()
        .claim_next(worker.worker_id(), &["fine_tune"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    // Drive the claimed job concurrently, not awaited inline, so the
    // spawned cancel-request watcher's own poll tick can actually run while
    // training is in progress — a synchronous "claim, then run to
    // completion" drive would never leave a window to request a cancel
    // before the run finishes on its own.
    let session_for_task = Arc::clone(&session);
    let run = tokio::spawn(async move {
        worker.run_claimed_job(&session_for_task, claimed).await;
    });

    // Sanity gate: the run must still be in flight when the cancel lands, or
    // this test cannot distinguish "the fix works" from "the run happened to
    // finish and complete anyway" — a large `epochs` makes this vanishingly
    // unlikely; if it fires, the fix is to raise `epochs` further, never to
    // delete the gate.
    assert!(
        !run.is_finished(),
        "the spawned run_claimed_job task already completed before the test could request a \
         cancel -- raise `epochs` further so this genuinely races a live training run"
    );

    assert!(
        session
            .catalog()
            .cancel_request(&handle.job_id)
            .await
            .unwrap(),
        "a cancel on the running training job is recorded"
    );

    tokio::time::timeout(Duration::from_secs(15), run)
        .await
        .expect("the cancel-request watcher's next poll tick observes the request promptly")
        .unwrap();

    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(
        row.status,
        JobStatus::Failed.to_string(),
        "the cancelled training job must land `failed`, not stay `running` or reach `completed`"
    );
    assert!(
        row.error
            .as_deref()
            .unwrap_or("")
            .contains("cancelled at the executor's request checkpoint"),
        "the row must record JammiError::JobCancelled's message, got {:?}",
        row.error
    );
    assert!(
        row.cancel_requested,
        "the request stays recorded on the row"
    );
}

/// #485 BLOCK B1 (adversarial round 3): a bare `JoinHandle` for the
/// cancel-request watcher only DETACHES its task when dropped — it does not
/// stop it — so `EmbeddedWorker::drop` aborting the loop task while a
/// training job's `run_claimed_job` future is still in flight used to leak
/// the watcher forever, polling `catalog.get_job` on an `Arc<Catalog>`
/// clone that can outlive `Catalog::close`.
///
/// This reproduces exactly that action — abort the task holding
/// `run_claimed_job`'s future while it is still running, dropping the
/// future without any of its own code (the explicit
/// `drop(hold); drop(cancel_watcher);` included) ever running again — at a
/// checkpoint chosen so the reproduction is deterministic rather than a
/// wall-clock race: `training_test_hooks::arm_pause_before_spawn_blocking`
/// parks the run right after the lease hold and cancel-request watcher are
/// both live, and BEFORE any training thread (or its own `Arc<Catalog>`
/// clone) exists, so the only two holders of the per-attempt catalog handle
/// at that point are `run_claimed_job`'s own local and the watcher's clone.
///
/// Asserts the watcher (behind `CancelWatcherGuard`'s abort-on-drop) is
/// finished within about one poll interval of the abort, and that the
/// per-attempt catalog handle's strong count returns all the way to zero —
/// its pre-attempt value, since nothing outside this attempt ever held a
/// clone of it. Before the fix, the watcher's own clone pins that count at
/// (at least) one forever and `AbortHandle::is_finished()` never becomes
/// `true`.
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_run_claimed_jobs_future_leaves_no_leaked_cancel_watcher_or_catalog_handle() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    // Short heartbeat (matching the sibling cancel test above): bounds how
    // long the watcher-finished poll below needs to wait even if the fix
    // relied on the watcher's own next tick rather than `abort()` alone.
    config.lease = jammi_db::config::LeaseConfig {
        duration_secs: 30,
        heartbeat_secs: 1,
    };
    config.worker = jammi_db::config::WorkerConfig {
        idle_poll_secs: 1,
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let handle = session
        .enqueue(
            TrainingSpec::FineTune {
                source: "training".to_string(),
                columns: vec![
                    "text_a".to_string(),
                    "text_b".to_string(),
                    "score".to_string(),
                ],
                method: FineTuneMethod::Lora,
                task: ModelTask::TextEmbedding,
                common: TrainingCommon {
                    base_model: tiny_bert_model(),
                    config: FineTuneConfig {
                        epochs: 1,
                        batch_size: 8,
                        lora_rank: 4,
                        warmup_steps: 0,
                        ..Default::default()
                    },
                },
            }
            .into(),
            0,
        )
        .await
        .unwrap();
    let job_id = handle.job_id.clone();

    let worker = JobWorker::new(&session).expect("this config clears the worker margin");
    let claimed = session
        .catalog()
        .claim_next(worker.worker_id(), &["fine_tune"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    // Arm the one-shot pause BEFORE the run starts, so it is guaranteed to
    // be the one this specific attempt takes.
    let parked = training_test_hooks::arm_pause_before_spawn_blocking();

    let session_for_task = Arc::clone(&session);
    let run = tokio::spawn(async move {
        worker.run_claimed_job(&session_for_task, claimed).await;
    });

    // Wait for the run to actually reach the checkpoint — deterministic,
    // not a wall-clock race: by construction, the lease hold and
    // cancel-request watcher are live and no training thread exists yet.
    tokio::time::timeout(Duration::from_secs(30), parked)
        .await
        .expect("run_claimed_job never reached the pre-spawn_blocking checkpoint")
        .expect("the checkpoint's arrival sender was dropped without firing");

    assert!(
        !run.is_finished(),
        "the run must still be parked at the checkpoint, not finished"
    );
    assert_eq!(
        training_test_hooks::catalog_strong_count(&job_id),
        Some(2),
        "at the checkpoint the only holders of the per-attempt catalog handle must be \
         `run_claimed_job`'s own local and the cancel-request watcher's clone"
    );

    // Reproduce `EmbeddedWorker::drop`'s exact action: abort the task that
    // owns `run_claimed_job`'s future while it is parked mid-run.
    run.abort();
    let joined = run.await;
    assert!(
        joined.unwrap_err().is_cancelled(),
        "the task must have been cancelled by the abort, not have panicked or completed"
    );

    // The watcher must be gone within about one poll interval of the abort
    // — bounded and polled, never a fixed sleep.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if training_test_hooks::watcher_is_finished(&job_id) == Some(true) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the cancel-request watcher was not aborted within 5s of its owning future being \
             dropped -- BLOCK B1: `CancelWatcherGuard::drop` must abort it unconditionally"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // And the per-attempt catalog handle's strong count must return all the
    // way to zero — the pre-attempt value, since nothing outside this
    // attempt ever held one; a leaked watcher would hold it at (at least)
    // one forever.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if training_test_hooks::catalog_strong_count(&job_id) == Some(0) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the per-attempt `Arc<Catalog>` never returned to its pre-attempt strong count of \
             zero -- something is still holding a clone past the dropped future's teardown"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// unit #485 (round-3 fix-verifier gap): `run_claimed_job`'s shared `cancel`
/// flag has two writers — the lease keeper's own renewal (a genuine lease
/// loss) and `spawn_cancel_request_watcher` observing `jobs.cancel_requested`
/// (an operator's `CancelJob`) — and the `Cancelled` arm tells them apart
/// with `cancel_requested_seen` so a lease loss is left `running` for reclaim
/// while a real cancel request lands `failed` with
/// [`JammiError::JobCancelled`]'s message. `fine_tune.rs`'s
/// `worker_that_lost_lease_does_not_finalize` and
/// `cancelled_run_reclaims_epoch_checkpoints_that_actually_existed` both drive
/// a lease loss by letting a SECOND worker reclaim (and re-claim) the row
/// before the flag trips — so by the time the first worker's stale attempt
/// reaches ANY terminal write, `record_failed`'s own ownership CAS (identical
/// guard columns to the reclaim it raced) already fails on `claimed_by`/
/// `attempts` alone. Neither test can tell "the lease-lost arm correctly
/// skipped `record_failed`" apart from "a broken arm called `record_failed`
/// but its CAS silently no-op'd anyway" — both produce the exact same
/// observable row. This test closes that gap: NO second worker or reclaim
/// ever touches the row, so a version of `run_claimed_job` that (wrongly)
/// treated every `cancel` trip as a cancel request would reach
/// `record_failed`'s CAS with `claimed_by`/`status`/`attempts` still fully
/// intact — the CAS would MATCH and the row WOULD land `failed` with the
/// cancel message. Only the correct branch logic (reading
/// `cancel_requested_seen`, which stays `false` here) leaves the row
/// untouched.
///
/// The lease loss itself is manufactured with
/// [`jammi_db::catalog::lease_keeper::LeaseKeeper::kill_thread_for_test`] (the
/// keeper's own belt-and-braces death hook, forwarded through this crate's
/// `test-hooks` feature): killing the ONE dedicated keeper thread flips the
/// job's lease hold's `lost` flag via `ExitGuard` alone, with zero writes to
/// the `jobs` row — `claimed_by`, `status`, and `attempts` stay exactly what
/// they were the moment this worker claimed the job, i.e. "no reclaim has
/// happened" by construction, not by a race that might or might not resolve
/// that way. No DB-level trick (forcing `lease_expires_at` stale, bumping
/// `attempts`) can substitute: the keeper's own renewal and `record_failed`'s
/// CAS share the identical `claimed_by`/`status`/`attempts` guard, so any
/// mutation that trips the renewal miss would ALSO block `record_failed`'s
/// CAS — reproducing the exact masking this test exists to avoid.
#[tokio::test(flavor = "multi_thread")]
async fn a_lease_loss_on_the_owning_worker_lands_the_lease_lost_outcome_never_the_cancel_message() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    // Same minimum-legal heartbeat/lease margin `cancelled_run_reclaims_epoch_
    // checkpoints_that_actually_existed` uses: fast enough that the keeper's
    // death is observed (and the training loop's epoch-boundary check bails)
    // within a couple of seconds, never a wall-clock gamble.
    config.lease = jammi_db::config::LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    };
    config.worker = jammi_db::config::WorkerConfig {
        idle_poll_secs: 1,
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let handle = session
        .enqueue(
            TrainingSpec::FineTune {
                source: "training".to_string(),
                columns: vec![
                    "text_a".to_string(),
                    "text_b".to_string(),
                    "score".to_string(),
                ],
                method: FineTuneMethod::Lora,
                task: ModelTask::TextEmbedding,
                common: TrainingCommon {
                    base_model: tiny_bert_model(),
                    config: FineTuneConfig {
                        // Deliberately large, matching every other lease-loss
                        // drive in this suite: a tiny real epoch is fast
                        // enough (single-digit milliseconds) that the count
                        // must be big enough the run is CERTAINLY still
                        // training when the keeper thread is killed below.
                        epochs: 20_000,
                        batch_size: 8,
                        lora_rank: 4,
                        warmup_steps: 0,
                        ..Default::default()
                    },
                },
            }
            .into(),
            0,
        )
        .await
        .unwrap();

    let worker = JobWorker::new(&session).expect("this config clears the worker margin");
    let claimed = session
        .catalog()
        .claim_next(worker.worker_id(), &["fine_tune"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");
    let worker_id = worker.worker_id().to_string();

    // Drive the claimed job concurrently so the real lease keeper thread (and
    // the training loop's own epoch-boundary checks) can actually run while
    // training is in progress.
    let session_for_task = Arc::clone(&session);
    let run = tokio::spawn(async move {
        worker.run_claimed_job(&session_for_task, claimed).await;
    });

    // Sanity gate, mirroring every other lease-loss drive in this suite: the
    // run must still be in flight, or this test cannot distinguish "the fix
    // works" from "the run happened to finish on its own first".
    assert!(
        !run.is_finished(),
        "the spawned run_claimed_job task already completed before the test could kill the \
         lease keeper -- raise `epochs` further so this genuinely races a live training run"
    );

    // Kill the ONE dedicated keeper thread this session's every lease hold
    // renews on. No second worker, no `reclaim_expired_jobs`, no direct SQL
    // write to `jobs` — the row is never touched by this step at all.
    session.lease_keeper().kill_thread_for_test();

    tokio::time::timeout(Duration::from_secs(15), run)
        .await
        .expect(
            "the training loop's next epoch-boundary check must observe the keeper's death \
             promptly and bail",
        )
        .unwrap();

    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(
        row.status,
        JobStatus::Running.to_string(),
        "a lease loss on the owning worker must leave the job `running` for reclaim, never \
         `failed` or `completed`"
    );
    assert_eq!(
        row.claimed_by.as_deref(),
        Some(worker_id.as_str()),
        "no reclaim ever happened -- the row is still claimed by the SAME worker whose lease \
         died"
    );
    assert!(
        row.error.is_none(),
        "the lease-lost arm must never record a terminal error, got {:?}",
        row.error
    );
    assert!(
        row.result.is_none(),
        "the lease-lost arm must never record a terminal result"
    );
    assert!(
        !row.cancel_requested,
        "no cancel was ever requested on this row"
    );
}
