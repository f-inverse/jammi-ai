//! The cancel checkpoints, `run_now`'s finish compare-and-set, and
//! `enqueue`'s model-link derivation — each driven through the real
//! executor path with a compute run parked at a documented checkpoint
//! (`jammi_ai::jobs::compute_test_hooks`), never by re-implementing the
//! algorithm in the test.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::training_job::fine_tuned_model_id;
use jammi_ai::fine_tune::worker::{EmbeddedWorker, JobWorker};
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
