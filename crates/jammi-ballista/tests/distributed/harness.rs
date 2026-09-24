//! The three(+)-process Ballista lane's harness: the fleets it spawns
//! (`jammi_test_utils::fleet`, the one facility every fleet on one host is
//! launched through) and the submission and observation helpers around them
//! (`add_training_source`, `submit_fine_tune`, `JobSize`, `await_job`,
//! `unique_source_name`, `training_pairs_url`, `tiny_bert_model`,
//! `label_of`).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::session::InferenceSession;
use jammi_datafusion::ModelTask;
use jammi_db::config::{
    CatalogConfig, DistributedConfig, InferenceConfig, JammiConfig, LeaseConfig, StorageConfig,
    WorkerConfig,
};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_test_utils::DistributedBackends;
use tempfile::TempDir;

use jammi_test_utils::fleet::{jammi_server_binary, MAX_WORLD_SIZE};
pub use jammi_test_utils::fleet::{BallistaRole, Fleet, ProcSpec, WorkerRole};

const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;

/// This lane's fleets: the workspace's own `jammi-server` build, the
/// committed shape-d configs under the workspace root.
pub fn spawn_fleet(
    backends: &DistributedBackends,
    result_root: &str,
    specs: Vec<ProcSpec>,
) -> Fleet {
    Fleet::spawn(
        &jammi_server_binary(),
        &jammi_test_utils::workspace_root(),
        backends,
        result_root,
        specs,
    )
}

/// The generous terminal-state timeout — cold boot + Postgres connect +
/// migrate + a tiny CPU LoRA fine-tune + publish to the S3 store + finalize, under a
/// 3s lease with reclaim on a crash.
pub const TERMINAL_TIMEOUT: Duration = Duration::from_secs(150);
pub const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Poll the shared catalog for `job_id` until `want(&record)` holds. Fails
/// loudly (fleet diagnostics + final job row) on an early unexpected worker
/// exit or the timeout, never silently waiting it out. Also fails loudly the
/// moment the row reaches a TERMINAL status (`JobRecord::is_terminal`, the
/// ONE terminality predicate) `want` does not accept: a terminal
/// row never mutates further, so a fixture polling for one specific literal
/// status fails immediately naming the status it actually settled on,
/// rather than burning the rest of [`TERMINAL_TIMEOUT`] on a DIFFERENT
/// terminal status instead.
pub async fn await_job(
    fleet: &mut Fleet,
    session: &Arc<InferenceSession>,
    job_id: &str,
    label: &str,
    mut want: impl FnMut(&jammi_db::catalog::jobs_repo::JobRecord) -> bool,
) -> jammi_db::catalog::jobs_repo::JobRecord {
    let catalog = session.catalog().pinned_to_tenant(None);
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        if let Ok(record) = catalog.get_job(job_id).await {
            if want(&record) {
                return record;
            }
            if record.is_terminal() {
                fleet.dump_diagnostics(&format!(
                    "job {job_id} settled on status={:?} error={:?}, which does not satisfy: \
                     {label}",
                    record.status, record.error
                ));
                panic!(
                    "distributed ballista lane: job {job_id} reached a TERMINAL status ({:?}) \
                     that does not satisfy: {label} (error: {:?}) — a terminal row never \
                     mutates further, so waiting out the remaining timeout cannot help.",
                    record.status, record.error
                );
            }
        }
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!(
                "worker {worker} exited unexpectedly ({status}) while awaiting: {label}"
            ));
            panic!(
                "distributed ballista lane: worker {worker} exited unexpectedly ({status}) \
                 before the fleet could satisfy: {label}."
            );
        }
        if Instant::now() >= deadline {
            fleet.dump_diagnostics(&format!("timed out after {TERMINAL_TIMEOUT:?}: {label}"));
            panic!("distributed ballista lane: timed out after {TERMINAL_TIMEOUT:?}: {label}.");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Poll the captured log of the worker labelled `label` until it contains
/// `needle`, returning the log. A log line a worker writes AFTER the catalog
/// fact a test already awaited (the placed submitter's `HandedOff` line
/// lands only once the gang's result stream has drained, which is after the
/// coordinator committed `completed`) is never a single read: CI run
/// 35127543679 read the submitter's log 240 ms after the job finished and
/// found the line absent. Fails loudly (fleet diagnostics) on an early
/// unexpected worker exit or the timeout.
pub async fn await_log_contains(
    fleet: &mut Fleet,
    label: &str,
    needle: &str,
    what: &str,
) -> String {
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        let log = fleet.log_contents(label);
        if log.contains(needle) {
            return log;
        }
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!(
                "worker {worker} exited unexpectedly ({status}) while awaiting the log line: {what}"
            ));
            panic!(
                "distributed ballista lane: worker {worker} exited unexpectedly ({status}) \
                 before {label}'s log showed: {what}."
            );
        }
        if Instant::now() >= deadline {
            fleet.dump_diagnostics(&format!(
                "timed out after {TERMINAL_TIMEOUT:?} awaiting the log line: {what}"
            ));
            panic!(
                "distributed ballista lane: timed out after {TERMINAL_TIMEOUT:?}: {label}'s log \
                 never showed: {what}; log:\n{log}"
            );
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Poll for a plain condition over the catalog with no `Fleet` diagnostics
/// (used where no fleet failure is expected to interrupt the wait, e.g. a
/// `submit_physical_plan` result already resolved).
pub async fn await_condition(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Build the harness's own observer session against the shared Postgres +
/// the S3 store, rooted at `result_root`. `[worker] enabled = false`: it only
/// submits and observes.
pub async fn harness_session(
    backends: &DistributedBackends,
    result_root: &str,
) -> (Arc<InferenceSession>, TempDir) {
    harness_session_with(backends, result_root, InferenceConfig::default()).await
}

/// [`harness_session`] planning its inference with `inference` — the fan-out
/// and chunk size a submitted plan carries to the executors.
pub async fn harness_session_with(
    backends: &DistributedBackends,
    result_root: &str,
    inference: InferenceConfig,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().expect("harness artifact_dir");
    let config = JammiConfig {
        inference,
        artifact_dir: dir.path().to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
            device: Some(-1),
            ..Default::default()
        },
        catalog: CatalogConfig::Postgres {
            url: backends.pg_url.clone().into(),
            pool_size: 8,
            max_lifetime_secs: None,
        },
        storage: StorageConfig {
            result_root: Some(result_root.to_string()),
            cloud: Some(backends.cloud()),
        },
        lease: LeaseConfig {
            duration_secs: LEASE_SECS,
            heartbeat_secs: HEARTBEAT_SECS,
        },
        worker: WorkerConfig {
            enabled: false,
            idle_poll_secs: IDLE_POLL_SECS,
            ..Default::default()
        },
        distributed: DistributedConfig {
            max_world_size: MAX_WORLD_SIZE,
        },
        ..Default::default()
    };
    let session = InferenceSession::open(config)
        .await
        .expect("harness session connects to shared Postgres + the S3 store");
    (session, dir)
}

/// The `JAMMI_WORKER_ID` label of the fleet member whose minted instance id
/// is `instance_id`.
pub async fn label_of(session: &InferenceSession, instance_id: &str) -> String {
    let workers = session.catalog().list_workers().await.unwrap();
    workers
        .iter()
        .find(|w| w.instance_id == instance_id)
        .and_then(|w| w.label.clone())
        .unwrap_or_else(|| {
            panic!(
                "claimed_by {instance_id:?} is not a labelled fleet member; workers = {workers:?}"
            )
        })
}

pub fn training_pairs_url() -> String {
    jammi_test_utils::fixture_url("training_pairs.csv")
}

pub fn tiny_bert_model() -> String {
    format!(
        "local:{}",
        jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .expect("utf8 tiny_bert path")
    )
}

pub fn unique_source_name(role: &str) -> String {
    format!("{role}-{}", uuid::Uuid::new_v4().simple())
}

pub async fn add_training_source(session: &Arc<InferenceSession>, name: &str) {
    session
        .add_source(
            name,
            SourceType::File,
            SourceConnection {
                url: Some(training_pairs_url()),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_else(|e| panic!("register training source {name}: {e}"));
}

#[derive(Clone, Copy)]
pub enum JobSize {
    Quick,
    Crashable,
}

impl JobSize {
    fn epochs(self) -> usize {
        match self {
            JobSize::Quick => 3,
            // The placed path (a working Ballista fleet, unlike a plain
            // in-process claim) resolves the placement round-trip almost
            // instantly on localhost — 60 epochs of `tiny_bert` LoRA
            // complete inside the harness's own detect+kill window (the
            // kill test then reads a single-entry `claimed_by` sequence:
            // the job finished before the kill landed). The kill
            // lands ~1 s after the placed claim, so the job must outlive
            // that; the SUCCESSOR then runs the whole job again from
            // scratch and must finish inside the reclaim wait. Measured
            // rates: ~15 epochs/s locally, ~5 epochs/s on a CI runner
            // (900 epochs reach about epoch 734 at the 150 s deadline).
            // 450 epochs keeps the job crashable (≥ 30 s at
            // the fast rate) and completes in ~90 s at the slow one.
            JobSize::Crashable => 450,
        }
    }
}

fn lane_fine_tune_config(size: JobSize) -> FineTuneConfig {
    FineTuneConfig {
        epochs: size.epochs(),
        batch_size: 8,
        lora_rank: 4,
        warmup_steps: 0,
        ..Default::default()
    }
}

/// Submit one durable `world_size`-rank LoRA fine-tune over `source`.
/// Returns `(job_id, output_model_id)`.
pub async fn submit_fine_tune(
    session: &Arc<InferenceSession>,
    source: &str,
    size: JobSize,
    world_size: u32,
) -> (String, String) {
    let job = session
        .run_training_spec(TrainingSpec::FineTune {
            source: source.to_string(),
            columns: vec![
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: tiny_bert_model(),
                config: lane_fine_tune_config(size),
                world_size,
                cache: CachePolicy::Bypass,
            },
        })
        .await
        .expect("submit a queued gang fine-tune job to the shared catalog");
    (job.job_id.clone(), job.model_id().to_string())
}

/// Register a synthetic episodic meta-dataset as `source` — its source
/// parquet under `dir`, its embedding table under the harness's result root
/// — so every fleet member reads both through the shared catalog.
pub async fn add_episodes_source(session: &Arc<InferenceSession>, dir: &Path, source: &str) {
    use jammi_test_utils::meta_dataset;
    let rows = meta_dataset::linear_tasks(8, 18, 321);
    session
        .add_source(
            source,
            SourceType::File,
            meta_dataset::write_source(dir, &rows),
        )
        .await
        .unwrap_or_else(|e| panic!("register episodes source {source}: {e}"));
    meta_dataset::materialize_embeddings(&session.result_store(), session.context(), source, &rows)
        .await;
}

/// Submit one durable context-predictor job over `source`, registering its
/// predictor as `model_id`. Returns the job id.
pub async fn submit_context_predictor(
    session: &Arc<InferenceSession>,
    source: &str,
    model_id: &str,
) -> String {
    use jammi_ai::pipeline::context_predictor::{
        ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
    };
    let spec = ContextPredictorTrainConfig {
        model_id: model_id.to_string(),
        architecture: ContextArchitecture::Cnp,
        key_column: "_row_id".to_string(),
        task_column: "task".to_string(),
        value_column: "y".to_string(),
        context_k: 6,
        hidden_dim: 16,
        num_heads: 2,
        num_layers: 2,
        head: PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
        epochs: 8,
        learning_rate: 0.005,
        grad_clip: 1.0,
        test_task_fraction: 0.25,
        min_task_count: 4,
        seed: 7,
    };
    session
        .train_context_predictor(source, &spec)
        .await
        .expect("submit a queued context-predictor job to the shared catalog")
        .job_id
        .clone()
}

/// A 2-file parquet directory source, disjoint keys — the SAME shape
/// `crates/jammi-ai/tests/it/pipeline.rs::write_two_file_source` uses for
/// its own `build_embedding_plan` oracle, ported here so the scan stage has
/// `partition_count >= 2`.
pub fn write_two_file_source(dir: &Path) -> String {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let src_dir = dir.join("two_files");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    // 4 rows/file, disjoint id ranges. DataFusion's file-group builder
    // coalesces files below `datafusion.optimizer.repartition_file_min_size`
    // (10 MiB default) into ONE partition regardless of `target_partitions`
    // — every caller that needs `partition_count >= 2` out of this small a
    // source must first lower that threshold on its OWN
    // session (`SET datafusion.optimizer.repartition_file_min_size = 1`),
    // never inflate file size to work around it.
    const ROWS_PER_FILE: i64 = 4;
    for f in 0..2i64 {
        let base = f * ROWS_PER_FILE;
        let ids: Vec<i64> = (base..base + ROWS_PER_FILE).collect();
        let texts: Vec<String> = (0..ROWS_PER_FILE)
            .map(|i| format!("file {f} row {i} about topic {}", (base + i) % 37))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(src_dir.join(format!("part{f}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    format!("file://{}", src_dir.display())
}

/// A one-file parquet source of `rows` keyed rows, each text a distinct
/// in-vocabulary sequence of the tiny encoder
/// (`jammi_test_utils::tiny_vocab_text`) — the shape a test reaches for
/// when a table's WRITE must take a while (its sink's index build grows
/// with the row count) while each row's inference stays cheap.
pub fn write_many_row_source(dir: &Path, rows: usize) -> String {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let src_dir = dir.join("many_rows");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let ids: Vec<i64> = (0..rows as i64).collect();
    let texts: Vec<String> = (0..rows)
        .map(|i| jammi_test_utils::tiny_vocab_text('r', i))
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(src_dir.join("part0.parquet")).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    format!("file://{}", src_dir.display())
}
