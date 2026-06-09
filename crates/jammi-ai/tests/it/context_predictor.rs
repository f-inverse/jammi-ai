//! S19 episodic meta-training for the `AnyContextPredictor` family.
//!
//! Hermetic, CPU, synthetic meta-datasets. A meta-dataset is a family of linear
//! functions: each **task** `t` carries a weight vector `w_t`, every row's
//! outcome is `y = w_t · x` over a small feature vector `x`, and the embedding
//! table stores `x` as the row's vector. Same-task neighbours (retrieved by
//! x-similarity) form a context that *determines* `w_t`, so a predictor that
//! meta-learns "infer the task's map from its context, apply it at the target"
//! generalises to a held-out task in one forward pass — the NP success bar.
//!
//! These tests exercise the real engine path: the episodic sampler reads
//! per-member x-vectors and y-labels through the generic SQL surface, the
//! context is leakage-scoped (`exclude_self` + same-task split), tasks (not
//! points) are partitioned into train/test, and training drives the generalized
//! `train_loop` with the S18 proper-scoring objective.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tempfile::TempDir;

use jammi_ai::pipeline::context_predictor::{
    ConformalLevers, ContextPredictorTrainConfig, ContextServeOptions, EpisodeBatch,
    GaussianObjective, PredictiveHead, SampledEpisodes,
};
use jammi_ai::pipeline::context_set::ContextSourceKind;
use jammi_ai::pipeline::parallel_train::{train_loop, ParallelTrainConfig};
use jammi_ai::session::InferenceSession;
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_encoders::{AnyContextPredictor, ContextArchitecture, ContextPredictorConfig};
use parquet::arrow::ArrowWriter;

use crate::common;

const FEATURE_DIM: usize = 4;

/// splitmix64 — a deterministic generator so the synthetic meta-dataset is
/// reproducible without pulling a test-only rng dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [-1, 1).
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// One synthetic row: its key (`_row_id`, the same identity the embedding table
/// keys its vector by), the task it belongs to, its feature vector `x`, and its
/// outcome `y = w_task · x`.
struct Row {
    id: String,
    task: String,
    x: Vec<f32>,
    y: f64,
}

/// Build a linear-function meta-dataset: `n_tasks` tasks, each with a random
/// weight vector and `rows_per_task` rows. Returns the rows in (task-major)
/// order.
fn synthetic_meta_dataset(n_tasks: usize, rows_per_task: usize, seed: u64) -> Vec<Row> {
    let mut rng = Rng(seed);
    let mut rows = Vec::with_capacity(n_tasks * rows_per_task);
    for t in 0..n_tasks {
        let w: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
        for r in 0..rows_per_task {
            let x: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
            let y: f64 = x.iter().zip(&w).map(|(xi, wi)| (xi * wi) as f64).sum();
            rows.push(Row {
                id: format!("t{t}_r{r}"),
                task: format!("task_{t}"),
                x,
                y,
            });
        }
    }
    rows
}

/// Stand up a session over a synthetic meta-dataset: a source parquet carrying
/// `(id, task, y)` plus a hand-written embedding result table whose `vector`
/// column is the row's feature `x`, keyed by `id`. The embedding table is
/// registered + marked ready so `assemble_context` / vector search resolve it.
async fn session_with_meta_dataset(rows: &[Row]) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.register_query_functions();

    // Source parquet: `_row_id` (the key, shared with the embedding table's
    // identity), `task`, `y`. The split predicate scopes the context over this
    // source on the embedding table's key column, so the key column is a real
    // source column — naming it `_row_id` shares one identity end to end.
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let tasks: Vec<&str> = rows.iter().map(|r| r.task.as_str()).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(tasks)),
            Arc::new(Float64Array::from(ys)),
        ],
    )
    .unwrap();
    let source_path = dir.path().join("source.parquet");
    {
        let file = std::fs::File::create(&source_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&source_schema), None).unwrap();
        writer.write(&source_batch).unwrap();
        writer.close().unwrap();
    }
    session
        .add_source(
            "fns",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", source_path.to_str().unwrap())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Embedding result table keyed by `_row_id`, vector = the row's feature `x`.
    // Materialised through the engine's own embedding-table writer so it gets a
    // real sidecar ANN index — the same search path production uses, keyed
    // `_row_id` (which equals the source key column).
    let pairs: Vec<(String, Vec<f32>)> = rows.iter().map(|r| (r.id.clone(), r.x.clone())).collect();
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            "fns",
            "synthetic-embed",
            None,
            &pairs,
            FEATURE_DIM,
        )
        .await
        .unwrap();

    (session, dir)
}

/// A spec over the synthetic dataset for the given architecture / objective.
fn spec(architecture: ContextArchitecture, head: PredictiveHead) -> ContextPredictorTrainConfig {
    ContextPredictorTrainConfig {
        model_id: "ctx-predictor".to_string(),
        architecture,
        key_column: "_row_id".to_string(),
        task_column: "task".to_string(),
        value_column: "y".to_string(),
        context_k: 6,
        hidden_dim: 16,
        num_heads: 2,
        num_layers: 2,
        head,
        epochs: 60,
        learning_rate: 0.005,
        grad_clip: 1.0,
        test_task_fraction: 0.25,
        min_task_count: 4,
        seed: 7,
    }
}

/// Build the predictor a spec selects in a fresh `VarMap` (the same mapping the
/// pipeline's trainer uses), randomised off zero so a fresh head is
/// non-degenerate. Returns the predictor and its varmap.
fn build(spec: &ContextPredictorTrainConfig, device: &Device) -> (AnyContextPredictor, VarMap) {
    let varmap = VarMap::new();
    let cfg = ContextPredictorConfig {
        architecture: spec.architecture,
        context_k: spec.context_k,
        feature_dim: FEATURE_DIM,
        value_dim: 1,
        hidden_dim: spec.hidden_dim,
        num_heads: spec.num_heads,
        num_layers: spec.num_layers,
        head_width: match &spec.head {
            PredictiveHead::Gaussian { .. } => 2,
            PredictiveHead::Quantile { levels } => levels.len(),
        },
    };
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let predictor = AnyContextPredictor::new(&cfg, vb).unwrap();
    // A fresh varmap is zero-initialised — randomise so the init head is real.
    {
        let data = varmap.data().lock().unwrap();
        let mut rng = Rng(99);
        for var in data.values() {
            let n: usize = var.shape().elem_count();
            let vals: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.1).collect();
            let t = Tensor::from_vec(vals, var.shape().clone(), device).unwrap();
            var.set(&t).unwrap();
        }
    }
    (predictor, varmap)
}

/// Mean proper-score over a set of episode batches under the spec's objective —
/// the held-out-task evaluation metric.
fn mean_loss(
    predictor: &AnyContextPredictor,
    spec: &ContextPredictorTrainConfig,
    episodes: &[EpisodeBatch],
) -> f64 {
    let mut total = 0.0;
    for batch in episodes {
        let preds = predictor.forward(&batch.episode).unwrap();
        let loss = spec.head.score(&preds, &batch.target_y).unwrap();
        total += loss.to_scalar::<f32>().unwrap() as f64;
    }
    total / episodes.len().max(1) as f64
}

/// The held-out-task generalisation bar, for CNP and the attentive AttnCNP:
/// after meta-training on the train tasks, the predictive score on **held-out
/// tasks** (tasks never seen in training) is better than at initialisation.
/// Drives the real sampler (SQL per-member reads, leakage scoping) end to end.
async fn held_out_task_score_improves(architecture: ContextArchitecture) {
    let rows = synthetic_meta_dataset(28, 16, 123);
    let (session, _dir) = session_with_meta_dataset(&rows).await;
    let device = Device::Cpu;

    let spec = spec(
        architecture,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );

    let SampledEpisodes { train, test, .. } =
        session.sample_context_episodes("fns", &spec).await.unwrap();
    assert!(!train.is_empty(), "train tasks sampled");
    assert!(!test.is_empty(), "held-out test tasks sampled");

    let (predictor, varmap) = build(&spec, &device);
    let init_test_loss = mean_loss(&predictor, &spec, &test);

    let config = ParallelTrainConfig {
        epochs: spec.epochs,
        learning_rate: spec.learning_rate,
        weight_decay: 0.0,
        grad_clip: spec.grad_clip,
    };
    train_loop(
        &varmap,
        &train,
        &config,
        &std::sync::atomic::AtomicBool::new(false),
        |batch: &EpisodeBatch| {
            predictor
                .forward(&batch.episode)
                .map_err(|e| jammi_db::error::JammiError::FineTune(format!("{e}")))
        },
        |preds, batch: &EpisodeBatch| spec.head.score(preds, &batch.target_y),
    )
    .unwrap();

    let trained_test_loss = mean_loss(&predictor, &spec, &test);
    assert!(
        trained_test_loss < init_test_loss,
        "{architecture:?}: held-out-task score did not improve \
         (init {init_test_loss}, trained {trained_test_loss})"
    );
}

#[tokio::test]
async fn cnp_held_out_task_log_likelihood_rises() {
    held_out_task_score_improves(ContextArchitecture::Cnp).await;
}

#[tokio::test]
async fn attncnp_held_out_task_log_likelihood_rises() {
    held_out_task_score_improves(ContextArchitecture::AttnCnp).await;
}

/// The [HIGH] context-leakage contract: a target's own key never appears in its
/// own assembled context, across every episode of every task. Asserted directly
/// on the sampler's leakage-scoped reads — `exclude_self` + the same-task split.
#[tokio::test]
async fn target_never_appears_in_its_own_context() {
    let rows = synthetic_meta_dataset(8, 20, 55);
    let (session, _dir) = session_with_meta_dataset(&rows).await;

    let spec = spec(
        ContextArchitecture::Cnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );

    // Re-walk the same retrieval the sampler does, target by target, asserting
    // the target's own key is absent from its context. The sampler's episodes
    // carry only tensors (keys are dropped), so the contract is checked on the
    // assemble_context surface the sampler is built on, with identical scoping.
    use jammi_ai::pipeline::context_set::{ContextRequest, SetAggregator};

    for task in ["task_0", "task_3", "task_7"] {
        let split = format!("arrow_cast(\"task\", 'Utf8') = '{task}'");
        // Every target row in the task. The target's own feature `x` is the
        // retrieval query (the same vector the embedding table stores for it).
        for row in rows.iter().filter(|r| r.task == task) {
            let mut request = ContextRequest::new("fns", row.x.clone(), spec.context_k);
            request.exclude_self = true;
            request.exclude_key = Some(row.id.clone());
            request.split = Some(split.clone());
            request.aggregator = SetAggregator::Mean;
            let rep = session.assemble_context(&request).await.unwrap();

            assert!(
                !rep.context_keys.contains(&row.id),
                "target '{}' leaked into its own context: {:?}",
                row.id,
                rep.context_keys
            );
            // Same-task split: every context member belongs to the same task.
            for key in &rep.context_keys {
                let member_task = rows
                    .iter()
                    .find(|r| &r.id == key)
                    .map(|r| r.task.as_str())
                    .unwrap();
                assert_eq!(
                    member_task, task,
                    "context member '{key}' came from a different task than target '{}'",
                    row.id
                );
            }
        }
    }
}

/// The [HIGH] meta-overfitting guard: a meta-dataset with too few distinct
/// tasks is rejected with a typed error rather than meta-trained into
/// memorisation.
#[tokio::test]
async fn too_few_tasks_is_rejected() {
    // Two tasks, but a min_task_count of 4 — below the guard.
    let rows = synthetic_meta_dataset(2, 16, 9);
    let (session, _dir) = session_with_meta_dataset(&rows).await;

    let spec = spec(
        ContextArchitecture::Cnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Nll { beta: 0.5 },
        },
    );

    let err = session
        .sample_context_episodes("fns", &spec)
        .await
        .expect_err("a 2-task meta-dataset must be rejected under min_task_count 4");
    let msg = format!("{err}");
    assert!(
        msg.contains("min_task_count") && msg.contains("distinct task"),
        "error should name the meta-overfitting guard, got: {msg}"
    );
}

/// End-to-end: `train_context_predictor` trains, persists the weights, and
/// registers the artifact as a `ModelTask::Regression` model — the PR2 success
/// bar (a catalogued trained artifact through the model path).
#[tokio::test(flavor = "multi_thread")]
async fn train_context_predictor_persists_a_catalogued_artifact() {
    let rows = synthetic_meta_dataset(8, 18, 321);
    let (session, _dir) = session_with_meta_dataset(&rows).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let spec = spec(
        ContextArchitecture::Cnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );

    // Submit + run through the worker, then read the catalogued model row.
    let job = session.train_context_predictor("fns", &spec).await.unwrap();
    assert_eq!(job.model_id(), "ctx-predictor");
    job.wait().await.unwrap();

    let record = session
        .catalog()
        .get_model("ctx-predictor")
        .await
        .unwrap()
        .expect("predictor registered in catalog");
    assert_eq!(record.model_id, "ctx-predictor");
    assert_eq!(record.task, ModelTask::Regression);
    let artifact = record.artifact_path.expect("artifact path catalogued");

    // The recorded `artifact_path` is the object-store prefix the worker
    // published the weights under. Fetch the bundle (an in-place read for the
    // default `file://` root) and confirm the weights reload as a real tensor
    // map — usable, not an empty file.
    let prefix_url = jammi_db::storage::StorageUrl::parse(&artifact).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("published predictor weights fetch and verify");
    let weights = local.dir().join("model.safetensors");
    assert!(
        weights.exists(),
        "trained predictor weights persisted at {weights:?}"
    );
    let loaded = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();
    assert!(!loaded.is_empty(), "persisted weight map is non-empty");
}

/// End-to-end over the **real embedding path**: vectors are produced by
/// `generate_text_embeddings` (the same call production uses), which registers
/// the embedding model in the catalog and records the result table under the
/// model's bare canonical name. `train_context_predictor` then resolves that
/// catalogued model and uses its PK as the job's `base_model_id` FK — the arm
/// the synthetic-table tests never reach, because a hand-materialised table has
/// no model row. A job whose `base_model_id` does not name a real `models` PK
/// is rejected by the `training_jobs.base_model_id` foreign key, so this drives
/// the submit + worker run to completion as the proof the FK resolves.
#[tokio::test(flavor = "multi_thread")]
async fn train_context_predictor_over_generated_embeddings() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.register_query_functions();

    // A meta-dataset shaped source: `_row_id` (key), `task`, `y`, and the `text`
    // the embedding model encodes. Each task gets distinct, repeated text so the
    // encoder produces same-task-clustered vectors, mirroring the real flow.
    let rows = synthetic_meta_dataset(8, 18, 321);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let tasks: Vec<&str> = rows.iter().map(|r| r.task.as_str()).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let texts: Vec<String> = rows
        .iter()
        .map(|r| format!("a description belonging to {}", r.task))
        .collect();
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(tasks)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let source_path = dir.path().join("source.parquet");
    {
        let file = std::fs::File::create(&source_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&source_schema), None).unwrap();
        writer.write(&source_batch).unwrap();
        writer.close().unwrap();
    }
    session
        .add_source(
            "fns",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", source_path.to_str().unwrap())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // The real embedding path: encodes `text` with the tiny fixture model,
    // auto-registers the embedding model row (PK `name::version`), and records
    // the result table under the model's bare canonical name.
    let model_id = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();
    session
        .generate_text_embeddings("fns", &model_id, &["text".to_string()], "_row_id")
        .await
        .unwrap();

    // The catalogued model's bare name differs from its PK, so the previously
    // mis-resolving `Some` arm would submit a job whose `base_model_id` is the
    // bare name and trip the `training_jobs.base_model_id` foreign key.
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let spec = spec(
        ContextArchitecture::Cnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );
    let job = session.train_context_predictor("fns", &spec).await.unwrap();
    job.wait().await.unwrap();

    let record = session
        .catalog()
        .get_model("ctx-predictor")
        .await
        .unwrap()
        .expect("predictor registered after a completed run");
    assert_eq!(record.task, ModelTask::Regression);
}

// ---------------------------------------------------------------------------
// PR3: serving, the conformal calibration gate, epistemic widening.
// ---------------------------------------------------------------------------

/// Replicate the module's deterministic train/test **task** split so a test can
/// name the held-out tasks the predictor never trained on (the same splitmix64
/// Fisher-Yates + trailing-fraction the sampler uses). `tasks` is the sorted
/// distinct task list (lexicographic, the `distinct_tasks` order).
fn held_out_tasks(mut tasks: Vec<String>, seed: u64, test_fraction: f64) -> Vec<String> {
    // splitmix64 Fisher-Yates, identical to the module's `deterministic_shuffle`.
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    };
    let n = tasks.len();
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        tasks.swap(i, j);
    }
    let n_test = (((n as f64) * test_fraction).ceil() as usize).clamp(1, n - 1);
    tasks.split_off(n - n_test)
}

/// Train a predictor through the pipeline, returning the catalogued model id.
///
/// `train_context_predictor` now submits a durable job; a worker runs it. The
/// helper starts a worker over the session, submits, waits for completion, and
/// returns the model id the predictor registered under (the spec's `model_id`).
async fn train(session: &Arc<InferenceSession>, spec: &ContextPredictorTrainConfig) -> String {
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(session)
        .expect("default worker intervals are valid");
    let job = session.train_context_predictor("fns", spec).await.unwrap();
    job.wait().await.unwrap();
    job.model_id().to_string()
}

/// Per-point coverage indicators of a conformal-wrapped predictor over a set of
/// `(target_key, observed_y)` calibration/test pairs — `true` where the observed
/// outcome lands inside the served interval. The marginal coverage guarantee is a
/// property of these indicators *pooled*; returning them (rather than only their
/// mean) lets a caller pool across runs into one low-variance estimate.
async fn conformal_indicators(
    session: &Arc<InferenceSession>,
    served: &jammi_ai::pipeline::context_predictor::ServedContextPredictor,
    wrap: &jammi_ai::pipeline::context_predictor::ConformalContextPredictor,
    targets: &[(String, f64)],
) -> Vec<bool> {
    let mut covered = Vec::with_capacity(targets.len());
    for (key, y) in targets {
        let dist = session
            .predict_with_context_predictor(served, key)
            .await
            .unwrap();
        let (lo, hi) = wrap.interval(&dist, None).unwrap();
        covered.push(*y >= lo && *y <= hi);
    }
    covered
}

/// The empirical coverage fraction of a set of indicators.
fn coverage_fraction(indicators: &[bool]) -> f64 {
    indicators.iter().filter(|c| **c).count() as f64 / indicators.len().max(1) as f64
}

/// Empirical interval coverage of a conformal-wrapped predictor over a set of
/// `(target_key, observed_y)` calibration/test pairs — the fraction of observed
/// outcomes that land inside the served interval.
async fn conformal_coverage(
    session: &Arc<InferenceSession>,
    served: &jammi_ai::pipeline::context_predictor::ServedContextPredictor,
    wrap: &jammi_ai::pipeline::context_predictor::ConformalContextPredictor,
    targets: &[(String, f64)],
) -> f64 {
    coverage_fraction(&conformal_indicators(session, served, wrap, targets).await)
}

/// No-retrain adaptation: a `predict` is one forward pass with **zero** gradient
/// updates. Asserted at the strongest hermetic level — the served predictor's
/// trainable tensors are byte-identical before and after a `predict`, and a
/// repeated predict on the same target is byte-identical (deterministic), so the
/// in-context adaptation lives entirely in the forward, never in a weight update.
#[tokio::test(flavor = "multi_thread")]
async fn predict_is_inference_only_no_gradient_updates() {
    let rows = synthetic_meta_dataset(12, 16, 4242);
    let (session, _dir) = session_with_meta_dataset(&rows).await;

    let spec = spec(
        ContextArchitecture::AttnCnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );
    let model_id = train(&session, &spec).await;

    let served = session
        .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
        .await
        .unwrap();

    // The served predictor's weights, before any predict — fetched from the
    // artifact store under the recorded prefix.
    let artifact = session
        .catalog()
        .get_model(&model_id)
        .await
        .unwrap()
        .unwrap()
        .artifact_path
        .unwrap();
    let prefix_url = jammi_db::storage::StorageUrl::parse(&artifact).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .unwrap();
    let weights = local.dir().join("model.safetensors");
    let before = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();

    let target = &rows[0].id;
    let first = session
        .predict_with_context_predictor(&served, target)
        .await
        .unwrap();
    let second = session
        .predict_with_context_predictor(&served, target)
        .await
        .unwrap();

    // Inference is deterministic — the same target, the same served weights, the
    // same forward — so two predicts are byte-identical. A gradient update mid
    // predict would perturb the second.
    assert_eq!(
        first, second,
        "predict is not deterministic — a weight mutated"
    );

    // The on-disk weights are untouched — inference never writes the varmap back.
    let after = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();
    assert_eq!(before.len(), after.len());
    for (name, t0) in &before {
        let t1 = after.get(name).expect("a weight vanished after predict");
        let v0 = t0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v1 = t1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v0, v1, "weight '{name}' changed across a predict");
    }
}

/// The R2 calibration gate, ≥3 seeds: on **held-out tasks** the conformal-wrapped
/// interval's empirical coverage is ≈ the nominal `1 - alpha`. The predictor
/// meta-trains on the train tasks, the conformal wrap calibrates on one disjoint
/// held-out task, and coverage is measured on the *other* held-out tasks — three
/// disjoint splits, the conformal contract.
///
/// Conformal's guarantee is **marginal** coverage `>= 1 - alpha` — an expectation
/// over the exchangeable distribution, not a property every individual eval split
/// satisfies. A single small held-out split is a high-variance estimate of that
/// expectation, so asserting each seed clears the threshold individually tests a
/// stronger claim than conformal makes and reddens on an unlucky split. Instead we
/// **pool** every eval point's covered/not-covered indicator from all seeds into
/// one sample and assert the pooled fraction clears `1 - alpha - slack`: pooling
/// maximizes the sample size, giving the tightest, least-variance estimate of the
/// marginal coverage the guarantee is actually about.
#[tokio::test(flavor = "multi_thread")]
async fn conformal_wrap_hits_nominal_coverage_across_seeds() {
    let alpha = 0.2_f64;
    let seeds = [11u64, 23, 37];
    let mut pooled: Vec<bool> = Vec::new();
    for &seed in &seeds {
        // A larger meta-dataset so held-out tasks carry enough targets to
        // estimate coverage and calibrate a finite-sample quantile at alpha=0.2.
        let rows = synthetic_meta_dataset(16, 24, seed);
        let (session, _dir) = session_with_meta_dataset(&rows).await;

        let mut spec = spec(
            ContextArchitecture::AttnCnp,
            PredictiveHead::Quantile {
                levels: vec![0.1, 0.5, 0.9],
            },
        );
        spec.seed = seed;
        spec.test_task_fraction = 0.25;
        spec.min_task_count = 4;
        let model_id = train(&session, &spec).await;

        let served = session
            .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
            .await
            .unwrap();

        // The held-out tasks (never trained on). Split them in two: the first is
        // the conformal calibration set, the rest are the coverage-eval set —
        // disjoint from each other and from training.
        let distinct: Vec<String> = {
            let mut t: Vec<String> = rows.iter().map(|r| r.task.clone()).collect();
            t.sort();
            t.dedup();
            t
        };
        let test_tasks = held_out_tasks(distinct, spec.seed, spec.test_task_fraction);
        assert!(
            test_tasks.len() >= 2,
            "need ≥2 held-out tasks to split calibration from eval"
        );
        let cal_task = &test_tasks[0];
        let eval_tasks = &test_tasks[1..];

        let targets_of = |tasks: &[String]| -> Vec<(String, f64)> {
            rows.iter()
                .filter(|r| tasks.iter().any(|t| t == &r.task))
                .map(|r| (r.id.clone(), r.y))
                .collect()
        };
        let cal = targets_of(std::slice::from_ref(cal_task));
        let eval = targets_of(eval_tasks);

        let wrap = session
            .calibrate_context_predictor_conformal(&served, &cal, alpha, ConformalLevers::Marginal)
            .await
            .unwrap();
        // Pool this seed's per-point indicators into the cross-seed sample rather
        // than collapsing to a per-seed fraction first — the pooled fraction is the
        // direct estimate of marginal coverage.
        pooled.extend(conformal_indicators(&session, &served, &wrap, &eval).await);
    }

    assert!(!pooled.is_empty(), "no eval points collected across seeds");
    let pooled_cov = coverage_fraction(&pooled);

    // The guarantee is marginal coverage >= 1 - alpha under exchangeability. The
    // slack is finite-sample only and kept tight — well below the gap between this
    // conformal estimate and the under-covering raw band (see the sibling test),
    // so a non-conformal/under-covering predictor still fails this threshold.
    let slack = 0.05_f64;
    assert!(
        pooled_cov >= 1.0 - alpha - slack,
        "pooled conformal coverage {pooled_cov} over {} eval points across {} seeds \
         fell below the marginal floor {}",
        pooled.len(),
        seeds.len(),
        1.0 - alpha - slack,
    );
}

/// Conformal earns its place: the *raw* S19 quantile band under-covers on a
/// held-out task (the amortized posterior is overconfident off its training
/// tasks), and the conformal wrap restores coverage to ≥ nominal — the exact
/// reason S19 is wrapped by S17 rather than shipped bare.
#[tokio::test(flavor = "multi_thread")]
async fn conformal_restores_coverage_the_raw_band_loses() {
    let alpha = 0.2_f64;
    let rows = synthetic_meta_dataset(16, 24, 909);
    let (session, _dir) = session_with_meta_dataset(&rows).await;

    let mut spec = spec(
        ContextArchitecture::AttnCnp,
        PredictiveHead::Quantile {
            levels: vec![0.1, 0.5, 0.9],
        },
    );
    spec.test_task_fraction = 0.25;
    let model_id = train(&session, &spec).await;
    let served = session
        .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
        .await
        .unwrap();

    let distinct: Vec<String> = {
        let mut t: Vec<String> = rows.iter().map(|r| r.task.clone()).collect();
        t.sort();
        t.dedup();
        t
    };
    let test_tasks = held_out_tasks(distinct, spec.seed, spec.test_task_fraction);
    let cal_task = &test_tasks[0];
    let eval_tasks = &test_tasks[1..];
    let targets_of = |tasks: &[String]| -> Vec<(String, f64)> {
        rows.iter()
            .filter(|r| tasks.iter().any(|t| t == &r.task))
            .map(|r| (r.id.clone(), r.y))
            .collect()
    };
    let cal = targets_of(std::slice::from_ref(cal_task));
    let eval = targets_of(eval_tasks);

    // Raw band coverage: the served quantile band [q0.1, q0.9] without conformal.
    let mut raw_hits = 0usize;
    for (key, y) in &eval {
        let dist = session
            .predict_with_context_predictor(&served, key)
            .await
            .unwrap();
        if let jammi_ai::pipeline::context_predictor::PredictedDistribution::Quantile { levels } =
            dist
        {
            let lo = levels.first().unwrap().1 as f64;
            let hi = levels.last().unwrap().1 as f64;
            if *y >= lo && *y <= hi {
                raw_hits += 1;
            }
        }
    }
    let raw_cov = raw_hits as f64 / eval.len() as f64;

    let wrap = session
        .calibrate_context_predictor_conformal(&served, &cal, alpha, ConformalLevers::Marginal)
        .await
        .unwrap();
    let conf_cov = conformal_coverage(&session, &served, &wrap, &eval).await;

    // The raw band under-covers the nominal 1 - alpha = 0.8; the conformal wrap
    // restores it to at least nominal (minus finite-sample slack).
    assert!(
        raw_cov < 1.0 - alpha,
        "the raw S19 band was expected to under-cover the nominal {}, got {raw_cov}",
        1.0 - alpha
    );
    assert!(
        conf_cov >= raw_cov,
        "the conformal wrap should not lose coverage the raw band had: raw {raw_cov} conf {conf_cov}"
    );
    assert!(
        conf_cov >= 1.0 - alpha - 0.1,
        "conformal coverage {conf_cov} did not reach nominal {}",
        1.0 - alpha
    );
}

/// Epistemic widening: a sparse/unfamiliar context yields a wider predicted σ
/// than a dense/familiar one. This is primarily an **AttnCNP** property — the
/// attentive members and the learned prior path let the head widen when its
/// context is thin; a CNP's mean-pool barely widens and conditions instead on
/// the carried context size. Asserted on AttnCNP, comparing a target served with
/// a large context (`k` neighbours) against the same target served with a
/// 1-member context — the thinner context's σ is wider.
#[tokio::test(flavor = "multi_thread")]
async fn sparse_context_widens_sigma_attncnp() {
    let rows = synthetic_meta_dataset(14, 24, 7777);
    let (session, _dir) = session_with_meta_dataset(&rows).await;

    let spec = spec(
        ContextArchitecture::AttnCnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Crps,
        },
    );
    let model_id = train(&session, &spec).await;

    // Dense: the full context_k. Sparse: a 1-member context (k = 1).
    let dense = session
        .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
        .await
        .unwrap();

    // The same target, served once with the trained context_k and once with a
    // single neighbour. Average σ over several targets so the comparison is the
    // population property, not one target's noise.
    let mut dense_sigma = 0.0;
    let mut sparse_sigma = 0.0;
    let probe: Vec<&Row> = rows.iter().take(12).collect();
    for row in &probe {
        let d = session
            .predict_with_context_predictor(&dense, &row.id)
            .await
            .unwrap();
        if let jammi_ai::pipeline::context_predictor::PredictedDistribution::Gaussian {
            std, ..
        } = d
        {
            dense_sigma += std as f64;
        }
    }
    // Rebuild a served predictor capped to a 1-member context by retraining the
    // serving knob is not exposed per call, so we read the sparse σ by serving
    // the predictor against a context restricted to one neighbour through a
    // narrow split — a single same-task member.
    for row in &probe {
        // Restrict the live context to a single sibling: a split picking exactly
        // one other row of the same task. This is the sparse-context regime.
        let sibling = rows
            .iter()
            .find(|r| r.task == row.task && r.id != row.id)
            .unwrap();
        let split = format!("arrow_cast(\"_row_id\", 'Utf8') = '{}'", sibling.id);
        let sparse = session
            .load_context_predictor(
                &model_id,
                "fns",
                ContextServeOptions {
                    split: Some(split),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let s = session
            .predict_with_context_predictor(&sparse, &row.id)
            .await
            .unwrap();
        if let jammi_ai::pipeline::context_predictor::PredictedDistribution::Gaussian {
            std, ..
        } = s
        {
            sparse_sigma += std as f64;
        }
    }
    let dense_mean = dense_sigma / probe.len() as f64;
    let sparse_mean = sparse_sigma / probe.len() as f64;
    assert!(
        sparse_mean > dense_mean,
        "AttnCNP epistemic widening: sparse-context σ {sparse_mean} should exceed \
         dense-context σ {dense_mean}"
    );
}

/// The generalized `train_loop` still drives the simple `TensorBatch` case
/// (the P5 convergence proof) under the batch-generic signature — the
/// refactor did not break the flat path. (The full P5 parity + convergence
/// suite lives in `parallel_train.rs`; this is the cross-check that the episodic
/// generalization is source-compatible with the flat batch type.)
#[test]
fn generalized_train_loop_still_drives_tensor_batch() {
    use jammi_ai::pipeline::parallel_train::TensorBatch;

    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let linear = candle_nn::linear(2, 1, vb.pp("l")).unwrap();

    let n = 16usize;
    let mut feats = Vec::with_capacity(n * 2);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x0 = (i as f32 % 7.0) - 3.0;
        let x1 = (i as f32 % 5.0) - 2.0;
        feats.push(x0);
        feats.push(x1);
        targets.push(2.0 * x0 - 3.0 * x1 + 0.5);
    }
    let features = Tensor::from_vec(feats, (n, 2), &dev).unwrap();
    let targets = Tensor::from_vec(targets, (n, 1), &dev).unwrap();
    let batches = vec![TensorBatch { features, targets }];

    let report = train_loop(
        &varmap,
        &batches,
        &ParallelTrainConfig {
            epochs: 300,
            learning_rate: 0.05,
            weight_decay: 0.0,
            grad_clip: 1.0,
        },
        &std::sync::atomic::AtomicBool::new(false),
        |batch: &TensorBatch| {
            candle_nn::Module::forward(&linear, &batch.features)
                .map_err(|e| jammi_db::error::JammiError::FineTune(format!("{e}")))
        },
        |preds, batch: &TensorBatch| {
            let diff = (preds - &batch.targets)
                .map_err(|e| jammi_db::error::JammiError::FineTune(format!("{e}")))?;
            diff.sqr()
                .and_then(|d| d.mean_all())
                .map_err(|e| jammi_db::error::JammiError::FineTune(format!("{e}")))
        },
    )
    .unwrap();

    assert!(
        report.final_loss < 1e-2,
        "generic train_loop did not converge on the flat case: {}",
        report.final_loss
    );
}

/// A served prediction always carries its provenance: the assembly `source` fact
/// and the context member keys (never the target itself). This is the seam the
/// coverage layer attributes a prediction by — a graph-conditioned (or any)
/// prediction is never *unattributed*. The `source_kind()` accessor is the seam
/// governance reads off the served state.
#[tokio::test(flavor = "multi_thread")]
async fn predict_provenanced_carries_source_and_context_keys() {
    let rows = synthetic_meta_dataset(8, 16, 5);
    let (session, _dir) = session_with_meta_dataset(&rows).await;
    let spec = spec(
        ContextArchitecture::Cnp,
        PredictiveHead::Gaussian {
            objective: GaussianObjective::Nll { beta: 0.5 },
        },
    );
    let model_id = train(&session, &spec).await;
    let served = session
        .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
        .await
        .unwrap();

    // The served state's source fact (the E8 seam).
    assert_eq!(served.source_kind(), ContextSourceKind::Ann);

    let target = &rows[0].id;
    let prov = session
        .predict_with_context_predictor_provenanced(&served, target)
        .await
        .unwrap();
    assert_eq!(
        prov.source,
        ContextSourceKind::Ann,
        "an ANN serve source produces the Ann assembly fact"
    );
    assert!(
        !prov.context_keys.is_empty(),
        "the prediction carries its context provenance — never unattributed"
    );
    assert!(
        !prov.context_keys.iter().any(|k| k == target),
        "the target is never in its own context provenance (the leakage guard)"
    );
    // The bare-distribution wrapper agrees with the provenanced distribution.
    let bare = session
        .predict_with_context_predictor(&served, target)
        .await
        .unwrap();
    assert_eq!(bare, prov.distribution);
}

/// The conformal levers are *applied*, never *chosen*: marginal (the default)
/// always serves; a caller-supplied Mondrian cohort or weights route to the
/// group-conditional / weighted S17 constructors; and a cohort/point length
/// mismatch is a typed error, never a silent misalignment. The engine never
/// self-selects a cohort.
#[tokio::test(flavor = "multi_thread")]
async fn conformal_levers_apply_and_never_self_select() {
    let alpha = 0.2_f64;
    let rows = synthetic_meta_dataset(16, 24, 31);
    let (session, _dir) = session_with_meta_dataset(&rows).await;
    let mut spec = spec(
        ContextArchitecture::AttnCnp,
        PredictiveHead::Quantile {
            levels: vec![0.1, 0.5, 0.9],
        },
    );
    spec.test_task_fraction = 0.25;
    let model_id = train(&session, &spec).await;
    let served = session
        .load_context_predictor(&model_id, "fns", ContextServeOptions::default())
        .await
        .unwrap();

    let distinct: Vec<String> = {
        let mut t: Vec<String> = rows.iter().map(|r| r.task.clone()).collect();
        t.sort();
        t.dedup();
        t
    };
    let test_tasks = held_out_tasks(distinct, spec.seed, spec.test_task_fraction);
    let cal: Vec<(String, f64)> = rows
        .iter()
        .filter(|r| r.task == test_tasks[0])
        .map(|r| (r.id.clone(), r.y))
        .collect();
    assert!(cal.len() >= 4, "need a few calibration points");
    let probe = &cal[0].0;
    let dist = session
        .predict_with_context_predictor(&served, probe)
        .await
        .unwrap();

    // Marginal: serves with no cohort (the engine self-selects nothing).
    let marginal = session
        .calibrate_context_predictor_conformal(&served, &cal, alpha, ConformalLevers::Marginal)
        .await
        .unwrap();
    let (lo, hi) = marginal.interval(&dist, None).unwrap();
    assert!(lo <= hi, "the marginal wrap always serves an interval");

    // Mondrian: one caller-supplied cohort per calibration point; serves for a
    // supplied cohort.
    let groups: Vec<String> = (0..cal.len())
        .map(|i| if i % 2 == 0 { "a" } else { "b" }.to_string())
        .collect();
    let mondrian = session
        .calibrate_context_predictor_conformal(
            &served,
            &cal,
            alpha,
            ConformalLevers::Mondrian { groups },
        )
        .await
        .unwrap();
    let (lo_m, hi_m) = mondrian.interval(&dist, Some("a")).unwrap();
    assert!(
        lo_m <= hi_m,
        "the Mondrian wrap serves an interval for a governance-supplied cohort"
    );

    // Weighted: one weight per calibration point.
    let weights = vec![1.0_f64; cal.len()];
    let weighted = session
        .calibrate_context_predictor_conformal(
            &served,
            &cal,
            alpha,
            ConformalLevers::Weighted { weights },
        )
        .await
        .unwrap();
    let (lo_w, hi_w) = weighted.interval(&dist, None).unwrap();
    assert!(lo_w <= hi_w, "the weighted wrap serves an interval");

    // A cohort/point length mismatch is rejected, not silently misaligned.
    let mismatch = session
        .calibrate_context_predictor_conformal(
            &served,
            &cal,
            alpha,
            ConformalLevers::Mondrian {
                groups: vec!["only-one".to_string()],
            },
        )
        .await;
    assert!(
        mismatch.is_err(),
        "one cohort for many calibration points must be a typed error"
    );
}
