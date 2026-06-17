//! The CPU-hermetic context-predictor tier: the engine's episodic meta-training
//! ([`train_context_predictor`](InferenceSession::train_context_predictor)) and
//! in-context serving
//! ([`predict_with_context_predictor`](InferenceSession::predict_with_context_predictor)),
//! measured for training throughput and gated for predict determinism over a
//! committed weight bundle.
//!
//! This is the in-context-prediction peer of [`crate::train_scale`] and
//! [`crate::propagate`]: a same-box training *rate* on one side, a portable
//! same-machine determinism *digest* on the other.
//!
//! ## Why the predict digest gates over committed *weights*, not a train→predict run
//!
//! The engine's episodic `train_loop` over the candle CPU backward is **not**
//! byte-reproducible run-to-run (the CPU gemm reduction order floats), so a digest
//! of a full train→predict pipeline would never re-derive — gating it would be
//! the vacuity trap. What *is* deterministic is the engine's
//! inference-only-no-gradient serving contract: `predict_with_context_predictor`
//! over a fixed served weight set and a fixed target is byte-identical across runs
//! and across a fresh reload **on a machine** (the engine's
//! `predict_is_inference_only_no_gradient_updates` contract). So the committed
//! artifact is a real **trained weight bundle** — the safetensors the off-box
//! rebuild's training produced — and the gate re-derives the predict digest by
//! *loading that bundle and predicting* on the running box, never by retraining.
//!
//! The predicted distribution is `f32`, and an `f32` forward's exact bits are NOT
//! identical across CPUs (SIMD/FMA/BLAS reduction order differs). So the gate does
//! not assert equality to a committed cross-machine constant — that would
//! spuriously "fail" on any box but the rebuild box. It asserts the real,
//! same-machine property: predict twice on the running box and the two digests
//! agree. A regression in the serve/predict path (context assembly, the in-context
//! forward, the distribution adapter, the de-standardisation) is caught by the
//! relative perturbation teeth (a wrong `context_k` vs the in-process baseline).
//!
//! ## Two lanes
//!
//! * **Training throughput** — the meta-training work (`train_episodes · epochs`
//!   episode-steps) the engine's `train_context_predictor` drives per second on
//!   the CPU, gated against a committed same-box baseline by [`crate::rate_gate`].
//!   A *rate*, so it is the training tier's same-box discipline, never a portable
//!   floor.
//! * **The predict determinism digest** — a stable checksum of the predicted
//!   distributions over the committed targets; the gate folds it twice on the
//!   running box and asserts the two agree. Predict wall-time rides as an un-gated,
//!   machine-dependent reference.
//!
//! ## The committed artifact bundle
//!
//! The committed `baselines/context_predictor.json` carries the dataset
//! generation spec (so the gate regenerates the exact synthetic meta-dataset and
//! its embedding table deterministically), the trained predictor's `config_json`
//! (architecture, `context_k`, the persisted target scaler), the committed target
//! keys, the predict digest, and the same-box training baseline — never a
//! hand-written digest. The predict digest is a documented **same-box reference**
//! (the output is `f32`), recorded on the rebuild box. The trained weight bundle
//! (`model.safetensors` + `manifest.json`) lives beside it under
//! `baselines/context_predictor_weights/`. The rebuild re-derives every committed
//! value from a fresh train + predict.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::pipeline::context_predictor::{
    ContextArchitecture, ContextPredictorTrainConfig, ContextServeOptions, GaussianObjective,
    PredictedDistribution, PredictiveHead, SampledEpisodes,
};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::report::{ContextPredictorTier, DeterminismGate, Measurement, RateVerdict};

/// The feature (embedding) dimensionality of the synthetic meta-dataset — the
/// same small dim the engine's own context-predictor suite uses.
const FEATURE_DIM: usize = 4;
/// The source id the synthetic meta-dataset registers under. Generic — names no
/// consumer; the dataset is a neutral family of linear functions.
const SOURCE_ID: &str = "fns";
/// The model id the synthetic embedding table is stamped with.
const EMBED_MODEL_ID: &str = "synthetic-embed";
/// The model id the trained context predictor registers under.
const PREDICTOR_MODEL_ID: &str = "ctx-predictor";

/// The committed context-predictor spec: the dataset generation parameters, the
/// training spec knobs, the trained predictor's serialised config, the committed
/// target keys, and the gated values (the predict digest and the same-box
/// training baseline). The on-disk `baselines/context_predictor.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPredictorSpec {
    /// Distinct tasks the synthetic linear-function meta-dataset carries.
    pub n_tasks: usize,
    /// Rows per task.
    pub rows_per_task: usize,
    /// The dataset generation seed (the splitmix64 stream feeding the weights,
    /// features, and outcomes).
    pub dataset_seed: u64,
    /// The predictor architecture (`Cnp` / `AttnCnp`).
    pub architecture: String,
    /// The context width `k` the predictor trains and serves at.
    pub context_k: usize,
    /// The predictor hidden dimension.
    pub hidden_dim: usize,
    /// Attention heads (used by `AttnCnp`).
    pub num_heads: usize,
    /// Predictor layers.
    pub num_layers: usize,
    /// Meta-training epochs.
    pub epochs: usize,
    /// The training learning rate.
    pub learning_rate: f64,
    /// The gradient-clipping ceiling.
    pub grad_clip: f64,
    /// The held-out task fraction the train/test split uses.
    pub test_task_fraction: f64,
    /// The meta-overfitting guard: the minimum distinct task count.
    pub min_task_count: usize,
    /// The training spec seed (the deterministic task split anchor).
    pub spec_seed: u64,
    /// The trained predictor's serialised config (`load_context_predictor` reads
    /// the architecture, `context_k`, head form, and persisted target scaler from
    /// it). Captured from the rebuild's trained model; the gate registers a model
    /// row with it so the committed weights reload exactly as trained.
    pub config_json: String,
    /// The target row keys the predict digest is folded over — a deterministic
    /// prefix of the dataset's rows, so any box re-predicts the same set.
    pub target_keys: Vec<String>,
    /// The committed predict digest: the checksum the engine's
    /// `predict_with_context_predictor` produced over the committed targets and
    /// the committed weight bundle when the spec was cut, on the rebuild box. A
    /// documented same-box reference (the predicted distribution is `f32`, whose
    /// exact bits vary by CPU) — reported for human comparison, never asserted for
    /// cross-machine equality.
    pub predict_digest: String,
    /// The committed same-box training baseline, episode-steps/s.
    pub baseline_episode_steps_per_s: f64,
}

impl ContextPredictorSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("context_predictor.json")
    }

    /// The crate-relative directory the committed trained weight bundle
    /// (`model.safetensors` + `manifest.json`) lives in.
    pub fn weights_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("context_predictor_weights")
    }

    /// Load the committed spec from `baselines/context_predictor.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }

    /// The architecture enum this spec names.
    fn architecture(&self) -> Result<ContextArchitecture, Box<dyn std::error::Error>> {
        match self.architecture.as_str() {
            "Cnp" => Ok(ContextArchitecture::Cnp),
            "AttnCnp" => Ok(ContextArchitecture::AttnCnp),
            "Tnp" => Ok(ContextArchitecture::Tnp),
            other => Err(format!("unknown context-predictor architecture {other:?}").into()),
        }
    }

    /// The engine training spec this committed spec drives.
    fn train_config(&self) -> Result<ContextPredictorTrainConfig, Box<dyn std::error::Error>> {
        Ok(ContextPredictorTrainConfig {
            model_id: PREDICTOR_MODEL_ID.to_string(),
            architecture: self.architecture()?,
            key_column: "_row_id".to_string(),
            task_column: "task".to_string(),
            value_column: "y".to_string(),
            context_k: self.context_k,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            head: PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            },
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            grad_clip: self.grad_clip,
            test_task_fraction: self.test_task_fraction,
            min_task_count: self.min_task_count,
            seed: self.spec_seed,
        })
    }
}

/// splitmix64 — the deterministic generator the engine's own context-predictor
/// suite draws its synthetic meta-dataset from, so the bench regenerates a
/// byte-identical dataset from the committed seed.
struct SplitMix(u64);

impl SplitMix {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// One synthetic row: its key, the task it belongs to, its feature vector `x`,
/// and its linear outcome `y = w_task · x`.
struct Row {
    id: String,
    task: String,
    x: Vec<f32>,
    y: f64,
}

/// Build the synthetic linear-function meta-dataset from the committed seed: a
/// family of `n_tasks` linear maps, each with `rows_per_task` rows. Same shape and
/// generator the engine's context-predictor suite uses, regenerated here so the
/// gate stands up the exact dataset the committed digest was folded over.
fn build_dataset(spec: &ContextPredictorSpec) -> Vec<Row> {
    let mut rng = SplitMix(spec.dataset_seed);
    let mut rows = Vec::with_capacity(spec.n_tasks * spec.rows_per_task);
    for t in 0..spec.n_tasks {
        let w: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
        for r in 0..spec.rows_per_task {
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

/// Stand up a hermetic `Device::Cpu` session over the synthetic meta-dataset:
/// write the source parquet (`_row_id`, `task`, `y`), register it, and
/// materialise the embedding table whose `vector` is each row's feature `x` —
/// through the engine's own embedding-table writer, the same setup the engine's
/// context-predictor suite uses. Holds the [`tempfile::TempDir`] so the artifacts
/// outlive the session.
async fn dataset_session(
    rows: &[Row],
) -> Result<(Arc<InferenceSession>, tempfile::TempDir), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let config = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await?);
    session.register_query_functions();

    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let tasks: Vec<&str> = rows.iter().map(|r| r.task.as_str()).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(tasks)),
            Arc::new(Float64Array::from(ys)),
        ],
    )?;
    let path = dir.path().join("source.parquet");
    let url = StorageUrl::parse(path.to_str().ok_or("source path is not valid UTF-8")?)?;
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None)?;
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema)).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    session
        .add_source(
            SOURCE_ID,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", path.to_str().unwrap())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    let pairs: Vec<(String, Vec<f32>)> = rows.iter().map(|r| (r.id.clone(), r.x.clone())).collect();
    let descriptor = jammi_db::store::manifest::ProducingDescriptor::ContextSet {
        encoder_id: EMBED_MODEL_ID.to_string(),
        source_id: SOURCE_ID.to_string(),
        dimensions: FEATURE_DIM,
    };
    let env =
        jammi_db::store::manifest::MaterializationEnv::new(session.compute_device(), Vec::new());
    let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
        SOURCE_ID,
        "1970-01-01T00:00:00Z",
    )];
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            jammi_db::store::EmbeddingTableSpec {
                source_id: SOURCE_ID,
                model_id: EMBED_MODEL_ID,
                derived_from: None,
                dimensions: FEATURE_DIM,
            },
            &pairs,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
        )
        .await?;

    Ok((session, dir))
}

/// The stable checksum of a set of predicted distributions over the committed
/// targets, in target order — an FNV-1a hash over each distribution's raw `f32`
/// bits, rendered as a fixed-width hex string.
///
/// Pure arithmetic over the served floats, no crate. Because the engine's serving
/// path is byte-deterministic over a fixed weight set and target, this digest is
/// a stable reference: any change to a predicted mean/σ (or a quantile value)
/// flips it. A type tag byte distinguishes a Gaussian from a quantile serve so the
/// two head shapes cannot collide.
fn digest(predictions: &[PredictedDistribution]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for pred in predictions {
        match pred {
            PredictedDistribution::Gaussian { mean, std } => {
                mix(0x01);
                for b in mean.to_bits().to_le_bytes() {
                    mix(b);
                }
                for b in std.to_bits().to_le_bytes() {
                    mix(b);
                }
            }
            PredictedDistribution::Quantile { levels } => {
                mix(0x02);
                for (level, value) in levels {
                    for b in level.to_bits().to_le_bytes() {
                        mix(b);
                    }
                    for b in value.to_bits().to_le_bytes() {
                        mix(b);
                    }
                }
            }
        }
    }
    format!("{hash:016x}")
}

/// The committed trained-weight bundle file names the gate registers a model row
/// over: the safetensors weights and the artifact-store manifest the loader
/// verifies. The bundle is fetched by [`InferenceSession::artifact_store`] before
/// `load_context_predictor` reloads it, and that fetch verifies the manifest, so
/// both files travel.
const BUNDLE_FILES: [&str; 2] = ["model.safetensors", "manifest.json"];

/// Register a model row in `session`'s catalog pointing at a committed weight
/// bundle directory, so `load_context_predictor` reloads the committed weights
/// exactly as the rebuild's training produced them. `config_json` carries the
/// architecture / `context_k` / scaler the loader rebuilds the predictor from;
/// `model_id` lets the gate register a second row under a *perturbed* config for
/// the teeth test.
async fn register_committed_weights(
    session: &Arc<InferenceSession>,
    model_id: &str,
    weights_dir: &std::path::Path,
    config_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let artifact = format!(
        "file://{}",
        weights_dir
            .to_str()
            .ok_or("committed weights dir is not valid UTF-8")?
    );
    session
        .catalog()
        .register_model(RegisterModelParams {
            model_id,
            version: 1,
            model_type: "context-predictor",
            backend: "candle",
            task: ModelTask::Regression,
            base_model_id: None,
            artifact_path: Some(&artifact),
            config_json: Some(config_json),
        })
        .await?;
    Ok(())
}

/// Predict over the committed targets through the engine's real
/// `predict_with_context_predictor`, given a session that already carries the
/// dataset embedding table and a model row pointing at the committed weights.
/// Returns the predicted distributions in target order and the total predict
/// wall-time (the un-gated latency reference).
///
/// `model_id` selects which registered model row (and so which `config_json`) the
/// serve loads under — the committed config for the gate, a perturbed config for
/// the teeth test.
async fn predict_committed_targets(
    session: &Arc<InferenceSession>,
    model_id: &str,
    target_keys: &[String],
) -> Result<(Vec<PredictedDistribution>, f64), Box<dyn std::error::Error>> {
    let served = session
        .load_context_predictor(model_id, SOURCE_ID, ContextServeOptions::default())
        .await?;
    let start = Instant::now();
    let mut predictions = Vec::with_capacity(target_keys.len());
    for key in target_keys {
        predictions.push(session.predict_with_context_predictor(&served, key).await?);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    Ok((predictions, elapsed_ms))
}

/// Re-derive a predict digest: stand up the dataset session, register the committed
/// weight bundle, predict the committed targets through the real engine, and digest
/// the predictions. Used by `rebuild_spec` to record the same-box reference and by
/// the determinism test to fold twice on the running box.
///
/// `weights_dir` is passed explicitly so a caller can point at the committed bundle
/// under a perturbed config; the default uses [`ContextPredictorSpec::weights_dir`].
pub async fn fold_predict_digest(
    spec: &ContextPredictorSpec,
    weights_dir: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let rows = build_dataset(spec);
    let (session, _dir) = dataset_session(&rows).await?;
    register_committed_weights(&session, PREDICTOR_MODEL_ID, weights_dir, &spec.config_json)
        .await?;
    let (predictions, _ms) =
        predict_committed_targets(&session, PREDICTOR_MODEL_ID, &spec.target_keys).await?;
    Ok(digest(&predictions))
}

/// Measure the training throughput of the engine's `train_context_predictor` on
/// the CPU: submit the durable training job, run it to completion through an
/// embedded worker, and divide the total meta-training work (`train_episodes ·
/// epochs` episode-steps) by the job wall-clock.
///
/// A *rate*, so its absolute value is not gated for equality (unlike the predict
/// digest) — it rides into the report and is gated against the committed same-box
/// baseline by [`crate::rate_gate`]. Returns `(episode_steps_per_s, wall_ms,
/// train_episodes)`.
async fn measure_train_throughput(
    spec: &ContextPredictorSpec,
) -> Result<(f64, f64, usize), Box<dyn std::error::Error>> {
    let rows = build_dataset(spec);
    let (session, _dir) = dataset_session(&rows).await?;
    let config = spec.train_config()?;

    // The episode count the meta-training optimises over per epoch — measured
    // through the engine's own deterministic sampler, so the work the rate divides
    // is the real per-epoch episode count, not an estimate.
    let SampledEpisodes { train, .. } = session.sample_context_episodes(SOURCE_ID, &config).await?;
    let train_episodes = train.len();

    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)?;
    let start = Instant::now();
    let job = session.train_context_predictor(SOURCE_ID, &config).await?;
    job.wait().await?;
    let elapsed = start.elapsed();

    let wall_ms = elapsed.as_secs_f64() * 1_000.0;
    let episode_steps = (train_episodes * spec.epochs) as f64;
    let rate = if elapsed.as_secs_f64() > 0.0 {
        episode_steps / elapsed.as_secs_f64()
    } else {
        0.0
    };
    Ok((rate, wall_ms, train_episodes))
}

/// Run the context-predictor tier against the committed spec: measure the
/// training throughput of `train_context_predictor`, re-fold the predict digest
/// over the committed weight bundle through `predict_with_context_predictor`, and
/// assemble the tier with the rate gate and the digest gate.
///
/// This is the path the `context-predictor-scale` subcommand drives and the
/// `cargo test` gate asserts: the predict digest is the engine's own served-output
/// checksum, folded twice on this box and gated for same-machine determinism (the
/// output is `f32`, so the committed digest rides as a same-box reference, not a
/// cross-machine constant); the training throughput is a same-box rate gated by the
/// relative-drop [`crate::rate_gate`]; the predict latency rides as an un-gated
/// reference.
pub async fn run(
    spec: &ContextPredictorSpec,
) -> Result<ContextPredictorTier, Box<dyn std::error::Error>> {
    let (rate, train_wall_ms, train_episodes) = measure_train_throughput(spec).await?;

    let weights_dir = ContextPredictorSpec::weights_dir();

    // Two same-machine predict folds over the committed bundle: the gate asserts
    // they agree (the portable same-machine determinism contract). The first fold's
    // serve also yields the predict latency reference.
    let (first_predictions, predict_ms) = predict_once(spec, &weights_dir).await?;
    let (second_predictions, _ms) = predict_once(spec, &weights_dir).await?;
    let predict_gate = DeterminismGate::new(
        digest(&first_predictions),
        digest(&second_predictions),
        spec.predict_digest.clone(),
    );

    let gate = crate::rate_gate::RateGate::evaluate(
        rate,
        spec.baseline_episode_steps_per_s,
        crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD,
    );

    Ok(ContextPredictorTier {
        architecture: match spec.architecture()? {
            ContextArchitecture::Cnp => "Cnp",
            ContextArchitecture::AttnCnp => "AttnCnp",
            ContextArchitecture::Tnp => "Tnp",
        },
        context_k: spec.context_k,
        train_episodes,
        train_pairs_per_s: Measurement::measured(rate, "episode_steps_per_s"),
        train_wall_ms: Measurement::measured(train_wall_ms, "ms"),
        rate_gate: Some(RateVerdict {
            measured_pairs_per_s: gate.measured,
            baseline_pairs_per_s: gate.baseline,
            threshold: gate.threshold,
            floor_pairs_per_s: gate.floor,
            passed: gate.passed,
            detail: gate.detail(),
        }),
        predict_digest: predict_gate,
        predict_latency_ms: Measurement::measured(predict_ms, "ms"),
    })
}

/// One same-machine predict fold over the committed weight bundle: stand up a
/// fresh dataset session, register the committed weights, and predict the committed
/// targets through the real engine. Returns the predicted distributions and the
/// serve wall-time. The `run` gate calls this twice on the running box and asserts
/// the two digests agree (the portable same-machine determinism contract).
async fn predict_once(
    spec: &ContextPredictorSpec,
    weights_dir: &std::path::Path,
) -> Result<(Vec<PredictedDistribution>, f64), Box<dyn std::error::Error>> {
    let rows = build_dataset(spec);
    let (session, _dir) = dataset_session(&rows).await?;
    register_committed_weights(&session, PREDICTOR_MODEL_ID, weights_dir, &spec.config_json)
        .await?;
    predict_committed_targets(&session, PREDICTOR_MODEL_ID, &spec.target_keys).await
}

/// Whether both gates held — the verdict the subcommand maps to its exit code and
/// the `cargo test` gate asserts: the two same-machine predict folds agreed AND the
/// training throughput cleared the same-box floor.
pub fn gates_passed(tier: &ContextPredictorTier) -> bool {
    tier.predict_digest.passed && tier.rate_gate.as_ref().is_none_or(|v| v.passed)
}

/// Re-derive the committed spec and its weight bundle from a fresh train +
/// predict: regenerate the dataset, train a predictor through the engine's
/// `train_context_predictor`, copy the trained weight bundle into
/// `baselines/context_predictor_weights/`, capture its `config_json`, predict the
/// committed targets, and record the predict digest and the same-box training
/// baseline. The off-box one-shot that writes `baselines/context_predictor.json`
/// and the bundle; CI only ever loads and re-predicts them.
///
/// Returns the spec; the caller writes the JSON. The weight bundle is written here
/// (it is a directory of binary files, not part of the JSON spec).
pub async fn rebuild_spec(
    params: ContextPredictorParams,
) -> Result<ContextPredictorSpec, Box<dyn std::error::Error>> {
    // Stage 1: train a predictor on the synthetic dataset through the engine, and
    // capture both its config and the trained weight bundle.
    let mut spec = ContextPredictorSpec {
        n_tasks: params.n_tasks,
        rows_per_task: params.rows_per_task,
        dataset_seed: params.dataset_seed,
        architecture: params.architecture.to_string(),
        context_k: params.context_k,
        hidden_dim: params.hidden_dim,
        num_heads: params.num_heads,
        num_layers: params.num_layers,
        epochs: params.epochs,
        learning_rate: params.learning_rate,
        grad_clip: params.grad_clip,
        test_task_fraction: params.test_task_fraction,
        min_task_count: params.min_task_count,
        spec_seed: params.spec_seed,
        config_json: String::new(),
        target_keys: Vec::new(),
        predict_digest: String::new(),
        baseline_episode_steps_per_s: 0.0,
    };

    let rows = build_dataset(&spec);
    let (session, _dir) = dataset_session(&rows).await?;
    let config = spec.train_config()?;
    let SampledEpisodes { train, .. } = session.sample_context_episodes(SOURCE_ID, &config).await?;
    let train_episodes = train.len();

    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)?;
    let start = Instant::now();
    let job = session.train_context_predictor(SOURCE_ID, &config).await?;
    job.wait().await?;
    let elapsed = start.elapsed();
    spec.baseline_episode_steps_per_s = if elapsed.as_secs_f64() > 0.0 {
        (train_episodes * spec.epochs) as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let record = session
        .catalog()
        .get_model(PREDICTOR_MODEL_ID)
        .await?
        .ok_or("rebuild: trained predictor was not registered")?;
    spec.config_json = record
        .config_json
        .clone()
        .ok_or("rebuild: trained predictor carries no config_json")?;

    // Stage 2: copy the trained weight bundle into the committed weights dir.
    let prefix = record
        .artifact_path
        .as_deref()
        .ok_or("rebuild: trained predictor has no artifact path")?;
    let prefix_url = StorageUrl::parse(prefix)?;
    let local = session.artifact_store().fetch_artifact(&prefix_url).await?;
    let weights_dir = ContextPredictorSpec::weights_dir();
    if weights_dir.exists() {
        std::fs::remove_dir_all(&weights_dir)?;
    }
    std::fs::create_dir_all(&weights_dir)?;
    for file in BUNDLE_FILES {
        std::fs::copy(local.dir().join(file), weights_dir.join(file))?;
    }

    // Stage 3: the committed targets — a deterministic prefix of the dataset rows
    // — and the predict digest the committed weights produce over them.
    spec.target_keys = rows
        .iter()
        .take(params.target_count)
        .map(|r| r.id.clone())
        .collect();
    spec.predict_digest = fold_predict_digest(&spec, &weights_dir).await?;

    Ok(spec)
}

/// The generation parameters a rebuild draws the committed spec from — the
/// dataset shape, the predictor spec, and how many targets the predict digest
/// folds over. Passed as one struct so the rebuilder takes the shape as a unit.
#[derive(Debug, Clone, Copy)]
pub struct ContextPredictorParams {
    /// Distinct tasks the synthetic meta-dataset carries.
    pub n_tasks: usize,
    /// Rows per task.
    pub rows_per_task: usize,
    /// The dataset generation seed.
    pub dataset_seed: u64,
    /// The predictor architecture token (`Cnp` / `AttnCnp`).
    pub architecture: &'static str,
    /// The context width `k`.
    pub context_k: usize,
    /// The predictor hidden dimension.
    pub hidden_dim: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Predictor layers.
    pub num_layers: usize,
    /// Meta-training epochs.
    pub epochs: usize,
    /// The learning rate.
    pub learning_rate: f64,
    /// The gradient-clipping ceiling.
    pub grad_clip: f64,
    /// The held-out task fraction.
    pub test_task_fraction: f64,
    /// The meta-overfitting guard.
    pub min_task_count: usize,
    /// The training spec seed.
    pub spec_seed: u64,
    /// How many of the dataset's rows the predict digest folds over.
    pub target_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed spec is well-formed: a structured meta-dataset, a positive
    /// context width, a non-empty config and target set, a digest of the FNV
    /// width, and a positive baseline rate.
    #[test]
    fn committed_spec_is_well_formed() {
        let spec =
            ContextPredictorSpec::load().expect("baselines/context_predictor.json must be present");
        assert!(
            spec.n_tasks >= spec.min_task_count,
            "enough tasks to meta-train"
        );
        assert!(spec.rows_per_task >= 2);
        assert!(spec.context_k >= 1);
        assert!(spec.epochs >= 1);
        assert!(
            !spec.config_json.is_empty(),
            "config travels for the reload"
        );
        assert!(!spec.target_keys.is_empty(), "the digest needs targets");
        assert_eq!(
            spec.predict_digest.len(),
            16,
            "digest is a 64-bit FNV hex string"
        );
        assert!(
            spec.predict_digest.chars().all(|c| c.is_ascii_hexdigit()),
            "digest must be hex"
        );
        assert!(
            spec.baseline_episode_steps_per_s > 0.0,
            "committed baseline rate must be positive"
        );
    }

    /// The committed weight bundle is present and complete: both bundle files the
    /// artifact-store reload verifies exist under the committed weights dir. A
    /// missing bundle would make the digest gate fail to even reload, so this
    /// guards the committed artifact's completeness.
    #[test]
    fn committed_weight_bundle_is_present() {
        let dir = ContextPredictorSpec::weights_dir();
        for file in BUNDLE_FILES {
            assert!(
                dir.join(file).exists(),
                "committed weight bundle missing {file} under {}",
                dir.display()
            );
        }
    }

    /// The portable determinism gate (DIGEST-CLEARS direction): loading the
    /// committed weight bundle and predicting the committed targets through the
    /// engine's real `predict_with_context_predictor` twice on THIS box (each a fresh
    /// session + register + load + predict) produces the byte-identical digest. This
    /// is the engine's real contract — same-machine, reload-invariant byte-identity
    /// for an `f32` serve — and is true on any CI box by construction, unlike a
    /// committed cross-machine constant, which `f32` SIMD/FMA/BLAS differences would
    /// spuriously break.
    #[tokio::test(flavor = "multi_thread")]
    async fn repredict_is_deterministic_on_this_machine() {
        let spec =
            ContextPredictorSpec::load().expect("baselines/context_predictor.json must be present");
        let first = fold_predict_digest(&spec, &ContextPredictorSpec::weights_dir())
            .await
            .expect("first predict fold runs over the committed bundle");
        let second = fold_predict_digest(&spec, &ContextPredictorSpec::weights_dir())
            .await
            .expect("second predict fold runs over the committed bundle");
        assert_eq!(
            first, second,
            "two same-machine predict folds disagreed — predict is not deterministic on this box"
        );
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to fail).
    ///
    /// A perturbed serve — the SAME committed weights loaded under a regressed
    /// config — produces a different predict digest, proving the gate catches the
    /// serve/predict regressions it exists to catch. The perturbation is a
    /// **wrong `context_k`** in the loaded config: the predictor then assembles a
    /// different-width context for every target, so its in-context forward serves
    /// a different distribution — exactly the regression a mis-recorded serving
    /// knob would cause. The perturbed digest is compared against the IN-PROCESS
    /// baseline (the committed config served on THIS box), so the teeth are portable:
    /// both serves run on the same machine.
    #[tokio::test(flavor = "multi_thread")]
    async fn wrong_context_k_changes_the_predict_digest() {
        let spec =
            ContextPredictorSpec::load().expect("baselines/context_predictor.json must be present");
        let rows = build_dataset(&spec);
        let (session, _dir) = dataset_session(&rows).await.expect("dataset session");

        // The in-process baseline: the committed config served on THIS box.
        register_committed_weights(
            &session,
            PREDICTOR_MODEL_ID,
            &ContextPredictorSpec::weights_dir(),
            &spec.config_json,
        )
        .await
        .expect("register committed weights");
        let (correct, _) =
            predict_committed_targets(&session, PREDICTOR_MODEL_ID, &spec.target_keys)
                .await
                .expect("committed predict runs");
        let baseline = digest(&correct);

        // A perturbed config (a smaller context_k) serves a different distribution.
        let from = format!("\"context_k\":{}", spec.context_k);
        let to = format!("\"context_k\":{}", spec.context_k.saturating_sub(2).max(1));
        let bad_config = spec.config_json.replace(&from, &to);
        assert_ne!(
            bad_config, spec.config_json,
            "the perturbation must actually change the config (context_k must be present)"
        );
        register_committed_weights(
            &session,
            "ctx-predictor-perturbed",
            &ContextPredictorSpec::weights_dir(),
            &bad_config,
        )
        .await
        .expect("register perturbed config");
        let (perturbed, _) =
            predict_committed_targets(&session, "ctx-predictor-perturbed", &spec.target_keys)
                .await
                .expect("perturbed predict runs");
        assert_ne!(
            digest(&perturbed),
            baseline,
            "a wrong context_k must trip the predict digest (else a serving-knob regression slips)"
        );
    }

    /// The committed throughput baseline gates with teeth: a run at the baseline
    /// clears the gate, a run past the threshold fails it (RC1). Asserts the
    /// committed baseline is a well-formed, generously-thresholded same-box
    /// reference without re-measuring the (slow) CPU training in the test lane.
    #[test]
    fn committed_baseline_gates_with_teeth() {
        use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};

        let spec =
            ContextPredictorSpec::load().expect("baselines/context_predictor.json must be present");
        let rate = spec.baseline_episode_steps_per_s;
        assert!(rate > 0.0, "committed baseline rate must be positive");
        assert!(
            RateGate::evaluate(rate, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a run at the committed baseline must pass the gate"
        );
        let floor = rate * (1.0 - DEFAULT_REGRESSION_THRESHOLD);
        assert!(
            !RateGate::evaluate(floor - 1.0, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a rate below the floor must fail the gate"
        );
        assert!(
            RateGate::evaluate(rate * 1.1, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a faster-than-baseline run must pass"
        );
    }
}
