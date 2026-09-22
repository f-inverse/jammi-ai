//! The `predictor-train-run` workload's engine rung, `in-process`, and the
//! served predictor's determinism contract.
//!
//! ## The rung
//!
//! A context-predictor training job is a composite: sample the meta-dataset into
//! episodes (context assembly through `search`, leakage guards, the task split),
//! then meta-train over them. The episode set is its committed intermediate
//! artifact and the cut: this rung samples the episodes through the engine's own
//! [`sample_context_episodes`](InferenceSession::sample_context_episodes),
//! writes the train and held-out sets and the engine's seeded initial weights
//! under the legs directory's `input/<arch>/seed<N>/`, and trains over them
//! with the engine's own [`fit_context_predictor`] — for whichever member of the
//! family `--arch` names (`Cnp`, `AttnCnp`, `Tnp`). A PyTorch twin that loads
//! those files starts from the same parameters and sees the same batches in the
//! same order, so what remains between the two stacks is numerics. A unit is
//! `seed<N>`: the seed fixes the task split and the initial weights.
//!
//! A leg carries every optimizer step's wall-clock (`iter_wall_s`), the
//! process's peak resident set, the held-out loss of the untrained model
//! (`held_out_at_init`) and after every epoch (`trajectory`,
//! `held_out_example_mean` at the end), the train-side probe series the
//! learning premise reads (`train_probe_series`: the objective over the train
//! episodes at init and after every epoch), and the trained head's raw output on
//! every held-out target as `vectors_file` + `vector_dim`, keyed
//! `test{episode}_{row}`, with its digest.
//!
//! ## The served predictor's determinism
//!
//! `predict_with_context_predictor` over a fixed served weight set and a fixed
//! target is byte-identical across runs and across a fresh reload **on a
//! machine** (the engine's inference-only no-gradient contract). The predicted
//! distribution is `f32`, whose exact bits vary by CPU, so no digest is committed:
//! the hermetic tests load the committed trained weight bundle
//! (`baselines/context_predictor_weights/`), predict the committed targets twice
//! on the running box and assert the two digests agree, and assert a regressed
//! serving knob (a wrong `context_k`) moves the digest — so a regression in
//! context assembly, the in-context forward, the distribution adapter or the
//! de-standardisation is caught.
//!
//! `baselines/context_predictor.json` carries the dataset generation spec (so the
//! exact synthetic meta-dataset and its embedding table regenerate
//! deterministically), the training knobs, the trained predictor's `config_json`
//! (architecture, `context_k`, the persisted target scaler) and the committed
//! target keys. `rebuild-context-predictor-spec` re-derives the bundle and the
//! spec from a fresh engine training job.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use candle_core::{Device, Tensor};

use jammi_ai::pipeline::context_predictor::{
    build_context_predictor, fit_context_predictor, score_episodes, ContextArchitecture,
    ContextPredictorTrainConfig, EpisodeBatch, GaussianObjective, PredictiveHead,
};
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::capture::{
    artifact_of, cpu_provenance, file_leg, leg_report, leg_stem, legs_per_point,
    vector_rows_digest, write_vector_rows, Artifact, IterationSeries,
};
use crate::leg::{Facts, Leg, Measured, Measurement, Payload, TrajectoryPoint};
use crate::report::{Nullable, Tiers};

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
/// training spec knobs, the trained predictor's serialised config and the
/// committed target keys. The on-disk `baselines/context_predictor.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPredictorSpec {
    /// Distinct tasks the synthetic linear-function meta-dataset carries.
    pub n_tasks: usize,
    /// Rows per task.
    pub rows_per_task: usize,
    /// The dataset generation seed (the splitmix64 stream feeding the weights,
    /// features, and outcomes).
    pub dataset_seed: u64,
    /// The predictor architecture (`Cnp` / `AttnCnp` / `Tnp`).
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
    /// it). Captured from the rebuild's trained model; the tests register a model
    /// row with it so the committed weights reload exactly as trained.
    pub config_json: String,
    /// The target row keys the predict digest is folded over — a deterministic
    /// prefix of the dataset's rows, so any box re-predicts the same set.
    pub target_keys: Vec<String>,
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
/// tests stand up the exact dataset the committed weights were trained over.
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
    session.install_query_functions();

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
    let handle = registry.handle_for(&url, None)?;
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
        embedding_table: None,
        candidate_source: jammi_db::store::manifest::ContextCandidateSource::Ann { k: 5 },
        value_columns: Vec::new(),
        aggregator: jammi_db::store::manifest::ContextAggregator::Mean,
        exclude_self: true,
        split: None,
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
                key_column: Some("_row_id"),
                text_columns: None,
            },
            &pairs,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            None,
        )
        .await?;

    Ok((session, dir))
}

/// The committed trained-weight bundle file names a model row is registered
/// over: the safetensors weights and the artifact-store manifest the loader
/// verifies. The bundle is fetched by [`InferenceSession::artifact_store`] before
/// `load_context_predictor` reloads it, and that fetch verifies the manifest, so
/// both files travel.
const BUNDLE_FILES: [&str; 2] = ["model.safetensors", "manifest.json"];

/// The rung this producer serves.
pub const RUNG: &str = "in-process";

/// The training one `predictor-train-run` leg is asked to run.
#[derive(Debug, Clone)]
pub struct PredictorTrainParams {
    /// Where the leg and its predictions are filed, and the unit's inputs under
    /// `input/<arch>/seed<N>/`.
    pub legs_dir: PathBuf,
    /// The predictor architecture (`Cnp` / `AttnCnp` / `Tnp`); the committed
    /// spec's when `None`. Every other knob — dataset, episodes, widths, heads,
    /// layers, optimiser — is the committed spec's for every architecture.
    pub architecture: Option<String>,
    /// The run seed — the unit; the committed spec's when `None`.
    pub seed: Option<u64>,
    /// Passes over the train episodes; the committed spec's when `None`.
    pub epochs: Option<usize>,
    /// Leading optimizer steps kept out of the timing series (they still train).
    pub warmup_steps: usize,
    /// The measured repeat this leg is filed as.
    pub take: usize,
}

/// The `predictor-train-run` leg's payload: what two legs must agree on, and
/// what this rung records about its run.
#[derive(Debug, Clone, Serialize)]
pub struct PredictorTrainRunPayload {
    /// The run seed: the task split's and the initial weights'.
    pub seed: u64,
    /// The predictor architecture.
    pub architecture: String,
    /// The context width `k`.
    pub context_k: usize,
    /// The feature width of an episode's `x`.
    pub feature_dim: usize,
    /// The width of an outcome `y`.
    pub value_dim: usize,
    /// Hidden width.
    pub hidden_dim: usize,
    /// Attention heads the configuration names; `Cnp` builds no attention.
    pub num_heads: usize,
    /// Self-attention layers the configuration names; only `Tnp` builds them.
    pub num_layers: usize,
    /// The head's width.
    pub head_width: usize,
    /// sha256 of the initial weights both stacks start from.
    pub initial_weights_sha256: String,
    /// sha256 of the train episodes both stacks step over.
    pub train_episodes_sha256: String,
    /// sha256 of the held-out episodes both stacks score.
    pub heldout_episodes_sha256: String,
    /// Passes over the train episodes.
    pub epochs: usize,
    /// Targets per episode batch — one step.
    pub batch: usize,
    /// AdamW learning rate.
    pub lr: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// The learning-rate schedule.
    pub schedule: &'static str,
    /// The precision the parameters are held in.
    pub compute_precision: &'static str,
    /// The rung this leg claims.
    pub rung: &'static str,
    /// The unit, `seed<N>`.
    pub unit: String,
    /// The measured repeat.
    pub take: usize,
    /// Global-L2 gradient-clip norm.
    pub grad_clip: f64,
    /// The head and its training objective.
    pub objective: &'static str,
    /// Train episode batches per epoch.
    pub train_episodes: usize,
    /// Held-out test episode batches scored.
    pub heldout_episodes: usize,
    /// Leading steps excluded from the timing series.
    pub warmup_steps: usize,
    /// Optimizer steps timed.
    pub iters_measured: usize,
    /// The trained parameters.
    pub final_weights: Artifact,
    /// The implementation that trained.
    pub trainer: &'static str,
}

impl Payload for PredictorTrainRunPayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("seed", Nullable::NonNull),
        ("architecture", Nullable::NonNull),
        ("context_k", Nullable::NonNull),
        ("feature_dim", Nullable::NonNull),
        ("value_dim", Nullable::NonNull),
        ("hidden_dim", Nullable::NonNull),
        ("num_heads", Nullable::NonNull),
        ("num_layers", Nullable::NonNull),
        ("head_width", Nullable::NonNull),
        ("initial_weights_sha256", Nullable::NonNull),
        ("train_episodes_sha256", Nullable::NonNull),
        ("heldout_episodes_sha256", Nullable::NonNull),
        ("epochs", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("lr", Nullable::NonNull),
        ("weight_decay", Nullable::NonNull),
        ("schedule", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
    ];
}

/// The unit's input files under `legs_dir/input/<arch>/<unit>/`.
pub const INPUT_DIR: &str = "input";
/// The train episodes' file.
pub const TRAIN_EPISODES_FILE: &str = "train_episodes.safetensors";
/// The held-out episodes' file.
pub const HELDOUT_EPISODES_FILE: &str = "heldout_episodes.safetensors";
/// The initial weights' file.
pub const INITIAL_WEIGHTS_FILE: &str = "initial_weights.safetensors";

/// Write an episode set as one safetensors file: for batch `i`, the tensors
/// `{i}.target_x`, `.context_x`, `.context_y`, `.presence` and `.target_y`,
/// all `f32`.
fn write_episodes(
    path: &Path,
    batches: &[EpisodeBatch],
) -> Result<Artifact, Box<dyn std::error::Error>> {
    let tensors: HashMap<String, Tensor> = batches
        .iter()
        .enumerate()
        .flat_map(|(i, batch)| {
            [
                ("target_x", &batch.episode.target_x),
                ("context_x", &batch.episode.context_x),
                ("context_y", &batch.episode.context_y),
                ("presence", &batch.episode.presence),
                ("target_y", &batch.target_y),
            ]
            .map(|(field, tensor)| (format!("{i}.{field}"), tensor.clone()))
        })
        .collect();
    candle_core::safetensors::save(&tensors, path)?;
    artifact_of(path)
}

/// The one target count every batch of `episodes` carries — a step's batch.
fn batch_targets(episodes: &[EpisodeBatch]) -> Result<usize, Box<dyn std::error::Error>> {
    let mut counts = episodes.iter().map(|b| b.target_y.dims1());
    let first = counts.next().ok_or("no train episodes were sampled")??;
    for count in counts {
        if count? != first {
            return Err(
                "the train episodes do not share one target count, so `batch` names no one step"
                    .into(),
            );
        }
    }
    Ok(first)
}

/// Run one `predictor-train-run` leg and file it: sample the committed
/// meta-dataset into episodes through the engine, write them and the engine's
/// seeded initial weights, train with the engine's own fit while probing the
/// held-out and train sets at every epoch, and write the head's output on
/// every held-out target beside the leg. Returns the leg and its file name.
pub async fn run_leg(
    params: &PredictorTrainParams,
) -> Result<(Leg<PredictorTrainRunPayload>, String), Box<dyn std::error::Error>> {
    let committed = ContextPredictorSpec::load()?;
    let spec = ContextPredictorSpec {
        architecture: params
            .architecture
            .clone()
            .unwrap_or_else(|| committed.architecture.clone()),
        spec_seed: params.seed.unwrap_or(committed.spec_seed),
        ..committed
    };
    let config = ContextPredictorTrainConfig {
        epochs: params.epochs.unwrap_or(spec.epochs),
        ..spec.train_config()?
    };
    let rows = build_dataset(&spec);
    let (session, _dir) = dataset_session(&rows).await?;
    let sampled = session.sample_context_episodes(SOURCE_ID, &config).await?;
    let unit = format!("seed{}", config.seed);
    let stem = leg_stem(RUNG, &unit, params.take);

    let input_dir = params
        .legs_dir
        .join(INPUT_DIR)
        .join(&spec.architecture)
        .join(&unit);
    std::fs::create_dir_all(&input_dir)?;
    let train_episodes = write_episodes(&input_dir.join(TRAIN_EPISODES_FILE), &sampled.train)?;
    let heldout_episodes = write_episodes(&input_dir.join(HELDOUT_EPISODES_FILE), &sampled.test)?;
    let device = Device::Cpu;
    let (varmap, predictor) = build_context_predictor(&config, FEATURE_DIM, &device)?;
    let initial_path = input_dir.join(INITIAL_WEIGHTS_FILE);
    varmap.save(&initial_path)?;
    let initial_weights = artifact_of(&initial_path)?;
    let batch = batch_targets(&sampled.train)?;

    let held_out_at_init = score_episodes(&config, &predictor, &sampled.test)?;
    let mut train_probe_series = vec![score_episodes(&config, &predictor, &sampled.train)?];
    let mut trajectory = Vec::with_capacity(config.epochs);
    let started = Instant::now();
    let report = fit_context_predictor(
        &config,
        &varmap,
        &predictor,
        &sampled.train,
        &AtomicBool::new(false),
        |trained, epoch| {
            trajectory.push(TrajectoryPoint {
                epoch,
                held_out_mean: score_episodes(&config, trained, &sampled.test)?,
                run_wall_s_cumulative: Some(started.elapsed().as_secs_f64()),
                steps_wall_s_cumulative: None,
                held_out_tie_fraction: None,
                held_out_batch_partition_sha256: None,
            });
            train_probe_series.push(score_episodes(&config, trained, &sampled.train)?);
            Ok(())
        },
    )?;
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    let mut series = IterationSeries::new(
        params.warmup_steps,
        report.total_steps.saturating_sub(params.warmup_steps),
    );
    report
        .step_seconds
        .iter()
        .for_each(|s| series.record(Duration::from_secs_f64(*s)));

    let final_path = params
        .legs_dir
        .join(format!("{stem}.final_weights.safetensors"));
    varmap.save(&final_path)?;
    let predictions: Vec<(String, Vec<f32>)> = sampled
        .test
        .iter()
        .enumerate()
        .map(|(e, batch)| Ok((e, predictor.forward(&batch.episode)?.to_vec2::<f32>()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?
        .into_iter()
        .flat_map(|(e, heads)| {
            heads
                .into_iter()
                .enumerate()
                .map(move |(row, head)| (format!("test{e}_{row}"), head))
        })
        .collect();
    let vectors = write_vector_rows(&params.legs_dir, &stem, &predictions)?;
    let held_out_example_mean = trajectory
        .last()
        .map(|p| p.held_out_mean)
        .ok_or("a run of no epochs has no final held-out loss")?;

    let payload = PredictorTrainRunPayload {
        seed: config.seed,
        architecture: spec.architecture.clone(),
        context_k: config.context_k,
        feature_dim: FEATURE_DIM,
        value_dim: 1,
        hidden_dim: config.hidden_dim,
        num_heads: config.num_heads,
        num_layers: config.num_layers,
        head_width: vectors.dim,
        initial_weights_sha256: initial_weights.sha256,
        train_episodes_sha256: train_episodes.sha256,
        heldout_episodes_sha256: heldout_episodes.sha256,
        epochs: config.epochs,
        batch,
        lr: config.learning_rate,
        weight_decay: 0.0,
        schedule: "constant",
        compute_precision: "f32",
        rung: RUNG,
        unit,
        take: params.take,
        grad_clip: config.grad_clip,
        objective: "gaussian-crps",
        train_episodes: sampled.train.len(),
        heldout_episodes: sampled.test.len(),
        warmup_steps: params.warmup_steps,
        iters_measured: report.total_steps.saturating_sub(params.warmup_steps),
        final_weights: artifact_of(&final_path)?,
        trainer: "jammi_ai::pipeline::context_predictor::fit_context_predictor",
    };
    let measured = Measured {
        iter_wall_s: Some(series.into_seconds()),
        work: None,
        peak_rss_bytes,
        peak_vram_bytes: Measurement::not_yet_measured("bytes"),
        outcome_digest: Some(vector_rows_digest(&predictions)),
        held_out_example_mean: Some(held_out_example_mean),
        held_out_at_init: Some(held_out_at_init),
        trajectory,
        vectors_file: Some(format!("{stem}.vectors.f32")),
        vector_dim: Some(vectors.dim),
        ..Default::default()
    };
    let facts = Facts {
        train_probe_series: Some(train_probe_series),
        ..Default::default()
    };
    let leg = Leg::new(payload, cpu_provenance(), measured, facts);
    let report = leg_report("predictor-train-run", leg.clone(), |leg| Tiers {
        predictor_train_run: Some(leg),
        ..Default::default()
    });
    let file = file_leg(&params.legs_dir, &stem, &report)?;
    Ok((leg, file))
}

/// `predictor-train-run`'s flags.
#[derive(Debug, Clone, clap::Args)]
pub struct PredictorTrainArgs {
    /// Where the legs are filed, `in-process__seed<N>__r<take>.json` with the
    /// predictions beside, and the inputs under `input/<arch>/seed<N>/`.
    #[arg(long)]
    legs_dir: PathBuf,
    /// `Cnp`, `AttnCnp` or `Tnp`; defaults to the committed spec's.
    #[arg(long)]
    arch: Option<String>,
    /// The seeds to run, comma-separated — one unit each; the committed spec's
    /// seed when omitted.
    #[arg(long, value_delimiter = ',')]
    seeds: Vec<u64>,
    /// Passes over the train episodes; defaults to the committed spec's.
    #[arg(long)]
    epochs: Option<usize>,
    /// Leading optimizer steps kept out of the timing series.
    #[arg(long, default_value_t = 2)]
    warmup_steps: usize,
    /// Measured repeats of each seed, each in a process of its own.
    #[arg(long, default_value_t = 1)]
    takes: usize,
    /// The take a single seed's run is filed as.
    #[arg(long, default_value_t = 1)]
    take: usize,
}

impl PredictorTrainArgs {
    fn params(&self, seed: Option<u64>, take: usize) -> PredictorTrainParams {
        PredictorTrainParams {
            legs_dir: self.legs_dir.clone(),
            architecture: self.arch.clone(),
            seed,
            epochs: self.epochs,
            warmup_steps: self.warmup_steps,
            take,
        }
    }

    /// Run the subcommand: one leg per (seed, take), and print the file names.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let seeds: Vec<Option<u64>> = if self.seeds.is_empty() {
            vec![None]
        } else {
            self.seeds.iter().map(|s| Some(*s)).collect()
        };
        let points: Vec<(Option<u64>, usize)> = seeds
            .iter()
            .flat_map(|&s| (1..=self.takes).map(move |t| (s, t)))
            .collect();
        let first = self.params(points[0].0, if points.len() == 1 { self.take } else { 1 });
        let files = legs_per_point(
            &points,
            async move { run_leg(&first).await.map(|(_, file)| vec![file]) },
            |&(seed, take)| {
                let mut args: Vec<std::ffi::OsString> = vec![
                    "predictor-train-run".into(),
                    "--legs-dir".into(),
                    (&self.legs_dir).into(),
                    "--warmup-steps".into(),
                    self.warmup_steps.to_string().into(),
                    "--take".into(),
                    take.to_string().into(),
                ];
                if let Some(arch) = &self.arch {
                    args.extend(["--arch".into(), arch.into()]);
                }
                if let Some(seed) = seed {
                    args.extend(["--seeds".into(), seed.to_string().into()]);
                }
                if let Some(epochs) = self.epochs {
                    args.extend(["--epochs".into(), epochs.to_string().into()]);
                }
                args
            },
        )
        .await?;
        println!("{}", serde_json::to_string_pretty(&files)?);
        Ok(())
    }
}

/// Re-derive the committed spec and its weight bundle from a fresh training job:
/// regenerate the dataset, train a predictor through the engine's
/// `train_context_predictor`, copy the trained weight bundle into
/// `baselines/context_predictor_weights/`, and capture its `config_json` and the
/// committed target keys. The one-shot that writes
/// `baselines/context_predictor.json` and the bundle; the tests only ever load
/// and re-predict them.
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
    };

    let rows = build_dataset(&spec);
    let (session, _dir) = dataset_session(&rows).await?;
    let config = spec.train_config()?;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)?;
    let job = session.train_context_predictor(SOURCE_ID, &config).await?;
    job.wait().await?;

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
    let prefix_url = record
        .location
        .as_ref()
        .ok_or("rebuild: trained predictor has no location")?
        .bundle_url()?;
    let local = session.artifact_store().fetch_artifact(&prefix_url).await?;
    let weights_dir = ContextPredictorSpec::weights_dir();
    if weights_dir.exists() {
        std::fs::remove_dir_all(&weights_dir)?;
    }
    std::fs::create_dir_all(&weights_dir)?;
    for file in BUNDLE_FILES {
        std::fs::copy(local.dir().join(file), weights_dir.join(file))?;
    }

    // Stage 3: the committed targets — a deterministic prefix of the dataset rows.
    spec.target_keys = rows
        .iter()
        .take(params.target_count)
        .map(|r| r.id.clone())
        .collect();

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
    /// How many of the dataset's rows are committed as predict targets.
    pub target_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    use jammi_ai::pipeline::context_predictor::{ContextServeOptions, PredictedDistribution};
    use jammi_db::catalog::model_repo::RegisterModelParams;
    use jammi_db::model_task::ModelTask;

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
                external_location: Some(&artifact),
                config_json: Some(config_json),
            })
            .await?;
        Ok(())
    }

    /// Predict over the committed targets through the engine's real
    /// `predict_with_context_predictor`, given a session that already carries the
    /// dataset embedding table and a model row pointing at the committed weights.
    /// Returns the predicted distributions in target order.
    ///
    /// `model_id` selects which registered model row (and so which `config_json`) the
    /// serve loads under — the committed config for the gate, a perturbed config for
    /// the teeth test.
    async fn predict_committed_targets(
        session: &Arc<InferenceSession>,
        model_id: &str,
        target_keys: &[String],
    ) -> Result<Vec<PredictedDistribution>, Box<dyn std::error::Error>> {
        let served = session
            .load_context_predictor(model_id, SOURCE_ID, ContextServeOptions::default())
            .await?;
        let mut predictions = Vec::with_capacity(target_keys.len());
        for key in target_keys {
            predictions.push(session.predict_with_context_predictor(&served, key).await?);
        }
        Ok(predictions)
    }

    /// Fold a predict digest: stand up the dataset session, register the committed
    /// weight bundle, predict the committed targets through the real engine, and digest
    /// the predictions.
    ///
    /// `weights_dir` is passed explicitly so a caller can point at the committed bundle
    /// under a perturbed config; the default uses [`ContextPredictorSpec::weights_dir`].
    async fn fold_predict_digest(
        spec: &ContextPredictorSpec,
        weights_dir: &std::path::Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let rows = build_dataset(spec);
        let (session, _dir) = dataset_session(&rows).await?;
        register_committed_weights(&session, PREDICTOR_MODEL_ID, weights_dir, &spec.config_json)
            .await?;
        let predictions =
            predict_committed_targets(&session, PREDICTOR_MODEL_ID, &spec.target_keys).await?;
        Ok(digest(&predictions))
    }

    /// The committed spec is well-formed: a structured meta-dataset, a positive
    /// context width, a non-empty config and target set.
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

    /// The teeth, GATE-FAILS direction (an assertion must be able to fail).
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
        let correct = predict_committed_targets(&session, PREDICTOR_MODEL_ID, &spec.target_keys)
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
        let perturbed =
            predict_committed_targets(&session, "ctx-predictor-perturbed", &spec.target_keys)
                .await
                .expect("perturbed predict runs");
        assert_ne!(
            digest(&perturbed),
            baseline,
            "a wrong context_k must trip the predict digest (else a serving-knob regression slips)"
        );
    }

    /// For every architecture: a leg is filed under the ladder's name with its
    /// inputs and predictions, carries one timing per optimizer step and a
    /// held-out point per epoch anchored at init, starts from the seeded initial
    /// weights (the same file on every run), and its training moves the weights.
    #[tokio::test(flavor = "multi_thread")]
    async fn leg_carries_the_trajectory_from_seeded_initial_weights() {
        let run = |dir: &Path, architecture: &str| {
            let params = PredictorTrainParams {
                legs_dir: dir.to_path_buf(),
                architecture: Some(architecture.to_string()),
                seed: Some(7),
                epochs: Some(3),
                warmup_steps: 1,
                take: 1,
            };
            async move { run_leg(&params).await.expect("predictor leg runs") }
        };
        let mut initial_weights = std::collections::HashSet::new();
        for architecture in ["Cnp", "AttnCnp", "Tnp"] {
            let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            let (first, file) = run(a.path(), architecture).await;
            let (second, _) = run(b.path(), architecture).await;

            assert_eq!(file, "in-process__seed7__r1.json");
            assert!(a.path().join(&file).is_file());
            assert!(a
                .path()
                .join(INPUT_DIR)
                .join(architecture)
                .join("seed7")
                .join(INITIAL_WEIGHTS_FILE)
                .is_file());
            assert_eq!(first.payload.architecture, architecture);
            let steps = first.payload.train_episodes * first.payload.epochs;
            assert_eq!(
                first.measured.iter_wall_s.as_ref().unwrap().len(),
                steps - 1
            );
            assert_eq!(first.measured.trajectory.len(), 3);
            assert_eq!(first.facts.train_probe_series.as_ref().unwrap().len(), 4);
            assert!(first.measured.held_out_at_init.is_some());
            assert_eq!(
                first.measured.held_out_example_mean,
                Some(first.measured.trajectory[2].held_out_mean)
            );
            assert!(
                first.payload.heldout_episodes > 0,
                "held-out tasks to predict over"
            );
            assert_eq!(
                first.payload.initial_weights_sha256, second.payload.initial_weights_sha256,
                "{architecture}: the initial weights are a pure function of the seed"
            );
            assert_eq!(
                first.payload.train_episodes_sha256,
                second.payload.train_episodes_sha256
            );
            assert_eq!(
                first.measured.held_out_at_init, second.measured.held_out_at_init,
                "{architecture}: the untrained probe is a forward over identical inputs"
            );
            assert_ne!(
                first.payload.final_weights.sha256, first.payload.initial_weights_sha256,
                "{architecture}: training moved the weights"
            );
            initial_weights.insert(first.payload.initial_weights_sha256);
        }
        assert_eq!(
            initial_weights.len(),
            3,
            "each architecture has its own parameters"
        );
    }

    /// The ladder runs end to end over filed legs and reaches its seeded-loss
    /// verdict: the edge's twelve seed units each carry a held-out trajectory
    /// anchored at init and a train-side probe series that descends. The
    /// reference rung's legs are stand-ins — this rung's own legs filed under
    /// `torch` with every held-out loss lowered by a known shift — so the run
    /// exercises the comparator's path over the leg shape, not a second
    /// trainer, and the verdict it reaches is known: twelve paired differences,
    /// all in the direction of the shift.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_ladder_reaches_a_seeded_loss_verdict_over_filed_legs() {
        use crate::ladder::definition::Workload;
        use crate::ladder::verdict::OutcomeVerdict;
        use crate::ladder::{run_ladder, Axis, LadderArgs};
        let dir = tempfile::tempdir().unwrap();
        let legs = dir.path().join("legs");
        for seed in 1..=12u64 {
            let (_, file) = run_leg(&PredictorTrainParams {
                legs_dir: legs.clone(),
                architecture: None,
                seed: Some(seed),
                epochs: Some(2),
                warmup_steps: 0,
                take: 1,
            })
            .await
            .expect("predictor leg runs");
            let mut report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(legs.join(&file)).unwrap()).unwrap();
            let block = &mut report["tiers"]["predictor_train_run"];
            const SHIFT: f64 = 0.01;
            block["held_out_example_mean"] =
                (block["held_out_example_mean"].as_f64().unwrap() - SHIFT).into();
            for point in block["trajectory"].as_array_mut().unwrap() {
                point["held_out_mean"] = (point["held_out_mean"].as_f64().unwrap() - SHIFT).into();
            }
            std::fs::write(
                legs.join(file.replacen("in-process__", "torch__", 1)),
                serde_json::to_vec(&report).unwrap(),
            )
            .unwrap();
        }
        let verdict = run_ladder(&LadderArgs {
            workload: Workload::PredictorTrainRun,
            legs_dir: legs,
            out: None,
            from: None,
            to: None,
            axes: vec![Axis::Outcome],
            waive_control: true,
            law_dir: None,
            mutants: vec![],
            revision: None,
        })
        .unwrap();
        assert!(verdict.refusals.is_empty(), "{:?}", verdict.refusals);
        assert_eq!(verdict.edges.len(), 1);
        let edge = &verdict.edges[0];
        match &edge.outcome {
            Some(OutcomeVerdict::SeededLoss {
                clean_units,
                sign_test,
                mean_d,
                ..
            }) => {
                assert_eq!(*clean_units, 12);
                assert_eq!(sign_test.map(|s| s.n_pos), Some(12), "{sign_test:?}");
                assert!(
                    mean_d.unwrap() > 0.0,
                    "the engine rung's loss sits above the shifted reference's"
                );
            }
            other => panic!("expected a seeded-loss verdict, got {other:?}"),
        }
        for refusal in &edge.refusals {
            assert!(
                matches!(
                    refusal.refusal,
                    crate::ladder::refusal::Refusal::AssayInsensitive { .. }
                ),
                "an unexpected refusal on the shape: {refusal:?}"
            );
        }
    }
}
