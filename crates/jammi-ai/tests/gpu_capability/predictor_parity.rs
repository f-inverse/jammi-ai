//! P1 — CPU↔GPU parity for the context-predictor `predict` forward pass.
//!
//! One predictor is trained over a synthetic linear-function meta-dataset (CPU
//! is fine for *producing* the predictor — the parity check is on the served
//! forward, not on training), and its weights are persisted to the artifact
//! store. Two fresh sessions then open over the *same* artifact dir — one
//! CPU-pinned, one GPU-pinned — load that *same* persisted predictor, and serve
//! the *same* targets. Because both load byte-identical weights, the predicted
//! distributions must agree within the parity tolerance: the GPU forward
//! reproduces the CPU forward of the in-context predictor.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_ai::pipeline::context_predictor::{
    ContextPredictorTrainConfig, ContextServeOptions, GaussianObjective, PredictedDistribution,
    PredictiveHead,
};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_encoders::ContextArchitecture;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

const FEATURE_DIM: usize = 4;

/// splitmix64 — a deterministic generator (the same one the CPU `it` suite
/// uses), so the synthetic meta-dataset is reproducible with no rng dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

struct Row {
    id: String,
    task: String,
    x: Vec<f32>,
    y: f64,
}

/// A linear-function meta-dataset: `n_tasks` tasks, each with a random weight
/// vector and `rows_per_task` rows, `y = w_task · x`.
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

/// Stand up a session on `artifact_dir` pinned to `device_cpu`, registering the
/// meta-dataset source + its embedding table. Both the CPU and GPU serving
/// sessions reuse this so they read the identical source/embedding state the
/// training session wrote.
async fn meta_session(
    rows: &[Row],
    artifact_dir: &Path,
    device_cpu: bool,
) -> Arc<InferenceSession> {
    let session = if device_cpu {
        harness::cpu_session(artifact_dir).await
    } else {
        harness::gpu_session(artifact_dir).await
    };
    session.register_query_functions();

    let source_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let source_batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.task.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.y).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let source_path = artifact_dir.join("source.parquet");
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

    let pairs: Vec<(String, Vec<f32>)> = rows.iter().map(|r| (r.id.clone(), r.x.clone())).collect();
    let (__d, __e, __i) =
        jammi_test_utils::synthetic_seed_contract("synthetic-embed", "fns", FEATURE_DIM);
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            jammi_db::store::EmbeddingTableSpec {
                source_id: "fns",
                model_id: "synthetic-embed",
                derived_from: None,
                dimensions: FEATURE_DIM,
            },
            &pairs,
            jammi_db::store::manifest::Materialization::new(&__d, &__e, __i),
        )
        .await
        .unwrap();

    session
}

/// Open a *serving* session over an artifact dir the trainer already populated.
/// The source providers + embedding table + trained model are rehydrated from
/// the shared catalog at startup, so the session re-registers nothing — it only
/// needs the SQL functions wired and the device pinned. `load_context_predictor`
/// then builds the predictor on `select_device(self.device_config())`, so the
/// CPU and GPU variants load byte-identical weights onto their respective
/// devices and serve the same forward.
async fn serve_session(artifact_dir: &Path, device_cpu: bool) -> Arc<InferenceSession> {
    let session = if device_cpu {
        harness::cpu_session(artifact_dir).await
    } else {
        harness::gpu_session(artifact_dir).await
    };
    session.register_query_functions();
    session
}

fn spec() -> ContextPredictorTrainConfig {
    ContextPredictorTrainConfig {
        model_id: "ctx-predictor".to_string(),
        architecture: ContextArchitecture::AttnCnp,
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
        epochs: 40,
        learning_rate: 0.005,
        grad_clip: 1.0,
        test_task_fraction: 0.25,
        min_task_count: 4,
        seed: 7,
    }
}

/// Pull `(mean, std)` from a Gaussian served distribution.
fn gaussian(dist: &PredictedDistribution) -> (f32, f32) {
    match dist {
        PredictedDistribution::Gaussian { mean, std } => (*mean, *std),
        other => panic!("expected a Gaussian head, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_forward_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let rows = synthetic_meta_dataset(28, 16, 123);
    let spec = spec();

    // Train once on a CPU session over a shared artifact dir; the worker
    // persists the predictor weights + registers the model in the shared
    // catalog. (CPU is fine for producing the predictor — the parity proof is
    // on the served forward.)
    let shared = TempDir::new().unwrap();
    {
        let trainer = meta_session(&rows, shared.path(), true).await;
        let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&trainer)
            .expect("default worker intervals are valid");
        let job = trainer.train_context_predictor("fns", &spec).await.unwrap();
        job.wait().await.unwrap();
        assert_eq!(job.model_id(), "ctx-predictor");
    }

    // The targets to serve: one row per task (a spread across the held data).
    let targets: Vec<String> = (0..28).map(|t| format!("t{t}_r0")).collect();

    // CPU serving session over the shared dir: rehydrates the source + the
    // trained model, loads the persisted predictor onto the CPU.
    let cpu = serve_session(shared.path(), true).await;
    let cpu_served = cpu
        .load_context_predictor("ctx-predictor", "fns", ContextServeOptions::default())
        .await
        .unwrap();

    // GPU serving session over the *same* shared dir: loads the *same* persisted
    // weights onto the GPU and serves the identical forward.
    let gpu = serve_session(shared.path(), false).await;
    let gpu_served = gpu
        .load_context_predictor("ctx-predictor", "fns", ContextServeOptions::default())
        .await
        .unwrap();

    let mut worst_mean_abs = 0.0f64;
    let mut worst_std_abs = 0.0f64;
    let mut means_cpu = Vec::new();
    let mut means_gpu = Vec::new();
    for key in &targets {
        let c = gaussian(
            &cpu.predict_with_context_predictor(&cpu_served, key)
                .await
                .unwrap(),
        );
        let g = gaussian(
            &gpu.predict_with_context_predictor(&gpu_served, key)
                .await
                .unwrap(),
        );
        let mean_abs = (c.0 as f64 - g.0 as f64).abs();
        let std_abs = (c.1 as f64 - g.1 as f64).abs();
        tracing::info!(
            key,
            cpu_mean = c.0,
            gpu_mean = g.0,
            cpu_std = c.1,
            gpu_std = g.1,
            mean_abs,
            std_abs,
            "predict parity"
        );
        assert!(
            mean_abs <= harness::ELEMENTWISE_ABS_TOL,
            "predict[{key}]: μ diverged CPU {} vs GPU {} (|Δ| {mean_abs})",
            c.0,
            g.0
        );
        assert!(
            std_abs <= harness::ELEMENTWISE_ABS_TOL,
            "predict[{key}]: σ diverged CPU {} vs GPU {} (|Δ| {std_abs})",
            c.1,
            g.1
        );
        worst_mean_abs = worst_mean_abs.max(mean_abs);
        worst_std_abs = worst_std_abs.max(std_abs);
        means_cpu.push(c.0);
        means_gpu.push(g.0);
    }

    // The predicted-mean *vectors* over all targets must also align in direction
    // — a coarse cross-check alongside the per-target absolute bounds.
    harness::assert_parity("predict_means", &means_cpu, &means_gpu);
    tracing::info!(
        targets = targets.len(),
        worst_mean_abs,
        worst_std_abs,
        "context-predictor predict forward parity"
    );
}
