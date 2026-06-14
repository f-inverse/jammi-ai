//! W5-PR0b acceptance test — CPU fine-tuning init is bit-reproducible.
//!
//! A LoRA fine-tune on `Device::Cpu` must be a pure function of
//! `(seed, source rows, config)`: two runs with the SAME seed produce
//! BYTE-IDENTICAL adapter weights, and a DIFFERENT seed produces different
//! weights (so the seed is provably honoured, not ignored). The four
//! nondeterminism sources PR0b fixes — unseeded LoRA Kaiming/Gaussian init
//! (#1/#2), unseeded dropout (#3), and unstable source row order (#6) — would
//! each break this.
//!
//! SCOPE OF THIS FILE. These tests drive the loop with PRECOMPUTED
//! `TrainingBatch`es, which the trainer routes straight to `compute_loss` over
//! the raw embeddings — `LoraLinear::forward` is never called. So this file pins
//! only the **seeded init** halves of the contract (#1/#2): same-seed byte
//! equality and different-seed divergence of the initialised adapter, plus the
//! isolated init probe below. The **trained-forward** halves of the contract —
//! that the adapter genuinely TRAINS off zero-init and that the seeded **dropout**
//! mask (#3) is reproducible on the executed `forward` — are proven by the
//! in-crate module `fine_tune::trainer::determinism_through_forward`, which drives
//! the production `forward` → `regress` → `compute_loss` → AdamW dispatch (private
//! to the crate, hence in-crate). That module also carries the non-vacuity check:
//! swapping the seeded mask for candle's unseeded `ops::dropout` makes its
//! same-seed byte-equality assertion fail.
//!
//! The design (§4) measured candle CPU ops bit-identical across thread counts,
//! so these tests do NOT pin `RAYON_NUM_THREADS`; they run on a multi-thread
//! runtime. If a same-seed run is ever not byte-identical, that is a
//! design-invalidating finding — do not loosen to a tolerance.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use jammi_ai::fine_tune::{
    data::{TrainingBatch, TrainingDataLoader},
    lora::build_projection_head,
    target::TrainingTarget,
    trainer::TrainingLoopBuilder,
    EarlyStoppingMetric, FineTuneConfig,
};
use jammi_ai::model::ModelTask;

/// Catalog + claimed-job boilerplate so a `TrainingLoop` can run (it stamps
/// lease-guarded run-start metrics). Mirrors `ft_correctness_sweep`'s setup.
async fn claimed_loop_env(tag: &str) -> (Arc<jammi_db::catalog::Catalog>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    let model_id = format!("{tag}-model");
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
            job_id: tag,
            base_model_id: &format!("{model_id}::1"),
            training_source: "src",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    catalog
        .claim_next_training_job(&format!("{tag}-worker"), std::time::Duration::from_secs(60))
        .await
        .unwrap()
        .expect("queued job claimable");
    (catalog, dir)
}

/// A small, deterministic, learnable contrastive batch (orthogonal a/b with
/// score 1.0 → non-zero cosent loss, gradients flow).
fn learnable_batch(device: &Device) -> TrainingBatch {
    let a = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], device).unwrap();
    let b = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], device).unwrap();
    TrainingBatch::Contrastive {
        embeddings_a: a,
        embeddings_b: b,
        scores: Tensor::new(&[1.0f32], device).unwrap(),
    }
}

/// The fine-tune config under test. `lora_dropout > 0` is set for parity with a
/// real config, but note the precomputed-batch path here never calls
/// `LoraLinear::forward`, so dropout is NOT exercised by this file (see the
/// module header) — that coverage lives in `determinism_through_forward`.
/// Everything else is deterministic small-loop settings.
fn determinism_config(seed: u64) -> FineTuneConfig {
    FineTuneConfig {
        seed,
        epochs: 3,
        batch_size: 1,
        validation_fraction: 0.0,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        lora_dropout: 0.1,
        early_stopping_metric: EarlyStoppingMetric::TrainLoss,
        early_stopping_patience: 10_000,
        learning_rate: 1e-3,
        ..Default::default()
    }
}

/// Run one full fine-tune on CPU at `seed` and return the saved
/// `adapter.safetensors` bytes — the exact artifact the worker publishes. The
/// loop runs on a blocking thread (CPU-bound); the bytes are read back inside
/// the closure so the result tempdir need not escape.
async fn run_and_capture_adapter_bytes(tag: &str, seed: u64) -> Vec<u8> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = determinism_config(seed);
    let head = build_projection_head(4, &config, &varmap, &vb).unwrap();

    // 4 identical learnable batches per epoch — a fixed, ordered loader (no
    // source read here, so #6's ORDER BY is irrelevant to this in-memory path;
    // the init + dropout seeding is what these bytes prove).
    let batches: Vec<_> = (0..4).map(|_| learnable_batch(&device)).collect();
    let loader = TrainingDataLoader::from_precomputed(batches);

    let (catalog, dir) = claimed_loop_env(tag).await;

    let mut tl = TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
        .job_id(tag.to_string())
        .worker_id(format!("{tag}-worker"))
        .catalog(Arc::clone(&catalog))
        .artifact_dir(dir.path().to_path_buf())
        .build()
        .unwrap();

    tokio::task::spawn_blocking(move || {
        let result = tl.run(&loader).unwrap();
        std::fs::read(result.artifact_dir.path().join("adapter.safetensors")).unwrap()
    })
    .await
    .unwrap()
}

/// (a) Two fine-tunes with the SAME seed produce byte-identical adapter weights.
/// On this precomputed-batch path this pins the seeded INIT (#1/#2): it fails
/// unless the per-parameter init streams are keyed by name, not `VarMap` order
/// (so two independent processes' `VarMap` iteration order cannot perturb the
/// result). The seeded-dropout half (#3) is proven in
/// `determinism_through_forward`, which drives the real `forward`.
#[tokio::test(flavor = "multi_thread")]
async fn same_seed_produces_byte_identical_adapter() {
    let a = run_and_capture_adapter_bytes("det-same-a", 12345).await;
    let b = run_and_capture_adapter_bytes("det-same-b", 12345).await;
    assert_eq!(
        a,
        b,
        "same-seed CPU fine-tunes must produce byte-identical adapter.safetensors \
         ({} vs {} bytes) — an unseeded init/dropout source remains",
        a.len(),
        b.len()
    );
}

/// (b) A DIFFERENT seed produces different weights — guards against the seed
/// being silently ignored (which would make (a) pass vacuously).
#[tokio::test(flavor = "multi_thread")]
async fn different_seed_produces_different_adapter() {
    let a = run_and_capture_adapter_bytes("det-diff-a", 12345).await;
    let b = run_and_capture_adapter_bytes("det-diff-b", 67890).await;
    assert_ne!(
        a, b,
        "different seeds must produce different adapters — the seed is being ignored"
    );
}

/// Init-only determinism, isolated from the training loop: building the same
/// projection head twice at one seed yields byte-identical LoRA A (Kaiming) and
/// B tensors, and a different seed perturbs A. Pinpoints #1/#2 directly.
#[tokio::test(flavor = "multi_thread")]
async fn seeded_init_is_reproducible_and_seed_sensitive() {
    let device = Device::Cpu;

    let build = |seed: u64| {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = FineTuneConfig {
            seed,
            // Gaussian init draws BOTH A and B from seeded streams — the
            // strongest init-seeding probe.
            init_lora_weights: jammi_lora::LoraInitMode::Gaussian,
            ..Default::default()
        };
        let head = build_projection_head(8, &cfg, &varmap, &vb).unwrap();
        let w = TrainingTarget::ProjectionHead { head }
            .named_trainable_weights()
            .unwrap();
        let a: Vec<f32> = w["projection.lora_a"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = w["projection.lora_b"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        (a, b)
    };

    let (a1, b1) = build(2024);
    let (a2, b2) = build(2024);
    assert_eq!(a1, a2, "same-seed Gaussian LoRA A must be byte-identical");
    assert_eq!(b1, b2, "same-seed Gaussian LoRA B must be byte-identical");

    let (a3, _) = build(2025);
    assert_ne!(a1, a3, "a different seed must change the LoRA A draw");
}
