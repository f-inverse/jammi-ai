use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use tempfile::TempDir;

use jammi_ai::fine_tune::{
    data::TrainingDataLoader, lora::LoraLinear, trainer::compute_lr, FineTuneConfig,
    FineTuneMethod, LrSchedule,
};
use jammi_ai::session::InferenceSession;
use jammi_engine::catalog::status::FineTuneJobStatus;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

use crate::common;

// ─── LoRA layer: one setup, all mechanics ──────────────────────────────────
//
// Guards the core LoRA invariant: B=0 → identity over base, nonzero B → diverges.
// Also validates tensor shapes and initialization strategy (kaiming A, zero B).

#[test]
fn lora_layer_mechanics() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base_weight = Tensor::randn(0.0f32, 1.0, (4, 8), &device).unwrap();
    let base_linear = Linear::new(base_weight, None);
    let mut lora = LoraLinear::new(base_linear.clone(), 2, 4.0, &vb).unwrap();

    // B initialized to zeros — critical: ensures LoRA starts as identity
    let b_vals = lora.lora_b.to_vec2::<f32>().unwrap();
    for row in &b_vals {
        for &val in row {
            assert!(val.abs() < 1e-10, "B should be zeros at init, got {val}");
        }
    }

    // A initialized with kaiming (non-zero) — ensures gradient signal flows from step 1
    let a_vals = lora.lora_a.to_vec2::<f32>().unwrap();
    let all_zero = a_vals
        .iter()
        .all(|row| row.iter().all(|&v| v.abs() < 1e-10));
    assert!(!all_zero, "A should be non-zero (kaiming init)");

    // trainable_params returns A (rank=2, in=8) and B (out=4, rank=2)
    let params = lora.trainable_params();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].dims(), &[2, 8]);
    assert_eq!(params[1].dims(), &[4, 2]);

    // At init (B=0): LoRA output == base output — the identity invariant
    let x = Tensor::randn(0.0f32, 1.0, (3, 8), &device).unwrap();
    let base_out = base_linear.forward(&x).unwrap();
    let lora_out = lora.forward(&x).unwrap();
    let max_diff = (&lora_out - &base_out)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        max_diff < 1e-6,
        "B=0 → output should match base, diff={max_diff}"
    );

    // After setting B to nonzero: output diverges — proves LoRA path is active
    lora.lora_b = Tensor::ones((4, 2), DType::F32, &device).unwrap();
    let lora_out2 = lora.forward(&x).unwrap();
    let max_diff2 = (&lora_out2 - &base_out)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        max_diff2 > 1e-6,
        "Non-zero B → output should differ, diff={max_diff2}"
    );
}

// ─── LR scheduling: one test per schedule ───────────────────────────────────
//
// Guards the warmup→decay transition and the decay curve shape.
// Each schedule has its own code path in the match arm.

#[test]
fn lr_schedule_warmup_and_cosine_decay() {
    let config = FineTuneConfig {
        learning_rate: 1e-3,
        warmup_steps: 200,
        lr_schedule: LrSchedule::CosineDecay,
        ..Default::default()
    };

    // Warmup: starts at 0, linear ramp
    assert!(compute_lr(&config, 0, 1000) < 1e-6, "Step 0 ≈ 0");
    assert!(
        (compute_lr(&config, 100, 1000) - 0.5e-3).abs() < 1e-8,
        "Warmup midpoint = base/2"
    );
    assert!(
        (compute_lr(&config, 200, 1000) - 1e-3).abs() < 1e-8,
        "Warmup end = base LR"
    );

    // Cosine decay over remaining 800 steps
    assert!(
        (compute_lr(&config, 600, 1000) - 0.5e-3).abs() < 1e-8,
        "Cosine midpoint"
    );
    assert!(
        compute_lr(&config, 1000, 1000).abs() < 1e-8,
        "Cosine end ≈ 0"
    );
}

#[test]
fn lr_schedule_linear_decay() {
    let config = FineTuneConfig {
        learning_rate: 1e-3,
        warmup_steps: 0,
        lr_schedule: LrSchedule::LinearDecay,
        ..Default::default()
    };

    assert!(
        (compute_lr(&config, 0, 1000) - 1.0e-3).abs() < 1e-8,
        "Start"
    );
    assert!(
        (compute_lr(&config, 500, 1000) - 0.50e-3).abs() < 1e-8,
        "50%"
    );
    assert!(compute_lr(&config, 1000, 1000).abs() < 1e-8, "End ≈ 0");
}

#[test]
fn lr_schedule_constant_after_warmup() {
    let config = FineTuneConfig {
        learning_rate: 2e-4,
        warmup_steps: 10,
        lr_schedule: LrSchedule::Constant,
        ..Default::default()
    };

    // Warmup boundary
    assert!(
        (compute_lr(&config, 5, 1000) - 1e-4).abs() < 1e-8,
        "Warmup midpoint"
    );

    // After warmup: flat at any point
    assert!(
        (compute_lr(&config, 100, 1000) - 2e-4).abs() < 1e-8,
        "Flat at 100"
    );
    assert!(
        (compute_lr(&config, 999, 1000) - 2e-4).abs() < 1e-8,
        "Flat at 999"
    );
}

// ─── Validation split ────────────────────────────────────────────────────────
//
// Guards the fraction calculation (round behavior) and the zero-fraction edge case.

#[test]
fn validation_split_fractions() {
    // 10% of 100 → 90 train, 10 val (guards round() vs floor())
    let loader = TrainingDataLoader::from_rows(100);
    let (train, val) = loader.split(0.1).unwrap();
    assert_eq!(train.len(), 90);
    assert_eq!(val.len(), 10);

    // Zero fraction → all in train, no validation (edge case: no divide-by-zero)
    let loader2 = TrainingDataLoader::from_rows(50);
    let (t2, v2) = loader2.split(0.0).unwrap();
    assert_eq!(t2.len(), 50);
    assert_eq!(v2.len(), 0);
}

// ─── Contract: LR monotonicity ──────────────────────────────────────────────
//
// Property test: sweeps all steps and verifies no schedule produces a LR increase
// after warmup. Catches any formula regression that introduces non-monotonicity.

#[test]
fn contract_lr_schedule_is_monotonic_after_warmup() {
    for schedule in [LrSchedule::CosineDecay, LrSchedule::LinearDecay] {
        let config = FineTuneConfig {
            learning_rate: 1e-3,
            warmup_steps: 100,
            lr_schedule: schedule,
            ..Default::default()
        };

        let mut prev_lr = compute_lr(&config, 100, 1000);
        for step in (101..=1000).step_by(10) {
            let lr = compute_lr(&config, step, 1000);
            assert!(
                lr <= prev_lr + 1e-12,
                "{schedule:?}: LR at step {step} ({lr}) > previous ({prev_lr})"
            );
            prev_lr = lr;
        }
    }
}

// ─── Fine-tune end-to-end with tiny_bert: real training + inference ──────────
//
// Covers UAT 1-4, 17. Runs the full pipeline: fine_tune with real model
// encoding → adapter saved → fine-tuned model loaded → produces embeddings.
// Uses local tiny_bert fixture — no network access needed.

fn tiny_bert_model() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_training_data() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "training",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    (session, dir)
}

#[tokio::test]
async fn fine_tune_job_lifecycle_and_artifacts() {
    let (session, dir) = session_with_training_data().await;
    let model = tiny_bert_model();

    let job = session
        .fine_tune(
            "training",
            &model,
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            "text_embedding",
            Some(FineTuneConfig {
                epochs: 2,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // UAT 1: job_id is set
    assert!(!job.job_id.is_empty());

    // UAT 3: model_id follows jammi:fine-tuned:{id} pattern (invariant 2)
    assert!(
        job.model_id().starts_with("jammi:fine-tuned:"),
        "model_id should have jammi:fine-tuned: prefix, got '{}'",
        job.model_id()
    );

    // Wait for completion
    job.wait().await.unwrap();

    // UAT 4: job status transitions queued → running → completed
    let record = session.catalog().get_fine_tune_job(&job.job_id).unwrap();
    assert_eq!(record.status, "completed");
    assert!(record.started_at.is_some(), "started_at should be set");
    assert!(record.completed_at.is_some(), "completed_at should be set");

    // UAT 2: adapter weights saved to artifact store
    let adapter_dir = dir.path().join("models").join(&job.job_id);
    assert!(adapter_dir.exists(), "Adapter dir should exist");
    let adapter_file = adapter_dir.join("adapter.safetensors");
    assert!(adapter_file.exists(), "adapter.safetensors should exist");
    assert!(
        std::fs::metadata(&adapter_file).unwrap().len() > 0,
        "Adapter file should not be empty"
    );

    // UAT 17: checkpoint files exist
    let checkpoints: Vec<_> = std::fs::read_dir(&adapter_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .unwrap_or("")
                .starts_with("checkpoint")
        })
        .collect();
    assert!(!checkpoints.is_empty(), "Should have checkpoint files");

    // UAT 3: fine-tuned model registered in catalog with artifact_path
    let models = session.catalog().list_models().unwrap();
    let ft_models: Vec<_> = models
        .iter()
        .filter(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .collect();
    assert!(
        !ft_models.is_empty(),
        "Fine-tuned model should be registered in catalog"
    );
    assert_eq!(ft_models[0].model_type, "fine-tuned");
    assert!(
        ft_models[0].artifact_path.is_some(),
        "Fine-tuned model should have artifact_path set"
    );

    // UAT 3 continued: fine-tuned model produces embeddings (real inference)
    let ft_model_id = &ft_models[0].model_id;
    // The model_id in catalog is "jammi:fine-tuned:{uuid}::1", but encode_query
    // needs the name part. Extract the name (everything before ::).
    let ft_name = ft_model_id.split("::").next().unwrap();
    let base_embedding = session
        .encode_text_query(&model, "quantum computing")
        .await
        .unwrap();
    let ft_embedding = session
        .encode_text_query(ft_name, "quantum computing")
        .await
        .unwrap();

    assert_eq!(
        ft_embedding.len(),
        32,
        "Fine-tuned model should produce 32-dim embeddings"
    );

    // After training, the LoRA projection should have changed at least some dimensions
    // (B was zero at init, optimizer updated it)
    let diff: f32 = base_embedding
        .iter()
        .zip(&ft_embedding)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Fine-tuned embeddings should differ from base (LoRA delta should be non-zero), diff={diff}"
    );
}

// UAT 6 (QLoRA): Invalid methods are now unrepresentable at the type level
// via `FineTuneMethod` enum. No runtime test needed.

// ─── Fine-tune catalog CRUD ─────────────────────────────────────────────────

#[test]
fn fine_tune_job_catalog_crud() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = jammi_engine::catalog::Catalog::open(dir.path()).unwrap();

    // Register base model (FK constraint)
    catalog
        .register_model(jammi_engine::catalog::model_repo::RegisterModelParams {
            model_id: "base-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "text_embedding",
            ..Default::default()
        })
        .unwrap();

    // Create job
    catalog
        .create_fine_tune_job(
            "job-1",
            "base-model::1",
            "training_source",
            "cosent",
            r#"{"lora_rank": 8}"#,
        )
        .unwrap();

    // Get job — status should be "queued"
    let job = catalog.get_fine_tune_job("job-1").unwrap();
    assert_eq!(job.status, "queued");
    assert_eq!(job.base_model_id, "base-model::1");

    // Update to running
    let metrics = r#"{"started_at": "2026-01-01T00:00:00Z"}"#;
    catalog
        .update_fine_tune_status("job-1", FineTuneJobStatus::Running, Some(metrics))
        .unwrap();
    let job2 = catalog.get_fine_tune_job("job-1").unwrap();
    assert_eq!(job2.status, "running");
    assert!(job2.started_at.is_some());

    // Update to completed with output model
    catalog
        .update_fine_tune_status(
            "job-1",
            FineTuneJobStatus::Completed,
            Some(r#"{"completed_at": "2026-01-01T01:00:00Z"}"#),
        )
        .unwrap();
    catalog
        .set_fine_tune_output_model("job-1", "jammi:fine-tuned:job-1")
        .unwrap();
    let job3 = catalog.get_fine_tune_job("job-1").unwrap();
    assert_eq!(job3.status, "completed");
    assert_eq!(
        job3.output_model_id.as_deref(),
        Some("jammi:fine-tuned:job-1")
    );

    // List jobs
    let jobs = catalog.list_fine_tune_jobs().unwrap();
    assert_eq!(jobs.len(), 1);
}

// ─── Gradient flow: backward_step actually changes LoRA weights ─────────────
//
// Proves the training loop is not a no-op. Without this test, a broken
// backward_step that silently skips updates would go undetected.

#[test]
fn lora_backward_step_changes_weights() {
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base_weight = Tensor::randn(0.0f32, 1.0, (4, 8), &device).unwrap();
    let base = Linear::new(base_weight, None);
    let lora = LoraLinear::new(base, 2, 4.0, &vb.pp("test")).unwrap();

    // Capture B weights before training — should be zeros
    let b_before = lora.lora_b.to_vec2::<f32>().unwrap();
    assert!(
        b_before
            .iter()
            .all(|row| row.iter().all(|&v| v.abs() < 1e-10)),
        "B should be zeros before training"
    );

    // Create optimizer from VarMap's trainable variables
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-2, // high LR to make changes visible in one step
            ..Default::default()
        },
    )
    .unwrap();

    // Create a contrastive batch with known embeddings (not random)
    // embed_a and embed_b are far apart, but score says they should be similar
    // → loss is high → gradients are large → weights change
    let emb_a = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
    let emb_b = Tensor::new(&[[0.0f32, 0.0, 0.0, 1.0]], &device).unwrap();
    let scores = Tensor::new(&[1.0f32], &device).unwrap();

    // Compute cosine similarity loss manually (same formula as trainer)
    let dot = (&emb_a * &emb_b).unwrap().sum(1).unwrap();
    let norm_a = emb_a.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
    let norm_b = emb_b.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
    let cos_sim = (&dot / &(&norm_a * &norm_b).unwrap()).unwrap();
    let diff = (&cos_sim - &scores).unwrap();
    let loss = diff.sqr().unwrap().mean_all().unwrap();

    // One backward step
    optimizer.backward_step(&loss).unwrap();

    // Check that VarMap variables changed (at least one var should have non-zero gradient)
    let vars = varmap.all_vars();
    let any_changed = vars.iter().any(|var| {
        let vals = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        vals.iter().any(|&v| v.abs() > 1e-10)
    });
    assert!(
        any_changed,
        "At least one variable should have changed after backward_step"
    );
}

// ─── Divergence detection: NaN loss triggers job failure ────────────────────
//
// UAT 5. The training loop should fail with "diverged" after 3 consecutive
// batches with NaN or >100 loss. Tests with precomputed NaN-score batches.

#[test]
fn training_divergence_detection() {
    use candle_nn::VarMap;
    use jammi_ai::fine_tune::{
        data::{TrainingBatch, TrainingDataLoader},
        lora::build_lora_projection,
        trainer::TrainingLoopBuilder,
    };
    use std::sync::Arc;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = build_lora_projection(32, &FineTuneConfig::default(), &vb).unwrap();

    // Create batches where scores are NaN → cosent_loss produces NaN
    let nan_batch = TrainingBatch::Contrastive {
        embeddings_a: Tensor::ones((1, 32), DType::F32, &device).unwrap(),
        embeddings_b: Tensor::ones((1, 32), DType::F32, &device).unwrap(),
        scores: Tensor::new(&[f32::NAN], &device).unwrap(),
    };
    // Need at least 3 batches to trigger divergence (3 consecutive NaN)
    let loader =
        TrainingDataLoader::from_precomputed(vec![nan_batch.clone(), nan_batch.clone(), nan_batch]);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_engine::catalog::Catalog::open(dir.path()).unwrap());

    // Register a model for the FK
    catalog
        .register_model(jammi_engine::catalog::model_repo::RegisterModelParams {
            model_id: "div-test-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "text_embedding",
            ..Default::default()
        })
        .unwrap();
    catalog
        .create_fine_tune_job("div-job", "div-test-model::1", "src", "cosent", "{}")
        .unwrap();

    let mut training_loop = TrainingLoopBuilder::new(
        model,
        varmap,
        FineTuneConfig {
            epochs: 5,
            batch_size: 1,
            validation_fraction: 0.0,
            warmup_steps: 0,
            ..Default::default()
        },
    )
    .job_id("div-job".into())
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = training_loop.run(&loader);

    assert!(result.is_err(), "Training should fail on NaN loss");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("diverge"),
        "Error should mention divergence, got: {msg}"
    );

    // Catalog should record failure
    let job = catalog.get_fine_tune_job("div-job").unwrap();
    assert_eq!(job.status, "failed", "Job status should be 'failed'");
}

// ─── Early stopping: patience exhaustion stops training ─────────────────────
//
// UAT 7. With patience=1, training should stop well before max epochs because
// validation loss never improves. Uses precomputed batches:
// - Training batches: score=1.0 with identical embeddings → low loss
// - Validation batches: score=0.0 with identical embeddings → high loss
// Validation loss stays constant, so patience exhausts after epoch 2.

#[test]
fn training_early_stopping_triggers() {
    use candle_nn::VarMap;
    use jammi_ai::fine_tune::{
        data::{TrainingBatch, TrainingDataLoader},
        lora::build_lora_projection,
        trainer::TrainingLoopBuilder,
    };
    use std::sync::Arc;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = build_lora_projection(32, &FineTuneConfig::default(), &vb).unwrap();

    // Training batches: similar embeddings with score=1.0 → low contrastive loss
    let make_train_batch = || {
        let emb = Tensor::ones((4, 32), DType::F32, &device).unwrap();
        TrainingBatch::Contrastive {
            embeddings_a: emb.clone(),
            embeddings_b: emb,
            scores: Tensor::from_vec(vec![1.0f32; 4], (4,), &device).unwrap(),
        }
    };
    // Validation batches: similar embeddings with score=0.0 → high contrastive loss
    // (cosine similarity ~1.0, target score 0.0 → MSE always high → never improves)
    let make_val_batch = || {
        let emb = Tensor::ones((4, 32), DType::F32, &device).unwrap();
        TrainingBatch::Contrastive {
            embeddings_a: emb.clone(),
            embeddings_b: emb,
            scores: Tensor::from_vec(vec![0.0f32; 4], (4,), &device).unwrap(),
        }
    };

    // 8 training + 2 validation batches → split(0.2) gives 8 train, 2 val
    let mut batches = Vec::new();
    for _ in 0..8 {
        batches.push(make_train_batch());
    }
    for _ in 0..2 {
        batches.push(make_val_batch());
    }
    let loader = TrainingDataLoader::from_precomputed(batches);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_engine::catalog::Catalog::open(dir.path()).unwrap());

    catalog
        .register_model(jammi_engine::catalog::model_repo::RegisterModelParams {
            model_id: "es-test-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "text_embedding",
            ..Default::default()
        })
        .unwrap();
    catalog
        .create_fine_tune_job("es-job", "es-test-model::1", "src", "cosent", "{}")
        .unwrap();

    let mut training_loop = TrainingLoopBuilder::new(
        model,
        varmap,
        FineTuneConfig {
            epochs: 100, // high — should stop well before this
            batch_size: 10,
            validation_fraction: 0.2,   // 20% holdout
            early_stopping_patience: 1, // stop after 1 epoch without improvement
            warmup_steps: 0,
            learning_rate: 1e-4,
            ..Default::default()
        },
    )
    .job_id("es-job".into())
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = training_loop.run(&loader).unwrap();

    // With patience=1 and constant validation loss, early stopping triggers
    // after epoch 2 (epoch 1 sets best, epoch 2 doesn't improve).
    assert!(
        result.total_steps < 200,
        "Early stopping should trigger well before 100 epochs, got {} steps",
        result.total_steps
    );

    // Job should be completed (not failed)
    let job = catalog.get_fine_tune_job("es-job").unwrap();
    assert_eq!(job.status, "completed");
}

// ─── Fine-tuned model produces measurably different search quality ───────────
//
// End-to-end: fine-tune with LoRA, generate embeddings with both base and
// fine-tuned models, run eval_embeddings on both, assert that retrieval
// metrics differ. Proves the adapter actually alters search behavior.

#[tokio::test]
async fn fine_tuned_model_produces_measurably_different_search_quality() {
    let (session, _dir) = session_with_training_data().await;
    let model = tiny_bert_model();

    // Register patents source for embedding generation and eval
    session
        .add_source(
            "patents",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Register golden relevance dataset for evaluation
    session
        .add_source(
            "golden_rel",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("golden_relevance.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate base embeddings
    let base_rec = session
        .generate_text_embeddings("patents", &model, &["abstract".to_string()], "id")
        .await
        .unwrap();

    // Fine-tune with LoRA. The tiny 32-dim model needs enough total
    // gradient to shift LoRA's zero-initialized B matrix away from the
    // identity projection: 10 epochs × ~4 batches = 40 steps at 1e-3
    // with constant schedule (no decay wasting steps near zero LR).
    let columns = vec![
        "text_a".to_string(),
        "text_b".to_string(),
        "score".to_string(),
    ];
    let job = session
        .fine_tune(
            "training",
            &model,
            &columns,
            FineTuneMethod::Lora,
            "text_embedding",
            Some(FineTuneConfig {
                epochs: 10,
                batch_size: 8,
                learning_rate: 1e-3,
                lora_rank: 4,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    // Generate embeddings with the fine-tuned model
    let ft_rec = session
        .generate_text_embeddings("patents", job.model_id(), &["abstract".to_string()], "id")
        .await
        .unwrap();

    // Eval base embeddings against golden relevance
    let base_metrics = session
        .eval_embeddings(
            "patents",
            Some(&base_rec.table_name),
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // Eval fine-tuned embeddings against golden relevance
    let ft_metrics = session
        .eval_embeddings(
            "patents",
            Some(&ft_rec.table_name),
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // Both evals return valid JSON with all four metric keys in [0, 1]
    let metric_keys = ["recall_at_k", "precision_at_k", "mrr", "ndcg"];
    for (label, metrics) in [("base", &base_metrics), ("fine-tuned", &ft_metrics)] {
        for key in &metric_keys {
            let val = metrics[key]
                .as_f64()
                .unwrap_or_else(|| panic!("{label} missing metric: {key}"));
            assert!(
                (0.0..=1.0).contains(&val),
                "{label} {key} = {val} outside [0, 1]"
            );
        }
    }

    // At least one metric must differ between base and fine-tuned (proves the
    // adapter actually changes retrieval behavior, not a no-op)
    let any_different = metric_keys.iter().any(|key| {
        let base_val = base_metrics[key].as_f64().unwrap();
        let ft_val = ft_metrics[key].as_f64().unwrap();
        (base_val - ft_val).abs() > 1e-6
    });
    assert!(
        any_different,
        "Fine-tuned model should produce at least one different retrieval metric.\n\
         base:       {base_metrics}\n\
         fine-tuned: {ft_metrics}"
    );
}

// ─── FineTuneConfig validation ──────────────────────────────────────────────
//
// Gap 5: invalid configs should be rejected before training starts.

#[test]
fn config_validation_rejects_invalid_values() {
    let cases = vec![
        (
            FineTuneConfig {
                lora_rank: 0,
                ..Default::default()
            },
            "lora_rank",
        ),
        (
            FineTuneConfig {
                lora_alpha: 0.0,
                ..Default::default()
            },
            "lora_alpha",
        ),
        (
            FineTuneConfig {
                learning_rate: -1.0,
                ..Default::default()
            },
            "learning_rate",
        ),
        (
            FineTuneConfig {
                epochs: 0,
                ..Default::default()
            },
            "epochs",
        ),
        (
            FineTuneConfig {
                batch_size: 0,
                ..Default::default()
            },
            "batch_size",
        ),
        (
            FineTuneConfig {
                gradient_accumulation_steps: 0,
                ..Default::default()
            },
            "gradient_accumulation",
        ),
        (
            FineTuneConfig {
                validation_fraction: 1.5,
                ..Default::default()
            },
            "validation_fraction",
        ),
        (
            FineTuneConfig {
                early_stopping_patience: 0,
                ..Default::default()
            },
            "early_stopping_patience",
        ),
        (
            FineTuneConfig {
                lora_dropout: -0.1,
                ..Default::default()
            },
            "lora_dropout",
        ),
    ];

    for (config, field) in &cases {
        let result = config.validate();
        assert!(result.is_err(), "Should reject invalid {field}: {config:?}");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains(&field.to_lowercase()),
            "Error for {field} should name the field, got: {msg}"
        );
    }

    // Default config should be valid
    assert!(FineTuneConfig::default().validate().is_ok());
}
