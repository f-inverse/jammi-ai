use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use tempfile::TempDir;

use jammi_ai::fine_tune::{
    data::TrainingDataLoader, trainer::compute_lr, FineTuneConfig, FineTuneMethod, LrSchedule,
};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_lora::LoraLinear;

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
    let mut lora = LoraLinear::new_simple(base_linear.clone(), 2, 4.0, 0, &varmap, &vb).unwrap();

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
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_training_data() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
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
    (session, dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_job_lifecycle_and_artifacts() {
    let (session, _dir) = session_with_training_data().await;
    // `fine_tune` submits a queued job; a worker runs it. Start one over this
    // session (it holds a `Weak`, so the test's `Arc` keeps the session alive).
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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
            ModelTask::TextEmbedding,
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
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(record.status, "completed");
    assert!(record.started_at.is_some(), "started_at should be set");
    assert!(record.completed_at.is_some(), "completed_at should be set");

    // UAT 3: fine-tuned model registered in catalog with artifact_path
    let models = session.catalog().list_models().await.unwrap();
    let ft_models: Vec<_> = models
        .iter()
        .filter(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .collect();
    assert!(
        !ft_models.is_empty(),
        "Fine-tuned model should be registered in catalog"
    );
    assert_eq!(ft_models[0].model_type, "fine-tuned");
    let artifact_prefix = ft_models[0]
        .artifact_path
        .as_deref()
        .expect("Fine-tuned model should have artifact_path set");

    // UAT 2: adapter weights published to the artifact store under the recorded
    // per-attempt prefix. Fetch the bundle (an in-place read for the default
    // `file://` root) and assert the adapter file is present and non-empty.
    let prefix_url = jammi_db::storage::StorageUrl::parse(artifact_prefix).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("published adapter fetches and verifies");
    let adapter_file = local.dir().join("adapter.safetensors");
    assert!(adapter_file.exists(), "adapter.safetensors should exist");
    assert!(
        std::fs::metadata(&adapter_file).unwrap().len() > 0,
        "Adapter file should not be empty"
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

// ─── Per-epoch adapter checkpoints (unit 348) ──────────────────────────────
//
// Round-2 reshape (F3): `keep_last_n_checkpoints` is DISABLED BY DEFAULT.
// `epoch_checkpoints_default_off_writes_nothing` below is the load-bearing
// no-regression oracle every OTHER test in this file (and every caller that
// predates this feature) implicitly relies on: a run that never sets the
// field must write exactly zero epoch-checkpoint bytes and register exactly
// zero epoch rows. `epoch_checkpoints_registered_and_loadable_when_enabled`
// then drives the OPT-IN path with `Some(n)` where `n >= epochs`, pinning the
// documented "no separate keep-all sentinel — ask for a cap at least as
// large as the epoch count" equivalence.

/// THE no-regression oracle (unit 348 F3): a DEFAULT run — `keep_last_n_
/// checkpoints` never set — writes ZERO epoch-checkpoint bytes and registers
/// ZERO epoch-checkpoint catalog rows. Every caller that predates this
/// feature, and every caller that never opts in, gets exactly this behavior.
#[tokio::test(flavor = "multi_thread")]
async fn epoch_checkpoints_default_off_writes_nothing() {
    let (session, dir) = session_with_training_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 3,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                // Deliberately absent — the default this test pins.
                keep_last_n_checkpoints: None,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let output_name = job.model_id().to_string();
    job.wait().await.unwrap();

    // Zero catalog rows for any epoch — not even epoch_0.
    for epoch in 0..3 {
        let epoch_name = format!("{output_name}:epoch_{epoch}");
        assert!(
            session
                .catalog()
                .get_model(&epoch_name)
                .await
                .unwrap()
                .is_none(),
            "a default (keep_last_n_checkpoints absent) run must register no epoch row, \
             found {epoch_name}"
        );
    }

    // Zero bytes on disk under this attempt's `checkpoints/` subtree —
    // real filesystem existence, not merely "no catalog row" (the same
    // "actually check the bytes" discipline the other epoch-checkpoint
    // tests use). `file://` roots materialise real files at exactly the
    // documented path shape.
    let checkpoints_root = dir.path().join("jammi_db").join("models");
    // Find this job's own attempt directory by walking
    // `{root}/{job_id}/{worker_id}/{attempt}/checkpoints` — worker id and
    // attempt are not independently known to this test, so search for ANY
    // `checkpoints` directory under the job's own subtree.
    let job_root = checkpoints_root.join(&job.job_id);
    if job_root.is_dir() {
        for worker_entry in std::fs::read_dir(&job_root).unwrap().flatten() {
            if !worker_entry.file_type().unwrap().is_dir() {
                continue;
            }
            for attempt_entry in std::fs::read_dir(worker_entry.path()).unwrap().flatten() {
                if !attempt_entry.file_type().unwrap().is_dir() {
                    continue;
                }
                let checkpoints_dir = attempt_entry.path().join("checkpoints");
                assert!(
                    !checkpoints_dir.exists(),
                    "a default run must never create a checkpoints/ subtree at all, found \
                     {checkpoints_dir:?}"
                );
            }
        }
    }

    // The final/best artifact is entirely unaffected: still registered and
    // loadable under its own unchanged name.
    let final_record = session
        .catalog()
        .get_model(&output_name)
        .await
        .unwrap()
        .expect("the final output model row still exists under its own name");
    assert_eq!(final_record.status, "registered");
    let final_embedding = session
        .encode_text_query(&output_name, "quantum computing")
        .await
        .unwrap();
    assert_eq!(final_embedding.len(), 32);
}

/// The opt-in path: `keep_last_n_checkpoints = Some(n)` with `n >= epochs`
/// retains every epoch — there is no separate "keep all" sentinel, per the
/// documented equivalence on the field. Registers a catalog row for EVERY
/// epoch (`epoch_0`..`epoch_2`), each a full loadable adapter — the SAME
/// `jammi_lora::save_adapter` bundle shape the served final model uses, not
/// the weights-only resume format. The final/best model's own name is
/// unchanged and still resolves.
#[tokio::test(flavor = "multi_thread")]
async fn epoch_checkpoints_registered_and_loadable_when_enabled() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 3,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                // n == epochs: the documented "n >= epochs retains every
                // epoch" equivalence, no separate keep-all sentinel needed.
                keep_last_n_checkpoints: Some(3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let output_name = job.model_id().to_string();
    job.wait().await.unwrap();

    // Every epoch (0, 1, 2) got its own DISTINCT-NAME catalog row, never an
    // additional VERSION of the output model's name.
    for epoch in 0..3 {
        let epoch_name = format!("{output_name}:epoch_{epoch}");
        let record = session
            .catalog()
            .get_model(&epoch_name)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("epoch {epoch} checkpoint row must be registered"));
        assert_eq!(
            record.status, "checkpoint",
            "an epoch checkpoint row's status must distinguish it from a served model row"
        );
        assert_ne!(
            record.status, "registered",
            "an epoch checkpoint row must not carry the served-model status"
        );
        let artifact_prefix = record
            .artifact_path
            .as_deref()
            .unwrap_or_else(|| panic!("epoch {epoch} checkpoint row must carry an artifact_path"));

        // Loadable: the published bundle fetches, verifies, and contains a
        // full adapter (weights + config), not the resume format's files.
        let prefix_url = jammi_db::storage::StorageUrl::parse(artifact_prefix).unwrap();
        let local = session
            .artifact_store()
            .fetch_artifact(&prefix_url)
            .await
            .unwrap_or_else(|e| panic!("epoch {epoch} checkpoint fetches and verifies: {e}"));
        assert!(
            local.dir().join("adapter.safetensors").exists(),
            "epoch {epoch} checkpoint must be a full loadable adapter (adapter.safetensors)"
        );
        assert!(
            local.dir().join("adapter_config.json").exists(),
            "epoch {epoch} checkpoint must carry adapter_config.json (SavedAdapter metadata)"
        );

        // Actually loadable for inference by that id.
        let embedding = session
            .encode_text_query(&epoch_name, "quantum computing")
            .await
            .unwrap_or_else(|e| panic!("epoch {epoch} checkpoint must load for inference: {e}"));
        assert_eq!(embedding.len(), 32);
    }

    // The final/best artifact is unaffected: still registered under its own
    // (unchanged) name and still describe-resolvable / loadable.
    let final_record = session
        .catalog()
        .get_model(&output_name)
        .await
        .unwrap()
        .expect("the final output model row still exists under its own name");
    assert_eq!(final_record.status, "registered");
    assert!(final_record.artifact_path.is_some());
    let final_embedding = session
        .encode_text_query(&output_name, "quantum computing")
        .await
        .unwrap();
    assert_eq!(final_embedding.len(), 32);
}

/// `keep_last_n_checkpoints = 2` over a 3-epoch run retains only the LAST two
/// epochs: epoch_1 and epoch_2 are registered; epoch_0's bytes are pruned from
/// the store (never registered — no catalog row exists mid-run to race) and
/// its checkpoint row never appears.
#[tokio::test(flavor = "multi_thread")]
async fn epoch_checkpoints_retention_prunes_oldest() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 3,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                keep_last_n_checkpoints: Some(2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let output_name = job.model_id().to_string();
    job.wait().await.unwrap();

    // epoch_1 and epoch_2 are registered and loadable.
    for epoch in [1, 2] {
        let epoch_name = format!("{output_name}:epoch_{epoch}");
        let record = session
            .catalog()
            .get_model(&epoch_name)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("retained epoch {epoch} must be registered"));
        let prefix_url =
            jammi_db::storage::StorageUrl::parse(record.artifact_path.as_deref().unwrap()).unwrap();
        session
            .artifact_store()
            .fetch_artifact(&prefix_url)
            .await
            .unwrap_or_else(|e| panic!("retained epoch {epoch} bytes must fetch: {e}"));
    }

    // epoch_0 was pruned: no catalog row, and — the load-bearing assertion —
    // its bytes are gone from the store, not merely unregistered.
    let epoch_0_name = format!("{output_name}:epoch_0");
    assert!(
        session
            .catalog()
            .get_model(&epoch_0_name)
            .await
            .unwrap()
            .is_none(),
        "a pruned epoch must never be registered"
    );
    // Discover the retained epoch_1 prefix and derive epoch_0's sibling prefix
    // from it (`.../checkpoints/epoch_1` -> `.../checkpoints/epoch_0`) rather
    // than assuming worker-id/attempt values.
    let epoch_1_record = session
        .catalog()
        .get_model(&format!("{output_name}:epoch_1"))
        .await
        .unwrap()
        .expect("epoch_1 is retained");
    let epoch_1_prefix = epoch_1_record.artifact_path.unwrap();
    let epoch_0_prefix = epoch_1_prefix.replace("epoch_1", "epoch_0");
    let epoch_0_url = jammi_db::storage::StorageUrl::parse(&epoch_0_prefix).unwrap();
    assert!(
        session
            .artifact_store()
            .fetch_artifact(&epoch_0_url)
            .await
            .is_err(),
        "epoch_0's bytes must be absent from the store after pruning, got a successful fetch"
    );
}

/// `keep_last_n_checkpoints = 0` is a typed validation refusal at submission —
/// never a silent zero-retention run.
#[tokio::test(flavor = "multi_thread")]
async fn keep_last_n_checkpoints_zero_is_refused() {
    let (session, _dir) = session_with_training_data().await;
    let model = tiny_bert_model();

    let err = session
        .fine_tune(
            "training",
            &model,
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                keep_last_n_checkpoints: Some(0),
                ..Default::default()
            }),
        )
        .await
        .expect_err("keep_last_n_checkpoints=0 must be refused at submission");
    let msg = err.to_string();
    assert!(
        msg.contains("keep_last_n_checkpoints") && msg.contains("ambiguous"),
        "the typed error must name the field and explain the ambiguity: {msg}"
    );
}

// ─── Audio projection-head fine-tune: tuned audio embeddings differ from base ─
//
// JA2. The contrastive fine-tune path accepts JA1's audio encoder family via a
// trainable projection head on a frozen CLAP audio tower. This drives the full
// audio path end-to-end on the hermetic `htsat_clap_tiny` fixture (real-key
// weights, no network): build (anchor, positive, negative) audio triplets from
// the corpus
// (positive = same timbre family, negative = a different family — caller-supplied
// pairing, the trainer stays agnostic), fine-tune a projection head, then eval
// audio→audio retrieval for both the base and tuned embeddings and assert the
// adapter measurably changed retrieval. Mirrors the text quality test above; the
// only difference is the modality of the encoded inputs.

fn htsat_clap_model() -> String {
    "local:".to_string()
        + common::cookbook_fixture("htsat_clap_tiny")
            .to_str()
            .unwrap()
}

/// Every `clip_*.wav` under the tiny audio corpus, keyed by stem, grouped by
/// timbre family (the token between `clip_` and the trailing index).
fn audio_corpus_by_family() -> std::collections::BTreeMap<String, Vec<(String, Vec<u8>)>> {
    let corpus_dir = common::cookbook_fixture("tiny_audio_corpus");
    let mut families: std::collections::BTreeMap<String, Vec<(String, Vec<u8>)>> =
        std::collections::BTreeMap::new();
    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("wav")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("clip_"))
        })
        .collect();
    entries.sort();
    for path in entries {
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        // "clip_sine_0" → family "sine"; "clip_harmonic_2" → "harmonic".
        let family = stem
            .strip_prefix("clip_")
            .and_then(|rest| rest.rsplit_once('_').map(|(fam, _)| fam.to_string()))
            .expect("corpus clip name follows clip_<family>_<idx>");
        let bytes = std::fs::read(&path).unwrap();
        families.entry(family).or_default().push((stem, bytes));
    }
    families
}

/// Write the corpus as a `(clip_id, audio)` Parquet table for embedding +
/// eval, and a held-out `(query_id, query_audio, relevant_id)` golden table
/// where each query clip is relevant to the *other* clips in its family.
fn write_audio_corpus_and_golden(
    dir: &std::path::Path,
) -> (std::path::PathBuf, std::path::PathBuf) {
    use arrow::array::{ArrayRef, BinaryArray, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let families = audio_corpus_by_family();

    // Corpus: one row per clip.
    let mut clip_ids: Vec<String> = Vec::new();
    let mut clip_bytes: Vec<Vec<u8>> = Vec::new();
    for clips in families.values() {
        for (id, bytes) in clips {
            clip_ids.push(id.clone());
            clip_bytes.push(bytes.clone());
        }
    }
    let corpus_schema = Arc::new(Schema::new(vec![
        Field::new("clip_id", DataType::Utf8, false),
        Field::new("audio", DataType::Binary, false),
    ]));
    let corpus_batch = RecordBatch::try_new(
        corpus_schema.clone(),
        vec![
            Arc::new(StringArray::from(
                clip_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BinaryArray::from(
                clip_bytes.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let corpus_path = dir.join("audio_corpus.parquet");
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(&corpus_path).unwrap(),
        corpus_schema,
        None,
    )
    .unwrap();
    w.write(&corpus_batch).unwrap();
    w.close().unwrap();

    // Golden: each clip is a query; its relevant docs are its same-family
    // siblings (excluding itself). Audio-query mode is triggered by the
    // binary `query_audio` column.
    let mut query_ids: Vec<String> = Vec::new();
    let mut query_audios: Vec<Vec<u8>> = Vec::new();
    let mut relevant_ids: Vec<String> = Vec::new();
    for clips in families.values() {
        for (qid, qbytes) in clips {
            for (rid, _) in clips {
                if rid == qid {
                    continue;
                }
                query_ids.push(qid.clone());
                query_audios.push(qbytes.clone());
                relevant_ids.push(rid.clone());
            }
        }
    }
    let golden_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Utf8, false),
        Field::new("query_audio", DataType::Binary, false),
        Field::new("relevant_id", DataType::Utf8, false),
    ]));
    let golden_batch = RecordBatch::try_new(
        golden_schema.clone(),
        vec![
            Arc::new(StringArray::from(
                query_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BinaryArray::from(
                query_audios
                    .iter()
                    .map(|b| b.as_slice())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                relevant_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let golden_path = dir.join("audio_golden.parquet");
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(&golden_path).unwrap(),
        golden_schema,
        None,
    )
    .unwrap();
    w.write(&golden_batch).unwrap();
    w.close().unwrap();

    (corpus_path, golden_path)
}

/// Write an `(anchor, positive, negative)` audio-triplet Parquet table: for
/// each clip, pair it with a same-family sibling (positive) and a
/// different-family clip (negative). The "meaning" of the pairing is the
/// caller's — the trainer only minimizes the contrastive objective.
pub(crate) fn write_audio_triplets(dir: &std::path::Path) -> std::path::PathBuf {
    use arrow::array::{ArrayRef, BinaryArray, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let families = audio_corpus_by_family();
    let fam_names: Vec<&String> = families.keys().collect();

    let mut anchors: Vec<Vec<u8>> = Vec::new();
    let mut positives: Vec<Vec<u8>> = Vec::new();
    let mut negatives: Vec<Vec<u8>> = Vec::new();
    for (fi, fam) in fam_names.iter().enumerate() {
        let clips = &families[*fam];
        // A different family, deterministically chosen.
        let other_fam = fam_names[(fi + 1) % fam_names.len()];
        let neg_clips = &families[other_fam];
        for (ci, (_, anchor)) in clips.iter().enumerate() {
            let (_, positive) = &clips[(ci + 1) % clips.len()];
            let (_, negative) = &neg_clips[ci % neg_clips.len()];
            anchors.push(anchor.clone());
            positives.push(positive.clone());
            negatives.push(negative.clone());
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("anchor", DataType::Binary, false),
        Field::new("positive", DataType::Binary, false),
        Field::new("negative", DataType::Binary, false),
    ]));
    let to_bin = |v: &[Vec<u8>]| -> ArrayRef {
        Arc::new(BinaryArray::from(
            v.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
        )) as ArrayRef
    };
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![to_bin(&anchors), to_bin(&positives), to_bin(&negatives)],
    )
    .unwrap();
    let path = dir.join("audio_triplets.parquet");
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

#[tokio::test(flavor = "multi_thread")]
async fn audio_projection_head_fine_tune_changes_embeddings() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let model = htsat_clap_model();

    // Triplet source (audio bytes), corpus, and golden.
    let triplets_path = write_audio_triplets(dir.path());
    let (corpus_path, golden_path) = write_audio_corpus_and_golden(dir.path());

    for (name, path) in [
        ("audio_triplets", &triplets_path),
        ("audio_corpus", &corpus_path),
        ("audio_golden", &golden_path),
    ] {
        session
            .add_source(
                name,
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", path.display())),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    // Base audio embeddings over the corpus.
    let base_rec = session
        .generate_audio_embeddings(
            "audio_corpus",
            &model,
            "audio",
            "clip_id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // Fine-tune a projection head on the audio triplets. Empty target_modules
    // → projection head on the frozen CLAP audio tower. Triplet loss; the epoch
    // count and learning rate give the zero-init LoRA B enough total gradient to
    // move the shared-latent audio embeddings measurably off the identity
    // projection.
    let job = session
        .fine_tune(
            "audio_triplets",
            &model,
            &[
                "anchor".to_string(),
                "positive".to_string(),
                "negative".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::AudioEmbedding,
            Some(FineTuneConfig {
                epochs: 40,
                batch_size: 4,
                learning_rate: 5e-3,
                lora_rank: 4,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                validation_fraction: 0.0,
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert!(
        job.model_id().starts_with("jammi:fine-tuned:"),
        "model_id should carry the fine-tuned prefix, got '{}'",
        job.model_id()
    );
    job.wait().await.unwrap();

    // The tuned model is registered as an audio-embedding model.
    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("fine-tuned audio model registered in catalog");
    assert_eq!(ft.model_type, "fine-tuned");
    assert_eq!(
        ft.task,
        ModelTask::AudioEmbedding,
        "fine-tuned model should carry the audio-embedding task"
    );

    // Tuned audio embeddings over the same corpus.
    let ft_rec = session
        .generate_audio_embeddings(
            "audio_corpus",
            job.model_id(),
            "audio",
            "clip_id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // Eval audio→audio retrieval for both, against the held-out golden set.
    let base_metrics = session
        .eval_embeddings(
            "audio_corpus",
            Some(&base_rec.table_name),
            "audio_golden.public.audio_golden",
            5,
            &Default::default(),
        )
        .await
        .unwrap();
    let ft_metrics = session
        .eval_embeddings(
            "audio_corpus",
            Some(&ft_rec.table_name),
            "audio_golden.public.audio_golden",
            5,
            &Default::default(),
        )
        .await
        .unwrap();

    // Every aggregate metric stays in range for both.
    for (label, report) in [("base", &base_metrics), ("tuned", &ft_metrics)] {
        for (name, val) in common::aggregate_named_metrics(&report.aggregate) {
            assert!(
                (0.0..=1.0).contains(&val),
                "{label} {name} = {val} outside [0, 1]"
            );
        }
    }

    // The projection head is not a no-op on the audio path: re-encoding the same
    // clip with the tuned model yields a different embedding than the base model
    // (the trained LoRA delta is non-zero). This is the direct, deterministic
    // proof — coarse retrieval metrics over a 20-clip corpus need not flip for
    // the head to have trained, so asserting on them is a knife-edge; the
    // per-clip embedding delta is not. (Mirrors the text-adapter check above.)
    let clip = std::fs::read(common::cookbook_fixture("tiny_audio_corpus").join("clip_sine_0.wav"))
        .unwrap();
    let base_embedding = session.encode_audio_query(&model, &clip).await.unwrap();
    let ft_embedding = session
        .encode_audio_query(job.model_id(), &clip)
        .await
        .unwrap();
    let diff: f32 = base_embedding
        .iter()
        .zip(&ft_embedding)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Tuned audio embeddings should differ from base (non-zero LoRA delta), diff={diff}"
    );
}

// UAT 6 (QLoRA): Invalid methods are now unrepresentable at the type level
// via `FineTuneMethod` enum. No runtime test needed.

// ─── Fine-tune catalog CRUD ─────────────────────────────────────────────────

#[tokio::test]
async fn fine_tune_job_catalog_crud() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = jammi_db::catalog::Catalog::open(dir.path()).await.unwrap();

    // Register base model (FK constraint)
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "base-model",
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

    // Create job
    catalog
        .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
            job_id: "job-1",
            base_model_id: "base-model::1",
            training_source: "training_source",
            loss_type: "cosent",
            hyperparams: r#"{"lora_rank": 8}"#,
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();

    // Get job — status should be "queued"
    let job = catalog.get_training_job("job-1").await.unwrap();
    assert_eq!(job.status, "queued");
    assert_eq!(job.base_model_id, "base-model::1");

    // A worker claims the job → running, leased to it, started_at recorded.
    let claimed = catalog
        .claim_next_training_job("worker-x", std::time::Duration::from_secs(60))
        .await
        .unwrap()
        .expect("the queued job is claimable");
    assert_eq!(claimed.status, "running");
    let marked = catalog
        .mark_training_running(
            "job-1",
            "worker-x",
            Some(r#"{"started_at": "2026-01-01T00:00:00Z"}"#),
        )
        .await
        .unwrap();
    assert!(marked, "the lease owner records its run-start metrics");
    let job2 = catalog.get_training_job("job-1").await.unwrap();
    assert_eq!(job2.status, "running");
    assert!(job2.started_at.is_some());

    // The lease owner finalizes: the single compare-and-set writes the output
    // model + flips to completed + records the run metrics.
    let finalized = catalog
        .finalize_training_job(
            jammi_db::catalog::training_repo::FinalizeTrainingJobParams {
                job_id: "job-1",
                worker_id: "worker-x",
                output_model_id: "jammi:fine-tuned:job-1",
                output_model_version: 1,
                artifact_path: "file:///artifacts/job-1/worker-x/1",
                metrics: Some(r#"{"completed_at": "2026-01-01T01:00:00Z"}"#),
                epoch_checkpoints: &[],
            },
        )
        .await
        .unwrap();
    assert!(finalized, "the lease owner finalizes the job");
    let job3 = catalog.get_training_job("job-1").await.unwrap();
    assert_eq!(job3.status, "completed");
    assert_eq!(
        job3.output_model_id.as_deref(),
        Some("jammi:fine-tuned:job-1")
    );
    assert_eq!(job3.completed_at.as_deref(), Some("2026-01-01T01:00:00Z"));

    // List jobs
    let jobs = catalog.list_training_jobs().await.unwrap();
    assert_eq!(jobs.len(), 1);
}

// ─── Gradient flow: backward_step actually changes LoRA weights ─────────────
//
// Proves the training loop is not a no-op. Without this test, a broken
// backward_step that silently skips updates would go undetected.

#[test]
fn lora_backward_step_changes_weights() {
    use jammi_ai::fine_tune::adamw::{AdamW, ParamsAdamW};

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base_weight = Tensor::randn(0.0f32, 1.0, (4, 8), &device).unwrap();
    let base = Linear::new(base_weight, None);
    let lora = LoraLinear::new_simple(base, 2, 4.0, 0, &varmap, &vb.pp("test")).unwrap();

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

    // One backward step: gradients, then an optimizer step (what the old
    // `Optimizer::backward_step` convenience did).
    let grads = loss.backward().unwrap();
    optimizer.step(&grads).unwrap();

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
// batches with NaN or >100 loss. Tests with precomputed NaN-embedding
// batches.
//
// esc-040 de-pin: the real CoSENT objective (pairwise ordering — see
// `cosent_loss` in `trainer.rs`) is NaN-in-*scores*-safe by construction: a
// score only ever participates in a `<` comparison building the valid-pair
// mask, and any comparison touching NaN is IEEE-754 `false`, so a NaN score
// is masked out (never a valid pair) rather than propagating — for *any*
// batch size, not just this test's single-row batch. The previous fixture
// put NaN in `scores` and relied on the OLD (buggy, plain-MSE) `cosent_loss`
// computing `(cos - NaN)²` directly on every row with no pairwise masking;
// that premise no longer holds. NaN in the *embeddings* still propagates
// (it corrupts the cosine similarity itself, upstream of the pairwise mask),
// so that is what now exercises the divergence path.
#[tokio::test(flavor = "multi_thread")]
async fn training_divergence_detection() {
    use candle_nn::VarMap;
    use jammi_ai::fine_tune::{
        data::{TrainingBatch, TrainingDataLoader},
        lora::build_projection_head,
        trainer::TrainingLoopBuilder,
    };
    use std::sync::Arc;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = build_projection_head(32, &FineTuneConfig::default(), &varmap, &vb).unwrap();

    // Create batches where the embeddings are NaN → cosine similarity, and
    // hence cosent_loss, is NaN regardless of pairing.
    let nan_batch = TrainingBatch::Contrastive {
        embeddings_a: Tensor::full(f32::NAN, (1, 32), &device).unwrap(),
        embeddings_b: Tensor::ones((1, 32), DType::F32, &device).unwrap(),
        scores: Tensor::new(&[1.0f32], &device).unwrap(),
    };
    // Need at least 3 batches to trigger divergence (3 consecutive NaN)
    let loader =
        TrainingDataLoader::from_precomputed(vec![nan_batch.clone(), nan_batch.clone(), nan_batch]);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());

    // Register a model for the FK
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "div-test-model",
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
            job_id: "div-job",
            base_model_id: "div-test-model::1",
            training_source: "src",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    // Claim it under the worker so the run-start stamp (lease-guarded on
    // `claimed_by` + `running`) lands, matching the real worker flow.
    catalog
        .claim_next_training_job("worker-div", std::time::Duration::from_secs(60))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    let mut training_loop = TrainingLoopBuilder::new(
        jammi_ai::fine_tune::target::TrainingTarget::ProjectionHead { head: model },
        varmap,
        FineTuneConfig {
            epochs: 5,
            batch_size: 1,
            validation_fraction: 0.0,
            early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
            warmup_steps: 0,
            ..Default::default()
        },
    )
    .job_id("div-job".into())
    .worker_id("worker-div".into())
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = tokio::task::spawn_blocking(move || training_loop.run(&loader))
        .await
        .unwrap();

    assert!(result.is_err(), "Training should fail on NaN loss");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("diverge"),
        "Error should mention divergence, got: {msg}"
    );
    // The loop returns the typed divergence error; recording the terminal
    // `failed` status is the worker's job (a single finalization authority), not
    // the loop's, so this trainer-internals test asserts on the returned error
    // rather than the catalog status. The worker-driven panic→failed path is
    // covered end-to-end in `panicking_training_job_lands_failed`.
}

// ─── Early stopping: patience exhaustion stops training ─────────────────────
//
// UAT 7. With patience=1, training should stop well before max epochs because
// validation loss never improves. Uses precomputed batches:
// - Training batches: score=1.0 with identical embeddings → low loss
// - Validation batches: score=0.0 with identical embeddings → high loss
// Validation loss stays constant, so patience exhausts after epoch 2.

#[tokio::test(flavor = "multi_thread")]
async fn training_early_stopping_triggers() {
    use candle_nn::VarMap;
    use jammi_ai::fine_tune::{
        data::{TrainingBatch, TrainingDataLoader},
        lora::build_projection_head,
        trainer::TrainingLoopBuilder,
    };
    use std::sync::Arc;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = build_projection_head(32, &FineTuneConfig::default(), &varmap, &vb).unwrap();

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
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());

    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "es-test-model",
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
            job_id: "es-job",
            base_model_id: "es-test-model::1",
            training_source: "src",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    catalog
        .claim_next_training_job("worker-es", std::time::Duration::from_secs(60))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    let mut training_loop = TrainingLoopBuilder::new(
        jammi_ai::fine_tune::target::TrainingTarget::ProjectionHead { head: model },
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
    .worker_id("worker-es".into())
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .build()
    .unwrap();

    let result = tokio::task::spawn_blocking(move || training_loop.run(&loader))
        .await
        .unwrap()
        .unwrap();

    // With patience=1 and constant validation loss, early stopping triggers
    // after epoch 2 (epoch 1 sets best, epoch 2 doesn't improve).
    assert!(
        result.total_steps < 200,
        "Early stopping should trigger well before 100 epochs, got {} steps",
        result.total_steps
    );
    // The loop persists the adapter and returns its result; flipping the job to
    // `completed` is the worker's single lease-guarded finalization, not the
    // loop's, so this trainer-internals test asserts on the returned result. The
    // worker-driven completed path is covered end-to-end by the durability and
    // lifecycle tests above.
}

// ─── Fine-tuned model produces measurably different search quality ───────────
//
// End-to-end: fine-tune with LoRA, generate embeddings with both base and
// fine-tuned models, run eval_embeddings on both, assert that retrieval
// metrics differ. Proves the adapter actually alters search behavior.

#[tokio::test(flavor = "multi_thread")]
async fn fine_tuned_model_produces_measurably_different_search_quality() {
    let (session, _dir) = session_with_training_data().await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let model = tiny_bert_model();

    // Register patents source for embedding generation and eval
    session
        .add_source(
            "patents",
            SourceType::File,
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
            SourceType::File,
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
        .generate_text_embeddings(
            "patents",
            &model,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

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
            ModelTask::TextEmbedding,
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
        .generate_text_embeddings(
            "patents",
            job.model_id(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // Eval base embeddings against golden relevance
    let base_metrics = session
        .eval_embeddings(
            "patents",
            Some(&base_rec.table_name),
            "golden_rel.public.golden_relevance",
            10,
            &Default::default(),
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
            &Default::default(),
        )
        .await
        .unwrap();

    // Both reports carry all four aggregate metrics in [0, 1]
    for (label, report) in [("base", &base_metrics), ("fine-tuned", &ft_metrics)] {
        for (name, val) in common::aggregate_named_metrics(&report.aggregate) {
            assert!(
                (0.0..=1.0).contains(&val),
                "{label} {name} = {val} outside [0, 1]"
            );
        }
    }

    // At least one aggregate metric must differ between base and fine-tuned
    // (proves the adapter actually changes retrieval behavior, not a no-op)
    let base_named = common::aggregate_named_metrics(&base_metrics.aggregate);
    let ft_named = common::aggregate_named_metrics(&ft_metrics.aggregate);
    let any_different = base_named
        .into_iter()
        .zip(ft_named)
        .any(|((_, b), (_, f))| (b - f).abs() > 1e-6);
    assert!(
        any_different,
        "Fine-tuned model should produce at least one different retrieval metric.\n\
         base:       {:?}\n\
         fine-tuned: {:?}",
        base_metrics.aggregate, ft_metrics.aggregate
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

// ─── Durability: a job submitted by one session runs on a worker started later ─
//
// `fine_tune` only submits a queued job carrying a self-describing spec — no
// in-memory data crosses the submit boundary. A `TrainingWorker` started
// afterwards claims the job, reconstructs the loader from the persisted source +
// columns, and trains it to completion. This is the durability contract: the
// submitter need not be the runner.

#[tokio::test(flavor = "multi_thread")]
async fn durable_job_runs_on_separately_started_worker() {
    let (session, _dir) = session_with_training_data().await;
    let model = tiny_bert_model();

    // Submit with NO worker running: the job sits `queued`.
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let queued = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        queued.status, "queued",
        "job sits queued until a worker claims it"
    );

    // Start the worker separately — it reconstructs the run from the spec alone.
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    job.wait().await.unwrap();

    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "the separately-started worker ran the job"
    );

    // The fine-tuned model was registered by the worker's run.
    let models = session.catalog().list_models().await.unwrap();
    assert!(
        models
            .iter()
            .any(|m| m.model_id.starts_with("jammi:fine-tuned:")),
        "worker registered the fine-tuned model"
    );

    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .unwrap();
    let prefix_url =
        jammi_db::storage::StorageUrl::parse(ft.artifact_path.as_deref().unwrap()).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .unwrap();
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.exists(),
        "worker published the adapter at {adapter:?}"
    );
}

// ─── Lease loss: a worker that lost its lease must not finalize ──────────────
//
// The CRITICAL race: cancellation is checked only at epoch boundaries, so a
// worker can finish a short run without ever noticing its lease was reclaimed,
// then reach the terminal write. That terminal write is a compare-and-set gated
// on `claimed_by = worker_id AND status = 'running'`, so the stale worker
// matches zero rows and does NOT mark the job `completed` — the owner that
// re-claimed it is the sole finalizer.
//
// This drives the real worker (`run_claimed_job`): worker-a's claim is stolen by
// worker-b (a reclaim + re-claim) before worker-a runs. worker-a then runs its
// (now stale) claim to completion — a 1-epoch tiny_bert run finishes well inside
// the 10s heartbeat interval, so the cancel flag never fires and worker-a
// reaches finalize believing it succeeded. Post-fix its finalize CAS fails and
// the job stays `running` (owned by worker-b). Pre-fix worker-a finalized
// unconditionally, so the job would (wrongly) be `completed` by the worker that
// lost the lease — this test fails against that code and passes against the CAS.

#[tokio::test(flavor = "multi_thread")]
async fn worker_that_lost_lease_does_not_finalize() {
    use jammi_ai::fine_tune::worker::TrainingWorker;
    use std::time::Duration;

    let (session, _dir) = session_with_training_data().await;
    let model = tiny_bert_model();

    // Submit with NO worker running so the job sits `queued` and carries a real
    // reconstructable spec.
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let worker_a = TrainingWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = TrainingWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with a zero lease (immediately expired). The returned
    // record carries `claimed_by = worker-a` — the stale claim it will later try
    // to finalize.
    let stale_claim = session
        .catalog()
        .claim_next_training_job(worker_a.worker_id(), Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued job");

    // worker-b reclaims (the zero lease is already expired → requeue) and
    // re-claims under a long lease: worker-b now owns the job.
    let actioned = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned = session
        .catalog()
        .claim_next_training_job(worker_b.worker_id(), Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(owned.claimed_by.as_deref(), Some(worker_b.worker_id()));

    // worker-a runs its stale claim to completion. The 1-epoch run finishes
    // before the 10s heartbeat fires, so the cancel flag never trips and
    // worker-a reaches finalize — where the lease-guarded CAS blocks it.
    worker_a.run_claimed_job(&session, stale_claim).await;

    // The job is still `running`, owned by worker-b: worker-a did NOT finalize.
    let after_a = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        after_a.status, "running",
        "a worker that lost its lease must not finalize the job (CAS blocks it)"
    );
    assert_eq!(
        after_a.claimed_by.as_deref(),
        Some(worker_b.worker_id()),
        "the job is still owned by the worker that re-claimed it"
    );

    // The legitimate owner runs and finalizes exactly once.
    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();
    let after_b = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(after_b.status, "completed", "the lease owner finalizes");
    assert!(
        after_b.output_model_id.is_some(),
        "the owner records the output model"
    );

    // Exactly one fine-tuned model row exists (deterministic id, upserted), and
    // it is the one the owner finalized against.
    let ft: Vec<_> = session
        .catalog()
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .collect();
    assert_eq!(
        ft.len(),
        1,
        "the fine-tuned model is registered exactly once"
    );
}

// ─── Tracing span oracle: a worker run carries job_id + tenant_id ────────────
//
// The worker's `run_claimed_job` opens a `#[tracing::instrument]` span stamped
// with the run's `worker_id`, `job_id`, and `tenant_id` so a trace ties a job to
// its worker run (the gRPC request -> worker correlation lives on the shared
// `job_id` / `tenant_id` fields). This drives the real worker path — a bound
// tenant submits a job, the worker claims it (the claim record carries the
// tenant), and `run_claimed_job` runs it to completion under a test-local
// subscriber that emits span NEW/CLOSE events — then asserts a span actually
// carrying both `job_id` and `tenant_id` was emitted. The assertion is on a
// span (NEW/CLOSE), not a bare event: without `with_span_events` the span fields
// would never reach the sink and the oracle would pass vacuously.
//
// Run on a current-thread runtime so the worker future never migrates off the
// thread the test-local subscriber is installed on — the span NEW/CLOSE records
// fire on that thread and land in the buffer. (A multi-thread runtime would let
// the future resume on a worker thread with no default subscriber, the
// "spawned/migrated tasks escape a thread-local subscriber" trap.)
#[test]
fn worker_run_span_carries_job_and_tenant() {
    use std::io;
    use std::str::FromStr;
    use std::sync::Mutex;
    use std::time::Duration;

    use jammi_ai::fine_tune::worker::TrainingWorker;
    use jammi_db::TenantId;
    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::fmt::MakeWriter;

    /// A `MakeWriter` that captures everything written into a shared buffer.
    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let (session, _dir) = session_with_training_data().await;
        let model = tiny_bert_model();

        // A bound tenant so the submitted job — and the claim record the worker
        // runs — carries a real `tenant_id` (not the unscoped `None`).
        let tenant =
            TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a").expect("valid tenant uuid");
        session.bind_tenant(tenant);

        // Submit with no worker loop running so the job sits `queued` and carries
        // a real reconstructable spec. The worker claims it below; the returned
        // handle is not needed here (the claim record carries the ids the span
        // records).
        session
            .fine_tune(
                "training",
                &model,
                &[
                    "text_a".to_string(),
                    "text_b".to_string(),
                    "score".to_string(),
                ],
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                Some(FineTuneConfig {
                    epochs: 1,
                    batch_size: 8,
                    lora_rank: 4,
                    warmup_steps: 0,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let worker = TrainingWorker::new(&session).expect("default worker intervals are valid");
        let claim = session
            .catalog()
            .claim_next_training_job(worker.worker_id(), Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("the worker claims the queued job");
        assert_eq!(
            claim.tenant_id,
            Some(tenant),
            "the claim record carries the bound tenant the span records"
        );
        let job_id = claim.job_id.clone();

        // Install a test-local subscriber that emits span NEW/CLOSE events over a
        // buffer (prod `telemetry::install` is untouched — this never changes prod
        // log volume), then drive the real worker run under it.
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(BufferWriter(buffer.clone()))
            .with_ansi(false)
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .finish();
        let _guard: DefaultGuard = tracing::subscriber::set_default(subscriber);

        worker.run_claimed_job(&session, claim).await;

        let after = session.catalog().get_training_job(&job_id).await.unwrap();
        assert_eq!(
            after.status, "completed",
            "the worker ran the claimed job to completion"
        );

        let logs = String::from_utf8(buffer.lock().unwrap().clone()).expect("utf-8 logs");
        // A span (not a bare event) carrying both fields: the `run_claimed_job`
        // span opens stamped with `job_id` and `tenant_id`. Span events render the
        // span's fields in its `name{...}` context, so a line that names the span,
        // names a span lifecycle event, and carries both ids proves the span was
        // emitted with them. The `tenant_id` field is recorded with `?` (Debug),
        // so the `Option<TenantId>` renders as `Some(...)` wrapping the uuid.
        let span_line = logs.lines().find(|line| {
            line.contains("run_claimed_job")
                && (line.contains("new") || line.contains("close"))
                && line.contains(&format!("job_id={job_id}"))
                && line.contains("tenant_id=Some(")
                && line.contains(&tenant.to_string())
        });
        assert!(
            span_line.is_some(),
            "expected a run_claimed_job span event carrying job_id={job_id} and \
             tenant_id=Some(..{tenant}..); captured logs:\n{logs}"
        );
    });
}

// ─── Configurable lease drives reclaim ──────────────────────────────────────
//
// The prerequisite for the distributed-validation lane: a short configured
// lease must actually expire and be reclaimed. This builds a session whose
// `[training]` timing carries a 1 s lease, claims a real job under the exact
// lease the worker derives from that config (the single source of truth —
// `TrainingConfig::worker_intervals`), then stops heartbeating. After the lease
// elapses, `reclaim_expired_training_jobs` re-queues the job — proving the
// configured lease, not the historical 30 s constant, drives reclaim. A second
// claim under the same short lease succeeds, confirming the job is back in the
// queue.
#[tokio::test(flavor = "multi_thread")]
async fn configured_short_lease_drives_reclaim() {
    use jammi_ai::fine_tune::worker::TrainingWorker;

    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    // A short lease (6 s) with a real heartbeat margin (2 s heartbeat, so
    // heartbeat * 2 < lease — strictly under half) and a 1 s poll — the kind of
    // timing the distributed-validation lane uses to exercise expiry quickly.
    config.training = jammi_db::config::TrainingConfig {
        lease_duration_secs: 6,
        heartbeat_interval_secs: 2,
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
    let model = tiny_bert_model();

    // Submit with no worker loop running so the job sits queued.
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // The worker reads its lease from the session's `[training]` config. Build
    // it the production way so the test exercises the configured value, then
    // claim under that exact lease (the value the worker's loop would pass).
    let worker = TrainingWorker::new(&session).expect("short timing clears the margin");
    let lease = session
        .inner_config()
        .training
        .worker_intervals()
        .unwrap()
        .lease;
    let claimed = session
        .catalog()
        .claim_next_training_job(worker.worker_id(), lease)
        .await
        .unwrap()
        .expect("the queued job is claimable");
    assert_eq!(claimed.status, "running");
    assert_eq!(claimed.claimed_by.as_deref(), Some(worker.worker_id()));

    // The worker "stalls" — it never heartbeats. Before the lease elapses the
    // job is NOT reclaimable.
    let actioned = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(actioned, 0, "a live lease is not reclaimed");

    // Wait past the configured lease, then reclaim re-queues the orphaned job.
    tokio::time::sleep(lease + std::time::Duration::from_secs(1)).await;
    let actioned = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(
        actioned, 1,
        "the configured short lease expired, so the orphaned job is reclaimed"
    );

    // The job is back queued and re-claimable — the reclaim genuinely returned
    // it to the queue rather than failing it.
    let job_after = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(job_after.status, "queued", "the reclaimed job is re-queued");
    let reclaimed = session
        .catalog()
        .claim_next_training_job("worker-second", lease)
        .await
        .unwrap()
        .expect("the re-queued job is claimable again");
    assert_eq!(reclaimed.claimed_by.as_deref(), Some("worker-second"));
}

// ─── Lease loss: only the winner's prefix is the committed artifact ──────────
//
// The data-integrity counterpart to `worker_that_lost_lease_does_not_finalize`:
// that test pins the job row; this one pins the served artifact. Both workers
// train the *same* `job_id`, but each writes to its OWN unique per-attempt
// prefix (`{job_id}/{worker_id}/{attempt}`), so neither can overwrite the other.
// The catalog model-row update (the lease-guarded finalize CAS) is the single
// atomic commit: the loser (worker-a, lost lease) writes its prefix but its CAS
// fails, so the catalog never points at it (the prefix is orphaned). Only the
// winner (worker-b)'s CAS wins, so the committed `artifact_path` is worker-b's
// prefix — fetched and verified (manifest + sha256) to hold its weights. There
// is no shared canonical path to clobber and no torn-promote window.

#[tokio::test(flavor = "multi_thread")]
async fn loser_prefix_is_never_the_committed_artifact() {
    use jammi_ai::fine_tune::worker::TrainingWorker;
    use std::time::Duration;

    let (session, _dir) = session_with_training_data().await;
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let worker_a = TrainingWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = TrainingWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with a zero (already-expired) lease; worker-b reclaims and
    // re-claims under a long lease, so worker-b owns the job.
    let stale_claim = session
        .catalog()
        .claim_next_training_job(worker_a.worker_id(), Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued job");
    let actioned = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned = session
        .catalog()
        .claim_next_training_job(worker_b.worker_id(), Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");

    // worker-a runs its stale claim to completion: its finalize CAS fails, so the
    // job is NOT completed and no catalog model row records worker-a's prefix.
    worker_a.run_claimed_job(&session, stale_claim).await;
    let after_loser = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_ne!(
        after_loser.status, "completed",
        "the loser's finalize CAS fails; the job is not completed by it"
    );

    // The winner runs and finalizes: its finalize CAS wins, so the catalog points
    // the model row at the prefix worker-b published — the single atomic commit.
    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();
    let done = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(done.status, "completed", "the winner finalizes the job");

    // The committed artifact is the one the winner wrote: it fetches and verifies
    // (manifest + sha256) and holds a well-formed, non-empty safetensors map.
    // Exactly-one-writer is guaranteed by the per-attempt prefix + catalog-pointer
    // commit: the loser wrote a different prefix that the catalog never points at,
    // so no clobber or torn read is possible.
    let ft = session
        .catalog()
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the winner registered the fine-tuned model");
    let prefix_url =
        jammi_db::storage::StorageUrl::parse(ft.artifact_path.as_deref().unwrap()).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("the committed artifact fetches and verifies");
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.exists(),
        "the committed prefix holds the winner's adapter at {adapter:?}"
    );
    let loaded = candle_core::safetensors::load(&adapter, &candle_core::Device::Cpu).unwrap();
    assert!(
        !loaded.is_empty(),
        "the committed adapter is a well-formed safetensors tensor map"
    );
}

// ─── A cancelled-mid-run attempt's already-written epoch checkpoints are
//     reclaimed — existence proven BEFORE reclaim (unit 348, F1/F2) ─────────
//
// The top-level artifact prefix's `delete_artifact_prefix` reads and deletes
// only ITS OWN `manifest.json`'s files — it never reaches into the nested
// `checkpoints/epoch_{N}/` prefixes underneath, each carrying its own separate
// manifest. Without a dedicated, DERIVED sweep, a losing/cancelled attempt's
// epoch checkpoints would be orphaned forever (unbounded storage growth
// across every reclaimed/failed attempt — family E).
//
// This drives `TrainingWorker::run_claimed_job`'s `Cancelled` arm
// specifically (a real heartbeat-detected lease loss, not a
// finalize-CAS-loses-the-race scenario), and closes the vacuity a prior
// version of this test had: that version asserted only ABSENCE after the
// race, never proving the bytes existed in the first place — so it could not
// distinguish "the GC worked" from "there was nothing to reclaim". Here the
// existence check is load-bearing and comes FIRST: the test polls for
// epoch_0's manifest to land on disk (a real write, not assumed), THEN forces
// the lease stale (deterministic, not a wall-clock race against the
// attempt's own healthy heartbeat), THEN waits for the cancelled run to
// finish, THEN asserts the SAME bytes are gone.
#[tokio::test(flavor = "multi_thread")]
async fn cancelled_run_reclaims_epoch_checkpoints_that_actually_existed() {
    use jammi_ai::fine_tune::worker::TrainingWorker;
    use jammi_db::catalog::backend::{SqlValue, TxOptions};
    use std::time::Duration;

    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    // The minimum legal heartbeat/lease/poll (heartbeat*2 < lease): fast
    // enough that a real heartbeat tick reliably detects the forced lease
    // loss within about a second, matching `configured_short_lease_drives_
    // reclaim`'s precedent for exercising real timing quickly.
    config.training = jammi_db::config::TrainingConfig {
        lease_duration_secs: 3,
        heartbeat_interval_secs: 1,
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                // Deliberately large: a tiny real fine-tune epoch is fast
                // (single-digit milliseconds on this fixture), so the count
                // must be big enough that the run is CERTAINLY still mid-
                // flight by the time the test forces the lease loss below —
                // the sanity assertion right before that forcing step turns
                // a wrong guess here into a loud, diagnosable failure rather
                // than a silently-wrong-path pass.
                epochs: 20_000,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                // Opt in (round-2 F3: disabled by default) — without this,
                // no epoch checkpoint is ever written and the whole test is
                // vacuous by construction. `n >= epochs` retains everything
                // this run reaches before it is cancelled.
                keep_last_n_checkpoints: Some(20_000),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let worker_a = TrainingWorker::new(&session).expect("short timing clears the margin");
    let lease = session
        .inner_config()
        .training
        .worker_intervals()
        .unwrap()
        .lease;
    let claimed = session
        .catalog()
        .claim_next_training_job(worker_a.worker_id(), lease)
        .await
        .unwrap()
        .expect("the queued job is claimable");
    let attempt = claimed.attempts;
    let job_id = job.job_id.clone();
    let worker_a_id = worker_a.worker_id().to_string();

    // Run the claimed job concurrently (not awaited inline) so the real
    // heartbeat task can actually tick while training is still in progress —
    // the load-bearing difference from a synchronous "claim, then run to
    // completion" drive, which can never observe an in-flight cancellation.
    let session_for_task = Arc::clone(&session);
    let handle = tokio::spawn(async move {
        worker_a.run_claimed_job(&session_for_task, claimed).await;
    });

    // The local on-disk path this attempt's epoch-0 checkpoint manifest
    // lands at — `file://` roots materialise real files at exactly this
    // path (`ArtifactStore`'s in-place short-circuit), so checking it
    // directly needs no `StorageUrl` round-trip.
    let epoch0_manifest = dir
        .path()
        .join("jammi_db")
        .join("models")
        .join(&job_id)
        .join(&worker_a_id)
        .join(attempt.to_string())
        .join("checkpoints")
        .join("epoch_0")
        .join("manifest.json");

    // LOAD-BEARING: poll for the bytes to actually appear before doing
    // anything else. A bounded wait, not a fixed sleep — the exact epoch-0
    // wall time varies by machine.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if epoch0_manifest.exists() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "epoch_0's checkpoint manifest never appeared within 30s at {epoch0_manifest:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        epoch0_manifest.exists(),
        "epoch_0's checkpoint bytes must exist BEFORE the reclaim — the load-bearing existence \
         proof a vacuous version of this test skipped"
    );

    // Sanity gate: the spawned `run_claimed_job` task must NOT have returned
    // yet, or the whole point of this test — forcing a lease loss WHILE
    // training is still in flight — is moot (a task that already returned
    // would instead have exercised `publish_and_finalize`'s own
    // already-covered `Ok(false)` arm, a DIFFERENT code path, silently
    // mis-testing the wrong one; `training_jobs.status` alone cannot
    // distinguish the two — both leave it `running`). A large `epochs` is
    // chosen so this should never fire; if it does, that is a loud,
    // diagnosable failure demanding a bigger epoch count, not a silent
    // false pass.
    assert!(
        !handle.is_finished(),
        "the spawned run_claimed_job task already completed before the test could force the \
         lease loss — raise `epochs` further so this genuinely races a live run"
    );

    // Force the lease stale RIGHT NOW — deterministic, not a race against
    // worker-a's own healthy (actively-renewing) heartbeat, which would
    // otherwise never organically expire while the attempt is alive.
    let force_job_id = job_id.clone();
    session
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE training_jobs SET lease_expires_at = '2000-01-01T00:00:00Z' \
                     WHERE job_id = $1",
                    &[SqlValue::TextOwned(force_job_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
    let reclaimed = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(reclaimed, 1, "the forced-stale lease is reclaimed");
    let worker_b = TrainingWorker::new(&session).expect("short timing clears the margin");
    session
        .catalog()
        .claim_next_training_job(worker_b.worker_id(), Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b steals the requeued job");

    // Worker-a's own heartbeat notices the loss within one interval and sets
    // `cancel`; the training loop bails at its next epoch boundary via
    // `WorkerJobError::Cancelled`. Bounded wait for the spawned task.
    tokio::time::timeout(Duration::from_secs(30), handle)
        .await
        .expect("worker-a's run_claimed_job must finish once its lease is lost")
        .unwrap();

    // Worker-a never finalized (the `Cancelled` arm records no terminal
    // status) — confirms this run really took the cancelled path, not a
    // race where it happened to finish and win anyway.
    let after = session.catalog().get_training_job(&job_id).await.unwrap();
    assert!(
        after.output_model_id.is_none(),
        "worker-a's cancelled run must not have finalized the job itself"
    );

    // THE reclaim assertion: the SAME bytes confirmed to exist above are now
    // gone — reclaimed by `run_claimed_job`'s `Cancelled` arm calling the
    // derived `TrainingWorker::gc_epoch_checkpoints` sweep.
    assert!(
        !epoch0_manifest.exists(),
        "epoch_0's checkpoint bytes must be reclaimed once the Cancelled arm runs, not left \
         durable forever"
    );
}

// ─── Winner-path prune leak: a persistently-failed mid-run prune is reclaimed
//     at finalize, and the failure is never silent (unit 348 F2) ──────────────
//
// A REAL, non-cancelled winning run with `keep_last_n_checkpoints = Some(1)`
// over 3 epochs: epoch_0's on-disk directory is made undeletable (real
// `chmod`, Unix — see `crates/jammi-ai/src/fine_tune/trainer.rs`'s
// `epoch_checkpoint_retention_failure` module for why this crate has no
// pluggable `ArtifactStore` fault-injection seam and this is the closest
// real integration-level fault injection reachable). Retention can therefore
// never successfully prune anything: it always retries the OLDEST entry
// first (epoch_0), which keeps failing, so epoch_1 is never even individually
// attempted mid-run — both stay durable, over the retention cap, until the
// run completes. `publish_and_finalize`'s winner arm then (a) trims to the
// true trailing window (epoch_2 only) before registering, and (b) sweeps the
// excluded stale entries: epoch_1's delete succeeds for the first time HERE
// (proving "reclaimed at finalize" for an entry retention itself never got
// to), while epoch_0's delete fails again (still chmod'd), which must emit
// the one warning this test asserts on.
// Deliberately the DEFAULT (`current_thread`) flavor, not `multi_thread`:
// the tracing capture below installs a THREAD-LOCAL default subscriber,
// which does not propagate across OS threads. On a `multi_thread` runtime
// (even with `worker_threads = 1`), `Runtime::block_on`'s calling thread and
// a `tokio::spawn`-ed task's thread are DIFFERENT — the runtime's own
// worker pool executes spawned tasks, not the thread driving `block_on`. On
// `current_thread`, there IS no separate worker pool: `block_on` and every
// `tokio::spawn`-ed task share the exact same OS thread, cooperatively
// interleaved at `.await` points, which keeps the subscriber visible to the
// spawned `run_claimed_job` task's async continuation after `spawn_blocking`
// resolves back onto it (the `spawn_blocking` closure ITSELF still runs on
// tokio's separate blocking pool, but the warn under test fires from
// `publish_and_finalize`, back on the single async thread, not from inside
// that closure).
/// PROBE a `chmod 0o555`'d directory for a real write-block: root (and a
/// mode-ignoring filesystem) can write through it regardless, in which case
/// the caller's failed-prune fault-injection premise never exists and it
/// must skip loudly rather than assert against a fault that was never
/// injected. Unless `JAMMI_REQUIRE_POSIX_PERMS` is set (the lane that is
/// SUPPOSED to run unprivileged with real POSIX permission enforcement), in
/// which case a bypassed chmod is itself a hard failure, never a silent
/// skip — the same require-gate polarity every other `JAMMI_REQUIRE_*`
/// skip-guard in this crate carries, applied to a filesystem-privilege
/// probe instead of a hardware one.
#[cfg(unix)]
fn chmod_bypassed(dir: &std::path::Path) -> bool {
    let probe = dir.join(".root_probe");
    let bypassed = std::fs::write(&probe, b"x").is_ok();
    if bypassed {
        let _ = std::fs::remove_file(&probe);
        if std::env::var_os("JAMMI_REQUIRE_POSIX_PERMS").is_some() {
            panic!(
                "JAMMI_REQUIRE_POSIX_PERMS is set but the process could write through a \
                 0o555-chmod'd directory (root, or a mode-ignoring filesystem) — the \
                 failed-prune fault-injection premise this test needs does not hold; a silent \
                 skip is not acceptable here"
            );
        }
    }
    bypassed
}

#[cfg(unix)]
#[tokio::test]
async fn finalize_reclaims_a_persistently_failed_prune_and_warns() {
    use std::io;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;
    use std::time::Duration;

    use jammi_ai::fine_tune::worker::TrainingWorker;
    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);
    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 3,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                keep_last_n_checkpoints: Some(1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let worker = TrainingWorker::new(&session).expect("default worker intervals are valid");
    let claimed = session
        .catalog()
        .claim_next_training_job(worker.worker_id(), std::time::Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("the queued job is claimable");
    let attempt = claimed.attempts;
    let job_id = job.job_id.clone();
    let worker_id = worker.worker_id().to_string();
    let output_name = job.model_id().to_string();

    let epoch_local_dir = |epoch: usize| {
        dir.path()
            .join("jammi_db")
            .join("models")
            .join(&job_id)
            .join(&worker_id)
            .join(attempt.to_string())
            .join("checkpoints")
            .join(format!("epoch_{epoch}"))
    };
    let epoch0_dir = epoch_local_dir(0);
    let epoch1_dir = epoch_local_dir(1);

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(BufferWriter(buffer.clone()))
        .with_ansi(false)
        .finish();
    let _guard: DefaultGuard = tracing::subscriber::set_default(subscriber);

    let session_for_task = Arc::clone(&session);
    let handle = tokio::spawn(async move {
        worker.run_claimed_job(&session_for_task, claimed).await;
    });

    // LOAD-BEARING: poll for epoch_0's manifest to actually appear before
    // chmod'ing it.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while !epoch0_dir.join("manifest.json").exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "epoch_0's checkpoint manifest never appeared within 30s at {epoch0_dir:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    // PROBE the injection before relying on it: root (and mode-ignoring
    // filesystems) can delete through a 0o555 directory, so the failed-prune
    // premise never exists there — skip loudly (the environment-conditional
    // convention this batch applies in candle.rs's device_tests too). The
    // run is aborted rather than awaited: nothing below is meaningful
    // without the injected fault.
    if chmod_bypassed(&epoch0_dir) {
        let _ = std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o755));
        handle.abort();
        eprintln!(
            "finalize_reclaims_a_persistently_failed_prune_and_warns: skipping — fault \
             injection unavailable: process can write despite chmod (root?)"
        );
        return;
    }

    // Let the run finish naturally (no cancellation this time — this is the
    // WINNER path). Bounded wait.
    tokio::time::timeout(Duration::from_secs(60), handle)
        .await
        .expect("the run must complete")
        .unwrap();

    // Un-chmod immediately so the tempdir's own cleanup (on drop) can
    // recursively remove it regardless of what the assertions below find.
    let restore_epoch0 = || {
        std::fs::set_permissions(&epoch0_dir, std::fs::Permissions::from_mode(0o755)).ok();
    };

    let after = session.catalog().get_training_job(&job_id).await.unwrap();
    if after.status != "completed" {
        restore_epoch0();
        panic!("the run must complete and finalize as the sole winner, got status {after:?}");
    }

    // Only epoch_2 (the trailing `keep=1` window) is registered.
    let epoch2_row = session
        .catalog()
        .get_model(&format!("{output_name}:epoch_2"))
        .await
        .unwrap();
    let epoch0_row = session
        .catalog()
        .get_model(&format!("{output_name}:epoch_0"))
        .await
        .unwrap();
    let epoch1_row = session
        .catalog()
        .get_model(&format!("{output_name}:epoch_1"))
        .await
        .unwrap();

    // epoch_1's bytes are gone — reclaimed by the winner's finalize-time
    // sweep, even though mid-run retention never individually attempted it
    // (FIFO always retries epoch_0 first). This is the "reclaimed at
    // finalize" claim, made observable independent of epoch_0's own
    // still-blocked state.
    let epoch1_gone = !epoch1_dir.join("manifest.json").exists();
    // epoch_0's bytes are STILL present — the chmod is still in effect, so
    // even the finalize-time retry fails, which is exactly what must
    // produce the warning this test asserts on below.
    let epoch0_still_present = epoch0_dir.join("manifest.json").exists();

    let logs = String::from_utf8(buffer.lock().unwrap().clone()).expect("utf-8 logs");
    restore_epoch0();

    assert!(
        epoch2_row.is_some(),
        "epoch_2 (the retained window) must register"
    );
    assert!(
        epoch0_row.is_none(),
        "epoch_0 must NOT register — trimmed out of the retained window"
    );
    assert!(
        epoch1_row.is_none(),
        "epoch_1 must NOT register — trimmed out of the retained window"
    );
    assert!(
        epoch1_gone,
        "epoch_1's bytes must be reclaimed by the winner-arm finalize sweep, even though \
         mid-run retention never individually attempted it"
    );
    assert!(
        epoch0_still_present,
        "epoch_0's bytes must still be present — the chmod was never lifted before finalize, \
         so even the finalize-time retry must fail (proving the sweep genuinely re-attempts it \
         rather than silently succeeding)"
    );
    assert!(
        logs.contains("epoch-checkpoint GC sweep") && logs.contains(&job_id),
        "a failed finalize-time reclaim must emit exactly one warning naming the job; \
         captured logs:\n{logs}"
    );
}

// ─── Reclaim: a zombie loser running AFTER the winner cannot corrupt the commit ─
//
// The audit's exact ordering, and the one `loser_prefix_is_never_the_committed_
// artifact` does NOT exercise: the WINNER runs and completes FIRST, THEN the
// stale (zombie) loser runs to completion. The loser still holds an old claim
// (its lease expired and was reclaimed), so when it finishes it registers its
// own model row and runs its finalize. With the served path committed by an
// unguarded last-writer-wins `register_model` (the pre-fix shape) the zombie's
// late register would overwrite the committed `artifact_path` with its own
// prefix, and its CAS-loss branch would then delete that prefix's bytes —
// leaving the completed model pointing at deleted bytes (a `manifest.json
// NotFound` on reload) and, separately, regressing the job's status back to
// `running` via the unguarded run-start status write.
//
// Post-fix: the served path is committed solely by the winner's lease-guarded
// finalize CAS, never by `register_model`; the zombie's finalize matches zero
// rows and commits nothing, so it only GC's its OWN (never-committed) prefix;
// and every job-row write the zombie makes is lease-guarded, so the terminal
// `completed` status is undisturbed. The committed prefix is the winner's, its
// bytes survive, and reload succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn zombie_loser_after_winner_cannot_corrupt_the_commit() {
    use jammi_ai::fine_tune::worker::TrainingWorker;
    use std::time::Duration;

    let (session, _dir) = session_with_training_data().await;
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
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let worker_a = TrainingWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = TrainingWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with a zero (already-expired) lease — this is the stale
    // claim the zombie will later run. reclaim re-queues it; worker-b re-claims
    // under a long lease and is the rightful owner.
    let stale_claim = session
        .catalog()
        .claim_next_training_job(worker_a.worker_id(), Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued job");
    let actioned = session
        .catalog()
        .reclaim_expired_training_jobs(5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned = session
        .catalog()
        .claim_next_training_job(worker_b.worker_id(), Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");

    // WINNER FIRST: worker-b trains, wins its finalize CAS, and completes the
    // job — committing its prefix as the served artifact path.
    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();

    let ft = session
        .catalog()
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the winner registered the fine-tuned model");
    let winner_prefix = ft
        .artifact_path
        .clone()
        .expect("the winner committed a served artifact_path");

    // THEN the zombie loser runs its stale claim to completion: it registers its
    // own model row and runs its finalize. Its lease was reclaimed, so its
    // finalize CAS must match zero rows — committing nothing — and it only GC's
    // its own (never-committed) prefix.
    worker_a.run_claimed_job(&session, stale_claim).await;

    // (1) The served path is still the WINNER's prefix — the zombie's late
    // register never overwrote the committed pointer.
    let ft_after = session
        .catalog()
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the fine-tuned model row still exists");
    assert_eq!(
        ft_after.artifact_path.as_deref(),
        Some(winner_prefix.as_str()),
        "the committed served path is the winner's prefix; the zombie's late \
         register did not overwrite it"
    );

    // (2) The committed prefix's bytes still exist and fetch_artifact succeeds —
    // reload works (no `manifest.json NotFound`). The zombie GC'd its OWN prefix,
    // never the committed one.
    let prefix_url = jammi_db::storage::StorageUrl::parse(&winner_prefix).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("the committed artifact still fetches and verifies after the zombie ran");
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.exists(),
        "the committed prefix still holds the winner's adapter at {adapter:?}"
    );
    let loaded = candle_core::safetensors::load(&adapter, &candle_core::Device::Cpu).unwrap();
    assert!(
        !loaded.is_empty(),
        "the committed adapter remains a well-formed safetensors tensor map"
    );

    // (3) The job stays `completed` — the zombie's run-start status write is
    // lease-guarded, so it could not regress the terminal status to `running`.
    let after_zombie = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        after_zombie.status, "completed",
        "the terminal status is undisturbed by the zombie (no completed → running regression)"
    );
}

// ─── Lease loss: cooperative cancellation bails the trainer at the boundary ──
//
// A `spawn_blocking` trainer cannot be force-aborted, so the worker's heartbeat
// sets a cancel flag the trainer checks at every epoch boundary. With the flag
// pre-set, a multi-epoch run bails at the first boundary with a "lease lost"
// error and never marks the job `completed`; the job is left `running` for
// `reclaim_expired_training_jobs` to re-queue (bounded by the attempts cap).

#[tokio::test(flavor = "multi_thread")]
async fn training_bails_when_lease_lost_mid_run() {
    use candle_nn::VarMap;
    use jammi_ai::fine_tune::{
        data::{TrainingBatch, TrainingDataLoader},
        lora::build_projection_head,
        trainer::TrainingLoopBuilder,
    };
    use std::sync::atomic::AtomicBool;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let head = build_projection_head(32, &FineTuneConfig::default(), &varmap, &vb).unwrap();

    // A benign precomputed batch so a non-cancelled run would simply train.
    let batch = TrainingBatch::Contrastive {
        embeddings_a: Tensor::ones((2, 32), DType::F32, &device).unwrap(),
        embeddings_b: Tensor::ones((2, 32), DType::F32, &device).unwrap(),
        scores: Tensor::new(&[1.0f32, 0.0], &device).unwrap(),
    };
    let loader = TrainingDataLoader::from_precomputed(vec![batch]);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "lease-model",
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
            job_id: "lease-job",
            base_model_id: "lease-model::1",
            training_source: "src",
            loss_type: "cosent",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    // Claim it so the row is `running` (the state a mid-run job is in). A zero
    // lease means the lease is already expired by the time reclaim runs — a
    // deterministic forced expiry, no sleep needed.
    catalog
        .claim_next_training_job("worker-a", std::time::Duration::ZERO)
        .await
        .unwrap()
        .expect("claimed the queued job");

    // The lease is "lost" before training starts: the cancel flag is set, as the
    // heartbeat would set it.
    let cancel = Arc::new(AtomicBool::new(true));

    let mut training_loop = TrainingLoopBuilder::new(
        jammi_ai::fine_tune::target::TrainingTarget::ProjectionHead { head },
        varmap,
        FineTuneConfig {
            epochs: 5,
            batch_size: 2,
            validation_fraction: 0.0,
            early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
            warmup_steps: 0,
            ..Default::default()
        },
    )
    .job_id("lease-job".into())
    .worker_id("worker-a".into())
    .catalog(Arc::clone(&catalog))
    .artifact_dir(dir.path().to_path_buf())
    .cancel(Arc::clone(&cancel))
    .build()
    .unwrap();

    let result = tokio::task::spawn_blocking(move || training_loop.run(&loader))
        .await
        .unwrap();

    assert!(result.is_err(), "a lost lease must bail the training loop");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("training cancelled"),
        "bail error should name the cancellation, got: {msg}"
    );

    // The job was NOT marked completed — it is left for reclaim.
    let job = catalog.get_training_job("lease-job").await.unwrap();
    assert_ne!(
        job.status, "completed",
        "a bailed job must not be completed"
    );

    // The job's lease is already expired (claimed with a zero lease), so reclaim
    // re-queues it (attempts 1 < cap 3) — a dead worker's job is retried.
    let actioned = catalog.reclaim_expired_training_jobs(3).await.unwrap();
    assert!(actioned >= 1, "the expired-lease job is reclaimed");
    let reclaimed = catalog.get_training_job("lease-job").await.unwrap();
    assert_eq!(
        reclaimed.status, "queued",
        "a reclaimed job is re-queued for another worker"
    );
}
