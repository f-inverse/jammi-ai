//! End-to-end integration tests for the
//! [`TrainingTarget::EncoderAdapters`](jammi_ai::fine_tune::target::TrainingTarget::EncoderAdapters)
//! flavour of fine-tuning: LoRA injected into the encoder's internal
//! attention/FFN linears.
//!
//! Each test fine-tunes one of the local `tiny_bert` / `tiny_modernbert`
//! fixtures with `target_modules` populated, walks back through the saved
//! adapter on disk, and asserts the persisted `SavedAdapter` variant +
//! contents are correct. Uses the same fixtures and helpers as
//! `fine_tune.rs` — no network, no GPU.
//!
//! Pure-function jammi-lora helpers (`should_apply_lora`, `effective_rank`,
//! `AdapterConfig` JSON round-trip) are covered by jammi-lora's own
//! integration suite and intentionally not duplicated here.

use std::sync::Arc;

use tempfile::TempDir;

use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, LrSchedule};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

fn tiny_modernbert_model() -> String {
    "local:".to_string() + common::fixture("tiny_modernbert").to_str().unwrap()
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

fn training_columns() -> Vec<String> {
    vec![
        "text_a".to_string(),
        "text_b".to_string(),
        "score".to_string(),
    ]
}

/// Locate the saved adapter directory for a fine-tuned model id.
/// Models registered via `fine_tune` carry their artifact path on the
/// model record; the on-disk layout is `<artifact_path>/adapter.safetensors`
/// alongside `adapter_config.json`.
async fn adapter_dir_for_model(session: &InferenceSession, model_id: &str) -> std::path::PathBuf {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .expect("fine-tuned model registered in catalog");
    std::path::PathBuf::from(record.artifact_path.expect("artifact_path"))
}

#[tokio::test]
async fn encoder_adapters_bert_writes_adapter_marker() {
    let (session, _dir) = session_with_training_data().await;

    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 2,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                // Non-empty target_modules triggers the encoder-adapters target.
                target_modules: vec!["query".to_string(), "value".to_string()],
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    // Walk back through the saved adapter and confirm the encoder-adapters
    // discriminator + the persisted target modules / rank.
    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let cfg_path = adapter_dir.join("adapter_config.json");
    assert!(
        cfg_path.exists(),
        "adapter_config.json should exist at {cfg_path:?}"
    );
    let cfg_str = std::fs::read_to_string(&cfg_path).unwrap();
    let saved: jammi_ai::fine_tune::target::SavedAdapter = serde_json::from_str(&cfg_str).unwrap();
    let cfg = match saved {
        jammi_ai::fine_tune::target::SavedAdapter::EncoderAdapters(cfg) => *cfg,
        other => panic!("expected EncoderAdapters variant, got {other:?}"),
    };

    assert_eq!(cfg.model_type, "bert");
    assert_eq!(
        cfg.target_modules,
        vec!["query".to_string(), "value".to_string()]
    );
    assert_eq!(cfg.lora_rank, 4);

    // Weights file should land next to the config.
    assert!(
        adapter_dir.join("adapter.safetensors").exists(),
        "adapter.safetensors should exist at {:?}",
        adapter_dir.join("adapter.safetensors")
    );
}

#[tokio::test]
async fn encoder_adapters_modernbert_writes_adapter_marker() {
    let (session, _dir) = session_with_training_data().await;

    let job = session
        .fine_tune(
            "training",
            &tiny_modernbert_model(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 2,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                // ModernBERT-specific targets: fused QKV + output projection.
                target_modules: vec!["Wqkv".to_string(), "Wo".to_string()],
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let saved: jammi_ai::fine_tune::target::SavedAdapter = serde_json::from_str(
        &std::fs::read_to_string(adapter_dir.join("adapter_config.json")).unwrap(),
    )
    .unwrap();
    let cfg = match saved {
        jammi_ai::fine_tune::target::SavedAdapter::EncoderAdapters(cfg) => *cfg,
        other => panic!("expected EncoderAdapters variant, got {other:?}"),
    };

    assert_eq!(cfg.model_type, "modernbert");
    assert_eq!(
        cfg.target_modules,
        vec!["Wqkv".to_string(), "Wo".to_string()]
    );
}

#[tokio::test]
async fn encoder_adapters_changes_embeddings_versus_base() {
    let (session, _dir) = session_with_training_data().await;
    let base = tiny_bert_model();

    // Base embedding for a known input.
    let base_vec = session
        .encode_text_query(&base, "encoder adapters end to end smoke")
        .await
        .unwrap();

    let job = session
        .fine_tune(
            "training",
            &base,
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 5,
                batch_size: 4,
                lora_rank: 4,
                learning_rate: 1e-3,
                warmup_steps: 0,
                lr_schedule: LrSchedule::Constant,
                target_modules: vec!["query".to_string(), "value".to_string()],
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait().await.unwrap();

    let ft_vec = session
        .encode_text_query(job.model_id(), "encoder adapters end to end smoke")
        .await
        .unwrap();

    assert_eq!(
        base_vec.len(),
        ft_vec.len(),
        "embedding dimension must not change"
    );
    let max_abs_diff = base_vec
        .iter()
        .zip(ft_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff > 1e-6,
        "encoder-adapters fine-tune should shift the embedding (max |Δ| = {max_abs_diff})"
    );
}
