//! P2 — `fine_tune` learns on GPU.
//!
//! A tiny real LoRA run over the `training_pairs.csv` fixture, on a GPU-pinned
//! session (`gpu.device = 0`, `require_gpu = true`), must:
//!   (a) complete without error on the GPU,
//!   (b) decrease its training loss first→last epoch (the on-device training
//!       math actually descends — captured off the trainer's per-epoch tracing),
//!   (c) produce an adapter that *changes* embeddings vs the base model (a
//!       non-zero LoRA delta — the same assertion the CPU lifecycle test makes,
//!       run on GPU).

use std::sync::Arc;

use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

async fn add_training_source(session: &Arc<InferenceSession>) {
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(harness::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_learns_on_gpu() {
    skip_without_gpu!();
    harness::loss_capture::install();
    harness::loss_capture::reset();

    let dir = TempDir::new().unwrap();
    let session = harness::gpu_session(dir.path()).await;
    add_training_source(&session).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let model = harness::local_model_id("tiny_bert");

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
                // ≥2 epochs so first→last carries a decrease signal; tiny rank.
                epochs: 6,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // (a) completes on the GPU.
    job.wait().await.unwrap();
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "GPU fine-tune job should complete, got {}",
        record.status
    );

    // (b) loss decreases first→last epoch.
    let curve = harness::loss_capture::captured();
    let (first, last) = harness::assert_loss_decreases("fine_tune", &curve);

    // (c) the adapter changes embeddings vs the base model. The fine-tuned model
    // is served back through the same GPU session.
    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("fine-tuned model registered");
    let ft_name = ft.model_id.split("::").next().unwrap();

    let base = session
        .encode_text_query(&model, "quantum computing")
        .await
        .unwrap();
    let tuned = session
        .encode_text_query(ft_name, "quantum computing")
        .await
        .unwrap();
    let delta: f32 = base.iter().zip(&tuned).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        delta > 1e-6,
        "GPU-trained adapter must change embeddings (LoRA delta non-zero), delta={delta}"
    );

    tracing::info!(
        first_loss = first,
        last_loss = last,
        embed_delta = delta,
        "P2 fine_tune learns on GPU"
    );
}
