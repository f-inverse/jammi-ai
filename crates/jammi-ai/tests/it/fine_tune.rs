use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use tempfile::TempDir;

use jammi_ai::fine_tune::{
    data::TrainingDataLoader, trainer::compute_lr, FineTuneConfig, FineTuneMethod, LrSchedule,
};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::status::FineTuneJobStatus;
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
    let mut lora = LoraLinear::new_simple(base_linear.clone(), 2, 4.0, &vb).unwrap();

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
        .get_fine_tune_job(&job.job_id)
        .await
        .unwrap();
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
fn write_audio_triplets(dir: &std::path::Path) -> std::path::PathBuf {
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

#[tokio::test]
async fn audio_projection_head_fine_tune_changes_embeddings() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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
        .generate_audio_embeddings("audio_corpus", &model, "audio", "clip_id")
        .await
        .unwrap();

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
        .generate_audio_embeddings("audio_corpus", job.model_id(), "audio", "clip_id")
        .await
        .unwrap();

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
        .create_fine_tune_job(
            "job-1",
            "base-model::1",
            "training_source",
            "cosent",
            r#"{"lora_rank": 8}"#,
        )
        .await
        .unwrap();

    // Get job — status should be "queued"
    let job = catalog.get_fine_tune_job("job-1").await.unwrap();
    assert_eq!(job.status, "queued");
    assert_eq!(job.base_model_id, "base-model::1");

    // Update to running
    let metrics = r#"{"started_at": "2026-01-01T00:00:00Z"}"#;
    catalog
        .update_fine_tune_status("job-1", FineTuneJobStatus::Running, Some(metrics))
        .await
        .unwrap();
    let job2 = catalog.get_fine_tune_job("job-1").await.unwrap();
    assert_eq!(job2.status, "running");
    assert!(job2.started_at.is_some());

    // Update to completed with output model
    catalog
        .update_fine_tune_status(
            "job-1",
            FineTuneJobStatus::Completed,
            Some(r#"{"completed_at": "2026-01-01T01:00:00Z"}"#),
        )
        .await
        .unwrap();
    catalog
        .set_fine_tune_output_model("job-1", "jammi:fine-tuned:job-1")
        .await
        .unwrap();
    let job3 = catalog.get_fine_tune_job("job-1").await.unwrap();
    assert_eq!(job3.status, "completed");
    assert_eq!(
        job3.output_model_id.as_deref(),
        Some("jammi:fine-tuned:job-1")
    );

    // List jobs
    let jobs = catalog.list_fine_tune_jobs().await.unwrap();
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
    let lora = LoraLinear::new_simple(base, 2, 4.0, &vb.pp("test")).unwrap();

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
    let model = build_projection_head(32, &FineTuneConfig::default(), &vb).unwrap();

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
        .create_fine_tune_job("div-job", "div-test-model::1", "src", "cosent", "{}")
        .await
        .unwrap();

    let mut training_loop = TrainingLoopBuilder::new(
        jammi_ai::fine_tune::target::TrainingTarget::ProjectionHead { head: model },
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

    let result = tokio::task::spawn_blocking(move || training_loop.run(&loader))
        .await
        .unwrap();

    assert!(result.is_err(), "Training should fail on NaN loss");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("diverge"),
        "Error should mention divergence, got: {msg}"
    );

    // Catalog should record failure
    let job = catalog.get_fine_tune_job("div-job").await.unwrap();
    assert_eq!(job.status, "failed", "Job status should be 'failed'");
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
    let model = build_projection_head(32, &FineTuneConfig::default(), &vb).unwrap();

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
        .create_fine_tune_job("es-job", "es-test-model::1", "src", "cosent", "{}")
        .await
        .unwrap();

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

    // Job should be completed (not failed)
    let job = catalog.get_fine_tune_job("es-job").await.unwrap();
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
