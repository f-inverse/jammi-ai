//! Issue #421 A3/A4: LoRA adapters trained INSIDE the three cross-modal
//! towers — the OpenCLIP text tower, the OpenCLIP vision tower and the
//! HTSAT-Swin CLAP audio tower — round-trip through the real worker and are
//! actually applied when the fine-tuned model is served.
//!
//! # What each positive test proves, and why it needs three assertions
//!
//! A "the fine-tune ran" assertion is vacuous on its own: a run that trained
//! the WRONG tower, or whose adapter serving silently discarded, still
//! completes and still writes an `adapter_config.json`. Each A3 test therefore
//! pins three independent facts:
//!
//! 1. **Identity** — the saved adapter is
//!    [`SavedAdapter::EncoderAdapters`](jammi_ai::fine_tune::target::SavedAdapter)
//!    with the architecture id and the `tower` this base/task pair implies.
//!    A run that coerced the checkpoint to BERT, or that forgot which tower it
//!    adapted, fails here.
//! 2. **Non-vacuity** — the served embedding differs from the BASE model's for
//!    the same input. A zero LoRA delta, or an adapter serving dropped, fails
//!    here — the exact failure mode this assertion exists to catch on these
//!    three towers: an adapter loaded but silently ignored.
//! 3. **Mechanism, bit-for-bit** — the served embedding is bit-equal to an
//!    INDEPENDENTLY constructed tower: the base checkpoint plus the saved
//!    `adapter.safetensors`, built through `jammi-encoders`' own public
//!    builder, forwarded over the same preprocessed tensors, with the same
//!    post-forward step the serving path applies. This is the assertion that
//!    distinguishes "an adapter was applied" from "THIS adapter was applied to
//!    THIS tower". The reference reproduces the serving path's preprocessing
//!    through the SAME public front-end functions
//!    (`inference::{image_preprocess, audio_preprocess}`, `TokenizerWrapper`),
//!    so it is an independent construction, not a second call into the code
//!    under test.
//!
//! No test-only production hook is needed for any of this: every input the
//! serving path builds is reproducible from public API.
//!
//! # A4 — the refusals
//!
//! The negative tests pin the K2 boundary this unit installs: an unsupported
//! `model_type` (RED at base — the worker's `_ => BERT` arm coerced it and
//! trained), an unsupported `(family, task)` pair, a cross-family adapter at
//! load time, the two task/column-shape mismatches, and — the freeze fix
//! round's addition — a `target_modules` list that selects NO site on the
//! tower the dispatch picked, on two towers with disjoint site vocabularies.
//! Beside them sits the audio front end's mel-bin guard, which pins the
//! trainer to the SAME refusal the serving audio path already made.
//!
//! Beside them sits the one POSITIVE case that boundary must NOT swallow: a
//! checkpoint whose config declares no `model_type` at all, which has to
//! resolve to the same family for training as it does for serving.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use tempfile::TempDir;

use jammi_ai::fine_tune::target::SavedAdapter;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, LrSchedule};
use jammi_ai::inference::{audio_preprocess, image_preprocess};
use jammi_ai::model::tokenizer::TokenizerWrapper;
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_lora::{AdapterConfig, LoraBuildConfig, LoraInitMode, Tower};

use crate::common;

/// The OpenCLIP fixture: one 1-layer text tower and one 1-layer vision tower
/// in a single `open_clip_model.safetensors`, described by
/// `open_clip_config.json` (no `model_type` field at all — the checkpoint
/// family `EncoderFamily::OpenClip` exists to name).
fn tiny_open_clip_dir() -> PathBuf {
    common::fixture("tiny_open_clip")
}

fn tiny_open_clip_model() -> String {
    "local:".to_string() + tiny_open_clip_dir().to_str().unwrap()
}

/// The HF-CLAP fixture: a 4-stage HTSAT-Swin audio tower plus its
/// `preprocessor_config.json` front-end geometry.
fn htsat_clap_tiny_dir() -> PathBuf {
    common::cookbook_fixture("htsat_clap_tiny")
}

fn htsat_clap_tiny_model() -> String {
    "local:".to_string() + htsat_clap_tiny_dir().to_str().unwrap()
}

fn triplet_columns() -> Vec<String> {
    vec![
        "anchor".to_string(),
        "positive".to_string(),
        "negative".to_string(),
    ]
}

/// A short encoder-adapters config: `target_modules` non-empty is what selects
/// the encoder-adapters arm at all. Two epochs over a handful of tiny rows is
/// enough for a non-zero LoRA delta at this learning rate; the point of these
/// tests is the routing and the round-trip, never convergence.
fn tower_config(target_modules: &[&str], epochs: usize, learning_rate: f64) -> FineTuneConfig {
    FineTuneConfig {
        epochs,
        batch_size: 4,
        learning_rate,
        lora_rank: 4,
        warmup_steps: 0,
        lr_schedule: LrSchedule::Constant,
        validation_fraction: 0.0,
        early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
        target_modules: target_modules.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

/// Fetch a fine-tuned model's published adapter bundle into a local directory.
async fn adapter_dir_for_model(session: &InferenceSession, model_id: &str) -> PathBuf {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .expect("fine-tuned model registered in catalog");
    let prefix = record.artifact_path.expect("artifact_path");
    let prefix_url = jammi_db::storage::StorageUrl::parse(&prefix).unwrap();
    session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("published adapter fetches and verifies")
        .dir()
        .to_path_buf()
}

/// Read a published bundle's `adapter_config.json` and assert it is an
/// encoder-adapters record naming `model_type` and `tower`; return the parsed
/// config so the reference build below can reuse the EXACT LoRA geometry the
/// run trained with (rank, alpha, targets, rank pattern) rather than a
/// hand-written guess that could silently differ.
fn expect_encoder_adapter(dir: &Path, model_type: &str, tower: Tower) -> AdapterConfig {
    let raw = std::fs::read_to_string(dir.join("adapter_config.json"))
        .expect("a published encoder-adapters bundle carries adapter_config.json");
    let saved: SavedAdapter = serde_json::from_str(&raw).expect("adapter_config.json parses");
    let cfg = match saved {
        SavedAdapter::EncoderAdapters(cfg) => *cfg,
        SavedAdapter::ProjectionHead(_) => panic!(
            "a run with non-empty target_modules must save an EncoderAdapters adapter, \
             got a projection head: {raw}"
        ),
    };
    assert_eq!(
        cfg.model_type, model_type,
        "the adapter must record the base ARCHITECTURE family's canonical id"
    );
    assert_eq!(
        cfg.tower,
        Some(tower),
        "a multi-tower checkpoint's adapter must record WHICH tower it installs on"
    );
    cfg
}

/// The inference-shaped [`LoraBuildConfig`] for a saved adapter — the same
/// values `CandleBackend::load` reconstructs from the persisted config, so the
/// reference tower below is wrapped at the identical sites and ranks. `ZerosB`
/// init and seed `0` are inert: every A/B tensor is overwritten by the adapter
/// file this build reads.
fn inference_lora(cfg: &AdapterConfig) -> LoraBuildConfig<'_> {
    LoraBuildConfig {
        target_modules: &cfg.target_modules,
        layers_to_transform: &cfg.layers_to_transform,
        lora_rank: cfg.lora_rank,
        lora_alpha: cfg.lora_alpha,
        use_rslora: cfg.use_rslora,
        lora_dropout: None,
        rank_pattern: &cfg.rank_pattern,
        init_mode: LoraInitMode::ZerosB,
        seed: 0,
    }
}

/// The serving path's L2 normalization, op for op
/// (`model::backend::candle::l2_normalize`, which is crate-private). Only the
/// IMAGE arm needs it: the vision tower's `forward` output is unnormalized and
/// `forward_image_embedding` normalizes it, whereas the text and audio towers
/// emit already-normalized embeddings that serving passes through untouched.
/// Reproduced as the identical candle op sequence so the comparison stays
/// bit-exact.
fn l2_normalize(t: &Tensor) -> Tensor {
    let norm = t
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .clamp(1e-12, f64::MAX)
        .unwrap();
    t.broadcast_div(&norm).unwrap()
}

/// Read a checkpoint's `preprocessor_config.json` into the front-end geometry
/// the serving audio path loads. `CandleBackend`'s own reader is private, so
/// this is an independent transcription of the SAME field names — which is
/// what makes the resulting reference an independent oracle rather than a
/// second call into the code under test. A field-name divergence would show up
/// as a spectrogram mismatch, i.e. as a loud failure here, never as a silent
/// agreement.
fn clap_frontend(path: &Path) -> audio_preprocess::ClapFrontendConfig {
    let prep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let u = |k: &str| prep[k].as_u64().unwrap_or_else(|| panic!("missing '{k}'"));
    let f = |k: &str| prep[k].as_f64().unwrap_or_else(|| panic!("missing '{k}'"));
    audio_preprocess::ClapFrontendConfig {
        n_mels: u("feature_size") as usize,
        sample_rate: u("sampling_rate") as u32,
        fft_window_size: u("fft_window_size") as usize,
        hop_length: u("hop_length") as usize,
        frequency_min: f("frequency_min"),
        frequency_max: f("frequency_max"),
        max_length_s: u("max_length_s") as u32,
    }
}

fn row0(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()[0].clone()
}

fn assert_bit_equal(served: &[f32], reference: &[f32], label: &str) {
    assert_eq!(
        served.len(),
        reference.len(),
        "{label}: served and reference embeddings must have the same width"
    );
    for (i, (a, b)) in served.iter().zip(reference).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: component {i} differs bit-for-bit (served {a}, reference {b}) — the \
             served model is not running the adapter this run saved onto the tower it \
             names"
        );
    }
}

fn assert_differs(base: &[f32], tuned: &[f32], label: &str) {
    let diff: f32 = base
        .iter()
        .zip(tuned)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    assert!(
        diff > 1e-6,
        "{label}: the fine-tuned embedding is identical to the base model's — either the \
         LoRA delta is zero or serving discarded the adapter (sum |Δ| = {diff})"
    );
}

// =============================================================================
// A3 (text): the OpenCLIP TEXT tower
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn open_clip_text_tower_adapter_trains_and_serves() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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

    const PROBE: &str = "a photograph of a small round object";
    let base_model = tiny_open_clip_model();
    let base_embedding = session
        .encode_text_query(&base_model, PROBE)
        .await
        .expect("the OpenCLIP text tower serves the base model");

    let job = session
        .fine_tune(
            "training",
            &base_model,
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            // `in_proj` is the fused-QKV site, `c_fc` the MLP's first linear —
            // both CLIP-text site names, neither of which exists on a BERT
            // tower. A run that fell back to BERT could not even match them.
            Some(tower_config(&["in_proj", "c_fc"], 2, 5e-3)),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("the text-tower fine-tune completes");

    // (1) Identity.
    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let cfg = expect_encoder_adapter(&adapter_dir, "open_clip", Tower::Text);

    // (2) Non-vacuity.
    let tuned_embedding = session
        .encode_text_query(job.model_id(), PROBE)
        .await
        .expect("the fine-tuned OpenCLIP text model resolves and serves");
    assert_differs(&base_embedding, &tuned_embedding, "open_clip text");

    // (3) Mechanism, bit-for-bit.
    let device = Device::Cpu;
    let fixture = tiny_open_clip_dir();
    let open_clip_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.join("open_clip_config.json")).unwrap(),
    )
    .unwrap();
    let text_config =
        jammi_encoders::ClipTextConfig::from_open_clip_config(&open_clip_config).unwrap();
    let tower = jammi_encoders::ClipText::builder()
        .lora(inference_lora(&cfg))
        .backbone_dtype(DType::F32)
        .adapter(Some(&adapter_dir.join("adapter.safetensors")))
        .build(
            &[&fixture.join("open_clip_model.safetensors")],
            &text_config,
            &device,
            &VarMap::new(),
        )
        .expect("the saved adapter loads onto an independently built text tower");

    // The serving text path tokenizes with the model's own tokenizer,
    // truncated to the tower's `context_length`, then feeds the padded
    // `[1, seq]` id/mask pair straight to `forward_pooled` (which for the
    // OpenCLIP text wrapper IS `ClipText::forward`, with no pooling or
    // normalization on top).
    let tokenizer = TokenizerWrapper::from_file(&fixture.join("tokenizer.json")).unwrap();
    let encoding = tokenizer
        .encode_batch(&[PROBE], Some(text_config.context_length))
        .unwrap();
    let cols = encoding.input_ids[0].len();
    let input_ids = Tensor::from_vec(encoding.input_ids.concat(), (1, cols), &device).unwrap();
    let mask = Tensor::from_vec(encoding.attention_masks.concat(), (1, cols), &device).unwrap();
    let reference = row0(&tower.forward(&input_ids, &mask).unwrap());

    assert_bit_equal(&tuned_embedding, &reference, "open_clip text");
}

// =============================================================================
// A3 (image): the OpenCLIP VISION tower
// =============================================================================

/// Every image in the corpus, grouped by shape family (the `img_{family}_{n}`
/// naming), so a triplet's positive is a same-family sibling and its negative
/// belongs to another family. Deterministic (`BTreeMap`, sorted read) —
/// family J: no unseeded RNG anywhere in the fixture construction.
fn image_corpus_by_family() -> std::collections::BTreeMap<String, Vec<Vec<u8>>> {
    let dir = common::cookbook_fixture("tiny_image_corpus");
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    names.sort();

    let mut families: std::collections::BTreeMap<String, Vec<Vec<u8>>> = Default::default();
    for path in names {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        // `img_{family}_{index}`
        let family = stem
            .split('_')
            .nth(1)
            .expect("corpus images are named img_{family}_{n}")
            .to_string();
        families
            .entry(family)
            .or_default()
            .push(std::fs::read(&path).unwrap());
    }
    families
}

/// Write an `(anchor, positive, negative)` Binary Parquet of encoded PNGs —
/// the image peer of `fine_tune.rs`'s `write_audio_triplets`. Identical column
/// shape and cell type; only the payload's modality differs, which is exactly
/// the `MediaTriplet` design (the job's task, not the bytes, says which).
fn write_image_triplets(dir: &Path) -> PathBuf {
    use arrow::array::{ArrayRef, BinaryArray, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let families = image_corpus_by_family();
    let fam_names: Vec<&String> = families.keys().collect();

    let (mut anchors, mut positives, mut negatives) = (Vec::new(), Vec::new(), Vec::new());
    for (fi, fam) in fam_names.iter().enumerate() {
        let imgs = &families[*fam];
        let other = &families[fam_names[(fi + 1) % fam_names.len()]];
        for (ci, anchor) in imgs.iter().enumerate() {
            anchors.push(anchor.clone());
            positives.push(imgs[(ci + 1) % imgs.len()].clone());
            negatives.push(other[ci % other.len()].clone());
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
    let path = dir.join("image_triplets.parquet");
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

#[tokio::test(flavor = "multi_thread")]
async fn open_clip_vision_tower_adapter_trains_and_serves() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let triplets = write_image_triplets(dir.path());
    session
        .add_source(
            "image_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", triplets.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let probe_bytes =
        std::fs::read(common::cookbook_fixture("tiny_image_corpus").join("img_circle_0.png"))
            .unwrap();
    let base_model = tiny_open_clip_model();
    let base_embedding = session
        .encode_image_query(&base_model, &probe_bytes)
        .await
        .expect("the OpenCLIP vision tower serves the base model");

    let job = session
        .fine_tune(
            "image_triplets",
            &base_model,
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::ImageEmbedding,
            Some(tower_config(&["in_proj", "c_fc"], 2, 5e-3)),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("the vision-tower fine-tune completes");

    // (1) Identity.
    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let cfg = expect_encoder_adapter(&adapter_dir, "open_clip", Tower::Vision);

    // (2) Non-vacuity.
    let tuned_embedding = session
        .encode_image_query(job.model_id(), &probe_bytes)
        .await
        .expect("the fine-tuned OpenCLIP vision model resolves and serves");
    assert_differs(&base_embedding, &tuned_embedding, "open_clip vision");

    // (3) Mechanism, bit-for-bit.
    let device = Device::Cpu;
    let fixture = tiny_open_clip_dir();
    let open_clip_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.join("open_clip_config.json")).unwrap(),
    )
    .unwrap();
    let vision_config =
        jammi_encoders::OpenClipVisionConfig::from_open_clip_config(&open_clip_config).unwrap();
    let tower = jammi_encoders::OpenClipVisionTransformer::builder()
        .lora(inference_lora(&cfg))
        .backbone_dtype(DType::F32)
        .adapter(Some(&adapter_dir.join("adapter.safetensors")))
        .build(
            &[&fixture.join("open_clip_model.safetensors")],
            &vision_config,
            &device,
            &VarMap::new(),
        )
        .expect("the saved adapter loads onto an independently built vision tower");

    let image = image::load_from_memory(&probe_bytes).unwrap();
    let pixels = image_preprocess::preprocess_image_batch(
        &[image],
        vision_config.image_size as u32,
        &vision_config.preprocess_mean,
        &vision_config.preprocess_std,
        &device,
    )
    .unwrap();
    // The vision tower's own output is UNnormalized; the serving image path
    // L2-normalizes it before returning (unlike text/audio).
    let reference = row0(&l2_normalize(&tower.forward(&pixels).unwrap()));

    assert_bit_equal(&tuned_embedding, &reference, "open_clip vision");
}

// =============================================================================
// A3 (audio): the HTSAT-Swin CLAP audio tower
// =============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn clap_audio_tower_adapter_trains_and_serves() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let triplets = crate::fine_tune::write_audio_triplets(dir.path());
    session
        .add_source(
            "audio_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", triplets.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let probe_bytes =
        std::fs::read(common::cookbook_fixture("tiny_audio_corpus").join("clip_sine_0.wav"))
            .unwrap();
    let base_model = htsat_clap_tiny_model();
    let base_embedding = session
        .encode_audio_query(&base_model, &probe_bytes)
        .await
        .expect("the CLAP audio tower serves the base model");

    let job = session
        .fine_tune(
            "audio_triplets",
            &base_model,
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::AudioEmbedding,
            // `query`/`value` are the Swin blocks' attention sites (indexed by
            // STAGE), `linear1` the audio projection head's first linear — an
            // UNINDEXED site, so this also exercises the `layer_idx: None`
            // path end to end.
            Some(tower_config(&["query", "value", "linear1"], 2, 5e-3)),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("the audio-tower fine-tune completes");

    // (1) Identity.
    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let cfg = expect_encoder_adapter(&adapter_dir, "clap_audio_model", Tower::Audio);

    // (2) Non-vacuity.
    let tuned_embedding = session
        .encode_audio_query(job.model_id(), &probe_bytes)
        .await
        .expect("the fine-tuned CLAP audio model resolves and serves");
    assert_differs(&base_embedding, &tuned_embedding, "clap audio");

    // (3) Mechanism, bit-for-bit.
    let device = Device::Cpu;
    let fixture = htsat_clap_tiny_dir();
    let model_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("config.json")).unwrap())
            .unwrap();
    let audio_config =
        jammi_encoders::HtsatAudioConfig::from_hf_clap_config(&model_config).unwrap();
    let tower = jammi_encoders::HtsatAudio::builder()
        .lora(inference_lora(&cfg))
        .backbone_dtype(DType::F32)
        .adapter(Some(&adapter_dir.join("adapter.safetensors")))
        .build(
            &[&fixture.join("model.safetensors")],
            &audio_config,
            &device,
            &VarMap::new(),
        )
        .expect("the saved adapter loads onto an independently built audio tower");

    let frontend = clap_frontend(&fixture.join("preprocessor_config.json"));
    let decoded = audio_preprocess::decode_audio_bytes(&probe_bytes).unwrap();
    let (features, is_longer) =
        audio_preprocess::preprocess_clap_fusion(&[decoded], &frontend, &device).unwrap();
    // The CLAP audio tower emits L2-normalized embeddings itself, so the
    // serving path returns them untouched.
    let reference = row0(&tower.forward(&features, &is_longer).unwrap());

    assert_bit_equal(&tuned_embedding, &reference, "clap audio");
}

// =============================================================================
// A4 — the refusals
// =============================================================================

/// A4(a). RED at base: the worker's `_ => BERT` arm coerced any config that
/// deserialized as a `BertConfig` — a `tiny_bert` directory whose `model_type`
/// says `gpt2` does — and trained a BERT tower over it, publishing an adapter
/// that claimed the architecture `gpt2`. There is no GPT-2 loader in this
/// crate; the only honest answer is a typed refusal naming the model_type and
/// the supported set.
#[tokio::test(flavor = "multi_thread")]
async fn unsupported_model_type_refuses_instead_of_coercing_to_bert() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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

    // A byte-for-byte `tiny_bert` copy with ONE field changed. Everything else
    // still parses as a `BertConfig`, which is precisely why the old default
    // arm trained happily.
    let model_dir = dir.path().join("gpt2_flavoured_bert");
    std::fs::create_dir_all(&model_dir).unwrap();
    let fixture = common::cookbook_fixture("tiny_bert");
    for name in ["model.safetensors", "tokenizer.json"] {
        std::fs::copy(fixture.join(name), model_dir.join(name)).unwrap();
    }
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("config.json")).unwrap())
            .unwrap();
    config["model_type"] = serde_json::json!("gpt2");
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let job = session
        .fine_tune(
            "training",
            &format!("local:{}", model_dir.display()),
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(tower_config(&["query", "value"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("an unsupported model_type must fail the job, never train a coerced BERT");
    let msg = err.to_string();
    assert!(
        msg.contains("gpt2") && msg.contains("supported"),
        "the refusal must name the offending model_type and the supported set, got: {msg}"
    );
}

/// A4(a'). The COMPLEMENT of the refusal above, and the divergence W2c closed:
/// a `tiny_bert` copy whose `config.json` OMITS `model_type` entirely (the
/// older sentence-transformers / hand-written bare-export shape) must be ONE
/// architecture to BOTH readers.
///
/// RED at W2b: `EncoderFamily::from_config` answered `None` for an absent key,
/// so the fine-tune worker refused this directory outright while the serving
/// loader's own `unwrap_or("bert")` was simultaneously loading the identical
/// bytes as BERT — training and serving disagreeing on one file, which is the
/// single thing `model::arch` exists to prevent.
///
/// One deterministic assertion per side of that seam:
///
/// * SERVING — the embedding is bit-identical to the unmodified fixture's for
///   the same probe (K4: deleting a key the loader only reads to pick an arm
///   changes no loaded byte; an equality that would also hold if BOTH sides
///   silently failed is ruled out by the two `expect`s, either of which fires
///   first on a refusal).
/// * TRAINING — the job completes and publishes an encoder-adapters bundle
///   recording the `bert` architecture. A run that trained a projection head
///   instead, or that recorded some other architecture, fails here.
#[tokio::test(flavor = "multi_thread")]
async fn an_absent_model_type_trains_and_serves_as_bert() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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

    // A byte-for-byte `tiny_bert` copy with ONE key REMOVED — the mirror of
    // the test above, which changes that same key's VALUE.
    let fixture = common::cookbook_fixture("tiny_bert");
    let model_dir = dir.path().join("bert_without_model_type");
    std::fs::create_dir_all(&model_dir).unwrap();
    for name in ["model.safetensors", "tokenizer.json"] {
        std::fs::copy(fixture.join(name), model_dir.join(name)).unwrap();
    }
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("config.json")).unwrap())
            .unwrap();
    assert!(
        config
            .as_object_mut()
            .expect("config.json is a JSON object")
            .remove("model_type")
            .is_some(),
        "the fixture must DECLARE a model_type for its removal to be the thing under test"
    );
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    const PROBE: &str = "a short sentence about a small round object";
    let bare_model = format!("local:{}", model_dir.display());
    let declared_model = "local:".to_string() + fixture.to_str().unwrap();

    // (1) Serving.
    let served = session
        .encode_text_query(&bare_model, PROBE)
        .await
        .expect("a checkpoint that declares no model_type still serves");
    let reference = session
        .encode_text_query(&declared_model, PROBE)
        .await
        .expect("the unmodified fixture serves");
    assert_bit_equal(&served, &reference, "absent model_type");

    // (2) Training. `query`/`value` are BERT attention site names; a run that
    // refused the checkpoint, or routed it anywhere but the BERT text tower,
    // never gets here.
    let job = session
        .fine_tune(
            "training",
            &bare_model,
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(tower_config(&["query", "value"], 1, 1e-3)),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("a checkpoint that declares no model_type fine-tunes as BERT");

    let adapter_dir = adapter_dir_for_model(&session, job.model_id()).await;
    let raw = std::fs::read_to_string(adapter_dir.join("adapter_config.json"))
        .expect("a published encoder-adapters bundle carries adapter_config.json");
    let saved: SavedAdapter = serde_json::from_str(&raw).expect("adapter_config.json parses");
    match saved {
        // A single-tower text family records no `tower` (its adapter has
        // exactly one place to install); the architecture id is the assertion.
        SavedAdapter::EncoderAdapters(cfg) => assert_eq!(
            cfg.model_type, "bert",
            "the adapter must record the family the ONE predicate resolved, got: {raw}"
        ),
        SavedAdapter::ProjectionHead(_) => panic!(
            "a run with non-empty target_modules must save an EncoderAdapters adapter, \
             got a projection head: {raw}"
        ),
    }
}

/// A4(b). The OpenCLIP checkpoint has no audio tower; an `audio_embedding`
/// encoder-adapters job over it is a `(family, task)` refusal, not a silently
/// mis-routed text or vision build.
#[tokio::test(flavor = "multi_thread")]
async fn open_clip_base_with_audio_task_refuses() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let triplets = crate::fine_tune::write_audio_triplets(dir.path());
    session
        .add_source(
            "audio_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", triplets.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let job = session
        .fine_tune(
            "audio_triplets",
            &tiny_open_clip_model(),
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::AudioEmbedding,
            Some(tower_config(&["in_proj"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("audio_embedding on an OpenCLIP checkpoint has no tower to adapt");
    let msg = err.to_string();
    assert!(
        msg.contains("audio_embedding") && msg.contains("OpenClip"),
        "the refusal must name both the task and the base architecture family, got: {msg}"
    );
}

/// A4(c). A hand-written adapter claiming `clap_audio_model` installed on a
/// BERT base: the load seam must refuse on the FAMILY mismatch rather than
/// resolve the adapter and silently serve the unadapted base.
///
/// Written as an on-disk `adapter_config.json` (the shape
/// `content_digest.rs::write_projection_adapter` uses) and pushed through the
/// real `CandleBackend::load` by overriding `adapter_path` on an otherwise
/// real resolve.
#[tokio::test(flavor = "multi_thread")]
async fn cross_family_adapter_refuses_at_load() {
    use jammi_ai::model::backend::candle::CandleBackend;
    use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
    use jammi_ai::model::resolver::ModelResolver;
    use jammi_ai::model::ModelSource;

    let scratch = TempDir::new().unwrap();
    let adapter_dir = scratch.path().join("adapter");
    std::fs::create_dir_all(&adapter_dir).unwrap();
    std::fs::write(
        adapter_dir.join("adapter_config.json"),
        serde_json::to_string(&serde_json::json!({
            "adapter_type": "encoder_adapters",
            "model_type": "clap_audio_model",
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "use_rslora": false,
            "target_modules": ["query", "value"],
            "tower": "audio",
        }))
        .unwrap(),
    )
    .unwrap();
    let mut weights: std::collections::HashMap<String, Tensor> = Default::default();
    weights.insert(
        "marker".to_string(),
        Tensor::new(&[0.0f32], &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&weights, adapter_dir.join("adapter.safetensors")).unwrap();

    let catalog_dir = TempDir::new().unwrap();
    let catalog = Arc::new(
        jammi_db::catalog::Catalog::open(catalog_dir.path())
            .await
            .unwrap(),
    );
    let resolver = ModelResolver::new(catalog, common::test_artifact_store()).unwrap();
    let mut resolved = resolver
        .resolve(
            &ModelSource::local(common::cookbook_fixture("tiny_bert")),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();
    resolved.adapter_path = Some(adapter_dir);

    // `LoadedModel` is not `Debug`, so the error is taken through a match
    // rather than `expect_err`.
    let msg = match CandleBackend.load(
        &resolved,
        &DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        },
    ) {
        Ok(_) => panic!(
            "a clap_audio_model adapter on a BERT base must refuse, never silently serve \
             the unadapted base under the fine-tuned model's id"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("ClapAudio") && msg.contains("Bert"),
        "the refusal must name BOTH the adapter's family and the base's, got: {msg}"
    );
}

/// A4(d), arm 1: an image task over TEXT triplet columns. The columns are
/// strings, so there is nothing to decode; the refusal must name the task.
#[tokio::test(flavor = "multi_thread")]
async fn image_task_on_text_triplets_refuses_naming_the_task() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    session
        .add_source(
            "text_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_triplets.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let job = session
        .fine_tune(
            "text_triplets",
            &tiny_open_clip_model(),
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::ImageEmbedding,
            Some(tower_config(&["in_proj"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("an image job over string triplet columns has nothing to decode");
    let msg = err.to_string();
    assert!(
        msg.contains("image_embedding") || msg.contains("media triplets"),
        "the refusal must name the task and the expected column shape, got: {msg}"
    );
}

/// A4(d), arm 2: a TEXT task over BINARY triplet columns. The mirror image —
/// and the message must point the caller at the media tasks by name rather
/// than dying on an opaque UTF-8 cast failure.
#[tokio::test(flavor = "multi_thread")]
async fn text_task_on_binary_triplets_refuses_naming_the_media_tasks() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let triplets = write_image_triplets(dir.path());
    session
        .add_source(
            "image_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", triplets.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let job = session
        .fine_tune(
            "image_triplets",
            &tiny_open_clip_model(),
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(tower_config(&["in_proj"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("a text job over binary triplet columns cannot tokenize the cells");
    let msg = err.to_string();
    assert!(
        msg.contains("text_embedding") && msg.contains("image_embedding"),
        "the refusal must name the submitted task AND the media task the caller wants, \
         got: {msg}"
    );
}

/// A4(e), arm 1: a `target_modules` list that selects NOTHING on the tower the
/// job's `(family, task)` picked must FAIL the job.
///
/// `q_proj` is a real selector on plenty of decoder checkpoints and on nothing
/// in an OpenCLIP tower, whose four sites are `in_proj` / `out_proj` / `c_fc` /
/// `c_proj` — exactly the plausible-but-wrong string an operator carries over
/// from another architecture's recipe.
///
/// RED at 5bf8abdb, and not by a missing symbol: `build_encoder_adapters` never
/// consulted `trainable_params()` there, `optimizer::clip_and_step` treats an
/// EMPTY trainable set as the one unambiguously benign reading and does not
/// even warn, so the job ran its epoch over an empty `GradStore`, published an
/// `adapter.safetensors` with no A/B tensors and reported SUCCESS — `job.wait()`
/// returned `Ok`, so this test's `expect_err` panics there. Traced to the
/// mechanism rather than asserted: with the new `trainable_params().is_empty()`
/// refusal removed at this tip, both arms fail exactly that way
/// (`expect_err(..): ()`), which is the base behaviour restored.
///
/// The message assertion is the non-vacuous half (family F): "the job failed"
/// alone would also pass on a decode error, an OOM, or a missing fixture, so
/// the text must carry this tower's OWN site vocabulary — `in_proj` and
/// `c_proj`, which no other tower in this workspace offers.
#[tokio::test(flavor = "multi_thread")]
async fn unmatched_target_modules_refuse_instead_of_training_nothing() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    session
        .add_source(
            "text_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_triplets.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let job = session
        .fine_tune(
            "text_triplets",
            &tiny_open_clip_model(),
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(tower_config(&["q_proj"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job.wait().await.expect_err(
        "a target_modules list matching no site on the CLIP text tower must fail the job, \
         never publish an empty adapter under a fine-tuned model id",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("q_proj") && msg.contains("in_proj") && msg.contains("c_proj"),
        "the refusal must echo the submitted selector AND name this tower's real site \
         names, got: {msg}"
    );
}

/// A4(e), arm 2: the same refusal on a BERT-family tower, whose site
/// vocabulary is the DOTTED checkpoint path (`attention.self.query`, …) rather
/// than the short suffix form the guide's recipe table shows. A caller who
/// reads the message must be able to paste a name straight out of it, so the
/// dotted form is what the message has to print — the short `["query",
/// "value"]` form works only because `should_apply_lora` also accepts a
/// suffix.
///
/// RED at 5bf8abdb for the identical reason as arm 1: the job succeeded there.
#[tokio::test(flavor = "multi_thread")]
async fn unmatched_target_modules_on_bert_name_the_dotted_site_paths() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
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

    let job = session
        .fine_tune(
            "training",
            &format!("local:{}", common::cookbook_fixture("tiny_bert").display()),
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(tower_config(&["not_a_module"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("a nonsense selector on a BERT tower must fail the job");
    let msg = err.to_string();
    assert!(
        msg.contains("not_a_module") && msg.contains("attention.self.query"),
        "the refusal must echo the submitted selector AND name the BERT tower's real \
         (dotted) site names, got: {msg}"
    );
}

/// The training audio front end applies serving's own mel-bin guard.
///
/// The front-end geometry and the tower's input contract come from two
/// INDEPENDENT files in the checkpoint (`preprocessor_config.json`'s
/// `feature_size`, `config.json`'s `num_mel_bins`), so nothing but an explicit
/// comparison stops a `[B, 4, time, 33]` batch reaching a tower built for 32
/// mel bins. `CandleModel::embed_audio` has refused exactly this since the
/// audio serving path landed; the trainer's `audio_encoder_input` did not, so
/// the same misconfigured checkpoint served a typed error and trained a
/// silently mis-binned spectrogram.
///
/// One deterministic fixture: a byte-for-byte `htsat_clap_tiny` copy with ONE
/// field changed (`feature_size` 32 → 33), the same one-field-mutation shape
/// [`unsupported_model_type_refuses_instead_of_coercing_to_bert`] uses.
#[tokio::test(flavor = "multi_thread")]
async fn audio_training_refuses_a_mel_bin_mismatch_like_serving_does() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let triplets = crate::fine_tune::write_audio_triplets(dir.path());
    session
        .add_source(
            "audio_triplets",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", triplets.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let fixture = htsat_clap_tiny_dir();
    let model_dir = dir.path().join("htsat_wrong_mels");
    std::fs::create_dir_all(&model_dir).unwrap();
    for name in ["model.safetensors", "config.json"] {
        std::fs::copy(fixture.join(name), model_dir.join(name)).unwrap();
    }
    let mut prep: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.join("preprocessor_config.json")).unwrap(),
    )
    .unwrap();
    let good = prep["feature_size"].as_u64().unwrap();
    prep["feature_size"] = serde_json::json!(good + 1);
    std::fs::write(
        model_dir.join("preprocessor_config.json"),
        serde_json::to_string(&prep).unwrap(),
    )
    .unwrap();

    let job = session
        .fine_tune(
            "audio_triplets",
            &format!("local:{}", model_dir.display()),
            &triplet_columns(),
            FineTuneMethod::Lora,
            ModelTask::AudioEmbedding,
            Some(tower_config(&["query", "value"], 1, 1e-3)),
        )
        .await
        .unwrap();
    let err = job.wait().await.expect_err(
        "a feature_size that disagrees with the tower's num_mel_bins must fail the job",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("num_mel_bins") && msg.contains(&(good + 1).to_string()),
        "the refusal must name the mismatched quantity and echo the front end's own \
         feature_size, got: {msg}"
    );
}
