use crate::common;

use jammi_ai::inference::adapter::{
    BackendOutput, ClassificationAdapter, EmbeddingAdapter, OutputAdapter,
};

use arrow::array::Array;
#[cfg(feature = "live-hub-tests")]
use arrow::array::FixedSizeListArray;
#[cfg(feature = "live-hub-tests")]
use arrow::datatypes::DataType;

// ─── Hermetic tests (no network, no live models) ───────────────────────────

// --- Contract tests: OutputAdapter behavioral invariants ---

#[test]
fn contract_embedding_adapter_schema_matches_adapt_output() {
    let adapter = EmbeddingAdapter::new(384);
    let schema_fields = adapter.output_schema();

    let output = BackendOutput {
        float_outputs: vec![vec![0.0_f32; 384]],
        string_outputs: vec![],
        row_status: vec![true],
        row_errors: vec![String::new()],
        shapes: vec![(1, 384)],
    };

    let columns = adapter.adapt(&output, 1).unwrap();
    assert_eq!(
        columns.len(),
        schema_fields.len(),
        "adapt() column count must match output_schema() field count"
    );

    for (col, field) in columns.iter().zip(schema_fields.iter()) {
        assert_eq!(
            col.data_type(),
            field.data_type(),
            "Column '{}' type mismatch",
            field.name()
        );
    }
}

#[test]
fn contract_classification_adapter_schema_matches_adapt_output() {
    let adapter = ClassificationAdapter;
    let schema_fields = adapter.output_schema();

    let output = BackendOutput {
        float_outputs: vec![vec![0.95]],
        string_outputs: vec![vec!["physics".into()], vec![r#"{"physics":0.95}"#.into()]],
        row_status: vec![true],
        row_errors: vec![String::new()],
        shapes: vec![(1, 1)],
    };

    let columns = adapter.adapt(&output, 1).unwrap();
    assert_eq!(
        columns.len(),
        schema_fields.len(),
        "adapt() column count must match output_schema() field count"
    );

    for (col, field) in columns.iter().zip(schema_fields.iter()) {
        assert_eq!(
            col.data_type(),
            field.data_type(),
            "Column '{}' type mismatch",
            field.name()
        );
    }
}

// ─── Hermetic: OpenCLIP text tower via tiny fixture ─────────────────────────

/// The OpenCLIP text tower must emit text embeddings in the shared CLIP
/// latent space (`embed_dim`), not the per-token transformer `width`. This
/// is the cross-modal invariant: text embeddings and image embeddings must
/// share dimensionality so cosine similarity is well-defined.
#[tokio::test]
async fn text_embeddings_via_open_clip_share_latent_dim_with_vision() {
    use jammi_ai::session::InferenceSession;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use std::sync::Arc;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    let model_id = format!(
        "local:{}",
        common::fixture("tiny_open_clip").to_str().unwrap()
    );

    // Encode a single text query — same path as cross-modal search uses.
    let text_vec = session
        .encode_text_query(&model_id, "a small figure")
        .await
        .unwrap();

    // Encode a single image query through the vision tower.
    let img_bytes = {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([200, 100, 50]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    };
    let img_vec = session
        .encode_image_query(&model_id, &img_bytes)
        .await
        .unwrap();

    // Shared latent space invariant: text and image vectors have the same
    // dimensionality, set by the OpenCLIP config's `embed_dim`.
    assert_eq!(
        text_vec.len(),
        img_vec.len(),
        "OpenCLIP text and image embeddings must share latent dim"
    );
    assert_eq!(text_vec.len(), 16, "tiny_open_clip embed_dim is 16");

    // Both L2-normalized.
    let text_norm: f32 = text_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    let img_norm: f32 = img_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (text_norm - 1.0).abs() < 0.01 && (img_norm - 1.0).abs() < 0.01,
        "Both text ({text_norm}) and image ({img_norm}) vectors should be L2-normalized"
    );

    // Generate text embeddings for a parquet source the same way images are
    // generated. The output `vector` column must have the same FixedSize as
    // the image-side embedding.
    session
        .add_source(
            "captions",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let record = session
        .generate_text_embeddings("captions", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(
        record.dimensions,
        Some(16),
        "text-embedding result table must carry the shared embed_dim"
    );
}

// ─── Live tests (behind feature gate, require HF Hub downloads) ────────────

#[cfg(feature = "live-hub-tests")]
mod live {
    use super::*;
    use arrow::array::{Float32Array, StringArray};
    use candle_core::Device;
    use jammi_ai::model::{ModelSource, ModelTask};
    use jammi_ai::session::InferenceSession;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use serial_test::serial;
    use std::sync::Arc;
    use tempfile::tempdir;

    // PatentCLIP tests share the hf-hub on-disk cache for
    // `patentclip/PatentCLIP_Vit_B`. The per-`ModelCache` single-flight in
    // `cache::ModelCache::get_or_load` deduplicates concurrent loads inside one
    // cache, but each of these tests builds its own cache (and its own
    // `hf_hub::api::sync::Api`). With cargo's default parallel test runner
    // three threads race on the same on-disk weights file and hf-hub's sync
    // API does not coordinate concurrent first-time downloads across `Api`
    // instances. `#[serial]` forces them to run one at a time so the cache
    // populates cleanly on the first attempt.

    async fn setup_with_patents() -> (InferenceSession, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();
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
        (session, dir)
    }

    // Tests 384-dim real model — tiny_bert is only 32-dim, can't cover this.
    #[tokio::test]
    async fn real_model_produces_384_dim_embeddings() {
        let (session, _dir) = setup_with_patents().await;

        let results = session
            .infer(
                "patents",
                &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                ModelTask::TextEmbedding,
                &["abstract".to_string()],
                "id",
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
        let batch = &results[0];

        let vector_col = batch.column_by_name("vector").expect("vector column");
        match vector_col.data_type() {
            DataType::FixedSizeList(inner, 384) => {
                assert_eq!(inner.data_type(), &DataType::Float32);
            }
            other => panic!("Expected FixedSizeList(Float32, 384), got {other:?}"),
        }

        let fsl = vector_col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let status = batch
            .column_by_name("_status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..fsl.len() {
            if status.value(i) == "ok" {
                assert!(!fsl.is_null(i), "OK row {i} should have a vector");
            }
        }
    }

    // Tests batch memory estimation with real model config — tiny_bert doesn't have realistic dimensions.
    #[tokio::test]
    async fn real_model_batch_memory_estimation() {
        use jammi_ai::concurrency::GpuScheduler;
        use jammi_ai::model::backend::DeviceConfig;
        use jammi_ai::model::cache::ModelCache;
        use jammi_ai::model::resolver::ModelResolver;

        let dir = tempdir().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        let resolver =
            ModelResolver::new(Arc::clone(&catalog), crate::common::test_artifact_store()).unwrap();
        let device_config = DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
        };
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config, scheduler);

        let guard = cache
            .get_or_load(
                &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                ModelTask::TextEmbedding,
                None,
            )
            .await
            .unwrap();

        let batch_mem = guard.model.estimate_batch_memory(32, 128);
        assert!(batch_mem > 0, "Batch memory estimate should be positive");
        assert!(
            batch_mem > 20_000_000 && batch_mem < 30_000_000,
            "Expected ~25MB for batch(32,128), got {batch_mem}"
        );
    }

    #[tokio::test]
    async fn live_classification_produces_valid_labels() {
        let (session, _dir) = setup_with_patents().await;

        // Emotion classifier: BERT-based, 6 labels, has tokenizer.json + model.safetensors,
        // single-layer classifier head (classifier.weight + classifier.bias).
        let model = ModelSource::parse("bhadresh-savani/bert-base-uncased-emotion");
        let valid_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"];

        let batches = session
            .infer(
                "patents",
                &model,
                ModelTask::Classification,
                &["abstract".into()],
                "id",
            )
            .await
            .unwrap();

        assert!(!batches.is_empty());
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total_rows > 0);

        for batch in &batches {
            let status = batch
                .column_by_name("_status")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            let labels = batch
                .column_by_name("label")
                .expect("label column")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let confidence = batch
                .column_by_name("confidence")
                .expect("confidence column")
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                if status.value(i) == "ok" {
                    let label = labels.value(i);
                    assert!(
                        valid_labels.contains(&label),
                        "Label should be one of {valid_labels:?}, got '{label}'"
                    );
                    let conf = confidence.value(i);
                    assert!(conf > 0.0 && conf <= 1.0, "Confidence {conf} out of range");
                }
            }
        }

        // At least one row should have confidence > 0.2 (6-class → random baseline ~0.17)
        let any_confident = batches.iter().any(|b| {
            let conf_col = b.column_by_name("confidence").unwrap();
            let arr = conf_col.as_any().downcast_ref::<Float32Array>().unwrap();
            (0..arr.len()).any(|i| !arr.is_null(i) && arr.value(i) > 0.2)
        });
        assert!(
            any_confident,
            "At least one row should have confidence > 0.2"
        );
    }

    #[tokio::test]
    #[serial(patentclip)]
    async fn live_patentclip_produces_512_dim_image_embeddings() {
        use jammi_ai::model::backend::DeviceConfig;
        use jammi_ai::model::cache::ModelCache;
        use jammi_ai::model::resolver::ModelResolver;

        let dir = tempdir().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        let resolver =
            ModelResolver::new(Arc::clone(&catalog), crate::common::test_artifact_store()).unwrap();
        let device_config = DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
        };
        let scheduler = Arc::new(jammi_ai::concurrency::GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config, scheduler);

        // Load PatentCLIP
        let source = ModelSource::hf("patentclip/PatentCLIP_Vit_B");
        let guard = cache
            .get_or_load(&source, ModelTask::ImageEmbedding, None)
            .await
            .unwrap();

        // Verify embedding dimension
        let dim = guard.model.embedding_dim().unwrap();
        assert_eq!(dim, 512, "PatentCLIP should produce 512-dim embeddings");

        // Create a synthetic test image (white 100x150 with a black rectangle)
        let mut img = image::RgbImage::from_pixel(100, 150, image::Rgb([255, 255, 255]));
        for y in 30..80 {
            for x in 20..70 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        // Embed via forward pass
        let binary_array = Arc::new(arrow::array::BinaryArray::from(vec![png_bytes.as_slice()]))
            as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[binary_array], ModelTask::ImageEmbedding)
            .unwrap();

        // Verify output shape
        assert_eq!(output.shapes, vec![(1, 512)]);
        assert_eq!(output.float_outputs[0].len(), 512);
        assert!(output.row_status[0], "Row should be marked as OK");

        // Verify L2-normalized (norm ≈ 1.0)
        let norm: f32 = output.float_outputs[0]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be L2-normalized, got norm={norm}"
        );

        // Verify non-trivial (not all zeros)
        let max_val = output.float_outputs[0]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_val > 0.01,
            "Embedding should not be all zeros, max={max_val}"
        );
    }

    #[tokio::test]
    #[serial(patentclip)]
    async fn live_patentclip_encode_image_query() {
        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();

        // Create a small test image
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        let vector = session
            .encode_image_query("patentclip/PatentCLIP_Vit_B", &png_bytes)
            .await
            .unwrap();

        assert_eq!(vector.len(), 512);

        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Should be L2-normalized, got norm={norm}"
        );
    }

    #[tokio::test]
    #[serial(patentclip)]
    async fn live_patentclip_generate_image_embeddings_pipeline() {
        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();

        // Create a source with 2 small test images
        let mut images = Vec::new();
        for i in 0..2u8 {
            let img = image::RgbImage::from_pixel(20, 20, image::Rgb([i * 100, 100, 200]));
            let mut buf = Vec::new();
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
            images.push(buf);
        }

        let parquet_path = dir.path().join("test_images.parquet");
        {
            use arrow::array::{ArrayRef, BinaryArray, StringArray};
            use arrow::datatypes::{DataType, Field, Schema};
            use arrow::record_batch::RecordBatch;
            use parquet::arrow::ArrowWriter;

            let schema = Arc::new(Schema::new(vec![
                Field::new("fid", DataType::Utf8, false),
                Field::new("img", DataType::Binary, false),
            ]));
            let keys = Arc::new(StringArray::from(vec!["img_0", "img_1"])) as ArrayRef;
            let imgs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
            let img_array = Arc::new(BinaryArray::from(imgs)) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![keys, img_array]).unwrap();

            let file = std::fs::File::create(&parquet_path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        session
            .add_source(
                "test_imgs",
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", parquet_path.display())),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // This is the exact path that previously failed with PatentCLIP
        let record = session
            .generate_image_embeddings("test_imgs", "patentclip/PatentCLIP_Vit_B", "img", "fid")
            .await
            .unwrap();

        assert_eq!(record.status, "ready");
        assert_eq!(record.row_count, 2);
        assert_eq!(record.dimensions, Some(512));
    }

    // ── Real CLAP (laion/clap-htsat-fused) audio embedding ──────────────────
    //
    // The `jammi-encoders` live test isolates the TOWER: it feeds the committed
    // `input_features` straight to `HtsatAudio::forward`, so a divergence there
    // is a tower-vs-model bug. The two tests below close the loop through the
    // PRODUCTION audio path — `InferenceSession` → `ModelCache` resolves
    // `laion/clap-htsat-fused` → `forward_audio_embedding` (decode → resample →
    // `preprocess_clap_fusion` front-end → tower) — so they additionally
    // exercise the front-end DSP the tower test bypasses.

    const REAL_CLAP_ID: &str = "laion/clap-htsat-fused";

    /// Build little-endian 16-bit mono PCM WAV bytes at `sample_rate` from int16
    /// samples — the exact container `transformers`/the e2e golden's waveform was
    /// written as, so the production decode path reconstructs byte-identical PCM.
    fn wav_bytes_mono_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * 2; // mono, 2 bytes/sample
        let mut buf = Vec::with_capacity(44 + data_len as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels = 1
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    /// E2E front-end + tower: encode the SAME seeded 5s waveform the committed
    /// real golden was produced from — rebuilt in-test as 48 kHz mono 16-bit WAV
    /// from `waveform_i16` — through the production `encode_audio_query` path
    /// (`InferenceSession`/`ModelCache` → `forward_audio_embedding`) and assert
    /// the 512-d result matches the committed `embedding` by direction.
    ///
    /// Unlike the tower-isolation test, the front-end is in the loop: the Rust
    /// `preprocess_clap_fusion` decode → resample → log-mel runs end-to-end. The
    /// waveform is at the 48 kHz target rate, so resample is identity. Crucially,
    /// jammi's front-end now marks every clip `is_longer=true` (its deterministic
    /// analogue of `ClapFeatureExtractor`'s fusion promotion), so the tower runs
    /// the AFF fusion path and reproduces the committed CANONICAL
    /// `get_audio_features` embedding — the vector the CLAP ecosystem searches
    /// with. (Emitting the global-only flag instead lands ~0.73 cosine off.)
    ///
    /// The front-end's dB log-mel differs from `transformers` by ~1.6e-3 max-abs
    /// only on the worst near-floor low-energy cells; those cells carry no signal
    /// the tower keys on, so the propagated embedding is HIGH — in practice it
    /// rounds to cosine ≈1.0, indistinguishable from the tower-isolation test.
    ///
    /// MEASURED cosine on this box: 1.0000002 (fp32 cosine rounds just over 1.0;
    /// the front-end error does not perceptibly move the embedding direction).
    /// The floor is kept at 0.999 — well below the measured value yet high enough
    /// that a front-end DSP regression or a tower/gate regression (which collapses
    /// cosine well below 0.99, e.g. the global-only gate's ~0.73) fails, with
    /// ample margin for the resampler/DSP fp variation a different host may show.
    /// Derived from the measured value; never tuned to pass.
    #[tokio::test]
    #[serial(real_clap)]
    async fn live_real_clap_e2e_matches_committed_embedding() {
        const MIN_COS_E2E: f32 = 0.999;

        let real_dir = common::cookbook_fixture("htsat_clap_real");
        let goldens =
            candle_core::safetensors::load(real_dir.join("goldens.safetensors"), &Device::Cpu)
                .expect("load real goldens.safetensors");
        let waveform = goldens
            .get("waveform_i16")
            .expect("waveform_i16 golden")
            .to_vec1::<i16>()
            .expect("waveform_i16 to vec");
        let golden_embedding = goldens
            .get("embedding")
            .expect("embedding golden")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .expect("embedding to vec");

        // ClapProcessor sampling_rate is 48 kHz; the golden waveform is at that
        // rate, so the production decode → resample is identity (48k → 48k) and
        // the only e2e gap vs the golden is the Rust-vs-torch front-end DSP.
        let wav = wav_bytes_mono_i16(&waveform, 48_000);

        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();

        let vector = session
            .encode_audio_query(REAL_CLAP_ID, &wav)
            .await
            .unwrap();
        assert_eq!(vector.len(), 512, "real CLAP audio embedding is 512-d");

        // L2-normalized (the tower normalizes; encode_audio_query passes through).
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "audio embedding should be L2-normalized, got norm={norm}"
        );

        let cos = cosine(&vector, &golden_embedding);
        assert!(
            cos >= MIN_COS_E2E,
            "e2e front-end+tower vs committed real embedding: cosine = {cos} < {MIN_COS_E2E} \
             — a front-end or tower regression, not a tolerance issue"
        );
    }

    /// Behavioral retrieval sanity: embed the 20-clip synthetic timbre corpus (5
    /// families × 4) and the 5 held-out queries through jammi's REAL CLAP audio
    /// path, then check the embeddings are genuinely discriminating — same-family
    /// clips rank above other families.
    ///
    /// OBSERVED on this box (Rust production path, per query → top-1 clip,
    /// cosine):
    ///   q_harmonic → clip_harmonic_* top-1 (same family)
    ///   q_noise    → clip_noise_*    top-1 (same family)
    ///   q_saw      → clip_saw_*      top-1 (same family)
    ///   q_sine     → clip_sine_*     top-1 (same family)
    ///   q_square   → clip_square_*   top-1 (same family)
    /// → 5/5 queries rank a same-family clip top-1.
    ///   mean(intra-family cosine) = 0.9313, mean(inter-family) = 0.4575,
    ///   margin = 0.4738.
    ///
    /// The torch reference shows the identical 5/5 and a 0.4701 margin (the Rust
    /// resampler/front-end shifts cosines by <1e-2, not enough to flip a family).
    /// The assertions are set strictly below the observed values yet far above
    /// what a broken-but-normalizing tower yields: such a tower produces a
    /// near-zero intra-vs-inter margin and ~1-in-5 (random) top-1 hits, so
    /// `top1 ≥ 4/5` and `margin > 0.15` cannot pass on scrambled embeddings. K=4
    /// (not 5) leaves headroom for exactly one borderline family to flip under
    /// cross-host DSP fp variation — saw/square are the closest synthetic pair
    /// (q_saw's 3rd-ranked neighbour is a square clip at ~0.89) — without
    /// weakening the discriminating power. M=0.15 is ~3× below the observed 0.47.
    #[tokio::test]
    #[serial(real_clap)]
    async fn live_real_clap_retrieval_separates_timbre_families() {
        const MIN_TOP1_SAME_FAMILY: usize = 4;
        const MIN_INTRA_INTER_MARGIN: f32 = 0.15;

        let families = ["harmonic", "noise", "saw", "sine", "square"];
        let corpus = common::cookbook_fixture("tiny_audio_corpus");

        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();

        // Embed every corpus clip (4 per family) through the real audio path.
        let mut clip_ids: Vec<String> = Vec::new();
        let mut clip_vecs: Vec<Vec<f32>> = Vec::new();
        for fam in families {
            for variant in 0..4 {
                let id = format!("clip_{fam}_{variant}");
                let bytes = std::fs::read(corpus.join(format!("{id}.wav"))).unwrap();
                let v = session
                    .encode_audio_query(REAL_CLAP_ID, &bytes)
                    .await
                    .unwrap();
                assert_eq!(v.len(), 512);
                clip_ids.push(id);
                clip_vecs.push(v);
            }
        }

        let family_of = |id: &str| -> String { id.split('_').nth(1).unwrap().to_string() };

        // For each query: rank all clips by cosine, record the top-1 family, and
        // accumulate intra- vs inter-family cosines.
        let mut top1_same = 0usize;
        let mut intra: Vec<f32> = Vec::new();
        let mut inter: Vec<f32> = Vec::new();
        for fam in families {
            let qbytes =
                std::fs::read(corpus.join("queries").join(format!("q_{fam}.wav"))).unwrap();
            let qv = session
                .encode_audio_query(REAL_CLAP_ID, &qbytes)
                .await
                .unwrap();

            let mut scored: Vec<(f32, &str)> = clip_ids
                .iter()
                .zip(&clip_vecs)
                .map(|(id, cv)| (cosine(&qv, cv), id.as_str()))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let top1_family = family_of(scored[0].1);
            if top1_family == fam {
                top1_same += 1;
            }

            for (sim, id) in &scored {
                if family_of(id) == fam {
                    intra.push(*sim);
                } else {
                    inter.push(*sim);
                }
            }
        }

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let mean_intra = mean(&intra);
        let mean_inter = mean(&inter);
        let margin = mean_intra - mean_inter;

        assert!(
            top1_same >= MIN_TOP1_SAME_FAMILY,
            "only {top1_same}/5 queries rank a same-family clip top-1 (need ≥{MIN_TOP1_SAME_FAMILY}) \
             — real CLAP retrieval looks random, a front-end or tower bug"
        );
        assert!(
            margin > MIN_INTRA_INTER_MARGIN,
            "intra-family cosine ({mean_intra:.4}) − inter-family ({mean_inter:.4}) = {margin:.4} \
             ≤ {MIN_INTRA_INTER_MARGIN}; embeddings do not separate timbre families"
        );
    }
}
