#[cfg(feature = "live-hub-tests")]
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

// ─── Live tests (behind feature gate, require HF Hub downloads) ────────────

#[cfg(feature = "live-hub-tests")]
mod live {
    use super::*;
    use arrow::array::{Float32Array, StringArray};
    use jammi_ai::model::{ModelSource, ModelTask};
    use jammi_ai::session::InferenceSession;
    use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
    use std::sync::Arc;
    use tempfile::tempdir;

    async fn setup_with_patents() -> (InferenceSession, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let session = InferenceSession::new(config).await.unwrap();
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
                ModelTask::Embedding,
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
        let catalog = Arc::new(jammi_engine::catalog::Catalog::open(dir.path()).unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
        let device_config = DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
        };
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config, scheduler);

        let guard = cache
            .get_or_load(
                &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                ModelTask::Embedding,
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
    async fn live_patentclip_produces_512_dim_image_embeddings() {
        use jammi_ai::model::backend::DeviceConfig;
        use jammi_ai::model::cache::ModelCache;
        use jammi_ai::model::resolver::ModelResolver;

        let dir = tempdir().unwrap();
        let catalog = Arc::new(jammi_engine::catalog::Catalog::open(dir.path()).unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
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
}
