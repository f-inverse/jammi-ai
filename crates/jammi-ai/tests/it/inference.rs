#[cfg(feature = "live-hub-tests")]
use crate::common;

use jammi_ai::inference::adapter::{
    BackendOutput, ClassificationAdapter, EmbeddingAdapter, OutputAdapter,
};

use arrow::array::{Array, FixedSizeListArray};
use arrow::datatypes::DataType;

// ─── Hermetic tests (no network, no live models) ───────────────────────────

// --- Output adapter schemas match spec ---

#[test]
fn embedding_adapter_schema_has_vector_field() {
    let adapter = EmbeddingAdapter::new(384);
    let fields = adapter.output_schema();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name(), "vector");
    match fields[0].data_type() {
        DataType::FixedSizeList(inner, 384) => {
            assert_eq!(inner.data_type(), &DataType::Float32);
        }
        other => panic!("Expected FixedSizeList(Float32, 384), got {other:?}"),
    }
}

#[test]
fn classification_adapter_schema_has_label_confidence_scores() {
    let adapter = ClassificationAdapter;
    let fields = adapter.output_schema();
    let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["label", "confidence", "all_scores_json"]);
}

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

#[test]
fn contract_adapters_handle_zero_rows() {
    let embedding = EmbeddingAdapter::new(384);
    let classification = ClassificationAdapter;

    let empty_output = BackendOutput {
        float_outputs: vec![vec![]],
        string_outputs: vec![vec![], vec![]],
        row_status: vec![],
        row_errors: vec![],
        shapes: vec![(0, 384)],
    };

    let emb_cols = embedding.adapt(&empty_output, 0).unwrap();
    assert_eq!(
        emb_cols[0].len(),
        0,
        "Embedding adapter should handle 0 rows"
    );

    let cls_cols = classification.adapt(&empty_output, 0).unwrap();
    assert_eq!(
        cls_cols[0].len(),
        0,
        "Classification adapter should handle 0 rows"
    );
}

#[test]
fn contract_adapters_null_failed_rows() {
    let adapter = EmbeddingAdapter::new(384);

    let output = BackendOutput {
        float_outputs: vec![vec![0.0_f32; 768]], // 2 * 384
        string_outputs: vec![],
        row_status: vec![true, false],
        row_errors: vec![String::new(), "tokenization failed".into()],
        shapes: vec![(2, 384)],
    };

    let columns = adapter.adapt(&output, 2).unwrap();
    let vector_col = columns[0]
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();

    assert!(!vector_col.is_null(0), "Success row should have a vector");
    assert!(vector_col.is_null(1), "Failed row should have null vector");
}

// ─── Live tests (behind feature gate, require HF Hub downloads) ────────────

#[cfg(feature = "live-hub-tests")]
mod live {
    use super::*;
    use arrow::array::StringArray;
    use jammi_ai::inference::observer::InferenceObserver;
    use jammi_ai::model::{ModelSource, ModelTask};
    use jammi_ai::session::InferenceSession;
    use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    async fn setup_session() -> InferenceSession {
        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        InferenceSession::new(config).await.unwrap()
    }

    async fn setup_with_patents() -> InferenceSession {
        let session = setup_session().await;
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
        session
    }

    #[tokio::test]
    async fn inference_exec_embedding_produces_fixed_size_list_384() {
        let session = setup_with_patents().await;

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

        assert!(!results.is_empty(), "Should produce at least one batch");
        let batch = &results[0];

        // Check that "vector" column exists and is FixedSizeList(384)
        let vector_col = batch
            .column_by_name("vector")
            .expect("vector column should exist");
        match vector_col.data_type() {
            DataType::FixedSizeList(inner, 384) => {
                assert_eq!(inner.data_type(), &DataType::Float32);
            }
            other => panic!("Expected FixedSizeList(Float32, 384), got {other:?}"),
        }

        // Every OK row should have a 384-dim vector
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

    #[tokio::test]
    async fn model_registered_in_catalog_after_inference() {
        let session = setup_with_patents().await;

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

        // Verify the model was registered in the catalog
        let models = session.catalog().list_models().unwrap();
        assert!(
            models
                .iter()
                .any(|m| m.model_id.contains("all-MiniLM-L6-v2")),
            "Model should be registered in catalog after inference"
        );
    }

    #[tokio::test]
    async fn model_dimensions_retained_after_loading() {
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

        // MiniLM-L6-v2: hidden=384, heads=12, intermediate=1536
        let batch_mem = guard.model.estimate_batch_memory(32, 128);
        assert!(batch_mem > 0, "Batch memory estimate should be positive");
        // attention: 32*12*128*128*4 = 25,165,824
        // ffn: 32*128*1536*4 = 25,165,824
        assert!(
            batch_mem > 20_000_000 && batch_mem < 30_000_000,
            "Expected ~25MB for batch(32,128), got {batch_mem}"
        );
    }

    #[tokio::test]
    async fn null_text_row_produces_error_status() {
        let session = setup_session().await;
        session
            .add_source(
                "patents_nulls",
                SourceType::Local,
                SourceConnection {
                    url: Some(common::fixture_url("patents_with_nulls.parquet")),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = session
            .infer(
                "patents_nulls",
                &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                ModelTask::Embedding,
                &["abstract".to_string()],
                "id",
            )
            .await
            .unwrap();

        let mut ok_count = 0;
        let mut error_count = 0;

        for batch in &results {
            let status = batch
                .column_by_name("_status")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..status.len() {
                match status.value(i) {
                    "ok" => ok_count += 1,
                    "error" => error_count += 1,
                    other => panic!("Unexpected _status value: {other}"),
                }
            }
        }

        // patents_with_nulls.parquet has rows with null abstract
        assert!(
            error_count > 0,
            "Rows with null abstract should have _status='error'"
        );
        assert!(
            ok_count > 0,
            "Rows with valid abstract should have _status='ok'"
        );
    }

    #[tokio::test]
    async fn error_rows_have_null_vector_and_populated_error_message() {
        let session = setup_session().await;
        session
            .add_source(
                "patents_nulls",
                SourceType::Local,
                SourceConnection {
                    url: Some(common::fixture_url("patents_with_nulls.parquet")),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = session
            .infer(
                "patents_nulls",
                &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                ModelTask::Embedding,
                &["abstract".to_string()],
                "id",
            )
            .await
            .unwrap();

        for batch in &results {
            let status = batch
                .column_by_name("_status")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let error_col = batch
                .column_by_name("_error")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let vector_col = batch
                .column_by_name("vector")
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            for i in 0..status.len() {
                if status.value(i) == "error" {
                    assert!(
                        vector_col.is_null(i),
                        "Error row {i} should have null vector"
                    );
                    assert!(
                        !error_col.is_null(i),
                        "Error row {i} should have an error message"
                    );
                    assert!(
                        !error_col.value(i).is_empty(),
                        "Error message should not be empty"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn observer_receives_batch_notifications() {
        struct CountingObserver {
            batch_count: AtomicUsize,
        }

        impl InferenceObserver for CountingObserver {
            fn on_batch(
                &self,
                _batch: &arrow::record_batch::RecordBatch,
                _model_id: &str,
                _latency: Duration,
            ) {
                self.batch_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dir = tempdir().unwrap();
        let config = common::test_config(dir.path());
        let observer = Arc::new(CountingObserver {
            batch_count: AtomicUsize::new(0),
        });
        let session = InferenceSession::with_observer(
            config,
            Some(observer.clone() as Arc<dyn InferenceObserver>),
        )
        .await
        .unwrap();

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
        assert!(
            observer.batch_count.load(Ordering::Relaxed) > 0,
            "Observer should have been called at least once"
        );
    }
}
