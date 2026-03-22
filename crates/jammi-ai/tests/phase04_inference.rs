mod common;

#[allow(unused_imports)]
use jammi_ai::{
    inference::{
        adapter::{BackendOutput, ClassificationAdapter, EmbeddingAdapter, OutputAdapter},
        observer::InferenceObserver,
        schema::{build_output_schema, common_prefix_fields},
    },
    model::{
        backend::DeviceConfig, cache::ModelCache, resolver::ModelResolver, BackendType, ModelTask,
    },
    operator::inference_exec::InferenceExec,
};
#[allow(unused_imports)]
use jammi_engine::{
    config::JammiConfig,
    session::JammiSession,
    source::{FileFormat, SourceConnection, SourceType},
};

use arrow::array::{Array, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field};
use std::sync::Arc;
#[allow(unused_imports)]
use tempfile::tempdir;

// --- Embedding output schema ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_exec_embedding_produces_fixed_size_list_384() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

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
            "sentence-transformers/all-MiniLM-L6-v2",
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should produce at least one batch");
    let batch = &results[0];
    let vector_col = batch
        .column_by_name("vector")
        .expect("vector column should exist");
    match vector_col.data_type() {
        DataType::FixedSizeList(inner, 384) => {
            assert_eq!(inner.data_type(), &DataType::Float32);
        }
        other => panic!("Expected FixedSizeList(Float32, 384), got {other:?}"),
    }
}

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

// --- Common prefix columns ---

#[test]
fn common_prefix_fields_present() {
    let fields = common_prefix_fields();
    let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
    assert!(names.contains(&"_row_id"));
    assert!(names.contains(&"_source"));
    assert!(names.contains(&"_model"));
    assert!(names.contains(&"_status"));
    assert!(names.contains(&"_error"));
    assert!(names.contains(&"_latency_ms"));
}

// --- Output schema construction ---

#[test]
fn build_output_schema_embedding_has_prefix_plus_vector() {
    let input_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("abstract", DataType::Utf8, false),
    ]));

    let schema = build_output_schema(&ModelTask::Embedding, &input_schema, "id").unwrap();

    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"_row_id"));
    assert!(field_names.contains(&"_status"));
    assert!(field_names.contains(&"vector"));
}

#[test]
fn build_output_schema_classification_has_prefix_plus_task_cols() {
    let input_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("abstract", DataType::Utf8, false),
    ]));

    let schema = build_output_schema(&ModelTask::Classification, &input_schema, "id").unwrap();

    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"_row_id"));
    assert!(field_names.contains(&"_status"));
    assert!(field_names.contains(&"label"));
    assert!(field_names.contains(&"confidence"));
    assert!(field_names.contains(&"all_scores_json"));
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

// --- Live tests (behind feature gate) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_exec_classification_produces_label_and_confidence() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn null_text_row_produces_error_status() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn error_rows_have_null_vector_and_populated_error_message() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_output_contains_all_prefix_columns() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn process_large_input_without_oom() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn observer_receives_batch_notifications() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn no_observer_runs_without_error() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_with_nonexistent_model_returns_error() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_with_nonexistent_source_returns_error() {
    todo!("Requires live model loading")
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn inference_with_nonexistent_column_returns_error() {
    todo!("Requires live model loading")
}
