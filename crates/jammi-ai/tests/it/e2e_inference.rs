// End-to-end hermetic tests for the inference pipeline.
//
// These tests exercise the full path: InferenceSession → add_source → infer()
// → RecordBatch with vectors. They use a tiny BERT fixture (32-dim, 1 layer)
// checked into tests/fixtures/tiny_bert/ — no network access required.

use crate::common;

use arrow::array::{Array, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::DataType;
use jammi_ai::inference::observer::InferenceObserver;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_engine::index::cosine_distance;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn tiny_bert_source() -> ModelSource {
    ModelSource::local(common::fixture("tiny_bert"))
}

fn tiny_modernbert_source() -> ModelSource {
    ModelSource::local(common::fixture("tiny_modernbert"))
}

async fn session_with_patents() -> (InferenceSession, TempDir) {
    let dir = TempDir::new().unwrap();
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

// ─── Full pipeline: source → model → embedding vectors ─────────────────────

#[tokio::test]
async fn e2e_embedding_produces_vectors_with_correct_schema() {
    let (session, _dir) = session_with_patents().await;
    let model_source = tiny_bert_source();

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should produce at least one batch");
    let batch = &results[0];

    // Verify all prefix columns exist
    for col_name in &[
        "_row_id",
        "_source",
        "_model",
        "_status",
        "_error",
        "_latency_ms",
    ] {
        assert!(
            batch.schema().field_with_name(col_name).is_ok(),
            "Missing prefix column: {col_name}"
        );
    }

    // Verify vector column is FixedSizeList(Float32, 32) — tiny BERT has hidden_size=32
    let vector_col = batch
        .column_by_name("vector")
        .expect("vector column should exist");
    match vector_col.data_type() {
        DataType::FixedSizeList(inner, 32) => {
            assert_eq!(inner.data_type(), &DataType::Float32);
        }
        other => panic!("Expected FixedSizeList(Float32, 32), got {other:?}"),
    }
}

#[tokio::test]
async fn e2e_every_row_has_valid_status() {
    let (session, _dir) = session_with_patents().await;
    let model_source = tiny_bert_source();

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    let mut total_rows = 0;
    for batch in &results {
        let status = batch
            .column_by_name("_status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..status.len() {
            let s = status.value(i);
            assert!(
                s == "ok" || s == "error",
                "Row {i} has invalid _status: '{s}'"
            );
        }
        total_rows += batch.num_rows();
    }

    // patents.parquet has 20 rows
    assert_eq!(total_rows, 20, "All 20 rows should be processed");
}

#[tokio::test]
async fn e2e_provenance_columns_have_correct_values() {
    let (session, _dir) = session_with_patents().await;
    let model_source = tiny_bert_source();

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    let batch = &results[0];

    // _source should be the source_id
    let source_col = batch
        .column_by_name("_source")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(source_col.value(0), "patents");

    // _model should contain the model path
    let model_col = batch
        .column_by_name("_model")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(
        model_col.value(0).contains("tiny_bert"),
        "_model should reference the model used"
    );

    // _latency_ms should be positive
    let latency_col = batch
        .column_by_name("_latency_ms")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert!(latency_col.value(0) > 0.0, "Latency should be positive");
}

// ─── Null / error handling ──────────────────────────────────────────────────

#[tokio::test]
async fn e2e_null_text_rows_produce_error_status() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

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

    let model_source = tiny_bert_source();
    let results = session
        .infer(
            "patents_nulls",
            &model_source,
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
                other => panic!("Unexpected _status: '{other}'"),
            }
        }
    }

    // patents_with_nulls.parquet has 10 rows, 3 with null abstract
    assert!(
        error_count >= 3,
        "At least 3 null-abstract rows should have error status, got {error_count}"
    );
    assert!(
        ok_count >= 1,
        "At least some valid rows should have ok status, got {ok_count} (errors: {error_count})"
    );
}

#[tokio::test]
async fn e2e_error_rows_have_null_vector_and_error_message() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

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

    let model_source = tiny_bert_source();
    let results = session
        .infer(
            "patents_nulls",
            &model_source,
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

// ─── Observer integration ───────────────────────────────────────────────────

#[tokio::test]
async fn e2e_observer_receives_batch_notifications() {
    struct CountingObserver(AtomicUsize);
    impl InferenceObserver for CountingObserver {
        fn on_batch(&self, _: &arrow::record_batch::RecordBatch, _: &str, _: Duration) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let observer = Arc::new(CountingObserver(AtomicUsize::new(0)));
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

    let model_source = tiny_bert_source();
    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(
        observer.0.load(Ordering::Relaxed) > 0,
        "Observer should have been called at least once"
    );
}

// ─── Model catalog registration ─────────────────────────────────────────────

#[tokio::test]
async fn e2e_model_registered_in_catalog_after_inference() {
    let (session, _dir) = session_with_patents().await;
    let model_source = tiny_bert_source();

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();
    assert!(!results.is_empty());

    let models = session.catalog().list_models().unwrap();
    assert!(
        models.iter().any(|m| m.model_id.contains("tiny_bert")),
        "Model should be registered in catalog after inference. Found: {:?}",
        models.iter().map(|m| &m.model_id).collect::<Vec<_>>()
    );
}

// ─── Embedding semantic correctness and reproducibility ─────────────────────

#[tokio::test]
async fn embedding_vectors_are_semantically_meaningful_and_reproducible() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

    let model = "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap();

    // Encode four queries: two physics, one biology, and a repeat of the first
    let vec_physics_1 = session
        .encode_query(&model, "quantum computing in superconducting systems")
        .await
        .unwrap();

    let vec_physics_2 = session
        .encode_query(&model, "topological quantum error correction")
        .await
        .unwrap();

    let vec_biology = session
        .encode_query(&model, "CRISPR gene editing for disease treatment")
        .await
        .unwrap();

    let vec_physics_1_repeat = session
        .encode_query(&model, "quantum computing in superconducting systems")
        .await
        .unwrap();

    // Reproducibility: identical input must produce identical output
    assert_eq!(
        vec_physics_1, vec_physics_1_repeat,
        "Encoding the same text twice must produce identical vectors"
    );

    // Dimension: tiny_bert has hidden_size=32
    assert_eq!(
        vec_physics_1.len(),
        32,
        "Physics vector 1 should have 32 dimensions"
    );
    assert_eq!(
        vec_physics_2.len(),
        32,
        "Physics vector 2 should have 32 dimensions"
    );
    assert_eq!(
        vec_biology.len(),
        32,
        "Biology vector should have 32 dimensions"
    );
    assert_eq!(
        vec_physics_1_repeat.len(),
        32,
        "Repeat vector should have 32 dimensions"
    );

    // Non-trivial: vectors should not be all zeros
    assert!(
        vec_physics_1.iter().any(|&v| v != 0.0),
        "Physics vector 1 should have at least one non-zero element"
    );
    assert!(
        vec_physics_2.iter().any(|&v| v != 0.0),
        "Physics vector 2 should have at least one non-zero element"
    );
    assert!(
        vec_biology.iter().any(|&v| v != 0.0),
        "Biology vector should have at least one non-zero element"
    );
    assert!(
        vec_physics_1_repeat.iter().any(|&v| v != 0.0),
        "Repeat vector should have at least one non-zero element"
    );

    // Semantic coherence: within-category distance should be small
    let dist_physics = cosine_distance(&vec_physics_1, &vec_physics_2);
    assert!(
        dist_physics < 0.5,
        "Within-category (physics-to-physics) cosine distance should be < 0.5, got {dist_physics}"
    );

    // Cross-category distance should be larger than within-category distance
    let dist_cross = cosine_distance(&vec_physics_1, &vec_biology);
    assert!(
        dist_cross > dist_physics,
        "Cross-category distance ({dist_cross}) should exceed within-category distance ({dist_physics})"
    );
}

// ─── ModernBERT backend ──────────────────────────────────��──────────────────

#[tokio::test]
async fn e2e_modernbert_embedding_produces_vectors_with_correct_schema() {
    let (session, _dir) = session_with_patents().await;
    let model_source = tiny_modernbert_source();

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should produce at least one batch");
    let batch = &results[0];

    // Verify vector column is FixedSizeList(Float32, 32) — tiny_modernbert has hidden_size=32
    let vector_col = batch
        .column_by_name("vector")
        .expect("vector column should exist");
    match vector_col.data_type() {
        DataType::FixedSizeList(inner, 32) => {
            assert_eq!(inner.data_type(), &DataType::Float32);
        }
        other => panic!("Expected FixedSizeList(Float32, 32), got {other:?}"),
    }
}

#[tokio::test]
async fn e2e_modernbert_embedding_vectors_are_nonzero_and_reproducible() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

    let model = "local:".to_string() + common::fixture("tiny_modernbert").to_str().unwrap();

    let vec_a = session
        .encode_query(&model, "quantum computing in superconducting systems")
        .await
        .unwrap();

    let vec_b = session
        .encode_query(&model, "quantum computing in superconducting systems")
        .await
        .unwrap();

    // Reproducibility: identical input must produce identical output
    assert_eq!(
        vec_a, vec_b,
        "Encoding the same text twice must produce identical vectors"
    );

    // Correct dimension
    assert_eq!(
        vec_a.len(),
        32,
        "ModernBERT vector should have 32 dimensions"
    );

    // Non-trivial: vectors should not be all zeros
    assert!(
        vec_a.iter().any(|&v| v != 0.0),
        "Vector should have at least one non-zero element"
    );
}
