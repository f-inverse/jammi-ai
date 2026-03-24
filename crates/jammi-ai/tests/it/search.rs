use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray, StringArray};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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
        .generate_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    (session, dir)
}

// ─── Vector search: results, schema, provenance, ordering ────────────────────

#[tokio::test]
async fn search_returns_ranked_results_with_provenance() {
    let (session, _dir) = session_with_embeddings().await;

    // Use a dummy query vector matching tiny_bert's 32 dimensions
    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty(), "Search should return results");
    let batch = &results[0];

    // Correct number of results
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0 && total_rows <= 5);

    // Schema has required columns
    assert!(batch.schema().field_with_name("_row_id").is_ok());
    assert!(batch.schema().field_with_name("_source_id").is_ok());
    assert!(batch.schema().field_with_name("similarity").is_ok());
    assert!(batch.schema().field_with_name("retrieved_by").is_ok());
    assert!(batch.schema().field_with_name("annotated_by").is_ok());

    // Similarity is descending (AnnSearchExec returns by distance ascending → similarity descending)
    let sim = batch
        .column_by_name("similarity")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    for i in 1..sim.len() {
        assert!(
            sim.value(i - 1) >= sim.value(i),
            "Similarity should be descending: {} < {} at row {i}",
            sim.value(i - 1),
            sim.value(i)
        );
    }

    // retrieved_by = ["vector"] for all rows
    let retrieved_by = batch
        .column_by_name("retrieved_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..retrieved_by.len() {
        let values = retrieved_by.value(i);
        let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
        let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
        assert!(
            channels.contains(&"vector"),
            "retrieved_by should contain 'vector', got {channels:?}"
        );
    }

    // annotated_by should be empty (no annotation step)
    let annotated_by = batch
        .column_by_name("annotated_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..annotated_by.len() {
        let values = annotated_by.value(i);
        assert_eq!(
            values.len(),
            0,
            "annotated_by should be empty without annotation"
        );
    }
}

// ─── Sort + limit compose correctly ──────────────────────────────────────────

#[tokio::test]
async fn search_sort_and_limit_compose() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 10)
        .await
        .unwrap()
        .sort("similarity", true)
        .unwrap()
        .limit(3)
        .run()
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total <= 3, "Limit(3) should cap results, got {total}");

    // Verify descending sort maintained
    for batch in &results {
        let sim = batch
            .column_by_name("similarity")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 1..sim.len() {
            assert!(sim.value(i - 1) >= sim.value(i));
        }
    }
}

// ─── Search fails without embedding table ────────────────────────────────────

#[tokio::test]
async fn search_fails_without_embedding_table() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

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

    let result = session.search("patents", vec![0.0f32; 32], 5).await;
    assert!(
        result.is_err(),
        "Search should fail when no embedding table exists"
    );
}

// ─── encode_query returns a vector ───────────────────────────────────────────

#[tokio::test]
async fn encode_query_returns_vector_of_correct_dimension() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

    let vector = session
        .encode_query(&tiny_bert_model(), "quantum computing")
        .await
        .unwrap();

    // tiny_bert has hidden_size=32
    assert_eq!(vector.len(), 32, "Vector should be 32-dim for tiny_bert");
    assert!(
        vector.iter().any(|&v| v != 0.0),
        "Vector should not be all zeros"
    );
}
