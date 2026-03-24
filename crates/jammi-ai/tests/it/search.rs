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

// ─── Vector search: results, hydrated columns, provenance, ordering ──────────

#[tokio::test]
async fn search_returns_hydrated_results_with_provenance() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0 && total_rows <= 5);

    // Evidence columns from ANN search
    assert!(batch.schema().field_with_name("_row_id").is_ok());
    assert!(batch.schema().field_with_name("_source_id").is_ok());
    assert!(batch.schema().field_with_name("similarity").is_ok());
    assert!(batch.schema().field_with_name("retrieved_by").is_ok());
    assert!(batch.schema().field_with_name("annotated_by").is_ok());

    // Hydrated columns from original source
    assert!(
        batch.schema().field_with_name("abstract").is_ok(),
        "Hydration should include original 'abstract' column"
    );
    assert!(
        batch.schema().field_with_name("title").is_ok(),
        "Hydration should include original 'title' column"
    );
    assert!(
        batch.schema().field_with_name("assignee_id").is_ok(),
        "Hydration should include original 'assignee_id' column"
    );

    // Similarity descending
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

    // retrieved_by = ["vector"], annotated_by = []
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
        assert!(channels.contains(&"vector"));
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

// ─── Join on real foreign key ────────────────────────────────────────────────

#[tokio::test]
async fn search_with_join_on_real_foreign_key() {
    let (session, _dir) = session_with_embeddings().await;

    session
        .add_source(
            "assignees",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .join("assignees", "assignee_id=id", None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // Joined columns from assignees
    assert!(
        batch.schema().field_with_name("company_name").is_ok(),
        "Join should add company_name from assignees"
    );
    assert!(
        batch.schema().field_with_name("country").is_ok(),
        "Join should add country from assignees"
    );

    // At least one row should have matched (patent assignee_id 101-110 matches assignees id 101-110)
    let company = batch
        .column_by_name("company_name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let has_match = (0..company.len()).any(|i| !company.is_null(i));
    assert!(
        has_match,
        "At least one joined row should have a company_name match"
    );
}

// ─── Annotate on real content column ─────────────────────────────────────────

#[tokio::test]
async fn search_with_annotate_on_real_column() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 3)
        .await
        .unwrap()
        .annotate(&tiny_bert_model(), "embedding", &["abstract".to_string()])
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // annotated_by should contain "inference"
    let annotated_by = batch
        .column_by_name("annotated_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..annotated_by.len() {
        let values = annotated_by.value(i);
        let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
        let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
        assert!(channels.contains(&"inference"));
    }

    // retrieved_by should contain "vector" and NOT "inference"
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
        assert!(channels.contains(&"vector"));
        assert!(!channels.contains(&"inference"));
    }
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

    assert_eq!(vector.len(), 32, "Vector should be 32-dim for tiny_bert");
    assert!(
        vector.iter().any(|&v| v != 0.0),
        "Vector should not be all zeros"
    );
}
