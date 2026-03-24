use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray, StringArray};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// CP3 smoke test: full pipeline from source registration through vector search.
/// This is the cross-phase regression gate introduced at CP3.
#[tokio::test]
async fn smoke_cp3_full_pipeline() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    // Register source
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

    // Generate embeddings
    let tiny_bert = "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap();
    let record = session
        .generate_embeddings("patents", &tiny_bert, &["abstract".to_string()], "id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(std::path::Path::new(&record.parquet_path).exists());

    let base = std::path::Path::new(record.index_path.as_ref().unwrap());
    assert!(base.with_extension("usearch").exists());
    assert!(base.with_extension("rowmap").exists());
    assert!(base.with_extension("manifest.json").exists());

    // Vector search
    let results = session
        .search("patents", vec![0.5_f32; 32], 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // Evidence columns present
    assert!(batch.schema().field_with_name("_row_id").is_ok());
    assert!(batch.schema().field_with_name("_source_id").is_ok());
    assert!(batch.schema().field_with_name("similarity").is_ok());
    assert!(batch.schema().field_with_name("retrieved_by").is_ok());
    assert!(batch.schema().field_with_name("annotated_by").is_ok());

    // Provenance correct
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

    // Similarity descending
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
