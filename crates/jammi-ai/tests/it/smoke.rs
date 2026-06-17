use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray, StringArray};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
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
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate embeddings
    let tiny_bert = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();
    let record = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(common::url_to_path(&record.parquet_path).exists());

    let base = common::url_to_path(record.index_path.as_ref().unwrap());
    assert!(base.with_extension("usearch").exists());
    assert!(base.with_extension("rowmap").exists());
    assert!(base.with_extension("manifest.json").exists());

    // Vector search
    let results = session
        .search("patents", vec![0.5_f32; 32], 5, None)
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

    // Hydrated source columns present
    assert!(batch.schema().field_with_name("abstract").is_ok());
    assert!(batch.schema().field_with_name("title").is_ok());

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

    // The list-of-strings provenance columns themselves are non-null
    // (their *items* are nullable, but the list slot per row always
    // exists). Verifies the merger respects nullability of the suffix
    // fields it adds.
    let schema = batch.schema();
    let retrieved_by_field = schema.field_with_name("retrieved_by").unwrap();
    assert!(
        !retrieved_by_field.is_nullable(),
        "retrieved_by list column is non-null per the merger output contract"
    );

    // The merged-result `similarity` column comes from the vector
    // channel's catalog declaration, not from a hardcoded path.
    let declared = session
        .catalog()
        .channels()
        .get(&jammi_db::ChannelId::new("vector").unwrap())
        .await
        .unwrap()
        .expect("vector channel must be seeded by migration 006");
    assert_eq!(declared.columns[0].name, "similarity");
    assert!(
        batch
            .schema()
            .column_with_name(&declared.columns[0].name)
            .is_some(),
        "smoke: column declared by vector channel is present in result"
    );
}
