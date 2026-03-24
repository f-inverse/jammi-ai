use crate::common;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
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

// ─── Core generate_embeddings: one run, many assertions ──────────────────────
//
// Consolidates: writes Parquet, status=ready, dimensions match, sidecar index
// built, DataFusion queryable, metadata tracked, Parquet readable standalone.

#[tokio::test]
async fn generate_embeddings_produces_complete_result() {
    let (session, _dir) = session_with_patents().await;

    let record = session
        .generate_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    // Parquet file exists with rows
    assert!(record.row_count > 0);
    assert!(
        std::path::Path::new(&record.parquet_path).exists(),
        "Parquet file should exist at {}",
        record.parquet_path
    );

    // Status = ready with completed_at
    assert_eq!(record.status, "ready");
    assert!(record.completed_at.is_some());
    let from_catalog = session
        .catalog()
        .get_result_table(&record.table_name)
        .unwrap()
        .unwrap();
    assert_eq!(from_catalog.status, "ready");

    // Dimensions match tiny_bert (hidden_size=32)
    assert_eq!(record.dimensions, Some(32));

    // Metadata tracked
    assert_eq!(record.source_id, "patents");
    assert_eq!(record.task, "embedding");
    assert!(record.key_column.as_deref() == Some("id"));
    assert!(record.text_columns.as_deref() == Some("abstract"));

    // Sidecar index files exist
    let base_path = record
        .index_path
        .as_ref()
        .expect("Embedding table should have index_path");
    let base = std::path::Path::new(base_path);
    assert!(
        base.with_extension("usearch").exists(),
        "USearch index missing"
    );
    assert!(base.with_extension("rowmap").exists(), "Rowmap missing");
    assert!(
        base.with_extension("manifest.json").exists(),
        "Manifest missing"
    );

    // Queryable via DataFusion
    let results = session
        .sql(&format!(
            "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 5",
            record.table_name
        ))
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].num_rows() <= 5);
    assert!(results[0].schema().field_with_name("_row_id").is_ok());
    assert!(results[0].schema().field_with_name("_source_id").is_ok());

    // Readable by external Parquet tools (no Jammi context)
    let file = std::fs::File::open(&record.parquet_path).unwrap();
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    let mut total_rows = 0;
    for batch in reader {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();
        assert!(batch.schema().field_with_name("_row_id").is_ok());
        assert!(batch.schema().field_with_name("vector").is_ok());
    }
    assert_eq!(total_rows, record.row_count);
}

// ─── Multiple result tables coexist + sidecar fallback ───────────────────────

#[tokio::test]
async fn multiple_tables_and_sidecar_fallback() {
    let (session, _dir) = session_with_patents().await;

    let r1 = session
        .generate_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    let r2 = session
        .generate_embeddings("patents", &tiny_bert_model(), &["title".to_string()], "id")
        .await
        .unwrap();

    // Two distinct tables
    assert_ne!(r1.table_name, r2.table_name);
    let tables = session
        .catalog()
        .find_result_tables("patents", Some("embedding"), None)
        .unwrap();
    assert!(tables.len() >= 2);

    // Delete sidecar for r1 — Parquet still queryable
    if let Some(ref base) = r1.index_path {
        let base = std::path::Path::new(base);
        std::fs::remove_file(base.with_extension("usearch")).ok();
        std::fs::remove_file(base.with_extension("rowmap")).ok();
        std::fs::remove_file(base.with_extension("manifest.json")).ok();
    }
    let results = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\" LIMIT 3",
            r1.table_name
        ))
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "Parquet should still be queryable after sidecar deletion"
    );
}

// ─── Failed rows filtered in embedding output ────────────────────────────────

#[tokio::test]
async fn failed_rows_skipped_in_embedding_output() {
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

    let record = session
        .generate_embeddings(
            "patents_nulls",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    // patents_with_nulls has 10 rows, 3 with null abstract
    assert!(
        record.row_count < 10,
        "Should have fewer rows than source due to null filtering, got {}",
        record.row_count
    );
    assert_eq!(record.status, "ready");
}

// ─── infer() persists results to Parquet ─────────────────────────────────────

#[tokio::test]
async fn infer_persists_results_to_parquet() {
    let (session, _dir) = session_with_patents().await;
    let model_source = ModelSource::local(common::fixture("tiny_bert"));

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

    assert!(!results.is_empty(), "Should return batches to caller");

    let tables = session
        .catalog()
        .find_result_tables("patents", Some("embedding"), None)
        .unwrap();
    assert!(!tables.is_empty(), "Should have created a result table");
    assert_eq!(tables[0].status, "ready");
}

// ─── Existing tables loaded on new session ───────────────────────────────────

#[tokio::test]
async fn existing_tables_loaded_on_new_session() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());

    // First session: generate embeddings
    {
        let session = InferenceSession::new(config.clone()).await.unwrap();
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
    }

    // Second session: result table should be queryable without re-generating
    {
        let session = InferenceSession::new(config).await.unwrap();
        let tables = session
            .catalog()
            .find_result_tables("patents", Some("embedding"), None)
            .unwrap();
        assert!(!tables.is_empty());

        let record = &tables[0];
        let results = session
            .sql(&format!(
                "SELECT count(*) as cnt FROM \"jammi.{}\"",
                record.table_name
            ))
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
}
