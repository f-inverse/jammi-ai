use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_engine::catalog::result_repo::CreateResultTableParams;
use jammi_engine::catalog::Catalog;
use jammi_engine::store::reader::{count_parquet_rows, is_valid_parquet};
use jammi_engine::store::writer::ParquetResultWriter;
use jammi_engine::store::ResultStore;
use tempfile::tempdir;

// ─── ParquetResultWriter roundtrip ───────────────────────────────────────────

#[test]
fn parquet_write_read_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let mut writer = ParquetResultWriter::new(&path, Arc::clone(&schema)).unwrap();

    // Multiple batches accumulate correctly
    for i in 0..3 {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![format!("row_{i}")])) as ArrayRef,
                Arc::new(Float32Array::from(vec![i as f32])) as ArrayRef,
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();
    }
    let row_count = writer.close().unwrap();

    assert_eq!(row_count, 3);
    assert!(is_valid_parquet(path.to_str().unwrap()));
    assert_eq!(count_parquet_rows(path.to_str().unwrap()).unwrap(), 3);
}

#[test]
fn parquet_writer_empty_close_produces_valid_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));

    let writer = ParquetResultWriter::new(&path, schema).unwrap();
    assert_eq!(writer.close().unwrap(), 0);
    assert!(is_valid_parquet(path.to_str().unwrap()));
}

#[test]
fn is_valid_parquet_rejects_garbage_and_missing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("garbage.parquet");
    std::fs::write(&path, b"not a parquet file").unwrap();

    assert!(!is_valid_parquet(path.to_str().unwrap()));
    assert!(!is_valid_parquet("/nonexistent/path.parquet"));
}

// ─── Catalog result_tables lifecycle ─────────────────────────────────────────

#[test]
fn result_table_crud_lifecycle() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // Create with all fields
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "t1",
            source_id: "patents",
            model_id: "sentence-transformers/all-MiniLM-L6-v2",
            task: "embedding",
            parquet_path: "/tmp/test.parquet",
            index_path: Some("/tmp/test"),
            dimensions: Some(384),
            key_column: Some("id"),
            text_columns: Some("abstract"),
        })
        .unwrap();

    let record = catalog.get_result_table("t1").unwrap().unwrap();
    assert_eq!(record.status, "building");
    assert_eq!(record.dimensions, Some(384));
    assert_eq!(record.row_count, 0);

    // Transition to ready
    catalog
        .update_result_table_status("t1", "ready", 42)
        .unwrap();
    let record = catalog.get_result_table("t1").unwrap().unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 42);
    assert!(record.completed_at.is_some());

    // List by status — create more tables to filter
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "t2",
            source_id: "patents",
            model_id: "m",
            task: "classification",
            parquet_path: "/tmp/t2.parquet",
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .unwrap();

    let building = catalog.list_result_tables_by_status("building").unwrap();
    assert_eq!(building.len(), 1);
    assert_eq!(building[0].table_name, "t2");

    let ready = catalog.list_result_tables_by_status("ready").unwrap();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].table_name, "t1");
}

#[test]
fn find_result_tables_filters_by_source_and_task() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    for (name, source, task) in [
        ("t1", "patents", "embedding"),
        ("t2", "patents", "classification"),
        ("t3", "scores", "embedding"),
    ] {
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: name,
                source_id: source,
                model_id: "model",
                task,
                parquet_path: &format!("/tmp/{name}.parquet"),
                index_path: None,
                dimensions: None,
                key_column: None,
                text_columns: None,
            })
            .unwrap();
    }

    assert_eq!(
        catalog
            .find_result_tables("patents", None, None)
            .unwrap()
            .len(),
        2
    );
    let emb = catalog
        .find_result_tables("patents", Some("embedding"), None)
        .unwrap();
    assert_eq!(emb.len(), 1);
    assert_eq!(emb[0].table_name, "t1");
}

#[test]
fn resolve_embedding_table_latest_explicit_and_missing() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // No tables → error
    assert!(catalog.resolve_embedding_table("patents", None).is_err());

    // Create two ready tables — should resolve to latest
    for name in ["old", "new"] {
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: name,
                source_id: "patents",
                model_id: "model",
                task: "embedding",
                parquet_path: &format!("/tmp/{name}.parquet"),
                index_path: None,
                dimensions: Some(384),
                key_column: None,
                text_columns: None,
            })
            .unwrap();
        catalog
            .update_result_table_status(name, "ready", 10)
            .unwrap();
    }

    let resolved = catalog.resolve_embedding_table("patents", None).unwrap();
    assert_eq!(resolved.table_name, "new", "Should resolve to latest");

    // Explicit name bypasses latest logic
    let explicit = catalog
        .resolve_embedding_table("patents", Some("old"))
        .unwrap();
    assert_eq!(explicit.table_name, "old");
}

// ─── ResultStore table naming ────────────────────────────────────────────────

#[tokio::test]
async fn result_store_create_table_generates_correct_paths() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let store = ResultStore::new(dir.path(), catalog).unwrap();

    let info = store
        .create_table(
            "patents",
            "embedding",
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
            None,
            None,
        )
        .unwrap();

    assert!(info
        .table_name
        .starts_with("patents__embedding__sentence-transformers_all-MiniLM-L6-v2__"));
    assert!(info.parquet_path.parent().unwrap().exists());
    assert!(
        info.index_path.is_some(),
        "Embedding tables should have an index path"
    );
}

// ─── Crash recovery (3 branches) ────────────────────────────────────────────

#[tokio::test]
async fn recovery_marks_missing_parquet_as_failed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "orphan",
            source_id: "src",
            model_id: "model",
            task: "embedding",
            parquet_path: dir.path().join("nonexistent.parquet").to_str().unwrap(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    assert_eq!(
        catalog.get_result_table("orphan").unwrap().unwrap().status,
        "failed"
    );
}

#[tokio::test]
async fn recovery_deletes_invalid_parquet_and_marks_failed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let db_dir = dir.path().join("jammi_db");
    std::fs::create_dir_all(&db_dir).unwrap();

    let bad_path = db_dir.join("corrupt.parquet");
    std::fs::write(&bad_path, b"not valid parquet data").unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "corrupt",
            source_id: "src",
            model_id: "model",
            task: "embedding",
            parquet_path: bad_path.to_str().unwrap(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    assert_eq!(
        catalog.get_result_table("corrupt").unwrap().unwrap().status,
        "failed"
    );
    assert!(!bad_path.exists(), "Invalid Parquet should be deleted");
}

#[tokio::test]
async fn recovery_promotes_valid_parquet_to_ready() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let db_dir = dir.path().join("jammi_db");
    std::fs::create_dir_all(&db_dir).unwrap();

    let parquet_path = db_dir.join("valid.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_row_id",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["r1", "r2", "r3"])) as ArrayRef],
    )
    .unwrap();
    let mut writer = ParquetResultWriter::new(&parquet_path, schema).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "stuck",
            source_id: "src",
            model_id: "model",
            task: "classification",
            parquet_path: parquet_path.to_str().unwrap(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    let record = catalog.get_result_table("stuck").unwrap().unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 3);
}
