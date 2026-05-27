use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::{
    reader::{count_parquet_rows, is_valid_parquet},
    JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl,
};
use jammi_db::store::ResultStore;
use tempfile::tempdir;

// ─── ObjectParquetWriter roundtrip ───────────────────────────────────────────

#[tokio::test]
async fn parquet_write_read_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let url = StorageUrl::parse(path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());

    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();

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
        writer.write_batch(&batch).await.unwrap();
    }
    let row_count = writer.close().await.unwrap();

    assert_eq!(row_count, 3);
    assert!(is_valid_parquet(&handle).await.unwrap());
    assert_eq!(count_parquet_rows(&handle).await.unwrap(), 3);
}

// ─── Catalog result_tables lifecycle ─────────────────────────────────────────

#[tokio::test]
async fn result_table_crud_lifecycle() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "t1",
            source_id: "patents",
            model_id: "sentence-transformers/all-MiniLM-L6-v2",
            task: ModelTask::TextEmbedding,
            parquet_path: "file:///tmp/test.parquet",
            index_path: Some("file:///tmp/test.idx"),
            dimensions: Some(384),
            key_column: Some("id"),
            text_columns: Some("abstract"),
        })
        .await
        .unwrap();

    let record = catalog.get_result_table("t1").await.unwrap().unwrap();
    assert_eq!(record.status, "building");
    assert_eq!(record.dimensions, Some(384));
    assert_eq!(record.row_count, 0);

    catalog
        .update_result_table_status("t1", ResultTableStatus::Ready, 42)
        .await
        .unwrap();
    let record = catalog.get_result_table("t1").await.unwrap().unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 42);
    assert!(record.completed_at.is_some());

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "t2",
            source_id: "patents",
            model_id: "m",
            task: ModelTask::Classification,
            parquet_path: "file:///tmp/t2.parquet",
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();

    let building = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(building.len(), 1);
    assert_eq!(building[0].table_name, "t2");

    let ready = catalog
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .await
        .unwrap();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].table_name, "t1");
}

#[tokio::test]
async fn find_result_tables_filters_by_source_and_task() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    for (name, source, task) in [
        ("t1", "patents", ModelTask::TextEmbedding),
        ("t2", "patents", ModelTask::Classification),
        ("t3", "scores", ModelTask::TextEmbedding),
    ] {
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: name,
                source_id: source,
                model_id: "model",
                task,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                index_path: None,
                dimensions: None,
                key_column: None,
                text_columns: None,
            })
            .await
            .unwrap();
    }

    assert_eq!(
        catalog
            .find_result_tables("patents", None, None)
            .await
            .unwrap()
            .len(),
        2
    );
    let emb = catalog
        .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    assert_eq!(emb.len(), 1);
    assert_eq!(emb[0].table_name, "t1");
}

#[tokio::test]
async fn resolve_embedding_table_latest_explicit_and_missing() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    assert!(catalog
        .resolve_embedding_table("patents", None)
        .await
        .is_err());

    for name in ["old", "new"] {
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: name,
                source_id: "patents",
                model_id: "model",
                task: ModelTask::TextEmbedding,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                index_path: None,
                dimensions: Some(384),
                key_column: None,
                text_columns: None,
            })
            .await
            .unwrap();
        catalog
            .update_result_table_status(name, ResultTableStatus::Ready, 10)
            .await
            .unwrap();
    }

    let resolved = catalog
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    assert_eq!(resolved.table_name, "new", "Should resolve to latest");

    let explicit = catalog
        .resolve_embedding_table("patents", Some("old"))
        .await
        .unwrap();
    assert_eq!(explicit.table_name, "old");
}

/// `resolve_embedding_table` must consider every `ModelTask` variant for
/// which `is_embedding()` returns `true`, and must ignore non-embedding
/// tasks. Drives the seed loop off `ModelTask::ALL` so that adding a new
/// embedding variant in the future automatically extends coverage — the
/// previous version enumerated `TextEmbedding` and `ImageEmbedding` by
/// hand and would have masked the regression that the dynamic IN-clause
/// in `resolve_embedding_table` was introduced to fix.
#[tokio::test]
async fn resolve_embedding_table_accepts_every_embedding_variant() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    // Seed one Ready table per variant. Created-at ordering puts the
    // last-inserted embedding variant on top of the resolver's
    // `ORDER BY created_at DESC` tiebreaker.
    let mut expected_winner: Option<String> = None;
    for task in ModelTask::ALL {
        let name = format!("row_{}", task.as_db_str());
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: &name,
                source_id: "media",
                model_id: "model",
                task: *task,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                index_path: None,
                dimensions: Some(8),
                key_column: None,
                text_columns: None,
            })
            .await
            .unwrap();
        catalog
            .update_result_table_status(&name, ResultTableStatus::Ready, 4)
            .await
            .unwrap();
        if task.is_embedding() {
            expected_winner = Some(name);
        }
    }

    let resolved = catalog
        .resolve_embedding_table("media", None)
        .await
        .unwrap();
    assert!(
        resolved.task.is_embedding(),
        "resolver returned non-embedding task {:?}",
        resolved.task
    );
    assert_eq!(
        resolved.table_name,
        expected_winner.expect("ModelTask::ALL has at least one embedding variant"),
    );
}

/// Crash recovery rebuilds the ANN sidecar index only for embedding-task
/// rows. A classification table sitting in `Building` must promote to
/// `Ready` without the recovery path trying to read a non-existent
/// `vector` column. Regression guard for the prior literal-string
/// `task == "embedding" || task == "text_embedding" || task ==
/// "image_embedding"` branch.
#[tokio::test]
async fn recovery_skips_index_rebuild_for_non_embedding_task() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let db_dir = dir.path().join("jammi_db");
    std::fs::create_dir_all(&db_dir).unwrap();

    // Write a valid parquet that intentionally lacks the `vector` column
    // an embedding-table sidecar rebuild would expect — proves the
    // classification branch never reaches the rebuild path.
    let parquet_path = db_dir.join("classify.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["A", "B"])),
        ],
    )
    .unwrap();

    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "classify_recover",
            source_id: "src",
            model_id: "model",
            task: ModelTask::Classification,
            parquet_path: url.as_str(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    let record = catalog
        .get_result_table("classify_recover")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 2);
    assert!(
        !record.task.is_embedding(),
        "test fixture should be a non-embedding task"
    );
}

// ─── ResultStore table naming ────────────────────────────────────────────────

#[tokio::test]
async fn result_store_create_table_generates_correct_paths() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), catalog).unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::TextEmbedding,
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert!(info
        .table_name
        .starts_with("patents__text_embedding__sentence-transformers_all-MiniLM-L6-v2__"));
    // parquet_url is a StorageUrl pointing at a file://… path under the
    // jammi_db root we just created.
    assert!(info.parquet_url.as_str().contains("jammi_db"));
    assert!(
        info.index_url.is_some(),
        "Embedding tables should have an index URL"
    );
}

// ─── Crash recovery (3 branches) ────────────────────────────────────────────

#[tokio::test]
async fn recovery_marks_missing_parquet_as_failed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    let missing_url =
        StorageUrl::parse(dir.path().join("nonexistent.parquet").to_str().unwrap()).unwrap();
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "orphan",
            source_id: "src",
            model_id: "model",
            task: ModelTask::TextEmbedding,
            parquet_path: missing_url.as_str(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    assert_eq!(
        catalog
            .get_result_table("orphan")
            .await
            .unwrap()
            .unwrap()
            .status,
        "failed"
    );
}

#[tokio::test]
async fn recovery_deletes_invalid_parquet_and_marks_failed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let db_dir = dir.path().join("jammi_db");
    std::fs::create_dir_all(&db_dir).unwrap();

    let bad_path = db_dir.join("corrupt.parquet");
    std::fs::write(&bad_path, b"not valid parquet data").unwrap();
    let bad_url = StorageUrl::parse(bad_path.to_str().unwrap()).unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "corrupt",
            source_id: "src",
            model_id: "model",
            task: ModelTask::TextEmbedding,
            parquet_path: bad_url.as_str(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    assert_eq!(
        catalog
            .get_result_table("corrupt")
            .await
            .unwrap()
            .unwrap()
            .status,
        "failed"
    );
    assert!(!bad_path.exists(), "Invalid Parquet should be deleted");
}

#[tokio::test]
async fn recovery_promotes_valid_parquet_to_ready() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
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

    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "stuck",
            source_id: "src",
            model_id: "model",
            task: ModelTask::Classification,
            parquet_path: url.as_str(),
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();

    let store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    store.recover().await.unwrap();

    let record = catalog.get_result_table("stuck").await.unwrap().unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 3);
}
