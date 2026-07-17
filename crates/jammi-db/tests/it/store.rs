use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{
    reader::{count_parquet_rows, is_valid_parquet},
    JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl,
};
use jammi_db::store::ResultStore;
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;

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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: "file:///tmp/test.parquet",
            dimensions: Some(384),
            key_column: Some("id"),
            text_columns: Some("abstract"),
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: "file:///tmp/t2.parquet",
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
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
                kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
                derived_from: None,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                dimensions: None,
                key_column: None,
                text_columns: None,
                storage_precision: jammi_db::config::StoragePrecision::F32,
                oversample: 4,
                created_at: jammi_db::catalog::backend::now_sortable(),
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

    for (seq, name) in ["old", "new"].into_iter().enumerate() {
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: name,
                source_id: "patents",
                model_id: "model",
                task: ModelTask::TextEmbedding,
                kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
                derived_from: None,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                dimensions: Some(384),
                key_column: None,
                text_columns: None,
                storage_precision: jammi_db::config::StoragePrecision::F32,
                oversample: 4,
                // Explicit, strictly-increasing `created_at` — "new" must
                // resolve as newest regardless of wall-clock resolution.
                created_at: sortable_at(seq as u64 + 1),
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

    // Seed one Ready table per variant. Explicit, strictly-increasing
    // `created_at` values (independent of wall-clock resolution) put the
    // last-inserted embedding variant on top of the resolver's
    // `ORDER BY created_at DESC` tiebreaker.
    let mut expected_winner: Option<String> = None;
    for (seq, task) in ModelTask::ALL.iter().enumerate() {
        let name = format!("row_{}", task.as_db_str());
        catalog
            .create_result_table(CreateResultTableParams {
                table_name: &name,
                source_id: "media",
                model_id: "model",
                task: *task,
                kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
                derived_from: None,
                parquet_path: &format!("file:///tmp/{name}.parquet"),
                dimensions: Some(8),
                key_column: None,
                text_columns: None,
                storage_precision: jammi_db::config::StoragePrecision::F32,
                oversample: 4,
                created_at: sortable_at(seq as u64 + 1),
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

/// `resolve_embedding_table` must pick the temporally-newest ready `model`
/// table by `created_at`, not by `table_name` — on both SQLite and Postgres.
///
/// The table names are chosen so a `table_name`-only tiebreak would return
/// the WRONG row: a result table's name is
/// `{source}__{task}__{model}__{timestamp}_{suffix}`, and the model segment
/// sorts *before* the timestamp, so `zzz_model` (created first, older) sorts
/// after `aaa_model` (created second, newer) even though `aaa_model` is the
/// correct answer. This is the exact naive-fix regression a prior
/// pressure-test killed (`ORDER BY created_at DESC, table_name DESC` alone
/// is correct; `ORDER BY table_name DESC` alone is not) — it also
/// regression-guards the Postgres `rowid`-does-not-exist hard error, since
/// SQLite is the only backend with a `rowid` to (wrongly) fall back on.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn resolve_embedding_table_picks_newest_by_created_at_not_table_name(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = match make_test_session(backend, dir.path()).await {
        Some(s) => s,
        None => {
            eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
    let catalog = session.catalog();

    let suffix = unique_suffix();
    let source_id = format!("multi_model_src_{suffix}");
    let older_table = format!("{source_id}__text_embedding__zzz_model__{suffix}_1");
    let newer_table = format!("{source_id}__text_embedding__aaa_model__{suffix}_2");

    catalog
        .register_source(
            &source_id,
            SourceType::File,
            &SourceConnection {
                url: Some(format!("file:///tmp/{source_id}.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // First (older) table: model name sorts alphabetically LAST.
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: &older_table,
            source_id: &source_id,
            model_id: "zzz_model",
            task: ModelTask::TextEmbedding,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: &format!("file:///tmp/{older_table}.parquet"),
            dimensions: Some(8),
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();
    catalog
        .update_result_table_status(&older_table, ResultTableStatus::Ready, 4)
        .await
        .unwrap();

    // A real wall-clock gap so `created_at` differs unambiguously even on a
    // coarse system clock — the oracle must fail on the naive fix
    // regardless of how fast the two creates happen to run.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    // Second (newer) table: model name sorts alphabetically FIRST — a
    // `table_name DESC` tiebreak would wrongly return `older_table` above.
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: &newer_table,
            source_id: &source_id,
            model_id: "aaa_model",
            task: ModelTask::TextEmbedding,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: &format!("file:///tmp/{newer_table}.parquet"),
            dimensions: Some(8),
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();
    catalog
        .update_result_table_status(&newer_table, ResultTableStatus::Ready, 4)
        .await
        .unwrap();

    let resolved = catalog
        .resolve_embedding_table(&source_id, None)
        .await
        .unwrap();
    assert_eq!(
        resolved.table_name, newer_table,
        "resolver must return the temporally-newest ready model table, \
         not the alphabetically-greatest table_name"
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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: url.as_str(),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();

    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    // A promotable torn state: the manifest sidecar landed before the crash.
    jammi_test_utils::write_manifest_sidecar_for(&store, &url, "src", 0).await;
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
    let store = ResultStore::new(dir.path(), catalog, AnnIndexConfig::default()).unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::TextEmbedding,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
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
    // No ANN index is generated at table creation — the index materialises
    // lazily as segments, so a freshly created table has an empty segment set.
    assert!(
        store
            .catalog()
            .list_index_segments(&info.table_name)
            .await
            .unwrap()
            .is_empty(),
        "a freshly created embedding table has no index segments yet"
    );
}

/// A `Binary`-precision deployment default stamps a NEW table's catalog row
/// with `oversample = 32` (`StoragePrecision::Binary::default_oversample`),
/// not the shared `4` every other precision stamps under an untouched
/// deployment config — Binary's single-bit Hamming coarse stage needs the
/// much wider oversample the Wave 1.5 go/no-go spike measured, and a
/// deployment that never explicitly configured `oversample` must not
/// silently under-provision it.
#[tokio::test]
async fn binary_precision_table_stamps_precision_specific_oversample() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let ann = AnnIndexConfig {
        storage_precision: jammi_db::config::StoragePrecision::Binary,
        ..AnnIndexConfig::default()
    };
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), ann).unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::TextEmbedding,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            Some(64),
            None,
            None,
        )
        .await
        .unwrap();

    let record = catalog
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        record.storage_precision,
        Some(jammi_db::config::StoragePrecision::Binary)
    );
    assert_eq!(
        record.oversample,
        Some(32),
        "a Binary table must stamp oversample=32, not the shared default of 4"
    );
}

/// An explicit deployment override of `oversample` is honored verbatim even
/// at `Binary` precision — the precision-specific default only kicks in when
/// the deployment has left `oversample` unset (`None`).
#[tokio::test]
async fn binary_precision_table_honors_explicit_oversample_override() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let ann = AnnIndexConfig {
        storage_precision: jammi_db::config::StoragePrecision::Binary,
        oversample: Some(8),
        ..AnnIndexConfig::default()
    };
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), ann).unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::TextEmbedding,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            Some(64),
            None,
            None,
        )
        .await
        .unwrap();

    let record = catalog
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        record.oversample,
        Some(8),
        "an explicit deployment oversample override must be honored, not widened to 32"
    );
}

/// The exact case the adversarial audit flagged: a deployment that has
/// EXPLICITLY configured `oversample = Some(4)` on a `Binary` table must stamp
/// `4` verbatim — never silently widened to Binary's own per-precision
/// default of `32` just because `4` also happens to equal the shared
/// `DEFAULT_OVERSAMPLE` every other precision defaults to.
#[tokio::test]
async fn binary_precision_table_honors_an_explicit_four_not_widened_to_thirty_two() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let ann = AnnIndexConfig {
        storage_precision: jammi_db::config::StoragePrecision::Binary,
        oversample: Some(4),
        ..AnnIndexConfig::default()
    };
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), ann).unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::TextEmbedding,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            Some(64),
            None,
            None,
        )
        .await
        .unwrap();

    let record = catalog
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        record.oversample,
        Some(4),
        "an explicit oversample=4 on a Binary deployment must be honored verbatim, \
         not silently widened to Binary's own default of 32"
    );
}

/// TRIPWIRE for the identity-completeness argument that lets
/// `ProducingDescriptor::NeighborGraph` fold `index_storage_precision` but
/// omit `oversample`: every row `create_table` persists must couple the two
/// columns so a rescoring precision never carries a `None` oversample — a
/// live-mutable `oversample` can only go live under a precision that already
/// makes the merge inexact-and-rescored, never under a precision where the
/// merge is exact regardless of `oversample`. If this invariant ever broke,
/// the omission would let a definition-hash cache-hit silently replay a
/// stale oversample under a folded, rescoring precision.
#[tokio::test]
async fn create_table_couples_rescoring_precision_to_a_present_oversample() {
    for precision in [
        jammi_db::config::StoragePrecision::F32,
        jammi_db::config::StoragePrecision::F16,
        jammi_db::config::StoragePrecision::Int8,
        jammi_db::config::StoragePrecision::Binary,
    ] {
        let dir = tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
        let ann = AnnIndexConfig {
            storage_precision: precision,
            ..AnnIndexConfig::default()
        };
        let store = ResultStore::new(dir.path(), Arc::clone(&catalog), ann).unwrap();

        let info = store
            .create_table(
                "patents",
                ModelTask::TextEmbedding,
                jammi_db::catalog::result_repo::ResultTableKind::Model,
                None,
                "model",
                Some(64),
                None,
                None,
            )
            .await
            .unwrap();

        let record = catalog
            .get_result_table(&info.table_name)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            record.storage_precision,
            Some(precision),
            "create_table must stamp the precision it was configured at"
        );
        if precision.needs_rescore() {
            assert!(
                record.oversample.is_some(),
                "{precision:?} needs a rescore companion, so its persisted row must carry a \
                 present oversample — a None here would let a live-mutable oversample go live \
                 under a folded, rescoring precision"
            );
        }
    }
}

/// A `ResultStore` rooted at a `memory://` URL roots every created table
/// under that URL and round-trips a written batch back through the shared
/// in-memory object store — the hermetic stand-in for an `r2://`/`s3://`
/// result root, exercising the cloud code path with no network.
#[tokio::test]
async fn result_store_with_memory_root_roots_and_roundtrips() {
    let dir = tempdir().unwrap();
    // Catalog stays local (SQLite under artifact_dir); only the result
    // tables move to the in-memory "cloud" root.
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let registry = StorageRegistry::new();
    let root = StorageUrl::memory("jammi_results");
    let store = ResultStore::with_root(
        root,
        registry,
        Arc::clone(&catalog),
        AnnIndexConfig::default(),
        dir.path().join("index_cache"),
    )
    .unwrap();

    let info = store
        .create_table(
            "patents",
            ModelTask::Classification,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "model",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    // The table's parquet URL is rooted at the memory root, not local disk.
    assert!(
        info.parquet_url
            .as_str()
            .starts_with("memory:///jammi_results/"),
        "parquet_url not under memory root: {}",
        info.parquet_url
    );

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef],
    )
    .unwrap();
    let mut writer = store
        .open_writer(&info.parquet_url, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();
    assert_eq!(rows, 3);

    // Read back through the same registry-cached in-memory driver.
    let handle = store.open_parquet(&info.parquet_url).unwrap();
    assert!(is_valid_parquet(&handle).await.unwrap());
    assert_eq!(count_parquet_rows(&handle).await.unwrap(), 3);
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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: missing_url.as_str(),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();

    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: bad_url.as_str(),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();

    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
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
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: url.as_str(),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();

    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    // A promotable torn state: the manifest sidecar landed before the crash, so
    // recovery promotes it (a manifest-less valid Parquet would be reaped).
    jammi_test_utils::write_manifest_sidecar_for(&store, &url, "src", 0).await;
    store.recover().await.unwrap();

    let record = catalog.get_result_table("stuck").await.unwrap().unwrap();
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 3);
}

// ─── typed-null catalog bind (dimensions: Option<i32> -> INTEGER) ───────────

/// A deterministic, strictly-increasing `now_sortable`-format `created_at`
/// for tests that assert "newest wins" — `now_sortable` is wall-clock
/// (`chrono::Utc::now`), which is not monotonic and has no guaranteed
/// resolution, so back-to-back un-slept inserts must not rely on it to
/// order themselves. Binding explicit values here makes the ordering the
/// test asserts on independent of clock behavior entirely.
fn sortable_at(seq: u64) -> String {
    format!("2020-01-01T00:00:00.{seq:09}Z")
}

/// A dimensionless result table (e.g. a classifier, whose output isn't a
/// fixed-width vector) writes `dimensions: None`. That `None` must bind a
/// typed SQL null (`INTEGER` on the `dimensions` column), not a bare text
/// null — Postgres rejects a text null bound into an `INTEGER` column, so
/// this is the catalog-side anti-regression oracle for the shared
/// `SqlValue::Null` typed-bind fix; the mutable-table-side oracle lives in
/// `mutable_tables.rs`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn result_table_none_dimensions_round_trips_as_null(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = match make_test_session(backend, dir.path()).await {
        Some(s) => s,
        None => {
            eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
    let catalog = session.catalog();

    let suffix = unique_suffix();
    let source_id = format!("classifier_src_{suffix}");
    let table_name = format!("classifier_out_{suffix}");

    catalog
        .register_source(
            &source_id,
            SourceType::File,
            &SourceConnection {
                url: Some(format!("file:///tmp/{source_id}.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    catalog
        .create_result_table(CreateResultTableParams {
            table_name: &table_name,
            source_id: &source_id,
            model_id: "acme/sentiment-classifier",
            task: ModelTask::Classification,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: &format!("file:///tmp/{table_name}.parquet"),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();

    let record = catalog
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        record.dimensions, None,
        "a dimensionless result table must round-trip NULL, not error binding a typed null"
    );
}
