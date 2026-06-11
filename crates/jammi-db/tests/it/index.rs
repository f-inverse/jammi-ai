use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion::common::TableReference;
use datafusion::datasource::file_format::options::ParquetReadOptions;
use datafusion::prelude::SessionContext;
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;
use tempfile::tempdir;

// ─── SidecarIndex: add, search, edge cases ───────────────────────────────────

#[test]
fn sidecar_add_search_and_edge_cases() {
    // Core: add vectors, search returns correct nearest neighbor
    let mut index = SidecarIndex::new(3).unwrap();
    index.add("row_a", &[1.0, 0.0, 0.0]).unwrap();
    index.add("row_b", &[0.0, 1.0, 0.0]).unwrap();
    index.add("row_c", &[0.9, 0.1, 0.0]).unwrap();
    index.build().unwrap();

    assert_eq!(index.len(), 3);

    let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "row_a", "Nearest should be row_a");
    assert!(
        results[0].1 < results[1].1,
        "Results sorted by distance ascending"
    );

    // Edge: k > count returns all
    let results = index.search(&[1.0, 0.0, 0.0], 100).unwrap();
    assert_eq!(results.len(), 3);

    // Edge: empty index
    let empty = SidecarIndex::new(3).unwrap();
    assert!(empty.search(&[1.0, 0.0, 0.0], 5).unwrap().is_empty());
    assert!(empty.is_empty());
}

// ─── Save/load roundtrip with manifest verification ──────────────────────────

#[test]
fn sidecar_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("test_index");

    let mut index = SidecarIndex::new(3).unwrap();
    index.add("id_1", &[1.0, 0.0, 0.0]).unwrap();
    index.add("id_2", &[0.0, 1.0, 0.0]).unwrap();
    index.add("id_3", &[0.0, 0.0, 1.0]).unwrap();
    index.build().unwrap();
    index.save(&base_path).unwrap();

    // Sidecar bundle produced
    assert!(base_path.with_extension("usearch").exists());
    assert!(base_path.with_extension("rowmap").exists());
    assert!(base_path.with_extension("manifest.json").exists());

    // Manifest has required fields
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(base_path.with_extension("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["dimensions"], 3);
    assert_eq!(manifest["count"], 3);
    assert_eq!(manifest["metric"], "cosine");
    assert_eq!(manifest["backend"], "usearch");

    // Load and verify search still works (row_id mapping survives)
    let loaded = SidecarIndex::load(&base_path).unwrap();
    assert_eq!(loaded.len(), 3);
    let results = loaded.search(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].0, "id_1");
}

// ─── Corruption detection ────────────────────────────────────────────────────

#[test]
fn sidecar_load_rejects_corrupted_rowmap() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("bad_version");

    let mut index = SidecarIndex::new(2).unwrap();
    index.add("r1", &[1.0, 0.0]).unwrap();
    index.build().unwrap();
    index.save(&base_path).unwrap();

    // Corrupt the rowmap version byte
    let map_path = base_path.with_extension("rowmap");
    let mut data = std::fs::read(&map_path).unwrap();
    data[0..4].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&map_path, data).unwrap();

    assert!(
        SidecarIndex::load(&base_path).is_err(),
        "Should reject unknown rowmap version"
    );
}

// ─── exact_vector_search: the non-indexed fallback under default schema ───────
//
// `exact_vector_search` is the brute-force path the engine takes for any result
// table WITHOUT an ANN sidecar index (`resolve_search_mode` → `None`). The scan
// reads `_row_id`, a `Utf8` column. DataFusion's default
// `schema_force_view_types` surfaces parquet `Utf8` as `Utf8View`
// (`StringViewArray`), so a downcast that only accepts `StringArray` would
// silently miss the column and fail with "Missing _row_id" — breaking the real
// production fallback. This builds a real Parquet result table through the
// engine's writer, registers it under `jammi.{name}` in a DEFAULT
// `SessionContext` (force-view ON, exactly as production runs), and asserts
// exact search resolves the row ids and ranks the nearest neighbour first.
#[tokio::test]
async fn exact_search_resolves_row_ids_under_default_schema() {
    let dir = tempdir().unwrap();

    let dim = 4_i32;
    let schema = embedding_table_schema(dim as usize);
    // Four rows; row "near" is closest to the query direction below.
    let row_ids = vec![
        "far".to_string(),
        "near".to_string(),
        "mid".to_string(),
        "opp".to_string(),
    ];
    let vectors = [
        [0.0_f32, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ];
    let n = row_ids.len();

    let flat: Vec<f32> = vectors.iter().flat_map(|r| r.iter().copied()).collect();
    let values = Arc::new(Float32Array::from(flat));
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_col = FixedSizeListArray::try_new(item, dim, values, None).unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(row_ids.clone())) as ArrayRef,
            Arc::new(StringArray::from(vec!["src"; n])),
            Arc::new(StringArray::from(vec!["model"; n])),
            Arc::new(vector_col),
        ],
    )
    .unwrap();

    // Write through the engine's Parquet writer so the on-disk encoding matches
    // production — what the default reader then surfaces as `Utf8View`.
    let parquet_path = dir.path().join("exact_table.parquet");
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    // DEFAULT context: `schema_force_view_types` is ON, matching production.
    let ctx = SessionContext::new();
    let table_name = "exact_table";
    let table_ref = TableReference::bare(format!("jammi.{table_name}"));
    ctx.register_parquet(table_ref, url.as_str(), ParquetReadOptions::default())
        .await
        .unwrap();

    // Query points along the "near" direction; expect "near" ranked first and
    // every row id resolved (not lost to a failed downcast).
    let results = exact_vector_search(&ctx, table_name, &[1.0, 0.0, 0.0, 0.0], 4)
        .await
        .expect("exact search must resolve _row_id under default schema");

    assert_eq!(results.len(), n, "every row scored");
    assert_eq!(results[0].0, "near", "nearest neighbour ranked first");
    let resolved: std::collections::HashSet<&str> =
        results.iter().map(|(id, _)| id.as_str()).collect();
    for id in &row_ids {
        assert!(
            resolved.contains(id.as_str()),
            "row id '{id}' must be resolved from the Utf8View column"
        );
    }
}
