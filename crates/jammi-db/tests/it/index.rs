use jammi_test_utils::vq;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion::common::TableReference;
use datafusion::datasource::file_format::options::ParquetReadOptions;
use datafusion::prelude::SessionContext;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
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
    let mut index =
        SidecarIndex::new(3, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
    index.add("row_a", &[1.0, 0.0, 0.0]).unwrap();
    index.add("row_b", &[0.0, 1.0, 0.0]).unwrap();
    index.add("row_c", &[0.9, 0.1, 0.0]).unwrap();
    index.build().unwrap();

    assert_eq!(index.len(), 3);

    let results = index.search(&vq(&[1.0, 0.0, 0.0]), 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "row_a", "Nearest should be row_a");
    assert!(
        results[0].1 < results[1].1,
        "Results sorted by distance ascending"
    );

    // Edge: k > count returns all
    let results = index.search(&vq(&[1.0, 0.0, 0.0]), 100).unwrap();
    assert_eq!(results.len(), 3);

    // Edge: empty index
    let empty = SidecarIndex::new(3, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
    assert!(empty.search(&vq(&[1.0, 0.0, 0.0]), 5).unwrap().is_empty());
    assert!(empty.is_empty());
}

// ─── get: stored vectors readable back by id ─────────────────────────────────

#[test]
fn sidecar_get_returns_stored_vectors() {
    let mut index =
        SidecarIndex::new(3, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
    index.add("row_a", &[1.0, 0.0, 0.0]).unwrap();
    index.add("row_b", &[0.0, 1.0, 0.0]).unwrap();
    index.build().unwrap();

    // A stored vector is readable back by its id — the index is the single owner
    // of the embeddings, so callers need not keep a second id→vector copy.
    let a = index.get("row_a").unwrap().expect("row_a is indexed");
    assert_eq!(a, vec![1.0, 0.0, 0.0]);
    let b = index.get("row_b").unwrap().expect("row_b is indexed");
    assert_eq!(b, vec![0.0, 1.0, 0.0]);

    // An unknown id is `None`, not an error.
    assert!(index.get("missing").unwrap().is_none());

    // The id→key reverse map survives save/load.
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("get_roundtrip");
    index.save(&base_path).unwrap();
    let loaded = SidecarIndex::load(
        &base_path,
        &AnnIndexConfig::default(),
        StoragePrecision::F32,
    )
    .unwrap();
    assert_eq!(
        loaded.get("row_b").unwrap().expect("row_b after load"),
        vec![0.0, 1.0, 0.0]
    );
}

// ─── Save/load roundtrip with manifest verification ──────────────────────────

#[test]
fn sidecar_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("test_index");

    let mut index =
        SidecarIndex::new(3, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
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
    let loaded = SidecarIndex::load(
        &base_path,
        &AnnIndexConfig::default(),
        StoragePrecision::F32,
    )
    .unwrap();
    assert_eq!(loaded.len(), 3);
    let results = loaded.search(&vq(&[1.0, 0.0, 0.0]), 1).unwrap();
    assert_eq!(results[0].0, "id_1");
}

// ─── Corruption detection ────────────────────────────────────────────────────

#[test]
fn sidecar_load_rejects_corrupted_rowmap() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("bad_version");

    let mut index =
        SidecarIndex::new(2, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
    index.add("r1", &[1.0, 0.0]).unwrap();
    index.build().unwrap();
    index.save(&base_path).unwrap();

    // Corrupt the rowmap version byte
    let map_path = base_path.with_extension("rowmap");
    let mut data = std::fs::read(&map_path).unwrap();
    data[0..4].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&map_path, data).unwrap();

    assert!(
        SidecarIndex::load(
            &base_path,
            &AnnIndexConfig::default(),
            StoragePrecision::F32
        )
        .is_err(),
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
            jammi_db::store::content_hash::null_hash_column(n),
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
    let results = exact_vector_search(&ctx, table_name, &vq(&[1.0, 0.0, 0.0, 0.0]), 4, None)
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

// ─── The exact path: the query is validated against the SCAN width; a
// corrupt stored row is refused at the top-k SINK ───────────────────────────

/// Write a 4-wide embedding parquet with the given rows and register it into
/// a fresh context under `jammi.{name}` — the no-index exact fallback's
/// input, hand-built so a row can carry a non-finite component.
async fn exact_table(
    dir: &std::path::Path,
    name: &str,
    rows: &[(&str, [f32; 4])],
) -> SessionContext {
    let dim = 4_i32;
    let schema = embedding_table_schema(dim as usize);
    let n = rows.len();
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_col =
        FixedSizeListArray::try_new(item, dim, Arc::new(Float32Array::from(flat)), None).unwrap();
    let ids: Vec<String> = rows.iter().map(|(id, _)| id.to_string()).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(vec!["src"; n])),
            Arc::new(StringArray::from(vec!["model"; n])),
            Arc::new(vector_col),
            jammi_db::store::content_hash::null_hash_column(n),
        ],
    )
    .unwrap();
    let url = StorageUrl::parse(dir.join(format!("{name}.parquet")).to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let handle = JammiObjectStore::new(registry.driver_for(&url, None).unwrap(), url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_parquet(
        TableReference::bare(format!("jammi.{name}")),
        url.as_str(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx
}

const FOUR_ROWS: [(&str, [f32; 4]); 4] = [
    ("far", [0.0, 1.0, 0.0, 0.0]),
    ("near", [1.0, 0.0, 0.0, 0.0]),
    ("mid", [0.7, 0.7, 0.0, 0.0]),
    ("opp", [-1.0, 0.0, 0.0, 0.0]),
];

// A1 — a wrong-width query on the NO-INDEX exact path is a typed `Schema`
// error, never a panic. The width comes from the scan schema's
// `FixedSizeList` length (the authority on this path), so it holds even with
// no catalog width in hand (`None`).
#[tokio::test]
async fn exact_search_refuses_a_wrong_width_query_typed_not_panic() {
    let dir = tempdir().unwrap();
    let ctx = exact_table(dir.path(), "exact_width", &FOUR_ROWS).await;
    for width in [5usize, 3, 0] {
        let q = vq(&vec![1.0f32; width]);
        let err = exact_vector_search(&ctx, "exact_width", &q, 4, None)
            .await
            .expect_err("a wrong-width query must be refused, not panic");
        assert!(
            matches!(err, jammi_db::error::JammiError::Schema { .. }),
            "width {width}: {err:?}"
        );
        assert!(err.to_string().contains("4"), "{err}");
    }
    // The catalog width is a CROSS-CHECK against the scan: a disagreement is
    // its own typed, table-named error.
    let err = exact_vector_search(&ctx, "exact_width", &vq(&[1.0, 0.0, 0.0, 0.0]), 4, Some(5))
        .await
        .expect_err("catalog width 5 disagrees with the scan's 4");
    assert!(
        matches!(&err, jammi_db::error::JammiError::IncompatibleFormat { artifact, .. } if artifact.contains("exact_width")),
        "{err:?}"
    );
    // The conforming query still serves.
    let hits = exact_vector_search(&ctx, "exact_width", &vq(&[1.0, 0.0, 0.0, 0.0]), 4, Some(4))
        .await
        .unwrap();
    assert_eq!(hits[0].0, "near");
}

// A1b — a ZERO-width `FixedSizeList` scan column (a corrupt schema, not a
// user query) is refused with a typed error naming the COLUMN, never
// silently treated as a width of `0` against the query. The query here is
// deliberately NON-EMPTY: an empty query would trivially match a width-0
// column under the old, buggy conversion too (0 == 0), so it would prove
// nothing. A non-empty query against a 0-width column is exactly the case
// the old `usize::try_from(*n).unwrap_or(0)` mishandled — it would refuse
// the query as "expected 0 dimensions" (blaming the CALLER's width, column
// "query"), when the true defect is the corrupt scan schema itself (column
// "vector"). Those are different failures; only the second is the fix.
#[tokio::test]
async fn exact_search_refuses_a_zero_width_scan_column_typed_not_a_width_of_zero() {
    let dir = tempdir().unwrap();
    let schema = embedding_table_schema(0);
    // Only the SCHEMA's `FixedSizeList` width matters to the check under
    // test (`exact_vector_search` inspects the scan's schema before reading
    // any row), so an empty (zero-row) batch is enough — a zero-size list
    // array's own length is ambiguous with any non-zero row count.
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_col = FixedSizeListArray::try_new(
        item,
        0,
        Arc::new(Float32Array::from(Vec::<f32>::new())),
        None,
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef,
            Arc::new(StringArray::from(Vec::<String>::new())),
            Arc::new(StringArray::from(Vec::<String>::new())),
            Arc::new(vector_col),
            jammi_db::store::content_hash::null_hash_column(0),
        ],
    )
    .unwrap();
    let url = StorageUrl::parse(dir.path().join("zero_width.parquet").to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let handle = JammiObjectStore::new(registry.driver_for(&url, None).unwrap(), url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_parquet(
        TableReference::bare("jammi.zero_width"),
        url.as_str(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    // Non-empty: 4 components, against the corrupt 0-width scan column.
    let err = exact_vector_search(&ctx, "zero_width", &vq(&[1.0, 0.0, 0.0, 0.0]), 1, None)
        .await
        .expect_err("a zero-width scan column must be refused, never treated as width 0");
    match &err {
        jammi_db::error::JammiError::Schema {
            column, expected, ..
        } => {
            assert_eq!(
                column, "vector",
                "must name the corrupt SCAN column, not \"query\" (the old, wrong shape): {err:?}"
            );
            assert!(
                expected.contains("positive"),
                "names the corrupt column's own defect, not a query width like \"0 dimensions\": {expected}"
            );
        }
        other => panic!("expected a typed Schema error naming the column: {other:?}"),
    }
}

// B-c — a stored row with a non-finite component yields a non-finite
// distance; the top-k SINK refuses it as a typed, table-named error. Never a
// top-k slot (the comparator is non-transitive under NaN and could rank it
// first), never a silently dropped row (which would read as "fewer matches").
#[tokio::test]
async fn exact_search_refuses_a_corrupt_stored_row_at_the_sink() {
    let dir = tempdir().unwrap();
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let rows = [
            ("ok", [1.0, 0.0, 0.0, 0.0]),
            ("poisoned", [poison, 0.0, 0.0, 0.0]),
            ("other", [0.0, 1.0, 0.0, 0.0]),
        ];
        let name = format!(
            "exact_poison_{}",
            if poison.is_nan() {
                "nan"
            } else if poison > 0.0 {
                "inf"
            } else {
                "ninf"
            }
        );
        let ctx = exact_table(dir.path(), &name, &rows).await;
        let err = exact_vector_search(&ctx, &name, &vq(&[1.0, 0.0, 0.0, 0.0]), 3, None)
            .await
            .expect_err("a corrupt stored row must not become a result");
        match &err {
            jammi_db::error::JammiError::IncompatibleFormat {
                artifact, found, ..
            } => {
                assert!(artifact.contains(&name), "names the table: {artifact}");
                assert!(found.contains("poisoned"), "names the row: {found}");
            }
            other => panic!("expected the corrupt-artifact variant, got {other:?}"),
        }
    }
}

// A5 (provenance, unit level) — the SAME non-finite vector is a caller fault
// when the caller supplied it and a corrupt artifact named by its table when
// it was read back from storage.
#[test]
fn provenance_decides_the_error_class() {
    use jammi_db::error::JammiError;
    use jammi_db::index::{validate_query, QuerySource};
    let caller: JammiError = validate_query(vec![f32::NAN, 0.0], None, QuerySource::Caller)
        .unwrap_err()
        .into();
    assert!(matches!(caller, JammiError::Schema { .. }), "{caller:?}");
    let stored: JammiError = validate_query(
        vec![f32::NAN, 0.0],
        None,
        QuerySource::Stored {
            table: "docs_embeddings".into(),
        },
    )
    .unwrap_err()
    .into();
    match &stored {
        JammiError::IncompatibleFormat { artifact, .. } => {
            assert!(artifact.contains("docs_embeddings"), "{artifact}")
        }
        other => panic!("a stored NaN is a corrupt artifact, got {other:?}"),
    }
}
