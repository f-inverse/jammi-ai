//! `exact_vector_search` — brute-force cosine search over a registered
//! Parquet table. These tests pin the *determinism* of the top-k under tied
//! distances: when several rows are exactly equidistant from the query, the
//! returned prefix must order ties by ascending `_row_id` so the result is
//! independent of parquet scan order and stable across repeated calls. The
//! harness writes a tempdir-backed parquet through the engine's
//! `ObjectParquetWriter` and registers it under the same `jammi.{name}`
//! identifier the query side resolves, so the test exercises the real read
//! path rather than a synthetic in-memory table.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use jammi_db::index::exact::exact_vector_search;
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;
use tempfile::tempdir;

/// Build a `FixedSizeList<Float32>` of inner length `dim` from `rows`.
fn fixed_size_list_from(rows: &[Vec<f32>], dim: i32) -> FixedSizeListArray {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let values = Arc::new(Float32Array::from(flat));
    let field = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float32,
        false,
    ));
    FixedSizeListArray::try_new(field, dim, values, None).unwrap()
}

/// Write `(row_id, vector)` rows to a tempdir parquet through the engine's
/// object writer and register it under `jammi.{table_name}` — the exact
/// quoted-bare identifier `exact_vector_search` resolves — returning a live
/// `SessionContext` ready to query. `dir` is borrowed for the lifetime of the
/// parquet file the context reads lazily.
async fn register_embedding_table(
    dir: &std::path::Path,
    table_name: &str,
    dim: i32,
    row_ids: &[&str],
    vectors: &[Vec<f32>],
) -> SessionContext {
    let schema = embedding_table_schema(dim as usize);
    let n = row_ids.len();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(row_ids.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(vec!["src"; n])),
            Arc::new(StringArray::from(vec!["model"; n])),
            Arc::new(fixed_size_list_from(vectors, dim)),
        ],
    )
    .unwrap();

    let parquet_path = dir.join(format!("{table_name}.parquet"));
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    // Run under the engine's DEFAULT schema settings (`schema_force_view_types`
    // on): `_row_id` surfaces as a `StringViewArray`, which `exact_vector_search`
    // casts to `Utf8` before reading. The determinism check therefore exercises
    // the real production path, not a relaxed one.
    let ctx = SessionContext::new();
    ctx.register_parquet(
        TableReference::bare(format!("jammi.{table_name}")),
        url.as_str(),
        datafusion::datasource::file_format::options::ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx
}

/// Four one-hot embeddings against a query that lies equidistant from two of
/// them. With query `[1, 1, 0, 0]` the basis vectors `e0` and `e1` share an
/// identical cosine distance (dot 1, unit norms), while `e2` and `e3` are
/// orthogonal (dot 0) and strictly farther. The top-2 therefore selects the
/// `{e0, e1}` tie, whose order must be fixed by ascending `_row_id` — not by
/// the order rows happen to stream out of the parquet.
#[tokio::test]
async fn tied_distances_break_on_ascending_row_id() {
    let dir = tempdir().unwrap();
    let dim = 4_i32;
    let query = [1.0_f32, 1.0, 0.0, 0.0];

    // Row ids deliberately *not* in basis order: the tie winner is "a"/"b"
    // (the e0/e1 rows), and "a" < "b" must come first regardless of the fact
    // that "b" is written to the parquet ahead of "a".
    let row_ids = ["d", "b", "c", "a"];
    let vectors = vec![
        vec![0.0, 0.0, 0.0, 1.0], // d → e3, far (dot 0)
        vec![0.0, 1.0, 0.0, 0.0], // b → e1, tie  (dot 1)
        vec![0.0, 0.0, 1.0, 0.0], // c → e2, far (dot 0)
        vec![1.0, 0.0, 0.0, 0.0], // a → e0, tie  (dot 1)
    ];
    let ctx = register_embedding_table(dir.path(), "tie_break", dim, &row_ids, &vectors).await;

    let top2 = exact_vector_search(&ctx, "tie_break", &query, 2)
        .await
        .unwrap();

    // Both tied rows surface, ascending row id breaks the tie: "a" before "b".
    assert_eq!(
        top2.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "tied distances must order by ascending row id"
    );
    assert_eq!(
        top2[0].1, top2[1].1,
        "the two selected neighbours are genuinely equidistant"
    );
}

/// The tie-break must be stable: re-running the search, and re-registering the
/// same rows in a *shuffled* write order, both yield the identical row-id
/// ordered prefix. This is the property the distance-only sort violated — a
/// scan-order-dependent `truncate(k)` would surface different ties per run.
#[tokio::test]
async fn tied_top_k_is_stable_across_repeats_and_input_order() {
    let dir = tempdir().unwrap();
    let dim = 4_i32;
    let query = [1.0_f32, 1.0, 1.0, 0.0];

    // Three basis vectors e0,e1,e2 are all equidistant from this query (dot 1,
    // unit norm); e3 is orthogonal and farther. The top-3 is the full tie set,
    // whose deterministic order is the ascending row ids r1 < r2 < r3.
    let canonical = [
        ("r1", vec![1.0_f32, 0.0, 0.0, 0.0]), // e0, tie
        ("r2", vec![0.0, 1.0, 0.0, 0.0]),     // e1, tie
        ("r3", vec![0.0, 0.0, 1.0, 0.0]),     // e2, tie
        ("r4", vec![0.0, 0.0, 0.0, 1.0]),     // e3, far
    ];
    let expected = ["r1", "r2", "r3"];

    // A handful of distinct write orders — including reverse — must all read
    // back the same row-id ordered prefix.
    let orders: [&[usize]; 4] = [&[0, 1, 2, 3], &[3, 2, 1, 0], &[2, 0, 3, 1], &[1, 3, 0, 2]];
    for (run, order) in orders.iter().enumerate() {
        let row_ids: Vec<&str> = order.iter().map(|&i| canonical[i].0).collect();
        let vectors: Vec<Vec<f32>> = order.iter().map(|&i| canonical[i].1.clone()).collect();
        let table = format!("tie_stable_{run}");
        let ctx = register_embedding_table(dir.path(), &table, dim, &row_ids, &vectors).await;

        // Repeat the call on the same context to prove per-call stability too.
        for _ in 0..3 {
            let top3 = exact_vector_search(&ctx, &table, &query, 3).await.unwrap();
            assert_eq!(
                top3.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                expected,
                "shuffled write order {order:?} must still yield the row-id ordered tie prefix"
            );
        }
    }
}
