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
use jammi_numerics::distance::cosine_distance;
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

/// Write `(row_id, vector)` rows to a tempdir parquet in `batch_rows`-sized
/// chunks through the engine's object writer, registering it under
/// `jammi.{table_name}`. Splitting the write into many `RecordBatch`es forces
/// the streamed `exact_vector_search` to actually fold across batch boundaries
/// — the per-batch drop that keeps its footprint bounded only matters when the
/// scan yields more than one batch.
async fn register_embedding_table_chunked(
    dir: &std::path::Path,
    table_name: &str,
    dim: i32,
    row_ids: &[String],
    vectors: &[Vec<f32>],
    batch_rows: usize,
) -> SessionContext {
    let schema = embedding_table_schema(dim as usize);
    let parquet_path = dir.join(format!("{table_name}.parquet"));
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();

    for (ids_chunk, vecs_chunk) in row_ids.chunks(batch_rows).zip(vectors.chunks(batch_rows)) {
        let n = ids_chunk.len();
        let id_refs: Vec<&str> = ids_chunk.iter().map(String::as_str).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(id_refs)) as ArrayRef,
                Arc::new(StringArray::from(vec!["src"; n])),
                Arc::new(StringArray::from(vec!["model"; n])),
                Arc::new(fixed_size_list_from(vecs_chunk, dim)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
    }
    writer.close().await.unwrap();

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

/// Deterministic, hermetic corpus generator: a 64-bit LCG (the
/// Numerical-Recipes constants) seeded per call, mapped to `f32` in `[-1, 1)`.
/// No randomness crate, no entropy — the same seed always yields the same
/// corpus, so the bit-identical equivalence proof is reproducible bit-for-bit.
fn lcg_corpus(seed: u64, rows: usize, dim: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Take the high 24 bits as a fraction so the value is well-spread and
        // independent of the low-bit LCG correlation.
        let frac = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        frac * 2.0 - 1.0
    };
    let mut row_ids = Vec::with_capacity(rows);
    let mut vectors = Vec::with_capacity(rows);
    for r in 0..rows {
        // Zero-padded so lexical `_row_id` order matches numeric order, which
        // makes the tie-break order legible when distances coincide.
        row_ids.push(format!("row_{r:09}"));
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(next());
        }
        vectors.push(v);
    }
    (row_ids, vectors)
}

/// The reference oracle: collect *every* row, score it, and sort the whole set
/// by the same total order the streamed path uses — ascending cosine distance,
/// ties broken by ascending `_row_id` — then take the `k` prefix. This is the
/// pre-streaming `collect`-everything path, inlined here so the equivalence
/// test compares the bounded heap against an independent full materialisation
/// rather than against itself.
fn naive_top_k(
    query: &[f32],
    row_ids: &[String],
    vectors: &[Vec<f32>],
    k: usize,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = row_ids
        .iter()
        .zip(vectors)
        .map(|(id, v)| (id.clone(), cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored
}

/// Assert two `(row_id, distance)` result lists are *bit-identical*: equal
/// length, equal row ids, and — crucially — equal distance *bit patterns*
/// (`f32::to_bits`), not merely equal under `==` (which would let `-0.0` and
/// `0.0`, or any future tolerance, slip through). This is the determinism
/// contract: the streamed path must reproduce the reference exactly.
fn assert_bit_identical(streamed: &[(String, f32)], reference: &[(String, f32)], context: &str) {
    assert_eq!(
        streamed.len(),
        reference.len(),
        "{context}: result length differs"
    );
    for (i, (s, r)) in streamed.iter().zip(reference).enumerate() {
        assert_eq!(s.0, r.0, "{context}: row_id differs at position {i}");
        assert_eq!(
            s.1.to_bits(),
            r.1.to_bits(),
            "{context}: distance bits differ at position {i} ({} vs {})",
            s.1,
            r.1
        );
    }
}

/// The determinism proof. A fixed seeded corpus (5000 rows × dim 16) — plus a
/// constructed exact-tie cluster appended at known row ids — is streamed
/// through `exact_vector_search` for `k ∈ {1, 10, 100, full}` against several
/// queries, and every result is asserted bit-identical (row_id == AND
/// distance bits ==) to the inlined `naive_top_k` reference computed over the
/// same in-memory corpus. Because both paths score with the same
/// `cosine_distance` and resolve the same total order, agreement on the tie
/// cluster — where distances genuinely coincide and only the unique `_row_id`
/// decides the kept set and its order — is a true equivalence proof, not a
/// coincidence of distinct distances.
#[tokio::test]
async fn streamed_top_k_is_bit_identical_to_naive_collect() {
    let dir = tempdir().unwrap();
    let dim = 16_usize;
    let (mut row_ids, mut vectors) = lcg_corpus(0xA5A5_1234_DEAD_BEEF, 5000, dim);

    // Construct an exact-tie cluster: eight rows sharing one vector value all
    // lie at *identical* cosine distance from any query, so the kept set among
    // them — and their order — is decided purely by the unique `_row_id`. These
    // ids sort after the `row_*` corpus, exercising tie resolution across the
    // batch boundary too.
    let tie_vector: Vec<f32> = (0..dim)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
        .collect();
    for t in 0..8 {
        row_ids.push(format!("tie_{t:03}"));
        vectors.push(tie_vector.clone());
    }

    // Several queries, including one that *is* the tie vector (distance 0 to the
    // whole cluster — the tie sits at the very front of the result) and a
    // zero-magnitude query (every distance short-circuits to 1.0, so the entire
    // corpus is one giant tie broken solely by row_id).
    let queries: Vec<Vec<f32>> = vec![
        (0..dim).map(|i| (i as f32 * 0.13).sin()).collect(),
        (0..dim)
            .map(|i| if i < dim / 2 { 1.0 } else { -1.0 })
            .collect(),
        tie_vector.clone(),
        vec![0.0; dim],
    ];

    let ctx = register_embedding_table_chunked(
        dir.path(),
        "equivalence",
        dim as i32,
        &row_ids,
        &vectors,
        // Small batches so the stream yields many `RecordBatch`es and the fold
        // crosses batch boundaries repeatedly.
        512,
    )
    .await;

    let total = row_ids.len();
    for (q_idx, query) in queries.iter().enumerate() {
        for k in [1_usize, 10, 100, total] {
            let streamed = exact_vector_search(&ctx, "equivalence", query, k)
                .await
                .unwrap();
            let reference = naive_top_k(query, &row_ids, &vectors, k);
            assert_bit_identical(&streamed, &reference, &format!("query {q_idx}, k={k}"));
        }
    }
}

/// Bounded-memory completion proof. A corpus far larger than the resident
/// working set (1,000,000 rows × dim 4) is streamed through
/// `exact_vector_search` in modest batches and the search completes returning
/// the correct `k` — demonstrating the scan runs without ever materialising all
/// `N` vectors at once. The structural guarantee (the heap holds at most `k`
/// `(row_id, distance)` pairs and the per-batch `vectors` buffer is cleared
/// each iteration) bounds peak memory to `O(k + batch_rows·d)`; this test
/// pins that the bounded path *runs to completion at scale*. A hard
/// binding-tier RSS assertion (15M rows + a negative control proving the old
/// collect-all path breaches a ceiling) is deferred to the out-of-process W1
/// bench harness, where an RSS number can be measured reliably rather than
/// flakily in-process.
#[tokio::test]
async fn streamed_search_completes_over_large_corpus() {
    let dir = tempdir().unwrap();
    let dim = 4_usize;
    let rows = 1_000_000_usize;
    let (row_ids, vectors) = lcg_corpus(0x0BAD_F00D_1234_5678, rows, dim);

    let ctx = register_embedding_table_chunked(
        dir.path(),
        "large",
        dim as i32,
        &row_ids,
        &vectors,
        65_536,
    )
    .await;

    let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
    let top10 = exact_vector_search(&ctx, "large", &query, 10)
        .await
        .unwrap();

    // The streamed result must match the naive full-materialisation reference
    // exactly even at this scale — same correctness contract as the small
    // equivalence test, just over a corpus that would be costly to hold whole.
    let reference = naive_top_k(&query, &row_ids, &vectors, 10);
    assert_bit_identical(&top10, &reference, "large-corpus k=10");
    assert_eq!(top10.len(), 10, "k=10 over a million rows returns ten");
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
