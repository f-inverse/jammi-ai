//! The ANN-vs-exact recall mechanism: how the harness measures how well a
//! frozen sidecar index recovers the exact nearest neighbours.
//!
//! ## The two retrievers
//!
//! * **Exact oracle** — the engine's [`exact_vector_search`], a brute-force scan
//!   over every corpus vector returning the `k` closest under a `(dist, _row_id)`
//!   total order. It is deterministic and exhaustive, so its top-`k` *is* ground
//!   truth: recall is measured against it, never the other way round.
//! * **Frozen ANN** — a [`SidecarIndex`] **loaded** from a committed `.usearch`
//!   bundle. USearch's HNSW build is nondeterministic (default `IndexOptions`
//!   pins no seed and no thread count), so the index is built and frozen *once*
//!   on the emit box and committed; the recall gate only ever [`SidecarIndex::load`]s
//!   it. Rebuilding here would measure a different graph than the one shipped,
//!   and the number would not be reproducible — so this module loads and never
//!   builds.
//!
//! ## Recall as a set-intersection floor
//!
//! For one query, ANN recall@k is `|ANN_topk ∩ EXACT_topk| / k`, intersection
//! taken over `_row_id`s. It is a *set* intersection: a neighbour the ANN found
//! counts whether or not it sits at the same rank the oracle put it at, so the
//! measure is insensitive to within-top-k ordering — exactly the latitude an
//! approximate index is allowed. recall@k over a query *set* is the mean of the
//! per-query fractions. The gate asserts each recall@k stays at or above a
//! committed floor (a `>=`, never an equality and never a bit-compare), because
//! the meaningful claim is "the ANN recovers at least this fraction of the true
//! neighbours", not "the ANN reproduces a specific graph".
//!
//! ## What this module proves vs. what a later gate proves
//!
//! This module is the *mechanism*: load-frozen-ANN, run-exact-oracle,
//! set-intersect, average. Its hermetic tests drive that mechanism over a tiny
//! deterministic synthetic fixture, where the correctness is hand-checkable. The *meaningful* recall floor (recall@k ≥ 0.95 over the
//! committed 170k real-embedding corpus) is asserted by a committed-fixture gate
//! added after the on-box emit — a later PR, not this one. Synthetic vectors
//! prove the math here; they cannot stand in for real-embedding recall quality,
//! and this module fakes no scale number.

use std::collections::BTreeMap;
use std::path::Path;

use datafusion::prelude::SessionContext;

use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

use crate::corpus;
use crate::report::{Measurement, RECALL_KS};

/// File names of the committed recall fixture, relative to its bundle directory.
///
/// One directory holds the whole recall input: the corpus the oracle scans and
/// the frozen sidecar bundle (three USearch files sharing the `STEM`). Naming
/// them once here keeps the emit side (which writes them) and the gate side
/// (which reads them) on one definition.
const CORPUS_FILE: &str = "vectors.parquet";
const SIDECAR_STEM: &str = "ann";

/// How many corpus rows form the query-by-example set: the first
/// [`RECALL_QUERY_COUNT`] rows by sorted `_row_id`.
///
/// The query set is a deterministic projection of the committed corpus (the
/// sorted-`_row_id` subset), not a separately committed file — so it is fixed by
/// the corpus alone and reproduces on every box. Querying with corpus rows is a
/// real query-by-example recall measurement: each query's true nearest neighbour
/// is itself, and the ANN's job is to also surface the surrounding cluster the
/// oracle ranks next.
const RECALL_QUERY_COUNT: usize = 64;

/// The table name the recall corpus registers under inside its `SessionContext`.
const RECALL_TABLE: &str = "recall_corpus";

/// Recall@k for one query: the fraction of the exact top-`k` neighbours the ANN
/// also returned, as a set intersection over `_row_id`s.
///
/// Both inputs are `(row_id, dist)` lists; only the ids participate — distances
/// ride along from the retrievers but the intersection is id-on-id, so a
/// neighbour found at a different rank (or a different reported distance) still
/// counts. `k` is the denominator the recall is *defined* against, not the
/// length of either list: a degenerate retriever returning fewer than `k`
/// simply scores lower, never divides by a smaller number. `k == 0` yields 0.0
/// rather than dividing by zero.
fn recall_at_k_for_query(ann: &[(String, f32)], exact: &[(String, f32)], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let ann_ids: std::collections::HashSet<&str> = ann.iter().map(|(id, _)| id.as_str()).collect();
    let hits = exact
        .iter()
        .take(k)
        .filter(|(id, _)| ann_ids.contains(id.as_str()))
        .count();
    hits as f64 / k as f64
}

/// Mean recall@k over a query set: load the frozen sidecar once, and for each
/// query intersect its ANN top-`k` against the exact oracle's top-`k`.
///
/// `sidecar_base` is the bundle base path (the `.usearch`/`.rowmap`/`.manifest`
/// stem) — [`SidecarIndex::load`] reconstructs the *frozen* graph; it is never
/// rebuilt. `table_name` is the corpus already registered in `ctx`, over which
/// [`exact_vector_search`] computes ground truth. The two retrievers run over
/// the same vectors (the sidecar was frozen over this corpus), so the
/// intersection is meaningful.
///
/// An empty `queries` yields 0.0 — there is nothing to average, and a caller
/// asserting a floor over no queries is a bug the 0.0 surfaces rather than a
/// vacuous 1.0 hiding it.
pub async fn mean_recall_at_k(
    ctx: &SessionContext,
    table_name: &str,
    sidecar_base: &std::path::Path,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    if queries.is_empty() {
        return Ok(0.0);
    }
    // LOAD the frozen sidecar — never rebuild. The committed graph is the one
    // whose recall is being measured.
    let index = SidecarIndex::load(sidecar_base)?;

    let mut total = 0.0;
    for query in queries {
        let exact = exact_vector_search(ctx, table_name, query, k).await?;
        let ann = index.search(query, k)?;
        total += recall_at_k_for_query(&ann, &exact, k);
    }
    Ok(total / queries.len() as f64)
}

/// Measure the recall curve over a committed fixture bundle directory.
///
/// The directory holds the corpus parquet ([`CORPUS_FILE`]) and the frozen
/// sidecar bundle ([`SIDECAR_STEM`]`.usearch`/`.rowmap`/`.manifest.json`). The
/// query set is derived deterministically as the first [`RECALL_QUERY_COUNT`]
/// corpus rows by sorted `_row_id` — a fixed function of the committed corpus,
/// not a separate file. For each k in [`RECALL_KS`] this runs the exact oracle
/// over the corpus and the loaded (never rebuilt) sidecar over the same vectors,
/// and reports the mean set-intersection recall@k as a real [`Measurement`].
///
/// This is the recall *path* the `arxiv` subcommand drives. It is decoupled from
/// any one corpus: the bundle the directory holds is whatever was committed
/// there — the on-box real-embedding emit, or a deterministic subset of it. The
/// emit, and the assertion of the meaningful floor against a committed golden,
/// land in a later step; here the path computes a genuine curve from whatever
/// bundle is present, and reports the absence of the bundle as an error rather
/// than a faked number.
pub async fn recall_curve(
    fixture_dir: &Path,
) -> Result<BTreeMap<usize, Measurement>, Box<dyn std::error::Error>> {
    let corpus_path = fixture_dir.join(CORPUS_FILE);
    let sidecar_base = fixture_dir.join(SIDECAR_STEM);

    let url = corpus::storage_url(&corpus_path)?;
    let ctx = corpus::register(&url, RECALL_TABLE).await?;

    // The query set is the deterministic sorted-`_row_id` projection of the
    // corpus — committed implicitly by the corpus, reproducible on any box.
    let corpus_rows = corpus::load_vectors(&ctx, RECALL_TABLE).await?;
    let queries: Vec<Vec<f32>> = corpus::sorted_row_id_subset(corpus_rows, RECALL_QUERY_COUNT)
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err(format!(
            "recall fixture at {} has an empty corpus — no queries to project",
            fixture_dir.display()
        )
        .into());
    }

    let mut curve = BTreeMap::new();
    for &k in &RECALL_KS {
        let recall = mean_recall_at_k(&ctx, RECALL_TABLE, &sidecar_base, &queries, k).await?;
        curve.insert(k, Measurement::measured(recall, "fraction"));
    }
    Ok(curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
    use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
    use jammi_db::store::schema::embedding_table_schema;
    use tempfile::tempdir;

    use crate::corpus;

    /// A tiny deterministic corpus: `n` rows of width `dim`, each a distinct
    /// pseudo-random *direction* drawn from a seeded LCG (the same generator the
    /// synthetic scale corpus uses). Random high-dimensional directions are
    /// well-separated under cosine distance, so the exact nearest neighbour of
    /// any corpus row is unambiguously itself — the property the recall and
    /// oracle assertions hand-check. A scale-then-shift over near-collinear rows
    /// would instead collapse under cosine (which ignores magnitude), so the
    /// directions must genuinely differ, not just the lengths.
    fn tiny_rows(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        // Numerical-Recipes LCG constants — fully reproducible, no rng crate.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
        };
        (0..n)
            .map(|i| {
                let id = format!("row_{i:03}");
                let v = (0..dim).map(|_| next()).collect();
                (id, v)
            })
            .collect()
    }

    /// Write `rows` to a Parquet object at `path` through the engine writer, the
    /// same production path the synthetic and committed corpora use.
    async fn write_corpus(path: &std::path::Path, rows: &[(String, Vec<f32>)], dim: usize) {
        let schema = embedding_table_schema(dim);
        let url = StorageUrl::parse(path.to_str().unwrap()).unwrap();
        let registry = StorageRegistry::new();
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);
        let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
            .await
            .unwrap();

        let ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let values = Arc::new(Float32Array::from(flat));
        let item = Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::Float32,
            false,
        ));
        let vectors = FixedSizeListArray::try_new(item, dim as i32, values, None).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(vec!["src"; rows.len()])),
                Arc::new(StringArray::from(vec!["model"; rows.len()])),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.close().await.unwrap();
    }

    /// Build a sidecar over `rows` and freeze it to `base` (`.usearch`/`.rowmap`/
    /// `.manifest`). This is the *one* build — it stands in for the on-box emit;
    /// the recall path under test only ever loads what this writes.
    fn freeze_sidecar(base: &std::path::Path, rows: &[(String, Vec<f32>)], dim: usize) {
        let mut index = SidecarIndex::new(dim).unwrap();
        for (id, v) in rows {
            index.add(id, v).unwrap();
        }
        index.build().unwrap();
        VectorIndex::save(&index, base).unwrap();
    }

    /// The recall computation is correct: a sidecar frozen over the *same*
    /// vectors the exact oracle scores recovers the exact neighbours, so
    /// recall@k == 1.0 for every k. This proves the load-frozen-ANN /
    /// run-exact-oracle / set-intersect / average path end to end.
    ///
    /// The MEANINGFUL real-embedding recall FLOOR (recall@k ≥ 0.95 over the
    /// committed 170k corpus) is asserted by a committed-fixture gate added after
    /// the on-box emit — a later PR. Synthetic vectors here prove the mechanism
    /// is correct; they cannot stand in for real-embedding recall quality.
    #[tokio::test]
    async fn ann_over_same_corpus_recovers_exact_neighbours() {
        let dim = 8;
        let n = 64;
        let rows = tiny_rows(n, dim);

        let dir = tempdir().unwrap();
        let corpus_path = dir.path().join("tiny.parquet");
        let sidecar_base = dir.path().join("tiny");

        write_corpus(&corpus_path, &rows, dim).await;
        freeze_sidecar(&sidecar_base, &rows, dim);

        let url = StorageUrl::parse(corpus_path.to_str().unwrap()).unwrap();
        let table = "tiny_corpus";
        let ctx = corpus::register(&url, table).await.unwrap();

        // Queries are exact corpus rows, so the exact top-1 of each is itself —
        // a hand-checkable oracle. Use a handful spread across the corpus.
        let queries: Vec<Vec<f32>> = [0usize, 7, 31, 63]
            .iter()
            .map(|&i| rows[i].1.clone())
            .collect();

        // On the same corpus, exact and HNSW agree at this scale: recall is 1.0.
        for k in [1usize, 10] {
            let recall = mean_recall_at_k(&ctx, table, &sidecar_base, &queries, k)
                .await
                .unwrap();
            assert_eq!(
                recall, 1.0,
                "ANN over the same frozen corpus must recover the exact top-{k}"
            );
        }
    }

    /// The exact oracle reproduces a hand-computable top-k: querying with a
    /// corpus row returns that row first (distance ~0), then its nearest corpus
    /// neighbours in `_row_id` tie-break order.
    #[tokio::test]
    async fn exact_oracle_returns_hand_checkable_top_k() {
        let dim = 8;
        let n = 64;
        let rows = tiny_rows(n, dim);

        let dir = tempdir().unwrap();
        let corpus_path = dir.path().join("tiny.parquet");
        write_corpus(&corpus_path, &rows, dim).await;

        let url = StorageUrl::parse(corpus_path.to_str().unwrap()).unwrap();
        let table = "tiny_corpus";
        let ctx = corpus::register(&url, table).await.unwrap();

        // Query == row_005; its own cosine distance to itself is ~0, so it is
        // the unambiguous top-1 the oracle must return first.
        let query = rows[5].1.clone();
        let top = exact_vector_search(&ctx, table, &query, 3).await.unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "row_005", "nearest neighbour of a row is itself");
        assert!(
            top[0].1 <= top[1].1 && top[1].1 <= top[2].1,
            "exact results are sorted by ascending distance, got {top:?}"
        );
    }

    /// A retriever that misses half the exact neighbours scores recall 0.5 —
    /// the set-intersection arithmetic is the fraction recovered, order-blind.
    #[test]
    fn recall_is_the_set_intersection_fraction() {
        let exact: Vec<(String, f32)> = (0..10)
            .map(|i| (format!("row_{i:03}"), i as f32 * 0.1))
            .collect();
        // ANN found 5 of the 10 true neighbours (the even ids), in a scrambled
        // order and with different distances — recall must still be 0.5.
        let ann: Vec<(String, f32)> = [8usize, 0, 6, 2, 4]
            .iter()
            .map(|&i| (format!("row_{i:03}"), 0.42))
            .collect();
        assert_eq!(recall_at_k_for_query(&ann, &exact, 10), 0.5);
        // A perfect retriever scores 1.0; an empty one scores 0.0.
        assert_eq!(recall_at_k_for_query(&exact, &exact, 10), 1.0);
        assert_eq!(recall_at_k_for_query(&[], &exact, 10), 0.0);
    }

    /// The sorted-`_row_id` subset helper returns the deterministic
    /// first-`n`-by-sorted-id projection, independent of input order.
    #[test]
    fn sorted_subset_is_the_deterministic_projection() {
        // Insert rows out of id order; the helper must sort then truncate.
        let rows: Vec<(String, Vec<f32>)> = [3usize, 0, 4, 1, 2]
            .iter()
            .map(|&i| (format!("row_{i:03}"), vec![i as f32]))
            .collect();
        let subset = corpus::sorted_row_id_subset(rows, 3);
        let ids: Vec<&str> = subset.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["row_000", "row_001", "row_002"]);
        // The vectors travel with their ids — the projection is on whole rows.
        assert_eq!(subset[0].1, vec![0.0]);
        assert_eq!(subset[2].1, vec![2.0]);
    }
}
