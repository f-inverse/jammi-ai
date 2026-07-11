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
//! ## Corpus-as-query vs. held-out queries
//!
//! There are two ways to source the query set, and they measure different
//! things:
//!
//! * **Corpus-as-query** — the queries are corpus rows themselves. Each query's
//!   true nearest neighbour is itself (distance ~0), so recall@1 is structurally
//!   near-1.0 whatever the index quality. This exercises the
//!   load / oracle / intersect / average mechanism (see
//!   `ann_over_same_corpus_recovers_exact_neighbours`); it is *not* a meaningful
//!   quality floor, because a query finding itself says nothing about how the ANN
//!   handles unseen points.
//! * **Held-out queries** ([`recall_curve_held_out`]) — the queries come from a
//!   *separate* embedding set, disjoint from the indexed corpus by construction.
//!   No query is its own neighbour, so recall@k measures how well the frozen ANN
//!   recovers the exact neighbours of unseen points — the quantity a deployed
//!   index is actually judged on. This is the path the `arxiv` subcommand drives
//!   and the path a real recall floor is asserted against.
//!
//! Both run the *same* primitive ([`mean_recall_at_k`]); they differ only in
//! where the query vectors come from. The held-out path takes its queries from a
//! separate parquet rather than from the corpus rows.
//!
//! ## What the engine gate proves vs. what the cookbook proves
//!
//! The hermetic cargo-test gate (`held_out_recall_clears_committed_floor`)
//! loads a *small committed fixture* — a deterministic sorted-`_row_id` subset of
//! the real 170k cache (real embeddings: corpus rows + held-out query rows, with
//! a sidecar frozen over the subset once) — and asserts the held-out recall@k
//! clears a committed floor measured on that same slice. This proves the
//! held-out gate works hermetically on real embeddings, inside `cargo test`,
//! with no LFS dependency.
//!
//! The *full* 168k held-out recall gate runs in the cookbook chapter (a later
//! step), which reads the Git-LFS cache the fixture is subset from. The split is
//! deliberate: the engine repo carries no LFS, so the engine gate proves the
//! held-out floor holds on a small provable projection that ships in the git
//! object store, and the cookbook gate proves it at full scale on the same
//! artifacts the fixture is carved from.
//!
//! ## The precision axis: retrieve→rescore recovery
//!
//! [`mean_recall_at_k_rescored`] generalizes the held-out measurement to a
//! quantized [`StoragePrecision`]: at `Int8` the loaded graph's own vectors are
//! lossy, so a search is the engine's two-stage retrieve→rescore — an
//! oversampled candidate pool off the quantized graph, exactly re-ranked
//! against the `.rawf32` rescore companion. Because recall@k is order-blind
//! (see above), `oversample == 1` measures the quantized graph's own naive
//! top-k (nothing for the rescore to recover), while the deployment's default
//! oversample measures how much of the quantization loss the rescore recovers.
//! `precision_recall_rescore_recovers_and_clears_committed_floor` loads a
//! second frozen bundle (`frozen_int8`, built once over the SAME fixture
//! corpus by `fixture.rs`'s `build_precision_recall_fixture`) and asserts both
//! that each variant clears its own committed floor AND that the rescored
//! variant clears the no-rescore variant by a real margin — the measured proof
//! that oversampling-then-rescoring recovers neighbours the lossy graph alone
//! misses, not just a floor two numbers happen to both clear.

use std::collections::BTreeMap;
use std::path::Path;

use datafusion::prelude::SessionContext;

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

use crate::corpus;
use crate::report::{Measurement, RECALL_KS};

/// File names of the committed *held-out* recall fixture, relative to its bundle
/// directory.
///
/// The held-out bundle holds three inputs rather than two: the corpus the oracle
/// scans and the sidecar is frozen over ([`HELD_OUT_CORPUS_FILE`] +
/// [`HELD_OUT_SIDECAR_STEM`]), and a *separate* query parquet
/// ([`HELD_OUT_QUERY_FILE`]) whose rows are disjoint from the corpus. The
/// disjointness is what makes the recall a generalization measurement rather
/// than a query-by-example one. Naming the files once keeps the fixture builder
/// (which writes them) and the gate (which reads them) on one definition.
const HELD_OUT_CORPUS_FILE: &str = "corpus_vectors.parquet";
const HELD_OUT_QUERY_FILE: &str = "query_vectors.parquet";
const HELD_OUT_SIDECAR_STEM: &str = "frozen";

/// The table name the held-out query set registers under inside its
/// `SessionContext`, distinct from [`RECALL_TABLE`] so corpus and queries can
/// coexist in one context.
const HELD_OUT_QUERY_TABLE: &str = "recall_held_out_queries";

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
pub(crate) fn recall_at_k_for_query(
    ann: &[(String, f32)],
    exact: &[(String, f32)],
    k: usize,
) -> f64 {
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
///
/// A thin `F32`, single-stage wrapper over [`mean_recall_at_k_rescored`] — the
/// oversample is irrelevant at `F32` ([`StoragePrecision::needs_rescore`] is
/// `false`, so the rescore stage never runs), kept as its own name because it
/// is the path every existing F32-only caller (the arxiv tier, the
/// build/search sweep axes) already uses.
pub async fn mean_recall_at_k(
    ctx: &SessionContext,
    table_name: &str,
    sidecar_base: &std::path::Path,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    mean_recall_at_k_rescored(
        ctx,
        table_name,
        sidecar_base,
        StoragePrecision::F32,
        1,
        queries,
        k,
    )
    .await
}

/// Mean recall@k over a query set, for a frozen sidecar bundle loaded at
/// `precision` and queried through the engine's own two-stage
/// retrieve→rescore ([`crate::operator_mirror::retrieve_then_rescore`]) when
/// that precision is quantized.
///
/// At `F32` the loaded index's own stored vectors are already exact
/// ([`StoragePrecision::needs_rescore`] is `false`), so this stays the
/// original single-stage path: `index.search(query, k)` directly, `oversample`
/// unused. At a quantized precision (`F16`/`Int8`) this mirrors the production
/// path in `jammi_ai::operator::ann_search_exec`: the loaded graph's own
/// (lossy) `search` proposes `k * oversample` candidates, each candidate's
/// *exact* `f32` vector is read back via [`SidecarIndex::get_exact`] (the
/// mmap'd rescore companion, never the quantized graph's own reconstruction),
/// cosine distance is recomputed against it, and the re-ranked set is
/// truncated to `k`.
///
/// Because recall@k is a *set* intersection (order-blind, see
/// [`recall_at_k_for_query`]), `oversample == 1` measures exactly the
/// quantized graph's own naive top-`k` — the rescore recomputes distances for
/// the same `k` ids the lossy graph already chose, so it can re-rank them but
/// recover nothing the graph missed. `oversample > 1` is what lets the exact
/// rescore recover a true neighbour the quantized graph ranked just outside
/// its naive top-`k` but still surfaced within the wider `k * oversample`
/// candidate pool — the recall-recovery mechanism the retrieve→rescore design
/// exists for.
pub async fn mean_recall_at_k_rescored(
    ctx: &SessionContext,
    table_name: &str,
    sidecar_base: &std::path::Path,
    precision: StoragePrecision,
    oversample: usize,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    if queries.is_empty() {
        return Ok(0.0);
    }
    // LOAD the frozen sidecar — never rebuild. The committed graph is the one
    // whose recall is being measured.
    let index = SidecarIndex::load(sidecar_base, &AnnIndexConfig::default(), precision)?;

    let mut total = 0.0;
    for query in queries {
        let exact = exact_vector_search(ctx, table_name, query, k).await?;
        let ann = if precision.needs_rescore() {
            crate::operator_mirror::retrieve_then_rescore(&index, query, k, oversample.max(1))?
        } else {
            index.search(query, k)?
        };
        total += recall_at_k_for_query(&ann, &exact, k);
    }
    Ok(total / queries.len() as f64)
}

/// Measure the held-out recall curve over a committed fixture bundle directory.
///
/// The directory holds three inputs: the corpus parquet
/// ([`HELD_OUT_CORPUS_FILE`]), the frozen sidecar bundle over that corpus
/// ([`HELD_OUT_SIDECAR_STEM`]`.usearch`/`.rowmap`/`.manifest.json`), and a
/// *separate* held-out query parquet ([`HELD_OUT_QUERY_FILE`]) whose `_row_id`s
/// are disjoint from the corpus by construction.
///
/// Unlike a corpus-as-query measurement, the query vectors are *not* projected
/// out of the corpus — they come from the separate query parquet, so no query is
/// its own nearest neighbour and recall@k measures how well the frozen ANN
/// recovers the exact neighbours of unseen points. For each k in [`RECALL_KS`]
/// this runs the
/// exact oracle over the corpus (ground truth) and the loaded (never rebuilt)
/// sidecar over the same corpus, querying both with the held-out vectors, and
/// reports the mean set-intersection recall@k.
///
/// This is the real recall-floor path: the committed fixture is a deterministic
/// subset of the 170k cache, and the cargo-test gate asserts each recall@k
/// clears a floor measured on this same slice. The absence of any input is
/// reported as an error rather than a faked number.
pub async fn recall_curve_held_out(
    fixture_dir: &Path,
) -> Result<BTreeMap<usize, Measurement>, Box<dyn std::error::Error>> {
    let corpus_path = fixture_dir.join(HELD_OUT_CORPUS_FILE);
    let query_path = fixture_dir.join(HELD_OUT_QUERY_FILE);
    let sidecar_base = fixture_dir.join(HELD_OUT_SIDECAR_STEM);

    let corpus_url = corpus::storage_url(&corpus_path)?;
    let ctx = corpus::register(&corpus_url, RECALL_TABLE).await?;

    // The query set is a SEPARATE embedding set, disjoint from the corpus — read
    // its vectors back through the same load path the corpus uses, registered
    // under its own table so it never collides with the corpus.
    let query_url = corpus::storage_url(&query_path)?;
    let query_ctx = corpus::register(&query_url, HELD_OUT_QUERY_TABLE).await?;
    let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, HELD_OUT_QUERY_TABLE)
        .await?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err(format!(
            "held-out recall fixture at {} has an empty query set — no queries to measure recall over",
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

    use jammi_db::storage::StorageUrl;
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

    /// Build a sidecar over `rows` and freeze it to `base` (`.usearch`/`.rowmap`/
    /// `.manifest`). This is the *one* build — it stands in for the on-box emit;
    /// the recall path under test only ever loads what this writes.
    fn freeze_sidecar(base: &std::path::Path, rows: &[(String, Vec<f32>)], dim: usize) {
        let mut index =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
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

        corpus::write_vectors(&corpus_path, &rows, dim)
            .await
            .unwrap();
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
        corpus::write_vectors(&corpus_path, &rows, dim)
            .await
            .unwrap();

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

    /// The held-out recall gate: load the committed fixture (a real-embedding
    /// subset of the 170k scale cache — corpus + sidecar frozen over it + a
    /// SEPARATE disjoint query set) and assert each recall@k clears the floor
    /// committed in `floor.json`.
    ///
    /// This is the hermetic engine proof that the *held-out* recall path works on
    /// real embeddings: the queries are not corpus rows, so no query finds itself
    /// and the recall is a genuine generalization measurement. The floor is the
    /// recall measured on this same slice minus a safety margin, so the gate has
    /// headroom against USearch-version or load-path drift without going vacuous.
    /// The FULL 168k held-out gate runs in the cookbook chapter over the Git-LFS
    /// cache this fixture is subset from; the engine repo carries no LFS, so this
    /// gate proves the floor on the small committed projection.
    #[tokio::test]
    async fn held_out_recall_clears_committed_floor() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("scale");

        let floor_json = std::fs::read_to_string(fixture_dir.join("floor.json"))
            .expect("committed floor.json must be present in the fixture bundle");
        let floor: serde_json::Value = serde_json::from_str(&floor_json).unwrap();

        let curve = recall_curve_held_out(&fixture_dir)
            .await
            .expect("held-out recall path over the committed fixture must run");

        for &k in &RECALL_KS {
            let measured = curve
                .get(&k)
                .and_then(|m| m.value)
                .unwrap_or_else(|| panic!("recall@{k} missing from measured curve"));
            let floor_k = floor["recall"][k.to_string()]["floor"]
                .as_f64()
                .unwrap_or_else(|| panic!("floor.json missing recall.{k}.floor"));
            assert!(
                measured >= floor_k,
                "held-out recall@{k} = {measured} fell below committed floor {floor_k}"
            );
        }
    }

    /// The file stem of the committed frozen `Int8` sidecar bundle — the SAME
    /// held-out fixture corpus [`held_out_recall_clears_committed_floor`]
    /// measures the `F32` bundle over, quantized. See `fixture.rs`'s
    /// `build_precision_recall_fixture`, the off-box builder that froze it and
    /// wrote the `floor.json` `"precision"` section this gate reads.
    const FROZEN_INT8_STEM: &str = "frozen_int8";

    /// The file stem of the committed frozen `Binary` sidecar bundle — the
    /// SAME held-out fixture corpus, sign-quantized. See `fixture.rs`'s
    /// `build_binary_recall_fixture`, the off-box builder that froze it and
    /// wrote the `binary_*` keys in `floor.json`'s `"precision"` section this
    /// gate reads.
    const FROZEN_BINARY_STEM: &str = "frozen_binary";

    /// The margin-clearing precision-recall recovery assertion shared by every
    /// quantized precision's gate: load the committed frozen bundle at
    /// `precision` (frozen once over the SAME fixture corpus the `F32` gate
    /// measures) and assert two things against `floor.json`'s `"precision"`
    /// section, keyed by `rescored_key`/`no_rescore_key`.
    ///
    /// 1. **Each variant clears its own committed floor** — the two-stage
    ///    retrieve→rescore at `oversample_rescored` and the naive
    ///    no-widening `oversample = 1` baseline both hold at or above
    ///    `measured − margin` on this same frozen bundle, the same
    ///    discipline [`held_out_recall_clears_committed_floor`] applies to
    ///    `F32`.
    /// 2. **The rescore recovery is real, not vacuous** — at every k the
    ///    rescored recall clears the no-rescore recall by at least
    ///    [`RESCORE_RECOVERY_MARGIN`], the measured proof that oversampling
    ///    the quantized graph's candidate pool and rescoring against the
    ///    exact `.rawf32` companion recovers neighbours the lossy quantized
    ///    graph alone would have missed (recall@k is order-blind — see
    ///    [`recall_at_k_for_query`] — so this margin cannot be satisfied by
    ///    re-ranking the SAME candidate set; the widened set must actually
    ///    contain the recovered ids).
    async fn assert_precision_recall_rescore_recovers_and_clears_floor(
        sidecar_stem: &str,
        precision: StoragePrecision,
        oversample_key: &str,
        rescored_key: &str,
        no_rescore_key: &str,
    ) {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("scale");
        let sidecar_base = fixture_dir.join(sidecar_stem);

        let floor_json = std::fs::read_to_string(fixture_dir.join("floor.json"))
            .expect("committed floor.json must be present in the fixture bundle");
        let floor: serde_json::Value = serde_json::from_str(&floor_json).unwrap();
        let oversample_rescored = floor["precision"][oversample_key]
            .as_u64()
            .unwrap_or_else(|| panic!("floor.json missing precision.{oversample_key}"))
            as usize;

        let corpus_path = fixture_dir.join("corpus_vectors.parquet");
        let corpus_url = corpus::storage_url(&corpus_path).unwrap();
        let corpus_table = format!("{sidecar_stem}_recall_corpus");
        let ctx = corpus::register(&corpus_url, &corpus_table).await.unwrap();

        let query_path = fixture_dir.join("query_vectors.parquet");
        let query_url = corpus::storage_url(&query_path).unwrap();
        let query_table = format!("{sidecar_stem}_recall_queries");
        let query_ctx = corpus::register(&query_url, &query_table).await.unwrap();
        let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, &query_table)
            .await
            .unwrap()
            .into_iter()
            .map(|(_, v)| v)
            .collect();

        for &k in &RECALL_KS {
            let rescored = mean_recall_at_k_rescored(
                &ctx,
                &corpus_table,
                &sidecar_base,
                precision,
                oversample_rescored,
                &queries,
                k,
            )
            .await
            .unwrap_or_else(|e| {
                panic!("rescored precision-recall path over the frozen {sidecar_stem} bundle must run: {e}")
            });
            let no_rescore = mean_recall_at_k_rescored(
                &ctx,
                &corpus_table,
                &sidecar_base,
                precision,
                1,
                &queries,
                k,
            )
            .await
            .unwrap_or_else(|e| {
                panic!("no-rescore precision-recall path over the frozen {sidecar_stem} bundle must run: {e}")
            });

            let rescored_floor = floor["precision"][rescored_key][k.to_string()]["floor"]
                .as_f64()
                .unwrap_or_else(|| panic!("floor.json missing precision.{rescored_key}.{k}.floor"));
            let no_rescore_floor = floor["precision"][no_rescore_key][k.to_string()]["floor"]
                .as_f64()
                .unwrap_or_else(|| {
                    panic!("floor.json missing precision.{no_rescore_key}.{k}.floor")
                });

            assert!(
                rescored >= rescored_floor,
                "{rescored_key} recall@{k} = {rescored} fell below committed floor {rescored_floor}"
            );
            assert!(
                no_rescore >= no_rescore_floor,
                "{no_rescore_key} recall@{k} = {no_rescore} fell below committed floor {no_rescore_floor}"
            );
            assert!(
                rescored - no_rescore >= RESCORE_RECOVERY_MARGIN,
                "rescore recovery at k={k} was only {} (rescored={rescored}, no_rescore={no_rescore}) \
                 — below the {RESCORE_RECOVERY_MARGIN} margin the retrieve→rescore design must clear",
                rescored - no_rescore
            );
        }
    }

    /// The `Int8` precision-recall recovery gate — see
    /// [`assert_precision_recall_rescore_recovers_and_clears_floor`].
    ///
    /// On this fixture the measured gap is large (recall@1: 0.98 rescored vs
    /// 0.71 without — an 0.27 recovery), so [`RESCORE_RECOVERY_MARGIN`] is set
    /// well below that to leave headroom for load-path/USearch-version drift
    /// while still having real teeth: a rescore path that silently degenerated
    /// into "return the quantized graph's own top-k" (the bug this gate exists
    /// to catch) would collapse the gap to ~0 and trip it.
    #[tokio::test]
    async fn precision_recall_rescore_recovers_and_clears_committed_floor() {
        assert_precision_recall_rescore_recovers_and_clears_floor(
            FROZEN_INT8_STEM,
            StoragePrecision::Int8,
            "oversample_rescored",
            "int8_rescored",
            "int8_no_rescore",
        )
        .await;
    }

    /// The `Binary` precision-recall recovery gate — see
    /// [`assert_precision_recall_rescore_recovers_and_clears_floor`].
    ///
    /// `Binary`'s Hamming coarse stage is far lossier per-candidate than
    /// `Int8`'s, so it is measured at its own much wider default oversample
    /// (`32` vs Int8's `4`, see
    /// [`jammi_db::config::StoragePrecision::default_oversample`]); the
    /// rescore recovery gap this measures is correspondingly larger than
    /// Int8's.
    #[tokio::test]
    async fn binary_precision_recall_rescore_recovers_and_clears_committed_floor() {
        assert_precision_recall_rescore_recovers_and_clears_floor(
            FROZEN_BINARY_STEM,
            StoragePrecision::Binary,
            "binary_oversample_rescored",
            "binary_rescored",
            "binary_no_rescore",
        )
        .await;
    }

    /// The minimum recall@k gap `{precision}_rescored − {precision}_no_rescore`
    /// must clear for the retrieve→rescore recovery to count as real, shared
    /// by every quantized precision's gate. Measured on the committed fixture
    /// the Int8 gap is 0.27/0.17/0.10 at k=1/10/100 (Binary's is wider still,
    /// its Hamming coarse stage being lossier) — this margin is set an order
    /// of magnitude below the smallest Int8 gap, so it has real teeth (a
    /// rescore that silently returned the quantized graph's own top-k,
    /// recovering nothing, collapses the gap to ~0 and trips it) while
    /// leaving generous headroom against load-path or USearch-version drift.
    const RESCORE_RECOVERY_MARGIN: f64 = 0.03;

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
