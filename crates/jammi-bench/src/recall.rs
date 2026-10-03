//! The ANN-vs-exact recall mechanism: how the harness measures how well the
//! ANN index the engine builds for a table recovers the exact nearest
//! neighbours.
//!
//! ## The two retrievers
//!
//! * **Exact oracle** — the engine's [`exact_vector_search`], a brute-force scan
//!   over every corpus vector returning the `k` closest under a `(dist, _row_id)`
//!   total order. It is deterministic and exhaustive, so its top-`k` *is* ground
//!   truth: recall is measured against it, never the other way round.
//! * **Built ANN** — the table the engine under test builds over the committed
//!   corpus slice ([`Slice::build`]): each segment a [`SidecarIndex`] built over
//!   its rows in sorted `_row_id` order and saved as a bundle, then loaded back
//!   and searched through [`SegmentedIndex::search_final`] — the entry every
//!   table search goes through. A single graph is a table of one segment
//!   ([`SINGLE_GRAPH`]), not a separate path. A graph is a function of its
//!   rows, so the gate measures the graph this engine builds — never one an
//!   earlier engine left behind.
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
//!   build / load / oracle / intersect / average mechanism (see
//!   `ann_over_same_corpus_recovers_exact_neighbours`); it is *not* a meaningful
//!   quality floor, because a query finding itself says nothing about how the ANN
//!   handles unseen points.
//! * **Held-out queries** ([`recall_curve_held_out`]) — the queries come from a
//!   *separate* embedding set, disjoint from the indexed corpus by construction.
//!   No query is its own neighbour, so recall@k measures how well the ANN
//!   recovers the exact neighbours of unseen points — the quantity a deployed
//!   index is actually judged on. This is the path the `arxiv` subcommand drives
//!   and the path a real recall floor is asserted against.
//!
//! Both run the *same* primitive ([`Slice::recall`]); they differ only in where
//! the query vectors come from.
//!
//! ## What the engine gate proves vs. what the cookbook shows
//!
//! The hermetic cargo-test gate
//! (`tests::recall_floor_gates_clear_their_committed_floors`) builds over a
//! *small committed slice* — a deterministic sorted-`_row_id` subset of the
//! real 170k cache (real embeddings: corpus rows + held-out query rows) — and
//! asserts the held-out recall@k clears a committed floor measured on that
//! same slice (`fixture.rs`'s `measure_floors`), for every precision the gate
//! builds. This proves the held-out gate works hermetically on real
//! embeddings, inside `cargo test`, with no LFS dependency: the engine repo
//! carries no LFS, so the slice ships in the git object store.
//!
//! Recall on a whole published corpus is the cookbook's to show: its
//! ANN-recall chapter embeds the corpus and measures the engine's own search
//! against an exact scan, at whatever scale the reader runs.
//!
//! ## The precision axis: retrieve→rescore recovery
//!
//! At a quantized [`StoragePrecision`] (`Int8`/`Binary`) the loaded graph's
//! own vectors are lossy, so [`SegmentedIndex::search_final`] is the engine's
//! two-stage retrieve→rescore — an oversampled candidate pool off the
//! quantized graph, exactly re-ranked against the `.rawf32` rescore companion.
//! Because recall@k is order-blind (see above), `oversample == 1` measures the
//! quantized graph's own naive top-k (nothing for the rescore to recover),
//! while the deployment's default oversample measures how much of the
//! quantization loss the rescore recovers. Every quantized row in
//! `tests::RECALL_GATE_TABLE` pairs a `primary` (retrieve→rescore) variant
//! against a `baseline` (`oversample = 1`) variant, both searching the ONE
//! table built at that precision over the SAME slice — the gate asserts both
//! that each variant clears its own committed floor AND that the `primary`
//! variant clears the `baseline` by a real margin — the measured proof that
//! oversampling-then-rescoring recovers neighbours the lossy graph alone
//! misses, not just a floor two numbers happen to both clear.
//!
//! ## The binary gate is a confidence interval, not a point estimate
//!
//! A point recall on a small held-out query set can pass or fail on which
//! queries happen to land in this committed slice, independently of the
//! underlying index quality. [`Anchor::CiLower`] instead treats each query's
//! recall@k ([`recall_at_k_for_query`]) as one bootstrap sample: it resamples
//! the held-out query set with replacement (the engine's own
//! [`jammi_numerics::stats::bootstrap_ci`], seeded and deterministic — the
//! same percentile-bootstrap kernel `eval.rs`'s `eval_compare` significance CI
//! already uses) and checks the floor against the 95% CI's lower bound. The
//! `Binary` row in `tests::RECALL_GATE_TABLE` is anchored there rather than at
//! [`Anchor::Point`] — the ONLY difference from `Int8`'s row — so a noisy draw
//! of queries cannot pass (or fail) the gate on chance alone: the gate only
//! passes when the *worst plausible* mean over resamples of this query set
//! still clears the floor.
//!
//! ## The segment axis: does the merge recover what a lone graph would find
//!
//! The same measurement varies the table's *topology* as well as its storage
//! precision: instead of one segment over the whole corpus, `N` segments —
//! each built over a disjoint subset of the SAME corpus — are loaded and
//! assembled into one [`SegmentedIndex`], whose
//! [`SegmentedIndex::search_final`] fans the query across every segment and
//! merges the results (`jammi_db::index::segment`'s
//! `DEFAULT_SEGMENT_OVERFETCH_FACTOR` over-fetches from each segment before the
//! merge). One graph is one draw of HNSW's random levels, so a real recall
//! floor over this axis needs the draw varied, and the one lever this harness
//! can pull deterministically is not the HNSW level seed (USearch fixes it) but
//! the *partitioning*: which corpus rows land in which segment.
//! `tests::segment_merge_recall_clears_its_committed_floor_and_tracks_the_single_graph`
//! asserts the committed floor holds across every partitioning in
//! [`SEGMENT_PARTITIONINGS`] (more than one), each standing in for an
//! independent draw of the graph. Alongside the floor, that gate also measures
//! the SAME corpus as [`SINGLE_GRAPH`] at the SAME precision and asserts the
//! merge's recall does not fall more than a committed margin below it — the
//! merge-vs-single-graph tracking check that gives
//! `DEFAULT_SEGMENT_OVERFETCH_FACTOR` real teeth: an under-provisioned
//! over-fetch would surface here as the merge losing recall the single graph
//! recovers, exactly the failure mode the constant's SEAM note names.
//!
//! The committed floor is measured at `Int8` and `Binary`, never `F32`: at
//! `F32`, [`StoragePrecision::needs_rescore`] is `false`, so
//! [`SegmentedIndex::search_final`] never enters its retrieve→rescore stage —
//! there is no rescore-in-merge for the merge's over-fetch to feed, and this
//! fixture's real-embedding corpus is small enough per segment that an `F32`
//! HNSW segment's own recall already sits at (or above) `1.0`, so an `F32`
//! floor would guard nothing precision-specific (a near-tautological "recall
//! is always `1.0`" floor). `Int8`/`Binary` segments are lossy enough on
//! their own — see the committed single-graph
//! `int8_no_rescore`/`binary_no_rescore` numbers — that `search_final`'s
//! retrieve→rescore-in-merge design has real recall to recover, which is
//! exactly what this floor and the tracking margin measure.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::{SidecarBuilder, SidecarIndex};
use jammi_db::index::{validate_query, Admission, QuerySource};
use jammi_db::index::{SegmentId, SegmentedIndex, VectorIndex};
use jammi_numerics::stats::bootstrap_ci;

use crate::corpus;
use crate::report::{Measurement, RECALL_KS};

/// Bootstrap resamples an [`Anchor::CiLower`] measurement draws to build its
/// confidence interval.
///
/// Matches `eval.rs`'s `BOOTSTRAP_ITERATIONS` — the same percentile-bootstrap
/// kernel, the same order of iterations. At the committed fixture's 100
/// held-out queries, a percentile CI's Monte Carlo error falls off as
/// `1/sqrt(iterations)`: 2000 resamples put the 2.5th/97.5th percentile
/// estimates within a fraction of a percentage point of their converged
/// values (the interval bounds stop moving beyond noise well before 2000),
/// while keeping the hermetic gate fast — the loop is bounded by
/// `iterations * queries.len()`, a few hundred thousand array reads, not a
/// re-run of the search path.
const RECALL_BOOTSTRAP_ITERATIONS: usize = 2000;

/// Two-tailed significance level for an [`Anchor::CiLower`] measurement's CI —
/// a 95% interval, the same level `eval.rs`'s bootstrap significance CI uses.
const RECALL_BOOTSTRAP_ALPHA: f64 = 0.05;

/// Fixed seed for an [`Anchor::CiLower`] measurement's bootstrap resampling.
///
/// The bootstrap is a function of the sample *multiset*
/// ([`jammi_numerics::stats::bootstrap_ci`] canonicalizes its resample basis
/// by sorting), so a fixed seed is sufficient for a reproducible interval —
/// the same samples give the same interval on every box and rerun.
const RECALL_BOOTSTRAP_SEED: u64 = 0xB17A_5EED;

/// Which value of a recall@k measurement its floor is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Anchor {
    /// The mean recall over the query set.
    Point,
    /// The lower bound of the mean's 95% bootstrap confidence interval over
    /// the query set — for a measurement noisy enough at this query count that
    /// a point could pass or fail on the draw of queries alone.
    CiLower,
}

/// One recall@k measurement over a query set.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Recall {
    /// The mean of the per-query recalls.
    pub(crate) point: f64,
    /// The value the measurement's [`Anchor`] checks a floor against: the
    /// point itself, or its bootstrap CI's lower bound.
    pub(crate) anchored: f64,
}

/// File names of the committed *held-out* recall slice, relative to its
/// directory: the corpus every table is built over and the oracle scans, and a
/// *separate* query parquet whose rows are disjoint from the corpus. The
/// disjointness is what makes the recall a generalization measurement rather
/// than a query-by-example one. Named once for the slice carver (which writes
/// them) and the gate (which reads them).
pub(crate) const HELD_OUT_CORPUS_FILE: &str = "corpus_vectors.parquet";
pub(crate) const HELD_OUT_QUERY_FILE: &str = "query_vectors.parquet";

/// How many segments each of [`SEGMENT_PARTITIONINGS`] splits the slice into.
/// `2` is the smallest split that exercises the cross-segment merge at all.
pub(crate) const SEGMENT_COUNT: usize = 2;

/// A layout of the slice's rows into a table's segments — the deterministic
/// axis the segment gate varies in place of the HNSW level seed USearch does
/// not let it pin (see the module's "segment axis" section).
pub(crate) struct Partitioning {
    /// The partitioning's name: its `floor.json` key and its bundles' stem.
    pub(crate) name: &'static str,
    /// How many segments the table has.
    pub(crate) segments: usize,
    /// `(row, rows, segments)` → the segment slice row `row` of `rows` (0-based,
    /// in sorted `_row_id` order) lands in.
    assign: fn(usize, usize, usize) -> usize,
}

/// Even-width contiguous blocks: row `i` of `rows` lands in segment
/// `i * segments / rows`. Mirrors how an append-only table's segments actually
/// accumulate — each batch of newly-added rows is contiguous in insertion
/// order.
fn contiguous_blocks(i: usize, rows: usize, segments: usize) -> usize {
    ((i * segments) / rows.max(1)).min(segments - 1)
}

/// Round-robin: row `i` lands in segment `i % segments`. A structurally
/// different assignment from [`contiguous_blocks`] over the SAME rows — each
/// row's true nearest neighbours are scattered across segments differently, so
/// the two partitionings are independent draws of "which segment a row's
/// neighbours end up in", not a relabeling of the same split.
fn round_robin(i: usize, _rows: usize, segments: usize) -> usize {
    i % segments
}

/// The whole slice as one segment: the table a freshly built embedding table
/// is, and the baseline the segment merge is tracked against.
pub(crate) const SINGLE_GRAPH: Partitioning = Partitioning {
    name: "single_graph",
    segments: 1,
    assign: contiguous_blocks,
};

/// The partitionings the segment floor is measured and gated over — at least
/// two, so the floor holds across more than one draw of the graph.
pub(crate) const SEGMENT_PARTITIONINGS: &[Partitioning] = &[
    Partitioning {
        name: "seg_block",
        segments: SEGMENT_COUNT,
        assign: contiguous_blocks,
    },
    Partitioning {
        name: "seg_interleaved",
        segments: SEGMENT_COUNT,
        assign: round_robin,
    },
];

/// A table's segments, saved as bundles, and the precision they were built
/// at — the precision they are read back at.
pub(crate) struct SegmentBundles {
    precision: StoragePrecision,
    bases: Vec<PathBuf>,
}

/// The deepest k [`Slice::recall`] measures: the last of [`RECALL_KS`], which
/// ascends.
const TRUTH_DEPTH: usize = RECALL_KS[RECALL_KS.len() - 1];

/// A corpus and a query set, loaded for recall measurement: the corpus rows in
/// sorted `_row_id` order — the order every table over them is built in — the
/// query vectors, and each query's exact top-[`TRUTH_DEPTH`] neighbours.
///
/// The exact neighbours are a property of the corpus and the query alone, never
/// of the table under measurement, so they are computed once at load: the
/// oracle's `(dist, _row_id)` order is total, so a query's exact top-`k` is the
/// first `k` of its top-[`TRUTH_DEPTH`].
pub(crate) struct Slice {
    pub(crate) rows: Vec<(String, Vec<f32>)>,
    pub(crate) dim: usize,
    pub(crate) queries: Vec<Vec<f32>>,
    pub(crate) truth: Vec<Vec<(String, f32)>>,
}

/// Load the committed slice under `fixture_dir` ([`Slice::load`]).
pub(crate) async fn load_slice(
    fixture_dir: &Path,
    prefix: &str,
) -> Result<Slice, Box<dyn std::error::Error>> {
    Slice::load(
        &fixture_dir.join(HELD_OUT_CORPUS_FILE),
        &fixture_dir.join(HELD_OUT_QUERY_FILE),
        prefix,
    )
    .await
}

impl Slice {
    /// Load the corpus parquet at `corpus_path` and the query parquet at
    /// `query_path`, registering them as `{prefix}_corpus` and
    /// `{prefix}_queries` so slices loaded side by side never collide, and run
    /// the exact oracle once per query.
    pub(crate) async fn load(
        corpus_path: &Path,
        query_path: &Path,
        prefix: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let table = format!("{prefix}_corpus");
        let ctx = corpus::register(&corpus::storage_url(corpus_path)?, &table).await?;
        let mut rows = corpus::load_vectors(&ctx, &table).await?;
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let dim = rows
            .first()
            .map(|(_, v)| v.len())
            .ok_or_else(|| format!("the corpus {} is empty", corpus_path.display()))?;

        let query_table = format!("{prefix}_queries");
        let query_ctx = corpus::register(&corpus::storage_url(query_path)?, &query_table).await?;
        let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, &query_table)
            .await?
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        if queries.is_empty() {
            return Err(format!(
                "the query set {} is empty — no queries to measure recall over",
                query_path.display()
            )
            .into());
        }

        let mut truth = Vec::with_capacity(queries.len());
        for query in &queries {
            let query = validate_query(query.to_vec(), dim, QuerySource::Caller)?;
            truth.push(
                exact_vector_search(&ctx, &table, &query, TRUTH_DEPTH, None, &Admission::Every)
                    .await?,
            );
        }
        Ok(Self {
            rows,
            dim,
            queries,
            truth,
        })
    }

    /// Build the table `part` lays the slice's rows out as, at `precision`:
    /// each segment a sidecar built over its rows in order and saved as a
    /// bundle under `dir` — the build and save a table's segment goes through.
    pub(crate) fn build(
        &self,
        dir: &Path,
        precision: StoragePrecision,
        part: &Partitioning,
    ) -> Result<SegmentBundles, Box<dyn std::error::Error>> {
        let mut builders = (0..part.segments)
            .map(|_| SidecarBuilder::new(self.dim, &AnnIndexConfig::default(), precision))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, (id, v)) in self.rows.iter().enumerate() {
            builders[(part.assign)(i, self.rows.len(), part.segments)].add(id, v)?;
        }
        let bases = builders
            .into_iter()
            .enumerate()
            .map(|(idx, builder)| {
                if builder.is_empty() {
                    return Err(format!(
                        "partitioning {} left segment {idx} empty — a segment holds at least one row",
                        part.name
                    )
                    .into());
                }
                let base = dir.join(format!("{}_{precision:?}_seg{idx}", part.name));
                VectorIndex::save(&builder.build()?, &base)?;
                Ok(base)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok(SegmentBundles { precision, bases })
    }

    /// The table's recall@k over the slice's queries: each query's top-`k`
    /// from the table `bundles` hold — loaded and searched as a table is
    /// served, through [`SegmentedIndex::search_final`] at `oversample` —
    /// intersected with the exact oracle's top-`k` over the slice corpus
    /// ([`recall_at_k_for_query`]), then averaged, and anchored by `anchor`.
    ///
    /// Errors when `k` exceeds [`TRUTH_DEPTH`], the depth the exact
    /// neighbours were computed to.
    pub(crate) fn recall(
        &self,
        bundles: &SegmentBundles,
        oversample: usize,
        k: usize,
        anchor: Anchor,
    ) -> Result<Recall, Box<dyn std::error::Error>> {
        if k > TRUTH_DEPTH {
            return Err(format!(
                "recall@{k} is deeper than the {TRUTH_DEPTH} exact neighbours the slice holds"
            )
            .into());
        }
        let samples = self.recall_samples(bundles, oversample, k)?;
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
        let point = mean(&samples);
        let anchored = match anchor {
            Anchor::Point => point,
            Anchor::CiLower => {
                bootstrap_ci(
                    &samples,
                    mean,
                    RECALL_BOOTSTRAP_ITERATIONS,
                    RECALL_BOOTSTRAP_ALPHA,
                    RECALL_BOOTSTRAP_SEED,
                )?
                .lower
            }
        };
        Ok(Recall { point, anchored })
    }

    /// [`Self::recall`] at every k in [`RECALL_KS`], keyed by k.
    pub(crate) fn recall_curve(
        &self,
        bundles: &SegmentBundles,
        oversample: usize,
        anchor: Anchor,
    ) -> Result<BTreeMap<usize, Recall>, Box<dyn std::error::Error>> {
        let mut curve = BTreeMap::new();
        for &k in &RECALL_KS {
            curve.insert(k, self.recall(bundles, oversample, k, anchor)?);
        }
        Ok(curve)
    }

    /// One recall@k sample per query, in query order — what
    /// [`Self::recall`] averages and resamples.
    fn recall_samples(
        &self,
        bundles: &SegmentBundles,
        oversample: usize,
        k: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let segments = bundles
            .bases
            .iter()
            .enumerate()
            .map(|(i, base)| {
                let index =
                    SidecarIndex::load(base, &AnnIndexConfig::default(), bundles.precision)?;
                Ok::<_, Box<dyn std::error::Error>>((SegmentId(i as i64), index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let table = SegmentedIndex::new(segments)?;
        // The table is the artifact under measurement — its own width is the
        // authority every query is checked against.
        let dim = table.dimensions();

        self.queries
            .iter()
            .zip(&self.truth)
            .map(|(query, truth)| {
                let query = validate_query(query.to_vec(), dim, QuerySource::Caller)?;
                let ann = table.search_final(&query, k, oversample, &Admission::Every)?;
                Ok(recall_at_k_for_query(&ann, truth, k))
            })
            .collect()
    }
}

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

/// Measure the held-out `F32` recall curve over the committed slice under
/// `fixture_dir`.
///
/// Builds the slice's `F32` [`SINGLE_GRAPH`] and queries it with the slice's
/// *separate* held-out query vectors, whose `_row_id`s are disjoint from the
/// corpus by construction — so no query is its own nearest neighbour and
/// recall@k measures how well the ANN recovers the exact neighbours of unseen
/// points. For each k in [`RECALL_KS`] this runs the exact oracle over the
/// corpus (ground truth) and the table, and reports the mean set-intersection
/// recall@k.
///
/// A missing input is an error, never a faked number.
pub async fn recall_curve_held_out(
    fixture_dir: &Path,
) -> Result<BTreeMap<usize, Measurement>, Box<dyn std::error::Error>> {
    let slice = load_slice(fixture_dir, "recall").await?;
    let work = tempfile::tempdir()?;
    let table = slice.build(work.path(), StoragePrecision::F32, &SINGLE_GRAPH)?;
    Ok(slice
        .recall_curve(&table, 1, Anchor::Point)?
        .into_iter()
        .map(|(k, recall)| (k, Measurement::measured(recall.point, "fraction")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    use jammi_db::storage::StorageUrl;
    use tempfile::tempdir;

    use crate::corpus;
    use crate::fixture::{read_floor_json, FloorEntry, FloorRecord, QUANTIZED_PRECISIONS};

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

    /// The recall computation is correct: a table built over the *same*
    /// vectors the exact oracle scores recovers the exact neighbours, so
    /// recall@k == 1.0 for every k. This proves the load / oracle / build /
    /// save / load / search / set-intersect / average path end to end.
    /// Synthetic vectors prove the mechanism; the real-embedding recall floor
    /// is the committed-slice gate's.
    #[tokio::test]
    async fn ann_over_same_corpus_recovers_exact_neighbours() {
        let dim = 8;
        let n = 64;
        let rows = tiny_rows(n, dim);
        // Queries are exact corpus rows, so the exact top-1 of each is itself —
        // a hand-checkable oracle. Use a handful spread across the corpus.
        let queries: Vec<(String, Vec<f32>)> = [0usize, 7, 31, 63]
            .iter()
            .map(|&i| rows[i].clone())
            .collect();

        let dir = tempdir().unwrap();
        let corpus_path = dir.path().join("tiny.parquet");
        let query_path = dir.path().join("tiny_queries.parquet");
        corpus::write_vectors(&corpus_path, &rows, dim)
            .await
            .unwrap();
        corpus::write_vectors(&query_path, &queries, dim)
            .await
            .unwrap();
        let slice = Slice::load(&corpus_path, &query_path, "tiny")
            .await
            .unwrap();
        let built = slice
            .build(dir.path(), StoragePrecision::F32, &SINGLE_GRAPH)
            .unwrap();

        // On the same corpus, exact and HNSW agree at this scale: recall is 1.0.
        for k in [1usize, 10] {
            let recall = slice.recall(&built, 1, k, Anchor::Point).unwrap();
            assert_eq!(
                recall.point, 1.0,
                "ANN over the same corpus must recover the exact top-{k}"
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
        let query = validate_query(query.to_vec(), dim, QuerySource::Caller).unwrap();
        let top = exact_vector_search(&ctx, table, &query, 3, None, &Admission::Every)
            .await
            .unwrap();
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

    /// The minimum recall@k gap `rescored − no_rescore` a quantized
    /// precision's single graph must clear for the retrieve→rescore recovery
    /// to count as real. Measured on the committed slice the Int8 gap is
    /// 0.27/0.17/0.10 at k=1/10/100 (Binary's is wider still, its Hamming
    /// coarse stage being lossier) — this margin is set an order of
    /// magnitude below the smallest Int8 gap, so it has real teeth (a
    /// rescore that silently returned the quantized graph's own top-k,
    /// recovering nothing, collapses the gap to ~0 and trips it) while
    /// leaving generous headroom against platform or USearch-version drift.
    const RESCORE_RECOVERY_MARGIN: f64 = 0.03;

    /// Absolute path to the committed held-out recall slice
    /// (`fixtures/scale/` — corpus, held-out queries and `floor.json`), shared
    /// by every gate that reads it.
    fn scale_fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("scale")
    }

    /// Assert `recall`'s anchored value clears `floors`' floor at `k`.
    fn assert_clears(label: &str, k: usize, recall: Recall, floors: &BTreeMap<usize, FloorEntry>) {
        let floor = floors
            .get(&k)
            .unwrap_or_else(|| panic!("floor.json has no {label} floor at k={k}"))
            .floor;
        assert!(
            recall.anchored >= floor,
            "{label} recall@{k} = {} fell below committed floor {floor}",
            recall.anchored,
        );
    }

    /// The committed floors cover exactly [`QUANTIZED_PRECISIONS`], in order —
    /// so no precision's gate passes by being absent from the record.
    fn assert_covers_quantized_precisions(label: &str, committed: &[StoragePrecision]) {
        let expected: Vec<StoragePrecision> = QUANTIZED_PRECISIONS
            .iter()
            .map(|spec| spec.precision)
            .collect();
        assert_eq!(
            committed, expected,
            "floor.json's {label} floors must cover every quantized precision"
        );
    }

    /// The held-out recall-floor gate over each single graph: the `F32` graph
    /// clears its floors, and each quantized graph clears its `rescored` and
    /// `no_rescore` floors with `rescored` clearing `no_rescore` by
    /// [`RESCORE_RECOVERY_MARGIN`] — the retrieve→rescore recovery proof.
    /// Every table is built from the slice by the engine under test.
    #[tokio::test]
    async fn recall_floor_gates_clear_their_committed_floors() {
        let fixture_dir = scale_fixture_dir();
        let floors: FloorRecord = read_floor_json(&fixture_dir).unwrap();
        let slice = load_slice(&fixture_dir, "gate").await.unwrap();
        let work = tempdir().unwrap();

        let f32_graph = slice
            .build(work.path(), StoragePrecision::F32, &SINGLE_GRAPH)
            .unwrap();
        for (k, recall) in slice.recall_curve(&f32_graph, 1, Anchor::Point).unwrap() {
            assert_clears("F32", k, recall, &floors.recall);
        }

        assert_covers_quantized_precisions(
            "precision",
            &floors
                .precision
                .iter()
                .map(|q| q.precision)
                .collect::<Vec<_>>(),
        );
        for q in &floors.precision {
            let graph = slice
                .build(work.path(), q.precision, &SINGLE_GRAPH)
                .unwrap();
            let rescored = slice.recall_curve(&graph, q.oversample, q.anchor).unwrap();
            let no_rescore = slice.recall_curve(&graph, 1, q.anchor).unwrap();
            for &k in &RECALL_KS {
                let label = format!("{:?}", q.precision);
                assert_clears(&format!("{label} rescored"), k, rescored[&k], &q.rescored);
                assert_clears(
                    &format!("{label} no-rescore"),
                    k,
                    no_rescore[&k],
                    &q.no_rescore,
                );
                let recovered = rescored[&k].point - no_rescore[&k].point;
                assert!(
                    recovered >= RESCORE_RECOVERY_MARGIN,
                    "{label} rescore recovery at k={k} was only {recovered} (rescored={}, \
                     no_rescore={}) — below the {RESCORE_RECOVERY_MARGIN} margin the \
                     retrieve→rescore design must clear",
                    rescored[&k].point,
                    no_rescore[&k].point,
                );
            }
        }
    }

    /// The segment-merge recall floor gate: for every quantized precision
    /// (where [`StoragePrecision::needs_rescore`] is `true`, so `search_final`'s
    /// retrieve→rescore-in-merge stage actually runs; see the module-level
    /// "segment axis" section for why `F32` is not gated here) and every
    /// partitioning in [`SEGMENT_PARTITIONINGS`] (each a deterministic
    /// stand-in for an independent HNSW build seed USearch does not let this
    /// harness pin), builds the partitioning's table and asserts its recall
    /// clears the committed floor at every k in [`RECALL_KS`]. It ALSO builds
    /// the SAME corpus as [`SINGLE_GRAPH`] at that SAME precision and asserts
    /// the merge never falls more than the committed
    /// `single_graph_tracking_margin` below it — the check that gives
    /// `jammi_db::index::segment::DEFAULT_SEGMENT_OVERFETCH_FACTOR` real
    /// teeth over a precision where per-segment recall is genuinely below
    /// `1.0`.
    #[tokio::test]
    async fn segment_merge_recall_clears_its_committed_floor_and_tracks_the_single_graph() {
        let fixture_dir = scale_fixture_dir();
        let floors: FloorRecord = read_floor_json(&fixture_dir).unwrap();
        let segment_merge = &floors.segment_merge;
        assert_eq!(segment_merge.segment_count, SEGMENT_COUNT);
        assert_covers_quantized_precisions(
            "segment_merge",
            &segment_merge
                .precisions
                .iter()
                .map(|s| s.precision)
                .collect::<Vec<_>>(),
        );

        let slice = load_slice(&fixture_dir, "segment_merge").await.unwrap();
        let work = tempdir().unwrap();

        for spec in &segment_merge.precisions {
            let label = format!("{:?}", spec.precision);
            // The single graph the merge is tracked against, built and measured
            // here by the same engine as the segments.
            let single = slice
                .build(work.path(), spec.precision, &SINGLE_GRAPH)
                .unwrap();
            let single_graph = slice
                .recall_curve(&single, spec.oversample, Anchor::Point)
                .unwrap();

            for part in SEGMENT_PARTITIONINGS {
                let floors_k = spec.partitionings.get(part.name).unwrap_or_else(|| {
                    panic!(
                        "floor.json has no {label} floors for partitioning {}",
                        part.name
                    )
                });
                let segmented = slice.build(work.path(), spec.precision, part).unwrap();
                let merged = slice
                    .recall_curve(&segmented, spec.oversample, spec.anchor)
                    .unwrap();
                for &k in &RECALL_KS {
                    let merged_k = merged[&k];
                    assert_clears(&format!("{label} {}", part.name), k, merged_k, floors_k);
                    let single_k = single_graph[&k].point;
                    assert!(
                        single_k - merged_k.point <= spec.single_graph_tracking_margin,
                        "{label} {} recall@{k} = {} fell more than {} below the single-graph \
                         baseline {single_k} — the retrieve→rescore-in-merge design did not \
                         recover the per-segment loss",
                        part.name,
                        merged_k.point,
                        spec.single_graph_tracking_margin,
                    );
                }
            }
        }
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
