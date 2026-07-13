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
//! The hermetic cargo-test gate
//! (`tests::recall_floor_gates_clear_their_committed_floors`) loads a *small
//! committed fixture* — a deterministic sorted-`_row_id` subset of the real
//! 170k cache (real embeddings: corpus rows + held-out query rows, with a
//! sidecar frozen over the subset once) — and asserts the held-out recall@k
//! clears a committed floor measured on that same slice, for every precision
//! the fixture carries a frozen bundle for. This proves the held-out gate
//! works hermetically on real embeddings, inside `cargo test`, with no LFS
//! dependency.
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
//! quantized [`StoragePrecision`]: at `Int8`/`Binary` the loaded graph's own
//! vectors are lossy, so a search is the engine's two-stage retrieve→rescore —
//! an oversampled candidate pool off the quantized graph, exactly re-ranked
//! against the `.rawf32` rescore companion. Because recall@k is order-blind
//! (see above), `oversample == 1` measures the quantized graph's own naive
//! top-k (nothing for the rescore to recover), while the deployment's default
//! oversample measures how much of the quantization loss the rescore recovers.
//! Every quantized row in `tests::RECALL_GATE_TABLE` pairs a `primary`
//! (retrieve→rescore) variant against a `baseline` (`oversample = 1`) variant,
//! both loaded from a second frozen bundle (e.g. `frozen_int8`, built once
//! over the SAME fixture corpus by `fixture.rs`'s
//! `build_precision_recall_fixture`) — the gate asserts both that each variant
//! clears its own committed floor AND that the `primary` variant clears the
//! `baseline` by a real margin — the measured proof that
//! oversampling-then-rescoring recovers neighbours the lossy graph alone
//! misses, not just a floor two numbers happen to both clear.
//!
//! ## The binary gate is a confidence interval, not a point estimate
//!
//! A single-seed point recall on a small held-out query set can invert at
//! real scale — a lucky or unlucky draw of which queries happen to land in
//! this committed slice can pass or fail the gate independently of the
//! underlying index quality. [`recall_ci_at_k_rescored`] instead treats each
//! query's recall@k ([`recall_at_k_for_query`]) as one bootstrap sample: it
//! resamples the held-out query set with replacement (the engine's own
//! [`jammi_numerics::stats::bootstrap_ci`], seeded and deterministic — the
//! same percentile-bootstrap kernel `eval.rs`'s `eval_compare` significance CI
//! already uses) and reports the point mean alongside a 95% CI. The `Binary`
//! row in `tests::RECALL_GATE_TABLE` sets its `anchor` to
//! `tests::Anchor::CiLower` rather than `tests::Anchor::Point` — the ONLY
//! difference from `Int8`'s row — which is what makes
//! `tests::recall_floor_gates_clear_their_committed_floors` assert the CI's
//! LOWER bound rather than the point mean for that row, so a noisy
//! single-sample draw cannot pass (or fail) the gate on chance alone: the
//! gate only passes when the *worst plausible* mean over resamples of this
//! query set still clears the floor.

use std::collections::BTreeMap;
use std::path::Path;

use datafusion::prelude::SessionContext;

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_numerics::stats::{bootstrap_ci, Interval};

use crate::corpus;
use crate::report::{Measurement, RECALL_KS};

/// Bootstrap resamples [`recall_ci_at_k_rescored`] draws to build its
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
pub(crate) const RECALL_BOOTSTRAP_ITERATIONS: usize = 2000;

/// Two-tailed significance level for [`recall_ci_at_k_rescored`]'s CI — a 95%
/// interval, the same level `eval.rs`'s bootstrap significance CI uses.
pub(crate) const RECALL_BOOTSTRAP_ALPHA: f64 = 0.05;

/// Fixed seed for [`recall_ci_at_k_rescored`]'s bootstrap resampling.
///
/// The bootstrap is a function of the sample *multiset*
/// ([`jammi_numerics::stats::bootstrap_ci`] canonicalizes its resample basis
/// by sorting), so a fixed seed is sufficient for a reproducible interval —
/// the committed CI is deterministic across boxes and reruns, the same
/// determinism discipline every other committed number in this harness
/// carries.
pub(crate) const RECALL_BOOTSTRAP_SEED: u64 = 0xB17A_5EED;

/// A recall@k point estimate plus its bootstrap confidence interval — the
/// mean of [`recall_samples_at_k_rescored`]'s per-query samples, and the
/// `[lower, upper]` interval [`jammi_numerics::stats::bootstrap_ci`] derives
/// by resampling those same samples with replacement.
#[derive(Debug, Clone, Copy)]
pub struct RecallCi {
    /// The point recall@k — the plain mean over the query set, identical to
    /// what [`mean_recall_at_k_rescored`] returns for the same inputs.
    pub point: f64,
    /// The bootstrap confidence interval over the query-set mean.
    pub interval: Interval,
}

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
    let samples = recall_samples_at_k_rescored(
        ctx,
        table_name,
        sidecar_base,
        precision,
        oversample,
        queries,
        k,
    )
    .await?;
    if samples.is_empty() {
        return Ok(0.0);
    }
    Ok(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// Per-query recall@k samples over a query set, for a frozen sidecar bundle
/// loaded at `precision` — the SAME retrieve→rescore path
/// [`mean_recall_at_k_rescored`] averages, but returned as the individual
/// per-query fractions rather than collapsed to their mean. `mean_recall_at_k_rescored`
/// is exactly `samples.iter().sum() / samples.len()` over what this returns;
/// [`recall_ci_at_k_rescored`] is what these samples exist for — a bootstrap
/// CI resamples this exact multiset.
///
/// An empty `queries` yields an empty sample set, mirroring
/// [`mean_recall_at_k_rescored`]'s "no queries, nothing to measure" contract.
async fn recall_samples_at_k_rescored(
    ctx: &SessionContext,
    table_name: &str,
    sidecar_base: &std::path::Path,
    precision: StoragePrecision,
    oversample: usize,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    // LOAD the frozen sidecar — never rebuild. The committed graph is the one
    // whose recall is being measured.
    let index = SidecarIndex::load(sidecar_base, &AnnIndexConfig::default(), precision)?;

    let mut samples = Vec::with_capacity(queries.len());
    for query in queries {
        let exact = exact_vector_search(ctx, table_name, query, k).await?;
        let ann = if precision.needs_rescore() {
            crate::operator_mirror::retrieve_then_rescore(&index, query, k, oversample.max(1))?
        } else {
            index.search(query, k)?
        };
        samples.push(recall_at_k_for_query(&ann, &exact, k));
    }
    Ok(samples)
}

/// Recall@k point estimate AND bootstrap confidence interval over a query set,
/// for a frozen sidecar bundle loaded at `precision` — the statistically sound
/// alternative to [`mean_recall_at_k_rescored`]'s bare point estimate (see the
/// module-level "binary gate is a confidence interval" section).
///
/// Draws [`recall_samples_at_k_rescored`] (one sample per query — the SAME
/// retrieve→rescore path `mean_recall_at_k_rescored` averages) and bootstraps
/// their mean via the engine's own
/// [`jammi_numerics::stats::bootstrap_ci`]: [`RECALL_BOOTSTRAP_ITERATIONS`]
/// resamples of the query set (with replacement), under the fixed
/// [`RECALL_BOOTSTRAP_SEED`], at the [`RECALL_BOOTSTRAP_ALPHA`] two-tailed
/// level. `RecallCi::point` is the plain mean (identical to what
/// `mean_recall_at_k_rescored` returns for the same inputs); `RecallCi::interval`
/// is the `[2.5th, 97.5th]` percentile interval over that mean, resampled —
/// the width a caller should treat as "how much this point estimate could move
/// on a different draw of this same query set".
///
/// Errors — rather than reporting a vacuous interval — when `queries` is
/// empty: a CI over zero samples is not a measurement, it is a hidden 0.0/0.0
/// masquerading as a confidence interval.
pub async fn recall_ci_at_k_rescored(
    ctx: &SessionContext,
    table_name: &str,
    sidecar_base: &std::path::Path,
    precision: StoragePrecision,
    oversample: usize,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<RecallCi, Box<dyn std::error::Error>> {
    let samples = recall_samples_at_k_rescored(
        ctx,
        table_name,
        sidecar_base,
        precision,
        oversample,
        queries,
        k,
    )
    .await?;
    if samples.is_empty() {
        return Err(
            "recall_ci_at_k_rescored: empty query set — a bootstrap CI needs at least one sample"
                .into(),
        );
    }
    let point = samples.iter().sum::<f64>() / samples.len() as f64;
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let interval = bootstrap_ci(
        &samples,
        mean,
        RECALL_BOOTSTRAP_ITERATIONS,
        RECALL_BOOTSTRAP_ALPHA,
        RECALL_BOOTSTRAP_SEED,
    )?;
    Ok(RecallCi { point, interval })
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

    /// Where a [`RecallGateRow`]'s floor is checked against a measurement:
    /// the bare point mean, or the bootstrap-CI lower bound.
    ///
    /// This is the ONE bit of data that carries the CI-anchored discipline
    /// (see the module header's "binary gate is a confidence interval"
    /// section) on the `Binary` row: [`measure_variant`]'s `match anchor`
    /// is the only place this field changes behavior, and it changes ONLY
    /// which value is handed back for the floor check — a row's `precision`
    /// never enters that decision.
    #[derive(Debug, Clone, Copy)]
    enum Anchor {
        /// [`mean_recall_at_k_rescored`]'s bare mean.
        Point,
        /// [`recall_ci_at_k_rescored`]'s `interval.lower` — a noisy
        /// single-sample draw of the query set cannot pass on chance alone.
        CiLower,
    }

    /// Where a [`Variant`]'s query-time `oversample` comes from.
    #[derive(Debug, Clone, Copy)]
    enum Oversample {
        /// A literal — always `1` for a no-rescore baseline, and for `F32`
        /// (where the value is irrelevant: `StoragePrecision::needs_rescore`
        /// is `false`, so the rescore stage this parameter widens never
        /// runs).
        Fixed(usize),
        /// Read live from `floor.json`'s `precision.<key>` — the
        /// deployment's stamped default oversample for a quantized
        /// precision's retrieve→rescore stage.
        FromFloorKey(&'static str),
    }

    /// One measured variant of a [`RecallGateRow`]: the `oversample` its
    /// search runs at, and the `floor.json` path (object-key segments,
    /// before the per-k key) its floor lives under.
    struct Variant {
        oversample: Oversample,
        floor_path: &'static [&'static str],
    }

    /// One precision's recall-floor gate, expressed entirely as data: which
    /// frozen sidecar to load, which [`Anchor`] its floors are checked
    /// against, its always-measured `primary` variant, and an optional
    /// `baseline` variant.
    ///
    /// When `baseline` is `Some`, [`assert_recall_gate_row_clears_floor`]
    /// also asserts `primary` clears `baseline` by
    /// [`RESCORE_RECOVERY_MARGIN`] — the retrieve→rescore recovery proof.
    /// `F32` has no `baseline` (its single-stage `primary` measurement
    /// leaves nothing to recover from), so "recovery margin applicable" is
    /// never its own field — it is exactly `baseline.is_some()`.
    struct RecallGateRow {
        precision: StoragePrecision,
        sidecar_stem: &'static str,
        anchor: Anchor,
        primary: Variant,
        baseline: Option<Variant>,
    }

    /// The three precision-recall floor gates, as data: `F32` (single-stage,
    /// point-anchored, no recovery margin), `Int8` (retrieve→rescore,
    /// point-anchored), `Binary` (retrieve→rescore, CI-lower-anchored — the
    /// ONLY difference from `Int8`'s row is its `anchor`).
    const RECALL_GATE_TABLE: &[RecallGateRow] = &[
        RecallGateRow {
            precision: StoragePrecision::F32,
            sidecar_stem: HELD_OUT_SIDECAR_STEM,
            anchor: Anchor::Point,
            primary: Variant {
                oversample: Oversample::Fixed(1),
                floor_path: &["recall"],
            },
            baseline: None,
        },
        RecallGateRow {
            precision: StoragePrecision::Int8,
            sidecar_stem: FROZEN_INT8_STEM,
            anchor: Anchor::Point,
            primary: Variant {
                oversample: Oversample::FromFloorKey("oversample_rescored"),
                floor_path: &["precision", "int8_rescored"],
            },
            baseline: Some(Variant {
                oversample: Oversample::Fixed(1),
                floor_path: &["precision", "int8_no_rescore"],
            }),
        },
        RecallGateRow {
            precision: StoragePrecision::Binary,
            sidecar_stem: FROZEN_BINARY_STEM,
            anchor: Anchor::CiLower,
            primary: Variant {
                oversample: Oversample::FromFloorKey("binary_oversample_rescored"),
                floor_path: &["precision", "binary_rescored"],
            },
            baseline: Some(Variant {
                oversample: Oversample::Fixed(1),
                floor_path: &["precision", "binary_no_rescore"],
            }),
        },
    ];

    /// The file stem of the committed frozen `Int8` sidecar bundle — the
    /// SAME held-out fixture corpus [`HELD_OUT_SIDECAR_STEM`]'s `F32` bundle
    /// indexes, quantized. See `fixture.rs`'s `build_precision_recall_fixture`.
    const FROZEN_INT8_STEM: &str = "frozen_int8";

    /// The file stem of the committed frozen `Binary` sidecar bundle — the
    /// SAME held-out fixture corpus, sign-quantized. See `fixture.rs`'s
    /// `build_binary_recall_fixture`.
    const FROZEN_BINARY_STEM: &str = "frozen_binary";

    /// The minimum recall@k gap `primary − baseline` a [`RecallGateRow`]
    /// with a `baseline` must clear for the retrieve→rescore recovery to
    /// count as real. Measured on the committed fixture the Int8 gap is
    /// 0.27/0.17/0.10 at k=1/10/100 (Binary's is wider still, its Hamming
    /// coarse stage being lossier) — this margin is set an order of
    /// magnitude below the smallest Int8 gap, so it has real teeth (a
    /// rescore that silently returned the quantized graph's own top-k,
    /// recovering nothing, collapses the gap to ~0 and trips it) while
    /// leaving generous headroom against load-path or USearch-version drift.
    const RESCORE_RECOVERY_MARGIN: f64 = 0.03;

    /// Absolute path to the committed held-out recall fixture bundle
    /// (`fixtures/scale/` — corpus, held-out queries, and the frozen
    /// `F32`/`Int8`/`Binary` sidecar bundles), shared by every gate that
    /// reads it.
    fn scale_fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("scale")
    }

    /// Parse the committed `floor.json` out of a fixture bundle directory.
    fn load_floor_json(fixture_dir: &Path) -> serde_json::Value {
        let floor_json = std::fs::read_to_string(fixture_dir.join("floor.json"))
            .expect("committed floor.json must be present in the fixture bundle");
        serde_json::from_str(&floor_json).expect("floor.json must be valid JSON")
    }

    /// Register the fixture's corpus and held-out query tables under names
    /// prefixed by `table_prefix` (so concurrently-loaded rows never
    /// collide), and load the query vectors — the ONE fixture-loading path
    /// every [`RecallGateRow`] shares.
    async fn register_gate_fixture(
        fixture_dir: &Path,
        table_prefix: &str,
    ) -> (SessionContext, String, Vec<Vec<f32>>) {
        let corpus_path = fixture_dir.join(HELD_OUT_CORPUS_FILE);
        let corpus_url = corpus::storage_url(&corpus_path).unwrap();
        let corpus_table = format!("{table_prefix}_recall_corpus");
        let ctx = corpus::register(&corpus_url, &corpus_table).await.unwrap();

        let query_path = fixture_dir.join(HELD_OUT_QUERY_FILE);
        let query_url = corpus::storage_url(&query_path).unwrap();
        let query_table = format!("{table_prefix}_recall_queries");
        let query_ctx = corpus::register(&query_url, &query_table).await.unwrap();
        let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, &query_table)
            .await
            .unwrap()
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        (ctx, corpus_table, queries)
    }

    /// Read the floor at `path` (object-key segments) then `k` then
    /// `"floor"` out of a parsed `floor.json`.
    fn floor_at(floor: &serde_json::Value, path: &[&str], k: usize) -> f64 {
        let mut v = floor;
        for segment in path {
            v = &v[*segment];
        }
        v[k.to_string()]["floor"]
            .as_f64()
            .unwrap_or_else(|| panic!("floor.json missing {}.{k}.floor", path.join(".")))
    }

    /// Resolve a [`Variant`]'s `oversample` — either the literal, or a live
    /// read of `floor.json`'s stamped default.
    fn resolve_oversample(oversample: Oversample, floor: &serde_json::Value) -> usize {
        match oversample {
            Oversample::Fixed(n) => n,
            Oversample::FromFloorKey(key) => floor["precision"][key]
                .as_u64()
                .unwrap_or_else(|| panic!("floor.json missing precision.{key}"))
                as usize,
        }
    }

    /// The frozen bundle a [`Variant`] is measured over. Fixed for a whole
    /// [`RecallGateRow`] (`primary` and `baseline` search the SAME bundle) —
    /// bundled so [`measure_variant`] takes one "where to search" argument
    /// instead of four.
    struct SearchTarget<'a> {
        ctx: &'a SessionContext,
        table: &'a str,
        sidecar_base: &'a Path,
        precision: StoragePrecision,
    }

    /// Measure one [`Variant`] at `k`: the plain point mean (identical
    /// whichever `anchor` is used — only the returned "value a floor is
    /// checked against" differs), and that checked value — the SAME point
    /// mean at [`Anchor::Point`], or the bootstrap-CI lower bound at
    /// [`Anchor::CiLower`].
    async fn measure_variant(
        target: &SearchTarget<'_>,
        anchor: Anchor,
        oversample: usize,
        queries: &[Vec<f32>],
        k: usize,
    ) -> (f64, f64) {
        match anchor {
            Anchor::Point => {
                let point = mean_recall_at_k_rescored(
                    target.ctx,
                    target.table,
                    target.sidecar_base,
                    target.precision,
                    oversample,
                    queries,
                    k,
                )
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "point recall path over {} at k={k} must run: {e}",
                        target.sidecar_base.display()
                    )
                });
                (point, point)
            }
            Anchor::CiLower => {
                let ci = recall_ci_at_k_rescored(
                    target.ctx,
                    target.table,
                    target.sidecar_base,
                    target.precision,
                    oversample,
                    queries,
                    k,
                )
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "bootstrap-CI recall path over {} at k={k} must run: {e}",
                        target.sidecar_base.display()
                    )
                });
                (ci.point, ci.interval.lower)
            }
        }
    }

    /// Measure and assert one [`RecallGateRow`] over the committed fixture:
    /// `primary` clears its own floor at every k, and — when `baseline` is
    /// present — `baseline` clears its own floor AND `primary` clears
    /// `baseline` by [`RESCORE_RECOVERY_MARGIN`].
    ///
    /// This is the ONE parameterized assertion every precision's gate runs
    /// through: which value is compared against a floor (point vs CI lower)
    /// is driven purely by `row.anchor`, and whether the recovery-margin
    /// check runs at all is driven purely by whether `row.baseline` is
    /// `Some` — no branch here inspects `row.precision`.
    async fn assert_recall_gate_row_clears_floor(
        row: &RecallGateRow,
        fixture_dir: &Path,
        floor: &serde_json::Value,
    ) {
        let sidecar_base = fixture_dir.join(row.sidecar_stem);
        let (ctx, corpus_table, queries) =
            register_gate_fixture(fixture_dir, row.sidecar_stem).await;
        let target = SearchTarget {
            ctx: &ctx,
            table: &corpus_table,
            sidecar_base: &sidecar_base,
            precision: row.precision,
        };

        for &k in &RECALL_KS {
            let primary_oversample = resolve_oversample(row.primary.oversample, floor);
            let (primary_point, primary_anchor_value) =
                measure_variant(&target, row.anchor, primary_oversample, &queries, k).await;
            let primary_floor = floor_at(floor, row.primary.floor_path, k);
            assert!(
                primary_anchor_value >= primary_floor,
                "{:?} {} recall@{k} = {primary_anchor_value} fell below committed floor {primary_floor}",
                row.precision,
                row.primary.floor_path.join("."),
            );

            let Some(baseline) = &row.baseline else {
                continue;
            };
            let baseline_oversample = resolve_oversample(baseline.oversample, floor);
            let (baseline_point, baseline_anchor_value) =
                measure_variant(&target, row.anchor, baseline_oversample, &queries, k).await;
            let baseline_floor = floor_at(floor, baseline.floor_path, k);
            assert!(
                baseline_anchor_value >= baseline_floor,
                "{:?} {} recall@{k} = {baseline_anchor_value} fell below committed floor {baseline_floor}",
                row.precision,
                baseline.floor_path.join("."),
            );
            assert!(
                primary_point - baseline_point >= RESCORE_RECOVERY_MARGIN,
                "{:?} rescore recovery at k={k} was only {} (primary={primary_point}, baseline={baseline_point}) \
                 — below the {RESCORE_RECOVERY_MARGIN} margin the retrieve→rescore design must clear",
                row.precision,
                primary_point - baseline_point,
            );
        }
    }

    /// The unified held-out recall-floor gate: for every row in
    /// [`RECALL_GATE_TABLE`] (`F32` single-stage, `Int8` and `Binary`
    /// retrieve→rescore), measure over the committed fixture and assert it
    /// clears its committed floor, driven by one loop over explicit data
    /// rather than a hand-written function per precision.
    #[tokio::test]
    async fn recall_floor_gates_clear_their_committed_floors() {
        let fixture_dir = scale_fixture_dir();
        let floor = load_floor_json(&fixture_dir);
        for row in RECALL_GATE_TABLE {
            assert_recall_gate_row_clears_floor(row, &fixture_dir, &floor).await;
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
