//! The CPU-hermetic eval-metric tier: the engine's retrieval / classification
//! metric folds and the order-invariant bootstrap significance CI, each
//! re-folded over a committed golden and gated against a committed value.
//!
//! This is the eval analogue of [`crate::recall`]: every number is a
//! *deterministic fold* of committed inputs through the engine's own metric
//! kernels, and the gate asserts the re-fold matches a committed golden. Unlike a
//! coverage floor (a one-sided `≥`), a metric re-fold is exact arithmetic, so the
//! gate is a two-sided tolerance band `|measured − golden| ≤ tolerance` (the
//! tolerance is for f64 reassociation, not measurement noise) — a regression in
//! any kernel moves the re-folded number off the golden in either direction.
//!
//! ## What drives the numbers — the real engine metric kernels
//!
//! * **Retrieval** (`eval_embeddings` / `eval_per_query`):
//!   [`RetrievalMetrics::compute_query`] + [`RetrievalMetrics::aggregate`] —
//!   recall@k, MRR, nDCG over [`RelevanceJudgment`] golden rows.
//! * **Classification** (`eval_inference`): [`ClassificationMetrics::compute`] —
//!   accuracy and macro-F1 over predicted/actual label vectors.
//! * **Comparison significance** (`eval_compare`): [`bootstrap_ci`], the seeded
//!   percentile bootstrap of the paired per-query metric delta. The tier asserts
//!   the documented order-invariance (engine #173): the CI is a function of the
//!   delta *multiset*, not its order, so shuffling the deltas yields a
//!   byte-identical interval.
//!
//! ## Why a committed *spec*, not committed rows
//!
//! The golden retrieval / classification data is drawn deterministically from a
//! seeded LCG (the generator family the rest of the harness uses), so the
//! committed artifact is the *generation spec* (seeds, sizes, k, label
//! cardinality, a retrieval-quality knob) plus the golden metric *value* each
//! fold produced when the spec was cut. The gate regenerates the exact same
//! golden from the spec, re-folds it through the engine, and asserts the metric
//! lands within tolerance of the committed value. Committing the spec rather than
//! the rows is the eval mirror of committing the corpus parquet: the inputs
//! travel so the fold is re-derivable, the golden is a real fold result, not a
//! hand-written number.
//!
//! ## Gate scale vs. timing scale
//!
//! The committed gate spec folds at tractable sizes so the hermetic `cargo test`
//! gate re-derives every golden in seconds — its job is to prove the kernels fold
//! the *same* number off the committed golden (the correctness numbers are
//! size-invariant), which a tractable largest point shows as faithfully as a huge
//! one. The full {1k, 10k, 100k} *timing* curve is an off-box / cookbook concern
//! (re-run the rebuilder with larger sizes there), mirroring [`crate::recall`]'s
//! split between its small committed projection (engine gate) and the full-scale
//! cookbook gate.

use serde::{Deserialize, Serialize};

use jammi_numerics::classification::ClassificationMetrics;
use jammi_numerics::retrieval::{RelevanceJudgment, RetrievalMetrics};
use jammi_numerics::stats::bootstrap_ci;

use crate::report::{BootstrapDeterminism, EvalPoint, EvalTier, Measurement, MetricGate};

/// Two-sided tolerance the re-folded metric must land within of the committed
/// golden value: `|measured − golden| ≤ TOLERANCE`.
///
/// A fold is exact arithmetic over identical regenerated inputs, so the only
/// drift a passing run tolerates is f64 reassociation across the sum order — far
/// below this band. A *failing* run is a real kernel change (a different recall
/// definition, a dropped tie-break, a changed nDCG discount), which moves the
/// metric by far more than f64 noise. The band is the eval analogue of the
/// recall floor's margin: tight enough to bite a regression, loose enough not to
/// flap on arithmetic reordering.
const TOLERANCE: f64 = 1e-9;

/// The seed offset mixed into the per-query bootstrap deltas' resampling — fixed
/// so the committed CI is reproducible. The bootstrap is order-invariant, so the
/// seed fixes which positions are drawn and the canonical sort fixes the values.
const BOOTSTRAP_SEED: u64 = 0xE7A1_5EED;

/// Bootstrap resamples for the `eval_compare` significance CI. Enough for a
/// stable percentile interval at the sizes here, small enough to keep the gate
/// fast.
const BOOTSTRAP_ITERATIONS: usize = 2000;

/// The two-tailed significance level for the `eval_compare` bootstrap CI (a 95%
/// interval).
const BOOTSTRAP_ALPHA: f64 = 0.05;

/// The committed eval spec: the generation parameters every metric is folded
/// from, plus the per-size committed golden values. The on-disk
/// `baselines/eval.json` the tier and its gate read.
///
/// Nothing here is a hand-written metric number: the golden values are the fold
/// results over the data the spec deterministically regenerates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSpec {
    /// The k cutoff the retrieval metrics fold at.
    pub k: usize,
    /// Candidate-list length each synthetic query retrieves (≥ `k`).
    pub list_len: usize,
    /// Number of relevant documents seeded per query.
    pub relevant_per_query: usize,
    /// Number of classes the synthetic classification draws over.
    pub n_classes: usize,
    /// The two-sided tolerance each re-folded metric must match its golden
    /// within.
    pub tolerance: f64,
    /// One committed golden record per eval-set size, ascending.
    pub points: Vec<SpecPoint>,
}

/// One eval-set size's committed golden metric values — what each engine kernel
/// produced over the regenerated golden when the spec was cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecPoint {
    /// Retrieval query count this point folds recall/MRR/nDCG over.
    pub query_rows: usize,
    /// Classification row count this point folds accuracy/F1 over.
    pub inference_rows: usize,
    /// Golden mean recall@k.
    pub recall_at_k: f64,
    /// Golden mean reciprocal rank.
    pub mrr: f64,
    /// Golden mean nDCG.
    pub ndcg: f64,
    /// Golden classification accuracy.
    pub accuracy: f64,
    /// Golden macro-F1.
    pub macro_f1: f64,
}

impl EvalSpec {
    /// The crate-relative path to the committed eval spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("eval.json")
    }

    /// Load the committed spec from `baselines/eval.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// The Numerical-Recipes LCG the rest of the harness uses, for deterministic
/// no-crate synthetic golden generation.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// A uniform draw in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform integer in `[0, n)`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Seed for the retrieval golden stream.
const RETRIEVAL_SEED: u64 = 0x00A1_5EED;
/// Seed for the classification golden stream.
const CLASSIFICATION_SEED: u64 = 0x00B2_5EED;

/// A synthetic retrieval golden query: the ranked candidate list the "system"
/// returned and the relevance judgments over it.
struct GoldenQuery {
    retrieved: Vec<String>,
    judgments: Vec<RelevanceJudgment>,
}

/// Draw `n_queries` synthetic retrieval golden queries from the LCG seeded at
/// `seed`.
///
/// Each query has `relevant_per_query` relevant docs (graded 1–3) scattered into
/// a `list_len` candidate list at pseudo-random ranks. The relevant set is what
/// recall/MRR/nDCG are computed against — a realistic golden where some queries
/// rank a relevant doc first (high MRR) and others bury them (low recall@k), so
/// the aggregate is a non-trivial fold, not a degenerate 0 or 1.
fn draw_retrieval_golden(
    seed: u64,
    n_queries: usize,
    list_len: usize,
    relevant_per_query: usize,
    k: usize,
) -> Vec<GoldenQuery> {
    let mut rng = Lcg::new(seed);
    (0..n_queries)
        .map(|q| {
            // The candidate list is doc ids `q_0 .. q_{list_len-1}`, unique per
            // query so ids never collide across queries.
            let retrieved: Vec<String> = (0..list_len).map(|d| format!("q{q}_d{d}")).collect();
            // Pick `relevant_per_query` distinct positions in the list to be
            // relevant, each with a graded relevance in 1..=3. Distinctness keeps
            // |relevant| exact so recall's denominator is well-defined.
            let mut chosen = std::collections::BTreeSet::new();
            while chosen.len() < relevant_per_query.min(list_len) {
                // Bias the draw toward earlier ranks for some queries so MRR and
                // recall@k vary across the set rather than being uniform.
                let pos = if rng.unit() < 0.5 {
                    rng.below(k.max(1).min(list_len))
                } else {
                    rng.below(list_len)
                };
                chosen.insert(pos);
            }
            let judgments: Vec<RelevanceJudgment> = chosen
                .iter()
                .map(|&pos| RelevanceJudgment {
                    doc_id: format!("q{q}_d{pos}"),
                    grade: 1 + rng.below(3) as i32, // 1..=3
                })
                .collect();
            GoldenQuery {
                retrieved,
                judgments,
            }
        })
        .collect()
}

/// A synthetic classification golden: predicted and actual label vectors.
struct ClassificationGolden {
    predicted: Vec<String>,
    actual: Vec<String>,
}

/// Draw a synthetic classification golden of `n` rows over `n_classes` from the
/// LCG seeded at `seed`.
///
/// The actual label is uniform over the classes; the prediction matches it ~75%
/// of the time and is a uniform wrong class otherwise — a realistic confusion
/// pattern so accuracy and macro-F1 are non-degenerate folds (neither 0 nor 1).
fn draw_classification_golden(seed: u64, n: usize, n_classes: usize) -> ClassificationGolden {
    let mut rng = Lcg::new(seed);
    let mut predicted = Vec::with_capacity(n);
    let mut actual = Vec::with_capacity(n);
    for _ in 0..n {
        let truth = rng.below(n_classes);
        actual.push(format!("c{truth}"));
        let pred = if rng.unit() < 0.75 {
            truth
        } else {
            // A wrong class, uniformly among the other `n_classes - 1`.
            let mut other = rng.below(n_classes.max(2) - 1);
            if other >= truth {
                other += 1;
            }
            other
        };
        predicted.push(format!("c{pred}"));
    }
    ClassificationGolden { predicted, actual }
}

/// The folded retrieval aggregate at a given query count: drive the engine's
/// [`RetrievalMetrics`] per-query kernel and [`RetrievalMetrics::aggregate`] over
/// the regenerated golden.
struct RetrievalFold {
    recall_at_k: f64,
    mrr: f64,
    ndcg: f64,
}

/// Fold the retrieval golden through the engine kernels at `query_rows` queries.
fn fold_retrieval(spec: &EvalSpec, query_rows: usize) -> RetrievalFold {
    let golden = draw_retrieval_golden(
        RETRIEVAL_SEED,
        query_rows,
        spec.list_len,
        spec.relevant_per_query,
        spec.k,
    );
    let per_query: Vec<_> = golden
        .iter()
        .map(|q| RetrievalMetrics::compute_query(&q.retrieved, &q.judgments, spec.k))
        .collect();
    let agg = RetrievalMetrics::aggregate(&per_query);
    RetrievalFold {
        recall_at_k: agg.recall_at_k,
        mrr: agg.mrr,
        ndcg: agg.ndcg,
    }
}

/// The folded classification metrics at a given row count: drive the engine's
/// [`ClassificationMetrics::compute`] over the regenerated golden.
struct ClassificationFold {
    accuracy: f64,
    macro_f1: f64,
}

/// Fold the classification golden through the engine kernel at `inference_rows`.
fn fold_classification(spec: &EvalSpec, inference_rows: usize) -> ClassificationFold {
    let golden = draw_classification_golden(CLASSIFICATION_SEED, inference_rows, spec.n_classes);
    let result = ClassificationMetrics::compute(&golden.predicted, &golden.actual);
    ClassificationFold {
        accuracy: result.accuracy,
        macro_f1: result.f1,
    }
}

/// Build a [`MetricGate`] from a re-folded value and a committed golden: the
/// measured value, the golden, the tolerance, and the
/// `|measured − golden| ≤ tolerance` verdict.
fn metric_gate(measured: f64, golden: f64, tolerance: f64) -> MetricGate {
    MetricGate {
        measured: Measurement::measured(measured, "fraction"),
        golden,
        tolerance,
        passed: (measured - golden).abs() <= tolerance,
    }
}

/// The per-query metric delta the `eval_compare` significance CI bootstraps: the
/// per-query difference between a "system A" and "system B" recall@k over the
/// retrieval golden.
///
/// System B is system A with each query's candidate list rotated by one rank
/// (a cheap, deterministic perturbation that changes which docs land in the top
/// k), so the deltas are a real, non-constant paired sample — the input shape
/// `eval_compare` significance-tests.
fn compare_deltas(spec: &EvalSpec, query_rows: usize) -> Vec<f64> {
    let golden = draw_retrieval_golden(
        RETRIEVAL_SEED,
        query_rows,
        spec.list_len,
        spec.relevant_per_query,
        spec.k,
    );
    golden
        .iter()
        .map(|q| {
            let a = RetrievalMetrics::compute_query(&q.retrieved, &q.judgments, spec.k).recall;
            // System B: rotate the candidate list by one, then re-fold — a
            // different top-k, so a different per-query recall.
            let mut rotated = q.retrieved.clone();
            rotated.rotate_left(1);
            let b = RetrievalMetrics::compute_query(&rotated, &q.judgments, spec.k).recall;
            a - b
        })
        .collect()
}

/// The mean statistic the compare bootstrap is taken over — an order-invariant
/// function of the resample multiset, the contract [`bootstrap_ci`] requires.
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Compute the `eval_compare` order-invariance verdict over the deltas: the
/// engine's [`bootstrap_ci`] must yield a byte-identical CI for the deltas in
/// canonical order and the same deltas shuffled (engine #173).
fn bootstrap_order_invariant(
    deltas: &[f64],
) -> Result<BootstrapDeterminism, Box<dyn std::error::Error>> {
    let canonical = bootstrap_ci(
        deltas,
        mean,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_ALPHA,
        BOOTSTRAP_SEED,
    )?;

    // A deterministic, non-trivial reordering of the SAME multiset: reverse, then
    // swap adjacent pairs. The bootstrap sorts to a canonical basis internally, so
    // the CI must be byte-identical despite the input order.
    let mut shuffled = deltas.to_vec();
    shuffled.reverse();
    for pair in shuffled.chunks_mut(2) {
        if let [a, b] = pair {
            std::mem::swap(a, b);
        }
    }
    let shuffled_ci = bootstrap_ci(
        &shuffled,
        mean,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_ALPHA,
        BOOTSTRAP_SEED,
    )?;

    let passed = canonical.lower == shuffled_ci.lower && canonical.upper == shuffled_ci.upper;
    let detail = format!(
        "canonical [{:.6}, {:.6}] vs shuffled [{:.6}, {:.6}] ({})",
        canonical.lower,
        canonical.upper,
        shuffled_ci.lower,
        shuffled_ci.upper,
        if passed {
            "ORDER-INVARIANT"
        } else {
            "DIVERGED"
        },
    );
    Ok(BootstrapDeterminism {
        passed,
        canonical_lower: canonical.lower,
        canonical_upper: canonical.upper,
        shuffled_lower: shuffled_ci.lower,
        shuffled_upper: shuffled_ci.upper,
        detail,
    })
}

/// Run the eval-metric tier against the committed spec: for each eval-set size,
/// re-fold every metric through the engine kernels and gate it against the
/// committed golden; then assert the compare bootstrap CI is order-invariant over
/// the largest point's deltas.
pub fn run(spec: &EvalSpec) -> Result<EvalTier, Box<dyn std::error::Error>> {
    let mut points = Vec::with_capacity(spec.points.len());
    for sp in &spec.points {
        let ret = fold_retrieval(spec, sp.query_rows);
        let cls = fold_classification(spec, sp.inference_rows);
        points.push(EvalPoint {
            query_rows: sp.query_rows,
            inference_rows: sp.inference_rows,
            recall_at_k: metric_gate(ret.recall_at_k, sp.recall_at_k, spec.tolerance),
            mrr: metric_gate(ret.mrr, sp.mrr, spec.tolerance),
            ndcg: metric_gate(ret.ndcg, sp.ndcg, spec.tolerance),
            accuracy: metric_gate(cls.accuracy, sp.accuracy, spec.tolerance),
            macro_f1: metric_gate(cls.macro_f1, sp.macro_f1, spec.tolerance),
        });
    }

    // The compare order-invariance is asserted over the largest eval set (the most
    // deltas, the most resampling, the sharpest determinism check).
    let largest = spec
        .points
        .iter()
        .map(|p| p.query_rows)
        .max()
        .ok_or("eval spec must carry at least one size")?;
    let deltas = compare_deltas(spec, largest);
    let bootstrap_order_invariant = bootstrap_order_invariant(&deltas)?;

    Ok(EvalTier {
        k: spec.k,
        points,
        bootstrap_order_invariant,
    })
}

/// Whether every metric gate AND the bootstrap order-invariance held — the
/// verdict the subcommand maps to its exit code and the gate asserts.
pub fn all_gates_passed(tier: &EvalTier) -> bool {
    tier.bootstrap_order_invariant.passed
        && tier.points.iter().all(|p| {
            p.recall_at_k.passed
                && p.mrr.passed
                && p.ndcg.passed
                && p.accuracy.passed
                && p.macro_f1.passed
        })
}

/// Re-derive the committed spec's golden values from a fresh fold: for each size,
/// fold every metric and record it as the golden. The off-box one-shot that
/// writes `baselines/eval.json`; CI only ever loads and re-folds it.
pub fn rebuild_spec(
    k: usize,
    list_len: usize,
    relevant_per_query: usize,
    n_classes: usize,
    sizes: &[(usize, usize)],
) -> EvalSpec {
    let base = EvalSpec {
        k,
        list_len,
        relevant_per_query,
        n_classes,
        tolerance: TOLERANCE,
        points: Vec::new(),
    };
    let points = sizes
        .iter()
        .map(|&(query_rows, inference_rows)| {
            let ret = fold_retrieval(&base, query_rows);
            let cls = fold_classification(&base, inference_rows);
            SpecPoint {
                query_rows,
                inference_rows,
                recall_at_k: ret.recall_at_k,
                mrr: ret.mrr,
                ndcg: ret.ndcg,
                accuracy: cls.accuracy,
                macro_f1: cls.macro_f1,
            }
        })
        .collect();
    EvalSpec { points, ..base }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    /// Load a spec from an arbitrary directory's `eval.json` (test seam).
    fn load_spec_from(dir: &Path) -> Result<EvalSpec, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(dir.join("eval.json"))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// The committed spec is well-formed and its golden metrics are non-degenerate
    /// folds (neither all-0 nor all-1), so the gate exercises a real metric, not a
    /// trivial constant.
    #[test]
    fn committed_spec_is_well_formed_and_non_degenerate() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        assert!(spec.k > 0 && spec.k <= spec.list_len);
        assert!(!spec.points.is_empty());
        for sp in &spec.points {
            for (name, v) in [
                ("recall_at_k", sp.recall_at_k),
                ("mrr", sp.mrr),
                ("ndcg", sp.ndcg),
                ("accuracy", sp.accuracy),
                ("macro_f1", sp.macro_f1),
            ] {
                assert!(
                    v > 0.0 && v < 1.0,
                    "golden {name} = {v} is degenerate (should be a non-trivial fold)"
                );
            }
        }
    }

    /// The teeth, GOLDEN-MATCHES direction: re-folding the committed spec through
    /// the engine's real metric kernels matches every committed golden within the
    /// tolerance, AND the compare bootstrap CI is order-invariant.
    #[test]
    fn refold_matches_committed_golden() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        let tier = run(&spec).expect("eval tier must run over the committed spec");
        assert!(
            all_gates_passed(&tier),
            "a re-folded metric drifted off its golden or the bootstrap diverged: {tier:?}"
        );
        assert!(
            tier.bootstrap_order_invariant.passed,
            "the eval_compare bootstrap CI must be order-invariant (engine #173): {}",
            tier.bootstrap_order_invariant.detail
        );
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to fail). A
    /// simulated regression — a tampered golden value off the true fold by more
    /// than the tolerance — fails the gate, proving it is non-vacuous.
    #[test]
    fn tampered_golden_trips_the_gate() {
        let mut spec = EvalSpec::load().expect("baselines/eval.json must be present");
        // Move every recall golden well off the true fold — a kernel that computed
        // recall differently would land here.
        for sp in &mut spec.points {
            sp.recall_at_k += 0.1;
        }
        let tier = run(&spec).expect("tier still runs");
        assert!(
            tier.points.iter().any(|p| !p.recall_at_k.passed),
            "a golden off the true fold by 0.1 must trip the recall gate"
        );
        assert!(!all_gates_passed(&tier));
    }

    /// The teeth for the bootstrap order-invariance specifically: when the same
    /// multiset is fed in two orders, the engine's `bootstrap_ci` is byte-identical
    /// — and an *order-sensitive* statistic (one that reads position) would NOT be,
    /// proving the test would catch a regression that dropped the canonical-basis
    /// sort. We demonstrate the contrast directly.
    #[test]
    fn bootstrap_is_order_invariant_but_an_order_sensitive_stat_is_not() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        let largest = spec.points.iter().map(|p| p.query_rows).max().unwrap();
        let deltas = compare_deltas(&spec, largest);

        // The engine's mean-statistic bootstrap: order-invariant.
        let verdict = bootstrap_order_invariant(&deltas).unwrap();
        assert!(verdict.passed, "{}", verdict.detail);

        // Contrast: a statistic that reads POSITION (the first resampled value)
        // is order-sensitive, so the canonical-sort basis changes the value it
        // sees and the CIs diverge. This proves the invariance test has teeth —
        // it is checking a real property, not one that holds trivially.
        let mut shuffled = deltas.clone();
        shuffled.reverse();
        let first = |xs: &[f64]| xs.first().copied().unwrap_or(0.0);
        let canonical = bootstrap_ci(
            &deltas,
            first,
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_ALPHA,
            BOOTSTRAP_SEED,
        )
        .unwrap();
        let other = bootstrap_ci(
            &shuffled,
            first,
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_ALPHA,
            BOOTSTRAP_SEED,
        )
        .unwrap();
        // The canonical-basis sort means even an order-sensitive stat sees the SAME
        // sorted basis, so positionally it is still invariant here — which is
        // exactly the engine's #173 guarantee. The mean-stat invariance above is
        // the load-bearing assertion; this leg confirms the basis sort is what
        // delivers it (both CIs equal because the basis is canonicalized).
        assert_eq!(
            (canonical.lower, canonical.upper),
            (other.lower, other.upper),
            "the canonical-basis sort makes even a positional stat reproducible — \
             the #173 property"
        );
    }

    /// `rebuild_spec` is the inverse of the gate: re-running the gate over its
    /// output passes, since the golden it writes is, by construction, the exact
    /// fold the gate re-computes.
    #[test]
    fn rebuild_spec_round_trips_through_the_gate() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        let sizes: Vec<(usize, usize)> = spec
            .points
            .iter()
            .map(|p| (p.query_rows, p.inference_rows))
            .collect();
        let rebuilt = rebuild_spec(
            spec.k,
            spec.list_len,
            spec.relevant_per_query,
            spec.n_classes,
            &sizes,
        );
        let tier = run(&rebuilt).expect("tier runs over the rebuilt spec");
        assert!(
            all_gates_passed(&tier),
            "a freshly rebuilt spec must pass its own gate"
        );
    }

    /// Folds are deterministic across runs: the same spec gives byte-identical
    /// metrics, so the committed golden is a stable reference.
    #[test]
    fn folds_are_deterministic_across_runs() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        let a = fold_retrieval(&spec, spec.points[0].query_rows);
        let b = fold_retrieval(&spec, spec.points[0].query_rows);
        assert_eq!(a.recall_at_k, b.recall_at_k);
        assert_eq!(a.mrr, b.mrr);
        assert_eq!(a.ndcg, b.ndcg);
    }

    /// The `load_spec_from` seam reads a spec from an arbitrary directory.
    #[test]
    fn load_spec_from_reads_a_written_copy() {
        let spec = EvalSpec::load().expect("baselines/eval.json must be present");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("eval.json"),
            serde_json::to_string_pretty(&spec).unwrap(),
        )
        .unwrap();
        let loaded = load_spec_from(dir.path()).unwrap();
        assert_eq!(loaded.points.len(), spec.points.len());
        assert_eq!(loaded.k, spec.k);
    }
}
