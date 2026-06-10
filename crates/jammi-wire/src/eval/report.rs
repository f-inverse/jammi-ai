//! Typed reports for the eval API.
//!
//! Every eval entry point (`eval_embeddings`, `eval_inference`, `eval_compare`)
//! returns one of these structs. The shape carries both an aggregate (mean
//! across queries / records) and per-query / per-record arrays. The per-query
//! data is what sample-based statistical rules (Welch's t, Mann-Whitney U)
//! consume at gate time; the aggregate is what the catalog persists for
//! historical reporting.

use std::collections::BTreeMap;

use jammi_numerics::classification::ClassificationResult;
use jammi_numerics::ner::{Entity, NerMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
// The bootstrap / rank-sum kernels are pulled in only where significance is
// *computed* — the local engine's `eval_compare`. The significance *types*
// below stay transport-neutral so the thin wire client round-trips them.
use jammi_numerics::stats::{bootstrap_ci, mann_whitney_u, Interval};
use serde::{Deserialize, Serialize};

/// Result of one `eval_embeddings` invocation. Carries the mean over all
/// queries (`aggregate.recall_at_k`, etc.) and the per-query records that
/// sample-based statistical rules consume at gate time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEvalReport {
    /// The catalog id of the recorded run. The same id keys the persisted
    /// per-query rows (`_jammi_eval_per_query`), so a caller can read them
    /// back via `eval_per_query(eval_run_id)` (spec J9). Every embedding eval
    /// records a run, so this is always a valid, re-readable run id.
    #[serde(default)]
    pub eval_run_id: String,
    /// Mean metrics across all queries.
    pub aggregate: AggregateMetrics,
    /// One record per query, in golden-set order.
    pub per_query: Vec<PerQueryRecord>,
}

/// A single query's metrics plus its `query_id` join key.
///
/// `metrics` (the historical single-cutoff [`QueryMetrics`]) is unchanged so
/// existing consumers keep reading `metrics.recall` / `metrics.mrr` / etc.
/// The J9 additions — `recall_at_ks` (Recall@{1,3,5,10}), `distance` (the
/// top-1 result's score), and opaque `cohorts` tags — are additive fields that
/// older callers can ignore. These are the fields persisted to
/// `_jammi_eval_per_query`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryRecord {
    /// `query_id` from the golden source (the join key).
    pub query_id: String,
    pub metrics: QueryMetrics,
    /// Recall at multiple cutoffs as `(k, recall@k)` pairs, in ascending k.
    #[serde(default)]
    pub recall_at_ks: Vec<(usize, f64)>,
    /// The top-1 retrieved result's score (distance / similarity), or `0.0`
    /// when the query returned no results.
    #[serde(default)]
    pub distance: f64,
    /// Opaque per-query cohort tags supplied at eval time (`{}` when none).
    /// The substrate never interprets these — declaration/validation is a
    /// downstream consumer's concern.
    #[serde(default)]
    pub cohorts: BTreeMap<String, String>,
}

/// The cutoffs at which per-query Recall@k is recorded and persisted (spec J9).
pub const PER_QUERY_RECALL_KS: [usize; 4] = [1, 3, 5, 10];

/// Result of one `eval_inference` invocation. The aggregate variant matches
/// the task kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEvalReport {
    pub aggregate: InferenceAggregate,
    pub per_record: Vec<PerRecordPrediction>,
}

/// Aggregate metrics for `eval_inference`, tagged by task kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum InferenceAggregate {
    Classification(ClassificationResult),
    Ner(NerMetrics),
}

/// One predicted / gold prediction pair from `eval_inference`, tagged by
/// task kind so the per-record array carries task-shaped payloads instead
/// of a single string-pair shape.
///
/// Classification carries the predicted/gold label strings; NER carries
/// the predicted/gold entity-span sets so downstream consumers can compute
/// per-record precision/recall without re-decoding the JSON payload the
/// NER inference adapter wrote.
///
/// Wire shape mirrors [`InferenceAggregate`]: a serde-tagged enum with
/// `"task": "classification"` or `"task": "ner"` discriminating the
/// variant, so both wire formats round-trip through the same
/// `serializable_to_pydict` conversion used by every other eval response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum PerRecordPrediction {
    Classification {
        record_id: String,
        predicted: String,
        gold: String,
    },
    Ner {
        record_id: String,
        predicted: Vec<Entity>,
        gold: Vec<Entity>,
    },
}

// ─── Calibration / uncertainty eval (spec R2) ───────────────────────────────
//
// `eval_calibration` measures whether a predictor's *uncertainty* is honest,
// the orthogonal question to `eval_embeddings`/`eval_inference`'s accuracy. A
// predictor over a held-out `(prediction, outcome)` set emits either a
// parametric predictive Gaussian (NP3) or an ensemble of predictive draws
// (NP4); both yield a strictly proper score (CRPS / NLL), a calibration
// diagnostic (adaptive-bin debiased ECE over the PIT), and sharpness. The
// proper score is the headline — it is the only metric uniquely minimised by
// the true distribution, so it rewards calibration and sharpness jointly.
//
// All scoring routes through `jammi_numerics::calibration`; no calibration math
// lives here.

/// The nominal central-interval level at which coverage and sharpness are
/// reported (a 90% interval). Coverage of a calibrated predictor approaches
/// this value; sharpness is the mean width of the interval that achieves it.
pub const CALIBRATION_INTERVAL_LEVEL: f64 = 0.90;

/// The number of equal-mass bins for the adaptive, debiased ECE diagnostic.
pub const CALIBRATION_ECE_BINS: usize = 10;

/// One held-out predictive distribution paired with its realised outcome.
///
/// The variant carries the predictor's output shape: `Gaussian` for a
/// parametric predictive `Normal(mean, sd)` (NP3), `Sample` for an ensemble of
/// predictive draws (NP4). Both are scored on the same yardstick so the
/// distributional spectrum is comparable. `cohorts` are opaque per-record
/// segment tags the substrate never interprets — they drive the per-cohort
/// slice the same way `eval_embeddings` cohorts do.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum CalibrationPrediction {
    /// A parametric predictive Gaussian and the realised outcome.
    Gaussian {
        record_id: String,
        mean: f64,
        sd: f64,
        outcome: f64,
        #[serde(default)]
        cohorts: BTreeMap<String, String>,
    },
    /// An ensemble of predictive draws and the realised outcome.
    Sample {
        record_id: String,
        draws: Vec<f64>,
        outcome: f64,
        #[serde(default)]
        cohorts: BTreeMap<String, String>,
    },
}

// The accessors are consumed only by the local scoring path; gating the impl
// keeps a `wire`-only build free of dead-code warnings (mirroring the
// significance helpers below).
impl CalibrationPrediction {
    fn record_id(&self) -> &str {
        match self {
            Self::Gaussian { record_id, .. } | Self::Sample { record_id, .. } => record_id,
        }
    }

    fn outcome(&self) -> f64 {
        match self {
            Self::Gaussian { outcome, .. } | Self::Sample { outcome, .. } => *outcome,
        }
    }

    fn cohorts(&self) -> &BTreeMap<String, String> {
        match self {
            Self::Gaussian { cohorts, .. } | Self::Sample { cohorts, .. } => cohorts,
        }
    }
}

/// One record's calibration scores: the proper scores (`crps`, `nll`), the
/// probability-integral-transform value (`pit`, uniform on `[0, 1]` when
/// calibrated), and whether the nominal central interval covered the outcome.
///
/// These are the per-record rows persisted to `_jammi_eval_per_query` and the
/// paired sample a proper-score significance test consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRecordCalibration {
    pub record_id: String,
    /// Continuous ranked probability score — the proper-score headline, in the
    /// units of the outcome. Lower is better.
    pub crps: f64,
    /// Negative log-likelihood of the outcome under the predictive
    /// distribution, in nats. Lower is better.
    pub nll: f64,
    /// Predictive CDF at the outcome; uniform on `[0, 1]` under calibration.
    pub pit: f64,
    /// Whether the nominal [`CALIBRATION_INTERVAL_LEVEL`] central interval
    /// contained the outcome.
    pub covered: bool,
    /// Width of that nominal interval — the per-record sharpness.
    pub interval_width: f64,
    /// Opaque per-record cohort tags (`{}` when none).
    #[serde(default)]
    pub cohorts: BTreeMap<String, String>,
}

/// Mean calibration scores across every held-out record.
///
/// `crps` is the headline (a strictly proper score); `nll` is the second proper
/// score; `adaptive_ece` is the debiased, equal-mass calibration diagnostic;
/// `sharpness` and `coverage` are the calibration/sharpness pair read together
/// (sharper is better only at fixed coverage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAggregate {
    /// Number of held-out records scored.
    pub n: usize,
    /// Mean CRPS — the proper-score headline.
    pub crps: f64,
    /// Mean negative log-likelihood (nats).
    pub nll: f64,
    /// Debiased, equal-mass (adaptive-bin) expected calibration error over the
    /// PIT values — the calibration diagnostic, never the headline.
    pub adaptive_ece: f64,
    /// Mean width of the nominal [`CALIBRATION_INTERVAL_LEVEL`] interval.
    pub sharpness: f64,
    /// Empirical coverage of the nominal interval; approaches
    /// [`CALIBRATION_INTERVAL_LEVEL`] under calibration.
    pub coverage: f64,
}

/// Calibration scores sliced to one cohort segment, with the sample size and a
/// bootstrap confidence interval on the proper score.
///
/// Marginal coverage can hide conditional miscoverage — a predictor can hit its
/// nominal level globally while systematically under-covering a subgroup — so
/// the harness slices coverage and CRPS by cohort. `n` and `crps_ci` are
/// reported so a small cohort is not over-read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortCalibration {
    /// The cohort tag key this slice groups on.
    pub key: String,
    /// The cohort tag value this slice groups on.
    pub value: String,
    /// Records in this cohort.
    pub n: usize,
    /// Mean CRPS within the cohort.
    pub crps: f64,
    /// Lower bound of the bootstrap CI on the cohort's mean CRPS, or `None` for
    /// a singleton cohort (nothing to resample a spread from).
    #[serde(default)]
    pub crps_ci_lower: Option<f64>,
    /// Upper bound of the bootstrap CI on the cohort's mean CRPS.
    #[serde(default)]
    pub crps_ci_upper: Option<f64>,
    /// Coverage of the nominal interval within the cohort.
    pub coverage: f64,
}

/// Result of one `eval_calibration` invocation. Carries the aggregate proper
/// score + diagnostics, the per-cohort slices, and the per-record scores that a
/// proper-score significance test pairs against a baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEvalReport {
    /// The catalog id of the recorded run; keys the persisted per-record rows
    /// (`_jammi_eval_per_query`) so they read back via `eval_per_query`.
    #[serde(default)]
    pub eval_run_id: String,
    /// Mean scores across all held-out records.
    pub aggregate: CalibrationAggregate,
    /// Per-cohort coverage + CRPS with n + CI, in ascending `(key, value)`.
    pub per_cohort: Vec<CohortCalibration>,
    /// One record per held-out prediction, in input order.
    pub per_record: Vec<PerRecordCalibration>,
}

impl CalibrationEvalReport {
    /// Distribution-free paired significance of this run's CRPS against a
    /// `baseline` run (spec R2 §4): the per-record proper scores are paired by
    /// `record_id` and compared with the same bootstrap CI + Mann–Whitney U that
    /// `eval_compare` wires for retrieval — no new stats. CRPS is the proper
    /// headline, so its paired test is what turns "this predictor is
    /// better-calibrated than the baseline" into a p-value rather than a vibe.
    ///
    /// Returns `None` when the two runs share no `record_id` to pair on. A
    /// `ci_upper` below zero is the resampling analogue of "this run's CRPS is
    /// significantly lower (better) than the baseline's".
    pub fn significance_vs(&self, baseline: &CalibrationEvalReport) -> Option<MetricSignificance> {
        calibration_significance(&baseline.per_record, &self.per_record)
    }
}

/// Result of one `eval_compare` invocation across multiple embedding tables.
/// The first table in `per_table` is the baseline; all subsequent entries
/// carry their `delta` relative to that baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareEvalReport {
    pub per_table: Vec<TableEvalReport>,
}

/// One table's embedding eval plus its delta relative to the baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEvalReport {
    pub table_name: String,
    pub embedding_eval: EmbeddingEvalReport,
    /// Aggregate-metric deltas relative to the first table in `per_table`.
    /// `None` for the baseline itself.
    pub delta: Option<AggregateDelta>,
}

/// Per-metric deltas (absolute + relative) between two aggregate scores, plus
/// the distribution-free paired significance of each delta.
///
/// `significance` is computed from the two tables' per-query metric arrays
/// (paired by `query_id`), not from the aggregates: it is `None` only when the
/// two runs share no query in common, so there is nothing to pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateDelta {
    pub recall_at_k: MetricDelta,
    pub precision_at_k: MetricDelta,
    pub mrr: MetricDelta,
    pub ndcg: MetricDelta,
    /// Paired significance of each metric delta, or `None` when the baseline
    /// and treatment runs share no `query_id` to pair on.
    #[serde(default)]
    pub significance: Option<DeltaSignificance>,
}

/// Single-metric delta: `absolute = model - baseline`, `relative = absolute /
/// baseline` (or `0.0` when the baseline is zero).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricDelta {
    pub absolute: f64,
    pub relative: f64,
}

/// Distribution-free paired significance for each metric in an
/// [`AggregateDelta`], in the same metric layout as the delta itself.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeltaSignificance {
    pub recall_at_k: MetricSignificance,
    pub precision_at_k: MetricSignificance,
    pub mrr: MetricSignificance,
    pub ndcg: MetricSignificance,
}

/// Distribution-free significance of one metric's delta, computed from the
/// paired per-query values.
///
/// - `ci` is a percentile bootstrap confidence interval on the mean paired
///   difference (`treatment − baseline`); a CI that excludes zero is the
///   resampling analogue of "the delta is significant".
/// - `p_value` is the two-tailed Mann–Whitney U p-value comparing the
///   baseline and treatment per-query distributions — distribution-free and
///   robust to the bounded, tie-heavy shape of retrieval metrics.
///
/// Both are deterministic: the bootstrap runs under [`BOOTSTRAP_SEED`] and a
/// fixed iteration count, so identical inputs yield identical output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricSignificance {
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

/// Bootstrap resamples for the paired confidence interval. Fixed so the CI is
/// a deterministic function of its inputs.
pub const BOOTSTRAP_ITERATIONS: usize = 10_000;

/// Two-tailed significance level for the bootstrap CI (a 95% interval).
pub const BOOTSTRAP_ALPHA: f64 = 0.05;

/// Pinned RNG seed for the paired bootstrap. The seed is part of the
/// significance contract — it is what makes the CI reproducible across runs.
pub const BOOTSTRAP_SEED: u64 = 0x6a616d6d695f7031; // "jammi_p1"

/// One per-query metric value paired between baseline and treatment.
struct PairedMetric {
    baseline: f64,
    treatment: f64,
}

/// Pair two tables' per-query records by `query_id`, returning the paired
/// values for one metric (selected by `extract`) in ascending `query_id`
/// order. Queries present in only one table are dropped — significance is a
/// paired test and an unpaired query carries no difference to resample.
fn paired_metric<F>(
    baseline: &[PerQueryRecord],
    treatment: &[PerQueryRecord],
    extract: F,
) -> Vec<PairedMetric>
where
    F: Fn(&QueryMetrics) -> f64,
{
    let treatment_by_id: BTreeMap<&str, &QueryMetrics> = treatment
        .iter()
        .map(|r| (r.query_id.as_str(), &r.metrics))
        .collect();
    baseline
        .iter()
        .filter_map(|b| {
            treatment_by_id
                .get(b.query_id.as_str())
                .map(|t| PairedMetric {
                    baseline: extract(&b.metrics),
                    treatment: extract(t),
                })
        })
        .collect()
}

/// Compute the distribution-free significance of one metric's paired deltas:
/// a bootstrap CI on the mean paired difference plus a Mann–Whitney U p-value
/// between the baseline and treatment distributions. Returns `None` when the
/// pairing is empty (no shared `query_id`).
fn metric_significance(paired: &[PairedMetric]) -> Option<MetricSignificance> {
    if paired.is_empty() {
        return None;
    }
    let differences: Vec<f64> = paired.iter().map(|p| p.treatment - p.baseline).collect();
    let baseline: Vec<f64> = paired.iter().map(|p| p.baseline).collect();
    let treatment: Vec<f64> = paired.iter().map(|p| p.treatment).collect();

    let Interval { lower, upper } = bootstrap_ci(
        &differences,
        |resample| resample.iter().sum::<f64>() / resample.len() as f64,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_ALPHA,
        BOOTSTRAP_SEED,
    )
    .ok()?;
    let p_value = mann_whitney_u(&baseline, &treatment).ok()?.p_value;

    Some(MetricSignificance {
        p_value,
        ci_lower: lower,
        ci_upper: upper,
    })
}

/// Compute paired significance for all four metrics between a baseline and a
/// treatment run's per-query records. Returns `None` when the two runs share
/// no `query_id` (nothing to pair).
pub fn delta_significance(
    baseline: &[PerQueryRecord],
    treatment: &[PerQueryRecord],
) -> Option<DeltaSignificance> {
    Some(DeltaSignificance {
        recall_at_k: metric_significance(&paired_metric(baseline, treatment, |m| m.recall))?,
        precision_at_k: metric_significance(&paired_metric(baseline, treatment, |m| m.precision))?,
        mrr: metric_significance(&paired_metric(baseline, treatment, |m| m.mrr))?,
        ndcg: metric_significance(&paired_metric(baseline, treatment, |m| m.ndcg))?,
    })
}

/// The z-multiplier for the two-sided central interval at
/// [`CALIBRATION_INTERVAL_LEVEL`] under a Gaussian predictive (the 0.95 normal
/// quantile for a 90% interval).
const CALIBRATION_INTERVAL_Z: f64 = 1.6448536269514722;

/// Score one held-out prediction against its outcome: CRPS and NLL (proper
/// scores) via `jammi_numerics::calibration`, the PIT value, and the nominal
/// central interval (for coverage + sharpness). All math is the numerics
/// crate's — this only selects the family that matches the prediction's shape
/// and assembles the per-record row.
fn score_prediction(
    prediction: &CalibrationPrediction,
) -> jammi_db::error::Result<PerRecordCalibration> {
    use jammi_numerics::calibration::{
        crps_gaussian, crps_sample, gaussian_nll, pit_values, sample_nll,
    };

    let map_err = |e: jammi_numerics::NumericsError| {
        jammi_db::error::JammiError::Eval(format!(
            "calibration scoring failed for record '{}': {e}",
            prediction.record_id()
        ))
    };

    let (crps, nll, pit, lower, upper) = match prediction {
        CalibrationPrediction::Gaussian {
            mean, sd, outcome, ..
        } => {
            let crps = crps_gaussian(*outcome, *mean, *sd).map_err(map_err)?;
            let nll = gaussian_nll(*outcome, *mean, *sd).map_err(map_err)?;
            // PIT = Phi((y - mean)/sd) — the numerics `pit_values` over a
            // one-element forecast, so the predictive-CDF evaluation stays in
            // the numerics crate rather than reaching for a normal here.
            let pit = pit_values(&[*mean], &[*sd], &[*outcome]).map_err(map_err)?[0];
            let half = CALIBRATION_INTERVAL_Z * *sd;
            (crps, nll, pit, *mean - half, *mean + half)
        }
        CalibrationPrediction::Sample { draws, outcome, .. } => {
            let crps = crps_sample(draws, *outcome).map_err(map_err)?;
            let nll = sample_nll(draws, *outcome).map_err(map_err)?;
            // Empirical PIT: the fraction of draws at or below the outcome.
            let pit = draws.iter().filter(|&&x| x <= *outcome).count() as f64 / draws.len() as f64;
            let (lower, upper) = empirical_central_interval(draws);
            (crps, nll, pit, lower, upper)
        }
    };

    Ok(PerRecordCalibration {
        record_id: prediction.record_id().to_string(),
        crps,
        nll,
        pit,
        covered: lower <= prediction.outcome() && prediction.outcome() <= upper,
        interval_width: upper - lower,
        cohorts: prediction.cohorts().clone(),
    })
}

/// The empirical central [`CALIBRATION_INTERVAL_LEVEL`] interval of a draw
/// ensemble: the `(alpha/2, 1 - alpha/2)` empirical quantiles via linear
/// interpolation between order statistics.
fn empirical_central_interval(draws: &[f64]) -> (f64, f64) {
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let alpha = 1.0 - CALIBRATION_INTERVAL_LEVEL;
    (
        empirical_quantile(&sorted, alpha / 2.0),
        empirical_quantile(&sorted, 1.0 - alpha / 2.0),
    )
}

/// Linear-interpolated empirical quantile of an already-sorted slice.
fn empirical_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute the full calibration report from held-out predictions: per-record
/// proper scores, the aggregate proper-score headline + diagnostics, and the
/// per-cohort slices. Pure over its inputs (no session, no I/O) and
/// deterministic — the only randomness is the seeded cohort-CI bootstrap.
///
/// Returns an error when `predictions` is empty (a proper score is undefined
/// over no trials) or when a prediction is malformed for its family (e.g. a
/// non-positive `sd`, surfaced by the numerics scorer).
pub fn compute_calibration(
    eval_run_id: String,
    predictions: &[CalibrationPrediction],
) -> jammi_db::error::Result<CalibrationEvalReport> {
    if predictions.is_empty() {
        return Err(jammi_db::error::JammiError::Eval(
            "eval_calibration requires at least one held-out prediction".into(),
        ));
    }

    let per_record: Vec<PerRecordCalibration> = predictions
        .iter()
        .map(score_prediction)
        .collect::<jammi_db::error::Result<_>>()?;

    let n = per_record.len();
    let n_f = n as f64;
    let crps = per_record.iter().map(|r| r.crps).sum::<f64>() / n_f;
    let nll = per_record.iter().map(|r| r.nll).sum::<f64>() / n_f;
    let sharpness = per_record.iter().map(|r| r.interval_width).sum::<f64>() / n_f;
    let coverage = per_record.iter().filter(|r| r.covered).count() as f64 / n_f;

    // The calibration diagnostic is the adaptive (equal-mass), debiased ECE
    // over the PIT — PIT uniformity *is* calibration, and the numerics
    // `pit_calibration_error` scores the PIT's departure from the uniform
    // diagonal. It is a diagnostic, never the headline: the proper score is the
    // verdict.
    let pit: Vec<f64> = per_record.iter().map(|r| r.pit).collect();
    let adaptive_ece =
        jammi_numerics::calibration::pit_calibration_error(&pit, CALIBRATION_ECE_BINS).map_err(
            |e| jammi_db::error::JammiError::Eval(format!("calibration diagnostic failed: {e}")),
        )?;

    let aggregate = CalibrationAggregate {
        n,
        crps,
        nll,
        adaptive_ece,
        sharpness,
        coverage,
    };

    let per_cohort = compute_cohort_calibration(&per_record);

    Ok(CalibrationEvalReport {
        eval_run_id,
        aggregate,
        per_cohort,
        per_record,
    })
}

/// Slice per-record scores by every `(cohort_key, cohort_value)` pair present,
/// reporting each slice's n, mean CRPS with a bootstrap CI, and coverage. A
/// record contributes to one slice per cohort tag it carries.
fn compute_cohort_calibration(per_record: &[PerRecordCalibration]) -> Vec<CohortCalibration> {
    // Group record indices by the cohort tag they carry. BTreeMap keeps the
    // output in ascending (key, value) order — deterministic across runs.
    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, rec) in per_record.iter().enumerate() {
        for (key, value) in &rec.cohorts {
            groups
                .entry((key.clone(), value.clone()))
                .or_default()
                .push(i);
        }
    }

    groups
        .into_iter()
        .map(|((key, value), idxs)| {
            let crps_values: Vec<f64> = idxs.iter().map(|&i| per_record[i].crps).collect();
            let n = crps_values.len();
            let crps = crps_values.iter().sum::<f64>() / n as f64;
            let covered = idxs.iter().filter(|&&i| per_record[i].covered).count();
            let coverage = covered as f64 / n as f64;

            // A singleton cohort has no spread to resample, so its CI is `None`
            // rather than a degenerate [point, point]; n is reported so the
            // caller does not over-read it.
            let (crps_ci_lower, crps_ci_upper) = if n < 2 {
                (None, None)
            } else {
                match bootstrap_ci(
                    &crps_values,
                    |resample| resample.iter().sum::<f64>() / resample.len() as f64,
                    BOOTSTRAP_ITERATIONS,
                    BOOTSTRAP_ALPHA,
                    BOOTSTRAP_SEED,
                ) {
                    Ok(Interval { lower, upper }) => (Some(lower), Some(upper)),
                    Err(_) => (None, None),
                }
            };

            CohortCalibration {
                key,
                value,
                n,
                crps,
                crps_ci_lower,
                crps_ci_upper,
                coverage,
            }
        })
        .collect()
}

/// Distribution-free paired significance of the per-record CRPS deltas between
/// a baseline and a treatment calibration run (reusing R1's bootstrap CI +
/// Mann–Whitney U — no new stats). Records are paired by `record_id`; pairs
/// present in only one run are dropped. Returns `None` when no `record_id` is
/// shared. The CRPS is the proper-score headline, so its paired test is what
/// turns "B is better-calibrated than A" into a p-value.
pub(crate) fn calibration_significance(
    baseline: &[PerRecordCalibration],
    treatment: &[PerRecordCalibration],
) -> Option<MetricSignificance> {
    let treatment_by_id: BTreeMap<&str, f64> = treatment
        .iter()
        .map(|r| (r.record_id.as_str(), r.crps))
        .collect();
    let paired: Vec<(f64, f64)> = baseline
        .iter()
        .filter_map(|b| {
            treatment_by_id
                .get(b.record_id.as_str())
                .map(|&t| (b.crps, t))
        })
        .collect();
    if paired.is_empty() {
        return None;
    }
    let differences: Vec<f64> = paired.iter().map(|(b, t)| t - b).collect();
    let base: Vec<f64> = paired.iter().map(|(b, _)| *b).collect();
    let treat: Vec<f64> = paired.iter().map(|(_, t)| *t).collect();

    let Interval { lower, upper } = bootstrap_ci(
        &differences,
        |resample| resample.iter().sum::<f64>() / resample.len() as f64,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_ALPHA,
        BOOTSTRAP_SEED,
    )
    .ok()?;
    let p_value = mann_whitney_u(&base, &treat).ok()?.p_value;
    Some(MetricSignificance {
        p_value,
        ci_lower: lower,
        ci_upper: upper,
    })
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    fn gaussian(id: &str, mean: f64, sd: f64, outcome: f64) -> CalibrationPrediction {
        CalibrationPrediction::Gaussian {
            record_id: id.to_string(),
            mean,
            sd,
            outcome,
            cohorts: BTreeMap::new(),
        }
    }

    fn gaussian_with_cohort(
        id: &str,
        mean: f64,
        sd: f64,
        outcome: f64,
        key: &str,
        value: &str,
    ) -> CalibrationPrediction {
        let mut cohorts = BTreeMap::new();
        cohorts.insert(key.to_string(), value.to_string());
        CalibrationPrediction::Gaussian {
            record_id: id.to_string(),
            mean,
            sd,
            outcome,
            cohorts,
        }
    }

    /// A held-out set drawn so the predictive Normal(0,1) is exactly the data
    /// generator — calibrated by construction. Deterministic: a fixed grid of
    /// standard-normal quantiles, not a sampler.
    fn calibrated_standard_normal(n: usize) -> Vec<CalibrationPrediction> {
        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::standard();
        (0..n)
            .map(|i| {
                // Evenly spaced PIT levels mapped through the inverse CDF give a
                // sample whose empirical distribution matches Normal(0,1).
                let p = (i as f64 + 0.5) / n as f64;
                let outcome = normal.inverse_cdf(p);
                gaussian(&format!("r{i}"), 0.0, 1.0, outcome)
            })
            .collect()
    }

    #[test]
    fn empty_predictions_is_error() {
        assert!(compute_calibration("run".into(), &[]).is_err());
    }

    #[test]
    fn gaussian_proper_scores_match_numerics() {
        // The aggregate CRPS/NLL are the means of the numerics per-record
        // scores — the report must not re-derive them.
        let preds = vec![gaussian("a", 0.0, 1.0, 0.0), gaussian("b", 0.0, 1.0, 1.0)];
        let report = compute_calibration("run".into(), &preds).unwrap();
        let expected_crps = (jammi_numerics::calibration::crps_gaussian(0.0, 0.0, 1.0).unwrap()
            + jammi_numerics::calibration::crps_gaussian(1.0, 0.0, 1.0).unwrap())
            / 2.0;
        assert!((report.aggregate.crps - expected_crps).abs() < 1e-12);
        assert_eq!(report.aggregate.n, 2);
    }

    #[test]
    fn calibrated_predictor_hits_nominal_coverage() {
        // A calibrated Normal(0,1) over its own quantile grid covers ~90% at the
        // nominal 90% interval, and its PIT-based ECE is small.
        let preds = calibrated_standard_normal(400);
        let report = compute_calibration("run".into(), &preds).unwrap();
        assert!(
            (report.aggregate.coverage - CALIBRATION_INTERVAL_LEVEL).abs() < 0.02,
            "coverage {} should be near {CALIBRATION_INTERVAL_LEVEL}",
            report.aggregate.coverage
        );
        assert!(
            report.aggregate.adaptive_ece < 0.1,
            "calibrated predictor ECE should be small: {}",
            report.aggregate.adaptive_ece
        );
    }

    #[test]
    fn overconfident_predictor_undercovers() {
        // A predictive sd far too small for the spread of outcomes: coverage
        // collapses below the nominal level — the marginal-predictor degenerate
        // in reverse, caught by coverage even when a point metric would not.
        let preds: Vec<CalibrationPrediction> = (0..100)
            .map(|i| {
                let outcome = (i as f64 - 50.0) / 5.0; // spread ~ +/- 10
                gaussian(&format!("r{i}"), 0.0, 0.1, outcome)
            })
            .collect();
        let report = compute_calibration("run".into(), &preds).unwrap();
        assert!(
            report.aggregate.coverage < 0.2,
            "overconfident predictor should badly under-cover: {}",
            report.aggregate.coverage
        );
    }

    #[test]
    fn sample_and_gaussian_families_agree_on_a_wide_ensemble() {
        // A large ensemble from Normal(0,1) should score close to the
        // closed-form Gaussian on the same outcome — both routes go through the
        // numerics crate, so this guards the family selection, not the math.
        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::standard();
        let draws: Vec<f64> = (0..5000)
            .map(|i| normal.inverse_cdf((i as f64 + 0.5) / 5000.0))
            .collect();
        let sample = CalibrationPrediction::Sample {
            record_id: "s".into(),
            draws,
            outcome: 0.5,
            cohorts: BTreeMap::new(),
        };
        let gauss = gaussian("g", 0.0, 1.0, 0.5);
        let sample_report =
            compute_calibration("run".into(), std::slice::from_ref(&sample)).unwrap();
        let gauss_report = compute_calibration("run".into(), std::slice::from_ref(&gauss)).unwrap();
        assert!(
            (sample_report.aggregate.crps - gauss_report.aggregate.crps).abs() < 2e-2,
            "sample CRPS {} vs gaussian CRPS {}",
            sample_report.aggregate.crps,
            gauss_report.aggregate.crps
        );
    }

    #[test]
    fn cohorts_slice_coverage_and_crps() {
        // Two cohorts: "easy" is calibrated, "hard" is overconfident. The slice
        // must separate their coverage and carry an n + CI per cohort.
        let mut preds = Vec::new();
        for i in 0..30 {
            let outcome = ((i as f64 + 0.5) / 30.0 - 0.5) * 2.0;
            preds.push(gaussian_with_cohort(
                &format!("e{i}"),
                0.0,
                1.0,
                outcome,
                "tier",
                "easy",
            ));
        }
        for i in 0..30 {
            let outcome = (i as f64 - 15.0) * 2.0; // wide spread, tiny sd
            preds.push(gaussian_with_cohort(
                &format!("h{i}"),
                0.0,
                0.1,
                outcome,
                "tier",
                "hard",
            ));
        }
        let report = compute_calibration("run".into(), &preds).unwrap();
        assert_eq!(report.per_cohort.len(), 2);
        let easy = report
            .per_cohort
            .iter()
            .find(|c| c.value == "easy")
            .unwrap();
        let hard = report
            .per_cohort
            .iter()
            .find(|c| c.value == "hard")
            .unwrap();
        assert_eq!(easy.n, 30);
        assert_eq!(hard.n, 30);
        assert!(
            easy.coverage > hard.coverage,
            "easy cohort {} should cover more than hard {}",
            easy.coverage,
            hard.coverage
        );
        assert!(easy.crps_ci_lower.is_some(), "n=30 cohort carries a CI");
        assert!(easy.crps_ci_lower.unwrap() <= easy.crps);
        assert!(easy.crps <= easy.crps_ci_upper.unwrap());
    }

    #[test]
    fn singleton_cohort_has_no_ci() {
        let preds = vec![gaussian_with_cohort("a", 0.0, 1.0, 0.0, "tier", "solo")];
        let report = compute_calibration("run".into(), &preds).unwrap();
        let cohort = &report.per_cohort[0];
        assert_eq!(cohort.n, 1);
        assert!(cohort.crps_ci_lower.is_none());
    }

    #[test]
    fn compute_is_deterministic() {
        let preds = calibrated_standard_normal(60);
        let a = compute_calibration("run".into(), &preds).unwrap();
        let b = compute_calibration("run".into(), &preds).unwrap();
        assert_eq!(a.aggregate.crps, b.aggregate.crps);
        assert_eq!(a.aggregate.adaptive_ece, b.aggregate.adaptive_ece);
    }

    #[test]
    fn crps_significance_pairs_by_record_id() {
        // Treatment is sharper-and-calibrated, baseline is overconfident: the
        // paired CRPS delta is significant and its CI excludes zero.
        let baseline: Vec<PerRecordCalibration> = compute_calibration(
            "b".into(),
            &(0..40)
                .map(|i| gaussian(&format!("r{i}"), 0.0, 0.1, (i as f64 - 20.0) / 4.0))
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .per_record;
        let treatment: Vec<PerRecordCalibration> = compute_calibration(
            "t".into(),
            &(0..40)
                .map(|i| gaussian(&format!("r{i}"), 0.0, 5.0, (i as f64 - 20.0) / 4.0))
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .per_record;
        let sig = calibration_significance(&baseline, &treatment).expect("shared ids");
        // Treatment CRPS is lower, so treatment - baseline is negative: the CI
        // sits below zero.
        assert!(
            sig.ci_upper < 0.0,
            "treatment should be significantly better-scored: ci_upper {}",
            sig.ci_upper
        );
    }

    #[test]
    fn crps_significance_none_without_shared_ids() {
        let baseline = compute_calibration("b".into(), &[gaussian("a", 0.0, 1.0, 0.0)])
            .unwrap()
            .per_record;
        let treatment = compute_calibration("t".into(), &[gaussian("z", 0.0, 1.0, 0.0)])
            .unwrap()
            .per_record;
        assert!(calibration_significance(&baseline, &treatment).is_none());
    }
}

#[cfg(test)]
mod significance_tests {
    use super::*;

    /// Build a per-query record carrying only the four scalar metrics under
    /// test; the other fields are irrelevant to significance.
    fn record(query_id: &str, recall: f64, precision: f64, mrr: f64, ndcg: f64) -> PerQueryRecord {
        PerQueryRecord {
            query_id: query_id.to_string(),
            metrics: QueryMetrics {
                recall,
                precision,
                mrr,
                ndcg,
            },
            recall_at_ks: Vec::new(),
            distance: 0.0,
            cohorts: BTreeMap::new(),
        }
    }

    /// A query set where the treatment dominates the baseline on every query
    /// and metric — used to exercise the "significant improvement" path.
    fn improving_pair() -> (Vec<PerQueryRecord>, Vec<PerQueryRecord>) {
        let baseline: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.2, 0.2, 0.2, 0.2))
            .collect();
        let treatment: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.8, 0.8, 0.8, 0.8))
            .collect();
        (baseline, treatment)
    }

    #[test]
    fn pairs_only_shared_query_ids() {
        let baseline = vec![
            record("a", 0.1, 0.1, 0.1, 0.1),
            record("b", 0.2, 0.2, 0.2, 0.2),
        ];
        let treatment = vec![
            record("b", 0.5, 0.5, 0.5, 0.5),
            record("c", 0.9, 0.9, 0.9, 0.9),
        ];
        let paired = paired_metric(&baseline, &treatment, |m| m.recall);
        assert_eq!(paired.len(), 1, "only query 'b' is shared");
        assert_eq!(paired[0].baseline, 0.2);
        assert_eq!(paired[0].treatment, 0.5);
    }

    #[test]
    fn no_shared_queries_yields_none() {
        let baseline = vec![record("a", 0.1, 0.1, 0.1, 0.1)];
        let treatment = vec![record("z", 0.9, 0.9, 0.9, 0.9)];
        assert!(delta_significance(&baseline, &treatment).is_none());
    }

    #[test]
    fn deterministic_under_pinned_seed() {
        let (baseline, treatment) = improving_pair();
        let first = delta_significance(&baseline, &treatment).expect("paired");
        let second = delta_significance(&baseline, &treatment).expect("paired");
        // Bit-for-bit identical: the bootstrap is seeded and the U test is
        // closed-form, so repeated calls on the same input never diverge.
        assert_eq!(first.recall_at_k.p_value, second.recall_at_k.p_value);
        assert_eq!(first.recall_at_k.ci_lower, second.recall_at_k.ci_lower);
        assert_eq!(first.recall_at_k.ci_upper, second.recall_at_k.ci_upper);
        assert_eq!(first.ndcg.ci_upper, second.ndcg.ci_upper);
    }

    #[test]
    fn improvement_is_significant() {
        let (baseline, treatment) = improving_pair();
        let sig = delta_significance(&baseline, &treatment).expect("paired");
        // A uniform +0.6 lift on every query: the paired-difference CI sits
        // entirely above zero and the rank-sum p-value is tiny.
        assert!(
            sig.recall_at_k.ci_lower > 0.0,
            "CI lower bound should exclude zero: {}",
            sig.recall_at_k.ci_lower
        );
        assert!(
            sig.recall_at_k.p_value < 0.01,
            "p-value should be small for a clear lift: {}",
            sig.recall_at_k.p_value
        );
    }

    #[test]
    fn identical_runs_ci_brackets_zero() {
        // Baseline == treatment: every paired difference is exactly zero, so
        // the bootstrap CI collapses to [0, 0] and the distributions are
        // indistinguishable (p ≈ 1).
        let baseline: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.5, 0.5, 0.5, 0.5))
            .collect();
        let treatment = baseline.clone();
        let sig = delta_significance(&baseline, &treatment).expect("paired");
        assert_eq!(sig.mrr.ci_lower, 0.0);
        assert_eq!(sig.mrr.ci_upper, 0.0);
        assert!(
            sig.mrr.p_value > 0.99,
            "identical distributions should be indistinguishable: {}",
            sig.mrr.p_value
        );
    }
}
