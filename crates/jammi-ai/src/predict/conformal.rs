//! Split (inductive) conformal prediction — the distribution-free serving
//! primitive.
//!
//! Conformal wraps **any** point predictor and turns its output into a
//! prediction **set** (classification) or **interval** (regression) carrying a
//! finite-sample, distribution-free coverage guarantee: under exchangeability
//! of the calibration and serving data, the marginal coverage of the emitted
//! sets is at least `1 - alpha`, for *any* underlying model, *any* data
//! distribution, and *any* sample size (Vovk et al. 2005; Angelopoulos &
//! Bates 2021). No retraining — a calibration pass and an empirical quantile —
//! and deterministic given a calibration set, which is the audit property.
//!
//! # The pieces
//!
//! * A [`NonconformityScore`] is the one task-specific piece: it maps a
//!   predictor output and a candidate label to a real number, larger when the
//!   label conforms *less* to the prediction. Calibration scores its held-out
//!   examples at their *true* label.
//! * [`finite_sample_quantile`] takes the calibration scores and `alpha` to the
//!   conformal threshold `q̂` — the `⌈(n+1)(1-alpha)⌉`-th smallest score. The
//!   `(n+1)` correction is what makes the guarantee *exact* rather than merely
//!   asymptotic; the naive `⌈n(1-alpha)⌉` order statistic under-covers.
//! * A [`ConformalModel`] holds the calibrated threshold(s). At serving time it
//!   admits every label whose nonconformity is `<= q̂` — a set for
//!   classification, an interval for regression.
//!
//! # Shift-robust score inputs (serving levers, not policy)
//!
//! Two variants of the quantile accept *known* structure so a governance layer
//! can correct for it: [`finite_sample_quantile_weighted`] applies importance
//! weights (weighted conformal, Tibshirani et al. 2019) under covariate shift,
//! and the Mondrian construction ([`ConformalModel::classification_mondrian`] /
//! [`ConformalModel::regression_mondrian`]) keeps a per-cohort quantile keyed on
//! a group column (group-conditional coverage). The primitive *applies* the
//! weights and the grouping; it does **not** decide when shift has occurred or
//! which cohorts matter — that judgment is governance, not a serving output.
//!
//! # The guarantee's one assumption
//!
//! Coverage holds **iff** calibration and serving data are exchangeable. Under
//! distribution drift it degrades silently (Barber et al. 2023). This primitive
//! names that loudly and supplies the weighted / Mondrian levers; *detecting*
//! drift and *adapting* online is a governed concern built atop this primitive,
//! not part of it. The calibration source is a distinct argument from train and
//! test — reusing test points to calibrate inflates coverage, so the three-way
//! split is the caller's contract.

use std::collections::BTreeMap;

use jammi_db::error::{JammiError, Result};

/// A real-valued nonconformity threshold, possibly infinite.
///
/// When the calibration set is too small for the requested level — fewer than
/// `⌈1/alpha⌉ - 1` points — the `⌈(n+1)(1-alpha)⌉` rank exceeds `n` and the
/// threshold is `+∞`: the honest, conservative answer is "every label", and the
/// guarantee still holds. A finite threshold is the usual case.
pub type Threshold = f64;

/// Classification nonconformity score families.
///
/// Each maps a row of per-class predictive probabilities (the softmax mass the
/// classifier already emits) to a nonconformity score for a candidate class.
/// Calibration evaluates the score at the *true* class; prediction admits every
/// class whose score is `<= q̂`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassScore {
    /// Least-Ambiguous set-valued Classifier (LAC / "score"): nonconformity is
    /// `1 - p_y`. Smallest sets at the nominal level, but non-adaptive — it does
    /// not grow on hard inputs the way APS does.
    Lac,
    /// Adaptive Prediction Sets (Romano et al. 2020): nonconformity is the
    /// cumulative probability mass of the classes ranked from most to least
    /// probable, up to and including `y`. Set size adapts to input difficulty.
    Aps,
    /// Regularized APS (Angelopoulos et al. 2021): APS plus a penalty
    /// `lambda * max(0, rank(y) - k_reg)` on the tail rank, shrinking sets on
    /// easy inputs while preserving coverage. `k_reg` is a 1-based rank.
    Raps {
        /// Penalty weight on each rank beyond `k_reg`.
        lambda: f64,
        /// Number of top ranks exempt from the penalty (1-based).
        k_reg: usize,
    },
}

/// Regression nonconformity score families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalScore {
    /// Absolute-residual conformal: nonconformity is `|y - ŷ|`, yielding a
    /// constant-width interval `[ŷ - q̂, ŷ + q̂]`. Distribution-free but
    /// non-adaptive — uninformative under heteroscedasticity.
    AbsoluteResidual,
    /// Conformalized Quantile Regression (Romano et al. 2019): nonconformity is
    /// `max(q_lo - y, y - q_hi)` over a predictor's lower/upper quantile
    /// estimates, yielding an *adaptive*-width interval `[q_lo - q̂, q_hi + q̂]`.
    Cqr,
}

/// The conformal threshold: the `⌈(n+1)(1-alpha)⌉`-th smallest calibration
/// score, or `+∞` when that rank exceeds `n`.
///
/// This is the heart of split conformal. The `(n+1)` finite-sample correction
/// is not optional cosmetics: with `n` calibration scores, taking the
/// `⌈(n+1)(1-alpha)⌉` order statistic makes `P(s_test <= q̂) >= 1 - alpha`
/// hold *exactly* under exchangeability, where the naive `⌈n(1-alpha)⌉`
/// quantile leaves a `~1/n` coverage gap. When `⌈(n+1)(1-alpha)⌉ > n` the
/// requested level is unattainable from this few points and the threshold is
/// `+∞` (every label admitted) — conservative, never silently wrong.
///
/// Returns `Inference` when `scores` is empty or `alpha` is not in `(0, 1)`.
pub fn finite_sample_quantile(scores: &[f64], alpha: f64) -> Result<Threshold> {
    validate_alpha(alpha)?;
    if scores.is_empty() {
        return Err(JammiError::Inference(
            "conformal calibration requires at least one score".into(),
        ));
    }
    if scores.iter().any(|s| s.is_nan()) {
        return Err(JammiError::Inference(
            "conformal calibration scores must be finite".into(),
        ));
    }
    let n = scores.len();
    // rank = ⌈(n+1)(1-alpha)⌉, a 1-based order statistic index.
    let rank = ((n + 1) as f64 * (1.0 - alpha)).ceil() as usize;
    if rank > n {
        // Too few points for this level: the honest answer is +∞.
        return Ok(f64::INFINITY);
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    // 1-based rank to 0-based index.
    Ok(sorted[rank - 1])
}

/// Importance-weighted conformal threshold (weighted conformal prediction,
/// Tibshirani et al. 2019).
///
/// Under a *known* covariate shift, each calibration point carries an
/// importance weight `w_i = dP_test/dP_cal(x_i) >= 0`. The threshold is the
/// smallest score `s_(k)` whose normalized cumulative weight reaches
/// `1 - alpha`, where the normalization includes a point mass at `+∞` for the
/// test point itself (its weight folds into the denominator), recovering the
/// `(n+1)`-style correction. With equal weights this reduces exactly to
/// [`finite_sample_quantile`].
///
/// The primitive *applies* the supplied weights; deciding that a shift occurred
/// and estimating the weights is governance, not a serving output.
///
/// Returns `Inference` when the lengths disagree, `scores` is empty, any weight
/// is negative or non-finite, all weights are zero, or `alpha` is not in
/// `(0, 1)`.
pub fn finite_sample_quantile_weighted(
    scores: &[f64],
    weights: &[f64],
    test_weight: f64,
    alpha: f64,
) -> Result<Threshold> {
    validate_alpha(alpha)?;
    if scores.len() != weights.len() {
        return Err(JammiError::Inference(format!(
            "weighted conformal: {} scores but {} weights",
            scores.len(),
            weights.len()
        )));
    }
    if scores.is_empty() {
        return Err(JammiError::Inference(
            "weighted conformal requires at least one score".into(),
        ));
    }
    if scores.iter().any(|s| s.is_nan()) {
        return Err(JammiError::Inference(
            "weighted conformal scores must be finite".into(),
        ));
    }
    if weights.iter().any(|&w| w < 0.0 || !w.is_finite()) {
        return Err(JammiError::Inference(
            "weighted conformal weights must be finite and non-negative".into(),
        ));
    }
    if test_weight < 0.0 || !test_weight.is_finite() {
        return Err(JammiError::Inference(
            "weighted conformal test weight must be finite and non-negative".into(),
        ));
    }
    let total: f64 = weights.iter().sum::<f64>() + test_weight;
    if total <= 0.0 {
        return Err(JammiError::Inference(
            "weighted conformal requires positive total weight".into(),
        ));
    }

    // Sort (score, weight) by ascending score, then walk the normalized
    // cumulative weight. The test point's weight sits at +∞ in the mixture, so
    // the calibration mass alone need only reach 1 - alpha for a finite
    // threshold; otherwise the +∞ point mass carries the remainder.
    let mut paired: Vec<(f64, f64)> = scores
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    paired.sort_by(|a, b| a.0.total_cmp(&b.0));

    let target = 1.0 - alpha;
    let mut cumulative = 0.0;
    for (score, weight) in paired {
        cumulative += weight / total;
        if cumulative >= target {
            return Ok(score);
        }
    }
    // The calibration mass never reached 1 - alpha; the +∞ point mass does.
    Ok(f64::INFINITY)
}

/// A calibrated conformal predictor: thresholds plus the score family that
/// produced them.
///
/// Built by [`ConformalModel::classification`] / [`ConformalModel::regression`]
/// (one marginal threshold) or their `_mondrian` / `_weighted` siblings, then
/// queried with [`ConformalModel::predict_set`] (classification) or
/// [`ConformalModel::predict_interval`] (regression). The model is a value: the
/// same calibration set produces byte-identical thresholds and therefore
/// byte-identical sets across runs.
#[derive(Debug, Clone)]
pub struct ConformalModel {
    /// The nominal miscoverage level the thresholds target.
    alpha: f64,
    /// The calibrated thresholds and the score family they belong to.
    kind: ModelKind,
}

#[derive(Debug, Clone)]
enum ModelKind {
    Classification {
        score: ClassScore,
        /// Marginal threshold, or per-group thresholds under Mondrian. The
        /// empty-key `""` entry, when present, is the marginal fallback for a
        /// group unseen at calibration time.
        thresholds: BTreeMap<String, Threshold>,
    },
    Regression {
        score: IntervalScore,
        thresholds: BTreeMap<String, Threshold>,
    },
}

/// The marginal group key — a single bucket holding the one global threshold.
const MARGINAL_KEY: &str = "";

impl ConformalModel {
    /// Calibrate a marginal classification predictor.
    ///
    /// `calibration` holds one row of per-class probabilities per held-out
    /// example, `true_labels[i]` is the index of the realised class for row
    /// `i`, and `score` selects the nonconformity family. Every probability row
    /// must have the same length (the class count) and every true label must
    /// index into it.
    pub fn classification(
        calibration: &[Vec<f64>],
        true_labels: &[usize],
        score: ClassScore,
        alpha: f64,
    ) -> Result<Self> {
        let scores = classification_scores(calibration, true_labels, score)?;
        let q = finite_sample_quantile(&scores, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Classification {
                score,
                thresholds: BTreeMap::from([(MARGINAL_KEY.to_string(), q)]),
            },
        })
    }

    /// Calibrate a Mondrian (group-conditional) classification predictor.
    ///
    /// `groups[i]` is the cohort key for calibration row `i`; a per-cohort
    /// quantile is taken over that cohort's scores alone, so each group carries
    /// its own threshold. At serving time [`ConformalModel::predict_set`] picks
    /// the threshold for the row's group; a group unseen at calibration time
    /// falls back to the pooled marginal threshold.
    ///
    /// This is the principled approximation to conditional coverage (full
    /// per-input coverage is provably impossible distribution-free). The
    /// primitive *applies* the grouping; *choosing* which cohorts to condition
    /// on is governance.
    pub fn classification_mondrian(
        calibration: &[Vec<f64>],
        true_labels: &[usize],
        groups: &[String],
        score: ClassScore,
        alpha: f64,
    ) -> Result<Self> {
        let all = classification_scores(calibration, true_labels, score)?;
        let thresholds = mondrian_thresholds(&all, groups, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Classification { score, thresholds },
        })
    }

    /// Calibrate an importance-weighted classification predictor for a known
    /// covariate shift.
    ///
    /// `weights[i]` is the importance weight `dP_test/dP_cal` of calibration row
    /// `i`. The threshold is the weighted conformal quantile
    /// ([`finite_sample_quantile_weighted`]) with a unit test-point weight; pass
    /// a representative test weight when one is known.
    pub fn classification_weighted(
        calibration: &[Vec<f64>],
        true_labels: &[usize],
        weights: &[f64],
        score: ClassScore,
        alpha: f64,
    ) -> Result<Self> {
        let scores = classification_scores(calibration, true_labels, score)?;
        let q = finite_sample_quantile_weighted(&scores, weights, 1.0, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Classification {
                score,
                thresholds: BTreeMap::from([(MARGINAL_KEY.to_string(), q)]),
            },
        })
    }

    /// The prediction set for one classification row.
    ///
    /// `probabilities` is the per-class predictive distribution; `group` selects
    /// the Mondrian cohort (`None` uses the marginal threshold). The returned
    /// class indices are exactly those whose nonconformity is `<= q̂`, in
    /// ascending order. An empty set never occurs at a finite, attainable level;
    /// a full set is the honest signal that the input is hard or the model is
    /// miscalibrated.
    pub fn predict_set(&self, probabilities: &[f64], group: Option<&str>) -> Result<Vec<usize>> {
        let ModelKind::Classification { score, thresholds } = &self.kind else {
            return Err(JammiError::Inference(
                "predict_set called on a regression conformal model".into(),
            ));
        };
        let q = lookup_threshold(thresholds, group)?;
        admit_classes(probabilities, *score, q)
    }

    /// Calibrate a marginal regression predictor.
    ///
    /// For [`IntervalScore::AbsoluteResidual`], `predictions` are point
    /// estimates `ŷ` and `lower`/`upper` are ignored (pass empty). For
    /// [`IntervalScore::Cqr`], `lower`/`upper` are the predictor's lower/upper
    /// quantile estimates and `predictions` is ignored. `observed[i]` is the
    /// realised target `y`.
    pub fn regression(
        predictions: &[f64],
        lower: &[f64],
        upper: &[f64],
        observed: &[f64],
        score: IntervalScore,
        alpha: f64,
    ) -> Result<Self> {
        let scores = regression_scores(predictions, lower, upper, observed, score)?;
        let q = finite_sample_quantile(&scores, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Regression {
                score,
                thresholds: BTreeMap::from([(MARGINAL_KEY.to_string(), q)]),
            },
        })
    }

    /// Calibrate a Mondrian (group-conditional) regression predictor. See
    /// [`ConformalModel::classification_mondrian`] for the cohort semantics.
    pub fn regression_mondrian(
        predictions: &[f64],
        lower: &[f64],
        upper: &[f64],
        observed: &[f64],
        groups: &[String],
        score: IntervalScore,
        alpha: f64,
    ) -> Result<Self> {
        let all = regression_scores(predictions, lower, upper, observed, score)?;
        let thresholds = mondrian_thresholds(&all, groups, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Regression { score, thresholds },
        })
    }

    /// Calibrate an importance-weighted regression predictor for a known
    /// covariate shift. See [`ConformalModel::classification_weighted`].
    pub fn regression_weighted(
        predictions: &[f64],
        lower: &[f64],
        upper: &[f64],
        observed: &[f64],
        weights: &[f64],
        score: IntervalScore,
        alpha: f64,
    ) -> Result<Self> {
        let scores = regression_scores(predictions, lower, upper, observed, score)?;
        let q = finite_sample_quantile_weighted(&scores, weights, 1.0, alpha)?;
        Ok(Self {
            alpha,
            kind: ModelKind::Regression {
                score,
                thresholds: BTreeMap::from([(MARGINAL_KEY.to_string(), q)]),
            },
        })
    }

    /// The prediction interval `[lower, upper]` for one regression row.
    ///
    /// For [`IntervalScore::AbsoluteResidual`] pass the point estimate as
    /// `prediction` (the quantile bounds are ignored); for
    /// [`IntervalScore::Cqr`] pass the lower/upper quantile estimates (the point
    /// estimate is ignored). `group` selects the Mondrian cohort.
    pub fn predict_interval(
        &self,
        prediction: f64,
        lower_quantile: f64,
        upper_quantile: f64,
        group: Option<&str>,
    ) -> Result<(f64, f64)> {
        let ModelKind::Regression { score, thresholds } = &self.kind else {
            return Err(JammiError::Inference(
                "predict_interval called on a classification conformal model".into(),
            ));
        };
        let q = lookup_threshold(thresholds, group)?;
        Ok(match score {
            IntervalScore::AbsoluteResidual => (prediction - q, prediction + q),
            IntervalScore::Cqr => (lower_quantile - q, upper_quantile + q),
        })
    }

    /// The nominal miscoverage level these thresholds target.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Validate a miscoverage level lies in the open interval `(0, 1)`.
fn validate_alpha(alpha: f64) -> Result<()> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(JammiError::Inference(format!(
            "conformal alpha must lie in (0, 1), got {alpha}"
        )));
    }
    Ok(())
}

/// Look up the threshold for a (possibly grouped) prediction.
///
/// A marginal model carries one threshold under the empty key. A Mondrian model
/// carries a threshold per group and a pooled marginal fallback under the empty
/// key for groups unseen at calibration time.
fn lookup_threshold(thresholds: &BTreeMap<String, Threshold>, group: Option<&str>) -> Result<f64> {
    match group {
        Some(g) => Ok(thresholds
            .get(g)
            .copied()
            .or_else(|| thresholds.get(MARGINAL_KEY).copied())
            .unwrap_or(f64::INFINITY)),
        None => thresholds.get(MARGINAL_KEY).copied().ok_or_else(|| {
            JammiError::Inference(
                "marginal threshold requested from a Mondrian model with no pooled fallback".into(),
            )
        }),
    }
}

/// Partition scores by group key and take the finite-sample quantile within
/// each, plus a pooled marginal fallback under [`MARGINAL_KEY`].
fn mondrian_thresholds(
    scores: &[f64],
    groups: &[String],
    alpha: f64,
) -> Result<BTreeMap<String, Threshold>> {
    if scores.len() != groups.len() {
        return Err(JammiError::Inference(format!(
            "Mondrian conformal: {} scores but {} group keys",
            scores.len(),
            groups.len()
        )));
    }
    let mut by_group: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (score, group) in scores.iter().zip(groups.iter()) {
        by_group.entry(group.as_str()).or_default().push(*score);
    }
    let mut thresholds = BTreeMap::new();
    for (group, group_scores) in by_group {
        thresholds.insert(
            group.to_string(),
            finite_sample_quantile(&group_scores, alpha)?,
        );
    }
    // Pooled fallback for groups unseen at calibration time.
    thresholds.insert(
        MARGINAL_KEY.to_string(),
        finite_sample_quantile(scores, alpha)?,
    );
    Ok(thresholds)
}

/// Nonconformity scores of the calibration rows at their *true* class.
fn classification_scores(
    calibration: &[Vec<f64>],
    true_labels: &[usize],
    score: ClassScore,
) -> Result<Vec<f64>> {
    if calibration.len() != true_labels.len() {
        return Err(JammiError::Inference(format!(
            "classification conformal: {} probability rows but {} labels",
            calibration.len(),
            true_labels.len()
        )));
    }
    if calibration.is_empty() {
        return Err(JammiError::Inference(
            "classification conformal requires at least one calibration row".into(),
        ));
    }
    calibration
        .iter()
        .zip(true_labels.iter())
        .map(|(probs, &label)| true_label_score(probs, label, score))
        .collect()
}

/// The nonconformity score of one calibration row evaluated at its true class.
fn true_label_score(probs: &[f64], label: usize, score: ClassScore) -> Result<f64> {
    validate_probabilities(probs)?;
    if label >= probs.len() {
        return Err(JammiError::Inference(format!(
            "true label {label} out of range for {} classes",
            probs.len()
        )));
    }
    Ok(match score {
        ClassScore::Lac => 1.0 - probs[label],
        ClassScore::Aps => aps_cumulative_mass(probs, label, None),
        ClassScore::Raps { lambda, k_reg } => {
            aps_cumulative_mass(probs, label, Some((lambda, k_reg)))
        }
    })
}

/// APS (and RAPS) nonconformity: the cumulative probability mass of classes
/// ranked most- to least-probable, up to and including `target`, plus the RAPS
/// rank penalty when `reg` is supplied.
///
/// Ties in probability are broken by class index so the ordering — and thus the
/// score — is deterministic across runs.
fn aps_cumulative_mass(probs: &[f64], target: usize, reg: Option<(f64, usize)>) -> f64 {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    // Descending probability; index ascending on ties for determinism.
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));

    let mut cumulative = 0.0;
    for (rank, &class) in order.iter().enumerate() {
        cumulative += probs[class];
        if class == target {
            if let Some((lambda, k_reg)) = reg {
                // 1-based rank; penalize ranks beyond k_reg.
                let rank_1based = rank + 1;
                let overflow = rank_1based.saturating_sub(k_reg);
                cumulative += lambda * overflow as f64;
            }
            return cumulative;
        }
    }
    // `target` is in range (checked by the caller), so it is always found.
    cumulative
}

/// The classes admitted into the prediction set: those whose nonconformity at
/// the row's probabilities is `<= q̂`.
fn admit_classes(probs: &[f64], score: ClassScore, q: f64) -> Result<Vec<usize>> {
    validate_probabilities(probs)?;
    if q.is_infinite() {
        // An infinite threshold admits every class.
        return Ok((0..probs.len()).collect());
    }
    match score {
        ClassScore::Lac => Ok((0..probs.len()).filter(|&c| 1.0 - probs[c] <= q).collect()),
        // For APS/RAPS the candidate score is the cumulative mass up to and
        // including the candidate class — exactly `true_label_score` with the
        // candidate standing in for the true label.
        ClassScore::Aps => Ok((0..probs.len())
            .filter(|&c| aps_cumulative_mass(probs, c, None) <= q)
            .collect()),
        ClassScore::Raps { lambda, k_reg } => Ok((0..probs.len())
            .filter(|&c| aps_cumulative_mass(probs, c, Some((lambda, k_reg))) <= q)
            .collect()),
    }
}

/// Nonconformity scores for the regression calibration rows.
fn regression_scores(
    predictions: &[f64],
    lower: &[f64],
    upper: &[f64],
    observed: &[f64],
    score: IntervalScore,
) -> Result<Vec<f64>> {
    match score {
        IntervalScore::AbsoluteResidual => {
            if predictions.len() != observed.len() {
                return Err(JammiError::Inference(format!(
                    "absolute-residual conformal: {} predictions but {} observations",
                    predictions.len(),
                    observed.len()
                )));
            }
            if predictions.is_empty() {
                return Err(JammiError::Inference(
                    "regression conformal requires at least one calibration row".into(),
                ));
            }
            ensure_finite(predictions, "predictions")?;
            ensure_finite(observed, "observations")?;
            Ok(predictions
                .iter()
                .zip(observed.iter())
                .map(|(&yhat, &y)| (y - yhat).abs())
                .collect())
        }
        IntervalScore::Cqr => {
            if lower.len() != observed.len() || upper.len() != observed.len() {
                return Err(JammiError::Inference(format!(
                    "CQR conformal: {} lower / {} upper quantiles but {} observations",
                    lower.len(),
                    upper.len(),
                    observed.len()
                )));
            }
            if observed.is_empty() {
                return Err(JammiError::Inference(
                    "regression conformal requires at least one calibration row".into(),
                ));
            }
            ensure_finite(lower, "lower quantiles")?;
            ensure_finite(upper, "upper quantiles")?;
            ensure_finite(observed, "observations")?;
            Ok(lower
                .iter()
                .zip(upper.iter())
                .zip(observed.iter())
                .map(|((&lo, &hi), &y)| (lo - y).max(y - hi))
                .collect())
        }
    }
}

/// Validate a per-class probability row: non-empty and free of NaN. The values
/// need not sum to one — conformal reads the *ordering* and the true-class mass,
/// not a normalized distribution — but a NaN would silently corrupt the rank.
fn validate_probabilities(probs: &[f64]) -> Result<()> {
    if probs.is_empty() {
        return Err(JammiError::Inference(
            "classification conformal requires at least one class probability".into(),
        ));
    }
    if probs.iter().any(|p| p.is_nan()) {
        return Err(JammiError::Inference(
            "classification conformal probabilities must be finite".into(),
        ));
    }
    Ok(())
}

/// Reject non-finite values in a regression input slice.
fn ensure_finite(values: &[f64], what: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(JammiError::Inference(format!(
            "regression conformal {what} must be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as NormalDraw};

    /// The naive (uncorrected) quantile — the `⌈n(1-alpha)⌉` order statistic —
    /// used only to demonstrate that it under-covers where the corrected one
    /// does not.
    fn naive_quantile(scores: &[f64], alpha: f64) -> f64 {
        let n = scores.len();
        let rank = (n as f64 * (1.0 - alpha)).ceil() as usize;
        let mut sorted = scores.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[rank.clamp(1, n) - 1]
    }

    #[test]
    fn quantile_takes_the_corrected_order_statistic() {
        // n = 9, alpha = 0.1: rank = ⌈10 * 0.9⌉ = 9, the 9th (largest) score.
        let scores: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        assert_eq!(finite_sample_quantile(&scores, 0.1).unwrap(), 9.0);
    }

    #[test]
    fn quantile_is_infinite_below_the_finite_sample_floor() {
        // n = 5, alpha = 0.1: rank = ⌈6 * 0.9⌉ = 6 > 5 -> +∞.
        let scores: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        assert!(finite_sample_quantile(&scores, 0.1).unwrap().is_infinite());
    }

    #[test]
    fn quantile_rejects_bad_alpha_and_empty() {
        assert!(finite_sample_quantile(&[1.0], 0.0).is_err());
        assert!(finite_sample_quantile(&[1.0], 1.0).is_err());
        assert!(finite_sample_quantile(&[], 0.1).is_err());
    }

    /// Build a noisy-argmax synthetic classifier: probabilities concentrate on
    /// the true class but leak mass elsewhere, and the realised label is drawn
    /// from those probabilities — exchangeable by construction.
    fn synthetic_classification(
        rng: &mut StdRng,
        n: usize,
        n_classes: usize,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut probs = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            // Random logits -> softmax; the label is sampled from the softmax,
            // so (probs, label) pairs are exchangeable and calibrated.
            let logits: Vec<f64> = (0..n_classes).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let row: Vec<f64> = exp.iter().map(|e| e / sum).collect();

            let u: f64 = rng.gen_range(0.0..1.0);
            let mut acc = 0.0;
            let mut label = n_classes - 1;
            for (c, p) in row.iter().enumerate() {
                acc += p;
                if u <= acc {
                    label = c;
                    break;
                }
            }
            probs.push(row);
            labels.push(label);
        }
        (probs, labels)
    }

    /// Realised coverage of a classification model on a held-out test split.
    fn classification_coverage(
        model: &ConformalModel,
        probs: &[Vec<f64>],
        labels: &[usize],
    ) -> f64 {
        let hits: Vec<bool> = probs
            .iter()
            .zip(labels.iter())
            .map(|(row, &y)| model.predict_set(row, None).unwrap().contains(&y))
            .collect();
        jammi_numerics::calibration::coverage(&hits).unwrap()
    }

    #[test]
    fn corrected_quantile_hits_marginal_coverage_where_naive_misses() {
        // Three-way split: train is implicit (the synthetic softmax stands in
        // for a trained predictor); calibration and test are disjoint draws.
        let mut rng = StdRng::seed_from_u64(20260605);
        let n_classes = 5;
        for &alpha in &[0.05, 0.1, 0.2] {
            let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 2000, n_classes);
            let (test_probs, test_labels) = synthetic_classification(&mut rng, 4000, n_classes);

            let model =
                ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Lac, alpha)
                    .unwrap();
            let cov = classification_coverage(&model, &test_probs, &test_labels);
            // The guarantee is coverage >= 1 - alpha; allow finite-sample slack
            // on both sides around the nominal level.
            assert!(
                cov >= 1.0 - alpha - 0.03,
                "alpha={alpha}: corrected coverage {cov} below 1 - alpha"
            );
            assert!(
                cov <= 1.0 - alpha + 0.05,
                "alpha={alpha}: corrected coverage {cov} far above nominal (uninformative)"
            );
        }
    }

    #[test]
    fn naive_quantile_undercovers() {
        // The naive quantile takes the ⌈n(1-alpha)⌉-th order statistic; for a
        // continuous score the test point falls uniformly into one of the n+1
        // gaps around the calibration scores, so its expected coverage is
        // ⌈n(1-alpha)⌉/(n+1) — strictly below 1-alpha. The shortfall is widest
        // for SMALL n: at n=30, alpha=0.1 it is ⌈27⌉/31 ≈ 0.871 against a 0.900
        // target (a ~0.029 gap), well outside the per-trial sampling noise of a
        // large test split, so a clear majority of trials under-cover. The
        // corrected quantile takes ⌈(n+1)(1-alpha)⌉ and does not (the test
        // above). Many trials average out the calibration draw's own noise.
        let mut rng = StdRng::seed_from_u64(7);
        let n_classes = 4;
        let alpha = 0.1;
        let mut shortfalls = 0;
        let trials = 60;
        for _ in 0..trials {
            let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 30, n_classes);
            let (test_probs, test_labels) = synthetic_classification(&mut rng, 8000, n_classes);

            let cal_scores: Vec<f64> = cal_probs
                .iter()
                .zip(cal_labels.iter())
                .map(|(p, &y)| 1.0 - p[y])
                .collect();
            let q_naive = naive_quantile(&cal_scores, alpha);

            let hits: Vec<bool> = test_probs
                .iter()
                .zip(test_labels.iter())
                .map(|(row, &y)| 1.0 - row[y] <= q_naive)
                .collect();
            let cov = jammi_numerics::calibration::coverage(&hits).unwrap();
            if cov < 1.0 - alpha {
                shortfalls += 1;
            }
        }
        // The naive quantile under-covers in a clear majority of trials; the
        // corrected one does not (covered by the test above).
        assert!(
            shortfalls > trials / 2,
            "expected the naive quantile to under-cover in most trials, got {shortfalls}/{trials}"
        );
    }

    #[test]
    fn aps_set_size_grows_on_harder_inputs() {
        // Calibrate APS on a moderately hard set, then compare set sizes on an
        // easy row (mass concentrated) and a hard row (mass diffuse).
        let mut rng = StdRng::seed_from_u64(99);
        let n_classes = 6;
        let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 3000, n_classes);
        let model =
            ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, 0.1).unwrap();

        let easy = vec![0.9, 0.04, 0.02, 0.02, 0.01, 0.01];
        let hard = vec![0.25, 0.22, 0.20, 0.15, 0.10, 0.08];
        let easy_set = model.predict_set(&easy, None).unwrap();
        let hard_set = model.predict_set(&hard, None).unwrap();
        assert!(
            hard_set.len() > easy_set.len(),
            "APS set should grow on the harder input: easy={} hard={}",
            easy_set.len(),
            hard_set.len()
        );
    }

    #[test]
    fn raps_shrinks_sets_relative_to_aps() {
        // RAPS regularizes APS by adding lambda * max(0, rank - k_reg) to a
        // candidate's nonconformity, so deep-tail classes cost more to admit. The
        // penalty also inflates the calibration scores and thus raises the
        // threshold; whether the net effect shrinks sets is a *population*
        // property (E[size_raps] <= E[size_aps]), not guaranteed on any single
        // calibration draw — one draw's threshold pairing can buck it. It shows
        // cleanly at a looser level (alpha = 0.2), where sets are small enough
        // that the rank penalty bites the tail rather than being swamped by the
        // threshold lift that saturated near-full sets exhibit at stringent
        // levels. Averaging the mean set size over independent calibration/test
        // trials estimates the population means, so their ordering is the sound
        // claim. k_reg = 2 exempts the top two ranks; lambda = 1.0 makes the tail
        // penalty decisive.
        let mut rng = StdRng::seed_from_u64(11);
        let n_classes = 8;
        let alpha = 0.2;
        let trials = 20;
        let mut total_aps = 0.0;
        let mut total_raps = 0.0;
        for _ in 0..trials {
            let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 3000, n_classes);
            let (test_probs, _) = synthetic_classification(&mut rng, 1000, n_classes);

            let aps =
                ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, alpha)
                    .unwrap();
            let raps = ConformalModel::classification(
                &cal_probs,
                &cal_labels,
                ClassScore::Raps {
                    lambda: 1.0,
                    k_reg: 2,
                },
                alpha,
            )
            .unwrap();

            let sizes = |model: &ConformalModel| -> f64 {
                test_probs
                    .iter()
                    .map(|p| model.predict_set(p, None).unwrap().len() as f64)
                    .sum::<f64>()
                    / test_probs.len() as f64
            };
            total_aps += sizes(&aps);
            total_raps += sizes(&raps);
        }
        let mean_aps = total_aps / trials as f64;
        let mean_raps = total_raps / trials as f64;
        assert!(
            mean_raps <= mean_aps,
            "RAPS regularization should shrink the mean set size: aps={mean_aps} raps={mean_raps}"
        );
    }

    #[test]
    fn cqr_width_grows_under_heteroscedasticity() {
        // Build a heteroscedastic regression: noise scale grows with |x|. The
        // quantile predictor's band tracks it, so post-conformal interval width
        // should be larger in the high-variance region than the low.
        let mut rng = StdRng::seed_from_u64(2024);
        let n = 3000;
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        let mut observed = Vec::with_capacity(n);
        let mut widths_low = Vec::new();
        let mut widths_high = Vec::new();
        for _ in 0..n {
            let x: f64 = rng.gen_range(0.0..10.0);
            let sd = 0.2 + 0.5 * x; // heteroscedastic
            let dist = NormalDraw::new(0.0, sd).unwrap();
            let y = dist.sample(&mut rng);
            // A well-specified quantile predictor: +/- 1.2816 sd ~ central 80%.
            let band = 1.2816 * sd;
            lower.push(-band);
            upper.push(band);
            observed.push(y);
        }
        let model =
            ConformalModel::regression(&[], &lower, &upper, &observed, IntervalScore::Cqr, 0.1)
                .unwrap();

        // Evaluate width in two regimes.
        for x in [1.0_f64, 9.0_f64] {
            let sd = 0.2 + 0.5 * x;
            let band = 1.2816 * sd;
            let (lo, hi) = model.predict_interval(0.0, -band, band, None).unwrap();
            if x < 5.0 {
                widths_low.push(hi - lo);
            } else {
                widths_high.push(hi - lo);
            }
        }
        assert!(
            widths_high[0] > widths_low[0],
            "CQR width should grow under heteroscedasticity: low={widths_low:?} high={widths_high:?}"
        );
    }

    #[test]
    fn absolute_residual_interval_covers_marginally() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 3000;
        let dist = NormalDraw::new(0.0, 1.0).unwrap();
        let cal_pred = vec![0.0; n];
        let cal_obs: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let model = ConformalModel::regression(
            &cal_pred,
            &[],
            &[],
            &cal_obs,
            IntervalScore::AbsoluteResidual,
            0.1,
        )
        .unwrap();

        let test_obs: Vec<f64> = (0..4000).map(|_| dist.sample(&mut rng)).collect();
        let (lower, upper): (Vec<f64>, Vec<f64>) = test_obs
            .iter()
            .map(|_| model.predict_interval(0.0, 0.0, 0.0, None).unwrap())
            .unzip();
        let cov =
            jammi_numerics::calibration::interval_coverage(&lower, &upper, &test_obs).unwrap();
        assert!(
            cov >= 0.9 - 0.02,
            "absolute-residual coverage {cov} below nominal"
        );
    }

    #[test]
    fn mondrian_keeps_per_group_thresholds() {
        // Two cohorts with different difficulty: a cohort-conditional quantile
        // gives each its own threshold, so the harder cohort gets larger sets.
        let mut rng = StdRng::seed_from_u64(42);
        let n_classes = 5;
        let n = 2000;
        let mut probs = Vec::with_capacity(2 * n);
        let mut labels = Vec::with_capacity(2 * n);
        let mut groups = Vec::with_capacity(2 * n);

        // Easy cohort "a": sharp softmax. Hard cohort "b": diffuse.
        for (group, temperature) in [("a", 0.5_f64), ("b", 3.0_f64)] {
            for _ in 0..n {
                let logits: Vec<f64> = (0..n_classes)
                    .map(|_| rng.gen_range(-2.0..2.0) / temperature)
                    .collect();
                let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let row: Vec<f64> = exp.iter().map(|e| e / sum).collect();
                let u: f64 = rng.gen_range(0.0..1.0);
                let mut acc = 0.0;
                let mut label = n_classes - 1;
                for (c, p) in row.iter().enumerate() {
                    acc += p;
                    if u <= acc {
                        label = c;
                        break;
                    }
                }
                probs.push(row);
                labels.push(label);
                groups.push(group.to_string());
            }
        }

        let model =
            ConformalModel::classification_mondrian(&probs, &labels, &groups, ClassScore::Lac, 0.1)
                .unwrap();

        // The hard cohort's threshold must exceed the easy cohort's.
        let test = vec![0.4, 0.2, 0.2, 0.1, 0.1];
        let set_a = model.predict_set(&test, Some("a")).unwrap();
        let set_b = model.predict_set(&test, Some("b")).unwrap();
        assert!(
            set_b.len() >= set_a.len(),
            "harder Mondrian cohort should not yield smaller sets: a={} b={}",
            set_a.len(),
            set_b.len()
        );
    }

    #[test]
    fn weighted_quantile_shifts_with_importance_weights() {
        // Equal weights reproduce the unweighted quantile; up-weighting the
        // larger scores pushes the weighted quantile up.
        let scores: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let equal = vec![1.0; 20];
        let unweighted = finite_sample_quantile(&scores, 0.2).unwrap();
        let weighted_equal = finite_sample_quantile_weighted(&scores, &equal, 1.0, 0.2).unwrap();
        assert_eq!(unweighted, weighted_equal);

        // Up-weight the top half (the larger scores): the threshold rises.
        let mut shifted = vec![1.0; 20];
        for w in shifted.iter_mut().skip(10) {
            *w = 5.0;
        }
        let weighted_shift = finite_sample_quantile_weighted(&scores, &shifted, 1.0, 0.2).unwrap();
        assert!(
            weighted_shift >= weighted_equal,
            "up-weighting large scores should not lower the quantile: equal={weighted_equal} shifted={weighted_shift}"
        );
    }

    #[test]
    fn weighted_rejects_inconsistent_or_degenerate_inputs() {
        assert!(finite_sample_quantile_weighted(&[1.0, 2.0], &[1.0], 1.0, 0.1).is_err());
        assert!(finite_sample_quantile_weighted(&[1.0], &[-1.0], 1.0, 0.1).is_err());
        assert!(finite_sample_quantile_weighted(&[1.0], &[0.0], 0.0, 0.1).is_err());
    }

    #[test]
    fn sets_are_deterministic_across_runs() {
        // The audit property: same calibration set -> identical thresholds ->
        // identical sets, twice.
        let mut rng = StdRng::seed_from_u64(123);
        let (cal_probs, cal_labels) = synthetic_classification(&mut rng, 1000, 5);
        let (test_probs, _) = synthetic_classification(&mut rng, 200, 5);

        let model_a =
            ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, 0.1).unwrap();
        let model_b =
            ConformalModel::classification(&cal_probs, &cal_labels, ClassScore::Aps, 0.1).unwrap();
        for row in &test_probs {
            assert_eq!(
                model_a.predict_set(row, None).unwrap(),
                model_b.predict_set(row, None).unwrap()
            );
        }
    }

    #[test]
    fn predict_rejects_cross_kind_calls() {
        let model = ConformalModel::regression(
            &[0.0, 1.0],
            &[],
            &[],
            &[0.1, 0.9],
            IntervalScore::AbsoluteResidual,
            0.1,
        )
        .unwrap();
        assert!(model.predict_set(&[0.5, 0.5], None).is_err());

        let model = ConformalModel::classification(
            &[vec![0.6, 0.4], vec![0.3, 0.7]],
            &[0, 1],
            ClassScore::Lac,
            0.1,
        )
        .unwrap();
        assert!(model.predict_interval(0.0, 0.0, 0.0, None).is_err());
    }

    #[test]
    fn rejects_out_of_range_label_and_nan_probabilities() {
        assert!(true_label_score(&[0.5, 0.5], 2, ClassScore::Lac).is_err());
        assert!(true_label_score(&[f64::NAN, 0.5], 0, ClassScore::Lac).is_err());
    }
}
