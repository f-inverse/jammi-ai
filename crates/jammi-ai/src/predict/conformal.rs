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
//! * A nonconformity score is the one task-specific piece: it maps a predictor
//!   output and a candidate label to a real number, larger when the label
//!   conforms *less* to the prediction. Calibration scores its held-out
//!   examples at their *true* label. A classifier picks its score family
//!   ([`ClassScore`]); a regressor's follows from what it estimates
//!   ([`Estimates`]).
//! * [`finite_sample_quantile`] takes the calibration scores and `alpha` to the
//!   conformal threshold `q̂` — the `⌈(n+1)(1-alpha)⌉`-th smallest score. The
//!   `(n+1)` correction is what makes the guarantee *exact* rather than merely
//!   asymptotic; the naive `⌈n(1-alpha)⌉` order statistic under-covers.
//! * A [`ConformalModel`] holds the calibrated threshold(s). At serving time it
//!   admits every label whose nonconformity is `<= q̂` — a set for
//!   classification, an interval for regression.
//!
//! # Calibration under known structure
//!
//! A [`Calibration`] says how the calibration rows pool into the threshold a
//! serving row is held to. `Marginal` pools them all. `Mondrian` keeps one
//! quantile per cohort (group-conditional coverage), and a serving row names its
//! cohort. `Weighted` applies importance weights under a known covariate shift
//! (weighted conformal, Tibshirani et al. 2019), and a serving row brings its own
//! weight, which sets its threshold. The primitive *applies* the grouping and the
//! weights; it does **not** decide when shift has occurred or which cohorts
//! matter.
//!
//! # The guarantee's one assumption
//!
//! Coverage holds **iff** calibration and serving data are exchangeable (within
//! a cohort, or after reweighting). Under drift it degrades silently (Barber et
//! al. 2023); *detecting* drift and *adapting* online are policies a caller
//! builds on this primitive, not part of it. The calibration source is a
//! distinct argument from train and test — reusing test points to calibrate
//! inflates coverage, so the three-way split is the caller's contract.

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

/// A regression predictor's estimates over the calibration rows; what it
/// estimates decides its nonconformity score.
#[derive(Debug, Clone, Copy)]
pub enum Estimates<'a> {
    /// Point estimates `ŷ`, scored by the absolute residual `|y - ŷ|`: a
    /// constant-width interval `[ŷ - q̂, ŷ + q̂]`. Distribution-free but
    /// non-adaptive — uninformative under heteroscedasticity.
    Points(&'a [f64]),
    /// Lower/upper quantile estimates, scored by Conformalized Quantile
    /// Regression (Romano et al. 2019), `max(q_lo - y, y - q_hi)`: an
    /// *adaptive*-width interval `[q_lo - q̂, q_hi + q̂]`.
    Bands {
        /// The lower quantile estimate per row.
        lower: &'a [f64],
        /// The upper quantile estimate per row.
        upper: &'a [f64],
    },
}

/// One serving row's estimate, in the form its model was calibrated on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Estimate {
    /// A point estimate `ŷ`, for a model calibrated on [`Estimates::Points`].
    Point(f64),
    /// A quantile band, for a model calibrated on [`Estimates::Bands`].
    Band {
        /// The lower quantile estimate.
        lower: f64,
        /// The upper quantile estimate.
        upper: f64,
    },
}

/// How the calibration rows pool into the threshold a serving row is held to.
#[derive(Debug, Clone, PartialEq)]
pub enum Calibration {
    /// One threshold over every calibration row: marginal coverage under
    /// exchangeability.
    Marginal,
    /// One threshold per cohort (Mondrian, group-conditional coverage):
    /// `groups[i]` is calibration row `i`'s cohort. A serving row names its
    /// cohort; one unseen at calibration is held to the pooled threshold. The
    /// principled approximation to conditional coverage — full per-input
    /// coverage is provably impossible distribution-free.
    Mondrian {
        /// The cohort of each calibration row, in calibration order.
        groups: Vec<String>,
    },
    /// Importance-weighted (Tibshirani et al. 2019) for a known covariate shift:
    /// `weights[i]` is the likelihood ratio `dP_test/dP_cal` at calibration row
    /// `i`. A serving row brings its own ratio, which sets its threshold — see
    /// [`finite_sample_quantile_weighted`].
    Weighted {
        /// The likelihood ratio of each calibration row, in calibration order.
        weights: Vec<f64>,
    },
}

/// What a serving row brings beyond its prediction, matching the
/// [`Calibration`] its model was built with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestPoint<'a> {
    /// A row exchangeable with the calibration rows ([`Calibration::Marginal`]).
    Exchangeable,
    /// A row of this cohort ([`Calibration::Mondrian`]).
    Group(&'a str),
    /// A row with this likelihood ratio `dP_test/dP_cal`
    /// ([`Calibration::Weighted`]).
    Weight(f64),
}

impl Calibration {
    /// Refuse a cohort or weight list that is not one per calibration row.
    pub fn check_rows(&self, rows: usize) -> Result<()> {
        match self {
            Self::Marginal => Ok(()),
            Self::Mondrian { groups } => one_per_row("groups", "cohort", groups.len(), rows),
            Self::Weighted { weights } => one_per_row("weights", "weight", weights.len(), rows),
        }
    }
}

fn one_per_row(argument: &str, what: &str, given: usize, rows: usize) -> Result<()> {
    if given == rows {
        return Ok(());
    }
    Err(JammiError::InvalidArgument {
        argument: argument.into(),
        reason: format!("conformal needs one {what} per calibration row: {given} for {rows} rows"),
    })
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
/// Returns `InvalidArgument` when `scores` is empty or holds a NaN, or `alpha`
/// is not in `(0, 1)`.
pub fn finite_sample_quantile(scores: &[f64], alpha: f64) -> Result<Threshold> {
    validate_alpha(alpha)?;
    validate_scores(scores)?;
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
/// Under a *known* covariate shift, calibration row `i` carries the likelihood
/// ratio `w_i = dP_test/dP_cal(x_i) >= 0` and the test point its own `w_test`.
/// The threshold is the `1 - alpha` quantile of the mixture placing mass
/// `w_i / (Σw + w_test)` on each calibration score and `w_test / (Σw + w_test)`
/// at `+∞`: the smallest score whose cumulative weight reaches
/// `(1 - alpha)(Σw + w_test)`, or `+∞` when the calibration mass cannot. The
/// test weight is what makes the threshold row-specific; with every weight one
/// this is exactly [`finite_sample_quantile`].
///
/// Returns `InvalidArgument` when the lengths disagree, `scores` is empty or
/// holds a NaN, any weight is negative or non-finite, the total weight is zero,
/// or `alpha` is not in `(0, 1)`.
pub fn finite_sample_quantile_weighted(
    scores: &[f64],
    weights: &[f64],
    test_weight: f64,
    alpha: f64,
) -> Result<Threshold> {
    validate_alpha(alpha)?;
    one_per_row("weights", "weight", weights.len(), scores.len())?;
    WeightedScores::new(scores, weights)?.threshold(test_weight, alpha)
}

/// A calibrated conformal predictor: thresholds plus the score family that
/// produced them.
///
/// Built by [`ConformalModel::classification`] or
/// [`ConformalModel::regression`], then queried with
/// [`ConformalModel::predict_set`] or [`ConformalModel::predict_interval`]. The
/// model is a value: the same calibration set produces byte-identical
/// thresholds and therefore byte-identical sets across runs.
#[derive(Debug, Clone)]
pub struct ConformalModel {
    /// The nominal miscoverage level the thresholds target.
    alpha: f64,
    /// The score family the thresholds belong to.
    family: Family,
    /// The calibrated thresholds, pooled as the [`Calibration`] said.
    thresholds: Thresholds,
}

#[derive(Debug, Clone, Copy)]
enum Family {
    Classes(ClassScore),
    Points,
    Bands,
}

#[derive(Debug, Clone)]
enum Thresholds {
    Marginal(Threshold),
    Mondrian {
        by_group: BTreeMap<String, Threshold>,
        /// The threshold over every row, for a cohort unseen at calibration.
        pooled: Threshold,
    },
    Weighted(WeightedScores),
}

impl Thresholds {
    fn calibrate(scores: &[f64], calibration: &Calibration, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        calibration.check_rows(scores.len())?;
        match calibration {
            Calibration::Marginal => Ok(Self::Marginal(finite_sample_quantile(scores, alpha)?)),
            Calibration::Mondrian { groups } => mondrian(scores, groups, alpha),
            Calibration::Weighted { weights } => {
                Ok(Self::Weighted(WeightedScores::new(scores, weights)?))
            }
        }
    }

    /// The threshold one serving row is held to.
    fn at(&self, point: TestPoint<'_>, alpha: f64) -> Result<Threshold> {
        match (self, point) {
            (Self::Marginal(q), TestPoint::Exchangeable) => Ok(*q),
            (Self::Mondrian { by_group, pooled }, TestPoint::Group(group)) => {
                Ok(by_group.get(group).copied().unwrap_or(*pooled))
            }
            (Self::Weighted(calibration), TestPoint::Weight(weight)) => {
                calibration.threshold(weight, alpha)
            }
            (thresholds, point) => Err(JammiError::InvalidArgument {
                argument: "test_point".into(),
                reason: format!(
                    "a {} calibration serves a row by {}, not {point:?}",
                    thresholds.name(),
                    thresholds.serves_by()
                ),
            }),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Marginal(_) => "marginal",
            Self::Mondrian { .. } => "Mondrian",
            Self::Weighted(_) => "weighted",
        }
    }

    fn serves_by(&self) -> &'static str {
        match self {
            Self::Marginal(_) => "no cohort or weight",
            Self::Mondrian { .. } => "its cohort",
            Self::Weighted(_) => "its likelihood ratio",
        }
    }
}

/// Calibration scores in ascending order, each with the running total of the
/// weights up to and including it — what a weighted threshold walks.
#[derive(Debug, Clone)]
struct WeightedScores {
    sorted: Vec<f64>,
    cumulative: Vec<f64>,
}

impl WeightedScores {
    /// `weights` holds one weight per score; the caller has checked the lengths.
    fn new(scores: &[f64], weights: &[f64]) -> Result<Self> {
        validate_scores(scores)?;
        validate_weights(weights)?;
        let mut paired: Vec<(f64, f64)> = scores
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        paired.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (sorted, cumulative) = paired
            .into_iter()
            .scan(0.0, |running, (score, weight)| {
                *running += weight;
                Some((score, *running))
            })
            .unzip();
        Ok(Self { sorted, cumulative })
    }

    fn threshold(&self, test_weight: f64, alpha: f64) -> Result<Threshold> {
        validate_weights(&[test_weight])?;
        // `new` refuses an empty calibration, so a last running total exists.
        let total = self.cumulative.last().copied().unwrap_or(0.0) + test_weight;
        if total <= 0.0 {
            return Err(JammiError::InvalidArgument {
                argument: "weights".into(),
                reason: "weighted conformal requires a positive total weight".into(),
            });
        }
        let target = (1.0 - alpha) * total;
        let reached = self.cumulative.partition_point(|&mass| mass < target);
        // The calibration mass never reached `target`; the test point's mass at
        // +∞ carries the rest.
        Ok(self.sorted.get(reached).copied().unwrap_or(f64::INFINITY))
    }
}

/// One threshold per cohort over that cohort's scores alone, plus the pooled
/// threshold over every score for a cohort unseen at calibration. `groups`
/// holds one cohort per score; the caller has checked the lengths.
fn mondrian(scores: &[f64], groups: &[String], alpha: f64) -> Result<Thresholds> {
    let mut by_group: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (score, group) in scores.iter().zip(groups) {
        by_group.entry(group.as_str()).or_default().push(*score);
    }
    let by_group = by_group
        .into_iter()
        .map(|(group, group_scores)| {
            Ok((
                group.to_string(),
                finite_sample_quantile(&group_scores, alpha)?,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Thresholds::Mondrian {
        by_group,
        pooled: finite_sample_quantile(scores, alpha)?,
    })
}

impl ConformalModel {
    /// Calibrate a classification predictor.
    ///
    /// `probabilities` holds one row of per-class probabilities per held-out
    /// example, `true_labels[i]` is the index of the realised class for row
    /// `i`, and `score` selects the nonconformity family. Every probability row
    /// must have the same length (the class count) and every true label must
    /// index into it.
    pub fn classification(
        probabilities: &[Vec<f64>],
        true_labels: &[usize],
        score: ClassScore,
        calibration: &Calibration,
        alpha: f64,
    ) -> Result<Self> {
        let scores = classification_scores(probabilities, true_labels, score)?;
        Ok(Self {
            alpha,
            family: Family::Classes(score),
            thresholds: Thresholds::calibrate(&scores, calibration, alpha)?,
        })
    }

    /// The prediction set for one classification row: the class indices whose
    /// nonconformity at `probabilities` is `<= q̂`, in ascending order. An empty
    /// set never occurs at a finite, attainable level; a full set is the honest
    /// signal that the input is hard or the model is miscalibrated.
    pub fn predict_set(&self, probabilities: &[f64], at: TestPoint<'_>) -> Result<Vec<usize>> {
        let Family::Classes(score) = self.family else {
            return Err(JammiError::InvalidArgument {
                argument: "model".into(),
                reason: "predict_set called on a regression conformal model".into(),
            });
        };
        admit_classes(probabilities, score, self.thresholds.at(at, self.alpha)?)
    }

    /// Calibrate a regression predictor on its `estimates` of the realised
    /// targets `observed`.
    pub fn regression(
        estimates: Estimates<'_>,
        observed: &[f64],
        calibration: &Calibration,
        alpha: f64,
    ) -> Result<Self> {
        let (family, scores) = regression_scores(estimates, observed)?;
        Ok(Self {
            alpha,
            family,
            thresholds: Thresholds::calibrate(&scores, calibration, alpha)?,
        })
    }

    /// The prediction interval `[lower, upper]` for one regression row.
    pub fn predict_interval(&self, estimate: Estimate, at: TestPoint<'_>) -> Result<(f64, f64)> {
        let q = || self.thresholds.at(at, self.alpha);
        match (self.family, estimate) {
            (Family::Points, Estimate::Point(y)) => {
                let q = q()?;
                Ok((y - q, y + q))
            }
            (Family::Bands, Estimate::Band { lower, upper }) => {
                let q = q()?;
                Ok((lower - q, upper + q))
            }
            (Family::Classes(_), _) => Err(JammiError::InvalidArgument {
                argument: "model".into(),
                reason: "predict_interval called on a classification conformal model".into(),
            }),
            (family, estimate) => Err(JammiError::InvalidArgument {
                argument: "estimate".into(),
                reason: format!(
                    "a model calibrated on {} serves {}, not {estimate:?}",
                    family.estimates(),
                    family.estimates()
                ),
            }),
        }
    }

    /// The nominal miscoverage level these thresholds target.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Family {
    fn estimates(self) -> &'static str {
        match self {
            Self::Classes(_) => "class probabilities",
            Self::Points => "point estimates",
            Self::Bands => "quantile bands",
        }
    }
}

/// Validate a miscoverage level lies in the open interval `(0, 1)`.
fn validate_alpha(alpha: f64) -> Result<()> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(JammiError::InvalidArgument {
            argument: "alpha".into(),
            reason: format!("conformal alpha must lie in (0, 1), got {alpha}"),
        });
    }
    Ok(())
}

/// Validate a calibration score slice: non-empty and free of NaN (a NaN would
/// silently corrupt the rank).
fn validate_scores(scores: &[f64]) -> Result<()> {
    if scores.is_empty() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "conformal calibration requires at least one score".into(),
        });
    }
    if scores.iter().any(|s| s.is_nan()) {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "conformal calibration scores must be finite".into(),
        });
    }
    Ok(())
}

/// Validate likelihood ratios: each finite and non-negative.
fn validate_weights(weights: &[f64]) -> Result<()> {
    if weights.iter().any(|&w| w < 0.0 || !w.is_finite()) {
        return Err(JammiError::InvalidArgument {
            argument: "weights".into(),
            reason: "weighted conformal weights must be finite and non-negative".into(),
        });
    }
    Ok(())
}

/// Nonconformity scores of the calibration rows at their *true* class.
fn classification_scores(
    probabilities: &[Vec<f64>],
    true_labels: &[usize],
    score: ClassScore,
) -> Result<Vec<f64>> {
    if probabilities.len() != true_labels.len() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: format!(
                "classification conformal: {} probability rows but {} labels",
                probabilities.len(),
                true_labels.len()
            ),
        });
    }
    if probabilities.is_empty() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "classification conformal requires at least one calibration row".into(),
        });
    }
    probabilities
        .iter()
        .zip(true_labels)
        .map(|(probs, &label)| true_label_score(probs, label, score))
        .collect()
}

/// The nonconformity score of one calibration row evaluated at its true class.
fn true_label_score(probs: &[f64], label: usize, score: ClassScore) -> Result<f64> {
    validate_probabilities(probs)?;
    if label >= probs.len() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: format!(
                "true label {label} out of range for {} classes",
                probs.len()
            ),
        });
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

/// Nonconformity scores for the regression calibration rows, and the family
/// their estimates belong to.
fn regression_scores(estimates: Estimates<'_>, observed: &[f64]) -> Result<(Family, Vec<f64>)> {
    if observed.is_empty() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "regression conformal requires at least one calibration row".into(),
        });
    }
    ensure_finite(observed, "observations")?;
    match estimates {
        Estimates::Points(predictions) => {
            if predictions.len() != observed.len() {
                return Err(JammiError::InvalidArgument {
                    argument: "calibration".into(),
                    reason: format!(
                        "absolute-residual conformal: {} predictions but {} observations",
                        predictions.len(),
                        observed.len()
                    ),
                });
            }
            ensure_finite(predictions, "predictions")?;
            let scores = predictions
                .iter()
                .zip(observed)
                .map(|(&yhat, &y)| (y - yhat).abs())
                .collect();
            Ok((Family::Points, scores))
        }
        Estimates::Bands { lower, upper } => {
            if lower.len() != observed.len() || upper.len() != observed.len() {
                return Err(JammiError::InvalidArgument {
                    argument: "calibration".into(),
                    reason: format!(
                        "CQR conformal: {} lower / {} upper quantiles but {} observations",
                        lower.len(),
                        upper.len(),
                        observed.len()
                    ),
                });
            }
            ensure_finite(lower, "lower quantiles")?;
            ensure_finite(upper, "upper quantiles")?;
            let scores = lower
                .iter()
                .zip(upper)
                .zip(observed)
                .map(|((&lo, &hi), &y)| (lo - y).max(y - hi))
                .collect();
            Ok((Family::Bands, scores))
        }
    }
}

/// Validate a per-class probability row: non-empty and free of NaN. The values
/// need not sum to one — conformal reads the *ordering* and the true-class mass,
/// not a normalized distribution — but a NaN would silently corrupt the rank.
fn validate_probabilities(probs: &[f64]) -> Result<()> {
    if probs.is_empty() {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "classification conformal requires at least one class probability".into(),
        });
    }
    if probs.iter().any(|p| p.is_nan()) {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: "classification conformal probabilities must be finite".into(),
        });
    }
    Ok(())
}

/// Reject non-finite values in a regression input slice.
fn ensure_finite(values: &[f64], what: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(JammiError::InvalidArgument {
            argument: "calibration".into(),
            reason: format!("regression conformal {what} must be finite"),
        });
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
            .map(|(row, &y)| {
                model
                    .predict_set(row, TestPoint::Exchangeable)
                    .unwrap()
                    .contains(&y)
            })
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

            let model = ConformalModel::classification(
                &cal_probs,
                &cal_labels,
                ClassScore::Lac,
                &Calibration::Marginal,
                alpha,
            )
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
        let model = ConformalModel::classification(
            &cal_probs,
            &cal_labels,
            ClassScore::Aps,
            &Calibration::Marginal,
            0.1,
        )
        .unwrap();

        let easy = vec![0.9, 0.04, 0.02, 0.02, 0.01, 0.01];
        let hard = vec![0.25, 0.22, 0.20, 0.15, 0.10, 0.08];
        let easy_set = model.predict_set(&easy, TestPoint::Exchangeable).unwrap();
        let hard_set = model.predict_set(&hard, TestPoint::Exchangeable).unwrap();
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

            let aps = ConformalModel::classification(
                &cal_probs,
                &cal_labels,
                ClassScore::Aps,
                &Calibration::Marginal,
                alpha,
            )
            .unwrap();
            let raps = ConformalModel::classification(
                &cal_probs,
                &cal_labels,
                ClassScore::Raps {
                    lambda: 1.0,
                    k_reg: 2,
                },
                &Calibration::Marginal,
                alpha,
            )
            .unwrap();

            let sizes = |model: &ConformalModel| -> f64 {
                test_probs
                    .iter()
                    .map(|p| model.predict_set(p, TestPoint::Exchangeable).unwrap().len() as f64)
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
        let model = ConformalModel::regression(
            Estimates::Bands {
                lower: &lower,
                upper: &upper,
            },
            &observed,
            &Calibration::Marginal,
            0.1,
        )
        .unwrap();

        // Evaluate width in two regimes.
        for x in [1.0_f64, 9.0_f64] {
            let sd = 0.2 + 0.5 * x;
            let band = 1.2816 * sd;
            let (lo, hi) = model
                .predict_interval(
                    Estimate::Band {
                        lower: -band,
                        upper: band,
                    },
                    TestPoint::Exchangeable,
                )
                .unwrap();
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
            Estimates::Points(&cal_pred),
            &cal_obs,
            &Calibration::Marginal,
            0.1,
        )
        .unwrap();

        let test_obs: Vec<f64> = (0..4000).map(|_| dist.sample(&mut rng)).collect();
        let (lower, upper): (Vec<f64>, Vec<f64>) = test_obs
            .iter()
            .map(|_| {
                model
                    .predict_interval(Estimate::Point(0.0), TestPoint::Exchangeable)
                    .unwrap()
            })
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

        let model = ConformalModel::classification(
            &probs,
            &labels,
            ClassScore::Lac,
            &Calibration::Mondrian { groups },
            0.1,
        )
        .unwrap();

        // The hard cohort's threshold must exceed the easy cohort's.
        let test = vec![0.4, 0.2, 0.2, 0.1, 0.1];
        let set_a = model.predict_set(&test, TestPoint::Group("a")).unwrap();
        let set_b = model.predict_set(&test, TestPoint::Group("b")).unwrap();
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

        let calibrate = || {
            ConformalModel::classification(
                &cal_probs,
                &cal_labels,
                ClassScore::Aps,
                &Calibration::Marginal,
                0.1,
            )
            .unwrap()
        };
        let (model_a, model_b) = (calibrate(), calibrate());
        for row in &test_probs {
            assert_eq!(
                model_a.predict_set(row, TestPoint::Exchangeable).unwrap(),
                model_b.predict_set(row, TestPoint::Exchangeable).unwrap()
            );
        }
    }

    #[test]
    fn predict_rejects_cross_kind_calls() {
        let model = ConformalModel::regression(
            Estimates::Points(&[0.0, 1.0]),
            &[0.1, 0.9],
            &Calibration::Marginal,
            0.1,
        )
        .unwrap();
        assert!(model
            .predict_set(&[0.5, 0.5], TestPoint::Exchangeable)
            .is_err());
        // A point-estimate model is not handed a quantile band.
        let band = Estimate::Band {
            lower: 0.0,
            upper: 1.0,
        };
        assert!(model
            .predict_interval(band, TestPoint::Exchangeable)
            .is_err());

        let model = ConformalModel::classification(
            &[vec![0.6, 0.4], vec![0.3, 0.7]],
            &[0, 1],
            ClassScore::Lac,
            &Calibration::Marginal,
            0.1,
        )
        .unwrap();
        assert!(model
            .predict_interval(Estimate::Point(0.0), TestPoint::Exchangeable)
            .is_err());
    }

    #[test]
    fn rejects_out_of_range_label_and_nan_probabilities() {
        assert!(true_label_score(&[0.5, 0.5], 2, ClassScore::Lac).is_err());
        assert!(true_label_score(&[f64::NAN, 0.5], 0, ClassScore::Lac).is_err());
    }

    #[test]
    fn weighted_threshold_follows_the_test_points_own_weight() {
        // Twenty unit-weight calibration scores 1..=20 at alpha = 0.2: a unit test
        // weight puts the target at 0.8 * 21 = 16.8, the 17th score. A heavier
        // test point holds more mass at +∞, so the calibration scores must reach
        // further: weight 5 puts the target at 0.8 * 25 = 20.0, the 20th score.
        let scores: Vec<f64> = (1..=20).map(f64::from).collect();
        let model = ConformalModel::regression(
            Estimates::Points(&[0.0; 20]),
            &scores,
            &Calibration::Weighted {
                weights: vec![1.0; 20],
            },
            0.2,
        )
        .unwrap();
        let width = |weight| {
            let (lo, hi) = model
                .predict_interval(Estimate::Point(0.0), TestPoint::Weight(weight))
                .unwrap();
            (hi - lo) / 2.0
        };
        assert_eq!(width(1.0), 17.0);
        assert_eq!(width(5.0), 20.0);
        // Weight 6: target 0.8 * 26 = 20.8 exceeds the whole calibration mass.
        assert!(width(6.0).is_infinite());
    }

    #[test]
    fn exact_weights_restore_what_a_score_aligned_shift_breaks() {
        // A covariate x drives the score upward; the calibration set keeps each row
        // with probability π(x) = 1/(1+e^{2x}), leaning it toward easy rows, and
        // the likelihood ratio 1/π is known exactly. Coverage is an expectation
        // over the calibration draw, so it is averaged over draws.
        let mut rng = StdRng::seed_from_u64(7);
        let covariate = NormalDraw::new(0.0, 1.0).unwrap();
        let noise = NormalDraw::new(0.0, 0.5).unwrap();
        let ratio = |x: f64| 1.0 + (2.0 * x).exp();
        let alpha = 0.1;
        let draws = 300;
        let (mut marginal, mut weighted) = (0.0, 0.0);
        for _ in 0..draws {
            let mut x_cal = Vec::new();
            for _ in 0..400 {
                let x = covariate.sample(&mut rng);
                if rng.gen_range(0.0..1.0) < 1.0 / ratio(x) {
                    x_cal.push(x);
                }
            }
            let x_test: Vec<f64> = (0..400).map(|_| covariate.sample(&mut rng)).collect();
            let cal: Vec<f64> = x_cal.iter().map(|x| x + noise.sample(&mut rng)).collect();
            let test: Vec<f64> = x_test.iter().map(|x| x + noise.sample(&mut rng)).collect();

            let q = finite_sample_quantile(&cal, alpha).unwrap();
            let weights: Vec<f64> = x_cal.iter().map(|&x| ratio(x)).collect();
            let calibration = WeightedScores::new(&cal, &weights).unwrap();
            let rows = test.len() as f64;
            marginal += test.iter().filter(|&&s| s <= q).count() as f64 / rows;
            weighted += test
                .iter()
                .zip(&x_test)
                .filter(|&(&s, &x)| s <= calibration.threshold(ratio(x), alpha).unwrap())
                .count() as f64
                / rows;
        }
        let (marginal, weighted) = (marginal / draws as f64, weighted / draws as f64);
        // The guarantee is a floor: heavy weights leave few effective calibration
        // rows, and the test row's mass at +∞ makes the weighted pass conservative.
        assert!(marginal < 1.0 - alpha - 0.1, "marginal coverage {marginal}");
        assert!(
            (1.0 - alpha..1.0 - alpha + 0.05).contains(&weighted),
            "weighted coverage {weighted}"
        );
    }

    #[test]
    fn unit_weights_reproduce_the_marginal_threshold_exactly() {
        let mut rng = StdRng::seed_from_u64(31);
        for n in [5, 9, 30, 101] {
            let scores: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..3.0)).collect();
            for alpha in [0.05, 0.1, 0.2, 0.5] {
                let marginal = finite_sample_quantile(&scores, alpha).unwrap();
                let weighted =
                    finite_sample_quantile_weighted(&scores, &vec![1.0; n], 1.0, alpha).unwrap();
                assert_eq!(marginal, weighted, "n={n} alpha={alpha}");
            }
        }
    }

    #[test]
    fn a_test_point_must_match_its_calibration() {
        let probs = vec![vec![0.6, 0.4], vec![0.3, 0.7], vec![0.8, 0.2]];
        let labels = [0, 1, 0];
        let calibrate = |calibration: Calibration| {
            ConformalModel::classification(&probs, &labels, ClassScore::Lac, &calibration, 0.4)
                .unwrap()
        };
        let row = [0.5, 0.5];
        let marginal = calibrate(Calibration::Marginal);
        assert!(marginal.predict_set(&row, TestPoint::Group("a")).is_err());
        let mondrian = calibrate(Calibration::Mondrian {
            groups: vec!["a".into(), "b".into(), "a".into()],
        });
        assert!(mondrian.predict_set(&row, TestPoint::Exchangeable).is_err());
        // A cohort unseen at calibration is held to the pooled threshold.
        assert_eq!(
            mondrian.predict_set(&row, TestPoint::Group("c")).unwrap(),
            marginal.predict_set(&row, TestPoint::Exchangeable).unwrap()
        );
        let weighted = calibrate(Calibration::Weighted {
            weights: vec![1.0; 3],
        });
        assert!(weighted.predict_set(&row, TestPoint::Exchangeable).is_err());
    }
}
