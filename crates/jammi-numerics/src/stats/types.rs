//! Shared types for statistical tests.

use serde::{Deserialize, Serialize};

/// Result of a two-sample test: the test statistic and its two-tailed
/// p-value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

/// A confidence interval `[lower, upper]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

/// Result of an exact two-sided sign test (see
/// [`crate::stats::sign_test::sign_test`]).
///
/// `n_pos + n_neg == n`; `ties` is reported separately and is never folded
/// into `n` — a tie is neither evidence for nor against either sign, so
/// including it in `n` would understate the p-value's actual binomial
/// denominator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SignTestResult {
    /// Non-tied pairs used in the test (`n_pos + n_neg`).
    pub n: usize,
    /// Count of strictly positive differences.
    pub n_pos: usize,
    /// Count of strictly negative differences.
    pub n_neg: usize,
    /// Count of exact-zero differences, excluded from `n`.
    pub ties: usize,
    /// Exact two-sided p-value: `2 * P(X >= max(n_pos, n_neg))` capped at
    /// `1.0`, under `X ~ Binomial(n, 0.5)`.
    pub p_value: f64,
}

/// A least-squares line `y = intercept + slope * x` (see
/// [`crate::stats::linear_fit::linear_fit`]).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearFit {
    /// The fitted value at `x = 0`.
    pub intercept: f64,
    /// The fitted change in `y` per unit `x`.
    pub slope: f64,
    /// Root-mean-square of the residuals `y_i - (intercept + slope * x_i)`,
    /// in the units of `y`.
    pub residual_rms: f64,
    /// Number of points fitted.
    pub n: usize,
}

/// Result of a Mann-Kendall monotonic-trend test (see
/// [`crate::stats::trend::mann_kendall`]).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrendResult {
    /// Kendall's `S`: concordant minus discordant pairs over time order.
    pub s: i64,
    /// Continuity-corrected standard normal score of `S` under the
    /// tie-corrected null variance; `0.0` when that variance is zero (a
    /// constant series carries no trend evidence either way).
    pub z: f64,
    /// Two-sided p-value of `z`.
    pub p_value: f64,
    /// Theil-Sen slope: the median of all pairwise slopes, in series units
    /// per index step.
    pub sen_slope: f64,
}

/// Which direction of a quantity is the better one: what makes a one-sided
/// claim about a difference a claim of "no worse".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Better {
    /// A loss, a time, a byte count.
    Lower,
    /// A score, a throughput.
    Higher,
}

/// Result of the paired margin tests (see
/// [`crate::stats::margin::paired_margin_test`]): the two one-sided
/// tests of a mean paired difference `a - b` against `±delta`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarginTestResult {
    /// Mean of the paired differences.
    pub mean: f64,
    /// The `1 - 2 * alpha` bootstrap interval of that mean.
    pub interval: Interval,
    /// The margin the interval is judged against.
    pub delta: f64,
    /// The null "the true mean is at least `+delta`" is rejected:
    /// `interval.upper < +delta`.
    pub below_upper_margin: bool,
    /// The null "the true mean is at most `-delta`" is rejected:
    /// `-delta < interval.lower`.
    pub above_lower_margin: bool,
}

impl MarginTestResult {
    /// Both one-sided nulls rejected: the interval lies strictly inside
    /// `(-delta, +delta)`.
    pub fn equivalent(&self) -> bool {
        self.below_upper_margin && self.above_lower_margin
    }

    /// `a` is no worse than `b` by `delta`: the one one-sided null on the
    /// unfavourable side is rejected. A difference on the favourable side,
    /// however large, is not evidence against it.
    pub fn non_inferior(&self, better: Better) -> bool {
        match better {
            Better::Lower => self.below_upper_margin,
            Better::Higher => self.above_lower_margin,
        }
    }
}
