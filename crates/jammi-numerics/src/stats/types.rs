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
