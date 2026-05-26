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
