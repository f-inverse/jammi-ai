//! Statistical tests and resampling kernels: Welch's t-test, Mann-Whitney U,
//! and percentile bootstrap confidence intervals.

pub mod bootstrap;
pub mod mannwhitney;
pub mod types;
pub mod welch;

pub use bootstrap::bootstrap_ci;
pub use mannwhitney::mann_whitney_u;
pub use types::{Interval, TestResult};
pub use welch::welch_t_test;
