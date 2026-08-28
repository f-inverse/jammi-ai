//! Statistical tests and resampling kernels: Welch's t-test, Mann-Whitney U,
//! the exact sign test, and percentile bootstrap confidence intervals.

pub mod bootstrap;
pub mod mannwhitney;
pub mod sign_test;
pub mod types;
pub mod welch;

pub use bootstrap::bootstrap_ci;
pub use mannwhitney::mann_whitney_u;
pub use sign_test::sign_test;
pub use types::{Interval, SignTestResult, TestResult};
pub use welch::welch_t_test;
