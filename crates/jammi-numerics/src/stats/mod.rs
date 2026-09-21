//! Statistical tests and resampling kernels: Welch's t-test, Mann-Whitney U,
//! the exact sign test, percentile and block bootstrap confidence intervals,
//! the paired equivalence test, the Mann-Kendall trend test, the
//! least-squares line, multinomial goodness of fit, and location summaries.

pub mod block_bootstrap;
pub mod bootstrap;
pub mod equivalence;
pub mod goodness_of_fit;
pub mod linear_fit;
pub mod mannwhitney;
pub mod sign_test;
pub mod summary;
pub mod trend;
pub mod types;
pub mod welch;

pub use block_bootstrap::block_bootstrap_ci;
pub use bootstrap::bootstrap_ci;
pub use equivalence::paired_equivalence;
pub use goodness_of_fit::{goodness_of_fit, FitCell, FitStatistic, GoodnessOfFit};
pub use linear_fit::linear_fit;
pub use mannwhitney::mann_whitney_u;
pub use sign_test::{sign_test, sign_test_critical_count};
pub use summary::{geometric_mean, mean, median, minimum, population_std_dev};
pub use trend::mann_kendall;
pub use types::{EquivalenceResult, Interval, LinearFit, SignTestResult, TestResult, TrendResult};
pub use welch::welch_t_test;
