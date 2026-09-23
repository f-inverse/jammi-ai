//! Statistical tests and resampling kernels: Welch's t-test, Mann-Whitney U,
//! the exact sign test, percentile and block bootstrap confidence intervals,
//! the paired margin tests (non-inferiority and equivalence), the
//! Mann-Kendall trend test, the marginal standard error truncation rule, the
//! least-squares line, multinomial goodness of fit, and location summaries.

pub mod block_bootstrap;
pub mod bootstrap;
pub mod goodness_of_fit;
pub mod linear_fit;
pub mod mannwhitney;
pub mod margin;
pub mod sign_test;
pub mod summary;
pub mod trend;
pub mod truncation;
pub mod types;
pub mod welch;

pub use block_bootstrap::block_bootstrap_ci;
pub use bootstrap::bootstrap_ci;
pub use goodness_of_fit::{
    goodness_of_fit, FitCell, FitStatistic, GoodnessOfFit, MIN_EXPECTED_COUNT,
};
pub use linear_fit::linear_fit;
pub use mannwhitney::mann_whitney_u;
pub use margin::paired_margin_test;
pub use sign_test::{sign_test, sign_test_critical_count};
pub use summary::{geometric_mean, mean, median, minimum, population_std_dev};
pub use trend::mann_kendall;
pub use truncation::{mser_truncation, Truncation};
pub use types::{
    Better, Interval, LinearFit, MarginTestResult, SignTestResult, TestResult, TrendResult,
};
pub use welch::welch_t_test;
