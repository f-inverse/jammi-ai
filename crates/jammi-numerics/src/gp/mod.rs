//! Gaussian process primitives for Bayesian optimisation: RBF kernel, a
//! Cholesky factoriser with diagonal jitter retry, posterior mean/variance
//! at a query point, Expected Improvement acquisition (maximisation
//! convention), and grid-search hyperparameter MLE.

pub mod cholesky;
pub mod ei;
pub mod kernel;
pub mod mle;
pub mod posterior;

pub use cholesky::cholesky_with_jitter;
pub use ei::expected_improvement;
pub use kernel::{rbf, rbf_gram};
pub use mle::mle_hyperparams;
pub use posterior::posterior;
