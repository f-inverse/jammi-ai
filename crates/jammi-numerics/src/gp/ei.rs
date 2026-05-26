//! Expected Improvement acquisition function for Bayesian optimisation.
//!
//! **Convention: maximisation.** `expected_improvement(mu, sigma, f_best)`
//! is the expected improvement over `f_best` when higher is better. Callers
//! who want to minimise should negate their objective and the best-so-far
//! value before calling: a minimisation problem reduces to maximising
//! `-f`.

use statrs::distribution::{ContinuousCDF, Normal};

/// Expected Improvement under the maximisation convention.
///
/// `mu`     — posterior mean at the candidate point.
/// `sigma`  — posterior standard deviation at the candidate point. When
///            ≈ 0 (no posterior uncertainty), EI is 0.
/// `f_best` — best (largest) observed objective so far.
pub fn expected_improvement(mu: f64, sigma: f64, f_best: f64) -> f64 {
    if sigma < 1e-12 {
        return 0.0;
    }
    let z = (mu - f_best) / sigma;
    let n = Normal::standard();
    let cdf = n.cdf(z);
    let pdf = (-0.5 * z * z).exp() / (2.0_f64 * std::f64::consts::PI).sqrt();
    (mu - f_best) * cdf + sigma * pdf
}
