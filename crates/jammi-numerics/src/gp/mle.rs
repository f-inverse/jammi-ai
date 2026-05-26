//! Grid-search maximum-likelihood hyperparameter selection for the RBF
//! GP.
//!
//! Scores every `(signal_var, lengthscale, noise_var)` combination on
//! the supplied training set by the negative log marginal likelihood
//! `NLL = 0.5 yᵀ K⁻¹ y + sum(log L_ii) + (n/2) log(2π)` (using the
//! Cholesky factor `L = chol(K)` so `log|K| = 2 * Σ log L_ii`) and
//! returns the lowest-NLL combination.

use nalgebra::DVector;

use crate::error::{NumericsError, Result};
use crate::gp::cholesky::cholesky_with_jitter;
use crate::gp::kernel::rbf_gram;

const SIGNAL_VAR_GRID: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0];
const LENGTHSCALE_GRID: &[f64] = &[0.1, 0.3, 1.0, 3.0, 10.0];
const NOISE_VAR_GRID: &[f64] = &[1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1];

/// Grid-search the RBF hyperparameters minimising negative log marginal
/// likelihood on `(x_train, y_train)`. Returns
/// `(signal_var, lengthscale, noise_var)`.
///
/// Returns `NumericsError::IllConditioned` only if every grid point's
/// Cholesky factorisation fails the jitter retry — extremely unlikely
/// given the noise floor entries in `NOISE_VAR_GRID`.
pub fn mle_hyperparams(x_train: &[Vec<f64>], y_train: &DVector<f64>) -> Result<(f64, f64, f64)> {
    if x_train.is_empty() || x_train.len() != y_train.len() {
        return Err(NumericsError::InvalidInput(
            "mle_hyperparams: empty or mismatched training inputs".into(),
        ));
    }
    let n = x_train.len();
    let log_2pi_term = 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();

    let mut best: Option<((f64, f64, f64), f64)> = None;
    for &sv in SIGNAL_VAR_GRID {
        for &ls in LENGTHSCALE_GRID {
            for &nv in NOISE_VAR_GRID {
                let k = rbf_gram(x_train, sv, ls, nv);
                let chol = match cholesky_with_jitter(k) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let alpha = chol.solve(y_train);
                let quad = 0.5 * y_train.dot(&alpha);
                let l = chol.l();
                let log_det: f64 = (0..n).map(|i| l[(i, i)].ln()).sum();
                let nll = quad + log_det + log_2pi_term;
                if best.as_ref().map(|(_, b)| nll < *b).unwrap_or(true) {
                    best = Some(((sv, ls, nv), nll));
                }
            }
        }
    }
    best.map(|(p, _)| p).ok_or(NumericsError::IllConditioned {
        attempts: SIGNAL_VAR_GRID.len() * LENGTHSCALE_GRID.len() * NOISE_VAR_GRID.len(),
        final_jitter: NOISE_VAR_GRID[NOISE_VAR_GRID.len() - 1],
    })
}
