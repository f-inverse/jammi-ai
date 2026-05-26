//! GP posterior mean and variance at a single query point.

use nalgebra::{Cholesky, DVector, Dyn};

use crate::gp::kernel::rbf;

/// Compute the posterior mean and variance at `x_star` given the
/// pre-factorised training Gram matrix `k_chol`, training inputs `x_train`,
/// training targets `y_train`, and the RBF hyperparameters used to
/// construct `k_chol`.
///
/// Variance is clamped to `>= 0.0` to absorb tiny negative values that can
/// arise from roundoff in `k(x*, x*) - k_*^T K^{-1} k_*`.
pub fn posterior(
    k_chol: &Cholesky<f64, Dyn>,
    x_train: &[Vec<f64>],
    y_train: &DVector<f64>,
    x_star: &[f64],
    signal_var: f64,
    lengthscale: f64,
) -> (f64, f64) {
    let k_star = DVector::<f64>::from_iterator(
        x_train.len(),
        x_train
            .iter()
            .map(|x| rbf(x, x_star, signal_var, lengthscale)),
    );
    let alpha = k_chol.solve(y_train);
    let mean = k_star.dot(&alpha);

    let v = k_chol.solve(&k_star);
    let k_xx = rbf(x_star, x_star, signal_var, lengthscale);
    let variance = (k_xx - k_star.dot(&v)).max(0.0);
    (mean, variance)
}
