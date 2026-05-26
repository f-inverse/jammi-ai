//! Radial Basis Function (squared exponential) kernel.

use nalgebra::DMatrix;

/// RBF kernel `k(x, y) = signal_var * exp(-||x - y||² / (2 lengthscale²))`.
///
/// `lengthscale` controls how quickly the kernel decays with distance —
/// large values produce smoother priors. `signal_var` scales the kernel's
/// peak value (the prior variance at `x == y`, before any noise).
pub fn rbf(x: &[f64], y: &[f64], signal_var: f64, lengthscale: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let sq_dist: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    signal_var * (-sq_dist / (2.0 * lengthscale * lengthscale)).exp()
}

/// Build the full `n × n` Gram matrix for `points` under the RBF kernel,
/// adding `noise_var` to the diagonal. Output `K[i][j] = rbf(points[i],
/// points[j], signal_var, lengthscale)` with `K[i][i] += noise_var`.
pub fn rbf_gram(
    points: &[Vec<f64>],
    signal_var: f64,
    lengthscale: f64,
    noise_var: f64,
) -> DMatrix<f64> {
    let n = points.len();
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut v = rbf(&points[i], &points[j], signal_var, lengthscale);
            if i == j {
                v += noise_var;
            }
            k[(i, j)] = v;
        }
    }
    k
}
