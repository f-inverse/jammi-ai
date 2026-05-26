use approx::assert_abs_diff_eq;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use jammi_numerics::error::NumericsError;
use jammi_numerics::gp::{
    cholesky_with_jitter, expected_improvement, mle_hyperparams, posterior, rbf_gram,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![lo];
    }
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64))
        .collect()
}

#[test]
fn cholesky_with_jitter_recovers_from_rank_deficient_matrix() {
    // K = [[1, 1], [1, 1]] is positive semi-definite but singular.
    let k = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    let chol = cholesky_with_jitter(k).expect("jitter retry must succeed for tiny PSD case");
    // Reconstruct L L^T and confirm it's close to the original after jitter
    // is added (the diagonal will be slightly larger).
    let l = chol.l();
    let reconstructed = &l * l.transpose();
    assert!((reconstructed[(0, 0)] - 1.0).abs() < 1e-3);
    assert!((reconstructed[(0, 1)] - 1.0).abs() < 1e-3);
}

#[test]
fn cholesky_with_jitter_surfaces_typed_error_when_exhausted() {
    // Negative-definite matrices defeat any positive jitter we add.
    let k = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
    match cholesky_with_jitter(k) {
        Err(NumericsError::IllConditioned { .. }) => {}
        other => panic!("expected IllConditioned, got {other:?}"),
    }
}

#[test]
fn posterior_mean_interpolates_training_points() {
    // Sample a noiseless quadratic; the posterior at training points
    // should reproduce the training targets.
    let x_train: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_train = DVector::from_iterator(5, x_train.iter().map(|x| -((x[0] - 2.0).powi(2))));
    let signal_var = 1.0;
    let lengthscale = 1.0;
    let noise_var = 1e-6;
    let k = rbf_gram(&x_train, signal_var, lengthscale, noise_var);
    let chol = cholesky_with_jitter(k).unwrap();
    for (i, x) in x_train.iter().enumerate() {
        let (mean, _var) = posterior(&chol, &x_train, &y_train, x, signal_var, lengthscale);
        assert_abs_diff_eq!(mean, y_train[i], epsilon = 1e-3);
    }
}

#[test]
fn expected_improvement_zero_when_sigma_is_zero() {
    assert_eq!(expected_improvement(2.0, 0.0, 1.0), 0.0);
}

#[test]
fn expected_improvement_positive_above_best() {
    let ei = expected_improvement(1.5, 0.5, 1.0);
    assert!(ei > 0.0, "EI above best should be positive, got {ei}");
}

#[test]
fn mle_hyperparams_picks_finite_values() {
    let x_train: Vec<Vec<f64>> = linspace(-2.0, 2.0, 8)
        .into_iter()
        .map(|x| vec![x])
        .collect();
    let y_train = DVector::from_iterator(8, x_train.iter().map(|x| (-(x[0] * x[0]) / 2.0).exp()));
    let (sv, ls, nv) = mle_hyperparams(&x_train, &y_train).unwrap();
    assert!(sv.is_finite() && sv > 0.0);
    assert!(ls.is_finite() && ls > 0.0);
    assert!(nv.is_finite() && nv > 0.0);
}

#[test]
fn bayes_opt_converges_on_1d_quadratic() {
    // Maximise f(x) = -(x - 1.7)^2 over [-5, 5]. After 10 random + 20 EI
    // selections from a 200-point candidate grid, the best observation
    // should be within 0.3 of x = 1.7.
    let objective = |x: f64| -((x - 1.7).powi(2));

    let mut rng = StdRng::seed_from_u64(42);
    let mut observed_x = Vec::new();
    let mut observed_y = Vec::new();
    for _ in 0..10 {
        let x = rng.gen_range(-5.0..5.0_f64);
        observed_x.push(vec![x]);
        observed_y.push(objective(x));
    }

    let candidates: Vec<f64> = linspace(-5.0, 5.0, 200);
    let signal_var = 1.0;
    let lengthscale = 1.0;
    let noise_var = 1e-4;

    for _ in 0..20 {
        let k = rbf_gram(&observed_x, signal_var, lengthscale, noise_var);
        let chol = cholesky_with_jitter(k).unwrap();
        let y = DVector::from_vec(observed_y.clone());
        let f_best = observed_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Pick the candidate with the highest EI.
        let (best_x, _) = candidates
            .iter()
            .map(|&c| {
                let (mu, var) = posterior(&chol, &observed_x, &y, &[c], signal_var, lengthscale);
                let sigma = var.sqrt();
                (c, expected_improvement(mu, sigma, f_best))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        observed_x.push(vec![best_x]);
        observed_y.push(objective(best_x));
    }

    let argmax = observed_x
        .iter()
        .zip(observed_y.iter())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(x, _)| x[0])
        .unwrap();

    assert!(
        (argmax - 1.7).abs() < 0.3,
        "BO should converge near x=1.7, got x={argmax}"
    );
}
