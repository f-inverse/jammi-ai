//! Ordinary least-squares line through `(x_i, y_i)`, with its residual.
//!
//! The closed form over centered sums: `slope = Sxy / Sxx`, `intercept =
//! mean(y) - slope * mean(x)`. Centering before summing keeps the two
//! coefficients from cancelling catastrophically when `x` spans several
//! orders of magnitude (a size sweep from tens to millions of rows).

use crate::error::{NumericsError, Result};
use crate::stats::types::LinearFit;

/// Fit `y = intercept + slope * x` by least squares.
///
/// [`LinearFit::residual_rms`] is what tells a caller whether a line
/// describes the data at all: two coefficients are always returned, but they
/// only *mean* "a fixed cost and a per-unit cost" when the residual is small
/// against the fitted values.
///
/// # Errors
///
/// [`NumericsError::DimensionMismatch`] when `xs` and `ys` differ in length;
/// [`NumericsError::InvalidInput`] when fewer than two points are given, when
/// any coordinate is non-finite, or when every `x` is the same value (the
/// slope is then undetermined — a vertical scatter has no line through it).
pub fn linear_fit(xs: &[f64], ys: &[f64]) -> Result<LinearFit> {
    if xs.len() != ys.len() {
        return Err(NumericsError::DimensionMismatch {
            expected: xs.len(),
            got: ys.len(),
        });
    }
    let n = xs.len();
    if n < 2 {
        return Err(NumericsError::InvalidInput(format!(
            "a line needs at least 2 points, got {n}"
        )));
    }
    if xs.iter().chain(ys).any(|v| !v.is_finite()) {
        return Err(NumericsError::InvalidInput(
            "linear fit requires finite coordinates".into(),
        ));
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let (sxx, sxy) = xs.iter().zip(ys).fold((0.0, 0.0), |(sxx, sxy), (x, y)| {
        let dx = x - mean_x;
        (sxx + dx * dx, sxy + dx * (y - mean_y))
    });
    if sxx == 0.0 {
        return Err(NumericsError::InvalidInput(
            "linear fit requires at least two distinct x values".into(),
        ));
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let residual_rms = (xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum::<f64>()
        / nf)
        .sqrt();
    Ok(LinearFit {
        intercept,
        slope,
        residual_rms,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn recovers_an_exact_line_with_zero_residual() {
        let xs = [16.0, 256.0, 4096.0, 65536.0];
        let ys: Vec<f64> = xs.iter().map(|x| 0.003 + 2.5e-6 * x).collect();
        let fit = linear_fit(&xs, &ys).unwrap();
        assert_relative_eq!(fit.intercept, 0.003, epsilon = 1e-12);
        assert_relative_eq!(fit.slope, 2.5e-6, epsilon = 1e-15);
        assert!(fit.residual_rms < 1e-12);
        assert_eq!(fit.n, 4);
    }

    #[test]
    fn a_curve_leaves_a_residual_a_line_does_not() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let quadratic: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let fit = linear_fit(&xs, &quadratic).unwrap();
        assert!(fit.residual_rms > 0.5, "residual {}", fit.residual_rms);
    }

    #[test]
    fn a_shifted_intercept_leaves_the_slope_alone() {
        let xs = [10.0, 100.0, 1000.0];
        let base: Vec<f64> = xs.iter().map(|x| 1.0 + 0.01 * x).collect();
        let shifted: Vec<f64> = base.iter().map(|y| y + 5.0).collect();
        let a = linear_fit(&xs, &base).unwrap();
        let b = linear_fit(&xs, &shifted).unwrap();
        assert_relative_eq!(a.slope, b.slope, epsilon = 1e-12);
        assert_relative_eq!(b.intercept - a.intercept, 5.0, epsilon = 1e-9);
    }

    #[test]
    fn refuses_degenerate_input() {
        assert!(linear_fit(&[1.0], &[1.0]).is_err());
        assert!(linear_fit(&[1.0, 2.0], &[1.0]).is_err());
        assert!(linear_fit(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(linear_fit(&[1.0, f64::NAN], &[1.0, 2.0]).is_err());
    }
}
