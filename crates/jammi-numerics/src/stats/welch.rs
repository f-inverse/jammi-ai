//! Welch's two-sample t-test. Suitable for samples of unequal size and/or
//! unequal variance.

use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::error::{NumericsError, Result};
use crate::stats::types::TestResult;

/// Welch's t-test between two independent samples. Returns the
/// two-tailed p-value via the Student's t distribution evaluated at the
/// Satterthwaite-approximated degrees of freedom.
///
/// Returns `NumericsError::InvalidInput` if either sample has fewer than
/// two elements (sample variance is undefined). When both samples have
/// zero variance the test statistic is forced to `0.0` and the p-value
/// to `1.0` — the populations are constant and indistinguishable.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.len() < 2 || b.len() < 2 {
        return Err(NumericsError::InvalidInput(
            "Welch's t-test requires at least 2 samples per group".into(),
        ));
    }
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;
    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se_sq = var_a / n_a + var_b / n_b;
    if se_sq <= 0.0 {
        return Ok(TestResult {
            statistic: 0.0,
            p_value: 1.0,
        });
    }
    let se = se_sq.sqrt();
    let t = (mean_a - mean_b) / se;

    let df = se_sq.powi(2)
        / (var_a.powi(2) / (n_a * n_a * (n_a - 1.0)) + var_b.powi(2) / (n_b * n_b * (n_b - 1.0)));

    let dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        NumericsError::InvalidInput(format!("Student-t with df={df} unconstructible: {e}"))
    })?;
    let p = 2.0 * (1.0 - dist.cdf(t.abs()));

    Ok(TestResult {
        statistic: t,
        p_value: p.clamp(0.0, 1.0),
    })
}
