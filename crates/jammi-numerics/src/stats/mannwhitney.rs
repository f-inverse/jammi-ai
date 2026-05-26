//! Mann-Whitney U test (rank-sum) with tie correction.
//!
//! Non-parametric two-sample test: does not assume a distribution shape
//! and is robust to outliers. Uses the normal approximation for p-value
//! computation, which is appropriate when both samples have at least
//! ~8-10 elements.

use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{NumericsError, Result};
use crate::stats::types::TestResult;

/// Mann-Whitney U statistic with tie-corrected variance, returning a
/// two-tailed p-value via the normal approximation.
///
/// Returns `NumericsError::InvalidInput` if either sample is empty.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.is_empty() || b.is_empty() {
        return Err(NumericsError::InvalidInput(
            "Mann-Whitney U requires non-empty samples".into(),
        ));
    }
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    // Build ranks of the combined sample with average ranks for ties.
    let mut combined: Vec<(f64, usize)> = a
        .iter()
        .map(|&x| (x, 0))
        .chain(b.iter().map(|&x| (x, 1)))
        .collect();
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks to ties.
    let mut ranks = vec![0.0_f64; combined.len()];
    let mut tie_groups: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < combined.len() {
        let mut j = i + 1;
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // Indices [i, j) share rank (avg of 1-based positions).
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for slot in &mut ranks[i..j] {
            *slot = avg_rank;
        }
        if j - i > 1 {
            tie_groups.push(j - i);
        }
        i = j;
    }

    let sum_ranks_a: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter_map(|(&(_, g), &r)| (g == 0).then_some(r))
        .sum();
    let u_a = sum_ranks_a - n_a * (n_a + 1.0) / 2.0;
    let u_b = n_a * n_b - u_a;
    let u = u_a.min(u_b);

    let n = n_a + n_b;
    let tie_correction: f64 = tie_groups
        .iter()
        .map(|&t| {
            let tf = t as f64;
            tf.powi(3) - tf
        })
        .sum::<f64>()
        / (n * (n - 1.0)).max(f64::EPSILON);
    let mean_u = n_a * n_b / 2.0;
    let var_u = n_a * n_b * (n + 1.0 - tie_correction) / 12.0;

    if var_u <= 0.0 {
        return Ok(TestResult {
            statistic: u,
            p_value: 1.0,
        });
    }

    let z = (u - mean_u) / var_u.sqrt();
    let standard = Normal::standard();
    let p = 2.0 * (1.0 - standard.cdf(z.abs()));

    Ok(TestResult {
        statistic: u,
        p_value: p.clamp(0.0, 1.0),
    })
}
