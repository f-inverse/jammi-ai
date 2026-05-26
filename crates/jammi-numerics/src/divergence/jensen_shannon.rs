//! Jensen-Shannon divergence between two scalar populations, computed via
//! `log2`-based KL divergence so the result lies in `[0, 1]`.

use crate::histogram::binning::{bin_proportions, padded_range, smooth_and_renormalise};

/// Jensen-Shannon divergence between `reference` and `current` over
/// `num_bins` equal-width bins. Range is taken from `reference` (with
/// padding); both populations are smoothed before the ratio is computed.
/// Returns a value in `[0.0, 1.0]`: `0.0` when the populations are
/// identical, `1.0` when they are maximally disjoint.
pub fn jensen_shannon(reference: &[f32], current: &[f32], num_bins: usize) -> f64 {
    let (min, max) = padded_range(reference);
    let mut p = bin_proportions(reference, min, max, num_bins);
    let mut q = bin_proportions(current, min, max, num_bins);
    smooth_and_renormalise(&mut p);
    smooth_and_renormalise(&mut q);
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
    0.5 * kl_divergence_log2(&p, &m) + 0.5 * kl_divergence_log2(&q, &m)
}

/// Kullback-Leibler divergence computed in base 2. Internal helper.
fn kl_divergence_log2(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi > 0.0 && *qi > 0.0 {
                pi * (pi / qi).log2()
            } else {
                0.0
            }
        })
        .sum()
}
