//! Population Stability Index between two scalar populations.

use crate::histogram::binning::{bin_proportions, padded_range, smooth_and_renormalise};

/// Additive smoothing applied to bin proportions before PSI / JS log ratios
/// to avoid `log(0)` on empty bins. Same epsilon used by both kernels.
pub const PSI_EPSILON: f64 = 1e-6;

/// Population Stability Index between `reference` and `current` over
/// `num_bins` equal-width bins. Range is taken from `reference` (with
/// padding); both populations are smoothed before the ratio is computed.
///
/// Conventional thresholds in industry use: `< 0.1` stable, `< 0.25`
/// borderline, `>= 0.25` significant population shift. The kernel here
/// computes the raw value; thresholding is the caller's job.
pub fn psi(reference: &[f32], current: &[f32], num_bins: usize) -> f64 {
    let (min, max) = padded_range(reference);
    let mut p_ref = bin_proportions(reference, min, max, num_bins);
    let mut p_cur = bin_proportions(current, min, max, num_bins);
    smooth_and_renormalise(&mut p_ref);
    smooth_and_renormalise(&mut p_cur);
    p_ref
        .iter()
        .zip(p_cur.iter())
        .map(|(r, c)| (c - r) * (c / r).ln())
        .sum()
}
