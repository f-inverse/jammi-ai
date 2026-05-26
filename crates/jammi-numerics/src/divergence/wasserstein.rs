//! 1-D Wasserstein distance (Earth-Mover's distance) between two scalar
//! populations, normalised by the reference range so the result is
//! comparable across populations on different scales.

use crate::histogram::binning::padded_range;
use crate::histogram::interpolate::interpolate_to;

/// 1-D Wasserstein distance between `reference` and `current`. Both
/// populations are sorted and linearly interpolated onto a common grid
/// of length `max(|reference|, |current|, 1)`; the mean absolute
/// difference is then divided by the reference's padded range so the
/// kernel is scale-invariant.
pub fn wasserstein_1d(reference: &[f32], current: &[f32]) -> f64 {
    let mut sorted_ref: Vec<f64> = reference.iter().map(|x| *x as f64).collect();
    let mut sorted_cur: Vec<f64> = current.iter().map(|x| *x as f64).collect();
    sorted_ref.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_cur.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted_ref.len().max(sorted_cur.len()).max(1);
    let interp_ref = interpolate_to(&sorted_ref, n);
    let interp_cur = interpolate_to(&sorted_cur, n);
    let sum_abs: f64 = interp_ref
        .iter()
        .zip(interp_cur.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let mean = sum_abs / n as f64;
    let (min, max) = padded_range(reference);
    let scale = (max - min).abs().max(f64::EPSILON);
    mean / scale
}
