//! Histogram binning, smoothing, and range padding used by the divergence
//! kernels.

use crate::divergence::psi::PSI_EPSILON;

/// Default histogram bin count used by Jensen-Shannon, PSI, and 1D
/// Wasserstein. Callers may override per call site.
pub const NUM_BINS: usize = 50;

/// Fractional padding applied to a reference distribution's `(min, max)`
/// range before binning. Current vectors falling slightly past the reference
/// extremes still land in the outer bins instead of clipping a histogram
/// edge to zero (which would make every divergence kernel blow up).
pub const RANGE_PADDING: f64 = 0.01;

/// Bin a projected scalar population into `num_bins` equal-width bins over
/// `[min, max]`. Returns proportions (the bin counts divided by the total
/// element count). Returns all zeros if `projected` is empty.
pub fn bin_proportions(projected: &[f32], min: f64, max: f64, num_bins: usize) -> Vec<f64> {
    let mut bins = vec![0.0_f64; num_bins];
    if projected.is_empty() {
        return bins;
    }
    let range = (max - min).max(f64::EPSILON);
    for v in projected {
        let x = *v as f64;
        let idx = ((x - min) / range * num_bins as f64).floor() as isize;
        let clamped = idx.clamp(0, num_bins as isize - 1) as usize;
        bins[clamped] += 1.0;
    }
    let n = projected.len() as f64;
    for b in &mut bins {
        *b /= n;
    }
    bins
}

/// Additive smoothing: add `PSI_EPSILON` to every bin then renormalise so
/// the proportions sum to 1.0. Avoids `log(0)` in PSI / KL ratios when a
/// bin is empty in one population but populated in the other.
pub fn smooth_and_renormalise(props: &mut [f64]) {
    for p in props.iter_mut() {
        *p += PSI_EPSILON;
    }
    let total: f64 = props.iter().sum();
    if total > 0.0 {
        for p in props.iter_mut() {
            *p /= total;
        }
    }
}

/// Compute the `(min, max)` range over `reference` and pad both ends by
/// [`RANGE_PADDING`] of the span. Falls back to `(0.0, 1.0)` when the
/// population is empty or non-finite.
pub fn padded_range(reference: &[f32]) -> (f64, f64) {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in reference {
        let x = *v as f64;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    let span = (max - min).abs().max(f64::EPSILON);
    let pad = span * RANGE_PADDING;
    (min - pad, max + pad)
}
