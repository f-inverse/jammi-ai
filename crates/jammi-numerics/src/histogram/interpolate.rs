//! 1-D linear interpolation onto a fixed target length. Used by the
//! Wasserstein-1D kernel to align two sorted populations of potentially
//! different sizes onto a common grid.

/// Linearly interpolate `sorted` (already sorted ascending) onto `target_len`
/// evenly-spaced positions. Returns all zeros when `sorted` is empty;
/// returns a clone when the lengths already match.
pub fn interpolate_to(sorted: &[f64], target_len: usize) -> Vec<f64> {
    if sorted.is_empty() {
        return vec![0.0; target_len];
    }
    if sorted.len() == target_len {
        return sorted.to_vec();
    }
    let n_src = sorted.len();
    let mut out = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let pos = if target_len == 1 {
            0.0
        } else {
            i as f64 * (n_src - 1) as f64 / (target_len - 1) as f64
        };
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n_src - 1);
        let frac = pos - lo as f64;
        out.push(sorted[lo] * (1.0 - frac) + sorted[hi] * frac);
    }
    out
}
