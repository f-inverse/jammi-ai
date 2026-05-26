//! Cosine-similarity shift: mean within-reference cosine similarity
//! (computed once at snapshot time) minus the mean cosine similarity
//! across the current vector population.

use crate::distance::vector_norm;
use crate::divergence::projection::cosine_f64;

/// Cosine-similarity shift: `mean_ref` minus the mean pairwise cosine
/// similarity over `current`.
///
/// `mean_ref` is the within-reference mean pairwise cosine, computed once
/// when the reference snapshot was taken. Positive return value = the
/// current population is less self-similar than the reference; negative
/// = the current population is more self-similar than the reference. The
/// scale is `[-1, +2]` and is **not** comparable to JS / PSI / Wasserstein
/// — callers pick a threshold per metric.
pub fn cosine_similarity_shift(mean_ref: f64, current: &[Vec<f32>]) -> f64 {
    if current.is_empty() {
        return 0.0;
    }
    let cur_norms: Vec<f64> = current.iter().map(|v| vector_norm(v)).collect();
    let mut sum = 0.0_f64;
    let mut count = 0_f64;
    for i in 0..current.len() {
        for j in (i + 1)..current.len() {
            sum += cosine_f64(&current[i], &current[j], cur_norms[i], cur_norms[j]);
            count += 1.0;
        }
    }
    let mean_cross = if count == 0.0 { 1.0 } else { sum / count };
    mean_ref - mean_cross
}
