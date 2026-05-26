//! Vector primitives: cosine distance / similarity and `f64` norm.

/// Cosine distance between two equal-length vectors, defined as
/// `1.0 - cosine_similarity(a, b)`. Returns `1.0` (the maximum distance) when
/// either input has zero magnitude.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}

/// Cosine similarity between two equal-length vectors. Returns `0.0` when
/// either input has zero magnitude (matching the boundary behaviour of
/// [`cosine_distance`], so that `similarity + distance == 1.0` everywhere
/// except the all-zero degenerate case where similarity is taken to be 0 and
/// distance is taken to be 1 by convention).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// Euclidean norm of a single vector, computed in `f64` to keep
/// downstream divergence and similarity calculations numerically stable.
pub fn vector_norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}
