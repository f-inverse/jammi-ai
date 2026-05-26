//! Random 1-D projection of a high-dimensional vector population.
//!
//! Divergence kernels (Jensen-Shannon, PSI, 1D Wasserstein) operate on
//! scalar populations. To apply them to vectors, the reference and the
//! current population are both projected onto the same randomly chosen
//! unit hyperplane, producing two scalar populations that can then be
//! binned and compared.
//!
//! The projection is **seeded** ([`PROJECTION_SEED`]) so that the
//! baseline reference and every production tick project against an
//! identical hyperplane — making the resulting histograms comparable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::distance::vector_norm;

/// Random-projection seed. Pinned for reproducibility across the
/// resilience baseline and every monitor evaluation. ASCII "Jammi_DV".
pub const PROJECTION_SEED: u64 = 0x4A_61_6D_6D_69_5F_44_56;

/// Build a length-`dimensions` random unit projection vector seeded by
/// [`PROJECTION_SEED`]. Components are sampled uniformly in `[-1, 1)` and
/// L2-normalised in `f64` for precision before being cast back to `f32`.
pub fn build_projection_vector(dimensions: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(PROJECTION_SEED);
    let raw: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm = raw
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return raw;
    }
    raw.into_iter().map(|x| (x as f64 / norm) as f32).collect()
}

/// Project a single vector onto the supplied projection axis. The dot
/// product is accumulated in `f64` to limit roundoff before being cast
/// back to `f32`.
pub fn project(v: &[f32], projection: &[f32]) -> f32 {
    v.iter()
        .zip(projection.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum::<f64>() as f32
}

/// Project every vector in `vectors` onto `projection`, returning the
/// resulting scalar population in the same order.
pub fn project_all(vectors: &[Vec<f32>], projection: &[f32]) -> Vec<f32> {
    vectors.iter().map(|v| project(v, projection)).collect()
}

/// Mean pairwise cosine similarity over all `(i, j)` pairs with `i < j`.
/// Returns `1.0` for populations of fewer than two elements (the degenerate
/// "every element is similar to itself" answer).
pub fn mean_pairwise_cosine(vectors: &[Vec<f32>]) -> f64 {
    if vectors.len() < 2 {
        return 1.0;
    }
    let norms: Vec<f64> = vectors.iter().map(|v| vector_norm(v)).collect();
    let mut sum = 0.0_f64;
    let mut count = 0_f64;
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            sum += cosine_f64(&vectors[i], &vectors[j], norms[i], norms[j]);
            count += 1.0;
        }
    }
    if count == 0.0 {
        1.0
    } else {
        sum / count
    }
}

/// Cosine similarity computed in `f64` with the caller supplying the
/// pre-computed `f64` norms. Internal helper for the
/// pairwise-cosine kernels — public `cosine_distance` /
/// `cosine_similarity` live in [`crate::distance`].
pub(crate) fn cosine_f64(a: &[f32], b: &[f32], norm_a: f64, norm_b: f64) -> f64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>();
    dot / (norm_a * norm_b)
}
