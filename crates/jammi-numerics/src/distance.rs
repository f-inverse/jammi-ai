//! Vector primitives: cosine distance / similarity and `f64` norm.
//!
//! The query side of both cosine kernels is a [`ValidatedQuery`]: every
//! component finite and, by the time it reaches a kernel, exactly as wide as
//! the stored vector it is compared with. That is what makes the width
//! `assert!` inside them unreachable — the type is the guard, the assert is
//! the last line a compiler-missed producer would hit.

use crate::query::ValidatedQuery;

/// Cosine distance between two equal-length vectors, defined as
/// `1.0 - cosine_similarity(a, b)`. Returns `1.0` (the maximum distance) when
/// either input has zero MAGNITUDE.
///
/// The lengths must match, and that is enforced in EVERY build profile: the
/// loop reads `b[i]` over `a.len()`, so a shorter `b` reads out of bounds and
/// a longer `b` is silently scored over its prefix — the first is a panic the
/// caller cannot catch, the second a wrong answer nothing reports. This was a
/// `debug_assert_eq!`, i.e. a release no-op, which is exactly where it
/// mattered. The real assert costs nothing measurable: the per-iteration
/// bounds check already dominates (d=768: 514.7 ns debug_assert vs 534.7 ns
/// real assert; d=4: 3.28 vs 3.28).
///
/// Zero MAGNITUDE short-circuits to `1.0`, but a non-finite COMPONENT does
/// not: it makes `denom` NaN, `NaN < EPSILON` is false, and the result is
/// NaN. This function therefore does NOT guarantee a finite result for a
/// non-finite input — callers that feed stored vectors check their own output
/// (`jammi_db::index::distance_is_admissible`).
pub fn cosine_distance(a: &ValidatedQuery, b: &[f32]) -> f32 {
    let a: &[f32] = a;
    assert!(
        a.len() == b.len(),
        "cosine_distance: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
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
/// distance is taken to be 1 by convention). The length match is enforced in
/// every build profile, for the reason [`cosine_distance`] documents.
pub fn cosine_similarity(a: &ValidatedQuery, b: &[f32]) -> f32 {
    let a: &[f32] = a;
    assert!(
        a.len() == b.len(),
        "cosine_similarity: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
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

#[cfg(test)]
mod length_and_domain_tests {
    use super::*;
    use crate::query::{validate_query, QuerySource, ValidatedQuery};

    /// A test query validated with no width in hand — so the kernel's own
    /// width assert is the line under test.
    fn vq(v: &[f32]) -> ValidatedQuery {
        validate_query(v.to_vec(), None, QuerySource::Caller).unwrap()
    }

    /// A length mismatch is REFUSED, in every build profile. The loop reads
    /// `b[i]` over `a.len()`, so a shorter `b` indexed out of bounds (a panic
    /// only because the slice bounds-checks) and a longer `b` was silently
    /// scored over its prefix — and the old `debug_assert_eq!` was a release
    /// no-op, so neither was caught where it mattered. The measured cost of
    /// the real assert is nil: the per-iteration bounds check dominates.
    #[test]
    #[should_panic(expected = "cosine_distance: length mismatch")]
    fn cosine_distance_refuses_a_longer_b() {
        let _ = cosine_distance(&vq(&[1.0, 0.0]), &[1.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "cosine_distance: length mismatch")]
    fn cosine_distance_refuses_a_shorter_b() {
        let _ = cosine_distance(&vq(&[1.0, 0.0, 0.0]), &[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "cosine_similarity: length mismatch")]
    fn cosine_similarity_refuses_a_length_mismatch() {
        let _ = cosine_similarity(&vq(&[1.0, 0.0]), &[1.0, 0.0, 0.0]);
    }

    /// The measured self-hit: `cosine_distance(v, v)` on an ordinary
    /// high-dimensional vector is a SMALL NEGATIVE number, not `0.0` —
    /// `dot / denom` rounds just above `1.0`. Pinned because it is the reason
    /// the admissibility domain is `is_finite` and NOT a `[0, 2]` range: a
    /// range check would refuse the commonest query there is (a self-hit:
    /// `search_by_id`, `context_set`'s exclude-self, every oracle's own
    /// queries).
    #[test]
    fn a_self_hit_is_slightly_negative_and_finite() {
        let v: Vec<f32> = (0..384).map(|i| ((i % 17) as f32 + 1.0) * 0.031).collect();
        let d = cosine_distance(&vq(&v), &v);
        assert!(d.is_finite(), "a self-hit is finite: {d:?}");
        assert!(
            d < 0.0 && d > -1.0e-6,
            "a self-hit rounds just below zero (measured -1.19e-7 class): {d:?}"
        );
    }

    /// A NaN/inf COMPONENT in the STORED vector defeats the zero-magnitude
    /// guard: `denom` is NaN, `NaN < EPSILON` is false, and the result is NaN.
    /// The query side cannot carry one (it is a `ValidatedQuery`), which is
    /// exactly why every SINK admits its distances — the written invariant
    /// that `cosine_distance` never produces NaN was false.
    #[test]
    fn a_non_finite_component_yields_a_non_finite_distance() {
        assert!(cosine_distance(&vq(&[1.0, 1.0]), &[f32::NAN, 1.0]).is_nan());
        assert!(cosine_distance(&vq(&[1.0, 1.0]), &[f32::INFINITY, 1.0]).is_nan());
    }

    /// The zero-vs-zero boundary, pinned as a KNOWN DIVERGENCE rather than
    /// erased: jammi-numerics answers `1.0` (distance) / `0.0` (similarity)
    /// by the convention documented above, while usearch's own `cos` metric
    /// answers `0.0` for zero-vs-zero (its explicit zero-magnitude guard).
    /// Normalising the two would silently redefine F32 similarity, so the
    /// divergence is documented and tested, not removed.
    #[test]
    fn zero_vs_zero_is_one_here_and_zero_in_usearch() {
        assert_eq!(cosine_distance(&vq(&[0.0, 0.0]), &[0.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&vq(&[0.0, 0.0]), &[0.0, 0.0]), 0.0);
        // usearch's side of the divergence is pinned where a real index can
        // be built: `jammi_db::index::segment`'s zero-row oracle.
    }
}
