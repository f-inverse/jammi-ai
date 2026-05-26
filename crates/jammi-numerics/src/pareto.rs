//! Pareto dominance and Pareto-optimal frontier extraction.
//!
//! Under the **minimisation** convention: `a` dominates `b` iff `a[i] <=
//! b[i]` for every coordinate `i` AND `a[j] < b[j]` for at least one
//! coordinate `j`. Callers who want to maximise an objective should
//! negate the corresponding coordinate before calling.

/// Returns `true` iff `a` weakly dominates `b` under minimisation.
///
/// `a.len()` and `b.len()` must agree; mismatched lengths trigger a
/// debug assertion. The all-equal case returns `false` (no strict
/// dimension).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut any_strict = false;
    for (xa, xb) in a.iter().zip(b.iter()) {
        if xa > xb {
            return false;
        }
        if xa < xb {
            any_strict = true;
        }
    }
    any_strict
}

/// Returns the indices of points in `points` that are Pareto-optimal
/// under minimisation (no other point dominates them). Output is in
/// ascending order of index.
pub fn frontier(points: &[Vec<f64>]) -> Vec<usize> {
    (0..points.len())
        .filter(|&i| {
            !points
                .iter()
                .enumerate()
                .any(|(j, p)| j != i && dominates(p, &points[i]))
        })
        .collect()
}
