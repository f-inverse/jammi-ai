//! Reciprocal-rank fusion (RRF) of N ranked retrieval lists.
//!
//! Dense (cosine) and lexical (BM25) retrievers score on **incompatible
//! scales**, so fusing their raw scores is meaningless. RRF sidesteps the
//! scale problem entirely: it fuses on *rank*, summing `1 / (k_rrf + rank)`
//! across the lists a row appears in (Cormack et al., SIGIR 2009). A row's
//! fused score therefore depends only on where it landed in each list, never
//! on the magnitude of any underlying score — the property the substrate's
//! hybrid-retrieval gate asserts.
//!
//! The operator is N-ary: it fuses the ANN list, the lexical list, and any
//! further ranked channel (e.g. an S9 graph-retrieval list) uniformly. Each
//! input is a ranked list of `_row_id`s in best-first order; the fuser reads
//! rank from position, so callers need not pre-compute it.

use std::collections::HashMap;

/// Cormack's default `k_rrf`. Robust across `[40, 80]`; damps the contribution
/// of deep ranks so a row's top-of-list appearances dominate.
pub const DEFAULT_K_RRF: u32 = 60;

/// One fused result: a row id and its summed reciprocal-rank score.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedHit {
    pub row_id: String,
    pub rrf_score: f64,
}

/// Fuse `ranked_lists` by reciprocal-rank fusion under `k_rrf`.
///
/// Each entry of `ranked_lists` is one retriever's output: an ordered slice of
/// `_row_id`s, best-first (position 0 is rank 0). A row's fused score is
/// `Σ_lists 1 / (k_rrf + rank_in_list + 1)` — the `+1` makes the best rank
/// contribute `1/(k_rrf+1)` rather than `1/k_rrf`, matching the canonical
/// 1-based-rank formulation while keeping the call site 0-based.
///
/// The result is sorted by fused score descending; ties break ascending by
/// `_row_id`, so the output is **fully deterministic** and independent of the
/// order the lists are supplied in. A row repeated within a single list counts
/// only at its first (best) occurrence in that list.
///
/// `k_rrf` is exposed, not forced; [`DEFAULT_K_RRF`] is the recommended value.
pub fn rrf_fuse<S: AsRef<str>>(ranked_lists: &[Vec<S>], k_rrf: u32) -> Vec<FusedHit> {
    let k = f64::from(k_rrf);
    let mut scores: HashMap<&str, f64> = HashMap::new();

    for list in ranked_lists {
        // First occurrence wins within a list — a retriever ranking the same
        // row twice must not double-count it.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (rank, id) in list.iter().enumerate() {
            let id = id.as_ref();
            if !seen.insert(id) {
                continue;
            }
            let contribution = 1.0 / (k + (rank as f64) + 1.0);
            *scores.entry(id).or_insert(0.0) += contribution;
        }
    }

    let mut fused: Vec<FusedHit> = scores
        .into_iter()
        .map(|(id, rrf_score)| FusedHit {
            row_id: id.to_string(),
            rrf_score,
        })
        .collect();

    fused.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.row_id.cmp(&b.row_id))
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(hits: &[FusedHit]) -> Vec<&str> {
        hits.iter().map(|h| h.row_id.as_str()).collect()
    }

    #[test]
    fn single_list_preserves_order() {
        let fused = rrf_fuse(&[vec!["a", "b", "c"]], DEFAULT_K_RRF);
        assert_eq!(ids(&fused), vec!["a", "b", "c"]);
    }

    #[test]
    fn row_in_both_lists_outranks_row_in_one() {
        // `b` is mid-pack in each list but appears in both; `a`/`x` top one
        // list each but appear in only one. RRF rewards cross-list agreement.
        let dense = vec!["a", "b", "c"];
        let lexical = vec!["x", "b", "y"];
        let fused = rrf_fuse(&[dense, lexical], DEFAULT_K_RRF);
        assert_eq!(fused[0].row_id, "b");
    }

    #[test]
    fn fusion_is_independent_of_list_order() {
        let dense = vec!["a", "b", "c", "d"];
        let lexical = vec!["d", "c", "b", "a"];
        let one = rrf_fuse(&[dense.clone(), lexical.clone()], DEFAULT_K_RRF);
        let two = rrf_fuse(&[lexical, dense], DEFAULT_K_RRF);
        assert_eq!(one, two);
    }

    #[test]
    fn fusion_is_score_scale_free() {
        // Two retrievers, identical *rankings* but wildly different raw scales,
        // produce the same fused order — RRF never sees the scores, only ranks.
        // (We model the scale-invariance by feeding only ranks: any monotone
        // re-scaling of either retriever leaves its rank list unchanged, hence
        // the fused result unchanged.)
        let a = vec!["p", "q", "r"];
        let b = vec!["p", "q", "r"];
        let fused = rrf_fuse(&[a, b], DEFAULT_K_RRF);
        assert_eq!(ids(&fused), vec!["p", "q", "r"]);
    }

    #[test]
    fn deterministic_tie_break_by_row_id() {
        // `a` and `b` each appear once at the same rank in disjoint lists, so
        // their fused scores are equal; the tie breaks ascending by id.
        let fused = rrf_fuse(&[vec!["b"], vec!["a"]], DEFAULT_K_RRF);
        assert_eq!(ids(&fused), vec!["a", "b"]);
    }

    #[test]
    fn three_lists_fuse_uniformly() {
        // The third list models an S9 graph-retrieval channel — fused exactly
        // like the dense and lexical lists, no special-casing.
        let dense = vec!["a", "b"];
        let lexical = vec!["c", "a"];
        let graph = vec!["a", "d"];
        let fused = rrf_fuse(&[dense, lexical, graph], DEFAULT_K_RRF);
        // `a` is in all three; it must lead.
        assert_eq!(fused[0].row_id, "a");
    }

    #[test]
    fn empty_input_yields_empty() {
        let fused: Vec<FusedHit> = rrf_fuse::<&str>(&[], DEFAULT_K_RRF);
        assert!(fused.is_empty());
    }

    #[test]
    fn duplicate_within_list_counts_once_at_best_rank() {
        // `a` appears at rank 0 and rank 2 of the same list; it must score as
        // a single rank-0 appearance, not two.
        let with_dup = rrf_fuse(&[vec!["a", "b", "a"]], DEFAULT_K_RRF);
        let without = rrf_fuse(&[vec!["a", "b"]], DEFAULT_K_RRF);
        let score =
            |hits: &[FusedHit], id: &str| hits.iter().find(|h| h.row_id == id).unwrap().rrf_score;
        assert_eq!(score(&with_dup, "a"), score(&without, "a"));
    }

    #[test]
    fn k_rrf_scales_scores_and_damps_rank_gaps() {
        // `k_rrf` enters the score as `1 / (k + rank + 1)`, so two contracts
        // hold and this test pins both:
        //
        //   1. A larger k yields smaller per-appearance scores (the whole
        //      ranking's magnitudes shrink), and
        //   2. a larger k *flattens the gap* between adjacent ranks — the ratio
        //      of a rank-0 score to a rank-1 score shrinks toward 1 as k grows.
        let list = vec!["r0", "r1"];

        let small = rrf_fuse(&[list.clone()], 1);
        let large = rrf_fuse(&[list], 1000);

        let score =
            |hits: &[FusedHit], id: &str| hits.iter().find(|h| h.row_id == id).unwrap().rrf_score;

        // (1) Larger k damps the rank-0 contribution.
        assert!(score(&small, "r0") > score(&large, "r0"));

        // (2) Larger k flattens the rank-0 vs rank-1 gap.
        let small_ratio = score(&small, "r0") / score(&small, "r1");
        let large_ratio = score(&large, "r0") / score(&large, "r1");
        assert!(small_ratio > large_ratio);
        assert!(large_ratio > 1.0, "rank-0 still outscores rank-1 at any k");
    }
}
