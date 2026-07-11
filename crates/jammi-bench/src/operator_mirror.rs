//! A benchmark-side mirror of the engine's two-stage retrieve→rescore search,
//! built from the same public engine primitives production uses
//! ([`SidecarIndex::search`], [`SidecarIndex::get_exact`],
//! [`jammi_numerics::distance::cosine_distance`]).
//!
//! The real implementation lives in `jammi_ai::operator::ann_search_exec` as a
//! private function of a `DataFusion` physical-plan node — a server-execution
//! concern jammi-bench has no business depending on (the bench measures the
//! engine's storage/index *primitives*, not its query-planning surface). This
//! module recomputes the same two-stage arithmetic directly over those
//! primitives so the recall harness can measure the actual retrieve→rescore
//! *mechanism* a quantized sidecar search performs, not an approximation of
//! it — the oversampled candidate retrieval, the exact-vector rescore, the
//! re-sort, and the truncation are bit-for-bit the same steps production
//! runs, just invoked here instead of from inside a physical-plan node.

use jammi_db::error::{JammiError, Result};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

/// Retrieve→rescore over a quantized [`SidecarIndex`]: `search` an
/// oversampled `k * oversample` candidate set off the (possibly lossy)
/// stored graph, look up each candidate's *exact* `f32` vector via
/// [`SidecarIndex::get_exact`], recompute cosine distance against `query`,
/// then re-sort and truncate to `k`.
///
/// Ties on distance break on `row_id` ascending, matching the exact oracle's
/// tie-break, so the re-ranked order is deterministic given identical inputs.
/// A candidate whose exact vector is missing from the rescore companion (a
/// corrupted or torn bundle) is a hard error, never a silent drop — a result
/// set that quietly shrank below `k` would look like "fewer neighbours exist"
/// rather than "the index is broken".
pub fn retrieve_then_rescore(
    index: &SidecarIndex,
    query: &[f32],
    k: usize,
    oversample: usize,
) -> Result<Vec<(String, f32)>> {
    let candidate_k = k.saturating_mul(oversample).max(k);
    let candidates = index.search(query, candidate_k)?;

    let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidates.len());
    for (row_id, _quantized_distance) in candidates {
        let exact = index.get_exact(&row_id)?.ok_or_else(|| {
            JammiError::Other(format!(
                "rescore: candidate '{row_id}' has no exact vector in the rescore companion \
                 (corrupted or torn sidecar bundle)"
            ))
        })?;
        let distance = jammi_numerics::distance::cosine_distance(query, &exact);
        rescored.push((row_id, distance));
    }

    rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    rescored.truncate(k);
    Ok(rescored)
}
