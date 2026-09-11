pub mod exact;
pub mod peer;
pub mod placed;
pub mod segment;
pub mod sidecar;

pub use peer::{
    AllLocal, NoPeers, PeerAddr, PeerError, PeerFailureCounters, PeerFailureReason, PeerTransport,
    SegmentPlacement, SegmentSearchPhase, StaticPlacement, PEER_FAILURE_LABELS, PEER_RPC_DEADLINE,
};
pub use placed::{PlacedIndex, SegmentSource};

pub use segment::{SegmentId, SegmentedIndex, DEFAULT_SEGMENT_OVERFETCH_FACTOR};

use crate::error::Result;

/// Whether a distance may enter a merge, a rank, or a result.
///
/// The domain is exactly `is_finite`, and it is the SAME predicate on both
/// production surfaces — the remote one
/// ([`placed`]'s response reconciliation, where a violation is a peer's fault:
/// [`PeerFailureReason::Malformed`]) and the local one ([`segment`]'s kernels,
/// where it is a broken index: a [`crate::error::JammiError`] naming the
/// segment).
///
/// Why finiteness and nothing more, measured rather than assumed:
///
/// - A distance is the merge's SORT KEY (`total_cmp`) and the user-visible
///   similarity (`1.0 - distance`). `-NaN` and `-inf` sort before every honest
///   distance, so one poisoned value takes the whole top-`k`.
/// - No RANGE bound. `cosine_distance(v, v)` measures a small NEGATIVE number
///   on ordinary high-dimensional vectors (`dot / denom` rounds just above
///   `1.0`), and a self-hit is the commonest query there is — a `[0, 2]` check
///   would refuse honest owners on ordinary traffic.
/// - No normalisation. usearch's `cos` answers `0.0` for zero-vs-zero where
///   `cosine_distance` answers `1.0`; folding them together would silently
///   redefine similarity, so that divergence is documented and pinned on both
///   sides instead.
/// - Nothing here is needed to defend against a zero corpus ROW: usearch
///   2.25.1 guards zero magnitude explicitly, and every precision was measured
///   finite for one (F32 `1.0`, F16 `1.0`, Int8 `0.6464466`, Binary a finite
///   bit count). The real producers of a non-finite distance are a non-finite
///   COMPONENT in a stored vector and a peer that sends one.
///
/// [`PeerFailureReason::Malformed`]: peer::PeerFailureReason::Malformed
pub fn distance_is_admissible(distance: f32) -> bool {
    distance.is_finite()
}

/// The first `(row_id, distance)` whose distance is inadmissible, if any —
/// the shape both surfaces scan their own output with.
pub fn first_inadmissible_hit(hits: &[(String, f32)]) -> Option<&(String, f32)> {
    hits.iter().find(|(_, d)| !distance_is_admissible(*d))
}

/// The version string of the ANN index backend (USearch) this build links.
///
/// Recorded in sidecar manifests and measurement reports so a reader can reject
/// a recall curve or a loaded graph produced against a different backend —
/// recall and the serialized graph format are both backend-version-dependent,
/// and the serialized header only carries the major version.
pub fn backend_version() -> &'static str {
    usearch::version()
}

/// Trait for ANN vector indexes keyed by `_row_id`.
pub trait VectorIndex: Send + Sync {
    /// Add a vector with its row ID to the index.
    fn add(&mut self, row_id: &str, vector: &[f32]) -> Result<()>;

    /// Build the index graph. Must be called after all `add()` calls.
    fn build(&mut self) -> Result<()>;

    /// Search for the `k` nearest neighbors, returning `(row_id, cosine_distance)` sorted ascending.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>>;

    /// Persist the index to disk.
    fn save(&self, path: &std::path::Path) -> Result<()>;

    /// Number of vectors currently in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
