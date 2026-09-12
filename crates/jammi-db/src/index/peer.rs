//! The peer vocabulary of the distributed data plane: the types a coordinator
//! and a segment owner exchange, the transport seam they exchange them over,
//! and the placement seam that says which segment lives where.
//!
//! Nothing here speaks a wire protocol. [`PeerTransport`] is the trait the
//! coordinator ([`crate::index::placed::PlacedIndex`]) fans out through; the
//! tonic implementation lives in `jammi-wire` (which owns the generated
//! clients) and is handed to the [`crate::store::ResultStore`] by the session
//! that builds it. [`NoPeers`] — every call is [`PeerFailureReason::Unreachable`]
//! — is the default, so a library process that never wires a transport is
//! exactly today's single-node process.
//!
//! [`SegmentPlacement`] is the second seam: it answers "who owns segment `s` of
//! table `t`" as a rendezvous-ordered candidate list, empty meaning "local".
//! [`AllLocal`] (the default) maps every segment to this process;
//! [`StaticPlacement`] is the explicit library value tests and embedders use.
//! Placement is read on the query path, never from a background loop.
//!
//! Every hit that crosses this seam is `(row_id, distance)` — ids and
//! distances, never vectors.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use crate::config::StoragePrecision;
use crate::index::{SegmentId, ValidatedQuery};

/// The address a coordinator dials an owner at (`host:port`, plaintext gRPC —
/// transport encryption is the runtime's, never the engine's).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PeerAddr(pub String);

impl std::fmt::Display for PeerAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which stage of the per-precision protocol a segment search runs.
///
/// Derived from the table's [`StoragePrecision`] by [`Self::for_precision`]:
/// `F32` needs no rescore and `Binary` cannot merge across segments on its
/// raw per-segment-τ Hamming distances, so both run one `Final` phase (the
/// owner returns exact cosine distances — for `Binary`, after rescoring every
/// hit locally); `F16` / `Int8` approximate distances ARE cross-segment
/// comparable, so they run `Approximate` (the owner returns approximate
/// candidates; the coordinator merges, truncates, then issues one
/// `ExactRescore` per owner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentSearchPhase {
    /// Return the segment's approximate candidates; the caller will rescore
    /// the survivors of its merge through [`PeerTransport::exact_rescore`].
    Approximate,
    /// Return final exact cosine distances: `F32` as-is, `Binary` after a
    /// per-hit rescore at the owner.
    Final,
}

impl SegmentSearchPhase {
    /// The phase the protocol runs for a table at `precision`.
    pub fn for_precision(precision: StoragePrecision) -> Self {
        match precision {
            StoragePrecision::F32 | StoragePrecision::Binary => Self::Final,
            StoragePrecision::F16 | StoragePrecision::Int8 => Self::Approximate,
        }
    }
}

/// One segment's hits for one query: `(row_id, distance)` pairs, never
/// vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentUnit {
    /// The segment these hits came from — the merge's final tie-break and the
    /// key an `ExactRescore` groups survivors under.
    pub segment_id: SegmentId,
    /// `(row_id, distance)`, in the order the owner's search produced them.
    pub hits: Vec<(String, f32)>,
}

/// A search of a set of segments an owner holds for one table.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentSearchRequest {
    /// The table the segments belong to. The owner verifies every id in
    /// `segment_ids` against this table's catalog segment list and refuses the
    /// whole request otherwise.
    pub table_name: String,
    /// Every segment of `table_name` this owner should search — all of one
    /// owner's segments go in ONE request.
    pub segment_ids: Vec<SegmentId>,
    /// The table's persisted precision; the owner's strict load refuses a
    /// bundle stamped otherwise.
    pub storage_precision: StoragePrecision,
    /// The query vector — validated (finite) before it ever crosses the seam;
    /// the owner re-validates what it receives at its own edge.
    pub query: ValidatedQuery,
    /// The per-segment fetch width (`over_fetch(candidate_k, N)`).
    pub width: usize,
    /// Which stage of the protocol this search is.
    pub phase: SegmentSearchPhase,
}

/// The second stage of the `Approximate` protocol: rescore named candidates
/// against their exact vectors at the owner, returning exact cosine distances.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactRescoreRequest {
    /// The table the segments belong to (verified by the owner as above).
    pub table_name: String,
    /// The table's persisted precision.
    pub storage_precision: StoragePrecision,
    /// The query vector (validated, as above).
    pub query: ValidatedQuery,
    /// The candidates to rescore, grouped by the segment that owns each.
    pub row_ids_by_segment: Vec<(SegmentId, Vec<String>)>,
}

/// Why one peer call failed — the label the failure counters and the
/// [`crate::error::JammiError::Unavailable`] reason carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerFailureReason {
    /// The per-RPC deadline elapsed.
    Deadline,
    /// The owner could not be reached (connection refused, no route, the
    /// [`NoPeers`] transport).
    Unreachable,
    /// The owner refused the request at its input edge (an unknown segment
    /// id, a precision that mismatches the bundle).
    Refused,
    /// The owner's bundle is torn (a candidate with no exact vector).
    Torn,
    /// Any other transport-level failure.
    Transport,
    /// The owner answered, but the answer does not reconcile with the request
    /// it was for: a unit for a segment the coordinator never asked for, a
    /// requested segment with no unit or two, a rescore row the coordinator
    /// never named, a named row missing or duplicated. A non-conforming or
    /// version-skewed peer, classified at the coordinator's edge — never a
    /// panic, never a silently short or polluted result.
    Malformed,
    /// The owner refused the REQUEST as invalid (`INVALID_ARGUMENT`): a
    /// caller fault the coordinator missed at its own edge. TERMINAL — a
    /// caller fault is never a ladder rung: no retry at the next candidate,
    /// no local load, no `Unavailable`. Counted so a coordinator that keeps
    /// sending bad requests is visible.
    CallerFault,
}

impl PeerFailureReason {
    /// The stable label this reason is counted and reported under.
    pub fn label(self) -> &'static str {
        match self {
            Self::Deadline => "deadline",
            Self::Unreachable => "unreachable",
            Self::Refused => "refused",
            Self::Torn => "torn",
            Self::Transport => "transport",
            Self::Malformed => "malformed",
            Self::CallerFault => "caller_fault",
        }
    }
}

impl std::fmt::Display for PeerFailureReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// One failed peer call, naming the segment, the owner it was addressed to,
/// and why. Also the shape a non-conforming ANSWER is reported as
/// ([`PeerFailureReason::Malformed`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerError {
    /// The segment the call was for (the first requested id of a multi-segment
    /// request).
    pub segment: SegmentId,
    /// The owner the call was addressed to.
    pub owner: PeerAddr,
    /// The classified failure.
    pub reason: PeerFailureReason,
    /// The owner's own message, when the failure carries one (a real `Status`
    /// the owner returned). Empty for a reason with no owner-side text (a
    /// transport failure, a reconciliation fault the COORDINATOR detected).
    /// [`PeerFailureReason::CallerFault`]'s surfaced error names it.
    pub message: String,
}

impl std::fmt::Display for PeerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "peer {} for segment {}: {}",
            self.owner, self.segment.0, self.reason
        )?;
        if !self.message.is_empty() {
            write!(f, " ({})", self.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for PeerError {}

/// The transport a coordinator fans a segment search out through. Implemented
/// over gRPC in `jammi-wire`; [`NoPeers`] here.
#[async_trait]
pub trait PeerTransport: Send + Sync {
    /// Search `req.segment_ids` at `owner`, bounded by `deadline`.
    async fn segment_search(
        &self,
        owner: &PeerAddr,
        req: &SegmentSearchRequest,
        deadline: Duration,
    ) -> Result<Vec<SegmentUnit>, PeerError>;

    /// Rescore the named candidates at `owner`, bounded by `deadline`,
    /// returning `(row_id, exact_cosine_distance)` for every candidate.
    async fn exact_rescore(
        &self,
        owner: &PeerAddr,
        req: &ExactRescoreRequest,
        deadline: Duration,
    ) -> Result<Vec<(String, f32)>, PeerError>;
}

/// The default transport: no peer is reachable. A store built without a
/// transport behaves exactly as a single-node store — every remote call fails
/// [`PeerFailureReason::Unreachable`] and the failure ladder decides.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoPeers;

#[async_trait]
impl PeerTransport for NoPeers {
    async fn segment_search(
        &self,
        owner: &PeerAddr,
        req: &SegmentSearchRequest,
        _deadline: Duration,
    ) -> Result<Vec<SegmentUnit>, PeerError> {
        Err(PeerError {
            segment: req.segment_ids.first().copied().unwrap_or(SegmentId(-1)),
            owner: owner.clone(),
            reason: PeerFailureReason::Unreachable,
            message: String::new(),
        })
    }

    async fn exact_rescore(
        &self,
        owner: &PeerAddr,
        req: &ExactRescoreRequest,
        _deadline: Duration,
    ) -> Result<Vec<(String, f32)>, PeerError> {
        Err(PeerError {
            segment: req
                .row_ids_by_segment
                .first()
                .map(|(s, _)| *s)
                .unwrap_or(SegmentId(-1)),
            owner: owner.clone(),
            reason: PeerFailureReason::Unreachable,
            message: String::new(),
        })
    }
}

/// Which process owns which segment. Empty = local; non-empty = the
/// rendezvous-ordered candidate list (first = the owner, second = the one
/// retry). Read on the query path at every resolve — never cached, never
/// refreshed by a loop of the engine's own.
#[async_trait]
pub trait SegmentPlacement: Send + Sync {
    /// The owners of `segment` of `table`, nearest first; empty means this
    /// process serves it locally.
    async fn owners(&self, table: &str, segment: SegmentId) -> Vec<PeerAddr>;
}

/// The default placement: every segment is local. A process with this
/// placement is a single node regardless of what transport it holds.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllLocal;

#[async_trait]
impl SegmentPlacement for AllLocal {
    async fn owners(&self, _table: &str, _segment: SegmentId) -> Vec<PeerAddr> {
        Vec::new()
    }
}

/// An explicit placement table: `(table_name, segment_id)` → owners. The
/// library value for an embedder that knows its topology, and the fixture the
/// end-to-end oracles drive. A pair absent from the map is local.
#[derive(Debug, Default, Clone)]
pub struct StaticPlacement(pub BTreeMap<(String, SegmentId), Vec<PeerAddr>>);

#[async_trait]
impl SegmentPlacement for StaticPlacement {
    async fn owners(&self, table: &str, segment: SegmentId) -> Vec<PeerAddr> {
        self.0
            .get(&(table.to_string(), segment))
            .cloned()
            .unwrap_or_default()
    }
}

/// Per-RPC deadline for one peer call. The local-load rung of the failure
/// ladder gets twice this.
pub const PEER_RPC_DEADLINE: Duration = Duration::from_secs(2);

/// One counter per failure-ladder outcome, read at scrape by the server's
/// metrics registry as `jammi_peer_search_failures_total{reason}` and by the
/// oracles directly. Every label is monotonic.
#[derive(Debug, Default)]
pub struct PeerFailureCounters {
    /// A peer call's deadline elapsed.
    pub deadline: AtomicU64,
    /// A peer was unreachable.
    pub unreachable: AtomicU64,
    /// A peer refused the request at its input edge.
    pub refused: AtomicU64,
    /// A peer's bundle was torn.
    pub torn: AtomicU64,
    /// Any other transport failure.
    pub transport: AtomicU64,
    /// A peer's answer did not reconcile with the request it was for.
    pub malformed: AtomicU64,
    /// An owner refused the request as invalid — terminal, never a rung.
    pub caller_fault: AtomicU64,
    /// The retry at the second rendezvous candidate succeeded.
    pub retry_ok: AtomicU64,
    /// A remote segment was loaded locally under the admission budget.
    pub local_load: AtomicU64,
    /// The ladder was exhausted: the query failed `Unavailable`.
    pub unavailable: AtomicU64,
}

/// The label set [`PeerFailureCounters`] exposes, in a stable order.
pub const PEER_FAILURE_LABELS: [&str; 10] = [
    "deadline",
    "unreachable",
    "refused",
    "torn",
    "transport",
    "malformed",
    "caller_fault",
    "retry_ok",
    "local_load",
    "unavailable",
];

impl PeerFailureCounters {
    /// Count one failed peer call under its reason.
    pub fn record(&self, reason: PeerFailureReason) {
        let counter = match reason {
            PeerFailureReason::Deadline => &self.deadline,
            PeerFailureReason::Unreachable => &self.unreachable,
            PeerFailureReason::Refused => &self.refused,
            PeerFailureReason::Torn => &self.torn,
            PeerFailureReason::Transport => &self.transport,
            PeerFailureReason::Malformed => &self.malformed,
            PeerFailureReason::CallerFault => &self.caller_fault,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the counter under `label` (one of [`PEER_FAILURE_LABELS`]).
    /// `None` for an unknown label.
    pub fn get(&self, label: &str) -> Option<u64> {
        let counter = match label {
            "deadline" => &self.deadline,
            "unreachable" => &self.unreachable,
            "refused" => &self.refused,
            "torn" => &self.torn,
            "transport" => &self.transport,
            "malformed" => &self.malformed,
            "caller_fault" => &self.caller_fault,
            "retry_ok" => &self.retry_ok,
            "local_load" => &self.local_load,
            "unavailable" => &self.unavailable,
            _ => return None,
        };
        Some(counter.load(Ordering::Relaxed))
    }

    /// Every `(label, value)` pair, in [`PEER_FAILURE_LABELS`] order — the
    /// snapshot a scrape or a delta assertion takes.
    pub fn snapshot(&self) -> Vec<(&'static str, u64)> {
        PEER_FAILURE_LABELS
            .iter()
            .map(|label| (*label, self.get(label).expect("label is in the fixed set")))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_follows_precision() {
        assert_eq!(
            SegmentSearchPhase::for_precision(StoragePrecision::F32),
            SegmentSearchPhase::Final
        );
        assert_eq!(
            SegmentSearchPhase::for_precision(StoragePrecision::Binary),
            SegmentSearchPhase::Final
        );
        assert_eq!(
            SegmentSearchPhase::for_precision(StoragePrecision::F16),
            SegmentSearchPhase::Approximate
        );
        assert_eq!(
            SegmentSearchPhase::for_precision(StoragePrecision::Int8),
            SegmentSearchPhase::Approximate
        );
    }

    #[tokio::test]
    async fn no_peers_is_unreachable_and_all_local_owns_nothing() {
        let owner = PeerAddr("127.0.0.1:1".into());
        let req = SegmentSearchRequest {
            table_name: "t".into(),
            segment_ids: vec![SegmentId(3)],
            storage_precision: StoragePrecision::F32,
            query: crate::index::validate_query(vec![1.0], None, crate::index::QuerySource::Caller)
                .unwrap(),
            width: 1,
            phase: SegmentSearchPhase::Final,
        };
        let err = NoPeers
            .segment_search(&owner, &req, PEER_RPC_DEADLINE)
            .await
            .unwrap_err();
        assert_eq!(err.reason, PeerFailureReason::Unreachable);
        assert_eq!(err.segment, SegmentId(3));
        assert!(AllLocal.owners("t", SegmentId(0)).await.is_empty());
        let placed = StaticPlacement(BTreeMap::from([(
            ("t".to_string(), SegmentId(1)),
            vec![owner.clone()],
        )]));
        assert_eq!(placed.owners("t", SegmentId(1)).await, vec![owner]);
        assert!(placed.owners("t", SegmentId(0)).await.is_empty());
    }

    #[test]
    fn counters_record_under_their_label() {
        let c = PeerFailureCounters::default();
        c.record(PeerFailureReason::Unreachable);
        c.record(PeerFailureReason::Unreachable);
        c.record(PeerFailureReason::Torn);
        assert_eq!(c.get("unreachable"), Some(2));
        assert_eq!(c.get("torn"), Some(1));
        assert_eq!(c.get("deadline"), Some(0));
        assert_eq!(c.get("bogus"), None);
        assert_eq!(c.snapshot().len(), PEER_FAILURE_LABELS.len());
    }
}
