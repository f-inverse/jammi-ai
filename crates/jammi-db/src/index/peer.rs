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
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::catalog::Catalog;
use crate::config::StoragePrecision;
use crate::index::{SegmentId, ValidatedQuery};
use crate::store::content_hash::domain_hash;

/// The address a coordinator dials an owner at. Moved to
/// [`crate::catalog::instance::PeerAddr`] (§8 of
/// `docs/rigor/contracts/feat_500-C-U5b-1a.md`): the peer listener this
/// module's transport dials and the gang listener
/// [`crate::catalog::Catalog::list_gang_members`] advertises are the SAME
/// address, so two distinct types here and in the catalog would be a lie.
/// Re-exported so every existing `index::peer::PeerAddr` path keeps
/// resolving.
pub use crate::catalog::instance::PeerAddr;

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
///
/// [`Self::plan`] takes the WHOLE segment set of one table in one call — a
/// single ring read serving every segment of one query, so the coordinator
/// never sees segment 3 through one ring snapshot and segment 4 through a
/// second, later one (`crate::store::ResultStore::resolve_search_mode` calls
/// this exactly once per resolve, above its per-segment loop). The returned
/// `Vec` has EXACTLY one entry per requested segment, in the same order —
/// callers check the arity and treat a mismatch as a placement fault, never
/// silently zipping a short answer against a long request.
#[async_trait]
pub trait SegmentPlacement: Send + Sync {
    /// The owners of every one of `segments` of `table`, nearest first per
    /// segment; an empty per-segment list means this process serves that
    /// segment locally. `Err` on a catalog read failure — NEVER a silent
    /// per-segment fallback to local, which would exact-scan a multi-node
    /// table's segment behind its owner's back.
    async fn plan(
        &self,
        table: &str,
        segments: &[SegmentId],
    ) -> crate::error::Result<Vec<Vec<PeerAddr>>>;

    /// An OPTIONAL per-process observable this placement wants scraped: a
    /// server that mounts a placement returning `Some` registers it into its
    /// metrics registry (`jammi_placement_ring_empty_total` for
    /// [`RendezvousPlacement`]'s own counter — see
    /// [`RendezvousMetrics::record_ring_empty`]'s doc for what it counts and
    /// why). Every other placement (`AllLocal`, `StaticPlacement`, a
    /// library's own) has nothing to observe and returns `None` — this is a
    /// trait hook, never a downcast, so `ResultStore` (which holds only
    /// `Arc<dyn SegmentPlacement>`) can reach it without knowing the
    /// concrete type.
    fn ring_empty_metrics(&self) -> Option<Arc<RendezvousMetrics>> {
        None
    }
}

/// The default placement: every segment is local. A process with this
/// placement is a single node regardless of what transport it holds.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllLocal;

#[async_trait]
impl SegmentPlacement for AllLocal {
    async fn plan(
        &self,
        _table: &str,
        segments: &[SegmentId],
    ) -> crate::error::Result<Vec<Vec<PeerAddr>>> {
        Ok(vec![Vec::new(); segments.len()])
    }
}

/// An explicit placement table: `(table_name, segment_id)` → owners. The
/// library value for an embedder that knows its topology, and the fixture the
/// end-to-end oracles drive. A pair absent from the map is local.
#[derive(Debug, Default, Clone)]
pub struct StaticPlacement(pub BTreeMap<(String, SegmentId), Vec<PeerAddr>>);

#[async_trait]
impl SegmentPlacement for StaticPlacement {
    async fn plan(
        &self,
        table: &str,
        segments: &[SegmentId],
    ) -> crate::error::Result<Vec<Vec<PeerAddr>>> {
        Ok(segments
            .iter()
            .map(|segment| {
                self.0
                    .get(&(table.to_string(), *segment))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect())
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

/// How many times [`RendezvousPlacement`] fell back to all-local because its
/// ring read came back empty or its own row was absent from it — NEVER
/// silent (RV2): a caller reads this delta to know whether a deployment's
/// membership is actually working the way `[server] placement = "rendezvous"`
/// promised, or is quietly behaving like `AllLocal`.
#[derive(Debug, Default)]
pub struct RendezvousMetrics {
    /// The ring was empty, or did not contain this process's own row, on a
    /// [`RendezvousPlacement::plan`] call — every segment of that call fell
    /// back to local.
    pub ring_empty: AtomicU64,
}

impl RendezvousMetrics {
    /// Count one ring-empty fallback.
    pub fn record_ring_empty(&self) {
        self.ring_empty.fetch_add(1, Ordering::Relaxed);
    }

    /// The current count — `jammi_placement_ring_empty_total` at scrape.
    pub fn ring_empty_total(&self) -> u64 {
        self.ring_empty.load(Ordering::Relaxed)
    }
}

/// The rendezvous (highest-random-weight, HRW) placement over the LIVE
/// `instances` ring: for each segment, every ring member (self included) is
/// scored by [`domain_hash`] under this crate's own placement domain, and the
/// candidates are the members sorted by score DESCENDING with ties broken by
/// `instance_id` byte order — a deterministic total order independent of the
/// ring's SQL row order (a `SELECT` with no `ORDER BY` makes no row-order
/// promise; sorting is this type's own job, not the query's).
///
/// The ring is read FRESH on every [`Self::plan`] call — one single-statement
/// query joining no other table, filtering on liveness
/// (`crate::catalog::lease::instance_liveness_margin` applied to `margin`,
/// which the CALLER resolves once from `[lease] duration_secs` and threads
/// in — this type reads no config), a shared result root (via
/// `crate::catalog::instance::live_with_root_clause`, the SAME fragment
/// [`Catalog::list_gang_members`](crate::catalog::Catalog::list_gang_members)
/// evaluates), and presence of `peer_addr` — self INCLUDED (a self-referencing
/// subquery, never a second, separately-cached copy of this process's own
/// root identity: `self_instance_id` is the only identifying field this type
/// stores). An empty ring, or a ring that does not contain the caller's own
/// row (a construction race, or a stale/pruned self row), yields an all-local
/// plan for every requested segment and counts
/// [`RendezvousMetrics::record_ring_empty`] — never a silent single-node
/// fallback indistinguishable from a healthy one-node ring.
///
/// **A root-identity-mismatched row is excluded, and is NOT separately
/// counted.** The `live_with_root_clause` predicate excludes it in the SAME
/// `WHERE` that excludes a stale row or one with no `peer_addr` — there is
/// no per-reason breakdown at the SQL edge, and adding one (a second
/// `COUNT(*) ... WHERE result_root_identity <> ...` statement, or a
/// `CASE`-tagged row the client tallies) would be a SECOND statement this
/// type's own cost budget (RV6, below) forbids paying on every placed
/// search just to distinguish "excluded for staleness" from "excluded for
/// a foreign root" — a distinction only human debugging, never placement
/// correctness, would use. This is deliberately unlike
/// [`RendezvousMetrics::record_ring_empty`]: THAT counter exists because
/// "the whole ring is empty (or missing self)" changes this type's
/// BEHAVIOUR (every segment falls back to all-local) and would otherwise be
/// silently indistinguishable from a healthy one-node deployment — a
/// root-mismatched row changes nothing behavioural; the ring is simply
/// smaller by one candidate, the same as if that row had never advertised
/// at all. The exclusion itself is proven, not asserted:
/// `rendezvous_ring::a_root_identity_mismatched_row_is_excluded` (a member
/// rooted elsewhere never appears in the ring) is this type's own executed
/// oracle, and `gang_membership`'s whole root-mismatch suite
/// (`file_and_s3_rooted_members_are_not_gang_members_of_each_other`,
/// `a_row_with_a_root_but_no_identity_is_never_a_member`, and siblings) is
/// the SAME `live_with_root_clause` fragment's oracle from
/// `Catalog::list_gang_members`'s side — one shared predicate, proven
/// excluding on both callers.
///
/// **Cost (measured, RENDEZVOUS RV6).** One ring read per placed search —
/// `crate::store::ResultStore::resolve_search_mode` calls [`Self::plan`]
/// exactly once above its per-segment loop (RV1) — ONE statement, no
/// transaction wrapper at all:
/// `crate::catalog::backend::BackendImpl::query_untransacted` runs it
/// directly against the pool, never through
/// `crate::catalog::backend::BackendImpl::transaction`, which on Postgres
/// pays for `BEGIN` + two `SET TRANSACTION ...` statements + `COMMIT`
/// around the caller's own query regardless of `read_only` — four extra
/// round trips this read never needed to pay.
///
/// Measured on the scratch Postgres host, 5 consecutive runs each on a
/// fresh database with a warmed connection (isolating the query's own cost
/// from a process's first-connection TCP+auth handshake, which is itself
/// 3-4 ms and unrelated to either form): **~3.9-5.0 ms at 101 `instances`
/// rows (51 ring candidates)** — essentially UNCHANGED from the ~4.6 ms
/// single-sample measurement WITH the transaction wrapper; at this row
/// count neither form's cost is the wrapper's four extra statements (each
/// well under 1 ms on localhost) — it is the unavoidable one-round-trip
/// floor of issuing any statement at all. **~10.2-10.9 ms at 10,101 rows
/// (5,051 candidates)** — a real but modest ~20-25 % reduction from the
/// ~13.3 ms single-sample measurement WITH the wrapper (removing 4 round
/// trips saves a few ms here too), but the wrapper was never the DOMINANT
/// cost at this scale either: `EXPLAIN (ANALYZE, BUFFERS)` on the exact
/// rendered query, reproduced with the same 50 % root-matching fixture,
/// shows
///
/// ```text
/// Seq Scan on instances  (cost=8.30..410.83 rows=5050 width=25)
///                        (actual time=0.018..3.149 rows=5051 loops=1)
///   Filter: (peer_addr IS NOT NULL) AND (result_root_identity = $0)
///           AND (last_seen_at >= <canonical cutoff text>)
///   Rows Removed by Filter: 5050
///   InitPlan 1 (returns $0)
///     ->  Index Scan using instances_pkey on instances (cost=0.29..8.30)
/// Execution Time: 3.262 ms
/// ```
///
/// `idx_instances_seen` (the index on `last_seen_at` alone) is NOT used:
/// with 50 % of the table sharing this fixture's root identity, a Seq Scan
/// of the whole (10,101-row) table is genuinely cheaper for Postgres's own
/// planner than randomly heap-fetching half the rows through an index that
/// supports only the OTHER conjunct — `idx_instances_seen` has no entry for
/// `result_root_identity` at all, and the sargable rewrite (RV3) only
/// changed whether the `last_seen_at` conjunct COULD use an index, not
/// whether the planner's cost model prefers one here. The scan+filter
/// itself is 3.262 ms of Postgres's own `EXPLAIN ANALYZE` time; the
/// remaining ~7 ms of the measured ~10.5 ms median wall-clock figure is the
/// wire transfer and per-row decode of the 5,051-row RESULT SET back to the
/// client — proportional to how many candidates the ring actually returns,
/// not to the query mechanism, and not reducible by removing wrappers or
/// adding indexes (an index on `result_root_identity` could turn this Seq
/// Scan into an Index Scan, but that is a schema/migration change this unit
/// does not make — RV4's own K5 note is that it appends no migration).
///
/// That 5,051-candidate shape is a deliberately adversarial STRESS fixture
/// (half of 10,100 synthetic rows sharing one root), never a realistic
/// ring: a real ring is bounded by one deployment's own live replica count
/// (single- to low-double-digit in practice), which the 101-row/51-candidate
/// measurement (~4-5 ms, dominated by the one-round-trip floor, not by row
/// count) already over-covers. The stated 20 ms budget is therefore not
/// "an order of magnitude of headroom" over the measured number — at 10k
/// adversarial rows it is under 2x — but it is a deliberate bound on a
/// fleet that has accumulated thousands of long-dead, unpruned `instances`
/// rows still sharing one root (a real but pathological shape
/// `Catalog::prune_instances` exists to prevent), not on the realistic
/// operating point this ring is sized for. Exceeding the budget at a
/// REALISTIC ring size is a regression; exceeding it only in the
/// adversarial 10k-row shape means `Catalog::prune_instances` has not run —
/// an operational fact, not this predicate's own cost.
#[derive(Clone)]
pub struct RendezvousPlacement {
    catalog: Arc<Catalog>,
    self_instance_id: String,
    margin: Duration,
    metrics: Arc<RendezvousMetrics>,
}

impl std::fmt::Debug for RendezvousPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RendezvousPlacement")
            .field("self_instance_id", &self.self_instance_id)
            .field("margin", &self.margin)
            .finish_non_exhaustive()
    }
}

/// The domain tag every RENDEZVOUS placement hash folds under — distinct
/// from [`crate::store::content_hash::CONTENT_HASH_DOMAIN`] and any other
/// domain this crate ever hashes under (see [`domain_hash`]'s docs).
pub const PLACEMENT_HASH_DOMAIN: &[u8] = b"jammi.placement.v1";

/// Length-prefix `bytes` (an 8-byte little-endian length, then the payload)
/// — the one self-delimiting shape [`domain_hash`]'s callers in this module
/// use, so `(instance_id, table, segment_id)` can never collide across a
/// shifted field boundary.
fn framed(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// The HRW score of `instance_id` for `(table, segment)`: the big-endian
/// `u64` of the first 8 bytes of `domain_hash(PLACEMENT_HASH_DOMAIN,
/// [instance_id, table, segment_id])`. Exported (`pub(crate)`) so the RV5
/// minimal-disruption oracle can compute an INDEPENDENT expected ranking
/// without duplicating this module's own sort.
pub(crate) fn rendezvous_score(instance_id: &str, table: &str, segment: SegmentId) -> u64 {
    let instance_part = framed(instance_id.as_bytes());
    let table_part = framed(table.as_bytes());
    let segment_part = framed(&segment.0.to_be_bytes());
    let hash = domain_hash(
        PLACEMENT_HASH_DOMAIN,
        &[&instance_part, &table_part, &segment_part],
    );
    u64::from_be_bytes(hash[0..8].try_into().expect("8 bytes"))
}

impl RendezvousPlacement {
    /// `margin` is the LIVENESS margin already
    /// (`crate::catalog::lease::instance_liveness_margin` applied — see this
    /// type's docs) — this constructor performs no
    /// further multiplication, so the ONE call site (`crate::store`'s
    /// session-wiring caller) is unambiguous about what it passed.
    pub fn new(
        catalog: Arc<Catalog>,
        self_instance_id: impl Into<String>,
        margin: Duration,
    ) -> Self {
        Self {
            catalog,
            self_instance_id: self_instance_id.into(),
            margin,
            metrics: Arc::new(RendezvousMetrics::default()),
        }
    }

    /// This placement's ring-empty-fallback counter
    /// (`jammi_placement_ring_empty_total`).
    pub fn metrics(&self) -> Arc<RendezvousMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Rank `ring` (every live member sharing this process's root, INCLUDING
    /// this process) for `(table, segment)`: sort by [`rendezvous_score`]
    /// descending, ties broken by `instance_id` byte order — a pure function
    /// of the ring's CONTENT, never its encounter order (the RV1/RV5 oracle:
    /// shuffling `ring` before calling this must not change the output).
    fn rank<'a>(
        table: &str,
        segment: SegmentId,
        ring: &'a [crate::catalog::instance::RingMember],
    ) -> Vec<&'a crate::catalog::instance::RingMember> {
        let mut scored: Vec<(u64, &crate::catalog::instance::RingMember)> = ring
            .iter()
            .map(|m| (rendezvous_score(&m.instance_id, table, segment), m))
            .collect();
        scored.sort_by(|(score_a, a), (score_b, b)| {
            score_b
                .cmp(score_a)
                .then_with(|| a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()))
        });
        scored.into_iter().map(|(_, m)| m).collect()
    }
}

#[async_trait]
impl SegmentPlacement for RendezvousPlacement {
    async fn plan(
        &self,
        table: &str,
        segments: &[SegmentId],
    ) -> crate::error::Result<Vec<Vec<PeerAddr>>> {
        let ring = self
            .catalog
            .list_ring_members(&self.self_instance_id, self.margin)
            .await?;
        let self_in_ring = ring.iter().any(|m| m.instance_id == self.self_instance_id);
        if ring.is_empty() || !self_in_ring {
            self.metrics.record_ring_empty();
            return Ok(vec![Vec::new(); segments.len()]);
        }
        Ok(segments
            .iter()
            .map(|segment| {
                let ranked = Self::rank(table, *segment, &ring);
                // `ranked[0]` always exists: `self_in_ring` guarantees at
                // least one member.
                if ranked[0].instance_id == self.self_instance_id {
                    Vec::new()
                } else {
                    let mut owners = vec![ranked[0].peer_addr.clone()];
                    if let Some(second) = ranked.get(1) {
                        owners.push(second.peer_addr.clone());
                    }
                    owners
                }
            })
            .collect())
    }

    fn ring_empty_metrics(&self) -> Option<Arc<RendezvousMetrics>> {
        Some(self.metrics())
    }
}

#[cfg(test)]
mod rendezvous_tests {
    use super::*;
    use crate::catalog::instance::RingMember;

    fn member(id: &str) -> RingMember {
        RingMember {
            instance_id: id.to_string(),
            peer_addr: PeerAddr::parse(&format!("10.0.0.{}:9000", id.len() + 1)).unwrap(),
        }
    }

    /// RV1/RV5: the rank is a PURE function of the ring's CONTENT, computed
    /// by an INDEPENDENT second implementation here (this test does its OWN
    /// sort, never calling `RendezvousPlacement::rank`'s sort at all) —
    /// reordering the ring (a SQL `SELECT` with no `ORDER BY` makes no
    /// row-order promise) must not change the result. This test calls
    /// `rendezvous_score` itself for each member's score, so it is
    /// independent of `rank`'s SORT/tie-break, never of `rendezvous_score`'s
    /// own byte derivation — THAT independence (an oracle that never calls
    /// `rendezvous_score` at all) is
    /// `rendezvous_score_matches_a_hand_rolled_sha256_fold`, below.
    #[test]
    fn rank_is_independent_of_ring_encounter_order() {
        let ring: Vec<RingMember> = ["alpha", "bravo", "charlie", "delta", "echo"]
            .iter()
            .map(|s| member(s))
            .collect();

        // The independent oracle: score each member directly (never through
        // `RendezvousPlacement::rank`) and sort by the SAME rule stated in
        // `rank`'s own doc — descending score, ties broken by instance_id
        // byte order.
        let mut expected: Vec<(u64, &str)> = ring
            .iter()
            .map(|m| {
                (
                    rendezvous_score(&m.instance_id, "t", SegmentId(42)),
                    m.instance_id.as_str(),
                )
            })
            .collect();
        expected
            .sort_by(|(sa, a), (sb, b)| sb.cmp(sa).then_with(|| a.as_bytes().cmp(b.as_bytes())));
        let expected_order: Vec<&str> = expected.into_iter().map(|(_, id)| id).collect();

        for perm_seed in 0..5u64 {
            let mut shuffled = ring.clone();
            // A cheap deterministic shuffle (no extra dev-dependency): rotate
            // and reverse by a seed-derived amount.
            let rotate_by = (perm_seed as usize) % shuffled.len().max(1);
            shuffled.rotate_left(rotate_by);
            if perm_seed % 2 == 0 {
                shuffled.reverse();
            }
            let ranked = RendezvousPlacement::rank("t", SegmentId(42), &shuffled);
            let order: Vec<&str> = ranked.iter().map(|m| m.instance_id.as_str()).collect();
            assert_eq!(
                order, expected_order,
                "seed {perm_seed}: rank must not depend on the ring's encounter order"
            );
        }
    }

    /// RV5: `rendezvous_score`'s exact byte derivation, pinned against a
    /// HAND-ROLLED SHA-256 fold that calls neither `domain_hash` nor
    /// `framed` nor `rendezvous_score` itself — every other oracle in this
    /// module (the rank-order-independence test above, the minimal-
    /// disruption test below) calls `rendezvous_score` to build its own
    /// "independent" expectation, which makes them independent of `rank`'s
    /// SORT but blind to a bug inside `rendezvous_score` itself (a byte
    /// window shifted, an endianness flipped): every downstream computation
    /// would still agree with itself and every existing assertion would
    /// stay green. This test is the one oracle that reimplements the raw
    /// SHA-256 fold from scratch and pins the EXACT `u64` each of 3 fixed
    /// inputs produces.
    #[test]
    fn rendezvous_score_matches_a_hand_rolled_sha256_fold() {
        use sha2::{Digest, Sha256};

        // A from-scratch reimplementation of `PLACEMENT_HASH_DOMAIN ++
        // framed(instance_id) ++ framed(table) ++ framed(segment_be_bytes)`,
        // "big-endian u64 of hash[0..8]" — written independently against
        // this module's own doc comments, never by calling `domain_hash`,
        // `framed`, or `rendezvous_score`.
        fn hand_rolled_score(instance_id: &str, table: &str, segment: i64) -> u64 {
            let mut hasher = Sha256::new();
            hasher.update(b"jammi.placement.v1");
            hasher.update((instance_id.len() as u64).to_le_bytes());
            hasher.update(instance_id.as_bytes());
            hasher.update((table.len() as u64).to_le_bytes());
            hasher.update(table.as_bytes());
            let segment_bytes = segment.to_be_bytes();
            hasher.update((segment_bytes.len() as u64).to_le_bytes());
            hasher.update(segment_bytes);
            let digest = hasher.finalize();
            u64::from_be_bytes(digest[0..8].try_into().expect("32-byte digest"))
        }

        let cases: [(&str, &str, i64); 3] = [
            ("node-a", "table-1", 0),
            ("node-b", "table-1", 0),
            ("node-a", "table-2", 7),
        ];
        for (instance_id, table, segment) in cases {
            let hand_rolled = hand_rolled_score(instance_id, table, segment);
            let actual = rendezvous_score(instance_id, table, SegmentId(segment));
            assert_eq!(
                actual, hand_rolled,
                "rendezvous_score({instance_id:?}, {table:?}, {segment}) diverged from the \
                 independent hand-rolled fold"
            );
        }

        // Pinned GOLDEN values — computed once by this same hand-rolled
        // fold (verified above to already equal `rendezvous_score`'s own
        // output) and hardcoded here — so a mutation that happened to shift
        // BOTH `hand_rolled_score` and `rendezvous_score` identically
        // (impossible in practice, since the two are separate functions
        // written independently, but this is the belt to the assertion
        // above's suspenders) still reds.
        assert_eq!(
            hand_rolled_score("node-a", "table-1", 0),
            0xca049e14dd4ff86b,
            "golden value for (\"node-a\", \"table-1\", 0) changed"
        );
        assert_eq!(
            hand_rolled_score("node-b", "table-1", 0),
            0x7656f394ba7ce0ee,
            "golden value for (\"node-b\", \"table-1\", 0) changed"
        );
        assert_eq!(
            hand_rolled_score("node-a", "table-2", 7),
            0xeefb7b87e2ba7b78,
            "golden value for (\"node-a\", \"table-2\", 7) changed"
        );

        // The rank order these 3 golden scores imply (descending; no ties
        // among them) — an executed check that the golden values themselves
        // are a meaningful ranking oracle, not just opaque numbers.
        let mut by_score = cases
            .iter()
            .map(|&(id, table, seg)| (hand_rolled_score(id, table, seg), id))
            .collect::<Vec<_>>();
        by_score.sort_by(|a, b| b.0.cmp(&a.0));
        assert_eq!(
            by_score.into_iter().map(|(_, id)| id).collect::<Vec<_>>(),
            vec!["node-a", "node-a", "node-b"],
            "these 3 golden values, sorted descending, must name this exact order"
        );
    }

    /// RV5 minimal-disruption: over 100 segments, removing one member of a
    /// 3-member ring moves roughly `1/N` of the segments, and EVERY moved
    /// segment was previously owned by the LEAVER (never a segment that
    /// stays with an unrelated owner); adding a 4th member moves roughly
    /// `1/N` of the segments and EVERY moved segment goes to the NEWCOMER.
    #[test]
    fn minimal_disruption_on_membership_change() {
        let a = member("node-a");
        let b = member("node-b");
        let c = member("node-c");
        let d = member("node-d");
        let table = "minimal-disruption-table";
        let segments: Vec<SegmentId> = (0..100).map(SegmentId).collect();

        let owner_under = |ring: &[RingMember], seg: SegmentId| -> String {
            RendezvousPlacement::rank(table, seg, ring)[0]
                .instance_id
                .clone()
        };

        let ring3 = [a.clone(), b.clone(), c.clone()];
        let ring2 = [a.clone(), b.clone()]; // c leaves
        let ring4 = [a.clone(), b.clone(), c.clone(), d.clone()]; // d joins

        let owners3: Vec<String> = segments.iter().map(|&s| owner_under(&ring3, s)).collect();
        let owners2: Vec<String> = segments.iter().map(|&s| owner_under(&ring2, s)).collect();
        let owners4: Vec<String> = segments.iter().map(|&s| owner_under(&ring4, s)).collect();

        let moved_on_leave = owners3
            .iter()
            .zip(&owners2)
            .filter(|(before, after)| before != after)
            .count();
        assert!(
            (10..=60).contains(&moved_on_leave),
            "N=3->2 should move roughly 1/N of 100 segments, got {moved_on_leave}"
        );
        for (before, after) in owners3.iter().zip(&owners2) {
            if before != after {
                assert_eq!(
                    before, "node-c",
                    "every segment that moved when C left must have been owned by C"
                );
            }
        }

        let moved_on_join = owners3
            .iter()
            .zip(&owners4)
            .filter(|(before, after)| before != after)
            .count();
        assert!(
            (10..=60).contains(&moved_on_join),
            "N=3->4 should move roughly 1/N of 100 segments, got {moved_on_join}"
        );
        for (before, after) in owners3.iter().zip(&owners4) {
            if before != after {
                assert_eq!(
                    after, "node-d",
                    "every segment that moved when D joined must now be owned by D"
                );
            }
        }
    }

    /// RV5: `domain_hash` under `PLACEMENT_HASH_DOMAIN` never collides with
    /// `CONTENT_HASH_DOMAIN` over the same raw bytes, and the placement score
    /// is sensitive to every one of its three determinants (instance, table,
    /// segment) — changing any ONE moves the score.
    #[test]
    fn placement_score_is_sensitive_to_every_determinant() {
        let base = rendezvous_score("inst-1", "table-a", SegmentId(1));
        assert_ne!(base, rendezvous_score("inst-2", "table-a", SegmentId(1)));
        assert_ne!(base, rendezvous_score("inst-1", "table-b", SegmentId(1)));
        assert_ne!(base, rendezvous_score("inst-1", "table-a", SegmentId(2)));
        // A boundary-shift pair: "inst-1"+"table-a" vs "inst-" + "1table-a"
        // must not collide (the length-prefixing in `framed`).
        assert_ne!(
            rendezvous_score("inst-1", "table-a", SegmentId(1)),
            rendezvous_score("inst-", "1table-a", SegmentId(1))
        );
    }

    #[test]
    fn ring_empty_metric_starts_at_zero_and_counts() {
        let metrics = RendezvousMetrics::default();
        assert_eq!(metrics.ring_empty_total(), 0);
        metrics.record_ring_empty();
        metrics.record_ring_empty();
        assert_eq!(metrics.ring_empty_total(), 2);
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
        let owner = PeerAddr::parse("127.0.0.1:1").unwrap();
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
        assert_eq!(
            AllLocal
                .plan("t", &[SegmentId(0), SegmentId(1)])
                .await
                .unwrap(),
            vec![Vec::<PeerAddr>::new(), Vec::new()],
            "one empty entry per requested segment, in order"
        );
        let placed = StaticPlacement(BTreeMap::from([(
            ("t".to_string(), SegmentId(1)),
            vec![owner.clone()],
        )]));
        assert_eq!(
            placed
                .plan("t", &[SegmentId(1), SegmentId(0)])
                .await
                .unwrap(),
            vec![vec![owner], Vec::new()],
            "arity preserved and order preserved even when only one segment has an owner"
        );
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
