//! `PeerService` — the segment OWNER side of the distributed data plane.
//!
//! Served only on the internal `[server] peer_bind` listener (see
//! [`crate::runtime::OssServer::bind`]) — never on the public gRPC + Flight SQL
//! listener, never wrapped by the tenant-binding layer, never advertised by
//! `GetServerInfo`. A replica is an owner iff `peer_bind` is set.
//!
//! The owner is deliberately tenant-free (invariant I-PEER): the request
//! carries no tenant, and this handler reads no `result_tables` row. Tenant
//! scope was enforced by the COORDINATOR, which resolved the table through its
//! own tenant-scoped catalog read before fanning out. What the owner enforces,
//! at its input edge, splits on WHOSE fault a refusal is — never on how it
//! looks at a glance:
//!
//! - Genuine REQUEST malformation is `INVALID_ARGUMENT`, TERMINAL at the
//!   coordinator (never a ladder rung): an empty or duplicated segment id in
//!   the requested set, a duplicated `ExactRescore` row id, a query component
//!   that is not finite, a `width` that does not fit `usize`, and a
//!   precision/phase enum whose raw value is `0` (`UNSPECIFIED` — explicitly
//!   not set).
//! - A disagreement about the OWNER's OWN DATA — never the coordinator's
//!   fault — is `FAILED_PRECONDITION`, which LADDERS (a retry, then a local
//!   load): a requested segment id absent from this owner's own segment list
//!   (its `list_index_segments` read can race a concurrent
//!   `purge_segments`/append); the bundle's stamped precision not matching
//!   the requested one (the segment cache's strict load); the query's width
//!   not matching a loaded segment's own width (see "The wire query" below);
//!   an `ExactRescore` row id this owner's segment does not index
//!   (this owner reloads its segment per RPC, so a rebuild between phases can
//!   move ids out from under it); and a precision/phase enum raw value this
//!   build's generated `enum` has no variant for but is non-zero (a value a
//!   NEWER coordinator knows and this owner does not — rolling-upgrade
//!   version skew, not malformation).
//!
//! **The wire query.** The vector a request carries arrives as a
//! [`jammi_db::index::FiniteQuery`]: finiteness is checked at the edge, with
//! nothing but the request consulted, so a non-finite component is the
//! request's own malformation. Its WIDTH has no catalog authority here — an
//! owner reads no `result_tables` row (I-PEER) — so the only width an owner
//! holds is its own segments', and the query becomes a
//! [`jammi_db::index::ValidatedQuery`] against the first requested segment it
//! loads (`admit_query`). The coordinator validated the query against the
//! table's authority (its catalog row, or its own resident segment) before
//! any fan-out, so a disagreement found HERE — with that first segment or
//! any later one — is this owner's segment drifting from that authority:
//! own-data, `FAILED_PRECONDITION`, never the caller's fault.
//!
//! It then runs the same pure kernels a single node runs
//! ([`jammi_db::index::segment::search_unit`] / [`rescore`]) per segment and
//! returns `(row_id, distance)` units — ids and distances, never vectors. A
//! torn bundle (a candidate with no exact vector) is `DATA_LOSS`.
//!
//! Segments are loaded per RPC exactly as a coordinator loads per query —
//! through the content-addressed segment cache (a `file://` bundle loads in
//! place; a remote bundle is fetched once per process). An owner-side resident
//! segment set across RPCs is out of scope.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::catalog::segment_repo::IndexSegment;
use jammi_db::config::StoragePrecision;
use jammi_db::error::JammiError;
use jammi_db::index::segment::{rescore, search_unit};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::{
    FiniteQuery, QuerySource, QueryValidationError, SegmentId, SegmentSearchPhase, ValidatedQuery,
};
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_wire::peer::{phase_from_proto, precision_from_proto, ProtoEnumDecode};
use tonic::{Request, Response, Status};

use crate::grpc::proto::peer::peer_service_server::PeerService;
use crate::grpc::proto::peer::{
    ExactRescoreRequest, ExactRescoreResponse, Hit, SegmentSearchRequest, SegmentSearchResponse,
    SegmentUnit,
};
use crate::grpc::wire::map_engine_error;

/// Server-side handler for the peer segment-search surface. Holds the shared
/// engine session for its result store (segment cache + catalog + ANN knobs).
pub struct PeerServer {
    session: Arc<InferenceSession>,
}

impl PeerServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// The catalog's segment list for `table_name` — the set every requested
    /// id must be a member of. No tenant filter: see the module docs (I-PEER).
    async fn segments_of(
        &self,
        store: &ResultStore,
        table_name: &str,
    ) -> Result<Vec<IndexSegment>, Status> {
        store
            .catalog()
            .list_index_segments(table_name)
            .await
            .map_err(map_engine_error)
    }

    /// Load segment `id` of `table_name` through the segment cache at the
    /// requested precision. The strict manifest check refuses a bundle stamped
    /// at another precision (`FAILED_PRECONDITION`); any other load failure is
    /// the engine error mapped as usual.
    async fn load(
        store: &ResultStore,
        table_name: &str,
        segment: &IndexSegment,
        precision: StoragePrecision,
    ) -> Result<SidecarIndex, Status> {
        let url = StorageUrl::parse(&segment.index_path)
            .map_err(|e| map_engine_error(JammiError::from(e)))?;
        store
            .segment_cache()
            .load_segment(&url, store.ann_config(), precision)
            .await
            .map_err(|e| match e {
                JammiError::IncompatibleFormat { .. } => Status::failed_precondition(format!(
                    "segment {}/{}: {e}",
                    table_name, segment.segment_id
                )),
                other => map_engine_error(other),
            })
    }

    /// Run `each` over every requested segment in request order, loading
    /// each through [`Self::load`] and holding one at a time. The wire query
    /// is admitted against the FIRST segment loaded (`admit_query`) and
    /// width-checked against every one before `each` sees it, so a width
    /// disagreement is own-data and never reaches a kernel.
    async fn over_segments<P, T>(
        store: &ResultStore,
        table_name: &str,
        segments: &[IndexSegment],
        precision: StoragePrecision,
        query: FiniteQuery,
        requested: Vec<(i64, P)>,
        each: impl Fn(i64, P, &SidecarIndex, &ValidatedQuery) -> Result<T, Status>,
    ) -> Result<Vec<T>, Status> {
        let load = |id: i64| async move {
            Self::load(
                store,
                table_name,
                segment(table_name, segments, id)?,
                precision,
            )
            .await
        };
        let checked = |id: i64, payload: P, index: &SidecarIndex, query: &ValidatedQuery| {
            verify_query_width(table_name, id, query, index)?;
            each(id, payload, index, query)
        };
        let mut requested = requested.into_iter();
        let Some((first_id, first_payload)) = requested.next() else {
            return Err(none_named(table_name));
        };
        let first = load(first_id).await?;
        let query = admit_query(table_name, first_id, query, &first)?;
        let mut out = Vec::with_capacity(requested.len() + 1);
        out.push(checked(first_id, first_payload, &first, &query)?);
        drop(first);
        for (id, payload) in requested {
            out.push(checked(id, payload, &load(id).await?, &query)?);
        }
        Ok(out)
    }
}

/// A request that names no segment — request malformation,
/// `INVALID_ARGUMENT`.
fn none_named(table_name: &str) -> Status {
    Status::invalid_argument(format!("no segment of table '{table_name}' was named"))
}

/// Every requested id must be named once and at least one must be named
/// (request malformation, `INVALID_ARGUMENT`); each named id must ALSO be in
/// this owner's own segment list — own-data, `FAILED_PRECONDITION`, since the
/// owner's `list_index_segments` read can race a concurrent
/// `purge_segments`/append and disagree with the coordinator's. The first
/// violation refuses the WHOLE request (a unit-less or partial answer would
/// be a silent shrink at the coordinator).
fn verify_membership(
    table_name: &str,
    requested: &[i64],
    segments: &[IndexSegment],
) -> Result<(), Status> {
    if requested.is_empty() {
        return Err(none_named(table_name));
    }
    let mut seen = std::collections::BTreeSet::new();
    for id in requested {
        if !seen.insert(*id) {
            return Err(Status::invalid_argument(format!(
                "segment {id} of table '{table_name}' is named more than once"
            )));
        }
        if !segments.iter().any(|s| s.segment_id == *id) {
            return Err(Status::failed_precondition(format!(
                "segment {id} is not a segment of table '{table_name}'"
            )));
        }
    }
    Ok(())
}

/// The request's vector as a [`FiniteQuery`] — the owner's edge. A
/// non-finite component is `INVALID_ARGUMENT` (a coordinator that sent it
/// missed its own check), never a torn bundle.
fn finite_query(values: Vec<f32>) -> Result<FiniteQuery, Status> {
    FiniteQuery::new(values, QuerySource::Caller).map_err(|e| map_engine_error(e.into()))
}

/// A width disagreement between the wire query and one of this owner's
/// segments — own-data, `FAILED_PRECONDITION` (see the module docs' "The
/// wire query"), whichever check found it.
fn width_drift(table_name: &str, segment_id: i64, e: QueryValidationError) -> Status {
    match e {
        QueryValidationError::Width {
            expected, actual, ..
        }
        | QueryValidationError::ArtifactMismatch {
            expected, actual, ..
        } => Status::failed_precondition(format!(
            "query has width {actual} but segment {table_name}/{segment_id} has width {expected}"
        )),
        non_finite @ QueryValidationError::NonFinite { .. } => {
            map_engine_error(JammiError::from(non_finite))
        }
    }
}

/// The wire query's one transition at this owner: checked against the first
/// requested segment it loaded — the only width an owner holds.
fn admit_query(
    table_name: &str,
    segment_id: i64,
    query: FiniteQuery,
    first: &SidecarIndex,
) -> Result<ValidatedQuery, Status> {
    query
        .against_authority(first.dimensions())
        .map_err(|e| width_drift(table_name, segment_id, e))
}

/// The query must be exactly as wide as the segment it is searched against:
/// a longer query would index past the stored vector in `cosine_distance`
/// (a panic), a shorter one would be silently scored over a prefix. Checked
/// BEFORE any kernel runs, against the loaded segment's own width, so a
/// mismatch is never mistaken for a torn bundle.
fn verify_query_width(
    table_name: &str,
    segment_id: i64,
    query: &ValidatedQuery,
    index: &SidecarIndex,
) -> Result<(), Status> {
    query
        .require_width(index.dimensions(), format!("segment {segment_id}"))
        .map_err(|e| width_drift(table_name, segment_id, e))
}

/// Every row id an `ExactRescore` names must be named once (request
/// malformation, `INVALID_ARGUMENT`) and indexed by the segment it is named
/// under — an id this owner's segment does not index is own-data,
/// `FAILED_PRECONDITION`, since this owner reloads its segment per RPC and a
/// rebuild between phases can move ids out from under it. Never mistaken for
/// a torn bundle by the rescore kernel (whose `Ok(None)` is reserved for a
/// row the graph holds but the companion lost).
fn verify_row_ids(
    table_name: &str,
    segment_id: i64,
    row_ids: &[String],
    index: &SidecarIndex,
) -> Result<(), Status> {
    let mut seen = std::collections::BTreeSet::new();
    for row_id in row_ids {
        if !seen.insert(row_id.as_str()) {
            return Err(Status::invalid_argument(format!(
                "row '{row_id}' of segment {table_name}/{segment_id} is named more than once"
            )));
        }
        if !index.contains_row(row_id) {
            return Err(Status::failed_precondition(format!(
                "row '{row_id}' is not indexed by segment {table_name}/{segment_id}"
            )));
        }
    }
    Ok(())
}

/// The catalog row for a requested id. Membership was verified above, but
/// the id is a wire value re-looked-up against this owner's own segment
/// list, so a miss is own-data — `FAILED_PRECONDITION`, the same class as
/// [`verify_membership`]'s — never a panic.
fn segment<'a>(
    table_name: &str,
    segments: &'a [IndexSegment],
    id: i64,
) -> Result<&'a IndexSegment, Status> {
    segments.iter().find(|s| s.segment_id == id).ok_or_else(|| {
        Status::failed_precondition(format!(
            "segment {id} is not a segment of table '{table_name}'"
        ))
    })
}

/// A kernel failure after a successful load AND after the input edge above
/// accepted the request — a row the graph holds but the companion lost, a
/// companion read fault, a graph search fault — is a torn bundle at this
/// owner. Caller faults never reach here: width, membership and row ids are
/// refused first.
fn torn(table_name: &str, segment_id: i64, e: JammiError) -> Status {
    Status::data_loss(format!("segment {table_name}/{segment_id}: {e}"))
}

fn hits(pairs: Vec<(String, f32)>) -> Vec<Hit> {
    pairs
        .into_iter()
        .map(|(row_id, distance)| Hit { row_id, distance })
        .collect()
}

/// Decode `raw`'s storage precision, split on WHOSE fault an unrecognised
/// value is: `0` (`UNSPECIFIED`) is the wire's explicit "not set" — request
/// malformation, `INVALID_ARGUMENT`; a non-zero value this owner's generated
/// `enum` has no variant for is a value a NEWER coordinator knows and this
/// (older) owner does not — rolling-upgrade version skew, own-data,
/// `FAILED_PRECONDITION` (ladders).
fn decode_precision(raw: i32) -> Result<StoragePrecision, Status> {
    match precision_from_proto(raw) {
        ProtoEnumDecode::Known(precision) => Ok(precision),
        ProtoEnumDecode::Unspecified => {
            Err(Status::invalid_argument("storage_precision is unspecified"))
        }
        ProtoEnumDecode::Unknown => Err(Status::failed_precondition(format!(
            "storage_precision {raw} is not a value this owner's build recognises (likely a \
             newer coordinator's value during a rolling upgrade)"
        ))),
    }
}

/// Decode `raw`'s search phase, split the same way as [`decode_precision`].
fn decode_phase(raw: i32) -> Result<SegmentSearchPhase, Status> {
    match phase_from_proto(raw) {
        ProtoEnumDecode::Known(phase) => Ok(phase),
        ProtoEnumDecode::Unspecified => Err(Status::invalid_argument("phase is unspecified")),
        ProtoEnumDecode::Unknown => Err(Status::failed_precondition(format!(
            "phase {raw} is not a value this owner's build recognises (likely a newer \
             coordinator's value during a rolling upgrade)"
        ))),
    }
}

#[tonic::async_trait]
impl PeerService for PeerServer {
    #[tracing::instrument(skip(self, request))]
    async fn segment_search(
        &self,
        request: Request<SegmentSearchRequest>,
    ) -> Result<Response<SegmentSearchResponse>, Status> {
        let req = request.into_inner();
        let precision = decode_precision(req.storage_precision)?;
        let phase = decode_phase(req.phase)?;
        let width = usize::try_from(req.width)
            .map_err(|_| Status::invalid_argument("width does not fit usize"))?;
        let finite = finite_query(req.query)?;
        let store = self.session.result_store();
        let segments = self.segments_of(&store, &req.table_name).await?;
        verify_membership(&req.table_name, &req.segment_ids, &segments)?;
        let table_name = req.table_name.as_str();
        let requested = req.segment_ids.iter().map(|id| (*id, ())).collect();
        let units = Self::over_segments(
            &store,
            table_name,
            &segments,
            precision,
            finite,
            requested,
            |id, (), index, query| {
                search_unit(SegmentId(id), index, query, width, phase, &|row_id| {
                    index.get_exact(row_id)
                })
                .map(|unit| SegmentUnit {
                    segment_id: id,
                    hits: hits(unit),
                })
                .map_err(|e| torn(table_name, id, e))
            },
        )
        .await?;
        Ok(Response::new(SegmentSearchResponse { units }))
    }

    #[tracing::instrument(skip(self, request))]
    async fn exact_rescore(
        &self,
        request: Request<ExactRescoreRequest>,
    ) -> Result<Response<ExactRescoreResponse>, Status> {
        let req = request.into_inner();
        let precision = decode_precision(req.storage_precision)?;
        let finite = finite_query(req.query)?;
        let store = self.session.result_store();
        let segments = self.segments_of(&store, &req.table_name).await?;
        let requested: Vec<i64> = req
            .row_ids_by_segment
            .iter()
            .map(|g| g.segment_id)
            .collect();
        verify_membership(&req.table_name, &requested, &segments)?;
        let table_name = req.table_name.as_str();
        let requested = req
            .row_ids_by_segment
            .into_iter()
            .map(|g| (g.segment_id, g.row_ids))
            .collect();
        let out: Vec<(String, f32)> = Self::over_segments(
            &store,
            table_name,
            &segments,
            precision,
            finite,
            requested,
            |id, row_ids: Vec<String>, index, query| {
                verify_row_ids(table_name, id, &row_ids, index)?;
                let candidates = row_ids.into_iter().map(|r| (r, 0.0f32)).collect();
                rescore(
                    SegmentId(id),
                    candidates,
                    &|row_id| index.get_exact(row_id),
                    query,
                )
                .map_err(|e| torn(table_name, id, e))
            },
        )
        .await?
        .into_iter()
        .flatten()
        .collect();
        Ok(Response::new(ExactRescoreResponse { hits: hits(out) }))
    }
}
