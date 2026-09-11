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
//! at its input edge, is the one thing the coordinator cannot: (1) every
//! requested segment id belongs to the named table (else `INVALID_ARGUMENT`
//! naming the first unknown id — the whole request is refused, never a
//! partial unit) and (2) the bundle's stamped precision matches the requested
//! one (the segment cache's strict load refuses otherwise →
//! `FAILED_PRECONDITION`). It then runs the same pure kernels a single node
//! runs ([`jammi_db::index::segment::search_unit`] / [`rescore`]) per segment
//! and returns `(row_id, distance)` units — ids and distances, never vectors.
//! A torn bundle (a candidate with no exact vector) is `DATA_LOSS`.
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
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_wire::peer::{phase_from_proto, precision_from_proto};
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
}

/// Every requested id must be in the table's segment list; the first unknown
/// id refuses the WHOLE request.
fn verify_membership(
    table_name: &str,
    requested: &[i64],
    segments: &[IndexSegment],
) -> Result<(), Status> {
    if let Some(unknown) = requested
        .iter()
        .find(|id| !segments.iter().any(|s| s.segment_id == **id))
    {
        return Err(Status::invalid_argument(format!(
            "segment {unknown} is not a segment of table '{table_name}'"
        )));
    }
    Ok(())
}

fn segment(segments: &[IndexSegment], id: i64) -> &IndexSegment {
    segments
        .iter()
        .find(|s| s.segment_id == id)
        .expect("membership verified above")
}

/// A kernel failure after a successful load — a candidate with no exact
/// vector, a companion read fault — is a torn bundle at this owner.
fn torn(table_name: &str, segment_id: i64, e: JammiError) -> Status {
    Status::data_loss(format!("segment {table_name}/{segment_id}: {e}"))
}

fn hits(pairs: Vec<(String, f32)>) -> Vec<Hit> {
    pairs
        .into_iter()
        .map(|(row_id, distance)| Hit { row_id, distance })
        .collect()
}

#[tonic::async_trait]
impl PeerService for PeerServer {
    #[tracing::instrument(skip(self, request))]
    async fn segment_search(
        &self,
        request: Request<SegmentSearchRequest>,
    ) -> Result<Response<SegmentSearchResponse>, Status> {
        let req = request.into_inner();
        let precision = precision_from_proto(req.storage_precision)
            .ok_or_else(|| Status::invalid_argument("storage_precision is unspecified"))?;
        let phase = phase_from_proto(req.phase)
            .ok_or_else(|| Status::invalid_argument("phase is unspecified"))?;
        let width = usize::try_from(req.width)
            .map_err(|_| Status::invalid_argument("width does not fit usize"))?;
        let store = self.session.result_store();
        let segments = self.segments_of(&store, &req.table_name).await?;
        verify_membership(&req.table_name, &req.segment_ids, &segments)?;

        let mut units = Vec::with_capacity(req.segment_ids.len());
        for id in req.segment_ids {
            let index =
                Self::load(&store, &req.table_name, segment(&segments, id), precision).await?;
            let unit = search_unit(&index, &req.query, width, phase, &|row_id| {
                index.get_exact(row_id)
            })
            .map_err(|e| torn(&req.table_name, id, e))?;
            units.push(SegmentUnit {
                segment_id: id,
                hits: hits(unit),
            });
        }
        Ok(Response::new(SegmentSearchResponse { units }))
    }

    #[tracing::instrument(skip(self, request))]
    async fn exact_rescore(
        &self,
        request: Request<ExactRescoreRequest>,
    ) -> Result<Response<ExactRescoreResponse>, Status> {
        let req = request.into_inner();
        let precision = precision_from_proto(req.storage_precision)
            .ok_or_else(|| Status::invalid_argument("storage_precision is unspecified"))?;
        let store = self.session.result_store();
        let segments = self.segments_of(&store, &req.table_name).await?;
        let requested: Vec<i64> = req
            .row_ids_by_segment
            .iter()
            .map(|g| g.segment_id)
            .collect();
        verify_membership(&req.table_name, &requested, &segments)?;

        let mut out = Vec::new();
        for group in req.row_ids_by_segment {
            let id = group.segment_id;
            let index =
                Self::load(&store, &req.table_name, segment(&segments, id), precision).await?;
            let candidates = group.row_ids.into_iter().map(|r| (r, 0.0f32)).collect();
            let rescored = rescore(candidates, &|row_id| index.get_exact(row_id), &req.query)
                .map_err(|e| torn(&req.table_name, id, e))?;
            out.extend(rescored);
        }
        Ok(Response::new(ExactRescoreResponse { hits: hits(out) }))
    }
}
