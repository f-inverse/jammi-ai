//! The gRPC implementation of the engine's peer-search transport seam.
//!
//! [`GrpcPeerTransport`] implements [`jammi_db::index::PeerTransport`] over the
//! generated `jammi.v1.peer.PeerService` client: one lazily-connected
//! [`Channel`] per owner address, one unary call per phase, the caller's
//! per-RPC deadline applied BOTH as the `grpc-timeout` the owner honours and
//! as a client-side `tokio::time::timeout` (so a hung owner can never hold a
//! coordinator past its budget). Every failure is classified into a
//! [`PeerFailureReason`] the coordinator's ladder and counters key on.
//!
//! This file also owns the proto↔domain conversions for the peer vocabulary
//! ([`precision_to_proto`] / [`precision_from_proto`], [`phase_to_proto`] /
//! [`phase_from_proto`]) so the owner handler in `jammi-server` and this client
//! decode the same way.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Request, Status};

use jammi_db::config::StoragePrecision;
use jammi_db::index::peer::{
    ExactRescoreRequest, PeerAddr, PeerError, PeerFailureReason, PeerTransport, SegmentSearchPhase,
    SegmentSearchRequest, SegmentUnit,
};
use jammi_db::index::SegmentId;

use crate::proto::peer as pb;
use crate::proto::peer::peer_service_client::PeerServiceClient;

/// Encode a [`StoragePrecision`] as its wire enum value.
pub fn precision_to_proto(precision: StoragePrecision) -> pb::StoragePrecision {
    match precision {
        StoragePrecision::F32 => pb::StoragePrecision::F32,
        StoragePrecision::F16 => pb::StoragePrecision::F16,
        StoragePrecision::Int8 => pb::StoragePrecision::Int8,
        StoragePrecision::Binary => pb::StoragePrecision::Binary,
    }
}

/// Decode a wire precision (the raw `i32` prost carries). `None` for
/// `UNSPECIFIED` or an unknown value — the receiver refuses, never defaults.
pub fn precision_from_proto(raw: i32) -> Option<StoragePrecision> {
    match pb::StoragePrecision::try_from(raw).ok()? {
        pb::StoragePrecision::F32 => Some(StoragePrecision::F32),
        pb::StoragePrecision::F16 => Some(StoragePrecision::F16),
        pb::StoragePrecision::Int8 => Some(StoragePrecision::Int8),
        pb::StoragePrecision::Binary => Some(StoragePrecision::Binary),
        pb::StoragePrecision::Unspecified => None,
    }
}

/// Encode a [`SegmentSearchPhase`] as its wire enum value.
pub fn phase_to_proto(phase: SegmentSearchPhase) -> pb::SegmentSearchPhase {
    match phase {
        SegmentSearchPhase::Approximate => pb::SegmentSearchPhase::Approximate,
        SegmentSearchPhase::Final => pb::SegmentSearchPhase::Final,
    }
}

/// Decode a wire phase. `None` for `UNSPECIFIED` or an unknown value.
pub fn phase_from_proto(raw: i32) -> Option<SegmentSearchPhase> {
    match pb::SegmentSearchPhase::try_from(raw).ok()? {
        pb::SegmentSearchPhase::Approximate => Some(SegmentSearchPhase::Approximate),
        pb::SegmentSearchPhase::Final => Some(SegmentSearchPhase::Final),
        pb::SegmentSearchPhase::Unspecified => None,
    }
}

/// Classify a failed peer call. `DEADLINE_EXCEEDED` is the owner's own
/// deadline; `UNAVAILABLE` is tonic's transport-level failure (connection
/// refused, reset, no route); the owner's input-edge refusals
/// (`INVALID_ARGUMENT` for an unknown segment id, `FAILED_PRECONDITION` for a
/// precision the bundle is not stamped with) are `Refused`; `DATA_LOSS` is a
/// torn bundle at the owner; everything else is a generic transport fault.
pub fn classify_status(status: &Status) -> PeerFailureReason {
    match status.code() {
        Code::DeadlineExceeded => PeerFailureReason::Deadline,
        Code::Unavailable => PeerFailureReason::Unreachable,
        Code::InvalidArgument | Code::FailedPrecondition | Code::NotFound => {
            PeerFailureReason::Refused
        }
        Code::DataLoss => PeerFailureReason::Torn,
        _ => PeerFailureReason::Transport,
    }
}

/// The tonic [`PeerTransport`]: one lazily-connected channel per owner,
/// reused across calls for the life of the transport.
#[derive(Default)]
pub struct GrpcPeerTransport {
    channels: Mutex<HashMap<PeerAddr, Channel>>,
}

impl GrpcPeerTransport {
    /// A transport with no channels open yet; each owner's channel is created
    /// on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// The channel for `owner`, created lazily (`connect_lazy`: the TCP
    /// connection is attempted on the first call, so an unreachable owner
    /// surfaces as that call's `UNAVAILABLE`, not as a construction error).
    fn channel(&self, owner: &PeerAddr) -> Result<Channel, PeerFailureReason> {
        let mut channels = self
            .channels
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(channel) = channels.get(owner) {
            return Ok(channel.clone());
        }
        let endpoint = Endpoint::from_shared(format!("http://{}", owner.0))
            .map_err(|_| PeerFailureReason::Unreachable)?;
        let channel = endpoint.connect_lazy();
        channels.insert(owner.clone(), channel.clone());
        Ok(channel)
    }
}

/// Run one unary call under `deadline`: the header the owner honours plus a
/// client-side timeout, so an owner that never answers is `Deadline` too.
async fn bounded<T, F>(deadline: Duration, call: F) -> Result<T, PeerFailureReason>
where
    F: std::future::Future<Output = Result<tonic::Response<T>, Status>>,
{
    match tokio::time::timeout(deadline, call).await {
        Ok(Ok(response)) => Ok(response.into_inner()),
        Ok(Err(status)) => Err(classify_status(&status)),
        Err(_elapsed) => Err(PeerFailureReason::Deadline),
    }
}

#[tonic::async_trait]
impl PeerTransport for GrpcPeerTransport {
    async fn segment_search(
        &self,
        owner: &PeerAddr,
        req: &SegmentSearchRequest,
        deadline: Duration,
    ) -> Result<Vec<SegmentUnit>, PeerError> {
        let first = req.segment_ids.first().copied().unwrap_or(SegmentId(-1));
        let fail = |reason| PeerError {
            segment: first,
            owner: owner.clone(),
            reason,
        };
        let channel = self.channel(owner).map_err(fail)?;
        let mut client = PeerServiceClient::new(channel);
        let mut request = Request::new(pb::SegmentSearchRequest {
            table_name: req.table_name.clone(),
            segment_ids: req.segment_ids.iter().map(|s| s.0).collect(),
            storage_precision: precision_to_proto(req.storage_precision) as i32,
            query: req.query.clone(),
            width: req.width as u64,
            phase: phase_to_proto(req.phase) as i32,
        });
        request.set_timeout(deadline);
        let response: pb::SegmentSearchResponse = bounded(deadline, client.segment_search(request))
            .await
            .map_err(fail)?;
        Ok(response
            .units
            .into_iter()
            .map(|u| SegmentUnit {
                segment_id: SegmentId(u.segment_id),
                hits: u.hits.into_iter().map(|h| (h.row_id, h.distance)).collect(),
            })
            .collect())
    }

    async fn exact_rescore(
        &self,
        owner: &PeerAddr,
        req: &ExactRescoreRequest,
        deadline: Duration,
    ) -> Result<Vec<(String, f32)>, PeerError> {
        let first = req
            .row_ids_by_segment
            .first()
            .map(|(s, _)| *s)
            .unwrap_or(SegmentId(-1));
        let fail = |reason| PeerError {
            segment: first,
            owner: owner.clone(),
            reason,
        };
        let channel = self.channel(owner).map_err(fail)?;
        let mut client = PeerServiceClient::new(channel);
        let mut request = Request::new(pb::ExactRescoreRequest {
            table_name: req.table_name.clone(),
            storage_precision: precision_to_proto(req.storage_precision) as i32,
            query: req.query.clone(),
            row_ids_by_segment: req
                .row_ids_by_segment
                .iter()
                .map(|(segment, row_ids)| pb::SegmentRowIds {
                    segment_id: segment.0,
                    row_ids: row_ids.clone(),
                })
                .collect(),
        });
        request.set_timeout(deadline);
        let response: pb::ExactRescoreResponse = bounded(deadline, client.exact_rescore(request))
            .await
            .map_err(fail)?;
        Ok(response
            .hits
            .into_iter()
            .map(|h| (h.row_id, h.distance))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_and_phase_round_trip_and_refuse_unspecified() {
        for p in [
            StoragePrecision::F32,
            StoragePrecision::F16,
            StoragePrecision::Int8,
            StoragePrecision::Binary,
        ] {
            assert_eq!(precision_from_proto(precision_to_proto(p) as i32), Some(p));
        }
        assert_eq!(precision_from_proto(0), None);
        assert_eq!(precision_from_proto(99), None);
        for ph in [SegmentSearchPhase::Approximate, SegmentSearchPhase::Final] {
            assert_eq!(phase_from_proto(phase_to_proto(ph) as i32), Some(ph));
        }
        assert_eq!(phase_from_proto(0), None);
    }

    #[test]
    fn status_classification() {
        assert_eq!(
            classify_status(&Status::deadline_exceeded("x")),
            PeerFailureReason::Deadline
        );
        assert_eq!(
            classify_status(&Status::unavailable("x")),
            PeerFailureReason::Unreachable
        );
        assert_eq!(
            classify_status(&Status::invalid_argument("x")),
            PeerFailureReason::Refused
        );
        assert_eq!(
            classify_status(&Status::failed_precondition("x")),
            PeerFailureReason::Refused
        );
        assert_eq!(
            classify_status(&Status::data_loss("x")),
            PeerFailureReason::Torn
        );
        assert_eq!(
            classify_status(&Status::internal("x")),
            PeerFailureReason::Transport
        );
    }

    /// A port nothing listens on: connection refused is `Unreachable`, and
    /// the failure names the first requested segment and the owner.
    #[tokio::test]
    async fn unreachable_owner_is_classified_unreachable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let transport = GrpcPeerTransport::new();
        let owner = PeerAddr(addr.to_string());
        let req = SegmentSearchRequest {
            table_name: "t".into(),
            segment_ids: vec![SegmentId(7)],
            storage_precision: StoragePrecision::F32,
            query: vec![1.0],
            width: 1,
            phase: SegmentSearchPhase::Final,
        };
        let err = transport
            .segment_search(&owner, &req, Duration::from_secs(2))
            .await
            .unwrap_err();
        assert_eq!(err.reason, PeerFailureReason::Unreachable, "{err}");
        assert_eq!(err.segment, SegmentId(7));
        assert_eq!(err.owner, owner);
    }
}
