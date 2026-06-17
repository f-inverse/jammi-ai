//! `InferenceService` engine-spec conversions: the declared-edge gather and the
//! served predictive distribution.
//!
//! The candle-free `infer_result_to_proto` lives on the wire substrate
//! ([`jammi_wire`]); what stays here are the `Predict` conversions that touch the
//! engine-side `EdgeGather` / `PredictedDistribution` types, only reachable in a
//! `local` build.
//!
//! The `Predict` rpc carries an optional declared-edge gather (the "bring your
//! own graph" registered-source case the predict binding exposes) and returns the
//! served `PredictedDistribution`. These map the wire shapes onto the engine
//! types; the ANN-only default is an absent gather, mirroring the embed binding.

use prost::Message;
use tonic::Status;

use crate::model::ModelTask;
use crate::pipeline::context_predictor::PredictedDistribution;
use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};
use jammi_wire::model_task_from_proto;
use jammi_wire::proto::inference as pb;

/// The decoded identity + task an `Infer` request carries. The engine method
/// (`Session::infer`) takes these separately, so the decode returns them as a
/// struct the binding destructures; `model` is the model-id string the engine
/// parses into a [`ModelSource`](crate::model::ModelSource) at the call site,
/// matching the in-process binding.
pub struct InferArgs {
    pub source_id: String,
    pub model: String,
    pub task: ModelTask,
    pub columns: Vec<String>,
    pub key_column: String,
    /// The opt-in memoization policy (the engine method's `cache` arg). Inference
    /// anchors its source `UnpinnedAtInstant`, so `Use` is honestly always a
    /// miss; the field rides for surface uniformity with the cacheable producers.
    pub cache: jammi_db::store::CachePolicy,
}

/// Decode a serialized [`pb::InferRequest`] body into the engine [`InferArgs`].
/// The embedded binding builds the request with the same pure-Python assembly
/// the remote client uses, serializes it, and hands the bytes here — so the
/// in-process and remote infer paths decode through one shared seam
/// ([`infer_from_proto`]). A body that is not a valid `InferRequest` is a client
/// error (`InvalidArgument`).
pub fn infer_from_bytes(body: &[u8]) -> Result<InferArgs, Status> {
    let req = pb::InferRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed Infer request: {e}")))?;
    infer_from_proto(req)
}

/// Decode a [`pb::InferRequest`] into the engine [`InferArgs`]. The required
/// identity fields (`source_id` / `model_id` / `key_column`) and a non-empty
/// `columns` list are validated at decode rather than deferred to the engine,
/// matching the in-process binding's required arguments.
pub fn infer_from_proto(req: pb::InferRequest) -> Result<InferArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.model_id.is_empty() {
        return Err(Status::invalid_argument("model_id is required"));
    }
    if req.key_column.is_empty() {
        return Err(Status::invalid_argument("key_column is required"));
    }
    if req.columns.is_empty() {
        return Err(Status::invalid_argument("columns is required"));
    }
    Ok(InferArgs {
        source_id: req.source_id,
        model: req.model_id,
        task: model_task_from_proto(req.task)?,
        columns: req.columns,
        key_column: req.key_column,
        cache: crate::wire::cache_policy_from_proto(req.cache)?,
    })
}

/// Decode the optional wire [`pb::EdgeGather`] into an engine [`EdgeGather`] over
/// a registered edge source. `None` = the ANN-only default. `hops == 0` keeps the
/// gather's default hop count (the engine clamps to its hop cap).
pub fn edge_gather_from_proto(
    gather: Option<pb::EdgeGather>,
) -> Result<Option<EdgeGather>, Status> {
    let Some(g) = gather else {
        return Ok(None);
    };
    if g.edge_source.is_empty() {
        return Err(Status::invalid_argument(
            "edge gather carries no edge_source",
        ));
    }
    let mut out = EdgeGather::new(EdgeSourceRef::Registered {
        source_id: g.edge_source,
        src_column: g.src_column,
        dst_column: g.dst_column,
        type_column: g.type_column,
        weight_column: g.weight_column,
        as_of_column: None,
    });
    if g.hops != 0 {
        out.hops = g.hops as usize;
    }
    out.fanout = g.fanout.map(|f| f as usize);
    out.direction = edge_direction_from_proto(g.direction)?;
    out.edge_types = if g.edge_types.is_empty() {
        None
    } else {
        Some(g.edge_types)
    };
    out.min_weight = g.min_weight;
    Ok(Some(out))
}

/// Encode an engine [`EdgeGather`] over a registered edge source onto the wire,
/// for the remote send side. Only the registered-source case is reachable from
/// the predict binding; an S9-`NeighborGraph` source is not a remote predict
/// input (it has no client-facing column bindings), so it is rejected here.
pub fn edge_gather_to_proto(gather: &EdgeGather) -> Result<pb::EdgeGather, Status> {
    let EdgeSourceRef::Registered {
        source_id,
        src_column,
        dst_column,
        type_column,
        weight_column,
        as_of_column: _,
    } = &gather.edge_source
    else {
        return Err(Status::invalid_argument(
            "remote predict carries only a registered edge source",
        ));
    };
    Ok(pb::EdgeGather {
        edge_source: source_id.clone(),
        src_column: src_column.clone(),
        dst_column: dst_column.clone(),
        type_column: type_column.clone(),
        weight_column: weight_column.clone(),
        hops: gather.hops as u32,
        fanout: gather.fanout.map(|f| f as u32),
        direction: edge_direction_to_proto(gather.direction) as i32,
        edge_types: gather.edge_types.clone().unwrap_or_default(),
        min_weight: gather.min_weight,
    })
}

fn edge_direction_from_proto(direction: i32) -> Result<EdgeDirection, Status> {
    match pb::EdgeDirection::try_from(direction) {
        // An unspecified direction defaults to the engine's `Out` (the
        // `EdgeGather::new` default), so a client that omits it gets the same
        // behaviour as the embed binding's default.
        Ok(pb::EdgeDirection::Unspecified) | Ok(pb::EdgeDirection::Out) => Ok(EdgeDirection::Out),
        Ok(pb::EdgeDirection::In) => Ok(EdgeDirection::In),
        Ok(pb::EdgeDirection::Undirected) => Ok(EdgeDirection::Undirected),
        Err(_) => Err(Status::invalid_argument("unknown edge direction")),
    }
}

fn edge_direction_to_proto(direction: EdgeDirection) -> pb::EdgeDirection {
    match direction {
        EdgeDirection::Out => pb::EdgeDirection::Out,
        EdgeDirection::In => pb::EdgeDirection::In,
        EdgeDirection::Undirected => pb::EdgeDirection::Undirected,
    }
}

/// Encode a served [`PredictedDistribution`] onto the wire `oneof`. A Gaussian
/// head carries `(mean, std)`; a quantile head carries its ascending
/// `(level, value)` points — the same shape the embed binding's dict exposes.
pub fn predicted_distribution_to_proto(
    distribution: &PredictedDistribution,
) -> pb::predict_response::Distribution {
    match distribution {
        PredictedDistribution::Gaussian { mean, std } => {
            pb::predict_response::Distribution::Gaussian(pb::predict_response::Gaussian {
                mean: *mean as f64,
                std: *std as f64,
            })
        }
        PredictedDistribution::Quantile { levels } => {
            pb::predict_response::Distribution::Quantile(pb::predict_response::Quantile {
                points: levels
                    .iter()
                    .map(|(level, value)| pb::predict_response::QuantilePoint {
                        level: *level,
                        value: *value as f64,
                    })
                    .collect(),
            })
        }
    }
}
