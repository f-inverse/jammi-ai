//! `InferenceService` proto↔domain conversions.
//!
//! `Session::infer` returns `Vec<RecordBatch>`; the wire carries them as one
//! Arrow IPC stream in an `ArrowBatch` (the Flight-IPC pairing the trigger and
//! inference surfaces share). An empty result has no schema to encode, so it
//! round-trips as an empty `ArrowBatch`.

use arrow::record_batch::RecordBatch;
use tonic::Status;

use crate::wire::encode_ipc_stream;
use crate::wire::proto::trigger::ArrowBatch;

// The Predict edge-gather + distribution conversions touch the engine-side
// `EdgeGather` / `PredictedDistribution` types, which live behind the `local`
// feature; they are reachable only in a `local + wire` build.
#[cfg(feature = "local")]
use crate::pipeline::context_predictor::PredictedDistribution;
#[cfg(feature = "local")]
use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};
#[cfg(feature = "local")]
use crate::wire::proto::inference as pb;

/// Encode the engine's inference result rows into one `ArrowBatch`. Carries the
/// rows as a single self-describing IPC stream keyed on the first batch's
/// schema; an empty result (empty source) has no schema to encode, so it
/// becomes an empty `ArrowBatch`.
pub fn infer_result_to_proto(batches: Vec<RecordBatch>) -> Result<ArrowBatch, Status> {
    match batches.first() {
        Some(first) => {
            let body = encode_ipc_stream(&first.schema(), &batches)?;
            Ok(ArrowBatch {
                data_header: Vec::new(),
                data_body: body,
                app_metadata: Vec::new(),
            })
        }
        None => Ok(ArrowBatch::default()),
    }
}

// ─── Predict: declared-edge gather + predictive distribution ─────────────────
//
// The `Predict` rpc carries an optional declared-edge gather (the "bring your
// own graph" registered-source case the predict binding exposes) and returns the
// served `PredictedDistribution`. These map the wire shapes onto the engine
// types; the ANN-only default is an absent gather, mirroring the embed binding.

#[cfg(feature = "local")]
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

#[cfg(feature = "local")]
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

#[cfg(feature = "local")]
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

#[cfg(feature = "local")]
fn edge_direction_to_proto(direction: EdgeDirection) -> pb::EdgeDirection {
    match direction {
        EdgeDirection::Out => pb::EdgeDirection::Out,
        EdgeDirection::In => pb::EdgeDirection::In,
        EdgeDirection::Undirected => pb::EdgeDirection::Undirected,
    }
}

#[cfg(feature = "local")]
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
