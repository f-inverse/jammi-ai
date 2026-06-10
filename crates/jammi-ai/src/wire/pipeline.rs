//! `PipelineService` proto↔domain conversions.
//!
//! The request types reconstruct the engine's pipeline request structs
//! ([`BuildNeighborGraph`], [`PropagateRequest`], [`ContextRequest`])
//! field-for-field so a remote-submitted verb is byte-identical to one called
//! in-process. The engine is the single source of default values: the build
//! knobs that are not on the wire (`self_exclude` / `exact_max_rows` /
//! `resolve_keys`) resolve to [`BuildNeighborGraph::default`], every genuinely-
//! optional scalar carries explicit presence (`optional`), and an `UNSPECIFIED`
//! enum arm maps to the engine default variant. The `propagate` graph arm is a
//! proto `oneof`, so both/neither is a decode error — matching the in-process
//! binding. The edge gather reuses the shared [`super::edge_gather_from_proto`].
//!
//! The [`AssembleContext`](pb::AssembleContextResponse) response carries the
//! pooled context vector inline as `repeated float` (IEEE-754 binary32, bit-exact
//! for the engine's `Vec<f32>`), presence-wrapped so a degenerate empty context
//! (`None`) stays distinguishable from a present-but-empty vector; the hydrated
//! value rows ride the shared Arrow-IPC pairing.
//!
//! Every conversion here touches the engine-side pipeline vocabulary
//! (`PropagateRequest` / `ContextRequest` / [`ContextRepresentation`]), which
//! lives behind the `local` feature, so the whole module is `local`-gated at its
//! mount in [`super`] — reachable only in a `local + wire` build (the server and
//! the embedded SDK). A thin wire-only client carries no engine pipeline type to
//! reconstruct.

use tonic::Status;

use crate::pipeline::context_set::{
    ContextRepresentation, ContextRequest, ContextSource, ContextSourceKind, HybridMerge,
    SetAggregator,
};
use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use crate::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use crate::pipeline::neighbor_graph::BuildNeighborGraph;
use crate::wire::edge_gather_from_proto;
use jammi_wire::proto::pipeline as pb;
use jammi_wire::proto::trigger::ArrowBatch;
use jammi_wire::{decode_ipc_stream, encode_ipc_stream};

// ─── BuildNeighborGraph ──────────────────────────────────────────────────────

/// The decoded source identity + optional embedding-table selector + build
/// params a `BuildNeighborGraph` request carries. The engine method takes these
/// separately, so the decode returns them as a struct the handler destructures.
pub struct BuildNeighborGraphArgs {
    pub source_id: String,
    /// The specific embedding table to build the graph over, or `None` to
    /// resolve the source's default — the engine method's `embedding_table` arg.
    pub embedding_table: Option<String>,
    pub params: BuildNeighborGraph,
}

/// Decode a [`pb::BuildNeighborGraphRequest`] into the engine's build args. The
/// off-wire knobs (`self_exclude` / `exact_max_rows` / `resolve_keys`) come from
/// [`BuildNeighborGraph::default`] so the engine stays the single source of
/// those values; the optional wire scalars override only when present.
pub fn build_neighbor_graph_from_proto(
    req: pb::BuildNeighborGraphRequest,
) -> Result<BuildNeighborGraphArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    let params = BuildNeighborGraph {
        k: req.k as usize,
        min_similarity: req.min_similarity,
        mutual: req.mutual,
        exact: req.exact,
        ..Default::default()
    };
    Ok(BuildNeighborGraphArgs {
        source_id: req.source_id,
        embedding_table: req.table,
        params,
    })
}

// ─── PropagateEmbeddings ─────────────────────────────────────────────────────

/// Decode a [`pb::PropagateEmbeddingsRequest`] into the engine [`PropagateRequest`].
/// The `graph` oneof selects the edge source (an S9 neighbour-graph table or a
/// registered edge source); both/neither is a decode error. Each optional scalar
/// overrides only when present, so an unset field keeps the [`PropagateRequest`]
/// builder default — the engine's value.
pub fn propagate_request_from_proto(
    req: pb::PropagateEmbeddingsRequest,
) -> Result<PropagateRequest, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    let edge_source = match req.graph {
        Some(pb::propagate_embeddings_request::Graph::EdgeGraphTable(table_name)) => {
            if table_name.is_empty() {
                return Err(Status::invalid_argument("edge_graph_table is empty"));
            }
            EdgeSourceRef::NeighborGraph { table_name }
        }
        Some(pb::propagate_embeddings_request::Graph::EdgeSource(s)) => {
            if s.edge_source.is_empty() {
                return Err(Status::invalid_argument("edge_source is empty"));
            }
            EdgeSourceRef::Registered {
                source_id: s.edge_source,
                src_column: empty_or(s.src_column, "src"),
                dst_column: empty_or(s.dst_column, "dst"),
                type_column: None,
                weight_column: s.weight_column,
                as_of_column: None,
            }
        }
        None => {
            return Err(Status::invalid_argument(
                "propagate requires a graph: edge_graph_table or edge_source",
            ))
        }
    };

    let mut request = PropagateRequest::new(req.source_id, edge_source)
        .with_direction(propagation_direction_from_proto(req.direction)?)
        .with_weighting(propagation_weighting_from_proto(req.weighting)?)
        .with_output(propagation_output_from_proto(req.output)?);
    if let Some(table) = req.embedding_table {
        request = request.with_embedding_table(table);
    }
    if let Some(hops) = req.hops {
        request = request.with_hops(hops as usize);
    }
    if let Some(alpha) = req.alpha {
        request = request.with_alpha(alpha);
    }
    Ok(request)
}

/// `""` → the engine's binding default column name; a non-empty value is kept.
fn empty_or(value: String, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value
    }
}

/// Map the propagation edge direction (reusing the shared inference
/// [`EdgeDirection`] enum); `UNSPECIFIED` keeps the engine default (`Out`).
fn propagation_direction_from_proto(direction: i32) -> Result<EdgeDirection, Status> {
    use jammi_wire::proto::inference::EdgeDirection as ProtoDir;
    match ProtoDir::try_from(direction) {
        Ok(ProtoDir::Unspecified) | Ok(ProtoDir::Out) => Ok(EdgeDirection::Out),
        Ok(ProtoDir::In) => Ok(EdgeDirection::In),
        Ok(ProtoDir::Undirected) => Ok(EdgeDirection::Undirected),
        Err(_) => Err(Status::invalid_argument("unknown edge direction")),
    }
}

/// Map the wire [`pb::PropagationWeighting`]; `UNSPECIFIED` keeps the engine
/// default (`DegreeNormalized`).
fn propagation_weighting_from_proto(weighting: i32) -> Result<PropagationWeighting, Status> {
    match pb::PropagationWeighting::try_from(weighting) {
        Ok(pb::PropagationWeighting::Unspecified)
        | Ok(pb::PropagationWeighting::DegreeNormalized) => {
            Ok(PropagationWeighting::DegreeNormalized)
        }
        Ok(pb::PropagationWeighting::Uniform) => Ok(PropagationWeighting::Uniform),
        Ok(pb::PropagationWeighting::EdgeSimilarity) => Ok(PropagationWeighting::EdgeSimilarity),
        Err(_) => Err(Status::invalid_argument("unknown propagation weighting")),
    }
}

/// Map the wire [`pb::PropagationOutput`]; `UNSPECIFIED` keeps the engine
/// default (`Final`).
fn propagation_output_from_proto(output: i32) -> Result<PropagationOutput, Status> {
    match pb::PropagationOutput::try_from(output) {
        Ok(pb::PropagationOutput::Unspecified) | Ok(pb::PropagationOutput::Final) => {
            Ok(PropagationOutput::Final)
        }
        Ok(pb::PropagationOutput::JumpingKnowledge) => Ok(PropagationOutput::JumpingKnowledge),
        Err(_) => Err(Status::invalid_argument("unknown propagation output")),
    }
}

// ─── AssembleContext ─────────────────────────────────────────────────────────

/// Decode a [`pb::AssembleContextRequest`] into the engine [`ContextRequest`].
/// The optional edge gather decodes through the shared [`edge_gather_from_proto`]
/// (an absent gather is the ANN-only default, present is a declared-edge walk,
/// present with `hybrid` is the union) — the same selection the in-process
/// binding makes.
pub fn assemble_context_request_from_proto(
    req: pb::AssembleContextRequest,
) -> Result<ContextRequest, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    let aggregator = set_aggregator_from_proto(req.aggregator)?;
    let gather = edge_gather_from_proto(req.edges)?;
    let context_source = match gather {
        None => ContextSource::Ann { k: req.k as usize },
        Some(edges) if req.hybrid => ContextSource::Hybrid {
            ann_k: req.k as usize,
            edges,
            merge: HybridMerge::Union,
        },
        Some(edges) => ContextSource::Edges(edges),
    };

    let mut request = ContextRequest::new(req.source_id, req.query, req.k as usize);
    request.source = context_source;
    request.value_columns = req.value_columns;
    request.aggregator = aggregator;
    // `exclude_self` carries explicit presence: an unset field keeps the
    // leakage-safe `ContextRequest::new` default (`true`), the engine's single
    // source of that value, so a client that omits it never silently disables
    // the self-exclusion guard via the proto3 bare-bool wire default (`false`).
    if let Some(exclude_self) = req.exclude_self {
        request.exclude_self = exclude_self;
    }
    request.exclude_key = req.exclude_key;
    request.split = req.split;
    Ok(request)
}

/// Map the wire [`pb::SetAggregator`]; `UNSPECIFIED` keeps the engine default
/// (`Mean`).
fn set_aggregator_from_proto(aggregator: i32) -> Result<SetAggregator, Status> {
    match pb::SetAggregator::try_from(aggregator) {
        Ok(pb::SetAggregator::Unspecified) | Ok(pb::SetAggregator::Mean) => Ok(SetAggregator::Mean),
        Ok(pb::SetAggregator::Sum) => Ok(SetAggregator::Sum),
        Ok(pb::SetAggregator::Max) => Ok(SetAggregator::Max),
        Err(_) => Err(Status::invalid_argument("unknown set aggregator")),
    }
}

/// The string tag for a context's assembly fact ("ann" / "edges" / "hybrid").
/// The same vocabulary the embed binding's dict and the predict response expose.
pub fn context_source_tag(kind: ContextSourceKind) -> &'static str {
    match kind {
        ContextSourceKind::Ann => "ann",
        ContextSourceKind::Edges => "edges",
        ContextSourceKind::Hybrid => "hybrid",
    }
}

/// Encode an assembled [`ContextRepresentation`] onto the wire response. The
/// pooled vector rides the wire as `repeated float` (IEEE-754 binary32, bit-
/// exact for the engine's `Vec<f32>`); a degenerate empty context (`None`) is a
/// presence-wrapped *absence*, never a present-but-empty vector — the
/// correctness signal a decoder reads as low-confidence. The hydrated value rows
/// cross as one Arrow IPC stream in an [`ArrowBatch`].
pub fn assemble_context_to_proto(
    context: ContextRepresentation,
) -> Result<pb::AssembleContextResponse, Status> {
    let value_rows = match context.value_rows.first() {
        Some(first) => {
            let body = encode_ipc_stream(&first.schema(), &context.value_rows)?;
            Some(ArrowBatch {
                data_header: Vec::new(),
                data_body: body,
                app_metadata: Vec::new(),
            })
        }
        None => None,
    };
    Ok(pb::AssembleContextResponse {
        context_vector: context
            .context_vector
            .map(|values| pb::ContextVector { values }),
        context_size: context.context_size as u64,
        context_keys: context.context_keys,
        value_rows,
        source: context_source_tag(context.source).to_string(),
    })
}

/// Decode a [`pb::AssembleContextResponse`] into the engine
/// [`ContextRepresentation`] — the the remote client receive side. The
/// presence-wrapped vector decodes back to `Option<Vec<f32>>` (an absent wrapper
/// is the degenerate empty context), and the value rows decode from their IPC
/// stream. The assembly-source tag rejects an unknown value rather than guess.
pub fn assemble_context_from_proto(
    resp: pb::AssembleContextResponse,
) -> Result<ContextRepresentation, Status> {
    let value_rows = match resp.value_rows {
        Some(batch) => decode_ipc_stream(&batch.data_header, &batch.data_body)?,
        None => Vec::new(),
    };
    Ok(ContextRepresentation {
        context_vector: resp.context_vector.map(|v| v.values),
        context_size: resp.context_size as usize,
        context_keys: resp.context_keys,
        value_rows,
        source: context_source_kind_from_tag(&resp.source)?,
    })
}

/// Map the assembly-source tag back onto the engine [`ContextSourceKind`]. An
/// unknown tag is a corrupt/incompatible payload, rejected rather than guessed.
fn context_source_kind_from_tag(tag: &str) -> Result<ContextSourceKind, Status> {
    match tag {
        "ann" => Ok(ContextSourceKind::Ann),
        "edges" => Ok(ContextSourceKind::Edges),
        "hybrid" => Ok(ContextSourceKind::Hybrid),
        other => Err(Status::invalid_argument(format!(
            "unknown context source tag '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// The pooled context vector round-trips bit-exactly through the wire's
    /// `repeated float` (IEEE-754 binary32) field. A value chosen so any
    /// `f32 → f64 → f32` detour (a `double` wire field) would perturb its bits
    /// — `0.1f32` has no exact binary representation, so a widening round-trip
    /// is observable. The presence wrapper preserves a non-empty vector.
    #[test]
    fn context_vector_round_trips_bit_exact() {
        let values: Vec<f32> = vec![0.1, 0.2, 0.3, f32::MIN_POSITIVE, 1.0 / 3.0, -2.5e-7];
        let context = ContextRepresentation {
            context_vector: Some(values.clone()),
            context_size: 4,
            context_keys: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            value_rows: Vec::new(),
            source: ContextSourceKind::Ann,
        };

        let proto = assemble_context_to_proto(context).expect("encode");
        let decoded = assemble_context_from_proto(proto).expect("decode");

        let got = decoded.context_vector.expect("vector present");
        assert_eq!(got.len(), values.len());
        for (i, (g, w)) in got.iter().zip(&values).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "element {i} differs bit-for-bit: {g} vs {w}"
            );
        }
        assert_eq!(decoded.context_size, 4);
        assert_eq!(decoded.context_keys, vec!["a", "b", "c", "d"]);
        assert_eq!(decoded.source, ContextSourceKind::Ann);
    }

    /// A degenerate empty context (`None` vector) is a presence-wrapped absence
    /// on the wire and decodes back to `None` — distinguishable from a present-
    /// but-empty vector. `context_size = 0` rides alongside it.
    #[test]
    fn degenerate_empty_context_round_trips_as_none() {
        let context = ContextRepresentation {
            context_vector: None,
            context_size: 0,
            context_keys: Vec::new(),
            value_rows: Vec::new(),
            source: ContextSourceKind::Edges,
        };
        let proto = assemble_context_to_proto(context).expect("encode");
        assert!(
            proto.context_vector.is_none(),
            "degenerate context is an absent wrapper, not an empty vector"
        );
        let decoded = assemble_context_from_proto(proto).expect("decode");
        assert!(
            decoded.context_vector.is_none(),
            "absence survives the round-trip"
        );
        assert_eq!(decoded.context_size, 0);
        assert_eq!(decoded.source, ContextSourceKind::Edges);
    }

    /// A minimal `AssembleContextRequest` carrying just the required identity +
    /// query + `k`, with every optional field left at its proto default. The
    /// helper isolates the `exclude_self` decode from the rest of the surface.
    fn minimal_assemble_request(exclude_self: Option<bool>) -> pb::AssembleContextRequest {
        pb::AssembleContextRequest {
            source_id: "src".into(),
            query: vec![0.1, 0.2],
            k: 5,
            value_columns: Vec::new(),
            aggregator: pb::SetAggregator::Unspecified as i32,
            exclude_self,
            exclude_key: None,
            split: None,
            edges: None,
            hybrid: false,
        }
    }

    /// `exclude_self` is presence-carried: an **unset** wire field must decode to
    /// the engine's leakage-safe default (`true`), never the proto3 bare-bool
    /// wire default (`false`) that would silently pool the target's own row into
    /// its context (train/serve leakage). An explicit `false`/`true` is honoured.
    ///
    /// Pre-fix (bare `bool exclude_self`) the unset case decoded to `false`, so
    /// this test fails on the first assertion; post-fix (`optional bool`,
    /// override-only-when-present) it passes.
    #[test]
    fn exclude_self_unset_decodes_to_engine_default_true() {
        let unset = assemble_context_request_from_proto(minimal_assemble_request(None))
            .expect("decode unset");
        assert!(
            unset.exclude_self,
            "an omitted exclude_self must keep the engine default (true), not the \
             proto3 bare-bool wire default (false) — omitting it must not disable \
             the self-exclusion leakage guard"
        );

        let off = assemble_context_request_from_proto(minimal_assemble_request(Some(false)))
            .expect("decode false");
        assert!(!off.exclude_self, "explicit false is honoured");

        let on = assemble_context_request_from_proto(minimal_assemble_request(Some(true)))
            .expect("decode true");
        assert!(on.exclude_self, "explicit true is honoured");
    }

    /// The hydrated value rows cross the wire as an Arrow IPC stream and decode
    /// back to the identical batch (schema + values), and the assembly-source
    /// tag round-trips through the string vocabulary.
    #[test]
    fn value_rows_and_source_tag_round_trip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new("outcome", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["k1", "k2"])),
                Arc::new(Float64Array::from(vec![1.5, 2.5])),
            ],
        )
        .expect("batch");
        let context = ContextRepresentation {
            context_vector: Some(vec![0.5, 0.5]),
            context_size: 2,
            context_keys: vec!["k1".into(), "k2".into()],
            value_rows: vec![batch.clone()],
            source: ContextSourceKind::Hybrid,
        };

        let proto = assemble_context_to_proto(context).expect("encode");
        let decoded = assemble_context_from_proto(proto).expect("decode");

        assert_eq!(decoded.value_rows.len(), 1);
        assert_eq!(decoded.value_rows[0], batch);
        assert_eq!(decoded.source, ContextSourceKind::Hybrid);
    }
}
