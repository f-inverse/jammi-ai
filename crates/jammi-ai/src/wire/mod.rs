//! The engine's residual wire surface: the engine-spec proto↔domain conversions
//! the candle-free [`jammi_wire`] substrate cannot home.
//!
//! Everything candle-free — the generated `jammi.v1` stubs, the request / eval /
//! fine-tune vocabulary, the IPC framing, the `CatalogService` / embedding /
//! trigger / audit / error converters, and the `FineTuneConfig` decode — lives in
//! [`jammi_wire`]. What stays here are the conversions that touch the engine spec
//! vocabulary (`TrainingSpec`, the graph sampler + context-predictor config, the
//! declared-edge `EdgeGather`, the served `PredictedDistribution`, and the
//! pipeline request/response structs), which are only reachable in an engine
//! (`local`) build. They are built on jammi-wire's `proto` + helpers, so the
//! mapping is shared with the server's receive side rather than reimplemented.

// The engine-spec converters reconstruct / project the engine pipeline request +
// response structs (`BuildNeighborGraph` / `PropagateRequest` / `ContextRequest`
// / `ContextRepresentation`), which live behind the `local` feature — reachable
// only in an engine build (the server and the embedded SDK).
mod inference;
mod pipeline;
mod training;

pub use inference::{
    edge_gather_from_proto, edge_gather_to_proto, predicted_distribution_to_proto,
};
pub use pipeline::{
    assemble_context_from_proto, assemble_context_request_from_proto, assemble_context_to_proto,
    build_neighbor_graph_from_proto, context_source_tag, propagate_request_from_proto,
    BuildNeighborGraphArgs,
};
pub use training::{training_spec_from_proto, training_spec_to_proto};
