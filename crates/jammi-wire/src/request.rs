//! Request vocabulary the consumer surface shares with the wire.
//!
//! These owned, serialisable request shapes are what a session verb takes
//! (`Modality` / `QueryInput` / `SearchRequest`) and the addressable id a
//! fine-tune job returns (`FineTuneJobId`). They hold no engine state, so they
//! live on the wire substrate: the embedded session and the data-plane client
//! both build verbs from them, and the gRPC converters map them on/off the wire.

use std::num::NonZeroU32;

use jammi_db::ModelTask;

use crate::fine_tune::{FineTuneConfig, FineTuneMethod};

/// Which embedding tower an embeddings / encode-query call targets. Unifies the
/// three per-modality engine verbs (`text`/`image`/`audio`) into one parameter
/// so the consumer surface carries one embedding verb, not three.
///
/// `Serialize`/`Deserialize`: persisted on `jobs.spec` as part of
/// `jammi_ai::jobs::ComputeSpec::Embedding` — a generate-embeddings call
/// submitted through `InferenceSession::run_now` must round-trip its
/// modality byte-for-byte on a fresh process's replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Dense vectors of input text.
    Text,
    /// Dense vectors of input images.
    Image,
    /// Dense vectors of input audio clips.
    Audio,
}

/// A single query to encode into a vector. Text is encoded by the text tower;
/// raw bytes are encoded by the image or audio tower, with the [`Modality`] the
/// caller passes selecting which.
pub enum QueryInput {
    /// A text string to encode with the text tower.
    Text(String),
    /// Encoded bytes (an image file or an audio clip) for the vision/audio
    /// tower.
    Bytes(Vec<u8>),
}

/// The query side of a flattened search: either a caller-supplied vector or a
/// row key resolved to its stored vector inside the engine.
pub enum SearchQuery {
    /// Search against a caller-supplied query vector.
    Vector(Vec<f32>),
    /// Query-by-example: rank by the vector stored for this row key.
    RowKey(String),
}

/// A flattened vector-search request. Every knob a one-shot search exposes
/// (`embedding_table`, `filter`, `select`) is a field here, so a transport can
/// serialise the whole request rather than replay a chain of builder calls.
pub struct SearchRequest {
    /// Source whose embedding table is searched.
    pub source_id: String,
    /// The query vector or the row key to resolve into one.
    pub query: SearchQuery,
    /// Number of nearest neighbours to retrieve.
    pub k: usize,
    /// Which embedding table of the source to search. `None` selects the
    /// source's most-recent ready table; `Some(name)` searches that table.
    pub embedding_table: Option<String>,
    /// Optional SQL predicate applied to the hydrated results.
    pub filter: Option<String>,
    /// Columns to project. Empty keeps every hydrated column.
    pub select: Vec<String>,
    /// Per-request override of the table's own stamped retrieve→rescore
    /// oversample default (market parity with Qdrant's per-query
    /// oversampling knob). `None` defers to the table's stamped default,
    /// falling back to the deployment's current oversample default only for
    /// a pre-migration table with no stamped column. Irrelevant for a `F32`
    /// table (single-stage, no rescore).
    pub oversample: Option<usize>,
}

/// A flattened column-source fine-tune submission. Every knob the submit
/// carries — the spec's own fields, the hyperparameter block, and the
/// data-parallel rank count — is a field here, the same way [`SearchRequest`]
/// flattens a search: a transport serialises the whole request, and the
/// embedded session and the data-plane client build the identical
/// submission from one shape rather than from two parameter lists that can
/// drift apart.
///
/// The count sits beside `config` rather than inside it because that is where
/// it sits in both of the shapes this maps onto: the engine's
/// `TrainingCommon { base_model, config, world_size }` and
/// `jammi.v1.job.SubmitJobRequest`'s tag 9. Folding it into
/// [`FineTuneConfig`] would represent one value in two places.
pub struct FineTuneRequest {
    /// Source whose rows supply the training pairs.
    pub source: String,
    /// Base encoder to adapt (`local:<path>`, an HF repo id, or a fine-tuned
    /// id).
    pub base_model: String,
    /// Content columns the pairs are read from.
    pub columns: Vec<String>,
    /// Fine-tuning method.
    pub method: FineTuneMethod,
    /// Which task head the adapted model serves.
    pub task: ModelTask,
    /// Hyperparameters. `None` applies the engine's defaults.
    pub config: Option<FineTuneConfig>,
    /// Data-parallel rank count. `None` is UNSET and leaves the engine's
    /// default of a single rank; `Some(n)` requests `n` ranks.
    ///
    /// `Option<NonZeroU32>` rather than a bare `u32` so the two meanings a `0`
    /// would carry — "I did not choose" and "I chose zero ranks" — cannot be
    /// confused, and a zero-rank job is unrepresentable at this edge rather
    /// than refused after it is built. On the wire the count is an
    /// implicit-presence `uint32` where `0` IS the unset value, so `None`
    /// encodes as `0` and a request that never sets the count is byte-for-byte
    /// what a caller sent before the field existed. A count the deployment
    /// cannot serve is refused at submit with a typed error, never clamped.
    ///
    /// `Some(1)` and `None` are the SAME wire value — both encode as `0` — and
    /// the engine treats both as one rank: a single-rank job has one encoding,
    /// whichever client and whichever surface submits it (the embedded session
    /// resolves both to the engine default, and the Python client's
    /// `_wire_world_size` likewise writes the field only above one). The
    /// distinction survives in this struct for the caller that wants to say
    /// "one rank" out loud; it does not reach the wire.
    pub world_size: Option<NonZeroU32>,
}

/// Identifier of a fine-tune job. Returned in place of an in-process job handle
/// so the job is addressable across a transport boundary; poll it with a
/// fine-tune-status verb.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineTuneJobId(pub String);
