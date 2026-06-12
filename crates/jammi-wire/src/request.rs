//! Request vocabulary the consumer surface shares with the wire.
//!
//! These owned, serialisable request shapes are what a session verb takes
//! (`Modality` / `QueryInput` / `SearchRequest`) and the addressable id a
//! fine-tune job returns (`FineTuneJobId`). They hold no engine state, so they
//! live on the wire substrate: the embedded session and the data-plane client
//! both build verbs from them, and the gRPC converters map them on/off the wire.

/// Which embedding tower an embeddings / encode-query call targets. Unifies the
/// three per-modality engine verbs (`text`/`image`/`audio`) into one parameter
/// so the consumer surface carries one embedding verb, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

/// Identifier of a fine-tune job. Returned in place of an in-process job handle
/// so the job is addressable across a transport boundary; poll it with a
/// fine-tune-status verb.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineTuneJobId(pub String);
