//! The single `Entity` type used by both the BIO span decoder and the
//! entity-level NER evaluation metric.

use serde::{Deserialize, Serialize};

/// A decoded or gold named entity span.
///
/// Equality and hashing are defined on `(label, start, end)` only — `text`
/// and `confidence` are intentionally ignored so that a decoder output
/// (which carries both fields populated) can be compared set-wise against
/// a gold record (which constructs with empty `text` and zero
/// `confidence`). The eval metric's strict-matching contract is "did the
/// model predict an entity with this exact label and these exact byte
/// offsets," which the custom equality preserves.
///
/// `Deserialize` is derived so the eval runner can round-trip the JSON
/// payload the NER inference adapter writes to its `entities` column
/// back into typed `Entity` values for metric computation and per-record
/// reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity type without B-/I- prefix (e.g. "PER", "ORG", "LOC").
    pub label: String,
    /// Inclusive byte start in the original text.
    pub start: usize,
    /// Exclusive byte end in the original text.
    pub end: usize,
    /// The substring spanned by `[start, end)`. Populated by the decoder;
    /// empty for gold records constructed for evaluation.
    pub text: String,
    /// Mean softmax confidence across the entity's tokens. Populated by
    /// the decoder; `0.0` for gold records.
    pub confidence: f32,
}

impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.start == other.start && self.end == other.end
    }
}

impl Eq for Entity {}

impl std::hash::Hash for Entity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.start.hash(state);
        self.end.hash(state);
    }
}
