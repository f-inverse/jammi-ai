//! The two shapes of a named entity: the [`Span`] a gold annotation states
//! and strict matching compares, and the [`Entity`] a model predicts — a span
//! with the text it covers and the model's confidence in it.

use serde::{Deserialize, Serialize};

/// A named-entity span: a label over a byte range of a text. What a gold
/// annotation states, and what entity-level NER matching compares: a
/// prediction is right when its span equals a gold span exactly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// Entity type without B-/I- prefix (e.g. "PER", "ORG", "LOC").
    pub label: String,
    /// Inclusive byte start in the original text.
    pub start: usize,
    /// Exclusive byte end in the original text.
    pub end: usize,
}

/// A named entity a model predicted.
///
/// `Deserialize` is derived so the eval runner can round-trip the JSON
/// payload the NER inference adapter writes to its `entities` column back
/// into typed values for metric computation and per-record reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Entity type without B-/I- prefix (e.g. "PER", "ORG", "LOC").
    pub label: String,
    /// Inclusive byte start in the original text.
    pub start: usize,
    /// Exclusive byte end in the original text.
    pub end: usize,
    /// The substring spanned by `[start, end)`.
    pub text: String,
    /// Mean confidence across the entity's words, each word's being its
    /// first subtoken's softmax probability for the word's label.
    pub confidence: f32,
}

impl Entity {
    /// The span this entity covers — what strict matching compares.
    pub fn span(&self) -> Span {
        Span {
            label: self.label.clone(),
            start: self.start,
            end: self.end,
        }
    }
}
