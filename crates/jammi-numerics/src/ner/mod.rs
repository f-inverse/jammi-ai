//! Named-entity-recognition primitives: a word-level BIO decoder, entity-level
//! NER metrics (strict matching on label + boundaries), the [`Span`] a gold
//! annotation states and the [`Entity`] a model predicts.

pub mod decoding;
pub mod metrics;
pub mod types;

pub use decoding::{decode_bio_spans, TokenizedRow};
pub use metrics::{NerMetrics, TypeMetrics};
pub use types::{Entity, Span};
