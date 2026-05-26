//! Named-entity-recognition primitives: a BIO span decoder, entity-level
//! NER metrics (strict matching on label + boundaries), and the single
//! `Entity` type that serves both roles.

pub mod decoding;
pub mod metrics;
pub mod types;

pub use decoding::decode_bio_spans;
pub use metrics::{NerMetrics, TypeMetrics};
pub use types::Entity;
