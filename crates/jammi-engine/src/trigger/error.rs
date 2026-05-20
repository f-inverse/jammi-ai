//! Error taxonomy for the trigger-stream primitive.

use thiserror::Error;

use crate::catalog::backend::BackendError;
use crate::store::mutable::definition::MutableTableError;

#[derive(Debug, Error)]
pub enum TriggerError {
    #[error("topic not found: {0}")]
    TopicNotFound(String),

    #[error("topic schema conflict on {topic}: {detail}")]
    SchemaConflict { topic: String, detail: String },

    #[error("batch schema does not match topic schema: {0}")]
    BatchSchemaMismatch(String),

    #[error("predicate parse failure: {0}")]
    PredicateParse(String),

    #[error("predicate evaluation failure: {0}")]
    PredicateEval(String),

    #[error("predicate uses an unsupported construct: {0}")]
    PredicateUnsupported(String),

    #[error("offset {0} is older than broker retention")]
    OffsetEvicted(u64),

    #[error("backing table unavailable: {0}")]
    BackingTable(#[from] MutableTableError),

    #[error("catalog backend error: {0}")]
    Backend(#[from] BackendError),

    #[error("broker driver error: {0}")]
    Driver(String),

    #[error("topic catalog corruption: {0}")]
    Catalog(String),
}
