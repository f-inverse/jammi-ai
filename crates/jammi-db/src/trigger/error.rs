//! Error taxonomy for the trigger-stream primitive.

use thiserror::Error;

use crate::catalog::backend::BackendError;
use crate::store::mutable::definition::MutableTableError;
use crate::tenant::TenantId;

#[derive(Debug, Clone, Error)]
pub enum TriggerError {
    #[error("topic not found: {0}")]
    TopicNotFound(String),

    #[error("topic schema conflict on {topic}: {detail}")]
    SchemaConflict { topic: String, detail: String },

    #[error("unsupported topic schema type for column '{column}': {data_type}")]
    UnsupportedSchemaType { column: String, data_type: String },

    #[error("batch schema does not match topic schema: {0}")]
    BatchSchemaMismatch(String),

    #[error(
        "publish tenant mismatch on topic '{topic}': topic is pinned to {topic_tenant:?}, publish was scoped to {publish_tenant:?}"
    )]
    PublishTenantMismatch {
        topic: String,
        topic_tenant: Option<TenantId>,
        publish_tenant: Option<TenantId>,
    },

    #[error("predicate parse failure: {0}")]
    PredicateParse(String),

    #[error("predicate evaluation failure: {0}")]
    PredicateEval(String),

    #[error("predicate uses an unsupported construct: {0}")]
    PredicateUnsupported(String),

    #[error("backing table unavailable: {0}")]
    BackingTable(#[from] MutableTableError),

    #[error("catalog backend error: {0}")]
    Backend(#[from] BackendError),

    #[error("broker driver error: {0}")]
    Driver(String),

    #[error("topic catalog corruption: {0}")]
    Catalog(String),
}

impl TriggerError {
    /// The condition this error reports ([`crate::error::ErrorClass`]), the
    /// same whether the topic verb ran in process or behind a server.
    pub fn class(&self) -> crate::error::ErrorClass {
        use crate::error::ErrorClass;
        match self {
            Self::TopicNotFound(_) => ErrorClass::NotFound,
            Self::BatchSchemaMismatch(_)
            | Self::UnsupportedSchemaType { .. }
            | Self::PredicateParse(_)
            | Self::PredicateUnsupported(_) => ErrorClass::InvalidArgument,
            // A re-registration whose schema disagrees with the stored one: a
            // conflict against the stored declaration.
            Self::SchemaConflict { .. } => ErrorClass::FailedPrecondition,
            Self::PublishTenantMismatch { .. } => ErrorClass::PermissionDenied,
            Self::Driver(_) => ErrorClass::Unavailable,
            // The topic's own backing table, the catalog underneath it, or a
            // predicate the parser admitted failing on a row: engine faults.
            Self::PredicateEval(_)
            | Self::BackingTable(_)
            | Self::Backend(_)
            | Self::Catalog(_) => ErrorClass::Internal,
        }
    }
}
