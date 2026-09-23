//! This crate's error taxonomy. The operators implement DataFusion traits
//! whose methods return `datafusion::error::Result`, so an operator
//! converts a typed refusal into a [`DataFusionError`] at the point of
//! return ([`Error::into_df`]) — as an `External` error carrying this type,
//! which a consumer restores by walking the error's source chain.

use datafusion::error::DataFusionError;

/// This crate's error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A refusal the operators or adapters raise over the shape of their
    /// input or a model's output.
    #[error("inference error: {0}")]
    Inference(String),
    /// A keyed input's key column has nulls. Raised over the whole input
    /// with the exact total, before any row reaches a model: every row
    /// needs a key.
    #[error("key column `{column}` has {null_count} null value(s); every row needs a key")]
    InvalidKey {
        /// The key column.
        column: String,
        /// How many of its values are null.
        null_count: u64,
    },
    /// A task spelling no [`ModelTask`](crate::ModelTask) has.
    #[error(
        "unknown model task '{0}'; expected one of: text_embedding, image_embedding, \
         audio_embedding, classification, ner, regression"
    )]
    UnknownTask(String),
    /// A device-kind spelling no [`ComputeDeviceKind`](crate::ComputeDeviceKind) has.
    #[error("unknown device kind '{0}'; expected one of: cpu, cuda, metal")]
    UnknownDeviceKind(String),
    /// A wire buffer failed to decode (truncated, malformed, or naming a
    /// value no field admits).
    #[error("malformed inference operator buffer: {0}")]
    Decode(String),
    /// The model runtime's own failure — binding a model, costing or
    /// preparing rows, admitting or running a forward — carried as the
    /// runtime raised it, so a consumer that knows the runtime's error type
    /// finds it in the source chain.
    #[error(transparent)]
    Runtime(Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// This crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// A runtime's failure, carried as it was raised.
    pub fn runtime(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Runtime(Box::new(error))
    }

    /// This error as the `External` DataFusion error an operator returns.
    pub fn into_df(self) -> DataFusionError {
        DataFusionError::External(Box::new(self))
    }

    /// The typed error an operator raised, found in `error`'s source chain
    /// — how a consumer restores a refusal from the [`DataFusionError`] a
    /// plan's execution surfaced it as. `None` when the chain carries none.
    pub fn found_in<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a Self> {
        std::iter::successors(Some(error), |err| err.source())
            .find_map(|err| err.downcast_ref::<Self>())
    }
}

impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        error.into_df()
    }
}
