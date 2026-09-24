use std::num::NonZeroUsize;

use jammi_numerics::ChunkBudget;

use crate::device::ComputeDeviceKind;
use crate::inference::adapter::DistributionForm;
use crate::source::ModelSource;
use crate::task::ModelTask;

/// What an [`InferenceExec`](crate::InferenceExec) computes: plain data,
/// and everything about the node that crosses a process boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSpec {
    /// The model to run.
    pub source: ModelSource,
    /// The task it performs.
    pub task: ModelTask,
    /// The input columns whose content the model reads.
    pub content_columns: Vec<String>,
    /// The row-identity column, carried to the output as `_row_id`.
    pub key_column: String,
    /// The source id the output is attributed to, carried as `_source`.
    pub source_id: String,
    /// What bounds one forward chunk: its rows and its padded tokens.
    pub chunk: ChunkBudget,
    /// The embedding output width, for a task that produces one.
    pub embedding_dim: Option<usize>,
    /// The served regression head's persisted distribution form.
    pub regression_form: Option<DistributionForm>,
    /// Input columns copied verbatim to the end of every output batch.
    pub passthrough: Vec<String>,
    /// The device kind this node must run on. A submitter placing the plan
    /// onto a kind other than its own names that kind here; nothing
    /// downstream rewrites it.
    pub device_kind: ComputeDeviceKind,
    /// The fan-out: how many partitions forward chunks concurrently.
    pub partitions: NonZeroUsize,
}

/// The order a [`NumberedInputExec`](crate::NumberedInputExec) numbers its
/// rows in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowOrder {
    /// The total order `(key, tie_breakers...)`, each `ASC NULLS LAST`, null
    /// keys refused. The input must carry every named column. The tie
    /// breakers are what make the order total for rows that share a key —
    /// a content hash, a version — so the written bytes are invariant under
    /// permuting the input.
    Keyed {
        /// The row-identity column.
        key_column: String,
        /// The columns that order rows sharing a key, in order.
        tie_breakers: Vec<String>,
    },
    /// The order rows arrive in.
    Arrival,
}
