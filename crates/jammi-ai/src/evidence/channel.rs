//! Per-batch contribution from one channel to the merged result.

use arrow::array::ArrayRef;

use jammi_db::ChannelId;

/// One channel's column values for one `RecordBatch`.
///
/// `columns` aligns 1:1 with the channel's declared columns
/// (`ChannelSpec::columns`) by index, and every array has the same length
/// as the batch the contribution attaches to. The merger validates both
/// invariants and rejects mismatches with a typed `EvidenceChannel`
/// error.
#[derive(Debug, Clone)]
pub struct ChannelContribution {
    pub channel: ChannelId,
    pub columns: Vec<ArrayRef>,
}

impl ChannelContribution {
    /// Convenience constructor for the common single-column case (e.g.
    /// `vector` contributing only `similarity`).
    pub fn single(channel: ChannelId, column: ArrayRef) -> Self {
        Self {
            channel,
            columns: vec![column],
        }
    }
}
