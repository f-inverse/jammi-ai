//! `InferenceService` proto↔domain conversions.
//!
//! An inference verb returns `Vec<RecordBatch>`; the wire carries them as one
//! Arrow IPC stream in an `ArrowBatch` (the Flight-IPC pairing the trigger and
//! inference surfaces share). An empty result has no schema to encode, so it
//! round-trips as an empty `ArrowBatch`.

use arrow::record_batch::RecordBatch;
use tonic::Status;

use crate::encode_ipc_stream;
use crate::proto::trigger::ArrowBatch;

/// Encode the engine's inference result rows into one `ArrowBatch`. Carries the
/// rows as a single self-describing IPC stream keyed on the first batch's
/// schema; an empty result (empty source) has no schema to encode, so it
/// becomes an empty `ArrowBatch`.
pub fn infer_result_to_proto(batches: Vec<RecordBatch>) -> Result<ArrowBatch, Status> {
    match batches.first() {
        Some(first) => {
            let body = encode_ipc_stream(&first.schema(), &batches)?;
            Ok(ArrowBatch {
                data_header: Vec::new(),
                data_body: body,
                app_metadata: Vec::new(),
            })
        }
        None => Ok(ArrowBatch::default()),
    }
}
