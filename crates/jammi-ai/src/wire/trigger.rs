//! `TriggerService` proto↔domain conversions for the Arrow-batch payloads.
//!
//! A publish carries exactly one record batch as an `ArrowBatch`
//! (`data_header` + `data_body`, decoded by [`super::decode_ipc_stream`]); a
//! delivered batch rides back the same way. The schema-match check against the
//! topic happens at decode so a malformed publish is a typed client error, not
//! a silent partial write. The topic lifecycle (mint id, repo register / look
//! up) and the `TriggerError` → `Status` mapping stay in the `jammi-server`
//! handler — those are catalog and transport concerns, not conversions.

use arrow::record_batch::RecordBatch;
use jammi_db::trigger::{DeliveredBatch, Offset, TopicDefinition};
use prost_types::Timestamp;
use tonic::Status;

use crate::wire::proto::trigger::{ArrowBatch, SubscribedBatch};
use crate::wire::{decode_ipc_stream, encode_ipc_stream};

/// Decode the single record batch a `Publish` carries, checking it against the
/// topic schema. Per ADR-01 §5.1 the wire pairing is `data_header` +
/// `data_body`; the shared decoder concatenates them and reads the IPC stream. A
/// publish carries exactly one batch — an empty or multi-batch payload is a
/// typed client error, and a schema mismatch is rejected so a malformed publish
/// never partially writes.
pub fn decode_publish_batch(
    wire: &ArrowBatch,
    topic: &TopicDefinition,
) -> Result<RecordBatch, Status> {
    let mut batches = decode_ipc_stream(&wire.data_header, &wire.data_body)?;
    let batch = match batches.len() {
        1 => batches.pop().expect("len checked == 1"),
        0 => {
            return Err(Status::invalid_argument(
                "batch IPC stream contains no batch",
            ))
        }
        n => {
            return Err(Status::invalid_argument(format!(
                "publish carries exactly one batch, got {n}"
            )))
        }
    };
    if batch.schema().as_ref() != topic.schema.as_ref() {
        return Err(Status::invalid_argument(
            "batch schema does not match topic schema",
        ));
    }
    Ok(batch)
}

/// Encode an engine-delivered batch into the wire `SubscribedBatch`. The
/// `StreamWriter` format is a single contiguous IPC stream; surface it as one
/// `data_body` payload with `data_header` empty — the shared decoder
/// concatenates the two anyway, so the wire shape is symmetric.
pub fn encode_delivered_batch(
    schema: &arrow_schema::SchemaRef,
    delivered: DeliveredBatch,
) -> Result<SubscribedBatch, Status> {
    let buf = encode_ipc_stream(schema, std::slice::from_ref(&delivered.batch))?;
    Ok(SubscribedBatch {
        offset: delivered.offset.value(),
        produced_at: Some(to_proto_timestamp(delivered.produced_at)),
        batch: Some(ArrowBatch {
            data_header: Vec::new(),
            data_body: buf,
            app_metadata: Vec::new(),
        }),
    })
}

/// Encode a UTC instant as a protobuf well-known `Timestamp`.
pub fn to_proto_timestamp(dt: chrono::DateTime<chrono::Utc>) -> Timestamp {
    let seconds = dt.timestamp();
    let nanos = dt.timestamp_subsec_nanos() as i32;
    Timestamp { seconds, nanos }
}

/// Decode a protobuf well-known `Timestamp` back to a UTC instant. The inverse
/// of [`to_proto_timestamp`]; an out-of-range timestamp is a corrupt payload, so
/// the decode is fallible. Used by the remote subscribe path to rebuild each
/// delivered batch's `produced_at`.
pub fn from_proto_timestamp(ts: &Timestamp) -> Result<chrono::DateTime<chrono::Utc>, Status> {
    chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
        .ok_or_else(|| Status::invalid_argument("produced_at timestamp out of range"))
}

/// Encode a single topic-payload batch into the wire `ArrowBatch` a `Publish`
/// carries — the send-side inverse of [`decode_publish_batch`]. The batch rides
/// as one self-describing IPC stream in `data_body`; the schema is the batch's
/// own (which a faithful caller has already matched to the topic).
pub fn encode_publish_batch(batch: &RecordBatch) -> Result<ArrowBatch, Status> {
    let buf = encode_ipc_stream(&batch.schema(), std::slice::from_ref(batch))?;
    Ok(ArrowBatch {
        data_header: Vec::new(),
        data_body: buf,
        app_metadata: Vec::new(),
    })
}

/// Decode a streamed wire `SubscribedBatch` back into the engine
/// [`DeliveredBatch`] a local subscription yields — the receive-side inverse of
/// [`encode_delivered_batch`], for the remote subscribe stream.
///
/// A batch carries exactly one record batch (the publish unit); an absent or
/// multi-batch payload is a corrupt frame, surfaced as an `Err` so the stream's
/// item is a faithful failure rather than a fabricated batch. The offset's
/// `committed_at` is set to `produced_at` — the two are the same instant by the
/// broker's contract (see [`DeliveredBatch`]).
pub fn decode_subscribed_batch(wire: SubscribedBatch) -> Result<DeliveredBatch, Status> {
    let arrow = wire
        .batch
        .ok_or_else(|| Status::invalid_argument("subscribed batch missing payload"))?;
    let mut batches = decode_ipc_stream(&arrow.data_header, &arrow.data_body)?;
    let batch = match batches.len() {
        1 => batches.pop().expect("len checked == 1"),
        n => {
            return Err(Status::invalid_argument(format!(
                "subscribed batch carries exactly one record batch, got {n}"
            )))
        }
    };
    let produced_at = wire
        .produced_at
        .as_ref()
        .map(from_proto_timestamp)
        .transpose()?
        .ok_or_else(|| Status::invalid_argument("subscribed batch missing produced_at"))?;
    Ok(DeliveredBatch {
        offset: Offset::new(wire.offset, produced_at),
        produced_at,
        batch,
    })
}
