//! `EmbeddingService` proto↔domain conversions.
//!
//! Maps the embedding compute wire enums/messages onto the engine's
//! [`Modality`], [`QueryInput`], and the result-table record. Modality and
//! input are validated at decode: an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`.
//!
//! The source-registration and model-introspection conversions
//! (`SourceType` / `SourceConnection` / `FileFormat` / `SourceDescriptor` /
//! `Model`) live with the control-plane catalog wire surface
//! ([`super::catalog`]); only the compute verbs' shapes are here.

use jammi_db::catalog::result_repo::ResultTableRecord;
use tonic::Status;

use crate::wire::proto::embedding as pb;
use crate::{Modality, QueryInput};

/// Map the proto [`Modality`] onto the engine's [`Modality`]. An unspecified
/// modality is rejected — a request that names no tower is a client error, not
/// a silent default.
impl TryFrom<pb::Modality> for Modality {
    type Error = Status;

    fn try_from(modality: pb::Modality) -> Result<Self, Self::Error> {
        match modality {
            pb::Modality::Text => Ok(Modality::Text),
            pb::Modality::Image => Ok(Modality::Image),
            pb::Modality::Audio => Ok(Modality::Audio),
            pb::Modality::Unspecified => {
                Err(Status::invalid_argument("modality must be specified"))
            }
        }
    }
}

/// Decode the raw enum discriminant a request carries. An out-of-range value is
/// rejected with the same message an `UNSPECIFIED` modality is — the request
/// names no valid tower either way.
impl TryFrom<i32> for Modality {
    type Error = Status;

    fn try_from(modality: i32) -> Result<Self, Self::Error> {
        match pb::Modality::try_from(modality) {
            Ok(m) => Modality::try_from(m),
            Err(_) => Err(Status::invalid_argument("modality must be specified")),
        }
    }
}

/// The proto query oneof paired with its resolved [`Modality`]. The oneof alone
/// does not say which tower it feeds, so decode takes both: TEXT requires
/// `text`, IMAGE/AUDIO require `data` (raw bytes); a missing oneof or a mismatch
/// is a client error.
pub struct ProtoQueryInput {
    pub input: Option<pb::encode_query_request::Input>,
    pub modality: Modality,
}

impl TryFrom<ProtoQueryInput> for QueryInput {
    type Error = Status;

    fn try_from(value: ProtoQueryInput) -> Result<Self, Self::Error> {
        use pb::encode_query_request::Input as ProtoInput;
        let input = value
            .input
            .ok_or_else(|| Status::invalid_argument("input (text or data) is required"))?;
        match (value.modality, input) {
            (Modality::Text, ProtoInput::Text(text)) => {
                if text.is_empty() {
                    return Err(Status::invalid_argument("text is required"));
                }
                Ok(QueryInput::Text(text))
            }
            (Modality::Image | Modality::Audio, ProtoInput::Data(data)) => {
                if data.is_empty() {
                    return Err(Status::invalid_argument("data is required"));
                }
                Ok(QueryInput::Bytes(data))
            }
            (Modality::Text, ProtoInput::Data(_)) => Err(Status::invalid_argument(
                "TEXT modality requires text input, got data",
            )),
            (Modality::Image | Modality::Audio, ProtoInput::Text(_)) => Err(
                Status::invalid_argument("IMAGE/AUDIO modality requires data input, got text"),
            ),
        }
    }
}

/// Encode the engine's result-table record into the wire `ResultTable`. The
/// engine's optional `dimensions` is flattened to `0` for a non-embedding /
/// unset result, `row_count` widens to the wire's `u64`, and `task` rides the
/// shared [`super::model_task_to_proto`] task vocabulary.
///
/// The wire `ResultTable` carries its own `task` (the embedding tower), so the
/// reconstruction recovers it faithfully from the message itself — never from a
/// modality threaded in out of band, never a guess.
impl From<ResultTableRecord> for pb::ResultTable {
    fn from(record: ResultTableRecord) -> Self {
        pb::ResultTable {
            table_name: record.table_name,
            source_id: record.source_id,
            model_id: record.model_id,
            dimensions: record.dimensions.unwrap_or(0),
            row_count: record.row_count as u64,
            status: record.status,
            task: super::model_task_to_proto(record.task) as i32,
        }
    }
}

/// Reconstruct the engine's result-table record from the wire `ResultTable` a
/// `GenerateEmbeddings` or `DescribeSource` response carries.
///
/// The wire message is the client-observable projection: it carries the fields
/// a client needs to locate and query the persisted embedding table
/// (`table_name`, `source_id`, `model_id`, `dimensions`, `row_count`, `status`,
/// `task`). The engine's server-internal bookkeeping — storage/index paths,
/// timestamps, the originating columns — is intentionally not on the wire, so
/// the reconstruction leaves those at their "not carried" values (`String::new`
/// / `None`). A remote consumer keys off the same fields a local one reads back;
/// the dropped fields are server-side state, not result data. The message is
/// self-describing in `task`, so an out-of-range/unspecified task is the
/// faithful `invalid_argument` the shared decoder builds.
pub fn result_table_from_proto(table: pb::ResultTable) -> Result<ResultTableRecord, Status> {
    let task = super::model_task_from_proto(table.task)?;
    Ok(ResultTableRecord {
        table_name: table.table_name,
        source_id: table.source_id,
        model_id: table.model_id,
        task,
        // `kind`/`derived_from` are server-internal bookkeeping, not carried on
        // the wire — `GenerateEmbeddings` only ever returns a model output, so
        // the reconstruction defaults to that kind.
        kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
        derived_from: None,
        parquet_path: String::new(),
        index_path: None,
        dimensions: (table.dimensions != 0).then_some(table.dimensions),
        distance_metric: String::new(),
        row_count: table.row_count as usize,
        status: table.status,
        key_column: None,
        text_columns: None,
        created_at: String::new(),
        completed_at: None,
    })
}
