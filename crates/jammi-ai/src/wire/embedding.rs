//! `EmbeddingService` proto↔domain conversions.
//!
//! Maps the embedding wire enums/messages onto the engine's [`Modality`],
//! [`QueryInput`], [`SourceType`], [`FileFormat`], [`SourceConnection`], and the
//! result-table record. Modality and input are validated at decode: an
//! unspecified modality and a text/bytes-vs-modality mismatch are rejected with
//! `invalid_argument`. Only the URL + format cross the wire on a connection —
//! cloud credentials are server-side, so [`SourceConnection`] decode fills the
//! rest from `Default`.
//!
//! The engine's [`SourceType`] / [`SourceConnection`] are foreign types (they
//! live in `jammi-db`), so their decodes take the **proto** message as the
//! `From` side (a local generated type) rather than a raw `i32` — that is the
//! orphan-rule-clean shape and the handler passes the proto enum directly. The
//! send side (the remote `add_source` client) gets the inverse encodes:
//! [`source_type_to_proto`] for the kind and `From<SourceConnection>` for the
//! connection message, carrying the same URL + format the decode reads back.

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::ModelTask;
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

/// Map the proto `SourceKind` discriminant onto the engine's [`SourceType`].
/// An unspecified or unknown kind is rejected — a registration with no backend
/// is a client error, not a silent default.
///
/// The engine's `SourceType` lives in `jammi-db`, so this is a free function
/// taking the raw `i32` rather than `impl TryFrom<i32> for SourceType` (both
/// types would be foreign — orphan-rule-blocked); it mirrors
/// [`super::model_task_from_proto`].
pub fn source_type_from_proto(kind: i32) -> Result<SourceType, Status> {
    match pb::SourceKind::try_from(kind) {
        Ok(pb::SourceKind::File) => Ok(SourceType::File),
        Ok(pb::SourceKind::Postgres) => Ok(SourceType::Postgres),
        Ok(pb::SourceKind::Mysql) => Ok(SourceType::Mysql),
        Ok(pb::SourceKind::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("source_kind must be specified"))
        }
    }
}

/// Encode the engine's [`SourceType`] onto the proto `SourceKind` discriminant —
/// the inverse of [`source_type_from_proto`], for the [`crate::RemoteSession`]
/// send side. Total: every engine source type maps to a concrete wire variant
/// (the engine type has no unspecified state). Mirrors
/// [`super::model_task_to_proto`].
pub fn source_type_to_proto(source_type: SourceType) -> pb::SourceKind {
    match source_type {
        SourceType::File => pb::SourceKind::File,
        SourceType::Postgres => pb::SourceKind::Postgres,
        SourceType::Mysql => pb::SourceKind::Mysql,
    }
}

/// Build the engine's [`SourceConnection`] from the proto message. Only the URL
/// and format are carried on the wire; cloud credentials are server-side, so
/// the rest comes from `Default`.
impl TryFrom<pb::SourceConnection> for SourceConnection {
    type Error = Status;

    fn try_from(conn: pb::SourceConnection) -> Result<Self, Self::Error> {
        let url = if conn.url.is_empty() {
            None
        } else {
            Some(conn.url)
        };
        Ok(SourceConnection {
            url,
            format: file_format_from_proto(conn.format)?,
            ..Default::default()
        })
    }
}

/// Encode the engine's [`SourceConnection`] into the proto message for an
/// `AddSource` request — the inverse of the decode above, for the
/// [`crate::RemoteSession`] send side. Only the URL + format cross the wire
/// (matching what the decode reads back): cloud credentials, file-extension
/// overrides, and driver options are server-side and have no wire field, so the
/// send side does not carry them. A `None` URL encodes as the empty string the
/// decode reads back as `None`; an unset format encodes as
/// `FILE_FORMAT_UNSPECIFIED`, which the decode maps to "let the engine infer".
impl From<SourceConnection> for pb::SourceConnection {
    fn from(conn: SourceConnection) -> Self {
        pb::SourceConnection {
            url: conn.url.unwrap_or_default(),
            format: file_format_to_proto(conn.format) as i32,
        }
    }
}

/// Map the engine's [`FileFormat`] onto the proto enum — the inverse of
/// [`file_format_from_proto`]. An absent format encodes as
/// `FILE_FORMAT_UNSPECIFIED` (the decode reads that back as "let the engine
/// infer").
fn file_format_to_proto(format: Option<FileFormat>) -> pb::FileFormat {
    match format {
        Some(FileFormat::Parquet) => pb::FileFormat::Parquet,
        Some(FileFormat::Csv) => pb::FileFormat::Csv,
        Some(FileFormat::Json) => pb::FileFormat::Json,
        Some(FileFormat::Avro) => pb::FileFormat::Avro,
        None => pb::FileFormat::Unspecified,
    }
}

/// Map the proto [`FileFormat`] enum onto the engine's [`FileFormat`]; an
/// unspecified/unknown format means "let the engine infer" → `None`.
fn file_format_from_proto(format: i32) -> Result<Option<FileFormat>, Status> {
    match pb::FileFormat::try_from(format) {
        Ok(pb::FileFormat::Parquet) => Ok(Some(FileFormat::Parquet)),
        Ok(pb::FileFormat::Csv) => Ok(Some(FileFormat::Csv)),
        Ok(pb::FileFormat::Json) => Ok(Some(FileFormat::Json)),
        Ok(pb::FileFormat::Avro) => Ok(Some(FileFormat::Avro)),
        Ok(pb::FileFormat::Unspecified) | Err(_) => Ok(None),
    }
}

/// Map a [`Modality`] onto the embedding [`ModelTask`] its tower produces. The
/// wire `ResultTable` deliberately omits the engine's `task` (server-internal
/// bookkeeping), but a remote client knows the modality it requested, so the
/// reconstruction recovers `task` faithfully from that — never a guess.
pub fn embedding_task_for(modality: Modality) -> ModelTask {
    match modality {
        Modality::Text => ModelTask::TextEmbedding,
        Modality::Image => ModelTask::ImageEmbedding,
        Modality::Audio => ModelTask::AudioEmbedding,
    }
}

/// Encode the engine's result-table record into the wire `ResultTable`. The
/// engine's optional `dimensions` is flattened to `0` for a non-embedding /
/// unset result, and `row_count` widens to the wire's `u64`.
impl From<ResultTableRecord> for pb::ResultTable {
    fn from(record: ResultTableRecord) -> Self {
        pb::ResultTable {
            table_name: record.table_name,
            source_id: record.source_id,
            model_id: record.model_id,
            dimensions: record.dimensions.unwrap_or(0),
            row_count: record.row_count as u64,
            status: record.status,
        }
    }
}

/// Reconstruct the engine's result-table record from the wire `ResultTable` a
/// `GenerateEmbeddings` response carries, plus the `modality` the client
/// requested (which recovers the omitted `task`).
///
/// The wire message is the client-observable projection: it carries the fields
/// a client needs to locate and query the persisted embedding table
/// (`table_name`, `source_id`, `model_id`, `dimensions`, `row_count`,
/// `status`). The engine's server-internal bookkeeping — storage/index paths,
/// timestamps, the originating columns — is intentionally not on the wire, so
/// the reconstruction leaves those at their "not carried" values (`String::new`
/// / `None`). A remote consumer keys off the same fields a local one reads back
/// from this verb; the dropped fields are server-side state, not result data.
pub fn result_table_from_proto(table: pb::ResultTable, modality: Modality) -> ResultTableRecord {
    ResultTableRecord {
        table_name: table.table_name,
        source_id: table.source_id,
        model_id: table.model_id,
        task: embedding_task_for(modality),
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
    }
}
