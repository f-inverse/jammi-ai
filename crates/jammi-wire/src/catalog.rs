//! `CatalogService` proto↔domain conversions for the control-plane surface
//! that names sources, models, and topics.
//!
//! Sources / models: maps the wire `SourceKind` / `FileFormat` /
//! `SourceConnection` / `SourceDescriptor` / `Model` onto the engine's
//! [`SourceType`], [`FileFormat`], [`SourceConnection`], [`SourceDescriptor`],
//! and [`ModelRecord`]. Only the URL + format cross the wire on a connection —
//! cloud credentials are server-side, so the connection decode fills the rest
//! from `Default`.
//!
//! Topics: maps the wire `Topic` listing message onto the engine's
//! [`TopicDefinition`]. The schema rides as a schema-only Arrow IPC stream (the
//! same framing `RegisterTopicRequest.schema` uses).
//!
//! The engine types are foreign (they live in `jammi-db`), so the enum decodes
//! are free functions taking the **proto** message (a local generated type)
//! rather than orphan-rule-blocked `TryFrom<i32>` impls; the connection /
//! result-table message conversions are `From`/`TryFrom` impls because the
//! proto message is local. A `SourceDescriptor`'s embedded result tables reuse
//! the embedding service's self-describing `ResultTable` shape — the same shape
//! `GenerateEmbeddings` returns — so there is one source-of-truth for the
//! embedding numbers, not a parallel one.

use std::collections::BTreeMap;
use std::str::FromStr;

use jammi_db::catalog::model_repo::ModelRecord;
use jammi_db::catalog::source_repo::SourceDescriptor;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::TopicDefinition;
use jammi_db::TenantId;
use tonic::Status;

use crate::proto::catalog as pb;
use crate::proto::embedding as embedding_pb;
use crate::{decode_ipc_schema, encode_ipc_stream, result_table_from_proto};

// === sources / models =====================================================

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
/// the inverse of [`source_type_from_proto`], for the the remote client
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
/// the remote client send side. Only the URL + format cross the wire
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

/// Encode the engine's [`SourceDescriptor`] into the wire message: the registry
/// identity (`source_id` / `kind` / `status`) plus each embedding result table
/// in the same self-describing [`embedding_pb::ResultTable`] shape
/// `GenerateEmbeddings` returns — one source-of-truth for the embedding numbers,
/// not a parallel one.
impl From<SourceDescriptor> for pb::SourceDescriptor {
    fn from(descriptor: SourceDescriptor) -> Self {
        pb::SourceDescriptor {
            source_id: descriptor.source_id,
            kind: source_type_to_proto(descriptor.source_type) as i32,
            status: descriptor.status,
            result_tables: descriptor
                .result_tables
                .into_iter()
                .map(embedding_pb::ResultTable::from)
                .collect(),
        }
    }
}

/// Reconstruct the engine's [`SourceDescriptor`] from the wire message — the
/// inverse of the encode above, for the the remote client receive side.
/// The kind decodes through the shared [`source_type_from_proto`] (an
/// unspecified/unknown backend is the faithful `invalid_argument`), and each
/// result table through [`result_table_from_proto`], so a remote
/// `describe_source` rebuilds the same descriptor a local one returns.
pub fn source_descriptor_from_proto(
    descriptor: pb::SourceDescriptor,
) -> Result<SourceDescriptor, Status> {
    Ok(SourceDescriptor {
        source_id: descriptor.source_id,
        source_type: source_type_from_proto(descriptor.kind)?,
        status: descriptor.status,
        result_tables: descriptor
            .result_tables
            .into_iter()
            .map(result_table_from_proto)
            .collect::<Result<_, Status>>()?,
    })
}

/// Encode the engine's [`ModelRecord`] into the wire `Model` — the client-
/// observable projection a `ListModels` / `DescribeModel` response carries. Only
/// the registry-identity fields cross the wire (`model_id` / `backend` / `task`
/// / `status`); the version counter, derived-from lineage, artifact path, config
/// blob, and registration timestamp are server-internal bookkeeping a list
/// consumer does not key off, mirroring how [`ResultTableRecord`] projects onto
/// [`embedding_pb::ResultTable`]. The `task` rides the shared
/// [`super::model_task_to_proto`] vocabulary.
///
/// [`ResultTableRecord`]: jammi_db::catalog::result_repo::ResultTableRecord
pub fn model_to_proto(record: &ModelRecord) -> pb::Model {
    pb::Model {
        model_id: record.model_id.clone(),
        backend: record.backend.clone(),
        task: super::model_task_to_proto(record.task) as i32,
        status: record.status.clone(),
        // The projection exposes only whether the model is promoted, derived
        // from the presence of the server-side `promoted_at` flag; the raw
        // timestamp stays server-side.
        promoted: record.promoted_at.is_some(),
    }
}

/// Reconstruct the engine's [`ModelRecord`] from the wire `Model` — the inverse
/// of [`model_to_proto`], for the the remote client receive side. The
/// fields not carried on the wire are server-internal bookkeeping, so they
/// reconstruct at their "not carried" values (`version = 1` default, `None`,
/// `String::new` — including `catalog_pk`, a server-side row key a list
/// consumer never resolves against), exactly as [`result_table_from_proto`]
/// does for a result table. The message is self-describing in `task`, so an out-of-range /
/// unspecified task surfaces as the faithful `invalid_argument` the shared
/// decoder builds.
pub fn model_from_proto(model: pb::Model) -> Result<ModelRecord, Status> {
    let task = super::model_task_from_proto(model.task)?;
    Ok(ModelRecord {
        model_id: model.model_id,
        catalog_pk: String::new(),
        version: 1,
        model_type: String::new(),
        base_model_id: None,
        backend: model.backend,
        task,
        artifact_path: None,
        config_json: None,
        status: model.status,
        created_at: String::new(),
        // The wire carries only the `promoted` boolean (not the raw timestamp),
        // so the flag reconstructs to a present-but-empty marker when promoted —
        // faithful to the boolean projection, the same way the other
        // not-carried fields reconstruct to their defaults.
        promoted_at: model.promoted.then(String::new),
    })
}

// === topics ===============================================================

/// Encode a [`TopicDefinition`] onto the wire `Topic` a `ListTopics` page
/// carries — the send side of the materialized listing. The schema rides as a
/// schema-only Arrow IPC stream (the same framing `RegisterTopicRequest.schema`
/// uses). Fallible only on the schema encode.
pub fn topic_to_proto(topic: &TopicDefinition) -> Result<pb::Topic, Status> {
    let schema = encode_ipc_stream(&topic.schema, &[])?;
    Ok(pb::Topic {
        topic_id: topic.id.to_string(),
        name: topic.name.clone(),
        schema,
        tenant_id: topic.tenant.map(|t| t.to_string()).unwrap_or_default(),
        broker_metadata: topic.broker_metadata.clone().into_iter().collect(),
    })
}

/// Reconstruct the [`TopicDefinition`] from the wire `Topic` — the inverse of
/// [`topic_to_proto`], for the the remote client `list_topics` read side.
/// Fallible: the id and (non-empty) tenant are re-parsed and the schema is
/// decoded from its IPC framing, so a corrupt page surfaces as a `Status` rather
/// than a fabricated definition.
pub fn topic_from_proto(wire: pb::Topic) -> Result<TopicDefinition, Status> {
    let id = TopicId::from_str(&wire.topic_id)
        .map_err(|e| Status::invalid_argument(format!("invalid topic_id: {e}")))?;
    let schema = decode_ipc_schema(&wire.schema)?;
    let tenant = if wire.tenant_id.is_empty() {
        None
    } else {
        Some(
            TenantId::from_str(&wire.tenant_id)
                .map_err(|e| Status::invalid_argument(format!("invalid tenant id: {e}")))?,
        )
    };
    let broker_metadata: BTreeMap<String, String> = wire.broker_metadata.into_iter().collect();
    Ok(TopicDefinition {
        id,
        name: wire.name,
        schema,
        tenant,
        broker_metadata,
    })
}
