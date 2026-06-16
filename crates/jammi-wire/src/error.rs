//! Faithful engine-error transport: the `JammiError` ↔ [`pb::JammiErrorDetail`]
//! conversions and the [`Status`] detail attach/extract helpers that make a
//! remote transport reconstruct the *exact* [`JammiError`] the in-process path
//! returns, not a lossy gRPC-code-category guess.
//!
//! The two directions are a pair:
//!
//! * encode — [`From<&JammiError> for pb::JammiErrorDetail`]; the server's
//!   central error mapping calls it once for every `JammiError` (DRY) and
//!   attaches the detail to the [`Status`] via [`attach_error_detail`], which
//!   wraps it in the canonical gRPC rich-error envelope ([`pb::RpcStatus`], the
//!   `google.rpc.Status` shape) so the `grpc-status-details-bin` trailer is
//!   spec-compliant and a gRPC-web client reads the real `code` + typed detail.
//! * decode — [`From<pb::JammiErrorDetail> for JammiError`]; a remote client
//!   reads the detail back off a [`Status`] via [`error_from_status`], which
//!   unwraps the envelope's `Any`, and reconstructs the precise variant.
//!
//! Both impls are orphan-rule-clean: `pb::JammiErrorDetail` is a local
//! generated type, so `From` to/from the foreign `JammiError` is allowed
//! without a newtype.
//!
//! The contract's fidelity boundary is precise, and faithfulness is a property
//! of the error type — not of any one verb surface — so the mapping is complete
//! over `JammiError`: every owned-shape variant (the String- and struct-carrying
//! ones — `Source`, `Model`, `Inference`, `Catalog`, `Schema`, `Config`, `Eval`,
//! `Tenant`, `FineTune`, `Gpu`, `Backend`, `ChannelAssembly`) reconstructs
//! exactly, field for field. So do [`JammiError::MutableTable`] and
//! [`JammiError::ChannelCatalog`]: their inner errors ([`MutableTableError`],
//! [`ChannelCatalogError`]) are engine-owned and every variant's fields
//! reconstruct, so they carry structured details ([`pb::MutableTableErrorDetail`]
//! — which nests an engine-owned [`pb::BackendErrorDetail`] for its `Backend`
//! arm — and [`pb::ChannelCatalogErrorDetail`]) and cross the wire faithfully —
//! never a fold. The only variants that fold into `other`
//! are the genuinely-foreign `#[from]` ones whose inner error cannot cross a
//! process boundary (`Io`, `BackendDriver`, `Toml`, `Json`, `DataFusion`,
//! `Trigger`, `Storage`, and the lone [`BackendError::Sqlx`] nested under
//! `MutableTable`): they reconstruct as [`JammiError::Other`] (or, for `Sqlx`, a
//! backend-detail string arm) carrying the faithful `Display` string — the
//! genuine limit, not a lossy guess.

use jammi_db::catalog::backend::BackendError;
use jammi_db::catalog::channel_repo::{ChannelCatalogError, ChannelColumnType};
use jammi_db::error::JammiError;
use jammi_db::store::mutable::{MutableTableError, MutableTableId};
use jammi_db::trigger::TriggerError;
use jammi_db::{AuditError, TenantId};
use prost::bytes::Bytes;
use prost::{Message, Name};
use prost_types::Any;
use tonic::{Code, Status};
use uuid::Uuid;

use crate::proto::error as pb;

/// Build the structured wire detail for an engine error. One match over every
/// `JammiError` variant so the server's `map_engine_error` emits a faithful
/// detail for the whole enum from one place.
impl From<&JammiError> for pb::JammiErrorDetail {
    fn from(err: &JammiError) -> Self {
        use pb::jammi_error_detail::Variant;
        let variant = match err {
            JammiError::Source { source_id, message } => Variant::Source(pb::SourceError {
                source_id: source_id.clone(),
                message: message.clone(),
            }),
            JammiError::Model { model_id, message } => Variant::Model(pb::ModelError {
                model_id: model_id.clone(),
                message: message.clone(),
            }),
            JammiError::ModelNotFound { model_id } => {
                Variant::ModelNotFound(pb::ModelNotFoundError {
                    model_id: model_id.clone(),
                })
            }
            JammiError::ModelRetired { model_id } => Variant::ModelRetired(pb::ModelRetiredError {
                model_id: model_id.clone(),
            }),
            JammiError::ModelReferenced {
                model_id,
                referenced_by,
            } => Variant::ModelReferenced(pb::ModelReferencedError {
                model_id: model_id.clone(),
                referenced_by: referenced_by.clone(),
            }),
            JammiError::Inference(message) => Variant::Inference(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Catalog(message) => Variant::Catalog(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Schema {
                table,
                column,
                expected,
                actual,
            } => Variant::Schema(pb::SchemaError {
                table: table.clone(),
                column: column.clone(),
                expected: expected.clone(),
                actual: actual.clone(),
            }),
            JammiError::Config(message) => Variant::Config(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Eval(message) => Variant::Eval(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Tenant(message) => Variant::Tenant(pb::StringError {
                message: message.clone(),
            }),
            JammiError::FineTune(message) => Variant::FineTune(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Gpu(message) => Variant::Gpu(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Backend(message) => Variant::Backend(pb::StringError {
                message: message.clone(),
            }),
            JammiError::ChannelCatalog(e) => Variant::ChannelCatalog(e.into()),
            JammiError::ChannelAssembly(message) => Variant::ChannelAssembly(pb::StringError {
                message: message.clone(),
            }),
            JammiError::MutableTable(e) => Variant::MutableTable(e.into()),
            // The fold reaches ONLY the genuinely-foreign `#[from]` variants
            // (`Io`, `BackendDriver`, `Toml`, `Json`, `DataFusion`, `Trigger`,
            // `Storage`) and the existing `Other`: every owned-shape variant —
            // and `MutableTable`, whose engine-owned inner error reconstructs
            // faithfully above — has an explicit arm. A foreign source error
            // cannot cross a process boundary, so it reconstructs as
            // `JammiError::Other` carrying the faithful `Display` string — the
            // genuine fidelity limit. `to_string()` is that string.
            other => Variant::Other(pb::StringError {
                message: other.to_string(),
            }),
        };
        pb::JammiErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the engine error from the wire detail. The inverse of the encode
/// above; a detail with no `variant` set (an older / corrupt payload) maps to
/// [`JammiError::Other`] carrying the empty marker so decode is total.
impl From<pb::JammiErrorDetail> for JammiError {
    fn from(detail: pb::JammiErrorDetail) -> Self {
        use pb::jammi_error_detail::Variant;
        match detail.variant {
            Some(Variant::Source(e)) => JammiError::Source {
                source_id: e.source_id,
                message: e.message,
            },
            Some(Variant::Model(e)) => JammiError::Model {
                model_id: e.model_id,
                message: e.message,
            },
            Some(Variant::ModelNotFound(e)) => JammiError::ModelNotFound {
                model_id: e.model_id,
            },
            Some(Variant::ModelRetired(e)) => JammiError::ModelRetired {
                model_id: e.model_id,
            },
            Some(Variant::ModelReferenced(e)) => JammiError::ModelReferenced {
                model_id: e.model_id,
                referenced_by: e.referenced_by,
            },
            Some(Variant::Inference(e)) => JammiError::Inference(e.message),
            Some(Variant::Catalog(e)) => JammiError::Catalog(e.message),
            Some(Variant::Schema(e)) => JammiError::Schema {
                table: e.table,
                column: e.column,
                expected: e.expected,
                actual: e.actual,
            },
            Some(Variant::Config(e)) => JammiError::Config(e.message),
            Some(Variant::Eval(e)) => JammiError::Eval(e.message),
            Some(Variant::Tenant(e)) => JammiError::Tenant(e.message),
            Some(Variant::FineTune(e)) => JammiError::FineTune(e.message),
            Some(Variant::Gpu(e)) => JammiError::Gpu(e.message),
            Some(Variant::Backend(e)) => JammiError::Backend(e.message),
            Some(Variant::ChannelCatalog(e)) => JammiError::ChannelCatalog(e.into()),
            Some(Variant::ChannelAssembly(e)) => JammiError::ChannelAssembly(e.message),
            Some(Variant::MutableTable(e)) => JammiError::MutableTable(e.into()),
            Some(Variant::Other(e)) => JammiError::Other(e.message),
            None => JammiError::Other(String::new()),
        }
    }
}

/// Encode the engine-owned [`MutableTableError`] into its structured wire
/// detail. Every variant carries exactly the fields it holds; the `Backend` arm
/// recurses into [`pb::BackendErrorDetail`]. No arm folds — the inner taxonomy
/// is engine-owned and fully reconstructable (the one genuinely-foreign leaf,
/// `BackendError::Sqlx`, is handled inside the backend encode as a `Display`
/// string).
impl From<&MutableTableError> for pb::MutableTableErrorDetail {
    fn from(err: &MutableTableError) -> Self {
        use pb::mutable_table_error_detail::Variant;
        let variant = match err {
            MutableTableError::InvalidId(m) => Variant::InvalidId(m.clone()),
            MutableTableError::Schema(m) => Variant::Schema(m.clone()),
            MutableTableError::MissingPrimaryKey(m) => Variant::MissingPrimaryKey(m.clone()),
            MutableTableError::ReservedColumn(m) => Variant::ReservedColumn(m.clone()),
            MutableTableError::NotFound(id) => Variant::NotFound(id.to_string()),
            MutableTableError::AlreadyExists(id) => Variant::AlreadyExists(id.to_string()),
            MutableTableError::NoOrderColumn => Variant::NoOrderColumn(true),
            MutableTableError::Backend(e) => Variant::Backend(e.into()),
        };
        pb::MutableTableErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`MutableTableError`] from its wire detail — the inverse of
/// the encode above. The id-carrying arms (`not_found`/`already_exists`)
/// re-validate the id string through [`MutableTableId::new`]; a forged id that
/// fails validation surfaces as `InvalidId` carrying the offending string,
/// which is exactly the variant the engine itself produces for such a string,
/// so decode stays total without a panic. A detail with no variant (an older or
/// corrupt payload) reconstructs as `Schema(String::new())` — the empty marker
/// kept inside the engine-owned taxonomy rather than escaping to `Other`.
impl From<pb::MutableTableErrorDetail> for MutableTableError {
    fn from(detail: pb::MutableTableErrorDetail) -> Self {
        use pb::mutable_table_error_detail::Variant;
        let reconstruct_id = |s: String, wrap: fn(MutableTableId) -> MutableTableError| {
            match MutableTableId::new(&s) {
                Ok(id) => wrap(id),
                Err(_) => MutableTableError::InvalidId(s),
            }
        };
        match detail.variant {
            Some(Variant::InvalidId(m)) => MutableTableError::InvalidId(m),
            Some(Variant::Schema(m)) => MutableTableError::Schema(m),
            Some(Variant::MissingPrimaryKey(m)) => MutableTableError::MissingPrimaryKey(m),
            Some(Variant::ReservedColumn(m)) => MutableTableError::ReservedColumn(m),
            Some(Variant::NotFound(s)) => reconstruct_id(s, MutableTableError::NotFound),
            Some(Variant::AlreadyExists(s)) => reconstruct_id(s, MutableTableError::AlreadyExists),
            Some(Variant::NoOrderColumn(_)) => MutableTableError::NoOrderColumn,
            Some(Variant::Backend(e)) => MutableTableError::Backend(e.into()),
            None => MutableTableError::Schema(String::new()),
        }
    }
}

/// Encode the engine-owned [`ChannelCatalogError`] into its structured wire
/// detail. The two struct variants carry per-field messages (channel + column +
/// canonical PascalCase type tokens) so `to_string()` rebuilds the exact Display
/// the in-process error produces; no arm folds — every field reconstructs.
impl From<&ChannelCatalogError> for pb::ChannelCatalogErrorDetail {
    fn from(err: &ChannelCatalogError) -> Self {
        use pb::channel_catalog_error_detail::Variant;
        let variant = match err {
            ChannelCatalogError::AlreadyExists(c) => Variant::AlreadyExists(c.clone()),
            ChannelCatalogError::NotRegistered(c) => Variant::NotRegistered(c.clone()),
            ChannelCatalogError::ColumnAlreadyDeclared {
                channel,
                column,
                ty,
            } => Variant::ColumnAlreadyDeclared(pb::ColumnAlreadyDeclared {
                channel: channel.clone(),
                column: column.clone(),
                ty: ty.as_str().to_string(),
            }),
            ChannelCatalogError::ColumnConflict {
                channel,
                column,
                existing,
                requested,
            } => Variant::ColumnConflict(pb::ColumnConflict {
                channel: channel.clone(),
                column: column.clone(),
                existing: existing.as_str().to_string(),
                requested: requested.as_str().to_string(),
            }),
            ChannelCatalogError::InvalidId(m) => Variant::InvalidId(m.clone()),
            ChannelCatalogError::InvalidColumnType(m) => Variant::InvalidColumnType(m.clone()),
        };
        pb::ChannelCatalogErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`ChannelCatalogError`] from its wire detail — the inverse of
/// the encode above. The struct variants re-parse the canonical PascalCase type
/// token; a forged token that does not parse reconstructs as
/// `InvalidColumnType` carrying the offending string — exactly the variant the
/// engine produces for an unknown token, so decode stays total. A detail with no
/// variant set reconstructs as an empty `NotRegistered` — kept inside the
/// channel-catalog taxonomy rather than escaping to `Other`.
impl From<pb::ChannelCatalogErrorDetail> for ChannelCatalogError {
    fn from(detail: pb::ChannelCatalogErrorDetail) -> Self {
        use pb::channel_catalog_error_detail::Variant;
        // A forged or corrupt type token cannot reconstruct a `ChannelColumnType`;
        // it surfaces as `InvalidColumnType` (carrying the token) — the same
        // variant the engine yields for an unknown token, keeping decode total.
        let parse_ty = |token: String| ChannelColumnType::from_sql_str(&token).map_err(|_| token);
        match detail.variant {
            Some(Variant::AlreadyExists(c)) => ChannelCatalogError::AlreadyExists(c),
            Some(Variant::NotRegistered(c)) => ChannelCatalogError::NotRegistered(c),
            Some(Variant::ColumnAlreadyDeclared(d)) => match parse_ty(d.ty) {
                Ok(ty) => ChannelCatalogError::ColumnAlreadyDeclared {
                    channel: d.channel,
                    column: d.column,
                    ty,
                },
                Err(token) => ChannelCatalogError::InvalidColumnType(token),
            },
            Some(Variant::ColumnConflict(d)) => match (parse_ty(d.existing), parse_ty(d.requested))
            {
                (Ok(existing), Ok(requested)) => ChannelCatalogError::ColumnConflict {
                    channel: d.channel,
                    column: d.column,
                    existing,
                    requested,
                },
                (Err(token), _) | (_, Err(token)) => ChannelCatalogError::InvalidColumnType(token),
            },
            Some(Variant::InvalidId(m)) => ChannelCatalogError::InvalidId(m),
            Some(Variant::InvalidColumnType(m)) => ChannelCatalogError::InvalidColumnType(m),
            None => ChannelCatalogError::NotRegistered(String::new()),
        }
    }
}

/// Encode the engine-owned [`BackendError`] into its structured wire detail.
/// Every variant but `Sqlx` reconstructs field-for-field; `Sqlx` wraps a raw
/// `sqlx::Error` that cannot cross a process boundary, so it folds to its
/// faithful `Display` string — the genuine fidelity limit, mirroring how the
/// top-level detail folds its own foreign `#[from]` variants.
impl From<&BackendError> for pb::BackendErrorDetail {
    fn from(err: &BackendError) -> Self {
        use pb::backend_error_detail::Variant;
        let variant = match err {
            BackendError::Execution(m) => Variant::Execution(m.clone()),
            BackendError::Constraint { table, detail } => {
                Variant::Constraint(pb::ConstraintViolation {
                    table: table.clone(),
                    detail: detail.clone(),
                })
            }
            BackendError::Unavailable(m) => Variant::Unavailable(m.clone()),
            BackendError::Retry(m) => Variant::Retry(m.clone()),
            BackendError::Migration(m) => Variant::Migration(m.clone()),
            BackendError::TypeConversion { column, detail } => {
                Variant::TypeConversion(pb::TypeConversion {
                    column: column.clone(),
                    detail: detail.clone(),
                })
            }
            BackendError::TenantMismatch {
                table,
                expected,
                got,
            } => Variant::TenantMismatch(pb::TenantMismatch {
                table: table.clone(),
                expected: expected.map(|t| t.to_string()).unwrap_or_default(),
                got: got.map(|t| t.to_string()).unwrap_or_default(),
            }),
            BackendError::Sqlx(e) => Variant::Sqlx(e.to_string()),
        };
        pb::BackendErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`BackendError`] from its wire detail — the inverse of the
/// encode above. The `tenant_mismatch` arm re-parses the UUID strings (empty ==
/// `None`); a non-empty string that fails to parse reconstructs as `None`, the
/// only total option for a forged payload, since the variant's faithful path
/// always carries a valid UUID. `Sqlx` reconstructs as `Execution` carrying the
/// original `Display` string — the raw `sqlx::Error` cannot be rebuilt, so the
/// faithful message lands in the nearest backend-owned string arm rather than
/// escaping the taxonomy. A missing variant reconstructs as an empty
/// `Execution`.
impl From<pb::BackendErrorDetail> for BackendError {
    fn from(detail: pb::BackendErrorDetail) -> Self {
        use pb::backend_error_detail::Variant;
        let parse_tenant = |s: String| -> Option<TenantId> {
            if s.is_empty() {
                None
            } else {
                s.parse().ok()
            }
        };
        match detail.variant {
            Some(Variant::Execution(m)) => BackendError::Execution(m),
            Some(Variant::Constraint(c)) => BackendError::Constraint {
                table: c.table,
                detail: c.detail,
            },
            Some(Variant::Unavailable(m)) => BackendError::Unavailable(m),
            Some(Variant::Retry(m)) => BackendError::Retry(m),
            Some(Variant::Migration(m)) => BackendError::Migration(m),
            Some(Variant::TypeConversion(t)) => BackendError::TypeConversion {
                column: t.column,
                detail: t.detail,
            },
            Some(Variant::TenantMismatch(t)) => BackendError::TenantMismatch {
                table: t.table,
                expected: parse_tenant(t.expected),
                got: parse_tenant(t.got),
            },
            Some(Variant::Sqlx(m)) => BackendError::Execution(m),
            None => BackendError::Execution(String::new()),
        }
    }
}

/// Encode an optional tenant onto its wire string form — a UUID string, or the
/// empty string for `None`. Shared by the tenant-carrying error arms; the
/// inverse [`parse_optional_tenant`] reads it back.
fn tenant_to_wire(t: Option<TenantId>) -> String {
    t.map(|t| t.to_string()).unwrap_or_default()
}

/// Parse a wire tenant string back to `Option<TenantId>` (empty == `None`). A
/// non-empty string that fails to parse reconstructs as `None` — the only total
/// option for a forged payload, since the faithful path always carries a valid
/// UUID or the empty string.
fn parse_optional_tenant(s: String) -> Option<TenantId> {
    if s.is_empty() {
        None
    } else {
        s.parse().ok()
    }
}

/// Build the structured wire detail for a [`TriggerError`]. One match over every
/// variant so the server's `map_trigger_error` emits a faithful detail for the
/// whole enum from one place.
///
/// No arm folds: every owned-shape variant has its own message, and the two
/// `#[from]` variants (`BackingTable`, `Backend`) wrap engine-owned errors that
/// reconstruct faithfully through the shared [`pb::MutableTableErrorDetail`] /
/// [`pb::BackendErrorDetail`] taxonomies. The whole `TriggerError` enum crosses
/// the wire without loss.
impl From<&TriggerError> for pb::TriggerErrorDetail {
    fn from(err: &TriggerError) -> Self {
        use pb::trigger_error_detail::Variant;
        let variant = match err {
            TriggerError::TopicNotFound(m) => Variant::TopicNotFound(m.clone()),
            TriggerError::SchemaConflict { topic, detail } => {
                Variant::SchemaConflict(pb::SchemaConflict {
                    topic: topic.clone(),
                    detail: detail.clone(),
                })
            }
            TriggerError::UnsupportedSchemaType { column, data_type } => {
                Variant::UnsupportedSchemaType(pb::UnsupportedSchemaType {
                    column: column.clone(),
                    data_type: data_type.clone(),
                })
            }
            TriggerError::BatchSchemaMismatch(m) => Variant::BatchSchemaMismatch(m.clone()),
            TriggerError::PublishTenantMismatch {
                topic,
                topic_tenant,
                publish_tenant,
            } => Variant::PublishTenantMismatch(pb::PublishTenantMismatch {
                topic: topic.clone(),
                topic_tenant: tenant_to_wire(*topic_tenant),
                publish_tenant: tenant_to_wire(*publish_tenant),
            }),
            TriggerError::PredicateParse(m) => Variant::PredicateParse(m.clone()),
            TriggerError::PredicateEval(m) => Variant::PredicateEval(m.clone()),
            TriggerError::PredicateUnsupported(m) => Variant::PredicateUnsupported(m.clone()),
            TriggerError::OffsetEvicted(n) => Variant::OffsetEvicted(*n),
            TriggerError::BackingTable(e) => Variant::BackingTable(e.into()),
            TriggerError::Backend(e) => Variant::Backend(e.into()),
            TriggerError::Driver(m) => Variant::Driver(m.clone()),
            TriggerError::Catalog(m) => Variant::Catalog(m.clone()),
        };
        pb::TriggerErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`TriggerError`] from its wire detail — the inverse of the
/// encode above. The nested engine-owned details (`backing_table`, `backend`)
/// reconstruct through their own `From<pb>` impls. A detail with no variant set
/// (an older / corrupt payload) reconstructs as `TriggerError::Catalog` carrying
/// the empty marker — kept inside the trigger taxonomy rather than escaping.
impl From<pb::TriggerErrorDetail> for TriggerError {
    fn from(detail: pb::TriggerErrorDetail) -> Self {
        use pb::trigger_error_detail::Variant;
        match detail.variant {
            Some(Variant::TopicNotFound(m)) => TriggerError::TopicNotFound(m),
            Some(Variant::SchemaConflict(c)) => TriggerError::SchemaConflict {
                topic: c.topic,
                detail: c.detail,
            },
            Some(Variant::UnsupportedSchemaType(u)) => TriggerError::UnsupportedSchemaType {
                column: u.column,
                data_type: u.data_type,
            },
            Some(Variant::BatchSchemaMismatch(m)) => TriggerError::BatchSchemaMismatch(m),
            Some(Variant::PublishTenantMismatch(p)) => TriggerError::PublishTenantMismatch {
                topic: p.topic,
                topic_tenant: parse_optional_tenant(p.topic_tenant),
                publish_tenant: parse_optional_tenant(p.publish_tenant),
            },
            Some(Variant::PredicateParse(m)) => TriggerError::PredicateParse(m),
            Some(Variant::PredicateEval(m)) => TriggerError::PredicateEval(m),
            Some(Variant::PredicateUnsupported(m)) => TriggerError::PredicateUnsupported(m),
            Some(Variant::OffsetEvicted(n)) => TriggerError::OffsetEvicted(n),
            Some(Variant::BackingTable(e)) => TriggerError::BackingTable(e.into()),
            Some(Variant::Backend(e)) => TriggerError::Backend(e.into()),
            Some(Variant::Driver(m)) => TriggerError::Driver(m),
            Some(Variant::Catalog(m)) => TriggerError::Catalog(m),
            None => TriggerError::Catalog(String::new()),
        }
    }
}

/// Build the structured wire detail for an [`AuditError`]. One match over every
/// variant so the server's `map_audit_error` emits a faithful detail for the
/// whole enum from one place.
///
/// Every owned-shape variant has its own message. The lone fold is `Serde`:
/// `AuditError::Serde(#[from] serde_json::Error)` wraps a foreign error that
/// cannot cross a process boundary, so it carries its faithful `Display` string
/// — the genuine fidelity limit, mirroring the foreign-source fold in
/// `JammiErrorDetail`.
impl From<&AuditError> for pb::AuditErrorDetail {
    fn from(err: &AuditError) -> Self {
        use pb::audit_error_detail::Variant;
        let variant = match err {
            AuditError::LengthMismatch { ids, scores } => {
                Variant::LengthMismatch(pb::LengthMismatch {
                    ids: *ids as u64,
                    scores: *scores as u64,
                })
            }
            AuditError::LineageTooLarge { actual, max } => {
                Variant::LineageTooLarge(pb::LineageTooLarge {
                    actual: *actual as u64,
                    max: *max as u64,
                })
            }
            AuditError::NoTenantBinding => Variant::NoTenantBinding(true),
            AuditError::SignatureMismatch(id) => Variant::SignatureMismatch(id.to_string()),
            AuditError::MasterKey(m) => Variant::MasterKey(m.clone()),
            // The one genuinely-foreign arm: a `serde_json::Error` cannot be
            // rebuilt across the wire, so its faithful `Display` string lands in
            // the dedicated `serde` arm. `to_string()` is that string.
            AuditError::Serde(e) => Variant::Serde(e.to_string()),
            AuditError::Storage(m) => Variant::Storage(m.clone()),
            AuditError::Broker(m) => Variant::Broker(m.clone()),
        };
        pb::AuditErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`AuditError`] from its wire detail — the inverse of the
/// encode above. The `signature_mismatch` arm re-parses the query-id UUID; a
/// forged string that fails to parse reconstructs as the nil UUID (total, since
/// the faithful path always carries a valid UUID). The `serde` arm reconstructs
/// as `AuditError::Storage` carrying the original `Display` string — the raw
/// `serde_json::Error` cannot be rebuilt, so the faithful message lands in the
/// nearest audit-owned string arm rather than escaping the taxonomy. A detail
/// with no variant reconstructs as an empty `Storage`.
impl From<pb::AuditErrorDetail> for AuditError {
    fn from(detail: pb::AuditErrorDetail) -> Self {
        use pb::audit_error_detail::Variant;
        match detail.variant {
            Some(Variant::LengthMismatch(l)) => AuditError::LengthMismatch {
                ids: l.ids as usize,
                scores: l.scores as usize,
            },
            Some(Variant::LineageTooLarge(l)) => AuditError::LineageTooLarge {
                actual: l.actual as usize,
                max: l.max as usize,
            },
            Some(Variant::NoTenantBinding(_)) => AuditError::NoTenantBinding,
            Some(Variant::SignatureMismatch(id)) => {
                AuditError::SignatureMismatch(Uuid::parse_str(&id).unwrap_or_default())
            }
            Some(Variant::MasterKey(m)) => AuditError::MasterKey(m),
            Some(Variant::Serde(m)) => AuditError::Storage(m),
            Some(Variant::Storage(m)) => AuditError::Storage(m),
            Some(Variant::Broker(m)) => AuditError::Broker(m),
            None => AuditError::Storage(String::new()),
        }
    }
}

/// Wrap a typed engine detail in the canonical gRPC rich-error envelope and
/// attach it to a [`Status`]. `tonic::Status::with_details` writes the bytes it
/// is given verbatim into the `grpc-status-details-bin` trailer; the gRPC
/// rich-error contract — which every gRPC-web client (Connect-ES, grpc-web) and
/// the canonical spec assume — requires those bytes to be a serialized
/// `google.rpc.Status` whose `code` mirrors the `grpc-status` trailer and whose
/// `details` is a list of `Any`. So this is the single emission point that
/// builds the [`pb::RpcStatus`] envelope, packs `detail` as its lone `Any`, and
/// hands the encoded envelope to `with_details`. A client that decodes the
/// envelope reconstructs the precise typed variant from the `Any`; a client that
/// only reads `code` + `message` (or `grpc-status` + `grpc-message`) still gets
/// the idiomatic gRPC surface.
///
/// `M: Name` supplies the `Any.type_url` (`type.googleapis.com/<full.name>`,
/// generated by `build.rs`'s `enable_type_names`) that [`extract_detail`] keys
/// on to read the same detail back.
fn attach_detail<M: Message + Name>(code: Code, message: String, detail: &M) -> Status {
    // A typed engine detail always encodes into a well-formed `Any`; the only
    // error path is an oversized message that cannot happen for these bounded
    // error payloads, so a failure there folds to an envelope carrying just the
    // code + message — never a panic on an error response.
    let details = Any::from_msg(detail)
        .map(|any| vec![any])
        .unwrap_or_default();
    let envelope = pb::RpcStatus {
        code: code as i32,
        message: message.clone(),
        details,
    };
    Status::with_details(code, message, Bytes::from(envelope.encode_to_vec()))
}

/// Read the typed engine detail a server attached via [`attach_detail`] back out
/// of a [`Status`]. Decodes the [`pb::RpcStatus`] envelope from the status
/// `details` bytes and unpacks the `Any` whose `type_url` matches `M`. Returns
/// `None` when the status carries no detail, the envelope is undecodable, or it
/// holds no `Any` of type `M` — every caller then folds to the status `message`,
/// the faithful fallback for a status that by construction carries no such
/// detail.
fn extract_detail<M: Message + Name + Default>(status: &Status) -> Option<M> {
    let details = status.details();
    if details.is_empty() {
        return None;
    }
    let envelope = pb::RpcStatus::decode(details).ok()?;
    let type_url = M::type_url();
    envelope
        .details
        .iter()
        .find(|any| any.type_url == type_url)
        .and_then(|any| any.to_msg::<M>().ok())
}

/// Attach a faithful [`pb::JammiErrorDetail`] for `err` to a [`Status`] of the
/// given `code` and `message`, inside the canonical gRPC rich-error envelope
/// (a `google.rpc.Status` carrying the detail as an `Any`) so a decoding client
/// reconstructs the precise variant while the `code` + `message` keep the
/// idiomatic gRPC surface.
pub fn attach_error_detail(code: Code, message: String, err: &JammiError) -> Status {
    attach_detail(code, message, &pb::JammiErrorDetail::from(err))
}

/// Reconstruct the engine error a server attached to a [`Status`]. When the
/// status carries a decodable [`pb::JammiErrorDetail`] the exact variant is
/// rebuilt; otherwise (no detail, or undecodable bytes) the status `message`
/// stands in as [`JammiError::Other`] — the faithful fallback, since a status
/// without a Jammi detail is by construction not an engine `JammiError`.
pub fn error_from_status(status: &Status) -> JammiError {
    match extract_detail::<pb::JammiErrorDetail>(status) {
        Some(detail) => JammiError::from(detail),
        None => JammiError::Other(status.message().to_string()),
    }
}

/// Attach a faithful [`pb::TriggerErrorDetail`] for `err` to a [`Status`]. The
/// trigger verbs surface [`TriggerError`] directly (not `JammiError`), so the
/// server's `map_trigger_error` is the single emission point for this detail —
/// the trigger analogue of [`attach_error_detail`].
pub fn attach_trigger_detail(code: Code, message: String, err: &TriggerError) -> Status {
    attach_detail(code, message, &pb::TriggerErrorDetail::from(err))
}

/// Reconstruct the [`TriggerError`] a server attached to a [`Status`]. When the
/// status carries a decodable [`pb::TriggerErrorDetail`] the exact variant is
/// rebuilt; otherwise (no detail, or undecodable bytes) the status `message`
/// stands in as [`TriggerError::Driver`] — the faithful fallback for a status
/// that by construction carries no trigger detail. This is also the path a
/// mid-stream subscribe failure takes: a terminal `tonic::Status` reconstructs
/// to the faithful variant, never a gRPC-code-category guess.
pub fn trigger_error_from_status(status: &Status) -> TriggerError {
    match extract_detail::<pb::TriggerErrorDetail>(status) {
        Some(detail) => TriggerError::from(detail),
        None => TriggerError::Driver(status.message().to_string()),
    }
}

/// Attach a faithful [`pb::AuditErrorDetail`] for `err` to a [`Status`]. The
/// audit verbs surface [`AuditError`] directly, so the server's
/// `map_audit_error` is the single emission point for this detail — the audit
/// analogue of [`attach_error_detail`].
pub fn attach_audit_detail(code: Code, message: String, err: &AuditError) -> Status {
    attach_detail(code, message, &pb::AuditErrorDetail::from(err))
}

/// Reconstruct the [`AuditError`] a server attached to a [`Status`]. When the
/// status carries a decodable [`pb::AuditErrorDetail`] the exact variant is
/// rebuilt; otherwise (no detail, or undecodable bytes) the status `message`
/// stands in as [`AuditError::Storage`] — the faithful fallback for a status
/// that by construction carries no audit detail.
pub fn audit_error_from_status(status: &Status) -> AuditError {
    match extract_detail::<pb::AuditErrorDetail>(status) {
        Some(detail) => AuditError::from(detail),
        None => AuditError::Storage(status.message().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Send an error through the full Status round-trip a real server/client
    /// pair uses (`attach_error_detail` → encode → decode → `error_from_status`).
    fn round_trip(err: &JammiError) -> JammiError {
        let status = attach_error_detail(Code::Internal, err.to_string(), err);
        error_from_status(&status)
    }

    /// Every owned-shape variant — the String- and struct-carrying ones — must
    /// reconstruct to the IDENTICAL variant and fields after a wire round-trip.
    /// This is the completeness proof: the contract is faithful over the whole
    /// owned-shape surface of `JammiError`, not just the verbs one stage wires.
    #[test]
    fn every_owned_shape_variant_round_trips_to_itself() {
        let owned = [
            JammiError::Config("missing api key".into()),
            JammiError::Catalog("no embedding table for source".into()),
            JammiError::Source {
                source_id: "patents".into(),
                message: "scan failed".into(),
            },
            JammiError::Model {
                model_id: "local:/models/tiny_bert".into(),
                message: "Model directory does not exist".into(),
            },
            JammiError::ModelNotFound {
                model_id: "local:/models/tiny_bert".into(),
            },
            JammiError::ModelRetired {
                model_id: "local:/models/tiny_bert".into(),
            },
            JammiError::ModelReferenced {
                model_id: "local:/models/tiny_bert".into(),
                referenced_by: vec!["result_tables".into(), "training_jobs.base_model_id".into()],
            },
            JammiError::Inference("encode_query forward: shape mismatch".into()),
            JammiError::FineTune("checkpoint epoch 3 diverged".into()),
            JammiError::Eval("golden NER fixture row 7 mismatch".into()),
            JammiError::Gpu("no CUDA device visible".into()),
            JammiError::Backend("vLLM returned HTTP 503".into()),
            JammiError::Tenant("nil UUID is not a valid tenant".into()),
            JammiError::ChannelAssembly(
                "batch 0: channel 'vector' column 'similarity' has dtype Int32".into(),
            ),
            JammiError::Schema {
                table: "patents_embeddings".into(),
                column: "vector".into(),
                expected: "FixedSizeList<Float32>".into(),
                actual: "missing".into(),
            },
            JammiError::Other("an error with no more specific shape".into()),
        ];
        for err in &owned {
            let back = round_trip(err);
            assert_eq!(
                std::mem::discriminant(&back),
                std::mem::discriminant(err),
                "owned-shape variant must reconstruct as itself: {err:?} -> {back:?}"
            );
            assert_eq!(
                back.to_string(),
                err.to_string(),
                "owned-shape variant must reconstruct its fields faithfully: {err:?} -> {back:?}"
            );
        }
    }

    /// The genuinely-foreign `#[from]` variants carry an inner error that cannot
    /// cross a process boundary, so they reconstruct as `JammiError::Other`
    /// carrying the faithful `Display` string — the genuine fidelity limit. `Io`
    /// is the representative case; the fold is identical for the rest
    /// (`BackendDriver`, `Toml`, `Json`, `DataFusion`, `Trigger`, `Storage`).
    /// `MutableTable` is deliberately NOT in this set — it reconstructs
    /// faithfully (see `mutable_table_variant_round_trips_faithfully`).
    #[test]
    fn foreign_source_variant_folds_to_other_with_faithful_display() {
        let io = JammiError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "model.safetensors not found",
        ));
        let display = io.to_string();
        match round_trip(&io) {
            JammiError::Other(message) => assert_eq!(
                message, display,
                "the foreign-source fold carries the faithful Display string"
            ),
            other => panic!("a foreign-source variant must fold to Other, got {other:?}"),
        }
    }

    /// `JammiError::MutableTable` is engine-owned and reconstructs faithfully —
    /// it must NOT fold to `Other`. Every `MutableTableError` variant (including
    /// the id-carrying `NotFound`/`AlreadyExists`, the payload-free
    /// `NoOrderColumn`, and the nested engine-owned `Backend`) round-trips to
    /// the identical variant and `Display` after the full Status round-trip. The
    /// lone genuinely-foreign leaf, `BackendError::Sqlx`, is exercised in
    /// `backend_sqlx_leaf_folds_to_faithful_string`.
    #[test]
    fn mutable_table_variant_round_trips_faithfully() {
        use jammi_db::catalog::backend::BackendError;
        use jammi_db::store::mutable::{MutableTableError, MutableTableId};

        let table_id = MutableTableId::new("patents_dim").expect("valid id");
        let cases = [
            MutableTableError::InvalidId(
                "table name '_jammi_audit' is reserved for the Jammi substrate".into(),
            ),
            MutableTableError::Schema("order_column 'seq' not in schema".into()),
            MutableTableError::MissingPrimaryKey("row_key".into()),
            MutableTableError::ReservedColumn("tenant_id".into()),
            MutableTableError::NotFound(table_id.clone()),
            MutableTableError::AlreadyExists(table_id.clone()),
            MutableTableError::NoOrderColumn,
            MutableTableError::Backend(BackendError::Constraint {
                table: "patents_dim".into(),
                detail: "duplicate key value violates unique constraint".into(),
            }),
            MutableTableError::Backend(BackendError::TenantMismatch {
                table: "patents_dim".into(),
                expected: Some(
                    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
                        .parse()
                        .expect("uuid"),
                ),
                got: None,
            }),
        ];

        for inner in cases {
            let err = JammiError::MutableTable(inner);
            let back = round_trip(&err);
            match (&err, &back) {
                (JammiError::MutableTable(_), JammiError::MutableTable(_)) => {}
                other => panic!(
                    "MutableTable must reconstruct as itself, never fold to Other: {other:?}"
                ),
            }
            assert_eq!(
                back.to_string(),
                err.to_string(),
                "MutableTable variant must reconstruct its fields faithfully: {err:?} -> {back:?}"
            );
        }
    }

    /// `JammiError::ChannelCatalog` is engine-owned and reconstructs faithfully —
    /// it must NOT fold to `Other`. Every `ChannelCatalogError` variant (the two
    /// string-carrying ids and the two struct variants carrying canonical
    /// PascalCase type tokens) round-trips to the identical variant and `Display`
    /// after the full Status round-trip. The contiguous "cannot redeclare as
    /// <type>" Display of `ColumnConflict` is load-bearing for the CLI / cookbook
    /// / db it-tests, so the `Display` equality below pins it across the wire.
    #[test]
    fn channel_catalog_variant_round_trips_faithfully() {
        let cases = [
            ChannelCatalogError::AlreadyExists("scored_by".into()),
            ChannelCatalogError::NotRegistered("vector".into()),
            ChannelCatalogError::ColumnAlreadyDeclared {
                channel: "scored_by".into(),
                column: "ranker".into(),
                ty: ChannelColumnType::Utf8,
            },
            ChannelCatalogError::ColumnConflict {
                channel: "scored_by".into(),
                column: "ranker".into(),
                existing: ChannelColumnType::Utf8,
                requested: ChannelColumnType::Int32,
            },
            ChannelCatalogError::InvalidId("invalid channel id 'Bad': must be [a-z0-9_]".into()),
            ChannelCatalogError::InvalidColumnType("Decimal".into()),
        ];
        for inner in cases {
            let err = JammiError::ChannelCatalog(inner);
            let back = round_trip(&err);
            match (&err, &back) {
                (JammiError::ChannelCatalog(_), JammiError::ChannelCatalog(_)) => {}
                other => panic!(
                    "ChannelCatalog must reconstruct as itself, never fold to Other: {other:?}"
                ),
            }
            assert_eq!(
                back.to_string(),
                err.to_string(),
                "ChannelCatalog variant must reconstruct its fields faithfully: {err:?} -> {back:?}"
            );
        }
        // The contiguous redeclare-conflict Display crosses the wire intact.
        let conflict = JammiError::ChannelCatalog(ChannelCatalogError::ColumnConflict {
            channel: "scored_by".into(),
            column: "ranker".into(),
            existing: ChannelColumnType::Utf8,
            requested: ChannelColumnType::Int32,
        });
        assert!(round_trip(&conflict)
            .to_string()
            .contains("cannot redeclare as Int32"));
    }

    /// Round-trip a [`TriggerError`] through the full Status path
    /// (`attach_trigger_detail` → encode → decode → `trigger_error_from_status`).
    fn round_trip_trigger(err: &TriggerError) -> TriggerError {
        let status = attach_trigger_detail(Code::Internal, err.to_string(), err);
        trigger_error_from_status(&status)
    }

    /// Every `TriggerError` variant — including the two engine-owned `#[from]`
    /// nests (`BackingTable`, `Backend`) — must reconstruct to the IDENTICAL
    /// variant and `Display` after a wire round-trip. No `TriggerError` variant
    /// folds to a lossy string; this is the completeness proof for the error
    /// type the topics/subscribe surface returns.
    #[test]
    fn every_trigger_variant_round_trips_to_itself() {
        let tenant_a: TenantId = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
            .parse()
            .expect("uuid");
        let cases = [
            TriggerError::TopicNotFound("events.changes".into()),
            TriggerError::SchemaConflict {
                topic: "events.changes".into(),
                detail: "column 'kind' type changed Utf8 -> Int64".into(),
            },
            TriggerError::UnsupportedSchemaType {
                column: "payload".into(),
                data_type: "Struct".into(),
            },
            TriggerError::BatchSchemaMismatch("publish has 3 columns, topic has 2".into()),
            TriggerError::PublishTenantMismatch {
                topic: "events.changes".into(),
                topic_tenant: Some(tenant_a),
                publish_tenant: None,
            },
            TriggerError::PredicateParse("unexpected token at column 4".into()),
            TriggerError::PredicateEval("predicate did not produce Boolean array".into()),
            TriggerError::PredicateUnsupported("aggregate functions are not allowed".into()),
            TriggerError::OffsetEvicted(42),
            TriggerError::BackingTable(MutableTableError::AlreadyExists(
                MutableTableId::new("__topic_abc").expect("valid id"),
            )),
            TriggerError::Backend(BackendError::Constraint {
                table: "topics".into(),
                detail: "duplicate key value violates unique constraint".into(),
            }),
            TriggerError::Driver("nats: connection closed".into()),
            TriggerError::Catalog("topic_id parse: invalid".into()),
        ];
        for err in &cases {
            let back = round_trip_trigger(err);
            assert_eq!(
                std::mem::discriminant(&back),
                std::mem::discriminant(err),
                "TriggerError variant must reconstruct as itself: {err:?} -> {back:?}"
            );
            assert_eq!(
                back.to_string(),
                err.to_string(),
                "TriggerError variant must reconstruct its fields faithfully: {err:?} -> {back:?}"
            );
        }
    }

    /// Round-trip an [`AuditError`] through the full Status path.
    fn round_trip_audit(err: &AuditError) -> AuditError {
        let status = attach_audit_detail(Code::Internal, err.to_string(), err);
        audit_error_from_status(&status)
    }

    /// Every owned-shape `AuditError` variant must reconstruct to the IDENTICAL
    /// variant and `Display` after a wire round-trip. `Serde` is deliberately
    /// NOT here — it is the one genuinely-foreign leaf and folds (see
    /// `audit_serde_leaf_folds_to_faithful_string`).
    #[test]
    fn every_owned_shape_audit_variant_round_trips_to_itself() {
        let cases = [
            AuditError::LengthMismatch { ids: 3, scores: 2 },
            AuditError::LineageTooLarge {
                actual: 70_000,
                max: 65_536,
            },
            AuditError::NoTenantBinding,
            AuditError::SignatureMismatch(
                "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
                    .parse()
                    .expect("uuid"),
            ),
            AuditError::MasterKey("expected 64 hex chars, got 10".into()),
            AuditError::Storage("mutable-table registry unavailable".into()),
            AuditError::Broker("audit topic publish failed".into()),
        ];
        for err in &cases {
            let back = round_trip_audit(err);
            assert_eq!(
                std::mem::discriminant(&back),
                std::mem::discriminant(err),
                "AuditError variant must reconstruct as itself: {err:?} -> {back:?}"
            );
            assert_eq!(
                back.to_string(),
                err.to_string(),
                "AuditError variant must reconstruct its fields faithfully: {err:?} -> {back:?}"
            );
        }
    }

    /// `AuditError::Serde` wraps a foreign `serde_json::Error` that cannot cross
    /// a process boundary, so it folds — carrying the inner error's faithful
    /// `Display` string in the nearest audit-owned arm (`Storage`), mirroring how
    /// `BackendError::Sqlx` carries its inner `Display`. The inner message is
    /// preserved verbatim (without the `AuditError::Serde` wrapper's "serde: "
    /// prefix, which the typed arm would otherwise re-add on reconstruction).
    #[test]
    fn audit_serde_leaf_folds_to_faithful_string() {
        let serde_err = serde_json::from_str::<serde_json::Value>("{not json")
            .expect_err("malformed JSON must fail to parse");
        let inner_display = serde_err.to_string();
        let err = AuditError::Serde(serde_err);
        match round_trip_audit(&err) {
            AuditError::Storage(message) => assert_eq!(
                message, inner_display,
                "the foreign serde leaf carries the inner error's faithful Display string"
            ),
            other => panic!("a foreign serde leaf must fold to Storage, got {other:?}"),
        }
    }
}
