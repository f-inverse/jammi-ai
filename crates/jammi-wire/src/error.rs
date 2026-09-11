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
//! * decode — [`jammi_error_from_detail`] (a free fn, not a `From` impl: an
//!   unknown-oneof detail needs the enclosing `Status`'s own `message`
//!   threaded in to reconstruct faithfully, and a `From` impl has nowhere to
//!   take that second argument); a remote client reads the detail back off a
//!   [`Status`] via [`error_from_status`], which unwraps the envelope's
//!   `Any` and calls it to reconstruct the precise variant. The same shape —
//!   a free fn taking `(detail, message)` — is used for every nested
//!   engine-owned detail this file decodes ([`mutable_table_error_from_detail`],
//!   `backend_error_from_detail`, [`channel_catalog_error_from_detail`],
//!   [`trigger_error_from_detail`], `audit_error_from_detail`), for the same
//!   reason.
//!
//! The encode impl is orphan-rule-clean: `pb::JammiErrorDetail` is a local
//! generated type, so `From<&JammiError>` for it is allowed without a
//! newtype.
//!
//! The contract's fidelity boundary is precise, and faithfulness is a property
//! of the error type — not of any one verb surface — so the mapping is complete
//! over `JammiError`: every owned-shape variant (the String- and struct-carrying
//! ones — `Source`, `Model`, `ModelNotFound`, `ModelReferenced`, `Inference`,
//! `Catalog`, `Schema`, `Config`, `Eval`, `Tenant`, `FineTune`, `Gpu`, `Backend`,
//! `ChannelAssembly`, `Lexical`, `IncompatibleFormat`, `DependencyCycle`,
//! `NotRecomputable`, `RowGone`, `TenantMismatch`, `LeaseLost`, `CasFailed`,
//! `JobAttemptSuperseded`, `JobCancelled`, `SourceBusy`, `InvalidKey`,
//! `VersionUnavailable`, `NotRefreshable`, `DefinitionDrift`, `NonUniqueKey`)
//! reconstructs exactly,
//! field for field — `tests::every_owned_shape_variant_round_trips_to_itself`
//! is the completeness proof, backed by an exhaustive match with no catch-all
//! so a NEW owned-shape variant fails to compile here until it is listed. So
//! do [`JammiError::MutableTable`] and
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
use jammi_db::error::{JammiError, NonUniqueScan, NotRefreshableReason};
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
            JammiError::RowGone { table } => Variant::RowGone(pb::RowGoneError {
                table: table.clone(),
            }),
            JammiError::TenantMismatch { table } => {
                Variant::TenantMismatch(pb::TenantMismatchError {
                    table: table.clone(),
                })
            }
            JammiError::LeaseLost { table } => Variant::LeaseLost(pb::LeaseLostError {
                table: table.clone(),
            }),
            JammiError::CasFailed { table, status } => Variant::CasFailed(pb::CasFailedError {
                table: table.clone(),
                status: status.clone(),
            }),
            JammiError::SourceBusy { source_id, table } => {
                Variant::SourceBusy(pb::SourceBusyError {
                    source_id: source_id.clone(),
                    table: table.clone(),
                })
            }
            JammiError::JobAttemptSuperseded { job_id } => {
                Variant::JobAttemptSuperseded(pb::JobAttemptSupersededError {
                    job_id: job_id.clone(),
                })
            }
            JammiError::JobCancelled { job_id } => Variant::JobCancelled(pb::JobCancelledError {
                job_id: job_id.clone(),
            }),
            JammiError::Lexical(message) => Variant::Lexical(pb::StringError {
                message: message.clone(),
            }),
            JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            } => Variant::IncompatibleFormat(pb::IncompatibleFormatError {
                artifact: artifact.clone(),
                found: found.clone(),
                supported: supported.clone(),
            }),
            JammiError::DependencyCycle { table } => {
                Variant::DependencyCycle(pb::DependencyCycleError {
                    table: table.clone(),
                })
            }
            JammiError::NotRecomputable { table } => {
                Variant::NotRecomputable(pb::NotRecomputableError {
                    table: table.clone(),
                })
            }
            JammiError::InvalidKey { column, null_count } => {
                Variant::InvalidKey(pb::InvalidKeyError {
                    column: column.clone(),
                    null_count: *null_count,
                })
            }
            JammiError::VersionUnavailable { table, version } => {
                Variant::VersionUnavailable(pb::VersionUnavailableError {
                    table: table.clone(),
                    version: *version,
                })
            }
            JammiError::NotRefreshable { table, reason } => {
                Variant::NotRefreshable(pb::NotRefreshableError {
                    table: table.clone(),
                    reason: reason.as_str().to_string(),
                })
            }
            JammiError::DefinitionDrift {
                table,
                recorded,
                current,
            } => Variant::DefinitionDrift(pb::DefinitionDriftError {
                table: table.clone(),
                recorded: recorded.clone(),
                current: current.clone(),
            }),
            JammiError::NonUniqueKey {
                table,
                scan,
                keys,
                total,
            } => Variant::NonUniqueKey(pb::NonUniqueKeyError {
                table: table.clone(),
                scan: scan.as_str().to_string(),
                keys: keys
                    .iter()
                    .map(|(key, count)| pb::KeyCount {
                        key: key.clone(),
                        count: *count,
                    })
                    .collect(),
                total: *total,
            }),
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

/// Reconstruct the engine error from the wire detail. The inverse of the
/// encode above; `message` is the enclosing `Status`'s own message, threaded
/// in so a detail whose `variant` is unset (a peer built against a NEWER
/// contract that added a variant this build's codegen does not know — an
/// "unknown oneof", NOT the "no detail at all" case [`error_from_status`]
/// handles separately) reconstructs as [`JammiError::Other`] carrying that
/// faithful message, never the empty string a bare `String::new()` would
/// silently substitute. Decode is total either way.
fn jammi_error_from_detail(detail: pb::JammiErrorDetail, message: &str) -> JammiError {
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
        Some(Variant::ChannelCatalog(e)) => {
            JammiError::ChannelCatalog(channel_catalog_error_from_detail(e, message))
        }
        Some(Variant::ChannelAssembly(e)) => JammiError::ChannelAssembly(e.message),
        Some(Variant::MutableTable(e)) => {
            JammiError::MutableTable(mutable_table_error_from_detail(e, message))
        }
        Some(Variant::RowGone(e)) => JammiError::RowGone { table: e.table },
        Some(Variant::TenantMismatch(e)) => JammiError::TenantMismatch { table: e.table },
        Some(Variant::LeaseLost(e)) => JammiError::LeaseLost { table: e.table },
        Some(Variant::CasFailed(e)) => JammiError::CasFailed {
            table: e.table,
            status: e.status,
        },
        Some(Variant::SourceBusy(e)) => JammiError::SourceBusy {
            source_id: e.source_id,
            table: e.table,
        },
        Some(Variant::JobAttemptSuperseded(e)) => {
            JammiError::JobAttemptSuperseded { job_id: e.job_id }
        }
        Some(Variant::JobCancelled(e)) => JammiError::JobCancelled { job_id: e.job_id },
        Some(Variant::Lexical(e)) => JammiError::Lexical(e.message),
        Some(Variant::IncompatibleFormat(e)) => JammiError::IncompatibleFormat {
            artifact: e.artifact,
            found: e.found,
            supported: e.supported,
        },
        Some(Variant::DependencyCycle(e)) => JammiError::DependencyCycle { table: e.table },
        Some(Variant::NotRecomputable(e)) => JammiError::NotRecomputable { table: e.table },
        Some(Variant::InvalidKey(e)) => JammiError::InvalidKey {
            column: e.column,
            null_count: e.null_count,
        },
        Some(Variant::VersionUnavailable(e)) => JammiError::VersionUnavailable {
            table: e.table,
            version: e.version,
        },
        // An unknown `reason` / `scan` token (a newer peer) reconstructs as
        // `Other` carrying the Status message, the same total-decode stance
        // as the unknown-oneof arm — never a fabricated token.
        Some(Variant::NotRefreshable(e)) => match NotRefreshableReason::parse(&e.reason) {
            Some(reason) => JammiError::NotRefreshable {
                table: e.table,
                reason,
            },
            None => JammiError::Other(message.to_string()),
        },
        Some(Variant::DefinitionDrift(e)) => JammiError::DefinitionDrift {
            table: e.table,
            recorded: e.recorded,
            current: e.current,
        },
        Some(Variant::NonUniqueKey(e)) => match NonUniqueScan::parse(&e.scan) {
            Some(scan) => JammiError::NonUniqueKey {
                table: e.table,
                scan,
                keys: e.keys.into_iter().map(|k| (k.key, k.count)).collect(),
                total: e.total,
            },
            None => JammiError::Other(message.to_string()),
        },
        Some(Variant::Other(e)) => JammiError::Other(e.message),
        // The unknown-oneof case (B5): `message` is the enclosing `Status`'s
        // own text, so the reconstructed error still carries the real fault
        // description even though this build cannot recover which specific
        // variant a newer peer set.
        None => JammiError::Other(message.to_string()),
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
/// so decode stays total without a panic. `message` is the enclosing
/// `Status`'s own text (see [`jammi_error_from_detail`]'s doc): a detail with
/// no variant set (an unknown oneof — a peer built against a newer contract)
/// reconstructs as `Schema(message)` — kept inside the engine-owned taxonomy
/// rather than escaping to `Other`, but carrying the real fault text instead
/// of a fabricated empty string.
fn mutable_table_error_from_detail(
    detail: pb::MutableTableErrorDetail,
    message: &str,
) -> MutableTableError {
    use pb::mutable_table_error_detail::Variant;
    let reconstruct_id =
        |s: String, wrap: fn(MutableTableId) -> MutableTableError| match MutableTableId::new(&s) {
            Ok(id) => wrap(id),
            Err(_) => MutableTableError::InvalidId(s),
        };
    match detail.variant {
        Some(Variant::InvalidId(m)) => MutableTableError::InvalidId(m),
        Some(Variant::Schema(m)) => MutableTableError::Schema(m),
        Some(Variant::MissingPrimaryKey(m)) => MutableTableError::MissingPrimaryKey(m),
        Some(Variant::ReservedColumn(m)) => MutableTableError::ReservedColumn(m),
        Some(Variant::NotFound(s)) => reconstruct_id(s, MutableTableError::NotFound),
        Some(Variant::AlreadyExists(s)) => reconstruct_id(s, MutableTableError::AlreadyExists),
        Some(Variant::NoOrderColumn(_)) => MutableTableError::NoOrderColumn,
        Some(Variant::Backend(e)) => {
            MutableTableError::Backend(backend_error_from_detail(e, message))
        }
        None => MutableTableError::Schema(message.to_string()),
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

/// Reconstruct the [`ChannelCatalogError`] from its wire detail — the inverse
/// of the encode above. The struct variants re-parse the canonical PascalCase
/// type token; a forged token that does not parse reconstructs as
/// `InvalidColumnType` carrying the offending string — exactly the variant
/// the engine produces for an unknown token, so decode stays total.
/// `message` is the enclosing `Status`'s own text (see
/// [`jammi_error_from_detail`]'s doc): a detail with no variant set (an
/// unknown oneof) reconstructs as `NotRegistered(message)` — kept inside the
/// channel-catalog taxonomy rather than escaping to `Other`, carrying the
/// real fault text instead of a fabricated empty string.
fn channel_catalog_error_from_detail(
    detail: pb::ChannelCatalogErrorDetail,
    message: &str,
) -> ChannelCatalogError {
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
        Some(Variant::ColumnConflict(d)) => match (parse_ty(d.existing), parse_ty(d.requested)) {
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
        None => ChannelCatalogError::NotRegistered(message.to_string()),
    }
}

/// Encode the engine-owned [`BackendError`] into its structured wire detail.
/// Every variant but `Sqlx` and `Busy` reconstructs field-for-field. `Sqlx`
/// wraps a raw `sqlx::Error` that cannot cross a process boundary, so it
/// folds to its faithful `Display` string — the genuine fidelity limit,
/// mirroring how the top-level detail folds its own foreign `#[from]`
/// variants. `Busy` is a transaction-internal rollback sentinel
/// (`jammi_db::catalog::backend::BackendError::Busy`'s own doc comment):
/// every producer intercepts it before its `Result` ever leaves
/// the catalog method that returned it (mapping it to a typed
/// [`jammi_db::error::JammiError::SourceBusy`] or similar), so in practice
/// it never reaches this encoder — folded to `Execution` for the same
/// reason `Sqlx` is: exhaustiveness, not an expected wire path.
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
            BackendError::Busy(m) => Variant::Execution(m.clone()),
        };
        pb::BackendErrorDetail {
            variant: Some(variant),
        }
    }
}

/// Reconstruct the [`BackendError`] from its wire detail — the inverse of the
/// encode above. A free fn, not a `From` impl: `message` is the enclosing
/// `Status`'s own text (threaded through by both nesting callers,
/// [`mutable_table_error_from_detail`] and [`trigger_error_from_detail`]),
/// so a detail whose `variant` oneof is unset (an unknown oneof — a peer
/// built against a newer contract that added a `BackendErrorDetail` variant
/// this build's codegen does not know) reconstructs as
/// `BackendError::Execution(message)` carrying that faithful text, never the
/// empty string a bare `String::new()` would silently substitute. The
/// `tenant_mismatch` arm re-parses the UUID strings (empty == `None`); a
/// non-empty string that fails to parse reconstructs as `None`, the only
/// total option for a forged payload, since the variant's faithful path
/// always carries a valid UUID. `Sqlx` reconstructs as `Execution` carrying
/// the original `Display` string — the raw `sqlx::Error` cannot be rebuilt,
/// so the faithful message lands in the nearest backend-owned string arm
/// rather than escaping the taxonomy.
fn backend_error_from_detail(detail: pb::BackendErrorDetail, message: &str) -> BackendError {
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
        None => BackendError::Execution(message.to_string()),
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
/// encode above. The nested engine-owned `backing_table` reconstructs through
/// [`mutable_table_error_from_detail`] (also threaded `message`); `backend`
/// reconstructs through `backend_error_from_detail`, also threaded
/// `message` — so an unknown `BackendErrorDetail` oneof nested under
/// `TriggerError::Backend` carries the real fault text too, not an empty
/// string. `message` is the enclosing `Status`'s own text: a detail with no
/// variant set (an unknown oneof) reconstructs as `TriggerError::Catalog(message)`
/// — kept inside the trigger taxonomy rather than escaping, carrying the real
/// fault text instead of a fabricated empty string.
fn trigger_error_from_detail(detail: pb::TriggerErrorDetail, message: &str) -> TriggerError {
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
        Some(Variant::BackingTable(e)) => {
            TriggerError::BackingTable(mutable_table_error_from_detail(e, message))
        }
        Some(Variant::Backend(e)) => TriggerError::Backend(backend_error_from_detail(e, message)),
        Some(Variant::Driver(m)) => TriggerError::Driver(m),
        Some(Variant::Catalog(m)) => TriggerError::Catalog(m),
        None => TriggerError::Catalog(message.to_string()),
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
/// forged string that fails to parse cannot be rebuilt as the typed variant
/// (there is no id to carry), so — mirroring how the `serde` arm folds a
/// foreign error it cannot rebuild — it reconstructs as `AuditError::Storage`
/// carrying the malformed value verbatim rather than fabricating the nil UUID,
/// which would silently claim a specific (and wrong) offending query. The
/// `serde` arm itself reconstructs as `AuditError::Storage` carrying the
/// original `Display` string — the raw `serde_json::Error` cannot be rebuilt,
/// so the faithful message lands in the nearest audit-owned string arm rather
/// than escaping the taxonomy. `message` is the enclosing `Status`'s own
/// text: a detail with no variant set (an unknown oneof) reconstructs as
/// `Storage(message)` rather than a fabricated empty string.
fn audit_error_from_detail(detail: pb::AuditErrorDetail, message: &str) -> AuditError {
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
        Some(Variant::SignatureMismatch(id)) => match Uuid::parse_str(&id) {
            Ok(uuid) => AuditError::SignatureMismatch(uuid),
            Err(e) => AuditError::Storage(format!(
                "signature_mismatch: malformed query id {id:?}: {e}"
            )),
        },
        Some(Variant::MasterKey(m)) => AuditError::MasterKey(m),
        Some(Variant::Serde(m)) => AuditError::Storage(m),
        Some(Variant::Storage(m)) => AuditError::Storage(m),
        Some(Variant::Broker(m)) => AuditError::Broker(m),
        None => AuditError::Storage(message.to_string()),
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
        Some(detail) => jammi_error_from_detail(detail, status.message()),
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
        Some(detail) => trigger_error_from_detail(detail, status.message()),
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
        Some(detail) => audit_error_from_detail(detail, status.message()),
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

    /// Compile-time half of the completeness proof (K4 error-parity oracle):
    /// exhaustive over EVERY `JammiError` variant, own-shape or genuinely-
    /// foreign, with NO catch-all arm. A new variant added to `JammiError`
    /// fails to compile here until it is listed — forcing the author to
    /// decide, in this same match, whether it is owned-shape (add it to
    /// `every_owned_shape_variant_round_trips_to_itself`'s `owned` array too)
    /// or a genuine foreign-source fold (leave it here with no round-trip
    /// case, mirroring `Io`/`Toml`/`Json`/`DataFusion`/`Trigger`/`Storage`/
    /// `BackendDriver`). Every arm is a no-op; the value is the match's
    /// EXHAUSTIVENESS, not its body.
    fn assert_exhaustive_variant_coverage(err: &JammiError) {
        match err {
            JammiError::Config(_)
            | JammiError::Catalog(_)
            | JammiError::Source { .. }
            | JammiError::Model { .. }
            | JammiError::ModelNotFound { .. }
            | JammiError::ModelReferenced { .. }
            | JammiError::Inference(_)
            | JammiError::FineTune(_)
            | JammiError::Eval(_)
            | JammiError::Gpu(_)
            | JammiError::Backend(_)
            | JammiError::Io(_)
            | JammiError::BackendDriver(_)
            | JammiError::Tenant(_)
            | JammiError::Toml(_)
            | JammiError::Json(_)
            | JammiError::DataFusion(_)
            | JammiError::ChannelCatalog(_)
            | JammiError::ChannelAssembly(_)
            | JammiError::Lexical(_)
            | JammiError::MutableTable(_)
            | JammiError::Trigger(_)
            | JammiError::Storage(_)
            | JammiError::Schema { .. }
            | JammiError::IncompatibleFormat { .. }
            | JammiError::DependencyCycle { .. }
            | JammiError::NotRecomputable { .. }
            | JammiError::RowGone { .. }
            | JammiError::TenantMismatch { .. }
            | JammiError::LeaseLost { .. }
            | JammiError::CasFailed { .. }
            | JammiError::JobAttemptSuperseded { .. }
            | JammiError::JobCancelled { .. }
            | JammiError::SourceBusy { .. }
            | JammiError::InvalidKey { .. }
            | JammiError::VersionUnavailable { .. }
            | JammiError::NotRefreshable { .. }
            | JammiError::DefinitionDrift { .. }
            | JammiError::NonUniqueKey { .. }
            | JammiError::Unavailable { .. }
            | JammiError::Other(_) => {}
        }
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
            JammiError::RowGone {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::TenantMismatch {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::LeaseLost {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::CasFailed {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
                status: "ready".into(),
            },
            JammiError::SourceBusy {
                source_id: "src1".into(),
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::JobAttemptSuperseded {
                job_id: "job-fine-tune-1".into(),
            },
            JammiError::JobCancelled {
                job_id: "job-fine-tune-1".into(),
            },
            JammiError::Lexical("bm25 sidecar build: tantivy index open failed".into()),
            JammiError::IncompatibleFormat {
                artifact: "ann-manifest".into(),
                found: "3".into(),
                supported: "2".into(),
            },
            JammiError::DependencyCycle {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::NotRecomputable {
                table: "src1__text_embedding__m__20260101T000000_deadbeef".into(),
            },
            JammiError::InvalidKey {
                column: "id".into(),
                null_count: 3,
            },
            JammiError::VersionUnavailable {
                table: "patents__embedding__m".into(),
                version: 4,
            },
            JammiError::NotRefreshable {
                table: "patents__embedding__m".into(),
                reason: NotRefreshableReason::MissingContentHash,
            },
            JammiError::DefinitionDrift {
                table: "patents__embedding__m".into(),
                recorded: "abc".into(),
                current: "def".into(),
            },
            JammiError::NonUniqueKey {
                table: "patents__embedding__m".into(),
                scan: NonUniqueScan::Source,
                keys: vec![("k1".into(), 2), ("k2".into(), 3)],
                total: 2,
            },
            JammiError::Other("an error with no more specific shape".into()),
        ];
        for err in &owned {
            // Compile-time exhaustiveness: see `assert_exhaustive_variant_coverage`.
            assert_exhaustive_variant_coverage(err);
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

    /// A decodable `pb::JammiErrorDetail` whose `variant` oneof is unset --
    /// the shape a NEWER peer's payload produces when it sets a oneof tag
    /// this build's codegen does not know (built here with a synthetic
    /// field number no `JammiErrorDetail` variant ever uses, so prost's
    /// decoder skips it rather than erroring, leaving `variant: None`
    /// exactly like a real cross-version drift would) -- must reconstruct
    /// as `JammiError::Other` carrying the enclosing `Status`'s own message
    /// verbatim, never `Other(String::new())` discarding it.
    #[test]
    fn unknown_oneof_variant_reconstructs_other_carrying_the_status_message_not_empty() {
        /// A shadow message sharing NO field number with `pb::JammiErrorDetail`'s
        /// oneof (1-29, minus the reserved 15) -- decoding its bytes AS a
        /// `JammiErrorDetail` therefore always leaves `variant` unset, the
        /// same shape prost produces for a genuinely newer, unrecognized
        /// oneof tag.
        #[derive(Clone, PartialEq, ::prost::Message)]
        struct ShadowDetailFromANewerPeer {
            #[prost(string, tag = "9001")]
            a_variant_this_build_does_not_know: String,
        }

        let shadow = ShadowDetailFromANewerPeer {
            a_variant_this_build_does_not_know: "payload only a newer peer understands".into(),
        };
        let bytes = shadow.encode_to_vec();
        let detail = pb::JammiErrorDetail::decode(bytes.as_slice())
            .expect("an unrecognized field number is skipped, not a decode error");
        assert!(
            detail.variant.is_none(),
            "field 9001 is outside JammiErrorDetail's oneof, so decode must leave variant unset"
        );

        let peer_message = "the real fault text a newer peer attached";
        let status = attach_detail(Code::Internal, peer_message.to_string(), &detail);
        match error_from_status(&status) {
            JammiError::Other(message) => assert_eq!(
                message, peer_message,
                "an unknown oneof variant must reconstruct JammiError::Other carrying the \
                 Status's own message, never an empty string"
            ),
            other => panic!("expected JammiError::Other, got {other:?}"),
        }
    }

    /// A shadow message sharing NO field number with `pb::BackendErrorDetail`'s
    /// oneof (1-8) -- decoding its bytes AS a `BackendErrorDetail` therefore
    /// always leaves `variant` unset, the same shape prost produces for a
    /// genuinely newer, unrecognized oneof tag on that nested detail.
    #[derive(Clone, PartialEq, ::prost::Message)]
    struct ShadowBackendDetailFromANewerPeer {
        #[prost(string, tag = "9001")]
        a_variant_this_build_does_not_know: String,
    }

    /// Build a decodable `pb::BackendErrorDetail` whose `variant` oneof is
    /// unset, via the same shadow-message trick
    /// [`unknown_oneof_variant_reconstructs_other_carrying_the_status_message_not_empty`]
    /// uses at the top level -- proves this is genuinely what decode produces
    /// from an unrecognized `BackendErrorDetail` variant, not a hand-built
    /// `None` no real peer could ever send.
    fn unknown_backend_detail() -> pb::BackendErrorDetail {
        let shadow = ShadowBackendDetailFromANewerPeer {
            a_variant_this_build_does_not_know: "payload only a newer peer understands".into(),
        };
        let bytes = shadow.encode_to_vec();
        let detail = pb::BackendErrorDetail::decode(bytes.as_slice())
            .expect("an unrecognized field number is skipped, not a decode error");
        assert!(
            detail.variant.is_none(),
            "field 9001 is outside BackendErrorDetail's oneof, so decode must leave variant unset"
        );
        detail
    }

    /// An unknown `BackendErrorDetail` oneof nested under
    /// `MutableTableError::Backend` (a NEWER peer's `BackendErrorDetail`
    /// variant this build's codegen does not know) must reconstruct as
    /// `BackendError::Execution` carrying the enclosing `Status`'s own
    /// message verbatim, via `backend_error_from_detail` -- never
    /// `Execution(String::new())` silently discarding it.
    #[test]
    fn unknown_backend_variant_nested_in_mutable_table_carries_the_status_message() {
        use pb::mutable_table_error_detail::Variant;

        let mt_detail = pb::MutableTableErrorDetail {
            variant: Some(Variant::Backend(unknown_backend_detail())),
        };
        let peer_message = "the real fault text a newer peer attached";
        match mutable_table_error_from_detail(mt_detail, peer_message) {
            MutableTableError::Backend(BackendError::Execution(message)) => assert_eq!(
                message, peer_message,
                "an unknown BackendErrorDetail variant nested under MutableTableError::Backend \
                 must reconstruct BackendError::Execution carrying the Status's own message, \
                 never an empty string"
            ),
            other => panic!(
                "expected MutableTableError::Backend(BackendError::Execution(_)), got {other:?}"
            ),
        }
    }

    /// The `TriggerError::Backend` analogue of
    /// [`unknown_backend_variant_nested_in_mutable_table_carries_the_status_message`]:
    /// an unknown `BackendErrorDetail` oneof nested under `TriggerError::Backend`
    /// must also carry the enclosing `Status`'s own message, never an empty
    /// string.
    #[test]
    fn unknown_backend_variant_nested_in_trigger_carries_the_status_message() {
        use pb::trigger_error_detail::Variant;

        let trigger_detail = pb::TriggerErrorDetail {
            variant: Some(Variant::Backend(unknown_backend_detail())),
        };
        let peer_message = "the real fault text a newer peer attached";
        match trigger_error_from_detail(trigger_detail, peer_message) {
            TriggerError::Backend(BackendError::Execution(message)) => assert_eq!(
                message, peer_message,
                "an unknown BackendErrorDetail variant nested under TriggerError::Backend must \
                 reconstruct BackendError::Execution carrying the Status's own message, never an \
                 empty string"
            ),
            other => {
                panic!("expected TriggerError::Backend(BackendError::Execution(_)), got {other:?}")
            }
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

    /// A `signature_mismatch` detail carrying an un-parseable id (a forged or
    /// corrupted wire message — the faithful encode path never emits one) must
    /// not silently coerce to the nil UUID: that would fabricate a valid-looking
    /// id and point blame at a specific (wrong) query. It folds to `Storage`
    /// carrying the malformed value verbatim, the same "cannot rebuild the typed
    /// variant" fold the foreign `serde` leaf takes.
    #[test]
    fn signature_mismatch_malformed_id_folds_to_storage_not_nil_uuid() {
        use pb::audit_error_detail::Variant;
        let malformed = "not-a-uuid";
        let detail = pb::AuditErrorDetail {
            variant: Some(Variant::SignatureMismatch(malformed.to_string())),
        };
        match audit_error_from_detail(detail, "") {
            AuditError::Storage(message) => {
                assert!(
                    message.contains(malformed),
                    "the malformed id is preserved verbatim in the fold: {message:?}"
                );
            }
            AuditError::SignatureMismatch(uuid) => panic!(
                "a malformed id must not coerce to a fabricated UUID (got {uuid}), \
                 it must fold to Storage instead"
            ),
            other => panic!("expected a Storage fold, got {other:?}"),
        }
    }
}
