//! Faithful engine-error transport: the `JammiError` ↔ [`pb::JammiErrorDetail`]
//! conversions and the [`Status`] detail attach/extract helpers that make a
//! remote transport reconstruct the *exact* [`JammiError`] the in-process path
//! returns, not a lossy gRPC-code-category guess.
//!
//! The two directions are a pair:
//!
//! * encode — [`From<&JammiError> for pb::JammiErrorDetail`]; the server's
//!   central error mapping calls it once for every `JammiError` (DRY) and
//!   attaches the prost-encoded detail to the [`Status`] via
//!   [`attach_error_detail`].
//! * decode — [`From<pb::JammiErrorDetail> for JammiError`]; a remote client
//!   reads the detail back off a [`Status`] via [`error_from_status`] and
//!   reconstructs the precise variant.
//!
//! Both impls are orphan-rule-clean: `pb::JammiErrorDetail` is a local
//! generated type, so `From` to/from the foreign `JammiError` is allowed
//! without a newtype.
//!
//! `JammiError` variants that wrap a foreign source error (`Io`, `DataFusion`,
//! `Storage`, the `#[from]` arms) carry a non-serialisable payload and have no
//! faithful field-level reconstruction; the encode folds them into the `other`
//! arm carrying `err.to_string()`, which decodes to [`JammiError::Other`] — the
//! same string the in-process `Display` surfaces. These arms are unreachable as
//! *distinct* targets on the embedding/search verb surface this stage wires, so
//! the fold loses nothing a caller on this path could observe.

use jammi_db::error::JammiError;
use prost::bytes::Bytes;
use prost::Message;
use tonic::{Code, Status};

use crate::wire::proto::error as pb;

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
            // Every other variant either carries a non-serialisable foreign
            // source error or is the existing catch-all; both reconstruct
            // faithfully only as the rendered string the in-process `Display`
            // would surface. `to_string()` is that string.
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
            Some(Variant::Other(e)) => JammiError::Other(e.message),
            None => JammiError::Other(String::new()),
        }
    }
}

/// Attach a faithful [`pb::JammiErrorDetail`] for `err` to a [`Status`] of the
/// given `code` and `message`. The detail rides in the Status `details` bytes
/// (the gRPC richer-error model) so a decoding client reconstructs the precise
/// variant, while the `code` + `message` keep the idiomatic gRPC surface for a
/// client that does not decode the detail.
pub fn attach_error_detail(code: Code, message: String, err: &JammiError) -> Status {
    let detail = pb::JammiErrorDetail::from(err);
    Status::with_details(code, message, Bytes::from(detail.encode_to_vec()))
}

/// Reconstruct the engine error a server attached to a [`Status`]. When the
/// status carries a decodable [`pb::JammiErrorDetail`] the exact variant is
/// rebuilt; otherwise (no detail, or undecodable bytes) the status `message`
/// stands in as [`JammiError::Other`] — the faithful fallback, since a status
/// without a Jammi detail is by construction not an engine `JammiError`.
pub fn error_from_status(status: &Status) -> JammiError {
    let details = status.details();
    if details.is_empty() {
        return JammiError::Other(status.message().to_string());
    }
    match pb::JammiErrorDetail::decode(details) {
        Ok(detail) => JammiError::from(detail),
        Err(_) => JammiError::Other(status.message().to_string()),
    }
}
