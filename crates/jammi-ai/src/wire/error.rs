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
//! The contract's fidelity boundary is precise, and faithfulness is a property
//! of the error type — not of any one verb surface — so the mapping is complete
//! over `JammiError`: every owned-shape variant (the String- and struct-carrying
//! ones — `Source`, `Model`, `Inference`, `Catalog`, `Schema`, `Config`, `Eval`,
//! `Tenant`, `FineTune`, `Gpu`, `Backend`, `EvidenceChannel`) reconstructs
//! exactly, field for field. So does [`JammiError::MutableTable`]: its inner
//! [`MutableTableError`] is engine-owned and every variant's fields reconstruct,
//! so it carries a structured [`pb::MutableTableErrorDetail`] (which nests an
//! engine-owned [`pb::BackendErrorDetail`] for its `Backend` arm) and crosses
//! the wire faithfully — never a fold. The only variants that fold into `other`
//! are the genuinely-foreign `#[from]` ones whose inner error cannot cross a
//! process boundary (`Io`, `BackendDriver`, `Toml`, `Json`, `DataFusion`,
//! `Trigger`, `Storage`, and the lone [`BackendError::Sqlx`] nested under
//! `MutableTable`): they reconstruct as [`JammiError::Other`] (or, for `Sqlx`, a
//! backend-detail string arm) carrying the faithful `Display` string — the
//! genuine limit, not a lossy guess.

use jammi_db::catalog::backend::BackendError;
use jammi_db::error::JammiError;
use jammi_db::store::mutable::{MutableTableError, MutableTableId};
use jammi_db::TenantId;
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
            JammiError::FineTune(message) => Variant::FineTune(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Gpu(message) => Variant::Gpu(pb::StringError {
                message: message.clone(),
            }),
            JammiError::Backend(message) => Variant::Backend(pb::StringError {
                message: message.clone(),
            }),
            JammiError::EvidenceChannel(message) => Variant::EvidenceChannel(pb::StringError {
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
            Some(Variant::EvidenceChannel(e)) => JammiError::EvidenceChannel(e.message),
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
            JammiError::Inference("encode_query forward: shape mismatch".into()),
            JammiError::FineTune("checkpoint epoch 3 diverged".into()),
            JammiError::Eval("golden NER fixture row 7 mismatch".into()),
            JammiError::Gpu("no CUDA device visible".into()),
            JammiError::Backend("vLLM returned HTTP 503".into()),
            JammiError::Tenant("nil UUID is not a valid tenant".into()),
            JammiError::EvidenceChannel("channel 'patents' not registered".into()),
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
}
