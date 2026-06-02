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
//! exactly, field for field. The only variants that fold into `other` are the
//! eight `#[from]` foreign-source ones (`Io`, `BackendDriver`, `Toml`, `Json`,
//! `DataFusion`, `MutableTable`, `Trigger`, `Storage`): their inner foreign
//! error cannot cross a process boundary, so they reconstruct as
//! [`JammiError::Other`] carrying the faithful `Display` string — this is the
//! genuine limit, not a lossy guess.

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
            // The fold reaches ONLY the eight `#[from]` foreign-source variants
            // (`Io`, `BackendDriver`, `Toml`, `Json`, `DataFusion`,
            // `MutableTable`, `Trigger`, `Storage`) and the existing `Other`:
            // every owned-shape variant has an explicit arm above. A foreign
            // source error cannot cross a process boundary, so it reconstructs
            // as `JammiError::Other` carrying the faithful `Display` string —
            // the genuine fidelity limit. `to_string()` is that string.
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

    /// The eight `#[from]` foreign-source variants carry an inner error that
    /// cannot cross a process boundary, so they reconstruct as
    /// `JammiError::Other` carrying the faithful `Display` string — the genuine
    /// fidelity limit. `Io` is the representative case; the fold is identical
    /// for all eight (`BackendDriver`, `Toml`, `Json`, `DataFusion`,
    /// `MutableTable`, `Trigger`, `Storage`).
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
}
