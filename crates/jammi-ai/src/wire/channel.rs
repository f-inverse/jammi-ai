//! `ChannelService` proto↔domain conversions.
//!
//! The messages mirror the engine's [`ChannelColumn`] field for field; the
//! proto [`pb::ChannelColumnType`] mirrors the engine's closed
//! [`ChannelColumnType`] enum. The engine types live in `jammi-db`, so the
//! decodes are free functions taking the raw wire shapes rather than orphan-
//! rule-blocked `TryFrom<i32>` impls.

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType};
use jammi_db::ChannelId;
use tonic::Status;

use crate::wire::proto::channel as pb;

/// Parse a wire channel id into the validated [`ChannelId`] newtype.
pub fn parse_channel_id(id: &str) -> Result<ChannelId, Status> {
    if id.is_empty() {
        return Err(Status::invalid_argument("channel_id is required"));
    }
    ChannelId::new(id).map_err(|e| Status::invalid_argument(e.to_string()))
}

/// Map the wire columns onto the engine's [`ChannelColumn`], rejecting a missing
/// or unspecified column type — a column that names no type is a client error,
/// not a silent default.
pub fn columns_from_proto(columns: Vec<pb::ChannelColumn>) -> Result<Vec<ChannelColumn>, Status> {
    columns
        .into_iter()
        .map(|c| {
            if c.name.is_empty() {
                return Err(Status::invalid_argument("column name is required"));
            }
            Ok(ChannelColumn {
                name: c.name,
                data_type: column_type_from_proto(c.data_type)?,
            })
        })
        .collect()
}

/// Encode the engine's [`ChannelColumn`] list onto the wire — the inverse of
/// [`columns_from_proto`], for the [`crate::RemoteSession`] send side. Total:
/// every engine column maps to a fully-specified wire column (the engine's
/// `ChannelColumnType` is a closed enum with no unspecified state).
pub fn columns_to_proto(columns: &[ChannelColumn]) -> Vec<pb::ChannelColumn> {
    columns
        .iter()
        .map(|c| pb::ChannelColumn {
            name: c.name.clone(),
            data_type: column_type_to_proto(c.data_type) as i32,
        })
        .collect()
}

/// Map the engine's closed [`ChannelColumnType`] onto the wire enum. Total — the
/// engine type has no unspecified variant, so this never emits `UNSPECIFIED`.
fn column_type_to_proto(ty: ChannelColumnType) -> pb::ChannelColumnType {
    match ty {
        ChannelColumnType::Float32 => pb::ChannelColumnType::Float32,
        ChannelColumnType::Float64 => pb::ChannelColumnType::Float64,
        ChannelColumnType::Int32 => pb::ChannelColumnType::Int32,
        ChannelColumnType::Int64 => pb::ChannelColumnType::Int64,
        ChannelColumnType::Utf8 => pb::ChannelColumnType::Utf8,
        ChannelColumnType::Boolean => pb::ChannelColumnType::Boolean,
    }
}

/// Map the proto [`pb::ChannelColumnType`] discriminant onto the engine's closed
/// [`ChannelColumnType`]. An unspecified or unknown type is rejected.
fn column_type_from_proto(ty: i32) -> Result<ChannelColumnType, Status> {
    match pb::ChannelColumnType::try_from(ty) {
        Ok(pb::ChannelColumnType::Float32) => Ok(ChannelColumnType::Float32),
        Ok(pb::ChannelColumnType::Float64) => Ok(ChannelColumnType::Float64),
        Ok(pb::ChannelColumnType::Int32) => Ok(ChannelColumnType::Int32),
        Ok(pb::ChannelColumnType::Int64) => Ok(ChannelColumnType::Int64),
        Ok(pb::ChannelColumnType::Utf8) => Ok(ChannelColumnType::Utf8),
        Ok(pb::ChannelColumnType::Boolean) => Ok(ChannelColumnType::Boolean),
        Ok(pb::ChannelColumnType::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "column data_type must be specified",
        )),
    }
}
