//! The incremental-embedding verbs' wire vocabulary: the `RefreshEmbeddings`
//! / `CompactEmbeddings` / `ExpireVersions` request decoders shared by the
//! gRPC handler and the embedded binding, and the report types every
//! transport returns (`RefreshReport`, `ExpiryReport`) with their proto
//! conversions. The engine's actuators produce these types directly, so the
//! remote client and a local session hand a caller the identical value.

use jammi_db::store::DeletePolicy;
use tonic::Status;

use crate::proto::embedding as pb;

/// The options of one refresh.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefreshOptions {
    /// What a key the source no longer has becomes (default: tombstoned).
    pub deletes: DeletePolicy,
}

/// Whether a refresh published a new version or found nothing to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshOutcome {
    Published,
    NoChange,
}

/// What a refresh or compaction did. Realized counts: `inferred_rows` is the
/// number of rows the model was invoked on, `dropped_rows` the keys asked for
/// that the model did not realize (a per-row input failure), so `inferred_rows
/// - dropped_rows` is the fragment's row count.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RefreshReport {
    pub table: String,
    /// The published version (`Published`), or the current one (`NoChange`).
    pub version: Option<i64>,
    pub parent_version: Option<i64>,
    pub inferred_rows: u64,
    pub added: u64,
    pub changed: u64,
    pub deleted: u64,
    pub unchanged: u64,
    pub dropped_rows: u64,
    pub live_rows: u64,
    pub masked_rows: u64,
    pub outcome: RefreshOutcome,
}

/// What `expire_versions` removed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExpiryReport {
    pub table: String,
    /// The version rows deleted, ascending.
    pub expired_versions: Vec<i64>,
    /// The objects (fragments, deletes, manifests, segment siblings) deleted.
    pub objects_deleted: u64,
}

/// The decoded `RefreshEmbeddings` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshEmbeddingsArgs {
    pub table: String,
    pub options: RefreshOptions,
}

/// The decoded `ExpireVersions` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpireVersionsArgs {
    pub table: String,
    pub before: i64,
}

/// Map the wire `DeletePolicy`; `UNSPECIFIED` keeps the engine default
/// (`Tombstone`). An out-of-range value is rejected loudly.
pub fn delete_policy_from_proto(policy: i32) -> Result<DeletePolicy, Status> {
    match pb::DeletePolicy::try_from(policy) {
        Ok(pb::DeletePolicy::Unspecified) | Ok(pb::DeletePolicy::Tombstone) => {
            Ok(DeletePolicy::Tombstone)
        }
        Ok(pb::DeletePolicy::Retain) => Ok(DeletePolicy::Retain),
        Err(_) => Err(Status::invalid_argument("unknown delete policy")),
    }
}

/// The wire `DeletePolicy` of an engine policy.
pub fn delete_policy_to_proto(policy: DeletePolicy) -> pb::DeletePolicy {
    match policy {
        DeletePolicy::Tombstone => pb::DeletePolicy::Tombstone,
        DeletePolicy::Retain => pb::DeletePolicy::Retain,
    }
}

/// Decode a `RefreshEmbeddingsRequest`: an empty `table` is a client error.
pub fn refresh_embeddings_from_proto(
    req: pb::RefreshEmbeddingsRequest,
) -> Result<RefreshEmbeddingsArgs, Status> {
    if req.table.is_empty() {
        return Err(Status::invalid_argument("table is required"));
    }
    Ok(RefreshEmbeddingsArgs {
        table: req.table,
        options: RefreshOptions {
            deletes: delete_policy_from_proto(req.deletes)?,
        },
    })
}

/// Decode a `CompactEmbeddingsRequest` into the table name.
pub fn compact_embeddings_from_proto(req: pb::CompactEmbeddingsRequest) -> Result<String, Status> {
    if req.table.is_empty() {
        return Err(Status::invalid_argument("table is required"));
    }
    Ok(req.table)
}

/// Decode an `ExpireVersionsRequest`.
pub fn expire_versions_from_proto(
    req: pb::ExpireVersionsRequest,
) -> Result<ExpireVersionsArgs, Status> {
    if req.table.is_empty() {
        return Err(Status::invalid_argument("table is required"));
    }
    Ok(ExpireVersionsArgs {
        table: req.table,
        before: req.before,
    })
}

/// Encode a report onto the wire.
pub fn refresh_report_to_proto(report: &RefreshReport) -> pb::RefreshReport {
    pb::RefreshReport {
        table: report.table.clone(),
        version: report.version,
        parent_version: report.parent_version,
        inferred_rows: report.inferred_rows,
        added: report.added,
        changed: report.changed,
        deleted: report.deleted,
        unchanged: report.unchanged,
        dropped_rows: report.dropped_rows,
        live_rows: report.live_rows,
        masked_rows: report.masked_rows,
        outcome: match report.outcome {
            RefreshOutcome::Published => pb::RefreshOutcome::Published,
            RefreshOutcome::NoChange => pb::RefreshOutcome::NoChange,
        } as i32,
    }
}

/// Reconstruct a report from the wire — total: an unspecified outcome is a
/// faithful `invalid_argument`, never a guessed `Published`.
pub fn refresh_report_from_proto(report: pb::RefreshReport) -> Result<RefreshReport, Status> {
    let outcome = match pb::RefreshOutcome::try_from(report.outcome) {
        Ok(pb::RefreshOutcome::Published) => RefreshOutcome::Published,
        Ok(pb::RefreshOutcome::NoChange) => RefreshOutcome::NoChange,
        _ => return Err(Status::invalid_argument("unknown refresh outcome")),
    };
    Ok(RefreshReport {
        table: report.table,
        version: report.version,
        parent_version: report.parent_version,
        inferred_rows: report.inferred_rows,
        added: report.added,
        changed: report.changed,
        deleted: report.deleted,
        unchanged: report.unchanged,
        dropped_rows: report.dropped_rows,
        live_rows: report.live_rows,
        masked_rows: report.masked_rows,
        outcome,
    })
}

/// Encode an expiry report onto the wire.
pub fn expiry_report_to_proto(report: &ExpiryReport) -> pb::ExpiryReport {
    pb::ExpiryReport {
        table: report.table.clone(),
        expired_versions: report.expired_versions.clone(),
        objects_deleted: report.objects_deleted,
    }
}

/// Reconstruct an expiry report from the wire.
pub fn expiry_report_from_proto(report: pb::ExpiryReport) -> ExpiryReport {
    ExpiryReport {
        table: report.table,
        expired_versions: report.expired_versions,
        objects_deleted: report.objects_deleted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_round_trip() {
        let report = RefreshReport {
            table: "t".into(),
            version: Some(3),
            parent_version: Some(2),
            inferred_rows: 5,
            added: 1,
            changed: 4,
            deleted: 2,
            unchanged: 100,
            dropped_rows: 1,
            live_rows: 103,
            masked_rows: 6,
            outcome: RefreshOutcome::Published,
        };
        assert_eq!(
            refresh_report_from_proto(refresh_report_to_proto(&report)).unwrap(),
            report
        );
        let expiry = ExpiryReport {
            table: "t".into(),
            expired_versions: vec![0, 1],
            objects_deleted: 7,
        };
        assert_eq!(
            expiry_report_from_proto(expiry_report_to_proto(&expiry)),
            expiry
        );
    }

    #[test]
    fn delete_policy_defaults_to_tombstone() {
        assert_eq!(
            delete_policy_from_proto(0).unwrap(),
            DeletePolicy::Tombstone
        );
        assert_eq!(
            delete_policy_from_proto(pb::DeletePolicy::Retain as i32).unwrap(),
            DeletePolicy::Retain
        );
        assert!(delete_policy_from_proto(99).is_err());
    }
}
