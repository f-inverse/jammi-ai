//! Schema definition and DDL helpers for the audit table.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::session::JammiSession;
use crate::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};

use super::error::AuditError;

/// Reserved name of the per-query audit table.
///
/// The `_jammi_` prefix marks it substrate-owned; users may read it via SQL but
/// cannot create or directly write it (see [`is_reserved_table_name`]).
pub const AUDIT_TABLE_NAME: &str = "_jammi_search_audit";

/// Trigger topic every logged record is published to. Registered (idempotently)
/// through the catalog [`crate::catalog::topic_repo::TopicRepo`] on first log,
/// which mints the topic id and provisions its backing table.
pub const AUDIT_TOPIC: &str = "jammi.audit.search.v1";

/// Returns `true` if `name` is reserved for substrate-internal use.
///
/// The `_jammi_` prefix marks tables the substrate creates and maintains
/// implicitly. Users may read them via SQL but must not create or directly
/// write them, because doing so would bypass substrate-enforced invariants
/// such as audit signing.
pub fn is_reserved_table_name(name: &str) -> bool {
    name.starts_with("_jammi_")
}

/// Arrow schema for the audit table.
///
/// JSON-typed fields (`query_lineage`, `top_k_result_ids`, `retrieval_scores`)
/// are stored as `Utf8` JSON text; `executed_at` is `Int64` epoch microseconds
/// (matching the trigger backing-table convention so both SQLite and Postgres
/// round-trip identically). The implicit `tenant_id` column is added by the
/// mutable-table backend and is intentionally absent here.
pub fn audit_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("model_version", DataType::Utf8, false),
        Field::new("query_lineage", DataType::Utf8, false),
        Field::new("top_k_result_ids", DataType::Utf8, false),
        Field::new("retrieval_scores", DataType::Utf8, false),
        Field::new("executed_at", DataType::Int64, false),
        Field::new("signature", DataType::Utf8, false),
    ]))
}

/// Build the `MutableTableId` for the audit table.
///
/// The id is a compile-time constant shape so this never fails in practice; an
/// error is surfaced rather than panicked to keep the call site honest.
pub fn audit_table_id() -> Result<MutableTableId, AuditError> {
    MutableTableId::new(AUDIT_TABLE_NAME)
        .map_err(|e| AuditError::Storage(format!("audit table id: {e}")))
}

/// Create the audit table if it does not already exist.
///
/// Uses the substrate-internal `register_mutable_table_unchecked` path because
/// the audit table name is reserved and the user-facing
/// `create_mutable_table` would (correctly) refuse it. The `order_column` is
/// `executed_at` so reads can sort newest-first.
pub async fn ensure_table_exists(session: &JammiSession) -> Result<(), AuditError> {
    let id = audit_table_id()?;
    let exists = session
        .mutable_tables()
        .list_all()
        .await
        .map_err(|e| AuditError::Storage(e.to_string()))?
        .iter()
        .any(|def| def.id == id);
    if exists {
        return Ok(());
    }

    let def = MutableTableDefinitionBuilder::new(id, audit_schema())
        .primary_key(vec!["query_id".to_string()])
        .order_column("executed_at")
        .build()
        .map_err(|e| AuditError::Storage(e.to_string()))?;

    session
        .register_mutable_table_unchecked(def)
        .await
        .map_err(|e| AuditError::Storage(e.to_string()))?;
    Ok(())
}
