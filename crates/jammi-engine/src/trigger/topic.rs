//! `TopicDefinition` — the catalog-registered shape of one trigger-stream topic.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::tenant::TenantId;
use crate::trigger::ids::TopicId;

/// Engine-controlled column that carries the broker-assigned offset.
/// Stable across every row of one publish call so the subscribe path can
/// group the row-set back into the originally-published `RecordBatch`.
pub const OFFSET_COLUMN: &str = "_offset";

/// Engine-controlled column that carries the per-row position within a
/// publish call. Combined with `_offset` it forms the composite primary
/// key so multiple rows can share an offset without violating uniqueness.
pub const ROW_INDEX_COLUMN: &str = "_row_idx";

/// Engine-controlled column that carries the publish-time UTC instant.
/// Stored as `Int64` microseconds since the Unix epoch so the backing
/// table's storage type works identically across both backends — Postgres
/// stores it as `BIGINT`, SQLite as `INTEGER`, and the read path decodes
/// to `chrono::DateTime<Utc>` via `DateTime::from_timestamp_micros`.
pub const PRODUCED_AT_COLUMN: &str = "_produced_at";

/// Catalog-registered topic. The schema is the contract every published
/// batch must satisfy. Topic identity is `name` + `tenant`; the `id` field
/// is engine-minted.
#[derive(Debug, Clone)]
pub struct TopicDefinition {
    pub id: TopicId,
    /// Fully-qualified catalog name. Opaque to the engine beyond catalog
    /// lookup.
    pub name: String,
    /// Arrow schema for batch payloads. Validated on every publish.
    pub schema: SchemaRef,
    /// Tenant scope per ADR-00. `None` is the engine-default global topic.
    pub tenant: Option<TenantId>,
    /// Broker-driver-specific configuration (retention, replication, etc.).
    /// Opaque to the engine; surfaced to the driver via
    /// [`crate::trigger::broker::TriggerBroker::register_topic`].
    pub broker_metadata: BTreeMap<String, String>,
}

impl TopicDefinition {
    /// Conventional name for the backing mutable table that persists the
    /// topic's event log.
    pub fn backing_table_name(&self) -> String {
        format!("__topic_{}", self.id.as_uuid().simple())
    }

    /// Schema of the backing mutable table: engine-controlled offset and
    /// timestamp columns prepended to the topic's payload schema. Phase 2's
    /// mutable backend adds the implicit `tenant_id` column on its own; it
    /// is not included here.
    pub fn backing_table_schema(&self) -> SchemaRef {
        Arc::new(augment_schema_for_backing(&self.schema))
    }
}

/// Prepend the three engine-controlled columns to a user schema. Used by
/// both the backing-table builder and the publisher's batch augmentation.
pub fn augment_schema_for_backing(user: &SchemaRef) -> Schema {
    let mut fields: Vec<Field> = Vec::with_capacity(user.fields().len() + 3);
    fields.push(Field::new(OFFSET_COLUMN, DataType::Int64, false));
    fields.push(Field::new(ROW_INDEX_COLUMN, DataType::Int64, false));
    fields.push(Field::new(PRODUCED_AT_COLUMN, DataType::Int64, false));
    for f in user.fields() {
        fields.push(f.as_ref().clone());
    }
    Schema::new(fields)
}
