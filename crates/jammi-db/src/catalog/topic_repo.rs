//! Catalog-row operations for trigger-stream topics.
//!
//! `topic_repo` persists the [`TopicDefinition`] tuple — id, name, Arrow
//! schema (serialised as JSON in a `TEXT` column to keep the DDL identical
//! on SQLite and Postgres), tenant scope, broker metadata, backing
//! mutable-table name — and the matching Phase-2 backing table inside one
//! `CatalogBackend::transaction`. The encoding mirrors what `mutable_repo`
//! does for `mutable_tables.schema_json` so the two catalog tables decode
//! through the same lens.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
use crate::catalog::Catalog;
use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use crate::tenant::TenantId;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::TopicId;
use crate::trigger::topic::{
    augment_schema_for_backing, TopicDefinition, OFFSET_COLUMN, ROW_INDEX_COLUMN,
};

/// CRUD over the `topics` catalog table. Holds shared references to the
/// catalog backend and the mutable-table registry so `register_topic` can
/// commit both the topic row and the backing-table provisioning atomically.
pub struct TopicRepo {
    catalog: Arc<Catalog>,
    mutable: Arc<MutableTableRegistry>,
}

impl TopicRepo {
    pub fn new(catalog: Arc<Catalog>, mutable: Arc<MutableTableRegistry>) -> Self {
        Self { catalog, mutable }
    }

    /// Register a topic: insert the `topics` row and provision the Phase-2
    /// backing table. The backing table is registered via the mutable-table
    /// registry, which runs its own transaction; on failure the partial
    /// topics-row insert is rolled back via the engine's higher-level
    /// transaction so the catalog stays consistent.
    pub async fn register_topic(&self, topic: &TopicDefinition) -> Result<(), TriggerError> {
        // 1. Create the backing mutable table first — that gives us the
        //    storage rows that the topic catalog points at via FK.
        let backing_id = MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
        let augmented = Arc::new(augment_schema_for_backing(&topic.schema));
        let backing_def = MutableTableDefinitionBuilder::new(backing_id.clone(), augmented)
            .allow_internal_columns()
            .primary_key(vec![
                OFFSET_COLUMN.to_string(),
                ROW_INDEX_COLUMN.to_string(),
            ])
            .order_column(OFFSET_COLUMN)
            .tenant(topic.tenant)
            .build()
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
        self.mutable
            .register(backing_def)
            .await
            .map_err(TriggerError::BackingTable)?;

        // 2. Insert the topic catalog row referencing the backing table.
        let topic_id = topic.id.to_string();
        let name = topic.name.clone();
        let schema_json = encode_schema_json(topic.schema.as_ref())?;
        let tenant_str = topic.tenant.map(|t| t.to_string());
        let broker_metadata = serde_json::to_string(&topic.broker_metadata)
            .map_err(|e| TriggerError::Catalog(format!("broker_metadata serialisation: {e}")))?;
        let backing_table = backing_id.as_str().to_string();

        let backend = self.catalog.backend_arc();
        backend
            .transaction(TxOptions::default(), move |tx| {
                let topic_id = topic_id.clone();
                let name = name.clone();
                let schema_json = schema_json.clone();
                let tenant_str = tenant_str.clone();
                let broker_metadata = broker_metadata.clone();
                let backing_table = backing_table.clone();
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO topics \
                         (topic_id, name, schema_json, tenant_id, broker_metadata, backing_table) \
                         VALUES ($1, $2, $3, $4, $5, $6)",
                        &[
                            SqlValue::TextOwned(topic_id),
                            SqlValue::TextOwned(name),
                            SqlValue::TextOwned(schema_json),
                            SqlValue::from(tenant_str),
                            SqlValue::TextOwned(broker_metadata),
                            SqlValue::TextOwned(backing_table),
                        ],
                    )
                    .await?;
                    Ok::<(), BackendError>(())
                })
            })
            .await
            .map_err(TriggerError::Backend)
    }

    /// Look up a topic by its fully-qualified name. Tenant-filtered: a
    /// scoped session sees only its own topics plus the global (`NULL`)
    /// rows, per ADR-00's `tenant_id = $session OR tenant_id IS NULL`
    /// predicate.
    pub async fn lookup_by_name(
        &self,
        name: &str,
        tenant: Option<TenantId>,
    ) -> Result<Option<TopicDefinition>, TriggerError> {
        let name_owned = name.to_string();
        let tenant_str = tenant.map(|t| t.to_string());
        let backend = self.catalog.backend_arc();
        let row = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    let name_owned = name_owned.clone();
                    let tenant_str = tenant_str.clone();
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT topic_id, name, schema_json, tenant_id, \
                                    broker_metadata, backing_table \
                             FROM topics \
                             WHERE name = $1 AND (tenant_id = $2 OR tenant_id IS NULL) \
                             LIMIT 1",
                            &[SqlValue::TextOwned(name_owned), SqlValue::from(tenant_str)],
                            parse_topic_row,
                        )
                        .await
                    })
                },
            )
            .await
            .map_err(TriggerError::Backend)?;
        match row {
            Some(raw) => Ok(Some(materialize_topic(raw)?)),
            None => Ok(None),
        }
    }

    /// List every topic visible to the given tenant, in registration order.
    pub async fn list_topics(
        &self,
        tenant: Option<TenantId>,
    ) -> Result<Vec<TopicDefinition>, TriggerError> {
        let tenant_str = tenant.map(|t| t.to_string());
        let backend = self.catalog.backend_arc();
        let raw_rows: Vec<RawTopicRow> = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    let tenant_str = tenant_str.clone();
                    Box::pin(async move {
                        tx.query(
                            "SELECT topic_id, name, schema_json, tenant_id, \
                                    broker_metadata, backing_table \
                             FROM topics \
                             WHERE (tenant_id = $1 OR tenant_id IS NULL) \
                             ORDER BY created_at, topic_id",
                            &[SqlValue::from(tenant_str)],
                            parse_topic_row,
                        )
                        .await
                    })
                },
            )
            .await
            .map_err(TriggerError::Backend)?;
        raw_rows.into_iter().map(materialize_topic).collect()
    }

    /// Drop a topic: remove the catalog row and the matching backing
    /// mutable table.
    pub async fn drop_topic(
        &self,
        topic_id: TopicId,
        tenant: Option<TenantId>,
    ) -> Result<(), TriggerError> {
        // 1. Look up the topic to find the backing-table name. Dropping the
        //    catalog row before the backing table would trip the FK
        //    constraint (`ON DELETE RESTRICT`).
        //
        //    The lookup carries the SAME STRICT predicate as the delete below:
        //    a tenant session resolves only a topic it owns, never a shared
        //    GLOBAL (`tenant_id IS NULL`) one it did not create, so it surfaces
        //    `TopicNotFound` and never reaches the delete + `DROP TABLE`. Keeping
        //    the lookup loose would let a non-owner pass step 1 (resolving the
        //    backing-table name), have the strict delete match zero rows, and
        //    still drop the shared backing table in step 3 — orphaning the
        //    GLOBAL catalog row and destroying shared data.
        let id_str = topic_id.to_string();
        let tenant_str = tenant.map(|t| t.to_string());
        let backend = self.catalog.backend_arc();
        let raw = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    let id_str = id_str.clone();
                    let tenant_str = tenant_str.clone();
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT topic_id, name, schema_json, tenant_id, \
                                    broker_metadata, backing_table \
                             FROM topics \
                             WHERE topic_id = $1 \
                               AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                            &[SqlValue::TextOwned(id_str), SqlValue::from(tenant_str)],
                            parse_topic_row,
                        )
                        .await
                    })
                },
            )
            .await
            .map_err(TriggerError::Backend)?;
        let raw = raw.ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;

        // 2. Delete the topic row first so the FK no longer pins the table.
        //    STRICT delete predicate (mirrors `delete_model`): a tenant deletes
        //    only its own row, never a GLOBAL one — only an unscoped session
        //    (`$2 IS NULL`) manages GLOBAL rows. Step 1 already gated on the
        //    same predicate, so this is the matching delete for the row we just
        //    confirmed we own.
        let id_str_for_delete = topic_id.to_string();
        let tenant_for_delete = tenant.map(|t| t.to_string());
        backend
            .transaction(TxOptions::default(), move |tx| {
                let id_str = id_str_for_delete.clone();
                let tenant_str = tenant_for_delete.clone();
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM topics WHERE topic_id = $1 \
                         AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                        &[SqlValue::TextOwned(id_str), SqlValue::from(tenant_str)],
                    )
                    .await?;
                    Ok::<(), BackendError>(())
                })
            })
            .await
            .map_err(TriggerError::Backend)?;

        // 3. Now drop the backing mutable table.
        let backing_id = MutableTableId::new(raw.backing_table)
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
        self.mutable
            .drop_table(&backing_id)
            .await
            .map_err(TriggerError::BackingTable)?;
        Ok(())
    }
}

/// Raw row read straight out of `SELECT` against `topics`, before schema
/// decode + tenant parse.
struct RawTopicRow {
    topic_id: String,
    name: String,
    schema_json: String,
    tenant_id: Option<String>,
    broker_metadata: String,
    backing_table: String,
}

fn parse_topic_row(row: &Row<'_>) -> Result<RawTopicRow, BackendError> {
    Ok(RawTopicRow {
        topic_id: row.get("topic_id")?,
        name: row.get("name")?,
        schema_json: row.get("schema_json")?,
        tenant_id: row.try_get("tenant_id")?,
        broker_metadata: row.get("broker_metadata")?,
        backing_table: row.get("backing_table")?,
    })
}

fn materialize_topic(raw: RawTopicRow) -> Result<TopicDefinition, TriggerError> {
    let id = TopicId::from_str(&raw.topic_id)
        .map_err(|e| TriggerError::Catalog(format!("topic_id parse: {e}")))?;
    let schema = Arc::new(decode_schema_json(&raw.schema_json)?);
    let tenant = match raw.tenant_id {
        Some(s) => Some(
            TenantId::from_str(&s).map_err(|e| TriggerError::Catalog(format!("tenant: {e}")))?,
        ),
        None => None,
    };
    let broker_metadata: BTreeMap<String, String> = serde_json::from_str(&raw.broker_metadata)
        .map_err(|e| TriggerError::Catalog(format!("broker_metadata parse: {e}")))?;
    Ok(TopicDefinition {
        id,
        name: raw.name,
        schema,
        tenant,
        broker_metadata,
    })
}

/// Encode an Arrow schema as JSON — the same compact closed-type encoding
/// `mutable_repo` uses for `mutable_tables.schema_json`. Keeping the format
/// identical means the same backends decode both columns without a second
/// parser.
fn encode_schema_json(schema: &Schema) -> Result<String, TriggerError> {
    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|f| {
            let type_name = data_type_name(f.data_type()).ok_or_else(|| {
                TriggerError::UnsupportedSchemaType {
                    column: f.name().clone(),
                    data_type: format!("{:?}", f.data_type()),
                }
            })?;
            Ok(serde_json::json!({
                "name": f.name(),
                "type": type_name,
                "nullable": f.is_nullable(),
            }))
        })
        .collect::<Result<_, TriggerError>>()?;
    Ok(serde_json::json!({ "fields": fields }).to_string())
}

fn decode_schema_json(json: &str) -> Result<Schema, TriggerError> {
    #[derive(serde::Deserialize)]
    struct Wire {
        fields: Vec<WireField>,
    }
    #[derive(serde::Deserialize)]
    struct WireField {
        name: String,
        #[serde(rename = "type")]
        ty: String,
        nullable: bool,
    }
    let wire: Wire = serde_json::from_str(json)
        .map_err(|e| TriggerError::Catalog(format!("schema_json parse: {e}")))?;
    let fields: Result<Vec<Field>, TriggerError> = wire
        .fields
        .into_iter()
        .map(|w| Ok(Field::new(&w.name, data_type_from_name(&w.ty)?, w.nullable)))
        .collect();
    Ok(Schema::new(fields?))
}

fn data_type_name(ty: &DataType) -> Option<&'static str> {
    Some(match ty {
        DataType::Boolean => "Boolean",
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Utf8 => "Utf8",
        DataType::Binary => "Binary",
        _ => return None,
    })
}

fn data_type_from_name(name: &str) -> Result<DataType, TriggerError> {
    Ok(match name {
        "Boolean" => DataType::Boolean,
        "Int8" => DataType::Int8,
        "Int16" => DataType::Int16,
        "Int32" => DataType::Int32,
        "Int64" => DataType::Int64,
        "UInt8" => DataType::UInt8,
        "UInt16" => DataType::UInt16,
        "UInt32" => DataType::UInt32,
        "UInt64" => DataType::UInt64,
        "Float32" => DataType::Float32,
        "Float64" => DataType::Float64,
        "Utf8" => DataType::Utf8,
        "Binary" => DataType::Binary,
        other => {
            return Err(TriggerError::Catalog(format!(
                "unsupported topic schema type: {other}"
            )))
        }
    })
}
