//! Catalog-row operations for trigger-stream topics.
//!
//! `topic_repo` persists the [`TopicDefinition`] tuple — id, name, Arrow
//! schema (serialised as IPC bytes), tenant scope, broker metadata, backing
//! mutable-table name — and the matching Phase-2 backing table inside one
//! `CatalogBackend::transaction`. Schemas are IPC-encoded so the read path
//! reconstructs the exact `arrow_schema::Schema` the publisher saw, including
//! metadata that JSON encoders would drop.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::str::FromStr;
use std::sync::Arc;

use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;

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
        let schema_ipc = encode_schema_ipc(&topic.schema)?;
        let tenant_str = topic.tenant.map(|t| t.to_string());
        let broker_metadata = serde_json::to_string(&topic.broker_metadata)
            .map_err(|e| TriggerError::Catalog(format!("broker_metadata serialisation: {e}")))?;
        let backing_table = backing_id.as_str().to_string();

        let backend = self.catalog.backend_arc();
        backend
            .transaction(TxOptions::default(), move |tx| {
                let topic_id = topic_id.clone();
                let name = name.clone();
                let schema_ipc = schema_ipc.clone();
                let tenant_str = tenant_str.clone();
                let broker_metadata = broker_metadata.clone();
                let backing_table = backing_table.clone();
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO topics \
                         (topic_id, name, schema_arrow_ipc, tenant_id, broker_metadata, backing_table) \
                         VALUES ($1, $2, $3, $4, $5, $6)",
                        &[
                            SqlValue::TextOwned(topic_id),
                            SqlValue::TextOwned(name),
                            SqlValue::BytesOwned(schema_ipc),
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
                            "SELECT topic_id, name, schema_arrow_ipc, tenant_id, \
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
                            "SELECT topic_id, name, schema_arrow_ipc, tenant_id, \
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
                            "SELECT topic_id, name, schema_arrow_ipc, tenant_id, \
                                    broker_metadata, backing_table \
                             FROM topics \
                             WHERE topic_id = $1 AND (tenant_id = $2 OR tenant_id IS NULL)",
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
        let id_str_for_delete = topic_id.to_string();
        let tenant_for_delete = tenant.map(|t| t.to_string());
        backend
            .transaction(TxOptions::default(), move |tx| {
                let id_str = id_str_for_delete.clone();
                let tenant_str = tenant_for_delete.clone();
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM topics WHERE topic_id = $1 \
                         AND (tenant_id = $2 OR tenant_id IS NULL)",
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
    schema_ipc: Vec<u8>,
    tenant_id: Option<String>,
    broker_metadata: String,
    backing_table: String,
}

fn parse_topic_row(row: &Row<'_>) -> Result<RawTopicRow, BackendError> {
    Ok(RawTopicRow {
        topic_id: row.get("topic_id")?,
        name: row.get("name")?,
        schema_ipc: row.get("schema_arrow_ipc")?,
        tenant_id: row.try_get("tenant_id")?,
        broker_metadata: row.get("broker_metadata")?,
        backing_table: row.get("backing_table")?,
    })
}

fn materialize_topic(raw: RawTopicRow) -> Result<TopicDefinition, TriggerError> {
    let id = TopicId::from_str(&raw.topic_id)
        .map_err(|e| TriggerError::Catalog(format!("topic_id parse: {e}")))?;
    let schema = decode_schema_ipc(&raw.schema_ipc)?;
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

fn encode_schema_ipc(schema: &SchemaRef) -> Result<Vec<u8>, TriggerError> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| TriggerError::Catalog(format!("schema IPC encode: {e}")))?;
        writer
            .finish()
            .map_err(|e| TriggerError::Catalog(format!("schema IPC finish: {e}")))?;
    }
    Ok(buf)
}

fn decode_schema_ipc(bytes: &[u8]) -> Result<SchemaRef, TriggerError> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| TriggerError::Catalog(format!("schema IPC decode: {e}")))?;
    Ok(reader.schema())
}
