//! Catalog-row operations for mutable companion tables.
//!
//! Methods land on [`Catalog`] directly (mirroring `source_repo` /
//! `result_repo` patterns). The catalog rows are the engine's registration
//! state; the *storage* tables themselves are created/dropped by the
//! [`MutableTableRegistry`](crate::source::mutable::MutableTableRegistry).

use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
use crate::store::mutable::definition::{
    MutableIndexDef, MutableTableDefinition, MutableTableError, MutableTableId,
};
use crate::tenant::TenantId;

use super::Catalog;

/// Raw `mutable_tables` row tuple — `(schema_json, primary_key_json,
/// tenant_id, user_metadata, order_column)`. Aliased so the repo's
/// internal closures stay readable.
type MutableRowTuple = (String, String, Option<String>, String, Option<String>);

/// Raw `mutable_tables` listing tuple — `(id, schema_json,
/// primary_key_json, tenant_id, user_metadata, order_column)`.
type MutableListingTuple = (
    String,
    String,
    String,
    Option<String>,
    String,
    Option<String>,
);

impl Catalog {
    /// Insert the registration row + index rows for `def` atomically.
    pub async fn create_mutable_table(
        &self,
        def: &MutableTableDefinition,
    ) -> Result<(), MutableTableError> {
        let id_str = def.id.as_str().to_string();
        let schema_json = encode_schema(def.schema.as_ref())?;
        let primary_key_json = serde_json::to_string(&def.primary_key)
            .map_err(|e| MutableTableError::Schema(e.to_string()))?;
        let user_metadata = def.user_metadata.to_string();
        let backend_kind = format!("{:?}", self.backend().backend_kind()).to_lowercase();
        let tenant_str = def.tenant.map(|t| t.to_string());
        let order_column = def.order_column.clone();
        let indexes = def.indexes.clone();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO mutable_tables \
                         (id, schema_json, primary_key, tenant_id, user_metadata, backend_kind, order_column) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7)",
                        &[
                            SqlValue::TextOwned(id_str.clone()),
                            SqlValue::TextOwned(schema_json),
                            SqlValue::TextOwned(primary_key_json),
                            SqlValue::from(tenant_str),
                            SqlValue::TextOwned(user_metadata),
                            SqlValue::TextOwned(backend_kind),
                            SqlValue::from(order_column),
                        ],
                    )
                    .await?;

                    for idx in &indexes {
                        let cols_json = serde_json::to_string(&idx.columns).map_err(|e| {
                            BackendError::Execution(format!("index columns JSON: {e}"))
                        })?;
                        tx.execute(
                            "INSERT INTO mutable_table_indexes \
                             (table_id, index_name, columns, is_unique) \
                             VALUES ($1, $2, $3, $4)",
                            &[
                                SqlValue::TextOwned(id_str.clone()),
                                SqlValue::TextOwned(idx.name.clone()),
                                SqlValue::TextOwned(cols_json),
                                SqlValue::Int(if idx.unique { 1 } else { 0 }),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Look up a mutable table by id, scoped to the catalog's current tenant.
    /// Returns `None` if no row with that id is owned by the current tenant or
    /// is global (`tenant_id IS NULL`). A tenant therefore sees its own table
    /// plus shared global tables, but never another tenant's — the read-path
    /// counterpart of [`Self::get_model`]'s `(tenant_id = $N OR tenant_id IS
    /// NULL)` predicate.
    pub async fn get_mutable_table(
        &self,
        id: &MutableTableId,
    ) -> Result<Option<MutableTableDefinition>, MutableTableError> {
        let tenant = self.current_tenant();
        self.get_mutable_table_for_tenant(id, tenant).await
    }

    /// Look up a mutable table by id, scoped to an explicit `tenant` rather
    /// than the catalog's session binding.
    ///
    /// The trigger publish/subscribe paths resolve and authorize a topic
    /// against the tenant-scoped [`crate::catalog::topic_repo::TopicRepo`]
    /// first, then resolve the topic's backing `__topic_*` table as an
    /// already-authorized, engine-internal step. The backing table is stamped
    /// with the *topic's* tenant at registration, but the session binding in
    /// effect at resolution time is the caller's request scope — which the
    /// data-plane handlers thread explicitly rather than via a task-local.
    /// Reading the binding here would scope the lookup to the wrong tenant, so
    /// those paths pass the topic's tenant directly. [`Self::get_mutable_table`]
    /// is the session-binding variant for ordinary catalog reads.
    pub async fn get_mutable_table_for_tenant(
        &self,
        id: &MutableTableId,
        tenant: Option<TenantId>,
    ) -> Result<Option<MutableTableDefinition>, MutableTableError> {
        let id_str = id.as_str().to_string();
        let row = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let id_str = id_str.clone();
                    let tenant_str = tenant.map(|t| t.to_string());
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT schema_json, primary_key, tenant_id, user_metadata, order_column \
                             FROM mutable_tables \
                             WHERE id = $1 AND (tenant_id = $2 OR tenant_id IS NULL)",
                            &[SqlValue::TextOwned(id_str), SqlValue::from(tenant_str)],
                            read_mutable_row,
                        )
                        .await
                    })
                },
            )
            .await?;

        let Some((schema_json, pk_json, tenant_str, metadata_json, order_column)) = row else {
            return Ok(None);
        };

        let indexes = self.list_mutable_table_indexes(id).await?;
        Ok(Some(materialize(
            id.clone(),
            schema_json,
            pk_json,
            tenant_str,
            metadata_json,
            order_column,
            indexes,
        )?))
    }

    /// List every registered mutable table across all tenants. Used by
    /// session startup to register a `TableProvider` for each persisted
    /// row so DataFusion can resolve `mutable.public.<id>` regardless of
    /// which tenant the session later binds to.
    pub async fn list_all_mutable_tables(
        &self,
    ) -> Result<Vec<MutableTableDefinition>, MutableTableError> {
        let entries: Vec<MutableListingTuple> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT id, schema_json, primary_key, tenant_id, user_metadata, order_column \
                             FROM mutable_tables ORDER BY id",
                            &[],
                            read_listed_row,
                        )
                        .await
                    })
                },
            )
            .await?;

        let mut defs = Vec::with_capacity(entries.len());
        for (id_str, schema_json, pk_json, tenant_str, metadata_json, order_column) in entries {
            let id = MutableTableId::new(id_str)?;
            let indexes = self.list_mutable_table_indexes(&id).await?;
            defs.push(materialize(
                id,
                schema_json,
                pk_json,
                tenant_str,
                metadata_json,
                order_column,
                indexes,
            )?);
        }
        Ok(defs)
    }

    /// List mutable tables, optionally filtered by tenant. When `tenant`
    /// is `None`, returns only rows whose `tenant_id` is `NULL`.
    pub async fn list_mutable_tables(
        &self,
        tenant: Option<TenantId>,
    ) -> Result<Vec<MutableTableDefinition>, MutableTableError> {
        let tenant_str = tenant.map(|t| t.to_string());
        let entries: Vec<MutableListingTuple> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let tenant_str = tenant_str.clone();
                    Box::pin(async move {
                        if let Some(ts) = tenant_str {
                            tx.query(
                                "SELECT id, schema_json, primary_key, tenant_id, user_metadata, order_column \
                                 FROM mutable_tables WHERE tenant_id = $1 ORDER BY id",
                                &[SqlValue::TextOwned(ts)],
                                read_listed_row,
                            )
                            .await
                        } else {
                            tx.query(
                                "SELECT id, schema_json, primary_key, tenant_id, user_metadata, order_column \
                                 FROM mutable_tables WHERE tenant_id IS NULL ORDER BY id",
                                &[],
                                read_listed_row,
                            )
                            .await
                        }
                    })
                },
            )
            .await?;

        let mut defs = Vec::with_capacity(entries.len());
        for (id_str, schema_json, pk_json, tenant_str, metadata_json, order_column) in entries {
            let id = MutableTableId::new(id_str)?;
            let indexes = self.list_mutable_table_indexes(&id).await?;
            defs.push(materialize(
                id,
                schema_json,
                pk_json,
                tenant_str,
                metadata_json,
                order_column,
                indexes,
            )?);
        }
        Ok(defs)
    }

    /// Delete the catalog row + index rows for `id`, scoped strictly to the
    /// catalog's current tenant. Caller is responsible for issuing the backend
    /// `DROP TABLE` first.
    ///
    /// The tenant predicate is the STRICT form (`tenant_id = $cur OR (tenant_id
    /// IS NULL AND $cur IS NULL)`) used by [`Self::delete_model`], not the read
    /// path's `OR tenant_id IS NULL` leak: a tenant session (non-NULL `$cur`)
    /// deletes only a row it owns, never another tenant's and never a shared
    /// GLOBAL (`tenant_id IS NULL`) table it did not create; an unscoped session
    /// deletes only its own GLOBAL rows. The index-row delete is filtered by the
    /// same predicate against the parent so a non-matching table leaves both its
    /// catalog row and index rows untouched.
    pub async fn delete_mutable_table(&self, id: &MutableTableId) -> Result<(), MutableTableError> {
        let id_str = id.as_str().to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                let id_str = id_str.clone();
                let tenant_str = tenant.map(|t| t.to_string());
                Box::pin(async move {
                    // Index rows cascade via the FK, but we delete explicitly
                    // for backends that don't enforce CASCADE on DELETE. The
                    // index delete is gated on the parent row being owned by the
                    // current tenant, so a foreign or shared table keeps its
                    // index rows when the catalog-row delete below matches none.
                    tx.execute(
                        "DELETE FROM mutable_table_indexes WHERE table_id = $1 AND EXISTS ( \
                             SELECT 1 FROM mutable_tables \
                             WHERE id = $1 \
                               AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL)))",
                        &[
                            SqlValue::TextOwned(id_str.clone()),
                            SqlValue::from(tenant_str.clone()),
                        ],
                    )
                    .await?;
                    tx.execute(
                        "DELETE FROM mutable_tables \
                         WHERE id = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                        &[SqlValue::TextOwned(id_str), SqlValue::from(tenant_str)],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Internal: load index definitions for one mutable table.
    async fn list_mutable_table_indexes(
        &self,
        id: &MutableTableId,
    ) -> Result<Vec<MutableIndexDef>, MutableTableError> {
        let id_str = id.as_str().to_string();
        let rows: Vec<(String, String, i64)> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let id_str = id_str.clone();
                    Box::pin(async move {
                        tx.query(
                            "SELECT index_name, columns, is_unique \
                             FROM mutable_table_indexes WHERE table_id = $1 ORDER BY index_name",
                            &[SqlValue::TextOwned(id_str)],
                            |row| {
                                Ok((
                                    row.get::<String>("index_name")?,
                                    row.get::<String>("columns")?,
                                    row.get::<i64>("is_unique")?,
                                ))
                            },
                        )
                        .await
                    })
                },
            )
            .await?;

        rows.into_iter()
            .map(|(name, cols_json, is_unique)| {
                let columns: Vec<String> = serde_json::from_str(&cols_json)
                    .map_err(|e| MutableTableError::Schema(format!("index columns JSON: {e}")))?;
                Ok(MutableIndexDef {
                    name,
                    columns,
                    unique: is_unique != 0,
                })
            })
            .collect()
    }
}

fn read_mutable_row(row: &Row<'_>) -> Result<MutableRowTuple, BackendError> {
    Ok((
        row.get::<String>("schema_json")?,
        row.get::<String>("primary_key")?,
        row.try_get::<String>("tenant_id")?,
        row.get::<String>("user_metadata")?,
        row.try_get::<String>("order_column")?,
    ))
}

fn read_listed_row(row: &Row<'_>) -> Result<MutableListingTuple, BackendError> {
    Ok((
        row.get::<String>("id")?,
        row.get::<String>("schema_json")?,
        row.get::<String>("primary_key")?,
        row.try_get::<String>("tenant_id")?,
        row.get::<String>("user_metadata")?,
        row.try_get::<String>("order_column")?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialize(
    id: MutableTableId,
    schema_json: String,
    pk_json: String,
    tenant_str: Option<String>,
    metadata_json: String,
    order_column: Option<String>,
    indexes: Vec<MutableIndexDef>,
) -> Result<MutableTableDefinition, MutableTableError> {
    let schema = decode_schema(&schema_json)
        .map_err(|e| MutableTableError::Schema(format!("schema_json: {e}")))?;
    let primary_key: Vec<String> = serde_json::from_str(&pk_json)
        .map_err(|e| MutableTableError::Schema(format!("primary_key JSON: {e}")))?;
    let user_metadata: serde_json::Value = serde_json::from_str(&metadata_json)
        .map_err(|e| MutableTableError::Schema(format!("user_metadata JSON: {e}")))?;
    let tenant = tenant_str
        .as_deref()
        .map(TenantId::from_str)
        .transpose()
        .map_err(|e| MutableTableError::Schema(format!("tenant_id: {e}")))?;
    Ok(MutableTableDefinition {
        id,
        schema: Arc::new(schema),
        primary_key,
        tenant,
        indexes,
        user_metadata,
        order_column,
        chunk_size: 8192,
    })
}

/// Encode an Arrow `Schema` as a compact JSON blob.
///
/// `arrow_schema::Schema` doesn't implement `Serialize` directly under our
/// feature set; this encoder captures the column subset Phase 2 supports
/// (the closed set of primitive types accepted by the SQLite/Postgres
/// `MutableBackend` impls).
fn encode_schema(schema: &Schema) -> Result<String, MutableTableError> {
    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|f| {
            let type_name = data_type_name(f.data_type()).ok_or_else(|| {
                MutableTableError::Schema(format!(
                    "unsupported mutable-table column type for '{}': {:?}",
                    f.name(),
                    f.data_type()
                ))
            })?;
            Ok(serde_json::json!({
                "name": f.name(),
                "type": type_name,
                "nullable": f.is_nullable(),
            }))
        })
        .collect::<Result<_, MutableTableError>>()?;
    Ok(serde_json::json!({ "fields": fields }).to_string())
}

fn decode_schema(json: &str) -> Result<Schema, String> {
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
    let wire: Wire = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let fields: Result<Vec<Field>, String> = wire
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

fn data_type_from_name(name: &str) -> Result<DataType, String> {
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
        other => return Err(format!("unsupported column type: {other}")),
    })
}
