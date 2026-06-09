use crate::error::{JammiError, Result};
use crate::source::{SourceConnection, SourceType};

use super::backend::{SqlValue, TxOptions};
use super::result_repo::ResultTableRecord;
use super::Catalog;

/// Registry introspection for one registered source: its registration identity
/// joined with the embedding result tables produced out of it.
///
/// The descriptor holds no copy of the embedding numbers: `source_id` /
/// `source_type` / `status` come from the [`SourceRecord`], and the embedding
/// `status` / `row_count` / `dimensions` ride on the [`ResultTableRecord`]s in
/// `result_tables` — the same records [`Catalog::find_result_tables`] and a
/// `generate_embeddings` call return, so there is one source-of-truth for those
/// numbers rather than a parallel registry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceDescriptor {
    /// The source's stable id.
    pub source_id: String,
    /// The storage backend the source was registered with.
    pub source_type: SourceType,
    /// Registration lifecycle status from the source catalog (e.g. `"active"`).
    pub status: String,
    /// Every embedding result table produced from this source, in registration
    /// order. Empty until a `generate_embeddings` call persists one.
    pub result_tables: Vec<ResultTableRecord>,
}

/// Materialized row from the `sources` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceRecord {
    /// Unique identifier for this data source.
    pub source_id: String,
    /// Backend type (e.g., `File`, `Postgres`, `Mysql`).
    pub source_type: SourceType,
    /// Deserialized connection parameters.
    pub connection: SourceConnection,
    /// Cached schema JSON (populated after first introspection).
    pub schema_json: Option<String>,
    /// Lifecycle status (e.g., `"active"`).
    pub status: String,
    /// ISO-8601 timestamp of initial registration.
    pub created_at: String,
    /// ISO-8601 timestamp of last update.
    pub updated_at: String,
}

impl Catalog {
    /// Persist a new source to the catalog. The session's bound tenant is
    /// written to `tenant_id` and asserted via
    /// [`crate::catalog::backend::Transaction::assert_tenant_matches`] before
    /// the INSERT (SPEC-03 §7 defence-in-depth).
    pub async fn register_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: &SourceConnection,
    ) -> Result<()> {
        let type_str =
            serde_json::to_string(&source_type).map_err(|e| JammiError::Catalog(e.to_string()))?;
        let uri = connection.url.as_deref().unwrap_or("").to_string();
        let options =
            serde_json::to_string(connection).map_err(|e| JammiError::Catalog(e.to_string()))?;
        let sid = source_id.to_string();
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "sources")?;
                    tx.execute(
                        "INSERT INTO sources (source_id, name, source_type, uri, options, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, $6)",
                        &[
                            SqlValue::TextOwned(sid.clone()),
                            SqlValue::TextOwned(sid),
                            SqlValue::TextOwned(type_str),
                            SqlValue::TextOwned(uri),
                            SqlValue::TextOwned(options),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Look up a source by ID. Filtered to the session's tenant (own rows
    /// plus globally-scoped `tenant_id IS NULL` rows).
    pub async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>> {
        let sid = source_id.to_string();
        let tenant = self.current_tenant();
        let raw = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT source_id, source_type, options, schema_json, \
                                'active' AS status, created_at, updated_at \
                             FROM sources WHERE source_id = $1 \
                               AND (tenant_id = $2 OR tenant_id IS NULL)",
                            &[
                                SqlValue::TextOwned(sid),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            read_source_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        raw.map(parse_source_row).transpose()
    }

    /// List sources visible to the session's tenant.
    pub async fn list_sources(&self) -> Result<Vec<SourceRecord>> {
        let tenant = self.current_tenant();
        let raws = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT source_id, source_type, options, schema_json, \
                                'active' AS status, created_at, updated_at \
                             FROM sources \
                             WHERE tenant_id = $1 OR tenant_id IS NULL \
                             ORDER BY created_at",
                            &[SqlValue::from(tenant.map(|t| t.to_string()))],
                            read_source_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        raws.into_iter().map(parse_source_row).collect()
    }

    /// List every registered source across all tenants, in registration
    /// order. Used by session startup to re-register a `TableProvider` for
    /// each persisted source so DataFusion can resolve the source's catalog
    /// regardless of which tenant the session later binds to. Source provider
    /// registration is tenant-agnostic; tenant isolation is enforced at query
    /// time, not by which providers exist in the context.
    pub async fn list_all_sources(&self) -> Result<Vec<SourceRecord>> {
        let raws = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT source_id, source_type, options, schema_json, \
                                'active' AS status, created_at, updated_at \
                             FROM sources \
                             ORDER BY created_at",
                            &[],
                            read_source_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        raws.into_iter().map(parse_source_row).collect()
    }

    /// Describe one registered source: its registry record joined with the
    /// embedding result tables produced from it. Tenant-scoped (own rows plus
    /// globally-scoped). Returns `None` when no source with that id is visible
    /// to the session's tenant.
    ///
    /// The result tables carry the embedding `status` / `row_count` /
    /// `dimensions` — read from [`Self::find_result_tables`], the same records
    /// `generate_embeddings` returns — so the descriptor reports those numbers
    /// from their source-of-truth rather than a parallel store.
    pub async fn describe_source(&self, source_id: &str) -> Result<Option<SourceDescriptor>> {
        let record = match self.get_source(source_id).await? {
            Some(r) => r,
            None => return Ok(None),
        };
        Ok(Some(self.descriptor_for(record).await?))
    }

    /// Describe every source visible to the session's tenant, in registration
    /// order. This is [`Self::describe_source`] applied over [`Self::list_sources`]
    /// — one descriptor shape, one operator, no per-source special-casing.
    pub async fn list_source_descriptors(&self) -> Result<Vec<SourceDescriptor>> {
        let records = self.list_sources().await?;
        let mut descriptors = Vec::with_capacity(records.len());
        for record in records {
            descriptors.push(self.descriptor_for(record).await?);
        }
        Ok(descriptors)
    }

    /// Join one [`SourceRecord`] with its embedding result tables into a
    /// [`SourceDescriptor`]. The single operator both descriptor verbs descend
    /// through.
    async fn descriptor_for(&self, record: SourceRecord) -> Result<SourceDescriptor> {
        let result_tables = self
            .find_result_tables(&record.source_id, None, None)
            .await?;
        Ok(SourceDescriptor {
            source_id: record.source_id,
            source_type: record.source_type,
            status: record.status,
            result_tables,
        })
    }

    /// Remove a source from the catalog. Scoped to the session's tenant —
    /// a tenant cannot delete another tenant's source.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        let sid = source_id.to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "DELETE FROM sources WHERE source_id = $1 \
                           AND (tenant_id = $2 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(sid),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }
}

struct RawSourceRow {
    source_id: String,
    source_type: String,
    options: Option<String>,
    schema_json: Option<String>,
    status: String,
    created_at: String,
    updated_at: String,
}

fn read_source_row(
    row: &super::backend::Row<'_>,
) -> std::result::Result<RawSourceRow, super::backend::BackendError> {
    Ok(RawSourceRow {
        source_id: row.get("source_id")?,
        source_type: row.get("source_type")?,
        options: row.try_get("options")?,
        schema_json: row.try_get("schema_json")?,
        status: row.get("status")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
    })
}

fn parse_source_row(raw: RawSourceRow) -> Result<SourceRecord> {
    let source_type: SourceType = serde_json::from_str(&raw.source_type)
        .map_err(|e| JammiError::Catalog(format!("Invalid source_type: {e}")))?;
    let connection: SourceConnection = raw
        .options
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| JammiError::Catalog(format!("Invalid options: {e}")))?
        .unwrap_or_default();

    Ok(SourceRecord {
        source_id: raw.source_id,
        source_type,
        connection,
        schema_json: raw.schema_json,
        status: raw.status,
        created_at: raw.created_at,
        updated_at: raw.updated_at,
    })
}
