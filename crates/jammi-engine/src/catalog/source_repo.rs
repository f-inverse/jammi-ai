use crate::error::{JammiError, Result};
use crate::source::{SourceConnection, SourceType};

use super::backend::{SqlValue, TxOptions};
use super::Catalog;

/// Materialized row from the `sources` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceRecord {
    /// Unique identifier for this data source.
    pub source_id: String,
    /// Backend type (e.g., `Local`, `Postgres`, `S3`).
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
    /// Persist a new source to the catalog.
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

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO sources (source_id, name, source_type, uri, options) \
                         VALUES ($1, $2, $3, $4, $5)",
                        &[
                            SqlValue::TextOwned(sid.clone()),
                            SqlValue::TextOwned(sid),
                            SqlValue::TextOwned(type_str),
                            SqlValue::TextOwned(uri),
                            SqlValue::TextOwned(options),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Look up a source by ID.
    pub async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>> {
        let sid = source_id.to_string();
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
                         FROM sources WHERE source_id = $1",
                            &[SqlValue::TextOwned(sid)],
                            read_source_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        raw.map(parse_source_row).transpose()
    }

    /// List all registered sources.
    pub async fn list_sources(&self) -> Result<Vec<SourceRecord>> {
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
                         FROM sources ORDER BY created_at",
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

    /// Remove a source from the catalog.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        let sid = source_id.to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM sources WHERE source_id = $1",
                        &[SqlValue::TextOwned(sid)],
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
