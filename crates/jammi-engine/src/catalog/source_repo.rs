use crate::error::{JammiError, Result};
use crate::source::{SourceConnection, SourceType};

use super::Catalog;

/// Materialized row from the `sources` catalog table.
#[derive(Debug, Clone)]
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
    pub fn register_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: &SourceConnection,
    ) -> Result<()> {
        let conn = self.conn()?;
        let type_str =
            serde_json::to_string(&source_type).map_err(|e| JammiError::Catalog(e.to_string()))?;
        let uri = connection.url.as_deref().unwrap_or("");
        let options =
            serde_json::to_string(connection).map_err(|e| JammiError::Catalog(e.to_string()))?;

        conn.execute(
            "INSERT INTO sources (source_id, name, source_type, uri, options)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![source_id, source_id, type_str, uri, options],
        )?;
        Ok(())
    }

    /// Look up a source by ID.
    pub fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT source_id, source_type, options, schema_json, 'active', created_at, updated_at
             FROM sources WHERE source_id = ?1",
        )?;

        let mut rows = stmt.query_map(rusqlite::params![source_id], |row| {
            Ok(RawSourceRow {
                source_id: row.get(0)?,
                source_type: row.get(1)?,
                options: row.get(2)?,
                schema_json: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;

        match rows.next() {
            Some(Ok(raw)) => Ok(Some(parse_source_row(raw)?)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List all registered sources.
    pub fn list_sources(&self) -> Result<Vec<SourceRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT source_id, source_type, options, schema_json, 'active', created_at, updated_at
             FROM sources ORDER BY created_at",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(RawSourceRow {
                source_id: row.get(0)?,
                source_type: row.get(1)?,
                options: row.get(2)?,
                schema_json: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;

        rows.map(|r| {
            let raw = r?;
            parse_source_row(raw)
        })
        .collect()
    }

    /// Remove a source from the catalog.
    pub fn remove_source(&self, source_id: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM sources WHERE source_id = ?1",
            rusqlite::params![source_id],
        )?;
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
